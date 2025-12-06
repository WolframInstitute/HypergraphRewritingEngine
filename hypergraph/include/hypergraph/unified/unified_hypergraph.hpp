#pragma once

#include <cstdint>
#include <cstring>
#include <atomic>
#include <vector>
#include <memory>

#include "types.hpp"
#include "signature.hpp"
#include "index.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "causal_graph.hpp"
#include "uniqueness_tree.hpp"
#include "shared_uniqueness_tree.hpp"
#include "concurrent_map.hpp"
#include "edge_equivalence.hpp"

// Include v1 canonicalizer for exact isomorphism checking
#include "../canonicalization.hpp"

namespace hypergraph::unified {

// =============================================================================
// UnifiedHypergraph
// =============================================================================
// Central storage for all hypergraph data in the multiway system.
//
// Key design principles:
// - All edges are stored once (unified storage)
// - States are SparseBitset views over the edge pool
// - Thread-safe allocation via atomic counters
// - Arena allocation for cache-friendly memory layout
// - Lock-free indices for concurrent pattern matching
//
// Thread safety:
// - Edge/state/event/match creation: Lock-free via atomic counters
// - Index updates: Lock-free via ConcurrentMap and LockFreeList
// - Reading: Always safe (immutable after creation)

class UnifiedHypergraph {
    // Global ID counters (thread-safe)
    GlobalCounters counters_;

    // Arena for all allocations (thread-safe for parallel evolution)
    ConcurrentHeterogeneousArena arena_;

    // Edge storage
    SegmentedArray<Edge> edges_;

    // State storage
    SegmentedArray<State> states_;

    // Event storage
    SegmentedArray<Event> events_;

    // Match storage
    SegmentedArray<Match> matches_;

    // Pattern matching indices
    PatternMatchingIndex match_index_;

    // Per-state child tracking for match cascading
    SegmentedArray<LockFreeList<StateId>> state_children_;

    // Per-state match lists
    SegmentedArray<LockFreeList<MatchId>> state_matches_;

    // Causal and branchial graph
    CausalGraph causal_graph_;

    // Edge equivalence manager for Level 2 canonicalization
    EdgeEquivalenceManager edge_equiv_manager_;

    // Canonical state deduplication map: canonical_hash -> StateId
    // Used to find existing equivalent states before creating new ones
    ConcurrentMap<uint64_t, StateId> canonical_state_map_;

    // TEMP: Mutex to test if concurrent insert is causing non-determinism
    mutable std::mutex canonical_mutex_;

    // Level 2 canonicalization enabled flag
    bool level2_enabled_{false};

    // Cached StateCanonicalInfo for each state (for correspondence computation)
    SegmentedArray<StateCanonicalInfo> state_canonical_info_;

    // Shared uniqueness tree for efficient state hashing
    std::unique_ptr<SharedUniquenessTree> shared_tree_;

    // Flag to use shared tree vs exact canonicalization
    // Enabled by default: WL hashing is O(V²×E) vs O(g!) factorial for exact canonicalization
    bool use_shared_tree_{true};

public:
    UnifiedHypergraph()
        : shared_tree_(std::make_unique<SharedUniquenessTree>(&arena_))
    {
        causal_graph_.set_arena(&arena_);
        edge_equiv_manager_.set_arena(&arena_);

        // Set up callback for cross-branch causal edges
        edge_equiv_manager_.set_causal_edge_callback(
            [this](EventId producer, EventId consumer) {
                causal_graph_.add_causal_edge(producer, consumer, INVALID_ID);
            }
        );
    }

    // Non-copyable
    UnifiedHypergraph(const UnifiedHypergraph&) = delete;
    UnifiedHypergraph& operator=(const UnifiedHypergraph&) = delete;

    // =========================================================================
    // Vertex Management
    // =========================================================================

    // Allocate a new vertex ID
    VertexId alloc_vertex() {
        return counters_.alloc_vertex();
    }

    // Allocate N consecutive vertex IDs
    VertexId alloc_vertices(uint32_t count) {
        VertexId first = counters_.next_vertex.fetch_add(count, std::memory_order_relaxed);
        return first;
    }

    // Get current vertex count (upper bound)
    uint32_t num_vertices() const {
        return counters_.next_vertex.load(std::memory_order_relaxed);
    }

    // Ensure vertex ID space is at least `max_id + 1`
    void reserve_vertices(VertexId max_id) {
        VertexId current = counters_.next_vertex.load(std::memory_order_relaxed);
        while (current <= max_id) {
            if (counters_.next_vertex.compare_exchange_weak(
                    current, max_id + 1, std::memory_order_relaxed)) {
                break;
            }
        }
    }

    // =========================================================================
    // Edge Management
    // =========================================================================

    // Create a new edge
    EdgeId create_edge(
        const VertexId* vertices,
        uint8_t arity,
        EventId creator_event = INVALID_ID,
        uint32_t step = 0
    ) {
        EdgeId eid = counters_.alloc_edge();

        // Allocate and copy vertex array
        VertexId* verts = arena_.allocate_array<VertexId>(arity);
        std::memcpy(verts, vertices, arity * sizeof(VertexId));

        // Directly construct edge at slot eid using emplace_at
        // This avoids the race condition in ensure_size where another thread's
        // emplace might be in the middle of constructing our slot
        edges_.emplace_at(eid, arena_, eid, verts, arity, creator_event, step);

        // CRITICAL: Release fence to ensure vertex data (from memcpy above) and
        // edge struct are visible to other threads before the edge ID escapes.
        // Without this, other threads reading edges_[eid] might see stale vertex data.
        std::atomic_thread_fence(std::memory_order_release);

        // Update indices
        match_index_.add_edge(eid, vertices, arity, arena_);

        // Register with shared uniqueness tree
        if (shared_tree_) {
            shared_tree_->register_edge(eid, vertices, arity);
        }

        return eid;
    }

    // Create edge from initializer list (convenience)
    EdgeId create_edge(std::initializer_list<VertexId> vertices,
                       EventId creator_event = INVALID_ID,
                       uint32_t step = 0) {
        VertexId verts[MAX_ARITY];
        uint8_t arity = 0;
        for (VertexId v : vertices) {
            if (arity < MAX_ARITY) {
                verts[arity++] = v;
            }
        }
        return create_edge(verts, arity, creator_event, step);
    }

    // Get edge by ID
    const Edge& get_edge(EdgeId eid) const {
        return edges_[eid];
    }

    Edge& get_edge(EdgeId eid) {
        return edges_[eid];
    }

    // Edge accessor (for pattern matching)
    auto edge_accessor() const {
        return [this](EdgeId eid) -> const Edge& {
            return edges_[eid];
        };
    }

    // Number of edges
    uint32_t num_edges() const {
        return counters_.next_edge.load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Edge Accessors for UniquenessTree
    // =========================================================================
    // These provide the interface needed by UniquenessTree::compute()

    // Get vertex array for an edge (returns pointer to vertices)
    const VertexId* edge_vertices(EdgeId eid) const {
        return edges_[eid].vertices;
    }

    // Get arity of an edge
    uint8_t edge_arity(EdgeId eid) const {
        return edges_[eid].arity;
    }

    // Helper class to provide indexed access to edge vertices
    class EdgeVertexAccessor {
        const UnifiedHypergraph* hg_;
    public:
        explicit EdgeVertexAccessor(const UnifiedHypergraph* hg) : hg_(hg) {}
        const VertexId* operator[](EdgeId eid) const {
            return hg_->edge_vertices(eid);
        }
    };

    // Helper class to provide indexed access to edge arities
    class EdgeArityAccessor {
        const UnifiedHypergraph* hg_;
    public:
        explicit EdgeArityAccessor(const UnifiedHypergraph* hg) : hg_(hg) {}
        uint8_t operator[](EdgeId eid) const {
            return hg_->edge_arity(eid);
        }
    };

    EdgeVertexAccessor edge_vertex_accessor() const {
        return EdgeVertexAccessor(this);
    }

    EdgeArityAccessor edge_arity_accessor() const {
        return EdgeArityAccessor(this);
    }

    // Raw accessor that returns a pointer-indexable object
    // This creates a temporary vector of vertex pointers for UniquenessTree
    class EdgeVertexAccessorRaw {
        mutable std::vector<const VertexId*> vertex_ptrs_;
        const UnifiedHypergraph* hg_;
    public:
        explicit EdgeVertexAccessorRaw(const UnifiedHypergraph* hg) : hg_(hg) {
            uint32_t num_edges = hg->num_edges();
            vertex_ptrs_.resize(num_edges);
            for (uint32_t i = 0; i < num_edges; ++i) {
                vertex_ptrs_[i] = hg->edge_vertices(i);
            }
        }
        const VertexId* operator[](EdgeId eid) const {
            return vertex_ptrs_[eid];
        }
        operator const VertexId* const*() const {
            return vertex_ptrs_.data();
        }
    };

    class EdgeArityAccessorRaw {
        mutable std::vector<uint8_t> arities_;
        const UnifiedHypergraph* hg_;
    public:
        explicit EdgeArityAccessorRaw(const UnifiedHypergraph* hg) : hg_(hg) {
            uint32_t num_edges = hg->num_edges();
            arities_.resize(num_edges);
            for (uint32_t i = 0; i < num_edges; ++i) {
                arities_[i] = hg->edge_arity(i);
            }
        }
        uint8_t operator[](EdgeId eid) const {
            return arities_[eid];
        }
        operator const uint8_t*() const {
            return arities_.data();
        }
    };

    EdgeVertexAccessorRaw edge_vertex_accessor_raw() const {
        return EdgeVertexAccessorRaw(this);
    }

    EdgeArityAccessorRaw edge_arity_accessor_raw() const {
        return EdgeArityAccessorRaw(this);
    }

    // =========================================================================
    // State Management
    // =========================================================================

    // Create a new state from edge set
    StateId create_state(
        SparseBitset&& edge_set,
        uint32_t step = 0,
        uint64_t canonical_hash = 0,
        EventId parent_event = INVALID_ID
    ) {
        StateId sid = counters_.alloc_state();

        // Ensure auxiliary arrays are large enough (thread-safe)
        // These are LockFreeLists which only need default construction
        state_children_.ensure_size(sid + 1, arena_);
        state_matches_.ensure_size(sid + 1, arena_);

        // Directly construct state at slot sid using emplace_at
        // This avoids the race condition in ensure_size where another thread's
        // emplace might be in the middle of constructing our slot while we're
        // trying to move-assign over it
        states_.emplace_at(sid, arena_, sid, std::move(edge_set), step, canonical_hash, parent_event);

        // CRITICAL: Release fence to ensure state data (including SparseBitset's
        // internal pointers and the chunk data they point to) is visible to other
        // threads before the state ID escapes.
        std::atomic_thread_fence(std::memory_order_release);

        return sid;
    }

    // Create state from edge IDs (convenience)
    StateId create_state(
        const EdgeId* edge_ids,
        uint32_t num_edges,
        uint32_t step = 0,
        uint64_t canonical_hash = 0,
        EventId parent_event = INVALID_ID
    ) {
        SparseBitset edge_set;
        for (uint32_t i = 0; i < num_edges; ++i) {
            edge_set.set(edge_ids[i], arena_);
        }
        return create_state(std::move(edge_set), step, canonical_hash, parent_event);
    }

    // Create state from initializer list (convenience)
    StateId create_state(std::initializer_list<EdgeId> edge_ids,
                         uint32_t step = 0,
                         uint64_t canonical_hash = 0,
                         EventId parent_event = INVALID_ID) {
        SparseBitset edge_set;
        for (EdgeId eid : edge_ids) {
            edge_set.set(eid, arena_);
        }
        return create_state(std::move(edge_set), step, canonical_hash, parent_event);
    }

    // Get state by ID
    const State& get_state(StateId sid) const {
        return states_[sid];
    }

    State& get_state(StateId sid) {
        return states_[sid];
    }

    // Get state's edge set
    const SparseBitset& get_state_edges(StateId sid) const {
        // CRITICAL: Acquire fence to ensure we see all state data written by
        // the thread that created this state. Pairs with release fence in create_state.
        std::atomic_thread_fence(std::memory_order_acquire);
        return states_[sid].edges;
    }

    // Number of states
    uint32_t num_states() const {
        return counters_.next_state.load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Canonical State Deduplication
    // =========================================================================

    // Result of trying to create a canonical state
    struct CanonicalStateResult {
        StateId state;       // The canonical state ID (existing or new)
        StateId raw_state;   // The raw state ID we created (always new, with actual edges)
        bool was_new;        // true if new state was created, false if existing found
    };

    // Create state if no equivalent exists, otherwise return existing
    // This is the main API for state creation with canonicalization.
    // If Level 2 is enabled and a duplicate is found, edge correspondence is computed.
    //
    // Thread safety: Fully linearizable. We create the state first, then try to
    // insert into the canonical map. If another thread wins, we return their state
    // (the created state becomes "wasted" but this is correct).
    CanonicalStateResult create_or_get_canonical_state(
        SparseBitset&& edge_set,
        uint64_t canonical_hash,
        uint32_t step = 0,
        EventId parent_event = INVALID_ID
    ) {
        // First, create the state unconditionally
        // This ensures the StateId we insert is always valid
        StateId new_sid = create_state(std::move(edge_set), step, canonical_hash, parent_event);

        // Try to insert into canonical map
        // TEMP: Use mutex to test if concurrent insert is causing non-determinism
        std::pair<StateId, bool> result;
        {
            std::lock_guard<std::mutex> lock(canonical_mutex_);
            result = canonical_state_map_.insert_if_absent(canonical_hash, new_sid);
        }
        auto [existing_or_new, was_inserted] = result;

        if (!was_inserted) {
            // Another thread beat us - they have the canonical representative
            // Our new_sid is the raw state with actual edges, existing_or_new is the canonical

            if (level2_enabled_) {
                // Compute edge correspondence between our state and existing state
                register_edge_correspondence(new_sid, existing_or_new);
            }

            return {existing_or_new, new_sid, false};
        }

        // We won the race - our state is the canonical representative
        // raw_state == state in this case
        return {new_sid, new_sid, true};
    }

    // Lookup existing canonical state by hash
    std::optional<StateId> find_canonical_state(uint64_t canonical_hash) const {
        return canonical_state_map_.lookup(canonical_hash);
    }

    // Get the canonical representative for a given state
    // Returns the state itself if it's already the canonical representative
    StateId get_canonical_state(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;
        const State& s = get_state(raw_state);
        if (s.canonical_hash == 0) return raw_state;  // No hash computed
        auto canonical = find_canonical_state(s.canonical_hash);
        return canonical.value_or(raw_state);
    }

    // Number of unique canonical states
    // Uses count_unique() for accurate counting after evolution completes,
    // handling the case where ConcurrentMap may have duplicate keys due to
    // concurrent insertions of the same canonical hash.
    size_t num_canonical_states() const {
        return canonical_state_map_.count_unique();
    }

    // =========================================================================
    // Level 2 Canonicalization: Edge Correspondence
    // =========================================================================

    // Enable Level 2 canonicalization (edge correspondence, event canonicalization)
    void enable_level2() {
        level2_enabled_ = true;
    }

    // Check if Level 2 is enabled
    bool level2_enabled() const {
        return level2_enabled_;
    }

    // Enable shared uniqueness tree for faster hashing with incremental computation
    void enable_shared_tree() {
        use_shared_tree_ = true;
    }

    // Disable shared uniqueness tree (use exact canonicalization)
    void disable_shared_tree() {
        use_shared_tree_ = false;
    }

    // Check if shared tree is enabled
    bool shared_tree_enabled() const {
        return use_shared_tree_;
    }

    // Get or compute canonical info for a state
    // The info is cached after first computation
    StateCanonicalInfo& get_or_compute_canonical_info(StateId sid) {
        // Ensure canonical info storage is large enough (thread-safe)
        state_canonical_info_.ensure_size(sid + 1, arena_);

        StateCanonicalInfo& info = state_canonical_info_[sid];

        // If already computed, return cached version
        if (info.canonical_hash != 0 || states_[sid].edges.empty()) {
            return info;
        }

        // Compute canonical info
        info = compute_canonical_info(states_[sid].edges);
        return info;
    }

    // Register edge correspondence between two isomorphic states
    // This merges corresponding edges into the same equivalence class
    void register_edge_correspondence(StateId new_state, StateId existing_state) {
        if (!level2_enabled_) return;

        // Get canonical info for both states
        StateCanonicalInfo& new_info = get_or_compute_canonical_info(new_state);
        StateCanonicalInfo& existing_info = get_or_compute_canonical_info(existing_state);

        // Find edge correspondence
        EdgeCorrespondence correspondence = UniquenessTree::find_correspondence(
            new_info,
            states_[new_state].edges,
            existing_info,
            states_[existing_state].edges,
            edge_vertex_accessor_raw(),
            edge_arity_accessor_raw(),
            arena_
        );

        if (!correspondence.valid) {
            // States not actually isomorphic (hash collision)
            return;
        }

        // Merge corresponding edges into equivalence classes
        for (uint32_t i = 0; i < correspondence.count; ++i) {
            edge_equiv_manager_.merge(
                correspondence.state1_edges[i],
                correspondence.state2_edges[i]
            );
        }
    }

    // Get EdgeEquivalenceManager (for edge producer/consumer tracking)
    EdgeEquivalenceManager& edge_equiv_manager() {
        return edge_equiv_manager_;
    }

    const EdgeEquivalenceManager& edge_equiv_manager() const {
        return edge_equiv_manager_;
    }

    // =========================================================================
    // State Parent-Child Relationships
    // =========================================================================

    // Add child to parent state
    void add_state_child(StateId parent, StateId child) {
        if (parent >= state_children_.size()) return;
        state_children_[parent].push(child, arena_);
    }

    // Iterate over children of a state
    template<typename Visitor>
    void for_each_child(StateId parent, Visitor&& visit) const {
        if (parent >= state_children_.size()) return;
        state_children_[parent].for_each([&](StateId child) {
            visit(child);
        });
    }

    // =========================================================================
    // Event Management
    // =========================================================================

    // Create a new event
    EventId create_event(
        StateId input_state,
        StateId output_state,
        RuleIndex rule_index,
        const EdgeId* consumed,
        uint8_t num_consumed,
        const EdgeId* produced,
        uint8_t num_produced,
        const VariableBinding& binding
    ) {
        EventId eid = counters_.alloc_event();

        // Allocate and copy edge arrays
        EdgeId* cons = arena_.allocate_array<EdgeId>(num_consumed);
        std::memcpy(cons, consumed, num_consumed * sizeof(EdgeId));

        EdgeId* prod = arena_.allocate_array<EdgeId>(num_produced);
        std::memcpy(prod, produced, num_produced * sizeof(EdgeId));

        // Directly construct event at slot eid using emplace_at
        // This avoids the race condition in ensure_size where another thread might
        // read the event before our assignment completes
        events_.emplace_at(eid, arena_, eid, input_state, output_state, rule_index,
                           cons, num_consumed, prod, num_produced, binding);

        // Track parent-child relationship
        add_state_child(input_state, output_state);

        return eid;
    }

    // Get event by ID
    const Event& get_event(EventId eid) const {
        return events_[eid];
    }

    Event& get_event(EventId eid) {
        return events_[eid];
    }

    // Number of events
    uint32_t num_events() const {
        return counters_.next_event.load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Match Management
    // =========================================================================

    // Register a new match
    MatchId register_match(
        RuleIndex rule_index,
        const EdgeId* matched_edges,
        uint8_t num_edges,
        const VariableBinding& binding,
        StateId origin_state
    ) {
        MatchId mid = counters_.alloc_match();

        // Allocate and copy edge array
        EdgeId* edges = arena_.allocate_array<EdgeId>(num_edges);
        std::memcpy(edges, matched_edges, num_edges * sizeof(EdgeId));

        // Directly construct match at slot mid using emplace_at
        // This avoids the race condition in ensure_size where another thread's
        // emplace might be constructing our slot while we're trying to assign
        matches_.emplace_at(mid, arena_, mid, rule_index, edges, num_edges, binding, origin_state);

        // Add to state's match list
        if (origin_state < state_matches_.size()) {
            state_matches_[origin_state].push(mid, arena_);
        }

        return mid;
    }

    // Get match by ID
    const Match& get_match(MatchId mid) const {
        return matches_[mid];
    }

    Match& get_match(MatchId mid) {
        return matches_[mid];
    }

    // Iterate over matches in a state
    template<typename Visitor>
    void for_each_match(StateId sid, Visitor&& visit) const {
        if (sid >= state_matches_.size()) return;
        state_matches_[sid].for_each([&](MatchId mid) {
            visit(mid);
        });
    }

    // Number of matches
    uint32_t num_matches() const {
        return counters_.next_match.load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Index Access
    // =========================================================================

    const SignatureIndex& signature_index() const {
        return match_index_.signature_index();
    }

    const InvertedVertexIndex& inverted_index() const {
        return match_index_.inverted_index();
    }

    const PatternMatchingIndex& match_index() const {
        return match_index_;
    }

    // =========================================================================
    // Causal Graph Access
    // =========================================================================

    CausalGraph& causal_graph() { return causal_graph_; }
    const CausalGraph& causal_graph() const { return causal_graph_; }

    // Set edge producer (called when edge is created by an event)
    void set_edge_producer(EdgeId edge, EventId producer) {
        causal_graph_.set_edge_producer(edge, producer);

        // Level 2: Also track in edge equivalence manager for cross-branch causal edges
        if (level2_enabled_) {
            edge_equiv_manager_.add_producer(edge, producer);
        }
    }

    // Add edge consumer (called when edge is consumed by an event)
    void add_edge_consumer(EdgeId edge, EventId consumer) {
        causal_graph_.add_edge_consumer(edge, consumer);

        // Level 2: Also track in edge equivalence manager for cross-branch causal edges
        if (level2_enabled_) {
            edge_equiv_manager_.add_consumer(edge, consumer);
        }
    }

    // Register event for branchial tracking
    void register_event_for_branchial(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed
    ) {
        causal_graph_.register_event_from_state_with_overlap_check(
            event, input_state, consumed_edges, num_consumed,
            [this](EventId eid, const EdgeId*& edges, uint8_t& num) {
                const Event& ev = events_[eid];
                edges = ev.consumed_edges;
                num = ev.num_consumed;
            }
        );
    }

    // Get causal/branchial statistics
    size_t num_causal_edges() const { return causal_graph_.num_causal_edges(); }
    size_t num_branchial_edges() const { return causal_graph_.num_branchial_edges(); }

    // =========================================================================
    // Arena Access
    // =========================================================================

    ConcurrentHeterogeneousArena& arena() { return arena_; }
    const ConcurrentHeterogeneousArena& arena() const { return arena_; }

    // =========================================================================
    // Counter Access
    // =========================================================================

    GlobalCounters& counters() { return counters_; }
    const GlobalCounters& counters() const { return counters_; }

    // =========================================================================
    // Utility
    // =========================================================================

    // Compute simple hash for a state's edge set (fast but not isomorphism-invariant)
    static uint64_t compute_state_hash(const SparseBitset& edges) {
        uint64_t h = 14695981039346656037ULL;
        edges.for_each([&](EdgeId eid) {
            h ^= eid;
            h *= 1099511628211ULL;
        });
        return h;
    }

    // Compute canonical hash using exact canonicalization (isomorphism-invariant)
    // Uses factorial-time algorithm for correctness, but fast for small graphs
    // If shared_tree is enabled, uses faster uniqueness tree hashing instead
    uint64_t compute_canonical_hash(const SparseBitset& edges) const {
        // Use shared uniqueness tree if enabled (faster with incremental computation)
        if (use_shared_tree_ && shared_tree_) {
            return compute_canonical_hash_shared(edges);
        }

        // Build edge vectors for canonicalizer (use std::size_t for v1 compatibility)
        std::vector<std::vector<std::size_t>> edge_vectors;

        // CRITICAL: Acquire fence to ensure we see all edge data written by other threads.
        // Pairs with release fence in create_edge after edge construction.
        std::atomic_thread_fence(std::memory_order_acquire);

        edges.for_each([&](EdgeId eid) {
            const Edge& e = edges_[eid];

            std::vector<std::size_t> verts;
            verts.reserve(e.arity);
            for (uint8_t i = 0; i < e.arity; ++i) {
                verts.push_back(static_cast<std::size_t>(e.vertices[i]));
            }
            edge_vectors.push_back(std::move(verts));
        });

        if (edge_vectors.empty()) {
            return 0;
        }

        // Use exact canonicalization (same as v1)
        hypergraph::Canonicalizer canonicalizer;
        auto result = canonicalizer.canonicalize_edges(edge_vectors);

        // FNV-style hash of canonical form
        uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
        constexpr uint64_t FNV_PRIME = 1099511628211ULL;

        for (const auto& edge : result.canonical_form.edges) {
            for (auto vertex : edge) {
                hash ^= static_cast<uint64_t>(vertex);
                hash *= FNV_PRIME;
            }
            // Add separator between edges
            hash ^= 0xDEADBEEF;
            hash *= FNV_PRIME;
        }

        return hash;
    }

    // Debug: Get canonical form as string
    std::string get_canonical_form_string(const SparseBitset& edges) const {
        std::vector<std::vector<std::size_t>> edge_vectors;
        edges.for_each([&](EdgeId eid) {
            const Edge& e = edges_[eid];
            std::vector<std::size_t> verts;
            verts.reserve(e.arity);
            for (uint8_t i = 0; i < e.arity; ++i) {
                verts.push_back(static_cast<std::size_t>(e.vertices[i]));
            }
            edge_vectors.push_back(std::move(verts));
        });

        if (edge_vectors.empty()) {
            return "{}";
        }

        hypergraph::Canonicalizer canonicalizer;
        auto result = canonicalizer.canonicalize_edges(edge_vectors);

        std::string s = "{";
        for (size_t i = 0; i < result.canonical_form.edges.size(); ++i) {
            if (i > 0) s += ", ";
            s += "{";
            for (size_t j = 0; j < result.canonical_form.edges[i].size(); ++j) {
                if (j > 0) s += ",";
                s += std::to_string(result.canonical_form.edges[i][j]);
            }
            s += "}";
        }
        s += "}";
        return s;
    }

    // Debug: Get raw edges as string
    std::string get_raw_edges_string(const SparseBitset& edges) const {
        std::string s = "{";
        bool first = true;
        edges.for_each([&](EdgeId eid) {
            if (!first) s += ", ";
            first = false;
            const Edge& e = edges_[eid];
            s += "{";
            for (uint8_t i = 0; i < e.arity; ++i) {
                if (i > 0) s += ",";
                s += std::to_string(e.vertices[i]);
            }
            s += "}";
        });
        s += "}";
        return s;
    }

    // Compute canonical hash using SharedUniquenessTree
    // Uses globally cached vertex tree data for incremental computation
    // This is faster than full canonicalization and correctly identifies isomorphism
    uint64_t compute_canonical_hash_shared(const SparseBitset& edges) const {
        if (!shared_tree_ || edges.empty()) {
            return 0;
        }

        // Build accessors for SharedUniquenessTree
        std::vector<EdgeId> edge_list;
        edges.for_each([&](EdgeId eid) {
            edge_list.push_back(eid);
        });

        uint32_t max_eid = 0;
        for (EdgeId eid : edge_list) {
            if (eid > max_eid) max_eid = eid;
        }

        // Create pointer arrays for the shared tree
        std::vector<const VertexId*> edge_verts(max_eid + 1, nullptr);
        std::vector<uint8_t> edge_arities(max_eid + 1, 0);

        // CRITICAL: Acquire fence to ensure we see all edge data written by other threads.
        // Pairs with release fence in create_edge after edge construction.
        std::atomic_thread_fence(std::memory_order_acquire);

        for (EdgeId eid : edge_list) {
            edge_verts[eid] = edges_[eid].vertices;
            edge_arities[eid] = edges_[eid].arity;
        }

        return shared_tree_->compute_state_hash(
            edges,
            edge_verts.data(),
            edge_arities.data()
        );
    }

    // Compute canonical hash using UniquenessTree (polynomial-time, approximate)
    // This is faster but may have rare false positives/negatives
    uint64_t compute_canonical_hash_wl(const SparseBitset& edges) {
        // Build edge arrays for UniquenessTree
        std::vector<EdgeId> edge_list;
        edges.for_each([&](EdgeId eid) {
            edge_list.push_back(eid);
        });

        if (edge_list.empty()) {
            return 0;
        }

        uint32_t max_eid = 0;
        for (EdgeId eid : edge_list) {
            if (eid > max_eid) max_eid = eid;
        }

        std::vector<const VertexId*> edge_verts(max_eid + 1, nullptr);
        std::vector<uint8_t> edge_arities(max_eid + 1, 0);

        for (EdgeId eid : edge_list) {
            edge_verts[eid] = edges_[eid].vertices;
            edge_arities[eid] = edges_[eid].arity;
        }

        StateCanonicalInfo info = UniquenessTree::compute(
            edges,
            edge_list.data(),
            static_cast<uint32_t>(edge_list.size()),
            edge_verts.data(),
            edge_arities.data(),
            arena_
        );

        return info.canonical_hash;
    }

    // Compute full canonical info (hash + vertex classes)
    StateCanonicalInfo compute_canonical_info(const SparseBitset& edges) {
        std::vector<EdgeId> edge_list;
        edges.for_each([&](EdgeId eid) {
            edge_list.push_back(eid);
        });

        if (edge_list.empty()) {
            return StateCanonicalInfo();
        }

        uint32_t max_eid = 0;
        for (EdgeId eid : edge_list) {
            if (eid > max_eid) max_eid = eid;
        }

        std::vector<const VertexId*> edge_verts(max_eid + 1, nullptr);
        std::vector<uint8_t> edge_arities(max_eid + 1, 0);

        for (EdgeId eid : edge_list) {
            edge_verts[eid] = edges_[eid].vertices;
            edge_arities[eid] = edges_[eid].arity;
        }

        return UniquenessTree::compute(
            edges,
            edge_list.data(),
            static_cast<uint32_t>(edge_list.size()),
            edge_verts.data(),
            edge_arities.data(),
            arena_
        );
    }

    // Count edges in a state
    uint32_t count_state_edges(StateId sid) const {
        uint32_t count = 0;
        states_[sid].edges.for_each([&](EdgeId) {
            count++;
        });
        return count;
    }
};

}  // namespace hypergraph::unified
