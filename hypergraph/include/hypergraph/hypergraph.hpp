#pragma once

#include <cstdint>
#include <cstring>
#include <atomic>
#include <vector>
#include <memory>
#include <unordered_map>

#include "types.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "index.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "causal_graph.hpp"
#include "wl_hash.hpp"
#include "concurrent_map.hpp"

// Shared types: CanonicalizationResult, CanonicalForm, VertexMapping
#include "canonical_types.hpp"

namespace hypergraph {

// =============================================================================
// DirectAdjacencyWithArity: wraps a raw adjacency map (vertex -> list of
// (edge, position)) to provide for_each_adjacent and for_each_occurrence
// interfaces expected by the WL hash computation.
// =============================================================================
template<typename AdjacencyMapT, typename ArityAccessor>
class DirectAdjacencyWithArity {
public:
    DirectAdjacencyWithArity(const AdjacencyMapT& adj, const ArityAccessor& arities)
        : adj_(adj), arities_(arities) {}

    template<typename Callback>
    void for_each_adjacent(VertexId v, Callback&& cb) const {
        auto it = adj_.find(v);
        if (it != adj_.end()) {
            for (const auto& [eid, pos] : it->second) {
                cb(eid, pos);
            }
        }
    }

    template<typename Callback>
    void for_each_occurrence(VertexId v, Callback&& cb) const {
        auto it = adj_.find(v);
        if (it != adj_.end()) {
            for (const auto& [eid, pos] : it->second) {
                EdgeOccurrence occ(eid, pos, arities_[eid]);
                cb(occ);
            }
        }
    }

private:
    const AdjacencyMapT& adj_;
    const ArityAccessor& arities_;
};

// =============================================================================
// Hypergraph
// =============================================================================
// Central storage for all hypergraph data in the multiway system.
//
// Key design principles:
// - All edges are stored once (shared storage)
// - States are SparseBitset views over the edge pool
// - Thread-safe allocation via atomic counters
// - Arena allocation for cache-friendly memory layout
// - Lock-free indices for concurrent pattern matching
//
// Thread safety:
// - Edge/state/event/match creation: Lock-free via atomic counters
// - Index updates: Lock-free via ConcurrentMap and LockFreeList
// - Reading: Always safe (immutable after creation)

class Hypergraph {
    // Global ID counters (thread-safe)
    GlobalCounters counters_;

    // Arena for all allocations (thread-safe for parallel evolution)
    ConcurrentHeterogeneousArena arena_;

    // Edge storage
    SegmentedArray<Edge> edges_;

    // Cached edge signatures (computed once at edge creation, immutable)
    SegmentedArray<EdgeSignature> edge_signatures_;

    // State storage
    SegmentedArray<State> states_;

    // Event storage
    SegmentedArray<Event> events_;

    // Pattern matching indices
    PatternMatchingIndex match_index_;

    // Per-state child tracking for match cascading
    SegmentedArray<LockFreeList<StateId>> state_children_;

    // Causal and branchial graph
    CausalGraph causal_graph_;

    // Canonical state deduplication map: canonical_hash -> StateId
    // Used to find existing equivalent states before creating new ones
    ConcurrentMap<uint64_t, StateId> canonical_state_map_;

    // Event canonicalization state map: always keyed by isomorphism-invariant hash
    // Unlike canonical_state_map_ (keyed differently based on state_canonicalization_mode_),
    // this map is ALWAYS keyed by canonical_hash (WL/IR) regardless of state mode.
    // Used by event signature computation to find canonical representatives for
    // edge correspondence when state_canonicalization_mode_ is None or Automatic.
    ConcurrentMap<uint64_t, StateId> event_canonical_state_map_;

    // State canonicalization mode: controls how states are deduplicated
    // None: tree mode - no deduplication, each state is unique
    // Automatic: content-ordered hash (not yet implemented, behaves like Full)
    // Full: isomorphism-invariant hash via WL approximate or IR exact
    // NOTE: Must be atomic for ARM64 memory ordering - ensures visibility to worker threads
    std::atomic<StateCanonicalizationMode> state_canonicalization_mode_{StateCanonicalizationMode::None};

    // ==========================================================================
    // Global Vertex Adjacency Index
    // ==========================================================================
    // Maps each vertex to the list of edges it appears in, with position info.
    // This is THE canonical adjacency structure for the entire hypergraph.
    //
    // Thread safety: Append-only via LockFreeList. Edges are immutable after
    // creation, so once registered here, entries never change or get removed.
    //
    // Usage: When computing state hashes, iterate this and filter by
    // state_edges.contains(edge_id) to get per-state adjacency without copying.
    SegmentedArray<LockFreeList<EdgeOccurrence>> vertex_adjacency_;

    // Weisfeiler-Leman hash implementation (fast approximate state hash)
    std::unique_ptr<WLHash> wl_hash_;

    // Selects the algorithm for compute_canonical_hash:
    //   true  -> WL approximate hash (fast hot path)
    //   false -> IR exact canonicalization (isomorphism-invariant)
    bool use_shared_tree_{true};


    // Event canonicalization: maps event signature to first EventId
    // Signature computed from keys specified by event_signature_keys_ bitflag
    ConcurrentMap<uint64_t, EventId> canonical_event_map_;
    std::atomic<uint32_t> canonical_event_count_{0};
    EventSignatureKeys event_signature_keys_{EVENT_SIG_NONE};

    // Genesis state: the empty state (no edges) from which all initial states originate
    // Created lazily on first call to get_or_create_genesis_state()
    // Uses lock-free initialization: 0=uninit, 1=in_progress, 2=done
    StateId genesis_state_{INVALID_ID};
    std::atomic<int> genesis_state_init_{0};

public:
    Hypergraph()
        : wl_hash_(std::make_unique<WLHash>(&arena_))
    {
        causal_graph_.set_arena(&arena_);
    }

    // Non-copyable
    Hypergraph(const Hypergraph&) = delete;
    Hypergraph& operator=(const Hypergraph&) = delete;

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
    );

    // Create edge from initializer list (convenience)
    EdgeId create_edge(std::initializer_list<VertexId> vertices,
                       EventId creator_event = INVALID_ID,
                       uint32_t step = 0);

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
    // Edge Accessors for the WL hash
    // =========================================================================
    // These provide the interface needed by WLHash::compute_state_hash*()

    // Get vertex array for an edge (returns pointer to vertices)
    const VertexId* edge_vertices(EdgeId eid) const {
        return edges_[eid].vertices;
    }

    // Get arity of an edge
    uint8_t edge_arity(EdgeId eid) const {
        return edges_[eid].arity;
    }

    // Get cached signature for an edge (computed once at creation)
    const EdgeSignature& edge_signature(EdgeId eid) const {
        return edge_signatures_[eid];
    }

    // Lightweight accessor for the WL hash that provides pointer indexing
    // Returns pointer to edge's inline vertex array - no copying or allocation
    class EdgeVertexAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeVertexAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

        const VertexId* operator[](EdgeId eid) const {
            return hg_->edges_[eid].vertices;
        }
    };

    // Direct arity accessor - reads from struct field, O(1)
    class EdgeArityAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeArityAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

        uint8_t operator[](EdgeId eid) const {
            return hg_->edges_[eid].arity;
        }
    };

    EdgeVertexAccessorRaw edge_vertex_accessor_raw() const {
        return EdgeVertexAccessorRaw(this);
    }

    EdgeArityAccessorRaw edge_arity_accessor_raw() const {
        return EdgeArityAccessorRaw(this);
    }

    // =========================================================================
    // Global Vertex Adjacency Access
    // =========================================================================
    // Provides access to the global vertex-to-edge adjacency index.
    // Use for state hash computation: iterate occurrences and filter by
    // state_edges.contains(edge_id) to get per-state adjacency without copying.

    // Get the edge occurrences for a vertex (may be empty if vertex not seen)
    const LockFreeList<EdgeOccurrence>* get_vertex_adjacency(VertexId v) const {
        if (v >= vertex_adjacency_.size()) {
            return nullptr;
        }
        return &vertex_adjacency_[v];
    }

    // Iterate over edges containing vertex v that are also in state_edges
    // This is the key operation for the WL hash computation without copying adjacency
    template<typename F>
    void for_each_adjacent_edge_in_state(
        VertexId v,
        const SparseBitset& state_edges,
        F&& f
    ) const {
        if (v >= vertex_adjacency_.size()) return;

        vertex_adjacency_[v].for_each([&](const EdgeOccurrence& occ) {
            if (state_edges.contains(occ.edge_id)) {
                f(occ);
            }
        });
    }

    // Iterate over ALL edge occurrences for a vertex (unfiltered)
    // Used by hash strategies with their own filtering
    template<typename F>
    void for_each_occurrence(VertexId v, F&& f) const {
        if (v >= vertex_adjacency_.size()) return;
        vertex_adjacency_[v].for_each(std::forward<F>(f));
    }

    // Adjacency provider wrapper for use with hash strategies
    // This wraps the global adjacency index into an object that can be passed
    // to the incremental WL hash's adjacency queries
    class GlobalAdjacencyProvider {
        const Hypergraph* hg_;
    public:
        explicit GlobalAdjacencyProvider(const Hypergraph* hg) : hg_(hg) {}

        template<typename F>
        void for_each_occurrence(VertexId v, F&& f) const {
            hg_->for_each_occurrence(v, std::forward<F>(f));
        }
    };

    GlobalAdjacencyProvider adjacency_provider() const {
        return GlobalAdjacencyProvider(this);
    }

    // State-filtered adjacency provider: uses global index but filters by state's edge set
    // This is the key optimization: O(vertex_degree) iteration with O(1) contains check
    // instead of O(state_edges) adjacency rebuild
    class StateFilteredAdjacencyProvider {
        const SegmentedArray<LockFreeList<EdgeOccurrence>>* vertex_adjacency_;
        const SparseBitset* state_edges_;
    public:
        StateFilteredAdjacencyProvider(
            const SegmentedArray<LockFreeList<EdgeOccurrence>>* vertex_adjacency,
            const SparseBitset* state_edges)
            : vertex_adjacency_(vertex_adjacency), state_edges_(state_edges) {}

        template<typename Callback>
        void for_each_adjacent(VertexId v, Callback&& cb) const {
            if (v >= vertex_adjacency_->size()) return;
            (*vertex_adjacency_)[v].for_each([&](const EdgeOccurrence& occ) {
                if (state_edges_->contains(occ.edge_id)) {
                    cb(occ.edge_id, occ.position);
                }
            });
        }

        // For compute_tree_hash_external_adjacency which expects EdgeOccurrence
        template<typename Callback>
        void for_each_occurrence(VertexId v, Callback&& cb) const {
            if (v >= vertex_adjacency_->size()) return;
            (*vertex_adjacency_)[v].for_each([&](const EdgeOccurrence& occ) {
                if (state_edges_->contains(occ.edge_id)) {
                    cb(occ);
                }
            });
        }
    };

    StateFilteredAdjacencyProvider state_filtered_adjacency(const SparseBitset& state_edges) const {
        return StateFilteredAdjacencyProvider(&vertex_adjacency_, &state_edges);
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
    );

    // Create state from edge IDs (convenience)
    StateId create_state(
        const EdgeId* edge_ids,
        uint32_t num_edges,
        uint32_t step = 0,
        uint64_t canonical_hash = 0,
        EventId parent_event = INVALID_ID
    );

    // Create state from initializer list (convenience)
    StateId create_state(std::initializer_list<EdgeId> edge_ids,
                         uint32_t step = 0,
                         uint64_t canonical_hash = 0,
                         EventId parent_event = INVALID_ID);

    // Get state by ID
    const State& get_state(StateId sid) const {
        // CRITICAL: Acquire fence to ensure we see all state data written by
        // the thread that created this state. Pairs with release fence in create_state.
        std::atomic_thread_fence(std::memory_order_acquire);
        return states_[sid];
    }

    State& get_state(StateId sid) {
        std::atomic_thread_fence(std::memory_order_acquire);
        return states_[sid];
    }

    // Get state's edge set
    const SparseBitset& get_state_edges(StateId sid) const {
        // CRITICAL: Acquire fence to ensure we see all state data written by
        // the thread that created this state. Pairs with release fence in create_state.
        std::atomic_thread_fence(std::memory_order_acquire);
        return states_[sid].edges;
    }

    // Get content-ordered hash for a state (for Automatic state canonicalization)
    // This is the same hash function used during evolution for state deduplication
    // in Automatic mode, ensuring consistency between evolution and display.
    uint64_t get_state_content_hash(StateId sid) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return compute_content_ordered_hash(states_[sid].edges);
    }

    // Number of states
    uint32_t num_states() const {
        return counters_.next_state.load(std::memory_order_relaxed);
    }

    // Get the genesis state ID (creates it lazily if needed)
    // The genesis state is an empty state (no edges) that serves as the origin
    // for all initial states via genesis events.
    StateId get_or_create_genesis_state();

    // Check if a state is the genesis state
    bool is_genesis_state(StateId sid) const {
        return genesis_state_init_.load(std::memory_order_acquire) == 2 && sid == genesis_state_;
    }

    // Check if an event is a genesis event (connects from genesis state to initial state)
    bool is_genesis_event(EventId eid) const {
        if (genesis_state_init_.load(std::memory_order_acquire) != 2) {
            return false;
        }
        if (eid >= events_.size()) {
            return false;
        }
        const Event& event = events_[eid];
        return event.input_state == genesis_state_;
    }

    // Get genesis state ID (returns INVALID_ID if not created)
    StateId genesis_state() const {
        if (genesis_state_init_.load(std::memory_order_acquire) == 2) {
            return genesis_state_;
        }
        return INVALID_ID;
    }

    // =========================================================================
    // Canonical State Deduplication
    // =========================================================================

    // Result of trying to create a canonical state
    struct CanonicalStateResult {
        StateId canonical_state_id;  // The canonical state ID (existing or new)
        StateId created_state_id;    // The state ID we created (always new, with actual edges)
        bool was_new;                // true if new canonical state, false if existing found
    };

    // Create state if no equivalent exists, otherwise return existing
    // This is the main API for state creation with canonicalization.
    // If Level 2 is enabled and a duplicate is found, edge correspondence is computed.
    //
    // Thread safety: Fully linearizable. We create the state first, then try to
    // insert into the canonical map. If another thread wins, we return their state
    // (the created state becomes "wasted" but this is correct).
    // canonical_hash is computed internally (mode-aware): the exact IR hash in Full
    // mode (reused as both identity and dedup key), the fast WL hash otherwise.
    // The optional incr_* delta (parent state + consumed/produced edges) lets the WL
    // hash be computed incrementally from the parent's cached history when
    // incremental WL is enabled; it is bit-identical, so dedup is unaffected.
    CanonicalStateResult create_or_get_canonical_state(
        SparseBitset&& edge_set,
        uint32_t step = 0,
        EventId parent_event = INVALID_ID,
        StateId incr_parent = INVALID_ID,
        const EdgeId* incr_consumed = nullptr, uint8_t incr_num_consumed = 0,
        const EdgeId* incr_produced = nullptr, uint8_t incr_num_produced = 0
    );


    // Lookup existing canonical state by hash (waits for concurrent inserts)
    std::optional<StateId> find_canonical_state(uint64_t canonical_hash) const {
        return canonical_state_map_.lookup_waiting(canonical_hash);
    }

    // Get the canonical representative for a given state
    // Behavior depends on state_canonicalization_mode_:
    // - None: returns raw_state (no canonicalization)
    // - Automatic/Full: returns cached canonical_id (may differ from raw_state)
    // NOTE: Uses acquire fence to ensure visibility of canonical_id on ARM64
    StateId get_canonical_state(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;
        if (state_canonicalization_mode_.load(std::memory_order_acquire) == StateCanonicalizationMode::None) {
            return raw_state;
        }
        // Acquire fence ensures we see the canonical_id write from create_or_get_canonical_state
        // This is critical for ARM64's weak memory model
        std::atomic_thread_fence(std::memory_order_acquire);
        const State& state = get_state(raw_state);
        return state.canonical_id;
    }

    // Get the canonical state for event canonicalization purposes.
    // Always uses the isomorphism-invariant hash (WL/IR) to find the canonical
    // representative, regardless of state_canonicalization_mode_.
    // This is needed for computing edge correspondence when state mode is None.
    StateId get_canonical_state_for_event(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;

        // Get the isomorphism-invariant hash for this state
        const State& state = get_state(raw_state);
        uint64_t hash = state.canonical_hash;

        // If hash is 0, the state's hash wasn't computed - fall back to raw state
        if (hash == 0) return raw_state;

        // Lookup in event_canonical_state_map_ which is always keyed by canonical_hash
        auto result = event_canonical_state_map_.lookup_waiting(hash);
        return result.value_or(raw_state);
    }

    // Get the canonical hash for a state (compute on-demand if not available)
    // This is used for event canonicalization, which needs isomorphism-invariant
    // state hashes regardless of whether state_canonicalization_mode_ is None.
    uint64_t get_or_compute_canonical_hash(StateId state_id);

    // Quotient exploration support. try_lower_explore_depth records a shorter path to a
    // canonical state, returning true only when it improved on what was known. Depth is a
    // shortest-path label, a property of the graph, so the set of states reachable within
    // the step budget does not depend on the order paths are found. try_claim_expanded
    // succeeds exactly once per canonical state, so its matches are computed once and the
    // matches-per-instance it records are well defined.
    bool try_lower_explore_depth(StateId canonical_id, uint32_t depth);
    bool try_claim_expanded(StateId canonical_id);

    // Number of unique canonical states
    // Uses count_unique() for accurate counting after evolution completes,
    // handling the case where ConcurrentMap may have duplicate keys due to
    // concurrent insertions of the same canonical hash.
    size_t num_canonical_states() const {
        return canonical_state_map_.count_unique();
    }

    // =========================================================================
    // State Canonicalization Configuration
    // =========================================================================

    // State canonicalization mode: controls state deduplication strategy
    // Uses release semantics to ensure visibility to worker threads on ARM64
    void set_state_canonicalization_mode(StateCanonicalizationMode mode) {
        state_canonicalization_mode_.store(mode, std::memory_order_release);
    }

    // Uses acquire semantics to see updates from main thread on ARM64
    StateCanonicalizationMode state_canonicalization_mode() const {
        return state_canonicalization_mode_.load(std::memory_order_acquire);
    }

    // Select the WL approximate hash for compute_canonical_hash (fast hot path)
    void enable_shared_tree() {
        use_shared_tree_ = true;
    }

    // Select IR exact canonicalization for compute_canonical_hash
    void disable_shared_tree() {
        use_shared_tree_ = false;
    }

    // Whether compute_canonical_hash uses the WL approximate hash
    bool shared_tree_enabled() const {
        return use_shared_tree_;
    }

    // Full canonicalization mode: IR-based exact dedup, edge correspondence, and canonical output
    bool is_full_canonicalization() const {
        return state_canonicalization_mode_.load(std::memory_order_acquire) == StateCanonicalizationMode::Full;
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

    // Create a new event with optional canonicalization
    // Returns: (event_id, canonical_event_id, is_canonical)
    // - event_id: the ID of the created event
    // - canonical_event_id: for duplicate events, points to the first event with same signature
    // - is_canonical: true if this is a new canonical event, false if duplicate
    struct CreateEventResult {
        EventId event_id;
        EventId canonical_event_id;  // Same as event_id if is_canonical, otherwise first event
        bool is_canonical;
    };

    CreateEventResult create_event(
        StateId input_state,
        StateId output_state,
        RuleIndex rule_index,
        const EdgeId* consumed,
        uint8_t num_consumed,
        const EdgeId* produced,
        uint8_t num_produced,
        const VariableBinding& binding
    );

    // Get event by ID
    const Event& get_event(EventId eid) const {
        return events_[eid];
    }

    Event& get_event(EventId eid) {
        return events_[eid];
    }

    // Number of events (returns canonical count when canonicalization enabled)
    uint32_t num_events() const {
        if (event_signature_keys_ != EVENT_SIG_NONE) {
            return canonical_event_count_.load(std::memory_order_acquire);
        }
        // Use acquire to synchronize with release stores in alloc_event
        return counters_.next_event.load(std::memory_order_acquire);
    }

    // Number of raw events (always returns total count)
    uint32_t num_raw_events() const {
        return counters_.next_event.load(std::memory_order_acquire);
    }

    // Iterate over canonical events only (skips duplicates)
    // Callback signature: void(EventId eid, const Event& event)
    template<typename Callback>
    void for_each_canonical_event(Callback&& callback) const {
        uint32_t count = num_raw_events();
        for (uint32_t eid = 0; eid < count; ++eid) {
            const Event& event = events_[eid];
            if (event.id == INVALID_ID) continue;
            if (!event.is_canonical()) continue;
            callback(eid, event);
        }
    }

    // Check if an event is canonical (not a duplicate)
    bool is_event_canonical(EventId eid) const {
        if (eid >= num_raw_events()) return false;
        return events_[eid].is_canonical();
    }

    // Get the canonical event ID for a raw event ID
    EventId get_canonical_event(EventId eid) const {
        if (eid >= num_raw_events()) return INVALID_ID;
        const Event& event = events_[eid];
        return event.is_canonical() ? eid : event.canonical_event_id;
    }

    // Event signature keys (bitflag controlling event equivalence)
    void set_event_signature_keys(EventSignatureKeys keys) {
        event_signature_keys_ = keys;
    }

    EventSignatureKeys event_signature_keys() const {
        return event_signature_keys_;
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
    }

    // Get edge producer (returns INVALID_ID if edge has no producer yet)
    EventId get_edge_producer(EdgeId edge) const {
        return causal_graph_.get_edge_producer(edge);
    }

    // Add edge consumer (called when edge is consumed by an event)
    void add_edge_consumer(EdgeId edge, EventId consumer) {
        causal_graph_.add_edge_consumer(edge, consumer);
    }

    // Create a genesis event for an initial state.
    // This synthetic event connects the empty genesis state to the initial state.
    // It "produces" all edges in the initial state, enabling causal tracking from gen 0.
    // Returns the genesis event ID.
    EventId create_genesis_event(StateId initial_state, const EdgeId* edges, uint8_t num_edges);

    // Register event for branchial tracking
    // When event canonicalization is enabled, uses edge equivalence for overlap detection
    // and skips branchial edges between canonically equivalent events
    void register_event_for_branchial(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed,
        EventId canonical_event = INVALID_ID  // Pass canonical_event_id for deduplication
    );

    // Get causal/branchial statistics
    size_t num_causal_edges() const { return causal_graph_.num_causal_edges(); }
    size_t num_causal_event_pairs() const { return causal_graph_.num_causal_event_pairs(); }
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

    // Compute content-ordered hash for Automatic state canonicalization mode
    // Hashes edge contents in order by edge ID: (arity, v1, v2, ...) for each edge
    // Fast but not isomorphism-invariant.
    uint64_t compute_content_ordered_hash(const SparseBitset& edges) const;

    // Compute canonical hash (isomorphism-invariant).
    // With the WL hash enabled (use_shared_tree_), uses the fast approximate hash;
    // otherwise falls back to IR exact canonicalization.
    uint64_t compute_canonical_hash(const SparseBitset& edges) const;

    // Compute the Weisfeiler-Leman approximate canonical hash for a set of
    // edges. This is the fast hot-path hash backing compute_canonical_hash (in
    // WL mode), the per-state canonical_hash recorded during evolution, and the
    // isomorphism-invariant key for event canonicalization.
    uint64_t compute_wl_hash(const SparseBitset& edges) const;


    // Find edge correspondence between two isomorphic states. Uses IR in Full
    // canonicalization mode, WL subtree hashes otherwise.
    // Returns mapping from state1 edges to state2 edges.
    EdgeCorrespondence find_edge_correspondence_dispatch(
        const SparseBitset& state1_edges,
        const SparseBitset& state2_edges
    ) const;

    // Count edges in a state
    uint32_t count_state_edges(StateId sid) const {
        uint32_t count = 0;
        states_[sid].edges.for_each([&](EdgeId) {
            count++;
        });
        return count;
    }
};

}  // namespace hypergraph
