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
#include "uniqueness_tree.hpp"
#include "incremental_uniqueness_tree.hpp"
#include "wl_hash.hpp"
#include "concurrent_map.hpp"

// Include v1 canonicalizer for exact isomorphism checking
#include "canonicalization.hpp"

namespace hypergraph {

// =============================================================================
// HashStrategy: Controls which hashing algorithm to use
// =============================================================================

enum class HashStrategy {
    UniquenessTree,            // True Gorard-style uniqueness trees (BFS tree structure)
    IncrementalUniquenessTree, // Incremental version that caches subtree hashes
    WL,                        // Weisfeiler-Lehman style iterative refinement
    IncrementalWL              // Incremental WL that reuses parent vertex hashes
};

// =============================================================================
// DirectAdjacency: Simple wrapper around adjacency map for use with tree hash
// =============================================================================
//
// Wraps a raw adjacency map (vertex -> list of (edge, position)) to provide
// the for_each_adjacent interface expected by compute_tree_hash_with_adjacency_provider.
// Also provides for_each_occurrence for compute_tree_hash_external_adjacency.
//
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

    // For compute_tree_hash_external_adjacency which expects EdgeOccurrence
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

// Simple version without arity (for backward compatibility)
template<typename AdjacencyMapT>
class DirectAdjacency {
public:
    explicit DirectAdjacency(const AdjacencyMapT& adj) : adj_(adj) {}

    template<typename Callback>
    void for_each_adjacent(VertexId v, Callback&& cb) const {
        auto it = adj_.find(v);
        if (it != adj_.end()) {
            for (const auto& [eid, pos] : it->second) {
                cb(eid, pos);
            }
        }
    }

private:
    const AdjacencyMapT& adj_;
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

    // Match storage
    SegmentedArray<Match> matches_;

    // Pattern matching indices
    PatternMatchingIndex match_index_;

    // Per-state child tracking for match cascading
    SegmentedArray<LockFreeList<StateId>> state_children_;

    // Per-state match lists
    SegmentedArray<LockFreeList<MatchId>> state_matches_;

    // Per-state incremental cache for IncrementalUniquenessTree strategy
    SegmentedArray<StateIncrementalCache> state_incremental_cache_;

    // Per-state WL hash cache (atomic pointer to VertexHashCache)
    // Used to memoize compute_state_hash_with_cache for canonical states
    // Uses atomic pointer for thread-safe initialization without torn writes
    struct WLHashCacheEntry {
        std::atomic<VertexHashCache*> cache_ptr{nullptr};
    };
    SegmentedArray<WLHashCacheEntry> wl_hash_cache_;

    // Causal and branchial graph
    CausalGraph causal_graph_;

    // Canonical state deduplication map: canonical_hash -> StateId
    // Used to find existing equivalent states before creating new ones
    ConcurrentMap<uint64_t, StateId> canonical_state_map_;

    // Event canonicalization state map: always keyed by isomorphism-invariant hash
    // Unlike canonical_state_map_ (keyed differently based on state_canonicalization_mode_),
    // this map is ALWAYS keyed by canonical_hash (WL/UT) regardless of state mode.
    // Used by event signature computation to find canonical representatives for
    // edge correspondence when state_canonicalization_mode_ is None or Automatic.
    ConcurrentMap<uint64_t, StateId> event_canonical_state_map_;

    // State canonicalization mode: controls how states are deduplicated
    // None: tree mode - no deduplication, each state is unique
    // Automatic: content-ordered hash (not yet implemented, behaves like Full)
    // Full: isomorphism-invariant hash via WL/UT
    StateCanonicalizationMode state_canonicalization_mode_{StateCanonicalizationMode::None};

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

    // Hash strategy implementations
    std::unique_ptr<UniquenessTree> unified_tree_;                 // Gorard-style uniqueness trees
    std::unique_ptr<IncrementalUniquenessTree> incremental_tree_;  // Incremental version
    std::unique_ptr<WLHash> wl_hash_;                                     // Weisfeiler-Lehman hashing
    HashStrategy hash_strategy_{HashStrategy::WL}; // UT-Inc with cached adjacency

    // Stats for bloom filter-based vertex hash reuse in compute_canonical_hash_incremental
    mutable std::atomic<size_t> bloom_reused_{0};
    mutable std::atomic<size_t> bloom_recomputed_{0};

    // Flag to use shared tree vs exact canonicalization
    // Enabled by default: WL hashing is O(V²×E) vs O(g!) factorial for exact canonicalization
    bool use_shared_tree_{true};

    // Event canonicalization: maps event signature to first EventId
    // Signature computed from keys specified by event_signature_keys_ bitflag
    ConcurrentMap<uint64_t, EventId> canonical_event_map_;
    std::atomic<uint32_t> canonical_event_count_{0};
    EventSignatureKeys event_signature_keys_{EVENT_SIG_NONE};

    // Abort flag for long-running hash computations (set by evolution engine)
    std::atomic<bool>* abort_flag_{nullptr};

    // Genesis state: the empty state (no edges) from which all initial states originate
    // Created lazily on first call to get_or_create_genesis_state()
    StateId genesis_state_{INVALID_ID};
    std::atomic<bool> genesis_state_created_{false};
    std::mutex genesis_state_mutex_;

public:
    Hypergraph()
        : unified_tree_(std::make_unique<UniquenessTree>(&arena_))
        , incremental_tree_(std::make_unique<IncrementalUniquenessTree>(&arena_))
        , wl_hash_(std::make_unique<WLHash>(&arena_))
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

        // Compute and cache edge signature (immutable after creation)
        edge_signatures_.emplace_at(eid, arena_, EdgeSignature::from_edge(vertices, arity));

        // Update indices
        match_index_.add_edge(eid, vertices, arity, arena_);

        // Register in global vertex adjacency index
        // This is the canonical adjacency structure used by all hash strategies
        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = vertices[i];
            EdgeOccurrence occ(eid, i, arity);
            vertex_adjacency_.get_or_default(v, arena_).push(occ, arena_);
        }

        // Register with hash implementations (for any additional per-strategy state)
        if (unified_tree_) {
            unified_tree_->register_edge(eid, vertices, arity);
        }
        if (incremental_tree_) {
            incremental_tree_->register_edge(eid, vertices, arity);
        }
        if (wl_hash_) {
            wl_hash_->register_edge(eid, vertices, arity);
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

    // Get cached signature for an edge (computed once at creation)
    const EdgeSignature& edge_signature(EdgeId eid) const {
        return edge_signatures_[eid];
    }

    // Helper class to provide indexed access to edge vertices
    class EdgeVertexAccessor {
        const Hypergraph* hg_;
    public:
        explicit EdgeVertexAccessor(const Hypergraph* hg) : hg_(hg) {}
        const VertexId* operator[](EdgeId eid) const {
            return hg_->edge_vertices(eid);
        }
    };

    // Helper class to provide indexed access to edge arities
    class EdgeArityAccessor {
        const Hypergraph* hg_;
    public:
        explicit EdgeArityAccessor(const Hypergraph* hg) : hg_(hg) {}
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
    // Optimized accessor: direct access without caching overhead
    class EdgeVertexAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeVertexAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

        const VertexId* operator[](EdgeId eid) const {
            return hg_->edges_[eid].vertices;  // Direct access
        }
    };

    // Optimized accessor: direct access without caching overhead
    // edge_arity() is already O(1), caching added more overhead than it saved
    class EdgeArityAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeArityAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

        uint8_t operator[](EdgeId eid) const {
            return hg_->edges_[eid].arity;  // Direct access, no function call
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
    // This is the key operation for tree hash computation without copying adjacency
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
    // to IncrementalUniquenessTree's external adjacency methods
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
    // Incremental Vertex Collection
    // =========================================================================
    // Computes child vertices from parent in O(delta) instead of O(E).
    // child_vertices = parent_vertices - orphaned + new_from_produced
    //
    // orphaned = vertices in consumed edges that don't appear in any child edge
    // new = vertices in produced edges that weren't in parent

    template<typename EdgeAccessor, typename ArityAccessor>
    ArenaVector<VertexId> compute_child_vertices_incremental(
        const VertexHashCache& parent_cache,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const SparseBitset& child_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        // Build set of parent vertices for fast lookup
        std::unordered_set<VertexId> parent_vertex_set;
        parent_vertex_set.reserve(parent_cache.count);
        for (uint32_t i = 0; i < parent_cache.count; ++i) {
            parent_vertex_set.insert(parent_cache.vertices[i]);
        }

        // Collect vertices from consumed edges that might be orphaned
        std::unordered_set<VertexId> maybe_orphaned;
        for (uint8_t i = 0; i < num_consumed; ++i) {
            EdgeId eid = consumed_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                maybe_orphaned.insert(verts[j]);
            }
        }

        // Check which are actually orphaned (no edges in child_edges)
        std::unordered_set<VertexId> orphaned;
        for (VertexId v : maybe_orphaned) {
            bool has_edge_in_child = false;
            // Use global adjacency to check vertex's edges
            if (v < vertex_adjacency_.size()) {
                vertex_adjacency_[v].for_each([&](const EdgeOccurrence& occ) {
                    if (!has_edge_in_child && child_edges.contains(occ.edge_id)) {
                        has_edge_in_child = true;
                    }
                });
            }
            if (!has_edge_in_child) {
                orphaned.insert(v);
            }
        }

        // Collect new vertices from produced edges
        std::unordered_set<VertexId> new_vertices;
        for (uint8_t i = 0; i < num_produced; ++i) {
            EdgeId eid = produced_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                VertexId v = verts[j];
                if (parent_vertex_set.find(v) == parent_vertex_set.end()) {
                    new_vertices.insert(v);
                }
            }
        }

        // Build result: parent - orphaned + new
        ArenaVector<VertexId> result(arena_);
        result.reserve(parent_cache.count - orphaned.size() + new_vertices.size());

        // Add parent vertices that aren't orphaned
        for (uint32_t i = 0; i < parent_cache.count; ++i) {
            VertexId v = parent_cache.vertices[i];
            if (orphaned.find(v) == orphaned.end()) {
                result.push_back(v);
            }
        }

        // Add new vertices
        for (VertexId v : new_vertices) {
            result.push_back(v);
        }

        // Sort for consistency
        std::sort(result.begin(), result.end());

        return result;
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
    StateId get_or_create_genesis_state() {
        // Fast path: already created
        if (genesis_state_created_.load(std::memory_order_acquire)) {
            return genesis_state_;
        }

        // Slow path: create under lock
        std::lock_guard<std::mutex> lock(genesis_state_mutex_);

        // Double-check after acquiring lock
        if (genesis_state_created_.load(std::memory_order_relaxed)) {
            return genesis_state_;
        }

        // Create empty state (no edges, step 0, hash 0)
        SparseBitset empty_edges;
        genesis_state_ = create_state(std::move(empty_edges), 0, 0, INVALID_ID);

        genesis_state_created_.store(true, std::memory_order_release);
        return genesis_state_;
    }

    // Check if a state is the genesis state
    bool is_genesis_state(StateId sid) const {
        return genesis_state_created_.load(std::memory_order_acquire) && sid == genesis_state_;
    }

    // Check if an event is a genesis event (connects from genesis state to initial state)
    bool is_genesis_event(EventId eid) const {
        if (!genesis_state_created_.load(std::memory_order_acquire)) {
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
        if (genesis_state_created_.load(std::memory_order_acquire)) {
            return genesis_state_;
        }
        return INVALID_ID;
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

        // Determine the key for canonical map based on mode:
        // - None: use state ID as key (each state is unique, no deduplication)
        // - Automatic: use content-ordered hash (fast but not isomorphism-invariant)
        // - Full: use canonical_hash from WL/UT (isomorphism-invariant)
        uint64_t map_key;
        switch (state_canonicalization_mode_) {
            case StateCanonicalizationMode::None:
                map_key = static_cast<uint64_t>(new_sid);
                break;
            case StateCanonicalizationMode::Automatic:
                // Content-ordered hash: hash edges in order by edge ID
                map_key = compute_content_ordered_hash(get_state(new_sid).edges);
                break;
            case StateCanonicalizationMode::Full:
            default:
                map_key = canonical_hash;
                break;
        }

        // Try to insert into canonical map (lock-free, waiting for LOCKED slots)
        // Must use waiting version to handle concurrent inserts during resize
        auto [existing_or_new, was_inserted] = canonical_state_map_.insert_if_absent_waiting(map_key, new_sid);

        // Also insert into event_canonical_state_map_ using the isomorphism-invariant hash.
        // This is always keyed by canonical_hash regardless of state_canonicalization_mode_,
        // ensuring event canonicalization can find canonical representatives for edge
        // correspondence computation even when state mode is None or Automatic.
        event_canonical_state_map_.insert_if_absent_waiting(canonical_hash, new_sid);

        // Cache the canonical ID in the state for fast lookup
        // This avoids race conditions and map lookups in get_canonical_state
        states_[new_sid].canonical_id = existing_or_new;

        if (!was_inserted) {
            // Another thread beat us - they have the canonical representative
            // Our new_sid is the raw state with actual edges, existing_or_new is the canonical
            return {existing_or_new, new_sid, false};
        }

        // We won the race - our state is the canonical representative
        // raw_state == state in this case
        return {new_sid, new_sid, true};
    }

    // Lookup existing canonical state by hash (waits for concurrent inserts)
    std::optional<StateId> find_canonical_state(uint64_t canonical_hash) const {
        return canonical_state_map_.lookup_waiting(canonical_hash);
    }

    // Get the canonical representative for a given state
    // Behavior depends on state_canonicalization_mode_:
    // - None: returns raw_state (no canonicalization)
    // - Automatic/Full: returns cached canonical_id (may differ from raw_state)
    StateId get_canonical_state(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;
        if (state_canonicalization_mode_ == StateCanonicalizationMode::None) {
            return raw_state;
        }
        const State& state = get_state(raw_state);
        return state.canonical_id;
    }

    // Get the canonical state for event canonicalization purposes.
    // Always uses the isomorphism-invariant hash (WL/UT) to find the canonical
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
    uint64_t get_or_compute_canonical_hash(StateId state_id) {
        if (state_id == INVALID_ID) return 0;

        State& state = states_[state_id];

        // If hash is already computed, return it
        if (state.canonical_hash != 0) {
            return state.canonical_hash;
        }

        // Compute hash on-demand using hash dispatch
        auto [hash, cache] = compute_hash_with_cache_dispatch(state.edges);

        // Cache the hash for future use (not thread-safe, but hash is idempotent)
        state.canonical_hash = hash;
        return hash;
    }

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
    void set_state_canonicalization_mode(StateCanonicalizationMode mode) {
        state_canonicalization_mode_ = mode;
    }

    StateCanonicalizationMode state_canonicalization_mode() const {
        return state_canonicalization_mode_;
    }

    // Legacy setter for backward compatibility
    void set_state_canonicalization(bool enabled) {
        state_canonicalization_mode_ = enabled ? StateCanonicalizationMode::Full : StateCanonicalizationMode::None;
    }

    bool state_canonicalization_enabled() const {
        return state_canonicalization_mode_ != StateCanonicalizationMode::None;
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
    ) {
        // Allocate event ID
        EventId eid = counters_.alloc_event();

        bool is_canonical = true;
        EventId canonical_eid = eid;

        // Event canonicalization: check if this event signature already exists
        // Signature is built from components specified by event_signature_keys_ bitflag
        if (event_signature_keys_ != EVENT_SIG_NONE) {
            const EventSignatureKeys keys = event_signature_keys_;

            // Get canonical state IDs for event canonicalization (isomorphism-based)
            // Use get_canonical_state_for_event() which ALWAYS uses isomorphism hash,
            // because the reference's CanonicalEventFunction ALWAYS uses
            // CanonicalLinkedHypergraph (isomorphism) regardless of state mode.
            StateId canonical_input = get_canonical_state_for_event(input_state);
            StateId canonical_output = get_canonical_state_for_event(output_state);
            const State& canonical_out_state = get_state(canonical_output);

            uint64_t sig_key = FNV_OFFSET;

            // Add isomorphism-invariant state hashes to signature if requested
            if (keys & EventKey_InputState) {
                uint64_t input_hash = get_or_compute_canonical_hash(input_state);
                sig_key = fnv_hash(sig_key, input_hash);
            }
            if (keys & EventKey_OutputState) {
                uint64_t output_hash = get_or_compute_canonical_hash(output_state);
                sig_key = fnv_hash(sig_key, output_hash);
            }
            if (keys & EventKey_Step) {
                sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_out_state.step));
            }
            if (keys & EventKey_Rule) {
                sig_key = fnv_hash(sig_key, static_cast<uint64_t>(rule_index));
            }

            // Add edge signatures if requested (requires edge correspondence computation)
            if (keys & (EventKey_ConsumedEdges | EventKey_ProducedEdges)) {
                const State& in_state = get_state(input_state);
                const State& out_state = get_state(output_state);
                const State& canonical_in_state = get_state(canonical_input);

                // Compute edge correspondence using hash dispatch
                EdgeCorrespondence input_correspondence = find_edge_correspondence_dispatch(
                    in_state.edges, canonical_in_state.edges);
                EdgeCorrespondence output_correspondence = find_edge_correspondence_dispatch(
                    out_state.edges, canonical_out_state.edges);

                // Build edge mappings: raw_edge_id -> canonical_edge_id
                std::unordered_map<EdgeId, EdgeId> input_edge_map, output_edge_map;
                if (input_correspondence.valid) {
                    for (uint32_t i = 0; i < input_correspondence.count; ++i) {
                        input_edge_map[input_correspondence.state1_edges[i]] =
                            input_correspondence.state2_edges[i];
                    }
                }
                if (output_correspondence.valid) {
                    for (uint32_t i = 0; i < output_correspondence.count; ++i) {
                        output_edge_map[output_correspondence.state1_edges[i]] =
                            output_correspondence.state2_edges[i];
                    }
                }

                // Map edges to canonical equivalents and compute signatures
                if (keys & EventKey_ConsumedEdges) {
                    for (uint8_t i = 0; i < num_consumed; ++i) {
                        auto it = input_edge_map.find(consumed[i]);
                        EdgeId canonical_edge = (it != input_edge_map.end()) ? it->second : consumed[i];
                        sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_edge));
                    }
                }

                if (keys & EventKey_ProducedEdges) {
                    for (uint8_t i = 0; i < num_produced; ++i) {
                        auto it = output_edge_map.find(produced[i]);
                        EdgeId canonical_edge = (it != output_edge_map.end()) ? it->second : produced[i];
                        sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_edge));
                    }
                }
            }

            // Avoid key=0 (reserved as EMPTY_KEY in ConcurrentMap)
            if (sig_key == 0 || sig_key == FNV_OFFSET) sig_key = 1;

            // Try to insert this signature - if it exists, we have a duplicate
            auto [existing_or_new, was_inserted] = canonical_event_map_.insert_if_absent_waiting(sig_key, eid);

            if (!was_inserted) {
                is_canonical = false;
                canonical_eid = existing_or_new;
            } else {
                canonical_event_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Allocate and copy edge arrays
        EdgeId* cons = arena_.allocate_array<EdgeId>(num_consumed);
        std::memcpy(cons, consumed, num_consumed * sizeof(EdgeId));

        EdgeId* prod = arena_.allocate_array<EdgeId>(num_produced);
        std::memcpy(prod, produced, num_produced * sizeof(EdgeId));

        // Directly construct event at slot eid using emplace_at
        // Pass canonical_event_id: INVALID_ID if this event is canonical, otherwise the canonical event's ID
        EventId canonical_id_for_event = is_canonical ? INVALID_ID : canonical_eid;
        events_.emplace_at(eid, arena_, eid, input_state, output_state, rule_index,
                           cons, num_consumed, prod, num_produced, binding, canonical_id_for_event);

        // CRITICAL: Release fence to ensure event data is visible to other threads
        // before the event ID escapes via the return value or is used in concurrent
        // callbacks (e.g., branchial tracking iterates events and accesses event data).
        // Without this, another thread may see the event ID in a list but not see
        // the event's data in the SegmentedArray due to memory ordering.
        std::atomic_thread_fence(std::memory_order_release);

        // Track parent-child relationship
        add_state_child(input_state, output_state);

        return {eid, canonical_eid, is_canonical};
    }

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

    // Hash strategy (UniquenessTree vs WL)
    void set_hash_strategy(HashStrategy strategy) {
        hash_strategy_ = strategy;
    }

    HashStrategy hash_strategy() const {
        return hash_strategy_;
    }

    // Abort flag for long-running operations (e.g., tree hash computation)
    // Set by evolution engine to allow early termination on user abort
    void set_abort_flag(std::atomic<bool>* flag) {
        abort_flag_ = flag;
        // Also propagate to tree implementations
        if (unified_tree_) unified_tree_->set_abort_flag(flag);
        if (incremental_tree_) incremental_tree_->set_abort_flag(flag);
    }

    bool should_abort() const {
        return abort_flag_ && abort_flag_->load(std::memory_order_relaxed);
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
    EventId create_genesis_event(StateId initial_state, const EdgeId* edges, uint8_t num_edges) {
        // Ensure genesis state exists
        StateId genesis = get_or_create_genesis_state();

        // Allocate event ID
        EventId eid = counters_.alloc_event();

        // Event canonicalization for genesis events
        bool is_canonical = true;
        EventId canonical_eid = eid;

        if (event_signature_keys_ != EVENT_SIG_NONE) {
            const EventSignatureKeys keys = event_signature_keys_;

            // Get canonical state IDs (used for edge correspondence if needed)
            StateId canonical_output = get_canonical_state(initial_state);
            const State& canonical_out_state = get_state(canonical_output);

            // Build signature from selected keys
            // (genesis events don't have consumed edges or rule, only produced edges)
            uint64_t sig_key = FNV_OFFSET;

            // Use isomorphism-invariant state hashes for event signature
            if (keys & EventKey_InputState) {
                uint64_t input_hash = get_or_compute_canonical_hash(genesis);
                sig_key = fnv_hash(sig_key, input_hash);
            }
            if (keys & EventKey_OutputState) {
                uint64_t output_hash = get_or_compute_canonical_hash(initial_state);
                sig_key = fnv_hash(sig_key, output_hash);
            }
            if (keys & EventKey_Step) {
                sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_out_state.step));
            }
            // Note: Rule key not applicable for genesis events (rule_index = -1)
            // Note: ConsumedEdges not applicable (genesis consumes nothing)
            if (keys & EventKey_ProducedEdges) {
                // For genesis, produced edges are the initial state's edges
                for (uint8_t i = 0; i < num_edges; ++i) {
                    sig_key = fnv_hash(sig_key, static_cast<uint64_t>(edges[i]));
                }
            }

            if (sig_key == 0 || sig_key == FNV_OFFSET) sig_key = 1;

            // Try to insert - if it exists, we have a duplicate genesis event
            auto [existing_or_new, was_inserted] = canonical_event_map_.insert_if_absent_waiting(sig_key, eid);

            if (!was_inserted) {
                is_canonical = false;
                canonical_eid = existing_or_new;
            } else {
                canonical_event_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Allocate produced edges array
        EdgeId* produced = arena_.allocate_array<EdgeId>(num_edges);
        std::memcpy(produced, edges, num_edges * sizeof(EdgeId));

        // Directly construct event at slot eid using emplace_at
        // Genesis event: input_state = genesis state (empty), output_state = initial_state
        // Rule index = -1 (no rule applied), consumes nothing, produces all initial edges
        EventId canonical_id_for_event = is_canonical ? INVALID_ID : canonical_eid;
        events_.emplace_at(eid, arena_, eid, genesis, initial_state,
                           static_cast<RuleIndex>(-1),
                           nullptr, 0,  // consumed_edges (none)
                           produced, num_edges,  // produced_edges
                           VariableBinding{},  // empty binding
                           canonical_id_for_event);

        // CRITICAL: Release fence to ensure event data is visible to other threads
        std::atomic_thread_fence(std::memory_order_release);

        // Register this event as the producer of all initial edges
        for (uint8_t i = 0; i < num_edges; ++i) {
            set_edge_producer(edges[i], eid);
        }

        return eid;
    }

    // Register event for branchial tracking
    // When event canonicalization is enabled, uses edge equivalence for overlap detection
    // and skips branchial edges between canonically equivalent events
    void register_event_for_branchial(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed,
        EventId canonical_event = INVALID_ID  // Pass canonical_event_id for deduplication
    ) {
        if (event_signature_keys_ != EVENT_SIG_NONE) {
            // Use edge equivalence-aware branchial registration
            causal_graph_.register_event_from_state_with_canonicalization(
                event, input_state, consumed_edges, num_consumed,
                // Get consumed edges callback
                [this](EventId eid, const EdgeId*& edges, uint8_t& num) {
                    const Event& ev = events_[eid];
                    edges = ev.consumed_edges;
                    num = ev.num_consumed;
                },
                // Same canonical event check - for v1 compatibility, don't skip any events
                // v1 creates branchial edges between all event pairs with overlap,
                // even if they're canonically equivalent
                [this, canonical_event](EventId e1, EventId e2) -> bool {
                    // Return false = don't skip = create branchial edge
                    // This matches v1's behavior
                    (void)e1; (void)e2; (void)canonical_event;
                    return false;
                },
                // Edge equivalence check - v1 uses raw edge ID comparison
                [](EdgeId e1, EdgeId e2) -> bool {
                    return e1 == e2;
                }
            );
        } else {
            // No canonicalization - use simple overlap check with raw edge IDs
            causal_graph_.register_event_from_state_with_overlap_check(
                event, input_state, consumed_edges, num_consumed,
                [this](EventId eid, const EdgeId*& edges, uint8_t& num) {
                    const Event& ev = events_[eid];
                    edges = ev.consumed_edges;
                    num = ev.num_consumed;
                }
            );
        }
    }

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
    // This preserves edge ordering - states with same content in different order
    // will have different hashes (matching reference MultiwaySystem behavior).
    // Fast but not isomorphism-invariant - states with same content but different
    // vertex numbering will have different hashes.
    uint64_t compute_content_ordered_hash(const SparseBitset& edges) const {
        uint64_t h = FNV_OFFSET;

        // Hash edge count first for extra differentiation
        h = fnv_hash(h, mix64(edges.count()));

        // SparseBitset iteration is ordered by edge ID - this preserves edge order
        edges.for_each([&](EdgeId eid) {
            const Edge& e = edges_[eid];
            // Hash arity with mixing for better avalanche on small values
            h = fnv_hash(h, mix64(static_cast<uint64_t>(e.arity)));
            // Hash each vertex in order with mixing
            for (uint8_t i = 0; i < e.arity; ++i) {
                h = fnv_hash(h, mix64(static_cast<uint64_t>(e.vertices[i])));
            }
            // Edge separator to prevent boundary ambiguity
            h = fnv_hash(h, 0xDEADBEEFCAFEBABEULL);
        });

        return h;
    }

    // Compute canonical hash using exact canonicalization (isomorphism-invariant)
    // Uses factorial-time algorithm for correctness, but fast for small graphs
    // If shared_tree is enabled, uses faster uniqueness tree hashing instead
    uint64_t compute_canonical_hash(const SparseBitset& edges) const {
        // Use unified uniqueness tree if enabled (faster with incremental computation)
        if (use_shared_tree_ && unified_tree_) {
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

    // Compute canonical hash using the selected hash strategy
    // Uses globally cached vertex tree data for incremental computation
    // This is faster than full canonicalization and correctly identifies isomorphism
    uint64_t compute_canonical_hash_shared(const SparseBitset& edges) const {
        if (edges.empty()) {
            return 0;
        }

        // CRITICAL: Acquire fence to ensure we see all edge data written by other threads.
        // Pairs with release fence in create_edge after edge construction.
        std::atomic_thread_fence(std::memory_order_acquire);

        // Use hash dispatch
        auto [hash, cache] = compute_hash_with_cache_dispatch(edges);
        return hash;
    }

    // Compute canonical hash using UniquenessTree (polynomial-time, approximate)
    // This is faster but may have rare false positives/negatives
    uint64_t compute_canonical_hash_wl(const SparseBitset& edges) {
        // Delegate to compute_canonical_hash_shared
        return compute_canonical_hash_shared(edges);
    }

    // Get or compute WL hash cache for a state (memoized)
    // Thread-safe: uses atomic pointer with compare-exchange to prevent torn writes
    VertexHashCache get_or_compute_wl_cache(StateId state_id) {
        WLHashCacheEntry& entry = wl_hash_cache_.get_or_default(state_id, arena_);

        // Fast path: already computed
        VertexHashCache* cached = entry.cache_ptr.load(std::memory_order_acquire);
        if (cached) {
            return *cached;
        }

        // Slow path: compute cache
        const State& state = get_state(state_id);
        EdgeVertexAccessorRaw vert_acc(this);
        EdgeArityAccessorRaw arity_acc(this);

        auto [hash, cache] = wl_hash_->compute_state_hash_with_cache(
            state.edges, vert_acc, arity_acc);

        // Allocate cache on arena and copy data
        VertexHashCache* new_cache = arena_.create<VertexHashCache>(cache);

        // Try to set the pointer atomically - if someone else beat us, use theirs
        VertexHashCache* expected = nullptr;
        if (entry.cache_ptr.compare_exchange_strong(expected, new_cache,
                                                     std::memory_order_release,
                                                     std::memory_order_acquire)) {
            // We won the race - our cache is now the canonical one
            return *new_cache;
        } else {
            // Someone else won - expected now contains their pointer
            // Our new_cache allocation is wasted but that's OK (arena memory)
            return *expected;
        }
    }

    // Get or compute UT-Inc hash cache for a state (memoized)
    // Thread-safe: uses atomic pointer with compare-exchange to prevent torn writes
    VertexHashCache get_or_compute_ut_cache(StateId state_id) {
        StateIncrementalCache& entry = state_incremental_cache_.get_or_default(state_id, arena_);

        // Fast path: already computed
        StateIncrementalCacheData* cached = entry.data_ptr.load(std::memory_order_acquire);
        if (cached) {
            return cached->vertex_cache;
        }

        // Slow path: compute cache
        const State& state = get_state(state_id);
        EdgeVertexAccessorRaw vert_acc(this);
        EdgeArityAccessorRaw arity_acc(this);

        auto [hash, cache] = incremental_tree_->compute_state_hash_with_cache(
            state.edges, vert_acc, arity_acc);

        // Allocate cache data on arena
        StateIncrementalCacheData* new_data = arena_.create<StateIncrementalCacheData>();
        new_data->vertex_cache = cache;

        // Try to set the pointer atomically - if someone else beat us, use theirs
        StateIncrementalCacheData* expected = nullptr;
        if (entry.data_ptr.compare_exchange_strong(expected, new_data,
                                                    std::memory_order_release,
                                                    std::memory_order_acquire)) {
            // We won the race
            return new_data->vertex_cache;
        } else {
            // Someone else won - expected now contains their pointer
            return expected->vertex_cache;
        }
    }

    // Get or compute plain UT hash cache for a state (memoized)
    // Thread-safe: uses atomic pointer with compare-exchange to prevent torn writes
    // Reuses wl_hash_cache_ since WL and UT are mutually exclusive
    VertexHashCache get_or_compute_ut_plain_cache(StateId state_id) {
        WLHashCacheEntry& entry = wl_hash_cache_.get_or_default(state_id, arena_);

        // Fast path: already computed
        VertexHashCache* cached = entry.cache_ptr.load(std::memory_order_acquire);
        if (cached) {
            return *cached;
        }

        // Slow path: compute cache
        const State& state = get_state(state_id);
        EdgeVertexAccessorRaw vert_acc(this);
        EdgeArityAccessorRaw arity_acc(this);

        auto [hash, cache] = unified_tree_->compute_state_hash_with_cache(
            state.edges, vert_acc, arity_acc);

        // Allocate cache on arena and copy data
        VertexHashCache* new_cache = arena_.create<VertexHashCache>(cache);

        // Try to set the pointer atomically - if someone else beat us, use theirs
        VertexHashCache* expected = nullptr;
        if (entry.cache_ptr.compare_exchange_strong(expected, new_cache,
                                                     std::memory_order_release,
                                                     std::memory_order_acquire)) {
            return *new_cache;
        } else {
            return *expected;
        }
    }

    // =========================================================================
    // Unified Hash Strategy Dispatch Helpers
    // =========================================================================
    // These methods provide a single interface regardless of which hash strategy
    // is active, eliminating the need for repeated if/else dispatch chains.

    // Compute hash with cache using the active hash strategy
    // Returns {hash, vertex_cache} pair
    std::pair<uint64_t, VertexHashCache> compute_hash_with_cache_dispatch(
        const SparseBitset& edges
    ) const {
        if (edges.empty()) {
            return {0, VertexHashCache()};
        }

        EdgeVertexAccessorRaw vert_acc(this);
        EdgeArityAccessorRaw arity_acc(this);

        if ((hash_strategy_ == HashStrategy::WL || hash_strategy_ == HashStrategy::IncrementalWL) && wl_hash_) {
            return wl_hash_->compute_state_hash_with_cache(edges, vert_acc, arity_acc);
        } else if (hash_strategy_ == HashStrategy::IncrementalUniquenessTree && incremental_tree_) {
            return incremental_tree_->compute_state_hash_with_cache(edges, vert_acc, arity_acc);
        } else if (unified_tree_) {
            return unified_tree_->compute_state_hash_with_cache(edges, vert_acc, arity_acc);
        }
        return {0, VertexHashCache()};
    }

    // Get or compute hash cache for a state using the active hash strategy
    // Thread-safe: memoizes the result per state
    VertexHashCache get_or_compute_hash_cache_dispatch(StateId state_id) {
        switch (hash_strategy_) {
            case HashStrategy::WL:
            case HashStrategy::IncrementalWL:
                return get_or_compute_wl_cache(state_id);
            case HashStrategy::IncrementalUniquenessTree:
                return get_or_compute_ut_cache(state_id);
            case HashStrategy::UniquenessTree:
            default:
                return get_or_compute_ut_plain_cache(state_id);
        }
    }

    // Find edge correspondence between two isomorphic states using active strategy
    // Returns mapping from state1 edges to state2 edges
    EdgeCorrespondence find_edge_correspondence_dispatch(
        const SparseBitset& state1_edges,
        const SparseBitset& state2_edges
    ) const {
        EdgeVertexAccessorRaw vert_acc(this);
        EdgeArityAccessorRaw arity_acc(this);

        if ((hash_strategy_ == HashStrategy::WL || hash_strategy_ == HashStrategy::IncrementalWL) && wl_hash_) {
            return wl_hash_->find_edge_correspondence(state1_edges, state2_edges, vert_acc, arity_acc);
        } else if (hash_strategy_ == HashStrategy::IncrementalUniquenessTree && incremental_tree_) {
            return incremental_tree_->find_edge_correspondence(state1_edges, state2_edges, vert_acc, arity_acc);
        } else if (unified_tree_) {
            return unified_tree_->find_edge_correspondence(state1_edges, state2_edges, vert_acc, arity_acc);
        }
        return EdgeCorrespondence{};  // Invalid
    }

    // =========================================================================
    // Incremental Hash Computation
    // =========================================================================
    // These methods enable O(delta) hash computation for child states by
    // reusing parent state's vertex hash cache.

    // Compute canonical hash incrementally using parent state's cache
    // Returns (hash, cache_for_new_state)
    //
    // BLOOM FILTER STRATEGY:
    // 1. Build per-state adjacency once (like UT does) - O(edges_in_state)
    // 2. Compute tree hash with bloom filter for each vertex's subtree
    // 3. For child states, check if vertex's bloom filter contains any affected vertex
    // 4. If bloom filter says no affected vertices → reuse parent's cached hash
    // 5. If bloom filter says maybe affected → recompute with new adjacency
    //
    // This achieves O(affected_subtrees) for most rewrites on sparse graphs.
    //
    std::pair<uint64_t, VertexHashCache> compute_canonical_hash_incremental(
        const SparseBitset& new_edges,
        StateId parent_state,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced
    ) {
        if (new_edges.empty()) {
            return {0, VertexHashCache()};
        }

        // NOTE: We always compute the isomorphism-invariant hash regardless of
        // state_canonicalization_mode_. This is needed for EVENT canonicalization,
        // which must find canonical representatives even when state mode is None.
        // The state mode only affects whether STATES are deduplicated, not whether
        // their canonical hashes are computed.

        // Handle WL strategy (non-incremental)
        if (hash_strategy_ == HashStrategy::WL && wl_hash_) {
            EdgeVertexAccessorRaw verts_accessor(this);
            EdgeArityAccessorRaw arities_accessor(this);
            auto [hash, cache] = wl_hash_->compute_state_hash_with_cache(
                new_edges, verts_accessor, arities_accessor);
            return {hash, cache};
        }

        // Handle IncrementalWL strategy
        if (hash_strategy_ == HashStrategy::IncrementalWL && wl_hash_) {
            EdgeVertexAccessorRaw verts_accessor(this);
            EdgeArityAccessorRaw arities_accessor(this);

            // Try to get parent cache for incremental computation
            const VertexHashCache* parent_wl_cache = nullptr;
            if (parent_state != INVALID_ID && parent_state < wl_hash_cache_.size()) {
                const WLHashCacheEntry& entry = wl_hash_cache_[parent_state];
                VertexHashCache* cached = entry.cache_ptr.load(std::memory_order_acquire);
                if (cached && cached->count > 0) {
                    parent_wl_cache = cached;
                }
            }

            if (parent_wl_cache) {
                // Use incremental WL computation
                auto [hash, cache] = wl_hash_->compute_state_hash_incremental_with_cache(
                    new_edges, *parent_wl_cache,
                    consumed_edges, num_consumed,
                    produced_edges, num_produced,
                    verts_accessor, arities_accessor);
                return {hash, cache};
            } else {
                // No parent cache, use full computation
                auto [hash, cache] = wl_hash_->compute_state_hash_with_cache(
                    new_edges, verts_accessor, arities_accessor);
                return {hash, cache};
            }
        }

        // Use incremental path only for IncrementalUniquenessTree strategy
        if (hash_strategy_ != HashStrategy::IncrementalUniquenessTree || !incremental_tree_) {
            // Fall back to non-incremental computation for other strategies
            EdgeVertexAccessorRaw verts_accessor(this);
            EdgeArityAccessorRaw arities_accessor(this);

            if (unified_tree_) {
                auto [hash, cache] = unified_tree_->compute_state_hash_with_cache(
                    new_edges, verts_accessor, arities_accessor);
                return {hash, cache};
            }
            return {0, VertexHashCache()};
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        EdgeVertexAccessorRaw verts_accessor(this);
        EdgeArityAccessorRaw arities_accessor(this);

        // Get parent's vertex hash cache for incremental reuse
        // Thread-safe: uses acquire semantics to synchronize with store_state_cache's release.
        const VertexHashCache* parent_vertex_cache = nullptr;
        if (parent_state != INVALID_ID && parent_state < state_incremental_cache_.size()) {
            const StateIncrementalCache* pcache = state_incremental_cache_.get(parent_state);
            // Acquire load synchronizes with release store in store_state_cache
            if (pcache) {
                StateIncrementalCacheData* data = pcache->data_ptr.load(std::memory_order_acquire);
                if (data) {
                    const VertexHashCache& vc = data->vertex_cache;
                    // Verify ALL required pointers are non-null before using
                    if (vc.count > 0 && vc.vertices != nullptr &&
                        vc.hashes != nullptr && vc.subtree_filters != nullptr) {
                        parent_vertex_cache = &vc;
                    }
                }
            }
        }

        // Collect directly affected vertices (vertices in consumed or produced edges)
        ArenaVector<VertexId> affected_vertices(arena_);
        for (uint8_t i = 0; i < num_consumed; ++i) {
            EdgeId eid = consumed_edges[i];
            uint8_t arity = arities_accessor[eid];
            const VertexId* verts = verts_accessor[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                affected_vertices.push_back(verts[j]);
            }
        }
        for (uint8_t i = 0; i < num_produced; ++i) {
            EdgeId eid = produced_edges[i];
            uint8_t arity = arities_accessor[eid];
            const VertexId* verts = verts_accessor[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                affected_vertices.push_back(verts[j]);
            }
        }
        // Remove duplicates for efficient bloom filter checking
        std::sort(affected_vertices.begin(), affected_vertices.end());
        auto new_end = std::unique(affected_vertices.begin(), affected_vertices.end());
        affected_vertices.resize(new_end - affected_vertices.begin());

        // If no parent cache with bloom filters, use the full computation
        // which builds bloom filters for future children
        if (!parent_vertex_cache) {
            return incremental_tree_->compute_state_hash_with_cache(
                new_edges, verts_accessor, arities_accessor);
        }

        // Bloom filter reuse path - use parent's vertex hash cache for incremental computation

        // Collect all vertices in child state (O(E) scan)
        // TODO: Consider incremental vertex collection once performance is validated
        ArenaVector<VertexId> vertices(arena_);
        std::unordered_set<VertexId> seen_vertices;
        new_edges.for_each([&](EdgeId eid) {
            uint8_t arity = arities_accessor[eid];
            const VertexId* verts = verts_accessor[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                if (seen_vertices.insert(verts[j]).second) {
                    vertices.push_back(verts[j]);
                }
            }
        });
        std::sort(vertices.begin(), vertices.end());

        if (vertices.empty()) {
            return {0, VertexHashCache()};
        }

        // Prepare result cache with space for bloom filters
        VertexHashCache result_cache;
        result_cache.capacity = static_cast<uint32_t>(vertices.size());
        result_cache.vertices = arena_.allocate_array<VertexId>(result_cache.capacity);
        result_cache.hashes = arena_.allocate_array<uint64_t>(result_cache.capacity);
        result_cache.subtree_filters = arena_.allocate_array<SubtreeBloomFilter>(result_cache.capacity);
        result_cache.count = 0;

        ArenaVector<uint64_t> tree_hashes(arena_, vertices.size());

        // Track stats via member atomics for reporting
        size_t local_reused = 0;
        size_t local_recomputed = 0;

        // Lazy adjacency building: only build if we need to recompute vertices
        // This is O(E) but only when needed, and subsequent lookups are O(1)
        std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>> adjacency;
        bool adjacency_built = false;

        auto build_adjacency_if_needed = [&]() {
            if (adjacency_built) return;
            adjacency_built = true;

            new_edges.for_each([&](EdgeId eid) {
                uint8_t arity = arities_accessor[eid];
                const VertexId* verts = verts_accessor[eid];
                for (uint8_t i = 0; i < arity; ++i) {
                    VertexId v = verts[i];
                    auto it = adjacency.find(v);
                    if (it == adjacency.end()) {
                        it = adjacency.emplace(v, ArenaVector<std::pair<EdgeId, uint8_t>>(arena_)).first;
                    }
                    it->second.push_back({eid, i});
                }
            });
        };

        // Build O(1) lookup map from parent cache for efficient vertex hash lookup
        std::unordered_map<VertexId, uint32_t> parent_cache_index;
        parent_cache_index.reserve(parent_vertex_cache->count);
        for (uint32_t i = 0; i < parent_vertex_cache->count; ++i) {
            parent_cache_index[parent_vertex_cache->vertices[i]] = i;
        }

        // Compute tree hash for each vertex, reusing where possible
        for (VertexId root : vertices) {
            // Check if we can reuse parent's hash via bloom filter (O(1) lookup)
            auto cache_it = parent_cache_index.find(root);
            if (cache_it != parent_cache_index.end()) {
                uint32_t idx = cache_it->second;
                uint64_t parent_hash = parent_vertex_cache->hashes[idx];
                const SubtreeBloomFilter* bloom = parent_vertex_cache->subtree_filters
                    ? &parent_vertex_cache->subtree_filters[idx] : nullptr;

                if (bloom != nullptr && parent_hash != 0) {
                    // Check if any affected vertex might be in this subtree
                    bool might_be_affected = false;
                    for (VertexId affected : affected_vertices) {
                        if (bloom->might_contain(affected)) {
                            might_be_affected = true;
                            break;
                        }
                    }

                    if (!might_be_affected) {
                        // Bloom filter says no affected vertices in subtree - reuse hash!
                        tree_hashes.push_back(parent_hash);
                        result_cache.insert_with_subtree(root, parent_hash, *bloom);
                        ++local_reused;
                        continue;
                    }
                }
            }

            // Need to recompute this vertex's hash
            build_adjacency_if_needed();
            ++local_recomputed;

            // Use DirectAdjacencyWithArity wrapper for the tree hash computation
            DirectAdjacencyWithArity<decltype(adjacency), EdgeArityAccessorRaw> adj_provider(adjacency, arities_accessor);

            // Compute tree hash with bloom filter
            SparseBitset visited;
            SubtreeBloomFilter new_bloom;
            new_bloom.clear();

            uint64_t tree_hash = incremental_tree_->compute_tree_hash_with_bloom(
                root, new_edges, verts_accessor, arities_accessor,
                adj_provider, visited, new_bloom);

            tree_hashes.push_back(tree_hash);
            result_cache.insert_with_subtree(root, tree_hash, new_bloom);
        }

        // Update member atomics for stats reporting
        bloom_reused_.fetch_add(local_reused, std::memory_order_relaxed);
        bloom_recomputed_.fetch_add(local_recomputed, std::memory_order_relaxed);

        // Combine tree hashes into state hash
        std::sort(tree_hashes.begin(), tree_hashes.end());
        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, result_cache};
    }

    // Store computed cache for a state (call after creating state)
    // Thread-safe: uses compare-exchange on atomic pointer to ensure only one thread writes.
    void store_state_cache(StateId state, const VertexHashCache& cache) {
        // Store for IncrementalWL strategy
        if (hash_strategy_ == HashStrategy::IncrementalWL) {
            WLHashCacheEntry& slot = wl_hash_cache_.get_or_default(state, arena_);

            // Allocate cache on arena
            VertexHashCache* new_cache = arena_.create<VertexHashCache>(cache);

            // Try to set the pointer atomically - if someone else beat us, skip
            VertexHashCache* expected = nullptr;
            slot.cache_ptr.compare_exchange_strong(expected, new_cache,
                    std::memory_order_release, std::memory_order_relaxed);
            // If CAS fails, new_cache is wasted but that's OK (arena memory)
            return;
        }

        // Store for iUT strategy
        if (hash_strategy_ != HashStrategy::IncrementalUniquenessTree) {
            return;  // Only store for incremental strategies (iUT, iWL)
        }

        StateIncrementalCache& slot = state_incremental_cache_.get_or_default(state, arena_);

        // Allocate cache data on arena
        StateIncrementalCacheData* new_data = arena_.create<StateIncrementalCacheData>();
        new_data->vertex_cache = cache;

        // Try to set the pointer atomically - if someone else beat us, skip
        StateIncrementalCacheData* expected = nullptr;
        slot.data_ptr.compare_exchange_strong(expected, new_data,
                std::memory_order_release, std::memory_order_relaxed);
        // If CAS fails, new_data is wasted but that's OK (arena memory)
    }

    // Get number of stored caches (for debugging/profiling)
    size_t num_stored_caches() const {
        size_t count = 0;
        for (size_t i = 0; i < state_incremental_cache_.size(); ++i) {
            if (state_incremental_cache_[i].data_ptr.load(std::memory_order_relaxed) != nullptr) {
                ++count;
            }
        }
        return count;
    }

    // Get incremental tree stats (for profiling)
    // Returns stats from bloom filter reuse path + incremental tree fallback path
    std::pair<size_t, size_t> incremental_tree_stats() const {
        size_t reused = bloom_reused_.load(std::memory_order_relaxed);
        size_t recomputed = bloom_recomputed_.load(std::memory_order_relaxed);
        if (incremental_tree_) {
            reused += incremental_tree_->stats_reused();
            recomputed += incremental_tree_->stats_recomputed();
        }
        return {reused, recomputed};
    }

    void reset_incremental_tree_stats() {
        bloom_reused_.store(0, std::memory_order_relaxed);
        bloom_recomputed_.store(0, std::memory_order_relaxed);
        if (incremental_tree_) {
            incremental_tree_->reset_stats();
        }
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

}  // namespace hypergraph
