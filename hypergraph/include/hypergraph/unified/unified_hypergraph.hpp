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
#include "unified_uniqueness_tree.hpp"
#include "incremental_unified_uniqueness_tree.hpp"
#include "wl_hash.hpp"
#include "concurrent_map.hpp"
#include "edge_equivalence.hpp"

// Include v1 canonicalizer for exact isomorphism checking
#include "../canonicalization.hpp"

namespace hypergraph::unified {

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

    // Per-state WL hash cache (VertexHashCache + validity flag)
    // Used to memoize compute_state_hash_with_cache for canonical states
    struct WLHashCacheEntry {
        VertexHashCache cache;
        std::atomic<bool> valid{false};
    };
    SegmentedArray<WLHashCacheEntry> wl_hash_cache_;

    // Causal and branchial graph
    CausalGraph causal_graph_;

    // Edge equivalence manager for Level 2 canonicalization
    EdgeEquivalenceManager edge_equiv_manager_;

    // Canonical state deduplication map: canonical_hash -> StateId
    // Used to find existing equivalent states before creating new ones
    ConcurrentMap<uint64_t, StateId> canonical_state_map_;

    // Level 2 canonicalization enabled flag
    bool level2_enabled_{false};

    // ==========================================================================
    // Global Vertex Adjacency Index
    // ==========================================================================
    // Maps each vertex to the list of edges it appears in, with position info.
    // This is THE canonical adjacency structure for the entire unified hypergraph.
    //
    // Thread safety: Append-only via LockFreeList. Edges are immutable after
    // creation, so once registered here, entries never change or get removed.
    //
    // Usage: When computing state hashes, iterate this and filter by
    // state_edges.contains(edge_id) to get per-state adjacency without copying.
    SegmentedArray<LockFreeList<EdgeOccurrence>> vertex_adjacency_;

    // Hash strategy implementations
    std::unique_ptr<UnifiedUniquenessTree> unified_tree_;  // Gorard-style uniqueness trees
    std::unique_ptr<IncrementalUnifiedUniquenessTree> incremental_tree_;  // Incremental version
    std::unique_ptr<WLHash> wl_hash_;                      // Weisfeiler-Lehman hashing
    HashStrategy hash_strategy_{HashStrategy::IncrementalWL};  // UT-Inc with cached adjacency

    // Stats for bloom filter-based vertex hash reuse in compute_canonical_hash_incremental
    mutable std::atomic<size_t> bloom_reused_{0};
    mutable std::atomic<size_t> bloom_recomputed_{0};

    // Flag to use shared tree vs exact canonicalization
    // Enabled by default: WL hashing is O(V²×E) vs O(g!) factorial for exact canonicalization
    bool use_shared_tree_{true};

    // Event canonicalization: maps event signature to first EventId
    // For ByState mode: key is (canonical_input_id << 32) | canonical_output_id
    // For ByStateAndEdges mode: key is hash of full EventSignature
    ConcurrentMap<uint64_t, EventId> canonical_event_map_;
    std::atomic<uint32_t> canonical_event_count_{0};
    EventCanonicalizationMode event_canonicalization_mode_{EventCanonicalizationMode::ByState};

public:
    UnifiedHypergraph()
        : unified_tree_(std::make_unique<UnifiedUniquenessTree>(&arena_))
        , incremental_tree_(std::make_unique<IncrementalUnifiedUniquenessTree>(&arena_))
        , wl_hash_(std::make_unique<WLHash>(&arena_))
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
    // Optimized accessor: direct access without caching overhead
    class EdgeVertexAccessorRaw {
        const UnifiedHypergraph* hg_;
    public:
        explicit EdgeVertexAccessorRaw(const UnifiedHypergraph* hg) : hg_(hg) {}

        const VertexId* operator[](EdgeId eid) const {
            return hg_->edges_[eid].vertices;  // Direct access
        }
    };

    // Optimized accessor: direct access without caching overhead
    // edge_arity() is already O(1), caching added more overhead than it saved
    class EdgeArityAccessorRaw {
        const UnifiedHypergraph* hg_;
    public:
        explicit EdgeArityAccessorRaw(const UnifiedHypergraph* hg) : hg_(hg) {}

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
    // to IncrementalUnifiedUniquenessTree's external adjacency methods
    class GlobalAdjacencyProvider {
        const UnifiedHypergraph* hg_;
    public:
        explicit GlobalAdjacencyProvider(const UnifiedHypergraph* hg) : hg_(hg) {}

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

        // Try to insert into canonical map (lock-free, waiting for LOCKED slots)
        // Must use waiting version to handle concurrent inserts during resize
        auto [existing_or_new, was_inserted] = canonical_state_map_.insert_if_absent_waiting(canonical_hash, new_sid);

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

    // Lookup existing canonical state by hash (waits for concurrent inserts)
    std::optional<StateId> find_canonical_state(uint64_t canonical_hash) const {
        return canonical_state_map_.lookup_waiting(canonical_hash);
    }

    // Get the canonical representative for a given state
    // Returns the state itself if it's already the canonical representative
    // Uses waiting lookup to handle concurrent inserts
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

    // Register edge correspondence between two isomorphic states
    // This merges corresponding edges into the same equivalence class
    void register_edge_correspondence(StateId new_state, StateId existing_state) {
        if (!level2_enabled_) return;
        if (!unified_tree_) return;

        // Build accessors for the edge data (use references to allow dynamic extension)
        EdgeVertexAccessorRaw verts_accessor(this);
        EdgeArityAccessorRaw arities_accessor(this);

        // Find edge correspondence using the unified uniqueness tree
        EdgeCorrespondence correspondence = unified_tree_->find_edge_correspondence(
            states_[new_state].edges,
            states_[existing_state].edges,
            verts_accessor,
            arities_accessor
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
        if (event_canonicalization_mode_ != EventCanonicalizationMode::None) {
            // Get canonical state IDs for event signature
            StateId canonical_input = get_canonical_state(input_state);
            StateId canonical_output = get_canonical_state(output_state);


            uint64_t sig_key = 0;

            if (event_canonicalization_mode_ == EventCanonicalizationMode::ByState) {
                // Simple key: just canonical state IDs
                // Add 1 to both to avoid key=0 (reserved as EMPTY_KEY in ConcurrentMap)
                sig_key = (static_cast<uint64_t>(canonical_input + 1) << 32) |
                          static_cast<uint64_t>(canonical_output + 1);
            } else {
                // ByStateAndEdges: compute full signature with edge correspondence
                //
                // CRITICAL: We must use the CANONICAL REPRESENTATIVE states for
                // computing edge signatures, not the raw states. Here's why:
                //
                // When multiple raw states are isomorphic (same canonical hash),
                // they may have different vertex IDs and edge IDs. For example:
                //   Raw state 13: edges {E3[1,2], E4[2,3], E25[0,1], E26[1,14]}
                //   Raw state 19: edges {E2[1,2], E35[0,1], E37[1,19], E38[19,20]}
                //   Raw state 29: edges {E35[0,1], E36[1,19], E57[1,2], E58[2,30]}
                //
                // All three are isomorphic, but if we compute vertex hashes from each
                // raw state, the edge signatures will differ (different vertex IDs
                // produce different hashes). This causes events that consume the
                // "same" canonical edge position to have different signatures.
                //
                // The fix (matching v1's Automatic mode):
                // 1. Get the canonical representative states (first state seen with each hash)
                // 2. Compute vertex hashes for the CANONICAL states only
                // 3. Find edge correspondence: raw_state edges -> canonical_state edges
                // 4. Map consumed/produced edges through this correspondence
                // 5. Use canonical state's vertex hashes for edge signatures
                //
                // This ensures all isomorphic states map edges to the same canonical
                // positions, producing identical signatures for equivalent events.

                const State& in_state = get_state(input_state);
                const State& out_state = get_state(output_state);
                const State& canonical_in_state = get_state(canonical_input);
                const State& canonical_out_state = get_state(canonical_output);

                EdgeVertexAccessorRaw vert_acc(this);
                EdgeArityAccessorRaw arity_acc(this);

                // Compute vertex hashes for the CANONICAL states (not raw states)
                // These hashes will be consistent across all isomorphic raw states
                // Use the selected hash strategy
                VertexHashCache canonical_input_cache, canonical_output_cache;
                EdgeCorrespondence input_correspondence, output_correspondence;

                if (hash_strategy_ == HashStrategy::WL && wl_hash_) {
                    // Use memoized cache - many events share the same canonical states
                    canonical_input_cache = get_or_compute_wl_cache(canonical_input);
                    canonical_output_cache = get_or_compute_wl_cache(canonical_output);

                    input_correspondence = wl_hash_->find_edge_correspondence(
                        in_state.edges, canonical_in_state.edges, vert_acc, arity_acc);
                    output_correspondence = wl_hash_->find_edge_correspondence(
                        out_state.edges, canonical_out_state.edges, vert_acc, arity_acc);
                } else if (hash_strategy_ == HashStrategy::IncrementalUniquenessTree && incremental_tree_) {
                    auto [in_hash, in_cache] = incremental_tree_->compute_state_hash_with_cache(
                        canonical_in_state.edges, vert_acc, arity_acc);
                    auto [out_hash, out_cache] = incremental_tree_->compute_state_hash_with_cache(
                        canonical_out_state.edges, vert_acc, arity_acc);
                    canonical_input_cache = in_cache;
                    canonical_output_cache = out_cache;

                    input_correspondence = incremental_tree_->find_edge_correspondence(
                        in_state.edges, canonical_in_state.edges, vert_acc, arity_acc);
                    output_correspondence = incremental_tree_->find_edge_correspondence(
                        out_state.edges, canonical_out_state.edges, vert_acc, arity_acc);
                } else if (unified_tree_) {
                    auto [in_hash, in_cache] = unified_tree_->compute_state_hash_with_cache(
                        canonical_in_state.edges, vert_acc, arity_acc);
                    auto [out_hash, out_cache] = unified_tree_->compute_state_hash_with_cache(
                        canonical_out_state.edges, vert_acc, arity_acc);
                    canonical_input_cache = in_cache;
                    canonical_output_cache = out_cache;

                    input_correspondence = unified_tree_->find_edge_correspondence(
                        in_state.edges, canonical_in_state.edges, vert_acc, arity_acc);
                    output_correspondence = unified_tree_->find_edge_correspondence(
                        out_state.edges, canonical_out_state.edges, vert_acc, arity_acc);
                }

                // Build index mappings: raw_edge_id -> canonical_edge_id
                // Note: For the same raw state as canonical (input_state == canonical_input),
                // the correspondence maps edges to themselves, so this still works.
                std::unordered_map<EdgeId, EdgeId> input_edge_map;
                std::unordered_map<EdgeId, EdgeId> output_edge_map;

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

                // Map consumed edges to their canonical equivalents
                // These are the edges in the CANONICAL input state that correspond
                // to the consumed edges in the raw input state
                EdgeId canonical_consumed[MAX_PATTERN_EDGES];
                uint8_t num_canonical_consumed = 0;
                for (uint8_t i = 0; i < num_consumed; ++i) {
                    auto it = input_edge_map.find(consumed[i]);
                    if (it != input_edge_map.end()) {
                        canonical_consumed[num_canonical_consumed++] = it->second;
                    } else {
                        // Edge not found in correspondence - use original
                        // (This shouldn't happen for valid isomorphic states)
                        canonical_consumed[num_canonical_consumed++] = consumed[i];
                    }
                }

                // Map produced edges to their canonical equivalents
                EdgeId canonical_produced[MAX_PATTERN_EDGES];
                uint8_t num_canonical_produced = 0;
                for (uint8_t i = 0; i < num_produced; ++i) {
                    auto it = output_edge_map.find(produced[i]);
                    if (it != output_edge_map.end()) {
                        canonical_produced[num_canonical_produced++] = it->second;
                    } else {
                        canonical_produced[num_canonical_produced++] = produced[i];
                    }
                }

                // Compute event signature using:
                // - Canonical state hashes (for state identification)
                // - Canonical edge IDs with canonical vertex hashes (for edge positions)
                EventSignature sig;
                if (hash_strategy_ == HashStrategy::WL && wl_hash_) {
                    sig = wl_hash_->compute_event_signature(
                        canonical_in_state.canonical_hash, canonical_out_state.canonical_hash,
                        canonical_consumed, num_canonical_consumed,
                        canonical_produced, num_canonical_produced,
                        canonical_input_cache, canonical_output_cache, vert_acc, arity_acc);
                } else if (hash_strategy_ == HashStrategy::IncrementalUniquenessTree && incremental_tree_) {
                    sig = incremental_tree_->compute_event_signature(
                        canonical_in_state.canonical_hash, canonical_out_state.canonical_hash,
                        canonical_consumed, num_canonical_consumed,
                        canonical_produced, num_canonical_produced,
                        canonical_input_cache, canonical_output_cache, vert_acc, arity_acc);
                } else if (unified_tree_) {
                    sig = unified_tree_->compute_event_signature(
                        canonical_in_state.canonical_hash, canonical_out_state.canonical_hash,
                        canonical_consumed, num_canonical_consumed,
                        canonical_produced, num_canonical_produced,
                        canonical_input_cache, canonical_output_cache, vert_acc, arity_acc);
                }

                sig_key = sig.hash();
                // Avoid key=0 (reserved as EMPTY_KEY in ConcurrentMap)
                if (sig_key == 0) sig_key = 1;
            }

            // Try to insert this signature - if it exists, we have a duplicate
            // Use waiting version to handle concurrent inserts
            auto [existing_or_new, was_inserted] = canonical_event_map_.insert_if_absent_waiting(sig_key, eid);

            if (!was_inserted) {
                // Duplicate event - an event with this signature already exists
                is_canonical = false;
                canonical_eid = existing_or_new;
            } else {
                // New canonical event
                canonical_event_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Allocate and copy edge arrays
        EdgeId* cons = arena_.allocate_array<EdgeId>(num_consumed);
        std::memcpy(cons, consumed, num_consumed * sizeof(EdgeId));

        EdgeId* prod = arena_.allocate_array<EdgeId>(num_produced);
        std::memcpy(prod, produced, num_produced * sizeof(EdgeId));

        // Directly construct event at slot eid using emplace_at
        events_.emplace_at(eid, arena_, eid, input_state, output_state, rule_index,
                           cons, num_consumed, prod, num_produced, binding);

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
        if (event_canonicalization_mode_ != EventCanonicalizationMode::None) {
            return canonical_event_count_.load(std::memory_order_acquire);
        }
        // Use acquire to synchronize with release stores in alloc_event
        return counters_.next_event.load(std::memory_order_acquire);
    }

    // Number of raw events (always returns total count)
    uint32_t num_raw_events() const {
        return counters_.next_event.load(std::memory_order_acquire);
    }

    // Event canonicalization mode
    void set_event_canonicalization_mode(EventCanonicalizationMode mode) {
        event_canonicalization_mode_ = mode;
    }

    EventCanonicalizationMode event_canonicalization_mode() const {
        return event_canonicalization_mode_;
    }

    // Hash strategy (UniquenessTree vs WL)
    void set_hash_strategy(HashStrategy strategy) {
        hash_strategy_ = strategy;
    }

    HashStrategy hash_strategy() const {
        return hash_strategy_;
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

    // Create a genesis event for an initial state.
    // This synthetic event has input_state = INVALID_ID and output_state = initial_state.
    // It "produces" all edges in the initial state, enabling causal tracking from gen 0.
    // Returns the genesis event ID.
    EventId create_genesis_event(StateId initial_state, const EdgeId* edges, uint8_t num_edges) {
        // Allocate event ID
        EventId eid = counters_.alloc_event();

        // Allocate produced edges array
        EdgeId* produced = arena_.allocate_array<EdgeId>(num_edges);
        std::memcpy(produced, edges, num_edges * sizeof(EdgeId));

        // Directly construct event at slot eid using emplace_at
        // Genesis event: input_state = INVALID_ID, output_state = initial_state
        // Rule index = -1 (no rule applied), consumes nothing, produces all initial edges
        events_.emplace_at(eid, arena_, eid, INVALID_ID, initial_state,
                           static_cast<RuleIndex>(-1),
                           nullptr, 0,  // consumed_edges (none)
                           produced, num_edges,  // produced_edges
                           VariableBinding{});  // empty binding

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
        if (event_canonicalization_mode_ != EventCanonicalizationMode::None) {
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
                // Edge equivalence check - use union-find to check if edges are equivalent
                [this](EdgeId e1, EdgeId e2) -> bool {
                    // Two edges are equivalent if they belong to the same equivalence class
                    // This is determined by the uniqueness tree correspondences that have
                    // been merged via edge_equiv_manager_
                    if (e1 == e2) return true;
                    // Check if in same equivalence class via union-find
                    // Note: edge_equiv_manager_ tracks Level 2 equivalences
                    // For basic edge correspondence, we can also check via unified_tree_
                    return edge_equiv_manager_.find(e1) == edge_equiv_manager_.find(e2);
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

        // Build accessors (pass as references for dynamic extension)
        EdgeVertexAccessorRaw verts_accessor(this);
        EdgeArityAccessorRaw arities_accessor(this);

        if ((hash_strategy_ == HashStrategy::WL || hash_strategy_ == HashStrategy::IncrementalWL) && wl_hash_) {
            return wl_hash_->compute_state_hash(edges, verts_accessor, arities_accessor);
        } else if (hash_strategy_ == HashStrategy::IncrementalUniquenessTree && incremental_tree_) {
            // Note: For standalone hash computation (no parent state), incremental tree
            // behaves the same as non-incremental. The speedup comes from the incremental
            // API which requires parent state info.
            return incremental_tree_->compute_state_hash(edges, verts_accessor, arities_accessor);
        } else if (unified_tree_) {
            return unified_tree_->compute_state_hash(edges, verts_accessor, arities_accessor);
        }
        return 0;
    }

    // Compute canonical hash using UnifiedUniquenessTree (polynomial-time, approximate)
    // This is faster but may have rare false positives/negatives
    uint64_t compute_canonical_hash_wl(const SparseBitset& edges) {
        // Delegate to compute_canonical_hash_shared
        return compute_canonical_hash_shared(edges);
    }

    // Get or compute WL hash cache for a state (memoized)
    // Thread-safe: uses atomic valid flag for synchronization
    const VertexHashCache& get_or_compute_wl_cache(StateId state_id) {
        WLHashCacheEntry& entry = wl_hash_cache_.get_or_default(state_id, arena_);

        // Fast path: already computed
        if (entry.valid.load(std::memory_order_acquire)) {
            return entry.cache;
        }

        // Slow path: compute and cache
        const State& state = get_state(state_id);
        EdgeVertexAccessorRaw vert_acc(this);
        EdgeArityAccessorRaw arity_acc(this);

        auto [hash, cache] = wl_hash_->compute_state_hash_with_cache(
            state.edges, vert_acc, arity_acc);

        // Store the cache (may race with other threads, but result is same)
        entry.cache = cache;
        entry.valid.store(true, std::memory_order_release);

        return entry.cache;
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
                if (entry.valid.load(std::memory_order_acquire) && entry.cache.count > 0) {
                    parent_wl_cache = &entry.cache;
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
            // This ensures we see the vertex_cache data if valid == true
            if (pcache && pcache->valid.load(std::memory_order_acquire)) {
                const VertexHashCache& vc = pcache->vertex_cache;
                // Verify ALL required pointers are non-null before using
                if (vc.count > 0 && vc.vertices != nullptr &&
                    vc.hashes != nullptr && vc.subtree_filters != nullptr) {
                    parent_vertex_cache = &vc;
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

        // Heuristic 1: if many vertices are affected, the bloom filter path is unlikely
        // to provide reuse benefit but has significant overhead. Use fast path instead.
        // Threshold: if affected vertices >= 25% of parent cache, use fast path.
        if (affected_vertices.size() * 4 >= parent_vertex_cache->count) {
            return incremental_tree_->compute_state_hash_with_cache(
                new_edges, verts_accessor, arities_accessor);
        }

        // Heuristic 2: if parent cache is small, overhead of incremental path
        // exceeds any potential benefit. Use fast path for small states.
        if (parent_vertex_cache->count < 20) {
            return incremental_tree_->compute_state_hash_with_cache(
                new_edges, verts_accessor, arities_accessor);
        }

        // Heuristic 3: Sample a few bloom filters to estimate reuse potential.
        // If all samples indicate affected vertices, skip incremental path.
        // This catches connected graphs where bloom filter reuse won't help.
        {
            const uint32_t SAMPLE_SIZE = std::min(8u, parent_vertex_cache->count);
            uint32_t samples_affected = 0;
            uint32_t step = parent_vertex_cache->count / SAMPLE_SIZE;
            if (step == 0) step = 1;

            for (uint32_t i = 0; i < parent_vertex_cache->count && samples_affected < SAMPLE_SIZE; i += step) {
                const SubtreeBloomFilter* bloom = parent_vertex_cache->subtree_filters
                    ? &parent_vertex_cache->subtree_filters[i] : nullptr;
                if (bloom) {
                    for (VertexId affected : affected_vertices) {
                        if (bloom->might_contain(affected)) {
                            ++samples_affected;
                            break;
                        }
                    }
                }
            }

            // If all samples are affected, likely a connected graph - use fast path
            if (samples_affected >= SAMPLE_SIZE) {
                return incremental_tree_->compute_state_hash_with_cache(
                    new_edges, verts_accessor, arities_accessor);
            }
        }

        // Bloom filter reuse path - should have good reuse potential if we reach here

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
    // Thread-safe: uses compare-exchange to ensure only one thread writes.
    // Prevents race condition where two threads try to store the same state,
    // which could cause torn reads of the vertex_cache struct.
    void store_state_cache(StateId state, const VertexHashCache& cache) {
        // Store for IncrementalWL strategy
        if (hash_strategy_ == HashStrategy::IncrementalWL) {
            WLHashCacheEntry& slot = wl_hash_cache_.get_or_default(state, arena_);

            bool expected = false;
            if (!slot.valid.compare_exchange_strong(expected, false,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                return;  // Another thread already stored - skip
            }

            slot.cache = cache;
            slot.valid.store(true, std::memory_order_release);
            return;
        }

        // Store for iUT strategy
        if (hash_strategy_ != HashStrategy::IncrementalUniquenessTree) {
            return;  // Only store for incremental strategies (iUT, iWL)
        }

        StateIncrementalCache& slot = state_incremental_cache_.get_or_default(state, arena_);

        // Try to claim this slot atomically. If another thread already set valid=true,
        // skip - we don't want to race on the non-atomic vertex_cache struct.
        bool expected = false;
        if (!slot.valid.compare_exchange_strong(expected, false,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
            return;  // Another thread already stored (or is storing) - skip
        }

        // We claimed the slot (valid is still false), now populate it
        slot.vertex_cache = cache;

        // Release store ensures all data writes are visible before valid becomes true
        slot.valid.store(true, std::memory_order_release);
    }

    // Get number of stored caches (for debugging/profiling)
    size_t num_stored_caches() const {
        size_t count = 0;
        for (size_t i = 0; i < state_incremental_cache_.size(); ++i) {
            if (state_incremental_cache_[i].valid.load(std::memory_order_relaxed)) {
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

}  // namespace hypergraph::unified
