#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "types.hpp"
#include "bitset.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"

namespace hypergraph {

// =============================================================================
// IncrementalUniquenessTree: Incremental Gorard-style Uniqueness Trees
// =============================================================================
//
// This is an INCREMENTAL version of UniquenessTree that caches:
// 1. Persistent adjacency index (updated incrementally on edge add/remove)
// 2. Per-vertex subtree hashes (invalidated only when local neighborhood changes)
// 3. Dirty vertex tracking (only recompute affected vertices)
//
// For deep multiway graphs with large states and small rewrites, this can be
// 100-200x faster than recomputing everything from scratch.
//
// Key insight: When a rewrite consumes 2 edges and produces 3, only vertices
// within MAX_TREE_DEPTH hops of those edges need their subtree hashes recomputed.
//

class IncrementalUniquenessTree {
public:
    // Must match UniquenessTree::MAX_TREE_DEPTH for correct hash values!
    // The incremental benefit comes from sparse graphs with disconnected components,
    // not from reduced depth.
    static constexpr uint32_t MAX_TREE_DEPTH = 100;

    // NEW APPROACH: Lazy dirty detection during tree traversal
    // Instead of pre-propagating dirty flags (which marks everything on connected
    // graphs), we detect affected subtrees during traversal:
    //
    // 1. Mark "directly affected" vertices (on consumed/produced edges)
    // 2. During tree hash computation, track if subtree contains affected vertex
    // 3. If subtree is clean (no affected vertices), reuse parent's cached hash
    // 4. If subtree is dirty, recompute and cache for future reuse
    //
    // This gives O(affected_subtree_size) work instead of O(entire_graph).

    explicit IncrementalUniquenessTree(ConcurrentHeterogeneousArena* arena)
        : arena_(arena) {}

    // Abort flag for early termination of long-running hash computations
    void set_abort_flag(std::atomic<bool>* flag) { abort_flag_ = flag; }
    bool should_abort() const {
        return abort_flag_ && abort_flag_->load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Edge Registration (called when edges are added to hypergraph)
    // =========================================================================

    void register_edge(EdgeId edge_id, const VertexId* vertices, uint8_t arity) {
        // Store edge data for adjacency lookups (grows as needed)
        if (edge_id >= edge_vertices_.size()) {
            edge_vertices_.get_or_default(edge_id, *arena_);
            edge_arities_.get_or_default(edge_id, *arena_);
        }

        // Copy vertices to arena-allocated storage
        VertexId* verts = arena_->allocate_array<VertexId>(arity);
        for (uint8_t i = 0; i < arity; ++i) {
            verts[i] = vertices[i];
        }
        edge_vertices_[edge_id] = verts;
        edge_arities_[edge_id] = arity;

        // Record edge occurrence for each vertex
        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = vertices[i];
            EdgeOccurrence occ(edge_id, i, arity);
            vertex_occurrences_.get_or_default(v, *arena_).push(occ, *arena_);
        }
    }

    // =========================================================================
    // State Hash Computation (non-incremental, for initial states or fallback)
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_state_hash(
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        auto [hash, cache] = compute_state_hash_with_cache(state_edges, edge_vertices, edge_arities);
        return hash;
    }

    template<typename EdgeAccessor, typename ArityAccessor>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_with_cache(
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        VertexHashCache cache;

        if (state_edges.empty()) {
            return {0, cache};
        }

        // Collect vertices and track min/max for dense index optimization
        ArenaVector<VertexId> vertices(*arena_);
        VertexId max_vertex = 0;
        VertexId min_vertex = UINT32_MAX;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                vertices.push_back(verts[i]);
                if (verts[i] > max_vertex) max_vertex = verts[i];
                if (verts[i] < min_vertex) min_vertex = verts[i];
            }
        });

        // Remove duplicates
        std::sort(vertices.begin(), vertices.end());
        auto new_end = std::unique(vertices.begin(), vertices.end());
        vertices.resize(new_end - vertices.begin());

        if (vertices.empty()) {
            return {0, cache};
        }

        const size_t num_vertices = vertices.size();

        // Always use sparse index for simplicity and predictable behavior
        // Dense index optimization was removed as an unprincipled heuristic
        const bool use_dense_index = false;
        ArenaVector<size_t> dense_index(*arena_);  // Empty, unused

        std::unordered_map<VertexId, size_t> sparse_index;
        sparse_index.reserve(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            sparse_index[vertices[i]] = i;
        }

        // Lambda for vertex lookup
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            return sparse_index[v];
        };

        // Count total edge occurrences for pre-allocation
        size_t total_occurrences = 0;
        state_edges.for_each([&](EdgeId eid) {
            total_occurrences += edge_arities[eid];
        });

        // Arena-allocate flat adjacency structure
        ArenaVector<std::pair<EdgeId, uint8_t>> adj_data(*arena_, total_occurrences);
        ArenaVector<uint32_t> adj_counts(*arena_, num_vertices);
        adj_counts.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) adj_counts[i] = 0;

        // First pass: count occurrences per vertex
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                adj_counts[idx]++;
            }
        });

        // Compute offsets (prefix sum)
        ArenaVector<uint32_t> adj_offsets(*arena_, num_vertices + 1);
        adj_offsets.resize(num_vertices + 1);
        adj_offsets[0] = 0;
        for (size_t i = 0; i < num_vertices; ++i) {
            adj_offsets[i + 1] = adj_offsets[i] + adj_counts[i];
        }

        // Reset counts for second pass
        for (size_t i = 0; i < num_vertices; ++i) adj_counts[i] = 0;

        // Second pass: fill adjacency data
        adj_data.resize(total_occurrences);
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                size_t pos = adj_offsets[idx] + adj_counts[idx];
                adj_data[pos] = {eid, i};
                adj_counts[idx]++;
            }
        });

        // Allocate cache arrays including subtree bloom filters
        cache.capacity = static_cast<uint32_t>(num_vertices);
        cache.vertices = arena_->allocate_array<VertexId>(cache.capacity);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.capacity);
        cache.subtree_filters = arena_->allocate_array<SubtreeBloomFilter>(cache.capacity);
        cache.count = 0;

        // Compute uniqueness tree hash for each vertex
        ArenaVector<uint64_t> tree_hashes(*arena_, num_vertices);

        // Shared visited set (cleared automatically by tree hash after each use)
        SparseBitset visited;

        // Reusable buffers for tree hash computation - allocated once for all vertices
        TreeHashBuffers buffers(*arena_);
        buffers.init_if_needed(*arena_, MAX_TREE_DEPTH);

        for (size_t vi = 0; vi < num_vertices; ++vi) {
            if (should_abort()) throw AbortedException{};
            VertexId root = vertices[vi];
            SubtreeBloomFilter filter;
            filter.clear();

            uint64_t tree_hash = compute_tree_hash_flat_with_bloom(
                root, min_vertex, use_dense_index, dense_index, sparse_index,
                visited, buffers, filter,
                adj_data, adj_offsets, state_edges, edge_vertices, edge_arities);
            tree_hashes.push_back(tree_hash);
            cache.insert_with_subtree(root, tree_hash, filter);
        }

        // Combine tree hashes for state hash (sorted for canonical form)
        insertion_sort(tree_hashes.begin(), tree_hashes.end());

        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, cache};
    }

    // =========================================================================
    // INCREMENTAL State Hash Computation
    // =========================================================================
    //
    // Given a parent state's cached data, compute the child state's hash by
    // only recomputing subtree hashes for vertices whose subtrees contain
    // affected vertices.
    //
    // Key insight: Instead of pre-propagating dirty flags (which marks everything
    // on connected graphs), we detect affected subtrees during traversal using
    // lazy evaluation with memoization.
    //

    // =========================================================================
    // Incremental computation WITHOUT pre-built adjacency (uses vertex_occurrences_)
    // =========================================================================
    // This version avoids the O(parent_size) adjacency build cost by using the
    // global vertex_occurrences_ index combined with state_edges.test() checks.
    // This is the preferred method for incremental computation.

    template<typename EdgeAccessor, typename ArityAccessor>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_incremental_no_adjacency(
        const SparseBitset& child_edges,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& parent_cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        VertexHashCache cache;

        if (child_edges.empty()) {
            return {0, cache};
        }

        // Track DIRECTLY affected vertices (on consumed/produced edges only)
        std::unordered_set<VertexId> directly_affected;

        // Mark vertices on consumed edges as directly affected
        for (uint8_t i = 0; i < num_consumed; ++i) {
            EdgeId eid = consumed_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                directly_affected.insert(verts[j]);
            }
        }

        // Mark vertices on produced edges as directly affected
        for (uint8_t i = 0; i < num_produced; ++i) {
            EdgeId eid = produced_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                directly_affected.insert(verts[j]);
            }
        }

        // Collect all vertices in child state (using the global index)
        ArenaVector<VertexId> vertices(*arena_);
        std::unordered_set<VertexId> seen_vertices;

        child_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                if (seen_vertices.insert(verts[j]).second) {
                    vertices.push_back(verts[j]);
                }
            }
        });

        std::sort(vertices.begin(), vertices.end());

        if (vertices.size() == 0) {
            return {0, cache};
        }

        // Allocate cache arrays
        cache.capacity = static_cast<uint32_t>(vertices.size());
        cache.vertices = arena_->allocate_array<VertexId>(cache.capacity);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.capacity);
        cache.count = 0;

        // Compute tree hashes with lazy dirty detection
        std::unordered_map<VertexId, std::pair<uint64_t, bool>> child_memo;
        child_memo.reserve(vertices.size());  // Pre-size to avoid rehashing
        ArenaVector<uint64_t> tree_hashes(*arena_, vertices.size());

        size_t reused = 0;
        size_t recomputed = 0;

        for (VertexId root : vertices) {
            if (should_abort()) throw AbortedException{};
            auto [tree_hash, is_dirty] = compute_tree_hash_incremental_no_adjacency(
                root, child_edges, edge_vertices, edge_arities,
                directly_affected, parent_cache, child_memo);

            if (is_dirty) {
                ++recomputed;
            } else {
                ++reused;
            }

            tree_hashes.push_back(tree_hash);
            cache.insert(root, tree_hash);
        }

        // Update stats
        stats_reused_ += reused;
        stats_recomputed_ += recomputed;

        // Combine tree hashes for state hash (sorted for canonical form)
        std::sort(tree_hashes.begin(), tree_hashes.end());

        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, cache};
    }

    // =========================================================================
    // Incremental computation with external adjacency provider (PREFERRED)
    // =========================================================================
    // This version uses an external adjacency provider (e.g., Hypergraph's
    // global vertex_adjacency_ index) instead of the internal vertex_occurrences_.
    // This enables true global adjacency without per-strategy duplication.
    //
    // AdjacencyProvider must have: for_each_occurrence(VertexId v, Callback f)
    // where Callback takes EdgeOccurrence&

    template<typename EdgeAccessor, typename ArityAccessor, typename AdjacencyProvider>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_with_external_adjacency(
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const AdjacencyProvider& adjacency_provider
    ) {
        VertexHashCache cache;

        if (state_edges.empty()) {
            return {0, cache};
        }

        // Collect vertices and track min/max for dense index optimization
        ArenaVector<VertexId> vertices(*arena_);
        VertexId max_vertex = 0;
        VertexId min_vertex = UINT32_MAX;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                vertices.push_back(verts[i]);
                if (verts[i] > max_vertex) max_vertex = verts[i];
                if (verts[i] < min_vertex) min_vertex = verts[i];
            }
        });

        // Remove duplicates
        std::sort(vertices.begin(), vertices.end());
        auto new_end = std::unique(vertices.begin(), vertices.end());
        vertices.resize(new_end - vertices.begin());

        if (vertices.empty()) {
            return {0, cache};
        }

        // Allocate cache arrays
        cache.capacity = static_cast<uint32_t>(vertices.size());
        cache.vertices = arena_->allocate_array<VertexId>(cache.capacity);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.capacity);
        cache.subtree_filters = arena_->allocate_array<SubtreeBloomFilter>(cache.capacity);
        cache.count = 0;

        // Compute uniqueness tree hash for each vertex
        ArenaVector<uint64_t> tree_hashes(*arena_, vertices.size());
        SparseBitset visited;

        for (VertexId root : vertices) {
            if (should_abort()) throw AbortedException{};
            SubtreeBloomFilter filter;
            filter.clear();

            uint64_t tree_hash = compute_tree_hash_external_adjacency(
                root, state_edges, edge_vertices, edge_arities,
                adjacency_provider, visited, filter);
            tree_hashes.push_back(tree_hash);
            cache.insert_with_subtree(root, tree_hash, filter);
        }

        // Combine tree hashes for state hash (sorted for canonical form)
        insertion_sort(tree_hashes.begin(), tree_hashes.end());

        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, cache};
    }

    template<typename EdgeAccessor, typename ArityAccessor, typename AdjacencyProvider>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_incremental_external(
        const SparseBitset& child_edges,
        const ArenaVector<VertexId>& child_vertices,  // Pre-computed incrementally
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& parent_cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const AdjacencyProvider& adjacency_provider
    ) {
        VertexHashCache cache;

        if (child_vertices.empty()) {
            return {0, cache};
        }

        // Track DIRECTLY affected vertices (on consumed/produced edges only)
        std::unordered_set<VertexId> directly_affected;

        // Mark vertices on consumed edges as directly affected
        for (uint8_t i = 0; i < num_consumed; ++i) {
            EdgeId eid = consumed_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                directly_affected.insert(verts[j]);
            }
        }

        // Mark vertices on produced edges as directly affected
        for (uint8_t i = 0; i < num_produced; ++i) {
            EdgeId eid = produced_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                directly_affected.insert(verts[j]);
            }
        }

        // Use pre-computed vertices (already sorted)
        const ArenaVector<VertexId>& vertices = child_vertices;

        // Allocate cache arrays
        cache.capacity = static_cast<uint32_t>(vertices.size());
        cache.vertices = arena_->allocate_array<VertexId>(cache.capacity);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.capacity);
        cache.count = 0;

        // Compute tree hashes with lazy dirty detection
        std::unordered_map<VertexId, std::pair<uint64_t, bool>> child_memo;
        child_memo.reserve(vertices.size());  // Pre-size to avoid rehashing
        ArenaVector<uint64_t> tree_hashes(*arena_, vertices.size());

        size_t reused = 0;
        size_t recomputed = 0;

        for (VertexId root : vertices) {
            if (should_abort()) throw AbortedException{};
            auto [tree_hash, is_dirty] = compute_tree_hash_incremental_external(
                root, child_edges, edge_vertices, edge_arities,
                adjacency_provider, directly_affected, parent_cache, child_memo);

            if (is_dirty) {
                ++recomputed;
            } else {
                ++reused;
            }

            tree_hashes.push_back(tree_hash);
            cache.insert(root, tree_hash);
        }

        // Update stats
        stats_reused_ += reused;
        stats_recomputed_ += recomputed;

        // Combine tree hashes for state hash (sorted for canonical form)
        std::sort(tree_hashes.begin(), tree_hashes.end());

        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, cache};
    }

    // =========================================================================
    // Original incremental computation (requires pre-built adjacency)
    // =========================================================================
    // DEPRECATED: prefer compute_state_hash_incremental_external with global adjacency

    // =========================================================================
    // DeltaAdjacency: Zero-copy adjacency view with delta applied
    // =========================================================================
    // Provides child state's adjacency by wrapping parent adjacency and applying
    // consumed/produced edges as a delta. O(delta) construction, O(1) per lookup.
    //
    template<typename EdgeAccessor, typename ArityAccessor>
    class DeltaAdjacency {
    public:
        using AdjList = ArenaVector<std::pair<EdgeId, uint8_t>>;
        using AdjMap = std::unordered_map<VertexId, AdjList>;

        DeltaAdjacency(
            const AdjMap& parent_adj,
            const EdgeId* consumed, uint8_t num_consumed,
            const EdgeId* produced, uint8_t num_produced,
            const EdgeAccessor& edge_vertices,
            const ArityAccessor& edge_arities,
            ConcurrentHeterogeneousArena& arena
        ) : parent_adj_(parent_adj), arena_(arena) {
            // Build set of consumed edge IDs for O(1) lookup
            for (uint8_t i = 0; i < num_consumed; ++i) {
                consumed_edges_.insert(consumed[i]);
            }

            // Build additions per vertex from produced edges
            for (uint8_t i = 0; i < num_produced; ++i) {
                EdgeId eid = produced[i];
                uint8_t arity = edge_arities[eid];
                const VertexId* verts = edge_vertices[eid];
                for (uint8_t j = 0; j < arity; ++j) {
                    VertexId v = verts[j];
                    auto it = additions_.find(v);
                    if (it == additions_.end()) {
                        it = additions_.emplace(v, AdjList(arena_)).first;
                    }
                    it->second.push_back({eid, j});
                }
            }
        }

        // Get adjacency list for vertex, applying delta on-the-fly
        // Returns a temporary view - caller should not store the pointer
        template<typename Callback>
        void for_each_adjacent(VertexId v, Callback&& cb) const {
            // First, iterate parent's adjacency (filtering out consumed)
            auto parent_it = parent_adj_.find(v);
            if (parent_it != parent_adj_.end()) {
                for (const auto& [eid, pos] : parent_it->second) {
                    if (consumed_edges_.find(eid) == consumed_edges_.end()) {
                        cb(eid, pos);
                    }
                }
            }

            // Then, add produced edges
            auto add_it = additions_.find(v);
            if (add_it != additions_.end()) {
                for (const auto& [eid, pos] : add_it->second) {
                    cb(eid, pos);
                }
            }
        }

        // Check if vertex has any adjacency
        bool has_adjacency(VertexId v) const {
            // Check parent (excluding consumed)
            auto parent_it = parent_adj_.find(v);
            if (parent_it != parent_adj_.end()) {
                for (const auto& [eid, _] : parent_it->second) {
                    if (consumed_edges_.find(eid) == consumed_edges_.end()) {
                        return true;
                    }
                }
            }
            // Check additions
            auto add_it = additions_.find(v);
            return add_it != additions_.end() && !add_it->second.empty();
        }

    private:
        const AdjMap& parent_adj_;
        ConcurrentHeterogeneousArena& arena_;
        std::unordered_set<EdgeId> consumed_edges_;
        std::unordered_map<VertexId, AdjList> additions_;
    };

    template<typename EdgeAccessor, typename ArityAccessor>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_incremental(
        const SparseBitset& child_edges,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& parent_cache,
        const std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>>& parent_adjacency,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        VertexHashCache cache;

        // Create delta-based adjacency view (O(delta) not O(parent_size))
        DeltaAdjacency<EdgeAccessor, ArityAccessor> adjacency(
            parent_adjacency,
            consumed_edges, num_consumed,
            produced_edges, num_produced,
            edge_vertices, edge_arities,
            *arena_
        );

        // Track DIRECTLY affected vertices (on consumed/produced edges only)
        std::unordered_set<VertexId> directly_affected;

        for (uint8_t i = 0; i < num_consumed; ++i) {
            EdgeId eid = consumed_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                directly_affected.insert(verts[j]);
            }
        }

        for (uint8_t i = 0; i < num_produced; ++i) {
            EdgeId eid = produced_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                directly_affected.insert(verts[j]);
            }
        }

        // Collect all vertices in child state from child_edges
        ArenaVector<VertexId> vertices(*arena_);
        std::unordered_set<VertexId> seen_vertices;
        child_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                if (seen_vertices.insert(verts[j]).second) {
                    vertices.push_back(verts[j]);
                }
            }
        });
        std::sort(vertices.begin(), vertices.end());

        if (vertices.size() == 0) {
            return {0, cache};
        }

        // Allocate cache arrays
        cache.capacity = static_cast<uint32_t>(vertices.size());
        cache.vertices = arena_->allocate_array<VertexId>(cache.capacity);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.capacity);
        cache.count = 0;

        // Step 4: Compute tree hashes with lazy dirty detection
        // Memoization map for vertices computed in this child state
        std::unordered_map<VertexId, std::pair<uint64_t, bool>> child_memo;  // vertex -> (hash, is_dirty)
        child_memo.reserve(vertices.size());  // Pre-size to avoid rehashing

        ArenaVector<uint64_t> tree_hashes(*arena_, vertices.size());

        size_t reused = 0;
        size_t recomputed = 0;

        for (VertexId root : vertices) {
            auto [tree_hash, is_dirty] = compute_tree_hash_incremental_delta(
                root, child_edges, adjacency, edge_vertices, edge_arities,
                directly_affected, parent_cache, child_memo);

            if (is_dirty) {
                ++recomputed;
            } else {
                ++reused;
            }

            tree_hashes.push_back(tree_hash);
            cache.insert(root, tree_hash);
        }

        // Update stats
        stats_reused_ += reused;
        stats_recomputed_ += recomputed;

        // Combine tree hashes for state hash (sorted for canonical form)
        std::sort(tree_hashes.begin(), tree_hashes.end());

        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, cache};
    }

    // =========================================================================
    // Incremental computation with adjacency output (for chaining)
    // =========================================================================
    // DEPRECATED: This version copies parent adjacency which is O(parent_size).
    // Use compute_state_hash_incremental instead which uses DeltaAdjacency.

    template<typename EdgeAccessor, typename ArityAccessor>
    std::tuple<uint64_t, VertexHashCache, std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>>>
    compute_state_hash_incremental_with_adjacency(
        const SparseBitset& child_edges,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& parent_cache,
        const std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>>& parent_adjacency,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        VertexHashCache cache;
        std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>> adjacency;

        // Copy parent adjacency
        for (const auto& [v, adj_list] : parent_adjacency) {
            auto& new_list = adjacency.emplace(v, ArenaVector<std::pair<EdgeId, uint8_t>>(*arena_)).first->second;
            for (size_t i = 0; i < adj_list.size(); ++i) {
                new_list.push_back(adj_list[i]);
            }
        }

        // Track DIRECTLY affected vertices only (no propagation)
        std::unordered_set<VertexId> directly_affected;

        // Remove consumed edges
        for (uint8_t i = 0; i < num_consumed; ++i) {
            EdgeId eid = consumed_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            for (uint8_t j = 0; j < arity; ++j) {
                VertexId v = verts[j];
                directly_affected.insert(v);

                auto it = adjacency.find(v);
                if (it != adjacency.end()) {
                    auto& adj_list = it->second;
                    for (size_t k = 0; k < adj_list.size(); ) {
                        if (adj_list[k].first == eid) {
                            adj_list[k] = adj_list[adj_list.size() - 1];
                            adj_list.resize(adj_list.size() - 1);
                        } else {
                            ++k;
                        }
                    }
                    if (adj_list.size() == 0) {
                        adjacency.erase(it);
                    }
                }
            }
        }

        // Add produced edges
        for (uint8_t i = 0; i < num_produced; ++i) {
            EdgeId eid = produced_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            for (uint8_t j = 0; j < arity; ++j) {
                VertexId v = verts[j];
                directly_affected.insert(v);

                auto it = adjacency.find(v);
                if (it == adjacency.end()) {
                    it = adjacency.emplace(v, ArenaVector<std::pair<EdgeId, uint8_t>>(*arena_)).first;
                }
                it->second.push_back({eid, j});
            }
        }

        // Collect vertices
        ArenaVector<VertexId> vertices(*arena_);
        vertices.reserve(adjacency.size());
        for (const auto& [v, _] : adjacency) {
            vertices.push_back(v);
        }
        std::sort(vertices.begin(), vertices.end());

        if (vertices.size() == 0) {
            return {0, cache, adjacency};
        }

        cache.capacity = static_cast<uint32_t>(vertices.size());
        cache.vertices = arena_->allocate_array<VertexId>(cache.capacity);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.capacity);
        cache.count = 0;

        // Use lazy dirty detection
        std::unordered_map<VertexId, std::pair<uint64_t, bool>> child_memo;
        child_memo.reserve(vertices.size());  // Pre-size to avoid rehashing
        ArenaVector<uint64_t> tree_hashes(*arena_, vertices.size());

        for (VertexId root : vertices) {
            auto [tree_hash, is_dirty] = compute_tree_hash_incremental(
                root, child_edges, adjacency, edge_vertices, edge_arities,
                directly_affected, parent_cache, child_memo);

            if (is_dirty) {
                ++stats_recomputed_;
            } else {
                ++stats_reused_;
            }

            tree_hashes.push_back(tree_hash);
            cache.insert(root, tree_hash);
        }

        std::sort(tree_hashes.begin(), tree_hashes.end());

        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, cache, std::move(adjacency)};
    }

    // =========================================================================
    // Build adjacency from scratch (for initial states)
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>> build_adjacency(
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>> adjacency;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                VertexId v = verts[i];
                auto it = adjacency.find(v);
                if (it == adjacency.end()) {
                    it = adjacency.emplace(v, ArenaVector<std::pair<EdgeId, uint8_t>>(*arena_)).first;
                }
                it->second.push_back({eid, i});
            }
        });

        return adjacency;
    }

    // =========================================================================
    // Edge Correspondence (same as non-incremental version)
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    EdgeCorrespondence find_edge_correspondence(
        const SparseBitset& state1_edges,
        const SparseBitset& state2_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        EdgeCorrespondence result;

        auto [hash1, cache1] = compute_state_hash_with_cache(state1_edges, edge_vertices, edge_arities);
        auto [hash2, cache2] = compute_state_hash_with_cache(state2_edges, edge_vertices, edge_arities);

        if (hash1 != hash2) {
            return result;
        }

        std::unordered_map<uint64_t, ArenaVector<EdgeId>> edge2_by_sig;

        state2_edges.for_each([&](EdgeId eid) {
            uint64_t sig = compute_edge_signature(eid, cache2, edge_vertices, edge_arities);
            auto it = edge2_by_sig.find(sig);
            if (it == edge2_by_sig.end()) {
                it = edge2_by_sig.emplace(sig, ArenaVector<EdgeId>(*arena_)).first;
            }
            it->second.push_back(eid);
        });

        ArenaVector<EdgeId> edges1(*arena_);
        state1_edges.for_each([&](EdgeId eid) {
            edges1.push_back(eid);
        });

        size_t edge2_count = 0;
        state2_edges.for_each([&](EdgeId) { ++edge2_count; });

        if (edges1.size() != edge2_count) {
            return result;
        }

        result.count = static_cast<uint32_t>(edges1.size());
        result.state1_edges = arena_->allocate_array<EdgeId>(result.count);
        result.state2_edges = arena_->allocate_array<EdgeId>(result.count);

        std::unordered_map<uint64_t, size_t> sig_next_idx;

        for (size_t i = 0; i < edges1.size(); ++i) {
            EdgeId e1 = edges1[i];
            uint64_t sig = compute_edge_signature(e1, cache1, edge_vertices, edge_arities);

            auto e2_it = edge2_by_sig.find(sig);
            if (e2_it == edge2_by_sig.end()) {
                result.valid = false;
                return result;
            }

            size_t& idx = sig_next_idx[sig];
            if (idx >= e2_it->second.size()) {
                result.valid = false;
                return result;
            }

            result.state1_edges[i] = e1;
            result.state2_edges[i] = e2_it->second[idx];
            ++idx;
        }

        result.valid = true;
        return result;
    }

    // =========================================================================
    // Event Signature Computation
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    EventSignature compute_event_signature(
        uint64_t input_state_hash,
        uint64_t output_state_hash,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& input_cache,
        const VertexHashCache& output_cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        EventSignature sig;
        sig.input_state_hash = input_state_hash;
        sig.output_state_hash = output_state_hash;

        sig.consumed_edges_sig = compute_edge_set_signature(
            consumed_edges, num_consumed, input_cache, edge_vertices, edge_arities);
        sig.produced_edges_sig = compute_edge_set_signature(
            produced_edges, num_produced, output_cache, edge_vertices, edge_arities);

        return sig;
    }

    // =========================================================================
    // Tree Hash with External Adjacency Provider (for chained delta adjacency)
    // =========================================================================
    // Computes a single vertex's tree hash using an external adjacency provider.
    // The provider must support: for_each_adjacent(vertex, callback)
    // where callback takes (EdgeId, position).

    template<typename EdgeAccessor, typename ArityAccessor, typename AdjacencyProvider>
    uint64_t compute_tree_hash_with_adjacency_provider(
        VertexId root,
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const AdjacencyProvider& adjacency
    ) {
        SparseBitset visited;

        ArenaVector<DFSFrame> stack(*arena_);
        stack.reserve(MAX_TREE_DEPTH + 1);

        ArenaVector<ArenaVector<ChildInfo>> children_by_level(*arena_);
        children_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            children_by_level.push_back(ArenaVector<ChildInfo>(*arena_));
        }

        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level(*arena_);
        child_hashes_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            child_hashes_by_level.push_back(ArenaVector<uint64_t>(*arena_));
        }

        ArenaVector<uint8_t> scratch_positions(*arena_);
        ArenaVector<uint16_t> scratch_occ_positions(*arena_);

        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            // Check for abort periodically
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);

                scratch_positions.clear();
                adjacency.for_each_adjacent(frame.vertex, [&](EdgeId eid, uint8_t pos) {
                    scratch_positions.push_back(pos);
                });
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);
                frame.own_positions_count = scratch_positions.size();

                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                std::unordered_map<VertexId, ArenaVector<std::pair<uint8_t, uint8_t>>> adjacent_map;

                adjacency.for_each_adjacent(frame.vertex, [&](EdgeId eid, uint8_t my_pos) {
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t i = 0; i < arity; ++i) {
                        if (i != my_pos) {
                            VertexId adj_v = verts[i];
                            if (!visited.contains(adj_v)) {
                                auto it = adjacent_map.find(adj_v);
                                if (it == adjacent_map.end()) {
                                    it = adjacent_map.emplace(adj_v, ArenaVector<std::pair<uint8_t, uint8_t>>(*arena_)).first;
                                }
                                it->second.push_back({arity, i});
                            }
                        }
                    }
                });

                for (const auto& [adj_v, occs] : adjacent_map) {
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.is_unique = (occs.size() == 1);
                    child.num_occurrences = static_cast<uint8_t>(occs.size());

                    scratch_occ_positions.clear();
                    for (size_t k = 0; k < occs.size(); ++k) {
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(occs[k].first) << 8) | occs[k].second);
                    }
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);
                    child_hash = fnv_hash(child_hash, 0);
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            visited.clear(frame.vertex);

            scratch_positions.clear();
            adjacency.for_each_adjacent(frame.vertex, [&](EdgeId eid, uint8_t pos) {
                scratch_positions.push_back(pos);
            });
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);
            hash = fnv_hash(hash, child_hashes.size());

            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            if (!stack.empty()) {
                DFSFrame& parent_frame = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent_frame.level];
                auto& parent_child_hashes = child_hashes_by_level[parent_frame.level];

                ChildInfo& child_info = parent_children[parent_frame.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent_frame.child_idx++;
            }
        }

        return result;
    }

    // =========================================================================
    // Statistics (for profiling)
    // =========================================================================

    size_t stats_reused() const { return stats_reused_; }
    size_t stats_recomputed() const { return stats_recomputed_; }
    void reset_stats() { stats_reused_ = 0; stats_recomputed_ = 0; }

    double reuse_ratio() const {
        size_t total = stats_reused_ + stats_recomputed_;
        return total > 0 ? static_cast<double>(stats_reused_) / total : 0.0;
    }

    // =========================================================================
    // Public Tree Hash with External Adjacency (for bloom filter reuse)
    // =========================================================================
    // Computes a single vertex's tree hash using an external adjacency provider.
    // This is a public wrapper around the private compute_tree_hash_external_adjacency.
    //
    // The provider must support: for_each_occurrence(vertex, callback)
    // where callback takes EdgeOccurrence&
    //
    // Also populates a bloom filter with all vertices visited in the subtree.

    template<typename EdgeAccessor, typename ArityAccessor, typename AdjacencyProvider>
    uint64_t compute_tree_hash_with_bloom(
        VertexId root,
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const AdjacencyProvider& adjacency_provider,
        SparseBitset& visited,
        SubtreeBloomFilter& bloom_filter
    ) {
        return compute_tree_hash_external_adjacency(
            root, state_edges, edge_vertices, edge_arities,
            adjacency_provider, visited, bloom_filter);
    }

private:
    ConcurrentHeterogeneousArena* arena_;
    std::atomic<bool>* abort_flag_{nullptr};

    // Per-vertex edge occurrences
    SegmentedArray<LockFreeList<EdgeOccurrence>> vertex_occurrences_;

    // Edge data
    SegmentedArray<VertexId*> edge_vertices_;
    SegmentedArray<uint8_t> edge_arities_;

    // Statistics
    mutable size_t stats_reused_ = 0;
    mutable size_t stats_recomputed_ = 0;

    // =========================================================================
    // Incremental Tree Hash with Bloom Filter Dirty Detection
    // =========================================================================
    //
    // Returns (hash, is_dirty) where is_dirty indicates if this subtree
    // contains any directly affected vertices.
    //
    // Uses bloom filters for O(num_affected) dirty detection instead of O(graph_size) BFS.
    // The bloom filter stored with each vertex's cached hash records which vertices
    // are in that vertex's DFS subtree. We check if any affected vertex might be
    // in the subtree using the bloom filter.
    //
    // False positives are possible (may recompute when not needed), but no false
    // negatives (never incorrectly reuse a stale hash).
    //

    // Version that works with DeltaAdjacency (zero-copy, O(delta) construction)
    template<typename EdgeAccessor, typename ArityAccessor>
    std::pair<uint64_t, bool> compute_tree_hash_incremental_delta(
        VertexId root,
        const SparseBitset& state_edges,
        const DeltaAdjacency<EdgeAccessor, ArityAccessor>& adjacency,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const std::unordered_set<VertexId>& directly_affected,
        const VertexHashCache& parent_cache,
        std::unordered_map<VertexId, std::pair<uint64_t, bool>>& child_memo
    ) {
        // Check memo first
        auto memo_it = child_memo.find(root);
        if (memo_it != child_memo.end()) {
            return memo_it->second;
        }

        // Check if this vertex is directly affected
        bool is_directly_affected = (directly_affected.find(root) != directly_affected.end());

        // FAST PATH: Use bloom filter for O(num_affected) dirty detection
        auto [cached_hash, bloom_filter] = parent_cache.lookup_with_subtree(root);

        if (!is_directly_affected && cached_hash != 0 && bloom_filter != nullptr) {
            // O(num_affected) check: does the bloom filter contain any affected vertex?
            bool might_be_dirty = bloom_filter->might_contain_any(directly_affected);

            if (!might_be_dirty) {
                // Bloom filter says NO affected vertices in subtree - safe to reuse
                auto result = std::make_pair(cached_hash, false);
                child_memo[root] = result;
                return result;
            }
            // Bloom filter says MAYBE - could be false positive, must recompute to be safe
        }

        // SLOW PATH: Vertex is affected OR bloom filter says maybe dirty - recompute
        uint64_t hash = compute_tree_hash_delta(root, state_edges, adjacency, edge_vertices, edge_arities);
        auto result = std::make_pair(hash, true);  // Mark as dirty
        child_memo[root] = result;
        return result;
    }

    // Tree hash computation using DeltaAdjacency
    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_tree_hash_delta(
        VertexId root,
        const SparseBitset& state_edges,
        const DeltaAdjacency<EdgeAccessor, ArityAccessor>& adjacency,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        SparseBitset visited;

        ArenaVector<DFSFrame> stack(*arena_);
        stack.reserve(MAX_TREE_DEPTH + 1);

        ArenaVector<ArenaVector<ChildInfo>> children_by_level(*arena_);
        children_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            children_by_level.push_back(ArenaVector<ChildInfo>(*arena_));
        }

        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level(*arena_);
        child_hashes_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            child_hashes_by_level.push_back(ArenaVector<uint64_t>(*arena_));
        }

        ArenaVector<uint8_t> scratch_positions(*arena_);
        ArenaVector<uint16_t> scratch_occ_positions(*arena_);

        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            // Check for abort periodically
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            // First time visiting this frame?
            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);

                // Collect own positions
                scratch_positions.clear();
                adjacency.for_each_adjacent(frame.vertex, [&](EdgeId eid, uint8_t pos) {
                    scratch_positions.push_back(pos);
                });
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);
                frame.own_positions_count = scratch_positions.size();

                // Build children list using DeltaAdjacency
                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                std::unordered_map<VertexId, ArenaVector<std::pair<uint8_t, uint8_t>>> adjacent_map;

                adjacency.for_each_adjacent(frame.vertex, [&](EdgeId eid, uint8_t my_pos) {
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t i = 0; i < arity; ++i) {
                        if (i != my_pos) {
                            VertexId adj_v = verts[i];
                            if (!visited.contains(adj_v)) {
                                auto it = adjacent_map.find(adj_v);
                                if (it == adjacent_map.end()) {
                                    it = adjacent_map.emplace(adj_v, ArenaVector<std::pair<uint8_t, uint8_t>>(*arena_)).first;
                                }
                                it->second.push_back({arity, i});
                            }
                        }
                    }
                });

                // Convert to children list
                for (const auto& [adj_v, occs] : adjacent_map) {
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.is_unique = (occs.size() == 1);
                    child.num_occurrences = static_cast<uint8_t>(occs.size());

                    scratch_occ_positions.clear();
                    for (size_t k = 0; k < occs.size(); ++k) {
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(occs[k].first) << 8) | occs[k].second);
                    }
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            // Process next child
            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);
                    child_hash = fnv_hash(child_hash, 0);
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            // All children processed - compute hash
            visited.clear(frame.vertex);

            // Re-collect own positions for final hash
            scratch_positions.clear();
            adjacency.for_each_adjacent(frame.vertex, [&](EdgeId eid, uint8_t pos) {
                scratch_positions.push_back(pos);
            });
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);
            hash = fnv_hash(hash, child_hashes.size());

            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            if (!stack.empty()) {
                DFSFrame& parent_frame = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent_frame.level];
                auto& parent_child_hashes = child_hashes_by_level[parent_frame.level];

                ChildInfo& child_info = parent_children[parent_frame.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent_frame.child_idx++;
            }
        }

        return result;
    }

    // Original version that works with std::unordered_map adjacency
    template<typename EdgeAccessor, typename ArityAccessor>
    std::pair<uint64_t, bool> compute_tree_hash_incremental(
        VertexId root,
        const SparseBitset& state_edges,
        const std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>>& adjacency,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const std::unordered_set<VertexId>& directly_affected,
        const VertexHashCache& parent_cache,
        std::unordered_map<VertexId, std::pair<uint64_t, bool>>& child_memo
    ) {
        // Check memo first
        auto memo_it = child_memo.find(root);
        if (memo_it != child_memo.end()) {
            return memo_it->second;
        }

        // Check if this vertex is directly affected
        bool is_directly_affected = (directly_affected.find(root) != directly_affected.end());

        // FAST PATH: Use bloom filter for O(num_affected) dirty detection
        auto [cached_hash, bloom_filter] = parent_cache.lookup_with_subtree(root);

        if (!is_directly_affected && cached_hash != 0 && bloom_filter != nullptr) {
            // O(num_affected) check: does the bloom filter contain any affected vertex?
            bool might_be_dirty = bloom_filter->might_contain_any(directly_affected);

            if (!might_be_dirty) {
                // Bloom filter says NO affected vertices in subtree - safe to reuse
                auto result = std::make_pair(cached_hash, false);
                child_memo[root] = result;
                return result;
            }
            // Bloom filter says MAYBE - could be false positive, must recompute to be safe
        }

        // SLOW PATH: Vertex is affected OR bloom filter says maybe dirty - recompute
        uint64_t hash = compute_tree_hash(root, state_edges, adjacency, edge_vertices, edge_arities);
        auto result = std::make_pair(hash, true);  // Mark as dirty
        child_memo[root] = result;
        return result;
    }

    // =========================================================================
    // Tree Hash Computation - NO ADJACENCY VERSION (uses vertex_occurrences_)
    // =========================================================================
    // This version uses the global vertex_occurrences_ index with state_edges.test()
    // instead of a pre-built per-state adjacency map.

    template<typename EdgeAccessor, typename ArityAccessor>
    std::pair<uint64_t, bool> compute_tree_hash_incremental_no_adjacency(
        VertexId root,
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const std::unordered_set<VertexId>& directly_affected,
        const VertexHashCache& parent_cache,
        std::unordered_map<VertexId, std::pair<uint64_t, bool>>& child_memo
    ) {
        // Check memo first
        auto memo_it = child_memo.find(root);
        if (memo_it != child_memo.end()) {
            return memo_it->second;
        }

        // Check if this vertex is directly affected
        bool is_directly_affected = (directly_affected.find(root) != directly_affected.end());

        // FAST PATH: Use bloom filter for O(num_affected) dirty detection
        auto [cached_hash, bloom_filter] = parent_cache.lookup_with_subtree(root);

        if (!is_directly_affected && cached_hash != 0 && bloom_filter != nullptr) {
            // O(num_affected) check: does the bloom filter contain any affected vertex?
            bool might_be_dirty = bloom_filter->might_contain_any(directly_affected);

            if (!might_be_dirty) {
                // Bloom filter says NO affected vertices in subtree - safe to reuse
                auto result = std::make_pair(cached_hash, false);
                child_memo[root] = result;
                return result;
            }
            // Bloom filter says MAYBE - could be false positive, must recompute to be safe
        }

        // SLOW PATH: Vertex is affected OR bloom filter says maybe dirty - recompute
        uint64_t hash = compute_tree_hash_no_adjacency(root, state_edges, edge_vertices, edge_arities);
        auto result = std::make_pair(hash, true);  // Mark as dirty
        child_memo[root] = result;
        return result;
    }

    // =========================================================================
    // Tree Hash Computation - EXTERNAL ADJACENCY PROVIDER VERSION
    // =========================================================================
    // Uses an external adjacency provider (e.g., Hypergraph::vertex_adjacency_)
    // instead of the internal vertex_occurrences_. Provider must support:
    //   provider.for_each_occurrence(vertex, [](const EdgeOccurrence& occ) { ... });

    template<typename EdgeAccessor, typename ArityAccessor, typename AdjacencyProvider>
    std::pair<uint64_t, bool> compute_tree_hash_incremental_external(
        VertexId root,
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const AdjacencyProvider& adjacency_provider,
        const std::unordered_set<VertexId>& directly_affected,
        const VertexHashCache& parent_cache,
        std::unordered_map<VertexId, std::pair<uint64_t, bool>>& child_memo
    ) {
        // Check memo first
        auto memo_it = child_memo.find(root);
        if (memo_it != child_memo.end()) {
            return memo_it->second;
        }

        // Check if this vertex is directly affected
        bool is_directly_affected = (directly_affected.find(root) != directly_affected.end());

        // FAST PATH: Use bloom filter for O(num_affected) dirty detection
        auto [cached_hash, bloom_filter] = parent_cache.lookup_with_subtree(root);

        if (!is_directly_affected && cached_hash != 0 && bloom_filter != nullptr) {
            // O(num_affected) check: does the bloom filter contain any affected vertex?
            bool might_be_dirty = bloom_filter->might_contain_any(directly_affected);

            if (!might_be_dirty) {
                // Bloom filter says NO affected vertices in subtree - safe to reuse
                auto result = std::make_pair(cached_hash, false);
                child_memo[root] = result;
                return result;
            }
            // Bloom filter says MAYBE - could be false positive, must recompute to be safe
        }

        // SLOW PATH: Vertex is affected OR bloom filter says maybe dirty - recompute
        SparseBitset visited;
        SubtreeBloomFilter unused_filter;  // Not used for incremental path
        uint64_t hash = compute_tree_hash_external_adjacency(
            root, state_edges, edge_vertices, edge_arities,
            adjacency_provider, visited, unused_filter);
        auto result = std::make_pair(hash, true);  // Mark as dirty
        child_memo[root] = result;
        return result;
    }

    // Tree hash computation using external adjacency provider
    template<typename EdgeAccessor, typename ArityAccessor, typename AdjacencyProvider>
    uint64_t compute_tree_hash_external_adjacency(
        VertexId root,
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        const AdjacencyProvider& adjacency_provider,
        SparseBitset& visited,
        SubtreeBloomFilter& bloom_filter
    ) {
        ArenaVector<DFSFrame> stack(*arena_);
        stack.reserve(MAX_TREE_DEPTH + 1);

        ArenaVector<ArenaVector<ChildInfo>> children_by_level(*arena_);
        children_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            children_by_level.push_back(ArenaVector<ChildInfo>(*arena_));
        }

        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level(*arena_);
        child_hashes_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            child_hashes_by_level.push_back(ArenaVector<uint64_t>(*arena_));
        }

        ArenaVector<uint8_t> scratch_positions(*arena_);
        ArenaVector<uint16_t> scratch_occ_positions(*arena_);

        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            // Check for abort periodically
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            // First time visiting this frame?
            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);
                bloom_filter.add(frame.vertex);

                // Collect own positions using external adjacency provider + state_edges check
                scratch_positions.clear();
                adjacency_provider.for_each_occurrence(frame.vertex, [&](const EdgeOccurrence& occ) {
                    if (state_edges.contains(occ.edge_id)) {
                        scratch_positions.push_back(occ.position);
                    }
                });
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);
                frame.own_positions_count = scratch_positions.size();

                // Build children list using external adjacency provider
                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                std::unordered_map<VertexId, ArenaVector<std::pair<uint8_t, uint8_t>>> adjacent_map;

                adjacency_provider.for_each_occurrence(frame.vertex, [&](const EdgeOccurrence& occ) {
                    if (!state_edges.contains(occ.edge_id)) {
                        return;  // Edge not in this state
                    }

                    EdgeId eid = occ.edge_id;
                    uint8_t my_pos = occ.position;
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t i = 0; i < arity; ++i) {
                        if (i != my_pos) {
                            VertexId adj_v = verts[i];
                            if (!visited.contains(adj_v)) {
                                auto it = adjacent_map.find(adj_v);
                                if (it == adjacent_map.end()) {
                                    it = adjacent_map.emplace(adj_v, ArenaVector<std::pair<uint8_t, uint8_t>>(*arena_)).first;
                                }
                                it->second.push_back({arity, i});
                            }
                        }
                    }
                });

                // Convert to children list
                for (const auto& [adj_v, occs] : adjacent_map) {
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.is_unique = (occs.size() == 1);
                    child.num_occurrences = static_cast<uint8_t>(occs.size());

                    scratch_occ_positions.clear();
                    for (size_t k = 0; k < occs.size(); ++k) {
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(occs[k].first) << 8) | occs[k].second);
                    }
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    // Non-unique children are also in the subtree (as leaves)
                    if (!child.is_unique) {
                        bloom_filter.add(adj_v);
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            // Process next child
            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);
                    child_hash = fnv_hash(child_hash, 0);
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            // All children processed - compute hash
            visited.clear(frame.vertex);

            // Re-collect own positions for final hash
            scratch_positions.clear();
            adjacency_provider.for_each_occurrence(frame.vertex, [&](const EdgeOccurrence& occ) {
                if (state_edges.contains(occ.edge_id)) {
                    scratch_positions.push_back(occ.position);
                }
            });
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);
            hash = fnv_hash(hash, child_hashes.size());

            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            if (!stack.empty()) {
                DFSFrame& parent_frame = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent_frame.level];
                auto& parent_child_hashes = child_hashes_by_level[parent_frame.level];

                ChildInfo& child_info = parent_children[parent_frame.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent_frame.child_idx++;
            }
        }

        return result;
    }

    // Tree hash computation using vertex_occurrences_ instead of adjacency map
    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_tree_hash_no_adjacency(
        VertexId root,
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        SparseBitset visited;

        ArenaVector<DFSFrame> stack(*arena_);
        stack.reserve(MAX_TREE_DEPTH + 1);

        ArenaVector<ArenaVector<ChildInfo>> children_by_level(*arena_);
        children_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            children_by_level.push_back(ArenaVector<ChildInfo>(*arena_));
        }

        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level(*arena_);
        child_hashes_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            child_hashes_by_level.push_back(ArenaVector<uint64_t>(*arena_));
        }

        ArenaVector<uint8_t> scratch_positions(*arena_);
        ArenaVector<uint16_t> scratch_occ_positions(*arena_);

        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            // Check for abort periodically
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            // First time visiting this frame?
            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);

                // Collect own positions using vertex_occurrences_ + state_edges check
                scratch_positions.clear();
                if (frame.vertex < vertex_occurrences_.size()) {
                    vertex_occurrences_[frame.vertex].for_each([&](const EdgeOccurrence& occ) {
                        if (state_edges.contains(occ.edge_id)) {
                            scratch_positions.push_back(occ.position);
                        }
                    });
                }
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);
                frame.own_positions_count = scratch_positions.size();

                // Build children list using vertex_occurrences_
                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                std::unordered_map<VertexId, ArenaVector<std::pair<uint8_t, uint8_t>>> adjacent_map;

                if (frame.vertex < vertex_occurrences_.size()) {
                    vertex_occurrences_[frame.vertex].for_each([&](const EdgeOccurrence& occ) {
                        if (!state_edges.contains(occ.edge_id)) {
                            return;  // Edge not in this state
                        }

                        EdgeId eid = occ.edge_id;
                        uint8_t my_pos = occ.position;
                        uint8_t arity = edge_arities[eid];
                        const VertexId* verts = edge_vertices[eid];

                        for (uint8_t i = 0; i < arity; ++i) {
                            if (i != my_pos) {
                                VertexId adj_v = verts[i];
                                if (!visited.contains(adj_v)) {
                                    auto it = adjacent_map.find(adj_v);
                                    if (it == adjacent_map.end()) {
                                        it = adjacent_map.emplace(adj_v, ArenaVector<std::pair<uint8_t, uint8_t>>(*arena_)).first;
                                    }
                                    it->second.push_back({arity, i});
                                }
                            }
                        }
                    });
                }

                // Convert to children list
                for (const auto& [adj_v, occs] : adjacent_map) {
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.is_unique = (occs.size() == 1);
                    child.num_occurrences = static_cast<uint8_t>(occs.size());

                    scratch_occ_positions.clear();
                    for (size_t k = 0; k < occs.size(); ++k) {
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(occs[k].first) << 8) | occs[k].second);
                    }
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            // Process next child
            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                DFSFrame child_frame;
                child_frame.vertex = child.vertex;
                child_frame.level = level + 1;
                child_frame.child_idx = 0;
                child_frame.num_children = 0;
                child_frame.partial_hash = 0;
                child_frame.own_positions_count = 0;
                stack.push_back(child_frame);
                continue;
            }

            // All children processed - compute hash
            insertion_sort(child_hashes.begin(), child_hashes.end());

            uint64_t hash = frame.partial_hash;
            for (size_t i = 0; i < frame.own_positions_count; ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }
            for (uint64_t ch : child_hashes) {
                hash = fnv_hash(hash, ch);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            if (!stack.empty()) {
                DFSFrame& parent_frame = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent_frame.level];
                auto& parent_child_hashes = child_hashes_by_level[parent_frame.level];

                ChildInfo& child_info = parent_children[parent_frame.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent_frame.child_idx++;
            }
        }

        return result;
    }

    // =========================================================================
    // Tree Hash Computation - ITERATIVE VERSION
    // =========================================================================
    // Uses explicit stack instead of recursion to avoid per-call allocations.
    // Matches UniquenessTree::compute_tree_hash exactly.

    // Stack frame for iterative DFS
    struct DFSFrame {
        VertexId vertex;
        uint32_t level;
        uint32_t child_idx;           // Which child we're currently processing
        uint32_t num_children;        // Total number of children
        uint64_t partial_hash;        // Hash accumulated so far
        size_t own_positions_count;   // Number of own positions
    };

    // Child descriptor for DFS
    struct ChildInfo {
        VertexId vertex;
        bool is_unique;
        uint8_t num_occurrences;           // Number of occurrences
        uint16_t occurrence_positions[8];  // Sorted (arity << 8) | position values
    };

    // Reusable buffers for tree hash computation - allocated once per state hash
    struct TreeHashBuffers {
        ArenaVector<DFSFrame> stack;
        ArenaVector<ArenaVector<ChildInfo>> children_by_level;
        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level;
        ArenaVector<uint8_t> scratch_positions;
        ArenaVector<uint16_t> scratch_occ_positions;
        ArenaVector<std::pair<VertexId, std::pair<uint8_t, uint8_t>>> adj_vertices;
        bool initialized;

        explicit TreeHashBuffers(ConcurrentHeterogeneousArena& arena)
            : stack(arena)
            , children_by_level(arena)
            , child_hashes_by_level(arena)
            , scratch_positions(arena)
            , scratch_occ_positions(arena)
            , adj_vertices(arena)
            , initialized(false)
        {}

        void init_if_needed(ConcurrentHeterogeneousArena& arena, uint32_t max_depth) {
            if (initialized) return;
            stack.reserve(max_depth + 1);
            children_by_level.reserve(max_depth + 1);
            child_hashes_by_level.reserve(max_depth + 1);
            for (uint32_t i = 0; i <= max_depth; ++i) {
                children_by_level.push_back(ArenaVector<ChildInfo>(arena));
                child_hashes_by_level.push_back(ArenaVector<uint64_t>(arena));
            }
            initialized = true;
        }

        void reset_for_new_tree() {
            stack.clear();
            scratch_positions.clear();
            scratch_occ_positions.clear();
            adj_vertices.clear();
        }
    };

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_tree_hash(
        VertexId root,
        const SparseBitset& state_edges,
        const std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>>& adjacency,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        // Use SparseBitset for visited tracking with arena allocation
        SparseBitset visited;

        // Explicit stack
        ArenaVector<DFSFrame> stack(*arena_);
        stack.reserve(MAX_TREE_DEPTH + 1);

        // Per-frame children storage (indexed by stack depth)
        ArenaVector<ArenaVector<ChildInfo>> children_by_level(*arena_);
        children_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            children_by_level.push_back(ArenaVector<ChildInfo>(*arena_));
        }

        // Per-frame child hashes storage
        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level(*arena_);
        child_hashes_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            child_hashes_by_level.push_back(ArenaVector<uint64_t>(*arena_));
        }

        // Scratch buffers (reused)
        ArenaVector<uint8_t> scratch_positions(*arena_);
        ArenaVector<uint16_t> scratch_occ_positions(*arena_);

        // Push initial frame
        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            // Check for abort periodically
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            // First time visiting this frame?
            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);

                // Collect own positions
                scratch_positions.clear();
                auto adj_it = adjacency.find(frame.vertex);
                if (adj_it != adjacency.end()) {
                    for (size_t i = 0; i < adj_it->second.size(); ++i) {
                        scratch_positions.push_back(adj_it->second[i].second);
                    }
                }
                // Use insertion sort for small arrays
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);
                frame.own_positions_count = scratch_positions.size();

                // Build children list
                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                // Group adjacent vertices
                std::unordered_map<VertexId, ArenaVector<std::pair<uint8_t, uint8_t>>> adjacent_map;

                if (adj_it != adjacency.end()) {
                    for (size_t j = 0; j < adj_it->second.size(); ++j) {
                        EdgeId eid = adj_it->second[j].first;
                        uint8_t my_pos = adj_it->second[j].second;
                        uint8_t arity = edge_arities[eid];
                        const VertexId* verts = edge_vertices[eid];

                        for (uint8_t i = 0; i < arity; ++i) {
                            if (i != my_pos) {
                                VertexId adj_v = verts[i];
                                if (!visited.contains(adj_v)) {
                                    auto it = adjacent_map.find(adj_v);
                                    if (it == adjacent_map.end()) {
                                        it = adjacent_map.emplace(adj_v, ArenaVector<std::pair<uint8_t, uint8_t>>(*arena_)).first;
                                    }
                                    it->second.push_back({arity, i});
                                }
                            }
                        }
                    }
                }

                // Convert to children list
                for (const auto& [adj_v, occs] : adjacent_map) {
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.is_unique = (occs.size() == 1);
                    child.num_occurrences = static_cast<uint8_t>(occs.size());

                    scratch_occ_positions.clear();
                    for (size_t k = 0; k < occs.size(); ++k) {
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(occs[k].first) << 8) | occs[k].second);
                    }
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            // Process next child
            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);
                    child_hash = fnv_hash(child_hash, 0);
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            // All children processed - finalize
            visited.clear(frame.vertex);

            scratch_positions.clear();
            auto adj_it = adjacency.find(frame.vertex);
            if (adj_it != adjacency.end()) {
                for (size_t i = 0; i < adj_it->second.size(); ++i) {
                    scratch_positions.push_back(adj_it->second[i].second);
                }
            }
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);
            hash = fnv_hash(hash, child_hashes.size());

            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            if (!stack.empty()) {
                DFSFrame& parent = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent.level];
                auto& parent_child_hashes = child_hashes_by_level[parent.level];

                ChildInfo& child_info = parent_children[parent.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent.child_idx++;
            }
        }

        return result;
    }

    // Insertion sort for small arrays (faster than std::sort for n < 16)
    template<typename Iter>
    static void insertion_sort(Iter begin, Iter end) {
        for (Iter i = begin; i != end; ++i) {
            auto key = *i;
            Iter j = i;
            while (j != begin && *(j - 1) > key) {
                *j = *(j - 1);
                --j;
            }
            *j = key;
        }
    }

    // =========================================================================
    // Tree Hash Computation - FLAT ADJACENCY VERSION WITH BLOOM FILTER
    // =========================================================================
    // This version also populates a bloom filter with all vertices in the subtree.

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_tree_hash_flat_with_bloom(
        VertexId root,
        VertexId min_vertex,
        bool use_dense_index,
        const ArenaVector<size_t>& dense_index,
        const std::unordered_map<VertexId, size_t>& sparse_index,
        SparseBitset& visited,
        TreeHashBuffers& buffers,
        SubtreeBloomFilter& bloom_filter,  // OUTPUT: populated with subtree vertices
        const ArenaVector<std::pair<EdgeId, uint8_t>>& adj_data,
        const ArenaVector<uint32_t>& adj_offsets,
        [[maybe_unused]] const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            if (use_dense_index) {
                size_t idx = v - min_vertex;
                return (idx < dense_index.size()) ? dense_index[idx] : SIZE_MAX;
            } else {
                auto it = sparse_index.find(v);
                return (it != sparse_index.end()) ? it->second : SIZE_MAX;
            }
        };

        buffers.reset_for_new_tree();
        auto& stack = buffers.stack;
        auto& children_by_level = buffers.children_by_level;
        auto& child_hashes_by_level = buffers.child_hashes_by_level;
        auto& scratch_positions = buffers.scratch_positions;
        auto& scratch_occ_positions = buffers.scratch_occ_positions;

        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            // Check for abort periodically
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);
                bloom_filter.add(frame.vertex);  // Add to bloom filter

                size_t v_idx = get_vertex_idx(frame.vertex);
                uint32_t adj_start = adj_offsets[v_idx];
                uint32_t adj_end = adj_offsets[v_idx + 1];

                scratch_positions.clear();
                for (uint32_t j = adj_start; j < adj_end; ++j) {
                    scratch_positions.push_back(adj_data[j].second);
                }
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);
                frame.own_positions_count = scratch_positions.size();

                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                auto& adj_vertices = buffers.adj_vertices;
                adj_vertices.clear();

                for (uint32_t j = adj_start; j < adj_end; ++j) {
                    EdgeId eid = adj_data[j].first;
                    uint8_t my_pos = adj_data[j].second;
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t i = 0; i < arity; ++i) {
                        if (i != my_pos) {
                            VertexId adj_v = verts[i];
                            if (!visited.contains(adj_v)) {
                                adj_vertices.push_back({adj_v, {arity, i}});
                            }
                        }
                    }
                }

                std::sort(adj_vertices.begin(), adj_vertices.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

                size_t i = 0;
                while (i < adj_vertices.size()) {
                    VertexId adj_v = adj_vertices[i].first;
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.num_occurrences = 0;

                    scratch_occ_positions.clear();

                    while (i < adj_vertices.size() && adj_vertices[i].first == adj_v) {
                        auto [arity, pos] = adj_vertices[i].second;
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(arity) << 8) | pos);
                        child.num_occurrences++;
                        ++i;
                    }

                    child.is_unique = (child.num_occurrences == 1);
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    // Non-unique children are also in the subtree (as leaves)
                    if (!child.is_unique) {
                        bloom_filter.add(adj_v);
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);
                    child_hash = fnv_hash(child_hash, 0);
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            visited.clear(frame.vertex);

            size_t v_idx = get_vertex_idx(frame.vertex);
            uint32_t adj_start = adj_offsets[v_idx];
            uint32_t adj_end = adj_offsets[v_idx + 1];

            scratch_positions.clear();
            for (uint32_t j = adj_start; j < adj_end; ++j) {
                scratch_positions.push_back(adj_data[j].second);
            }
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);
            hash = fnv_hash(hash, child_hashes.size());

            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            if (!stack.empty()) {
                DFSFrame& parent = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent.level];
                auto& parent_child_hashes = child_hashes_by_level[parent.level];

                ChildInfo& child_info = parent_children[parent.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent.child_idx++;
            }
        }

        return result;
    }

    // =========================================================================
    // Tree Hash Computation - FLAT ADJACENCY VERSION (without bloom)
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_tree_hash_flat(
        VertexId root,
        VertexId min_vertex,
        bool use_dense_index,
        const ArenaVector<size_t>& dense_index,
        const std::unordered_map<VertexId, size_t>& sparse_index,
        SparseBitset& visited,  // Passed from caller, reused across calls
        TreeHashBuffers& buffers,  // Reusable buffers passed from caller
        const ArenaVector<std::pair<EdgeId, uint8_t>>& adj_data,
        const ArenaVector<uint32_t>& adj_offsets,
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        // Lambda for vertex lookup
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            if (use_dense_index) {
                size_t idx = v - min_vertex;
                return (idx < dense_index.size()) ? dense_index[idx] : SIZE_MAX;
            } else {
                auto it = sparse_index.find(v);
                return (it != sparse_index.end()) ? it->second : SIZE_MAX;
            }
        };

        // Use reusable buffers from caller
        buffers.reset_for_new_tree();
        auto& stack = buffers.stack;
        auto& children_by_level = buffers.children_by_level;
        auto& child_hashes_by_level = buffers.child_hashes_by_level;
        auto& scratch_positions = buffers.scratch_positions;
        auto& scratch_occ_positions = buffers.scratch_occ_positions;

        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            // Check for abort periodically
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);

                size_t v_idx = get_vertex_idx(frame.vertex);
                uint32_t adj_start = adj_offsets[v_idx];
                uint32_t adj_end = adj_offsets[v_idx + 1];

                scratch_positions.clear();
                for (uint32_t j = adj_start; j < adj_end; ++j) {
                    scratch_positions.push_back(adj_data[j].second);
                }
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);
                frame.own_positions_count = scratch_positions.size();

                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                // Use reusable buffer for adjacent vertices
                auto& adj_vertices = buffers.adj_vertices;
                adj_vertices.clear();

                for (uint32_t j = adj_start; j < adj_end; ++j) {
                    EdgeId eid = adj_data[j].first;
                    uint8_t my_pos = adj_data[j].second;
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t i = 0; i < arity; ++i) {
                        if (i != my_pos) {
                            VertexId adj_v = verts[i];
                            if (!visited.contains(adj_v)) {
                                adj_vertices.push_back({adj_v, {arity, i}});
                            }
                        }
                    }
                }

                std::sort(adj_vertices.begin(), adj_vertices.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

                size_t i = 0;
                while (i < adj_vertices.size()) {
                    VertexId adj_v = adj_vertices[i].first;
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.num_occurrences = 0;

                    scratch_occ_positions.clear();

                    while (i < adj_vertices.size() && adj_vertices[i].first == adj_v) {
                        auto [arity, pos] = adj_vertices[i].second;
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(arity) << 8) | pos);
                        child.num_occurrences++;
                        ++i;
                    }

                    child.is_unique = (child.num_occurrences == 1);
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);
                    child_hash = fnv_hash(child_hash, 0);
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            visited.clear(frame.vertex);

            size_t v_idx = get_vertex_idx(frame.vertex);
            uint32_t adj_start = adj_offsets[v_idx];
            uint32_t adj_end = adj_offsets[v_idx + 1];

            scratch_positions.clear();
            for (uint32_t j = adj_start; j < adj_end; ++j) {
                scratch_positions.push_back(adj_data[j].second);
            }
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);
            hash = fnv_hash(hash, child_hashes.size());

            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            if (!stack.empty()) {
                DFSFrame& parent = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent.level];
                auto& parent_child_hashes = child_hashes_by_level[parent.level];

                ChildInfo& child_info = parent_children[parent.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent.child_idx++;
            }
        }

        return result;
    }

    // =========================================================================
    // Edge Signature Computation
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_edge_signature(
        EdgeId eid,
        const VertexHashCache& cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        uint8_t arity = edge_arities[eid];
        const VertexId* verts = edge_vertices[eid];

        uint64_t sig = FNV_OFFSET;
        sig = fnv_hash(sig, arity);

        for (uint8_t i = 0; i < arity; ++i) {
            sig = fnv_hash(sig, cache.lookup(verts[i]));
        }

        return sig;
    }

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_edge_set_signature(
        const EdgeId* edges,
        uint8_t count,
        const VertexHashCache& cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        ArenaVector<uint64_t> edge_sigs(*arena_, count);

        for (uint8_t i = 0; i < count; ++i) {
            edge_sigs.push_back(compute_edge_signature(edges[i], cache, edge_vertices, edge_arities));
        }

        std::sort(edge_sigs.begin(), edge_sigs.end());

        uint64_t sig = FNV_OFFSET;
        sig = fnv_hash(sig, count);
        for (uint64_t es : edge_sigs) {
            sig = fnv_hash(sig, es);
        }

        return sig;
    }
};

}  // namespace hypergraph
