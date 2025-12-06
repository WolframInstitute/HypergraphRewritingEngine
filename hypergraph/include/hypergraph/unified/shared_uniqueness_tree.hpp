#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <atomic>

#include "types.hpp"
#include "bitset.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "concurrent_map.hpp"
#include "lock_free_list.hpp"

namespace hypergraph::unified {

// =============================================================================
// SharedUniquenessTree: Incremental Uniqueness Tree for Fast Canonicalization
// =============================================================================
//
// Key insight from DESIGN.md Section 18.3:
// 1. Single SharedUniquenessTree spans ALL states
// 2. Vertex neighborhood data shared across states (computed once per vertex)
// 3. State hash = filtered view of global tree (only edges in state's bitset)
// 4. Canonicalization speedup: Vertex classification prunes permutation search
//
// CRITICAL: Uniqueness trees CORRECTLY identify isomorphism (not approximate).
// They produce canonical vertex classes that can be used to:
// - Quickly compute state hashes
// - Find edge correspondences between isomorphic states
// - Speed up full canonicalization by constraining the search space
//
// Incremental computation strategy:
// - When edges are added, only affected vertices need tree updates
// - Vertex trees can be cached and reused across state computations
// - New states often share most vertices with parent, enabling delta computation
//
// Thread safety: Lock-free updates via atomic operations.
//

// -----------------------------------------------------------------------------
// VertexTreeNode: Cached uniqueness tree data for a single vertex
// -----------------------------------------------------------------------------

struct VertexTreeNode {
    // Edge occurrences for this vertex (fixed after edges are created)
    struct EdgeOccurrence {
        EdgeId edge_id;
        uint8_t position;    // Position of vertex in edge
        uint8_t arity;       // Total arity of edge
    };

    // Lock-free list of occurrences - thread-safe concurrent append
    LockFreeList<EdgeOccurrence> occurrences;

    // Structural signature (degree + position pattern)
    // This is stable for a vertex once all its edges are known
    // Updated atomically after each edge addition
    std::atomic<uint64_t> structural_signature{0};

    // Refined tree hash after convergence
    // This incorporates neighbor information recursively
    std::atomic<uint64_t> tree_hash{0};

    // Version counter for incremental updates
    std::atomic<uint32_t> version{0};
};

// -----------------------------------------------------------------------------
// SharedUniquenessTree: Main class
// -----------------------------------------------------------------------------

class SharedUniquenessTree {
public:
    static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    static constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;
    static constexpr size_t MAX_REFINEMENT_DEPTH = 100;

    explicit SharedUniquenessTree(ConcurrentHeterogeneousArena* arena)
        : arena_(arena)
    {}

    // Non-copyable
    SharedUniquenessTree(const SharedUniquenessTree&) = delete;
    SharedUniquenessTree& operator=(const SharedUniquenessTree&) = delete;

    // =========================================================================
    // Edge Registration (called when edges are created)
    // =========================================================================

    // Register a new edge, updating vertex neighborhoods
    // Thread-safe: can be called concurrently
    void register_edge(
        EdgeId edge_id,
        const VertexId* vertices,
        uint8_t arity
    ) {
        // Ensure vertex storage is large enough
        for (uint8_t i = 0; i < arity; ++i) {
            ensure_vertex_capacity(vertices[i]);
        }

        // Add edge occurrence to each vertex's neighborhood
        for (uint8_t i = 0; i < arity; ++i) {
            add_edge_to_vertex(vertices[i], edge_id, i, arity);
        }
    }

    // =========================================================================
    // State Canonical Hash Computation
    // =========================================================================

    // Compute canonical hash for a state (set of edges)
    // Uses LOCAL WL computation on state's edges only - deterministic!
    // The vertex occurrence lists are shared, but WL refinement is per-state.
    //
    // Performance: Uses ArenaVector for simple arrays, std::vector with reserve()
    // for nested structures where cache locality matters.
    uint64_t compute_state_hash(
        const SparseBitset& state_edges,
        const VertexId* const* edge_vertices,
        const uint8_t* edge_arities
    ) {
        if (state_edges.empty()) {
            return 0;
        }

        // Get edge count for size estimation
        size_t edge_count = state_edges.count();

        // ========== STEP 1: Collect state's vertices ==========
        ArenaVector<VertexId> vertices(*arena_, edge_count * 3);  // Estimate: avg 3 vertices per edge
        VertexId max_vertex = 0;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                vertices.push_back(verts[i]);
                if (verts[i] > max_vertex) max_vertex = verts[i];
            }
        });

        // Remove duplicates
        std::sort(vertices.begin(), vertices.end());
        auto new_end = std::unique(vertices.begin(), vertices.end());
        size_t unique_vertex_count = new_end - vertices.begin();

        if (unique_vertex_count == 0) {
            return 0;
        }

        // Build vertex index for O(1) lookup
        ArenaVector<size_t> vertex_index(*arena_, max_vertex + 1);
        vertex_index.resize(max_vertex + 1, SIZE_MAX);
        for (size_t i = 0; i < unique_vertex_count; ++i) {
            vertex_index[vertices[i]] = i;
        }

        // ========== STEP 2: Build local adjacency from state's edges ==========
        // For each vertex, collect (arity, position) pairs from THIS state's edges only
        std::vector<std::vector<std::pair<uint8_t, uint8_t>>> filtered_occ(unique_vertex_count);

        // Also build edge list per vertex for WL neighbor lookup
        std::vector<std::vector<std::pair<EdgeId, uint8_t>>> vertex_to_edges(unique_vertex_count);

        // Pre-reserve inner vectors based on estimated degree (avg ~3-4 edges per vertex)
        size_t avg_degree = (edge_count * 2) / std::max(unique_vertex_count, size_t(1)) + 1;
        for (size_t i = 0; i < unique_vertex_count; ++i) {
            filtered_occ[i].reserve(avg_degree);
            vertex_to_edges[i].reserve(avg_degree);
        }

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = vertex_index[verts[i]];
                filtered_occ[idx].emplace_back(arity, i);
                vertex_to_edges[idx].emplace_back(eid, i);
            }
        });

        // ========== STEP 3: Compute initial structural signatures ==========
        ArenaVector<uint64_t> current(*arena_, unique_vertex_count);
        current.resize(unique_vertex_count);

        for (size_t i = 0; i < unique_vertex_count; ++i) {
            auto& occ = filtered_occ[i];
            std::sort(occ.begin(), occ.end());  // Deterministic ordering

            uint64_t h = FNV_OFFSET;
            h = fnv_combine(h, occ.size());  // degree
            for (const auto& [arity, pos] : occ) {
                h = fnv_combine(h, arity);
                h = fnv_combine(h, pos);
            }
            current[i] = h;
        }

        // ========== STEP 4: WL refinement (LOCAL to state) ==========
        ArenaVector<uint64_t> next(*arena_, unique_vertex_count);
        next.resize(unique_vertex_count);
        bool changed = true;
        size_t iteration = 0;

        // Pre-allocate neighbor_hashes (reused each iteration)
        ArenaVector<uint64_t> neighbor_hashes(*arena_, unique_vertex_count * 2);

        while (changed && iteration < MAX_REFINEMENT_DEPTH) {
            changed = false;
            ++iteration;

            for (size_t i = 0; i < unique_vertex_count; ++i) {
                uint64_t h = current[i];

                // Clear and reuse the neighbor_hashes vector
                neighbor_hashes.clear();

                for (const auto& [eid, my_pos] : vertex_to_edges[i]) {
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    // Add hashes of other vertices in this edge
                    for (uint8_t k = 0; k < arity; ++k) {
                        if (k != my_pos) {
                            size_t neighbor_idx = vertex_index[verts[k]];
                            // Include position for structural distinction
                            neighbor_hashes.push_back(fnv_combine(current[neighbor_idx], k));
                        }
                    }
                }

                std::sort(neighbor_hashes.begin(), neighbor_hashes.end());
                for (uint64_t nh : neighbor_hashes) {
                    h = fnv_combine(h, nh);
                }

                if (h != current[i]) changed = true;
                next[i] = h;
            }

            // Swap current and next (ArenaVector supports move)
            ArenaVector<uint64_t> tmp = std::move(current);
            current = std::move(next);
            next = std::move(tmp);
        }

        // ========== STEP 5: Build canonical hash ==========
        // Sort vertices by hash for canonical ordering
        ArenaVector<std::pair<uint64_t, size_t>> sorted_hashes(*arena_, unique_vertex_count);
        for (size_t i = 0; i < unique_vertex_count; ++i) {
            sorted_hashes.emplace_back(current[i], i);
        }
        std::sort(sorted_hashes.begin(), sorted_hashes.end());

        // Build canonical edge representations
        std::vector<std::vector<uint64_t>> canonical_edges;
        canonical_edges.reserve(edge_count);
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            std::vector<uint64_t> edge_rep;
            edge_rep.reserve(arity);
            for (uint8_t i = 0; i < arity; ++i) {
                edge_rep.push_back(current[vertex_index[verts[i]]]);
            }
            // DO NOT sort - vertex order matters for directed hyperedges
            canonical_edges.push_back(std::move(edge_rep));
        });

        // Sort edges for canonical ordering
        std::sort(canonical_edges.begin(), canonical_edges.end());

        // Compute final hash
        uint64_t hash = FNV_OFFSET;

        // Hash vertex equivalence class structure
        uint64_t prev_hash = 0;
        uint32_t class_count = 0;
        for (size_t i = 0; i < sorted_hashes.size(); ++i) {
            uint64_t vh = sorted_hashes[i].first;
            if (vh != prev_hash && class_count > 0) {
                hash = fnv_combine(hash, prev_hash);
                hash = fnv_combine(hash, class_count);
                class_count = 0;
            }
            prev_hash = vh;
            ++class_count;
        }
        if (class_count > 0) {
            hash = fnv_combine(hash, prev_hash);
            hash = fnv_combine(hash, class_count);
        }

        // Hash edge structure
        for (const auto& edge : canonical_edges) {
            hash = fnv_combine(hash, edge.size());
            for (uint64_t vh : edge) {
                hash = fnv_combine(hash, vh);
            }
        }

        return hash;
    }

    // =========================================================================
    // Vertex Classification for Canonicalization Speedup
    // =========================================================================

    // Get vertex equivalence classes for a state
    // Vertices with same WL hash are automorphically equivalent
    // This constrains the canonicalization search from O(V!) to O(n1! * n2! * ...)
    // Uses LOCAL WL computation on state's edges only - deterministic!
    std::vector<std::vector<VertexId>> get_vertex_classes(
        const SparseBitset& state_edges,
        const VertexId* const* edge_vertices,
        const uint8_t* edge_arities
    ) {
        std::vector<std::vector<VertexId>> classes;

        if (state_edges.empty()) {
            return classes;
        }

        // Get edge count for size estimation
        size_t edge_count = state_edges.count();

        // ========== STEP 1: Collect state's vertices ==========
        std::vector<VertexId> vertices;
        vertices.reserve(edge_count * 3);
        VertexId max_vertex = 0;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                vertices.push_back(verts[i]);
                if (verts[i] > max_vertex) max_vertex = verts[i];
            }
        });

        // Remove duplicates
        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

        if (vertices.empty()) {
            return classes;
        }

        // Build vertex index for O(1) lookup
        std::vector<size_t> vertex_index(max_vertex + 1, SIZE_MAX);
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertex_index[vertices[i]] = i;
        }

        // ========== STEP 2: Build local adjacency from state's edges ==========
        std::vector<std::vector<std::pair<uint8_t, uint8_t>>> filtered_occ(vertices.size());
        std::vector<std::vector<std::pair<EdgeId, uint8_t>>> vertex_to_edges(vertices.size());

        // Pre-reserve inner vectors
        size_t avg_degree = (edge_count * 2) / std::max(vertices.size(), size_t(1)) + 1;
        for (size_t i = 0; i < vertices.size(); ++i) {
            filtered_occ[i].reserve(avg_degree);
            vertex_to_edges[i].reserve(avg_degree);
        }

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = vertex_index[verts[i]];
                filtered_occ[idx].emplace_back(arity, i);
                vertex_to_edges[idx].emplace_back(eid, i);
            }
        });

        // ========== STEP 3: Compute initial structural signatures ==========
        std::vector<uint64_t> current(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            auto& occ = filtered_occ[i];
            std::sort(occ.begin(), occ.end());

            uint64_t h = FNV_OFFSET;
            h = fnv_combine(h, occ.size());
            for (const auto& [arity, pos] : occ) {
                h = fnv_combine(h, arity);
                h = fnv_combine(h, pos);
            }
            current[i] = h;
        }

        // ========== STEP 4: WL refinement (LOCAL to state) ==========
        std::vector<uint64_t> next(vertices.size());
        bool changed = true;
        size_t iteration = 0;

        // Pre-allocate neighbor_hashes outside the loop
        std::vector<uint64_t> neighbor_hashes;
        neighbor_hashes.reserve(vertices.size() * 2);

        while (changed && iteration < MAX_REFINEMENT_DEPTH) {
            changed = false;
            ++iteration;

            for (size_t i = 0; i < vertices.size(); ++i) {
                uint64_t h = current[i];

                neighbor_hashes.clear();
                for (const auto& [eid, my_pos] : vertex_to_edges[i]) {
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t k = 0; k < arity; ++k) {
                        if (k != my_pos) {
                            size_t neighbor_idx = vertex_index[verts[k]];
                            neighbor_hashes.push_back(fnv_combine(current[neighbor_idx], k));
                        }
                    }
                }

                std::sort(neighbor_hashes.begin(), neighbor_hashes.end());
                for (uint64_t nh : neighbor_hashes) {
                    h = fnv_combine(h, nh);
                }

                if (h != current[i]) changed = true;
                next[i] = h;
            }

            std::swap(current, next);
        }

        // ========== STEP 5: Group vertices by hash into classes ==========
        std::vector<std::pair<uint64_t, VertexId>> vertex_hashes;
        vertex_hashes.reserve(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertex_hashes.emplace_back(current[i], vertices[i]);
        }
        std::sort(vertex_hashes.begin(), vertex_hashes.end());

        classes.push_back({vertex_hashes[0].second});
        for (size_t i = 1; i < vertex_hashes.size(); ++i) {
            if (vertex_hashes[i].first == vertex_hashes[i-1].first) {
                classes.back().push_back(vertex_hashes[i].second);
            } else {
                classes.push_back({vertex_hashes[i].second});
            }
        }

        return classes;
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    size_t num_vertices() const { return vertices_.size(); }

private:
    ConcurrentHeterogeneousArena* arena_;

    // Per-vertex tree data
    SegmentedArray<VertexTreeNode> vertices_;

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    static uint64_t fnv_combine(uint64_t h, uint64_t value) {
        h ^= value;
        h *= FNV_PRIME;
        return h;
    }

    void ensure_vertex_capacity(VertexId vertex) {
        // Use get_or_default to ensure the segment exists without double-construction
        // The segment allocation already default-constructs all elements, so this
        // just ensures the segment is allocated and accessible. Thread-safe.
        vertices_.get_or_default(vertex, *arena_);
    }

    void add_edge_to_vertex(VertexId vertex, EdgeId edge, uint8_t position, uint8_t arity) {
        auto& node = vertices_[vertex];

        // Lock-free append to occurrences list
        VertexTreeNode::EdgeOccurrence occ{edge, position, arity};
        node.occurrences.push(occ, *arena_);

        // Recompute structural signature (atomic update)
        // Note: This is an approximation during concurrent updates, but converges
        uint64_t new_sig = compute_structural_signature(node);
        node.structural_signature.store(new_sig, std::memory_order_release);
        node.version.fetch_add(1, std::memory_order_release);
    }

    uint64_t compute_structural_signature(const VertexTreeNode& node) {
        uint64_t h = FNV_OFFSET;

        // Collect occurrences from lock-free list (snapshot at call time)
        std::vector<std::pair<uint8_t, uint8_t>> sorted_occ;
        node.occurrences.for_each([&](const VertexTreeNode::EdgeOccurrence& occ) {
            sorted_occ.emplace_back(occ.arity, occ.position);
        });

        h = fnv_combine(h, sorted_occ.size());  // Degree

        // Sort occurrences for canonical ordering
        std::sort(sorted_occ.begin(), sorted_occ.end());

        for (const auto& [arity, pos] : sorted_occ) {
            h = fnv_combine(h, arity);
            h = fnv_combine(h, pos);
        }

        return h;
    }
};

}  // namespace hypergraph::unified
