#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <atomic>
#include <unordered_map>

#include "types.hpp"
#include "bitset.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "concurrent_map.hpp"
#include "lock_free_list.hpp"

namespace hypergraph::unified {

// =============================================================================
// WLHash: Weisfeiler-Lehman style hashing for hypergraph isomorphism
// =============================================================================
//
// This implements WL (color refinement) style hashing, NOT uniqueness trees.
// For true Gorard-style uniqueness trees, see unified_uniqueness_tree.hpp.
//
// WL hashing works by:
// 1. Assigning initial colors based on vertex degree/position structure
// 2. Iteratively refining colors by aggregating neighbor colors
// 3. Converging when colors stabilize (no changes between iterations)
//
// Key characteristics:
// - Iterative message passing (all vertices update simultaneously)
// - Polynomial time: O(V^2) per iteration, typically O(log V) iterations
// - May not distinguish all non-isomorphic graphs (weaker than uniqueness trees)
//
// Thread safety:
// - Edge registration: Lock-free via LockFreeList
// - State hash computation: Thread-safe, uses shared read-only data
// - Hash caching: Lock-free via ConcurrentMap
//
// Note: Common types (EdgeOccurrence, VertexHashCache, EdgeCorrespondence,
// EventSignature, FNV constants) are defined in types.hpp

// Forward declarations
class UnifiedHypergraph;

// =============================================================================
// WLHash: Main class
// =============================================================================

class WLHash {
public:
    // FNV constants are defined in types.hpp
    static constexpr size_t MAX_REFINEMENT_DEPTH = 100;

    explicit WLHash(ConcurrentHeterogeneousArena* arena)
        : arena_(arena)
    {}

    // Non-copyable
    WLHash(const WLHash&) = delete;
    WLHash& operator=(const WLHash&) = delete;

    // =========================================================================
    // Edge Registration (called when edges are created)
    // =========================================================================

    // Register a new edge, updating vertex neighborhoods
    // Thread-safe: can be called concurrently
    void register_edge(EdgeId edge_id, const VertexId* vertices, uint8_t arity) {
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
    // This is the full computation from scratch
    // Uses accessors that can dynamically extend for thread-safety
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t compute_state_hash(
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        if (state_edges.empty()) {
            return 0;
        }

        // Collect vertices and track max for dense index optimization
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

        if (vertices.size() == 0) {
            return 0;
        }

        const size_t num_vertices = vertices.size();
        const VertexId range = max_vertex - min_vertex + 1;

        // Always use arena-allocated dense array for O(1) lookup without hash overhead
        ArenaVector<size_t> vertex_index(*arena_, range);
        vertex_index.resize(range);
        for (size_t i = 0; i < range; ++i) {
            vertex_index[i] = SIZE_MAX;
        }
        for (size_t i = 0; i < num_vertices; ++i) {
            vertex_index[vertices[i] - min_vertex] = i;
        }

        // Direct array lookup - no hash overhead
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            return vertex_index[v - min_vertex];
        };

        // Count total edge occurrences for pre-allocation
        size_t total_occurrences = 0;
        state_edges.for_each([&](EdgeId eid) {
            total_occurrences += edge_arities[eid];
        });

        // Arena-allocate flat adjacency structure
        // filtered_occ_data: flat array of (arity, pos) pairs
        // vertex_to_edges_data: flat array of (edge_id, pos) pairs
        // offsets: start index for each vertex
        ArenaVector<std::pair<uint8_t, uint8_t>> filtered_occ_data(*arena_, total_occurrences);
        ArenaVector<std::pair<EdgeId, uint8_t>> vertex_to_edges_data(*arena_, total_occurrences);
        ArenaVector<uint32_t> occ_counts(*arena_, num_vertices);
        occ_counts.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) occ_counts[i] = 0;

        // First pass: count occurrences per vertex
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                occ_counts[idx]++;
            }
        });

        // Compute offsets (prefix sum)
        ArenaVector<uint32_t> occ_offsets(*arena_, num_vertices + 1);
        occ_offsets.resize(num_vertices + 1);
        occ_offsets[0] = 0;
        for (size_t i = 0; i < num_vertices; ++i) {
            occ_offsets[i + 1] = occ_offsets[i] + occ_counts[i];
        }

        // Reset counts for second pass
        for (size_t i = 0; i < num_vertices; ++i) occ_counts[i] = 0;

        // Second pass: fill adjacency data
        filtered_occ_data.resize(total_occurrences);
        vertex_to_edges_data.resize(total_occurrences);
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                size_t pos = occ_offsets[idx] + occ_counts[idx];
                filtered_occ_data[pos] = {arity, i};
                vertex_to_edges_data[pos] = {eid, i};
                occ_counts[idx]++;
            }
        });

        // Sort each vertex's occurrences for canonical form
        for (size_t i = 0; i < num_vertices; ++i) {
            auto begin = filtered_occ_data.begin() + occ_offsets[i];
            auto end = filtered_occ_data.begin() + occ_offsets[i + 1];
            insertion_sort(begin, end);
        }

        // Initial structural signatures
        ArenaVector<uint64_t> current(*arena_, num_vertices);
        current.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            uint64_t h = FNV_OFFSET;
            size_t count = occ_offsets[i + 1] - occ_offsets[i];
            h = fnv_combine(h, count);  // degree
            for (size_t j = occ_offsets[i]; j < occ_offsets[i + 1]; ++j) {
                h = fnv_combine(h, filtered_occ_data[j].first);   // arity
                h = fnv_combine(h, filtered_occ_data[j].second);  // pos
            }
            current[i] = h;
        }

        // WL refinement - terminate when no colors change
        ArenaVector<uint64_t> next(*arena_, num_vertices);
        next.resize(num_vertices);
        ArenaVector<uint64_t> neighbor_hashes(*arena_);
        bool changed = true;
        size_t iteration = 0;

        while (changed && iteration < MAX_REFINEMENT_DEPTH) {
            changed = false;
            ++iteration;

            for (size_t i = 0; i < num_vertices; ++i) {
                uint64_t h = current[i];
                neighbor_hashes.clear();

                for (size_t j = occ_offsets[i]; j < occ_offsets[i + 1]; ++j) {
                    EdgeId eid = vertex_to_edges_data[j].first;
                    uint8_t my_pos = vertex_to_edges_data[j].second;
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t k = 0; k < arity; ++k) {
                        if (k != my_pos) {
                            size_t neighbor_idx = get_vertex_idx(verts[k]);
                            neighbor_hashes.push_back(fnv_combine(current[neighbor_idx], k));
                        }
                    }
                }

                insertion_sort(neighbor_hashes.begin(), neighbor_hashes.end());
                for (size_t j = 0; j < neighbor_hashes.size(); ++j) {
                    h = fnv_combine(h, neighbor_hashes[j]);
                }

                if (h != current[i]) changed = true;
                next[i] = h;
            }

            std::swap(current, next);
        }

        // Build canonical hash from vertex and edge structure
        return build_canonical_hash_dense(current, vertices, min_vertex,
                                          vertex_index,
                                          state_edges, edge_vertices, edge_arities);
    }

    // Compute state hash with caching of vertex hashes
    // Returns both the hash and a cache that can be used for child states
    // Uses arena allocation for all temporary storage (consistent with compute_state_hash)
    template<typename VertexAccessor, typename ArityAccessor>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_with_cache(
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
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

        if (vertices.size() == 0) {
            return {0, cache};
        }

        const size_t num_vertices = vertices.size();
        const VertexId range = max_vertex - min_vertex + 1;

        // Always use arena-allocated dense array for O(1) lookup without hash overhead
        // 1M vertices = 8MB which is acceptable for performance
        ArenaVector<size_t> vertex_index(*arena_, range);
        vertex_index.resize(range);
        for (size_t i = 0; i < range; ++i) {
            vertex_index[i] = SIZE_MAX;
        }
        for (size_t i = 0; i < num_vertices; ++i) {
            vertex_index[vertices[i] - min_vertex] = i;
        }

        // Direct array lookup - no hash overhead
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            return vertex_index[v - min_vertex];
        };

        // Count total edge occurrences for pre-allocation
        size_t total_occurrences = 0;
        state_edges.for_each([&](EdgeId eid) {
            total_occurrences += edge_arities[eid];
        });

        // Arena-allocate flat adjacency structure
        ArenaVector<std::pair<uint8_t, uint8_t>> filtered_occ_data(*arena_, total_occurrences);
        ArenaVector<std::pair<EdgeId, uint8_t>> vertex_to_edges_data(*arena_, total_occurrences);
        ArenaVector<uint32_t> occ_counts(*arena_, num_vertices);
        occ_counts.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) occ_counts[i] = 0;

        // First pass: count occurrences per vertex
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                occ_counts[idx]++;
            }
        });

        // Compute offsets (prefix sum)
        ArenaVector<uint32_t> occ_offsets(*arena_, num_vertices + 1);
        occ_offsets.resize(num_vertices + 1);
        occ_offsets[0] = 0;
        for (size_t i = 0; i < num_vertices; ++i) {
            occ_offsets[i + 1] = occ_offsets[i] + occ_counts[i];
        }

        // Reset counts for second pass
        for (size_t i = 0; i < num_vertices; ++i) occ_counts[i] = 0;

        // Second pass: fill adjacency data
        filtered_occ_data.resize(total_occurrences);
        vertex_to_edges_data.resize(total_occurrences);
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                size_t pos = occ_offsets[idx] + occ_counts[idx];
                filtered_occ_data[pos] = {arity, i};
                vertex_to_edges_data[pos] = {eid, i};
                occ_counts[idx]++;
            }
        });

        // Sort each vertex's occurrences for canonical form
        for (size_t i = 0; i < num_vertices; ++i) {
            auto begin = filtered_occ_data.begin() + occ_offsets[i];
            auto end = filtered_occ_data.begin() + occ_offsets[i + 1];
            insertion_sort(begin, end);
        }

        // Initial structural signatures
        ArenaVector<uint64_t> current(*arena_, num_vertices);
        current.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            uint64_t h = FNV_OFFSET;
            size_t count = occ_offsets[i + 1] - occ_offsets[i];
            h = fnv_combine(h, count);  // degree
            for (size_t j = occ_offsets[i]; j < occ_offsets[i + 1]; ++j) {
                h = fnv_combine(h, filtered_occ_data[j].first);   // arity
                h = fnv_combine(h, filtered_occ_data[j].second);  // pos
            }
            current[i] = h;
        }

        // WL refinement - terminate when no colors change
        ArenaVector<uint64_t> next(*arena_, num_vertices);
        next.resize(num_vertices);
        ArenaVector<uint64_t> neighbor_hashes(*arena_);
        bool changed = true;
        size_t iteration = 0;

        while (changed && iteration < MAX_REFINEMENT_DEPTH) {
            changed = false;
            ++iteration;

            for (size_t i = 0; i < num_vertices; ++i) {
                uint64_t h = current[i];
                neighbor_hashes.clear();

                for (size_t j = occ_offsets[i]; j < occ_offsets[i + 1]; ++j) {
                    EdgeId eid = vertex_to_edges_data[j].first;
                    uint8_t my_pos = vertex_to_edges_data[j].second;
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t k = 0; k < arity; ++k) {
                        if (k != my_pos) {
                            size_t neighbor_idx = get_vertex_idx(verts[k]);
                            neighbor_hashes.push_back(fnv_combine(current[neighbor_idx], k));
                        }
                    }
                }

                insertion_sort(neighbor_hashes.begin(), neighbor_hashes.end());
                for (size_t j = 0; j < neighbor_hashes.size(); ++j) {
                    h = fnv_combine(h, neighbor_hashes[j]);
                }

                if (h != current[i]) changed = true;
                next[i] = h;
            }

            std::swap(current, next);
        }

        // Build vertex hash cache (vertices already sorted)
        cache.count = static_cast<uint32_t>(num_vertices);
        cache.vertices = arena_->allocate_array<VertexId>(cache.count);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.count);

        for (size_t i = 0; i < num_vertices; ++i) {
            cache.vertices[i] = vertices[i];
            cache.hashes[i] = current[i];
        }

        uint64_t hash = build_canonical_hash_dense(current, vertices, min_vertex,
                                                   vertex_index,
                                                   state_edges, edge_vertices, edge_arities);

        return {hash, cache};
    }

    // Compute state hash incrementally from parent's cached hashes
    // Only recomputes hashes for vertices affected by removed/added edges
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t compute_state_hash_incremental(
        const SparseBitset& state_edges,
        const VertexHashCache& parent_cache,
        const EdgeId* removed_edges, uint8_t num_removed,
        const EdgeId* added_edges, uint8_t num_added,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        if (state_edges.empty()) {
            return 0;
        }

        // Find affected vertices (those incident to removed or added edges)
        std::vector<VertexId> affected_vertices;

        for (uint8_t i = 0; i < num_removed; ++i) {
            EdgeId eid = removed_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                affected_vertices.push_back(verts[j]);
            }
        }

        for (uint8_t i = 0; i < num_added; ++i) {
            EdgeId eid = added_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                affected_vertices.push_back(verts[j]);
            }
        }

        std::sort(affected_vertices.begin(), affected_vertices.end());
        affected_vertices.erase(std::unique(affected_vertices.begin(), affected_vertices.end()),
                               affected_vertices.end());

        // If many vertices affected, fall back to full computation
        // (incremental becomes less beneficial with larger delta)
        if (affected_vertices.size() > parent_cache.count / 2) {
            return compute_state_hash(state_edges, edge_vertices, edge_arities);
        }

        // Collect all vertices in the new state
        std::vector<VertexId> vertices;
        VertexId max_vertex = 0;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                vertices.push_back(verts[i]);
                if (verts[i] > max_vertex) max_vertex = verts[i];
            }
        });

        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

        if (vertices.empty()) {
            return 0;
        }

        // Build vertex index
        std::vector<size_t> vertex_index(max_vertex + 1, SIZE_MAX);
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertex_index[vertices[i]] = i;
        }

        // Mark which vertices are affected
        std::vector<bool> is_affected(vertices.size(), false);
        for (VertexId v : affected_vertices) {
            if (v <= max_vertex && vertex_index[v] != SIZE_MAX) {
                is_affected[vertex_index[v]] = true;
            }
        }

        // Build adjacency
        std::vector<std::vector<std::pair<uint8_t, uint8_t>>> filtered_occ(vertices.size());
        std::vector<std::vector<std::pair<EdgeId, uint8_t>>> vertex_to_edges(vertices.size());

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = vertex_index[verts[i]];
                filtered_occ[idx].emplace_back(arity, i);
                vertex_to_edges[idx].emplace_back(eid, i);
            }
        });

        // Initialize hashes - ALWAYS compute initial hash from current state's structure
        // NOTE: We cannot use parent's cached FINAL hashes as INITIAL hashes here.
        // Parent cache contains hashes after N refinement iterations, but we need
        // iteration-0 hashes (degree + edge positions) for correct WL refinement.
        std::vector<uint64_t> current(vertices.size());

        for (size_t i = 0; i < vertices.size(); ++i) {
            // Compute initial hash for this vertex based on current state's structure
            auto& occ = filtered_occ[i];
            insertion_sort(occ.begin(), occ.end());

            uint64_t h = FNV_OFFSET;
            h = fnv_combine(h, occ.size());
            for (const auto& [arity, pos] : occ) {
                h = fnv_combine(h, arity);
                h = fnv_combine(h, pos);
            }
            current[i] = h;
        }

        // WL refinement with active vertex tracking
        std::vector<uint64_t> next(vertices.size());
        std::vector<uint64_t> neighbor_hashes;
        size_t iteration = 0;

        // NOTE: We must mark ALL vertices as active, not just affected ones.
        // WL requires all vertices to refine synchronously - a vertex's iteration-N
        // hash depends on neighbors' iteration-(N-1) hashes. If we only refine
        // affected vertices first, they get "ahead" of unaffected vertices, breaking
        // the synchronization and producing incorrect hashes.
        std::vector<bool> active(vertices.size(), true);
        std::vector<bool> next_active(vertices.size(), false);

        // Count active for early termination check
        size_t num_active = 0;
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (active[i]) ++num_active;
        }

        while (num_active > 0 && iteration < MAX_REFINEMENT_DEPTH) {
            ++iteration;
            size_t next_num_active = 0;

            for (size_t i = 0; i < vertices.size(); ++i) {
                if (!active[i]) {
                    // Vertex not active - keep current hash
                    next[i] = current[i];
                    continue;
                }

                // Active vertex - recompute hash
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

                insertion_sort(neighbor_hashes.begin(), neighbor_hashes.end());
                for (uint64_t nh : neighbor_hashes) {
                    h = fnv_combine(h, nh);
                }

                next[i] = h;

                // If hash changed, mark neighbors as active for next iteration
                if (h != current[i]) {
                    for (const auto& [eid, my_pos] : vertex_to_edges[i]) {
                        uint8_t arity = edge_arities[eid];
                        const VertexId* verts = edge_vertices[eid];
                        for (uint8_t k = 0; k < arity; ++k) {
                            if (k != my_pos) {
                                size_t neighbor_idx = vertex_index[verts[k]];
                                if (!next_active[neighbor_idx]) {
                                    next_active[neighbor_idx] = true;
                                    ++next_num_active;
                                }
                            }
                        }
                    }
                }
            }

            std::swap(current, next);
            std::swap(active, next_active);
            std::fill(next_active.begin(), next_active.end(), false);
            num_active = next_num_active;
        }

        return build_canonical_hash(current, vertices, vertex_index, state_edges,
                                   edge_vertices, edge_arities);
    }

    // Compute state hash incrementally with cache for children to use
    // Returns both hash and cache (similar to compute_state_hash_with_cache)
    template<typename VertexAccessor, typename ArityAccessor>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_incremental_with_cache(
        const SparseBitset& state_edges,
        const VertexHashCache& parent_cache,
        const EdgeId* removed_edges, uint8_t num_removed,
        const EdgeId* added_edges, uint8_t num_added,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        VertexHashCache cache;

        if (state_edges.empty()) {
            return {0, cache};
        }

        // Find affected vertices (those incident to removed or added edges)
        ArenaVector<VertexId> affected_vertices(*arena_);

        for (uint8_t i = 0; i < num_removed; ++i) {
            EdgeId eid = removed_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                affected_vertices.push_back(verts[j]);
            }
        }

        for (uint8_t i = 0; i < num_added; ++i) {
            EdgeId eid = added_edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t j = 0; j < arity; ++j) {
                affected_vertices.push_back(verts[j]);
            }
        }

        std::sort(affected_vertices.begin(), affected_vertices.end());
        auto new_end = std::unique(affected_vertices.begin(), affected_vertices.end());
        affected_vertices.resize(new_end - affected_vertices.begin());

        // If many vertices affected, fall back to full computation
        if (affected_vertices.size() > parent_cache.count / 2) {
            return compute_state_hash_with_cache(state_edges, edge_vertices, edge_arities);
        }

        // Collect all vertices in the new state
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

        std::sort(vertices.begin(), vertices.end());
        auto vert_end = std::unique(vertices.begin(), vertices.end());
        vertices.resize(vert_end - vertices.begin());

        if (vertices.empty()) {
            return {0, cache};
        }

        const size_t num_vertices = vertices.size();
        const VertexId range = max_vertex - min_vertex + 1;

        // Build vertex index (dense)
        ArenaVector<size_t> vertex_index(*arena_, range);
        vertex_index.resize(range);
        for (size_t i = 0; i < range; ++i) {
            vertex_index[i] = SIZE_MAX;
        }
        for (size_t i = 0; i < num_vertices; ++i) {
            vertex_index[vertices[i] - min_vertex] = i;
        }

        auto get_vertex_idx = [&](VertexId v) -> size_t {
            return vertex_index[v - min_vertex];
        };

        // Mark which vertices are affected
        ArenaVector<bool> is_affected(*arena_, num_vertices);
        is_affected.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) is_affected[i] = false;
        for (VertexId v : affected_vertices) {
            if (v >= min_vertex && v <= max_vertex && vertex_index[v - min_vertex] != SIZE_MAX) {
                is_affected[vertex_index[v - min_vertex]] = true;
            }
        }

        // Count occurrences for adjacency building
        size_t total_occurrences = 0;
        state_edges.for_each([&](EdgeId eid) {
            total_occurrences += edge_arities[eid];
        });

        // Build flat adjacency structure
        ArenaVector<std::pair<uint8_t, uint8_t>> filtered_occ_data(*arena_, total_occurrences);
        ArenaVector<std::pair<EdgeId, uint8_t>> vertex_to_edges_data(*arena_, total_occurrences);
        ArenaVector<uint32_t> occ_counts(*arena_, num_vertices);
        occ_counts.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) occ_counts[i] = 0;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                occ_counts[idx]++;
            }
        });

        ArenaVector<uint32_t> occ_offsets(*arena_, num_vertices + 1);
        occ_offsets.resize(num_vertices + 1);
        occ_offsets[0] = 0;
        for (size_t i = 0; i < num_vertices; ++i) {
            occ_offsets[i + 1] = occ_offsets[i] + occ_counts[i];
        }

        for (size_t i = 0; i < num_vertices; ++i) occ_counts[i] = 0;

        filtered_occ_data.resize(total_occurrences);
        vertex_to_edges_data.resize(total_occurrences);
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                size_t pos = occ_offsets[idx] + occ_counts[idx];
                filtered_occ_data[pos] = {arity, i};
                vertex_to_edges_data[pos] = {eid, i};
                occ_counts[idx]++;
            }
        });

        for (size_t i = 0; i < num_vertices; ++i) {
            auto begin = filtered_occ_data.begin() + occ_offsets[i];
            auto end = filtered_occ_data.begin() + occ_offsets[i + 1];
            insertion_sort(begin, end);
        }

        // Initialize hashes - ALWAYS compute initial hash from current state's structure
        // NOTE: We cannot use parent's cached FINAL hashes as INITIAL hashes here.
        // Parent cache contains hashes after N refinement iterations, but we need
        // iteration-0 hashes (degree + edge positions) for correct WL refinement.
        // The incremental benefit comes from the active vertex tracking in refinement,
        // not from reusing parent's final hashes as initial values.
        ArenaVector<uint64_t> current(*arena_, num_vertices);
        current.resize(num_vertices);

        for (size_t i = 0; i < num_vertices; ++i) {
            // Compute initial hash for this vertex based on current state's structure
            uint64_t h = FNV_OFFSET;
            size_t count = occ_offsets[i + 1] - occ_offsets[i];
            h = fnv_combine(h, count);
            for (size_t j = occ_offsets[i]; j < occ_offsets[i + 1]; ++j) {
                h = fnv_combine(h, filtered_occ_data[j].first);
                h = fnv_combine(h, filtered_occ_data[j].second);
            }
            current[i] = h;
        }

        // WL refinement with active vertex tracking
        ArenaVector<uint64_t> next(*arena_, num_vertices);
        next.resize(num_vertices);
        ArenaVector<uint64_t> neighbor_hashes(*arena_);
        size_t iteration = 0;

        // NOTE: We must mark ALL vertices as active, not just affected ones.
        // WL requires all vertices to refine synchronously - a vertex's iteration-N
        // hash depends on neighbors' iteration-(N-1) hashes. If we only refine
        // affected vertices first, they get "ahead" of unaffected vertices, breaking
        // the synchronization and producing incorrect hashes.
        ArenaVector<bool> active(*arena_, num_vertices);
        active.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) active[i] = true;

        ArenaVector<bool> next_active(*arena_, num_vertices);
        next_active.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) next_active[i] = false;

        size_t num_active = 0;
        for (size_t i = 0; i < num_vertices; ++i) {
            if (active[i]) ++num_active;
        }

        while (num_active > 0 && iteration < MAX_REFINEMENT_DEPTH) {
            ++iteration;
            size_t next_num_active = 0;

            for (size_t i = 0; i < num_vertices; ++i) {
                if (!active[i]) {
                    next[i] = current[i];
                    continue;
                }

                uint64_t h = current[i];
                neighbor_hashes.clear();

                for (size_t j = occ_offsets[i]; j < occ_offsets[i + 1]; ++j) {
                    EdgeId eid = vertex_to_edges_data[j].first;
                    uint8_t my_pos = vertex_to_edges_data[j].second;
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t k = 0; k < arity; ++k) {
                        if (k != my_pos) {
                            size_t neighbor_idx = get_vertex_idx(verts[k]);
                            neighbor_hashes.push_back(fnv_combine(current[neighbor_idx], k));
                        }
                    }
                }

                insertion_sort(neighbor_hashes.begin(), neighbor_hashes.end());
                for (uint64_t nh : neighbor_hashes) {
                    h = fnv_combine(h, nh);
                }

                next[i] = h;

                if (h != current[i]) {
                    for (size_t j = occ_offsets[i]; j < occ_offsets[i + 1]; ++j) {
                        EdgeId eid = vertex_to_edges_data[j].first;
                        uint8_t my_pos = vertex_to_edges_data[j].second;
                        uint8_t arity = edge_arities[eid];
                        const VertexId* verts = edge_vertices[eid];
                        for (uint8_t k = 0; k < arity; ++k) {
                            if (k != my_pos) {
                                size_t neighbor_idx = get_vertex_idx(verts[k]);
                                if (!next_active[neighbor_idx]) {
                                    next_active[neighbor_idx] = true;
                                    ++next_num_active;
                                }
                            }
                        }
                    }
                }
            }

            std::swap(current, next);
            std::swap(active, next_active);
            for (size_t i = 0; i < num_vertices; ++i) next_active[i] = false;
            num_active = next_num_active;
        }

        // Build cache for children
        cache.count = static_cast<uint32_t>(num_vertices);
        cache.vertices = arena_->allocate_array<VertexId>(cache.count);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.count);

        for (size_t i = 0; i < num_vertices; ++i) {
            cache.vertices[i] = vertices[i];
            cache.hashes[i] = current[i];
        }

        uint64_t hash = build_canonical_hash_dense(current, vertices, min_vertex, vertex_index,
                                                    state_edges, edge_vertices, edge_arities);

        return {hash, cache};
    }

    // =========================================================================
    // Edge Correspondence (O(E) algorithm)
    // =========================================================================

    // Find edge correspondence between two isomorphic states
    // Uses vertex subtree hashes - no O(E^2) search needed
    template<typename VertexAccessor, typename ArityAccessor>
    EdgeCorrespondence find_edge_correspondence(
        const SparseBitset& state1_edges,
        const SparseBitset& state2_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        EdgeCorrespondence result;

        // Compute vertex hashes for both states
        auto [hash1, cache1] = compute_state_hash_with_cache(state1_edges, edge_vertices, edge_arities);
        auto [hash2, cache2] = compute_state_hash_with_cache(state2_edges, edge_vertices, edge_arities);

        // Quick check: must have same canonical hash
        if (hash1 != hash2) {
            return result;  // Not isomorphic
        }

        // Must have same number of vertices
        if (cache1.count != cache2.count) {
            return result;
        }

        // Build edge signature -> list of edges map for state2
        // Edge signature = tuple of vertex subtree hashes (in order)
        // Multiple edges can have the same signature (automorphic edges)
        std::unordered_map<uint64_t, std::vector<EdgeId>> edge2_by_sig;

        state2_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            uint64_t sig = compute_edge_signature(verts, arity, cache2);
            edge2_by_sig[sig].push_back(eid);
        });

        // Collect edges from state1
        std::vector<EdgeId> edges1;
        state1_edges.for_each([&](EdgeId eid) {
            edges1.push_back(eid);
        });

        // Collect edges from state2
        size_t edge2_count = 0;
        state2_edges.for_each([&](EdgeId) { ++edge2_count; });

        if (edges1.size() != edge2_count) {
            return result;  // Different edge counts
        }

        // Allocate result arrays
        result.count = static_cast<uint32_t>(edges1.size());
        result.state1_edges = arena_->allocate_array<EdgeId>(result.count);
        result.state2_edges = arena_->allocate_array<EdgeId>(result.count);

        // Track which edges from state2 have been used
        std::unordered_map<uint64_t, size_t> sig_next_idx;  // For automorphic edges, track next available

        // Find correspondence for each edge in state1
        for (size_t i = 0; i < edges1.size(); ++i) {
            EdgeId e1 = edges1[i];
            uint8_t arity = edge_arities[e1];
            const VertexId* verts = edge_vertices[e1];

            uint64_t sig = compute_edge_signature(verts, arity, cache1);

            auto e2_it = edge2_by_sig.find(sig);
            if (e2_it == edge2_by_sig.end()) {
                // No matching edge found - states not actually isomorphic
                result.valid = false;
                return result;
            }

            // For automorphic edges (same signature), use next available
            // This is valid because automorphic edges are interchangeable
            size_t& idx = sig_next_idx[sig];
            if (idx >= e2_it->second.size()) {
                // More edges in state1 with this signature than in state2
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
    // Event Signature Computation (for Full deduplication)
    // =========================================================================

    // Compute event signature for Full deduplication mode
    template<typename VertexAccessor, typename ArityAccessor>
    EventSignature compute_event_signature(
        uint64_t input_state_hash,
        uint64_t output_state_hash,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& input_cache,
        const VertexHashCache& output_cache,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        EventSignature sig;
        sig.input_state_hash = input_state_hash;
        sig.output_state_hash = output_state_hash;

        // Compute consumed edges signature
        sig.consumed_edges_sig = compute_edge_set_signature(
            consumed_edges, num_consumed, input_cache, edge_vertices, edge_arities);

        // Compute produced edges signature
        sig.produced_edges_sig = compute_edge_set_signature(
            produced_edges, num_produced, output_cache, edge_vertices, edge_arities);

        return sig;
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    size_t num_vertices() const { return vertices_.size(); }

private:
    ConcurrentHeterogeneousArena* arena_;

    // Per-vertex edge occurrences (global, grows as edges are added)
    SegmentedArray<LockFreeList<EdgeOccurrence>> vertices_;

    // Version counters for invalidation (incremented on edge addition)
    SegmentedArray<std::atomic<uint32_t>> vertex_versions_;

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    // Use fnv_hash from types.hpp - alias for compatibility
    static uint64_t fnv_combine(uint64_t h, uint64_t value) {
        return fnv_hash(h, value);
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

    void ensure_vertex_capacity(VertexId vertex) {
        vertices_.get_or_default(vertex, *arena_);
        vertex_versions_.get_or_default(vertex, *arena_);
    }

    void add_edge_to_vertex(VertexId vertex, EdgeId edge, uint8_t position, uint8_t arity) {
        EdgeOccurrence occ(edge, position, arity);
        vertices_[vertex].push(occ, *arena_);
        vertex_versions_[vertex].fetch_add(1, std::memory_order_release);
    }

    // Build canonical hash from vertex hashes and edge structure
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t build_canonical_hash(
        const std::vector<uint64_t>& vertex_hashes,
        const std::vector<VertexId>& vertices,
        const std::vector<size_t>& vertex_index,
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        // Sort vertices by hash for canonical ordering
        std::vector<std::pair<uint64_t, size_t>> sorted_hashes;
        sorted_hashes.reserve(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            sorted_hashes.emplace_back(vertex_hashes[i], i);
        }
        std::sort(sorted_hashes.begin(), sorted_hashes.end());

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

        // Build and sort canonical edge representations
        std::vector<std::vector<uint64_t>> canonical_edges;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            std::vector<uint64_t> edge_rep;
            edge_rep.reserve(arity);
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = vertex_index[verts[i]];
                edge_rep.push_back(vertex_hashes[idx]);
            }
            // DO NOT sort - vertex order matters for directed hyperedges
            canonical_edges.push_back(std::move(edge_rep));
        });

        // Sort edges for canonical ordering
        std::sort(canonical_edges.begin(), canonical_edges.end());

        // Hash edge structure
        for (const auto& edge : canonical_edges) {
            hash = fnv_combine(hash, edge.size());
            for (uint64_t vh : edge) {
                hash = fnv_combine(hash, vh);
            }
        }

        return hash;
    }

    // Arena-based version for optimized compute_state_hash with dense index
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t build_canonical_hash_dense(
        const ArenaVector<uint64_t>& vertex_hashes,
        const ArenaVector<VertexId>& vertices,
        VertexId min_vertex,
        const ArenaVector<size_t>& vertex_index,
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        // Direct array lookup - no hash overhead
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            return vertex_index[v - min_vertex];
        };

        // Sort vertices by hash for canonical ordering
        ArenaVector<std::pair<uint64_t, size_t>> sorted_hashes(*arena_, vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            sorted_hashes.push_back({vertex_hashes[i], i});
        }
        std::sort(sorted_hashes.begin(), sorted_hashes.end());

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

        // Build canonical edge hashes (just the hash, not full representation)
        ArenaVector<uint64_t> edge_hashes(*arena_);

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            // Compute edge hash directly without storing full representation
            uint64_t edge_hash = FNV_OFFSET;
            edge_hash = fnv_combine(edge_hash, arity);
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                edge_hash = fnv_combine(edge_hash, vertex_hashes[idx]);
            }
            edge_hashes.push_back(edge_hash);
        });

        // Sort edge hashes for canonical ordering
        std::sort(edge_hashes.begin(), edge_hashes.end());

        // Hash edge structure
        hash = fnv_combine(hash, edge_hashes.size());
        for (size_t i = 0; i < edge_hashes.size(); ++i) {
            hash = fnv_combine(hash, edge_hashes[i]);
        }

        return hash;
    }

    // Arena-based version for optimized compute_state_hash (legacy, uses sparse index)
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t build_canonical_hash_arena(
        const ArenaVector<uint64_t>& vertex_hashes,
        const ArenaVector<VertexId>& vertices,
        const std::unordered_map<VertexId, size_t>& vertex_index,
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        // Sort vertices by hash for canonical ordering
        ArenaVector<std::pair<uint64_t, size_t>> sorted_hashes(*arena_, vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            sorted_hashes.push_back({vertex_hashes[i], i});
        }
        std::sort(sorted_hashes.begin(), sorted_hashes.end());

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

        // Build canonical edge hashes (just the hash, not full representation)
        ArenaVector<uint64_t> edge_hashes(*arena_);

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            // Compute edge hash directly without storing full representation
            uint64_t edge_hash = FNV_OFFSET;
            edge_hash = fnv_combine(edge_hash, arity);
            for (uint8_t i = 0; i < arity; ++i) {
                auto it = vertex_index.find(verts[i]);
                size_t idx = it->second;
                edge_hash = fnv_combine(edge_hash, vertex_hashes[idx]);
            }
            edge_hashes.push_back(edge_hash);
        });

        // Sort edge hashes for canonical ordering
        std::sort(edge_hashes.begin(), edge_hashes.end());

        // Hash edge structure
        hash = fnv_combine(hash, edge_hashes.size());
        for (size_t i = 0; i < edge_hashes.size(); ++i) {
            hash = fnv_combine(hash, edge_hashes[i]);
        }

        return hash;
    }

    // Compute edge signature from vertex hashes
    uint64_t compute_edge_signature(
        const VertexId* vertices,
        uint8_t arity,
        const VertexHashCache& cache
    ) const {
        uint64_t sig = FNV_OFFSET;
        sig = fnv_combine(sig, arity);

        for (uint8_t i = 0; i < arity; ++i) {
            uint64_t vh = cache.lookup(vertices[i]);
            sig = fnv_combine(sig, vh);
        }

        return sig;
    }

    // Compute signature for a set of edges
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t compute_edge_set_signature(
        const EdgeId* edges,
        uint8_t count,
        const VertexHashCache& cache,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        // Collect individual edge signatures
        std::vector<uint64_t> edge_sigs;
        edge_sigs.reserve(count);

        for (uint8_t i = 0; i < count; ++i) {
            EdgeId eid = edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            edge_sigs.push_back(compute_edge_signature(verts, arity, cache));
        }

        // Sort for canonical ordering (order of edges in event shouldn't matter)
        insertion_sort(edge_sigs.begin(), edge_sigs.end());

        // Combine into single signature
        uint64_t sig = FNV_OFFSET;
        sig = fnv_combine(sig, count);
        for (uint64_t es : edge_sigs) {
            sig = fnv_combine(sig, es);
        }

        return sig;
    }
};

}  // namespace hypergraph::unified
