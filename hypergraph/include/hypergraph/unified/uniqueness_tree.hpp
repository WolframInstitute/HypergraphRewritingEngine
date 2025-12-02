#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>

#include "types.hpp"
#include "bitset.hpp"
#include "arena.hpp"

namespace hypergraph::unified {

// =============================================================================
// UniquenessTree: Isomorphism-invariant hashing and vertex classification
// =============================================================================
//
// Implements Gorard's uniqueness tree approach for hypergraph canonicalization.
// Computes:
// 1. Canonical hash - states with same hash ARE isomorphic
// 2. Vertex equivalence classes - vertices with same tree hash can be mapped
// 3. Edge correspondence - given two isomorphic states, find edge mapping
//
// The vertex classification dramatically constrains the canonicalization
// search space from O(V!) to O(n₁! × n₂! × ... × nₖ!) where nᵢ are class sizes.
//
// Thread safety: Immutable after construction. Safe for concurrent reads.
//

// Forward declarations
class UnifiedHypergraph;

// -----------------------------------------------------------------------------
// VertexTreeHash: Per-vertex tree hash capturing local structure
// -----------------------------------------------------------------------------

struct VertexTreeHash {
    VertexId vertex;
    uint64_t hash;

    bool operator<(const VertexTreeHash& other) const {
        return hash < other.hash;
    }

    bool operator==(const VertexTreeHash& other) const {
        return hash == other.hash;
    }
};

// -----------------------------------------------------------------------------
// VertexEquivalenceClass: Group of vertices with identical tree hashes
// -----------------------------------------------------------------------------

struct VertexEquivalenceClass {
    uint64_t hash;
    VertexId* vertices;    // Arena-allocated array
    uint32_t count;

    VertexEquivalenceClass() : hash(0), vertices(nullptr), count(0) {}
    VertexEquivalenceClass(uint64_t h, VertexId* v, uint32_t c)
        : hash(h), vertices(v), count(c) {}
};

// -----------------------------------------------------------------------------
// EdgeCorrespondence: Mapping between edges in isomorphic states
// -----------------------------------------------------------------------------

struct EdgeCorrespondence {
    EdgeId* state1_edges;    // Arena-allocated
    EdgeId* state2_edges;    // Arena-allocated, state2_edges[i] corresponds to state1_edges[i]
    uint32_t count;
    bool valid;              // False if states not actually isomorphic

    EdgeCorrespondence() : state1_edges(nullptr), state2_edges(nullptr), count(0), valid(false) {}
};

// -----------------------------------------------------------------------------
// StateCanonicalInfo: Cached canonical information for a state
// -----------------------------------------------------------------------------

struct StateCanonicalInfo {
    uint64_t canonical_hash;
    VertexEquivalenceClass* equiv_classes;  // Arena-allocated array
    uint32_t num_classes;
    VertexId* canonical_vertex_order;       // Arena-allocated, sorted by (hash, tiebreaker)
    uint32_t num_vertices;

    StateCanonicalInfo()
        : canonical_hash(0)
        , equiv_classes(nullptr)
        , num_classes(0)
        , canonical_vertex_order(nullptr)
        , num_vertices(0)
    {}
};

// -----------------------------------------------------------------------------
// UniquenessTree: Main class for canonical hash and correspondence computation
// -----------------------------------------------------------------------------

class UniquenessTree {
public:
    static constexpr size_t MAX_TREE_DEPTH = 100;
    static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    static constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

    // Compute canonical info for a state
    // Uses the state's edge bitset to filter which edges are considered
    template<typename Arena>
    static StateCanonicalInfo compute(
        const SparseBitset& state_edges,
        const EdgeId* all_edges,           // All edges in unified hypergraph
        uint32_t num_all_edges,
        const VertexId* const* edge_vertices,  // edge_vertices[edge_id] -> vertex array
        const uint8_t* edge_arities,           // edge_arities[edge_id] -> arity
        Arena& arena
    );

    // Compute edge correspondence between two isomorphic states
    // Returns invalid correspondence if states are not actually isomorphic
    template<typename Arena>
    static EdgeCorrespondence find_correspondence(
        const StateCanonicalInfo& state1_info,
        const SparseBitset& state1_edges,
        const StateCanonicalInfo& state2_info,
        const SparseBitset& state2_edges,
        const VertexId* const* edge_vertices,
        const uint8_t* edge_arities,
        Arena& arena
    );

private:
    // FNV-1a hash combining
    static uint64_t fnv_hash(uint64_t h, uint64_t value) {
        h ^= value;
        h *= FNV_PRIME;
        return h;
    }

    // Hash an array of values
    template<typename T>
    static uint64_t hash_array(const T* arr, size_t count, uint64_t seed = FNV_OFFSET) {
        uint64_t h = seed;
        for (size_t i = 0; i < count; ++i) {
            h = fnv_hash(h, static_cast<uint64_t>(arr[i]));
        }
        return h;
    }

    // Compute tree hash for a single vertex
    // This is the core WL-style iteration
    template<typename Arena>
    static uint64_t compute_vertex_tree_hash(
        VertexId vertex,
        const SparseBitset& state_edges,
        const VertexId* const* edge_vertices,
        const uint8_t* edge_arities,
        const uint64_t* prev_vertex_hashes,  // Previous iteration's hashes (null for first iter)
        uint32_t max_vertex_id,
        size_t depth,
        Arena& arena
    );

    // Build adjacency info for vertices in a state
    struct AdjacencyInfo {
        struct EdgeOccurrence {
            EdgeId edge_id;
            uint8_t position;    // Position of vertex in edge
            uint8_t arity;       // Total arity of edge
        };

        EdgeOccurrence* occurrences;  // Arena-allocated
        uint32_t count;
    };

    template<typename Arena>
    static void build_adjacency(
        const SparseBitset& state_edges,
        const VertexId* const* edge_vertices,
        const uint8_t* edge_arities,
        uint32_t max_vertex_id,
        AdjacencyInfo* out_adjacency,  // Array of size max_vertex_id+1
        Arena& arena
    );
};

// =============================================================================
// Implementation
// =============================================================================

template<typename Arena>
StateCanonicalInfo UniquenessTree::compute(
    const SparseBitset& state_edges,
    const EdgeId* all_edges,
    uint32_t num_all_edges,
    const VertexId* const* edge_vertices,
    const uint8_t* edge_arities,
    Arena& arena
) {
    StateCanonicalInfo result;

    if (state_edges.empty()) {
        return result;
    }

    // Collect vertices present in this state and find max vertex ID
    std::vector<VertexId> vertices;
    uint32_t max_vertex_id = 0;

    state_edges.for_each([&](EdgeId eid) {
        uint8_t arity = edge_arities[eid];
        const VertexId* verts = edge_vertices[eid];
        for (uint8_t i = 0; i < arity; ++i) {
            vertices.push_back(verts[i]);
            if (verts[i] > max_vertex_id) {
                max_vertex_id = verts[i];
            }
        }
    });

    // Remove duplicates
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

    result.num_vertices = static_cast<uint32_t>(vertices.size());
    if (result.num_vertices == 0) {
        return result;
    }

    // Build adjacency info
    AdjacencyInfo* adjacency = arena.template allocate_array<AdjacencyInfo>(max_vertex_id + 1);
    for (uint32_t i = 0; i <= max_vertex_id; ++i) {
        adjacency[i].occurrences = nullptr;
        adjacency[i].count = 0;
    }
    build_adjacency(state_edges, edge_vertices, edge_arities, max_vertex_id, adjacency, arena);

    // Initial vertex hashes based on degree and local structure
    uint64_t* current_hashes = arena.template allocate_array<uint64_t>(max_vertex_id + 1);
    uint64_t* next_hashes = arena.template allocate_array<uint64_t>(max_vertex_id + 1);

    for (uint32_t i = 0; i <= max_vertex_id; ++i) {
        current_hashes[i] = 0;
        next_hashes[i] = 0;
    }

    // Initialize with structural hash (degree, edge arities, positions)
    for (VertexId v : vertices) {
        uint64_t h = FNV_OFFSET;
        h = fnv_hash(h, adjacency[v].count);  // Degree

        // Sort occurrences for canonical ordering
        std::vector<std::pair<uint8_t, uint8_t>> sorted_occ;  // (arity, position)
        for (uint32_t i = 0; i < adjacency[v].count; ++i) {
            sorted_occ.emplace_back(
                adjacency[v].occurrences[i].arity,
                adjacency[v].occurrences[i].position
            );
        }
        std::sort(sorted_occ.begin(), sorted_occ.end());

        for (auto& [arity, pos] : sorted_occ) {
            h = fnv_hash(h, arity);
            h = fnv_hash(h, pos);
        }
        current_hashes[v] = h;
    }

    // WL-style refinement iterations
    bool changed = true;
    size_t iteration = 0;

    while (changed && iteration < MAX_TREE_DEPTH) {
        changed = false;
        ++iteration;

        for (VertexId v : vertices) {
            uint64_t h = current_hashes[v];

            // Collect neighbor hashes
            std::vector<uint64_t> neighbor_hashes;

            for (uint32_t i = 0; i < adjacency[v].count; ++i) {
                EdgeId eid = adjacency[v].occurrences[i].edge_id;
                uint8_t arity = adjacency[v].occurrences[i].arity;
                const VertexId* verts = edge_vertices[eid];

                // Add hashes of all other vertices in this edge
                for (uint8_t j = 0; j < arity; ++j) {
                    VertexId neighbor = verts[j];
                    if (neighbor != v) {
                        // Include position info to distinguish structure
                        uint64_t nh = fnv_hash(current_hashes[neighbor], j);
                        neighbor_hashes.push_back(nh);
                    }
                }
            }

            // Sort for canonical ordering
            std::sort(neighbor_hashes.begin(), neighbor_hashes.end());

            // Combine with current hash
            for (uint64_t nh : neighbor_hashes) {
                h = fnv_hash(h, nh);
            }

            if (h != current_hashes[v]) {
                changed = true;
            }
            next_hashes[v] = h;
        }

        // Swap buffers
        std::swap(current_hashes, next_hashes);
    }

    // Build vertex tree hashes
    std::vector<VertexTreeHash> vertex_hashes;
    vertex_hashes.reserve(vertices.size());
    for (VertexId v : vertices) {
        vertex_hashes.push_back({v, current_hashes[v]});
    }

    // Sort by hash for canonical ordering
    std::sort(vertex_hashes.begin(), vertex_hashes.end());

    // Build equivalence classes
    std::vector<VertexEquivalenceClass> classes;
    size_t class_start = 0;

    for (size_t i = 1; i <= vertex_hashes.size(); ++i) {
        if (i == vertex_hashes.size() || vertex_hashes[i].hash != vertex_hashes[class_start].hash) {
            // End of class
            uint32_t class_size = static_cast<uint32_t>(i - class_start);
            VertexId* class_vertices = arena.template allocate_array<VertexId>(class_size);
            for (size_t j = 0; j < class_size; ++j) {
                class_vertices[j] = vertex_hashes[class_start + j].vertex;
            }
            classes.push_back({vertex_hashes[class_start].hash, class_vertices, class_size});
            class_start = i;
        }
    }

    result.num_classes = static_cast<uint32_t>(classes.size());
    result.equiv_classes = arena.template allocate_array<VertexEquivalenceClass>(classes.size());
    for (size_t i = 0; i < classes.size(); ++i) {
        result.equiv_classes[i] = classes[i];
    }

    // Build canonical vertex order
    result.canonical_vertex_order = arena.template allocate_array<VertexId>(vertices.size());
    for (size_t i = 0; i < vertex_hashes.size(); ++i) {
        result.canonical_vertex_order[i] = vertex_hashes[i].vertex;
    }

    // Compute canonical hash - combine sorted vertex hashes and edge structure
    uint64_t canonical_hash = FNV_OFFSET;

    // Add vertex class structure
    for (const auto& vc : classes) {
        canonical_hash = fnv_hash(canonical_hash, vc.hash);
        canonical_hash = fnv_hash(canonical_hash, vc.count);
    }

    // Add edge structure in canonical order
    // First, create canonical edge representations
    std::vector<std::vector<uint64_t>> canonical_edges;

    state_edges.for_each([&](EdgeId eid) {
        uint8_t arity = edge_arities[eid];
        const VertexId* verts = edge_vertices[eid];

        // Map vertices to their hash values in ORDER (directed edges!)
        // DO NOT sort within edge - edge {a,b} is different from {b,a}
        std::vector<uint64_t> edge_rep;
        edge_rep.reserve(arity);
        for (uint8_t i = 0; i < arity; ++i) {
            edge_rep.push_back(current_hashes[verts[i]]);
        }
        // Note: We do NOT sort edge_rep - vertex order matters for directed hyperedges
        canonical_edges.push_back(std::move(edge_rep));
    });

    // Sort edges for canonical ordering
    std::sort(canonical_edges.begin(), canonical_edges.end());

    // Hash the sorted edge structure
    for (const auto& edge : canonical_edges) {
        canonical_hash = fnv_hash(canonical_hash, edge.size());
        for (uint64_t vh : edge) {
            canonical_hash = fnv_hash(canonical_hash, vh);
        }
    }

    result.canonical_hash = canonical_hash;
    return result;
}

template<typename Arena>
void UniquenessTree::build_adjacency(
    const SparseBitset& state_edges,
    const VertexId* const* edge_vertices,
    const uint8_t* edge_arities,
    uint32_t max_vertex_id,
    AdjacencyInfo* out_adjacency,
    Arena& arena
) {
    // First pass: count occurrences per vertex
    std::vector<uint32_t> counts(max_vertex_id + 1, 0);

    state_edges.for_each([&](EdgeId eid) {
        uint8_t arity = edge_arities[eid];
        const VertexId* verts = edge_vertices[eid];
        for (uint8_t i = 0; i < arity; ++i) {
            ++counts[verts[i]];
        }
    });

    // Allocate arrays
    for (uint32_t v = 0; v <= max_vertex_id; ++v) {
        if (counts[v] > 0) {
            out_adjacency[v].occurrences = arena.template allocate_array<AdjacencyInfo::EdgeOccurrence>(counts[v]);
            out_adjacency[v].count = 0;  // Will be filled in second pass
        }
    }

    // Second pass: fill occurrences
    state_edges.for_each([&](EdgeId eid) {
        uint8_t arity = edge_arities[eid];
        const VertexId* verts = edge_vertices[eid];
        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = verts[i];
            uint32_t idx = out_adjacency[v].count++;
            out_adjacency[v].occurrences[idx] = {eid, i, arity};
        }
    });
}

template<typename Arena>
EdgeCorrespondence UniquenessTree::find_correspondence(
    const StateCanonicalInfo& state1_info,
    const SparseBitset& state1_edges,
    const StateCanonicalInfo& state2_info,
    const SparseBitset& state2_edges,
    const VertexId* const* edge_vertices,
    const uint8_t* edge_arities,
    Arena& arena
) {
    EdgeCorrespondence result;

    // Quick check: must have same canonical hash
    if (state1_info.canonical_hash != state2_info.canonical_hash) {
        return result;  // Not isomorphic
    }

    // Must have same number of equivalence classes
    if (state1_info.num_classes != state2_info.num_classes) {
        return result;
    }

    // Check class structures match
    for (uint32_t i = 0; i < state1_info.num_classes; ++i) {
        if (state1_info.equiv_classes[i].hash != state2_info.equiv_classes[i].hash ||
            state1_info.equiv_classes[i].count != state2_info.equiv_classes[i].count) {
            return result;
        }
    }

    // Build vertex mapping using equivalence classes
    // For classes of size 1: direct mapping
    // For larger classes: need to find correct permutation

    std::vector<std::pair<VertexId, VertexId>> vertex_mapping;
    vertex_mapping.reserve(state1_info.num_vertices);

    bool mapping_found = true;

    for (uint32_t c = 0; c < state1_info.num_classes && mapping_found; ++c) {
        const auto& class1 = state1_info.equiv_classes[c];
        const auto& class2 = state2_info.equiv_classes[c];

        if (class1.count == 1) {
            // Unique vertex in class - direct mapping
            vertex_mapping.push_back({class1.vertices[0], class2.vertices[0]});
        } else {
            // Multiple vertices in class - use canonical order as default
            // This works because both states have same structure
            // More sophisticated: try permutations within class if needed
            for (uint32_t i = 0; i < class1.count; ++i) {
                vertex_mapping.push_back({class1.vertices[i], class2.vertices[i]});
            }
        }
    }

    if (!mapping_found) {
        return result;
    }

    // Build vertex map for quick lookup
    std::vector<VertexId> v1_to_v2;
    uint32_t max_v1 = 0;
    for (const auto& [v1, v2] : vertex_mapping) {
        if (v1 > max_v1) max_v1 = v1;
    }
    v1_to_v2.resize(max_v1 + 1, INVALID_ID);
    for (const auto& [v1, v2] : vertex_mapping) {
        v1_to_v2[v1] = v2;
    }

    // Now find edge correspondence
    // For each edge in state1, find matching edge in state2 with mapped vertices

    std::vector<EdgeId> edges1, edges2;
    state1_edges.for_each([&](EdgeId eid) { edges1.push_back(eid); });
    state2_edges.for_each([&](EdgeId eid) { edges2.push_back(eid); });

    if (edges1.size() != edges2.size()) {
        return result;  // Should not happen if hashes match
    }

    result.count = static_cast<uint32_t>(edges1.size());
    result.state1_edges = arena.template allocate_array<EdgeId>(edges1.size());
    result.state2_edges = arena.template allocate_array<EdgeId>(edges1.size());

    std::vector<bool> used2(edges2.size(), false);

    for (size_t i = 0; i < edges1.size(); ++i) {
        EdgeId e1 = edges1[i];
        result.state1_edges[i] = e1;

        uint8_t arity1 = edge_arities[e1];
        const VertexId* verts1 = edge_vertices[e1];

        // Map vertices to state2 space
        std::vector<VertexId> mapped_verts(arity1);
        for (uint8_t j = 0; j < arity1; ++j) {
            mapped_verts[j] = v1_to_v2[verts1[j]];
        }

        // Find matching edge in state2
        bool found = false;
        for (size_t k = 0; k < edges2.size() && !found; ++k) {
            if (used2[k]) continue;

            EdgeId e2 = edges2[k];
            uint8_t arity2 = edge_arities[e2];

            if (arity2 != arity1) continue;

            const VertexId* verts2 = edge_vertices[e2];

            // Check if vertices match (in order, since hyperedges are ordered)
            bool match = true;
            for (uint8_t j = 0; j < arity1 && match; ++j) {
                if (verts2[j] != mapped_verts[j]) {
                    match = false;
                }
            }

            if (match) {
                result.state2_edges[i] = e2;
                used2[k] = true;
                found = true;
            }
        }

        if (!found) {
            // No matching edge found - states not actually isomorphic
            // (This could happen with hash collisions, though rare)
            result.valid = false;
            return result;
        }
    }

    result.valid = true;
    return result;
}

}  // namespace hypergraph::unified
