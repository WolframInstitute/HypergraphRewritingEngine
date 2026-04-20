#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>

#include "types.hpp"
#include "canonical_types.hpp"

namespace hypergraph {

// Ordered partition of vertex indices for IR refinement
struct IRPartition {
    std::vector<std::vector<uint32_t>> cells;
    std::vector<uint32_t> vertex_to_cell;  // vertex_index -> cell index

    bool is_discrete() const;
    size_t first_non_singleton() const;
};

// McKay-style Individualization-Refinement canonicalizer for directed hypergraphs.
// Works directly on the original vertex set using hypergraph-aware refinement.
// Produces the lexicographically smallest canonical form (matching brute-force).
class IRCanonicalizer {
public:
    CanonicalizationResult canonicalize_edges(
        const std::vector<std::vector<VertexId>>& edges) const;

    uint64_t compute_canonical_hash(
        const std::vector<std::vector<VertexId>>& edges) const;

    bool are_isomorphic(
        const std::vector<std::vector<VertexId>>& edges1,
        const std::vector<std::vector<VertexId>>& edges2) const;

    EdgeCorrespondence find_edge_correspondence(
        const std::vector<std::vector<VertexId>>& edges1,
        const std::vector<std::vector<VertexId>>& edges2) const;

private:
    // Vertex occurrence in an edge: (edge_index, position, arity)
    struct VertexOccurrence {
        uint32_t edge_idx;
        uint8_t position;
        uint8_t arity;
    };

    // Precomputed adjacency for refinement.
    //
    // Vertex indexing: the canonicalizer works on a compact 0..num_vertices-1 index
    // space, derived from the original VertexIds by sorting them. idx_to_orig is a
    // sorted vector of the original VertexIds; to map original → index we do a
    // binary search on idx_to_orig (O(log V)) instead of a hash-map lookup — faster
    // and more cache-friendly for the typical V ≤ few hundred on the hot path.
    struct HypergraphAdj {
        uint32_t num_vertices;
        std::vector<std::vector<VertexOccurrence>> vertex_edges;  // index -> occurrences
        const std::vector<std::vector<VertexId>>* edges;
        std::vector<VertexId> idx_to_orig;  // sorted ascending

        uint32_t index_of(VertexId v) const {
            auto it = std::lower_bound(idx_to_orig.begin(), idx_to_orig.end(), v);
            // Caller guarantees v is present; asserting would add a branch in a hot
            // loop. The invariant is maintained by build_adjacency.
            return static_cast<uint32_t>(it - idx_to_orig.begin());
        }
    };

    HypergraphAdj build_adjacency(const std::vector<std::vector<VertexId>>& edges) const;
    IRPartition initial_partition(const HypergraphAdj& adj) const;
    bool refine(const HypergraphAdj& adj, IRPartition& pi) const;
    IRPartition individualize(const IRPartition& pi, size_t cell_idx, uint32_t v) const;
    std::vector<uint32_t> extract_labeling(const IRPartition& pi) const;

    std::vector<std::vector<VertexId>> apply_labeling(
        const std::vector<std::vector<VertexId>>& edges,
        const HypergraphAdj& adj,
        const std::vector<uint32_t>& labeling) const;

    CanonicalizationResult build_result(
        const std::vector<std::vector<VertexId>>& edges,
        const HypergraphAdj& adj,
        const std::vector<uint32_t>& labeling) const;
};

}  // namespace hypergraph
