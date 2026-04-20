#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>

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

    // Precomputed adjacency for refinement
    struct HypergraphAdj {
        uint32_t num_vertices;
        std::vector<std::vector<VertexOccurrence>> vertex_edges; // vertex_index -> occurrences
        const std::vector<std::vector<VertexId>>* edges;
        std::unordered_map<VertexId, uint32_t> orig_to_idx;
        std::vector<VertexId> idx_to_orig;
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
