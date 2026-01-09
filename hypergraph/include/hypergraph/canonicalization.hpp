#ifndef HYPERGRAPH_CANONICALIZATION_HPP
#define HYPERGRAPH_CANONICALIZATION_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <limits>
#include <cstdint>

#include "types.hpp"

namespace hypergraph {

// Use VertexId from types.hpp (uint32_t)
// Use INVALID_ID from types.hpp for invalid vertex marker
constexpr VertexId INVALID_VERTEX = INVALID_ID;

/**
 * Canonical representation of a hypergraph.
 * Two hypergraphs with the same canonical form are structurally identical.
 */
struct CanonicalForm {
    std::vector<std::vector<VertexId>> edges;  // Edges with canonical vertex indices
    VertexId vertex_count;

    bool operator==(const CanonicalForm& other) const {
        return vertex_count == other.vertex_count && edges == other.edges;
    }

    bool operator!=(const CanonicalForm& other) const {
        return !(*this == other);
    }

    // String representation for debugging
    std::string to_string() const;
};

/**
 * Vertex mapping from original hypergraph to canonical form.
 */
struct VertexMapping {
    std::unordered_map<VertexId, VertexId> original_to_canonical;
    std::vector<VertexId> canonical_to_original;

    // Edge permutation: original_edge_index -> canonical_edge_index
    std::unordered_map<std::size_t, std::size_t> original_edge_to_canonical;
    std::vector<std::size_t> canonical_edge_to_original;

    // Apply the mapping to a vertex ID
    VertexId map_vertex(VertexId original) const {
        auto it = original_to_canonical.find(original);
        return (it != original_to_canonical.end()) ? it->second : INVALID_VERTEX;
    }

    // Get original vertex from canonical index
    VertexId get_original(VertexId canonical) const {
        return (canonical < canonical_to_original.size()) ?
               canonical_to_original[canonical] : INVALID_VERTEX;
    }

    // Apply edge permutation: original edge index -> canonical edge index
    std::size_t map_edge(std::size_t original_idx) const {
        auto it = original_edge_to_canonical.find(original_idx);
        return (it != original_edge_to_canonical.end()) ? it->second : static_cast<std::size_t>(-1);
    }
};

/**
 * Result of canonicalization containing both the canonical form and mapping.
 */
struct CanonicalizationResult {
    CanonicalForm canonical_form;
    VertexMapping vertex_mapping;

    // Check if two hypergraphs are isomorphic via their canonical forms
    static bool are_isomorphic(const CanonicalizationResult& a, const CanonicalizationResult& b) {
        return a.canonical_form == b.canonical_form;
    }
};

/**
 * Canonicalization algorithm using Wolfram's approach.
 *
 * The algorithm works as follows:
 * 1. Try all permutations of vertices
 * 2. For each permutation, map vertices to 0, 1, 2, ...
 * 3. Sort the resulting edges
 * 4. Keep the lexicographically smallest result
 *
 * This finds the true canonical form but has factorial complexity.
 */
class Canonicalizer {
private:
    /**
     * Apply the Wolfram canonicalization algorithm using permutations.
     * Returns the canonical edges and populates the vertex mapping.
     */
    template<typename VertexType>
    std::vector<std::vector<VertexType>> wolfram_canonical_hypergraph(
        const std::vector<std::vector<VertexType>>& edges,
        VertexMapping& mapping) const;

    /**
     * Helper to convert edge list with vertex IDs to size_t.
     */
    std::vector<std::vector<std::size_t>> edges_to_size_t(
        const std::vector<std::vector<VertexId>>& edges) const;

public:
    /**
     * Canonicalize using raw edge vectors.
     */
    template<typename VertexType>
    CanonicalizationResult canonicalize_edges(const std::vector<std::vector<VertexType>>& edges) const;
};

} // namespace hypergraph

#endif // HYPERGRAPH_CANONICALIZATION_HPP