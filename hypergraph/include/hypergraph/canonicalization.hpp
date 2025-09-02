#ifndef HYPERGRAPH_CANONICALIZATION_HPP
#define HYPERGRAPH_CANONICALIZATION_HPP

#include <hypergraph/hypergraph.hpp>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <limits>

namespace hypergraph {

/**
 * Canonical representation of a hypergraph.
 * Two hypergraphs with the same canonical form are structurally identical.
 */
struct CanonicalForm {
    std::vector<std::vector<std::size_t>> edges;  // Edges with canonical vertex indices
    std::size_t vertex_count;
    
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
    std::unordered_map<VertexId, std::size_t> original_to_canonical;
    std::vector<VertexId> canonical_to_original;
    
    // Apply the mapping to a vertex ID
    std::size_t map_vertex(VertexId original) const {
        auto it = original_to_canonical.find(original);
        return (it != original_to_canonical.end()) ? it->second : std::numeric_limits<std::size_t>::max();
    }
    
    // Get original vertex from canonical index
    VertexId get_original(std::size_t canonical) const {
        return (canonical < canonical_to_original.size()) ? 
               canonical_to_original[canonical] : INVALID_VERTEX;
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
 * Canonicalization algorithm based on vertex degree and edge connectivity.
 * 
 * The algorithm works as follows:
 * 1. Compute vertex signatures (degree, edge arities, connectivity patterns)
 * 2. Sort vertices by their signatures to establish canonical ordering
 * 3. Handle ties using stable tiebreakers (vertex ID, edge structure)
 * 4. Remap all vertices to canonical indices 0, 1, 2, ...
 * 5. Sort edges by their canonical vertex lists
 */
class Canonicalizer {
private:
    /**
     * Vertex signature used for canonical ordering.
     */
    struct VertexSignature {
        VertexId vertex_id;
        std::size_t degree;                           // Number of edges containing this vertex
        std::vector<std::size_t> edge_arities;        // Sorted arities of incident edges
        std::vector<std::size_t> position_counts;     // Count of positions in each arity
        
        // For comparison (lexicographic ordering)
        bool operator<(const VertexSignature& other) const;
        bool operator==(const VertexSignature& other) const;
    };
    
    /**
     * Compute signature for a vertex based on its connectivity.
     */
    VertexSignature compute_vertex_signature(const Hypergraph& hg, VertexId vertex_id) const;
    
    /**
     * Compute signatures for all vertices.
     */
    std::vector<VertexSignature> compute_all_signatures(const Hypergraph& hg) const;
    
    /**
     * Establish canonical vertex ordering based on signatures.
     */
    VertexMapping compute_vertex_mapping(const std::vector<VertexSignature>& signatures) const;
    
    /**
     * Apply vertex mapping to create canonical edge list.
     */
    std::vector<std::vector<std::size_t>> map_edges_to_canonical(
        const Hypergraph& hg, const VertexMapping& mapping) const;
    
    /**
     * Apply vertex mapping using insertion sort (optimized for nearly-sorted edge lists).
     */
    std::vector<std::vector<std::size_t>> map_edges_to_canonical_incremental(
        const Hypergraph& hg, const VertexMapping& mapping) const;
    
public:
    /**
     * Canonicalize a hypergraph, returning both canonical form and vertex mapping.
     */
    CanonicalizationResult canonicalize(const Hypergraph& hg) const;
    
    /**
     * Optimized canonicalization for rewritten graphs using insertion sort.
     * Assumes the input hypergraph is already close to canonical form (e.g., after localized edits).
     * This is much faster than full canonicalization when only a few edges have changed.
     */
    CanonicalizationResult canonicalize_rewritten(const Hypergraph& hg, 
                                                  const VertexMapping& previous_mapping) const;
    
    /**
     * Quick check if two hypergraphs are isomorphic (same canonical form).
     */
    bool are_isomorphic(const Hypergraph& a, const Hypergraph& b) const {
        return canonicalize(a).canonical_form == canonicalize(b).canonical_form;
    }
};

} // namespace hypergraph

// Hash function for CanonicalForm
namespace std {
template<>
struct hash<hypergraph::CanonicalForm> {
    std::size_t operator()(const hypergraph::CanonicalForm& canonical) const {
        std::size_t hash_value = std::hash<std::size_t>{}(canonical.vertex_count);
        
        // Hash the edges
        std::hash<std::size_t> size_hasher;
        for (const auto& edge : canonical.edges) {
            std::size_t edge_hash = size_hasher(edge.size());
            for (std::size_t vertex : edge) {
                edge_hash ^= size_hasher(vertex) + 0x9e3779b9 + (edge_hash << 6) + (edge_hash >> 2);
            }
            hash_value ^= edge_hash + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }
        
        return hash_value;
    }
};
}

#endif // HYPERGRAPH_CANONICALIZATION_HPP