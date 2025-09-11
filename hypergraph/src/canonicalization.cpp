#include <hypergraph/canonicalization.hpp>
#include <sstream>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>
#include <unordered_set>

namespace hypergraph {

std::string CanonicalForm::to_string() const {
    std::ostringstream oss;
    oss << "CanonicalForm(vertices=" << vertex_count << ", edges=[";
    for (std::size_t i = 0; i < edges.size(); ++i) {
        oss << "[";
        for (std::size_t j = 0; j < edges[i].size(); ++j) {
            oss << edges[i][j];
            if (j < edges[i].size() - 1) oss << ",";
        }
        oss << "]";
        if (i < edges.size() - 1) oss << ", ";
    }
    oss << "])";
    return oss.str();
}

template<typename VertexType>
std::vector<std::vector<VertexType>> Canonicalizer::wolfram_canonical_hypergraph(
    const std::vector<std::vector<VertexType>>& edges,
    VertexMapping& mapping) const {
    
    // Get all unique vertices in order of first appearance
    std::vector<VertexType> vertices;
    std::set<VertexType> seen;
    for (const auto& edge : edges) {
        for (auto v : edge) {
            if (seen.insert(v).second) {
                vertices.push_back(v);
            }
        }
    }
    
    // Try all permutations of vertices and find the lexicographically smallest result
    std::vector<std::vector<VertexType>> best_result;
    std::vector<VertexType> best_mapping;
    bool first = true;
    
    // Create indices for permutation
    std::vector<std::size_t> indices(vertices.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    do {
        // Create mapping for this permutation
        std::map<VertexType, VertexType> vertex_map;
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            vertex_map[vertices[indices[i]]] = static_cast<VertexType>(i);
        }
        
        // Apply mapping to edges
        std::vector<std::vector<VertexType>> permuted_edges;
        for (const auto& edge : edges) {
            std::vector<VertexType> new_edge;
            for (auto v : edge) {
                new_edge.push_back(vertex_map[v]);
            }
            permuted_edges.push_back(new_edge);
        }
        
        // Sort the edges for this permutation
        std::sort(permuted_edges.begin(), permuted_edges.end());
        
        // Keep the lexicographically smallest result
        if (first || permuted_edges < best_result) {
            best_result = permuted_edges;
            
            // Store the vertex mapping for this best result
            best_mapping.clear();
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                best_mapping.push_back(vertices[indices[i]]);
            }
            first = false;
        }
    } while (std::next_permutation(indices.begin(), indices.end()));
    
    // Build the final mapping
    mapping.canonical_to_original = best_mapping;
    mapping.original_to_canonical.clear();
    for (std::size_t i = 0; i < best_mapping.size(); ++i) {
        mapping.original_to_canonical[best_mapping[i]] = static_cast<VertexType>(i);
    }
    
    return best_result;
}

std::vector<std::vector<std::size_t>> Canonicalizer::edges_to_size_t(
    const std::vector<std::vector<VertexId>>& edges) const {
    
    std::vector<std::vector<std::size_t>> result;
    for (const auto& edge : edges) {
        std::vector<std::size_t> new_edge;
        for (auto v : edge) {
            new_edge.push_back(static_cast<std::size_t>(v));
        }
        result.push_back(new_edge);
    }
    return result;
}

CanonicalizationResult Canonicalizer::canonicalize(const Hypergraph& hg) const {
    CanonicalizationResult result;
    
    if (hg.num_vertices() == 0) {
        result.canonical_form.vertex_count = 0;
        return result;
    }
    
    // Extract edges as vectors
    std::vector<std::vector<VertexId>> edges;
    for (const Hyperedge& edge : hg.edges()) {
        std::vector<VertexId> edge_vertices;
        for (VertexId v : edge.vertices()) {
            edge_vertices.push_back(v);
        }
        edges.push_back(edge_vertices);
    }
    
    // Apply Wolfram canonicalization
    result.canonical_form.edges = wolfram_canonical_hypergraph(edges, result.vertex_mapping);
    result.canonical_form.vertex_count = result.vertex_mapping.canonical_to_original.size();
    
    return result;
}

template<typename VertexType>
CanonicalizationResult Canonicalizer::canonicalize_edges(const std::vector<std::vector<VertexType>>& edges) const {
    CanonicalizationResult result;
    
    if (edges.empty()) {
        result.canonical_form.vertex_count = 0;
        return result;
    }
    
    // Apply Wolfram canonicalization directly
    result.canonical_form.edges = wolfram_canonical_hypergraph(edges, result.vertex_mapping);
    result.canonical_form.vertex_count = result.vertex_mapping.canonical_to_original.size();
    
    return result;
}

// Explicit template instantiations for the types we use
// Note: VertexId and GlobalVertexId are both std::size_t, so only instantiate once
template CanonicalizationResult Canonicalizer::canonicalize_edges<VertexId>(const std::vector<std::vector<VertexId>>& edges) const;

template std::vector<std::vector<VertexId>> Canonicalizer::wolfram_canonical_hypergraph<VertexId>(
    const std::vector<std::vector<VertexId>>& edges, VertexMapping& mapping) const;

} // namespace hypergraph