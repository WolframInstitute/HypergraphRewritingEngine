#include <hypergraph/canonicalization.hpp>
#include <sstream>
#include <unordered_map>
#include <map>
#include <set>
#include <limits>

namespace hypergraph {

std::string CanonicalForm::to_string() const {
    std::ostringstream oss;
    oss << "CanonicalForm{vertices=" << vertex_count << ", edges=[";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{";
        for (size_t j = 0; j < edges[i].size(); ++j) {
            if (j > 0) oss << ",";
            oss << edges[i][j];
        }
        oss << "}";
    }
    oss << "]}";
    return oss.str();
}

bool Canonicalizer::VertexSignature::operator<(const VertexSignature& other) const {
    // Primary: degree (fewer edges = smaller)
    if (degree != other.degree) {
        return degree < other.degree;
    }
    
    // Secondary: edge arities (lexicographic)
    if (edge_arities != other.edge_arities) {
        return edge_arities < other.edge_arities;
    }
    
    // Tertiary: position counts (lexicographic)
    if (position_counts != other.position_counts) {
        return position_counts < other.position_counts;
    }
    
    // Final tiebreaker: vertex ID (for stability)
    return vertex_id < other.vertex_id;
}

bool Canonicalizer::VertexSignature::operator==(const VertexSignature& other) const {
    return degree == other.degree && 
           edge_arities == other.edge_arities &&
           position_counts == other.position_counts &&
           vertex_id == other.vertex_id;
}

Canonicalizer::VertexSignature Canonicalizer::compute_vertex_signature(
    const Hypergraph& hg, VertexId vertex_id) const {
    
    VertexSignature sig;
    sig.vertex_id = vertex_id;
    
    auto edge_ids = hg.edges_containing(vertex_id);
    sig.degree = edge_ids.size();
    
    // Count positions in edges of different arities
    std::map<std::size_t, std::size_t> arity_to_count;
    std::map<std::size_t, std::map<std::size_t, std::size_t>> arity_to_position_counts;
    
    for (EdgeId edge_id : edge_ids) {
        const Hyperedge* edge = hg.get_edge(edge_id);
        if (!edge) continue;
        
        std::size_t arity = edge->arity();
        arity_to_count[arity]++;
        
        // Count positions where this vertex appears in this arity
        for (std::size_t pos = 0; pos < arity; ++pos) {
            if (edge->vertex(pos) == vertex_id) {
                arity_to_position_counts[arity][pos]++;
            }
        }
    }
    
    // Convert to sorted vectors
    for (const auto& [arity, count] : arity_to_count) {
        for (std::size_t i = 0; i < count; ++i) {
            sig.edge_arities.push_back(arity);
        }
    }
    // Use insertion sort for edge arities
    for (std::size_t i = 1; i < sig.edge_arities.size(); ++i) {
        std::size_t key = sig.edge_arities[i];
        std::size_t j = i;
        
        while (j > 0 && key < sig.edge_arities[j - 1]) {
            sig.edge_arities[j] = sig.edge_arities[j - 1];
            --j;
        }
        sig.edge_arities[j] = key;
    }
    
    // Position counts as a flattened vector: [arity1, pos0_count, pos1_count, ..., arity2, ...]
    for (const auto& [arity, pos_counts] : arity_to_position_counts) {
        sig.position_counts.push_back(arity);
        for (const auto& [pos, count] : pos_counts) {
            sig.position_counts.push_back(pos);
            sig.position_counts.push_back(count);
        }
    }
    
    return sig;
}

std::vector<Canonicalizer::VertexSignature> Canonicalizer::compute_all_signatures(
    const Hypergraph& hg) const {
    
    std::vector<VertexSignature> signatures;
    for (VertexId vertex_id : hg.vertices()) {
        signatures.push_back(compute_vertex_signature(hg, vertex_id));
    }
    
    return signatures;
}

VertexMapping Canonicalizer::compute_vertex_mapping(
    const std::vector<VertexSignature>& signatures) const {
    
    VertexMapping mapping;
    
    // Sort signatures to establish canonical ordering using insertion sort
    std::vector<VertexSignature> sorted_signatures = signatures;
    for (std::size_t i = 1; i < sorted_signatures.size(); ++i) {
        VertexSignature key = sorted_signatures[i];
        std::size_t j = i;
        
        while (j > 0 && key < sorted_signatures[j - 1]) {
            sorted_signatures[j] = sorted_signatures[j - 1];
            --j;
        }
        sorted_signatures[j] = key;
    }
    
    // Create mapping
    mapping.canonical_to_original.reserve(sorted_signatures.size());
    for (std::size_t canonical_idx = 0; canonical_idx < sorted_signatures.size(); ++canonical_idx) {
        VertexId original_vertex = sorted_signatures[canonical_idx].vertex_id;
        mapping.original_to_canonical[original_vertex] = canonical_idx;
        mapping.canonical_to_original.push_back(original_vertex);
    }
    
    return mapping;
}

std::vector<std::vector<std::size_t>> Canonicalizer::map_edges_to_canonical(
    const Hypergraph& hg, const VertexMapping& mapping) const {
    
    std::vector<std::vector<std::size_t>> canonical_edges;
    
    for (const Hyperedge& edge : hg.edges()) {
        std::vector<std::size_t> canonical_vertices;
        canonical_vertices.reserve(edge.arity());
        
        for (VertexId original_vertex : edge.vertices()) {
            std::size_t canonical_vertex = mapping.map_vertex(original_vertex);
            if (canonical_vertex != std::numeric_limits<std::size_t>::max()) {
                canonical_vertices.push_back(canonical_vertex);
            }
        }
        
        if (!canonical_vertices.empty()) {
            // Sort vertices within each edge for canonical ordering
            std::sort(canonical_vertices.begin(), canonical_vertices.end());
            canonical_edges.push_back(std::move(canonical_vertices));
        }
    }
    
    // Sort edges for canonical ordering using insertion sort
    for (std::size_t i = 1; i < canonical_edges.size(); ++i) {
        std::vector<std::size_t> key = canonical_edges[i];
        std::size_t j = i;
        
        while (j > 0 && key < canonical_edges[j - 1]) {
            canonical_edges[j] = canonical_edges[j - 1];
            --j;
        }
        canonical_edges[j] = key;
    }
    
    return canonical_edges;
}

CanonicalizationResult Canonicalizer::canonicalize(const Hypergraph& hg) const {
    CanonicalizationResult result;
    
    if (hg.num_vertices() == 0) {
        result.canonical_form.vertex_count = 0;
        return result;
    }
    
    // Step 1: Extract all edges and collect all vertices
    std::vector<std::vector<VertexId>> edges;
    std::set<VertexId> all_vertices;
    
    for (const Hyperedge& edge : hg.edges()) {
        std::vector<VertexId> edge_vertices;
        for (VertexId v : edge.vertices()) {
            edge_vertices.push_back(v);
            all_vertices.insert(v);
        }
        edges.push_back(edge_vertices);
    }
    
    // Step 2: DelDup - relabel vertices to consecutive integers starting from 0
    std::vector<VertexId> vertex_list(all_vertices.begin(), all_vertices.end());
    std::sort(vertex_list.begin(), vertex_list.end());
    
    std::unordered_map<VertexId, std::size_t> vertex_relabeling;
    for (std::size_t i = 0; i < vertex_list.size(); ++i) {
        vertex_relabeling[vertex_list[i]] = i;
        result.vertex_mapping.original_to_canonical[vertex_list[i]] = i;
        result.vertex_mapping.canonical_to_original.push_back(vertex_list[i]);
    }
    
    // Step 3: Apply relabeling to edges
    std::vector<std::vector<std::size_t>> relabeled_edges;
    for (const auto& edge : edges) {
        std::vector<std::size_t> relabeled_edge;
        for (VertexId v : edge) {
            relabeled_edge.push_back(vertex_relabeling[v]);
        }
        relabeled_edges.push_back(relabeled_edge);
    }
    
    // Step 4: CanonicalizeParts - group by edge length (arity)
    std::map<std::size_t, std::vector<std::vector<std::size_t>>> edges_by_arity;
    for (const auto& edge : relabeled_edges) {
        edges_by_arity[edge.size()].push_back(edge);
    }
    
    // Step 5: Sort each arity group, then combine
    result.canonical_form.edges.clear();
    
    // Process in descending arity order (like ReverseSort in Mathematica)
    for (auto it = edges_by_arity.rbegin(); it != edges_by_arity.rend(); ++it) {
        auto& arity_edges = it->second;
        
        // Do NOT sort vertices within each edge - preserve order
        // Mathematica treats {20,30} and {30,20} as different edges
        
        // Then sort the edges lexicographically
        std::sort(arity_edges.begin(), arity_edges.end());
        
        // Add to result
        for (const auto& edge : arity_edges) {
            result.canonical_form.edges.push_back(edge);
        }
    }
    
    result.canonical_form.vertex_count = vertex_list.size();
    
    return result;
}

CanonicalizationResult Canonicalizer::canonicalize_rewritten(
    const Hypergraph& hg, const VertexMapping& previous_mapping) const {
    
    CanonicalizationResult result;
    
    if (hg.num_vertices() == 0) {
        result.canonical_form.vertex_count = 0;
        return result;
    }
    
    // For rewritten graphs, we assume the vertex order is mostly preserved
    // Only a few edges have been added/removed, so sorting is nearly done
    
    // Step 1: Check if previous mapping is still valid
    bool mapping_valid = true;
    if (previous_mapping.canonical_to_original.size() != hg.num_vertices()) {
        mapping_valid = false;
    } else {
        // Check if all vertices in previous mapping still exist
        for (VertexId vertex : previous_mapping.canonical_to_original) {
            if (hg.vertices().find(vertex) == hg.vertices().end()) {
                mapping_valid = false;
                break;
            }
        }
    }
    
    if (!mapping_valid) {
        // Fall back to full canonicalization if mapping is invalid
        return canonicalize(hg);
    }
    
    // Step 2: Recompute signatures (some may have changed due to edge additions/deletions)
    auto signatures = compute_all_signatures(hg);
    
    // Step 3: Use insertion sort since the order is likely mostly preserved
    std::vector<VertexSignature> sorted_signatures;
    sorted_signatures.reserve(signatures.size());
    
    // Start with the previous canonical order
    for (VertexId original_vertex : previous_mapping.canonical_to_original) {
        // Find the updated signature for this vertex
        auto it = std::find_if(signatures.begin(), signatures.end(),
            [original_vertex](const VertexSignature& sig) {
                return sig.vertex_id == original_vertex;
            });
        if (it != signatures.end()) {
            sorted_signatures.push_back(*it);
        }
    }
    
    // Handle any new vertices that weren't in the previous mapping
    for (const auto& sig : signatures) {
        if (std::find_if(sorted_signatures.begin(), sorted_signatures.end(),
                [&sig](const VertexSignature& existing) {
                    return existing.vertex_id == sig.vertex_id;
                }) == sorted_signatures.end()) {
            sorted_signatures.push_back(sig);
        }
    }
    
    // Step 4: Apply insertion sort - much faster than full sort for nearly-sorted data
    for (std::size_t i = 1; i < sorted_signatures.size(); ++i) {
        VertexSignature key = sorted_signatures[i];
        std::size_t j = i;
        
        // Shift elements that are greater than key to the right
        while (j > 0 && key < sorted_signatures[j - 1]) {
            sorted_signatures[j] = sorted_signatures[j - 1];
            --j;
        }
        sorted_signatures[j] = key;
    }
    
    // Step 5: Create mapping from sorted signatures
    result.vertex_mapping.canonical_to_original.reserve(sorted_signatures.size());
    for (std::size_t canonical_idx = 0; canonical_idx < sorted_signatures.size(); ++canonical_idx) {
        VertexId original_vertex = sorted_signatures[canonical_idx].vertex_id;
        result.vertex_mapping.original_to_canonical[original_vertex] = canonical_idx;
        result.vertex_mapping.canonical_to_original.push_back(original_vertex);
    }
    
    // Step 6: Map edges to canonical form (also use insertion sort for edge ordering)
    result.canonical_form.edges = map_edges_to_canonical_incremental(hg, result.vertex_mapping);
    result.canonical_form.vertex_count = hg.num_vertices();
    
    return result;
}

std::vector<std::vector<std::size_t>> Canonicalizer::map_edges_to_canonical_incremental(
    const Hypergraph& hg, const VertexMapping& mapping) const {
    
    std::vector<std::vector<std::size_t>> canonical_edges;
    
    for (const Hyperedge& edge : hg.edges()) {
        std::vector<std::size_t> canonical_vertices;
        canonical_vertices.reserve(edge.arity());
        
        for (VertexId original_vertex : edge.vertices()) {
            std::size_t canonical_vertex = mapping.map_vertex(original_vertex);
            if (canonical_vertex != std::numeric_limits<std::size_t>::max()) {
                canonical_vertices.push_back(canonical_vertex);
            }
        }
        
        if (!canonical_vertices.empty()) {
            // Sort vertices within each edge for canonical ordering
            std::sort(canonical_vertices.begin(), canonical_vertices.end());
            canonical_edges.push_back(std::move(canonical_vertices));
        }
    }
    
    // Use insertion sort for edge ordering (faster for nearly-sorted edge lists)
    for (std::size_t i = 1; i < canonical_edges.size(); ++i) {
        std::vector<std::size_t> key = canonical_edges[i];
        std::size_t j = i;
        
        while (j > 0 && key < canonical_edges[j - 1]) {
            canonical_edges[j] = canonical_edges[j - 1];
            --j;
        }
        canonical_edges[j] = key;
    }
    
    return canonical_edges;
}

} // namespace hypergraph