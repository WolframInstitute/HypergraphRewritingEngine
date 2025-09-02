#include <hypergraph/pattern_matching.hpp>
#include <algorithm>
#include <queue>

namespace hypergraph {

bool PatternMatcher::edge_matches(const Hyperedge& concrete_edge, 
                                 const PatternEdge& pattern_edge,
                                 VariableAssignment& assignment) const {
    if (concrete_edge.arity() != pattern_edge.arity()) {
        return false;
    }
    
    // Try to match each position
    VariableAssignment temp_assignment = assignment;
    
    for (std::size_t i = 0; i < pattern_edge.arity(); ++i) {
        const PatternVertex& pattern_vertex = pattern_edge.vertices[i];
        VertexId concrete_vertex = concrete_edge.vertex(i);
        
        if (pattern_vertex.is_concrete()) {
            if (pattern_vertex.id != concrete_vertex) {
                return false;  // Concrete mismatch
            }
        } else {
            // Variable vertex - check assignment consistency
            if (!temp_assignment.assign(pattern_vertex.id, concrete_vertex)) {
                return false;  // Inconsistent assignment
            }
        }
    }
    
    // If we get here, the match is valid
    assignment = temp_assignment;
    return true;
}

std::vector<VariableAssignment> PatternMatcher::generate_assignments(
    const Hyperedge& concrete_edge,
    const PatternEdge& pattern_edge,
    const VariableAssignment& base_assignment) const {
    
    std::vector<VariableAssignment> assignments;
    
    if (concrete_edge.arity() != pattern_edge.arity()) {
        return assignments;  // Empty - no match possible
    }
    
    // For simplicity, we'll use a direct matching approach
    // In a full implementation, you might want permutation-based matching
    VariableAssignment assignment = base_assignment;
    if (edge_matches(concrete_edge, pattern_edge, assignment)) {
        assignments.push_back(assignment);
    }
    
    return assignments;
}

bool PatternMatcher::match_remaining_edges(
    const Hypergraph& target,
    const std::vector<PatternEdge>& remaining_pattern_edges,
    const std::unordered_set<EdgeId>& available_edges,
    std::vector<EdgeId>& matched_edges,
    VariableAssignment& assignment) const {
    
    if (remaining_pattern_edges.empty()) {
        return true;  // All edges matched
    }
    
    const PatternEdge& current_pattern = remaining_pattern_edges[0];
    std::vector<PatternEdge> remaining(remaining_pattern_edges.begin() + 1, 
                                      remaining_pattern_edges.end());
    
    // Try to match current pattern edge with each available concrete edge
    for (EdgeId edge_id : available_edges) {
        const Hyperedge* concrete_edge = target.get_edge(edge_id);
        if (!concrete_edge) continue;
        
        VariableAssignment temp_assignment = assignment;
        if (edge_matches(*concrete_edge, current_pattern, temp_assignment)) {
            // This edge matches - try to match remaining edges
            matched_edges.push_back(edge_id);
            
            std::unordered_set<EdgeId> remaining_available = available_edges;
            remaining_available.erase(edge_id);
            
            if (match_remaining_edges(target, remaining, remaining_available, 
                                    matched_edges, temp_assignment)) {
                assignment = temp_assignment;
                return true;  // Found complete match
            }
            
            // Backtrack
            matched_edges.pop_back();
        }
    }
    
    return false;  // No match found
}

std::vector<PatternMatch> PatternMatcher::find_matches_from_anchor(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    VertexId anchor_vertex,
    std::size_t search_radius) const {
    
    std::vector<PatternMatch> matches;
    
    if (!target.has_vertex(anchor_vertex)) {
        return matches;  // Anchor doesn't exist
    }
    
    // Get edges within radius of anchor
    auto nearby_edges = target.edges_within_radius(anchor_vertex, search_radius);
    
    if (nearby_edges.size() < pattern.num_edges()) {
        return matches;  // Not enough edges to match pattern
    }
    
    // Try to match pattern edges
    const auto& pattern_edges = pattern.edges();
    if (pattern_edges.empty()) {
        return matches;  // Empty pattern
    }
    
    // For each possible assignment of the first pattern edge
    for (EdgeId edge_id : nearby_edges) {
        const Hyperedge* concrete_edge = target.get_edge(edge_id);
        if (!concrete_edge) continue;
        
        // Try to match first pattern edge
        auto assignments = generate_assignments(*concrete_edge, pattern_edges[0], 
                                              VariableAssignment{});
        
        for (const auto& initial_assignment : assignments) {
            std::vector<EdgeId> matched_edges = {edge_id};
            VariableAssignment assignment = initial_assignment;
            
            // Try to match remaining pattern edges
            std::vector<PatternEdge> remaining_patterns(pattern_edges.begin() + 1, 
                                                       pattern_edges.end());
            std::unordered_set<EdgeId> available_edges = nearby_edges;
            available_edges.erase(edge_id);
            
            if (match_remaining_edges(target, remaining_patterns, available_edges,
                                    matched_edges, assignment)) {
                // Found a complete match
                PatternMatch match;
                match.matched_edges = matched_edges;
                match.assignment = assignment;
                match.anchor_vertex = anchor_vertex;
                
                // Build edge_map: pattern edge index -> target edge ID
                for (std::size_t i = 0; i < matched_edges.size(); ++i) {
                    match.edge_map[i] = matched_edges[i];
                }
                
                matches.push_back(match);
            }
        }
    }
    
    return matches;
}

std::vector<PatternMatch> PatternMatcher::find_matches_around(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    VertexId anchor_vertex,
    std::size_t search_radius) const {
    
    return find_matches_from_anchor(target, pattern, anchor_vertex, search_radius);
}

std::vector<PatternMatch> PatternMatcher::find_all_matches(
    const Hypergraph& target,
    const PatternHypergraph& pattern) const {
    
    // Pure exhaustive pattern matching on the entire hypergraph
    std::vector<PatternMatch> all_matches;
    
    // Try every possible starting point for pattern matching
    for (VertexId vertex : target.vertices()) {
        auto matches = find_matches_from_anchor(target, pattern, vertex, target.num_edges());
        all_matches.insert(all_matches.end(), matches.begin(), matches.end());
    }
    
    // Remove duplicates (same set of matched edges)
    std::sort(all_matches.begin(), all_matches.end(),
        [](const PatternMatch& a, const PatternMatch& b) {
            return a.matched_edges < b.matched_edges;
        });
    
    all_matches.erase(
        std::unique(all_matches.begin(), all_matches.end(),
            [](const PatternMatch& a, const PatternMatch& b) {
                return a.matched_edges == b.matched_edges;
            }),
        all_matches.end());
    
    return all_matches;
}

bool PatternMatcher::matches_at(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    const VariableAssignment& assignment) const {
    
    // Check if assignment is complete
    if (!assignment.is_complete(pattern.variable_vertices())) {
        return false;
    }
    
    // For each pattern edge, check if corresponding concrete edge exists
    for (const auto& pattern_edge : pattern.edges()) {
        std::vector<VertexId> concrete_vertices;
        
        for (const auto& pattern_vertex : pattern_edge.vertices) {
            auto resolved = assignment.resolve(pattern_vertex);
            if (!resolved) {
                return false;  // Unresolved variable
            }
            concrete_vertices.push_back(*resolved);
        }
        
        // Check if this concrete edge exists in target
        bool found = false;
        for (const auto& target_edge : target.edges()) {
            if (target_edge.vertices() == concrete_vertices) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            return false;  // Pattern edge not found in target
        }
    }
    
    return true;
}

std::optional<PatternMatch> PatternMatcher::find_first_match_around(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    VertexId anchor_vertex,
    std::size_t search_radius) const {
    
    auto matches = find_matches_around(target, pattern, anchor_vertex, search_radius);
    return matches.empty() ? std::nullopt : std::optional<PatternMatch>(matches[0]);
}

} // namespace hypergraph