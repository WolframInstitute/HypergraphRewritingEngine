/**
 * Pattern Matching Example
 * 
 * Demonstrates hypergraph pattern matching:
 * - Creating patterns with variables and concrete vertices
 * - Finding matches in target hypergraphs
 * - Variable assignments and edge mappings
 */

#include <hypergraph/pattern_matching.hpp>
#include <hypergraph/hypergraph.hpp>
#include <iostream>

using namespace hypergraph;

int main() {
    std::cout << "=== Pattern Matching Example ===\n\n";
    
    // Create target hypergraph: triangle + extra edges
    Hypergraph target;
    target.add_edge({1, 2});
    target.add_edge({2, 3});
    target.add_edge({3, 1});
    target.add_edge({4, 5});
    
    std::cout << "Target hypergraph has " << target.num_edges() << " edges:\n";
    for (std::size_t i = 0; i < target.num_edges(); ++i) {
        const auto* edge = target.get_edge(i);
        if (edge) {
            std::cout << "  Edge " << i << ": {";
            for (std::size_t j = 0; j < edge->arity(); ++j) {
                std::cout << edge->vertex(j);
                if (j < edge->arity() - 1) std::cout << ", ";
            }
            std::cout << "}\n";
        }
    }
    std::cout << "\n";
    
    // Create pattern: two connected edges {X, Y}, {Y, Z}
    std::cout << "Creating pattern: {X, Y}, {Y, Z} where X, Y, Z are variables\n";
    PatternHypergraph pattern;
    
    pattern.add_edge(PatternEdge{
        PatternVertex::variable(100),  // X
        PatternVertex::variable(101)   // Y
    });
    
    pattern.add_edge(PatternEdge{
        PatternVertex::variable(101),  // Y (shared variable)
        PatternVertex::variable(102)   // Z
    });
    
    std::cout << "Pattern has " << pattern.num_edges() << " edges, "
              << pattern.num_variable_vertices() << " variables\n\n";
    
    // Find matches
    PatternMatcher matcher;
    
    std::cout << "Finding matches around vertex 2 with radius 2:\n";
    auto matches = matcher.find_matches_around(target, pattern, 2, 2);
    
    std::cout << "Found " << matches.size() << " matches:\n";
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];
        std::cout << "\nMatch " << i << ":\n";
        
        std::cout << "  Matched edges: ";
        for (EdgeId edge_id : match.matched_edges) {
            std::cout << edge_id << " ";
        }
        std::cout << "\n";
        
        std::cout << "  Variable assignments:\n";
        for (const auto& [var, vertex] : match.assignment.variable_to_concrete) {
            std::cout << "    Variable " << var << " -> Vertex " << vertex << "\n";
        }
        
        std::cout << "  Edge mapping:\n";
        for (const auto& [pattern_idx, target_edge] : match.edge_map) {
            std::cout << "    Pattern edge " << pattern_idx << " -> Target edge " << target_edge << "\n";
        }
    }
    
    // Try pattern with concrete vertices
    std::cout << "\n" << std::string(50, '=') << "\n\n";
    std::cout << "Creating mixed pattern: {1, X} where 1 is concrete, X is variable\n";
    
    PatternHypergraph concrete_pattern;
    concrete_pattern.add_edge(PatternEdge{
        PatternVertex::concrete(1),    // Concrete vertex 1
        PatternVertex::variable(200)   // Variable X
    });
    
    auto concrete_matches = matcher.find_matches_around(target, concrete_pattern, 1, 2);
    std::cout << "Found " << concrete_matches.size() << " matches:\n";
    
    for (std::size_t i = 0; i < concrete_matches.size(); ++i) {
        const auto& match = concrete_matches[i];
        std::cout << "\nMatch " << i << ":\n";
        std::cout << "  Variable 200 -> Vertex " << match.assignment.variable_to_concrete.at(200) << "\n";
        std::cout << "  Matched edge: " << match.matched_edges[0] << "\n";
    }
    
    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}