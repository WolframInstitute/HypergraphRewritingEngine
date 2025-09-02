/**
 * Rewriting Engine Example
 * 
 * Demonstrates hypergraph rewriting operations:
 * - Creating rewriting rules
 * - Applying rules to hypergraphs
 * - Observing structural changes
 */

#include <hypergraph/rewriting.hpp>
#include <hypergraph/hypergraph.hpp>
#include <iostream>

using namespace hypergraph;

int main() {
    std::cout << "=== Rewriting Engine Example ===\n\n";
    
    // Create initial hypergraph
    Hypergraph graph;
    graph.add_edge({1, 2});
    graph.add_edge({2, 3});
    
    std::cout << "Initial graph:\n";
    std::cout << "  Vertices: " << graph.num_vertices() << ", Edges: " << graph.num_edges() << "\n";
    for (std::size_t i = 0; i < graph.num_edges(); ++i) {
        const auto* edge = graph.get_edge(i);
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
    
    // Create rule: {X, Y} -> {{X, Z}, {Z, Y}} (edge split with fresh vertex Z)
    std::cout << "Creating rule: {X, Y} -> {{X, Z}, {Z, Y}}\n";
    std::cout << "This splits an edge by inserting a fresh vertex Z\n\n";
    
    PatternHypergraph lhs;
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1),  // X
        PatternVertex::variable(2)   // Y
    });
    
    PatternHypergraph rhs;
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1),  // X
        PatternVertex::variable(3)   // Z (fresh variable)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(3),  // Z
        PatternVertex::variable(2)   // Y
    });
    
    RewritingRule rule(lhs, rhs);
    std::cout << "Rule is well-formed: " << (rule.is_well_formed() ? "Yes" : "No") << "\n\n";
    
    // Create rewriting engine
    RewritingEngine engine;
    
    // Apply rule to first edge
    std::cout << "Applying rule to edge containing vertex 1:\n";
    RewritingResult result = engine.apply_rule_at(graph, rule, 1, 2);
    
    if (result.was_applied()) {
        std::cout << "Rule applied successfully!\n";
        std::cout << "  Removed " << result.removed_edges.size() << " edges\n";
        std::cout << "  Added " << result.added_edges.size() << " edges\n";
        
        std::cout << "  Variable assignments:\n";
        for (const auto& [var, vertex] : result.variable_assignment.variable_to_concrete) {
            std::cout << "    Variable " << var << " -> Vertex " << vertex << "\n";
        }
    } else {
        std::cout << "Rule could not be applied\n";
    }
    
    std::cout << "\nGraph after rewriting:\n";
    std::cout << "  Vertices: " << graph.num_vertices() << ", Edges: " << graph.num_edges() << "\n";
    for (std::size_t i = 0; i < graph.num_edges(); ++i) {
        const auto* edge = graph.get_edge(i);
        if (edge) {
            std::cout << "  Edge " << i << ": {";
            for (std::size_t j = 0; j < edge->arity(); ++j) {
                std::cout << edge->vertex(j);
                if (j < edge->arity() - 1) std::cout << ", ";
            }
            std::cout << "}\n";
        }
    }
    
    // Apply rule again
    std::cout << "\n" << std::string(50, '=') << "\n\n";
    std::cout << "Applying rule again to see continued evolution:\n";
    
    RewritingResult result2 = engine.apply_rule_at(graph, rule, 2, 2);
    if (result2.was_applied()) {
        std::cout << "Rule applied again!\n";
        std::cout << "  Removed " << result2.removed_edges.size() << " edges\n";
        std::cout << "  Added " << result2.added_edges.size() << " edges\n";
        
        std::cout << "\nFinal graph:\n";
        std::cout << "  Vertices: " << graph.num_vertices() << ", Edges: " << graph.num_edges() << "\n";
        for (std::size_t i = 0; i < graph.num_edges(); ++i) {
            const auto* edge = graph.get_edge(i);
            if (edge) {
                std::cout << "  Edge " << i << ": {";
                for (std::size_t j = 0; j < edge->arity(); ++j) {
                    std::cout << edge->vertex(j);
                    if (j < edge->arity() - 1) std::cout << ", ";
                }
                std::cout << "}\n";
            }
        }
    } else {
        std::cout << "Rule could not be applied again\n";
    }
    
    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}