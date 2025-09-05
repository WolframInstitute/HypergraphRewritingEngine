#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>
#include <iostream>

using namespace hypergraph;

/**
 * Multiway Graph Usage Example
 * 
 * This example demonstrates how to use the WolframEvolution class
 * to perform hypergraph rewriting and explore multiway spaces.
 * 
 * Key concepts:
 * - Creating evolution with rules
 * - Running evolution steps
 * - Accessing multiway graph results
 */
int main() {
    std::cout << "Multiway Graph Usage Example\n";
    std::cout << "================================\n\n";
    
    try {
        // Create a Wolfram evolution system - TEST: 2 steps to see patch-based matching
        std::cout << "Creating WolframEvolution (2 steps, 1 thread)...\n";
        WolframEvolution evolution(2, 1, true, false);  // 2 steps, 1 thread, canonicalization on
        
        // Create a SIMPLE rewriting rule: {{x, y}} -> {{x, y}, {y, z}}
        std::cout << "Setting up SIMPLE rewriting rule: {{x, y}} -> {{x, y}, {y, z}}\n";
        PatternHypergraph lhs, rhs;
        
        // Left-hand side: {{x, y}} - simple single-edge pattern
        lhs.add_edge(PatternEdge{
            PatternVertex::variable(1),  // x
            PatternVertex::variable(2)   // y
        });
        
        // Right-hand side: {{x, y}, {y, z}} - keep original edge, add one new edge
        rhs.add_edge(PatternEdge{
            PatternVertex::variable(1),  // x
            PatternVertex::variable(2)   // y
        });
        rhs.add_edge(PatternEdge{
            PatternVertex::variable(2),  // y  
            PatternVertex::variable(3)   // z (fresh variable - doesn't appear in LHS)
        });
        
        RewritingRule rule(lhs, rhs);
        evolution.add_rule(rule);
        
        // Set initial state: single edge {{1, 2}}
        std::cout << "Initial state: {{1, 2}}\n";
        std::vector<std::vector<GlobalVertexId>> initial_state = {{1, 2}};
        
        // Run evolution
        std::cout << "Running evolution for 2 steps...\n\n";
        evolution.evolve(initial_state);
        
        // Get results from multiway graph
        const MultiwayGraph& graph = evolution.get_multiway_graph();
        
        std::cout << "Evolution Results:\n";
        std::cout << "------------------\n";
        std::cout << "Total states: " << graph.num_states() << "\n";
        std::cout << "Total events: " << graph.num_events() << "\n";
        std::cout << "Causal edges: " << graph.get_causal_edge_count() << "\n";
        std::cout << "Branchial edges: " << graph.get_branchial_edge_count() << "\n";
        
        // Explain what happened
        std::cout << "\nWhat happened:\n";
        std::cout << "1. Initial state {{1, 2}} was created\n";
        std::cout << "2. Rule applied: {{1, 2}} -> {{1, 2}, {2, z}}\n";
        std::cout << "3. This creates one new edge from vertex 2 to a new vertex z\n";
        std::cout << "4. Evolution stopped after 2 steps as requested\n";
        std::cout << "5. Multiway graph tracks the evolution path\n";
        
        std::cout << "\nMultiway graph exploration complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}