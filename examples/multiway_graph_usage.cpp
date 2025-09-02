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
        // Create a Wolfram evolution system
        std::cout << "Creating WolframEvolution (2 steps, 1 thread)...\n";
        WolframEvolution evolution(2, 1, true, false);  // 2 steps, 1 thread, canonicalization on
        
        // Create a rewriting rule: {{x, y}, {y, z}} -> {{x, y}, {y, z}, {y, w}}
        std::cout << "Setting up rewriting rule: {{x, y}, {y, z}} -> {{x, y}, {y, z}, {y, w}}\n";
        PatternHypergraph lhs, rhs;
        
        // Left-hand side: {{x, y}, {y, z}} - pattern with path of length 2
        lhs.add_edge(PatternEdge{
            PatternVertex::variable(1),  // x
            PatternVertex::variable(2)   // y
        });
        lhs.add_edge(PatternEdge{
            PatternVertex::variable(2),  // y
            PatternVertex::variable(3)   // z
        });
        
        // Right-hand side: {{x, y}, {y, z}, {y, w}} - keep original edges, add branching
        rhs.add_edge(PatternEdge{
            PatternVertex::variable(1),  // x
            PatternVertex::variable(2)   // y
        });
        rhs.add_edge(PatternEdge{
            PatternVertex::variable(2),  // y
            PatternVertex::variable(3)   // z
        });
        rhs.add_edge(PatternEdge{
            PatternVertex::variable(2),  // y  
            PatternVertex::variable(4)   // w (fresh variable - doesn't appear in LHS)
        });
        
        RewritingRule rule(lhs, rhs);
        evolution.add_rule(rule);
        
        // Set initial state: path {{1, 2}, {2, 3}}
        std::cout << "Initial state: {{1, 2}, {2, 3}}\n";
        std::vector<std::vector<GlobalVertexId>> initial_state = {{1, 2}, {2, 3}};
        
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
        std::cout << "1. Initial state {{1, 2}, {2, 3}} was created\n";
        std::cout << "2. Rule applied: {{1, 2}, {2, 3}} -> {{1, 2}, {2, 3}, {2, w}}\n";
        std::cout << "3. This creates branching from vertex 2 to a new vertex w\n";
        std::cout << "4. Rule applied again to new states, creating more branching\n";
        std::cout << "5. Multiway graph tracks all possible evolution paths\n";
        
        std::cout << "\nMultiway graph exploration complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}