/**
 * Basic Wolfram Evolution Example
 *
 * Demonstrates simple hypergraph evolution:
 * - Creating evolution rules
 * - Running single-step evolution
 * - Analyzing results (states, events, edges)
 */

#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <iostream>

using namespace hypergraph;

int main() {
    std::cout << "=== Basic Wolfram Evolution Example ===\n\n";

    // Create multi-threaded evolution (4 steps, 4 threads)
    std::cout << "Creating Wolfram evolution system:\n";
    std::cout << "  Max steps: 4\n";
    std::cout << "  Threads: 4\n";
    std::cout << "  Canonicalization: enabled\n\n";
    WolframEvolution evolution(4, 4, true, false);

    // Create rule: {{X,Y},{Y,Z}} -> {{X,Y},{Y,Z},{Y,W}} (TestCase2 rule)
    std::cout << "Creating rewriting rule (TestCase2):\n";
    std::cout << "  LHS: {{X,Y},{Y,Z}}\n";
    std::cout << "  RHS: {{X,Y},{Y,Z},{Y,W}}\n";
    std::cout << "  Effect: Adds a new edge from shared vertex\n\n";

    PatternHypergraph lhs, rhs;

    // LHS: two edges sharing a vertex
    lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    lhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});

    // RHS: original edges plus new edge from shared vertex
    rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(4)});

    RewritingRule rule(lhs, rhs);
    evolution.add_rule(rule);

    // Initial state: two edges {{1, 2}, {2, 3}}
    std::cout << "Initial hypergraph state: {{1, 2}, {2, 3}}\n\n";
    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}};

    std::cout << "Running evolution...\n";

    try {
        evolution.evolve(initial);

        const auto& graph = evolution.get_multiway_graph();

        std::cout << "\nEvolution Results:\n";
        std::cout << "  States: " << graph.num_states() << " (different hypergraph configurations)\n";
        std::cout << "  Events: " << graph.num_events() << " (rule applications)\n";
        std::cout << "  Causal edges: " << graph.get_causal_edge_count() << " (temporal dependencies)\n";
        std::cout << "  Branchial edges: " << graph.get_branchial_edge_count() << " (space-like relations)\n\n";

        if (graph.num_states() >= 2 && graph.num_events() >= 1) {
            std::cout << "Evolution successful!\n";
            std::cout << "The rule created new states from the initial configuration.\n";
            return 0;
        } else {
            std::cout << "Evolution did not generate expected results\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception during evolution: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n=== Example completed successfully ===\n";
}