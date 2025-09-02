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

    // Create simple single-threaded evolution (1 step, 1 thread)
    std::cout << "Creating Wolfram evolution system:\n";
    std::cout << "  Max steps: 1\n";
    std::cout << "  Threads: 1\n";
    std::cout << "  Canonicalization: enabled\n\n";
    WolframEvolution evolution(1, 1, true, false);

    // Create rule: {X, Y} -> {{X, Y}, {Y, Z}} (edge growth)
    std::cout << "Creating rewriting rule:\n";
    std::cout << "  LHS: {X, Y}\n";
    std::cout << "  RHS: {{X, Y}, {Y, Z}}\n";
    std::cout << "  Effect: Adds a new edge extending from the matched edge\n\n";

    PatternHypergraph lhs, rhs;

    // LHS: single edge with two variables
    lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});

    // RHS: original edge plus new edge (Z is fresh variable)
    rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});

    RewritingRule rule(lhs, rhs);
    evolution.add_rule(rule);

    // Initial state: single edge {{1, 2}}
    std::cout << "Initial hypergraph state: {{1, 2}}\n\n";
    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}};

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