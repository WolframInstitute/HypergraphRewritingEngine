#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>
#include <iostream>

using namespace hypergraph;

int main() {
    std::cout << "Multiway Graph Usage Example\n";
    std::cout << "================================\n\n";

    try {
        const auto nSteps = 4;
        const auto nThreads = std::thread::hardware_concurrency();

        std::cout << "Creating WolframEvolution (" << nSteps << " steps, " << nThreads << " threads)...\n";
        WolframEvolution evolution(nSteps, std::thread::hardware_concurrency(), true, false);

        {
            // Create rule: {{1,2,3}} -> {{1,2},{1,3},{1,4}}
            std::cout << "Setting up rewriting rule: {{1,2,3}} -> {{1,2},{1,3},{1,4}}\n";
            PatternHypergraph lhs, rhs;

            // Left-hand side: {{1,2,3}}
            lhs.add_edge(PatternEdge{
                PatternVertex::variable(1),
                PatternVertex::variable(2),
                PatternVertex::variable(3)
            });

            // Right-hand side: {{1,2},{1,3},{1,4}}
            rhs.add_edge(PatternEdge{
                PatternVertex::variable(1),
                PatternVertex::variable(2)
            });
            rhs.add_edge(PatternEdge{
                PatternVertex::variable(1),
                PatternVertex::variable(3)
            });
            rhs.add_edge(PatternEdge{
                PatternVertex::variable(1),
                PatternVertex::variable(4),
            });

            RewritingRule rule(lhs, rhs);
            evolution.add_rule(rule);
        }

        {
            // Create rule: {{1,2}} -> {{1,2},{1,3}}
            std::cout << "Setting up rewriting rule: {{1,2}} -> {{1,2},{1,3}}\n";
            PatternHypergraph lhs, rhs;

            lhs.add_edge(PatternEdge{
                PatternVertex::variable(1),
                PatternVertex::variable(2)
            });

            rhs.add_edge(PatternEdge{
                PatternVertex::variable(1),
                PatternVertex::variable(2)
            });
            rhs.add_edge(PatternEdge{
                PatternVertex::variable(1),
                PatternVertex::variable(3)
            });

            RewritingRule rule(lhs, rhs);
            evolution.add_rule(rule);
        }

        // Set initial state: {{1,2,3}}
        std::cout << "Initial state: {{1,2,3}}\n";
        std::vector<std::vector<GlobalVertexId>> initial_state = {{1,2,3}};

        // Run evolution
        std::cout << "Running evolution for " << nSteps << " steps...\n\n";
        evolution.evolve(initial_state);
        std::cout << "Evolution.evolve() completed\n";

        // Get results from multiway graph
        const MultiwayGraph& graph = evolution.get_multiway_graph();
        std::cout << "Got multiway graph reference\n";

        std::cout << "Evolution Results:\n";
        std::cout << "------------------\n";
        std::cout << "Total states: " << graph.num_states() << "\n";
        std::cout << "Total events: " << graph.num_events() << "\n";
        std::cout << "Causal edges: " << graph.get_causal_edge_count() << "\n";
        std::cout << "Branchial edges: " << graph.get_branchial_edge_count() << "\n";

        // Print detailed states and events
        std::cout << "\n";
        graph.print_summary();

        std::cout << "\nMultiway graph exploration complete!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}