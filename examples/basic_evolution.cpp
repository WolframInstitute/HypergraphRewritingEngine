/**
 * Basic Evolution Example
 *
 * Demonstrates simple hypergraph evolution using the API:
 * - Creating evolution rules with make_rule()
 * - Running evolution steps
 * - Analyzing results (states, events)
 */

#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>
#include <iostream>

using namespace hypergraph;

int main() {
    std::cout << "=== Basic Evolution Example ===\n\n";

    // Create hypergraph and evolution engine
    std::cout << "Creating hypergraph and evolution engine:\n";
    std::cout << "  Threads: 4\n";
    std::cout << "  Steps: 4\n\n";

    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);

    // Create rule: {{x,y},{y,z}} -> {{x,y},{y,z},{y,w}}
    // This rule finds two connected edges and adds a new edge from their shared vertex
    std::cout << "Creating rewriting rule:\n";
    std::cout << "  LHS: {{x,y},{y,z}} - two edges sharing vertex y\n";
    std::cout << "  RHS: {{x,y},{y,z},{y,w}} - adds new edge from y\n\n";

    // Variables: x=0, y=1, z=2, w=3 (fresh)
    auto rule = make_rule(0)
        .lhs({0, 1})      // {x, y}
        .lhs({1, 2})      // {y, z}
        .rhs({0, 1})      // {x, y}
        .rhs({1, 2})      // {y, z}
        .rhs({1, 3})      // {y, w} - new edge with fresh vertex
        .build();

    engine.add_rule(rule);

    // Initial state: {{1, 2}, {2, 3}} - matches the LHS pattern
    std::cout << "Initial hypergraph state: {{1, 2}, {2, 3}}\n\n";
    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}};

    std::cout << "Running evolution for 4 steps...\n";
    engine.evolve(initial, 4);

    std::cout << "\nEvolution Results:\n";
    std::cout << "  States: " << hg.num_states() << " (different hypergraph configurations)\n";
    std::cout << "  Events: " << hg.num_events() << " (rule applications)\n";

    auto causal_edges = hg.causal_graph().get_causal_edges();
    auto branchial_edges = hg.causal_graph().get_branchial_edges();
    std::cout << "  Causal edges: " << causal_edges.size() << " (temporal dependencies)\n";
    std::cout << "  Branchial edges: " << branchial_edges.size() << " (space-like relations)\n\n";

    if (hg.num_states() >= 2 && hg.num_events() >= 1) {
        std::cout << "Evolution successful!\n";
        std::cout << "The rule created new states from the initial configuration.\n";
    } else {
        std::cout << "Evolution did not generate expected results\n";
        return 1;
    }

    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}
