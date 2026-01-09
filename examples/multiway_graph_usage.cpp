/**
 * Multiway Graph Usage Example
 *
 * Demonstrates multiway system evolution using the API:
 * - Multiple rules creating branching evolution
 * - Exploring the causal graph structure
 * - Analyzing states and events
 */

#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>
#include <iostream>
#include <thread>

using namespace hypergraph;

int main() {
    std::cout << "Multiway Graph Usage Example\n";
    std::cout << "================================\n\n";

    const size_t nSteps = 3;
    const size_t nThreads = std::thread::hardware_concurrency();

    std::cout << "Creating evolution system (" << nSteps << " steps, "
              << nThreads << " threads)...\n\n";

    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, nThreads);

    // Rule 1: {{x,y,z}} -> {{x,y},{x,z},{x,w}}
    // Expands a ternary edge into three binary edges
    std::cout << "Rule 1: {{x,y,z}} -> {{x,y},{x,z},{x,w}}\n";
    std::cout << "  Expands ternary edge into binary edges with fresh vertex\n";

    auto rule1 = make_rule(0)
        .lhs({0, 1, 2})       // {x, y, z}
        .rhs({0, 1})          // {x, y}
        .rhs({0, 2})          // {x, z}
        .rhs({0, 3})          // {x, w} - fresh vertex w=3
        .build();

    engine.add_rule(rule1);

    // Rule 2: {{x,y}} -> {{x,y},{x,z}}
    // Adds a new edge from first vertex
    std::cout << "Rule 2: {{x,y}} -> {{x,y},{x,z}}\n";
    std::cout << "  Preserves edge and adds new edge with fresh vertex\n\n";

    auto rule2 = make_rule(1)
        .lhs({0, 1})          // {x, y}
        .rhs({0, 1})          // {x, y}
        .rhs({0, 2})          // {x, z} - fresh vertex z=2
        .build();

    engine.add_rule(rule2);

    // Initial state: {{1,2,3}} - a single ternary edge
    std::cout << "Initial state: {{1,2,3}} (single ternary edge)\n";
    std::cout << "Running evolution for " << nSteps << " steps...\n\n";

    std::vector<std::vector<VertexId>> initial_state = {{1, 2, 3}};
    engine.evolve(initial_state, nSteps);

    std::cout << "Evolution Results:\n";
    std::cout << "------------------\n";
    std::cout << "Total states: " << hg.num_states() << "\n";
    std::cout << "Total events: " << hg.num_events() << "\n";

    // Get causal structure
    auto causal_edges = hg.causal_graph().get_causal_edges();
    auto branchial_edges = hg.causal_graph().get_branchial_edges();

    std::cout << "Causal edges: " << causal_edges.size() << " (time-like connections)\n";
    std::cout << "Branchial edges: " << branchial_edges.size() << " (space-like connections)\n\n";

    // Print step-by-step evolution
    std::cout << "Evolution by step:\n";
    for (size_t step = 0; step <= nSteps; ++step) {
        size_t states_at_step = 0;
        for (StateId sid = 0; sid < hg.num_states(); ++sid) {
            const auto& state = hg.get_state(sid);
            if (state.id != INVALID_ID && state.step == step) {
                states_at_step++;
            }
        }
        std::cout << "  Step " << step << ": " << states_at_step << " state(s)\n";
    }

    std::cout << "\n";

    // Print sample event information
    if (hg.num_events() > 0) {
        std::cout << "Sample events:\n";
        size_t shown = 0;
        for (EventId eid = 0; eid < hg.num_events() && shown < 5; ++eid) {
            const auto& event = hg.get_event(eid);
            if (event.id != INVALID_ID) {
                std::cout << "  Event " << eid << ": "
                          << "Rule " << (int)event.rule_index << ", "
                          << "Input state " << event.input_state << " -> "
                          << "Output state " << event.output_state << "\n";
                shown++;
            }
        }
        if (hg.num_events() > 5) {
            std::cout << "  ... (" << (hg.num_events() - 5) << " more events)\n";
        }
    }

    std::cout << "\nMultiway graph exploration complete!\n";
    std::cout << "\nThis example demonstrates how multiple rules can be applied\n";
    std::cout << "to create a branching multiway evolution system.\n";

    return 0;
}
