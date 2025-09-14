#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <iostream>
#include "test_helpers.hpp"

// Debug test to trace exactly what edges are being selected
class DebugEdgeSelectionTest : public ::testing::Test {
protected:
    void test_edge_selection_debug() {
        std::cout << "\n=== DEBUG: Edge Selection Tracing ===\n";

        // Use the exact problematic state: RAW: {2,3} {1,2} {2,4} {4,7}
        std::vector<std::vector<hypergraph::GlobalVertexId>> problematic_state = {
            {2, 3}, {1, 2}, {2, 4}, {4, 7}
        };

        std::cout << "Problematic state edges: ";
        for (size_t i = 0; i < problematic_state.size(); ++i) {
            std::cout << "[" << i << "] {";
            for (size_t j = 0; j < problematic_state[i].size(); ++j) {
                std::cout << problematic_state[i][j];
                if (j < problematic_state[i].size() - 1) std::cout << ",";
            }
            std::cout << "} ";
        }
        std::cout << "\n";

        // Use many workers to increase chance of showing the problem
        hypergraph::WolframEvolution evolution(1, 16, true, false, true);

        hypergraph::PatternHypergraph lhs, rhs;
        lhs.add_edge(hypergraph::PatternEdge{
            {hypergraph::PatternVertex::variable(1), hypergraph::PatternVertex::variable(2)}
        });
        rhs.add_edge(hypergraph::PatternEdge{
            {hypergraph::PatternVertex::variable(1), hypergraph::PatternVertex::variable(2)}
        });
        rhs.add_edge(hypergraph::PatternEdge{
            {hypergraph::PatternVertex::variable(2), hypergraph::PatternVertex::variable(3)}
        });

        hypergraph::RewritingRule rule(lhs, rhs);
        evolution.add_rule(rule);

        std::cout << "Running 1-step evolution with 16 workers...\n";
        evolution.evolve(problematic_state);

        const auto& graph = evolution.get_multiway_graph();
        std::cout << "Results: " << graph.num_states() << " states, " << graph.num_events() << " events\n";

        // With 1 initial state + 4 unique matches, we should get exactly 5 states
        std::cout << "Expected: 5 states (1 initial + 4 matches)\n";

        if (graph.num_states() != 5) {
            std::cout << "❌ DUPLICATE EVENTS DETECTED!\n";
            std::cout << "Expected 5 states, got " << graph.num_states() << "\n";
            FAIL() << "Pattern matching still producing duplicates";
        } else {
            std::cout << "✅ Correct state count - pattern matching working properly\n";
        }
    }
};

TEST_F(DebugEdgeSelectionTest, TraceEdgeSelection) {
    test_edge_selection_debug();
}