#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <atomic>
#include <iostream>
#include "test_helpers.hpp"

// Focused test to reproduce the duplicate match issue using exact state from logs
class IsolatedDuplicateBugTest : public ::testing::Test {
protected:
    // Test exact state 47542529087130026 from debug logs: RAW: {1,2} {2,4} {2,3} {3,9}
    void test_duplicate_matches_in_state() {
        std::cout << "\n=== Testing Duplicate Matches in State 47542529087130026 ===\n";

        // Exact edges from debug logs for state 47542529087130026
        std::vector<std::vector<hypergraph::GlobalVertexId>> problematic_state = {
            {1, 2}, {2, 4}, {2, 3}, {3, 9}
        };

        std::cout << "Using exact edges from state 47542529087130026: ";
        for (const auto& edge : problematic_state) {
            std::cout << "{";
            for (size_t i = 0; i < edge.size(); ++i) {
                std::cout << edge[i];
                if (i < edge.size() - 1) std::cout << ",";
            }
            std::cout << "} ";
        }
        std::cout << "\n";

        // Use single-edge LHS rule: {{x,y}} -> {{x,y}, {y,z}}
        // This should produce exactly 4 events (one per edge) with NO duplicates
        hypergraph::WolframEvolution evolution(1, 4, true, false, true);

        // Create the same rule from TestCase1
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

        std::cout << "Running 1-step evolution from this exact state...\n";
        evolution.evolve(problematic_state);

        const auto& graph = evolution.get_multiway_graph();
        std::cout << "Results: " << graph.num_states() << " states, " << graph.num_events() << " events\n";

        // For a 4-edge state with single-edge LHS rule, we should get exactly 5 states:
        // 1 initial state + 4 new states (one per edge matched)
        std::cout << "Expected: 5 states (1 initial + 4 from matches)\n";
        std::cout << "Actual: " << graph.num_states() << " states\n";

        if (graph.num_states() != 5) {
            std::cout << "❌ DUPLICATE MATCHES DETECTED!\n";
            std::cout << "Expected 5 total states, got " << graph.num_states() << "\n";
            FAIL() << "Duplicate matches still present - fix incomplete";
        } else {
            std::cout << "✅ No duplicate matches - pattern matching working correctly\n";
        }
    }
};

TEST_F(IsolatedDuplicateBugTest, ExactStateReproduction) {
    test_duplicate_matches_in_state();
}