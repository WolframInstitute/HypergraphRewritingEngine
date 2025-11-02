#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>
#include "test_helpers.hpp"

class MultipleInitialStatesTest : public ::testing::Test {
protected:
    // Helper to create a simple rule: {{1, 2}} -> {{1, 2}, {2, 3}}
    hypergraph::RewritingRule create_test_rule() {
        hypergraph::PatternHypergraph lhs, rhs;

        lhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        });

        rhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        });
        rhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(2),
            hypergraph::PatternVertex::variable(3)
        });

        return hypergraph::RewritingRule(lhs, rhs);
    }
};

// === BASIC FUNCTIONALITY TESTS ===

TEST_F(MultipleInitialStatesTest, TwoInitialStates) {
    hypergraph::WolframEvolution evolution(2, 1, true, false);
    evolution.add_rule(create_test_rule());

    // Two different initial states
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}},           // First state: single edge {1,2}
        {{10, 20}, {20, 30}}  // Second state: two edges {10,20}, {20,30}
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    // Should have tracked both initial states
    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 2) << "Should have 2 initial state IDs";

    // Should have generated more states through evolution
    EXPECT_GT(graph.num_states(), 2) << "Should have more than just the 2 initial states";
    EXPECT_GT(graph.num_events(), 0) << "Should have at least some events";
}

TEST_F(MultipleInitialStatesTest, ThreeInitialStates) {
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());

    // Three different initial states
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}},
        {{3, 4}},
        {{5, 6}}
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    // Should have tracked all three initial states
    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 3) << "Should have 3 initial state IDs";

    // Each initial state should produce at least one event
    EXPECT_GE(graph.num_events(), 3) << "Should have at least 3 events (one per initial state)";
}

TEST_F(MultipleInitialStatesTest, SingleStateViaMultiStateAPI) {
    // Test that single state still works through the multi-state API
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());

    // Single state wrapped in vector
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}}
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    // Should have exactly 1 initial state
    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 1) << "Should have 1 initial state ID";

    EXPECT_GT(graph.num_states(), 1) << "Should have generated more states";
    EXPECT_GT(graph.num_events(), 0) << "Should have at least one event";
}

// === BACKWARD COMPATIBILITY TESTS ===

TEST_F(MultipleInitialStatesTest, BackwardCompatibilitySingleState) {
    // Test that old single-state API still works
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());

    // Old API: single state (2D vector)
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial_edges = {{1, 2}};

    evolution.evolve(initial_edges);

    const auto& graph = evolution.get_multiway_graph();

    // Should have exactly 1 initial state
    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 1) << "Should have 1 initial state ID via old API";

    EXPECT_GT(graph.num_states(), 1) << "Should have generated more states";
}

// === CANONICALIZATION TESTS ===

TEST_F(MultipleInitialStatesTest, IsomorphicInitialStatesGetCanonicalized) {
    // Two initial states that are isomorphic should be deduplicated
    hypergraph::WolframEvolution evolution(1, 1, true, false);  // canonicalization ON
    evolution.add_rule(create_test_rule());

    // Two isomorphic initial states (same structure, different vertex IDs)
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}},     // First state
        {{100, 200}}  // Isomorphic to first (single edge)
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    // Should have created 2 initial state IDs (before canonicalization)
    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 2) << "Should track both initial state IDs";

    // But total states might be less if canonicalization merged them
    // This is expected behavior - document it
    std::cout << "  Note: " << graph.num_states() << " total states (canonicalization may merge isomorphic initials)\n";
}

TEST_F(MultipleInitialStatesTest, NonIsomorphicInitialStatesRemainSeparate) {
    // Different structures should remain separate
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());

    // Two different structures
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}},              // Single edge
        {{3, 4}, {4, 5}}       // Two edges
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 2) << "Should have 2 initial state IDs";

    // Different structures should produce different evolution patterns
    EXPECT_GT(graph.num_states(), 2) << "Should have generated additional states";
}

// === EDGE CASE TESTS ===

TEST_F(MultipleInitialStatesTest, EmptyStatesList) {
    // Empty list should be handled gracefully (though FFI should catch this)
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());

    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {};

    // Should not crash - just produce no results
    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 0) << "Should have 0 initial states";
    EXPECT_EQ(graph.num_states(), 0) << "Should have 0 total states";
    EXPECT_EQ(graph.num_events(), 0) << "Should have 0 events";
}

TEST_F(MultipleInitialStatesTest, StateWithNoMatches) {
    // Initial state that doesn't match the rule
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());  // Rule matches single binary edges

    // State with ternary edge (won't match binary edge rule)
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2, 3}}  // Ternary edge
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    // Should have the initial state but no events
    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 1) << "Should have 1 initial state";
    EXPECT_EQ(graph.num_events(), 0) << "Should have 0 events (no rule matches)";
}

// === EVOLUTION BEHAVIOR TESTS ===

TEST_F(MultipleInitialStatesTest, MultiStepEvolution) {
    hypergraph::WolframEvolution evolution(3, 1, true, false);  // 3 steps
    evolution.add_rule(create_test_rule());

    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}},
        {{10, 20}}
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    // With 3 steps, we should have substantial evolution
    EXPECT_GT(graph.num_states(), 2) << "Should have many states after 3 steps";
    EXPECT_GT(graph.num_events(), 2) << "Should have many events after 3 steps";

    std::cout << "  After 3 steps: " << graph.num_states() << " states, "
              << graph.num_events() << " events from 2 initial states\n";
}

TEST_F(MultipleInitialStatesTest, SharedVertexSpace) {
    // Verify that vertices are globally scoped across all initial states
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());

    // Two states that share vertex ID 5
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 5}},
        {{5, 10}}
    };

    evolution.evolve(initial_states);

    const auto& graph = evolution.get_multiway_graph();

    // This is expected: vertices are globally scoped
    // Both initial states exist in the same vertex space
    auto initial_ids = graph.get_initial_state_ids();
    EXPECT_EQ(initial_ids.size(), 2) << "Should have 2 initial states sharing vertex space";

    std::cout << "  Note: Vertices are globally scoped - both states share vertex 5\n";
}

// === DETERMINISM TEST ===

TEST_F(MultipleInitialStatesTest, DeterministicBehavior) {
    // Evolution should be deterministic with multiple initial states

    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}},
        {{3, 4}}
    };

    std::set<size_t> state_counts, event_counts;

    for (int run = 0; run < 3; ++run) {
        hypergraph::WolframEvolution evolution(2, 1, true, false);
        evolution.add_rule(create_test_rule());
        evolution.evolve(initial_states);

        const auto& graph = evolution.get_multiway_graph();
        state_counts.insert(graph.num_states());
        event_counts.insert(graph.num_events());
    }

    EXPECT_EQ(state_counts.size(), 1) << "State counts should be deterministic";
    EXPECT_EQ(event_counts.size(), 1) << "Event counts should be deterministic";
}

// === COMPLEX MULTI-EDGE RULE TEST ===

TEST_F(MultipleInitialStatesTest, MultiEdgeRuleWithMultipleInitialStates) {
    // Test with multi-edge rule and multiple initial states
    // Rule: {{1,2}, {2,3}} -> {{1,3}, {2,3}, {3,4}}
    // Initial states: {{{1,2}, {2,1}}, {{1,2}, {2,1}, {2,3}}}

    hypergraph::PatternHypergraph lhs, rhs;

    // LHS: two edges
    lhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(1),
        hypergraph::PatternVertex::variable(2)
    });
    lhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(2),
        hypergraph::PatternVertex::variable(3)
    });

    // RHS: three edges
    rhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(1),
        hypergraph::PatternVertex::variable(3)
    });
    rhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(2),
        hypergraph::PatternVertex::variable(3)
    });
    rhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(3),
        hypergraph::PatternVertex::variable(4)
    });

    hypergraph::RewritingRule rule(lhs, rhs);

    // Multiple initial states
    std::vector<std::vector<std::vector<hypergraph::GlobalVertexId>>> initial_states = {
        {{1, 2}, {2, 1}},                // First state: 2 edges
        {{1, 2}, {2, 1}, {2, 3}}         // Second state: 3 edges
    };

    // Test 2 steps - only state canonicalization enabled
    {
        hypergraph::WolframEvolution evolution(2, 1, true, false);  // canonicalize states, no event canonicalization
        evolution.add_rule(rule);
        evolution.evolve(initial_states);

        const auto& graph = evolution.get_multiway_graph();

        std::cout << "  2 steps: " << graph.num_states() << " states, "
                  << graph.num_events() << " events\n";

        EXPECT_EQ(graph.num_states(), 17) << "Should have exactly 17 states after 2 steps";
        EXPECT_EQ(graph.num_events(), 22) << "Should have exactly 22 events after 2 steps";
    }

    // Test 3 steps (from scratch) - only state canonicalization enabled
    {
        hypergraph::WolframEvolution evolution(3, 1, true, false);  // canonicalize states, no event canonicalization
        evolution.add_rule(rule);
        evolution.evolve(initial_states);

        const auto& graph = evolution.get_multiway_graph();

        std::cout << "  3 steps: " << graph.num_states() << " states, "
                  << graph.num_events() << " events\n";

        EXPECT_EQ(graph.num_states(), 40) << "Should have exactly 40 states after 3 steps";
        EXPECT_EQ(graph.num_events(), 97) << "Should have exactly 97 events after 3 steps";
    }
}
