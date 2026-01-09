#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"
#include <set>
#include <iostream>

using namespace hypergraph;

class UnifiedMultipleInitialStatesTest : public ::testing::Test {
protected:
    // Helper to create a simple rule: {{x, y}} -> {{x, y}, {y, z}}
    static RewriteRule create_test_rule() {
        return make_rule(0)
            .lhs({0, 1})
            .rhs({0, 1})
            .rhs({1, 2})
            .build();
    }
};

// === BASIC FUNCTIONALITY TESTS ===

TEST_F(UnifiedMultipleInitialStatesTest, TwoInitialStates) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(create_test_rule());

    // Two different initial states
    std::vector<std::vector<std::vector<VertexId>>> initial_states = {
        {{1, 2}},             // First state: single edge {1,2}
        {{10, 20}, {20, 30}}  // Second state: two edges
    };

    engine.evolve(initial_states, 2);

    EXPECT_GT(hg.num_states(), 2) << "Should have more than just the 2 initial states";
    EXPECT_GT(hg.num_events(), 0) << "Should have at least some events";
}

TEST_F(UnifiedMultipleInitialStatesTest, ThreeInitialStates) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(create_test_rule());

    // Three different initial states
    std::vector<std::vector<std::vector<VertexId>>> initial_states = {
        {{1, 2}},
        {{3, 4}},
        {{5, 6}}
    };

    engine.evolve(initial_states, 1);

    // Each initial state should produce at least one event
    EXPECT_GE(hg.num_events(), 3) << "Should have at least 3 events (one per initial state)";
}

TEST_F(UnifiedMultipleInitialStatesTest, SingleStateViaMultiStateAPI) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(create_test_rule());

    // Single state wrapped in vector
    std::vector<std::vector<std::vector<VertexId>>> initial_states = {
        {{1, 2}}
    };

    engine.evolve(initial_states, 1);

    EXPECT_GT(hg.num_states(), 1) << "Should have generated more states";
    EXPECT_GT(hg.num_events(), 0) << "Should have at least one event";
}

// === EDGE CASE TESTS ===

TEST_F(UnifiedMultipleInitialStatesTest, EmptyStatesList) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(create_test_rule());

    std::vector<std::vector<std::vector<VertexId>>> initial_states = {};

    // Should not crash - just produce no results
    engine.evolve(initial_states, 1);

    EXPECT_EQ(hg.num_states(), 0) << "Should have 0 total states";
    EXPECT_EQ(hg.num_events(), 0) << "Should have 0 events";
}

TEST_F(UnifiedMultipleInitialStatesTest, StateWithNoMatches) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(create_test_rule());  // Rule matches binary edges

    // State with ternary edge (won't match binary edge rule)
    std::vector<std::vector<std::vector<VertexId>>> initial_states = {
        {{1, 2, 3}}  // Ternary edge
    };

    engine.evolve(initial_states, 1);

    // Should have the initial state but no events (no rule matches)
    EXPECT_GE(hg.num_states(), 1) << "Should have initial state";
    EXPECT_EQ(hg.num_events(), 0) << "Should have 0 events (no rule matches)";
}

// === EVOLUTION BEHAVIOR TESTS ===

TEST_F(UnifiedMultipleInitialStatesTest, MultiStepEvolution) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);
    engine.add_rule(create_test_rule());

    std::vector<std::vector<std::vector<VertexId>>> initial_states = {
        {{1, 2}},
        {{10, 20}}
    };

    engine.evolve(initial_states, 3);

    // With 3 steps, we should have substantial evolution
    EXPECT_GT(hg.num_states(), 2) << "Should have many states after 3 steps";
    EXPECT_GT(hg.num_events(), 2) << "Should have many events after 3 steps";

    std::cout << "  After 3 steps: " << hg.num_states() << " states, "
              << hg.num_events() << " events from 2 initial states\n";
}

// === DETERMINISM TEST ===

TEST_F(UnifiedMultipleInitialStatesTest, DeterministicBehavior) {
    std::vector<std::vector<std::vector<VertexId>>> initial_states = {
        {{1, 2}},
        {{3, 4}}
    };

    std::set<size_t> state_counts, event_counts;

    for (int run = 0; run < 3; ++run) {
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, 4);
        engine.add_rule(create_test_rule());
        engine.evolve(initial_states, 2);

        state_counts.insert(hg.num_states());
        event_counts.insert(hg.num_events());
    }

    EXPECT_EQ(state_counts.size(), 1) << "State counts should be deterministic";
    EXPECT_EQ(event_counts.size(), 1) << "Event counts should be deterministic";
}

// === COMPLEX MULTI-EDGE RULE TEST ===

TEST_F(UnifiedMultipleInitialStatesTest, MultiEdgeRuleWithMultipleInitialStates) {
    // Rule: {{x,y}, {y,z}} -> {{x,z}, {y,z}, {z,w}}
    auto rule = make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 2})
        .rhs({1, 2})
        .rhs({2, 3})
        .build();

    std::vector<std::vector<std::vector<VertexId>>> initial_states = {
        {{1, 2}, {2, 1}},            // First state: 2 edges
        {{1, 2}, {2, 1}, {2, 3}}     // Second state: 3 edges
    };

    // Test 2 steps
    {
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, 4);
        engine.add_rule(rule);
        engine.evolve(initial_states, 2);

        std::cout << "  2 steps: " << hg.num_states() << " states, "
                  << hg.num_events() << " events\n";

        EXPECT_GT(hg.num_states(), 2) << "Should have generated states";
        EXPECT_GT(hg.num_events(), 0) << "Should have generated events";
    }

    // Test 3 steps
    {
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, 4);
        engine.add_rule(rule);
        engine.evolve(initial_states, 3);

        std::cout << "  3 steps: " << hg.num_states() << " states, "
                  << hg.num_events() << " events\n";

        EXPECT_GT(hg.num_states(), 2) << "Should have generated states";
    }
}
