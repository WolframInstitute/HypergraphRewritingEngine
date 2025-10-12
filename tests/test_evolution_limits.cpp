#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>
#include <algorithm>
#include <map>

using namespace hypergraph;

// Helper function to count successors of a state
int count_successors(const MultiwayGraph& graph, StateID state_id) {
    auto events = graph.get_all_events();
    return std::count_if(events.begin(), events.end(), [state_id](const WolframEvent& e) {
        return e.input_state_id == state_id;
    });
}

// Helper function to count states at a specific step
int count_states_at_step(const MultiwayGraph& graph, std::size_t step) {
    auto events = graph.get_all_events();
    std::set<StateID> states_at_step;
    for (const auto& event : events) {
        if (event.step == step) {
            states_at_step.insert(event.output_state_id);
        }
    }
    return states_at_step.size();
}

// Test 1: Max Successors Per Parent - Hard Limit
TEST(EvolutionLimits, MaxSuccessorsPerParentHardLimit) {
    // Rule: {{1,2}} -> {{1,3},{2,3}} (generates 2 children from 1 parent when matched multiple ways)
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    // Initial state: {{0,1}}
    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}};

    // Limit: 1 successor per parent
    WolframEvolution evolution(
        /*max_steps=*/2,
        /*num_threads=*/4,
        /*canonicalization=*/true,
        /*full_capture=*/false,
        /*event_dedup=*/true,
        /*transitive_reduction=*/true,
        /*early_termination=*/false,
        /*full_capture_non_canonicalised=*/false,
        /*max_successor_states_per_parent=*/1  // HARD LIMIT
    );

    evolution.add_rule(rule);
    evolution.evolve(initial_edges);

    auto& graph = evolution.get_multiway_graph();
    auto events = graph.get_all_events();

    // Count events from initial state (state 0)
    int events_from_initial = std::count_if(events.begin(), events.end(), [](const WolframEvent& e) {
        return e.input_state_id.value == 0;  // Initial state has ID 0
    });

    EXPECT_EQ(events_from_initial, 1) << "Hard limit violated - expected exactly 1 successor from initial state";
}

// Test 2: Max States Per Step - Hard Limit
TEST(EvolutionLimits, MaxStatesPerStepHardLimit) {
    // Rule that can create multiple successors: {{1,2}} -> {{1,3},{2,3}}
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    // Initial edges that could generate many states
    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}, {1, 2}};

    // Limit: 3 states at step 1
    WolframEvolution evolution(
        /*max_steps=*/2,
        /*num_threads=*/8,
        /*canonicalization=*/true,
        /*full_capture=*/false,
        /*event_dedup=*/true,
        /*transitive_reduction=*/true,
        /*early_termination=*/false,
        /*full_capture_non_canonicalised=*/false,
        /*max_successor_states_per_parent=*/0,  // Unlimited successors per parent
        /*max_states_per_step=*/3  // HARD LIMIT: max 3 states at any step
    );

    evolution.add_rule(rule);
    evolution.evolve(initial_edges);

    auto& graph = evolution.get_multiway_graph();
    int states_at_step_1 = count_states_at_step(graph, 1);

    EXPECT_LE(states_at_step_1, 3) << "Hard limit violated - too many states at step 1";
}

// Test 3: Random Exploration Probability (Statistical Test)
TEST(EvolutionLimits, RandomExplorationProbability) {
    // Simple rule: {{1,2}} -> {{1,3},{2,3}}
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}};

    // Run with p=0.5 multiple times and check distribution
    std::vector<size_t> event_counts;
    for (int trial = 0; trial < 20; ++trial) {
        WolframEvolution evolution(
            /*max_steps=*/1,
            /*num_threads=*/1,
            /*canonicalization=*/true,
            /*full_capture=*/false,
            /*event_dedup=*/true,
            /*transitive_reduction=*/true,
            /*early_termination=*/false,
            /*full_capture_non_canonicalised=*/false,
            /*max_successor_states_per_parent=*/0,
            /*max_states_per_step=*/0,
            /*exploration_probability=*/0.5  // 50% chance
        );

        evolution.add_rule(rule);
        evolution.evolve(initial_edges);
        event_counts.push_back(evolution.get_multiway_graph().num_events());
    }

    // Calculate mean
    double mean = std::accumulate(event_counts.begin(), event_counts.end(), 0.0) / event_counts.size();

    // With p=0.5, we expect roughly half to be rejected
    // So mean should be around 0.5 events (some runs 0, some runs 1)
    EXPECT_GE(mean, 0.2) << "Too few events - probability seems too low";
    EXPECT_LE(mean, 0.8) << "Too many events - probability seems too high";
}

// Test 4: Combined Limits
TEST(EvolutionLimits, CombinedLimitsInteraction) {
    // Rule: {{1,2}} -> {{1,3},{2,3}}
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}, {1, 2}, {2, 3}};

    // Test with all three limits active
    WolframEvolution evolution(
        /*max_steps=*/3,
        /*num_threads=*/8,
        /*canonicalization=*/true,
        /*full_capture=*/false,
        /*event_dedup=*/true,
        /*transitive_reduction=*/true,
        /*early_termination=*/false,
        /*full_capture_non_canonicalised=*/false,
        /*max_successor_states_per_parent=*/2,  // Max 2 successors per parent
        /*max_states_per_step=*/5,              // Max 5 states per step
        /*exploration_probability=*/0.8         // 80% exploration
    );

    evolution.add_rule(rule);
    evolution.evolve(initial_edges);

    auto& graph = evolution.get_multiway_graph();
    auto events = graph.get_all_events();

    // Verify successor limit
    std::map<StateID, int> successor_counts;
    for (const auto& event : events) {
        successor_counts[event.input_state_id]++;
    }

    for (const auto& [state_id, count] : successor_counts) {
        EXPECT_LE(count, 2) << "Successor limit violated for state " << state_id.value;
    }

    // Verify step limit
    for (size_t step = 0; step < 3; ++step) {
        int states = count_states_at_step(graph, step);
        EXPECT_LE(states, 5) << "Step limit violated at step " << step;
    }
}

// Test 5: Edge Cases - Unlimited (defaults)
TEST(EvolutionLimits, UnlimitedDefaults) {
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}};

    // Default values (0 = unlimited, 1.0 = always explore)
    WolframEvolution evolution(
        /*max_steps=*/2,
        /*num_threads=*/4,
        /*canonicalization=*/true,
        /*full_capture=*/false,
        /*event_dedup=*/true,
        /*transitive_reduction=*/true,
        /*early_termination=*/false,
        /*full_capture_non_canonicalised=*/false
        // max_successor_states_per_parent defaults to 0 (unlimited)
        // max_states_per_step defaults to 0 (unlimited)
        // exploration_probability defaults to 1.0 (always explore)
    );

    evolution.add_rule(rule);
    evolution.evolve(initial_edges);

    auto& graph = evolution.get_multiway_graph();
    EXPECT_GT(graph.num_events(), 0) << "Should have created events with unlimited settings";
}

// Test 6: Probability = 0 (reject all)
TEST(EvolutionLimits, ExplorationProbabilityZero) {
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}};

    WolframEvolution evolution(
        /*max_steps=*/1,
        /*num_threads=*/1,
        /*canonicalization=*/true,
        /*full_capture=*/false,
        /*event_dedup=*/true,
        /*transitive_reduction=*/true,
        /*early_termination=*/false,
        /*full_capture_non_canonicalised=*/false,
        /*max_successor_states_per_parent=*/0,
        /*max_states_per_step=*/0,
        /*exploration_probability=*/0.0  // Reject all
    );

    evolution.add_rule(rule);
    evolution.evolve(initial_edges);

    auto& graph = evolution.get_multiway_graph();
    EXPECT_EQ(graph.num_events(), 0) << "With probability 0, no events should be created";
}

// Test 7: Limit = 1 (degenerate case)
TEST(EvolutionLimits, LimitOfOne) {
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}, {1, 2}};

    WolframEvolution evolution(
        /*max_steps=*/2,
        /*num_threads=*/4,
        /*canonicalization=*/true,
        /*full_capture=*/false,
        /*event_dedup=*/true,
        /*transitive_reduction=*/true,
        /*early_termination=*/false,
        /*full_capture_non_canonicalised=*/false,
        /*max_successor_states_per_parent=*/1,  // Only 1 successor allowed
        /*max_states_per_step=*/1              // Only 1 state per step
    );

    evolution.add_rule(rule);
    evolution.evolve(initial_edges);

    auto& graph = evolution.get_multiway_graph();
    auto events = graph.get_all_events();

    // With such strict limits, evolution should be very restricted
    for (const auto& event : events) {
        int successors = count_successors(graph, event.input_state_id);
        EXPECT_LE(successors, 1) << "Violated limit of 1 successor per parent";
    }

    for (size_t step = 0; step < 2; ++step) {
        int states = count_states_at_step(graph, step);
        EXPECT_LE(states, 1) << "Violated limit of 1 state per step";
    }
}

// Test 8: Early Termination + Limits (composition)
TEST(EvolutionLimits, EarlyTerminationWithLimits) {
    // Rule that creates cycles: {{1,2}} -> {{2,1}}
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(1)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial_edges = {{0, 1}};

    WolframEvolution evolution(
        /*max_steps=*/5,
        /*num_threads=*/4,
        /*canonicalization=*/true,
        /*full_capture=*/false,
        /*event_dedup=*/true,
        /*transitive_reduction=*/true,
        /*early_termination=*/true,  // Stop on duplicate states
        /*full_capture_non_canonicalised=*/false,
        /*max_successor_states_per_parent=*/2,
        /*max_states_per_step=*/3,
        /*exploration_probability=*/1.0
    );

    evolution.add_rule(rule);
    evolution.evolve(initial_edges);

    auto& graph = evolution.get_multiway_graph();

    // Early termination should prevent infinite loops
    // Limits should still be respected
    EXPECT_LT(graph.num_events(), 100) << "Early termination should have stopped the evolution";
}
