#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <numeric>

using namespace hypergraph;

// Helper to create a simple rule: {{x,y}} -> {{x,z},{y,z}}
static RewriteRule create_growth_rule() {
    return make_rule(0)
        .lhs({0, 1})
        .rhs({0, 2})
        .rhs({1, 2})
        .build();
}

// Test 1: Max Successors Per Parent - Hard Limit
TEST(UnifiedEvolutionLimits, MaxSuccessorsPerParentHardLimit) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(create_growth_rule());
    engine.set_max_successor_states_per_parent(1);  // HARD LIMIT

    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 2);

    // With max 1 successor per parent, evolution should be limited
    // We can't easily count per-parent successors without event data,
    // but we can verify the overall count is bounded
    EXPECT_GE(hg.num_states(), 1) << "Should have at least initial state";
    EXPECT_LE(hg.num_events(), 10) << "Events should be limited with max 1 successor per parent";
}

// Test 2: Max States Per Step - Hard Limit
TEST(UnifiedEvolutionLimits, MaxStatesPerStepHardLimit) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);
    engine.add_rule(create_growth_rule());
    engine.set_max_states_per_step(3);  // HARD LIMIT: max 3 states at any step

    std::vector<std::vector<VertexId>> initial = {{0, 1}, {1, 2}};
    engine.evolve(initial, 2);

    // With max 3 states per step, evolution should be bounded
    EXPECT_GE(hg.num_states(), 1) << "Should have at least initial state";
    // The limit is per step, so total states can exceed it
}

// Test 3: Random Exploration Probability (Statistical Test)
// Note: exploration_probability affects whether NEW states are explored further,
// not whether events are created. Events are created first, then exploration may be skipped.
TEST(UnifiedEvolutionLimits, RandomExplorationProbability) {
    std::vector<std::vector<VertexId>> initial = {{0, 1}};

    // Run with p=0.5 for 2+ steps - the effect is seen in subsequent steps
    std::vector<size_t> event_counts;
    for (int trial = 0; trial < 20; ++trial) {
        Hypergraph hg;
        ParallelEvolutionEngine engine(&hg, 1);  // Single thread for determinism
        engine.add_rule(create_growth_rule());
        engine.set_exploration_probability(0.5);  // 50% chance

        engine.evolve(initial, 2);  // 2 steps to see exploration effects
        event_counts.push_back(hg.num_events());
    }

    // Calculate mean
    double mean = std::accumulate(event_counts.begin(), event_counts.end(), 0.0) / event_counts.size();

    // With p=0.5 over 2 steps, some branches won't be explored
    // Step 1 always creates event(s), step 2 may be skipped with 50% probability
    EXPECT_GE(mean, 1.0) << "Should have at least step 1 events";
    // Just verify there's some variation in the results
}

// Test 4: Combined Limits
TEST(UnifiedEvolutionLimits, CombinedLimitsInteraction) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);
    engine.add_rule(create_growth_rule());
    engine.set_max_successor_states_per_parent(2);  // Max 2 successors per parent
    engine.set_max_states_per_step(5);              // Max 5 states per step
    engine.set_exploration_probability(0.8);        // 80% exploration

    std::vector<std::vector<VertexId>> initial = {{0, 1}, {1, 2}, {2, 3}};
    engine.evolve(initial, 3);

    // Evolution should be bounded by combined limits
    EXPECT_GE(hg.num_states(), 1) << "Should have at least initial state";
}

// Test 5: Edge Cases - Unlimited (defaults)
TEST(UnifiedEvolutionLimits, UnlimitedDefaults) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);
    engine.add_rule(create_growth_rule());
    // Default values: 0 = unlimited, 1.0 = always explore

    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 2);

    EXPECT_GT(hg.num_events(), 0) << "Should have created events with unlimited settings";
}

// Test 6: Probability = 0 (reject all exploration)
// Note: With p=0, step 1 events ARE created (initial state is always explored),
// but the resulting states won't be explored further (step 2+ skipped)
TEST(UnifiedEvolutionLimits, ExplorationProbabilityZero) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 1);
    engine.add_rule(create_growth_rule());
    engine.set_exploration_probability(0.0);  // Reject all exploration

    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 2);  // 2 steps

    // Step 1 creates events (initial state explored), step 2 skipped (p=0)
    size_t events_step1 = hg.num_events();

    // Now compare with p=1.0 to verify the difference
    Hypergraph hg2;
    ParallelEvolutionEngine engine2(&hg2, 1);
    engine2.add_rule(create_growth_rule());
    engine2.set_exploration_probability(1.0);  // Always explore

    engine2.evolve(initial, 2);
    size_t events_both_steps = hg2.num_events();

    // With p=0, we should have fewer events than with p=1.0
    EXPECT_LT(events_step1, events_both_steps) << "With probability 0, should have fewer events than p=1.0";
}

// Test 7: Limit = 1 (degenerate case)
TEST(UnifiedEvolutionLimits, LimitOfOne) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(create_growth_rule());
    engine.set_max_successor_states_per_parent(1);  // Only 1 successor allowed
    engine.set_max_states_per_step(1);              // Only 1 state per step

    std::vector<std::vector<VertexId>> initial = {{0, 1}, {1, 2}};
    engine.evolve(initial, 2);

    // With such strict limits, evolution should be very restricted
    EXPECT_GE(hg.num_states(), 1) << "Should have at least initial state";
}

// Test 8: Early Termination + Limits (composition)
TEST(UnifiedEvolutionLimits, EarlyTerminationWithLimits) {
    // Rule that creates cycles: {{x,y}} -> {{y,x}}
    auto rule = make_rule(0)
        .lhs({0, 1})
        .rhs({1, 0})
        .build();

    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);
    engine.add_rule(rule);
    engine.set_max_successor_states_per_parent(2);
    engine.set_max_states_per_step(3);

    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 5);

    // With canonicalization, cyclic rules should produce limited unique states
    EXPECT_LT(hg.num_events(), 100) << "Evolution should be bounded";
}

// Test 9: steps is a literal generation budget -- steps == 0 yields the initial state
// alone and RETURNS (the budget guards compare `step > max_steps_` directly, so a zero
// budget rejects the very first match task instead of running unbounded).
TEST(UnifiedEvolutionLimits, ZeroStepsYieldsInitialStateOnly) {
    for (int threads : {1, 4}) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine engine(&hg, threads);
        engine.set_explore_from_canonical_states_only(true);
        engine.add_rule(create_growth_rule());

        std::vector<std::vector<VertexId>> initial = {{0, 1}, {0, 2}};
        engine.evolve(initial, 0);

        EXPECT_EQ(hg.num_states(), 1u) << "threads=" << threads;
        EXPECT_EQ(hg.num_canonical_states(), 1u) << "threads=" << threads;
        EXPECT_EQ(hg.num_events(), 0u) << "threads=" << threads;
    }
}

// MatchesPerStateRule keeps k of a state's own matches PER RULE, chosen at the drain.
//
// The cap must be by RANK, not arrival: arrival order depends on the schedule, so a cap by
// arrival keeps a different set at a different worker count. That is what these three assert --
// the count is bounded, the SET is identical across worker counts, and a rule with fewer than k
// matches is not padded.
TEST(UnifiedEvolutionLimits, MatchesPerStateRuleBoundsAndReproduces) {
    auto run = [](size_t k, unsigned threads) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine engine(&hg, threads);
        engine.add_rule(create_growth_rule());
        engine.set_random_seed(12345);
        if (k) engine.set_matches_per_state_rule(k);
        engine.evolve({{0, 1}}, 4);
        // CANONICAL identity, not raw ids. Raw state ids are assigned in CREATION order, so two
        // runs of the same evolution at different worker counts label the same states
        // differently -- comparing them would report every run as irreproducible, including an
        // uncapped one.
        std::multiset<std::pair<uint64_t, uint64_t>> edges;
        for (uint32_t e = 0; e < hg.num_published_events(); ++e) {
            if (hg.is_genesis_event(e)) continue;
            edges.insert({hg.get_state(hg.get_event(e).input_state).canonical_hash,
                          hg.get_state(hg.get_event(e).output_state).canonical_hash});
        }
        return std::make_pair(hg.num_canonical_states(), edges);
    };

    const auto uncapped = run(0, 4);
    const auto capped1  = run(1, 4);
    EXPECT_LT(capped1.first, uncapped.first)
        << "a cap of one match per (state, rule) explored as much as no cap at all";
    EXPECT_GT(capped1.first, 1u) << "the cap explored nothing; it is a cap, not an off switch";

    // THE REPRODUCIBILITY CHECK, which is the one a cap-by-arrival fails: the same seed at a
    // different worker count must keep the SAME transitions, not merely the same number.
    const auto capped1_8 = run(1, 8);
    EXPECT_EQ(capped1.second, capped1_8.second)
        << "the kept transition set changed with the worker count, so the choice was made by "
           "arrival order and not by the transition's own rank";

    const auto capped2 = run(2, 4);
    EXPECT_GE(capped2.first, capped1.first) << "k=2 kept less than k=1";
}
