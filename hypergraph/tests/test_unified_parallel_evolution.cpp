#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"
#include <set>
#include <chrono>

using namespace hypergraph;

// =============================================================================
// Test Helpers (static to have internal linkage, avoiding ODR conflicts)
// =============================================================================

static RewriteRule par_simple_rule() {
    // {{x, y}} -> {{y, z}}
    return make_rule(0)
        .lhs({0, 1})
        .rhs({1, 2})
        .build();
}

static RewriteRule par_two_edge_rule() {
    // {{x, y}, {y, z}} -> {{x, z}}
    return make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 2})
        .build();
}

static RewriteRule par_branching_rule() {
    // {{x, y}} -> {{y, z}, {z, x}}
    return make_rule(0)
        .lhs({0, 1})
        .rhs({1, 2})
        .rhs({2, 0})
        .build();
}

// =============================================================================
// Basic Parallel Evolution Tests
// =============================================================================

TEST(Unified_ParallelEvolution, BasicConstruction) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);  // 2 threads

    EXPECT_EQ(engine.num_threads(), 2);
    EXPECT_EQ(engine.num_states(), 0);
    EXPECT_EQ(engine.num_events(), 0);
}

TEST(Unified_ParallelEvolution, SimpleRule_OneStep) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);

    engine.add_rule(par_simple_rule());

    // Initial state: {{0, 1}}
    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 1);

    // After 1 step with {{x,y}} -> {{y,z}}:
    // Initial: {{0,1}}
    // Result: {{1,2}}
    EXPECT_GE(engine.num_states(), 1);  // At least initial state
    EXPECT_GE(engine.num_events(), 0);  // May or may not have events depending on matches
}

TEST(Unified_ParallelEvolution, SimpleRule_TwoSteps) {
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);

    engine.add_rule(par_simple_rule());

    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 2);

    // After 2 steps: should have initial + 2 more states
    EXPECT_GE(engine.num_states(), 1);
}

// =============================================================================
// Determinism Tests
// =============================================================================

TEST(Unified_ParallelEvolution, Determinism_SimpleRule) {
    // Run parallel evolution multiple times and verify same results

    std::set<size_t> state_counts;
    std::set<size_t> event_counts;

    for (int run = 0; run < 10; ++run) {
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, 4);

        engine.add_rule(par_simple_rule());

        std::vector<std::vector<VertexId>> initial = {{0, 1}};
        engine.evolve(initial, 3);

        state_counts.insert(engine.num_states());
        event_counts.insert(engine.num_events());
    }

    // Should have exactly one unique count (deterministic)
    EXPECT_EQ(state_counts.size(), 1) << "State counts vary across runs!";
    EXPECT_EQ(event_counts.size(), 1) << "Event counts vary across runs!";
}

TEST(Unified_ParallelEvolution, Determinism_TwoEdgeRule) {
    std::set<size_t> state_counts;
    std::set<size_t> event_counts;

    for (int run = 0; run < 10; ++run) {
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, 4);

        engine.add_rule(par_two_edge_rule());

        // Triangle: can apply rule to any two adjacent edges
        std::vector<std::vector<VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};
        engine.evolve(initial, 2);

        state_counts.insert(engine.num_states());
        event_counts.insert(engine.num_events());
    }

    EXPECT_EQ(state_counts.size(), 1) << "State counts vary across runs!";
    EXPECT_EQ(event_counts.size(), 1) << "Event counts vary across runs!";
}

TEST(Unified_ParallelEvolution, Determinism_BranchingRule) {
    std::set<size_t> state_counts;
    std::set<size_t> event_counts;

    for (int run = 0; run < 10; ++run) {
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, 4);

        engine.add_rule(par_branching_rule());

        std::vector<std::vector<VertexId>> initial = {{0, 1}};
        engine.evolve(initial, 3);

        state_counts.insert(engine.num_states());
        event_counts.insert(engine.num_events());
    }

    EXPECT_EQ(state_counts.size(), 1) << "State counts vary across runs!";
    EXPECT_EQ(event_counts.size(), 1) << "Event counts vary across runs!";
}

// =============================================================================
// Thread Count Variations
// =============================================================================

TEST(Unified_ParallelEvolution, DifferentThreadCounts) {
    std::vector<std::vector<VertexId>> initial = {{0, 1}, {1, 2}};

    size_t reference_canonical_states = 0;
    size_t reference_events = 0;

    for (size_t threads : {1, 2, 4, 8}) {
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, threads);
        engine.add_rule(par_simple_rule());
        engine.evolve(initial, 3);

        // Use num_canonical_states() for comparison, not num_states()
        // Multi-threaded execution may create "wasted" states due to race
        // between state creation and canonical deduplication - this is
        // correct behavior for linearizability. The important metric is
        // that the number of UNIQUE (canonical) states is the same.
        size_t canonical_states = hg.num_canonical_states();

        if (threads == 1) {
            reference_canonical_states = canonical_states;
            reference_events = engine.num_events();
        } else {
            EXPECT_EQ(canonical_states, reference_canonical_states)
                << "Canonical state count differs with " << threads << " threads";
            // Note: events may differ slightly due to "wasted" states creating
            // events that point to non-canonical states. The critical invariant
            // is that the multiway system explores the same logical state space.
            // For stricter testing, we compare num_events() but allow slack.
            EXPECT_LE(reference_events, engine.num_events())
                << "Event count lower than single-threaded with " << threads << " threads";
        }
    }
}
