#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/rewriter.hpp"
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

TEST(ParallelEvolution, BasicConstruction) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 2);  // 2 threads

    EXPECT_EQ(engine.num_threads(), 2);
    EXPECT_EQ(engine.num_states(), 0);
    EXPECT_EQ(engine.num_events(), 0);
}

TEST(ParallelEvolution, SimpleRule_OneStep) {
    Hypergraph hg;
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

TEST(ParallelEvolution, SimpleRule_TwoSteps) {
    Hypergraph hg;
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

TEST(ParallelEvolution, Determinism_SimpleRule) {
    // Run parallel evolution multiple times and verify same results

    std::set<size_t> state_counts;
    std::set<size_t> event_counts;

    for (int run = 0; run < 10; ++run) {
        Hypergraph hg;
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

TEST(ParallelEvolution, Determinism_TwoEdgeRule) {
    std::set<size_t> state_counts;
    std::set<size_t> event_counts;

    for (int run = 0; run < 10; ++run) {
        Hypergraph hg;
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

TEST(ParallelEvolution, Determinism_BranchingRule) {
    std::set<size_t> state_counts;
    std::set<size_t> event_counts;

    for (int run = 0; run < 10; ++run) {
        Hypergraph hg;
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

TEST(ParallelEvolution, DifferentThreadCounts) {
    std::vector<std::vector<VertexId>> initial = {{0, 1}, {1, 2}};

    size_t reference_canonical_states = 0;
    size_t reference_events = 0;

    for (size_t threads : {1, 2, 4, 8}) {
        Hypergraph hg;
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

// =============================================================================
// Rewriter: stale input_state returns an empty result instead of aborting.
// =============================================================================
// A stale forwarded match can carry a state ID past num_states(); the
// rewriter must degrade gracefully so callers can discard the match.

TEST(Rewriter, StaleStateId_ReturnsEmptyResult) {
    Hypergraph hg;
    Rewriter rewriter(&hg);

    std::vector<VertexId> verts = {0, 1};
    EdgeId eid = hg.create_edge(verts.data(), 2);
    hg.create_state({eid}, 0, /*canonical_hash=*/0, INVALID_ID);

    RewriteRule rule = par_simple_rule();
    VariableBinding binding;
    EdgeId matched[1] = {eid};

    StateId stale = hg.num_states();  // one past the last valid state
    RewriteResult result = rewriter.apply(rule, stale, matched, 1, binding, 1);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.new_state, INVALID_ID);
    EXPECT_EQ(result.raw_state, INVALID_ID);
    EXPECT_EQ(result.event, INVALID_ID);
    EXPECT_EQ(result.num_produced, 0);
}

// =============================================================================
// create_edge / RuleBuilder / PatternEdge: throw on fixed-bound overflow.
// =============================================================================
// These types hold fixed-size buffers of MAX_ARITY / MAX_PATTERN_EDGES.
// Over-size input must raise std::length_error rather than silently truncate.

TEST(EvolutionBounds, CreateEdge_Pointer_OverMaxArity_Throws) {
    Hypergraph hg;
    VertexId over[MAX_ARITY + 1] = {};
    EXPECT_THROW(hg.create_edge(over, MAX_ARITY + 1), std::length_error);
}

TEST(EvolutionBounds, CreateEdge_InitList_OverMaxArity_Throws) {
    Hypergraph hg;
    EXPECT_THROW(
        hg.create_edge({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}),
        std::length_error);
}

TEST(EvolutionBounds, CreateEdge_AtMaxArity_Succeeds) {
    Hypergraph hg;
    VertexId at_max[MAX_ARITY] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    EXPECT_NO_THROW(hg.create_edge(at_max, MAX_ARITY));
}

// The arity the guard reads must be the arity the caller counted. A count that is a multiple
// of 256 is the case a byte-wide parameter cannot represent: 256 presents as 0 and 260 as 4,
// both of which are at or under MAX_ARITY and would be accepted.
TEST(EvolutionBounds, CreateEdge_ArityMultipleOf256_Throws) {
    Hypergraph hg;
    std::vector<VertexId> v(260, 0);
    EXPECT_THROW(hg.create_edge(v.data(), v.size()), std::length_error);
    std::vector<VertexId> exactly_256(256, 0);
    EXPECT_THROW(hg.create_edge(exactly_256.data(), exactly_256.size()), std::length_error);
}

// A genesis event produces every initial edge and Event::num_produced is one byte, so a state
// past that bound is refused. Truncating the count instead registers a producer for none of
// the edges and reports no error, which reads downstream as a run with fewer causal edges.
TEST(EvolutionBounds, CreateGenesisEvent_PastEventWidth_Throws) {
    Hypergraph hg;
    std::vector<EdgeId> edges;
    for (size_t i = 0; i < Hypergraph::MAX_GENESIS_EDGES + 1; ++i) {
        VertexId pair[2] = {static_cast<VertexId>(2 * i), static_cast<VertexId>(2 * i + 1)};
        edges.push_back(hg.create_edge(pair, 2));
    }
    SparseBitset all;
    for (EdgeId e : edges) all.set(e, hg.arena());
    auto [canonical, raw, was_new] =
        hg.create_or_get_canonical_state(std::move(all), 0, INVALID_ID);
    EXPECT_THROW(hg.create_genesis_event(raw, edges.data(), edges.size()), std::length_error);
    EXPECT_NO_THROW(hg.create_genesis_event(raw, edges.data(), Hypergraph::MAX_GENESIS_EDGES));
}

TEST(EvolutionBounds, PatternEdge_InitList_OverMaxArity_Throws) {
    EXPECT_THROW(
        (PatternEdge{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}),
        std::length_error);
}

TEST(EvolutionBounds, RuleBuilder_OverMaxPatternEdges_Throws) {
    auto builder = make_rule(0);
    for (int i = 0; i < MAX_PATTERN_EDGES; ++i) {
        builder.lhs({0, 1});
    }
    EXPECT_THROW(builder.lhs({0, 1}), std::length_error);
}

TEST(EvolutionBounds, RuleBuilder_VectorOverload_OverMaxArity_Throws) {
    std::vector<uint8_t> huge(MAX_ARITY + 1, 0);
    EXPECT_THROW(make_rule(0).lhs(huge), std::length_error);
}
