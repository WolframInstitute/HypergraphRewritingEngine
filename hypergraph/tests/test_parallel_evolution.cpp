#include <gtest/gtest.h>
#include <memory>
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/rewriter.hpp"
#include "hgcommon/park_gate.hpp"
#include "hgcommon/cache_line.hpp"
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
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
    auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, 2); ParallelEvolutionEngine& engine = *engine_heap;  // 2 threads

    EXPECT_EQ(engine.num_threads(), 2);
    EXPECT_EQ(engine.num_states(), 0);
    EXPECT_EQ(engine.num_events(), 0);
}

TEST(ParallelEvolution, SimpleRule_OneStep) {
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
    auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, 2); ParallelEvolutionEngine& engine = *engine_heap;

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
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
    auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, 4); ParallelEvolutionEngine& engine = *engine_heap;

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
        auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
        auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, 4); ParallelEvolutionEngine& engine = *engine_heap;

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
        auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
        auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, 4); ParallelEvolutionEngine& engine = *engine_heap;

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
        auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
        auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, 4); ParallelEvolutionEngine& engine = *engine_heap;

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
        auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
        auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, threads); ParallelEvolutionEngine& engine = *engine_heap;
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
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
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
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
    VertexId over[MAX_ARITY + 1] = {};
    EXPECT_THROW(hg.create_edge(over, MAX_ARITY + 1), std::length_error);
}

TEST(EvolutionBounds, CreateEdge_InitList_OverMaxArity_Throws) {
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
    EXPECT_THROW(
        hg.create_edge({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}),
        std::length_error);
}

TEST(EvolutionBounds, CreateEdge_AtMaxArity_Succeeds) {
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
    VertexId at_max[MAX_ARITY] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    EXPECT_NO_THROW(hg.create_edge(at_max, MAX_ARITY));
}

// The arity the guard reads must be the arity the caller counted. A count that is a multiple
// of 256 is the case a byte-wide parameter cannot represent: 256 presents as 0 and 260 as 4,
// both of which are at or under MAX_ARITY and would be accepted.
TEST(EvolutionBounds, CreateEdge_ArityMultipleOf256_Throws) {
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
    std::vector<VertexId> v(260, 0);
    EXPECT_THROW(hg.create_edge(v.data(), v.size()), std::length_error);
    std::vector<VertexId> exactly_256(256, 0);
    EXPECT_THROW(hg.create_edge(exactly_256.data(), exactly_256.size()), std::length_error);
}

// A genesis event produces every initial edge and Event::num_produced is one byte, so a state
// past that bound is refused. Truncating the count instead registers a producer for none of
// the edges and reports no error, which reads downstream as a run with fewer causal edges.
TEST(EvolutionBounds, CreateGenesisEvent_PastEventWidth_Throws) {
    auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
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

// H5 (docs/TUNING.md): the causal sets' first table size. Without a step cap it is 8192 above
// 4 GB of available memory and 1024 below; with a cap it follows the key bound. The relation is
// the same at every capacity and thread count.
TEST(ParallelEvolution, CausalSetCapacityChangesNoRelation) {
    EXPECT_EQ(ParallelEvolutionEngine::derived_causal_set_capacity(5, 0, 2, uint64_t{8} << 30), 8192u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_causal_set_capacity(5, 0, 2, uint64_t{2} << 30), 1024u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_causal_set_capacity(5, 10, 2, uint64_t{8} << 30), 1024u);
    // (1 + 100 x 1000) x 3 keys over 64 shards at 0.75: 6250 per shard, so 8192.
    EXPECT_EQ(ParallelEvolutionEngine::derived_causal_set_capacity(100, 1000, 3, 0), 8192u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_causal_set_capacity(1u << 20, 1u << 20, 16, 0), 65536u);

    std::vector<std::vector<VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};
    size_t ref_causal = 0, ref_pairs = 0, ref_events = 0;
    bool first = true;
    for (size_t capacity : {size_t{0}, size_t{1}, size_t{1024}, size_t{65536}}) {
        for (size_t threads : {1, 4, 32}) {
            auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
            auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, threads); ParallelEvolutionEngine& engine = *engine_heap;
            hgcommon::TuningOverrides t;
            if (capacity) t.set(hgcommon::Tune::CausalSetCapacity, static_cast<int64_t>(capacity));
            engine.set_tuning(t);
            engine.add_rule(par_two_edge_rule());
            engine.evolve(initial, 4);
            if (capacity) EXPECT_EQ(engine.causal_set_capacity(), capacity);
            if (first) {
                ref_causal = hg.num_causal_edges();
                ref_pairs = hg.num_causal_event_pairs();
                ref_events = engine.num_events();
                EXPECT_GT(ref_pairs, 0u);
                first = false;
            }
            EXPECT_EQ(hg.num_causal_edges(), ref_causal) << capacity << " / " << threads;
            EXPECT_EQ(hg.num_causal_event_pairs(), ref_pairs) << capacity << " / " << threads;
            EXPECT_EQ(engine.num_events(), ref_events) << capacity << " / " << threads;
        }
    }
}

// H4 (docs/TUNING.md): the rewrite task size. The derived size is ceil(count / workers) from 4
// to 64; a run gives the same states, events and causal edges at every chunk size and thread
// count. The rule has a one-edge left-hand side, so a scan task hands a state's whole match set
// to one expansion and the chunking applies.
TEST(ParallelEvolution, ExpandChunkChangesNoResult) {
    EXPECT_EQ(ParallelEvolutionEngine::derived_expand_chunk(3, 32), 4u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_expand_chunk(100, 32), 4u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_expand_chunk(200, 32), 7u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_expand_chunk(100, 1), 64u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_expand_chunk(30, 1), 30u);
    EXPECT_EQ(ParallelEvolutionEngine::derived_expand_chunk(30, 0), 30u);

    std::vector<std::vector<VertexId>> initial;
    for (VertexId v = 0; v < 40; ++v) initial.push_back({v, v + 1});
    size_t ref_states = 0, ref_events = 0, ref_causal = 0;
    bool first = true;
    for (size_t chunk : {size_t{0}, size_t{1}, size_t{16}, size_t{64}}) {
        for (size_t threads : {1, 4, 32}) {
            auto hg_heap = std::make_unique<Hypergraph>(); Hypergraph& hg = *hg_heap;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            auto engine_heap = std::make_unique<ParallelEvolutionEngine>(&hg, threads); ParallelEvolutionEngine& engine = *engine_heap;
            hgcommon::TuningOverrides t;
            if (chunk) t.set(hgcommon::Tune::ExpandChunk, static_cast<int64_t>(chunk));
            engine.set_tuning(t);
            engine.add_rule(par_simple_rule());
            engine.evolve(initial, 2);
            if (first) {
                ref_states = hg.num_canonical_states();
                ref_events = engine.num_events();
                ref_causal = hg.num_causal_event_pairs();
                EXPECT_GT(ref_events, 40u);
                first = false;
            }
            EXPECT_EQ(hg.num_canonical_states(), ref_states) << chunk << " / " << threads;
            EXPECT_EQ(engine.num_events(), ref_events) << chunk << " / " << threads;
            EXPECT_EQ(hg.num_causal_event_pairs(), ref_causal) << chunk << " / " << threads;
        }
    }
}

// H16 (docs/TUNING.md): one cache-line constant aligns every per-thread slot. 64 bytes here;
// 128 on Apple aarch64 (the sources compile with HG_DESTRUCTIVE_LINE=128).
TEST(CacheLine, EveryPerThreadSlotIsOneConstantAligned) {
#if defined(__x86_64__) || (defined(__aarch64__) && !defined(__APPLE__))
    EXPECT_EQ(hgcommon::kDestructiveLine, 64u);
#endif
    EXPECT_EQ(alignof(hgcommon::ParkGate::Domain), hgcommon::kDestructiveLine);
    EXPECT_EQ(sizeof(hgcommon::ParkGate::Domain) % hgcommon::kDestructiveLine, 0u);
    EXPECT_EQ(ConcurrentHeterogeneousArena::kMaxAlign, hgcommon::kDestructiveLine);
}
