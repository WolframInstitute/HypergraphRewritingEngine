#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"

#include <array>
#include <cstddef>

using namespace hypergraph;

// These tests exercise the sampling / pruning code paths that are otherwise only
// reached through the paclet FFI: evolve_uniform_random() and the
// ExplorationProbability draw on the default dataflow path. They pin down the
// determinism contract: with a nonzero random seed and a single worker thread,
// both paths must be bit-reproducible run-to-run.

namespace {

// Growth rule {{x,y}} -> {{x,y},{y,z}}: consumes the matched edge and produces it
// again plus a new edge on a fresh vertex, so the frontier branches every step.
RewriteRule make_growth_rule() {
    return make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();
}

struct RunMetrics {
    size_t canonical_states;
    size_t events;
    size_t causal_edges;
    size_t branchial_edges;

    bool operator==(const RunMetrics& o) const {
        return canonical_states == o.canonical_states &&
               events == o.events &&
               causal_edges == o.causal_edges &&
               branchial_edges == o.branchial_edges;
    }
};

// Run evolve_uniform_random in its own scope so the engine (and its worker
// thread) is fully torn down before the next run, then report the metrics.
RunMetrics run_uniform_random(uint64_t seed, size_t steps, size_t matches_per_step,
                              size_t num_threads) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, num_threads);
    engine.add_rule(make_growth_rule());
    engine.set_random_seed(seed);

    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve_uniform_random(initial, steps, matches_per_step);

    return RunMetrics{hg.num_canonical_states(), hg.num_events(),
                      hg.num_causal_edges(), hg.num_branchial_edges()};
}

// Run the default dataflow evolve() with an ExplorationProbability draw.
RunMetrics run_exploration(uint64_t seed, double probability, size_t steps,
                           size_t num_threads) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, num_threads);
    engine.add_rule(make_growth_rule());
    engine.set_random_seed(seed);
    engine.set_exploration_probability(probability);

    std::vector<std::vector<VertexId>> initial = {{0, 1}};
    engine.evolve(initial, steps);

    return RunMetrics{hg.num_canonical_states(), hg.num_events(),
                      hg.num_causal_edges(), hg.num_branchial_edges()};
}

}  // namespace

// evolve_uniform_random with the same seed and one thread must reproduce exactly.
TEST(SamplingReproducibility, UniformRandomSameSeedReproducible) {
    RunMetrics a = run_uniform_random(/*seed=*/12345, /*steps=*/6,
                                      /*matches_per_step=*/3, /*num_threads=*/1);
    RunMetrics b = run_uniform_random(/*seed=*/12345, /*steps=*/6,
                                      /*matches_per_step=*/3, /*num_threads=*/1);

    EXPECT_EQ(a.canonical_states, b.canonical_states);
    EXPECT_EQ(a.events, b.events);
    EXPECT_EQ(a.causal_edges, b.causal_edges);
    EXPECT_EQ(a.branchial_edges, b.branchial_edges);
    EXPECT_TRUE(a == b) << "Same-seed uniform-random runs must be identical";
}

// A different seed must still produce a valid, bounded run (it may coincidentally
// match on such a tiny graph, so we only require it stays bounded and non-empty).
TEST(SamplingReproducibility, UniformRandomDifferentSeedBounded) {
    RunMetrics r = run_uniform_random(/*seed=*/98765, /*steps=*/6,
                                      /*matches_per_step=*/3, /*num_threads=*/1);
    EXPECT_GE(r.canonical_states, 1u) << "Should retain at least the initial state";
    EXPECT_LT(r.canonical_states, 500u) << "Sampling must keep growth bounded";
}

// With a small matches_per_step, new states per step are capped, so total growth
// stays bounded over several steps (target_states == matches_per_step in the loop).
TEST(SamplingReproducibility, UniformRandomBounded) {
    const size_t steps = 5;
    const size_t matches_per_step = 4;
    RunMetrics r = run_uniform_random(/*seed=*/7, steps, matches_per_step,
                                      /*num_threads=*/1);

    // At most matches_per_step new states are accepted per step, plus the initial.
    EXPECT_GE(r.canonical_states, 1u);
    EXPECT_LE(r.canonical_states, matches_per_step * steps + 1)
        << "States per step must be bounded by matches_per_step";
}

// Part (a) regression: the ExplorationProbability draw is seeded from
// random_seed_, so the default dataflow path is reproducible single-threaded.
// Before the fix this used a thread_local random_device RNG and would flake.
TEST(SamplingReproducibility, ExplorationProbabilityReproducible) {
    RunMetrics a = run_exploration(/*seed=*/999, /*probability=*/0.5,
                                   /*steps=*/4, /*num_threads=*/1);
    RunMetrics b = run_exploration(/*seed=*/999, /*probability=*/0.5,
                                   /*steps=*/4, /*num_threads=*/1);

    EXPECT_EQ(a.canonical_states, b.canonical_states)
        << "ExplorationProbability draw must be deterministic for a fixed seed";
    EXPECT_EQ(a.events, b.events)
        << "ExplorationProbability draw must be deterministic for a fixed seed";
    EXPECT_EQ(a.causal_edges, b.causal_edges);
    EXPECT_EQ(a.branchial_edges, b.branchial_edges);
}

// Multi-threaded sampling is not required to be bit-reproducible (task scheduling
// perturbs which successor gets which draw), but it must not crash and must stay
// bounded under the exploration probability.
TEST(SamplingReproducibility, MultiThreadExplorationBounded) {
    RunMetrics r = run_exploration(/*seed=*/2024, /*probability=*/0.5,
                                   /*steps=*/4, /*num_threads=*/4);
    EXPECT_GE(r.canonical_states, 1u);
    EXPECT_LT(r.canonical_states, 2000u)
        << "Multi-threaded exploration must remain bounded";
}

// Unbiasedness: the whole point of reservoir sampling is a UNIFORM subsample,
// which the reproducibility/boundedness tests above do not check. Within a single
// (state, rule) stratum, evolve_uniform_random keeps k of M matches; each match
// must be selected with probability k/M. Initial state = M disconnected edges; the
// rule matches any single edge and appends a fresh edge to its second vertex, so
// each match extends a distinct component and the produced state names the chosen
// match (its appended edge is {odd-vertex, fresh}). Over many seeds each component
// should be chosen ~equally; a chi-square well above its d.o.f. would signal bias.
TEST(SamplingReproducibility, ReservoirUniformWithinStratum) {
    constexpr int M = 20;      // matches available
    constexpr int k = 5;       // reservoir size
    constexpr int R = 3000;    // seeds
    RewriteRule rule = make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();
    std::vector<std::vector<VertexId>> init;
    for (int i = 0; i < M; ++i)
        init.push_back({static_cast<VertexId>(2*i), static_cast<VertexId>(2*i + 1)});

    std::array<long, M> freq{};
    long total = 0;
    for (int seed = 1; seed <= R; ++seed) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::None);
        ParallelEvolutionEngine e(&hg, 1);
        e.set_random_seed(static_cast<uint64_t>(seed));
        e.add_rule(rule);
        e.evolve_uniform_random(init, 1, static_cast<size_t>(k));
        for (uint32_t s = 0; s < hg.num_states(); ++s) {
            if (hg.get_state(s).id == INVALID_ID || hg.get_state(s).step != 1) continue;
            hg.get_state(s).edges.for_each([&](EdgeId eid) {
                const auto& ed = hg.get_edge(eid);
                if (ed.arity == 2 && (ed.vertices[0] % 2 == 1)) {  // appended {odd, fresh}
                    int comp = (ed.vertices[0] - 1) / 2;
                    if (comp >= 0 && comp < M) { freq[comp]++; total++; }
                }
            });
        }
    }
    EXPECT_EQ(total, static_cast<long>(R) * k)
        << "reservoir must pick exactly k matches per step";
    const double expected = static_cast<double>(R) * k / M;
    double chisq = 0;
    for (int i = 0; i < M; ++i) {
        double d = freq[i] - expected;
        chisq += d * d / expected;
    }
    // df = M-1 = 19; chi-square ~ df under the null. 2x df is a generous bound
    // (p < ~0.001 of a false positive at this threshold with a correct sampler).
    EXPECT_LT(chisq, 2.0 * (M - 1))
        << "within-stratum reservoir selection is non-uniform; chi-square=" << chisq
        << " for df=" << (M - 1);
}

// exploration_probability=p must keep each CANONICAL state with probability p
// (node sampling), independent of how many transitions reach it. On a symmetric
// N-cycle, a pendant-adding rule makes all N single-edge rewrites produce one
// canonical child (in-degree N). Under quotient exploration, P(child explored)
// is p if the coin is flipped once per canonical state, or 1-(1-p)^N if it is
// flipped per transition (the bias this guards against). Explored means the
// child was expanded, i.e. some state exists at step 2.
TEST(SamplingReproducibility, ExplorationProbabilityIsPerCanonicalState) {
    constexpr int N = 6;
    constexpr int R = 400;
    RewriteRule rule = make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();
    std::vector<std::vector<VertexId>> cyc;
    for (int i = 0; i < N; ++i)
        cyc.push_back({static_cast<VertexId>(i), static_cast<VertexId>((i + 1) % N)});

    for (double p : {0.25, 0.5}) {
        int explored = 0;
        for (int seed = 1; seed <= R; ++seed) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine e(&hg, 1);
            e.set_explore_from_canonical_states_only(true);
            e.set_exploration_probability(p);
            e.set_random_seed(static_cast<uint64_t>(seed));
            e.add_rule(rule);
            e.evolve(cyc, 2);
            bool has_step2 = false;
            for (uint32_t s = 0; s < hg.num_states(); ++s) {
                if (hg.get_state(s).id != INVALID_ID && hg.get_state(s).step == 2) {
                    has_step2 = true; break;
                }
            }
            if (has_step2) ++explored;
        }
        double frac = static_cast<double>(explored) / R;
        // Per-state expectation p; per-transition would be ~0.82 (p=.25) / ~0.98
        // (p=.5). A 0.12 tolerance separates the two hypotheses comfortably.
        EXPECT_NEAR(frac, p, 0.12)
            << "exploration_probability is not per-canonical-state at p=" << p
            << " (observed " << frac << "); a per-transition coin would bias high";
    }
}
