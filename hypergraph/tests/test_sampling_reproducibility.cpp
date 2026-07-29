#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"

#include <array>
#include <cstddef>
#include <mutex>
#include <unordered_map>

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

// TransitionRate thins the multiway graph and must keep doing so AT DEPTH, with match
// forwarding on. That last clause is the whole test: a match reaches a state either by
// discovery in its own SCAN/EXPAND tree or by forwarding from an ancestor, forwarding
// dominates in a deep run, and a sampler that only sees discoveries bounds nothing. The
// per-state reservoir failed exactly here -- 2,038,505 states instead of 1,365 on this shape --
// while passing a one-step uniformity test, so depth with forwarding on is the discriminating
// case and not an extra one.
TEST(SamplingReproducibility, TransitionRateThinsAtDepthWithForwardingOn) {
    RewriteRule rule = make_growth_rule();
    std::vector<std::vector<VertexId>> init;
    for (int i = 0; i < 24; ++i)
        init.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});

    // Measure the KEPT FRACTION rather than a size. A size can collapse to the root by chance
    // when the root has few matches, which says nothing about whether the rate is reaching
    // every acceptance point; events/matches is the rate itself. If forwarding bypassed the
    // sampler the ratio would sit far above q, since forwarded matches dominate at depth.
    struct Kept { size_t matches; size_t events; };
    auto run = [&](double q, uint64_t seed) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, 4);
        e.set_random_seed(seed);
        e.set_match_forwarding(true);      // the condition under test, stated not assumed
        e.set_transition_rate(q);
        e.add_rule(rule);
        e.evolve(init, 4);
        return Kept{e.total_matches(), hg.num_events()};
    };

    const Kept full = run(1.0, 1);
    ASSERT_GT(full.matches, 1000u) << "the unthinned run is too small to measure a rate against";
    EXPECT_NEAR(static_cast<double>(full.events) / full.matches, 1.0, 0.02)
        << "q=1 dropped transitions, so the sampler is not a no-op at its identity";

    // Several seeds: one seed's root can die by chance, an average over seeds cannot.
    for (double q : {0.25, 0.5}) {
        size_t matches = 0, events = 0;
        for (uint64_t seed = 1; seed <= 12; ++seed) {
            const Kept k = run(q, seed);
            matches += k.matches;
            events += k.events;
        }
        ASSERT_GT(matches, 0u);
        const double kept = static_cast<double>(events) / matches;
        EXPECT_NEAR(kept, q, 0.05)
            << "at q=" << q << " the run kept " << kept << " of its transitions; a rate that "
            << "misses the forwarding dispatches reads high, one applied twice reads low";
    }
}

// Thinning must be a property of the TRANSITION, not of the worker that happened to reach it.
// Drawing from a per-thread RNG would make the surviving subgraph depend on the schedule, and
// a sample nobody can reproduce cannot be checked against the full evolution it claims to
// represent. Same seed and same thread count must give the same graph, every time.
TEST(SamplingReproducibility, TransitionRateIsReproducibleForAGivenSeed) {
    RewriteRule rule = make_growth_rule();
    // Wide enough that the root survives thinning: a run that dies at the root is trivially
    // reproducible and would prove nothing.
    std::vector<std::vector<VertexId>> init;
    for (int i = 0; i < 12; ++i)
        init.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});

    auto run = [&]() {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, 1);
        e.set_random_seed(777);
        e.set_transition_rate(0.35);
        e.add_rule(rule);
        e.evolve(init, 4);
        return std::make_pair(hg.num_states(), hg.num_events());
    };

    const auto first = run();
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(run(), first) << "the sampled subgraph changed between identical runs";
    }
    EXPECT_GT(first.first, 1u) << "the run produced nothing, so equality is vacuous";
}

// MatchesPerState is a UNIFORM subsample of one state's matches, taken with no barrier
// anywhere. Same measurement as the strided-reservoir gate above, on the path that replaces it:
// M available, k kept, each chosen with probability k/M. Initial state = M disconnected edges;
// the rule matches any single edge and appends a fresh edge to its second vertex, so each match
// extends a distinct component and the produced state names the chosen match.
//
// Run at several thread counts. Algorithm R is correct for any input ORDER, and a slot here is
// won by the highest stream position rather than by whoever stores last, so the distribution
// must not move when the schedule does. A sampler that resolved slots by store order would
// pass at 1 thread and drift at 8.
TEST(SamplingReproducibility, MatchesPerStateIsUniformOverThatStatesMatches) {
    constexpr int M = 20;      // matches available
    constexpr int k = 5;       // reservoir size
    constexpr int R = 3000;    // seeds
    RewriteRule rule = make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();
    std::vector<std::vector<VertexId>> init;
    for (int i = 0; i < M; ++i)
        init.push_back({static_cast<VertexId>(2*i), static_cast<VertexId>(2*i + 1)});

    for (size_t threads : {size_t(1), size_t(8)}) {
        std::array<long, M> freq{};
        long total = 0;
        for (int seed = 1; seed <= R; ++seed) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::None);
            ParallelEvolutionEngine e(&hg, threads);
            e.set_random_seed(static_cast<uint64_t>(seed));
            e.set_matches_per_state(k);
            e.add_rule(rule);
            e.evolve(init, 1);
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
            << "the reservoir kept a different number than k per state at "
            << threads << " threads";
        const double expected = static_cast<double>(R) * k / M;
        double chisq = 0;
        for (int i = 0; i < M; ++i) {
            double d = freq[i] - expected;
            chisq += d * d / expected;
        }
        EXPECT_LT(chisq, 2.0 * (M - 1))
            << "per-state reservoir selection is non-uniform at " << threads
            << " threads; chi-square=" << chisq << " for df=" << (M - 1);
    }
}

// The per-state match join: a state's drain must fire exactly once, and strictly after that
// state's last match. It is what lets anything be keyed on "the matches of one state" AS A SET
// -- a reservoir first of all -- without a step barrier (docs/ASYNC_SAMPLING_DESIGN.md §5).
//
// Both halves are load-bearing and fail differently. Firing twice would finalise a population
// twice and rewrite the second sample on top of the first. Firing early would finalise over
// part of the population, which is invisible in the output -- the run still looks complete,
// it just explored a subset -- so it is checked here rather than left to be noticed.
//
// Run multi-threaded: with one worker every task drains in submission order and the ordering
// invariants are never tested.
TEST(SamplingReproducibility, StateMatchJoinDrainsOncePerStateAfterTheLastMatch) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 4);
    e.add_rule(make_growth_rule());

    std::mutex m;
    std::unordered_map<StateId, int> drains;
    std::unordered_map<StateId, size_t> matches_at_drain;
    e.set_on_state_matches_complete([&](StateId s, uint32_t) {
        std::lock_guard<std::mutex> lock(m);
        drains[s]++;
        matches_at_drain[s] = e.matches_found_for_state(s);
    });

    e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}, {2u, 3u}}, 4);

    ASSERT_FALSE(drains.empty()) << "no state drained, so the join was never exercised";
    EXPECT_EQ(e.states_drained(), drains.size())
        << "the engine counted a different number of drains than the callback saw";

    for (const auto& [state, count] : drains) {
        EXPECT_EQ(count, 1) << "state " << state << " drained " << count << " times";
        EXPECT_EQ(matches_at_drain[state], e.matches_found_for_state(state))
            << "state " << state << " drained with " << matches_at_drain[state]
            << " matches but finished with " << e.matches_found_for_state(state)
            << ", so the drain fired before its match tree had finished";
    }
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
