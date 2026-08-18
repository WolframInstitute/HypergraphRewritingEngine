#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <mutex>
#include <set>
#include <unordered_map>

using namespace hypergraph;

// The sampling and pruning surface, which is otherwise reached only through the paclet FFI.
//
// Everything here runs on the ONE evolve(): sampling has no scheduler of its own, because a
// sample needs no barrier once it is a rate rather than a count (docs/ASYNC_SAMPLING_DESIGN.md).
//
// The contract these pin down is that a sampled run is a SUBGRAPH OF THE FULL ONE that can be
// reproduced: the same seed selects the same subgraph at any worker count, in either
// exploration mode. Reproducibility is not a convenience here -- a sample nobody can reproduce
// cannot be compared against the evolution it claims to represent.

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

// ExplorationProbability keeps the SAME states at any worker count.
//
// This used to assert only that a multi-threaded run stayed bounded, because the draw came
// from a per-worker RNG and which state survived depended on which worker got there. The draw
// is keyed on isomorphism-invariant identity now -- the class's canonical hash under quotient,
// and the canonical key of the creating transition under full capture, a raw state having no
// other invariant name -- so the surviving set is a property of the run and not of the
// schedule, and the test can say so.
TEST(SamplingReproducibility, ExplorationProbabilityKeepsTheSameStatesAtEveryWorkerCount) {
    for (bool quotient : {false, true}) {
        auto run = [&](size_t threads) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine e(&hg, threads);
            e.set_explore_from_canonical_states_only(quotient);
            e.set_exploration_probability(0.5);
            e.set_random_seed(2024);
            e.add_rule(make_growth_rule());
            e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}, {2u, 3u}}, 5);
            std::multiset<uint64_t> hashes;
            for (uint32_t s = 0; s < hg.num_published_states(); ++s) {
                if (hg.get_state(s).id == INVALID_ID) continue;
                hashes.insert(hg.get_or_compute_canonical_hash(s));
            }
            return hashes;
        };

        const auto one = run(1);
        ASSERT_GT(one.size(), 5u)
            << "quotient=" << quotient << ": too small a sample to compare";
        for (size_t threads : {size_t(2), size_t(4), size_t(8)}) {
            EXPECT_EQ(run(threads), one)
                << "quotient=" << quotient << ": the explored set differs at " << threads
                << " workers, so the coin is keyed on something the schedule decides";
        }
    }
}

// Evolution is step-ASYNCHRONOUS: no depth waits for another to finish.
//
// Asserted behaviourally rather than by counting synchronisation calls, because the absence of
// a barrier is what matters and a barrier can be spelled many ways. With no per-depth
// synchronisation a state at depth d+1 completes before some state at depth d has, so the order
// states drain in is NOT sorted by depth. Under a step-synchronised scheduler it necessarily
// would be, which is what makes this discriminating rather than merely descriptive.
TEST(SamplingReproducibility, DepthsOverlapBecauseNothingWaitsForAStep) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 4);
    e.add_rule(make_growth_rule());

    std::mutex m;
    std::vector<uint32_t> drain_order;
    e.set_on_state_matches_complete([&](StateId, uint32_t step) {
        std::lock_guard<std::mutex> lock(m);
        drain_order.push_back(step);
    });

    e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}, {2u, 3u}}, 4);

    ASSERT_GT(drain_order.size(), 10u) << "too few states drained to see an overlap";
    const bool depth_sorted = std::is_sorted(drain_order.begin(), drain_order.end());
    EXPECT_FALSE(depth_sorted)
        << "every state drained in depth order, which is what a step barrier produces; "
        << "with none, a deeper state finishes before a shallower one";
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
    struct Kept { size_t matches; size_t events; size_t draws; size_t survived; };
    auto run = [&](double q, uint64_t seed) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, 4);
        e.set_random_seed(seed);
        e.set_match_forwarding(true);      // the condition under test, stated not assumed
        e.set_transition_rate(q);
        e.add_rule(rule);
        e.evolve(init, 4);
        return Kept{e.total_matches(), hg.num_events(), e.draws_taken(), e.draws_survived()};
    };

    const Kept full = run(1.0, 1);
    ASSERT_GT(full.matches, 1000u) << "the unthinned run is too small to measure a rate against";
    ASSERT_GT(full.events, 0u);
    // q = 1 must be a no-op, stated as what that MEANS rather than as a ratio: no thinning
    // decision is taken at all, because transition_survives returns on the >= 1.0 fast path
    // before it ever counts a draw. A ratio here would be reading total_matches(), which counts
    // push-path bookkeeping and is not comparable across submission modes.
    EXPECT_EQ(full.draws, 0u)
        << "q=1 took " << full.draws << " thinning draws, so the identity rate is not a no-op";

    // Two assertions, because "the sampler works" is two claims and only one of them is a rate.
    //
    // MEASURE THE SAMPLER, NOT A PROXY. events/total_matches was the original metric and it is
    // NOT comparable across submission modes: total_matches() counts push-path work, and
    // push_match_to_children draws 112 times under eager against 89,523 under batched on this
    // very workload while producing byte-identical events. The ratio therefore reads ~q under
    // eager and ~q/2 under batched for a sampler that is exactly correct in both.
    // draws_survived/draws_taken is the rate itself and lands on q in either mode.
    //
    // NO DISPATCH MAY BYPASS THE SAMPLER. That is what the original metric was really guarding,
    // and it is the point of this test: a match reaches a state either by discovery or by
    // forwarding from an ancestor, forwarding dominates at depth, and a sampler that only saw
    // discoveries would bound nothing. Every event must be preceded by a SURVIVING draw, so
    // survivors can exceed events (one transition drawn at two sites agrees with itself and
    // rewrites once) but can never fall short. A bypassed dispatch shows up immediately as
    // events outrunning survivors.
    for (double q : {0.25, 0.5}) {
        size_t draws = 0, survived = 0, events = 0;
        for (uint64_t seed = 1; seed <= 12; ++seed) {
            const Kept k = run(q, seed);
            draws += k.draws;
            survived += k.survived;
            events += k.events;
        }
        ASSERT_GT(draws, 0u) << "no draw was taken at all, so the sampler never ran";

        const double rate = static_cast<double>(survived) / draws;
        EXPECT_NEAR(rate, q, 0.02)
            << "at q=" << q << " the sampler kept " << rate << " of the transitions it drew on";

        EXPECT_GE(survived, events)
            << "at q=" << q << " there were " << events << " events but only " << survived
            << " surviving draws, so some dispatch produced a rewrite without consulting the "
            << "sampler -- forwarding dominates at depth and a sampler that only sees "
            << "discoveries bounds nothing";
    }
}

// The sample must be the SAME SUBGRAPH at any worker count.
//
// This is the property that makes a sparse sample checkable at all: it is compared against the
// unpruned evolution it claims to represent, and a subgraph that differs every run has nothing
// to compare. It is also strictly stronger than same-seed reproducibility, and it is what the
// draw's key has to earn -- keyed on a raw state id it fails, because work-stealing assigns
// those in whatever order the workers arrived. The key is the isomorphism-invariant transition
// identity (input state's canonical hash, rule, canonical ranks of the consumed edges), which
// is the same object on any schedule and, being hgcommon's, on either device.
TEST(SamplingReproducibility, SampledSubgraphIsTheSameAtEveryWorkerCount) {
    RewriteRule rule = make_growth_rule();
    std::vector<std::vector<VertexId>> init;
    for (int i = 0; i < 12; ++i)
        init.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});

    // Compare the CANONICAL content, not the counts: two runs could agree on how many states
    // they kept while keeping different ones.
    auto run = [&](size_t threads) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, threads);
        e.set_random_seed(20260729);
        e.set_transition_rate(0.4);
        e.add_rule(rule);
        e.evolve(init, 4);
        std::multiset<uint64_t> hashes;
        for (uint32_t s = 0; s < hg.num_published_states(); ++s) {
            if (hg.get_state(s).id == INVALID_ID) continue;
            hashes.insert(hg.get_or_compute_canonical_hash(s));
        }
        return hashes;
    };

    const auto one = run(1);
    ASSERT_GT(one.size(), 20u) << "too small a sample for the comparison to mean anything";
    for (size_t threads : {size_t(2), size_t(4), size_t(8)}) {
        EXPECT_EQ(run(threads), one)
            << "the sampled subgraph differs at " << threads << " workers, so the draw is "
            << "keyed on something the schedule decides rather than on the transition";
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
            for (uint32_t s = 0; s < hg.num_published_states(); ++s) {
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

// A depth reports complete exactly once, in order, and only after every state at it has drained.
//
// The signal is derived from the per-state drain with no barrier, so the properties that matter
// are the ones a barrier would have given for free: fired once, fired in depth order, and never
// fired while a state at that depth was still matching. Run at several worker counts, because
// with one worker every task drains in submission order and none of it is tested.
TEST(SamplingReproducibility, DepthCompletesOnceAfterEveryStateAtItHasDrained) {
    for (unsigned threads : {1u, 4u, 16u}) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, threads);
        e.add_rule(make_growth_rule());

        std::mutex m;
        std::vector<uint32_t> depth_order;
        std::unordered_map<uint32_t, int> depth_fires;
        std::unordered_map<uint32_t, size_t> drained_at_depth;
        std::unordered_map<uint32_t, size_t> drained_when_depth_fired;
        std::unordered_map<uint32_t, size_t> arrived_at_depth;

        e.set_on_state_matches_complete([&](StateId, uint32_t step) {
            std::lock_guard<std::mutex> lock(m);
            drained_at_depth[step]++;
        });
        e.set_on_depth_complete([&](uint32_t depth) {
            std::lock_guard<std::mutex> lock(m);
            depth_order.push_back(depth);
            depth_fires[depth]++;
            drained_when_depth_fired[depth] = drained_at_depth[depth];
        });
        ASSERT_TRUE(e.depth_signal_available())
            << "full capture must offer the depth signal, or this test asserts nothing";

        e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}, {2u, 3u}}, 4);

        // Every state that drained belongs to some depth; a depth that fired must have seen all
        // of its own drains BEFORE it fired, which is the whole content of the signal.
        {
            std::lock_guard<std::mutex> lock(m);
            arrived_at_depth = drained_at_depth;   // final tally, after the run
        }

        ASSERT_FALSE(depth_order.empty())
            << "no depth completed at " << threads << " threads, so nothing was tested";
        EXPECT_EQ(e.depth_late_arrivals(), 0u)
            << e.depth_late_arrivals() << " states arrived at a depth already reported complete, "
            << "at " << threads << " threads";

        for (const auto& [depth, fires] : depth_fires) {
            EXPECT_EQ(fires, 1) << "depth " << depth << " fired " << fires << " times at "
                                << threads << " threads";
            EXPECT_EQ(drained_when_depth_fired[depth], arrived_at_depth[depth])
                << "depth " << depth << " fired with " << drained_when_depth_fired[depth]
                << " of its " << arrived_at_depth[depth] << " states drained, at " << threads
                << " threads";
        }

        for (size_t i = 1; i < depth_order.size(); ++i) {
            EXPECT_LT(depth_order[i - 1], depth_order[i])
                << "depths completed out of order at " << threads << " threads: "
                << depth_order[i - 1] << " then " << depth_order[i];
        }
    }
}

// Under quotient exploration the signal is refused rather than made to lie: a child is
// submitted at its parent's LIVE MINIMUM depth plus one, so a relaxation can put an arrival at
// a shallow depth from a task running at a deep one, after that depth's predecessor settled.
TEST(SamplingReproducibility, DepthSignalIsRefusedUnderQuotientExploration) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 4);
    e.set_explore_from_canonical_states_only(true);
    e.add_rule(make_growth_rule());

    std::atomic<int> fires{0};
    e.set_on_depth_complete([&](uint32_t) { fires.fetch_add(1); });
    EXPECT_FALSE(e.depth_signal_available())
        << "quotient exploration cannot support the depth signal; see set_on_depth_complete";

    e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}, {2u, 3u}}, 4);
    EXPECT_EQ(fires.load(), 0)
        << "the depth signal fired under quotient, where its arrival invariant does not hold";
}

// ---------------------------------------------------------------------------------------
// RuleWeights: per-rule multipliers on TransitionRate.
//
// The two knobs COMPOSE rather than one overriding the other, so the rate a rule's transitions
// are drawn at is transition_rate x weight[rule]. That composition is the whole reason the
// draw sites had to stop testing `transition_rate_ < 1.0` directly: a caller who leaves the
// rate at 1 and weights one rule to zero is sampling, and that test said they were not.
namespace {

// Two rules that both apply to the seed, so weighting one is observable in the state set.
size_t run_weighted(const std::vector<double>& weights, uint64_t seed, size_t steps) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 1);
    e.set_random_seed(seed);
    e.add_rule(make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build());
    e.add_rule(make_rule(1).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).rhs({2, 3}).build());
    e.set_rule_weights(weights);
    const std::vector<std::vector<std::vector<VertexId>>> init = {{{0, 1}, {1, 2}}};
    e.evolve(init, steps);
    return hg.num_canonical_states();
}

}  // namespace

TEST(RuleWeights, AZeroWeightRemovesThatRuleAndTheOtherStillRuns) {
    // Weighting rule 1 to zero must leave rule 0's transitions untouched, which is what
    // distinguishes a per-rule weight from simply lowering the global rate.
    const size_t both  = run_weighted({}, 4242, 3);
    const size_t only0 = run_weighted({1.0, 0.0}, 4242, 3);
    const size_t only1 = run_weighted({0.0, 1.0}, 4242, 3);

    EXPECT_GT(both, 1u) << "the unweighted run explored nothing, so the comparison is vacuous";
    EXPECT_GT(only0, 1u) << "rule 0 was dropped even though its weight is 1";
    EXPECT_GT(only1, 1u) << "rule 1 was dropped even though its weight is 1";
    EXPECT_LT(only0, both) << "dropping rule 1 did not remove anything, so the weight did "
                              "nothing";
    EXPECT_LT(only1, both) << "dropping rule 0 did not remove anything, so the weight did "
                              "nothing";
}

TEST(RuleWeights, WeightsAreReproducibleForAFixedSeed) {
    EXPECT_EQ(run_weighted({1.0, 0.25}, 777, 4), run_weighted({1.0, 0.25}, 777, 4))
        << "a weighted draw must be a pure function of the transition and the seed";
}

TEST(RuleWeights, AWeightOfOneEverywhereChangesNothing) {
    // The knob must be inert at its default, or every unweighted run silently changes meaning.
    EXPECT_EQ(run_weighted({}, 31337, 4), run_weighted({1.0, 1.0}, 31337, 4));
    // And a SHORT vector is a partial override: rule 1 is unmentioned and takes weight 1.
    EXPECT_EQ(run_weighted({}, 31337, 4), run_weighted({1.0}, 31337, 4));
}
