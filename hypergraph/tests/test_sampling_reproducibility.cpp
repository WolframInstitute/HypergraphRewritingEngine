#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"
#include "hgcommon/state_cost_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>

using namespace hypergraph;

// The sampling and pruning surface, which is otherwise reached only through the paclet FFI.
//
// Everything here runs on the ONE evolve(): sampling has no scheduler of its own, because a
// sample needs no barrier once it is a rate rather than a count (docs/ARCHITECTURE.md, Sampling).
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

// TransitionRate thins the multiway graph at depth, and every event is either a surviving draw
// or a spine-forced transition. A per-state reservoir sampler passed a one-step uniformity test
// and kept 2,038,505 states instead of 1,365 on this shape, so the run goes to depth 4.
TEST(SamplingReproducibility, TransitionRateThinsAtDepth) {
    RewriteRule rule = make_growth_rule();
    std::vector<std::vector<VertexId>> init;
    for (int i = 0; i < 24; ++i)
        init.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});

    struct Kept { size_t matches; size_t events; size_t draws; size_t survived; size_t forced; };
    auto run = [&](double q, uint64_t seed) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, 4);
        e.set_random_seed(seed);
        e.set_transition_rate(q);
        e.add_rule(rule);
        e.evolve(init, 4);
        return Kept{e.total_matches(), hg.num_events(), e.draws_taken(), e.draws_survived(),
                    e.stats().total().spine_forced};
    };

    const Kept full = run(1.0, 1);
    ASSERT_GT(full.matches, 1000u) << "the unthinned run is too small to measure a rate against";
    ASSERT_GT(full.events, 0u);
    // At q = 1 transition_survives returns before counting a draw.
    EXPECT_EQ(full.draws, 0u)
        << "q=1 took " << full.draws << " thinning draws, so the identity rate is not a no-op";

    // draws_survived / draws_taken is the rate. A transition drawn at two sites rewrites once,
    // so survivors plus forced transitions may exceed events but never fall short of them.
    for (double q : {0.25, 0.5}) {
        size_t draws = 0, survived = 0, forced = 0, events = 0;
        for (uint64_t seed = 1; seed <= 12; ++seed) {
            const Kept k = run(q, seed);
            draws += k.draws;
            survived += k.survived;
            forced += k.forced;
            events += k.events;
        }
        ASSERT_GT(draws, 0u) << "no draw was taken at all, so the sampler never ran";

        const double rate = static_cast<double>(survived) / draws;
        EXPECT_NEAR(rate, q, 0.02)
            << "at q=" << q << " the sampler kept " << rate << " of the transitions it drew on";

        EXPECT_GE(survived + forced, events)
            << "at q=" << q << " there were " << events << " events but only " << survived
            << " surviving draws and " << forced << " spine-forced transitions, so a rewrite "
            << "was dispatched without a draw";
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

// THE DRAIN CAP KEEPS THE SAME STATES AT EVERY WORKER COUNT, with forwarding requested.
//
// cap_at_drain chooses k of a state's matches per rule by seeded rank, at the state's drain --
// the point its own matching completes. A match forwarded from an ancestor arrives after that
// point, so a run under the cap turns forwarding off and says so to a caller that asked for it.
// Unlike MaxStatesPerStep, this cap IS documented as reproducible, so if it varies with the
// worker count that is a defect and not a design choice. Forwarding is requested rather than
// left to the engine, which would leave it off for the rule's one-edge left-hand side.
TEST(SamplingReproducibility, TheDrainCapKeepsTheSameStatesAtEveryWorkerCount) {
    auto run = [](size_t threads) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, threads);
        e.set_random_seed(2024);
        e.set_match_forwarding(true);
        e.set_matches_per_state_rule(2);
        e.add_rule(make_growth_rule());
        std::vector<std::vector<VertexId>> init;
        for (int i = 0; i < 8; ++i)
            init.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
        e.evolve(init, 4);
        std::set<uint64_t> hashes;
        for (StateId sid = 0; sid < hg.num_states(); ++sid)
            hashes.insert(hg.get_or_compute_canonical_hash(sid));
        return hashes;
    };

    const std::set<uint64_t> one = run(1);
    ASSERT_FALSE(one.empty());
    for (size_t t : {2u, 4u, 8u}) {
        EXPECT_EQ(run(t), one)
            << "the drain cap kept a different set of states at " << t << " workers than at one";
    }

    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 2);
    e.set_match_forwarding(true);
    e.set_matches_per_state_rule(2);
    e.add_rule(make_growth_rule());
    e.evolve({{0, 1}}, 2);
    bool told = false;
    for (const std::string& w : e.warnings())
        told = told || w.find("turns forwarding off") != std::string::npos;
    EXPECT_TRUE(told) << "forwarding was requested with the cap, and the run did not say it turned "
                         "forwarding off";
}

// The per-state match join: a state's drain must fire exactly once, and strictly after that
// state's last match. It is what lets anything be keyed on "the matches of one state" AS A SET
// -- a reservoir first of all -- without a step barrier (docs/ARCHITECTURE.md, Sampling).
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

// The depth join's work and storage follow the depth a run reaches, not its step budget. The
// rule {{x,y}} -> {{y}} stops after one step; with a budget of 2,000,000 steps only the depths
// it reached are reported, under the caller's hook and under MaxStatesPerStep.
TEST(SamplingReproducibility, DepthJoinFollowsTheDepthReached) {
    for (const bool per_step_cap : {false, true}) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, 4);
        e.add_rule(make_rule(0).lhs({0, 1}).rhs({1}).build());
        if (per_step_cap) e.set_max_states_per_step(1);
        std::atomic<uint32_t> fires{0};
        e.set_on_depth_complete([&](uint32_t) { fires.fetch_add(1); });
        e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}}, 2000000);
        EXPECT_EQ(hg.num_states(), 2u);
        EXPECT_LE(fires.load(), 3u) << "per_step_cap=" << per_step_cap;
        EXPECT_EQ(e.depth_late_arrivals(), 0u);
        EXPECT_LE(e.depth_join_bytes(), 64u * 1024u) << "per_step_cap=" << per_step_cap;
    }
}

// The quotient reconstruction's per-match work follows the depth reached, not the step budget:
// a captured match is offered to the depths that hold an instance or a mass point. The rule
// {{x,y}} -> {{y}} stops after one step; the budget is 1,000,000,000 steps.
TEST(SamplingReproducibility, QuotientReplayFollowsTheDepthReached) {
    for (const bool quotient : {false, true}) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        hg.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
        ParallelEvolutionEngine e(&hg, 2);
        e.set_explore_from_canonical_states_only(quotient);
        e.add_rule(make_rule(0).lhs({0, 1}).rhs({1}).build());
        const auto t0 = std::chrono::steady_clock::now();
        e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}}, 1000000000);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        EXPECT_EQ(e.last_error(), job_system::ErrorType::None);
        EXPECT_LT(s, 1.0) << "quotient=" << quotient;
    }
}

namespace {
// An isomorphism-invariant fingerprint of the causal graph over full-capture events: each event
// is labelled (rule, canonical input hash, canonical output hash), then by its label with the
// sorted labels of its causal predecessors and successors; the result is the sorted list.
std::vector<uint64_t> causal_fingerprint(Hypergraph& hg) {
    std::map<EventId, uint64_t> label;
    std::map<EventId, std::vector<uint64_t>> pred, succ;
    auto label_of = [&](EventId e) {
        auto it = label.find(e);
        if (it != label.end()) return it->second;
        const Event& ev = hg.get_event(e);
        uint64_t h = hgcommon::fnv_hash(hgcommon::FNV_OFFSET, ev.rule_index);
        h = hgcommon::fnv_hash(h, ev.input_state == INVALID_ID ? 0 : hg.get_or_compute_canonical_hash(ev.input_state));
        h = hgcommon::fnv_hash(h, hg.get_or_compute_canonical_hash(ev.output_state));
        label[e] = h;
        return h;
    };
    hg.causal_graph().for_each_causal_edge([&](const CausalEdge& c) {
        pred[c.consumer].push_back(label_of(c.producer));
        succ[c.producer].push_back(label_of(c.consumer));
    });
    std::vector<uint64_t> out;
    for (uint32_t e = 0; e < hg.num_published_events(); ++e) {
        if (hg.get_event(e).id == INVALID_ID) continue;
        uint64_t h = label_of(e);
        auto& p = pred[e]; auto& s = succ[e];
        std::sort(p.begin(), p.end()); std::sort(s.begin(), s.end());
        for (uint64_t x : p) h = hgcommon::fnv_hash(h, x);
        h = hgcommon::fnv_hash(h, 0xFFu);
        for (uint64_t x : s) h = hgcommon::fnv_hash(h, x);
        out.push_back(h);
    }
    std::sort(out.begin(), out.end());
    return out;
}
}  // namespace

// MaxStatesPerStep chooses among transitions that tie on (rank, key) -- automorphic transitions
// of one state -- by a rule that does not read the schedule: the causal graph is the same, up
// to isomorphism, in every run. Rules {{4}} -> {{5}} and {{2}} -> {} from {{2},{2}}, 3 steps,
// cap 12, 16 threads.
TEST(SamplingReproducibility, StepCapTiesDoNotDependOnTheSchedule) {
    std::set<std::vector<uint64_t>> seen;
    for (int rep = 0; rep < 25; ++rep) {
        Hypergraph hg;
        ParallelEvolutionEngine e(&hg, 16);
        e.add_rule(make_rule(0).lhs({0}).rhs({1}).build());
        e.add_rule(make_rule(1).lhs({0}).build());
        e.set_max_states_per_step(12);
        e.evolve(std::vector<std::vector<VertexId>>{{0u}, {0u}}, 3);
        seen.insert(causal_fingerprint(hg));
    }
    EXPECT_EQ(seen.size(), 1u);
}

// Under MaxStatesPerStep N a state offers its step at most its N lowest-ranked transitions plus
// the ones tied with the N-th. Rule {{x,y},{y,z}} -> {{x,z},{y,z},{z,w}} has 4 matches on
// {{0,1},{1,2},{2,3},{3,4},{1,5}}, whose automorphism group is trivial, so no two transitions
// tie: cap 2 offers 2 candidates and keeps 2 transitions, at 1 and 16 threads.
TEST(SamplingReproducibility, StepCapOffersOnlyTheStatesLowestRankedTransitions) {
    for (unsigned threads : {1u, 16u}) {
        Hypergraph hg;
        ParallelEvolutionEngine e(&hg, threads);
        e.add_rule(make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).rhs({1, 2}).rhs({2, 3}).build());
        e.set_max_states_per_step(2);
        e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 4u}, {1u, 5u}}, 1);
        EXPECT_EQ(e.step_candidates_offered(), 2u) << threads << " threads";
        EXPECT_EQ(hg.num_events(), 2u) << threads << " threads";
    }
}

// Under MaxStatesPerStep a step's match storage is deleted when the next step settles, so at
// most two steps' matches exist at once, and the result equals a continuable run's, which keeps
// every match. Rule {{x,y},{y,z}} -> {{x,z},{y,z},{z,w}} from {{0,1},{1,2},{2,3},{3,4},{1,5}},
// cap 3, 8 steps, at 1 and 16 threads.
TEST(SamplingReproducibility, StepCapMatchStorageIsReleasedStepByStep) {
    auto run = [](unsigned threads, bool continuable, Hypergraph& hg) {
        auto e = std::make_unique<ParallelEvolutionEngine>(&hg, threads);
        e->add_rule(make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).rhs({1, 2}).rhs({2, 3}).build());
        e->set_max_states_per_step(3);
        e->set_continuable(continuable);
        e->evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 4u}, {1u, 5u}}, 8);
        return e;
    };
    for (unsigned threads : {1u, 16u}) {
        Hypergraph hg_kept, hg_released;
        auto kept = run(threads, true, hg_kept);
        auto released = run(threads, false, hg_released);
        EXPECT_EQ(kept->match_epochs_created(), 0u) << threads << " threads";
        EXPECT_EQ(released->match_epochs_created(), 8u) << threads << " threads";
        EXPECT_GE(released->match_epochs_retired(), 6u) << threads << " threads";
        EXPECT_LE(released->match_epochs_live_max(), 2u) << threads << " threads";
        EXPECT_EQ(hg_released.num_states(), hg_kept.num_states()) << threads << " threads";
        EXPECT_EQ(hg_released.num_events(), hg_kept.num_events()) << threads << " threads";
        EXPECT_EQ(hg_released.observable_num_causal_edges(), hg_kept.observable_num_causal_edges())
            << threads << " threads";
        EXPECT_EQ(causal_fingerprint(hg_released), causal_fingerprint(hg_kept))
            << threads << " threads";
    }
}

// The per-state caps and MaxStatesPerStep under the reconstruction route (Full states,
// Automatic events): the causal count is the same at every thread count. Rule
// {{4,1}} -> {{5,6},{4,1}} from {{5,4},{1,4}}, 2 steps.
TEST(SamplingReproducibility, CapTiesUnderTheReconstructionDoNotDependOnTheSchedule) {
    enum Cap { PER_RULE, PER_PARENT, PER_STEP };
    for (Cap cap : {PER_RULE, PER_PARENT, PER_STEP}) {
        std::set<std::array<uint64_t, 3>> seen;
        for (unsigned threads : {1u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u}) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            hg.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
            ParallelEvolutionEngine e(&hg, threads);
            e.add_rule(make_rule(0).lhs({0, 1}).rhs({2, 3}).rhs({0, 1}).build());
            if (cap == PER_RULE) e.set_matches_per_state_rule(2);
            if (cap == PER_PARENT) e.set_max_successor_states_per_parent(2);
            if (cap == PER_STEP) e.set_max_states_per_step(3);
            e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {2u, 1u}}, 2);
            seen.insert({hg.num_canonical_states(), hg.observable_num_events(),
                         hg.observable_num_causal_edges()});
        }
        std::string got;
        for (const auto& s : seen)
            got += " (" + std::to_string(s[0]) + "," + std::to_string(s[1]) + "," +
                   std::to_string(s[2]) + ")";
        EXPECT_EQ(seen.size(), 1u) << "cap " << static_cast<int>(cap) << ":" << got;
    }
}

// A sampled quotient run whose draws fail at every state, so the spine carries each step, from a
// state with automorphic transitions (fuzz seed 201768). The spine keeps every transition at the
// minimum rank, and automorphic transitions share it; full capture gives 2400 causal edges.
TEST(SamplingReproducibility, SpineTieUnderTheReconstructionMatchesFullCapture) {
    auto run = [](size_t threads, bool quotient) {
        Hypergraph hg;
        hg.set_event_signature_keys(hgcommon::EventKey_Step);
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, threads);
        e.set_max_steps(5);
        e.set_transitive_reduction(true);
        e.set_transition_rate(0.317);
        e.set_rule_weights({0.43, 0.24});
        e.set_random_seed(1855184222);
        e.set_explore_from_canonical_states_only(quotient);
        e.add_rule(make_rule(0).lhs({4, 3, 5, 2}).rhs({3, 5}).rhs({5, 4}).rhs({4, 4}).build());
        e.add_rule(make_rule(1).lhs({2, 4}).lhs({4, 2}).rhs({4}).rhs({4, 4}).rhs({2, 2}).build());
        e.evolve(std::vector<std::vector<VertexId>>{{0u, 1u}, {1u, 1u}, {2u, 1u, 1u, 0u},
                                                    {1u, 0u, 1u, 1u}},
                 5);
        return std::vector<size_t>{hg.num_canonical_states(), hg.observable_num_events(),
                                   hg.observable_num_causal_pairs(true),
                                   hg.observable_num_causal_pairs(false)};
    };
    // Full capture with the reduction on stores the reduced pairs only, so its unreduced count is
    // the reduced one; the quotient's unreduced count is compared against its own first run.
    auto one = run(1, false);
    ASSERT_EQ(one, (std::vector<size_t>{6, 5, 2400, 2400})) << "full capture changed";
    one[3] = run(1, true)[3];
    for (size_t threads : {size_t(1), size_t(2), size_t(4), size_t(8)}) {
        for (int rep = 0; rep < 20; ++rep) {
            const auto got = run(threads, true);
            ASSERT_EQ(got, one) << threads << " workers, repetition " << rep << ": states "
                                << got[0] << " events " << got[1] << " causal " << got[2]
                                << " unreduced " << got[3] << " vs full capture " << one[0]
                                << " " << one[1] << " " << one[2] << " " << one[3];
        }
    }
}

// Every cap and every sampling draw keeps the same transitions under quotient exploration as
// under full capture (SPEC.md §5): states, events, causal and branchial counts are equal, at 1
// and 4 workers. Fuzz seed 201768 and two rules with symmetric states, each with
// MatchesPerStateRule, MaxSuccessorStatesPerParent and MaxStatesPerStep at 1 and 2, and with
// TransitionRate and RuleWeights at three seeds.
TEST(SamplingReproducibility, CapsAndDrawsUnderQuotientExplorationMatchFullCapture) {
    struct Case {
        const char* name;
        std::vector<RewriteRule> rules;
        std::vector<std::vector<VertexId>> init;
        size_t steps;
    };
    const std::vector<Case> cases = {
        {"fuzz201768",
         {make_rule(0).lhs({4, 3, 5, 2}).rhs({3, 5}).rhs({5, 4}).rhs({4, 4}).build(),
          make_rule(1).lhs({2, 4}).lhs({4, 2}).rhs({4}).rhs({4, 4}).rhs({2, 2}).build()},
         {{0u, 1u}, {1u, 1u}, {2u, 1u, 1u, 0u}, {1u, 0u, 1u, 1u}},
         5},
        {"pair_grow",
         {make_rule(0).lhs({0, 1}).rhs({2, 3}).rhs({0, 1}).build()},
         {{0u, 1u}, {2u, 1u}},
         3},
        {"wolfram",
         {make_rule(0).lhs({0, 1}).lhs({0, 2}).rhs({0, 1}).rhs({0, 3}).rhs({1, 3}).rhs({2, 3})
              .build()},
         {{0u, 1u}, {0u, 2u}},
         3},
    };
    enum Knob { PER_RULE, PER_PARENT, PER_STEP, RATE, WEIGHTS };
    struct Setting { Knob knob; size_t k; uint64_t seed; };
    std::vector<Setting> settings;
    for (Knob kn : {PER_RULE, PER_PARENT, PER_STEP})
        for (size_t k : {size_t(1), size_t(2)}) settings.push_back({kn, k, 0});
    for (Knob kn : {RATE, WEIGHTS})
        for (uint64_t seed : {uint64_t(1855184222), uint64_t(7), uint64_t(424242)})
            settings.push_back({kn, 0, seed});

    auto run = [](const Case& c, const Setting& s, size_t threads, bool quotient) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, threads);
        e.set_max_steps(c.steps);
        e.set_explore_from_canonical_states_only(quotient);
        for (const auto& r : c.rules) e.add_rule(r);
        if (s.knob == PER_RULE) e.set_matches_per_state_rule(s.k);
        if (s.knob == PER_PARENT) e.set_max_successor_states_per_parent(s.k);
        if (s.knob == PER_STEP) e.set_max_states_per_step(s.k);
        if (s.knob == RATE || s.knob == WEIGHTS) {
            e.set_transition_rate(s.knob == RATE ? 0.5 : 0.8);
            if (s.knob == WEIGHTS) e.set_rule_weights({0.43, 0.6});
            e.set_random_seed(s.seed);
        }
        e.evolve(c.init, c.steps);
        return std::array<uint64_t, 4>{hg.num_canonical_states(), hg.observable_num_events(),
                                       hg.observable_num_causal_edges(),
                                       hg.observable_num_branchial()};
    };
    auto str = [](const std::array<uint64_t, 4>& a) {
        return std::to_string(a[0]) + "," + std::to_string(a[1]) + "," + std::to_string(a[2]) +
               "," + std::to_string(a[3]);
    };
    for (const auto& c : cases) {
        for (const auto& s : settings) {
            const auto full = run(c, s, 1, false);
            for (size_t threads : {size_t(1), size_t(4)}) {
                const auto q = run(c, s, threads, true);
                EXPECT_EQ(q, full) << c.name << " knob " << static_cast<int>(s.knob) << " k "
                                   << s.k << " seed " << s.seed << " threads " << threads
                                   << ": quotient " << str(q) << " full capture " << str(full);
            }
        }
    }
}

namespace {
struct SuccessorCapRun {
    size_t states, events;
    // The most distinct successor states one parent's events reach, and one step's.
    size_t most_per_parent, most_per_step;
    uint64_t causal, branchial;
};

SuccessorCapRun run_successor_cap(StateCanonicalizationMode mode,
                                  const std::vector<RewriteRule>& rules,
                                  const std::vector<std::vector<VertexId>>& init, size_t steps,
                                  size_t per_parent, size_t per_step, size_t threads,
                                  uint64_t seed) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(mode);
    ParallelEvolutionEngine e(&hg, threads);
    for (const auto& r : rules) e.add_rule(r);
    if (per_parent) e.set_max_successor_states_per_parent(per_parent);
    if (per_step) e.set_max_states_per_step(per_step);
    e.set_random_seed(seed);
    e.evolve(init, steps);
    std::map<StateId, std::set<StateId>> by_parent;
    std::map<uint32_t, std::set<StateId>> by_step;
    for (uint32_t ev = 0; ev < hg.num_published_events(); ++ev) {
        const Event& x = hg.get_event(ev);
        if (x.id == INVALID_ID || hg.is_genesis_event(ev)) continue;
        const StateId out = hg.get_canonical_state(x.output_state);
        by_parent[x.input_state].insert(out);
        by_step[hg.get_state(x.output_state).step].insert(out);
    }
    SuccessorCapRun r{hg.num_canonical_states(), hg.num_events(), 0, 0,
                      hg.observable_num_causal_edges(), hg.observable_num_branchial()};
    for (const auto& p : by_parent) r.most_per_parent = std::max(r.most_per_parent, p.second.size());
    for (const auto& p : by_step) r.most_per_step = std::max(r.most_per_step, p.second.size());
    return r;
}
}  // namespace

// MaxSuccessorStatesPerParent and MaxStatesPerStep count distinct successor STATES under the
// state identity in force, and keep every transition into a kept successor. Hand-counted cases:
//  - {{x,y}} -> {} from the path {{1,2},{2,3},{3,4}}: deleting the first or the last edge makes
//    a path of two edges, the middle one two disjoint edges. Full: 2 successor states (one by 2
//    transitions); Automatic and None: 3 (the edge lists differ).
//  - {{x}} -> {} from {{1},{1},{2}}: deleting either {1} makes {{1},{2}}, deleting {2} makes
//    {{1},{1}}. Automatic and Full: 2 successor states, one by 2 transitions; None: 3.
TEST(SamplingReproducibility, SuccessorCapsCountDistinctSuccessorStates) {
    const auto del2 = std::vector<RewriteRule>{make_rule(0).lhs({0, 1}).build()};
    const auto del1 = std::vector<RewriteRule>{make_rule(0).lhs({0}).build()};
    const std::vector<std::vector<VertexId>> path = {{1, 2}, {2, 3}, {3, 4}};
    const std::vector<std::vector<VertexId>> units = {{1}, {1}, {2}};
    using M = StateCanonicalizationMode;

    // k = 2 keeps both successors of the path under Full: all 3 transitions.
    for (size_t threads : {size_t(1), size_t(4), size_t(16)}) {
        const auto f = run_successor_cap(M::Full, del2, path, 1, 2, 0, threads, 7);
        EXPECT_EQ(f.events, 3u) << threads;
        EXPECT_EQ(f.states, 3u) << threads;
        for (M m : {M::Automatic, M::None}) {
            const auto a = run_successor_cap(m, del2, path, 1, 2, 0, threads, 7);
            EXPECT_EQ(a.events, 2u) << threads << " mode " << static_cast<int>(m);
            EXPECT_EQ(a.states, 3u) << threads << " mode " << static_cast<int>(m);
        }
    }
    // k = 1 keeps one successor state, with every transition into it. Over seeds, the successor
    // made by two transitions is kept with both.
    for (M m : {M::Full, M::Automatic}) {
        const auto& rules = m == M::Full ? del2 : del1;
        const auto& init = m == M::Full ? path : units;
        std::set<size_t> events;
        for (uint64_t seed = 0; seed < 24; ++seed) {
            const auto one = run_successor_cap(m, rules, init, 1, 1, 0, 1, seed);
            EXPECT_EQ(one.states, 2u) << "mode " << static_cast<int>(m) << " seed " << seed;
            EXPECT_EQ(one.most_per_parent, 1u);
            events.insert(one.events);
            for (size_t threads : {size_t(4), size_t(16)}) {
                const auto t = run_successor_cap(m, rules, init, 1, 1, 0, threads, seed);
                EXPECT_EQ(t.events, one.events) << "seed " << seed << " threads " << threads;
            }
        }
        EXPECT_EQ(events, (std::set<size_t>{1u, 2u})) << "mode " << static_cast<int>(m);
    }
    // MaxStatesPerStep 1 on two steps of the path under Full: step 1 keeps one class and every
    // transition into it; step 2 reaches one class, the single edge, from every kept state:
    // 1 + 2 events (the disjoint pair kept) or 2 + 4 (the path kept).
    std::set<size_t> step_events;
    for (uint64_t seed = 0; seed < 24; ++seed) {
        const auto s = run_successor_cap(M::Full, del2, path, 2, 0, 1, 1, seed);
        EXPECT_EQ(s.states, 3u) << seed;
        EXPECT_EQ(s.most_per_step, 1u) << seed;
        step_events.insert(s.events);
        for (size_t threads : {size_t(4), size_t(16)})
            EXPECT_EQ(run_successor_cap(M::Full, del2, path, 2, 0, 1, threads, seed).events,
                      s.events)
                << "seed " << seed << " threads " << threads;
    }
    EXPECT_EQ(step_events, (std::set<size_t>{3u, 6u}));
    // Under Automatic the path's successors all differ: one transition per step.
    const auto a = run_successor_cap(M::Automatic, del2, path, 2, 0, 1, 1, 7);
    EXPECT_EQ(a.events, 2u);
    EXPECT_EQ(a.states, 3u);
    // MatchesPerStep counts transitions: 2 of the path's 3 under Full, whichever states they make.
    for (uint64_t seed = 0; seed < 8; ++seed) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(M::Full);
        ParallelEvolutionEngine e(&hg, 4);
        e.add_rule(del2[0]);
        e.set_matches_per_step(2);
        e.set_random_seed(seed);
        e.evolve(path, 1);
        EXPECT_EQ(hg.num_events(), 2u) << seed;
    }
}

// Each parent's events reach at most k distinct successor states and each step's at most N, at
// 1, 4 and 16 workers with equal counts, on growth rules under Automatic and Full states.
TEST(SamplingReproducibility, SuccessorCapsBoundEveryParentAndStep) {
    const std::vector<std::vector<RewriteRule>> rule_sets = {
        {make_rule(0).lhs({0, 1}).lhs({0, 2}).rhs({0, 1}).rhs({0, 3}).rhs({1, 3}).rhs({2, 3})
             .build()},
        {make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build(),
         make_rule(1).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).build()},
    };
    const std::vector<std::vector<VertexId>> init = {{0, 1}, {0, 2}, {1, 2}};
    using M = StateCanonicalizationMode;
    for (const auto& rules : rule_sets) {
        for (M m : {M::Automatic, M::Full}) {
            for (size_t k : {size_t(1), size_t(2), size_t(3)}) {
                const auto p = run_successor_cap(m, rules, init, 3, k, 0, 1, 11);
                EXPECT_LE(p.most_per_parent, k) << "mode " << static_cast<int>(m) << " k " << k;
                const auto s = run_successor_cap(m, rules, init, 3, 0, k, 1, 11);
                EXPECT_LE(s.most_per_step, k) << "mode " << static_cast<int>(m) << " N " << k;
                for (size_t threads : {size_t(4), size_t(16)}) {
                    const auto pt = run_successor_cap(m, rules, init, 3, k, 0, threads, 11);
                    const auto st = run_successor_cap(m, rules, init, 3, 0, k, threads, 11);
                    EXPECT_EQ(std::make_tuple(pt.states, pt.events, pt.causal, pt.branchial),
                              std::make_tuple(p.states, p.events, p.causal, p.branchial))
                        << "per parent, mode " << static_cast<int>(m) << " k " << k
                        << " threads " << threads;
                    EXPECT_EQ(std::make_tuple(st.states, st.events, st.causal, st.branchial),
                              std::make_tuple(s.states, s.events, s.causal, s.branchial))
                        << "per step, mode " << static_cast<int>(m) << " N " << k
                        << " threads " << threads;
                }
            }
        }
    }
}

// Under full capture with Full states and Automatic event identity, the causal and branchial
// relations come from the quotient reconstruction. Under MaxStatesPerStep a class keeps different
// transitions at different depths, and the reconstruction gives the relations of the raw
// rewrites, which None event identity reports directly: equal raw events, unreduced causal pairs
// and branchial pairs, at 1 and 4 workers. Rules whose classes recur at several depths.
TEST(SamplingReproducibility, StepCapReconstructionMatchesTheRawRewrites) {
    struct Case {
        const char* name;
        std::vector<RewriteRule> rules;
        std::vector<std::vector<VertexId>> init;
    };
    const std::vector<Case> cases = {
        {"flip", {make_rule(0).lhs({0, 1}).rhs({1, 0}).build()}, {{0u, 1u}, {1u, 2u}, {2u, 0u}}},
        {"shortcut",
         {make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).rhs({2, 1}).build()},
         {{0u, 1u}, {1u, 2u}, {2u, 0u}, {0u, 0u}}},
        {"two_rules",
         {make_rule(0).lhs({0, 1}).rhs({1, 0}).build(),
          make_rule(1).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).rhs({1, 1}).build()},
         {{0u, 1u}, {1u, 2u}, {2u, 0u}, {2u, 3u}}},
    };
    for (const auto& c : cases) {
        for (size_t k : {size_t(1), size_t(2), size_t(3), size_t(5)}) {
            auto run = [&](hgcommon::EventSignatureKeys keys, size_t threads) {
                Hypergraph hg;
                hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
                hg.set_event_signature_keys(keys);
                ParallelEvolutionEngine e(&hg, threads);
                e.set_transitive_reduction(false);
                for (const auto& r : c.rules) e.add_rule(r);
                e.set_max_states_per_step(k);
                e.evolve(c.init, 5);
                const uint64_t raw = keys == hgcommon::EVENT_SIG_NONE
                                         ? hg.num_events() : hg.num_reconstructed_raw_events();
                return std::array<uint64_t, 4>{hg.num_canonical_states(), raw,
                                               hg.observable_num_causal_pairs(false),
                                               hg.observable_num_branchial()};
            };
            const auto raw = run(hgcommon::EVENT_SIG_NONE, 1);
            for (size_t threads : {size_t(1), size_t(4)})
                EXPECT_EQ(run(hgcommon::EVENT_SIG_AUTOMATIC, threads), raw)
                    << c.name << " MaxStatesPerStep " << k << " threads " << threads;
        }
    }
}

// Every sampling and capping option chooses the same transitions at 1, 4 and 32 workers. The
// per-state choices complete at the state's drain and the per-step choices when the step's
// matching completes (the depth join), with no barrier between steps (SPEC §5, sampling and
// capping). Rule {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}} from {{1,2},{1,3}}, 4 steps, each
// option alone; compared by the isomorphism-invariant causal fingerprint.
TEST(SamplingReproducibility, EverySamplingOptionChoosesTheSameTransitionsAt1And4And32Workers) {
    struct Option {
        const char* name;
        std::function<void(ParallelEvolutionEngine&)> set;
    };
    const std::vector<Option> options = {
        {"TransitionRate 0.5", [](ParallelEvolutionEngine& e) {
             e.set_random_seed(7); e.set_transition_rate(0.5); }},
        {"ExplorationProbability 0.5", [](ParallelEvolutionEngine& e) {
             e.set_random_seed(7); e.set_exploration_probability(0.5); }},
        {"MatchesPerStateRule 2", [](ParallelEvolutionEngine& e) {
             e.set_random_seed(7); e.set_matches_per_state_rule(2); }},
        {"MaxSuccessorStatesPerParent 2", [](ParallelEvolutionEngine& e) {
             e.set_max_successor_states_per_parent(2); }},
        {"MaxStatesPerStep 5", [](ParallelEvolutionEngine& e) { e.set_max_states_per_step(5); }},
        {"MatchesPerStep 5", [](ParallelEvolutionEngine& e) {
             e.set_random_seed(7); e.set_matches_per_step(5); }},
    };
    for (const Option& o : options) {
        auto run = [&](size_t threads) {
            Hypergraph hg;
            ParallelEvolutionEngine e(&hg, threads);
            e.add_rule(make_rule(0).lhs({0, 1}).lhs({0, 2})
                           .rhs({0, 1}).rhs({0, 3}).rhs({1, 3}).rhs({2, 3}).build());
            o.set(e);
            e.evolve(std::vector<std::vector<VertexId>>{{1u, 2u}, {1u, 3u}}, 4);
            return causal_fingerprint(hg);
        };
        const auto one = run(1);
        ASSERT_FALSE(one.empty()) << o.name;
        for (size_t threads : {size_t(4), size_t(32)})
            EXPECT_EQ(run(threads), one) << o.name << " at " << threads << " workers";
    }
}

// "StateCost" (hgcommon/state_cost_core.hpp). The interpreter on fixed values: each operation, a
// statistic with no value, NaN, and program validation.
TEST(StateCost, InterpreterEvaluatesEveryOperation) {
    using namespace hgcommon;
    StateInvariantValues v;
    v.vertex_count = 9;
    v.edge_count = 4;
    v.mean_degree = 2.5;
    SgGeometry g;
    g.defined = SG_RADIUS;
    g.radius = 3;
    g.mean_eccentricity = 2.0;
    auto prog = [](std::initializer_list<std::pair<uint8_t, double>> ops) {
        StateCost c;
        for (const auto& [op, arg] : ops) {
            c.op[c.n] = op;
            c.arg[c.n] = arg;
            ++c.n;
        }
        return c;
    };
    auto eval = [&](StateCost c) {
        EXPECT_TRUE(state_cost_validate(c));
        return state_cost_eval(c, v, g);
    };
    // |VertexCount - 12| + 0.5 EdgeCount = 3 + 2.
    EXPECT_EQ(eval(prog({{SC_STAT, 0}, {SC_CONST, 12}, {SC_SUB, 0}, {SC_ABS, 0}, {SC_CONST, 0.5},
                         {SC_STAT, 1}, {SC_MUL, 0}, {SC_ADD, 0}})),
              5.0);
    EXPECT_EQ(eval(prog({{SC_STAT, 0}, {SC_STAT, 1}, {SC_DIV, 0}})), 9.0 / 4.0);
    EXPECT_EQ(eval(prog({{SC_STAT, 0}, {SC_STAT, 1}, {SC_MIN, 0}})), 4.0);
    EXPECT_EQ(eval(prog({{SC_STAT, 0}, {SC_STAT, 1}, {SC_MAX, 0}})), 9.0);
    EXPECT_EQ(eval(prog({{SC_STAT, 0}, {SC_NEG, 0}})), -9.0);
    EXPECT_EQ(eval(prog({{SC_STAT, 0}, {SC_SQRT, 0}})), 3.0);
    EXPECT_EQ(eval(prog({{SC_CONST, -2}, {SC_POW, 3}})), -8.0);
    EXPECT_EQ(eval(prog({{SC_STAT, 1}, {SC_POW, -2}})), 1.0 / 16.0);
    EXPECT_NEAR(eval(prog({{SC_STAT, 3}, {SC_POW, 0.5}})), std::sqrt(2.5), 1e-15);
    EXPECT_NEAR(eval(prog({{SC_STAT, 0}, {SC_LOG, 0}})), std::log(9.0), 1e-15);
    EXPECT_NEAR(eval(prog({{SC_CONST, 1}, {SC_EXP, 0}})), std::exp(1.0), 1e-15);
    // GraphRadius (slot 11) is defined; WolframHausdorffDimension (slot 13) is not.
    EXPECT_EQ(eval(prog({{SC_STAT, 11}})), 3.0);
    EXPECT_EQ(eval(prog({{SC_STAT, 13}, {SC_CONST, 1}, {SC_ADD, 0}})), HUGE_VAL);
    // NaN and the root of a negative number are +Infinity.
    EXPECT_EQ(eval(prog({{SC_CONST, -1}, {SC_SQRT, 0}})), HUGE_VAL);
    EXPECT_EQ(eval(prog({{SC_CONST, -1}, {SC_LOG, 0}})), HUGE_VAL);
    EXPECT_EQ(eval(prog({{SC_CONST, 0}, {SC_CONST, 0}, {SC_DIV, 0}})), HUGE_VAL);
    // The measures of the statistics read.
    StateCost m = prog({{SC_STAT, 25}, {SC_STAT, 11}, {SC_ADD, 0}});
    ASSERT_TRUE(state_cost_validate(m));
    EXPECT_EQ(m.measures, SG_VOLUME_GROWTH | SG_RADIUS);
    // Malformed programs.
    StateCost bad = prog({{SC_ADD, 0}});
    EXPECT_FALSE(state_cost_validate(bad));
    bad = prog({{SC_CONST, 1}, {SC_CONST, 2}});
    EXPECT_FALSE(state_cost_validate(bad));
    bad = prog({{SC_STAT, 42}});
    EXPECT_FALSE(state_cost_validate(bad));
    bad = prog({{SC_STAT, 1.5}});
    EXPECT_FALSE(state_cost_validate(bad));
    bad = prog({{SC_OPS, 0}});
    EXPECT_FALSE(state_cost_validate(bad));
}

// The cost order: (cost, successor rank) ascending, equal pairs at one position.
TEST(StateCost, CostOrderRanksByCostThenSuccessorRank) {
    const uint64_t cost[6] = {hgcommon::state_cost_key(2.0), hgcommon::state_cost_key(-1.0),
                              hgcommon::state_cost_key(2.0), hgcommon::state_cost_key(HUGE_VAL),
                              hgcommon::state_cost_key(-1.0), hgcommon::state_cost_key(2.0)};
    const uint64_t srank[6] = {7, 9, 3, 1, 9, 7};
    uint32_t order[6];
    uint64_t out[6];
    hgcommon::cost_successor_ranks(cost, srank, 6, order, out);
    const uint64_t want[6] = {2, 0, 1, 3, 0, 2};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], want[i]) << i;
}

