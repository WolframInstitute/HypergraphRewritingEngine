// NO MATCH IS EVER MISSED.
//
// Matching is incremental. A child state C = P - consumed + produced inherits its parent's
// matches that use no consumed edge, and separately looks for matches that use a produced edge
// (delta). The partition is exhaustive: a match in C either uses only edges C took from P, and was
// then a match in P, or it touches a produced edge. Completeness reduces to three obligations:
//
//   1. P's match set was complete            (induction; a root does a full scan)
//   2. inheritance transfers every surviving match of P
//   3. delta finds every match using a produced edge
//
// Inheritance runs once per child, when P's matching has drained, and the child's own drain
// waits for it. So at a state's drain every match it will ever hold is claimed, and a full
// rematch there (validate_match_forwarding) must find nothing unclaimed, in every submission
// mode. One missed match at depth d removes the subtree below it while the run stays
// self-consistent, so an end-to-end count cannot find it; this check can.
//
// Every corpus case, a spread of worker counts, repeated: a race is a rate, and one passing run
// shows only that one interleaving was clean.

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "reference/oracle_corpus.hpp"

using namespace hypergraph;

namespace {

struct Outcome {
    size_t mismatches = 0;     // matches unclaimed at their state's drain
    size_t validations = 0;    // drains the validator ran at
    size_t still_missing = 0;  // of the mismatches, absent at the end of the run
};

Outcome run_validated(const oracle::Case& c, unsigned threads, bool batched, bool task_based) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_match_forwarding(true);
    engine.set_validate_match_forwarding(true);
    engine.set_batched_matching(batched);
    engine.set_task_based_matching(task_based);
    for (const auto& r : c.rules) engine.add_rule(r);
    engine.evolve(c.init, c.oracle_steps);

    Outcome o;
    o.mismatches = engine.validation_mismatches();
    o.validations = engine.validations_performed();
    o.still_missing = engine.still_missing();
    return o;
}

}  // namespace

// The three submission modes: task-based (the default), and the synchronous path batched and
// eager. Workers up to 32, because a lost match is a race and the determinism gate's firings
// were all at 16 or 32 threads.
TEST(MatchCompleteness, EveryStateHoldsEveryMatchAtItsDrain) {
    struct Mode { const char* name; bool batched, task_based; };
    const Mode modes[] = {{"task-based", true, true}, {"batched", true, false},
                          {"eager", false, false}};
    const std::vector<unsigned> worker_counts = {1, 2, 4, 8, 16, 32};
    constexpr int kReps = 2;

    for (const Mode& m : modes) {
        size_t runs = 0, validated = 0, missed = 0;
        std::vector<std::string> offenders;
        for (const auto& c : oracle::corpus()) {
            for (unsigned w : worker_counts) {
                for (int rep = 0; rep < kReps; ++rep) {
                    const Outcome o = run_validated(c, w, m.batched, m.task_based);
                    ++runs;
                    if (o.validations > 0) ++validated;
                    missed += o.mismatches;
                    if (o.mismatches != 0)
                        offenders.push_back(std::string(c.name) + " w=" + std::to_string(w) +
                                            " missing=" + std::to_string(o.mismatches) +
                                            " still=" + std::to_string(o.still_missing));
                }
            }
        }
        std::printf("# %s: %zu runs, %zu validated, %zu matches unclaimed at their drain\n",
                    m.name, runs, validated, missed);
        for (const auto& s : offenders) std::printf("#   %s\n", s.c_str());
        EXPECT_GT(validated, 0u) << m.name << ": the validator never executed";
        EXPECT_EQ(missed, 0u) << m.name << ": a state drained without every match it holds";
    }
}

// A run advanced one step at a time with forwarding on reaches what one run to the same depth
// reaches. A state at the step budget is not registered for forwarding; a continuation resumes it
// with a full match (defer_match_task), so it must find every match the forwarded copies carried.
// Every corpus case, full and quotient exploration, one and four workers.
TEST(MatchCompleteness, ForwardedRunContinuedStepwiseMatchesOneRun) {
    auto run = [](const oracle::Case& c, unsigned threads, bool quotient, bool stepped) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, threads);
        e.set_match_forwarding(true);
        e.set_explore_from_canonical_states_only(quotient);
        for (const auto& r : c.rules) e.add_rule(r);
        const size_t depth = c.oracle_steps;
        if (stepped) {
            e.set_continuable(true);
            e.evolve(c.init, 1);
            for (size_t d = 1; d < depth; ++d) e.evolve_more(1);
        } else {
            e.evolve(c.init, depth);
        }
        return std::array<size_t, 3>{hg.num_states(), hg.num_canonical_states(), hg.num_events()};
    };
    for (const auto& c : oracle::corpus()) {
        for (bool quotient : {false, true}) {
            const auto base = run(c, 1, quotient, false);
            for (unsigned threads : {1u, 4u}) {
                const auto stepped = run(c, threads, quotient, true);
                EXPECT_EQ(stepped, base) << c.name << (quotient ? " quotient" : " full") << ", "
                                         << threads << " worker(s): stepwise states/classes/events "
                                         << stepped[0] << "/" << stepped[1] << "/" << stepped[2]
                                         << " against one run's " << base[0] << "/" << base[1]
                                         << "/" << base[2];
            }
        }
    }
}

// A stopped run, continued, reaches what one run reaches, repeated because the defect it guards is
// an interleaving. A stop cuts a state's matching and defers a full rematch; the state's
// inheritance can then complete in the continuation before the rematch is submitted, and a state
// that drained there handed its children a partial list (MatchJoin::resume_pending). Measured
// before the guard: 4% of repetitions at 4 workers on this rule missed 3 to 8 events.
TEST(MatchCompleteness, AStoppedRunContinuedMatchesOneRunUnderRepetition) {
    const auto rule = make_rule(0).lhs({0, 1}).lhs({1, 2})
                          .rhs({0, 1}).rhs({1, 2}).rhs({2, 3}).build();
    const std::vector<std::vector<VertexId>> init = {{0, 1}, {1, 2}};
    const size_t depth = 4;
    for (bool quotient : {false, true}) {
        Hypergraph whole_hg;
        whole_hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine whole(&whole_hg, 4);
        whole.set_explore_from_canonical_states_only(quotient);
        whole.add_rule(rule);
        whole.evolve(init, depth + 1);
        size_t bad = 0;
        for (int rep = 0; rep < 200; ++rep) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine e(&hg, 4);
            e.set_explore_from_canonical_states_only(quotient);
            e.add_rule(rule);
            e.set_continuable(true);
            e.set_max_events(3);
            e.evolve(init, depth);
            e.set_max_events(0);
            e.evolve_more(1);
            if (hg.num_events() != whole_hg.num_events() ||
                hg.num_canonical_states() != whole_hg.num_canonical_states())
                ++bad;
        }
        EXPECT_EQ(bad, 0u) << (quotient ? "quotient" : "full") << ": " << bad
                           << " of 200 continued runs differ from one run";
    }
}
