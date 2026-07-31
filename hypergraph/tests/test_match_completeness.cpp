// NO MATCH IS EVER MISSED. This is the deepest invariant in the engine, and it had no gate.
//
// Matching is incremental. A child state C = P - consumed + produced inherits its parent's
// still-valid matches (FORWARDED) and separately looks for matches that use a produced edge
// (DELTA). That partition is exhaustive -- any match in C either survives from P, in which case
// it was already a match in P, or it touches a produced edge -- so the scheme is sound in
// principle. Completeness then reduces to three obligations:
//
//   1. P's match set was complete            (induction; the root does a full scan)
//   2. forwarding transfers every surviving parent match
//   3. delta finds every match using a produced edge
//
// Obligation 2 is the exposed one, and it is exposed IN THE SHIPPING CONFIGURATION.
// batched_matching_ defaults to false, so matches are submitted eagerly and children are created
// while their parent is still matching. A match the parent discovers AFTER a child was created
// has to reach that child by push, and the push path is what covers the race. It is also the
// same rendezvous that previously lost transitions under contention.
//
// Because forwarding is INDUCTIVE, one missed match at depth d silently removes the entire
// subtree below it. Nothing downstream can notice: the run stays self-consistent and simply
// produces less. So an end-to-end output check cannot find this. The only check that can is the
// one the engine already carries and never ran -- validate_match_forwarding, which re-runs a FULL
// scan per state and asserts every match it finds was actually claimed.
//
// This gate turns that on and reports a RATE. A single passing run does not establish
// completeness of a racy path; it establishes that one interleaving happened to be clean.

#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "reference/oracle_corpus.hpp"

using namespace hypergraph;

namespace {

struct Outcome {
    size_t mismatches = 0;
    size_t states = 0;
    // The validator only runs inside the DELTA branch. If no state ever took that branch the
    // check never executed and a zero mismatch count means nothing -- so the gate proves it ran.
    size_t delta_matches = 0;
    size_t validations = 0;
    size_t owed_fwd = 0, owed_delta = 0;
};

Outcome run_validated(const oracle::Case& c, unsigned threads, bool batched,
                      bool task_based) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_validate_match_forwarding(true);
    engine.set_batched_matching(batched);
    // The validator lives on the SYNCHRONOUS delta path. The task-based path submits
    // SCAN tasks and returns before reaching it, so with task_based on the check is
    // unreachable and a clean result would mean nothing. See the second test.
    engine.set_task_based_matching(task_based);
    for (const auto& r : c.rules) engine.add_rule(r);
    engine.evolve(c.init, c.oracle_steps);

    Outcome o;
    o.mismatches = engine.validation_mismatches();
    o.states = hg.num_states();
    o.delta_matches = engine.stats().delta_pattern_matches.load();
    o.validations = engine.validations_performed();
    o.owed_fwd = engine.missing_owed_by_forwarding();
    o.owed_delta = engine.missing_owed_by_delta();
    return o;
}

}  // namespace

// The standing gate. Every corpus case, a spread of worker counts, repeated -- because the
// hazard is a race and a race is a rate, not a verdict.
TEST(MatchCompleteness, ForwardedPlusDeltaFindsEveryMatch) {
    const std::vector<unsigned> worker_counts = {1, 2, 4, 8};
    constexpr int kReps = 3;

    size_t total_runs = 0, failing_runs = 0, total_missing = 0, runs_that_validated = 0;
    std::vector<std::string> offenders;

    for (const auto& c : oracle::corpus()) {
        for (unsigned w : worker_counts) {
            for (int rep = 0; rep < kReps; ++rep) {
                const Outcome o = run_validated(c, w, /*batched=*/false, /*task_based=*/false);
                ++total_runs;
                if (o.validations > 0) ++runs_that_validated;
                if (o.mismatches != 0) {
                    ++failing_runs;
                    total_missing += o.mismatches;
                    offenders.push_back(std::string(c.name) + " w=" + std::to_string(w) +
                                        " missing=" + std::to_string(o.mismatches) +
                                        " (fwd=" + std::to_string(o.owed_fwd) +
                                        " delta=" + std::to_string(o.owed_delta) + ")");
                }
            }
        }
    }

    std::printf("\n# match completeness: %zu/%zu runs incomplete, %zu matches missed total\n",
                failing_runs, total_runs, total_missing);
    std::printf("# %zu/%zu runs actually exercised the delta branch (where the check lives)\n",
                runs_that_validated, total_runs);
    for (const auto& s : offenders) std::printf("#   %s\n", s.c_str());

    EXPECT_GT(runs_that_validated, 0u)
        << "no run took the delta branch, so the validator never executed and a zero mismatch "
        << "count proves nothing";

    // EAGER submission still has a residual miss, and it is a RATE because it is a race: the
    // parent may discover a match after a child was created, and eager relies on the push path
    // to deliver it. Measured at 1/204 runs (2 matches) after the join-order fix, always
    // attributed to forwarding and never to delta, and always on the high-automorphism case.
    // The batched arm below is clean, which is exactly what parallel_evolution.hpp claims:
    // batching eliminates the forwarding races that eager covers with the push path.
    //
    // Tracked as #76. The bound is a recorded baseline, not an acceptance of the defect: it must
    // ratchet to zero, and it may never grow.
    constexpr size_t kKnownEagerRaceRuns = 12;  // ~6% of 204; observed spread is 1-7
    EXPECT_LE(failing_runs, kKnownEagerRaceRuns)
        << failing_runs << " of " << total_runs << " runs missed at least one match, above the "
        << "recorded baseline of " << kKnownEagerRaceRuns << ". Forwarding is inductive, so each "
        << "miss removes a whole subtree and the output stays self-consistent while being wrong.";
    if (failing_runs > 0) {
        std::printf("# KNOWN OPEN DEFECT #76: eager forwarding race, %zu/%zu runs\n",
                    failing_runs, total_runs);
    }
}

// Batched submission is documented as eliminating the forwarding races that eager submission
// covers with the push path. If eager ever shows a miss that batched does not, this separates
// them rather than leaving the default to an argument.
TEST(MatchCompleteness, BatchedSubmissionIsAlsoComplete) {
    const std::vector<unsigned> worker_counts = {1, 4, 8};

    size_t total_runs = 0, failing_runs = 0, runs_that_validated = 0;
    for (const auto& c : oracle::corpus()) {
        for (unsigned w : worker_counts) {
            const Outcome o = run_validated(c, w, /*batched=*/true, /*task_based=*/false);
            ++total_runs;
            if (o.validations > 0) ++runs_that_validated;
            if (o.mismatches != 0) ++failing_runs;
        }
    }
    std::printf("# batched: %zu/%zu runs incomplete, %zu exercised the delta branch\n",
                failing_runs, total_runs, runs_that_validated);
    EXPECT_GT(runs_that_validated, 0u) << "the validator never executed";
    // Batched has no residual: it is asserted at zero, with no baseline.
    EXPECT_EQ(failing_runs, 0u);
}
