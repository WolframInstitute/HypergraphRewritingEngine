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
// Obligation 2 is the exposed one, and WHICH SUBMISSION MODE IS IN USE decides how exposed.
// Under EAGER submission a child is created while its parent is still matching, so a match the
// parent discovers afterwards has to reach that child by push -- the same rendezvous that
// previously lost transitions under contention. Under BATCHED submission the parent finishes
// matching before any child exists, and that window closes. batched_matching_ defaults to true
// (parallel_evolution.hpp:577), so the shipping path is the batched one; the eager arm below is
// measured because the mode is still selectable, not because it is what ships.
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
    // The validator runs INSIDE the child's match task and counts what is not there YET. A
    // match delivered after that instant is counted as missing and then arrives, so a miss is
    // only a LOST match if it never arrives. The engine already tracks the difference; not
    // reading it is how a delivery that is merely late reads as a delivery that failed.
    size_t late = 0;
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
    o.late = engine.late_arrivals();
    return o;
}

}  // namespace

// The standing gate. Every corpus case, a spread of worker counts, repeated -- because the
// hazard is a race and a race is a rate, not a verdict.
TEST(MatchCompleteness, ForwardedPlusDeltaFindsEveryMatch) {
    const std::vector<unsigned> worker_counts = {1, 2, 4, 8};
    constexpr int kReps = 3;

    size_t total_runs = 0, failing_runs = 0, total_missing = 0, runs_that_validated = 0;
    size_t total_late = 0, runs_lost = 0, total_lost = 0;
    std::vector<std::string> offenders;

    for (const auto& c : oracle::corpus()) {
        for (unsigned w : worker_counts) {
            for (int rep = 0; rep < kReps; ++rep) {
                const Outcome o = run_validated(c, w, /*batched=*/false, /*task_based=*/false);
                ++total_runs;
                if (o.validations > 0) ++runs_that_validated;
                total_late += o.late;
                // What the run actually LOST: counted as missing and never delivered.
                const size_t lost = o.mismatches > o.late ? o.mismatches - o.late : 0;
                total_lost += lost;
                if (lost != 0) ++runs_lost;
                if (o.mismatches != 0) {
                    ++failing_runs;
                    total_missing += o.mismatches;
                    offenders.push_back(std::string(c.name) + " w=" + std::to_string(w) +
                                        " missing=" + std::to_string(o.mismatches) +
                                        " (fwd=" + std::to_string(o.owed_fwd) +
                                        " delta=" + std::to_string(o.owed_delta) +
                                        " late=" + std::to_string(o.late) + ")");
                }
            }
        }
    }

    std::printf("\n# match completeness: %zu/%zu runs incomplete, %zu matches missed total\n",
                failing_runs, total_runs, total_missing);
    std::printf("# of those, %zu arrived after the validator looked; %zu were LOST, in %zu runs\n",
                total_late, total_lost, runs_lost);
    std::printf("# %zu/%zu runs actually exercised the delta branch (where the check lives)\n",
                runs_that_validated, total_runs);
    for (const auto& s : offenders) std::printf("#   %s\n", s.c_str());

    EXPECT_GT(runs_that_validated, 0u)
        << "no run took the delta branch, so the validator never executed and a zero mismatch "
        << "count proves nothing";

    // THE GATE IS ON WHAT IS LOST, NOT ON WHAT IS LATE.
    //
    // This arm runs EAGER, which is not the default. The validator runs inside the child's match
    // task and counts every match not present AT THAT INSTANT, so a match the push path delivers
    // a moment later is counted as missing and then arrives. Those two are different facts and
    // only one of them is a defect.
    //
    // This distinction is not new information the engine had to be taught: late_arrivals_ has
    // always counted a previously-missing hash showing up, and the hash mixes source_state
    // (MatchRecord::hash), so the arrival is for the state that was owed. The test simply did not
    // read it, and reported the sum as a defect rate -- carried for a long time as "#76, 1-6
    // failing runs of 204, ratchet it to zero", with a tolerance of 12 runs standing in for it.
    //
    // MEASURED at ec16f78 over 18 invocations (3,672 validated runs): 38 late, 4 LOST, and never
    // more than one lost in a single invocation. So the eager path does drop a match -- but at
    // roughly 0.1% of runs, not the 1-6 of 204 the old gate recorded, because that number counted
    // arrivals as losses. Separating them moved the defect two orders of magnitude and, more to
    // the point, made it a different defect: what has to be explained is a rare genuine loss, not
    // a common benign lateness, and the two would have different causes.
    //
    // The bound is a recorded baseline and not an acceptance: it must ratchet to zero and may
    // never grow. It is set at twice the observed maximum so a rate does not read as a
    // regression, which is the whole reason a race is gated as a rate (board #34).
    constexpr size_t kKnownLostBaseline = 2;
    EXPECT_LE(total_lost, kKnownLostBaseline)
        << total_lost << " matches were never delivered, in " << runs_lost << " of "
        << total_runs << " runs, above the recorded baseline of " << kKnownLostBaseline
        << ". Forwarding is inductive, so each genuinely lost match removes a whole subtree and "
        << "the output stays self-consistent while being wrong.";
    if (total_lost > 0) {
        std::printf("# KNOWN OPEN DEFECT #95: eager forwarding LOSES %zu match(es) in %zu runs\n",
                    total_lost, total_runs);
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
