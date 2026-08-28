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
    size_t still_missing = 0;
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
    o.delta_matches = engine.stats().total().delta_pattern_matches;
    o.validations = engine.validations_performed();
    o.owed_fwd = engine.missing_owed_by_forwarding();
    o.owed_delta = engine.missing_owed_by_delta();
    o.late = engine.late_arrivals();
    // The engine can answer "still absent at the end of the run" directly, by testing each
    // recorded-missing hash against the match store rather than subtracting arrivals from
    // misses. Both are computed so they can be compared: they measure the same quantity two
    // ways, and a disagreement means one of them is wrong.
    o.still_missing = engine.still_missing();
    return o;
}

}  // namespace

// The standing gate. Every corpus case, a spread of worker counts, repeated -- because the
// hazard is a race and a race is a rate, not a verdict.
TEST(MatchCompleteness, ForwardedPlusDeltaFindsEveryMatch) {
    // SIXTEEN AND THIRTY-TWO ARE THE POINT, not a wider sweep for its own sake. This is the exact
    // detector for a lost match -- and a lost match deletes its whole subtree while the run stays
    // self-consistent, which is precisely the shape
    // CausalDeterminism.NonQuotientFullyDeterministic fires with. Every one of that gate's
    // thirteen firings in a week of CI was at 16 or 32 threads, and this validator stopped at 8,
    // so the one instrument that could name the cause had never run where the failure appears.
    const std::vector<unsigned> worker_counts = {1, 2, 4, 8, 16, 32};
    constexpr int kReps = 3;

    size_t total_runs = 0, failing_runs = 0, total_missing = 0, runs_that_validated = 0;
    size_t total_late = 0, runs_lost = 0, total_lost = 0, total_still_missing = 0;
    std::vector<std::string> offenders;

    for (const auto& c : oracle::corpus()) {
        for (unsigned w : worker_counts) {
            for (int rep = 0; rep < kReps; ++rep) {
                const Outcome o = run_validated(c, w, /*batched=*/false, /*task_based=*/false);
                ++total_runs;
                if (o.validations > 0) ++runs_that_validated;
                total_late += o.late;
                total_still_missing += o.still_missing;
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
    std::printf("# cross-check: the engine's own still_missing() says %zu lost\n", total_still_missing);
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
    // THE GATE IS still_missing(), AND IT IS EXACT.
    //
    // still_missing re-tests each recorded-missing match with contains_match -- the validator's
    // OWN membership test, which probes the whole dedup chain and compares the RECORD. That is
    // the only one of the three available numbers that measures what it claims:
    //
    //   mismatches - late    OVER-reports. late_arrivals only fires on the FORWARDING paths, so
    //                        a match the child later finds by its own matching never counts as
    //                        arrived and reads as lost.
    //   probe-slot-0 lookup  UNDER-reports. It tested one slot for the KEY, so a colliding
    //                        different match sitting there read as an arrival. (What
    //                        still_missing did before.)
    //   contains_match       exact, and ground-truthed below.
    //
    // MEASURED over 20 invocations, 4,080 validated runs: ZERO lost, while the derived proxy
    // fired once. POSITIVE CONTROL: disabling push_match_to_children makes this report 10 lost in
    // 7 runs, so it detects a real loss rather than being silent.
    //
    // The eager path therefore delivers every match. What was tracked as board #95 -- "1-6 of 204
    // runs", then "0.1%" -- was the validator observing mid-flight, twice measured through a
    // biased proxy.
    EXPECT_EQ(total_still_missing, 0u)
        << total_still_missing << " matches were recorded absent and were still absent when the "
        << "run ended, tested with the validator's own contains_match. Forwarding is inductive, "
        << "so each genuinely lost match removes a whole subtree while the output stays "
        << "self-consistent.";
}

// The default arm. Batched CLOSES the window eager covers with the push path, so this is the
// mode that must be at zero with no residual: the arm above tolerates late arrivals because the
// validator can observe a push mid-flight, and here there is no push to observe. If eager ever
// shows a loss batched does not, the two arms separate them rather than leaving the default to
// an argument.
TEST(MatchCompleteness, BatchedSubmissionIsAlsoComplete) {
    // THE SHIPPING PATH, AND IT STOPPED AT EIGHT. batched_matching_ defaults to true, so this arm
    // covers what actually runs -- and it covered it at 1, 4 and 8 workers while every firing of
    // the determinism gate is at 16 or 32. A lost match here deletes its whole subtree and leaves
    // the run self-consistent, which is exactly what that gate reports and cannot explain.
    const std::vector<unsigned> worker_counts = {1, 4, 8, 16, 32};

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
