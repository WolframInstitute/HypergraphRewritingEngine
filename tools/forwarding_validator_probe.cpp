// Forwarding-validator probe: turns on the engine's built-in match-forwarding
// validator (set_validate_match_forwarding) across the oracle corpus at several
// depths and thread counts. The validator re-scans each child state with full
// pattern matching; a full-scan match whose hash never enters seen_match_hashes_
// is a permanent loss (counted by still_missing()), while a match merely flagged
// late at scan time but delivered afterwards via push shows up in late_arrivals()
// and is not a loss.
//
// Self-checked invariant (baseline-free, race-sensitive): still_missing() must be
// IDENTICAL across thread counts for a given (case, steps). Permanent coverage is
// a structural property of the forwarding split, independent of scheduling, so if
// multi-threaded forwarding dropped a match that the single-threaded run kept,
// still_missing would differ across thread counts. (The residual nonzero
// still_missing on multi-LHS-edge rules is pre-existing: the engine relies on
// canonical state dedup rather than enumerating every raw (state,match) pair, and
// the canonical state / causal / branchial outputs remain exact — see cost_matrix.)
//
// The validator is implemented on the synchronous delta path, so we disable
// task-based matching for it (set_task_based_matching(false)).

#include "../reference/oracle_corpus.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

struct Run {
    size_t mismatches;
    size_t late_arrivals;
    size_t still_missing;
    size_t canonical_states;
};

Run run_case(const oracle::Case& c, int steps, unsigned threads) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_task_based_matching(false);       // validator lives on the sync path
    engine.set_validate_match_forwarding(true);
    for (const auto& r : c.rules) engine.add_rule(r);
    engine.evolve(c.init, steps);
    Run r;
    r.mismatches       = engine.validation_mismatches();
    r.late_arrivals    = engine.late_arrivals();
    r.still_missing    = engine.still_missing();
    r.canonical_states = hg.num_canonical_states();
    return r;
}

}  // namespace

int main() {
    auto cases = oracle::corpus();
    const unsigned thread_counts[] = {1, 4, 8};

    std::printf("%-18s %7s %8s %6s %10s %12s %12s\n",
                "case", "steps", "threads", "canon", "mismatch", "late", "stillMissing");
    std::printf("%s\n", std::string(80, '-').c_str());

    bool ok = true;
    for (const auto& c : cases) {
        // A few depths per case, capped by the deeper measure depth.
        std::vector<int> depths = {c.oracle_steps, c.oracle_steps + 1, c.measure_steps};
        for (int steps : depths) {
            size_t single_thread_missing = 0;
            for (unsigned t : thread_counts) {
                Run r = run_case(c, steps, t);
                if (t == 1) single_thread_missing = r.still_missing;
                // Race-sensitive gate: permanent loss must not depend on thread count.
                bool row_ok = (r.still_missing == single_thread_missing);
                ok = ok && row_ok;
                std::printf("%-18s %7d %8u %6zu %10zu %12zu %12zu%s\n",
                            c.name, steps, t, r.canonical_states,
                            r.mismatches, r.late_arrivals, r.still_missing,
                            row_ok ? "" : "  <== THREAD-DIVERGENT LOSS");
            }
        }
    }

    std::printf("%s\n", std::string(80, '-').c_str());
    std::printf("forwarding validator (permanent loss thread-invariant): %s\n",
                ok ? "PASS" : "*** THREAD-DIVERGENT LOSS ***");
    return ok ? 0 : 1;
}
