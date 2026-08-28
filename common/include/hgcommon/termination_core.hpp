#pragma once
#include "hgcommon/namespace.hpp"
//
// THE TERMINATION DECISION, one body.
//
// A persistent kernel has no host to tell it the work is finished: one block watches the counters
// and decides. That decision was written twice in gpu/src/persistent.cu -- once for the matching
// kernel, once for the rewriting one -- differing only in WHICH cursor counts consumed work and
// in what each printed. The decision itself, which is the part that can be wrong, was duplicated.
//
// WHAT IT DECIDES, and why each condition is there:
//
//   QUIESCENT ONCE IS NOT ENOUGH. A match that has just completed may not have its records
//   visible yet, so a single balanced snapshot can be taken in the gap. The detector looks again
//   after a backoff and requires every observed quantity to be UNCHANGED across the window --
//   not merely to satisfy the conditions again. Each counter is monotone, so a worker that
//   started and finished inside the window necessarily moved one of them; re-testing the
//   conditions alone would accept two distinct quiescent moments with activity between them.
//
//   PRODUCED AND CONSUMED ARE BOTH REQUIRED. Checking only the role counters exits with work
//   outstanding that was produced but not yet taken; checking only the cursor exits before
//   anything has been produced at all.
//
//   THE BUDGET COUNTS LACK OF PROGRESS, NOT ELAPSED ROUNDS. A fixed round ceiling cannot tell a
//   deadlock from a workload that takes longer than the ceiling, and it fired on the second: a
//   disconnected left side produces a cartesian product of matches, every resident block sits
//   inside a long match, and the queue drains slowly. Measured on disc-l3a2g2r2 at depth 5, the
//   device was at 97% utilisation -- working, not stuck -- and a round ceiling gave up anyway.
//   The signatures differ: there, role0 read pushed=2972 completed=295, a queue nobody is popping
//   because every consumer is busy; a genuine stall has pushed == completed, because nobody is
//   working at all. So any movement in any observed counter resets the budget, and only rounds in
//   which nothing moves count against it. A deadlock still trips it -- nothing moves, by
//   definition -- while arbitrarily slow forward progress never does.
//
// The Ctx supplies WHERE the counters live and WHAT to do at the edges (stall, diagnostics,
// backoff, exit). It supplies no part of the decision.

#include "hgcommon/core.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace common {

// Snapshot buffers are the caller's: on the device they are registers or local memory sized by
// the detector's own role ceiling, and the core must not assume an allocator exists.
template <class Ctx>
HG_DEV void term_detect_loop(Ctx& c, uint64_t* p1, uint64_t* c1, uint64_t* p2, uint64_t* c2) {
    const uint32_t roles = c.num_roles();

    uint32_t stagnant  = 0;
    uint32_t last_prod = 0xFFFFFFFFu;
    uint32_t last_done = 0xFFFFFFFFu;
    uint64_t last_pc   = 0xFFFFFFFFFFFFFFFFull;

    for (uint32_t round = 0; ; ++round) {
        if (stagnant >= c.max_stagnant_rounds()) {
            // The last snapshot and the round are handed over: a stall report whose whole value
            // is naming WHICH counter pair failed to converge cannot reconstruct them.
            c.on_stall(round, p1, c1);
            c.signal_exit();
            return;
        }

        const bool     q1    = c.snapshot(p1, c1);
        const uint32_t prod1 = c.produced();
        const uint32_t done1 = c.consumed();

        c.on_round(round, prod1, done1);

        // Any movement resets the budget. The role counters are SUMMED rather than compared
        // elementwise here -- any move changes the sum, and the elementwise comparison that the
        // exit test needs is a stronger check made only when it matters.
        uint64_t pc = 0;
        for (uint32_t r = 0; r < roles; ++r) pc += p1[r] + c1[r];
        if (prod1 != last_prod || done1 != last_done || pc != last_pc) {
            last_prod = prod1;
            last_done = done1;
            last_pc   = pc;
            stagnant  = 0;
        } else {
            ++stagnant;
        }

        if (q1 && done1 >= prod1) {
            c.backoff_long();
            const bool     q2    = c.snapshot(p2, c2);
            const uint32_t prod2 = c.produced();
            const uint32_t done2 = c.consumed();

            bool unchanged = (prod1 == prod2) && (done1 == done2);
            for (uint32_t r = 0; r < roles && unchanged; ++r)
                unchanged = (p1[r] == p2[r]) && (c1[r] == c2[r]);

            if (q2 && done2 >= prod2 && unchanged) {
                c.signal_exit();
                return;
            }
        }
        c.backoff_short();
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
