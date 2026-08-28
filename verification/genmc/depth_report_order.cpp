// GenMC harness: a depth's completion is REPORTED in depth order, whatever order it is SETTLED in.
//
// Runs hgcommon/depth_join.hpp itself -- the same header the engine's depth signal is, not a
// model of it. The engine owns only the slot storage and the callback; the rule is there.
//
// THE PROPERTY. A consumer acts on a depth once nothing more can arrive in it, and acts on the
// depths in the order it is told about them. So a report of depth d+1 before a report of depth d
// describes a run in which d+1 drained first -- which the settle order forbids and which the
// consumer cannot detect. Reports must therefore be totally ordered by depth.
//
// THE SETUP. Two depths, one task each, and the two tasks finish concurrently. Depth 1 can only
// settle once the roots are in and its own task is done; depth 2 only once depth 1 has settled.
// Both threads run the cascade, so either may be the one to settle either depth -- which is the
// window: the thread that wins depth 1's settle owns depth 1's report and has not made it yet,
// and the other thread is free to walk past the now-complete depth 1 and settle depth 2.
//
// Each report appends to a shared log. The assertion is on the log's ORDER, not on which thread
// wrote it -- the protocol says nothing about who reports, only about the sequence.
//
// CALIBRATION. -DCALIBRATE_REPORT_AFTER_SETTLE reinstates the defect this closes: report inline
// at the point of the CAS, which is what the engine did when CI run 32982383954 (macOS,
// 2026-08-26) reported depth 3 before depth 2. Run
//
//   HG_HARNESS_DEFINES=-DCALIBRATE_REPORT_AFTER_SETTLE verification/genmc/run.sh depth_report_order
//
// and the checker must find the violation. A harness whose calibration arm passes is checking
// nothing, so the arm is a command rather than a claim in a comment.
#include "hgcommon/depth_join.hpp"

#include <atomic>
#include <cassert>
#include <pthread.h>

namespace {

hgcommon::DepthJoin::Slot g_slots[3];   // depths 0, 1, 2
hgcommon::DepthJoin       g_join;

// The report log. Sized to the number of depths that can be reported, which is two.
std::atomic<uint32_t> g_log[2];
std::atomic<int>      g_log_n{0};

void record(uint32_t depth) {
    const int i = g_log_n.fetch_add(1, std::memory_order_acq_rel);
    assert(i < 2 && "a depth was reported more than once");
    g_log[i].store(depth, std::memory_order_release);
}

#if defined(CALIBRATE_REPORT_AFTER_SETTLE)
// THE DEFECT, reinstated: settle, and report from inside the settling loop. Written against the
// same slots the shared protocol uses, so what differs between the arms is only where the report
// is issued from.
void settle_and_report_inline(uint32_t depth) {
    for (uint32_t d = (depth == 0 ? 1u : depth); d < 3; ++d) {
        if (g_slots[d].complete.load(std::memory_order_acquire)) continue;
        if (d > 1 && !g_slots[d - 1].complete.load(std::memory_order_acquire)) break;
        if (g_slots[d].live.load(std::memory_order_acquire) != 0) break;
        uint8_t expected = 0;
        if (!g_slots[d].complete.compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }
        record(d);      // a second step, and the window this harness exists for
    }
}
#endif

void finish(uint32_t depth) {
#if defined(CALIBRATE_REPORT_AFTER_SETTLE)
    if (g_slots[depth].live.fetch_sub(1, std::memory_order_acq_rel) == 1)
        settle_and_report_inline(depth);
#else
    g_join.done(depth, record);
#endif
}

void* worker1(void*) { finish(1); return nullptr; }
void* worker2(void*) { finish(2); return nullptr; }

}  // namespace

int main() {
    g_join.seat(g_slots, 3);

    // One task at each depth, booked before either can run -- the discipline the protocol
    // requires of its caller, and the reason depth 2 cannot settle early on an empty count.
    g_join.push(1);
    g_join.push(2);
    g_join.mark_roots_seeded();

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, worker1, nullptr);
    pthread_create(&t2, nullptr, worker2, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // Both depths settle in every execution: each has exactly one task, both finish, and depth 1
    // has its roots. So both are reported, and the order is the whole property.
    assert(g_log_n.load(std::memory_order_acquire) == 2);
    assert(g_log[0].load(std::memory_order_acquire) == 1);
    assert(g_log[1].load(std::memory_order_acquire) == 2);

    // Nothing arrived at a settled depth, which is the precondition the ordering rests on.
    assert(g_join.late_arrivals() == 0);
    return 0;
}
