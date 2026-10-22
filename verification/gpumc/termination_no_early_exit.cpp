// GPUMC harness: the persistent kernel's detector never signals exit while work is still owed.
//
// Runs hgcommon/termination_core.hpp's term_detect_loop -- the body gpu/src/persistent.cu drives
// for both of its detectors -- over gpu/include/hg_gpu/termination.hpp's
// TerminationDetector::DeviceView (snapshot_quiescent, signal_exit, mark_pushed, mark_completed),
// with the device's atomics (verification/gpumc/device). The produced and consumed counts are read
// as the kernels read them: acquire loads of the record pool's counter and of rewrites_done.
//
// THE PROPERTY. If the detector signals exit through the QUIESCENT path, every role's pushed
// equals its completed and every produced record has been consumed. Exiting through the stall
// path is a different outcome and is not this claim: a stall is a recorded defect that returns
// partial work deliberately.
//
// THE SHAPE THAT BREAKS IT. A worker that marks itself completed BEFORE marking its child pushed
// leaves a window in which every counter is balanced and unchanged while a child is still owed --
// the same complete-then-submit precondition the host's quiescence rests on, which
// verification/tla/Quiescence.tla reports as MCQuiescenceLateSubmit. -DCALIBRATE_COMPLETE_THEN_PUSH
// reverses the two and the checker must report the early exit.
//
// THE BOUND. One worker, one child, one detector, and a stagnation budget of three rounds so every
// execution terminates. That is the smallest shape containing the window: the child's push must be
// able to land on either side of the detector's two snapshots.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: complete_then_push FAIL -DCALIBRATE_COMPLETE_THEN_PUSH

#include "hg_gpumc.hpp"
#include "hg_gpu/termination.hpp"
#include "hgcommon/termination_core.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

constexpr uint32_t kRoles = 1;

uint64_t g_pushed[g::TerminationDetector::kMaxRoles];
uint64_t g_completed[g::TerminationDetector::kMaxRoles];
uint32_t g_should_exit;
uint32_t g_produced;   // the record pool's counter
uint32_t g_consumed;   // rewrites_done

// Everything this run will ever complete: the item the host pre-marked, and the one child that
// doing it owes. A quiescent exit asserts against this.
constexpr uint64_t kTotalCompletions = 2;

uint32_t g_exited_by_stall;

g::TerminationDetector::DeviceView term() {
    return {g_pushed, g_completed, &g_should_exit, kRoles};
}

uint32_t load_acquire(uint32_t* a) {
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*a).load(
        cuda::memory_order_acquire);
}

// The detector's context, as gpu/src/persistent.cu supplies it, on the device's view.
struct HarnessCtx {
    uint32_t num_roles() const { return kRoles; }
    // Three, so every execution terminates: the device's budget is ten million rounds, which is a
    // wall-clock choice and not part of what is checked.
    uint32_t max_stagnant_rounds() const { return 3; }
    bool snapshot(uint64_t* p, uint64_t* c) const { return term().snapshot_quiescent(p, c); }
    uint32_t produced() const { return load_acquire(&g_produced); }
    uint32_t consumed() const { return load_acquire(&g_consumed); }
    uint64_t work_progress() const { return 0; }
    // No sleep: the checker explores every interleaving the backoff admits.
    void backoff_long() const {}
    void backoff_short() const {}
    void on_round(uint32_t, uint32_t, uint32_t) const {}
    void on_stall(uint32_t, const uint64_t*, const uint64_t*) const {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(g_exited_by_stall)
            .store(1u, cuda::memory_order_relaxed);
    }
    // THE CLAIM IS TESTED AT THE INSTANT OF THE DECISION: after the joins the worker has always
    // finished, so a check at the end of main cannot see the window. It is stated in the counters:
    // a quiescent exit claims the work is finished, so at that instant every completion owed has
    // been booked.
    void signal_exit() const {
        const bool stalled = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                                 g_exited_by_stall).load(cuda::memory_order_relaxed) != 0;
        if (!stalled) {
            const uint64_t done = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                                      g_completed[0]).load(cuda::memory_order_acquire);
            assert(done == kTotalCompletions &&
                   "the detector took the quiescent exit while a child was still owed");
        }
        term().signal_exit();
    }
};

void* detector(void*) {
    hg_gpumc::enter(0, 0);
    uint64_t p1[kRoles], c1[kRoles], p2[kRoles], c2[kRoles];
    HarnessCtx ctx;
    hgcommon::term_detect_loop(ctx, p1, c1, p2, c2);
    return nullptr;
}

void* worker(void*) {
    hg_gpumc::enter(1, 0);
    const g::TerminationDetector::DeviceView t = term();
    // The item the host pre-marked produces one record, and this worker consumes it: both cursors
    // level, the state the detector's produced/consumed test accepts. What is left to get wrong is
    // the role counters.
    atomicAdd(&g_produced, 1u);
    atomicAdd(&g_consumed, 1u);
#if defined(CALIBRATE_COMPLETE_THEN_PUSH)
    // Booked complete BEFORE the child it owes is announced.
    t.mark_completed(0);
    t.mark_pushed(0);
#else
    // The child is announced BEFORE this worker is booked complete: the persistent kernel's order.
    t.mark_pushed(0);
    t.mark_completed(0);
#endif
    // The child runs and finishes.
    t.mark_completed(0);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[2] = {1, 1};
    hg_gpumc::launch(2, lanes);
    // The host marks what it enqueued before launching (mark_pushed_host).
    g_pushed[0] = 1;
    pthread_t td, tw;
    pthread_create(&td, nullptr, detector, nullptr);
    pthread_create(&tw, nullptr, worker, nullptr);
    pthread_join(td, nullptr);
    pthread_join(tw, nullptr);
    assert(g_should_exit == 1);
    assert(g_pushed[0] == g_completed[0]);
    assert(g_consumed >= g_produced);
    return 0;
}
