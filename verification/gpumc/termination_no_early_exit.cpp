// GPUMC harness: the persistent kernel's detector never signals exit while work is still owed.
//
// Runs hgcommon/termination_core.hpp ITSELF -- the same body gpu/src/persistent.cu drives for
// both of its detectors -- rather than a model of it. That is the whole reason the decision was
// lifted out of the two kernels: a checker can be handed the decision without being handed a
// persistent CUDA kernel.
//
// WHY GPUMC RATHER THAN GenMC. The device's counters are scoped: the workers and the detector are
// separate CTAs communicating at device scope, and scoped-RC11 admits behaviours that RC11 does
// not. GenMC has no notion of a scope and would check a program the GPU does not run.
//
// HOW A SCOPE IS EXPRESSED. GPUMC reads an annotation call placed immediately BEFORE the access
// it qualifies -- __VERIFIER_memory_scope_device() -- and the access itself is an ordinary atomic.
// So the Ctx below is the device protocol with the annotation added, and the decision it drives is
// shared source.
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
#include "hgcommon/termination_core.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

extern "C" {
void __VERIFIER_memory_scope_device();
void __VERIFIER_thread_local_id(int);
void __VERIFIER_thread_group_id(int);
void __VERIFIER_thread_global_id(int);
void __VERIFIER_thread_kernel_id(int);
}

namespace {

constexpr uint32_t kRoles = 1;

uint64_t g_pushed[kRoles];
uint64_t g_completed[kRoles];
uint32_t g_should_exit;
uint32_t g_produced;
uint32_t g_consumed;

// Everything this run will ever complete: the item the host pre-marked, and the one child that
// doing it owes. A quiescent exit asserts against this rather than against a flag.
constexpr uint64_t kTotalCompletions = 2;

uint32_t g_exited_by_stall;

// Device-scope accesses, each preceded by the annotation that qualifies it.
uint64_t load_dev(uint64_t* a) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, __ATOMIC_ACQUIRE);
}
uint32_t load_dev32(uint32_t* a) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, __ATOMIC_ACQUIRE);
}
void add_dev(uint64_t* a, uint64_t n) {
    __VERIFIER_memory_scope_device();
    __atomic_fetch_add(a, n, __ATOMIC_RELEASE);
}
void add_dev32(uint32_t* a, uint32_t n) {
    __VERIFIER_memory_scope_device();
    __atomic_fetch_add(a, n, __ATOMIC_RELEASE);
}
void store_dev32(uint32_t* a, uint32_t v) {
    __VERIFIER_memory_scope_device();
    __atomic_store_n(a, v, __ATOMIC_RELEASE);
}

// The detector's storage face, as gpu/src/persistent.cu supplies it, with the scope annotations
// GPUMC needs. Nothing here is part of the decision.
struct HarnessCtx {
    uint32_t num_roles() const { return kRoles; }
    // Three, so every execution terminates: the real device budget is ten million rounds, which
    // is a wall-clock choice and not part of what is being checked.
    uint32_t max_stagnant_rounds() const { return 3; }

    bool snapshot(uint64_t* p, uint64_t* c) const {
        bool all_equal = true;
        for (uint32_t r = 0; r < kRoles; ++r) {
            p[r] = load_dev(&g_pushed[r]);
            c[r] = load_dev(&g_completed[r]);
            if (p[r] != c[r]) all_equal = false;
        }
        return all_equal;
    }
    uint32_t produced() const { return load_dev32(&g_produced); }
    uint32_t consumed() const { return load_dev32(&g_consumed); }

    // No sleep: the checker explores every interleaving the backoff could have admitted and more,
    // so a wait would only remove behaviours.
    void backoff_long() const {}
    void backoff_short() const {}

    void on_round(uint32_t, uint32_t, uint32_t) const {}
    void on_stall(uint32_t, const uint64_t*, const uint64_t*) const {
        store_dev32(&g_exited_by_stall, 1);
    }
    // THE CLAIM IS TESTED HERE, AT THE INSTANT OF THE DECISION, and it has to be: after the
    // threads are joined the worker has always finished, so a check at the end of main can never
    // see the transient this exists to catch. The first version of this harness did exactly that
    // and reported the defect arm as clean.
    //
    // A stall exit is a recorded defect returning partial work on purpose and claims nothing.
    void signal_exit() const {
        if (!load_dev32(&g_exited_by_stall)) {
            // STATED IN THE COUNTERS, not in a "finished" flag. A flag is set AFTER the last
            // counter write, so there is an instant where every piece of work is genuinely done
            // and the flag is not yet set -- and asserting on it fails the correct arm too, which
            // is what the first version of this harness did.
            //
            // kTotalCompletions is what this run will ever complete: the pre-marked item and the
            // one child it owes. A quiescent exit claims the work is finished, so at that instant
            // it must be.
            assert(load_dev(&g_completed[0]) == kTotalCompletions &&
                   "the detector took the quiescent exit while a child was still owed");
        }
        store_dev32(&g_should_exit, 1);
    }
};

void* detector(void*) {
    __VERIFIER_thread_global_id(0);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(0);
    __VERIFIER_thread_kernel_id(0);

    uint64_t p1[kRoles], c1[kRoles], p2[kRoles], c2[kRoles];
    HarnessCtx ctx;
    hgcommon::term_detect_loop(ctx, p1, c1, p2, c2);
    return nullptr;
}

void* worker(void*) {
    __VERIFIER_thread_global_id(1);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(1);
    __VERIFIER_thread_kernel_id(0);

    // The item the host pre-marked produces one record, and this worker consumes it. Both
    // cursors are then level, which is the state the detector's produced/consumed test accepts --
    // so what is left to get wrong is the ROLE counters, and that is the window.
    add_dev32(&g_produced, 1);
    add_dev32(&g_consumed, 1);

#if defined(CALIBRATE_COMPLETE_THEN_PUSH)
    // THE DEFECT: booked complete BEFORE the child it owes is announced. In between, every
    // quantity the detector reads agrees -- pushed equals completed, consumed equals produced --
    // and nothing moves, so two snapshots either side of that instant are identical and the
    // detector takes the quiescent exit with a child still owed.
    add_dev(&g_completed[0], 1);
    add_dev(&g_pushed[0], 1);
#else
    // The child is announced BEFORE this worker is booked complete, so there is no instant at
    // which the counters agree while a child is still owed. This is the order the persistent
    // kernel's workers keep, and it is what the detector's soundness rests on.
    add_dev(&g_pushed[0], 1);
    add_dev(&g_completed[0], 1);
#endif

    // The child runs and finishes.
    add_dev(&g_completed[0], 1);
    return nullptr;
}

}  // namespace

int main() {
    // The host marks what it enqueued before launching, so the counters start balanced against
    // what the workers will complete rather than at zero.
    g_pushed[0] = 1;

    pthread_t td, tw;
    pthread_create(&td, nullptr, detector, nullptr);
    pthread_create(&tw, nullptr, worker, nullptr);
    pthread_join(td, nullptr);
    pthread_join(tw, nullptr);

    // The decision itself was checked where it was made. What is left is that the run reached a
    // decision at all, and that the counters it left behind are consistent.
    assert(g_should_exit == 1);
    assert(g_pushed[0] == g_completed[0]);
    assert(g_consumed >= g_produced);
    return 0;
}
