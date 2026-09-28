// GPUMC harness: the persistent kernel's detector never takes the quiescent exit while a replay
// descent handed to another block (QeView::shares) is still owed.
//
// THE PROTOCOL TRANSCRIBED (gpu/include/hg_gpu/quotient_expansion.hpp qe_run, qe_share_hungry;
// gpu/src/persistent.cu k_persistent_evolve's shared-ring branch and idle path):
//   - A driver runs its descent stack (qe_run). Before driving a popped item, if more items are
//     pending and the grid is hungry (some block counted idle, ring below its low mark), it books
//     pushed[kQeShareRole], pushes its shallowest item, and on a full ring unbooks it
//     (completed[kQeShareRole]) and keeps it. Driving an item can push a child onto the driver's
//     own stack.
//   - A block in its idle path counts itself in idle_blocks once, and uncounts itself when it
//     finds work. On taking a ring item it drives it and the item's whole subtree on a fresh stack
//     (handing off in turn), and only then books completed[kQeShareRole].
//   - The driver does its replay while holding the record it rewrites: the record is role 0 here,
//     booked from the start and completed after the driver's qe_run. That is the kernel's cover
//     for work a block does inline: rewrites_done stays behind the readable records until then.
//   - The detector is hgcommon::term_detect_loop, the body the kernel runs, over two roles.
//
// THE PROPERTY. If the detector signals exit through the QUIESCENT path, every item was driven
// (kTotalDrives), both roles balance and the ring is empty.
//
// THE BOUND. Two workers; the driver starts holding two depth-0 items; an item below
// kMaxDepth - 1 produces one child. Four drives in all. A two-slot ring.
//
// EVERY PUSHER IS INSIDE A BOOKED UNIT: the driver pushes while its record is outstanding, and a
// taker pushes while the item it took is outstanding. So the detector cannot see both roles
// balanced between a push and its booking, and booking after the push (as the match queue must
// not) is not observable here: that variant verifies clean at this bound. The cover is what the
// property rests on, and the calibrations remove it.
//
// CALIBRATION. -DCALIBRATE_RECORD_BEFORE_REPLAY books the driver's record complete before its
// replay, and -DCALIBRATE_COMPLETE_BEFORE_DRIVE books a taken item complete before driving it.
// Each lets a snapshot see both roles balanced with work owed, and the quiescent assertion
// fires.
//
// HOW A SCOPE IS EXPRESSED: __VERIFIER_memory_scope_device() before the access it qualifies.
// The cursor words are 64-bit because this GPUMC build cannot complete a 32-bit
// compare-exchange (hash_insert_elects_one.cpp).

#include "hgcommon/ring_core.hpp"
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

constexpr uint32_t kRingCap    = 2;
constexpr uint32_t kRingMask   = kRingCap - 1;
constexpr uint32_t kMaxDepth   = 2;
constexpr uint32_t kTotalDrives = 4;
constexpr uint32_t kMaxIdle    = 2;
constexpr uint32_t kShareRole  = 1;
constexpr uint32_t kShareLow   = kRingCap;

uint32_t g_slots[kRingCap];   // an item is its depth
uint64_t g_seq[kRingCap];
uint64_t g_head;
uint64_t g_tail;

uint64_t g_pushed[2];
uint64_t g_completed[2];
uint32_t g_should_exit;
uint32_t g_exited_by_stall;
uint32_t g_idle_blocks;
uint32_t g_driven;

uint64_t load64_dev(uint64_t* a, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, order);
}
void store64_dev(uint64_t* a, uint64_t v, int order) {
    __VERIFIER_memory_scope_device();
    __atomic_store_n(a, v, order);
}
bool cas64_dev(uint64_t* a, uint64_t* expected, uint64_t desired) {
    __VERIFIER_memory_scope_device();
    return __atomic_compare_exchange_n(a, expected, desired, /*weak=*/true,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}
void add64_dev(uint64_t* a, uint64_t n, int order) {
    __VERIFIER_memory_scope_device();
    __atomic_fetch_add(a, n, order);
}
uint32_t load32_dev(uint32_t* a, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, order);
}
void store32_dev(uint32_t* a, uint32_t v, int order) {
    __VERIFIER_memory_scope_device();
    __atomic_store_n(a, v, order);
}
void add32_dev(uint32_t* a, int32_t n) {
    __VERIFIER_memory_scope_device();
    __atomic_fetch_add(a, static_cast<uint32_t>(n), __ATOMIC_RELAXED);
}

// ring_buffer.hpp's storage face for ring_claim.
template <bool kPush>
struct RingOps {
    const uint32_t* in;
    uint32_t*       out;
    uint32_t mask() const { return kRingMask; }
    uint64_t cursor_load() const {
        return load64_dev(kPush ? &g_tail : &g_head, __ATOMIC_RELAXED);
    }
    bool cursor_cas(uint64_t& expected, uint64_t desired) {
        return cas64_dev(kPush ? &g_tail : &g_head, &expected, desired);
    }
    uint64_t seq_load(uint32_t s) const { return load64_dev(&g_seq[s], __ATOMIC_ACQUIRE); }
    void seq_store(uint32_t s, uint64_t v) { store64_dev(&g_seq[s], v, __ATOMIC_RELEASE); }
    void transfer(uint32_t s) {
        if constexpr (kPush) g_slots[s] = *in; else *out = g_slots[s];
    }
};
bool try_push(uint32_t item) {
    RingOps<true> ops{&item, nullptr};
    return hgcommon::ring_claim(ops, /*want=*/0, /*leave=*/1);
}
bool try_pop(uint32_t& out) {
    RingOps<false> ops{nullptr, &out};
    return hgcommon::ring_claim(ops, /*want=*/1, /*leave=*/kRingMask + 1);
}
uint32_t ring_size_approx() {
    const uint64_t h = load64_dev(&g_head, __ATOMIC_RELAXED);
    const uint64_t t = load64_dev(&g_tail, __ATOMIC_RELAXED);
    return t > h ? static_cast<uint32_t>(t - h) : 0u;
}

void mark_pushed(uint32_t r)    { add64_dev(&g_pushed[r], 1, __ATOMIC_RELEASE); }
void mark_completed(uint32_t r) { add64_dev(&g_completed[r], 1, __ATOMIC_RELEASE); }
bool exit_requested() { return load32_dev(&g_should_exit, __ATOMIC_ACQUIRE) != 0; }

// qe_share_hungry.
bool hungry() {
    if (load32_dev(&g_idle_blocks, __ATOMIC_RELAXED) == 0) return false;
    return ring_size_approx() < kShareLow;
}

struct Stack {
    uint32_t items[4];
    uint32_t n = 0;
    void push(uint32_t v) { items[n++] = v; }
    bool pop(uint32_t& v) { if (n == 0) return false; v = items[--n]; return true; }
};

// One application's effect: the work is counted, and a child below the last level is pushed
// onto the driver's own stack (descend).
void drive(uint32_t depth, Stack& w) {
    add32_dev(&g_driven, 1);
    if (depth + 1 < kMaxDepth) w.push(depth + 1);
}

// qe_run with the hand-off.
void qe_run(Stack& w) {
    uint32_t it = 0;
    while (w.pop(it)) {
        if (w.n > 0 && hungry()) {
            mark_pushed(kShareRole);
            const bool pushed = try_push(w.items[0]);
            if (pushed) w.items[0] = w.items[--w.n];
            else mark_completed(kShareRole);
        }
        drive(it, w);
    }
}

// The persistent loop's shared-ring branch and idle path, for one block's thread 0.
void take_loop() {
    bool counted_idle = false;
    uint32_t idle = 0;
    for (;;) {
        uint32_t it = 0;
        if (try_pop(it)) {
            if (counted_idle) { add32_dev(&g_idle_blocks, -1); counted_idle = false; }
            idle = 0;
            Stack w;
#if defined(CALIBRATE_COMPLETE_BEFORE_DRIVE)
            mark_completed(kShareRole);
#endif
            drive(it, w);
            qe_run(w);
#if !defined(CALIBRATE_COMPLETE_BEFORE_DRIVE)
            mark_completed(kShareRole);
#endif
            continue;
        }
        if (exit_requested()) return;
        if (!counted_idle) { add32_dev(&g_idle_blocks, 1); counted_idle = true; }
        if (++idle >= kMaxIdle) return;
    }
}

struct DetectorCtx {
    uint32_t num_roles() const { return 2; }
    uint32_t max_stagnant_rounds() const { return 1; }
    bool snapshot(uint64_t* p, uint64_t* c) const {
        bool eq = true;
        for (uint32_t r = 0; r < 2; ++r) {
            p[r] = load64_dev(&g_pushed[r], __ATOMIC_ACQUIRE);
            c[r] = load64_dev(&g_completed[r], __ATOMIC_ACQUIRE);
            if (p[r] != c[r]) eq = false;
        }
        return eq;
    }
    uint32_t produced() const { return 0; }
    uint32_t consumed() const { return 0; }
    uint64_t work_progress() const { return load32_dev(&g_driven, __ATOMIC_RELAXED); }
    void backoff_long() const {}
    void backoff_short() const {}
    void on_round(uint32_t, uint32_t, uint32_t) const {}
    void on_stall(uint32_t, const uint64_t*, const uint64_t*) const {
        store32_dev(&g_exited_by_stall, 1u, __ATOMIC_RELEASE);
    }
    void signal_exit() const {
        if (!load32_dev(&g_exited_by_stall, __ATOMIC_ACQUIRE)) {
            assert(load32_dev(&g_driven, __ATOMIC_ACQUIRE) == kTotalDrives &&
                   "quiescent exit with a replay item still owed");
            assert(load64_dev(&g_head, __ATOMIC_ACQUIRE) ==
                   load64_dev(&g_tail, __ATOMIC_ACQUIRE) &&
                   "quiescent exit with an item still in the shared ring");
        }
        store32_dev(&g_should_exit, 1u, __ATOMIC_RELEASE);
    }
};

void* detector(void*) {
    __VERIFIER_thread_global_id(0);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(0);
    __VERIFIER_thread_kernel_id(0);
    uint64_t p1[2], c1[2], p2[2], c2[2];
    DetectorCtx ctx;
    hgcommon::term_detect_loop(ctx, p1, c1, p2, c2);
    return nullptr;
}

// The block whose record's capture produced two depth-0 descents: it runs them under its record
// (role 0), completes the record, then idles and takes ring items like any block.
void* driver(void*) {
    __VERIFIER_thread_global_id(1);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(1);
    __VERIFIER_thread_kernel_id(0);
    Stack w;
    w.push(0);
    w.push(0);
#if defined(CALIBRATE_RECORD_BEFORE_REPLAY)
    mark_completed(0);
    qe_run(w);
#else
    qe_run(w);
    mark_completed(0);
#endif
    take_loop();
    return nullptr;
}

void* taker(void*) {
    __VERIFIER_thread_global_id(2);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(2);
    __VERIFIER_thread_kernel_id(0);
    take_loop();
    return nullptr;
}

}  // namespace

int main() {
    for (uint32_t i = 0; i < kRingCap; ++i) g_seq[i] = i;
    g_pushed[0] = 1;   // the driver's record, booked before any block starts
    pthread_t td, t1, t2;
    pthread_create(&td, nullptr, detector, nullptr);
    pthread_create(&t1, nullptr, driver, nullptr);
    pthread_create(&t2, nullptr, taker, nullptr);
    pthread_join(td, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    return 0;
}
