// GPUMC harness: the persistent kernel's detector never takes the quiescent exit while a replay
// task (QeView::tasks) is claimed, unrun, or still owed by a task or record that is running.
//
// THE PROTOCOL TRANSCRIBED (gpu/include/hg_gpu/work_log.hpp WorkLogView, which
// qe_task_append and k_persistent_evolve's task branch drive; RewriteDetectorCtx):
//   - An append claims a slot with the pool's relaxed fetch-add on the task counter, writes the
//     task, and stores its published flag with release.
//   - A block's lane 0 reads the counter with acquire, clamps it to the capacity, and claims up
//     to kBatch consecutive tasks with one CAS on the task cursor. Each lane awaits its task's
//     published flag, runs it (which may append a child), and fences; lane 0 then adds the batch
//     size to tasks_done.
//   - A record's rewrite appends tasks (capture, seed) and books rewrites_done last, after a
//     fence.
//   - The detector is hgcommon::term_detect_loop with the kernel's accounting: produced is the
//     readable records plus the clamped task counter, consumed is rewrites_done plus tasks_done,
//     each read with acquire in the kernel's order.
//
// THE PROPERTY. If the detector signals exit through the QUIESCENT path, every task ran
// (kTotalTasks), the cursor reached the counter, and the record was rewritten.
//
// THE BOUND. HG_WORKERS workers (default 1): the rewriter, and with 2 a second block that only
// runs tasks, which adds the batch-claim race. The record's rewrite appends a depth-0 task and a depth-1 task; a
// depth-0 task appends one depth-1 child. Three tasks in all, a batch of two, a pool of four. One thread stands for a
// block and runs a batch's lanes in order. A lane never awaits a task appended by its own batch:
// every slot below the counter it read was claimed by a lane already running at the claim, and
// this block's lanes were all between batches then.
//
// THE COVER. An append happens inside a counted unit (the record being rewritten, or the task
// being run), so the counter moves before that unit is booked consumed.
//
// CALIBRATION. -DCALIBRATE_DONE_BEFORE_RUN books a batch in tasks_done before running it, and
// -DCALIBRATE_RECORD_BEFORE_CAPTURE books the record before its appends. Each removes the cover
// and the quiescent assertion must fire.
//
// HOW A SCOPE IS EXPRESSED: __VERIFIER_memory_scope_device() before the access it qualifies.
// The cursor is 64-bit because this GPUMC build cannot complete a 32-bit compare-exchange
// (hash_insert_elects_one.cpp).

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

#ifndef HG_WORKERS
#define HG_WORKERS 1
#endif

namespace {

constexpr uint32_t kWorkers    = HG_WORKERS;
constexpr uint32_t kCap        = 4;
constexpr uint32_t kBatch      = 2;
constexpr uint32_t kMaxDepth   = 2;
constexpr uint32_t kTotalTasks = 3;
constexpr uint32_t kMaxIdle    = 2;

uint32_t g_task_depth[kCap];
uint32_t g_task_published[kCap];
uint32_t g_task_counter;
uint64_t g_task_cursor;
uint32_t g_tasks_done;
uint32_t g_rewrites_done;
uint32_t g_records = 1;      // the record under rewrite, readable from the start

uint64_t g_pushed[1];
uint64_t g_completed[1];
uint32_t g_should_exit;
uint32_t g_exited_by_stall;
uint32_t g_ran;

uint64_t load64_dev(uint64_t* a, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, order);
}
bool cas64_dev(uint64_t* a, uint64_t* expected, uint64_t desired) {
    __VERIFIER_memory_scope_device();
    return __atomic_compare_exchange_n(a, expected, desired, /*weak=*/true,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}
uint32_t load32_dev(uint32_t* a, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, order);
}
void store32_dev(uint32_t* a, uint32_t v, int order) {
    __VERIFIER_memory_scope_device();
    __atomic_store_n(a, v, order);
}
uint32_t add32_dev(uint32_t* a, uint32_t n, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_fetch_add(a, n, order);
}
void fence_dev() {
    __VERIFIER_memory_scope_device();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

bool exit_requested() { return load32_dev(&g_should_exit, __ATOMIC_RELAXED) != 0; }

// qe_task_append. The pool never fills at this bound.
void append(uint32_t depth) {
    const uint32_t i = add32_dev(&g_task_counter, 1u, __ATOMIC_RELAXED);
    g_task_depth[i] = depth;
    store32_dev(&g_task_published[i], 1u, __ATOMIC_RELEASE);
}

// qe_apply's effect: the work is counted, and a child below the last level is appended.
void run_task(uint32_t i) {
    add32_dev(&g_ran, 1u, __ATOMIC_RELAXED);
    if (g_task_depth[i] + 1 < kMaxDepth) append(g_task_depth[i] + 1);
}

// Acquire for the detector, relaxed for a claimer (the published flag's acquire orders the data).
uint32_t readable_tasks(int order = __ATOMIC_ACQUIRE) {
    const uint32_t c = load32_dev(&g_task_counter, order);
    return c < kCap ? c : kCap;
}

// The task branch: claim a batch, run it, book it. False when there was nothing to claim.
bool task_batch() {
    const uint64_t readable = readable_tasks(__ATOMIC_RELAXED);
    uint64_t cur = load64_dev(&g_task_cursor, __ATOMIC_RELAXED);
    uint64_t base = 0;
    uint32_t count = 0;
    while (cur < readable) {
        const uint64_t k = readable - cur < kBatch ? readable - cur : kBatch;
        uint64_t expected = cur;
        if (cas64_dev(&g_task_cursor, &expected, cur + k)) {
            base = cur;
            count = static_cast<uint32_t>(k);
            break;
        }
        cur = expected;
    }
    if (count == 0) return false;
#if defined(CALIBRATE_DONE_BEFORE_RUN)
    add32_dev(&g_tasks_done, count, __ATOMIC_RELAXED);
#endif
    for (uint32_t l = 0; l < count; ++l) {
        const uint32_t i = static_cast<uint32_t>(base) + l;
        while (load32_dev(&g_task_published[i], __ATOMIC_ACQUIRE) == 0u) {}
        run_task(i);
        fence_dev();
    }
#if !defined(CALIBRATE_DONE_BEFORE_RUN)
    add32_dev(&g_tasks_done, count, __ATOMIC_RELAXED);
#endif
    return true;
}

void worker_loop() {
    uint32_t idle = 0;
    for (;;) {
        if (task_batch()) { idle = 0; continue; }
        if (exit_requested()) return;
        if (++idle >= kMaxIdle) return;
    }
}

struct DetectorCtx {
    uint32_t num_roles() const { return 1; }
    uint32_t max_stagnant_rounds() const { return 1; }
    bool snapshot(uint64_t* p, uint64_t* c) const {
        p[0] = load64_dev(&g_pushed[0], __ATOMIC_ACQUIRE);
        c[0] = load64_dev(&g_completed[0], __ATOMIC_ACQUIRE);
        return p[0] == c[0];
    }
    uint32_t produced() const {
        return load32_dev(&g_records, __ATOMIC_ACQUIRE) + readable_tasks();
    }
    uint32_t consumed() const {
        return load32_dev(&g_rewrites_done, __ATOMIC_ACQUIRE) +
               load32_dev(&g_tasks_done, __ATOMIC_ACQUIRE);
    }
    uint64_t work_progress() const { return load32_dev(&g_ran, __ATOMIC_RELAXED); }
    void backoff_long() const {}
    void backoff_short() const {}
    void on_round(uint32_t, uint32_t, uint32_t) const {}
    void on_stall(uint32_t, const uint64_t*, const uint64_t*) const {
        store32_dev(&g_exited_by_stall, 1u, __ATOMIC_RELEASE);
    }
    void signal_exit() const {
        if (!load32_dev(&g_exited_by_stall, __ATOMIC_ACQUIRE)) {
            assert(load32_dev(&g_ran, __ATOMIC_ACQUIRE) == kTotalTasks &&
                   "quiescent exit with a replay task still owed");
            assert(load64_dev(&g_task_cursor, __ATOMIC_ACQUIRE) ==
                   load32_dev(&g_task_counter, __ATOMIC_ACQUIRE) &&
                   "quiescent exit with a claimed task unrun");
            assert(load32_dev(&g_rewrites_done, __ATOMIC_ACQUIRE) == 1u &&
                   "quiescent exit with the record unrewritten");
        }
        store32_dev(&g_should_exit, 1u, __ATOMIC_RELEASE);
    }
};

void* detector(void*) {
    __VERIFIER_thread_global_id(0);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(0);
    __VERIFIER_thread_kernel_id(0);
    uint64_t p1[1], c1[1], p2[1], c2[1];
    DetectorCtx ctx;
    hgcommon::term_detect_loop(ctx, p1, c1, p2, c2);
    return nullptr;
}

// The block rewriting the record: its capture appends two tasks, then it books the record and
// runs tasks like any block.
void* rewriter(void*) {
    __VERIFIER_thread_global_id(1);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(1);
    __VERIFIER_thread_kernel_id(0);
#if defined(CALIBRATE_RECORD_BEFORE_CAPTURE)
    fence_dev();
    add32_dev(&g_rewrites_done, 1u, __ATOMIC_RELAXED);
    append(0);
    append(1);
#else
    append(0);
    append(1);
    fence_dev();
    add32_dev(&g_rewrites_done, 1u, __ATOMIC_RELAXED);
#endif
    worker_loop();
    return nullptr;
}

void* runner(void*) {
    __VERIFIER_thread_global_id(2);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(2);
    __VERIFIER_thread_kernel_id(0);
    worker_loop();
    return nullptr;
}

}  // namespace

int main() {
    pthread_t td, t1, t2;
    pthread_create(&td, nullptr, detector, nullptr);
    pthread_create(&t1, nullptr, rewriter, nullptr);
    if (kWorkers > 1) pthread_create(&t2, nullptr, runner, nullptr);
    pthread_join(td, nullptr);
    pthread_join(t1, nullptr);
    if (kWorkers > 1) pthread_join(t2, nullptr);
    return 0;
}
