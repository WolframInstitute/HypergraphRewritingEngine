// GPUMC harness: the persistent kernel's loop -- ring, record pool and detector composed -- never
// takes the quiescent exit while an item sits in the ring, a record is unrewritten, or a child
// is owed.
//
// WHAT IS COMPOSED. termination_no_early_exit.cpp checks hgcommon/termination_core.hpp against
// a worker that models its own accounting; ring_exactly_once.cpp checks hgcommon/ring_core.hpp
// against producers and consumers that do nothing else. gpu/src/persistent.cu runs both cores in
// ONE loop per block, and the order that loop books pushed/completed around the ring and the
// record pool is what the detector's decision rests on. This harness runs that loop's control
// flow -- the same cores, the same accounting order, the same claim/await/publish protocol on
// the record pool -- so the decision is checked against the composition and not against either
// part alone.
//
// THE LOOP TRANSCRIBED (k_persistent_evolve; one worker thread here stands for a block's thread
// 0, which is the only lane that touches the ring, the pool and the counters):
//   1. claim the next record (cursor CAS below the pool's readable count); if one is claimed,
//      await its published flag, "rewrite" it, and if the child is below the step budget book
//      pushed[match] BEFORE try_push for each rule, running the item inline (completion booked
//      first) when the ring is full; then fence and bump rewrites_done.
//   2. otherwise try_pop a match item; if one came, "match" it (claim a record, write it,
//      publish) and book completed[match] AFTER the record is published.
//   3. otherwise leave if the detector asked; idle rounds are bounded here as the kernel bounds
//      them with kMaxWorkerIdleSpins.
//
// THE PROPERTY. If the detector signals exit through the QUIESCENT path, every match item was
// completed, every record was rewritten and the ring is empty. Exiting through the stall path
// is not a defect of the decision and is not asserted against.
//
// THE BOUND. A two-slot ring and three rules, so the rewrite of the seed's record pushes two
// children and runs the third inline through the full-ring path; a step budget of two, so the
// children's records push nothing. One seed item, three child items, four records. HG_WORKERS
// (default 1) sets the worker count; the two-worker run adds the ring and pool races the unit
// harnesses cover on their own. The ring is never one slot: the sequence scheme's push-complete
// and pop-complete marks coincide at capacity one, and a second push overwrites a live item.
//
// CALIBRATION. -DCALIBRATE_PUSH_THEN_BOOK books pushed[match] AFTER the push. A snapshot taken
// between the push and the booking sees pushed == completed with the item in the ring, and the
// quiescent assertion fires. -DCALIBRATE_ONE_SLOT shrinks the ring to one slot, where the
// sequence scheme's push-complete and pop-complete marks coincide: the second push overwrites
// a live item, the pop after it never matches, and the quiescent assertion fires (9 executions).
// gpu/include/hg_gpu/ring_buffer.hpp rejects a capacity below two for this reason. The harness
// is only evidence if both arms fail.
//
// THE CURSOR IS 64-BIT because the checker cannot complete a 32-bit compare-exchange
// (hash_insert_elects_one.cpp records the measurement); the kernel's cursor is a 32-bit word.
//
// HOW A SCOPE IS EXPRESSED: __VERIFIER_memory_scope_device() before the access it qualifies.

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

#ifndef HG_WORKERS
#define HG_WORKERS 1
#endif
constexpr uint32_t kWorkers    = HG_WORKERS;
#if defined(CALIBRATE_ONE_SLOT)
constexpr uint32_t kRingCap    = 1;
#else
constexpr uint32_t kRingCap    = 2;
#endif
constexpr uint32_t kRingMask   = kRingCap - 1;
#ifndef HG_RULES
#define HG_RULES 2
#endif
constexpr uint32_t kRules      = HG_RULES;
constexpr uint32_t kMaxSteps   = 2;
constexpr uint32_t kMaxRecords = 4;
constexpr uint32_t kMaxIdle    = 1;
constexpr uint32_t kInvalid    = 0xFFFFFFFFu;
// The seed item, and one item per rule from the seed's record.
constexpr uint64_t kTotalItems   = 1 + kRules;
constexpr uint32_t kTotalRecords = 1 + kRules;

// The ring.
uint32_t g_slots[kRingCap];
uint64_t g_seq[kRingCap];
uint64_t g_head;
uint64_t g_tail;

// The detector's counters, one role (match items).
uint64_t g_pushed[1];
uint64_t g_completed[1];
uint32_t g_should_exit;
uint32_t g_exited_by_stall;

// The record pool: claims, per-record published flag and payload, the consume cursor and the
// rewrites-done count the detector reads as consumed().
uint32_t g_rec_counter;
uint32_t g_rec_published[kMaxRecords];
uint32_t g_rec_step[kMaxRecords];
uint64_t g_consume_cursor;
uint32_t g_rewrites_done;

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
uint32_t add32_dev(uint32_t* a, uint32_t n, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_fetch_add(a, n, order);
}
void fence_dev() {
    __VERIFIER_memory_scope_device();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// The ring's storage face, as gpu/include/hg_gpu/ring_buffer.hpp presents it to ring_claim:
// relaxed cursor load and CAS, acquire sequence load, release sequence store.
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

// TerminationDetector::DeviceView: release fetch_add on the counters, acquire loads.
void mark_pushed()    { add64_dev(&g_pushed[0], 1, __ATOMIC_RELEASE); }
void mark_completed() { add64_dev(&g_completed[0], 1, __ATOMIC_RELEASE); }
bool exit_requested() { return load32_dev(&g_should_exit, __ATOMIC_ACQUIRE) != 0; }

// readable_records / claim_next_record / publish_match / await_match, as persistent.cu and
// match.hpp define them.
uint32_t readable_records() {
    const uint32_t claimed = load32_dev(&g_rec_counter, __ATOMIC_ACQUIRE);
    return claimed < kMaxRecords ? claimed : kMaxRecords;
}
uint32_t claim_next_record() {
    uint64_t cur = load64_dev(&g_consume_cursor, __ATOMIC_RELAXED);
    for (;;) {
        if (cur >= readable_records()) return kInvalid;
        uint64_t expected = cur;
        if (cas64_dev(&g_consume_cursor, &expected, cur + 1u)) return static_cast<uint32_t>(cur);
        cur = expected;
    }
}
void await_match(uint32_t idx) {
    while (load32_dev(&g_rec_published[idx], __ATOMIC_ACQUIRE) == 0u) {}
}

// match_state_rule's effect on the pool: claim a slot, fill it, publish it.
void match(uint32_t step) {
    const uint32_t idx = add32_dev(&g_rec_counter, 1u, __ATOMIC_RELAXED);
    if (idx >= kMaxRecords) return;
    g_rec_step[idx] = step;
    store32_dev(&g_rec_published[idx], 1u, __ATOMIC_RELEASE);
}

struct DetectorCtx {
    uint32_t num_roles() const { return 1; }
    uint32_t max_stagnant_rounds() const { return 1; }
    bool snapshot(uint64_t* p, uint64_t* c) const {
        p[0] = load64_dev(&g_pushed[0], __ATOMIC_ACQUIRE);
        c[0] = load64_dev(&g_completed[0], __ATOMIC_ACQUIRE);
        return p[0] == c[0];
    }
    uint32_t produced() const { return readable_records(); }
    uint32_t consumed() const { return load32_dev(&g_rewrites_done, __ATOMIC_ACQUIRE); }
    void backoff_long() const {}
    void backoff_short() const {}
    void on_round(uint32_t, uint32_t, uint32_t) const {}
    void on_stall(uint32_t, const uint64_t*, const uint64_t*) const {
        store32_dev(&g_exited_by_stall, 1u, __ATOMIC_RELEASE);
    }
    void signal_exit() const {
        if (!load32_dev(&g_exited_by_stall, __ATOMIC_ACQUIRE)) {
            assert(load64_dev(&g_completed[0], __ATOMIC_ACQUIRE) == kTotalItems &&
                   "quiescent exit with a match item still owed");
            assert(load32_dev(&g_rewrites_done, __ATOMIC_ACQUIRE) == kTotalRecords &&
                   "quiescent exit with a record still unrewritten");
            assert(load64_dev(&g_head, __ATOMIC_ACQUIRE) ==
                   load64_dev(&g_tail, __ATOMIC_ACQUIRE) &&
                   "quiescent exit with an item still in the ring");
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

void* worker(void* arg) {
    const int id = static_cast<int>(reinterpret_cast<long>(arg));
    __VERIFIER_thread_global_id(id);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(id);
    __VERIFIER_thread_kernel_id(0);
    uint32_t idle = 0;
    for (;;) {
        const uint32_t claimed = claim_next_record();
        if (claimed != kInvalid) {
            idle = 0;
            await_match(claimed);
            const uint32_t child_step = g_rec_step[claimed] + 1u;
            if (child_step < kMaxSteps) {
                for (uint32_t r = 0; r < kRules; ++r) {
#if !defined(CALIBRATE_PUSH_THEN_BOOK)
                    mark_pushed();
#endif
                    const bool run_inline = !try_push(child_step);
#if defined(CALIBRATE_PUSH_THEN_BOOK)
                    mark_pushed();
#endif
                    if (run_inline) {
                        mark_completed();
                        match(child_step);
                    }
                }
            }
            fence_dev();
            add32_dev(&g_rewrites_done, 1u, __ATOMIC_RELAXED);
            continue;
        }
        uint32_t item = 0;
        if (try_pop(item)) {
            idle = 0;
            match(item);
            mark_completed();
            continue;
        }
        if (exit_requested()) return nullptr;
        if (++idle >= kMaxIdle) return nullptr;
    }
}

}  // namespace

int main() {
    for (uint32_t i = 0; i < kRingCap; ++i) g_seq[i] = i;
    // k_seed_match_queue and mark_pushed_host: the seed item is in the ring and booked before
    // any block starts.
    g_slots[0]  = 0;
    g_seq[0]    = 1;
    g_tail      = 1;
    g_pushed[0] = 1;

    pthread_t td, tw[kWorkers];
    pthread_create(&td, nullptr, detector, nullptr);
    for (long i = 0; i < static_cast<long>(kWorkers); ++i)
        pthread_create(&tw[i], nullptr, worker, reinterpret_cast<void*>(i + 1));
    pthread_join(td, nullptr);
    for (uint32_t i = 0; i < kWorkers; ++i) pthread_join(tw[i], nullptr);

    assert(g_should_exit == 1);
    if (!g_exited_by_stall) {
        assert(g_pushed[0] == g_completed[0]);
        assert(g_rewrites_done == kTotalRecords);
        assert(g_head == g_tail);
    }
    return 0;
}
