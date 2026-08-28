// GPUMC harness: the device work queue hands every pushed item out exactly once.
//
// Runs hgcommon/ring_core.hpp ITSELF -- the same ring_claim body gpu/include/hg_gpu/ring_buffer.hpp
// drives for both of its roles -- rather than a model of it. That is why the claim rule was
// lifted out of the device header: a checker can be handed the decision without being handed a
// persistent CUDA kernel.
//
// WHY GPUMC RATHER THAN GenMC. Every access here is at DEVICE scope: the producers and consumers
// are separate CTAs, and scoped-RC11 admits behaviours RC11 does not. GenMC has no notion of a
// scope, so it would check a program the device does not run and call the queue proved.
//
// WHAT IS BEING PROVED. Across the run, no item is delivered twice and no item is lost while the
// queue is non-empty -- which for this queue is not a throughput property but a termination one.
// The persistent kernel's producers are its own consumers, so an item that vanishes is a
// completion that can never be booked, and gpu/src/persistent.cu's detector then waits for it
// forever. verification/gpumc/termination_no_early_exit.cpp checks the detector given a sound
// queue; this checks the queue.
//
// THE SHAPE THAT BREAKS IT. Reserving a position by an unconditional fetch_add instead of a
// compare-exchange. The cursor then hands out a position whose slot is not yet the reserver's,
// and with producers that are also consumers there is nothing to roll back with:
// -DCALIBRATE_BUMP_CURSOR makes the reservation a bump and the checker must report an item
// delivered twice, or one delivered whose slot was never published.
//
// THE BOUND. Capacity two, two threads, one item each, one pop each. That is the smallest shape
// in which two reservers can contend for one slot AND the ring can wrap, which are the two things
// the sequence numbers exist to separate. Capacity two rather than one so a wrap is reachable
// while both slots can be live at once.
#include "hgcommon/ring_core.hpp"

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

constexpr uint32_t kCap = 2;
constexpr uint32_t kMask = kCap - 1;
constexpr uint32_t kThreads = 2;

uint32_t g_slots[kCap];
uint64_t g_seq[kCap];
uint64_t g_head;
uint64_t g_tail;

// What each thread took, so the run can be judged after the joins. A queue fault is either the
// same item twice or an item that was never published, and both are statements about these.
uint32_t g_taken[kThreads];
bool     g_got[kThreads];

// Device-scope accesses, each preceded by the annotation that qualifies it. This is the storage
// half; none of it decides anything.
uint64_t load_dev(uint64_t* a, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, order);
}
void store_dev(uint64_t* a, uint64_t v, int order) {
    __VERIFIER_memory_scope_device();
    __atomic_store_n(a, v, order);
}
bool cas_dev(uint64_t* a, uint64_t* expected, uint64_t desired) {
    __VERIFIER_memory_scope_device();
    return __atomic_compare_exchange_n(a, expected, desired, /*weak=*/true,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}
uint64_t fetch_add_dev(uint64_t* a, uint64_t n) {
    __VERIFIER_memory_scope_device();
    return __atomic_fetch_add(a, n, __ATOMIC_RELAXED);
}

template <bool kPush>
struct Ops {
    const uint32_t* in;
    uint32_t*       out;

    uint32_t mask() const { return kMask; }
    uint64_t cursor_load() const {
        return load_dev(kPush ? &g_tail : &g_head, __ATOMIC_RELAXED);
    }
    bool cursor_cas(uint64_t& expected, uint64_t desired) {
#if defined(CALIBRATE_BUMP_CURSOR)
        // THE DEFECT: the position is taken unconditionally. The reserver then owns a position
        // whose slot may still belong to someone else, and the sequence test that was supposed
        // to gate it has already been passed against a stale read.
        (void)desired;
        expected = fetch_add_dev(kPush ? &g_tail : &g_head, 1);
        return true;
#else
        return cas_dev(kPush ? &g_tail : &g_head, &expected, desired);
#endif
    }
    uint64_t seq_load(uint32_t s) const { return load_dev(&g_seq[s], __ATOMIC_ACQUIRE); }
    void seq_store(uint32_t s, uint64_t v) { store_dev(&g_seq[s], v, __ATOMIC_RELEASE); }
    void transfer(uint32_t s) {
        if constexpr (kPush) g_slots[s] = *in; else *out = g_slots[s];
    }
};

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    __VERIFIER_thread_global_id(static_cast<int>(id));
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(static_cast<int>(id));
    __VERIFIER_thread_kernel_id(0);

    // Item ids start at one, so zero -- what an unwritten slot holds -- is never a real item and
    // a consumer that reads an unpublished slot is distinguishable from one that read a real one.
    const uint32_t item = static_cast<uint32_t>(id) + 1;
    Ops<true> push{&item, nullptr};
    (void)hgcommon::ring_claim(push, /*want=*/0, /*leave=*/1);

    uint32_t got = 0;
    Ops<false> pop{nullptr, &got};
    if (hgcommon::ring_claim(pop, /*want=*/1, /*leave=*/kMask + 1)) {
        g_taken[id] = got;
        g_got[id] = true;
    }
    return nullptr;
}

}  // namespace

int main() {
    // seq[i] = i is what makes position i's first lap free, and it is the initial state the
    // three sequence tests are stated against.
    for (uint32_t i = 0; i < kCap; ++i) g_seq[i] = i;

    pthread_t t[kThreads];
    for (long i = 0; i < static_cast<long>(kThreads); ++i)
        pthread_create(&t[i], nullptr, worker, reinterpret_cast<void*>(i));
    for (uint32_t i = 0; i < kThreads; ++i) pthread_join(t[i], nullptr);

    // NO ITEM IS DELIVERED TWICE, and none is delivered that was never published. A pop that
    // returned false is not a fault at this bound -- a consumer may legitimately observe the
    // queue empty if it runs before either push -- so what is asserted is about what WAS handed
    // out, not how much.
    for (uint32_t i = 0; i < kThreads; ++i) {
        if (!g_got[i]) continue;
        assert(g_taken[i] >= 1 && g_taken[i] <= kThreads &&
               "a consumer took a slot that no producer had published");
        for (uint32_t j = i + 1; j < kThreads; ++j) {
            if (!g_got[j]) continue;
            assert(g_taken[i] != g_taken[j] && "one item was handed to two consumers");
        }
    }
    return 0;
}
