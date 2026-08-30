// GenMC harness: the tagged free-list hands every node to exactly one popper, across the
// speculative link read.
//
// WHAT IS BEING PROVED. hgcommon/pool_core.hpp is the claim rule of the arena's process-wide
// block pool: a popper reads the head, dereferences the CANDIDATE's link, and only then runs
// the CAS that makes the candidate its own -- so the link read races in TIME with a rival
// winning the node and rewriting that link (or handing it to an owner who does). The tag makes
// the stale value harmless; the accessors' atomicity makes the read DEFINED. ThreadSanitizer
// reported the plain-access version of exactly this against the shipped pool (CI tsan leg,
// 30/08), and the pool had no checker because its production caller is wrapped in mmap
// machinery a checker does not model. This harness runs the extracted rule itself.
//
// THE PROPERTY. Two workers pop, mark ownership, release ownership, and push back, twice
// each over a two-node pool. Ownership is a per-node CAS 0 -> tid asserted to succeed: a node
// popped by two threads at once fails it. After the joins the pool drains to exactly the two
// nodes, each exactly once.
//
// CALIBRATION. -DCALIBRATE_PLAIN_LINK compiles the link accessors as plain loads and stores;
// GenMC must report the data race the atomic accessors exist to remove. The harness is only
// evidence if that arm fails.
#include "hgcommon/pool_core.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

constexpr uint32_t kNodes   = 2;   // node ids 1..kNodes; 0 is the empty list
constexpr uint32_t kWorkers = 2;
constexpr uint32_t kRounds  = 2;

uint64_t g_head;
uint64_t g_link[kNodes + 1];
uint64_t g_owner[kNodes + 1];

struct Ops {
    uint64_t head_load() { return __atomic_load_n(&g_head, __ATOMIC_ACQUIRE); }
    bool head_cas(uint64_t& expected, uint64_t desired) {
        return __atomic_compare_exchange_n(&g_head, &expected, desired, /*weak=*/true,
                                           __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
#if defined(CALIBRATE_PLAIN_LINK)
    uint64_t link_load(uint64_t node) { return g_link[node]; }
    void link_store(uint64_t node, uint64_t v) { g_link[node] = v; }
#else
    uint64_t link_load(uint64_t node) {
        return __atomic_load_n(&g_link[node], __ATOMIC_RELAXED);
    }
    void link_store(uint64_t node, uint64_t v) {
        __atomic_store_n(&g_link[node], v, __ATOMIC_RELAXED);
    }
#endif
};

void* worker(void* arg) {
    const uint64_t tid = reinterpret_cast<uint64_t>(arg);
    Ops ops;
    for (uint32_t r = 0; r < kRounds; ++r) {
        const uint64_t node = hgcommon::pool_core_pop(ops);
        if (node == 0) continue;
        uint64_t expected = 0;
        const bool owned = __atomic_compare_exchange_n(&g_owner[node], &expected, tid,
                                                       /*weak=*/false,
                                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        assert(owned && "a node was handed to two poppers at once");
        __atomic_store_n(&g_owner[node], 0, __ATOMIC_RELEASE);
        hgcommon::pool_core_push(ops, node);
    }
    return nullptr;
}

}  // namespace

int main() {
    Ops ops;
    for (uint64_t n = 1; n <= kNodes; ++n) hgcommon::pool_core_push(ops, n);

    pthread_t t[kWorkers];
    for (uint64_t i = 0; i < kWorkers; ++i)
        pthread_create(&t[i], nullptr, worker, reinterpret_cast<void*>(i + 1));
    for (uint32_t i = 0; i < kWorkers; ++i) pthread_join(t[i], nullptr);

    // Inventory: every node back, each exactly once.
    uint32_t seen[kNodes + 1] = {0};
    uint32_t drained = 0;
    while (uint64_t node = hgcommon::pool_core_pop(ops)) {
        assert(node >= 1 && node <= kNodes);
        assert(seen[node] == 0 && "a node came back twice");
        seen[node] = 1;
        ++drained;
    }
    assert(drained == kNodes && "a node was lost");
    return 0;
}
