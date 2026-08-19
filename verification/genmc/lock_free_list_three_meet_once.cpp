// GenMC harness: THREE concurrent pushers meet pairwise exactly once.
//
// lock_free_list_pairs_meet_once bounds TWO pushers: of any two nodes one is older, so exactly
// one scan sees the other. That argument is about a PAIR, and it is the argument the branchial
// relation rests on -- but the relation is built over instances carrying MANY applications, and
// each scan runs while other pushes are still landing. Two threads cannot exhibit a scan that
// runs between two other pushes; three can.
//
// WHAT IS BEING PROVED. Each thread pushes one node and then visits the nodes linked strictly
// before its own, counting how many of the other two it sees. Summed over the three threads, the
// total must be exactly 3 = C(3,2): every unordered pair reported once and once only.
//
//   fewer than 3  a pair that never met, which is a branchial edge that does not exist
//   more than 3   a pair reported twice, which the removed dedup set used to absorb
//
// WHY IT MATTERS HERE. Nothing downstream deduplicates any more: the count is incremented per
// emission and the relation is re-derived by the same rule at read time, so this rule alone
// decides both. A miss or a double at three pushers would move the branchial pair count while
// leaving the application and event counts untouched -- which is the exact shape of the
// intermittent failure this is being read for.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh lock_free_list_three_meet_once

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <new>

#include "genmc_support.hpp"
#include "hypergraph/lock_free_list.hpp"

namespace {

using List = hypergraph::LockFreeList<uint64_t>;

struct StubArena {
    static constexpr int kCap = 8;
    alignas(16) unsigned char storage[kCap * 64];
    std::atomic<int> next{0};
    template <typename T, typename... Args>
    T* create(Args&&... args) {
        const int i = next.fetch_add(1, std::memory_order_relaxed);
        assert(i < kCap && sizeof(T) <= 64);
        return new (storage + i * 64) T(static_cast<Args&&>(args)...);
    }
};

// Distinct values, so a sighting names which pusher was seen.
constexpr uint64_t kVal[3] = {11, 22, 33};

List*      g_list;
StubArena* g_arena;
std::atomic<unsigned> g_saw[3];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    List::Node* at = g_list->push(kVal[id], *g_arena);

    unsigned seen = 0;
    g_list->for_each_before(at, [&](uint64_t v) {
        if (v != kVal[id]) ++seen;      // one of the other two
    });
    g_saw[id].store(seen, std::memory_order_relaxed);
    return nullptr;
}

}  // namespace

int main() {
    StubArena arena;
    List      list;
    g_arena = &arena;
    g_list  = &list;
    for (int i = 0; i < 3; ++i) g_saw[i].store(0, std::memory_order_relaxed);

    pthread_t t0, t1, t2;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_create(&t2, nullptr, worker, reinterpret_cast<void*>(2L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    const unsigned total = g_saw[0].load(std::memory_order_relaxed) +
                           g_saw[1].load(std::memory_order_relaxed) +
                           g_saw[2].load(std::memory_order_relaxed);

    // EXACTLY C(3,2). The three nodes are totally ordered by the order their CASes landed, so
    // each of the three pairs is seen by exactly one of them -- the later one.
    assert(total == 3);
    return 0;
}
