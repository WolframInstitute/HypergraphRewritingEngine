// GenMC harness: two concurrent pushers MEET EXACTLY ONCE under push + for_each_before.
//
// WHAT IS BEING PROVED. Every application that publishes itself to an instance's applied list
// then scans that list for siblings. The pair {A,B} must be reported once: zero times loses a
// branchial edge, twice reports one edge as two. The rule that gives it is that a scan visits
// only the nodes linked STRICTLY BEFORE its own -- of any two nodes exactly one is older, so
// exactly one of the two scans sees the other, whatever order the threads ran in.
//
// This is a different property from lock_free_list_completeness, which bounds a walk from the
// HEAD after the pushers have joined and says every node is visited once. That says nothing
// about two walks running CONCURRENTLY WITH the pushes and each seeing a different prefix,
// which is the situation here and the one the pairing depends on.
//
// THE DEFECT IT EXISTS TO EXCLUDE, and it is the code this replaced. Scanning from the head
// instead of from one's own node makes both pushers see each other whenever both pushes land
// before either scan, so the pair is emitted twice. That was survivable only because a set of
// pair keys deduplicated it afterwards -- 133,218,996 keys at ~1.07 GB on disc-l3a2g2r2 depth
// 3, and the device's 2^22 ceiling. Removing the set makes this rule load-bearing: nothing
// downstream can absorb a double report any more.
//
// WHAT IS BOUNDED. Two threads, one push each, each scanning the nodes before its own and
// counting how many of the OTHER thread's values it sees. Their two counts must sum to exactly
// one. This is a statement about every execution of THIS program under RC11, not about
// unbounded thread counts.
//
// CALIBRATION -- the harness must be able to fail. Replacing for_each_before(mine, ...) with
// for_each(...) -- the walk from the head -- must make this report the violation, because the
// interleaving where both pushes precede both scans is then a sum of two. A harness that
// cannot fail proves nothing.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh lock_free_list_pairs_meet_once

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <new>

#include "genmc_support.hpp"
#include "hypergraph/lock_free_list.hpp"

namespace {

using List = hypergraph::LockFreeList<uint64_t>;

// Same stub allocator, and the same scope, as lock_free_list_completeness: exclusive by
// construction, so this harness bounds the list's linkage and not the arena's.
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
using Arena = StubArena;

constexpr uint64_t kA = 11, kB = 21;

List*  g_list;
Arena* g_arena;
// How many of the OTHER thread's values each thread saw among the nodes older than its own.
std::atomic<unsigned> g_saw[2];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    const uint64_t mine  = (id == 0) ? kA : kB;
    const uint64_t other = (id == 0) ? kB : kA;

    List::Node* at = g_list->push(mine, *g_arena);

    unsigned seen = 0;
    g_list->for_each_before(at, [&](uint64_t v) { if (v == other) ++seen; });
    g_saw[id].store(seen, std::memory_order_relaxed);
    return nullptr;
}

}  // namespace

int main() {
    Arena arena;
    List  list;
    g_arena = &arena;
    g_list  = &list;
    g_saw[0].store(0, std::memory_order_relaxed);
    g_saw[1].store(0, std::memory_order_relaxed);

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    const unsigned total = g_saw[0].load(std::memory_order_relaxed) +
                           g_saw[1].load(std::memory_order_relaxed);

    // EXACTLY ONE, in both directions. Zero is a lost pair: neither saw the other, which the
    // prev chain forbids because the later push links over the earlier node. Two is the pair
    // reported twice, which is what a walk from the head produces.
    assert(total == 1);
    return 0;
}
