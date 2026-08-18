// GenMC harness: LockFreeList::for_each visits EVERY pushed node, exactly once.
//
// WHAT IS BEING PROVED, and why it is a completeness property. Every other harness here bounds
// EXCLUSIVITY -- at most one caller wins a claim (g_ins[0] != g_ins[1], g_won[0] != g_won[1],
// at most one consumer takes the item). None of them bounds the opposite direction on a list:
// that a walk yields everything that was pushed. The two failure modes are not the same defect
// and do not have the same fix, which key_set_enumeration already records for the key set --
// its first attempt emitted 79,215 keys for 30,063 distinct ones, and its second reported one
// key inserted twice.
//
// THE DEFECT IT EXISTS TO EXCLUDE. A node that was pushed and is not reachable from head_ when
// the walk runs. push is a Treiber push: prev is written, then a release CAS publishes the node.
// If a node's prev does not hold the head that its own CAS displaced -- a lost update, a stale
// prev after a retry, or two pushers handed the same node by the arena -- then every node below
// the break is orphaned, the walk is short, and nothing reports it. The counter beside the list
// still counts the push, so the loss is silent at the call site.
//
// WHY IT IS NOT HYPOTHETICAL. branchial_edges_ is a LockFreeList, and add_branchial_edge pushes
// one node per winning claim and increments num_branchial_edges_. Measured on WPP at 8 threads:
// 30,063 distinct pairs claimed, 30,063 edges stored by that counter, and 24,580 edges returned
// by the walk -- 5,483 nodes pushed and not visited. This structure had NO harness at all when
// that was found.
//
// WHAT IS BOUNDED. Two pushers, two pushes each, one walk after both join. Distinct values, so
// the walk's multiset is fully determined: four values, each exactly once, and nothing else.
// This is a statement about every execution of THIS program under RC11, not about unbounded
// thread counts or list lengths.
//
// CALIBRATION -- the harness must be able to fail. Replacing the CAS loop in push with a
// non-retrying publish (write prev once from a stale head_ load, then an unconditional store to
// head_) must make this harness report the safety violation. A harness that cannot fail proves
// nothing, and this one asserts a property no existing harness states.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh lock_free_list_completeness

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <new>

#include "genmc_support.hpp"
#include "hypergraph/lock_free_list.hpp"

namespace {

using List = hypergraph::LockFreeList<uint64_t>;

// A STUB ALLOCATOR, AND THE SCOPE THAT BUYS. The real arena reaches aligned operator new, which
// the checker cannot execute, so push is given a bump allocator that is exclusive BY
// CONSTRUCTION: one atomic fetch_add, one slot per call, no reuse.
//
// That is deliberately a precondition rather than a thing under test. This harness therefore
// bounds the LIST'S OWN LOGIC -- prev linkage and the walk -- given distinct nodes. Whether the
// arena in fact hands distinct nodes to concurrent callers is a separate property with its own
// harness (arena_disjoint_allocation). Proving them together in one harness would leave a
// failure unable to say which half was wrong, which is the argument key_set_enumeration already
// makes for keeping enumeration apart from the insert verdict.
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

// Distinct per thread, so a missing value names which pusher lost it rather than only that the
// count is wrong.
constexpr uint64_t kA0 = 11, kA1 = 12;
constexpr uint64_t kB0 = 21, kB1 = 22;

List*  g_list;
Arena* g_arena;

void* pusher(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    // TWO pushes per thread, not one: a single push each cannot produce a chain where one
    // node's prev must span another thread's node, which is the shape that orphans a tail.
    g_list->push(id == 0 ? kA0 : kB0, *g_arena);
    g_list->push(id == 0 ? kA1 : kB1, *g_arena);
    return nullptr;
}

}  // namespace

int main() {
    Arena arena;
    List  list;
    g_arena = &arena;
    g_list  = &list;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, pusher, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, pusher, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // Both pushers have joined, so the list is quiescent and the walk is over a fixed structure.
    // Any shortfall here is a node that was pushed and cannot be reached, not a race with a
    // push still in flight.
    unsigned n_a0 = 0, n_a1 = 0, n_b0 = 0, n_b1 = 0, n_other = 0;
    list.for_each([&](uint64_t v) {
        if      (v == kA0) ++n_a0;
        else if (v == kA1) ++n_a1;
        else if (v == kB0) ++n_b0;
        else if (v == kB1) ++n_b1;
        else               ++n_other;
    });

    // EXACTLY ONCE, in both directions. Zero is a lost node -- the defect measured on
    // branchial_edges_. More than one is a cycle or a re-emitted prefix, which the walk's
    // seen_up_to bound is what prevents.
    assert(n_a0 == 1);
    assert(n_a1 == 1);
    assert(n_b0 == 1);
    assert(n_b1 == 1);
    assert(n_other == 0);
    return 0;
}
