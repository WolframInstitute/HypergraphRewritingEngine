// GenMC harness: the quotient replay's (instance, match) rendezvous never drops a pair.
//
// WHAT IS BEING PROVED, and why nothing proved it. Under quotient exploration every raw event is
// reconstructed by applying a captured MATCH to an INSTANCE, and neither side exists first. Both
// sides therefore publish and then scan for the other:
//
//   instance side (Hypergraph::qc_add_instance)      match side (Hypergraph::qc_capture_expansion)
//     insert the instance list into qc_instances_      insert the match list into qc_expansion_
//     push the instance                                push the match
//     seq_cst fence                                    seq_cst fence
//     look up qc_expansion_ and scan it                look up qc_instances_ and scan it
//
// If BOTH scans miss, the pair is never applied: one fewer raw event, and with it every causal
// and branchial pair that event belonged to. The canonical state and event counts are untouched,
// because the state was still explored -- so the loss is invisible to every count a caller reads.
//
// claim_match_rendezvous models the MATCH-DEDUP rendezvous in parallel_evolution.hpp. This one
// is a different rendezvous in a different file over two different maps, and it had no harness.
//
// WHY THE FENCES ARE NOT OBVIOUSLY ENOUGH. The scan does not read the peer's list directly: it
// reaches it through a ConcurrentMap lookup, and that lookup answers ABSENT for an entry whose
// value has been claimed but not yet settled. So a scan can miss its peer for a reason the
// fences say nothing about, and the informal argument -- a lookup that misses implies the peer
// has not published, so the peer's own scan runs later and catches it -- chains a liveness claim
// onto an ordering one. That is the argument this harness exists to check rather than believe.
//
// WHAT IS BOUNDED. Two threads, one class, one instance and one match, over the REAL
// ConcurrentMap and the REAL LockFreeList. A statement about every execution of THIS program
// under RC11, not about unbounded thread counts.
//
// CALIBRATION -- the harness must be able to fail. Removing either seq_cst fence must make it
// report: the two publishes and the two scans then interleave so that each scan runs before the
// other's publish is visible. A harness that cannot fail proves nothing.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh quotient_instance_match_rendezvous

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <new>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"
#include "hypergraph/lock_free_list.hpp"

namespace {

using List = hypergraph::LockFreeList<uint64_t>;
using Map  = hypergraph::ConcurrentMap<uint64_t, List*>;

// Exclusive by construction, as in the other list harnesses: one slot per call, no reuse. The
// arena's own disjointness is a separate property with its own harness.
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

constexpr uint64_t kClass = 0x51ull;   // the one class both sides key on
constexpr uint64_t kInst  = 11;
constexpr uint64_t kMatch = 22;

Map*  g_instances;   // qc_instances_ : class -> list of instances
Map*  g_matches;     // qc_expansion_ : class -> list of matches
List* g_inst_list;
List* g_match_list;
StubArena* g_arena;

// Did each side see its peer? The pair is applied if EITHER did -- qc_apply's per-pair claim
// makes a double sighting harmless, so the property is "at least one", not "exactly one".
std::atomic<int> g_saw_match{0};
std::atomic<int> g_saw_instance{0};

// The instance side. Publish the list, publish the instance, fence, then scan for matches.
void* instance_side(void*) {
    g_instances->insert_if_absent(kClass, g_inst_list);
    g_inst_list->push(kInst, *g_arena);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (auto r = g_matches->lookup(kClass)) {
        (*r)->for_each([&](uint64_t v) { if (v == kMatch) g_saw_match.store(1, std::memory_order_relaxed); });
    }
    return nullptr;
}

// The match side. Same shape, opposite maps.
void* match_side(void*) {
    g_matches->insert_if_absent(kClass, g_match_list);
    g_match_list->push(kMatch, *g_arena);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (auto r = g_instances->lookup(kClass)) {
        (*r)->for_each([&](uint64_t v) { if (v == kInst) g_saw_instance.store(1, std::memory_order_relaxed); });
    }
    return nullptr;
}

}  // namespace

int main() {
    StubArena arena;
    Map instances(8), matches(8);
    List inst_list, match_list;

    g_arena = &arena;
    g_instances = &instances;
    g_matches = &matches;
    g_inst_list = &inst_list;
    g_match_list = &match_list;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, instance_side, nullptr);
    pthread_create(&t1, nullptr, match_side, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // THE PAIR IS APPLIED. Both missing is one raw event that never happens, and with it every
    // causal and branchial pair it belonged to -- while the state and canonical event counts a
    // caller reads stay exactly as they were.
    assert(g_saw_match.load(std::memory_order_relaxed) == 1 ||
           g_saw_instance.load(std::memory_order_relaxed) == 1);
    return 0;
}
