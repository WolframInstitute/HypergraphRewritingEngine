// GenMC harness: a child inherits its parent's matches exactly once, and sees all of them.
//
// WHAT IS BEING PROVED. Under match forwarding a child takes its parent's finished match list at
// the parent's drain. Neither side exists first, so both publish and then scan for the other
// (hgcommon::rendezvous<rv::ChildInheritance>):
//
//   child side (register_child_with_parent)         parent side (the drain, note_match_task_done)
//     get or create the parent's children list        store drained = 1 (release)
//     push the child                                  seq_cst fence
//     seq_cst fence                                   look up the children list and walk it
//     read drained (acquire); if set, inherit         inherit for every child it finds
//
// Every inheritance first claims the child's `inherited` flag, so of two sightings one is a no-op.
// Two properties, under RC11:
//   ONCE:     exactly one inheritance runs. Both scans missing leaves the child without the
//             parent's matches; the claim is what keeps two sightings from applying them twice.
//   COMPLETE: the inheritance sees every match the parent stored before draining. The parent
//             stores into its list and then publishes `drained`; a child that reads `drained`
//             set must read the list with those stores visible.
//
// verification/tla/MatchForwarding.tla checks the protocol under sequential consistency; this
// runs the engine's hgcommon::rendezvous and memory orders over the REAL ConcurrentMap and
// LockFreeList.
//
// WHAT IS BOUNDED. Two threads, one parent with one stored match, one child.
//
// CALIBRATION. -DCALIBRATE_NO_FENCE replaces hgcommon::rendezvous with publish(); scan(); and no
// fence: each side can then read the other as absent and no inheritance runs. -DCALIBRATE_RELAXED_DRAINED stores and loads `drained`
// relaxed: a child that sees it set need not see the parent's stored match.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh child_inheritance_rendezvous

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <new>

#include "genmc_support.hpp"
#include "hgcommon/rendezvous.hpp"
#include "hypergraph/concurrent_map.hpp"
#include "hypergraph/lock_free_list.hpp"

namespace {

using List = hypergraph::LockFreeList<uint64_t>;
using Map  = hypergraph::ConcurrentMap<uint64_t, List*>;

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

constexpr uint64_t kParent = 0x71ull;   // the parent's key in the children map
constexpr uint64_t kChild  = 5;
constexpr uint64_t kMatch  = 9;

#if defined(CALIBRATE_RELAXED_DRAINED)
constexpr std::memory_order kStore = std::memory_order_relaxed;
constexpr std::memory_order kLoad  = std::memory_order_relaxed;
#else
constexpr std::memory_order kStore = std::memory_order_release;
constexpr std::memory_order kLoad  = std::memory_order_acquire;
#endif

template <class Publish, class Scan>
void meet(Publish&& publish, Scan&& scan) {
#if defined(CALIBRATE_NO_FENCE)
    publish();
    scan();
#else
    hgcommon::rendezvous<hgcommon::rv::ChildInheritance>(publish, scan);
#endif
}

Map*  g_children;          // state_children_ : parent -> list of children
List* g_child_list;        // the list the child side creates for the parent
List* g_parent_matches;    // state_matches_[parent]
StubArena* g_arena;
std::atomic<uint32_t> g_drained{0};     // MatchJoin::drained of the parent
std::atomic<uint32_t> g_inherited{0};   // MatchJoin::inherited of the child
std::atomic<int> g_inherits{0};         // inheritances that passed the claim
std::atomic<int> g_saw_match{0};        // what the winning inheritance read

void inherit() {
    uint32_t unclaimed = 0;
    if (!g_inherited.compare_exchange_strong(unclaimed, 1u, std::memory_order_acq_rel)) return;
    g_inherits.fetch_add(1, std::memory_order_relaxed);
    g_parent_matches->for_each([&](uint64_t m) {
        if (m == kMatch) g_saw_match.store(1, std::memory_order_relaxed);
    });
}

// register_child_with_parent: get or create the children list, then push / read drained.
void* child_side(void*) {
    auto ins = g_children->insert_if_absent(kParent, g_child_list);
    List* kids = ins.second ? g_child_list : ins.first;
    bool drained = false;
    meet([&] { kids->push(kChild, *g_arena); },
         [&] { drained = g_drained.load(kLoad) != 0; });
    if (drained) inherit();
    return nullptr;
}

// The parent: store its match, then drain: store drained / look up the children list.
void* parent_side(void*) {
    g_parent_matches->push(kMatch, *g_arena);
    List* kids = nullptr;
    meet([&] { g_drained.store(1, kStore); },
         [&] { if (auto r = g_children->lookup(kParent)) kids = *r; });
    if (kids) kids->for_each([&](uint64_t c) { if (c == kChild) inherit(); });
    return nullptr;
}

}  // namespace

int main() {
    StubArena arena;
    Map children(8);
    List child_list, parent_matches;
    g_arena = &arena;
    g_children = &children;
    g_child_list = &child_list;
    g_parent_matches = &parent_matches;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, child_side, nullptr);
    pthread_create(&t1, nullptr, parent_side, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // ONCE: the child inherited, and only once.
    assert(g_inherits.load(std::memory_order_relaxed) == 1);
    // COMPLETE: the inheritance saw the match the parent stored before it drained.
    assert(g_saw_match.load(std::memory_order_relaxed) == 1);
    return 0;
}
