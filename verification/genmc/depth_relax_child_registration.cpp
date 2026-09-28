// GenMC harness: a child never strands at a stale depth when its parent is relaxed concurrently.
//
// WHAT IS BEING PROVED. A run's budget applies to the SHORTEST path from an initial state to a
// canonical state, not to the path the state was first reached along. So a parent whose depth
// falls must pull its children down with it, and a child registering itself must learn its
// parent's CURRENT minimum rather than whatever it happened to read. Those two things race:
//
//   registrar (explore_register_child)          relaxer (explore_relax)
//     push the child into the parent's list       lower the parent's depth
//     seq_cst fence                               seq_cst fence
//     read the parent's depth                     scan the parent's child list
//
// The outcome that must not exist is BOTH misses: the scan does not see the child AND the read
// does not see the lowered depth. The child is then stranded at a depth its parent no longer
// has, and nothing revisits it -- relaxation is driven by the store that already happened. What
// that costs is expansion decided by which side won a race, which is the one thing
// Section "Determinism Contract" says the observable output never depends on.
//
// WHAT IS DRIVEN. hgcommon/explore_depth_core.hpp, the body both engines call, over the host's
// face: the host LockFreeList for the child list, atomics for the depths, the seq_cst fence
// hgcommon::rendezvous_barrier issues. verification/gpumc/depth_relax_child_registration.cpp
// drives the same core over the device's list and fences.
//
// THE PROPERTY. The child ends at one past the parent's lowered depth, whichever side wins.
//
// WHAT IS BOUNDED. Two threads, one parent, one child, one relaxation. A statement about every
// execution of THIS program under RC11, not about unbounded worker counts.
//
// CALIBRATION. -DCALIBRATE_NO_FENCE makes the Ctx's fence a no-op, and the checker must report
// the child stranded at the parent's old depth plus one.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh depth_relax_child_registration

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <new>

#include "genmc_support.hpp"
#include "hgcommon/explore_depth_core.hpp"
#include "hypergraph/lock_free_list.hpp"

namespace {

using List = hypergraph::LockFreeList<uint32_t>;

// One slot per call, no reuse. The arena's own disjointness is a separate property with its own
// harness (arena_worker_index_exclusive).
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

constexpr uint32_t kParent    = 0;
constexpr uint32_t kChild     = 1;
constexpr uint32_t kOldDepth  = 5;   // the parent's depth when the child arrives
constexpr uint32_t kNewDepth  = 2;   // what the relaxation lowers it to

std::atomic<uint32_t> g_depth[2];
List* g_children[2];
StubArena* g_arena;

// ParallelEvolutionEngine::ExploreCtx, with a fixed frame store and no expansion.
struct Ctx {
    using Node = const List::Node*;
    Node     frame_at[4];
    uint32_t frame_depth[4];
    uint32_t frames = 0;

    uint32_t depth_load(uint32_t s) const { return g_depth[s].load(std::memory_order_acquire); }
    bool depth_cas(uint32_t s, uint32_t& e, uint32_t d) {
        return g_depth[s].compare_exchange_weak(e, d, std::memory_order_acq_rel,
                                                std::memory_order_acquire);
    }
    void children_push(uint32_t p, uint32_t c) { g_children[p]->push(c, *g_arena); }
    Node children_head(uint32_t s) const { return g_children[s]->head_node(); }
    static bool children_end(Node n) { return n == nullptr; }
    static uint32_t children_value(Node n) { return n->value; }
    static Node children_next(Node n) { return n->prev; }
    void fence() const {
#if !defined(CALIBRATE_NO_FENCE)
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    }
    void admit(uint32_t, uint32_t) {}
    bool frame_push(Node at, uint32_t d) {
        if (frames == 4) return false;
        frame_at[frames] = at; frame_depth[frames] = d; ++frames;
        return true;
    }
    bool frame_top(Node*& at, uint32_t& d) {
        if (frames == 0) return false;
        at = &frame_at[frames - 1]; d = frame_depth[frames - 1];
        return true;
    }
    void frame_pop() { --frames; }
};

// The rewrite that creates the child: register it, and walk from it if it was lowered.
void* registrar(void*) {
    Ctx c;
    const uint32_t d = hgcommon::explore_register_child(c, kParent, kChild, kOldDepth + 1u);
    if (d != hgcommon::kExploreNoDepth) hgcommon::explore_relax(c, kChild, d);
    return nullptr;
}

// A shorter path to the parent: lower it and walk its children.
void* relaxer(void*) {
    Ctx c;
    if (hgcommon::explore_try_lower(c, kParent, kNewDepth))
        hgcommon::explore_relax(c, kParent, kNewDepth);
    return nullptr;
}

}  // namespace

int main() {
    StubArena arena;
    List parent_children, child_children;
    g_arena = &arena;
    g_children[kParent] = &parent_children;
    g_children[kChild] = &child_children;
    g_depth[kParent].store(kOldDepth, std::memory_order_relaxed);
    g_depth[kChild].store(hgcommon::kExploreNoDepth, std::memory_order_relaxed);

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, registrar, nullptr);
    pthread_create(&t1, nullptr, relaxer, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    assert(g_depth[kChild].load(std::memory_order_relaxed) == kNewDepth + 1u &&
           "the child stranded at a depth its parent no longer has");
    return 0;
}
