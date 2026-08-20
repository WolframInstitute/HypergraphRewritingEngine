// GenMC harness: a child never strands at a stale depth when its parent is relaxed concurrently.
//
// WHAT IS BEING PROVED. A run's budget applies to the SHORTEST path from an initial state to a
// canonical state, not to the path the state was first reached along. So a parent whose depth
// falls must pull its children down with it, and a child registering itself must learn its
// parent's CURRENT minimum rather than whatever it happened to read. Those two things race:
//
//   registrar (execute_rewrite_task)            relaxer (propagate_explore_depth)
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
// WHY IT NEEDS CHECKING RATHER THAN READING. The two fences make this a Dekker store-load pair,
// and the argument that a seq_cst pair forbids the double miss is exactly the argument that is
// easy to state and easy to get wrong: it holds only if BOTH sides fence BETWEEN their store and
// their load, and only if the store and the load are on the same locations the peer touches.
// The engine's registrar pushes to a LockFreeList and loads a plain atomic; the relaxer stores
// that atomic and walks the list. Those are different mechanisms on the two sides, which is what
// makes the pairing worth checking against the memory model rather than asserting.
//
// THIS PROTOCOL HAS ALREADY FAILED ONCE, which is why it earns a harness rather than a comment.
// fb884d7 fixed a quotient run that was nondeterministic and incomplete under a truncated budget
// because a child's depth came from the parent's CLAIM depth instead of its live minimum. The
// fix added explore_depth_of and these fences, and was gated by a regression test in
// test_feature_matrix.cpp -- a test that samples schedules, over a property about all of them.
//
// TRANSCRIBED, and the reason is stated rather than glossed. The protocol lives inside
// ParallelEvolutionEngine methods that need a Hypergraph, a job system and an arena to call at
// all, so the harness cannot include the engine's own function the way the container harnesses
// include theirs. The CHILD LIST is the real LockFreeList; the depth is a plain atomic, which is
// what the engine uses. What is transcribed is the ordering, and it is transcribed line for line
// from the two sites named above.
//
// WHAT IS BOUNDED. Two threads, one parent, one child, one relaxation. A statement about every
// execution of THIS program under RC11, not about unbounded worker counts.
//
// CALIBRATION -- the harness must be able to fail. Removing either fence must make it report the
// double miss. Checked both ways before this was committed.
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
#include "hypergraph/lock_free_list.hpp"

namespace {

using List = hypergraph::LockFreeList<uint64_t>;

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

constexpr uint64_t kChild      = 7;
constexpr uint32_t kOldDepth   = 5;   // what the parent's depth was
constexpr uint32_t kNewDepth   = 2;   // what the relaxation lowers it to

List* g_children;                     // canon_children_[parent]
std::atomic<uint32_t>* g_parent_depth;
StubArena* g_arena;

// What each side observed. The property is over the CONJUNCTION of the two misses.
std::atomic<int> g_scan_saw_child{0};
std::atomic<int> g_read_saw_lowered{0};

// execute_rewrite_task: publish the child into the parent's list, fence, then read the parent's
// live minimum depth. The child's own depth is derived from what this read returns.
void* registrar(void*) {
    g_children->push(kChild, *g_arena);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const uint32_t d = g_parent_depth->load(std::memory_order_relaxed);
    if (d == kNewDepth) g_read_saw_lowered.store(1, std::memory_order_relaxed);
    return nullptr;
}

// propagate_explore_depth: lower the parent's depth, fence, then scan the child list so every
// child already registered is pulled down with it.
void* relaxer(void*) {
    g_parent_depth->store(kNewDepth, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    g_children->for_each([&](uint64_t c) {
        if (c == kChild) g_scan_saw_child.store(1, std::memory_order_relaxed);
    });
    return nullptr;
}

}  // namespace

int main() {
    StubArena arena;
    List children;
    std::atomic<uint32_t> parent_depth{kOldDepth};

    g_arena = &arena;
    g_children = &children;
    g_parent_depth = &parent_depth;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, registrar, nullptr);
    pthread_create(&t1, nullptr, relaxer, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // The child learns its parent's true minimum if EITHER side saw the other: the registrar
    // read the lowered depth itself, or the relaxer's scan found the child and lowered it. The
    // forbidden state is neither.
    assert(g_scan_saw_child.load(std::memory_order_relaxed) ||
           g_read_saw_lowered.load(std::memory_order_relaxed));
    return 0;
}
