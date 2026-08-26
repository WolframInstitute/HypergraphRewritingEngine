// GenMC harness: no element of a worker's deque is taken twice, and none is lost.
//
// WHY THIS EXISTS. Every worker owns a Chase-Lev deque: it pushes and pops its own bottom, and
// idle workers steal the top. The two ends are lock-free and independent, and each carries a
// seq_cst fence -- pop stores bottom then reads top, steal reads top then reads bottom -- placed
// so neither end can observe the other as absent. Those fences are the algorithm's correctness
// argument, and nothing was checking that they are load-bearing.
//
// Nothing checked this. `deque_no_double_extraction` checks `lockfree::Deque`, which is the shared
// INJECTOR -- a different structure with a different algorithm (a packed index word with a tag)
// -- and no harness included work_stealing_deque.hpp at all. The per-worker deque is the one on
// the hot path: every job a worker runs comes out of pop(), and every steal in find_work comes out
// of steal().
//
// TWO ELEMENTS AND TWO THIEVES, NOT ONE OF EACH, and the reason is the whole point of the
// harness. With a single element the two ends contest the SAME slot and both go through a
// compare-exchange on top, so exactly one wins whatever the fences do -- a one-element harness
// passes with the fences deleted, which was checked before this one was written. The hazard the
// fences exist for is pop's FAST PATH, where the owner takes an element with no CAS at all:
//
//   thief A steals slot 0, advancing top to 1
//   owner pops: stores bottom = 1, and reads top. Reading it STALE as 0 gives 0 < 1, so the
//     owner takes slot 1 by the fast path -- no compare-exchange
//   thief B steals: reads top = 1, and reads bottom. Reading it STALE as 2 gives 1 < 2, so it
//     takes slot 1 and its CAS on top succeeds
//
// Both took slot 1. Each fence forbids one half of that: pop's orders its bottom store before its
// top load, steal's orders its top load before its bottom load.
//
// WHAT IS BOUNDED. Three threads, capacity 4, two items pushed single-threaded before any starts,
// one attempt each. Concurrent push against steal is a different question -- push publishes the
// slot then moves bottom, and a thief reading the old bottom cannot see the slot -- and belongs in
// its own harness. This is a statement about every execution of THIS program under RC11 at this
// bound, not about unbounded thread counts.
//
// CALIBRATED, which is what makes the bound worth stating: deleting either seq_cst fence must
// make this report the violation, and it does.
//
// GENMC-EXPECT: pass

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "job_system/work_stealing_deque.hpp"

namespace {

// Stored inline in its slot, so the deque touches no allocator here and the harness measures the
// index algorithm rather than the boxing path -- the same choice its injector sibling makes.
int g_a = 11;
int g_b = 22;

hg::jobs::WorkStealingDeque<int*>* g_deque;
int* g_got[3] = {nullptr, nullptr, nullptr};

// The OWNER end. Only the owning thread may call pop(), which is the algorithm's precondition and
// the reason bottom needs no atomic read-modify-write on the uncontended path.
void* owner_pop(void*) {
    g_got[0] = g_deque->pop();
    return nullptr;
}

// The THIEF end. Any thread may call steal().
void* thief_steal_a(void*) {
    g_got[1] = g_deque->steal();
    return nullptr;
}

void* thief_steal_b(void*) {
    g_got[2] = g_deque->steal();
    return nullptr;
}

}  // namespace

int main() {
    hg::jobs::WorkStealingDeque<int*> deque(4);
    g_deque = &deque;

    // Populated single-threaded: this harness is about the ends racing, not about push.
    assert(deque.push(&g_a));
    assert(deque.push(&g_b));

    pthread_t t0, t1, t2;
    pthread_create(&t0, nullptr, owner_pop, nullptr);
    pthread_create(&t1, nullptr, thief_steal_a, nullptr);
    pthread_create(&t2, nullptr, thief_steal_b, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // A1: nothing is invented.
    for (int i = 0; i < 3; ++i)
        assert(g_got[i] == nullptr || g_got[i] == &g_a || g_got[i] == &g_b);

    // A2: NO ELEMENT WAS TAKEN TWICE. Two ends returning the same pointer is a job running
    // twice -- a state created twice, an event minted twice. Distinct payloads are what make
    // this checkable: with one element the question collapses to "did both succeed", which the
    // CAS on top settles by itself; with two, the owner's fast path can hand out an element a
    // thief is simultaneously taking, and only the fences forbid it.
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            assert(g_got[i] == nullptr || g_got[i] != g_got[j]);

    // A3: AND IT IS NOT LOST. Either an end took it or it is still there. A run where both ends
    // decline while the element is gone from the deque is the mirror defect of A2, and it is the
    // one that does not announce itself: a job silently never runs, and the system waits for a
    // completion that cannot arrive. Losing an element is what makes a fork-join hang rather
    // than crash.
    const int taken = (g_got[0] != nullptr) + (g_got[1] != nullptr) + (g_got[2] != nullptr);
    assert(deque.size() == static_cast<std::size_t>(2 - taken));

    return 0;
}
