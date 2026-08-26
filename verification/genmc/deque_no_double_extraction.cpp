// GenMC harness: the work-stealing deque never hands one item to two consumers.
//
// WHAT IS BEING PROVED. An owner pops from the back while a thief pops from the front, with
// EXACTLY ONE item in the deque. That is the only configuration in which the two ends contend:
// with head=0 and tail=1, pop_front resolves to slot 0 and pop_back to slot (1-1)&mask = 0, the
// same slot. Push two items instead and the two ends address slots 0 and 1, never meet, and the
// harness explores a handful of trivially-independent executions while appearing to test the
// race. The single-item deque is the case this structure exists to get right, and the case a
// stress test is least likely to hit, since it needs the two pops to overlap on a deque of size
// one.
//
// The deque commits both ends with a SINGLE compare-exchange on one packed word
// {tag:32, head:16, tail:16}, so the claim is that two pops racing for the last element see the
// same word and only one CAS succeeds.
//
// THE TAG IS NOT WHAT THIS HARNESS CHECKS, and the boundary is worth being exact about. The
// packed word carries a 32-bit tag against ABA, but this configuration cannot exercise it:
// with one item pushed before the consumers start and one pop attempt each, no (head,tail)
// pair recurs inside a popper's load-to-exchange window, so the tag is never what makes a
// compare-exchange fail here. Measured, not argued -- freezing the tag at all four commit
// sites leaves this harness passing in the same 6 executions. The tag has its own harness,
// deque_tag_defeats_aba, where freezing it does produce a violation.
//
// CALIBRATED. Committing the pop without winning the compare-exchange -- the double extraction
// A1 names -- makes this harness report a safety violation (genmc exit 42). What it verifies,
// it verifies; it is the tag that is out of its reach.
//
// The assertions cover both directions of the conservation argument:
//   A1  the one item is handed to AT MOST ONE consumer
//   A2  nothing is invented -- a consumer that reports an item reports the one that was pushed
//   A3  what remains agrees with what left
//
// A pop is allowed to return nullopt even when the deque is non-empty: the index move commits
// before the slot is published or cleared, so a consumer that arrives inside that window reports
// the transient lag rather than blocking. That is the documented non-blocking contract, so the
// assertions bound what may be returned rather than requiring that something is.
//
// WHAT IS BOUNDED. Two threads, capacity 4, one item pushed single-threaded before the consumers
// start, one pop attempt each. Concurrent PUSH against POP is a different question -- push
// publishes the slot after moving the index, pop clears it after moving the index, and the two
// windows interact -- and belongs in its own harness rather than inflating this one.

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "lockfree_deque/deque.hpp"

namespace {

// A pointer payload is stored INLINE in its slot, so the deque touches no allocator here and the
// harness measures the index algorithm rather than the boxing path.
int g_item = 11;

lockfree::Deque<int*>* g_deque;
int* g_got[2] = {nullptr, nullptr};

void* pop_front_worker(void*) {
    auto r = g_deque->try_pop_front();
    g_got[0] = r.has_value() ? *r : nullptr;
    return nullptr;
}

void* pop_back_worker(void*) {
    auto r = g_deque->try_pop_back();
    g_got[1] = r.has_value() ? *r : nullptr;
    return nullptr;
}

}  // namespace

int main() {
    lockfree::Deque<int*> deque(4);
    g_deque = &deque;

    // Populated single-threaded: this harness is about two CONSUMERS racing for one item.
    assert(deque.try_push_back(&g_item));

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, pop_front_worker, nullptr);
    pthread_create(&t1, nullptr, pop_back_worker, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // A2: nothing is invented.
    assert(g_got[0] == nullptr || g_got[0] == &g_item);
    assert(g_got[1] == nullptr || g_got[1] == &g_item);

    // A1: at most one consumer got it. Both getting it is the double extraction the packed word
    // and its tag exist to prevent -- two pops racing for the last element load the same word,
    // and only one compare-exchange can commit against it.
    assert(g_got[0] == nullptr || g_got[1] == nullptr);

    // A3: what remains agrees with what left.
    const int taken = (g_got[0] != nullptr) + (g_got[1] != nullptr);
    assert(deque.size() == static_cast<std::size_t>(1 - taken));

    return 0;
}
