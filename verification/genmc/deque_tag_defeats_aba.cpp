// GenMC harness: the packed word's tag stops a stale pop from claiming an item twice.
//
// WHAT IS BEING PROVED, AND WHY IT NEEDS ITS OWN HARNESS. deque_no_double_extraction covers two
// consumers racing for one item, and it covers it: injecting a double extraction there makes it
// report a violation. What it does NOT cover is the tag. Freeze the tag -- replace
// `pack(tag_of(v) + 1, ...)` with `pack(tag_of(v), ...)` at all four commit sites, deleting the
// ABA defence outright -- and that harness still passes, in the same 6 executions. It cannot see
// the tag, because with one item pushed before the consumers start and one pop attempt each, no
// (head,tail) pair can recur inside a popper's load-to-exchange window. So the tag was carried by
// the header's argument and by nothing that runs.
//
// THE WINDOW THE TAG DEFENDS. A popper loads the packed word, reads the item pointer out of the
// slot, and only then compare-exchanges. Between the load and the exchange the pair (head,tail)
// can come back to what it read, because BOTH ENDS MOVE IN BOTH DIRECTIONS: pop_front increments
// head, push_front decrements it. Two operations suffice. Starting from {h=0, t=1}:
//
//     pop_front   commits {h=1, t=1}   and clears slot 0
//     push_front  commits {h=0, t=1}   and publishes a NEW item into slot 0
//
// The pair is back to (0,1). A popper that loaded {h=0, t=1} and the OLD slot pointer now finds
// its expected value sitting in the word again. Without the tag its compare-exchange commits, and
// it returns an item another thread already returned -- while the new item's slot is cleared
// underneath it and that item is lost. That is ABA on this structure, and it needs two operations
// rather than the 2^32 a wrap would take.
//
// THE TAG IS WHAT DISTINGUISHES THE TWO WORDS. It is incremented on every successful operation,
// so the word after the cycle is {tag+2, h=0, t=1}, the stale exchange fails its comparison, and
// the popper retries from a fresh load. The header states a separate bound -- 32 bits wrap after
// 2^32 successful operations -- and that is not what this harness is about: it shows the defence
// working inside the window, not the window's outer limit.
//
// CALIBRATED. Freezing the tag at all four commit sites must make this harness report a
// violation, and it does. With the tag restored it is clean.
//
// WHAT IS BOUNDED. Two threads, capacity 4, one item present before they start. One thread makes
// a single pop_front; the other performs the pop/push cycle that returns the index pair. GenMC
// supplies the delay -- there is no sleep and no stall here, only the interleavings of RC11, one
// of which places the second thread's whole cycle inside the first thread's window.

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "lockfree_deque/deque.hpp"

namespace {

// Two distinct payloads, so "which item came out" is answerable rather than inferred from a
// count. As in the neighbouring harness, a pointer payload lives inline in its slot and the
// allocator is never touched.
int g_x = 11;   // present before the threads start
int g_y = 22;   // pushed by the cycling thread, into the slot g_x vacated

lockfree::Deque<int*>* g_deque;
int* g_got[2] = {nullptr, nullptr};
bool g_pushed_y = false;

// The thread whose load-to-exchange window the cycle has to fit inside. It is an ordinary
// try_pop_front: nothing here is special-cased for the harness.
void* stale_popper(void*) {
    auto r = g_deque->try_pop_front();
    g_got[0] = r.has_value() ? *r : nullptr;
    return nullptr;
}

// Pop then push, which advances the tag twice and returns (head,tail) to where it started.
// The push is conditional on the pop having succeeded, so this thread never adds a second item
// to a deque that still holds the first -- the index pair only recurs when the pop actually
// moved head.
void* cycler(void*) {
    auto r = g_deque->try_pop_front();
    g_got[1] = r.has_value() ? *r : nullptr;
    if (g_got[1] != nullptr) g_pushed_y = g_deque->try_push_front(&g_y);
    return nullptr;
}

}  // namespace

int main() {
    lockfree::Deque<int*> deque(4);
    g_deque = &deque;

    // {h=0, t=1}: one item, and the pair the cycle will return to.
    assert(deque.try_push_back(&g_x));

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, stale_popper, nullptr);
    pthread_create(&t1, nullptr, cycler, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // Nothing is invented: every value handed out is one of the two that were pushed.
    assert(g_got[0] == nullptr || g_got[0] == &g_x || g_got[0] == &g_y);
    assert(g_got[1] == nullptr || g_got[1] == &g_x || g_got[1] == &g_y);

    // THE ABA ASSERTION. g_x was pushed once, so it can be handed to at most one consumer. With
    // the tag frozen, the stale popper's exchange commits against the recurred pair and both
    // threads return it.
    assert(!(g_got[0] == &g_x && g_got[1] == &g_x));

    // g_y is pushed at most once and is subject to the same rule.
    assert(!(g_got[0] == &g_y && g_got[1] == &g_y));

    // Conservation, which is what catches the OTHER half of the ABA damage: the stale exchange
    // also clears the slot g_y was just published into, so g_y goes missing from a deque that
    // reports it was pushed. What left plus what remains must equal what went in.
    const int taken = (g_got[0] != nullptr) + (g_got[1] != nullptr);
    const int pushed = 1 + (g_pushed_y ? 1 : 0);
    assert(taken + static_cast<int>(deque.size()) == pushed);

    return 0;
}
