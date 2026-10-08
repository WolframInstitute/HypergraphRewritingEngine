// Expect: Number of complete executions explored: 48
// Three threads push onto a Treiber stack with compare_exchange_weak, storing the link before
// each attempt. A spurious failure returns the expected value, so the retry stores the same link
// and attempts the same CAS: the checker drops that repeat and the run ends without --unroll.
// The 48 executions are the push orders with their genuinely failed attempts. A checker that
// explores the repeats does not end (HG_GENMC_NO_STUTTER=1: no verdict in 30 s).
#include <pthread.h>
#include <atomic>
#include <cassert>

struct Node { Node* next; int v; };
std::atomic<Node*> head{nullptr};
Node nodes[3];

void* push(void* arg) {
    Node* n = &nodes[reinterpret_cast<long>(arg)];
    n->v = 1;
    Node* old = head.load(std::memory_order_relaxed);
    do {
        n->next = old;
    } while (!head.compare_exchange_weak(old, n, std::memory_order_release,
                                         std::memory_order_relaxed));
    return nullptr;
}

int main() {
    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, push, reinterpret_cast<void*>(i));
    for (auto& x : t) pthread_join(x, nullptr);
    int sum = 0;
    for (Node* n = head.load(std::memory_order_acquire); n; n = n->next) sum += n->v;
    assert(sum == 3);
    return 0;
}
