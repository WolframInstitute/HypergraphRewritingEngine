// Expect: Assertion violation
// Args: --weak-cas-spurious-total=2
// Two single-shot compare_exchange_weak calls on two words nobody else writes. Both fail only
// when each fails spuriously, which --weak-cas-spurious-total=2 explores, so the checker reports
// the assertion.
#include <pthread.h>
#include <atomic>
#include <cassert>

std::atomic<int> a{0};
std::atomic<int> b{0};

void* claim(void*) {
    int ea = 0;
    int eb = 0;
    const bool wa = a.compare_exchange_weak(ea, 1, std::memory_order_relaxed);
    const bool wb = b.compare_exchange_weak(eb, 1, std::memory_order_relaxed);
    assert(wa || wb);
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, claim, nullptr);
    pthread_join(t, nullptr);
    return 0;
}
