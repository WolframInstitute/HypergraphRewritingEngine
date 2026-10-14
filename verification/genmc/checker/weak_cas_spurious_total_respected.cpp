// Expect: Weak CAS spurious failures past the bound of 1 per execution: 1
// Args: --weak-cas-spurious-total=1
// The program of weak_cas_spurious_total_reached.cpp under --weak-cas-spurious-total=1. Both CASes
// failing needs two spurious failures in one execution, so that execution is not explored: no
// error, and the one option past the bound is counted.
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
