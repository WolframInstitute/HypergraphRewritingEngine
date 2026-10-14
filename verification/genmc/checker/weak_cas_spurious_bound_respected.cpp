// Expect: Weak CAS spurious failures past the bound of 1: 1
// Args: --weak-cas-spurious-bound=1
// The program of weak_cas_spurious_bound_reached.cpp under --weak-cas-spurious-bound=1. The
// third attempt needs two spurious failures in a row, so it is not explored: no error, and the one
// spurious failure past the bound is counted.
#include <pthread.h>
#include <atomic>
#include <cassert>

std::atomic<int> word{0};

void* bump(void*) {
    int expected = 0;
    int attempts = 1;
    while (!word.compare_exchange_weak(expected, 1, std::memory_order_relaxed))
        ++attempts;
    assert(attempts < 3);
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, bump, nullptr);
    pthread_join(t, nullptr);
    return 0;
}
