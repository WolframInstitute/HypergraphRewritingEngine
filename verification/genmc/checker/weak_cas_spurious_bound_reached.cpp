// Expect: Assertion violation
// Args: --weak-cas-spurious-bound=2
// A compare_exchange_weak retry loop with nobody else writing the word. Its third attempt runs
// only after two spurious failures in a row, which --weak-cas-spurious-bound=2 explores, so the
// checker reports the assertion.
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
