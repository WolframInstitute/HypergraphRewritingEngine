// Expect: Assertion violation
// A compare_exchange_weak retry loop that counts its attempts, with nobody else writing the word.
// Only a spurious failure makes a second attempt; the checker explores it and reports the
// assertion on the count, with no bound on spurious failures.
#include <pthread.h>
#include <atomic>
#include <cassert>

std::atomic<int> word{0};

void* bump(void*) {
    int expected = word.load(std::memory_order_relaxed);
    int attempts = 1;
    while (!word.compare_exchange_weak(expected, expected + 1, std::memory_order_relaxed))
        ++attempts;
    assert(attempts == 1);
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, bump, nullptr);
    pthread_join(t, nullptr);
    return 0;
}
