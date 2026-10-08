// Expect: Assertion violation
// A compare_exchange_weak retry loop that counts its attempts, with nobody else writing the word.
// The count is carried round the loop and changes on a failure, so a spurious failure is not a
// repeat of the attempt before it: the checker keeps the CAS weak, explores the spurious failure
// and reports the assertion on the count.
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
