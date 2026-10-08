// Expect: Assertion violation
// A compare_exchange_weak tried once, with nobody else writing the word. Only a spurious failure
// makes it fail, and the program treats a failure as having lost to another thread. The checker
// explores spurious failures of a weak CAS whose failure leaves the retry loop, so it reports the
// assertion; a checker that makes every weak CAS strong reports no error.
#include <pthread.h>
#include <atomic>
#include <cassert>

std::atomic<int> word{0};

void* claim(void*) {
    int expected = 0;
    const bool won = word.compare_exchange_weak(expected, 1, std::memory_order_acq_rel,
                                                std::memory_order_acquire);
    assert(won);
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, claim, nullptr);
    pthread_join(t, nullptr);
    return 0;
}
