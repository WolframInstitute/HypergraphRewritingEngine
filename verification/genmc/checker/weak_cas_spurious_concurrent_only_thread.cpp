// Expect: Assertion violation
// Args: --weak-cas-spurious-total=1 --weak-cas-spurious-concurrent-only
// The same single-shot weak CAS, executed after the first thread is created: its spurious
// failure is explored under --weak-cas-spurious-concurrent-only, and the assertion fails.
#include <atomic>
#include <cassert>
#include <pthread.h>

std::atomic<int> x{0};

void* other(void*) { return nullptr; }

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, other, nullptr);
    int expected = 0;
    assert(x.compare_exchange_weak(expected, 1));
    pthread_join(t, nullptr);
    return 0;
}
