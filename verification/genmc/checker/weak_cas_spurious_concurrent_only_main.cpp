// Args: --weak-cas-spurious-total=1 --weak-cas-spurious-concurrent-only
// A single-shot weak CAS that main executes before creating a thread. With
// --weak-cas-spurious-concurrent-only its spurious failure is not explored, so the assertion that
// it succeeded holds; the dropped option is counted.
#include <atomic>
#include <cassert>
#include <pthread.h>

std::atomic<int> x{0};

void* other(void*) { return nullptr; }

int main() {
    int expected = 0;
    assert(x.compare_exchange_weak(expected, 1));
    pthread_t t;
    pthread_create(&t, nullptr, other, nullptr);
    pthread_join(t, nullptr);
    return 0;
}
