// Expect: HG-SETUP-CAS main
// Args: --weak-cas-spurious-total=1 --weak-cas-spurious-concurrent-only
// Env: HG_GENMC_SETUP_CAS_SITES=1
// A single-shot weak CAS that main executes before creating a thread. With
// HG_GENMC_SETUP_CAS_SITES set, the checker prints the CAS site once as "HG-SETUP-CAS <site>"
// when --weak-cas-spurious-concurrent-only drops its spurious failure.
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
