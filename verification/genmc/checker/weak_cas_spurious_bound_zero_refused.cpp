// Expect: --weak-cas-spurious-bound must be at least 1
// Args: --weak-cas-spurious-bound=0
// A bound of 0 makes every weak CAS strong, and the checker refuses it.
#include <atomic>

std::atomic<int> word{0};

int main() {
    int expected = 0;
    word.compare_exchange_weak(expected, 1, std::memory_order_relaxed);
    return 0;
}
