// A spin loop whose path from the polled load to its exit carries a floating-point constant
// through a PHI. The load annotation substitutes every PHI on that path, and a double constant
// stopped the transform with "Only integer and null constants currently allowed in assume()
// expressions".
#include <atomic>
#include <cassert>
#include <pthread.h>

std::atomic<int> flag{0};

void* setter(void*) {
    flag.store(1, std::memory_order_release);
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, setter, nullptr);
    double weight;
    int v;
    do {
        v = flag.load(std::memory_order_acquire);
        weight = v ? 1.5 : 0.25;
    } while (v == 0);
    pthread_join(t, nullptr);
    assert(weight == 1.5);
    return 0;
}
