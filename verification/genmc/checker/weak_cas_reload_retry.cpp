// Expect: HG-STUTTER strong _Z3popPv
// Env: HG_GENMC_STUTTER_REPORT=1
// Two threads pop from a two-slot stack: the index is carried round the loop (the CAS refreshes
// it), each iteration loads the slot and can leave, and the compare_exchange_weak claims the
// index. An iteration that ends in a spurious failure holds only reads before the CAS and carries
// the index back unchanged, so the checker drops it: the report names the CAS strong. Upstream's
// spin-assume also bounds this loop when the checker compiles it itself; the engine's deque,
// compiled through run.sh, needs the drop (deque_no_double_extraction: no verdict in 90 s
// without it, 6 executions with it).
#include <pthread.h>
#include <atomic>
#include <cassert>

std::atomic<int> top{2};
std::atomic<int> slots[2] = {{10}, {20}};
int got[2];

void* pop(void* arg) {
    const long me = reinterpret_cast<long>(arg);
    int v = top.load(std::memory_order_acquire);
    while (v != 0) {
        const int item = slots[v - 1].load(std::memory_order_acquire);
        if (item == 0) break;
        if (top.compare_exchange_weak(v, v - 1, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
            got[me] = item;
            return nullptr;
        }
    }
    got[me] = -1;
    return nullptr;
}

int main() {
    pthread_t t[2];
    for (long i = 0; i < 2; ++i) pthread_create(&t[i], nullptr, pop, reinterpret_cast<void*>(i));
    for (auto& x : t) pthread_join(x, nullptr);
    assert(got[0] != got[1]);
    return 0;
}
