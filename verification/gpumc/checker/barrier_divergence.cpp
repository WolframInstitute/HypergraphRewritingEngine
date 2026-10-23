// Checker test for the barrier divergence rule in gpumc-fixes.patch: a thread waiting at its block's
// barrier diverges when a thread of its block has finished, and not when the block-mate is
// blocked elsewhere.
//
// Lane 1 of a two-lane block waits at the block barrier. -DEXITED: lane 0 returns without
// reaching it, and the run must report the divergence. Default: lane 0 spins on a word nothing
// sets, which GPUMC's spin-loop assume blocks, and the run must report no error and no complete
// execution.
//
// gpumc_docker:latest reports a divergence in both; gpumc_docker:hg-fixes in the first only.
// GPUMC-CELL: exited FAIL -DEXITED
// GPUMC-CELL: spinning BLOCKED
#include <cstdint>
#include <pthread.h>

extern "C" {
void __VERIFIER_memory_scope_device(void);
void __VERIFIER_memory_scope_work_group(void);
void __VERIFIER_thread_local_id(int);
void __VERIFIER_thread_group_id(int);
void __VERIFIER_thread_global_id(int);
void __VERIFIER_thread_kernel_id(int);
}

namespace {

pthread_barrier_t g_bar;
uint32_t g_never;

void* lane(void* arg) {
    const int id = static_cast<int>(reinterpret_cast<long>(arg));
    __VERIFIER_thread_global_id(id);
    __VERIFIER_thread_local_id(id);
    __VERIFIER_thread_group_id(0);
    __VERIFIER_thread_kernel_id(0);
    if (id == 0) {
#if defined(EXITED)
        return nullptr;
#else
        for (;;) {
            __VERIFIER_memory_scope_device();
            if (__atomic_load_n(&g_never, __ATOMIC_RELAXED) != 0) break;
        }
#endif
    }
    __VERIFIER_memory_scope_work_group();
    pthread_barrier_wait(&g_bar);
    return nullptr;
}

}  // namespace

int main() {
    pthread_barrier_init(&g_bar, nullptr, 2);
    pthread_t t[2];
    for (long i = 0; i < 2; ++i) pthread_create(&t[i], nullptr, lane, reinterpret_cast<void*>(i));
    for (int i = 0; i < 2; ++i) pthread_join(t[i], nullptr);
    return 0;
}
