// Checker test for gpumc-fixes.patch: a 32-bit compare-exchange succeeds when the expected value
// has bit 31 set.
//
// Thread 1 exchanges 0xFFFFFFFF (written by a plain store) for 1. Thread 2 exchanges 0x80000000
// (written by an atomic add, which GPUMC keeps sign-extended) for 2. Each exchange is the only
// writer of its word, so it must succeed in every execution.
//
// GPUMC-CELL: patched PASS
//
// Expected: gpumc_docker:latest reports the assertion with no complete execution;
// gpumc_docker:hg-fixes reports 1 complete execution and no error.
#include <cassert>
#include <cstdint>
#include <pthread.h>

extern "C" {
void __VERIFIER_memory_scope_device();
void __VERIFIER_thread_local_id(int);
void __VERIFIER_thread_group_id(int);
void __VERIFIER_thread_global_id(int);
void __VERIFIER_thread_kernel_id(int);
}

namespace {

uint32_t g_store;
uint32_t g_added;
bool g_ok[2];

void ids(int id) {
    __VERIFIER_thread_global_id(id);
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(id);
    __VERIFIER_thread_kernel_id(0);
}

void* by_store(void*) {
    ids(0);
    uint32_t expected = 0xFFFFFFFFu;
    __VERIFIER_memory_scope_device();
    g_ok[0] = __atomic_compare_exchange_n(&g_store, &expected, 1u, false, __ATOMIC_RELAXED,
                                          __ATOMIC_RELAXED);
    return nullptr;
}

void* by_add(void*) {
    ids(1);
    __VERIFIER_memory_scope_device();
    __atomic_fetch_add(&g_added, 1u, __ATOMIC_RELAXED);
    uint32_t expected = 0x80000000u;
    __VERIFIER_memory_scope_device();
    g_ok[1] = __atomic_compare_exchange_n(&g_added, &expected, 2u, false, __ATOMIC_RELAXED,
                                          __ATOMIC_RELAXED);
    return nullptr;
}

}  // namespace

int main() {
    g_store = 0xFFFFFFFFu;
    g_added = 0x7FFFFFFFu;
    pthread_t t[2];
    pthread_create(&t[0], nullptr, by_store, nullptr);
    pthread_create(&t[1], nullptr, by_add, nullptr);
    pthread_join(t[0], nullptr);
    pthread_join(t[1], nullptr);
    assert(g_ok[0] && g_store == 1u && "exchange of an expected 0xFFFFFFFF failed");
    assert(g_ok[1] && g_added == 2u && "exchange of an expected 0x80000000 failed");
    return 0;
}
