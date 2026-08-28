// GPUMC harness: the quotient replay's (instance, match) rendezvous on the DEVICE never drops a
// pair.
//
// Runs hgcommon/list_core.hpp ITSELF -- the push and walk gpu/include/hg_gpu/lock_free_list.hpp
// drives -- with the same shape the device replay has around it (quotient_expansion.hpp,
// qe_add_instance / qe_capture_expansion and the two qe_drive_* scans):
//
//   instance side                       match side
//     push the instance                   push the match
//     __threadfence()                     __threadfence()
//     walk the match list                 walk the instance list
//
// If BOTH walks miss, the pair is never applied: one fewer raw event, and with it every causal
// and branchial pair it belonged to, while the canonical counts are untouched. The host's twin,
// verification/genmc/quotient_instance_match_rendezvous.cpp, checks the same rule under RC11;
// this checks it under the memory model the device runs, where the two CTAs synchronise only
// through device-scope accesses and a device-scope fence.
//
// __threadfence() is cuda::atomic_thread_fence(memory_order_seq_cst, thread_scope_device), and
// that is what the fence below is: a seq_cst fence preceded by the device-scope annotation.
//
// CALIBRATED. -DCALIBRATE_NO_FENCE removes both fences, leaving the acq_rel exchange and the
// acquire head load, and the checker must report both walks missing.
#include "hgcommon/list_core.hpp"

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

constexpr uint32_t kInvalid = 0xFFFFFFFFu;
constexpr uint32_t kNodes = 4;

// Two lists over one node pool: index 0 is the instance node, index 1 the match node.
struct Node { uint32_t value; uint32_t next; };
Node     g_nodes[kNodes];
uint32_t g_head_inst = kInvalid;
uint32_t g_head_match = kInvalid;

uint32_t load_dev(uint32_t* a, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, order);
}
bool cas_dev(uint32_t* a, uint32_t* expected, uint32_t desired) {
    __VERIFIER_memory_scope_device();
    return __atomic_compare_exchange_n(a, expected, desired, /*weak=*/true,
                                       __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
}
void threadfence() {
    __VERIFIER_memory_scope_device();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

struct Ops {
    uint32_t* head;
    uint32_t invalid() const { return kInvalid; }
    uint32_t head_load_relaxed() const { return load_dev(head, __ATOMIC_RELAXED); }
    uint32_t head_load_acquire() const { return load_dev(head, __ATOMIC_ACQUIRE); }
    bool head_cas(uint32_t& expected, uint32_t desired) { return cas_dev(head, &expected, desired); }
    void set_next(uint32_t node, uint32_t next) { g_nodes[node].next = next; }
    uint32_t next_of(uint32_t node) const { return g_nodes[node].next; }
};

bool g_inst_saw_match = false;
bool g_match_saw_inst = false;

void* instance_side(void*) {
    __VERIFIER_thread_global_id(0); __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(0);  __VERIFIER_thread_kernel_id(0);
    g_nodes[0].value = 11;
    Ops mine{&g_head_inst};
    hgcommon::list_push(mine, 0);
#if !defined(CALIBRATE_NO_FENCE)
    threadfence();
#endif
    Ops theirs{&g_head_match};
    hgcommon::list_for_each(theirs, [&](uint32_t idx) { if (g_nodes[idx].value == 22) g_inst_saw_match = true; });
    return nullptr;
}

void* match_side(void*) {
    __VERIFIER_thread_global_id(1); __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(1);  __VERIFIER_thread_kernel_id(0);
    g_nodes[1].value = 22;
    Ops mine{&g_head_match};
    hgcommon::list_push(mine, 1);
#if !defined(CALIBRATE_NO_FENCE)
    threadfence();
#endif
    Ops theirs{&g_head_inst};
    hgcommon::list_for_each(theirs, [&](uint32_t idx) { if (g_nodes[idx].value == 11) g_match_saw_inst = true; });
    return nullptr;
}

}  // namespace

int main() {
    pthread_t a, b;
    pthread_create(&a, nullptr, instance_side, nullptr);
    pthread_create(&b, nullptr, match_side, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);
    // AT LEAST ONE side sees the other; both missing is the dropped pair.
    assert((g_inst_saw_match || g_match_saw_inst) && "instance and match both missed each other");
    return 0;
}
