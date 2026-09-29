// GPUMC harness: on the device, a child never strands at a stale depth when its parent is
// relaxed concurrently.
//
// THE PROTOCOL. hgcommon/explore_depth_core.hpp, the body both engines call, driven over the
// device's face (gpu/src/persistent.cu DeviceExploreCtx): child lists are
// gpu/include/hg_gpu/lock_free_list.hpp's (hgcommon/list_core.hpp push with a relaxed head load
// and an ACQ_REL exchange, walks from an acquire head load), depths are device-scope atomics with
// ACQ_REL compare-exchange and acquire loads, and the fence is __threadfence(), a seq_cst fence
// at device scope. The two threads are in different blocks.
//
//   registrar (explore_register_child)          relaxer (explore_try_lower, explore_relax)
//     push the child into the parent's list       lower the parent's depth
//     __threadfence                               __threadfence
//     read the parent's depth                     walk the parent's child list
//
// THE PROPERTY. The child ends at one past the parent's lowered depth, whichever side wins.
// verification/genmc/depth_relax_child_registration.cpp checks the same core over the host's
// list under RC11.
//
// THE BOUND. Two threads, one parent, one child, one relaxation.
//
// CALIBRATION. -DCALIBRATE_NO_FENCE makes the fence a no-op; the checker must report the child
// stranded at the parent's old depth plus one.
//
// HOW A SCOPE IS EXPRESSED: __VERIFIER_memory_scope_device() before the access it qualifies.
// The list heads and depths are 64-bit words because this GPUMC build cannot complete a 32-bit
// compare-exchange (hash_insert_elects_one.cpp); the device's are 32-bit.

#include "hgcommon/explore_depth_core.hpp"
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

constexpr uint32_t kInvalid   = 0xFFFFFFFFu;
constexpr uint32_t kParent    = 0;
constexpr uint32_t kChild     = 1;
constexpr uint32_t kOldDepth  = 5;
constexpr uint32_t kNewDepth  = 2;

struct Node { uint32_t value; uint32_t next; };
Node     g_nodes[2];
uint32_t g_next_node;               // node claims, as the list's pool claims them
uint64_t g_head[2]  = {kInvalid, kInvalid};
uint64_t g_depth[2] = {kOldDepth, hgcommon::kExploreNoDepth};

uint64_t load64_dev(uint64_t* a, int order) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, order);
}
bool cas64_dev(uint64_t* a, uint64_t* expected, uint64_t desired, int success, int failure) {
    __VERIFIER_memory_scope_device();
    return __atomic_compare_exchange_n(a, expected, desired, /*weak=*/false, success, failure);
}
uint32_t add32_dev(uint32_t* a, uint32_t n) {
    __VERIFIER_memory_scope_device();
    return __atomic_fetch_add(a, n, __ATOMIC_RELAXED);
}
void threadfence() {
#if !defined(CALIBRATE_NO_FENCE)
    __VERIFIER_memory_scope_device();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

// LockFreeList<T>::DeviceView::Ops.
struct ListOps {
    uint64_t* head;
    uint32_t invalid() const { return kInvalid; }
    uint32_t head_load_relaxed() const {
        return static_cast<uint32_t>(load64_dev(head, __ATOMIC_RELAXED));
    }
    uint32_t head_load_acquire() const {
        return static_cast<uint32_t>(load64_dev(head, __ATOMIC_ACQUIRE));
    }
    bool head_cas(uint32_t& expected, uint32_t desired) {
        uint64_t e = expected;
        const bool ok = cas64_dev(head, &e, desired, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
        expected = static_cast<uint32_t>(e);
        return ok;
    }
    void set_next(uint32_t node, uint32_t next) { g_nodes[node].next = next; }
    uint32_t next_of(uint32_t node) const { return g_nodes[node].next; }
};

// DeviceExploreCtx, with a fixed frame store and no expansion.
struct Ctx {
    using Node = uint32_t;
    uint32_t frame_node[4];
    uint32_t frame_depth[4];
    uint32_t frames = 0;

    uint32_t depth_load(uint32_t s) const {
        return static_cast<uint32_t>(load64_dev(&g_depth[s], __ATOMIC_ACQUIRE));
    }
    bool depth_cas(uint32_t s, uint32_t& expected, uint32_t desired) {
        uint64_t e = expected;
        const bool ok = cas64_dev(&g_depth[s], &e, desired, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        expected = static_cast<uint32_t>(e);
        return ok;
    }
    void children_push(uint32_t parent, uint32_t child) {
        const uint32_t idx = add32_dev(&g_next_node, 1u);
        g_nodes[idx].value = child;
        ListOps ops{&g_head[parent]};
        hgcommon::list_push(ops, idx);
    }
    Node children_head(uint32_t s) const { return ListOps{&g_head[s]}.head_load_acquire(); }
    static bool children_end(Node n) { return n == kInvalid; }
    uint32_t children_value(Node n) const { return g_nodes[n].value; }
    Node children_next(Node n) const { return g_nodes[n].next; }
    void fence() const { threadfence(); }
    void admit(uint32_t, uint32_t) {}
    bool frame_push(Node at, uint32_t d) {
        if (frames == 4) return false;
        frame_node[frames] = at; frame_depth[frames] = d; ++frames;
        return true;
    }
    bool frame_top(Node*& at, uint32_t& d) {
        if (frames == 0) return false;
        at = &frame_node[frames - 1]; d = frame_depth[frames - 1];
        return true;
    }
    void frame_pop() { --frames; }
};

void* registrar(void*) {
    __VERIFIER_thread_global_id(0); __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(0);  __VERIFIER_thread_kernel_id(0);
    Ctx c;
    const uint32_t d = hgcommon::explore_register_child(c, kParent, kChild, kOldDepth + 1u);
    if (d != hgcommon::kExploreNoDepth) hgcommon::explore_relax(c, kChild, d);
    return nullptr;
}

void* relaxer(void*) {
    __VERIFIER_thread_global_id(1); __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(1);  __VERIFIER_thread_kernel_id(0);
    Ctx c;
    if (hgcommon::explore_try_lower(c, kParent, kNewDepth))
        hgcommon::explore_relax(c, kParent, kNewDepth);
    return nullptr;
}

}  // namespace

int main() {
    pthread_t a, b;
    pthread_create(&a, nullptr, registrar, nullptr);
    pthread_create(&b, nullptr, relaxer, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);
    assert(load64_dev(&g_depth[kChild], __ATOMIC_ACQUIRE) == kNewDepth + 1u &&
           "the child stranded at a depth its parent no longer has");
    return 0;
}
