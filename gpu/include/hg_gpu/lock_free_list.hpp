#pragma once
#include <vector>
#include "hgcommon/namespace.hpp"

#include "hg_gpu/atomic_pool.hpp"
#include "hgcommon/list_core.hpp"

#include "hg_gpu/types.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

// Lock-free per-key list. For each key in [0, num_keys), maintains an
// append-only stack of nodes. Mirrors the CPU lock_free_list.hpp pattern
// (claim-node-then-CAS-link).
//
// Used for per-edge consumer lists (causal rendezvous) and per-state event
// lists (branchial scan). Nodes are claimed from a global pre-allocated pool;
// no node is ever freed during evolution (bulk-freed at end).
//
// Iteration walks the linked list from head; safe concurrent with appenders
// because nodes are immutable after the linking CAS publishes them.
//
// Memory-ordering audit:
//   Pusher:   node.value, node.next set (non-atomic) →
//             heads[key].CAS(prev, idx, release, relaxed)
//   Walker:   heads[key].load(acquire) → walk next pointers
//   The release on the CAS pairs with the acquire on the head load to make
//   the node's value and next fields visible to the walker. Subsequent
//   pointer-chasing along next is safe because each node was itself
//   published via a release-CAS on the head at some earlier time, and the
//   current walker already performed an acquire that synchronized-with the
//   most recent publish (which transitively happens-after all earlier
//   publishes via the CAS loop's prev chain).
template <typename T>
class LockFreeList {
public:
    struct Node {
        T        value;
        uint32_t next;   // index into node pool, INVALID_ID for tail
    };

    static constexpr uint32_t kEmptyHead = INVALID_ID;

    struct DeviceView {
        typename Pool<Node>::DeviceView pool;
        uint32_t* heads;     // size = num_keys
        uint32_t  num_keys;

        // The storage face hgcommon::list_push / list_for_each drive: how the head word and a
        // node's next field are touched and at what scope. Nothing here decides anything.
        struct Ops {
            DeviceView* v;   // the walks only read; they cast away const to share one face
            uint32_t key;
            __device__ uint32_t invalid() const { return Pool<Node>::kInvalid; }
            __device__ uint32_t head_load_relaxed() const {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> h(v->heads[key]);
                return h.load(cuda::memory_order_relaxed);
            }
            __device__ uint32_t head_load_acquire() const {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> h(v->heads[key]);
                return h.load(cuda::memory_order_acquire);
            }
            __device__ bool head_cas(uint32_t& expected, uint32_t desired) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> h(v->heads[key]);
                return h.compare_exchange_weak(expected, desired, cuda::memory_order_acq_rel,
                                               cuda::memory_order_relaxed);
            }
            __device__ void set_next(uint32_t node, uint32_t next) { v->pool.at(node).next = next; }
            __device__ uint32_t next_of(uint32_t node) const { return v->pool.at(node).next; }
        };

        // Push value onto list[key]. Returns the node index, or kInvalid on pool exhaustion or
        // out-of-range key. The exchange is ACQ_REL for the reason hgcommon/list_core.hpp gives:
        // release publishes the node, acquire covers the pusher's own walk below it.
        __device__ uint32_t push(uint32_t key, const T& value) {
            if (key >= num_keys) return Pool<Node>::kInvalid;
            uint32_t idx = pool.claim();
            if (idx == Pool<Node>::kInvalid) return Pool<Node>::kInvalid;
            pool.at(idx).value = value;
            Ops ops{this, key};
            hgcommon::list_push(ops, idx);
            return idx;
        }

        __device__ uint32_t head_index(uint32_t key) const {
            if (key >= num_keys) return Pool<Node>::kInvalid;
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> href(heads[key]);
            return href.load(cuda::memory_order_acquire);
        }

        __device__ const Node* node(uint32_t idx) const {
            return (idx == Pool<Node>::kInvalid) ? nullptr : &pool.at(idx);
        }

        // Every node linked STRICTLY BEFORE `mine`, most-recent-first. This is what lets two
        // pushers meet exactly once: of any two, exactly one is older, so exactly one of the two
        // scans sees the other. Walking from the head instead reports a pair twice whenever the
        // pushes and the scans interleave. `key` is only what the walk needs to name its pool.
        template <typename Fn>
        __device__ void for_each_before(uint32_t mine, Fn fn) const {
            Ops ops{const_cast<DeviceView*>(this), 0u};
            hgcommon::list_for_each_before(ops, mine, [&](uint32_t idx) { fn(pool.at(idx).value); });
        }

        // Every node in list[key], most-recent-first: exactly the nodes published before the
        // head was loaded, so it is safe concurrent with pushes.
        template <typename Fn>
        __device__ void for_each(uint32_t key, Fn fn) const {
            if (key >= num_keys) return;
            Ops ops{const_cast<DeviceView*>(this), key};
            hgcommon::list_for_each(ops, [&](uint32_t idx) { fn(pool.at(idx).value); });
        }
    };

    LockFreeList(uint32_t num_keys, uint32_t pool_capacity)
        : num_keys_(num_keys), pool_(pool_capacity) {
        HG_CUDA_CHECK(cudaMalloc(&heads_, sizeof(uint32_t) * num_keys_), "LockFreeList heads alloc");
        clear();
    }

    ~LockFreeList() {
        if (heads_) cudaFree(heads_);
    }

    LockFreeList(const LockFreeList&)            = delete;
    LockFreeList& operator=(const LockFreeList&) = delete;

    DeviceView view() const {
        return DeviceView{pool_.view(), heads_, num_keys_};
    }

    // Set every head to kEmptyHead (= INVALID_ID = 0xFFFFFFFF).
    //
    // `used_keys` bounds the reset to the prefix a run can have touched, for the lists whose KEY
    // IS A DENSE ID -- a vertex, an edge, an event -- allocated from a monotone counter. Those
    // are sized from the workload estimate and cleared in full every call, so a run producing a
    // handful of states paid for the estimate: the head arrays are what remains of the gigabytes
    // of cudaMemset a depth-3 run was measured issuing. Lists keyed by a HASH BUCKET cannot use
    // this -- their entries are scattered across the whole array -- and pass nothing.
    void clear(uint32_t used_keys = 0xFFFFFFFFu) {
        const uint32_t n = used_keys < num_keys_ ? used_keys : num_keys_;
        if (n) HG_CUDA_CHECK(cudaMemset(heads_, 0xFF, sizeof(uint32_t) * n),
                             "LockFreeList clear heads");
        pool_.reset();
    }

    uint32_t num_keys()       const { return num_keys_; }
    // Every node the pool holds, in claim order rather than list order. A host caller that
    // wants the RELATION a list encodes does not need the chains: it needs the records, which
    // it can group itself. Handing back nodes rather than a materialised relation is what keeps
    // the device from having to store an expansion of what it already has.
    void copy_nodes_to_host(std::vector<Node>& out) const { pool_.copy_to_host(out); }

    uint32_t pool_capacity()  const { return pool_.capacity(); }
    uint32_t pool_used_host() const { return pool_.size_host(); }

private:

    uint32_t   num_keys_ = 0;
    Pool<Node> pool_;
    uint32_t*  heads_ = nullptr;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE