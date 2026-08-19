#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/atomic_pool.hpp"
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

        // Push value onto list[key]. Returns the node index, or kInvalid on
        // pool exhaustion or out-of-range key.
        __device__ uint32_t push(uint32_t key, const T& value) {
            if (key >= num_keys) return Pool<Node>::kInvalid;
            uint32_t idx = pool.claim();
            if (idx == Pool<Node>::kInvalid) return Pool<Node>::kInvalid;

            Node& n = pool.at(idx);
            n.value = value;

            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> href(heads[key]);
            uint32_t prev = href.load(cuda::memory_order_relaxed);
            while (true) {
                n.next = prev;
                // Publish the node (with release) before swinging the head.
                // The release on compare_exchange ensures the next field is
                // visible to walkers that load head with acquire.
                if (href.compare_exchange_weak(
                        prev, idx,
                        cuda::memory_order_release,
                        cuda::memory_order_relaxed)) {
                    return idx;
                }
                // prev was updated by the failed CAS; retry with new prev.
            }
        }

        __device__ uint32_t head_index(uint32_t key) const {
            if (key >= num_keys) return Pool<Node>::kInvalid;
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> href(heads[key]);
            return href.load(cuda::memory_order_acquire);
        }

        __device__ const Node* node(uint32_t idx) const {
            return (idx == Pool<Node>::kInvalid) ? nullptr : &pool.at(idx);
        }

        // Every node linked STRICTLY BEFORE `mine`, most-recent-first.
        //
        // This is what lets two pushers meet exactly once. If each visits only the nodes older
        // than its own, then of any two exactly one is older, so exactly one of the two scans
        // sees the other -- no dedup structure, and no dependence on which warp ran first.
        // Walking from the HEAD instead makes both see each other whenever the pushes and the
        // scans interleave, which is a pair reported twice.
        //
        // The next chain below `mine` is fixed once `mine` is linked, since push only prepends.
        template <typename Fn>
        __device__ void for_each_before(uint32_t mine, Fn fn) const {
            if (mine == Pool<Node>::kInvalid) return;
            uint32_t idx = pool.at(mine).next;
            while (idx != Pool<Node>::kInvalid) {
                const Node& n = pool.at(idx);
                fn(n.value);
                idx = n.next;
            }
        }

        // Functional iteration: invoke fn(value) for each node in list[key].
        // Order is most-recent-first (stack semantics). Safe concurrent with
        // pushes from other threads — visits exactly the nodes published
        // before head was loaded.
        template <typename Fn>
        __device__ void for_each(uint32_t key, Fn fn) const {
            if (key >= num_keys) return;
            uint32_t idx = head_index(key);
            while (idx != Pool<Node>::kInvalid) {
                const Node& n = pool.at(idx);
                fn(n.value);
                idx = n.next;
            }
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
    uint32_t pool_capacity()  const { return pool_.capacity(); }
    uint32_t pool_used_host() const { return pool_.size_host(); }

private:

    uint32_t   num_keys_ = 0;
    Pool<Node> pool_;
    uint32_t*  heads_ = nullptr;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE