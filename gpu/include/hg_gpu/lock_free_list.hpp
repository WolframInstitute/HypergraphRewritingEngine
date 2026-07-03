#pragma once

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/types.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hg_gpu {

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
        check(cudaMalloc(&heads_, sizeof(uint32_t) * num_keys_), "LockFreeList heads alloc");
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
    void clear() {
        check(cudaMemset(heads_, 0xFF, sizeof(uint32_t) * num_keys_),
              "LockFreeList clear heads");
        pool_.reset();
    }

    uint32_t num_keys()       const { return num_keys_; }
    uint32_t pool_capacity()  const { return pool_.capacity(); }
    uint32_t pool_used_host() const { return pool_.size_host(); }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::LockFreeList ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    uint32_t   num_keys_ = 0;
    Pool<Node> pool_;
    uint32_t*  heads_ = nullptr;
};

}  // namespace hg_gpu
