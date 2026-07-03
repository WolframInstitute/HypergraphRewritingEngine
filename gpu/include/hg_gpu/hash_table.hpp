#pragma once

#include "hg_gpu/types.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hg_gpu {

// Open-addressing linear-probe concurrent hash table with EMPTY/LOCKED key
// sentinels. Mirrors hypergraph/include/hypergraph/concurrent_map.hpp:
//
//   insert: atomicCAS(keys[slot], EMPTY, LOCKED) → write value → release-store
//           the key. Readers skip LOCKED slots without spinning (non-waiting
//           variant) or __nanosleep-spin (waiting variant). First writer wins;
//           later inserters with the same key see the published key on retry
//           and return the existing value.
//
// Capacity is fixed at construction; auto-tuner sizes it. No resize.
//
// Template parameters K and V should be trivially copyable. EMPTY and LOCKED
// must be values that valid keys never take (e.g. hash 0 and hash ~0 are
// reserved here).
//
// Memory-ordering audit:
//   Writer:  values[slot].store(release)  ──┐
//            keys[slot].store(release)   ──┼─> pair with
//   Reader:  keys[slot].load(acquire)    <─┤  acquire load
//            values[slot].load(acquire)  <─┘  of the key
//   CAS EMPTY→LOCKED: acq_rel (writer's reservation is release; losers'
//                     observation is acquire).
//   The publish store on keys must happen AFTER the values write to
//   establish happens-before; otherwise a reader observing our key could
//   load a stale value. This is enforced by the release semantics on the
//   keys store paired with the reader's acquire load of the key (which
//   synchronizes-with the values write).
template <typename K, typename V, K EMPTY = K{0}, K LOCKED = static_cast<K>(~K{0})>
class ConcurrentMap {
public:
    struct LookupResult {
        V    value;
        bool found;
    };

    struct InsertResult {
        V    value;        // existing-or-newly-inserted value
        bool inserted;     // true iff this thread won the slot
    };

    struct DeviceView {
        K*       keys;
        V*       values;
        uint32_t capacity;

        // Hash key → starting slot. Caller hashes its own key — the table
        // treats K as opaque. Default mixer is sufficient for already-hashed
        // uint64_t keys.
        __device__ static uint32_t mix(uint64_t k) {
            // splitmix64 finalizer; cheap, well-distributed
            k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
            k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
            k ^= k >> 33;
            return static_cast<uint32_t>(k);
        }

        __device__ uint32_t initial_slot(K key) const {
            return mix(static_cast<uint64_t>(key)) % capacity;
        }

        // Lookup that skips LOCKED slots without waiting. May report not-found
        // if a concurrent insert is mid-publish; callers needing strong
        // visibility should use lookup_waiting.
        __device__ LookupResult lookup(K key) const {
            uint32_t slot = initial_slot(key);
            for (uint32_t i = 0; i < capacity; ++i) {
                cuda::atomic_ref<K, cuda::thread_scope_device> kref(keys[slot]);
                K cur = kref.load(cuda::memory_order_acquire);
                if (cur == key) {
                    cuda::atomic_ref<V, cuda::thread_scope_device> vref(values[slot]);
                    return LookupResult{vref.load(cuda::memory_order_acquire), true};
                }
                if (cur == EMPTY) return LookupResult{V{}, false};
                // LOCKED → skip without spin
                slot = (slot + 1) % capacity;
            }
            return LookupResult{V{}, false};
        }

        // Lookup that waits on LOCKED slots until they publish (the publish
        // either matches our key — found — or doesn't, in which case we
        // continue probing).
        __device__ LookupResult lookup_waiting(K key) const {
            uint32_t slot = initial_slot(key);
            for (uint32_t i = 0; i < capacity; ++i) {
                cuda::atomic_ref<K, cuda::thread_scope_device> kref(keys[slot]);
                K cur = kref.load(cuda::memory_order_acquire);
                while (cur == LOCKED) {
                    __nanosleep(32);
                    cur = kref.load(cuda::memory_order_acquire);
                }
                if (cur == key) {
                    cuda::atomic_ref<V, cuda::thread_scope_device> vref(values[slot]);
                    return LookupResult{vref.load(cuda::memory_order_acquire), true};
                }
                if (cur == EMPTY) return LookupResult{V{}, false};
                slot = (slot + 1) % capacity;
            }
            return LookupResult{V{}, false};
        }

        // Insert-if-absent. Returns the existing value if the key is already
        // present; otherwise atomically claims the slot, writes the value,
        // and returns it. inserted == true iff this thread won the slot.
        //
        // Per-slot inner loop handles three resolutions: (1) slot already holds
        // our key → return existing value, (2) slot is LOCKED → wait for the
        // owner to publish, (3) slot is EMPTY → CAS to LOCKED, write, publish.
        // The outer loop only advances when the slot is firmly held by a
        // different key.
        __device__ InsertResult insert_if_absent(K key, V value) {
            uint32_t slot = initial_slot(key);
            for (uint32_t i = 0; i < capacity; ++i) {
                cuda::atomic_ref<K, cuda::thread_scope_device> kref(keys[slot]);

                while (true) {
                    K cur = kref.load(cuda::memory_order_acquire);

                    if (cur == key) {
                        cuda::atomic_ref<V, cuda::thread_scope_device> vref(values[slot]);
                        return InsertResult{vref.load(cuda::memory_order_acquire), false};
                    }

                    if (cur == LOCKED) {
                        __nanosleep(32);
                        continue;
                    }

                    if (cur == EMPTY) {
                        K expected = EMPTY;
                        bool ok = kref.compare_exchange_strong(
                            expected, LOCKED,
                            cuda::memory_order_acq_rel, cuda::memory_order_acquire);
                        if (ok) {
                            cuda::atomic_ref<V, cuda::thread_scope_device> vref(values[slot]);
                            vref.store(value, cuda::memory_order_release);
                            kref.store(key, cuda::memory_order_release);
                            return InsertResult{value, true};
                        }
                        // CAS failed — expected now holds whatever raced ahead
                        // of us (could be our key, LOCKED, or another key).
                        // Loop to re-evaluate.
                        continue;
                    }

                    // cur is some other published key — advance probe.
                    break;
                }

                slot = (slot + 1) % capacity;
            }
            return InsertResult{V{}, false};  // capacity exceeded
        }
    };

    explicit ConcurrentMap(uint32_t capacity) : capacity_(capacity) {
        check(cudaMalloc(&keys_,   sizeof(K) * capacity_), "ConcurrentMap keys alloc");
        check(cudaMalloc(&values_, sizeof(V) * capacity_), "ConcurrentMap values alloc");
        clear();
    }

    ~ConcurrentMap() {
        if (keys_)   cudaFree(keys_);
        if (values_) cudaFree(values_);
    }

    ConcurrentMap(const ConcurrentMap&)            = delete;
    ConcurrentMap& operator=(const ConcurrentMap&) = delete;

    ConcurrentMap(ConcurrentMap&& o) noexcept
        : keys_(o.keys_), values_(o.values_), capacity_(o.capacity_) {
        o.keys_ = nullptr; o.values_ = nullptr; o.capacity_ = 0;
    }

    DeviceView view() const { return DeviceView{keys_, values_, capacity_}; }

    uint32_t capacity() const { return capacity_; }

    void clear() {
        // EMPTY is K{0}; we memset the keys array. (For non-zero EMPTY this
        // would need a fill kernel — not currently used in the codebase.)
        static_assert(EMPTY == K{0},
            "clear() relies on EMPTY == 0; provide a fill kernel for other sentinels");
        check(cudaMemset(keys_,   0, sizeof(K) * capacity_), "ConcurrentMap clear keys");
        check(cudaMemset(values_, 0, sizeof(V) * capacity_), "ConcurrentMap clear values");
    }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::ConcurrentMap ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    K*       keys_     = nullptr;
    V*       values_   = nullptr;
    uint32_t capacity_ = 0;
};

}  // namespace hg_gpu
