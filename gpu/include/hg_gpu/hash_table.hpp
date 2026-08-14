#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/types.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

// Open-addressing linear-probe concurrent hash table with EMPTY/LOCKED key
// sentinels. Mirrors hypergraph/include/hypergraph/concurrent_map.hpp:
//
//   insert: atomicCAS(keys[slot], EMPTY, LOCKED) → write value → release-store
//           the key. Readers skip LOCKED slots without spinning (non-waiting
//           variant) or __nanosleep-spin (waiting variant). First writer wins;
//           later inserters with the same key see the published key on retry
//           and return the existing value.
//
// Capacity is fixed at construction and there is no resize: the host sizes it from
// EngineConfig, and a run that exhausts it is retried by the host at a larger size.
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
// Gather the occupied slots of a key array into a dense prefix. One thread per slot; the order
// of the result is unspecified, which every caller already assumes because slot order is a
// function of the hash rather than of anything meaningful.
template <typename K, K EMPTY, K LOCKED>
__global__ void k_gather_keys(const K* __restrict__ keys, uint32_t capacity,
                              K* __restrict__ out, uint32_t* __restrict__ count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    const K k = keys[i];
    if (k == EMPTY || k == LOCKED) return;
    out[atomicAdd(count, 1u)] = k;
}

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
        // Fold the two reserved sentinels onto neighbouring keys.
        //
        // EMPTY marks a free slot and LOCKED marks one mid-publication, so a genuine key equal to
        // either is not merely mis-hashed -- it is INVISIBLE. An insert of EMPTY leaves the slot
        // reading as free and the entry is silently never stored; an insert of LOCKED leaves
        // readers waiting on a publication that already happened. The host map answers this by
        // throwing (reject_sentinel_key); device code cannot, so it folds instead.
        //
        // The keys here are 64-bit hashes -- canonical state hashes, hash_causal_triple -- so both
        // values are reachable, not merely representable. This same class cost four correctness
        // bugs on the host before the guard existed.
        //
        // Applied inside the map rather than at each call site, because a normalisation that
        // insert applies and lookup forgets is worse than none: the entry is stored where nothing
        // will ever look for it. Folding costs one comparison and collides two keys with their
        // neighbours, which is what a hash table already handles.
        __device__ static K normalize(K key) {
            if (key == EMPTY)  return static_cast<K>(EMPTY + K{1});
            if (key == LOCKED) return static_cast<K>(LOCKED - K{1});
            return key;
        }

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
            key = normalize(key);
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
            key = normalize(key);
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
            key = normalize(key);
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
        HG_CUDA_CHECK(cudaMalloc(&keys_,   sizeof(K) * capacity_), "ConcurrentMap keys alloc");
        HG_CUDA_CHECK(cudaMalloc(&values_, sizeof(V) * capacity_), "ConcurrentMap values alloc");
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

    // Every key the map holds. EMPTY and LOCKED are slot states rather than keys, so they are
    // dropped -- a caller enumerating a set must not be handed them. Only meaningful once the
    // kernels writing the map have completed.
    //
    // COMPACTED ON THE DEVICE, so the cost is the number of KEYS rather than the table's
    // CAPACITY. Copying the slot array whole made a run's cost scale with how large its pools
    // were configured rather than with the work it did: a depth-7 run whose pools were sized for
    // depth 8 copied eight times the bytes and, worse, value-initialised a host vector of
    // capacity elements for each of the three maps drained per call. A gather kernel writes the
    // occupied slots into a dense buffer and the host reads that prefix.
    void copy_keys_to_host(std::vector<K>& out) const {
        uint32_t* d_count = nullptr;
        K* d_dense = nullptr;
        HG_CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint32_t)), "key gather count alloc");
        HG_CUDA_CHECK(cudaMalloc(&d_dense, sizeof(K) * capacity_), "key gather dense alloc");
        HG_CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)), "key gather count clear");

        const uint32_t block = 256;
        const uint32_t grid = (capacity_ + block - 1) / block;
        k_gather_keys<K, EMPTY, LOCKED><<<grid, block>>>(keys_, capacity_, d_dense, d_count);
        HG_CUDA_CHECK(cudaGetLastError(), "key gather launch");

        uint32_t n = 0;
        HG_CUDA_CHECK(cudaMemcpy(&n, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                      "key gather count read");
        out.resize(n);
        if (n)
            HG_CUDA_CHECK(cudaMemcpy(out.data(), d_dense, sizeof(K) * n,
                                     cudaMemcpyDeviceToHost), "ConcurrentMap key readback");
        cudaFree(d_dense);
        cudaFree(d_count);
    }

    void clear() {
        // THE KEYS ARE THE STATE; THE VALUES ARE NOT.
        //
        // A slot is empty because its KEY says so, and every read of a value is guarded by a key
        // comparison that has already matched: lookup returns a value only under `cur == key`,
        // and insert stores into a slot it has just claimed by exchanging EMPTY for LOCKED. So a
        // value is always written before it is read, and zeroing the value array leaves the map
        // in exactly the state zeroing the keys already put it in.
        //
        // It is not free to do anyway. These maps are cleared once per evolve call and sized from
        // the workload estimate rather than from use, so the clear is charged to a run that may
        // touch a handful of slots: nsys on a depth-3 run producing thirteen states measured
        // gigabytes of cudaMemset, and the dedup maps are what remains of it after the per-edge
        // arrays were bounded.
        //
        // EMPTY is K{0}, so the keys go down with a memset. A non-zero sentinel would need a fill
        // kernel, which is why the assertion stands rather than a comment saying to be careful.
        static_assert(EMPTY == K{0},
            "clear() relies on EMPTY == 0; provide a fill kernel for other sentinels");
        HG_CUDA_CHECK(cudaMemset(keys_, 0, sizeof(K) * capacity_), "ConcurrentMap clear keys");
    }

private:

    K*       keys_     = nullptr;
    V*       values_   = nullptr;
    uint32_t capacity_ = 0;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE