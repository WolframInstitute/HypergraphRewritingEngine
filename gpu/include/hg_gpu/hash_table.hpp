#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/types.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

// Open-addressing linear-probe concurrent hash table. LOCK-FREE: NO OPERATION WAITS ON ANOTHER
// THREAD.
//
// A lane of a warp cannot wait for another lane without holding it back -- they advance
// together -- so a spinning lane can stall the very lane that owns the slot it waits on, and a
// spinning warp occupies an SM doing nothing while hammering one line. Every operation here
// completes without reference to any other thread's progress.
//
// The key goes straight from EMPTY to its final value in one exchange, so there is no
// intermediate state for anyone to observe, and the value is published under a SECOND exchange
// against UNPUBLISHED. A thread that finds the key present but the value unpublished OFFERS its
// own value rather than waiting, so the window between the two closes at the first arrival
// instead of lasting until the key's claimant is scheduled again.
//
// THE VALUE EXCHANGE IS THE ELECTION. UNPUBLISHED -> something happens exactly once per slot, so
// exactly one offer succeeds however many are made and whatever they carry, and the thread it
// elects is by construction the one whose value is stored. Neither of the alternatives holds:
// the KEY exchange elects a thread too, but a different one -- it can lose the value exchange
// and then report inserted while carrying a stranger's value -- and comparing the stored value
// against the caller's holds only while every caller for a key offers a DISTINCT value, which
// most callers here do not (they offer a constant presence marker). The host map carries the
// same rule for the same reason, with a model-checked harness for it.
//
// THE TWO STEPS ARE WHY BOTH ARRAYS CARRY STATE: a claimed slot whose value is not yet
// published must READ as unpublished, so clear() fills the values as well as the keys.
//
// Reserved keys: EMPTY marks a free slot and LOCKED is the second reserved key; a genuine key
// equal to either is folded onto a neighbour rather than lost. Mirrors
// hypergraph/include/hypergraph/concurrent_map.hpp:
//
//   insert: atomicCAS(keys[slot], EMPTY, key) publishes the key; atomicCAS(values[slot],
//           UNPUBLISHED, value) publishes the value AND elects the inserter, because that
//           exchange succeeds exactly once per slot. Everyone who meets an unpublished value
//           offers its own rather than waiting, so the window closes at the first arrival.
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
//   CAS EMPTY→key:    acq_rel (the winner's publication is release; the
//                     losers' observation is acquire).
//   The publish store on keys must happen AFTER the values write to
//   establish happens-before; otherwise a reader observing our key could
//   load a stale value. This is enforced by the release semantics on the
//   keys store paired with the reader's acquire load of the key (which
//   synchronizes-with the values write).
// Gather the occupied slots of a key array into a dense prefix. One thread per slot; the order
// of the result is unspecified, which every caller already assumes because slot order is a
// function of the hash rather than of anything meaningful.
//
// The anonymous namespace is what keeps this kernel to one registration per translation unit.
// A `__global__` template has external linkage, and this header reaches most of the .cu set
// through engine_state.hpp, so a shared specialization would be registered once per unit under
// one name; the runtime then keeps whichever was registered first and reports the rest as
// duplicates. Internal linkage gives each unit its own kernel and leaves the choice to nobody.
namespace {
template <typename K, K EMPTY, K LOCKED>
__global__ void k_gather_keys(const K* __restrict__ keys, uint32_t capacity,
                              K* __restrict__ out, uint32_t* __restrict__ count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    const K k = keys[i];
    if (k == EMPTY || k == LOCKED) return;
    out[atomicAdd(count, 1u)] = k;
}
}  // namespace

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
        // TRUE IFF THE PROBE RUN WAS EXHAUSTED WITHOUT DECIDING.
        //
        // Without this a full table is indistinguishable from a hit: the exhaustion path returned
        // {V{}, false}, which is byte-identical to finding an existing key whose value is 0. Every
        // caller here reads `inserted`, so a full table silently answered "already present" --
        // and the dedup map is what decides whether a state has been SEEN, so new states were
        // dropped from the answer with nothing recorded.
        //
        // It cannot be inferred from the other two fields and it cannot be avoided by bounding the
        // probe: with linear probing a key lives anywhere in its contiguous run, so giving up
        // early would miss existing keys and insert duplicates instead -- a silent double-count in
        // place of a silent drop. The caller has to be told, and callers that can record it do.
        bool overflowed = false;

    };

    struct DeviceView {
        // THE VALUE THAT MEANS "CLAIMED, NOT YET PUBLISHED", and the reason the value can be
        // published under its own exchange instead of behind a lock: a thread that finds it
        // unpublished OFFERS its own rather than waiting for the claimant.
        //
        // IT MUST BE A VALUE NO CALLER CAN STORE, and that rules zero out. Callers store state
        // ids and event ids RAW -- exploration.hpp inserts `sid`, event_identity.hpp inserts
        // `eid`, persistent.cu inserts `sid` -- and the first state and the first event are
        // numbered 0. Under a zero sentinel those two entries are stored and then read as
        // never-published for the rest of the run: lookup reports not-found and insert reports
        // the value unreadable, so the identity of event 0 is invisible to every caller that
        // asks for it, permanently and silently.
        static constexpr V UNPUBLISHED = static_cast<V>(~V{0});

        // IT IS ALL ONES SO clear() CAN STILL BE A MEMSET, and it is the top of the range so no
        // caller can reach it. The values stored here are state ids, event ids, `+1`-biased ids
        // and presence markers, all bounded by a table capacity that is orders below 2^32 --
        // whereas ZERO is reached by the first state id and the first event id, which is what a
        // sentinel must never be.
        static_assert(UNPUBLISHED == static_cast<V>(~V{0}),
            "clear() fills the values with a 0xFF memset; UNPUBLISHED must be all ones");

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
        // SET ONCE A PROBE RUN HAS EXHAUSTED THE TABLE, read before every insert.
        //
        // Without it a full table costs O(capacity) PER INSERT, because the probe walks every
        // slot before it can report overflow. Affordable once, ruinous in bulk: the branchial
        // relation on disc-l3a2g2r2 at depth three is 133,351,476 pairs against a default
        // capacity of 524,288, so the device faced ~10^13 probe steps and never returned --
        // where the policy everywhere else here is to return partial work with a warning.
        // Latching turns the second and every later overflow into O(1), so the run completes
        // and reports truncation instead of hanging.
        //
        // Racy by construction and harmless: a few threads may finish a full scan before the
        // flag is visible, and a stale zero costs only the scan that would have happened
        // anyway. Never cleared mid-kernel, so it cannot make a table with room reject an
        // insert.
        uint32_t* saturated;

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

        // Reports not-found while a concurrent insert holds the key but has not published a
        // value, which is the only window in which an answer is owed and missing. No thread
        // waits for another here.
        __device__ LookupResult lookup(K key) const {
            key = normalize(key);
            uint32_t slot = initial_slot(key);
            for (uint32_t i = 0; i < capacity; ++i) {
                cuda::atomic_ref<K, cuda::thread_scope_device> kref(keys[slot]);
                K cur = kref.load(cuda::memory_order_acquire);
                if (cur == key) {
                    cuda::atomic_ref<V, cuda::thread_scope_device> vref(values[slot]);
                    const V v = vref.load(cuda::memory_order_acquire);
                    // THE KEY IS PUBLISHED BEFORE THE VALUE, so a matching key is a claim and
                    // not yet an answer. Returning UNPUBLISHED as data hands the caller the
                    // sentinel itself.
                    if (v == UNPUBLISHED) return LookupResult{V{}, false};
                    return LookupResult{v, true};
                }
                if (cur == EMPTY) return LookupResult{V{}, false};
                // LOCKED → skip without spin
                slot = (slot + 1) % capacity;
            }
            return LookupResult{V{}, false};
        }

        // NOTHING WAITS HERE. The key exchange publishes the key and the value follows under
        // its own exchange, so the only thing a caller could wait FOR is a value no thread has
        // offered yet, and it is told not-found instead. The name is kept because its callers
        // are asking about visibility, and the host map answers the same question the same way
        // (concurrent_map.hpp: lookup_waiting forwards to lookup).
        __device__ LookupResult lookup_waiting(K key) const {
            return lookup(key);
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
            // A caller storing the reserved value would publish a slot that every reader takes
            // for unclaimed. Keys can be folded onto a neighbour; a value cannot, because
            // folding it IS the corruption, and device code cannot throw the way the host map
            // does -- so this is a DEBUG TRIPWIRE for a future caller, not the guarantee.
            //
            // The guarantee is the choice of sentinel: every value stored here is a state id,
            // an event id, a `+1`-biased id or a presence marker, all bounded by a table
            // capacity orders below 2^32, so all-ones is outside the domain by construction.
            // -O3 -DNDEBUG compiles the assert out, which is why the domain argument has to
            // hold on its own.
            assert(value != UNPUBLISHED && "value collides with the UNPUBLISHED sentinel");
            if (saturated) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> sref(*saturated);
                if (sref.load(cuda::memory_order_relaxed)) return InsertResult{V{}, false, true};
            }
            key = normalize(key);
            uint32_t slot = initial_slot(key);
            for (uint32_t i = 0; i < capacity; ++i) {
                cuda::atomic_ref<K, cuda::thread_scope_device> kref(keys[slot]);

                while (true) {
                    K cur = kref.load(cuda::memory_order_acquire);

                    if (cur == key) {
                        // OFFER, do not wait. The value exchange is what elects the inserter:
                        // UNPUBLISHED -> something happens exactly ONCE per slot, so exactly one
                        // thread's exchange succeeds however many offer and whatever they offer.
                        // That is the election, and the thread it elects is by construction the
                        // one whose value is stored -- which is the property `inserted` has to
                        // carry, because event_identity marks itself canonical on `inserted` and
                        // points every later event at the STORED value.
                        //
                        // It cannot be the KEY exchange. That elects one thread too, but a
                        // different one: the key winner can lose the value exchange, and then it
                        // reports inserted while carrying a stranger's value and the value's
                        // owner reports not-inserted -- one signature, two canonical events.
                        // Nor can it be a comparison of the stored value against the caller's:
                        // most callers here offer a constant presence marker, so every one of
                        // them would match and every one would be told it inserted, and
                        // qe.applied gates an APPLICATION on that flag. The host map states the
                        // same rule for the same reason.
                        cuda::atomic_ref<V, cuda::thread_scope_device> vref(values[slot]);
                        V seen = vref.load(cuda::memory_order_acquire);
                        if (seen == UNPUBLISHED) {
                            V expect_v = UNPUBLISHED;
                            if (vref.compare_exchange_strong(expect_v, value,
                                    cuda::memory_order_acq_rel, cuda::memory_order_acquire))
                                return InsertResult{value, true};
                            seen = expect_v;   // someone else's exchange won; take its value
                        }
                        return InsertResult{seen, false};
                    }

                    if (cur == EMPTY) {
                        // STRAIGHT TO THE KEY, no reservation state. The key exchange alone
                        // decides the winner, and the winner then owns the value slot, so no
                        // thread ever waits on another to publish.
                        K expected = EMPTY;
                        bool ok = kref.compare_exchange_strong(
                            expected, key,
                            cuda::memory_order_acq_rel, cuda::memory_order_acquire);
                        if (ok) {
                            // Publishing the KEY is not the election -- see above. This thread
                            // now offers its value like any other and is the inserter only if
                            // that exchange is the one that succeeds.
                            cuda::atomic_ref<V, cuda::thread_scope_device> vref(values[slot]);
                            V expect_v = UNPUBLISHED;
                            if (vref.compare_exchange_strong(expect_v, value,
                                    cuda::memory_order_acq_rel, cuda::memory_order_acquire))
                                return InsertResult{value, true};
                            return InsertResult{expect_v, false};
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
            // Exhausted every slot without finding the key or a free one. NOT a hit: say so, so
            // the caller can record a capacity overflow instead of treating this as "seen".
            // Latch it, so the next caller pays one load rather than another full scan.
            if (saturated) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> sref(*saturated);
                sref.store(1u, cuda::memory_order_relaxed);
            }
            return InsertResult{V{}, false, true};
        }
    };

    explicit ConcurrentMap(uint32_t capacity) : capacity_(capacity) {
        HG_CUDA_CHECK(cudaMalloc(&keys_,   sizeof(K) * capacity_), "ConcurrentMap keys alloc");
        HG_CUDA_CHECK(cudaMalloc(&values_, sizeof(V) * capacity_), "ConcurrentMap values alloc");
        HG_CUDA_CHECK(cudaMalloc(&saturated_, sizeof(uint32_t)), "ConcurrentMap saturated alloc");
        clear();
    }

    ~ConcurrentMap() {
        if (keys_)      cudaFree(keys_);
        if (values_)    cudaFree(values_);
        if (saturated_) cudaFree(saturated_);
    }

    ConcurrentMap(const ConcurrentMap&)            = delete;
    ConcurrentMap& operator=(const ConcurrentMap&) = delete;

    ConcurrentMap(ConcurrentMap&& o) noexcept
        : keys_(o.keys_), values_(o.values_), capacity_(o.capacity_), saturated_(o.saturated_) {
        o.keys_ = nullptr; o.values_ = nullptr; o.capacity_ = 0; o.saturated_ = nullptr;
    }

    DeviceView view() const { return DeviceView{keys_, values_, capacity_, saturated_}; }

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
        // BOTH ARRAYS CARRY STATE, and the values are the half that is easy to get wrong.
        //
        // A slot is free because its KEY says so, and a value is read only under a key that has
        // already matched. That does not make the value array incidental: the key is published
        // FIRST, in one exchange, and the value follows under its own exchange against
        // UNPUBLISHED. Between those two steps a reader sees the key and reads the value, so the
        // value a claimed slot holds before its owner publishes has to mean "not yet" -- and if
        // the array still held what a previous run left there, it would mean that run's answer.
        // These maps are cleared and reused per evolve call, so that is one call reading
        // another's values.
        //
        // Both sentinels are chosen so this stays two memsets rather than two fill kernels:
        // EMPTY is all zeroes and UNPUBLISHED is all ones.
        static_assert(EMPTY == K{0},
            "clear() relies on EMPTY == 0; provide a fill kernel for other sentinels");
        HG_CUDA_CHECK(cudaMemset(keys_, 0, sizeof(K) * capacity_), "ConcurrentMap clear keys");
        HG_CUDA_CHECK(cudaMemset(values_, 0xFF, sizeof(V) * capacity_),
                      "ConcurrentMap clear values");
        // The table has room again, so the latch must go with the keys. Leaving it set would
        // make a reused map reject every insert for the remainder of the run.
        if (saturated_)
            HG_CUDA_CHECK(cudaMemset(saturated_, 0, sizeof(uint32_t)),
                          "ConcurrentMap clear saturated");
    }

private:

    K*        keys_      = nullptr;
    V*        values_    = nullptr;
    uint32_t  capacity_  = 0;
    // Latched when a probe run exhausts the table; cleared with the keys, since a cleared table
    // has room again and a stale flag would refuse every insert for the rest of the run.
    uint32_t* saturated_ = nullptr;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE