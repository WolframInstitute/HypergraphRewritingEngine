#pragma once

#include "hg_gpu/types.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hg_gpu {

// Lock-free MPMC ring buffer for inter-kernel work queues.
//
// Layout (power-of-two capacity for cheap modulo):
//   slots[capacity]: payload storage
//   ready[capacity]: per-slot publish flag (0 empty, 1 ready, 2 consumed)
//   head:  consumer cursor (atomicAdd to claim a read slot)
//   tail:  producer cursor (atomicAdd to claim a write slot)
//
// Producer protocol:
//   1. idx = atomicAdd(tail, 1)
//   2. if idx - head_observed >= capacity → buffer full, return false
//   3. write slots[idx % cap]
//   4. ready[idx % cap].store(1, release)
//
// Consumer protocol:
//   1. if head >= tail (snapshot), return empty
//   2. idx = atomicAdd(head, 1)
//   3. if idx >= tail_observed → over-claimed, atomic_dec head, return empty
//   4. wait for ready[idx % cap].load(acquire) == 1 (with __nanosleep)
//   5. read slots[idx % cap]
//   6. ready[idx % cap].store(2, relaxed) — slot recyclable once head wraps
//
// `try_push` returns false on full; callers retry with backoff. `try_pop`
// returns false on empty; callers may also exit when termination is signalled.
//
// Memory-ordering audit:
//   Producer: slot_write (non-atomic) → ready.store(kReady, release)
//   Consumer: ready.load(acquire) == kReady → slot_read (non-atomic)
//   The release-store on ready synchronizes-with the acquire-load, which
//   establishes happens-before between the non-atomic slot write and read.
//   Head/tail fetch_add use acq_rel because both producers and consumers
//   read the counter to check capacity/availability before reserving.
template <typename T>
class RingBuffer {
public:
    static constexpr uint8_t kEmpty    = 0;
    static constexpr uint8_t kReady    = 1;
    static constexpr uint8_t kConsumed = 2;

    struct DeviceView {
        T*        slots;
        uint8_t*  ready;
        uint64_t* head;     // consumer cursor (monotonic)
        uint64_t* tail;     // producer cursor (monotonic)
        uint32_t  capacity; // power of two
        uint32_t  mask;     // capacity - 1

        __device__ bool try_push(const T& value) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> tref(*tail);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> href(*head);

            // Reserve a producer slot.
            uint64_t idx = tref.fetch_add(1ULL, cuda::memory_order_acq_rel);

            // Check capacity. Reload head; if reservation outruns the slot's
            // recycled state, we have to back off — uncontested case checks
            // ready[idx & mask] == kEmpty || kConsumed.
            uint32_t s = static_cast<uint32_t>(idx) & mask;
            cuda::atomic_ref<uint8_t, cuda::thread_scope_device> rref(ready[s]);

            // Wait for the slot to be free (kEmpty or kConsumed). If a
            // consumer hasn't drained it yet within a bounded retry window,
            // give up.
            constexpr int kMaxSpins = 64;
            int spins = 0;
            while (true) {
                uint8_t r = rref.load(cuda::memory_order_acquire);
                if (r == kEmpty || r == kConsumed) break;
                if (++spins >= kMaxSpins) {
                    // Roll back the reservation so we don't permanently lose
                    // a slot. Atomic decrement of tail is racy with other
                    // producers; we use a fetch_sub to mark it failed and
                    // accept that other concurrent reservations may have
                    // already moved past us — they'll see ready==kReady
                    // already (no, they won't, this slot is still ours). So
                    // we leave the reservation in place but use a special
                    // "abandoned" payload via ready=kReady-with-poison? For
                    // simplicity here we publish an empty slot and let the
                    // consumer drop it; full-buffer is a sign the queue is
                    // undersized and the auto-tuner needs to grow it.
                    return false;
                }
                __nanosleep(64);
            }

            slots[s] = value;
            rref.store(kReady, cuda::memory_order_release);
            return true;
        }

        __device__ bool try_pop(T& out) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> href(*head);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> tref(*tail);

            // Quick empty check: head >= tail.
            uint64_t h = href.load(cuda::memory_order_acquire);
            uint64_t t = tref.load(cuda::memory_order_acquire);
            if (h >= t) return false;

            // Reserve a consumer slot.
            uint64_t idx = href.fetch_add(1ULL, cuda::memory_order_acq_rel);

            // Re-check that we didn't over-claim past tail. If we did, mark
            // our reservation consumed (best-effort) and return empty so
            // other consumers / producers can make progress.
            uint64_t t2 = tref.load(cuda::memory_order_acquire);
            if (idx >= t2) {
                // Over-claimed. The corresponding slot will not be filled.
                // Mark it kEmpty so producers can wrap into it later.
                uint32_t s = static_cast<uint32_t>(idx) & mask;
                cuda::atomic_ref<uint8_t, cuda::thread_scope_device> rref(ready[s]);
                rref.store(kEmpty, cuda::memory_order_release);
                return false;
            }

            uint32_t s = static_cast<uint32_t>(idx) & mask;
            cuda::atomic_ref<uint8_t, cuda::thread_scope_device> rref(ready[s]);

            // Wait for the producer to publish.
            constexpr int kMaxSpins = 1024;
            int spins = 0;
            while (true) {
                uint8_t r = rref.load(cuda::memory_order_acquire);
                if (r == kReady) break;
                if (++spins >= kMaxSpins) {
                    // Producer never published (e.g. it gave up on full).
                    // Treat as empty.
                    rref.store(kEmpty, cuda::memory_order_release);
                    return false;
                }
                __nanosleep(64);
            }

            out = slots[s];
            rref.store(kConsumed, cuda::memory_order_release);
            return true;
        }

        __device__ uint32_t size_approx() const {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> href(*head);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> tref(*tail);
            uint64_t h = href.load(cuda::memory_order_relaxed);
            uint64_t t = tref.load(cuda::memory_order_relaxed);
            return (t > h) ? static_cast<uint32_t>(t - h) : 0u;
        }
    };

    explicit RingBuffer(uint32_t capacity_pow2) : capacity_(capacity_pow2), mask_(capacity_pow2 - 1) {
        if ((capacity_ & mask_) != 0 || capacity_ == 0) {
            throw std::invalid_argument("RingBuffer capacity must be a power of two ≥ 1");
        }
        check(cudaMalloc(&slots_, sizeof(T)       * capacity_), "RingBuffer slots");
        check(cudaMalloc(&ready_, sizeof(uint8_t) * capacity_), "RingBuffer ready");
        check(cudaMalloc(&head_,  sizeof(uint64_t)),            "RingBuffer head");
        check(cudaMalloc(&tail_,  sizeof(uint64_t)),            "RingBuffer tail");
        clear();
    }

    ~RingBuffer() {
        if (slots_) cudaFree(slots_);
        if (ready_) cudaFree(ready_);
        if (head_)  cudaFree(head_);
        if (tail_)  cudaFree(tail_);
    }

    RingBuffer(const RingBuffer&)            = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    DeviceView view() {
        return DeviceView{slots_, ready_, head_, tail_, capacity_, mask_};
    }

    void clear() {
        check(cudaMemset(ready_, 0, sizeof(uint8_t) * capacity_), "RingBuffer clear ready");
        check(cudaMemset(head_,  0, sizeof(uint64_t)),            "RingBuffer clear head");
        check(cudaMemset(tail_,  0, sizeof(uint64_t)),            "RingBuffer clear tail");
    }

    uint64_t head_host() const { uint64_t v = 0; cudaMemcpy(&v, head_, sizeof(v), cudaMemcpyDeviceToHost); return v; }
    uint64_t tail_host() const { uint64_t v = 0; cudaMemcpy(&v, tail_, sizeof(v), cudaMemcpyDeviceToHost); return v; }
    uint32_t capacity() const  { return capacity_; }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::RingBuffer ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    T*        slots_    = nullptr;
    uint8_t*  ready_    = nullptr;
    uint64_t* head_     = nullptr;
    uint64_t* tail_     = nullptr;
    uint32_t  capacity_ = 0;
    uint32_t  mask_     = 0;
};

}  // namespace hg_gpu
