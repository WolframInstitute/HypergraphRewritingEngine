#pragma once
#include "hgcommon/namespace.hpp"

#include "hgcommon/ring_core.hpp"

#include "hg_gpu/types.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// Writes seq[i] = i on the device; the launching wrapper is defined in persistent.cu because
// this header reaches host-only translation units that cannot hold a kernel.
void ring_seq_ramp_device(uint64_t* seq, uint32_t n);

// Bounded lock-free MPMC ring buffer for device-resident work queues.
//
// Every slot carries a SEQUENCE NUMBER, and that number alone says whose turn the slot is:
//
//   seq[s] == pos          the slot is free for the producer reserving position pos
//   seq[s] == pos + 1      the slot holds the item for the consumer reserving position pos
//   seq[s] <  pos          the queue is full (producer) / empty (consumer)
//
// seq[i] starts at i, a producer publishes with seq = pos + 1, and a consumer releases with
// seq = pos + capacity, so a slot advances by exactly `capacity` per lap and the three tests
// above stay exact across wraps. The capacity is at least two: at one, the value a consumer
// releases with (pos + 1) is the value the NEXT producer position tests as "holds an item"
// (pos + 1), so a second push lands on a live slot and the pop that follows never matches.
//
// The reservation is a CAS on head/tail rather than an unconditional bump. That is what makes
// the queue safe when the SAME workers both produce and consume, which is how a device-resident
// scheduler uses it: a bump has nothing to undo with, and a rollback (or a "give the slot back"
// store) can land on a slot another thread has already taken, which either loses an item or
// hands one out twice. An item lost from a queue whose producers are its consumers is not a
// dropped unit of work -- it is a run that never terminates, because the termination detector
// waits forever for a completion that can no longer happen.
//
// try_push returns false only when the queue was observed full, and try_pop false only when
// observed empty. Neither ever mutates a slot it did not win.
//
// Memory-ordering audit:
//   Producer: slot write (non-atomic) -> seq.store(pos + 1, release)
//   Consumer: seq.load(acquire) == pos + 1 -> slot read (non-atomic) -> seq.store(pos + cap)
//   The release-store synchronizes-with the acquire-load, establishing happens-before between
//   the non-atomic slot write and read. head/tail CAS is relaxed: the seq handshake carries the
//   ordering, and the cursors only allocate positions.
template <typename T>
class RingBuffer {
public:
    struct DeviceView {
        T*        slots;
        uint64_t* seq;      // [capacity]
        uint64_t* head;     // consumer cursor (monotonic)
        uint64_t* tail;     // producer cursor (monotonic)
        uint32_t  capacity; // power of two
        uint32_t  mask;     // capacity - 1

        // The storage face hgcommon::ring_claim drives. Everything here is HOW a word is
        // touched and at what scope; nothing here decides anything. A producer carries the
        // value it is publishing and a consumer the destination it is filling, and transfer()
        // is the one place the two roles differ.
        template <bool kPush>
        struct Ops {
            DeviceView* v;
            const T*    in;    // the value a producer is publishing
            T*          out;   // where a consumer puts what it took

            __device__ uint32_t mask() const { return v->mask; }
            __device__ uint64_t cursor_load() const {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> c(kPush ? *v->tail : *v->head);
                return c.load(cuda::memory_order_relaxed);
            }
            __device__ bool cursor_cas(uint64_t& expected, uint64_t desired) {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> c(kPush ? *v->tail : *v->head);
                return c.compare_exchange_weak(expected, desired, cuda::memory_order_relaxed);
            }
            __device__ uint64_t seq_load(uint32_t s) const {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> r(v->seq[s]);
                return r.load(cuda::memory_order_acquire);
            }
            __device__ void seq_store(uint32_t s, uint64_t value) {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> r(v->seq[s]);
                r.store(value, cuda::memory_order_release);
            }
            __device__ void transfer(uint32_t s) {
                if constexpr (kPush) v->slots[s] = *in; else *out = v->slots[s];
            }
        };

        __device__ bool try_push(const T& value) {
            Ops<true> ops{this, &value, nullptr};
            return hgcommon::ring_claim(ops, /*want=*/0, /*leave=*/1);
        }

        __device__ bool try_pop(T& out) {
            Ops<false> ops{this, nullptr, &out};
            return hgcommon::ring_claim(ops, /*want=*/1, /*leave=*/mask + 1);
        }

        __device__ uint32_t size_approx() const {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> href(*head);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> tref(*tail);
            const uint64_t h = href.load(cuda::memory_order_relaxed);
            const uint64_t t = tref.load(cuda::memory_order_relaxed);
            return (t > h) ? static_cast<uint32_t>(t - h) : 0u;
        }
    };

    explicit RingBuffer(uint32_t capacity_pow2)
        : capacity_(capacity_pow2), mask_(capacity_pow2 - 1) {
        if ((capacity_ & mask_) != 0 || capacity_ < 2) {
            throw std::invalid_argument("RingBuffer capacity must be a power of two >= 2");
        }
        HG_CUDA_CHECK(cudaMalloc(&slots_, sizeof(T)        * capacity_), "RingBuffer slots");
        HG_CUDA_CHECK(cudaMalloc(&seq_,   sizeof(uint64_t) * capacity_), "RingBuffer seq");
        HG_CUDA_CHECK(cudaMalloc(&head_,  sizeof(uint64_t)),             "RingBuffer head");
        HG_CUDA_CHECK(cudaMalloc(&tail_,  sizeof(uint64_t)),             "RingBuffer tail");
        clear();
    }

    ~RingBuffer() {
        if (slots_) cudaFree(slots_);
        if (seq_)   cudaFree(seq_);
        if (head_)  cudaFree(head_);
        if (tail_)  cudaFree(tail_);
    }

    RingBuffer(const RingBuffer&)            = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    DeviceView view() {
        return DeviceView{slots_, seq_, head_, tail_, capacity_, mask_};
    }

    // seq[i] = i, so position i is the first producer turn for slot i. A memset cannot express
    // that, and a fill kernel defined in this header would be registered once per including
    // translation unit, so the ramp is built on the host and uploaded.
    void clear() {
        HG_CUDA_CHECK(cudaMemset(head_, 0, sizeof(uint64_t)), "RingBuffer clear head");
        HG_CUDA_CHECK(cudaMemset(tail_, 0, sizeof(uint64_t)), "RingBuffer clear tail");
        // The ramp is written ON DEVICE. Building it host-side and copying it up cost eight
        // bytes per slot per clear -- 8 MB and 1.45 ms of H2D per run at the launch chain's
        // 2^20 capacity, about a quarter of the device's per-call floor -- for values that
        // are just the slot indices. Defined in persistent.cu because this header reaches
        // host-only translation units.
        ring_seq_ramp_device(seq_, capacity_);
    }

    uint64_t head_host() const {
        uint64_t v = 0; cudaMemcpy(&v, head_, sizeof(v), cudaMemcpyDeviceToHost); return v;
    }
    uint64_t tail_host() const {
        uint64_t v = 0; cudaMemcpy(&v, tail_, sizeof(v), cudaMemcpyDeviceToHost); return v;
    }
    uint32_t capacity() const { return capacity_; }

    T*        slots_    = nullptr;
    uint64_t* seq_      = nullptr;
    uint64_t* head_     = nullptr;
    uint64_t* tail_     = nullptr;
    uint32_t  capacity_ = 0;
    uint32_t  mask_     = 0;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE