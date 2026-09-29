#pragma once
#include "hgcommon/namespace.hpp"
//
// AN APPEND-ONLY WORK LOG for the persistent kernel.
//
// A producer claims a slot with the pool's fetch-add, writes the entry, and stores its
// `published` word with release. A consumer claims up to `max` consecutive entries with one
// compare-exchange on the cursor, below the readable count, and waits for each entry's
// published flag before reading it; the only wait is on a producer that has already claimed the
// slot and is writing it. When the claimed entries are done, and everything they produced is
// published, the consumer adds their number to `done`.
//
// The termination detector counts readable() as produced and done_count() as consumed. That is
// sound when every append happens inside a unit the detector still counts as unconsumed (the
// record being rewritten, or the entry being run): verification/gpumc/replay_task_termination.cpp.
//
// T carries a uint32_t `published`, zero in a fresh slot (the owner's reset_and_clear).

#include "hg_gpu/atomic_pool.hpp"

#include <cuda/atomic>
#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

template <class T>
struct WorkLogView {
    typename Pool<T>::DeviceView items{};
    uint32_t* cursor = nullptr;
    uint32_t* done   = nullptr;

    // False when the log is full.
    __device__ bool append(const T& v) {
        const uint32_t i = items.claim();
        if (i == Pool<T>::kInvalid) return false;
        T& t = items.at(i);
        t = v;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> pub(t.published);
        pub.store(1u, cuda::memory_order_release);
        return true;
    }

    // Claimed slots, clamped to the capacity. Acquire, for the detector's snapshot.
    __device__ uint32_t readable() const {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> c(*items.counter);
        const uint32_t n = c.load(cuda::memory_order_acquire);
        return n < items.capacity ? n : items.capacity;
    }

    __device__ uint32_t done_count() const {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> d(*done);
        return d.load(cuda::memory_order_acquire);
    }

    // Up to `max` consecutive entries from the cursor; returns how many, 0 when none are
    // readable, with the first index in `base`.
    __device__ uint32_t claim(uint32_t max, uint32_t& base) {
        const uint32_t readable_now = readable();
        uint32_t cur = *cursor;
        while (cur < readable_now) {
            const uint32_t k = min(readable_now - cur, max);
            const uint32_t prev = atomicCAS(cursor, cur, cur + k);
            if (prev == cur) { base = cur; return k; }
            cur = prev;
        }
        return 0;
    }

    __device__ T& await(uint32_t i) {
        T& t = items.at(i);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> pub(t.published);
        while (pub.load(cuda::memory_order_acquire) == 0u) __nanosleep(64);
        return t;
    }

    __device__ void book(uint32_t n) { atomicAdd(done, n); }
};

}  // namespace gpu
}  // namespace HG_NAMESPACE
