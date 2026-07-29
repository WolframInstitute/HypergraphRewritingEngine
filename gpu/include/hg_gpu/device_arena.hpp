#pragma once
// A bump allocator the DEVICE owns, for scratch whose size is only known once the work is in
// hand.
//
// It exists because of one constraint: no host-device communication during evolution
// (docs/GPU_PERSISTENT_DESIGN.md). The level-synchronous path can size the IR scratch slot by
// measuring the largest state in a batch on the host, because it has batches. A persistent
// loop does not -- states arrive continuously, and the largest is not knowable before the run
// starts. Sizing from a configured maximum instead would reintroduce the wrong-dedup-key
// exposure that fixing that measurement closed: above the bound, states fall back to 1-WL,
// which MERGES non-isomorphic states.
//
// So a worker sizes its scratch from its own state's edge and occurrence counts and claims
// exactly that, with no host involved and no fixed ceiling per state.
//
// There is no free. A worker finishes with its scratch before taking the next item, so blocks
// reuse a slot and only claim again when they need a LARGER one, abandoning the smaller. With
// doubling that costs at most one wasted slot per block, bounded by the peak, and the whole
// arena resets at the end of the run.
//
// Exhaustion is a capacity overflow like any other: the claim fails, the caller records it and
// returns partial work. It cannot grow, because growing needs the host.

#include "hg_gpu/types.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hg_gpu {

class DeviceArena {
public:
    struct View {
        uint32_t* base;
        uint64_t* cursor;      // words handed out so far
        uint64_t  capacity;    // words

        // Claim `words`, 8-byte aligned so the callers' uint64 views inside the block are
        // valid. Returns nullptr when the arena is exhausted -- never partially satisfies.
        __device__ uint32_t* claim(uint64_t words) {
            const uint64_t padded = (words + 1ull) & ~1ull;   // keep every claim even
            const uint64_t off = atomicAdd(reinterpret_cast<unsigned long long*>(cursor),
                                           static_cast<unsigned long long>(padded));
            if (off + padded > capacity) return nullptr;
            return base + off;
        }
    };

    explicit DeviceArena(uint64_t capacity_words) : capacity_(capacity_words) {
        check(cudaMalloc(&base_, capacity_ * sizeof(uint32_t)), "arena alloc");
        check(cudaMalloc(&cursor_, sizeof(uint64_t)), "arena cursor alloc");
        reset();
    }

    ~DeviceArena() {
        if (base_)   cudaFree(base_);
        if (cursor_) cudaFree(cursor_);
    }

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    void reset() {
        check(cudaMemset(cursor_, 0, sizeof(uint64_t)), "arena cursor clear");
    }

    View view() { return View{base_, cursor_, capacity_}; }

    uint64_t capacity_words() const { return capacity_; }

    // Words handed out. Reads across the boundary, so it is for AFTER a run, not during one.
    uint64_t used_words_host() const {
        uint64_t v = 0;
        cudaMemcpy(&v, cursor_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return v;
    }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::DeviceArena ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    uint32_t* base_    = nullptr;
    uint64_t* cursor_  = nullptr;
    uint64_t  capacity_ = 0;
};

}  // namespace hg_gpu
