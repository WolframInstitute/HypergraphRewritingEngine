#pragma once
#include "hgcommon/namespace.hpp"
// A bump allocator the DEVICE owns, for scratch whose size is only known once the work is in
// hand.
//
// It exists because of one constraint: no host-device communication during evolution
// (gpu/ARCHITECTURE.md sec 3). Sizing the IR scratch slot by measuring the largest state
// on the host needs batches to measure; the evolution has none -- states arrive continuously
// and the largest is not knowable before the run starts. Sizing from a configured maximum instead would reintroduce the wrong-dedup-key
// exposure that fixing that measurement closed: above the bound, states fall back to 1-WL,
// which MERGES non-isomorphic states.
//
// So a worker sizes its scratch from its own state's edge and occurrence counts and claims
// exactly that, with no host involved and no fixed ceiling per state.
//
// There is no free. The persistent kernel lays the arena out as one region per block and a pool
// behind them (run_persistent_evolve): a block's scratch for each claim is its region, and only a
// state larger than its share of the region claims from the pool. Other callers reuse a slot and
// claim again only for a larger one. The whole arena resets at the end of the run.
//
// Exhaustion is a capacity overflow like any other: the claim fails, the caller records it and
// returns partial work. It cannot grow, because growing needs the host.

#include "hg_gpu/types.hpp"
#include "hg_gpu/cuda_check.hpp"
#include "hg_gpu/clear_batch.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

class DeviceArena {
public:
    struct View {
        uint32_t* base;
        uint64_t* cursor;      // words handed out so far
        uint64_t  capacity;    // words

        // Claim `words`, 8-byte aligned so the callers' uint64 views inside the block are
        // valid. Returns nullptr when the arena cannot hold them, and then takes nothing, so a
        // refused large claim leaves the rest for smaller ones. The cursor moves only by an
        // exchange that fits; claims are per state, not per item, so the retry on a lost
        // exchange is rare.
        __device__ uint32_t* claim(uint64_t words) {
            const unsigned long long padded = (words + 1ull) & ~1ull;   // keep every claim even
            unsigned long long* cur = reinterpret_cast<unsigned long long*>(cursor);
            unsigned long long off = *reinterpret_cast<volatile unsigned long long*>(cur);
            for (;;) {
                if (off > capacity || padded > capacity - off) return nullptr;
                const unsigned long long seen = atomicCAS(cur, off, off + padded);
                if (seen == off) return base + off;
                off = seen;
            }
        }
    };

    explicit DeviceArena(uint64_t capacity_words);

    ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    void reset(ClearBatch* batch = nullptr);

    View view();

    uint64_t capacity_words() const;

    // Words handed out. Reads across the boundary, so it is for AFTER a run, not during one.
    uint64_t used_words_host() const;

private:

    uint32_t* base_    = nullptr;
    uint64_t* cursor_  = nullptr;
    uint64_t  capacity_ = 0;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE