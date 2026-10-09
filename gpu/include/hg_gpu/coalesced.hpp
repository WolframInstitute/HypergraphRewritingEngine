#pragma once
#include "hgcommon/namespace.hpp"
// Counter reservations that one lane makes for a warp. apply_one_match (rewrite.cu) reserves its
// state, event, edges, slice, vertex slots and fresh ids through these; test_rewrite.cu drives
// them from subsets of one warp.
//
// THEY AGGREGATE OVER A FULL WARP ONLY. When all 32 lanes reach one together on one counter,
// lane 0 makes the atomic and a shuffle over the whole warp hands out the shares; any other set
// of threads makes one atomicAdd per thread. A coalesced subset of a warp (cg::coalesced_threads)
// is not used: nvcc compiles a shuffle over a subset's mask as a convergence test on one lane's
// copy of the mask (R2UR, then BRA.DIV). When lanes of two subsets of one warp reach that shuffle
// together while one subset's first lane is still in its atomic, the test passes on the other
// subset's mask, the waiting lanes read an inactive lane, and the first lane then waits in
// WARPSYNC for lanes that have left. The full-warp mask is the same on every lane.

#include "hg_gpu/types.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

constexpr uint32_t kFullWarp = 0xFFFFFFFFu;

// True on every lane when all 32 lanes are here and pass the same counter, false on every lane
// otherwise. Lanes of one warp at one instruction can come from different loop iterations or
// merged call sites, so a full warp alone does not mean one counter.
__device__ __forceinline__ bool whole_warp_on(const uint32_t* counter) {
    if (__activemask() != kFullWarp) return false;
    int same = 0;
    __match_all_sync(kFullWarp, reinterpret_cast<unsigned long long>(counter), &same);
    return same != 0;
}

// An add of `n` to `counter`: one atomicAdd for a full warp, and each lane's share starts at the
// exclusive prefix of the shares of the lanes below it. Any other set of threads makes one
// atomicAdd per thread, and a thread asking for 0 makes none. Returns what the thread's own
// atomicAdd would have returned had the warp's adds run in lane order; a thread asking for 0 gets
// an unspecified value.
__device__ __forceinline__ uint32_t coalesced_add(uint32_t* counter, uint32_t n) {
    namespace cg = cooperative_groups;
    if (!whole_warp_on(counter)) return n ? atomicAdd(counter, n) : 0u;
    const cg::thread_block_tile<32> w = cg::tiled_partition<32>(cg::this_thread_block());
    const uint32_t prefix = cg::exclusive_scan(w, n);
    const uint32_t total = w.shfl(prefix + n, 31);
    uint32_t base = 0;
    if (w.thread_rank() == 0 && total) base = atomicAdd(counter, total);
    return w.shfl(base, 0) + prefix;
}

// A claim of one slot from `counter`, which ends every call at or below `limit`: a full warp's lane
// 0 takes as many of the 32 slots as fit in one exchange, and a lane past them gets none. Any other
// set of threads makes one atomicAdd per thread, and a thread whose add lands at or past `limit`
// lowers the counter back to `limit` (Pool::settle's rule); between those two atomics the counter
// can read above `limit`, and no device code reads it. Returns the slot, or INVALID_ID when none
// fit.
__device__ __forceinline__ uint32_t coalesced_bounded_claim(uint32_t* counter, uint32_t limit) {
    namespace cg = cooperative_groups;
    const auto take_from = [&](uint32_t want, uint32_t& base) {
        uint32_t cur = *counter;
        for (;;) {
            const uint32_t take = cur >= limit ? 0u : min(want, limit - cur);
            if (take == 0) return 0u;
            const uint32_t prev = atomicCAS(counter, cur, cur + take);
            if (prev == cur) { base = cur; return take; }
            cur = prev;
        }
    };
    if (!whole_warp_on(counter)) {
        const uint32_t at = atomicAdd(counter, 1u);
        if (at < limit) return at;
        atomicMin(counter, limit);
        return INVALID_ID;
    }
    uint32_t base = 0;
    const cg::thread_block_tile<32> w = cg::tiled_partition<32>(cg::this_thread_block());
    uint32_t take = 0;
    if (w.thread_rank() == 0) take = take_from(32u, base);
    base = w.shfl(base, 0);
    take = w.shfl(take, 0);
    return w.thread_rank() < take ? base + w.thread_rank() : INVALID_ID;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
