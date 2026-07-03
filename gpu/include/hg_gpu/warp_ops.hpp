#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#include <cstdint>

namespace hg_gpu {

namespace cg = cooperative_groups;

// Warp-cooperative primitives built on Cooperative Groups
// thread_block_tile<N>. Tile size N ∈ {8, 16, 32} selects the virtual-warp
// width (Hong et al. "Accelerating CUDA Graph Algorithms at Maximum Warp",
// PPoPP'11): 32 maximises SIMD utilisation when per-task work is uniform; 8
// or 16 reduces intra-tile divergence at the cost of SIMD efficiency. The
// match kernel picks N via the auto-tuner.
//
// All operations assume all threads in the tile participate.
template <uint32_t N>
struct VWarp {
    static_assert(N == 8 || N == 16 || N == 32, "VWarp N must be 8, 16 or 32");

    using Tile = cg::thread_block_tile<N>;

    __device__ static Tile tile() {
        return cg::tiled_partition<N>(cg::this_thread_block());
    }

    // Any/all ballots over a per-lane predicate.
    __device__ static bool any(Tile t, bool pred) {
        return t.any(pred);
    }
    __device__ static bool all(Tile t, bool pred) {
        return t.all(pred);
    }

    // Per-lane broadcast of the value from `src_lane`.
    template <typename T>
    __device__ static T broadcast(Tile t, T value, uint32_t src_lane) {
        return t.shfl(value, src_lane);
    }

    // Reductions.
    template <typename T>
    __device__ static T reduce_sum(Tile t, T value) {
        return cg::reduce(t, value, cg::plus<T>());
    }
    template <typename T>
    __device__ static T reduce_min(Tile t, T value) {
        return cg::reduce(t, value, cg::less<T>());  // cg::less for min
    }
    template <typename T>
    __device__ static T reduce_max(Tile t, T value) {
        return cg::reduce(t, value, cg::greater<T>());  // cg::greater for max
    }

    // Inclusive / exclusive prefix-sum over per-lane values.
    template <typename T>
    __device__ static T scan_inclusive(Tile t, T value) {
        return cg::inclusive_scan(t, value, cg::plus<T>());
    }
    template <typename T>
    __device__ static T scan_exclusive(Tile t, T value) {
        return cg::exclusive_scan(t, value, cg::plus<T>());
    }

    // Parallel compact: each lane holds (value, keep). Lanes with keep=true
    // write their value into out[base + rank] where rank is the exclusive
    // scan of keep across the tile. Returns the total number of kept
    // elements (broadcast to all lanes).
    template <typename T>
    __device__ static uint32_t compact(Tile t, T value, bool keep, T* out, uint32_t base) {
        uint32_t k   = keep ? 1u : 0u;
        uint32_t rnk = scan_exclusive(t, k);
        uint32_t tot = reduce_sum(t, k);
        if (keep) out[base + rnk] = value;
        return tot;
    }

    // Intersect two sorted uint32_t ranges (in shared or global memory) and
    // write the intersection into out. Merge-intersect with bounded loops.
    // Returns the intersection size (broadcast to all lanes).
    //
    // All lanes of the tile cooperate: each lane binary-searches one chunk
    // of A for a B-partition boundary, and the ranks are scanned to emit a
    // compacted output.
    //
    // Simplified implementation: serial pass on lane 0 of the tile. Adequate
    // for small candidate sets; replace with a true parallel intersect when
    // profiling shows it matters (see M4.7).
    __device__ static uint32_t intersect_sorted(Tile t,
                                                const uint32_t* a, uint32_t na,
                                                const uint32_t* b, uint32_t nb,
                                                uint32_t* out) {
        uint32_t n = 0;
        if (t.thread_rank() == 0) {
            uint32_t i = 0, j = 0;
            while (i < na && j < nb) {
                if      (a[i] < b[j]) ++i;
                else if (a[i] > b[j]) ++j;
                else                  { out[n++] = a[i]; ++i; ++j; }
            }
        }
        n = t.shfl(n, 0);
        return n;
    }
};

}  // namespace hg_gpu
