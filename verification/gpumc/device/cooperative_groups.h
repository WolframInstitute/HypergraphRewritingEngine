#pragma once
// The cooperative-groups subset the device headers use, over the harness's block as one warp.
#include "hg_gpumc.hpp"

namespace cooperative_groups {

struct thread_block {};
inline thread_block this_thread_block() { return {}; }

template <unsigned N>
struct thread_block_tile {
    unsigned thread_rank() const { return hg_gpumc::t_lane % N; }
    unsigned size() const { return N; }
    void sync() const { hg_gpumc::block_barrier(); }
    template <class T>
    T shfl(T v, unsigned src) const {
        return hg_gpumc::shfl(v, src, N);
    }
};

template <unsigned N, class Parent>
inline thread_block_tile<N> tiled_partition(const Parent&) {
    return {};
}

template <unsigned N, class T>
inline T exclusive_scan(const thread_block_tile<N>&, T v) {
    return hg_gpumc::exclusive_sum(v);
}

}  // namespace cooperative_groups
