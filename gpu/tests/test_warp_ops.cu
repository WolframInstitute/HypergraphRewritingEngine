#include <gtest/gtest.h>

#include "hg_gpu/warp_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace {

template <uint32_t N>
__global__ void k_reduce_sum(uint32_t* out, uint32_t value) {
    auto t = hg_gpu::VWarp<N>::tile();
    uint32_t lane = t.thread_rank();
    uint32_t s = hg_gpu::VWarp<N>::reduce_sum(t, lane * value);
    if (lane == 0) *out = s;
}

template <uint32_t N>
__global__ void k_scan_exclusive(uint32_t* out) {
    auto t = hg_gpu::VWarp<N>::tile();
    uint32_t lane = t.thread_rank();
    uint32_t v = 1;  // each lane contributes 1
    uint32_t rnk = hg_gpu::VWarp<N>::scan_exclusive(t, v);
    out[lane] = rnk;
}

template <uint32_t N>
__global__ void k_compact(const uint32_t* in, uint32_t* out, uint32_t* count) {
    auto t = hg_gpu::VWarp<N>::tile();
    uint32_t lane = t.thread_rank();
    uint32_t v = in[lane];
    bool keep = (v % 2 == 0);  // keep evens
    uint32_t k = hg_gpu::VWarp<N>::compact(t, v, keep, out, 0);
    if (lane == 0) *count = k;
}

template <uint32_t N>
__global__ void k_intersect(const uint32_t* a, uint32_t na,
                            const uint32_t* b, uint32_t nb,
                            uint32_t* out, uint32_t* out_n) {
    auto t = hg_gpu::VWarp<N>::tile();
    uint32_t n = hg_gpu::VWarp<N>::intersect_sorted(t, a, na, b, nb, out);
    if (t.thread_rank() == 0) *out_n = n;
}

TEST(WarpOps, ReduceSum32) {
    uint32_t* d = nullptr; cudaMalloc(&d, sizeof(uint32_t));
    k_reduce_sum<32><<<1, 32>>>(d, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t h = 0; cudaMemcpy(&h, d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h, 0u + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 +
                 14 + 15 + 16 + 17 + 18 + 19 + 20 + 21 + 22 + 23 + 24 + 25 +
                 26 + 27 + 28 + 29 + 30 + 31);
    cudaFree(d);
}

TEST(WarpOps, ReduceSum16) {
    uint32_t* d = nullptr; cudaMalloc(&d, sizeof(uint32_t));
    k_reduce_sum<16><<<1, 16>>>(d, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t h = 0; cudaMemcpy(&h, d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h, 0u + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15);
    cudaFree(d);
}

TEST(WarpOps, ScanExclusive32) {
    uint32_t* d = nullptr; cudaMalloc(&d, sizeof(uint32_t) * 32);
    k_scan_exclusive<32><<<1, 32>>>(d);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint32_t> h(32);
    cudaMemcpy(h.data(), d, sizeof(uint32_t) * 32, cudaMemcpyDeviceToHost);
    cudaFree(d);
    for (uint32_t i = 0; i < 32; ++i) EXPECT_EQ(h[i], i) << "lane " << i;
}

TEST(WarpOps, Compact32KeepsEvens) {
    std::vector<uint32_t> input(32);
    for (uint32_t i = 0; i < 32; ++i) input[i] = i;

    uint32_t* d_in   = nullptr; cudaMalloc(&d_in,   sizeof(uint32_t) * 32);
    uint32_t* d_out  = nullptr; cudaMalloc(&d_out,  sizeof(uint32_t) * 32);
    uint32_t* d_cnt  = nullptr; cudaMalloc(&d_cnt,  sizeof(uint32_t));
    cudaMemcpy(d_in, input.data(), sizeof(uint32_t) * 32, cudaMemcpyHostToDevice);

    k_compact<32><<<1, 32>>>(d_in, d_out, d_cnt);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t cnt = 0;
    cudaMemcpy(&cnt, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cnt, 16u);

    std::vector<uint32_t> got(16);
    cudaMemcpy(got.data(), d_out, sizeof(uint32_t) * 16, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_cnt);

    for (uint32_t i = 0; i < 16; ++i) EXPECT_EQ(got[i], i * 2);
}

TEST(WarpOps, IntersectSorted) {
    std::vector<uint32_t> a = {1, 3, 5, 7, 9, 11, 13};
    std::vector<uint32_t> b = {2, 3, 4, 5, 10, 11, 12};
    // expected intersection: {3, 5, 11}

    uint32_t *d_a = nullptr, *d_b = nullptr, *d_out = nullptr, *d_n = nullptr;
    cudaMalloc(&d_a,   sizeof(uint32_t) * a.size());
    cudaMalloc(&d_b,   sizeof(uint32_t) * b.size());
    cudaMalloc(&d_out, sizeof(uint32_t) * 8);
    cudaMalloc(&d_n,   sizeof(uint32_t));
    cudaMemcpy(d_a, a.data(), sizeof(uint32_t) * a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(uint32_t) * b.size(), cudaMemcpyHostToDevice);

    k_intersect<16><<<1, 16>>>(d_a, (uint32_t)a.size(), d_b, (uint32_t)b.size(), d_out, d_n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t n = 0; cudaMemcpy(&n, d_n, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(n, 3u);
    std::vector<uint32_t> out(3);
    cudaMemcpy(out.data(), d_out, sizeof(uint32_t) * 3, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_n);
    EXPECT_EQ(out, (std::vector<uint32_t>{3, 5, 11}));
}

}  // namespace
