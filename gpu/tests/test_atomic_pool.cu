#include <gtest/gtest.h>

#include "hg_gpu/atomic_pool.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace {

using hg_gpu::Pool;

__global__ void claim_one_each(Pool<uint32_t>::DeviceView view, uint32_t* out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = view.claim();
    out[tid] = idx;
    if (idx != Pool<uint32_t>::kInvalid) {
        view.at(idx) = tid;
    }
}

__global__ void claim_n_per_warp(Pool<uint32_t>::DeviceView view, uint32_t n_each, uint32_t* out) {
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t lane    = threadIdx.x & 31;
    if (lane != 0) return;
    uint32_t base = view.claim_n(n_each);
    out[warp_id] = base;
}

TEST(AtomicPool, SingleThreadClaimGivesSequentialIndices) {
    Pool<uint32_t> pool(4);
    uint32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t) * 4), cudaSuccess);

    claim_one_each<<<1, 4>>>(pool.view(), d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> got(4);
    ASSERT_EQ(cudaMemcpy(got.data(), d_out, sizeof(uint32_t) * 4, cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_out);

    EXPECT_EQ(pool.size_host(), 4u);
    std::vector<uint32_t> sorted = got;
    std::sort(sorted.begin(), sorted.end());
    for (uint32_t i = 0; i < 4; ++i) EXPECT_EQ(sorted[i], i);
}

TEST(AtomicPool, MultiWarpClaimNoDuplicates) {
    constexpr uint32_t kCapacity = 32 * 1024;
    constexpr uint32_t kThreads  = 4096;
    Pool<uint32_t> pool(kCapacity);

    uint32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t) * kThreads), cudaSuccess);

    int block = 256;
    int grid  = (int)(kThreads / block);
    claim_one_each<<<grid, block>>>(pool.view(), d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> got(kThreads);
    ASSERT_EQ(cudaMemcpy(got.data(), d_out, sizeof(uint32_t) * kThreads, cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_out);

    EXPECT_EQ(pool.size_host(), kThreads);
    std::sort(got.begin(), got.end());
    for (uint32_t i = 0; i < kThreads; ++i) {
        ASSERT_EQ(got[i], i) << "duplicate or gap at i=" << i;
    }
}

TEST(AtomicPool, OverflowReportsInvalid) {
    constexpr uint32_t kCapacity = 100;
    constexpr uint32_t kThreads  = 256;
    Pool<uint32_t> pool(kCapacity);

    uint32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t) * kThreads), cudaSuccess);

    claim_one_each<<<1, kThreads>>>(pool.view(), d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> got(kThreads);
    ASSERT_EQ(cudaMemcpy(got.data(), d_out, sizeof(uint32_t) * kThreads, cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_out);

    uint32_t valid = 0, invalid = 0;
    for (auto v : got) (v == Pool<uint32_t>::kInvalid ? ++invalid : ++valid);
    EXPECT_EQ(valid, kCapacity);
    EXPECT_EQ(invalid, kThreads - kCapacity);
}

TEST(AtomicPool, ClaimNAllocatesContiguous) {
    constexpr uint32_t kCapacity = 4096;
    constexpr uint32_t kWarps    = 32;
    constexpr uint32_t kPerWarp  = 7;
    Pool<uint32_t> pool(kCapacity);

    uint32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t) * kWarps), cudaSuccess);

    claim_n_per_warp<<<1, kWarps * 32>>>(pool.view(), kPerWarp, d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> bases(kWarps);
    ASSERT_EQ(cudaMemcpy(bases.data(), d_out, sizeof(uint32_t) * kWarps, cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_out);

    EXPECT_EQ(pool.size_host(), kWarps * kPerWarp);
    std::sort(bases.begin(), bases.end());
    for (uint32_t i = 0; i < kWarps; ++i) {
        EXPECT_EQ(bases[i], i * kPerWarp) << "non-contiguous bases";
    }
}

TEST(AtomicPool, ResetClearsCounter) {
    Pool<uint32_t> pool(16);
    uint32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t) * 4), cudaSuccess);

    claim_one_each<<<1, 4>>>(pool.view(), d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(pool.size_host(), 4u);

    pool.reset();
    EXPECT_EQ(pool.size_host(), 0u);

    claim_one_each<<<1, 4>>>(pool.view(), d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(pool.size_host(), 4u);

    cudaFree(d_out);
}

}  // namespace
