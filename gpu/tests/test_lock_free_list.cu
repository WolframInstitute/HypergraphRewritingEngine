#include <gtest/gtest.h>

#include "hg_gpu/lock_free_list.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <set>
#include <unordered_set>
#include <vector>

namespace {

using List = hg_gpu::LockFreeList<uint32_t>;

__global__ void k_push_per_thread(List::DeviceView v,
                                  const uint32_t* keys,
                                  const uint32_t* values,
                                  uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    v.push(keys[tid], values[tid]);
}

__global__ void k_collect(List::DeviceView v,
                          const uint32_t* keys_to_walk,
                          uint32_t* counts,
                          uint32_t* flat_values,
                          uint32_t  per_key_cap,
                          uint32_t  n_keys) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_keys) return;
    uint32_t key = keys_to_walk[tid];
    uint32_t* out_base = flat_values + tid * per_key_cap;
    uint32_t  cnt = 0;
    v.for_each(key, [&] (uint32_t value) {
        if (cnt < per_key_cap) out_base[cnt] = value;
        ++cnt;
    });
    counts[tid] = cnt;
}

TEST(LockFreeList, SingleKeyMultiplePushesIterateAll) {
    constexpr uint32_t kKeys = 1;
    constexpr uint32_t kPushes = 100;
    List list(kKeys, /*pool_capacity*/ kPushes);

    std::vector<uint32_t> keys(kPushes, 0);
    std::vector<uint32_t> vals(kPushes);
    for (uint32_t i = 0; i < kPushes; ++i) vals[i] = i + 1;  // avoid 0

    uint32_t* d_keys = nullptr; cudaMalloc(&d_keys, sizeof(uint32_t) * kPushes);
    uint32_t* d_vals = nullptr; cudaMalloc(&d_vals, sizeof(uint32_t) * kPushes);
    cudaMemcpy(d_keys, keys.data(), sizeof(uint32_t) * kPushes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals.data(), sizeof(uint32_t) * kPushes, cudaMemcpyHostToDevice);

    int block = 64;
    int grid = (int)((kPushes + block - 1) / block);
    k_push_per_thread<<<grid, block>>>(list.view(), d_keys, d_vals, kPushes);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(list.pool_used_host(), kPushes);

    uint32_t* d_walk_keys = nullptr; cudaMalloc(&d_walk_keys, sizeof(uint32_t));
    uint32_t k0 = 0; cudaMemcpy(d_walk_keys, &k0, sizeof(uint32_t), cudaMemcpyHostToDevice);
    uint32_t* d_counts = nullptr; cudaMalloc(&d_counts, sizeof(uint32_t));
    uint32_t* d_flat   = nullptr; cudaMalloc(&d_flat,   sizeof(uint32_t) * kPushes);

    k_collect<<<1, 1>>>(list.view(), d_walk_keys, d_counts, d_flat, kPushes, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t cnt = 0;
    cudaMemcpy(&cnt, d_counts, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cnt, kPushes);

    std::vector<uint32_t> got(kPushes);
    cudaMemcpy(got.data(), d_flat, sizeof(uint32_t) * kPushes, cudaMemcpyDeviceToHost);
    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_walk_keys); cudaFree(d_counts); cudaFree(d_flat);

    std::sort(got.begin(), got.end());
    for (uint32_t i = 0; i < kPushes; ++i) EXPECT_EQ(got[i], i + 1);
}

TEST(LockFreeList, MultipleKeysIsolated) {
    constexpr uint32_t kKeys = 4;
    constexpr uint32_t kPushesPerKey = 50;
    constexpr uint32_t kTotal = kKeys * kPushesPerKey;
    List list(kKeys, kTotal);

    std::vector<uint32_t> keys(kTotal);
    std::vector<uint32_t> vals(kTotal);
    for (uint32_t i = 0; i < kTotal; ++i) {
        keys[i] = i % kKeys;
        vals[i] = i + 1;
    }

    uint32_t* d_keys = nullptr; cudaMalloc(&d_keys, sizeof(uint32_t) * kTotal);
    uint32_t* d_vals = nullptr; cudaMalloc(&d_vals, sizeof(uint32_t) * kTotal);
    cudaMemcpy(d_keys, keys.data(), sizeof(uint32_t) * kTotal, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals.data(), sizeof(uint32_t) * kTotal, cudaMemcpyHostToDevice);

    int block = 64;
    int grid  = (int)((kTotal + block - 1) / block);
    k_push_per_thread<<<grid, block>>>(list.view(), d_keys, d_vals, kTotal);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> walk_keys(kKeys);
    for (uint32_t i = 0; i < kKeys; ++i) walk_keys[i] = i;

    uint32_t* d_walk = nullptr; cudaMalloc(&d_walk, sizeof(uint32_t) * kKeys);
    cudaMemcpy(d_walk, walk_keys.data(), sizeof(uint32_t) * kKeys, cudaMemcpyHostToDevice);
    uint32_t* d_counts = nullptr; cudaMalloc(&d_counts, sizeof(uint32_t) * kKeys);
    uint32_t* d_flat   = nullptr; cudaMalloc(&d_flat,   sizeof(uint32_t) * kKeys * kPushesPerKey);

    k_collect<<<1, kKeys>>>(list.view(), d_walk, d_counts, d_flat, kPushesPerKey, kKeys);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> counts(kKeys);
    cudaMemcpy(counts.data(), d_counts, sizeof(uint32_t) * kKeys, cudaMemcpyDeviceToHost);
    for (uint32_t k = 0; k < kKeys; ++k) {
        EXPECT_EQ(counts[k], kPushesPerKey) << "key " << k;
    }

    std::vector<uint32_t> flat(kKeys * kPushesPerKey);
    cudaMemcpy(flat.data(), d_flat, sizeof(uint32_t) * kKeys * kPushesPerKey, cudaMemcpyDeviceToHost);

    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_walk); cudaFree(d_counts); cudaFree(d_flat);

    // Verify per-key sets are disjoint and partition the input vals.
    std::unordered_set<uint32_t> all_seen;
    for (uint32_t k = 0; k < kKeys; ++k) {
        std::set<uint32_t> per_key(flat.begin() + k * kPushesPerKey,
                                   flat.begin() + (k + 1) * kPushesPerKey);
        EXPECT_EQ(per_key.size(), kPushesPerKey);
        for (uint32_t v : per_key) {
            EXPECT_EQ(((v - 1) % kKeys), k) << "value " << v << " under wrong key " << k;
            EXPECT_TRUE(all_seen.insert(v).second) << "value " << v << " seen twice";
        }
    }
    EXPECT_EQ(all_seen.size(), kTotal);
}

TEST(LockFreeList, ConcurrentPushesNoLostNodes) {
    constexpr uint32_t kKeys = 16;
    constexpr uint32_t kThreads = 4096;
    List list(kKeys, kThreads);

    std::vector<uint32_t> keys(kThreads);
    std::vector<uint32_t> vals(kThreads);
    for (uint32_t i = 0; i < kThreads; ++i) {
        keys[i] = i % kKeys;
        vals[i] = i + 1;
    }

    uint32_t* d_keys = nullptr; cudaMalloc(&d_keys, sizeof(uint32_t) * kThreads);
    uint32_t* d_vals = nullptr; cudaMalloc(&d_vals, sizeof(uint32_t) * kThreads);
    cudaMemcpy(d_keys, keys.data(), sizeof(uint32_t) * kThreads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals.data(), sizeof(uint32_t) * kThreads, cudaMemcpyHostToDevice);

    int block = 256;
    int grid  = (int)(kThreads / block);
    k_push_per_thread<<<grid, block>>>(list.view(), d_keys, d_vals, kThreads);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(list.pool_used_host(), kThreads);

    std::vector<uint32_t> walk_keys(kKeys);
    for (uint32_t i = 0; i < kKeys; ++i) walk_keys[i] = i;

    uint32_t per_key_cap = kThreads / kKeys + 8;
    uint32_t* d_walk = nullptr; cudaMalloc(&d_walk, sizeof(uint32_t) * kKeys);
    cudaMemcpy(d_walk, walk_keys.data(), sizeof(uint32_t) * kKeys, cudaMemcpyHostToDevice);
    uint32_t* d_counts = nullptr; cudaMalloc(&d_counts, sizeof(uint32_t) * kKeys);
    uint32_t* d_flat   = nullptr; cudaMalloc(&d_flat,   sizeof(uint32_t) * kKeys * per_key_cap);

    k_collect<<<1, kKeys>>>(list.view(), d_walk, d_counts, d_flat, per_key_cap, kKeys);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> counts(kKeys);
    cudaMemcpy(counts.data(), d_counts, sizeof(uint32_t) * kKeys, cudaMemcpyDeviceToHost);
    uint32_t total_seen = 0;
    for (uint32_t c : counts) total_seen += c;
    EXPECT_EQ(total_seen, kThreads);

    std::vector<uint32_t> flat(kKeys * per_key_cap);
    cudaMemcpy(flat.data(), d_flat, sizeof(uint32_t) * kKeys * per_key_cap, cudaMemcpyDeviceToHost);

    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_walk); cudaFree(d_counts); cudaFree(d_flat);

    std::unordered_set<uint32_t> all_seen;
    for (uint32_t k = 0; k < kKeys; ++k) {
        for (uint32_t i = 0; i < counts[k]; ++i) {
            uint32_t v = flat[k * per_key_cap + i];
            EXPECT_EQ(((v - 1) % kKeys), k);
            EXPECT_TRUE(all_seen.insert(v).second);
        }
    }
    EXPECT_EQ(all_seen.size(), kThreads);
}

}  // namespace
