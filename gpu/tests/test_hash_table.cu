#include <gtest/gtest.h>

#include "hg_gpu/hash_table.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

using Map = hg_gpu::ConcurrentMap<uint64_t, uint32_t>;

// =============================================================================
// Single-thread sanity
// =============================================================================

__global__ void k_single_insert_lookup(Map::DeviceView m,
                                       const uint64_t* keys,
                                       const uint32_t* vals,
                                       uint32_t n,
                                       uint32_t* out_inserted_count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t inserted = 0;
    for (uint32_t i = 0; i < n; ++i) {
        auto r = m.insert_if_absent(keys[i], vals[i]);
        if (r.inserted) ++inserted;
    }
    *out_inserted_count = inserted;
}

__global__ void k_single_lookup(Map::DeviceView m,
                                const uint64_t* keys,
                                uint32_t n,
                                uint32_t* out_values,
                                uint8_t* out_found) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (uint32_t i = 0; i < n; ++i) {
        auto r = m.lookup(keys[i]);
        out_values[i] = r.value;
        out_found[i]  = r.found ? 1 : 0;
    }
}

TEST(ConcurrentMap, SingleThreadInsertLookupRoundTrip) {
    Map map(64);
    constexpr uint32_t N = 10;

    std::vector<uint64_t> keys = {7, 13, 21, 1000, 99, 500, 17, 33, 71, 88};
    std::vector<uint32_t> vals = {70, 130, 210, 10000, 990, 5000, 170, 330, 710, 880};

    uint64_t* d_keys = nullptr;
    uint32_t* d_vals = nullptr;
    uint32_t* d_inserted = nullptr;
    cudaMalloc(&d_keys,    sizeof(uint64_t) * N);
    cudaMalloc(&d_vals,    sizeof(uint32_t) * N);
    cudaMalloc(&d_inserted, sizeof(uint32_t));
    cudaMemcpy(d_keys, keys.data(), sizeof(uint64_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals.data(), sizeof(uint32_t) * N, cudaMemcpyHostToDevice);

    k_single_insert_lookup<<<1, 1>>>(map.view(), d_keys, d_vals, N, d_inserted);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t inserted = 0;
    cudaMemcpy(&inserted, d_inserted, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(inserted, N);

    uint32_t* d_lookup_vals = nullptr;
    uint8_t*  d_lookup_found = nullptr;
    cudaMalloc(&d_lookup_vals,  sizeof(uint32_t) * N);
    cudaMalloc(&d_lookup_found, sizeof(uint8_t)  * N);

    k_single_lookup<<<1, 1>>>(map.view(), d_keys, N, d_lookup_vals, d_lookup_found);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> got_vals(N);
    std::vector<uint8_t>  got_found(N);
    cudaMemcpy(got_vals.data(),  d_lookup_vals,  sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_found.data(), d_lookup_found, sizeof(uint8_t)  * N, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_TRUE(got_found[i]) << "key " << keys[i] << " not found";
        EXPECT_EQ(got_vals[i], vals[i]) << "wrong value for key " << keys[i];
    }

    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_inserted);
    cudaFree(d_lookup_vals); cudaFree(d_lookup_found);
}

TEST(ConcurrentMap, LookupMissingReturnsNotFound) {
    Map map(32);
    auto v = map.view();
    uint64_t* d_keys = nullptr;
    cudaMalloc(&d_keys, sizeof(uint64_t) * 1);
    uint64_t key = 12345;
    cudaMemcpy(d_keys, &key, sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint32_t* d_lookup_vals = nullptr;
    uint8_t*  d_lookup_found = nullptr;
    cudaMalloc(&d_lookup_vals,  sizeof(uint32_t));
    cudaMalloc(&d_lookup_found, sizeof(uint8_t));

    k_single_lookup<<<1, 1>>>(v, d_keys, 1, d_lookup_vals, d_lookup_found);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t found = 0;
    cudaMemcpy(&found, d_lookup_found, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    EXPECT_FALSE(found);

    cudaFree(d_keys); cudaFree(d_lookup_vals); cudaFree(d_lookup_found);
}

// =============================================================================
// Concurrent insert: first-writer-wins. Every thread tries to insert (key, tid).
// All threads sharing a key must observe the same winner's value.
// =============================================================================

__global__ void k_concurrent_insert(Map::DeviceView m,
                                    const uint64_t* keys,
                                    uint32_t n,
                                    uint32_t* observed_value,
                                    uint8_t*  was_first) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    auto r = m.insert_if_absent(keys[tid], tid);
    observed_value[tid] = r.value;
    was_first[tid]      = r.inserted ? 1 : 0;
}

TEST(ConcurrentMap, ConcurrentInsertFirstWriterWinsNoDuplicates) {
    constexpr uint32_t kThreads     = 8192;
    constexpr uint32_t kUniqueKeys  = 1024;   // many threads contend on each key
    constexpr uint32_t kCapacity    = 4096;   // 4× load factor headroom
    Map map(kCapacity);

    std::vector<uint64_t> keys(kThreads);
    std::mt19937_64 rng(1234);
    for (uint32_t i = 0; i < kThreads; ++i) {
        // Avoid 0 (EMPTY) and ~0 (LOCKED). Spread across kUniqueKeys.
        keys[i] = (rng() % kUniqueKeys) + 1;
    }

    uint64_t* d_keys = nullptr;
    cudaMalloc(&d_keys, sizeof(uint64_t) * kThreads);
    cudaMemcpy(d_keys, keys.data(), sizeof(uint64_t) * kThreads, cudaMemcpyHostToDevice);

    uint32_t* d_observed = nullptr;
    uint8_t*  d_was_first = nullptr;
    cudaMalloc(&d_observed,  sizeof(uint32_t) * kThreads);
    cudaMalloc(&d_was_first, sizeof(uint8_t)  * kThreads);

    int block = 256;
    int grid  = (int)((kThreads + block - 1) / block);
    k_concurrent_insert<<<grid, block>>>(map.view(), d_keys, kThreads, d_observed, d_was_first);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> observed(kThreads);
    std::vector<uint8_t>  was_first(kThreads);
    cudaMemcpy(observed.data(),  d_observed,  sizeof(uint32_t) * kThreads, cudaMemcpyDeviceToHost);
    cudaMemcpy(was_first.data(), d_was_first, sizeof(uint8_t)  * kThreads, cudaMemcpyDeviceToHost);

    cudaFree(d_keys); cudaFree(d_observed); cudaFree(d_was_first);

    // Distinct keys actually inserted = number of `was_first` set.
    uint32_t winners = 0;
    for (uint8_t b : was_first) if (b) ++winners;
    EXPECT_LE(winners, kUniqueKeys);  // ≤ unique keys
    EXPECT_GE(winners, 1u);

    // For each key, all threads with that key must observe the SAME value
    // (the first writer's tid).
    std::vector<uint32_t> winning_value_for_key(kUniqueKeys + 1, 0xFFFFFFFFu);
    for (uint32_t i = 0; i < kThreads; ++i) {
        uint64_t k = keys[i];
        if (winning_value_for_key[k] == 0xFFFFFFFFu) {
            winning_value_for_key[k] = observed[i];
        } else {
            EXPECT_EQ(observed[i], winning_value_for_key[k])
                << "key " << k << " thread " << i << " saw different winner";
        }
    }

    // Every key that was claimed (some tid stored as winner) is unique.
    std::unordered_set<uint32_t> winner_tids;
    for (uint64_t k = 1; k <= kUniqueKeys; ++k) {
        uint32_t v = winning_value_for_key[k];
        if (v != 0xFFFFFFFFu) {
            EXPECT_TRUE(winner_tids.insert(v).second)
                << "winner tid " << v << " claimed by two distinct keys";
        }
    }
}

}  // namespace
