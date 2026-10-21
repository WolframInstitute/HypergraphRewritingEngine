#include <gtest/gtest.h>

#include "hg_gpu/device_probe.hpp"
#include "hg_gpu/evolve.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// No rules and an empty initial state: the empty initial state is the whole evolution, as on the
// host (DifferentialEvolution/empty_rules_empty_initial_zero_steps).
TEST(HgGpuSmoke, EmptyEvolveYieldsTheEmptyInitialState) {
    hg_gpu::EvolveInput in;
    in.num_steps = 0;
    auto result = hg_gpu::evolve(in);
    ASSERT_EQ(result.states.size(), 1u);
    EXPECT_EQ(result.states[0].num_edges, 0u);
    EXPECT_TRUE(result.events.empty());
    EXPECT_TRUE(result.causal_edges.empty());
    EXPECT_TRUE(result.branchial_edges.empty());
}

TEST(HgGpuSmoke, NoRulesYieldsJustInitialState) {
    hg_gpu::EvolveInput in;
    in.initial_state = {{0u, 1u}, {1u, 2u}};
    in.num_steps = 5;
    auto result = hg_gpu::evolve(in);
    // No rules means no matches/events, so evolution can't leave the initial
    // state. The initial state itself is tracked, so exactly one state.
    EXPECT_EQ(result.states.size(), 1u);
    EXPECT_TRUE(result.events.empty());
    if (!result.states.empty()) {
        EXPECT_EQ(result.states[0].num_edges, 2u);
    }
}

// The device probe against the runtime's attribute queries, which read the same properties by a
// different call. On the RTX 4090: cc 8.9, 128 SMs, 1536 threads and 24 blocks per SM, 72 MB L2.
TEST(HgGpuSmoke, DeviceProbeMatchesTheAttributeQueries) {
    const hg_gpu::DeviceProbe& p = hg_gpu::probe_device();
    ASSERT_TRUE(p.usable) << p.reason;
    std::printf("[          ] %s cc %d.%d, %d SMs, %d threads/SM, %d blocks/SM, %d regs/SM, "
                "%llu B shared/SM, %llu B L2, %llu B memory\n",
                p.name.c_str(), p.major, p.minor, p.sm_count, p.max_threads_per_sm,
                p.max_blocks_per_sm, p.regs_per_sm, (unsigned long long)p.shared_per_sm,
                (unsigned long long)p.l2_bytes, (unsigned long long)p.total_bytes);
    auto attr = [](cudaDeviceAttr a) {
        int v = 0;
        EXPECT_EQ(cudaDeviceGetAttribute(&v, a, 0), cudaSuccess);
        return v;
    };
    EXPECT_EQ(p.major, attr(cudaDevAttrComputeCapabilityMajor));
    EXPECT_EQ(p.minor, attr(cudaDevAttrComputeCapabilityMinor));
    EXPECT_EQ(p.sm_count, attr(cudaDevAttrMultiProcessorCount));
    EXPECT_EQ(p.max_threads_per_sm, attr(cudaDevAttrMaxThreadsPerMultiProcessor));
    EXPECT_EQ(p.max_blocks_per_sm, attr(cudaDevAttrMaxBlocksPerMultiprocessor));
    EXPECT_EQ(p.regs_per_sm, attr(cudaDevAttrMaxRegistersPerMultiprocessor));
    EXPECT_EQ(p.shared_per_sm, (uint64_t)attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor));
    EXPECT_EQ(p.shared_per_block_optin, (uint64_t)attr(cudaDevAttrMaxSharedMemoryPerBlockOptin));
    EXPECT_EQ(p.l2_bytes, (uint64_t)attr(cudaDevAttrL2CacheSize));
    EXPECT_EQ(p.persisting_l2_max, (uint64_t)attr(cudaDevAttrMaxPersistingL2CacheSize));
    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    EXPECT_EQ(p.total_bytes, (uint64_t)total_b);

    EXPECT_EQ(hg_gpu::device_resident_threads(),
              (uint64_t)p.sm_count * (uint64_t)p.max_threads_per_sm);
    if (!std::getenv("HG_GPU_PERSISTENT_BLOCKS"))
        EXPECT_EQ(hg_gpu::default_persistent_grid(), (uint32_t)p.sm_count * 8u);
    // The derived cap is at most 90% of total and at most free - 256 MB.
    const uint64_t ninety = (uint64_t)((double)p.total_bytes * 0.90);
    if (!std::getenv("HG_TUNE_GPU_MEMORY_LIMIT")) {
        const uint64_t cap = hg_gpu::resolved_memory_cap(0);
        EXPECT_LE(cap, ninety);
        EXPECT_LE(cap, (uint64_t)free_b);
        EXPECT_GE(hg_gpu::resolved_memory_cap(0, uint64_t{1} << 40), ninety);
    }
    EXPECT_EQ(hg_gpu::memory_cap_of(ninety, uint64_t{1} << 40), ninety);
    EXPECT_EQ(hg_gpu::memory_cap_of(ninety, uint64_t{1} << 30), (uint64_t{1} << 30) - (uint64_t{256} << 20));
    EXPECT_EQ(hg_gpu::memory_cap_of(ninety, uint64_t{100} << 20), 1u);
    EXPECT_EQ(hg_gpu::resolved_memory_cap(12345), 12345u);
}
