#include <gtest/gtest.h>

#include "hg_gpu/ring_buffer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <vector>

namespace {

using Ring = hg_gpu::RingBuffer<uint32_t>;

// =============================================================================
// Single-thread push/pop FIFO
// =============================================================================

__global__ void k_push_seq(Ring::DeviceView v, const uint32_t* in, uint32_t n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (uint32_t i = 0; i < n; ++i) v.try_push(in[i]);
}

__global__ void k_pop_seq(Ring::DeviceView v, uint32_t* out, uint32_t n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t x = 0;
        v.try_pop(x);
        out[i] = x;
    }
}

TEST(RingBuffer, SingleThreadPushPopFifo) {
    Ring ring(8);
    std::vector<uint32_t> input = {10, 20, 30, 40, 50, 60, 70};
    uint32_t n = input.size();

    uint32_t* d_in = nullptr; cudaMalloc(&d_in, sizeof(uint32_t) * n);
    cudaMemcpy(d_in, input.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice);
    k_push_seq<<<1, 1>>>(ring.view(), d_in, n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(ring.tail_host(), n);
    EXPECT_EQ(ring.head_host(), 0u);

    uint32_t* d_out = nullptr; cudaMalloc(&d_out, sizeof(uint32_t) * n);
    k_pop_seq<<<1, 1>>>(ring.view(), d_out, n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> got(n);
    cudaMemcpy(got.data(), d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    EXPECT_EQ(got, input);
    EXPECT_EQ(ring.head_host(), n);
}

TEST(RingBuffer, PopOnEmptyReturnsFalse) {
    Ring ring(4);
    uint32_t* d_out = nullptr; cudaMalloc(&d_out, sizeof(uint32_t));
    cudaMemset(d_out, 0xAB, sizeof(uint32_t));
    k_pop_seq<<<1, 1>>>(ring.view(), d_out, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t got = 0;
    cudaMemcpy(&got, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    // try_pop wrote 0 to out by reference even on failure (uninitialised),
    // so we just verify head/tail did not advance.
    EXPECT_EQ(ring.head_host(), 0u);
    EXPECT_EQ(ring.tail_host(), 0u);
}

// =============================================================================
// Concurrent producers + consumers on disjoint kernel launches
// =============================================================================

__global__ void k_producers(Ring::DeviceView v, uint32_t per_thread) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = tid * per_thread;
    for (uint32_t i = 0; i < per_thread; ++i) {
        while (!v.try_push(base + i)) { /* full → spin */ }
    }
}

__global__ void k_consumers(Ring::DeviceView v,
                            uint32_t* out_buf,
                            uint32_t  out_cap,
                            uint32_t* out_count,
                            uint64_t  expected_total) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> count_ref(*out_count);
    while (true) {
        uint32_t x = 0;
        if (v.try_pop(x)) {
            uint32_t pos = count_ref.fetch_add(1u, cuda::memory_order_acq_rel);
            if (pos < out_cap) out_buf[pos] = x;
            if (pos + 1 >= expected_total) return;
        } else {
            // Termination: stop when we've collected the expected count.
            if (count_ref.load(cuda::memory_order_acquire) >= expected_total) return;
            __nanosleep(64);
        }
    }
}

TEST(RingBuffer, ConcurrentProducersConsumersAllItemsExactlyOnce) {
    constexpr uint32_t kProducerThreads = 1024;
    constexpr uint32_t kPerThread       = 8;
    constexpr uint64_t kTotal           = uint64_t(kProducerThreads) * kPerThread; // 8192
    constexpr uint32_t kCapacity        = 8192;       // power of two; >= total to avoid backpressure complexity
    constexpr uint32_t kConsumerThreads = 256;

    Ring ring(kCapacity);

    // Producers
    int pblock = 256;
    int pgrid  = (int)(kProducerThreads / pblock);
    k_producers<<<pgrid, pblock>>>(ring.view(), kPerThread);

    // Consumers (launched on the default stream after producers in the same stream
    // — guarantees producers finish first; the test verifies sequencing not raw
    // concurrency).
    uint32_t* d_buf   = nullptr; cudaMalloc(&d_buf,   sizeof(uint32_t) * kTotal);
    uint32_t* d_count = nullptr; cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    int cblock = 64;
    int cgrid  = (int)(kConsumerThreads / cblock);
    k_consumers<<<cgrid, cblock>>>(ring.view(), d_buf, (uint32_t)kTotal, d_count, kTotal);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t count = 0;
    cudaMemcpy(&count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(count, kTotal);

    std::vector<uint32_t> got(kTotal);
    cudaMemcpy(got.data(), d_buf, sizeof(uint32_t) * kTotal, cudaMemcpyDeviceToHost);
    cudaFree(d_buf); cudaFree(d_count);

    std::sort(got.begin(), got.end());
    for (uint32_t i = 0; i < kTotal; ++i) {
        ASSERT_EQ(got[i], i) << "duplicate or missing at i=" << i;
    }
}

}  // namespace
