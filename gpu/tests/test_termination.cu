#include <gtest/gtest.h>

#include "hg_gpu/termination.hpp"

#include <cuda_runtime.h>

namespace {

using TD = hg_gpu::TerminationDetector;

__global__ void k_mark_pushed(TD::DeviceView v, uint32_t role, uint64_t n) {
    v.mark_pushed(role, n);
}

__global__ void k_mark_completed(TD::DeviceView v, uint32_t role, uint64_t n) {
    v.mark_completed(role, n);
}

__global__ void k_check_quiescent(TD::DeviceView v, uint8_t* out_q, uint64_t* out_p, uint64_t* out_c) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *out_q = v.snapshot_quiescent(out_p, out_c) ? 1 : 0;
}

__global__ void k_signal_exit(TD::DeviceView v) { v.signal_exit(); }
__global__ void k_check_exit (TD::DeviceView v, uint8_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *out = v.exit_requested() ? 1 : 0;
}

TEST(TerminationDetector, FreshIsQuiescent) {
    TD td(3);
    uint8_t*  d_q = nullptr; cudaMalloc(&d_q, sizeof(uint8_t));
    uint64_t* d_p = nullptr; cudaMalloc(&d_p, sizeof(uint64_t) * TD::kMaxRoles);
    uint64_t* d_c = nullptr; cudaMalloc(&d_c, sizeof(uint64_t) * TD::kMaxRoles);

    k_check_quiescent<<<1,1>>>(td.view(), d_q, d_p, d_c);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t q = 0; cudaMemcpy(&q, d_q, 1, cudaMemcpyDeviceToHost);
    EXPECT_EQ(q, 1);

    cudaFree(d_q); cudaFree(d_p); cudaFree(d_c);
}

TEST(TerminationDetector, PushedWithoutCompletedNotQuiescent) {
    TD td(3);
    k_mark_pushed<<<1, 32>>>(td.view(), /*role=*/1, /*n=*/2);  // 32 threads each push 2 → 64
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t*  d_q = nullptr; cudaMalloc(&d_q, sizeof(uint8_t));
    uint64_t* d_p = nullptr; cudaMalloc(&d_p, sizeof(uint64_t) * TD::kMaxRoles);
    uint64_t* d_c = nullptr; cudaMalloc(&d_c, sizeof(uint64_t) * TD::kMaxRoles);

    k_check_quiescent<<<1,1>>>(td.view(), d_q, d_p, d_c);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t q = 0; uint64_t p[TD::kMaxRoles] = {0}; uint64_t c[TD::kMaxRoles] = {0};
    cudaMemcpy(&q, d_q, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(p,  d_p, sizeof(p), cudaMemcpyDeviceToHost);
    cudaMemcpy(c,  d_c, sizeof(c), cudaMemcpyDeviceToHost);

    EXPECT_EQ(q, 0);
    EXPECT_EQ(p[1], 64u);
    EXPECT_EQ(c[1], 0u);

    cudaFree(d_q); cudaFree(d_p); cudaFree(d_c);
}

TEST(TerminationDetector, BalancedIsQuiescent) {
    TD td(3);
    k_mark_pushed   <<<1, 32>>>(td.view(), 0, 1);
    k_mark_completed<<<1, 32>>>(td.view(), 0, 1);
    k_mark_pushed   <<<1, 64>>>(td.view(), 2, 3);
    k_mark_completed<<<1, 64>>>(td.view(), 2, 3);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t*  d_q = nullptr; cudaMalloc(&d_q, sizeof(uint8_t));
    uint64_t* d_p = nullptr; cudaMalloc(&d_p, sizeof(uint64_t) * TD::kMaxRoles);
    uint64_t* d_c = nullptr; cudaMalloc(&d_c, sizeof(uint64_t) * TD::kMaxRoles);
    k_check_quiescent<<<1,1>>>(td.view(), d_q, d_p, d_c);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t q = 0; cudaMemcpy(&q, d_q, 1, cudaMemcpyDeviceToHost);
    EXPECT_EQ(q, 1);

    cudaFree(d_q); cudaFree(d_p); cudaFree(d_c);
}

TEST(TerminationDetector, SignalAndCheckExit) {
    TD td(2);
    EXPECT_FALSE(td.exit_requested_host());

    k_signal_exit<<<1, 1>>>(td.view());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_TRUE(td.exit_requested_host());

    uint8_t* d_e = nullptr; cudaMalloc(&d_e, 1);
    k_check_exit<<<1,1>>>(td.view(), d_e);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint8_t e = 0; cudaMemcpy(&e, d_e, 1, cudaMemcpyDeviceToHost);
    EXPECT_EQ(e, 1);
    cudaFree(d_e);

    td.clear();
    EXPECT_FALSE(td.exit_requested_host());
}

}  // namespace
