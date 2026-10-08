#include <gtest/gtest.h>

#include "hg_gpu/vertex_inverted_index.hpp"

#include <cuda_runtime.h>

#include <set>
#include <vector>

namespace {

using hg_gpu::EdgeId;
using hg_gpu::VertexId;

// =============================================================================
// VertexInvertedIndex
// =============================================================================

struct EdgeRef { EdgeId eid; VertexId v; };

__global__ void k_vidx_insert(hg_gpu::VertexInvertedIndex::DeviceView v,
                              const EdgeRef* refs, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    v.insert(refs[tid].v, refs[tid].eid);
}

__global__ void k_vidx_collect(hg_gpu::VertexInvertedIndex::DeviceView v,
                               VertexId q, EdgeId* out, uint32_t* out_n,
                               uint32_t cap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t cnt = 0;
    v.for_each_incident(q, [&](EdgeId eid) {
        if (cnt < cap) out[cnt] = eid;
        ++cnt;
    });
    *out_n = cnt;
}

TEST(VertexInvertedIndex, IncidenceTracking) {
    constexpr uint32_t kVerts = 4;
    constexpr uint32_t kCap   = 16;
    hg_gpu::VertexInvertedIndex idx(kVerts, kCap);

    // Edges:
    //   e0 = {0, 1}     → v0,v1
    //   e1 = {1, 2}     → v1,v2
    //   e2 = {0, 2, 3}  → v0,v2,v3
    //   e3 = {0, 0}     → v0 twice (self-loop)
    std::vector<EdgeRef> refs = {
        {0, 0}, {0, 1},
        {1, 1}, {1, 2},
        {2, 0}, {2, 2}, {2, 3},
        {3, 0}, {3, 0},
    };
    EdgeRef* d_refs = nullptr; cudaMalloc(&d_refs, sizeof(EdgeRef) * refs.size());
    cudaMemcpy(d_refs, refs.data(), sizeof(EdgeRef) * refs.size(), cudaMemcpyHostToDevice);
    k_vidx_insert<<<1, (uint32_t)refs.size()>>>(idx.view(), d_refs, (uint32_t)refs.size());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EdgeId*   d_out = nullptr; cudaMalloc(&d_out, sizeof(EdgeId) * kCap);
    uint32_t* d_n   = nullptr; cudaMalloc(&d_n,   sizeof(uint32_t));

    auto incident = [&](VertexId v) {
        k_vidx_collect<<<1, 1>>>(idx.view(), v, d_out, d_n, kCap);
        cudaDeviceSynchronize();
        uint32_t n = 0; cudaMemcpy(&n, d_n, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::vector<EdgeId> r(n);
        cudaMemcpy(r.data(), d_out, sizeof(EdgeId) * n, cudaMemcpyDeviceToHost);
        return r;
    };

    auto count_eq = [](const std::vector<EdgeId>& v, EdgeId e) {
        uint32_t c = 0; for (EdgeId x : v) if (x == e) ++c; return c;
    };

    auto v0 = incident(0); EXPECT_EQ(v0.size(), 4u);
    EXPECT_EQ(count_eq(v0, 0), 1u);
    EXPECT_EQ(count_eq(v0, 2), 1u);
    EXPECT_EQ(count_eq(v0, 3), 2u);  // self-loop
    auto v1 = incident(1); EXPECT_EQ(v1.size(), 2u);
    EXPECT_EQ(count_eq(v1, 0), 1u);
    EXPECT_EQ(count_eq(v1, 1), 1u);
    auto v2 = incident(2); EXPECT_EQ(v2.size(), 2u);
    auto v3 = incident(3); EXPECT_EQ(v3.size(), 1u);
    EXPECT_EQ(v3[0], 2u);

    cudaFree(d_refs); cudaFree(d_out); cudaFree(d_n);
}

}  // namespace
