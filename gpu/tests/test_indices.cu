#include <gtest/gtest.h>

#include "hg_gpu/edge_signature.hpp"
#include "hg_gpu/signature_index.hpp"
#include "hg_gpu/vertex_inverted_index.hpp"

#include <cuda_runtime.h>

#include <set>
#include <vector>

namespace {

using hg_gpu::EdgeId;
using hg_gpu::VertexId;

// =============================================================================
// SignatureIndex
// =============================================================================

__global__ void k_sig_insert_batch(hg_gpu::SignatureIndex::DeviceView v,
                                    const EdgeId*   eids,
                                    const uint64_t* sigs,
                                    uint32_t        n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    v.insert(eids[tid], sigs[tid]);
}

__global__ void k_sig_collect(hg_gpu::SignatureIndex::DeviceView v,
                              uint64_t  query_sig,
                              EdgeId*   out,
                              uint32_t* out_n,
                              uint32_t  cap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t cnt = 0;
    v.for_each_in_bucket(query_sig, [&](EdgeId eid) {
        if (cnt < cap) out[cnt] = eid;
        ++cnt;
    });
    *out_n = cnt;
}

TEST(SignatureIndex, InsertedEdgesAppearInBucket) {
    constexpr uint32_t kBuckets = 16;
    constexpr uint32_t kEdges   = 8;
    hg_gpu::SignatureIndex idx(kBuckets, kEdges);

    // All 8 edges share the same signature → one bucket holds all of them.
    std::vector<EdgeId>   eids(kEdges);
    std::vector<uint64_t> sigs(kEdges, 0xDEADBEEFCAFEULL);
    for (uint32_t i = 0; i < kEdges; ++i) eids[i] = i + 1;

    EdgeId*   d_eids = nullptr; cudaMalloc(&d_eids, sizeof(EdgeId)   * kEdges);
    uint64_t* d_sigs = nullptr; cudaMalloc(&d_sigs, sizeof(uint64_t) * kEdges);
    cudaMemcpy(d_eids, eids.data(), sizeof(EdgeId)   * kEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigs, sigs.data(), sizeof(uint64_t) * kEdges, cudaMemcpyHostToDevice);

    k_sig_insert_batch<<<1, kEdges>>>(idx.view(), d_eids, d_sigs, kEdges);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EdgeId*   d_out   = nullptr; cudaMalloc(&d_out,   sizeof(EdgeId) * kEdges);
    uint32_t* d_out_n = nullptr; cudaMalloc(&d_out_n, sizeof(uint32_t));
    k_sig_collect<<<1, 1>>>(idx.view(), 0xDEADBEEFCAFEULL, d_out, d_out_n, kEdges);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t n = 0; cudaMemcpy(&n, d_out_n, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(n, kEdges);
    std::vector<EdgeId> got(kEdges);
    cudaMemcpy(got.data(), d_out, sizeof(EdgeId) * kEdges, cudaMemcpyDeviceToHost);
    cudaFree(d_eids); cudaFree(d_sigs); cudaFree(d_out); cudaFree(d_out_n);

    std::set<EdgeId> got_set(got.begin(), got.end());
    EXPECT_EQ(got_set.size(), kEdges);
    for (uint32_t i = 0; i < kEdges; ++i) EXPECT_TRUE(got_set.count(i + 1)) << "missing " << (i+1);
}

TEST(SignatureIndex, DifferentSignaturesGoToDifferentBucketsUnlessCollide) {
    constexpr uint32_t kBuckets = 16;
    hg_gpu::SignatureIndex idx(kBuckets, 64);

    // Two edges with distinct signatures whose hashes & 15 differ.
    uint64_t sigA = 0x0000000000000000ULL;  // bucket 0
    uint64_t sigB = 0x0000000000000001ULL;  // bucket 1

    std::vector<EdgeId>   eids = {10, 20};
    std::vector<uint64_t> sigs = {sigA, sigB};

    EdgeId*   d_eids = nullptr; cudaMalloc(&d_eids, sizeof(EdgeId)   * 2);
    uint64_t* d_sigs = nullptr; cudaMalloc(&d_sigs, sizeof(uint64_t) * 2);
    cudaMemcpy(d_eids, eids.data(), sizeof(EdgeId)   * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigs, sigs.data(), sizeof(uint64_t) * 2, cudaMemcpyHostToDevice);
    k_sig_insert_batch<<<1, 2>>>(idx.view(), d_eids, d_sigs, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EdgeId*   d_out   = nullptr; cudaMalloc(&d_out,   sizeof(EdgeId) * 4);
    uint32_t* d_out_n = nullptr; cudaMalloc(&d_out_n, sizeof(uint32_t));

    k_sig_collect<<<1, 1>>>(idx.view(), sigA, d_out, d_out_n, 4);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t n = 0; cudaMemcpy(&n, d_out_n, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(n, 1u);
    EdgeId e = 0; cudaMemcpy(&e, d_out, sizeof(EdgeId), cudaMemcpyDeviceToHost);
    EXPECT_EQ(e, 10u);

    k_sig_collect<<<1, 1>>>(idx.view(), sigB, d_out, d_out_n, 4);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(&n, d_out_n, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&e, d_out,   sizeof(EdgeId),   cudaMemcpyDeviceToHost);
    EXPECT_EQ(n, 1u);
    EXPECT_EQ(e, 20u);

    cudaFree(d_eids); cudaFree(d_sigs); cudaFree(d_out); cudaFree(d_out_n);
}

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
