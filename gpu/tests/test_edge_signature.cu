#include <gtest/gtest.h>

#include "hg_gpu/edge_signature.hpp"

namespace {

using hg_gpu::EdgeSignature;
using hg_gpu::VertexId;
using hg_gpu::signature_from_vertices;
using hg_gpu::signature_hash;

TEST(EdgeSignature, AllDistinct) {
    VertexId v[] = {5, 6, 8};
    auto sig = signature_from_vertices(v, 3);
    EXPECT_EQ(sig.arity, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 1);
    EXPECT_EQ(sig.pattern[2], 2);
}

TEST(EdgeSignature, AdjacentRepeat) {
    VertexId v[] = {3, 3, 4};
    auto sig = signature_from_vertices(v, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 0);
    EXPECT_EQ(sig.pattern[2], 1);
}

TEST(EdgeSignature, AllSame) {
    VertexId v[] = {1, 1, 1};
    auto sig = signature_from_vertices(v, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 0);
    EXPECT_EQ(sig.pattern[2], 0);
}

TEST(EdgeSignature, AbARepeat) {
    VertexId v[] = {7, 8, 7};
    auto sig = signature_from_vertices(v, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 1);
    EXPECT_EQ(sig.pattern[2], 0);
}

TEST(EdgeSignature, EqualPatternEqualHash) {
    VertexId a[] = {3, 3, 4};
    VertexId b[] = {7, 7, 8};
    auto sa = signature_from_vertices(a, 3);
    auto sb = signature_from_vertices(b, 3);
    EXPECT_TRUE(sa == sb);
    EXPECT_EQ(signature_hash(sa), signature_hash(sb));
}

TEST(EdgeSignature, DifferentArityDifferentSignature) {
    VertexId a[] = {1, 2};
    VertexId b[] = {1, 2, 3};
    EXPECT_NE(signature_hash(signature_from_vertices(a, 2)),
              signature_hash(signature_from_vertices(b, 3)));
}

TEST(EdgeSignature, DifferentRepetitionDifferentHash) {
    VertexId a[] = {1, 2, 3};  // [0,1,2]
    VertexId b[] = {1, 2, 1};  // [0,1,0]
    EXPECT_NE(signature_hash(signature_from_vertices(a, 3)),
              signature_hash(signature_from_vertices(b, 3)));
}

TEST(EdgeSignature, EmptyEdgeIsLegal) {
    auto sig = signature_from_vertices(nullptr, 0);
    EXPECT_EQ(sig.arity, 0);
}

// Cross-check device-side computation against host: launch a kernel that
// invokes signature_hash_from_vertices on a batch of edges and compare.
__global__ void k_signature_batch(const VertexId* verts, const uint8_t* arities,
                                  const uint32_t* offsets, uint64_t* out, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = hg_gpu::signature_hash_from_vertices(verts + offsets[tid], arities[tid]);
}

TEST(EdgeSignature, DeviceMatchesHost) {
    // 5 edges of varying arity / repetition.
    std::vector<VertexId> verts;
    std::vector<uint8_t>  arities;
    std::vector<uint32_t> offsets;
    auto add_edge = [&](std::initializer_list<VertexId> e) {
        offsets.push_back(verts.size());
        for (auto v : e) verts.push_back(v);
        arities.push_back(static_cast<uint8_t>(e.size()));
    };
    add_edge({1, 2});
    add_edge({3, 3, 4});
    add_edge({5, 6, 7});
    add_edge({7, 8, 7});
    add_edge({9, 9, 9, 9});

    uint32_t n = arities.size();
    std::vector<uint64_t> host_hashes(n);
    for (uint32_t i = 0; i < n; ++i) {
        host_hashes[i] = hg_gpu::signature_hash_from_vertices(
            verts.data() + offsets[i], arities[i]);
    }

    VertexId* d_verts   = nullptr; cudaMalloc(&d_verts,   sizeof(VertexId) * verts.size());
    uint8_t*  d_arities = nullptr; cudaMalloc(&d_arities, sizeof(uint8_t)  * arities.size());
    uint32_t* d_offsets = nullptr; cudaMalloc(&d_offsets, sizeof(uint32_t) * offsets.size());
    uint64_t* d_out     = nullptr; cudaMalloc(&d_out,     sizeof(uint64_t) * n);
    cudaMemcpy(d_verts,   verts.data(),   sizeof(VertexId) * verts.size(),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_arities, arities.data(), sizeof(uint8_t)  * arities.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), sizeof(uint32_t) * offsets.size(), cudaMemcpyHostToDevice);

    k_signature_batch<<<1, 32>>>(d_verts, d_arities, d_offsets, d_out, n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint64_t> device_hashes(n);
    cudaMemcpy(device_hashes.data(), d_out, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_verts); cudaFree(d_arities); cudaFree(d_offsets); cudaFree(d_out);

    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(host_hashes[i], device_hashes[i]) << "edge " << i;
    }
}

}  // namespace
