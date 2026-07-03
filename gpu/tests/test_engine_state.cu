#include <gtest/gtest.h>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/initial_upload.hpp"

#include <cuda_runtime.h>

#include <set>
#include <vector>

namespace {

using hg_gpu::EdgeId;
using hg_gpu::VertexId;

TEST(EngineState, AllocateAndClear) {
    hg_gpu::EngineConfig cfg;
    cfg.max_edges            = 64;
    cfg.max_state_edge_total = 256;
    cfg.max_states           = 8;
    cfg.max_vertex_slots     = 256;
    cfg.max_vertices         = 64;
    cfg.sig_index_buckets    = 16;
    cfg.sig_index_pool       = 64;
    cfg.inverted_pool        = 256;

    hg_gpu::EngineState engine(cfg);
    EXPECT_EQ(engine.num_edges_host(), 0u);
    EXPECT_EQ(engine.num_states_host(), 0u);
    EXPECT_EQ(engine.vertex_high_water_host(), 0u);
}

TEST(EngineState, UploadInitialStateRoundTrip) {
    hg_gpu::EngineConfig cfg;
    cfg.max_edges              = 64;
    cfg.max_state_edge_total = 256;
    cfg.max_states             = 8;
    cfg.max_vertex_slots       = 256;
    cfg.max_vertices           = 64;
    cfg.sig_index_buckets      = 16;
    cfg.sig_index_pool         = 64;
    cfg.inverted_pool          = 256;

    hg_gpu::EngineState engine(cfg);

    // Initial state: edges {{0,1}, {1,2}, {2,0}} → triangle on {0,1,2}.
    std::vector<std::vector<VertexId>> initial = {{0u, 1u}, {1u, 2u}, {2u, 0u}};

    auto sid = hg_gpu::upload_initial_state(engine, initial);
    EXPECT_EQ(sid, 0u);

    EXPECT_EQ(engine.num_edges_host(), 3u);
    EXPECT_EQ(engine.num_states_host(), 1u);
    EXPECT_EQ(engine.vertex_high_water_host(), 3u);  // max(0,1,2) + 1

    // State 0 must contain edge IDs 0, 1, 2.
    auto edges = engine.state_edges_host(0);
    std::set<EdgeId> got(edges.begin(), edges.end());
    EXPECT_EQ(got, (std::set<EdgeId>{0, 1, 2}));

    // Each Edge's vertex tuple must round-trip exactly.
    for (uint32_t i = 0; i < 3; ++i) {
        auto verts = engine.edge_vertices_host(i);
        EXPECT_EQ(verts, initial[i]);
    }
}

TEST(EngineState, EmptyInitialStateIsLegal) {
    hg_gpu::EngineConfig cfg;
    cfg.max_edges              = 8;
    cfg.max_state_edge_total = 256;
    cfg.max_states             = 2;
    cfg.max_vertex_slots       = 16;
    cfg.max_vertices           = 8;
    cfg.sig_index_buckets      = 4;
    cfg.sig_index_pool         = 8;
    cfg.inverted_pool          = 16;

    hg_gpu::EngineState engine(cfg);
    auto sid = hg_gpu::upload_initial_state(engine, {});
    EXPECT_EQ(sid, 0u);
    EXPECT_EQ(engine.num_states_host(), 1u);
    EXPECT_EQ(engine.num_edges_host(), 0u);
    EXPECT_TRUE(engine.state_edges_host(0).empty());
}

// Indices are populated by the upload kernel. Verify by querying the
// signature index for each edge's signature and the vertex inverted index
// for each vertex — every edge must be reachable through both.

__global__ void k_query_sig(hg_gpu::SignatureIndex::DeviceView v,
                            uint64_t sig, uint32_t* count, EdgeId* out, uint32_t cap) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t c = 0;
    v.for_each_in_bucket(sig, [&](EdgeId eid) {
        if (c < cap) out[c] = eid;
        ++c;
    });
    *count = c;
}

__global__ void k_query_vert(hg_gpu::VertexInvertedIndex::DeviceView v,
                             VertexId q, uint32_t* count, EdgeId* out, uint32_t cap) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t c = 0;
    v.for_each_incident(q, [&](EdgeId eid) {
        if (c < cap) out[c] = eid;
        ++c;
    });
    *count = c;
}

TEST(EngineState, IndicesPopulatedByUpload) {
    hg_gpu::EngineConfig cfg;
    cfg.max_edges              = 32;
    cfg.max_state_edge_total = 256;
    cfg.max_states             = 4;
    cfg.max_vertex_slots       = 64;
    cfg.max_vertices           = 16;
    cfg.sig_index_buckets      = 16;
    cfg.sig_index_pool         = 32;
    cfg.inverted_pool          = 64;

    hg_gpu::EngineState engine(cfg);

    // Mixed-arity edges: two arity-2 and one arity-3.
    std::vector<std::vector<VertexId>> initial = {
        {0u, 1u},        // sig pattern [0,1]
        {2u, 3u},        // sig pattern [0,1]   (same signature as above)
        {4u, 5u, 6u},    // sig pattern [0,1,2] (different signature)
    };
    hg_gpu::upload_initial_state(engine, initial);

    auto ds = engine.device();

    // Query signature for {0,1} type — should return edges 0 and 1.
    uint64_t sig01 = hg_gpu::signature_hash_from_vertices(initial[0].data(), 2);
    uint32_t* d_cnt = nullptr; cudaMalloc(&d_cnt, sizeof(uint32_t));
    EdgeId*   d_out = nullptr; cudaMalloc(&d_out, sizeof(EdgeId) * 8);
    k_query_sig<<<1, 1>>>(ds.signature_index, sig01, d_cnt, d_out, 8);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t cnt = 0; cudaMemcpy(&cnt, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::vector<EdgeId> got(cnt);
    cudaMemcpy(got.data(), d_out, sizeof(EdgeId) * cnt, cudaMemcpyDeviceToHost);
    std::set<EdgeId> got_set(got.begin(), got.end());
    EXPECT_EQ(got_set, (std::set<EdgeId>{0u, 1u}));

    // Query inverted index for vertex 0 — must contain only edge 0.
    k_query_vert<<<1, 1>>>(ds.vertex_inverted_index, 0u, d_cnt, d_out, 8);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(&cnt, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cnt, 1u);
    EdgeId e = 0; cudaMemcpy(&e, d_out, sizeof(EdgeId), cudaMemcpyDeviceToHost);
    EXPECT_EQ(e, 0u);

    // Query inverted index for vertex 5 — only edge 2.
    k_query_vert<<<1, 1>>>(ds.vertex_inverted_index, 5u, d_cnt, d_out, 8);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(&cnt, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cnt, 1u);
    cudaMemcpy(&e, d_out, sizeof(EdgeId), cudaMemcpyDeviceToHost);
    EXPECT_EQ(e, 2u);

    cudaFree(d_cnt); cudaFree(d_out);
}

}  // namespace
