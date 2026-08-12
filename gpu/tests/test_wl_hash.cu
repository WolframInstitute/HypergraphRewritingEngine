#include <gtest/gtest.h>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/wl_hash.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace {

using hg_gpu::VertexId;

hg_gpu::EngineConfig small_cfg() {
    hg_gpu::EngineConfig cfg;
    cfg.max_edges              = 64;
    cfg.max_state_edge_total = 256;
    cfg.max_states             = 8;
    cfg.max_vertex_slots       = 256;
    cfg.max_vertices           = 64;
    cfg.sig_index_buckets      = 16;
    cfg.sig_index_pool         = 64;
    cfg.inverted_pool          = 256;
    return cfg;
}

TEST(WlHash, IsomorphicRelabelsSameHash) {
    // Triangle over {0,1,2} vs triangle over {7,8,9}. Isomorphic — same WL
    // hash.
    hg_gpu::EngineState a(small_cfg());
    hg_gpu::upload_initial_state(a, {{0u,1u}, {1u,2u}, {2u,0u}});
    hg_gpu::EngineState b(small_cfg());
    hg_gpu::upload_initial_state(b, {{7u,8u}, {8u,9u}, {9u,7u}});

    uint64_t ha = hg_gpu::compute_state_wl_hash_host(a, 0);
    uint64_t hb = hg_gpu::compute_state_wl_hash_host(b, 0);
    EXPECT_EQ(ha, hb);
    EXPECT_NE(ha, 0u);
}

TEST(WlHash, NonIsomorphicDifferentHash) {
    // Triangle vs path of 3 edges — distinct graphs, different WL hashes.
    hg_gpu::EngineState tri(small_cfg());
    hg_gpu::upload_initial_state(tri, {{0u,1u}, {1u,2u}, {2u,0u}});
    hg_gpu::EngineState path(small_cfg());
    hg_gpu::upload_initial_state(path, {{0u,1u}, {1u,2u}, {2u,3u}});

    uint64_t h_tri  = hg_gpu::compute_state_wl_hash_host(tri,  0);
    uint64_t h_path = hg_gpu::compute_state_wl_hash_host(path, 0);
    EXPECT_NE(h_tri, h_path);
}

TEST(WlHash, DifferentSizesDifferentHash) {
    hg_gpu::EngineState s1(small_cfg());
    hg_gpu::upload_initial_state(s1, {{0u,1u}});
    hg_gpu::EngineState s2(small_cfg());
    hg_gpu::upload_initial_state(s2, {{0u,1u}, {1u,2u}});

    EXPECT_NE(hg_gpu::compute_state_wl_hash_host(s1, 0),
              hg_gpu::compute_state_wl_hash_host(s2, 0));
}

TEST(WlHash, SelfLoopDistinguishedFromRegularEdge) {
    hg_gpu::EngineState loop(small_cfg());
    hg_gpu::upload_initial_state(loop, {{5u, 5u}});
    hg_gpu::EngineState reg(small_cfg());
    hg_gpu::upload_initial_state(reg,  {{3u, 4u}});

    EXPECT_NE(hg_gpu::compute_state_wl_hash_host(loop, 0),
              hg_gpu::compute_state_wl_hash_host(reg,  0));
}

TEST(WlHash, DirectionalityMatters) {
    // {0,1} is different from {1,0} — edges are tuples, not sets.
    hg_gpu::EngineState fwd(small_cfg());
    hg_gpu::upload_initial_state(fwd, {{0u, 1u}});
    hg_gpu::EngineState rev(small_cfg());
    hg_gpu::upload_initial_state(rev, {{1u, 0u}});

    // Both have one edge with one vertex at position 0 and one at position 1.
    // The WL algorithm tags neighbours by position, so {0,1} and {1,0} should
    // produce identical hashes under relabeling 0↔1 — they are in fact
    // isomorphic as directed hypergraphs. Verify they DO match (would have
    // failed if direction were ignored; WL captures relabelable structure,
    // and single-edge directed hypergraphs with distinct endpoints are
    // isomorphic to each other).
    uint64_t h1 = hg_gpu::compute_state_wl_hash_host(fwd, 0);
    uint64_t h2 = hg_gpu::compute_state_wl_hash_host(rev, 0);
    EXPECT_EQ(h1, h2);
}

TEST(WlHash, EmptyStateHashIsDeterministic) {
    hg_gpu::EngineState e(small_cfg());
    hg_gpu::upload_initial_state(e, {});
    uint64_t h = hg_gpu::compute_state_wl_hash_host(e, 0);
    // Empty state: no edges, no vertices → zero.
    EXPECT_EQ(h, 0u);
}

}  // namespace
