#include <gtest/gtest.h>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/match.hpp"

#include <cuda_runtime.h>

#include <set>
#include <vector>

namespace {

using hg_gpu::EdgeId;
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

hg_gpu::DeviceRule rule_one_edge_two_vars() {
    // LHS = {{a, b}} → 1 edge, 2 distinct vars.
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}};
    r.rhs = {};
    r.num_lhs_vars = 2;
    return hg_gpu::make_device_rule(r);
}

hg_gpu::DeviceRule rule_two_edges_three_vars() {
    // LHS = {{a, b}, {b, c}} → 2 edges sharing var 1.
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {};
    r.num_lhs_vars = 3;
    return hg_gpu::make_device_rule(r);
}

hg_gpu::DeviceRule rule_self_loop() {
    // LHS = {{a, a}} — pattern that matches edges where both vertices are
    // the same (self-loop).
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 0}};
    r.rhs = {};
    r.num_lhs_vars = 1;
    return hg_gpu::make_device_rule(r);
}

TEST(Match, OneEdgePatternOnTriangleYieldsThreeMatches) {
    hg_gpu::EngineState engine(small_cfg());
    // Triangle: edges {0,1}, {1,2}, {2,0}.
    hg_gpu::upload_initial_state(engine, {{0u,1u}, {1u,2u}, {2u,0u}});

    std::vector<hg_gpu::DeviceRule> rules = {rule_one_edge_two_vars()};

    hg_gpu::Pool<hg_gpu::MatchRecord> out(64);
    uint32_t n = hg_gpu::run_match_kernel(engine, rules, /*state=*/0, out);

    EXPECT_EQ(n, 3u);

    std::vector<hg_gpu::MatchRecord> got(n);
    cudaMemcpy(got.data(), out.view().data, sizeof(hg_gpu::MatchRecord) * n,
               cudaMemcpyDeviceToHost);
    std::set<EdgeId> matched;
    for (const auto& m : got) {
        EXPECT_EQ(m.rule_id, 0u);
        EXPECT_EQ(m.state_id, 0u);
        EXPECT_EQ(m.num_edges, 1u);
        matched.insert(m.matched_edges[0]);
    }
    EXPECT_EQ(matched, (std::set<EdgeId>{0u, 1u, 2u}));
}

TEST(Match, TwoEdgePatternOnPathYieldsConsistentBindings) {
    hg_gpu::EngineState engine(small_cfg());
    // Path of 3 edges: {0,1}, {1,2}, {2,3}.
    hg_gpu::upload_initial_state(engine, {{0u,1u}, {1u,2u}, {2u,3u}});

    std::vector<hg_gpu::DeviceRule> rules = {rule_two_edges_three_vars()};

    hg_gpu::Pool<hg_gpu::MatchRecord> out(64);
    uint32_t n = hg_gpu::run_match_kernel(engine, rules, 0, out);

    // Pattern {{a,b},{b,c}}. Expected matches:
    //   (a=0, b=1, c=2) → edges 0,1
    //   (a=1, b=2, c=3) → edges 1,2
    // Plus reverse iteration ordering may produce additional matches when
    // pattern edge 0 can also bind to edge 1 or edge 2 with different vars.
    // Let's check: pattern edge 0 = {a,b}. Try all 3 data edges.
    //   data edge 0 ({0,1}): a=0,b=1. Then pe1={b,c}={1,c}. Candidates with
    //     position 0 = 1 and {1,?} signature: data edge 1 ({1,2}). c=2. Match!
    //   data edge 1 ({1,2}): a=1,b=2. pe1={b,c}={2,c}. Need {2,?}: data edge
    //     2 ({2,3}). c=3. Match!
    //   data edge 2 ({2,3}): a=2,b=3. pe1={b,c}={3,c}. Need {3,?}: none.
    //
    // So expect exactly 2 matches. Each uses exactly 2 distinct edges.
    EXPECT_EQ(n, 2u);

    std::vector<hg_gpu::MatchRecord> got(n);
    cudaMemcpy(got.data(), out.view().data, sizeof(hg_gpu::MatchRecord) * n,
               cudaMemcpyDeviceToHost);

    std::set<std::pair<EdgeId, EdgeId>> match_pairs;
    for (const auto& m : got) {
        EXPECT_EQ(m.num_edges, 2u);
        EXPECT_NE(m.matched_edges[0], m.matched_edges[1]);  // one-to-one consumption
        match_pairs.insert({m.matched_edges[0], m.matched_edges[1]});
    }
    EXPECT_EQ(match_pairs, (std::set<std::pair<EdgeId,EdgeId>>{{0,1},{1,2}}));
}

TEST(Match, SelfLoopPatternMatchesOnlySelfLoops) {
    hg_gpu::EngineState engine(small_cfg());
    // Edges: one self-loop {0,0}, two non-self-loops {0,1}, {1,2}.
    hg_gpu::upload_initial_state(engine, {{0u,0u}, {0u,1u}, {1u,2u}});

    std::vector<hg_gpu::DeviceRule> rules = {rule_self_loop()};
    hg_gpu::Pool<hg_gpu::MatchRecord> out(64);
    uint32_t n = hg_gpu::run_match_kernel(engine, rules, 0, out);

    // Only edge 0 matches the self-loop pattern.
    EXPECT_EQ(n, 1u);
    hg_gpu::MatchRecord m{};
    cudaMemcpy(&m, out.view().data, sizeof(hg_gpu::MatchRecord), cudaMemcpyDeviceToHost);
    EXPECT_EQ(m.matched_edges[0], 0u);
}

TEST(Match, DistinctVarsCanBindToSameVertex) {
    // Wolfram-style: {{a, b}} should match a self-loop {0, 0} because a and b
    // are distinct variables that *may* (but need not) bind to the same vertex.
    hg_gpu::EngineState engine(small_cfg());
    hg_gpu::upload_initial_state(engine, {
        {0u, 0u},   // self-loop — a=0, b=0 should match
        {1u, 2u},   // non-self-loop — a=1, b=2 should also match
    });

    std::vector<hg_gpu::DeviceRule> rules = {rule_one_edge_two_vars()};
    hg_gpu::Pool<hg_gpu::MatchRecord> out(64);
    uint32_t n = hg_gpu::run_match_kernel(engine, rules, 0, out);

    EXPECT_EQ(n, 2u);
    std::vector<hg_gpu::MatchRecord> got(n);
    cudaMemcpy(got.data(), out.view().data, sizeof(hg_gpu::MatchRecord) * n,
               cudaMemcpyDeviceToHost);
    std::set<EdgeId> matched;
    for (const auto& m : got) matched.insert(m.matched_edges[0]);
    EXPECT_EQ(matched, (std::set<EdgeId>{0u, 1u}));
}

TEST(Match, NonDistinctBindingAllowedWhenPatternUsesSameVar) {
    // Pattern {{a, b, a}} matches edges where positions 0 and 2 hold the
    // same vertex. Wolfram-style: variables CAN bind same vertex when
    // pattern allows it.
    hg_gpu::EngineState engine(small_cfg());
    hg_gpu::upload_initial_state(engine, {
        {7u, 8u, 7u},   // matches: a=7, b=8
        {1u, 2u, 3u},   // does NOT match: positions 0,2 differ
        {4u, 5u, 4u},   // matches: a=4, b=5
    });

    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1, 0}};   // pattern positions: a, b, a
    r.rhs = {};
    r.num_lhs_vars = 2;
    auto dr = hg_gpu::make_device_rule(r);

    hg_gpu::Pool<hg_gpu::MatchRecord> out(64);
    uint32_t n = hg_gpu::run_match_kernel(engine, {dr}, 0, out);

    EXPECT_EQ(n, 2u);
    std::vector<hg_gpu::MatchRecord> got(n);
    cudaMemcpy(got.data(), out.view().data, sizeof(hg_gpu::MatchRecord) * n,
               cudaMemcpyDeviceToHost);
    std::set<EdgeId> matched;
    for (const auto& m : got) matched.insert(m.matched_edges[0]);
    EXPECT_EQ(matched, (std::set<EdgeId>{0u, 2u}));
}

}  // namespace
