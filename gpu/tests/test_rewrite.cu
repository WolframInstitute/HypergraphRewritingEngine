#include <gtest/gtest.h>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/rewrite.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <set>
#include <vector>

namespace {

using hg_gpu::EdgeId;
using hg_gpu::StateId;
using hg_gpu::VertexId;

hg_gpu::EngineConfig small_cfg() {
    hg_gpu::EngineConfig cfg;
    cfg.max_edges              = 128;
    cfg.max_state_edge_total = 256;
    cfg.max_states             = 32;
    cfg.max_vertex_slots       = 512;
    cfg.max_vertices           = 128;
    cfg.sig_index_buckets      = 32;
    cfg.sig_index_pool         = 256;
    cfg.inverted_pool          = 512;
    return cfg;
}

TEST(Rewrite, SimpleChainRuleOneStep) {
    // Rule: {{x,y}} -> {{x,y},{y,z}}  (adds a new edge consuming x,y's
    // shared vertex y; z is fresh).
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}};
    r.rhs = {{0, 1}, {1, 2}};
    r.num_lhs_vars = 2;
    r.num_rhs_vars = 3;

    hg_gpu::EngineState engine(small_cfg());
    hg_gpu::upload_initial_state(engine, {{0u, 1u}});

    auto dr = hg_gpu::make_device_rule(r);
    std::vector<hg_gpu::DeviceRule> rules = {dr};

    hg_gpu::Pool<hg_gpu::MatchRecord> matches(64);
    uint32_t n_matches = hg_gpu::run_match_kernel(engine, rules, /*state=*/0, matches);
    ASSERT_EQ(n_matches, 1u);

    uint32_t n_new = hg_gpu::run_rewrite_kernel(engine, rules, matches, n_matches, /*step=*/1);
    EXPECT_EQ(n_new, 1u);

    // After rewrite we should have:
    //   - 2 states (initial + new)
    //   - 3 edges total: original {0,1}=e0, plus the two RHS edges ({0,1}=e1,
    //     {1,2}=e2). Note: pattern var x=0, y=1, z=new.
    //     e1 = {x,y} = {0,1}  (same values as original but a NEW edge)
    //     e2 = {y,z} = {1, 2} (z is fresh VertexId 2)
    //   - State 1 bitset = {e1, e2} (e0 consumed from original)
    EXPECT_EQ(engine.num_states_host(), 2u);
    EXPECT_EQ(engine.num_edges_host(), 3u);
    EXPECT_EQ(engine.vertex_high_water_host(), 3u);  // max{0,1} + 1 new = 3

    auto state1_edges = engine.state_edges_host(1);
    EXPECT_EQ(state1_edges.size(), 2u);
    std::set<EdgeId> s1(state1_edges.begin(), state1_edges.end());
    EXPECT_EQ(s1, (std::set<EdgeId>{1u, 2u}));

    auto e1_verts = engine.edge_vertices_host(1);
    EXPECT_EQ(e1_verts, (std::vector<VertexId>{0u, 1u}));
    auto e2_verts = engine.edge_vertices_host(2);
    EXPECT_EQ(e2_verts, (std::vector<VertexId>{1u, 2u}));
}

TEST(Rewrite, WolframCanonicalRuleOneStep) {
    // {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}}
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {0, 2}};
    r.rhs = {{0, 1}, {0, 3}, {1, 3}, {2, 3}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    hg_gpu::EngineState engine(small_cfg());
    hg_gpu::upload_initial_state(engine, {{0u, 1u}, {0u, 2u}});

    auto dr = hg_gpu::make_device_rule(r);
    std::vector<hg_gpu::DeviceRule> rules = {dr};

    hg_gpu::Pool<hg_gpu::MatchRecord> matches(64);
    uint32_t n_matches = hg_gpu::run_match_kernel(engine, rules, 0, matches);
    // Pattern {{x,y},{x,z}}: x shared between both pattern edges. The
    // initial state {{0,1},{0,2}} has vertex 0 in both edges. So we can
    // bind x=0, y=1, z=2 (using edges 0 and 1) OR x=0, y=2, z=1 (swapping
    // which edge maps to which pattern edge). Both are valid under Wolfram
    // semantics (y != z is not required; they're independent vars).
    //
    // Pattern edge 0 = {x,y}: can bind to edge 0 (y=1) or edge 1 (y=2).
    // Pattern edge 1 = {x,z}: must share x. If pe0→e0 (x=0,y=1), pe1
    //   must have x=0 and not already consumed: e1 fits, z=2. Match.
    //   If pe0→e1 (x=0,y=2), pe1 must have x=0: e0 fits, z=1. Match.
    // So 2 matches total.
    ASSERT_EQ(n_matches, 2u);

    uint32_t n_new = hg_gpu::run_rewrite_kernel(engine, rules, matches, n_matches, 1);
    EXPECT_EQ(n_new, 2u);

    // 2 new states. Each has 4 RHS edges. Each uses one fresh vertex
    // (w), bumping vertex_high_water by 2.
    EXPECT_EQ(engine.num_states_host(), 3u);
    EXPECT_EQ(engine.vertex_high_water_host(), 5u);  // 3 (initial) + 2 fresh

    for (StateId sid : {1u, 2u}) {
        auto edges = engine.state_edges_host(sid);
        EXPECT_EQ(edges.size(), 4u) << "state " << sid;
    }
}

TEST(Rewrite, EventsAndCausalBranchialPopulated) {
    // Two-step Wolfram-style rule producing events with shared edges → both
    // causal and branchial edges should appear.
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}};
    r.rhs = {{0, 1}, {1, 2}};
    r.num_lhs_vars = 2;
    r.num_rhs_vars = 3;

    hg_gpu::EngineConfig cfg = small_cfg();
    hg_gpu::EvolveInput in;
    in.rules = {r};
    in.initial_state = {{0u, 1u}};
    in.num_steps = 3;

    auto result = hg_gpu::evolve(in);

    EXPECT_GT(result.events.size(), 0u);
    for (const auto& e : result.events) {
        EXPECT_EQ(e.consumed_edges.size(), 1u);
        EXPECT_EQ(e.produced_edges.size(), 2u);
    }

    // The branching structure should cause both causal (events chain
    // through produced/consumed edges) and branchial (same-state sibling
    // events) relationships to appear.
    EXPECT_GT(result.causal_edges.size(), 0u)
        << "expected at least one causal edge over " << result.events.size() << " events";
}

TEST(Rewrite, TriangleRuleIntroducesNewEdge) {
    // Rule: {{x,y},{y,z}} -> {{x,z}}  (contract two-path into a shortcut)
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 3;

    hg_gpu::EngineState engine(small_cfg());
    hg_gpu::upload_initial_state(engine, {{0u,1u}, {1u,2u}, {2u,3u}});

    auto dr = hg_gpu::make_device_rule(r);
    std::vector<hg_gpu::DeviceRule> rules = {dr};

    hg_gpu::Pool<hg_gpu::MatchRecord> matches(64);
    uint32_t n = hg_gpu::run_match_kernel(engine, rules, 0, matches);
    // Matches: {e0,e1} (x=0,y=1,z=2) and {e1,e2} (x=1,y=2,z=3). 2 matches.
    ASSERT_EQ(n, 2u);

    uint32_t n_new = hg_gpu::run_rewrite_kernel(engine, rules, matches, n, 1);
    EXPECT_EQ(n_new, 2u);

    // Each new state has:
    //   match 1 consumed e0,e1 → state has {e2} plus new edge (x,z)=(0,2)
    //   match 2 consumed e1,e2 → state has {e0} plus new edge (x,z)=(1,3)
    for (StateId sid : {1u, 2u}) {
        auto edges = engine.state_edges_host(sid);
        EXPECT_EQ(edges.size(), 2u) << "state " << sid;
    }
}

}  // namespace
