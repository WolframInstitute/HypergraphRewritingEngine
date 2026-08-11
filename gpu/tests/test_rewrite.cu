#include <gtest/gtest.h>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/rewrite.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace {

using hg_gpu::EdgeId;
using hg_gpu::StateId;
using hg_gpu::VertexId;

// Every state's exact canonical hash, as a multiset. This is the observable a scheduler must
// preserve: it is invariant under the vertex labelling and under the order states were created
// in, and both are things a scheduler is free to change.
std::multiset<uint64_t> canonical_hash_multiset(hg_gpu::EngineState& eng) {
    const uint32_t n = eng.num_states_host();
    std::multiset<uint64_t> out;
    if (n == 0) return out;
    uint64_t* d = nullptr;
    cudaMalloc(&d, sizeof(uint64_t) * n);
    hg_gpu::compute_state_ir_hashes_range(eng, 0, n, d);
    std::vector<uint64_t> h(n);
    cudaMemcpy(h.data(), d, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
    cudaFree(d);
    out.insert(h.begin(), h.end());
    return out;
}

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

TEST(Rewrite, SparseVariableNumberingBindsTheRightNewVariable) {
    // Rule: {{x,z}} -> {{x,z},{z,y}} with variable indices 0 and 2 on the LHS and 1 new.
    //
    // The new variables are rhs_vars & ~lhs_vars = {1}. They are NOT the index range
    // [num_lhs_vars, num_rhs_vars): num_lhs_vars is a count on the host and a max-index-plus-
    // one in the paclet's GPU backend, and neither reading names variable 1 here. Taking the
    // range would assign the fresh vertex to variable 2 -- which the match already bound --
    // and leave variable 1 unbound, so the produced edge would carry the fresh vertex in the
    // wrong position and an INVALID_ID in the other.
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 2}};
    r.rhs = {{0, 2}, {2, 1}};
    r.num_lhs_vars = 2;   // a COUNT of {0,2}, deliberately not a dense prefix
    r.num_rhs_vars = 3;

    hg_gpu::EngineState engine(small_cfg());
    hg_gpu::upload_initial_state(engine, {{0u, 1u}});

    auto dr = hg_gpu::make_device_rule(r);
    std::vector<hg_gpu::DeviceRule> rules = {dr};

    hg_gpu::Pool<hg_gpu::MatchRecord> matches(64);
    uint32_t n_matches = hg_gpu::run_match_kernel(engine, rules, /*state=*/0, matches);
    ASSERT_EQ(n_matches, 1u);
    uint32_t n_new = hg_gpu::run_rewrite_kernel(engine, rules, matches, n_matches, /*step=*/1);
    ASSERT_EQ(n_new, 1u);

    // x=0, z=1 from the match; y is the one fresh vertex, so 2.
    EXPECT_EQ(engine.vertex_high_water_host(), 3u);
    auto e1_verts = engine.edge_vertices_host(1);
    EXPECT_EQ(e1_verts, (std::vector<VertexId>{0u, 1u}));   // {x,z}
    auto e2_verts = engine.edge_vertices_host(2);
    EXPECT_EQ(e2_verts, (std::vector<VertexId>{1u, 2u}));   // {z,y}, y fresh
}

// The step budget is a PREDICATE on the item, not a loop bound. At max_steps == 1 only the
// roots are expanded, so the run must stop after one round of rewriting even though the
// children it produced are matchable.
TEST(Rewrite, PersistentEvolveStepBudgetStopsAtOne) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}};
    r.rhs = {{0, 1}, {1, 2}};
    r.num_lhs_vars = 2;
    r.num_rhs_vars = 3;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}};

    hg_gpu::EvolveInput in;
    in.rules = {r};
    in.initial_state = init;
    in.num_steps = 1;
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;

    hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
    hg_gpu::EngineState persistent(cfg);
    hg_gpu::upload_initial_state(persistent, init);

    std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
    hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
    matches.reset();
    hg_gpu::DeviceArena arena(8ull << 20);

    const auto stats = hg_gpu::run_persistent_evolve(
        persistent, rules, /*roots=*/{0u}, /*max_steps=*/1u, matches, arena, /*dedup=*/true,
        /*explore_threshold_u32=*/0xFFFFFFFFu, /*explore_seed=*/0,
        hg_gpu::CanonicalizationMode::Full, hgcommon::EVENT_SIG_NONE, /*blocks=*/5);

    // Two matches in the root (one per edge), so two children and no further expansion.
    EXPECT_EQ(stats.matches_found, 2u);
    EXPECT_EQ(stats.states_after, 3u);
}

// A whole multi-step evolution inside ONE launch must produce exactly what the level-
// synchronous Engine produces for the same input.
//
// Stage 3: the rewrite's output is hashed, deduplicated and re-enqueued on device, so there is
// with no host in the middle. That makes three things load-bearing that the kernel entry points
// and 2 never exercised, and each has its own failure signature:
//
//   THE STEP BUDGET. Depth rides on the item, so `max_steps` is a predicate. Getting it off by
//   one shows up as a state count above or below the reference.
//   THE EXACT HASH. Dedup uses the arena-backed IR hash, computed with no batch to size the
//   scratch from. A wrong key merges non-isomorphic states, which shows up as too FEW states.
//   TERMINATION. There is no phase boundary to fall back on, so a detector that fires early
//   truncates the run and one that never fires hangs. Both are caught here, the second only by
//   the test not returning.
TEST(Rewrite, PersistentEvolveMatchesTheLevelSynchronousEngine) {
    // {{x,y},{y,z}} -> {{x,y},{y,w},{w,z}}: branches, and its children are not all distinct,
    // so dedup does work rather than passing everything through.
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
    const uint32_t kSteps = 3;

    hg_gpu::EvolveInput in;
    in.rules = {r};
    in.initial_state = init;
    in.num_steps = kSteps;
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
    in.explore_from_canonical_states_only = true;

    hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
    hg_gpu::Engine reference(cfg);
    const auto ref = reference.run(in);
    ASSERT_TRUE(ref.warnings.empty()) << "reference run overflowed, so the comparison is unsound";
    ASSERT_GT(ref.states.size(), 1u) << "workload never branched, so the comparison is vacuous";

    hg_gpu::EngineState persistent(cfg);
    hg_gpu::upload_initial_state(persistent, init);

    std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
    hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
    matches.reset();
    hg_gpu::DeviceArena arena(64ull << 20);   // words, so 256 MB of scratch

    const auto stats = hg_gpu::run_persistent_evolve(
        persistent, rules, /*roots=*/{0u}, kSteps, matches, arena, /*dedup=*/true,
        /*explore_threshold_u32=*/0xFFFFFFFFu, /*explore_seed=*/0,
        hg_gpu::CanonicalizationMode::Full, hgcommon::EVENT_SIG_NONE, /*blocks=*/9);

    EXPECT_EQ(stats.states_after, ref.states.size());
    EXPECT_EQ(persistent.num_events_host(), ref.events.size());
    EXPECT_GT(stats.arena_words_used, 0u) << "no scratch was claimed, so no state was hashed";

    // The counts alone would pass on a run that explored a different state set of the same
    // size, so compare the canonical hashes themselves.
    std::multiset<uint64_t> ref_hashes;
    for (const auto& s : ref.states) ref_hashes.insert(s.canonical_hash);
    EXPECT_EQ(canonical_hash_multiset(persistent), ref_hashes);

    // The hash the device KEPT for each state must be the hash that state actually has.
    //
    // This is the prerequisite for a device-side event identity: an event's signature needs its
    // INPUT state's hash, computed when that state was created and read back much later, from
    // another block, for every transition out of it. If what was stored is not what the state
    // hashes to, every event identity built on it is wrong in a way no state-level check can
    // see -- the state set would still be right.
    const uint32_t n = persistent.num_states_host();
    std::vector<uint64_t> stored(n);
    cudaMemcpy(stored.data(), persistent.device().state_canonical_hash,
               sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);

    uint64_t* d_fresh = nullptr;
    cudaMalloc(&d_fresh, sizeof(uint64_t) * n);
    hg_gpu::compute_state_ir_hashes_range(persistent, 0, n, d_fresh);
    std::vector<uint64_t> fresh(n);
    cudaMemcpy(fresh.data(), d_fresh, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_fresh);

    size_t published = 0;
    for (uint32_t s = 0; s < n; ++s) {
        if (stored[s] == 0) continue;   // 0 means never computed
        ++published;
        EXPECT_EQ(stored[s], fresh[s])
            << "state " << s << ": the hash the device stored is not the hash it recomputes";
    }
    EXPECT_EQ(published, n)
        << "only " << published << " of " << n << " states carry a stored hash; a transition "
        << "out of an unhashed state would have no input hash to build an event identity from";
}

// The device computes an event identity, and it is the one hgcommon defines.
//
// A per-stage-launch scheduler cannot do this at all: it writes the event in the rewrite
// kernel, before the output state has been canonicalized, and by the time the hash exists the
// kernel that knew which event it belonged to has returned. The persistent scheduler has both
// at one point -- the input hash published when the parent was created, the output hash just
// computed for dedup -- which is what makes the identity reachable rather than merely desirable.
//
// Checked against hgcommon::event_signature recomputed on the host from the same inputs, so
// this compares the device against the SHARED RULE rather than against itself. A device that
// invented its own mixing would satisfy any self-consistency check and fail this one.
TEST(Rewrite, PersistentEvolveStampsTheSharedEventIdentity) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
    const uint32_t kSteps = 3;

    hg_gpu::EvolveInput in;
    in.rules = {r};
    in.initial_state = init;
    in.num_steps = kSteps;
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;

    hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
    hg_gpu::EngineState engine(cfg);
    hg_gpu::upload_initial_state(engine, init);

    std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
    hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
    matches.reset();
    hg_gpu::DeviceArena arena(64ull << 20);

    hg_gpu::run_persistent_evolve(engine, rules, /*roots=*/{0u}, kSteps, matches, arena,
                                  /*dedup=*/true, 0xFFFFFFFFu, 0,
                                  hg_gpu::CanonicalizationMode::Full,
                                  hgcommon::EVENT_SIG_FULL, /*blocks=*/9);

    const uint32_t ne = engine.num_events_host();
    ASSERT_GT(ne, 1u) << "no events, so the comparison is vacuous";

    std::vector<hg_gpu::DeviceEvent> events(ne);
    cudaMemcpy(events.data(), engine.device().event_pool.data,
               sizeof(hg_gpu::DeviceEvent) * ne, cudaMemcpyDeviceToHost);

    const uint32_t ns = engine.num_states_host();
    std::vector<uint64_t> hashes(ns);
    cudaMemcpy(hashes.data(), engine.device().state_canonical_hash,
               sizeof(uint64_t) * ns, cudaMemcpyDeviceToHost);

    size_t stamped = 0;
    for (uint32_t e = 0; e < ne; ++e) {
        const auto& ev = events[e];
        if (ev.id == hg_gpu::INVALID_ID) continue;
        ASSERT_LT(ev.input_state, ns);
        ASSERT_LT(ev.output_state, ns);
        const uint64_t expect = hgcommon::event_signature(
            hgcommon::EVENT_SIG_FULL, hashes[ev.input_state], hashes[ev.output_state],
            ev.step, ev.rule, nullptr, 0, nullptr, 0);
        EXPECT_EQ(ev.signature, expect)
            << "event " << e << ": the device's identity is not the shared rule's";
        ++stamped;
    }
    EXPECT_EQ(stamped, ne) << "some events carry no identity at all";

    // Isomorphic transitions must SHARE an identity -- that is what the identity is for. With a
    // branching rule at this depth the run necessarily reaches the same canonical state by more
    // than one route, so distinct events must collapse to fewer distinct signatures.
    std::set<uint64_t> distinct;
    for (const auto& ev : events) if (ev.id != hg_gpu::INVALID_ID) distinct.insert(ev.signature);
    EXPECT_LT(distinct.size(), static_cast<size_t>(ne))
        << "every event got its own identity, so the signature is distinguishing something it "
        << "should not -- an identity that never collapses is a serial number";
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

// Automatic event identity must be refused rather than answered coarsely.
// Automatic event identity keys on the canonical RANKS of the consumed and produced edges, on
// top of the two endpoint hashes, the step and the rule. Two properties make it worth having,
// and both are checked here:
//
//   - it SEPARATES applications that Full merges. Two rewrites out of one state into isomorphic
//     children are one event under Full and two under Automatic when they consumed different
//     edges, so Automatic's distinct-signature count is >= Full's on the same run.
//   - every rank resolved. A rank that was unavailable falls back to the raw edge id, which is
//     run-local, so a nonzero fallback count means some signature is not an isomorphism
//     invariant -- that is what the counter exists to make visible.
TEST(Rewrite, PersistentEvolveSeparatesApplicationsUnderAutomaticEventIdentity) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}};
    const uint32_t kSteps = 3;

    auto run = [&](hgcommon::EventSignatureKeys keys, uint32_t& fallbacks) {
        hg_gpu::EvolveInput in;
        in.rules = {r};
        in.initial_state = init;
        in.num_steps = kSteps;
        in.canonicalization = hg_gpu::CanonicalizationMode::Full;

        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
        hg_gpu::EngineState engine(cfg);
        hg_gpu::upload_initial_state(engine, init);
        if (keys & (hgcommon::EventKey_ConsumedEdges | hgcommon::EventKey_ProducedEdges))
            engine.ensure_edge_ranks();

        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(64ull << 20);

        hg_gpu::run_persistent_evolve(engine, rules, /*roots=*/{0u}, kSteps, matches, arena,
                                      /*dedup=*/true, 0xFFFFFFFFu, 0,
                                      hg_gpu::CanonicalizationMode::Full, keys, /*blocks=*/9);

        const uint32_t ne = engine.num_events_host();
        std::vector<hg_gpu::DeviceEvent> events(ne);
        cudaMemcpy(events.data(), engine.device().event_pool.data,
                   sizeof(hg_gpu::DeviceEvent) * ne, cudaMemcpyDeviceToHost);

        fallbacks = engine.event_sig_raw_fallbacks();
        std::set<uint64_t> sigs;
        for (const auto& ev : events)
            if (ev.id != hg_gpu::INVALID_ID && ev.signature != 0) sigs.insert(ev.signature);
        return sigs.size();
    };

    uint32_t fb_full = 0, fb_auto = 0;
    const size_t n_full = run(hgcommon::EVENT_SIG_FULL, fb_full);
    const size_t n_auto = run(hgcommon::EVENT_SIG_AUTOMATIC, fb_auto);

    ASSERT_GT(n_full, 0u) << "no signatures stamped, so the comparison is vacuous";
    EXPECT_EQ(fb_auto, 0u)
        << fb_auto << " edges had no canonical rank and were stamped with their raw edge id, "
        << "so those signatures are not isomorphism invariants";
    EXPECT_GE(n_auto, n_full)
        << "Automatic distinguished FEWER applications than Full (" << n_auto << " vs "
        << n_full << "), but its key set is a strict superset of Full's";
}

// Ranks are a property of a state's canonical labeling, so on a state whose canonical labeling
// is UNIQUE they cannot depend on how many blocks raced to produce it. The initial state here
// is a directed path, whose automorphism group is trivial; see the companion test below for
// what changes when it is not.
//
// Checked per key COMPONENT rather than only on the whole signature, so a failure names the
// component that carries the schedule dependence.
TEST(Rewrite, AutomaticEventIdentityIsTheSameAtEveryBlockCountOnRigidStates) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
    const uint32_t kSteps = 3;

    auto signatures = [&](uint32_t blocks, hgcommon::EventSignatureKeys keys) {
        hg_gpu::EvolveInput in;
        in.rules = {r};
        in.initial_state = init;
        in.num_steps = kSteps;
        in.canonicalization = hg_gpu::CanonicalizationMode::Full;

        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
        hg_gpu::EngineState engine(cfg);
        hg_gpu::upload_initial_state(engine, init);
        engine.ensure_edge_ranks();

        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(64ull << 20);

        hg_gpu::run_persistent_evolve(engine, rules, /*roots=*/{0u}, kSteps, matches, arena,
                                      /*dedup=*/true, 0xFFFFFFFFu, 0,
                                      hg_gpu::CanonicalizationMode::Full, keys, blocks);

        const uint32_t ne = engine.num_events_host();
        std::vector<hg_gpu::DeviceEvent> events(ne);
        cudaMemcpy(events.data(), engine.device().event_pool.data,
                   sizeof(hg_gpu::DeviceEvent) * ne, cudaMemcpyDeviceToHost);
        std::multiset<uint64_t> sigs;
        for (const auto& ev : events)
            if (ev.id != hg_gpu::INVALID_ID && ev.signature != 0) sigs.insert(ev.signature);
        return sigs;
    };

    // Checked per COMPONENT, so a failure names the key that carries the schedule dependence
    // instead of only reporting that the whole signature moved.
    struct Cell { const char* name; hgcommon::EventSignatureKeys keys; };
    const Cell cells[] = {
        {"endpoints",              hgcommon::EVENT_SIG_FULL},
        {"endpoints+rule",         hgcommon::EVENT_SIG_FULL | hgcommon::EventKey_Rule},
        {"endpoints+consumed",     hgcommon::EVENT_SIG_FULL | hgcommon::EventKey_ConsumedEdges},
        {"endpoints+produced",     hgcommon::EVENT_SIG_FULL | hgcommon::EventKey_ProducedEdges},
        {"endpoints+step",         hgcommon::EVENT_SIG_FULL | hgcommon::EventKey_Step},
        {"Automatic",              hgcommon::EVENT_SIG_AUTOMATIC},
    };
    for (const Cell& c : cells) {
        const auto a = signatures(3, c.keys);
        const auto b = signatures(17, c.keys);
        ASSERT_FALSE(a.empty()) << c.name << ": no signatures stamped, comparison is vacuous";
        EXPECT_EQ(b, a) << c.name << ": the same evolution stamped different identities at 3 "
                        << "blocks (" << a.size() << ") and at 17 (" << b.size() << ")";
    }
}

// An identity mode has to CHANGE THE ANSWER, not merely be accepted.
//
// The device computed hgcommon::event_signature per event and then dropped it: there was no
// signature map, canonical_id was written INVALID_ID at its one assignment site and never
// updated, so every event read as canonical and the reported count was the raw application
// count whatever mode was asked for. Nothing caught it because every gate compared signature
// VALUES, and the values were being computed correctly -- they just were not applied.
//
// The check is a strict ordering rather than fixed numbers, so it stays meaningful if the
// workload changes: the key sets form a refinement lattice, so a finer key set can only split
// events, never merge them.
//
//   None       computes no signature; every application is its own event  -> the raw count
//   Automatic  endpoints + step + rule-free + consumed/produced ranks      -> finest
//   Full       endpoints alone                                            -> coarsest
//
// so  canonical(Full) <= canonical(Automatic) <= raw,  with at least one strict on a workload
// that reaches the same state by two different applications.
TEST(Rewrite, EventIdentityModesActuallyMergeEvents) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}};
    const uint32_t kSteps = 3;

    struct Result { uint32_t raw; uint32_t canonical; };
    auto run = [&](hgcommon::EventSignatureKeys keys) {
        hg_gpu::EvolveInput in;
        in.rules = {r};
        in.initial_state = init;
        in.num_steps = kSteps;
        in.canonicalization = hg_gpu::CanonicalizationMode::Full;

        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
        hg_gpu::EngineState engine(cfg);
        hg_gpu::upload_initial_state(engine, init);
        if (keys & (hgcommon::EventKey_ConsumedEdges | hgcommon::EventKey_ProducedEdges))
            engine.ensure_edge_ranks();

        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(64ull << 20);

        auto st = hg_gpu::run_persistent_evolve(
            engine, rules, /*roots=*/{0u}, kSteps, matches, arena,
            /*dedup=*/true, 0xFFFFFFFFu, 0,
            hg_gpu::CanonicalizationMode::Full, keys, /*blocks=*/9);
        return Result{engine.num_events_host(), st.canonical_events};
    };

    const Result none = run(hgcommon::EVENT_SIG_NONE);
    const Result full = run(hgcommon::EVENT_SIG_FULL);
    const Result automatic = run(hgcommon::EVENT_SIG_AUTOMATIC);

    // Printed, not only asserted: the ordering can hold while nothing merges, and the numbers
    // are what show the modes are separated rather than merely ordered.
    std::printf("[ identity ] applications=%u  canonical: None=%u Automatic=%u Full=%u\n",
                none.raw, none.raw, automatic.canonical, full.canonical);

    ASSERT_GT(none.raw, 1u) << "no events, so the comparison is vacuous";
    EXPECT_EQ(none.canonical, 0u)
        << "None computes no signature, so nothing can have won a signature slot";

    EXPECT_GT(full.canonical, 0u) << "Full stamped signatures but none were applied";
    EXPECT_LE(full.canonical, full.raw);
    EXPECT_LE(automatic.canonical, automatic.raw);
    EXPECT_LE(full.canonical, automatic.canonical)
        << "Full (" << full.canonical << ") distinguished MORE events than Automatic ("
        << automatic.canonical << "), but its key set is a strict subset";

    // The workload is a 4-cycle under a rule that reaches the same state by several
    // applications, so at least one mode has to merge something. Without this the ordering
    // above is satisfied by a device that merges nothing at all -- which is the defect.
    EXPECT_LT(full.canonical, full.raw)
        << "Full merged nothing: " << full.raw << " applications, " << full.canonical
        << " events. The signature is being computed and not applied.";
}

// Arena exhaustion is a RECOVERABLE capacity failure, and must be reported as one.
//
// The persistent scheduler claims IR scratch from a DeviceArena sized as a multiple of
// cfg.max_states. When a worker cannot get a slot it used to record kScratchOverflow -- the same
// kind as a fixed per-thread bound in the TR closure -- and the host's grow_config_for marks
// that kind non-retryable, on the reasoning that the caller "must accept the soft accuracy
// degradation (1-WL fallback)".
//
// Both halves of that are false here. There is no 1-WL fallback on this path, deliberately: a
// fallback key MERGES non-isomorphic states, so the design records a capacity overflow and
// returns partial work instead (docs/GPU_PERSISTENT_DESIGN.md sec 3a). And the arena IS
// config-controlled, so growing recovers it. A run that could have completed returned a partial
// result and called the cause unfixable.
//
// This pins the distinction the fix rests on: a starved arena reports kIRArenaExhausted, never
// kScratchOverflow, and the same run with a sufficient arena reports nothing at all.
TEST(Rewrite, StarvedIRArenaReportsARetryableKind) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}};
    const uint32_t kSteps = 3;

    auto run_with_arena = [&](uint64_t arena_words) {
        hg_gpu::EvolveInput in;
        in.rules = {r};
        in.initial_state = init;
        in.num_steps = kSteps;
        in.canonicalization = hg_gpu::CanonicalizationMode::Full;

        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
        hg_gpu::EngineState engine(cfg);
        hg_gpu::upload_initial_state(engine, init);

        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(arena_words);

        hg_gpu::run_persistent_evolve(engine, rules, /*roots=*/{0u}, kSteps, matches, arena,
                                      /*dedup=*/true, 0xFFFFFFFFu, 0,
                                      hg_gpu::CanonicalizationMode::Full,
                                      hgcommon::EVENT_SIG_FULL, /*blocks=*/9);
        std::vector<hg_gpu::OverflowWarning> w;
        engine.collect_warnings_into(w, "starved arena probe");
        return std::make_pair(w, engine.num_states_host());
    };

    // Big enough for every state this workload reaches.
    const auto [healthy, healthy_states] = run_with_arena(64ull << 20);
    EXPECT_TRUE(healthy.empty())
        << "the control run overflowed, so the starved comparison below proves nothing";
    ASSERT_GT(healthy_states, 1u) << "workload never branched; the comparison is vacuous";

    // Too small for even one slot, so every worker that needs one is refused.
    const auto [starved, starved_states] = run_with_arena(8ull);
    bool saw_arena = false, saw_scratch = false;
    for (const auto& w : starved) {
        if (w.kind == hg_gpu::ErrorKind::kIRArenaExhausted) saw_arena = true;
        if (w.kind == hg_gpu::ErrorKind::kScratchOverflow)  saw_scratch = true;
    }
    EXPECT_TRUE(saw_arena)
        << "a starved arena did not report kIRArenaExhausted; the host cannot know the failure "
        << "is one that growing the config would fix";
    EXPECT_FALSE(saw_scratch)
        << "a starved arena reported kScratchOverflow, which grow_config_for treats as an "
        << "unfixable kernel limit -- this is the conflation the separate kind exists to end";
    EXPECT_LT(starved_states, healthy_states)
        << "the starved run reached as many states as the healthy one, so the arena was not "
        << "actually the binding constraint and this test is not measuring what it claims";
}

// The limit of edge ranks, pinned on the state that reaches it.
//
// A canonical rank is a position in a canonical LABELLING. When a state has a nontrivial
// automorphism group the canonical labelling is not one labelling but a coset of them: the
// canonical FORM is unique, and so is the state's hash, but which of several interchangeable
// edges lands at a given position depends on the order the edges were presented in -- and that
// order is the order the rewrites allocated them, which is the schedule. So an individual
// edge's rank is well defined only up to the automorphism group.
//
// The consequence for the identity lattice: EventKey_ConsumedEdges and EventKey_ProducedEdges,
// and therefore EVENT_SIG_AUTOMATIC, are isomorphism invariants on states with a trivial
// automorphism group and only orbit-wise invariants otherwise. That is a property of what the
// mode identifies events by, not of this scheduler -- SPEC.md sec 4.2 records the same limit
// for the reference implementation's Positional identity and defaults event identity to
// Canonical for it.
//
// What survives on such a state, and what this pins: the COUNT of distinct identities. The
// automorphism permutes which signature each event gets; it does not merge or split them.
//
// The initial state is a directed 4-cycle, whose automorphism group is the rotations, order 4.
TEST(Rewrite, AutomorphicStateEventIdentityIsTheSameAtEveryBlockCount) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}};
    const uint32_t kSteps = 3;

    auto run = [&](uint32_t blocks) {
        hg_gpu::EvolveInput in;
        in.rules = {r};
        in.initial_state = init;
        in.num_steps = kSteps;
        in.canonicalization = hg_gpu::CanonicalizationMode::Full;

        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
        hg_gpu::EngineState engine(cfg);
        hg_gpu::upload_initial_state(engine, init);
        engine.ensure_edge_ranks();

        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(64ull << 20);

        hg_gpu::run_persistent_evolve(engine, rules, /*roots=*/{0u}, kSteps, matches, arena,
                                      /*dedup=*/true, 0xFFFFFFFFu, 0,
                                      hg_gpu::CanonicalizationMode::Full,
                                      hgcommon::EVENT_SIG_AUTOMATIC, blocks);

        const uint32_t ne = engine.num_events_host();
        std::vector<hg_gpu::DeviceEvent> events(ne);
        cudaMemcpy(events.data(), engine.device().event_pool.data,
                   sizeof(hg_gpu::DeviceEvent) * ne, cudaMemcpyDeviceToHost);

        // CANONICAL events only -- the ones that won their signature slot. Folding every
        // application instead would compare the raw application set, which is a different
        // question from what identities the run produced, and it is the comparison that made
        // an earlier reading of this look like a rank permutation.
        std::multiset<uint64_t> sigs;
        for (const auto& ev : events)
            if (ev.id != hg_gpu::INVALID_ID && ev.signature != 0 &&
                ev.canonical_id == hg_gpu::INVALID_ID)
                sigs.insert(ev.signature);
        return sigs;
    };

    const auto a = run(3);
    const auto b = run(17);
    ASSERT_FALSE(a.empty()) << "no signatures stamped, so the comparison is vacuous";

    // THE INVARIANT. An automorphism permutes which edge holds which canonical rank; it cannot
    // create or destroy an identity. So however the ranks land, the number of distinct event
    // identities is the same.
    EXPECT_EQ(b.size(), a.size())
        << "the run produced a different NUMBER of distinct event identities at 3 blocks ("
        << a.size() << ") and at 17 (" << b.size() << "), which a permutation of ranks cannot "
        << "do -- something other than the labelling moved";

    // NOT asserted, because it is measured to be false: over 15 runs the VALUES differed in 3,
    // and in a captured failure 14 of 15 signatures matched with exactly one differing. Ranks
    // are positions in a canonical LABELLING, and on a state whose automorphism group is
    // nontrivial that labelling is a coset -- which member the search settles on follows the
    // presentation order, and the presentation order is the order the rewrites appended edges.
    //
    // Asserting it here would make a gate that fails one run in five while the engine is doing
    // what it is defined to do. It is reported instead, so the rate stays visible, and the work
    // to make the presentation order itself canonical is tracked rather than hidden.
    if (a != b) {
        size_t common = 0;
        for (uint64_t s : a) if (b.count(s)) ++common;
        std::printf("[ automorphism ] same count (%zu) but %zu of %zu identity values differ "
                    "between 3 and 17 blocks\n", a.size(), a.size() - common, a.size());
    }
}

// ---------------------------------------------------------------------------------------
// #121: a device SESSION continues an evolution instead of restarting it.
//
// Two cheaper designs were measured first and both failed, in opposite directions, against the
// 7 states / 6 events one run to depth 3 produces:
//
//   identity maps rebuilt per call   10 states, 9 events -- nothing recognised, so the second
//                                    call re-derived everything it already had as new.
//   identity maps owned by caller     5 states, 4 events -- everything recognised, so nothing
//                                    was re-expanded and depth 3 was never reached.
//
// One bit was answering two questions: the dedup map says "seen", and the worker reads that same
// answer as "already expanded". A session separates them, carrying the maps AND a frontier of
// the states whose expansion the BUDGET refused -- the device twin of defer_match_task.
TEST(Rewrite, ADeviceSessionExtendsToExactlyWhatOneRunOfTheSameBudgetProduces) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}};
    auto input_for = [&](uint32_t steps) {
        hg_gpu::EvolveInput in;
        in.rules = {r};
        in.initial_state = init;
        in.num_steps = steps;
        in.canonicalization = hg_gpu::CanonicalizationMode::Full;
        return in;
    };

    uint32_t ref_states = 0, ref_events = 0;
    {
        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(input_for(3));
        hg_gpu::EngineState eng(cfg);
        hg_gpu::upload_initial_state(eng, init);
        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(32ull << 20);
        const auto st = hg_gpu::run_persistent_evolve(
            eng, rules, /*roots=*/{0u}, /*max_steps=*/3u, matches, arena, /*dedup=*/true,
            0xFFFFFFFFu, 0, hg_gpu::CanonicalizationMode::Full, hgcommon::EVENT_SIG_AUTOMATIC);
        ref_states = st.states_after;
        ref_events = st.canonical_events;
    }

    uint32_t ext_states = 0, ext_events = 0, frontier_after_first = 0;
    {
        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(input_for(3));
        hg_gpu::EngineState eng(cfg);
        hg_gpu::upload_initial_state(eng, init);
        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(32ull << 20);

        hg_gpu::SessionState sess(cfg.max_states, cfg.max_events);
        hg_gpu::SessionView v = sess.view();

        hg_gpu::run_persistent_evolve(
            eng, rules, /*roots=*/{0u}, /*max_steps=*/2u, matches, arena, /*dedup=*/true,
            0xFFFFFFFFu, 0, hg_gpu::CanonicalizationMode::Full, hgcommon::EVENT_SIG_AUTOMATIC,
            /*blocks=*/0, /*quotient_roots=*/false, nullptr, nullptr, &v, /*start_step=*/0u);

        // The budget stopped somewhere, so it must have recorded where.
        frontier_after_first = sess.frontier_size();

        const auto st2 = hg_gpu::run_persistent_evolve(
            eng, rules, /*roots=*/{0u}, /*max_steps=*/3u, matches, arena, /*dedup=*/true,
            0xFFFFFFFFu, 0, hg_gpu::CanonicalizationMode::Full, hgcommon::EVENT_SIG_AUTOMATIC,
            /*blocks=*/0, /*quotient_roots=*/false, nullptr, nullptr, &v, /*start_step=*/2u);
        ext_states = st2.states_after;
        ext_events = st2.canonical_events;
    }

    EXPECT_GT(frontier_after_first, 0u)
        << "the first call stopped at a budget and recorded no boundary, so there is nothing "
           "to continue from";
    EXPECT_EQ(ext_states, ref_states)
        << "a session did not reach the same state set as one run of the same budget";
    EXPECT_EQ(ext_events, ref_events)
        << "a session did not reach the same event set as one run of the same budget";
}
