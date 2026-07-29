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

// Match and rewrite running as two persistent roles must produce exactly what the
// level-synchronous match-then-rewrite produces.
//
// Stage 2 of retiring the step loop. The roles feed each other with no barrier: a match is
// applied as soon as some worker claims it, not after every match in the step has been found.
// Both schedulers drive the same match_state_rule and apply_one_match, so any difference is in
// the scheduling, and scheduling must not change the result.
//
// "The result" is the CANONICAL form, which is the whole reason removing the barrier is sound
// (docs/GPU_PERSISTENT_DESIGN.md §2). Raw vertex ids are not part of it: fresh vertices come
// from a high-water bump, so which rewrite gets which id follows execution order, and the two
// schedulers legitimately produce the same hypergraph under a different labelling.
//
// The failure mode this guards against is not a wrong answer but a HANG: a detector that
// signals exit during a lull loses work, and one that never signals never returns. Run under a
// timeout.
TEST(Rewrite, PersistentTwoRoleSchedulerMatchesTheLevelSynchronousResult) {
    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 4u}};

    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;
    std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
    const std::vector<hg_gpu::StateId> states = {0u};

    // Level-synchronous: find every match, then apply every match.
    hg_gpu::EngineState lockstep(small_cfg());
    hg_gpu::upload_initial_state(lockstep, init);
    hg_gpu::Pool<hg_gpu::MatchRecord> ls_matches(256);
    ls_matches.reset();
    hg_gpu::DeviceRule* d_rules = nullptr;
    cudaMalloc(&d_rules, sizeof(hg_gpu::DeviceRule) * rules.size());
    cudaMemcpy(d_rules, rules.data(), sizeof(hg_gpu::DeviceRule) * rules.size(),
               cudaMemcpyHostToDevice);
    hg_gpu::StateId* d_states = nullptr;
    cudaMalloc(&d_states, sizeof(hg_gpu::StateId) * states.size());
    cudaMemcpy(d_states, states.data(), sizeof(hg_gpu::StateId) * states.size(),
               cudaMemcpyHostToDevice);
    const uint32_t ls_n = hg_gpu::run_match_kernel_batch(
        lockstep, d_rules, static_cast<uint32_t>(rules.size()), d_states,
        static_cast<uint32_t>(states.size()), ls_matches);
    cudaFree(d_states);
    cudaFree(d_rules);
    ASSERT_GT(ls_n, 0u) << "workload found no matches, so the comparison is vacuous";
    hg_gpu::run_rewrite_kernel(lockstep, rules, ls_matches, ls_n, /*step=*/1);

    // Persistent: the two roles interleave, with block 0 deciding when they are finished.
    hg_gpu::EngineState persistent(small_cfg());
    hg_gpu::upload_initial_state(persistent, init);
    hg_gpu::Pool<hg_gpu::MatchRecord> p_matches(256);
    p_matches.reset();
    const auto stats = hg_gpu::run_persistent_match_rewrite(
        persistent, rules, states, /*step=*/1, p_matches, /*blocks=*/5);

    EXPECT_EQ(stats.matches_found, ls_n);
    EXPECT_EQ(persistent.num_states_host(), lockstep.num_states_host());
    EXPECT_EQ(persistent.num_edges_host(), lockstep.num_edges_host());
    EXPECT_EQ(canonical_hash_multiset(persistent), canonical_hash_multiset(lockstep));
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
        /*explore_threshold_u32=*/0xFFFFFFFFu, /*explore_seed=*/0, /*blocks=*/5);

    // Two matches in the root (one per edge), so two children and no further expansion.
    EXPECT_EQ(stats.matches_found, 2u);
    EXPECT_EQ(stats.states_after, 3u);
}

// A whole multi-step evolution inside ONE launch must produce exactly what the level-
// synchronous Engine produces for the same input.
//
// Stage 3: the rewrite's output is hashed, deduplicated and re-enqueued on device, so there is
// no step loop and no host in the middle. That makes three things load-bearing that stages 1
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
        /*explore_threshold_u32=*/0xFFFFFFFFu, /*explore_seed=*/0, /*blocks=*/9);

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
// The level-synchronous scheduler cannot do this at all: it writes the event in the rewrite
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
