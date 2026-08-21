// The frame-slot rule, gated in isolation.
//
// A slot is an edge's rank when a state's edges are ordered by (Aut ORBIT, EdgeId) -- the
// host's EdgeOrbitTable::slot, same definition and same tie-break. Everything the expansion
// capture records is expressed in slots, and a wrong slot replays as a wrong event, so the rule
// is gated here rather than only end to end.
//
// The state is built by hand: DeviceState is a struct of pointers, so a slice plus an orbit
// array is a complete input for qe_slot_of and no engine has to run.

#include <gtest/gtest.h>

#include "hg_gpu/quotient_expansion.hpp"

#include <vector>

namespace {

using hg_gpu::DeviceState;
using hg_gpu::EdgeId;
using hg_gpu::StateEdgeSlice;

// One thread evaluates qe_slot_of for every edge of state 0 and writes the slots out.
__global__ void k_slots(DeviceState ds, const EdgeId* edges, uint32_t n, uint32_t* out) {
    for (uint32_t i = 0; i < n; ++i) out[i] = hg_gpu::qe_slot_of(ds, 0u, edges[i]);
}

struct Fixture {
    std::vector<uint32_t> run(const std::vector<EdgeId>& edges,
                              const std::vector<uint32_t>& orbits) {
        const uint32_t n = static_cast<uint32_t>(edges.size());
        EdgeId*   d_ids   = nullptr;
        uint32_t* d_orb   = nullptr;
        StateEdgeSlice* d_slices = nullptr;
        uint32_t* d_out   = nullptr;
        EdgeId*   d_query = nullptr;
        cudaMalloc(&d_ids,   sizeof(EdgeId) * n);
        cudaMalloc(&d_orb,   sizeof(uint32_t) * n);
        cudaMalloc(&d_slices, sizeof(StateEdgeSlice));
        cudaMalloc(&d_out,   sizeof(uint32_t) * n);
        cudaMalloc(&d_query, sizeof(EdgeId) * n);
        cudaMemcpy(d_ids, edges.data(), sizeof(EdgeId) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_orb, orbits.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_query, edges.data(), sizeof(EdgeId) * n, cudaMemcpyHostToDevice);
        StateEdgeSlice sl{}; sl.offset = 0; sl.count = n;
        cudaMemcpy(d_slices, &sl, sizeof(sl), cudaMemcpyHostToDevice);

        DeviceState ds{};
        ds.max_states        = 1;
        ds.state_edge_ids    = d_ids;
        ds.state_edge_orbit  = d_orb;
        ds.state_edge_slices = d_slices;

        k_slots<<<1, 1>>>(ds, d_query, n, d_out);
        cudaDeviceSynchronize();

        std::vector<uint32_t> out(n);
        cudaMemcpy(out.data(), d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost);
        cudaFree(d_ids); cudaFree(d_orb); cudaFree(d_slices);
        cudaFree(d_out); cudaFree(d_query);
        return out;
    }
};

// Distinct orbits: the slot is the orbit order, whatever the edge ids are.
TEST(QuotientExpansion, SlotFollowsOrbitOrder) {
    Fixture f;
    auto s = f.run(/*edges=*/{10, 20, 30}, /*orbits=*/{2, 0, 1});
    EXPECT_EQ(s[0], 2u);   // edge 10, orbit 2 -> last
    EXPECT_EQ(s[1], 0u);   // edge 20, orbit 0 -> first
    EXPECT_EQ(s[2], 1u);   // edge 30, orbit 1
}

// Ties inside one orbit break on EdgeId, and the slice is ascending, so index order is id
// order. This is the tie-break the replay depends on being the same in every instance.
TEST(QuotientExpansion, TiesInsideAnOrbitBreakOnEdgeId) {
    Fixture f;
    auto s = f.run({10, 20, 30, 40}, {1, 0, 1, 0});
    // orbit 0 holds edges 20 and 40 -> slots 0,1; orbit 1 holds 10 and 30 -> slots 2,3
    EXPECT_EQ(s[1], 0u);   // edge 20
    EXPECT_EQ(s[3], 1u);   // edge 40
    EXPECT_EQ(s[0], 2u);   // edge 10
    EXPECT_EQ(s[2], 3u);   // edge 30
}

// Slots are a permutation of [0, n): every edge gets exactly one, and none collide. This is
// what makes a slot vector a frame rather than a labelling with holes.
TEST(QuotientExpansion, SlotsAreAPermutation) {
    Fixture f;
    auto s = f.run({3, 5, 7, 11, 13}, {2, 2, 0, 1, 2});
    std::vector<uint32_t> seen(s.size(), 0);
    for (uint32_t v : s) {
        ASSERT_LT(v, s.size());
        ++seen[v];
    }
    for (uint32_t c : seen) EXPECT_EQ(c, 1u);
}

// An edge that is not in the state has no slot. The capture drops such a record rather than
// writing a slot that means nothing, so the sentinel has to be distinguishable.
TEST(QuotientExpansion, AbsentEdgeHasNoSlot) {
    const std::vector<EdgeId> edges{4, 8};
    const std::vector<uint32_t> orbits{0, 1};
    const uint32_t n = 2;
    EdgeId* d_ids = nullptr; uint32_t* d_orb = nullptr;
    StateEdgeSlice* d_slices = nullptr; uint32_t* d_out = nullptr; EdgeId* d_query = nullptr;
    cudaMalloc(&d_ids, sizeof(EdgeId) * n);
    cudaMalloc(&d_orb, sizeof(uint32_t) * n);
    cudaMalloc(&d_slices, sizeof(StateEdgeSlice));
    cudaMalloc(&d_out, sizeof(uint32_t));
    cudaMalloc(&d_query, sizeof(EdgeId));
    cudaMemcpy(d_ids, edges.data(), sizeof(EdgeId) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_orb, orbits.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice);
    const EdgeId absent = 9;
    cudaMemcpy(d_query, &absent, sizeof(EdgeId), cudaMemcpyHostToDevice);
    StateEdgeSlice sl{}; sl.offset = 0; sl.count = n;
    cudaMemcpy(d_slices, &sl, sizeof(sl), cudaMemcpyHostToDevice);

    DeviceState ds{};
    ds.max_states = 1;
    ds.state_edge_ids = d_ids;
    ds.state_edge_orbit = d_orb;
    ds.state_edge_slices = d_slices;

    k_slots<<<1, 1>>>(ds, d_query, 1u, d_out);
    cudaDeviceSynchronize();
    uint32_t got = 0;
    cudaMemcpy(&got, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got, UINT32_MAX);

    cudaFree(d_ids); cudaFree(d_orb); cudaFree(d_slices); cudaFree(d_out); cudaFree(d_query);
}

// No orbit array means no frame, so no slot -- the run did not ask for the reconstruction and
// the capture must not invent ranks from the raw ids.
TEST(QuotientExpansion, NoOrbitArrayMeansNoSlot) {
    EdgeId* d_query = nullptr; uint32_t* d_out = nullptr;
    StateEdgeSlice* d_slices = nullptr; EdgeId* d_ids = nullptr;
    cudaMalloc(&d_query, sizeof(EdgeId));
    cudaMalloc(&d_out, sizeof(uint32_t));
    cudaMalloc(&d_slices, sizeof(StateEdgeSlice));
    cudaMalloc(&d_ids, sizeof(EdgeId));
    const EdgeId e = 4;
    cudaMemcpy(d_query, &e, sizeof(EdgeId), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, &e, sizeof(EdgeId), cudaMemcpyHostToDevice);
    StateEdgeSlice sl{}; sl.offset = 0; sl.count = 1;
    cudaMemcpy(d_slices, &sl, sizeof(sl), cudaMemcpyHostToDevice);

    DeviceState ds{};
    ds.max_states = 1;
    ds.state_edge_ids = d_ids;
    ds.state_edge_orbit = nullptr;     // the run has no orbits
    ds.state_edge_slices = d_slices;

    k_slots<<<1, 1>>>(ds, d_query, 1u, d_out);
    cudaDeviceSynchronize();
    uint32_t got = 0;
    cudaMemcpy(&got, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got, UINT32_MAX);

    cudaFree(d_query); cudaFree(d_out); cudaFree(d_slices); cudaFree(d_ids);
}

}  // namespace

// =============================================================================
// The reconstruction does not depend on the recursion budget
// =============================================================================

#include "hg_gpu/evolve.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/match.hpp"

// THE DEPTH A RUN CAN RECONSTRUCT IS NOT A PROPERTY OF THE LAUNCH, and this is the assertion
// that says so without needing a workload deep enough to be infeasible.
//
// Both the replay and the causal DP used to descend by CALLING themselves, on a per-thread stack
// the driver reserves across every RESIDENT thread. That reservation cannot be made large -- it
// is an occupancy tax on the whole device, and paid by all 32 threads of a block for a path only
// thread 0 executes -- so it was capped, and past the cap the device recorded kScratchOverflow
// and returned a PARTIAL reconstruction. The host, recursing at 512 bytes a level on a thread
// stack with about a thousand levels of room, returned a complete one. The two engines answered
// different questions above the cap.
//
// Rather than reach that depth -- the replay is exponential in it by construction, which is why
// nobody hit this -- run the SAME evolution twice with two different recursion budgets and
// require the same answer. A build whose depth is bounded by the budget cannot pass this: the
// small-budget arm truncates. It is the property itself, at a depth that costs nothing.
TEST(QuotientExpansion, TheCausalRelationDoesNotDependOnTheRecursionBudget) {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}};
    r.rhs = {{0, 1}, {1, 2}};
    r.num_lhs_vars = 2;
    r.num_rhs_vars = 3;

    const std::vector<std::vector<hg_gpu::VertexId>> init = {{0u, 1u}};
    const uint32_t kSteps = 5;

    // One evolution, parameterised only by the budget the cascade is allowed to recurse to.
    auto run_with_budget = [&](uint32_t budget) {
        hg_gpu::EvolveInput in;
        in.rules         = {r};
        in.initial_state = init;
        in.num_steps     = kSteps;
        in.record.causal = true;

        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
        hg_gpu::EngineState engine(cfg);
        hg_gpu::upload_initial_state(engine, init);
        engine.ensure_edge_orbits();

        std::vector<hg_gpu::DeviceRule> rules = {hg_gpu::make_device_rule(r)};
        hg_gpu::Pool<hg_gpu::MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        hg_gpu::DeviceArena arena(64ull << 20);

        hg_gpu::QcState qc(/*on=*/true, cfg.max_events);
        qc.clear();
        qc.set_record_causal(true);
        qc.ensure_work(64u, kSteps);
        const hg_gpu::QcView qc_view = qc.view(kSteps, budget);

        engine.set_quotient_causal(true);
        hg_gpu::run_persistent_evolve(
            engine, rules, /*roots=*/{0u}, kSteps, matches, arena, /*dedup=*/true,
            /*explore_threshold_u32=*/0xFFFFFFFFu, /*explore_seed=*/0,
            hg_gpu::CanonicalizationMode::Full, hgcommon::EVENT_SIG_NONE, /*blocks=*/8,
            /*quotient_roots=*/true, &qc_view, /*qe_in=*/nullptr, /*session=*/nullptr,
            /*start_step=*/0);

        // The DP emits through the engine's causal pool, so the relation is read from there.
        return engine.num_causal_edges_host();
    };

    // A budget far below the depth the cascade reaches, and one far above it.
    const uint32_t shallow = run_with_budget(2u);
    const uint32_t deep    = run_with_budget(64u);

    EXPECT_GT(deep, 0u) << "the workload must actually produce a causal relation to compare";
    EXPECT_EQ(shallow, deep)
        << "the causal relation changed with the recursion budget: " << shallow << " against "
        << deep << ". The cascade is bounded by the per-thread stack rather than by the "
           "evolution, so a deep run answers a different question from the host's.";
}
