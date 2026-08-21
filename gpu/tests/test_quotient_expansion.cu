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

// The recursion-budget test that stood here is GONE BECAUSE ITS PARAMETER IS.
//
// It ran one evolution at two values of QcState::view's max_recursion_depth and required one
// answer, and it earned its keep: against the old policy it reported 2 causal edges at a short
// budget and 106 at a long one, which is what established that the bound truncated the causal
// relation rather than merely warning about it. There is no such argument now -- the cascade
// carries depth in a worklist and nothing can be set to a depth it refuses past -- so the test
// could only assert that a knob which does not exist has no effect.
//
// What it asserted OBSERVABLY is covered where CPU/GPU agreement belongs:
// QuotientReconstruction.PastTheOldStackDepthItReachesTheDepthInstead runs 80 deep, which is ten
// times the fixed nest budget and so exercises the deferral path, and requires that no
// kScratchOverflow is recorded; and the 28-workload differential corpus compares the causal and
// branchial relations against the host on every one.
