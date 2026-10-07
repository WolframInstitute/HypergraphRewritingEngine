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

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/quotient_expansion.hpp"

#include <vector>

namespace {

using hg_gpu::DeviceState;
using hg_gpu::EdgeId;
using hg_gpu::StateEdgeSlice;

// One thread evaluates qe_slot_of for every edge of state 0 and writes the slots out.
__global__ void k_alloc_words(const __grid_constant__ DeviceState ds, hg_gpu::QeView qe, uint32_t n, uint32_t* out) {
    *out = hg_gpu::qe_alloc_words(ds, qe, n);
}

// Push one instance reference for `key` into the bucket of `shard`, then count the instances
// qe_for_each_instance visits for (hash, depth).
__global__ void k_count_one_instance(hg_gpu::QeView qe, uint64_t hash, uint32_t depth,
                                     uint32_t shard, uint32_t* out) {
    const uint64_t key = hg_gpu::qe_inst_key(hash, depth);
    qe.by_key.push(hg_gpu::qe_inst_bucket(qe, key, shard), hg_gpu::QeInstRef{key, 0u});
    uint32_t n = 0;
    hg_gpu::qe_for_each_instance(qe, hash, depth, [&](const hg_gpu::DeviceQcInstance&) { ++n; });
    *out = n;
}

__global__ void k_slots(const __grid_constant__ DeviceState ds, const EdgeId* edges, uint32_t n, uint32_t* out) {
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


// An allocation that overflows the expansion arena advances its cursor past the capacity and
// writes nothing (qe_alloc_words). The readback of the reduced relation then reads the arena up
// to the capacity, not up to the cursor: a copy past the allocation fails, and the failure ends
// the grow-and-retry ladder instead of growing the arena.
TEST(QuotientExpansion, ReducedReadbackAfterArenaOverflowReadsOnlyTheArena) {
    hg_gpu::QeState qe(/*on=*/true, hg_gpu::QeEntries{1, 1, 1, 1, 16});   // an arena of 16 words
    const hg_gpu::QeView v = qe.view(/*max_steps=*/1, hgcommon::EVENT_SIG_NONE,
                                     /*replay=*/true, /*multiplicity=*/false,
                                     /*event_content=*/false);
    ASSERT_EQ(v.arr_capacity, 16u);

    // Raw event 0 kept four producers: three inline, the fourth spilled to arena word 0.
    const uint32_t one = 1;
    ASSERT_EQ(cudaMemcpy(v.next_raw_event, &one, sizeof(one), cudaMemcpyHostToDevice), cudaSuccess);
    const uint32_t kept[hg_gpu::kQeKeptStride] = {4u, 0u, 0u, 0u, 0u};
    ASSERT_EQ(cudaMemcpy(v.event_kept, kept, sizeof(kept), cudaMemcpyHostToDevice), cudaSuccess);
    const uint32_t spilled = 0;
    ASSERT_EQ(cudaMemcpy(v.arr_words, &spilled, sizeof(spilled), cudaMemcpyHostToDevice),
              cudaSuccess);
    // Later allocations overflowed and moved the cursor far past the 16 words.
    const uint32_t past = 1u << 26;
    ASSERT_EQ(cudaMemcpy(v.arr_cursor, &past, sizeof(past), cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<std::pair<uint64_t, uint64_t>> causal, reduced, branchial;
    EXPECT_NO_THROW(qe.reconstructed_pairs_host(causal, reduced, branchial,
                                                /*want_branchial=*/true, /*raw_events=*/1u,
                                                nullptr));
    EXPECT_EQ(reduced.size(), 4u);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}


// Failed arena allocations advance the cursor, and a long overflowing run drives it past 2^32.
// The capacity test must not wrap there: an offset near 2^32 plus the request lands back inside
// the capacity, and the caller writes 16 GB past the arena (bigpath and bigcycle at 3 steps).
TEST(QuotientExpansion, ArenaAllocationNearTheCursorWrapIsRefused) {
    hg_gpu::EngineConfig cfg;          // the smallest engine test_engine_state.cu builds
    cfg.max_edges            = 64;
    cfg.max_state_edge_total = 256;
    cfg.max_states           = 8;
    cfg.max_vertex_slots     = 256;
    cfg.max_vertices         = 64;
    cfg.sig_index_buckets    = 16;
    cfg.sig_index_pool       = 64;
    cfg.inverted_pool        = 256;
    hg_gpu::EngineState engine(cfg);   // for the error channel qe_alloc_words records into
    hg_gpu::QeState qe(/*on=*/true, hg_gpu::QeEntries{1, 1, 1, 1, 16});   // an arena of 16 words
    const hg_gpu::QeView v = qe.view(/*max_steps=*/1, hgcommon::EVENT_SIG_NONE,
                                     /*replay=*/true, /*multiplicity=*/false,
                                     /*event_content=*/false);
    ASSERT_EQ(v.arr_capacity, 16u);
    const uint32_t near_wrap = 0xFFFFFFF0u;
    ASSERT_EQ(cudaMemcpy(v.arr_cursor, &near_wrap, sizeof(near_wrap), cudaMemcpyHostToDevice),
              cudaSuccess);
    uint32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t)), cudaSuccess);
    k_alloc_words<<<1, 1>>>(engine.device(), v, 32u, d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t got = 0, cursor = 0;
    cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cursor, v.arr_cursor, sizeof(cursor), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    EXPECT_EQ(got, UINT32_MAX) << "an allocation whose end wraps past 2^32 was granted at " << got;
    EXPECT_LE(cursor, v.arr_capacity) << "a refused allocation left the cursor past the arena";
}


// Two shards of one (class, depth) key can hash to one bucket. qe_for_each_instance -- what the
// device's branchial count walks -- takes each bucket once: an instance in a shared bucket is
// visited once, not once per shard that maps there.
TEST(QuotientExpansion, AnInstanceInABucketTwoShardsShareIsVisitedOnce) {
    hg_gpu::QeState qe(/*on=*/true, hg_gpu::QeEntries{1, 1, 1, 1, 16});
    const hg_gpu::QeView v = qe.view(/*max_steps=*/1, hgcommon::EVENT_SIG_NONE,
                                     /*replay=*/true, /*multiplicity=*/false,
                                     /*event_content=*/false);
    const uint32_t buckets = v.by_key.num_keys;
    uint64_t hash = 0;
    uint32_t shard = 0;   // a shard whose bucket an earlier shard of the same key also takes
    for (uint64_t h = 1; h < 10000000ull && hash == 0; ++h)
        for (uint32_t s = 1; s < hg_gpu::kQeInstShards; ++s)
            if (!hg_gpu::qe_inst_shard_first(hg_gpu::qe_inst_key(h, 0), s, buckets)) {
                hash = h;
                shard = s;
                break;
            }
    ASSERT_NE(hash, 0u) << "no key below 10^7 has two shards in one of " << buckets << " buckets";
    uint32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t)), cudaSuccess);
    k_count_one_instance<<<1, 1>>>(v, hash, 0u, shard, d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t got = 0;
    cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    EXPECT_EQ(got, 1u) << "one instance was visited " << got << " times";
}

}  // namespace


