// GPUMC harness: the lanes of a branchial-rows warp count a row's pairs without a race and record
// every pair.
//
// Runs gpu/src/branchial_rows.cu's k_branchial_rows itself -- the row claim, hgcommon::br_row on
// the warp's lanes (IrTile: fan, sync, broadcast), the shared-memory tally and the record append --
// with the device's atomics, barriers and shuffles (verification/gpumc/device).
//
// THE PROTOCOL. Lane 0 claims a row with an add on next_row and broadcasts it. The lanes count the
// states sharing a vertex with the row's state (an add per holder, the first adder pushing the
// holder on `touched`), synchronise, optionally search the branchial graph a level at a time
// (claiming a node's distance by compare-exchange and pushing it on the queue), emit each pair
// into the warp's tally (a compare-exchange on a key slot, an add on its count), reset the
// scratch, synchronise, and append the tally's entries to the record buffer.
//
// THE SHAPES. One block, a warp of two lanes. The kernel strides its loops by 32 lanes; every loop
// here has at most two items, which lanes 0 and 1 take.
//   default          two states sharing one vertex: one pair, at distance 0.
//   -DSHAPE_GRAPH    the same, with a branchial graph of two adjacent nodes, one per state: the
//                    pair also at distance 1.
//   -DSHAPE_THREE    three states sharing one vertex: row 0's two pairs are counted by the two
//                    lanes at once, and the three pairs carry one key.
//
// THE PROPERTY. Summed over the records, every pair is counted once under its key: (distance 0,
// |b| 1, shared 1) for every pair, and (1, 1, 1) for the graph's. Every row is claimed. No access
// races; no barrier divergence.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW splits the adds into a load and a store: both lanes
// can push to one `touched` index, and the checker must report a lost pair.
// GPUMC-CELL: shared PASS
// GPUMC-CELL: graph PASS -DSHAPE_GRAPH
// GPUMC-CELL: three PASS -DSHAPE_THREE
// GPUMC-CELL: three_split_rmw FAIL -DSHAPE_THREE -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "branchial_rows.cu"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

#if defined(SHAPE_THREE)
constexpr uint32_t kStates = 3;
const uint32_t kSize[3] = {1, 1, 1};
const uint32_t kRowOff[4] = {0, 1, 2, 3};
const uint32_t kRowVert[3] = {0, 0, 0};
const uint32_t kHoldOff[2] = {0, 3};
const uint32_t kHold[3] = {0, 1, 2};
constexpr uint32_t kPairs0 = 3;   // pairs at distance 0
#else
constexpr uint32_t kStates = 2;
const uint32_t kSize[2] = {1, 1};
const uint32_t kRowOff[3] = {0, 1, 2};
const uint32_t kRowVert[2] = {0, 0};
const uint32_t kHoldOff[2] = {0, 2};
const uint32_t kHold[2] = {0, 1};
constexpr uint32_t kPairs0 = 1;
#endif
const uint32_t kAdjOff[3] = {0, 1, 2};
const uint32_t kAdj[2] = {1, 0};
const uint32_t kToGraph[2] = {0, 1};
const uint32_t kFromGraph[2] = {0, 1};

uint32_t g_next_row;
uint32_t g_count[kStates], g_touched[kStates], g_queue[kStates], g_counters[2];
int32_t g_dist[kStates] = {-1, -1};
g::BrRecord g_out[8];
uint32_t g_out_n, g_wide;

void* lane(void* arg) {
    const unsigned l = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(0, l);
    hgcommon::BrStep s;
    s.k = kStates;
    s.size = kSize;
    s.row_off = kRowOff;
    s.row_vert = kRowVert;
    s.vertices = 1;
    s.hold_off = kHoldOff;
    s.hold = kHold;
#if defined(SHAPE_GRAPH)
    s.n = 2;
    s.adj_off = kAdjOff;
    s.adj = kAdj;
    s.to_graph = kToGraph;
    s.from_graph = kFromGraph;
#endif
    g::k_branchial_rows(s, kStates, &g_next_row, g_count, g_touched, g_dist, g_queue, g_counters,
                        g_out, &g_out_n, 8, &g_wide, nullptr, true, hgcommon::BoMatrices{});
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[1] = {2};
    hg_gpumc::launch(1, lanes);
    pthread_t t[2];
    for (long l = 0; l < 2; ++l) pthread_create(&t[l], nullptr, lane, reinterpret_cast<void*>(l));
    for (int l = 0; l < 2; ++l) pthread_join(t[l], nullptr);

    unsigned long long k0 = 0, k1 = 0;
    assert(g::br_key(0, 1, 1, k0) && g::br_key(1, 1, 1, k1));
    uint32_t n0 = 0, n1 = 0;
    assert(g_out_n <= 8);
    for (uint32_t i = 0; i < g_out_n; ++i) {
        assert(g_out[i].size_a == 1u);
        if (g_out[i].key == k0) n0 += g_out[i].n;
        else if (g_out[i].key == k1) n1 += g_out[i].n;
        else assert(false && "a record carries a key no pair has");
    }
    assert(n0 == kPairs0 && "a pair at distance 0 was lost or counted twice");
#if defined(SHAPE_GRAPH)
    assert(n1 == 1u && "the pair at distance 1 was lost or counted twice");
#else
    assert(n1 == 0u);
#endif
    assert(g_wide == 0);
    assert(g_next_row >= kStates + 1 && "the warp stopped before claiming past the last row");
    return 0;
}
