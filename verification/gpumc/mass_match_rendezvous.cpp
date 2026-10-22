// GPUMC harness: the device's raw counts from class multiplicities lose no mass and pass none
// twice.
//
// Runs gpu/include/hg_gpu/quotient_expansion.hpp's qe_capture_multiplicity and DeviceQmCtx itself
// (hgcommon/quotient_multiplicity_core.hpp's qm_credit, qm_pass and qm_drain over the device's
// point and cell maps, saturating_add, the queued flags and the by_from bucket), with the device's
// atomics and fences (verification/gpumc/device). The host's twin,
// verification/genmc/quotient_mass_match_rendezvous.cpp, checks the core under RC11.
//
//   the capture of j:     record b_j (j is ready), fence, qm_pass(j), drain
//   mass arriving at c:   add to m(c, 0), fence, claim the queued flag, push if won, drain
//   a queued point runs:  pop clears the flag, fence, read m(c, 0), qm_pass every ready match
//
// THE SHAPE. Class c with one captured match j into class t, max_steps 1. The capture of j races
// an arrival of 1 unit at (c, 0) (one_arrival), or arrivals of 1 and 2 units (the long cells:
// no verdict in 570 s).
//
// THE PROPERTY. When every block has finished, j has passed on exactly the units that arrived:
// its consumed cell at depth 0, t's mass at depth 1 and the raw event count all equal them. No
// access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_NO_THREADFENCE makes the fences no-ops; the checker must report
// mass that j never passes on.
// GPUMC-CELL: one_arrival PASS -DSHAPE_ONE_ARRIVAL
// GPUMC-CELL: one_arrival_no_threadfence FAIL -DSHAPE_ONE_ARRIVAL -DHG_GPUMC_CALIBRATE_NO_THREADFENCE
// GPUMC-CELL-LONG: clean PASS
// GPUMC-CELL-LONG: no_threadfence FAIL -DHG_GPUMC_CALIBRATE_NO_THREADFENCE

#include "hg_gpumc.hpp"
#include "hg_gpu/quotient_expansion.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using MatchNode = g::LockFreeList<g::QeMatchRef>::Node;
using Slot = g::MapSlot<uint64_t, uint32_t>;

constexpr uint64_t kC = 0x5151, kT = 0x7272;
constexpr uint32_t kKeys = 2, kSlots = 4, kCells = 4, kItems = 4;
#if defined(SHAPE_ONE_ARRIVAL)
constexpr uint32_t kBlocks = 2;
constexpr uint64_t kUnits = 1;
#else
constexpr uint32_t kBlocks = 3;
constexpr uint64_t kUnits = 3;
#endif

g::DeviceState* g_ds;
g::QeView* g_qe;
g::DeviceSlotMatch g_j;
uint32_t g_matches_n = 1;
uint32_t g_words[2];
MatchNode g_match_nodes[1];
uint32_t g_match_nodes_n;
uint32_t g_match_heads[kKeys];
Slot g_points[kSlots], g_consumed[kSlots], g_overlaps[kSlots];
unsigned long long g_mass[kCells], g_point_class[kCells], g_consumed_cells[kCells];
uint32_t g_point_depth[kCells], g_queued[kCells], g_cursor[2];
unsigned long long g_counts[2];
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
g::QeWorkItem g_items[3][kItems];

g::QeWork work(unsigned block) { return g::QeWork{g_items[block], kItems, 0}; }

void* capture(void*) {
    hg_gpumc::enter(0, 0);
    g::QeWork w = work(0);
    g::qe_capture_multiplicity(*g_ds, *g_qe, g_j, kC, 0, 0, w);
    return nullptr;
}

void* arrive(void* arg) {
    const unsigned block = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(block, 0);
    g::QeWork w = work(block);
    g::DeviceQmCtx c{*g_ds, *g_qe, w};
    hgcommon::qm_credit(c, kC, 0u, block);
    hgcommon::qm_drain(c);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(kBlocks, lanes);
    for (uint32_t k = 0; k < kKeys; ++k) g_match_heads[k] = g::INVALID_ID;
    for (uint32_t s = 0; s < kSlots; ++s)
        g_points[s] = g_consumed[s] = g_overlaps[s] = {0, g::DedupMap::DeviceView::UNPUBLISHED};
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_qe = new (malloc(sizeof(g::QeView))) g::QeView{};
    g_qe->enabled = 1;
    g_qe->multiplicity = 1;
    g_qe->max_steps = 1;
    g_qe->keys = hgcommon::EVENT_SIG_NONE;
    g_qe->matches = {&g_j, &g_matches_n, 1};
    g_qe->arr_words = g_words;
    g_qe->by_from = {{g_match_nodes, &g_match_nodes_n, 1}, g_match_heads, kKeys};
    g_qe->qm_points = {g_points, kSlots, kSlots - 1, nullptr};
    g_qe->qm_consumed = {g_consumed, kSlots, kSlots - 1, nullptr};
    g_qe->qm_overlaps = {g_overlaps, kSlots, kSlots - 1, nullptr};
    g_qe->qm_mass = g_mass;
    g_qe->qm_point_class = g_point_class;
    g_qe->qm_point_depth = g_point_depth;
    g_qe->qm_queued = g_queued;
    g_qe->qm_consumed_cells = g_consumed_cells;
    g_qe->qm_cursor = g_cursor;
    g_qe->qm_capacity = kCells;
    g_qe->qm_counts = g_counts;
    g_j.id = 0;
    g_j.from_hash = kC;
    g_j.to_hash = kT;
    // j is in c's bucket before any block starts: qe_capture_expansion links it before the
    // capture runs, and an arrival that misses it is the capture's to pass on.
    const uint64_t key = g::qe_expansion_key(*g_ds, kC, 0);
    g_qe->by_from.push(g::qe_bucket(key, kKeys), g::QeMatchRef{key, 0});

    pthread_t t[kBlocks];
    pthread_create(&t[0], nullptr, capture, nullptr);
    for (long b = 1; b < kBlocks; ++b) pthread_create(&t[b], nullptr, arrive, reinterpret_cast<void*>(b));
    for (uint32_t b = 0; b < kBlocks; ++b) pthread_join(t[b], nullptr);

    g::QeWork w = work(0);
    g::DeviceQmCtx c{*g_ds, *g_qe, w};
    assert(c.consumed(g::QeMatchView(g_j, g_words), 0) == kUnits &&
           "mass that arrived was not passed on exactly once");
    assert(c.mass(kT, 1) == kUnits);
    assert(g_counts[0] == kUnits);
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
