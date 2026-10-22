// GPUMC harness: Automatic-mode state identity merges states of equal content into one class and
// gives a state of other content under the same hash its own class.
//
// Runs gpu/include/hg_gpu/exploration.hpp's state_claim_content itself (keyed_claim_device with
// hgcommon::content_equal over DeviceContentCursor), with the device's map, atomics and fences
// (verification/gpumc/device).
//
// THE PROTOCOL. A state's slice is written before its claim; make() fences before the exchange
// that publishes the state id under the content hash, and a claimant that finds a key fences and
// compares its content with the holder's slice.
//
// THE SHAPE. Edges 0 = {1, 2} and 1 = {1, 3}. Blocks 0 and 1 write states 0 and 1, each holding
// edge 0; block 2 writes state 2, holding edge 1. All three claim under one content hash.
//
// THE PROPERTY. States 0 and 1 share a class and key, one of them fresh; state 2 is fresh in a
// class of its own under another key. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the map's exchanges a load and a separate
// store; the checker must report two fresh states for one content. no_threadfence passes: the
// map's acq_rel value exchange and acquire lookups publish the slice without the fences.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: no_threadfence PASS -DHG_GPUMC_CALIBRATE_NO_THREADFENCE
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/exploration.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Map = g::DedupMap;

constexpr uint32_t kSlots = 4;
constexpr uint64_t kHash = 0x4321;

g::DeviceState* g_ds;
g::VertexId g_verts[4] = {1, 2, 1, 3};
g::Edge g_edges[2];
uint32_t g_edges_n = 2;
g::StateEdgeSlice g_slices[3];
g::EdgeId g_ids[3];
g::MapSlot<uint64_t, uint32_t> g_slots[kSlots];
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
g::StateClaim g_claim[3];

void* claimant(void* arg) {
    const unsigned sid = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(sid, 0);
    g_ids[sid] = sid < 2 ? 0u : 1u;
    g_slices[sid] = {sid, 1};
    const Map::DeviceView map{g_slots, kSlots, kSlots - 1, nullptr};
    g_claim[sid] = g::state_claim_content(*g_ds, sid, kHash, map);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    for (uint32_t s = 0; s < kSlots; ++s) g_slots[s] = {0, Map::DeviceView::UNPUBLISHED};
    for (uint32_t e = 0; e < 2; ++e) {
        g_edges[e].arity = 2;
        g_edges[e].vertex_offset = 2 * e;
        g_edges[e].creator_event = g::INVALID_ID;
    }
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->vertex_pool = {g_verts, nullptr, 4};
    g_ds->edge_pool = {g_edges, &g_edges_n, 2};
    g_ds->state_edge_slices = g_slices;
    g_ds->state_edge_ids = g_ids;
    g_ds->max_states = 3;
    g_ds->canonical_key_mask = ~uint64_t{0};

    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, claimant, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    assert(g_claim[0].canonical == g_claim[1].canonical && g_claim[0].key == g_claim[1].key &&
           "two states of equal content were not merged");
    assert(g_claim[0].fresh != g_claim[1].fresh);
    assert(g_claim[2].canonical == 2 && g_claim[2].fresh && g_claim[2].key != g_claim[0].key &&
           "a state of other content joined their class");
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
