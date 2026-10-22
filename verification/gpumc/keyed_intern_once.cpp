// GPUMC harness: keyed rewrites on the device give two applications of one rewrite one rewrite
// id, and an event interned on demand by another block the id its own block gives it.
//
// Runs gpu/include/hg_gpu/keyed.hpp's keyed_event_rid itself (the oldest-first walk,
// keyed_intern_event's claim on the rewrite map through keyed_claim_device, and the id's
// atomicExch into the event), with the device's map and atomics (verification/gpumc/device).
//
// THE PROTOCOL. An event's rewrite key is its rule and its consumed edges' tokens; an edge
// produced by an event has a token only once that event's rewrite id is set. The first event to
// claim a key holds it, and the rewrite id is the holder's event id + 1. A block that needs the
// token of an edge whose creator has no id interns the creator first.
//
// THE SHAPE. Initial edge 0; events 0 and 1 apply rule 0 to edge 0 (one rewrite); event 0
// produces edge 2; event 2 applies rule 0 to edge 2. Block 0 resolves event 0, block 1 event 1,
// block 2 event 2, which interns event 0 on demand.
//
// THE PROPERTY. Events 0 and 1 have one rewrite id, 1 or 2 (event 0's or event 1's id + 1);
// event 2's is 3. Each returned id is the one stored on the event. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the map's exchanges a load and a separate
// store; the checker must report two rewrite ids for one rewrite.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/keyed.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using RewriteMap = g::ConcurrentMap<uint64_t, uint32_t>;

constexpr uint32_t kSlots = 4;

g::DeviceState* g_ds;
g::Edge g_edges[3];
uint32_t g_edges_n = 3;
g::DeviceEvent g_events[3];
uint32_t g_events_n = 3;
g::EdgeId g_consumed[3] = {0, 0, 2};
g::MapSlot<uint64_t, uint32_t> g_rewrites[kSlots];
uint32_t g_keyed_words[4];
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
uint32_t g_rid[3];

void* resolver(void* arg) {
    const unsigned ev = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(ev, 0);
    g_rid[ev] = g::keyed_event_rid(*g_ds, ev);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    for (uint32_t s = 0; s < kSlots; ++s) g_rewrites[s] = {0, RewriteMap::DeviceView::UNPUBLISHED};
    for (uint32_t e = 0; e < 3; ++e) g_edges[e].creator_event = g::INVALID_ID;
    g_edges[2].creator_event = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        g_events[i].id = i;
        g_events[i].rule = 0;
        g_events[i].num_consumed = 1;
        g_events[i].consumed_at = i;
        g_events[i].rewrite_id = hgcommon::REWRITE_ID_UNSET;
    }
    g_events[0].num_produced = 1;
    g_events[0].first_produced = 2;
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->edge_pool = {g_edges, &g_edges_n, 3};
    g_ds->event_pool = {g_events, &g_events_n, 3};
    g_ds->event_consumed = g_consumed;
    g_ds->keyed.rewrites = {g_rewrites, kSlots, kSlots - 1, nullptr};
    g_ds->keyed.words = g_keyed_words;

    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, resolver, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    assert(g_rid[0] == g_rid[1] && "one rewrite received two rewrite ids");
    assert((g_rid[0] == 1u || g_rid[0] == 2u) && g_rid[2] == 3u);
    for (uint32_t i = 0; i < 3; ++i) assert(g_events[i].rewrite_id == g_rid[i]);
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
