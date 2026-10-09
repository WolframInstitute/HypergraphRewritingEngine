// GENMC-LINK: engine
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_ARENA_BLOCK_SIZE=512
// GENMC-ARGS: --disable-estimation
// GENMC-CALIBRATE: -DHG_CALIBRATE_BRANCHIAL_WALK_ALL
// GENMC-CALIBRATE: -DHG_CALIBRATE_BRANCHIAL_EVERY_BUCKET
//
// GenMC harness: events that consumed the same two edges of one input state are recorded as
// branchial pairs ONCE each, carrying the lower shared edge id.
//
// THE PROTOCOL. CausalGraph::record_branchial_overlaps keeps no set of recorded pairs. For each
// consumed edge an event pushes itself into the (state, edge) bucket and walks only the entries
// pushed before its own (LockFreeList::for_each_before), so in one bucket only the event pushed
// second reports the pair. A pair sharing several edges is reported only from the bucket of the
// lowest shared edge: an event that consumed an edge below the bucket's reads the other event's
// consumed edges and skips the pair when they share a lower one.
//
// THE SHAPE. Event 3 at state 0, consuming edges 5 and 7, is recorded by main first: it creates
// the two buckets, so the threads race on the bucket lists and not on the bucket map's inserts
// (the ConcurrentMap harnesses' subject). Then events 1 and 2, at state 0 and consuming 5 and 7,
// are recorded by two threads at once. Every interleaving of their pushes and walks in both
// buckets is explored.
//
// THE PROPERTY. Exactly three branchial edges, (1, 2), (1, 3) and (2, 3), each with shared edge 5.
//
// CALIBRATED. HG_CALIBRATE_BRANCHIAL_WALK_ALL walks the whole bucket, so an execution where both
// threads see each other reports (1, 2) twice. HG_CALIBRATE_BRANCHIAL_EVERY_BUCKET drops the
// lower-edge test, so buckets 5 and 7 both report every pair.
#include "hypergraph/causal_graph.hpp"
#include "hypergraph/arena.hpp"

#include <cassert>
#include <pthread.h>

namespace {

using namespace hg::engine;

CausalGraph* g_cg;

const EdgeId kConsumed[2] = {5, 7};

const EdgeId* consumed_of(const void*, EventId, uint8_t* n) {
    *n = 2;
    return kConsumed;
}

void* record(void* arg) {
    const EventId ev = static_cast<EventId>(reinterpret_cast<uintptr_t>(arg));
    g_cg->record_branchial_overlaps(ev, /*input_state=*/0, kConsumed, 2, consumed_of, nullptr);
    return nullptr;
}

}  // namespace

int main() {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);
    g_cg = &cg;
    cg.record_branchial_overlaps(3, /*input_state=*/0, kConsumed, 2, consumed_of, nullptr);

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, record, reinterpret_cast<void*>(uintptr_t{1}));
    pthread_create(&t2, nullptr, record, reinterpret_cast<void*>(uintptr_t{2}));
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    uint32_t n = 0, p12 = 0, p13 = 0, p23 = 0;
    bool on5 = true;
    cg.for_each_branchial_edge([&](const BranchialEdge& e) {
        ++n;
        if (e.event1 == 1 && e.event2 == 2) ++p12;
        if (e.event1 == 1 && e.event2 == 3) ++p13;
        if (e.event1 == 2 && e.event2 == 3) ++p23;
        if (e.shared_edge != 5) on5 = false;
    });
    assert(p12 == 1 && "the pair (1, 2) was not recorded exactly once");
    assert(p13 == 1 && p23 == 1 && "a pair with event 3 was not recorded exactly once");
    assert(n == 3 && "an extra branchial edge was recorded");
    assert(on5 && "a pair does not carry the lowest shared edge");
    return 0;
}
