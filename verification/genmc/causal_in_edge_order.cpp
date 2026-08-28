// GENMC-LINK: engine
// GENMC-DEFINES: -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=2
//
// GenMC harness: an event's in-edges are recorded in the one order the online transitive
// reduction is exact under, while the producer map grows underneath the registration.
//
// THE DEFECT THIS PINS. The reduction decides that a pair (p, c) is redundant by searching for a
// path p ->..-> c at the moment the pair is offered, and never revisits the answer. That is exact
// only if c's in-edges arrive from c's own thread in DESCENDING producer id, so the closer
// producer's edge is present when the farther producer's edge is judged. Rewriter::apply used to
// decide that order from one read of edge_producers_ (get_edge_producer) and emit the edges from
// a SECOND read (add_edge_consumer). The map grows concurrently, and when the first read answered
// "no producer" (sorted last) and the second found one, that edge was emitted after everything
// placed before it -- and a redundant pair was kept for good. Measured on the real engine: one
// extra causal edge per firing, three firings in 155,520 runs, gone after the fix.
//
// consume_edges is the fix: one rendezvous pass reads the producers, the order is decided over
// what that pass read, and the edges are emitted from the same set. This harness runs THAT body
// -- CausalGraph::consume_edges over the real ConcurrentMap and LockFreeList -- with the map's
// working capacity bounded to 2 so that a second thread inserting three unrelated producers
// forces two growths (2 -> 4 -> 8) while the consumer's registration is in flight. Every
// interleaving of the consumer's read with a growth's install, seal, carry and drain is explored.
//
// THE SHAPE. Event 0 produces e0. Event 1 consumes e0 and produces e1. Event 2 consumes e1 and
// produces e2. That is the chain 0 -> 1 -> 2, registered before the threads start, and each
// event's produced edge is registered before any event consumes it, which is the invariant the
// rewriter keeps. Then, concurrently:
//
//   consumer   event 3 consumes e0 and e2, so its in-edges are 0 -> 3 and 2 -> 3, and 0 -> 3 is
//              redundant through 0 -> 1 -> 2 -> 3. Exact only if 2 -> 3 is recorded first.
//   grower     three producers on three fresh edges, which is what makes the map grow.
//
// THE PROPERTY. The kept relation contains 2 -> 3 and does not contain 0 -> 3. Both halves
// matter: dropping 2 -> 3 as well would satisfy "0 -> 3 absent" and lose an edge.
//
// CALIBRATED. HG_CALIBRATE_IN_EDGE_ORDER_ASCENDING reverses the sort in consume_edges, so the
// farther producer's edge is judged before the closer one's is present, and the checker must
// report 0 -> 3 kept. The defect is reinstated in the shared body, in the real path.
#include "hypergraph/causal_graph.hpp"
#include "hypergraph/arena.hpp"

#include <cassert>
#include <pthread.h>

namespace {

using namespace hg::engine;

ConcurrentHeterogeneousArena* g_arena;
CausalGraph* g_cg;

constexpr EdgeId e0 = 10, e1 = 11, e2 = 12;

void* consumer(void*) {
    const CanonicalEdgeKey keys[2] = {CanonicalEdgeKey{e0}, CanonicalEdgeKey{e2}};
    const EdgeId raws[2] = {e0, e2};
    g_cg->consume_edges(keys, raws, 2, /*consumer=*/3);
    return nullptr;
}

void* grower(void*) {
    // Fresh keys, fresh producers: nothing here is an in-edge of event 3. What they do is push
    // edge_producers_ past its working capacity, twice.
    g_cg->set_edge_producer(CanonicalEdgeKey{20}, 7, 20);
    g_cg->set_edge_producer(CanonicalEdgeKey{21}, 8, 21);
    g_cg->set_edge_producer(CanonicalEdgeKey{22}, 9, 22);
    return nullptr;
}

}  // namespace

int main() {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);
    g_arena = &arena;
    g_cg = &cg;
    cg.set_transitive_reduction(true);
    cg.set_ids_are_topological(true);

    // The chain 0 -> 1 -> 2, each produced edge registered before it is consumed.
    cg.set_edge_producer(CanonicalEdgeKey{e0}, 0, e0);
    cg.set_edge_producer(CanonicalEdgeKey{e1}, 1, e1);
    { const CanonicalEdgeKey k[1] = {CanonicalEdgeKey{e0}}; const EdgeId r[1] = {e0};
      cg.consume_edges(k, r, 1, 1); }
    cg.set_edge_producer(CanonicalEdgeKey{e2}, 2, e2);
    { const CanonicalEdgeKey k[1] = {CanonicalEdgeKey{e1}}; const EdgeId r[1] = {e1};
      cg.consume_edges(k, r, 1, 2); }

    pthread_t tc, tg;
    pthread_create(&tc, nullptr, consumer, nullptr);
    pthread_create(&tg, nullptr, grower, nullptr);
    pthread_join(tc, nullptr);
    pthread_join(tg, nullptr);

    bool has_2_3 = false, has_0_3 = false;
    cg.for_each_causal_edge([&](const CausalEdge& e) {
        if (e.consumer == 3 && e.producer == 2) has_2_3 = true;
        if (e.consumer == 3 && e.producer == 0) has_0_3 = true;
    });
    assert(has_2_3 && "the direct in-edge 2 -> 3 was lost");
    assert(!has_0_3 && "0 -> 3 was kept although 0 -> 1 -> 2 -> 3 implies it");
    return 0;
}
