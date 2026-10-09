// GENMC-LINK: engine
// GENMC-ARGS: --disable-estimation --sc --bound=1 --bound-type=context
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_JOB_QUEUE_CAPACITY=16 -DHG_JOB_INJECTOR_CAPACITY=64 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_ARENA_BLOCK_SIZE=512 -DHG_HARNESS_DEFER_QUOTIENT_CAPTURE=1
// GENMC-CALIBRATE: -DHG_HARNESS_CALIBRATE_END
// GENMC-CALIBRATE: -DHG_CALIBRATE_LIST_NO_RETRY
//
// GenMC harness: the CAPTURE half of quotient_capture_composition. Main applies two rewrites of
// one parent through the real Rewriter::apply with the class-frame capture deferred
// (HG_HARNESS_DEFER_QUOTIENT_CAPTURE), so both children, their orbit tables and both events
// exist before any thread runs. Then two threads capture one event each
// (Hypergraph::register_quotient_transition): each reads the orbit tables of its event's two
// endpoints and records its match in the parent class's frame, which both share. The
// registration half is quotient_capture_register.
//
// THE PROPERTY. Both matches are captured: captured_matches() is 2, and the parent class's match
// list holds both. A capture that drops its match shows in the first, a push onto the list that
// is lost in the second.
//
// THE SEED. quotient_causal_seed(root, 1): the root's instance is expanded, so each capture
// applies its match to it (qr_apply: ids, content, relations) and descends to an instance at
// depth 1, which is at the bound and is recorded without expansion.
//
// WHAT IS BOUNDED. Sequential consistency with 1 context switch (`--sc --bound=1`): 4,392
// executions in 1,509 s. The RC11 estimate is 2^87 executions, 32% of its choices inside
// qr_apply. The RC11 behaviour of the rendezvous the captures meet in is
// quotient_instance_match_rendezvous's.
//
// CALIBRATED. -DHG_CALIBRATE_LIST_NO_RETRY publishes a LockFreeList push without retrying a
// failed exchange, so the two captures' pushes onto the class's match list can lose one.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/pattern.hpp"

#include <cassert>
#include <pthread.h>

namespace {
using namespace hg::engine;

Hypergraph* g_hg;
EventId g_event[2];

void* capture(void* arg) {
    const long which = reinterpret_cast<long>(arg);
    g_hg->register_quotient_transition(g_event[which]);
    return nullptr;
}
}  // namespace

int main() {
    Hypergraph hg;
    g_hg = &hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_quotient_causal(true);
    hg.set_quotient_reconstruction(true);
    const RewriteRule rule = make_rule(0).lhs({0, 1}).rhs({1, 2}).build();

    const VertexId v0 = hg.alloc_vertex(), v1 = hg.alloc_vertex(), v2 = hg.alloc_vertex();
    const EdgeId e0 = hg.create_edge({v0, v1});
    const EdgeId e1 = hg.create_edge({v1, v2});
    SparseBitset init; init.set(e0, hg.arena()); init.set(e1, hg.arena());
    auto r = hg.create_or_get_canonical_state(std::move(init), 0, INVALID_ID, INVALID_ID, nullptr, 0, nullptr, 0);
    const StateId parent = r.created_state_id;
    hg.quotient_causal_seed(r.canonical_state_id, 1);

    Rewriter rw(&hg);
    {
        VariableBinding b; b.bind(0, v0); b.bind(1, v1);
        EdgeId m[1] = {e0};
        g_event[0] = rw.apply(rule, parent, m, 1, b, 1).event;
    }
    {
        VariableBinding b; b.bind(0, v1); b.bind(1, v2);
        EdgeId m[1] = {e1};
        g_event[1] = rw.apply(rule, parent, m, 1, b, 1).event;
    }
    assert(hg.captured_matches() == 0);

    pthread_t a, b;
    pthread_create(&a, nullptr, capture, reinterpret_cast<void*>(0L));
    pthread_create(&b, nullptr, capture, reinterpret_cast<void*>(1L));
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);

    assert(hg.captured_matches() == 2 && "a match of the expanded representative was not captured");
    // captured_matches() counts the ids the captures took; the class's match list is what a
    // later instance of the class scans, so both matches must be in it.
    uint32_t listed = 0;
    hg.for_each_expansion_match(hg.get_state(parent).canonical_hash, [&](const SlotMatch&) { ++listed; });
    assert(listed == 2 && "a captured match is missing from its class's match list");
#if defined(HG_HARNESS_CALIBRATE_END)
    assert(!"the end of the harness is reachable under this bound");
#endif
    return 0;
}
