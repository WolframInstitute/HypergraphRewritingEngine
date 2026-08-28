// GENMC-LINK: engine
// GENMC-ARGS: --unroll=2
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16
//
// GenMC harness: TWO REWRITES OF ONE PARENT UNDER QUOTIENT RECONSTRUCTION, through the real
// Rewriter::apply. Each thread runs create_or_get_canonical_state (which fills the edge-orbit
// cache for its child), create_event, and register_quotient_transition (which reads the cache
// for BOTH endpoints and captures the match into the class frame). The orbit table of a state
// is a cache: a reader that misses rebuilds it from the state's own edges
// (qc_orbits_or_build), so a capture never depends on which thread filled the cache first.
//
// THE PROPERTY, stated on the counters the gate reads: no capture is dropped for want of an
// orbit table, both matches are captured, and the two events exist. The rebuild counter is NOT
// asserted -- a miss is a schedule fact and rebuilding is the correct response to it.
//
// WHAT IS BOUNDED. Two rewrites on one parent at depth 1; every loop unrolled twice, which ends
// a thread that exceeds it as blocked, never as an error.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/pattern.hpp"

#include <cassert>
#include <pthread.h>

namespace {
using namespace hg::engine;

Hypergraph* g_hg;
StateId g_parent;
EdgeId g_e0, g_e1;
VertexId g_v0, g_v1, g_v2;
RewriteRule g_rule;

void* rewrite(void* arg) {
    const long which = reinterpret_cast<long>(arg);
    Rewriter rw(g_hg);
    VariableBinding b;
    // Two distinct matches of {x,y} on the parent's two edges.
    if (which == 0) { b.bind(0, g_v0); b.bind(1, g_v1); EdgeId m[1] = {g_e0}; (void)rw.apply(g_rule, g_parent, m, 1, b, 1); }
    else            { b.bind(0, g_v1); b.bind(1, g_v2); EdgeId m[1] = {g_e1}; (void)rw.apply(g_rule, g_parent, m, 1, b, 1); }
    return nullptr;
}
}  // namespace

int main() {
    Hypergraph hg;
    g_hg = &hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_quotient_causal(true);
    hg.set_quotient_reconstruction(true);
    g_rule = make_rule(0).lhs({0, 1}).rhs({1, 2}).build();

    g_v0 = hg.alloc_vertex(); g_v1 = hg.alloc_vertex(); g_v2 = hg.alloc_vertex();
    g_e0 = hg.create_edge({g_v0, g_v1});
    g_e1 = hg.create_edge({g_v1, g_v2});
    SparseBitset init; init.set(g_e0, hg.arena()); init.set(g_e1, hg.arena());
    auto r = hg.create_or_get_canonical_state(std::move(init), 0, INVALID_ID, INVALID_ID, nullptr, 0, nullptr, 0);
    g_parent = r.created_state_id;
    hg.quotient_causal_seed(r.canonical_state_id, 2);

    pthread_t a, b;
    pthread_create(&a, nullptr, rewrite, reinterpret_cast<void*>(0L));
    pthread_create(&b, nullptr, rewrite, reinterpret_cast<void*>(1L));
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);

    assert(hg.capture_dropped_no_orbits() == 0 && "a capture was dropped for want of an orbit table");
    assert(hg.captured_matches() == 2 && "a match of the expanded representative was not captured");
    assert(hg.num_events() == 2);
    return 0;
}
