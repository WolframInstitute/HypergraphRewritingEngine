// GENMC-LINK: engine
// GENMC-ARGS: --disable-estimation --sc --bound=1 --bound-type=context
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_JOB_QUEUE_CAPACITY=16 -DHG_JOB_INJECTOR_CAPACITY=64 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_ARENA_BLOCK_SIZE=512 -DHG_HARNESS_DEFER_QUOTIENT_CAPTURE=1
// GENMC-CALIBRATE: -DHG_HARNESS_CALIBRATE_END
// GENMC-CALIBRATE: -DHG_CALIBRATE_DEDUP_HASH_ONLY
//
// GenMC harness: the REGISTRATION half of quotient_capture_composition. Two rewrites of one
// parent under quotient reconstruction, through the real Rewriter::apply, with the class-frame
// capture deferred (HG_HARNESS_DEFER_QUOTIENT_CAPTURE): each thread allocates its edge and
// vertex, creates its child through create_or_get_canonical_state (Full canonicalisation, the
// IR hash and the edge-orbit cache filled at creation) and creates its event. The capture half
// is quotient_capture_frame.
//
// THE SHAPE. The parent is the path v0-v1-v2 (e0, e1); the rule {x,y} -> {y,z} applied to e0
// and to e1 makes two children that are not isomorphic: {v1->v2, v1->z} (two edges out of v1)
// and {v0->v1, v2->z} (two disjoint edges). The canonical key
// mask is 0, so every state's first probe key is the same key and the two children's class
// claims meet on it: the collision path of the claim runs under contention.
//
// THE PROPERTY. Two events and three raw states; three classes; each child has its canonical
// hash and its orbit table published.
//
// WHAT IS BOUNDED. Sequential consistency with 1 context switch (`--sc --bound=1`): 2,684
// executions in 831 s. The RC11 estimate is 2^61 executions, 78% of its choices inside the two
// Rewriter::apply calls.
//
// CALIBRATED. -DHG_CALIBRATE_DEDUP_HASH_ONLY treats any key hit as a duplicate, so with every
// first key equal a child joins another state's class and the class count falls below three.
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
StateId g_child[2];

void* rewrite(void* arg) {
    const long which = reinterpret_cast<long>(arg);
    Rewriter rw(g_hg);
    VariableBinding b;
    RewriteResult r;
    if (which == 0) { b.bind(0, g_v0); b.bind(1, g_v1); EdgeId m[1] = {g_e0}; r = rw.apply(g_rule, g_parent, m, 1, b, 1); }
    else            { b.bind(0, g_v1); b.bind(1, g_v2); EdgeId m[1] = {g_e1}; r = rw.apply(g_rule, g_parent, m, 1, b, 1); }
    g_child[which] = r.raw_state;
    return nullptr;
}
}  // namespace

int main() {
    Hypergraph hg;
    g_hg = &hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_quotient_causal(true);
    hg.set_quotient_reconstruction(true);
    hg.set_canonical_key_mask(0);
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

    assert(hg.num_events() == 2);
    assert(hg.num_states() == 3);
    assert(hg.num_canonical_states() == 3 && "a child joined another state's class");
    for (StateId c : g_child) {
        assert(c != INVALID_ID);
        assert(hg.get_state(c).canonical_hash != 0 && "a child's canonical hash is unpublished");
        assert(hg.state_orbits(c) != nullptr && "a child's orbit table is unpublished");
    }
#if defined(HG_HARNESS_CALIBRATE_END)
    assert(!"the end of the harness is reachable under this bound");
#endif
    return 0;
}
