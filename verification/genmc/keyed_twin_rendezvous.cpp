// GENMC-LINK: engine
// GENMC-ARGS: --disable-estimation --sc --bound=1 --bound-type=context
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_JOB_QUEUE_CAPACITY=16 -DHG_JOB_INJECTOR_CAPACITY=64 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_ARENA_BLOCK_SIZE=512
// GENMC-CALIBRATE: -DHG_HARNESS_CALIBRATE_END
// GENMC-CALIBRATE: -DHG_CALIBRATE_TWIN_TAKE_UNPUBLISHED
//
// GenMC harness: KEYED REWRITES (protocol P27), the twin rendezvous under Full canonicalisation.
// Two threads create two children with the same token set through the real
// Hypergraph::create_or_get_canonical_state, each as a twin candidate: each computes its token
// sum (child_token_sum: edge_token through the token cache), claims the twin map (claim_twin,
// offer first, decided by same_tokens), and the child that finds the other takes its class key
// (take_twin) when the other has published it, or runs its own IR and class claim when not.
//
// THE SHAPE. Main does what Rewriter::apply does before create_or_get_canonical_state, for two
// applications of one rewrite (rule 0 on edge e0 of the parent {e0, e1}): it switches the run to
// INTERNING (note_inherited_rewrite, which installs the token cache), interns the rewrite
// (intern_rewrite), creates each application's produced edge p0, p1, and caches their tokens
// (cache_edge_token: both are token_produced(rid, 0)). The children are {e1, p0} and {e1, p1}.
// The intern race itself is keyed_intern_once.
//
// THE PROPERTY. Three raw states in two classes; both children carry the class's nonzero key.
//
// WHAT IS BOUNDED. Sequential consistency with 1 context switch (`--sc --bound=1`): 1,265
// executions in 231 s. The RC11 estimate is 2^51 executions, and 2 context switches gave no
// verdict in 1,500 s (11,000 executions).
//
// CALIBRATED. -DHG_CALIBRATE_TWIN_TAKE_UNPUBLISHED takes a twin's results without checking that
// it published them, so a child that finds its twin before the twin's key is stored records 0.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/pattern.hpp"

#include <cassert>
#include <pthread.h>

namespace {
using namespace hg::engine;

Hypergraph* g_hg;
StateId g_parent;
EdgeId g_e0, g_e1, g_p[2];
uint32_t g_keyed;
StateId g_child[2];

void* make_child(void* arg) {
    const long which = reinterpret_cast<long>(arg);
    SparseBitset edges;
    edges.set(g_e1, g_hg->arena());
    edges.set(g_p[which], g_hg->arena());
    const EdgeId consumed[1] = {g_e0};
    const EdgeId produced[1] = {g_p[which]};
    g_child[which] = g_hg->create_or_get_canonical_state(std::move(edges), 1, INVALID_ID, g_parent,
                                                         consumed, 1, produced, 1, g_keyed)
                         .created_state_id;
    return nullptr;
}
}  // namespace

int main() {
    Hypergraph hg;
    g_hg = &hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_keyed_rewrites(true);

    const VertexId v0 = hg.alloc_vertex(), v1 = hg.alloc_vertex(), v2 = hg.alloc_vertex();
    g_e0 = hg.create_edge({v0, v1});
    g_e1 = hg.create_edge({v1, v2});
    SparseBitset init; init.set(g_e0, hg.arena()); init.set(g_e1, hg.arena());
    g_parent = hg.create_or_get_canonical_state(std::move(init), 0, INVALID_ID, INVALID_ID, nullptr,
                                                0, nullptr, 0).created_state_id;
    assert(hg.keyed_state() == Hypergraph::KEYED_ARMED);

    // Rewriter::apply's keyed prefix, once per application.
    hg.note_inherited_rewrite();
    bool repeated = false;
    const uint32_t rid = hg.intern_rewrite(0, &g_e0, 1, repeated);
    assert(rid != hgcommon::REWRITE_ID_NONE && !repeated);
    g_keyed = rid | hgcommon::REWRITE_TWIN_CANDIDATE;
    for (int i = 0; i < 2; ++i) {
        const EventId ev = hg.reserve_event_id();
        const VertexId z = hg.alloc_vertex();
        g_p[i] = hg.create_edge({v1, z}, ev, 1);
        hg.cache_edge_token(g_p[i], hgcommon::token_produced(rid, 0));
    }

    pthread_t a, b;
    pthread_create(&a, nullptr, make_child, reinterpret_cast<void*>(0L));
    pthread_create(&b, nullptr, make_child, reinterpret_cast<void*>(1L));
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);

    assert(hg.num_states() == 3);
    assert(hg.num_canonical_states() == 2 && "the twins are not one class");
    const uint64_t k0 = hg.get_state(g_child[0]).canonical_hash;
    const uint64_t k1 = hg.get_state(g_child[1]).canonical_hash;
    assert(k0 != 0 && k1 != 0 && "a child has no class key");
    assert(k0 == k1 && "the twins carry different class keys");
#if defined(HG_HARNESS_CALIBRATE_END)
    assert(!"the end of the harness is reachable under this bound");
#endif
    return 0;
}
