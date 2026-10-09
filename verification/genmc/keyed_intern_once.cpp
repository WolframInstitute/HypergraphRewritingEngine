// GENMC-LINK: engine
// GENMC-ARGS: --disable-estimation --sc --bound=4 --bound-type=context
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_JOB_QUEUE_CAPACITY=16 -DHG_JOB_INJECTOR_CAPACITY=64 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_ARENA_BLOCK_SIZE=512
// GENMC-CALIBRATE: -DHG_CALIBRATE_MAP_LOSER_KEEPS_VALUE
//
// GenMC harness: KEYED REWRITES (protocol P27), the rewrite intern. Main switches the run to
// INTERNING (note_inherited_rewrite: the token cache installed, then ARMED -> INTERNING). Two
// threads then apply the same rewrite, doing what Rewriter::apply does first under keyed
// rewrites: intern_rewrite (the consumed edge's token read through the cache and filled on a
// miss, the rewrite key claimed on rewrite_map_, the id taken from next_rewrite_id_ by the
// claim's maker), then edge_token on the consumed edge.
//
// THE PROPERTY. Both threads get the same rewrite id, exactly one of them is told the rewrite is
// new, the run is INTERNING, and the consumed edge's token is the initial token.
//
// WHAT IS BOUNDED. Sequential consistency with at most 4 context switches (`--sc --bound=4`):
// 17,319 executions in 419 s. The RC11 exploration gave no verdict in 2,770 s. The map's RC11
// behaviour under a racing claim is the ConcurrentMap harnesses' (concurrent_map_agreement and
// the growth harnesses).
//
// CALIBRATED. -DHG_CALIBRATE_MAP_LOSER_KEEPS_VALUE makes the map's losing claimant answer as the
// inserter with its own record, so both threads are told the rewrite is new.
#include "hypergraph/hypergraph.hpp"

#include <cassert>
#include <pthread.h>

namespace {
using namespace hg::engine;

Hypergraph* g_hg;
EdgeId g_e0;
uint32_t g_rid[2];
bool g_repeated[2];
uint64_t g_token[2];

void* intern(void* arg) {
    const long which = reinterpret_cast<long>(arg);
    bool repeated = false;
    g_rid[which] = g_hg->intern_rewrite(0, &g_e0, 1, repeated);
    g_repeated[which] = repeated;
    g_token[which] = g_hg->edge_token(g_e0);
    return nullptr;
}
}  // namespace

int main() {
    Hypergraph hg;
    g_hg = &hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_keyed_rewrites(true);
    const VertexId v0 = hg.alloc_vertex(), v1 = hg.alloc_vertex();
    g_e0 = hg.create_edge({v0, v1});
    assert(hg.keyed_state() == Hypergraph::KEYED_ARMED);
    hg.note_inherited_rewrite();
    // The token cache segment that holds e0 is created here, through another edge in it: the
    // threads race on e0's slot, not on the segment install (segmented_array_published_read).
    const EdgeId other = hg.create_edge({v1, v0});
    (void)hg.edge_token(other);

    pthread_t a, b;
    pthread_create(&a, nullptr, intern, reinterpret_cast<void*>(0L));
    pthread_create(&b, nullptr, intern, reinterpret_cast<void*>(1L));
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);

    assert(hg.keyed_state() == Hypergraph::KEYED_INTERNING);
    assert(g_rid[0] != hgcommon::REWRITE_ID_NONE && g_rid[0] == g_rid[1] &&
           "the two applications got different rewrite ids");
    assert(g_repeated[0] != g_repeated[1] && "not exactly one application was told it is new");
    assert(g_token[0] == hgcommon::token_initial(g_e0) && g_token[1] == g_token[0]);
    return 0;
}
