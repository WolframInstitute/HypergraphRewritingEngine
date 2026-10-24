// GPUMC harness: a match block's per-(state, rule) cap and its draw keep the lowest-ranked
// transition of a state and emit it once.
//
// Runs gpu/src/match.cu's match_state_rule itself on a block of two threads -- the counting pass
// (emit_admit's shared count and rank buffer), thread 0's cut, the barriers between the passes
// and the emitting pass; and under a draw the shared minimum rank, the survivor flag and the
// spine pass -- with the device's atomics and barriers (verification/gpumc/device).
//
// THE SHAPE. State 0 holds three unary edges; rule 0 matches one unary edge, so the state has
// three transitions with distinct ranks (hgcommon::transition_rank over transition_key_device).
// The block's two threads take the edges by stride. MatchesPerStateRule 1 (cap), or a transition
// rate of 0 (spine: no transition survives its draw, so the minimum-rank ones are emitted).
//
// THE PROPERTY. The pool holds one published record, the transition of minimum rank, computed in
// main after the join. No access races; no barrier divergence.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the count's add and the minimum a load and a
// separate store; the checker must report a lost candidate (two records, or the wrong one).
//
// --disable-cast-elimination: GPUMC's EliminateCastsPass stops with BUG at
// EliminateCastsPass.cpp:214 (castToType) on match.cu.
// GPUMC-CELL: cap PASS -- --disable-cast-elimination
// GPUMC-CELL: spine PASS -DSHAPE_SPINE -- --disable-cast-elimination
// GPUMC-CELL: cap_split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW -- --disable-cast-elimination
// GPUMC-CELL: spine_split_rmw FAIL -DSHAPE_SPINE -DHG_GPUMC_CALIBRATE_SPLIT_RMW -- --disable-cast-elimination

#include "hg_gpumc.hpp"
#include "match.cu"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

constexpr uint32_t kEdges = 3;

g::DeviceState* g_ds;
g::DeviceRule* g_rule;
g::VertexId g_verts[kEdges] = {0, 1, 2};
g::Edge g_edges[kEdges];
uint32_t g_edges_n = kEdges;
g::StateEdgeSlice g_slices[1] = {{0, kEdges}};
g::EdgeId g_ids[kEdges] = {0, 1, 2};
uint32_t g_ranks[kEdges] = {0, 1, 2};
uint64_t g_exact[1] = {0x7777};
g::MatchRecord g_records[kEdges];
uint32_t g_records_n;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];

void* lane(void* arg) {
    hg_gpumc::enter(0, static_cast<unsigned>(reinterpret_cast<long>(arg)));
    g::match_state_rule(*g_ds, g_rule, 0, 0, 0, {g_records, &g_records_n, kEdges}, nullptr,
                        nullptr);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[1] = {2};
    hg_gpumc::launch(1, lanes);
    for (uint32_t e = 0; e < kEdges; ++e) {
        g_edges[e].arity = 1;
        g_edges[e].vertex_offset = e;
        g_edges[e].creator_event = g::INVALID_ID;
    }
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_rule = new (malloc(sizeof(g::DeviceRule))) g::DeviceRule{};
    g_rule->lhs[0].arity = 1;
    g_rule->num_lhs_edges = 1;
    g_rule->num_lhs_vars = 1;
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->vertex_pool = {g_verts, nullptr, kEdges};
    g_ds->edge_pool = {g_edges, &g_edges_n, kEdges};
    g_ds->state_edge_slices = g_slices;
    g_ds->state_edge_ids = g_ids;
    g_ds->state_edge_rank = g_ranks;
    g_ds->state_exact_hash = g_exact;
    g_ds->max_states = 1;
    g_ds->num_rules = 1;
    g_ds->slice_scan_max_edges = 64;
    g_ds->sampling_seed = 5;
#if defined(SHAPE_SPINE)
    g_ds->transition_rate = 0.0;
#else
    g_ds->transition_rate = 1.0;
    g_ds->matches_per_state_rule = 1;
#endif

    pthread_t t[2];
    for (long i = 0; i < 2; ++i) pthread_create(&t[i], nullptr, lane, reinterpret_cast<void*>(i));
    for (int i = 0; i < 2; ++i) pthread_join(t[i], nullptr);

    g::EdgeId best = g::INVALID_ID;
    uint64_t best_rank = ~uint64_t{0};
    for (g::EdgeId e = 0; e < kEdges; ++e) {
        g::EdgeId edges[g::kMaxPatternEdges];
        for (uint32_t i = 0; i < g::kMaxPatternEdges; ++i) edges[i] = g::INVALID_ID;
        edges[0] = e;
        const uint64_t r = hgcommon::transition_rank(
            g::transition_key_device(*g_ds, 0, 0, edges, 1), g_ds->sampling_seed, 0u);
        if (r < best_rank) { best_rank = r; best = e; }
    }
    assert(g_records_n == 1 && "the block did not emit exactly one transition");
    assert(g_records[0].published == 1u && g_records[0].matched_edges[0] == best &&
           "the emitted transition is not the lowest-ranked one");
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
