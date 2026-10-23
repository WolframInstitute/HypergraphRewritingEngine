// GPUMC harness: two keyed-rewrite children with one token set agree on their class: one claims
// the twin key and canonicalises, and the other takes its hash, class and edge ranks or waits on
// it and is handed on.
//
// Runs gpu/include/hg_gpu/keyed.hpp's keyed_take_twin (the token sums, the twin claim through
// keyed_claim_device, keyed_hash_or_follow, the rank copy) and keyed_close_followers, and
// gpu/include/hg_gpu/exploration.hpp's state_claim_form and publish_canonical_hash, with the
// device's maps, ring, pools and fences (verification/gpumc/device).
//
// THE PROTOCOL. A child made by a rewrite that may repeat stores its token sum, fences, and claims
// the sum in the twin map. The claimant of a fresh key canonicalises the state: its IR writes the
// edge ranks, it claims its canonical form (state_claim_form), publishes the class key as its
// canonical hash, and closes its follower stack (gpu/src/persistent.cu canonicalise_child). A
// child that finds a twin reads the twin's hash; when it is published the child fences, finds
// the class in the state map and copies the twin's ranks by token; when it is not, the child
// waits on the twin's stack (when its state has more than kFollowEdges edges, `may_follow`) or
// canonicalises itself.
//
// THE SHAPE. Parent state 0 holds initial edge 0. Events 0 and 1 apply rewrite 1 to it and make
// states 1 and 2, holding edges 1 and 2: one token set. Blocks 0 and 1 run the twin check of
// states 1 and 2 and, when they claim or may not wait, canonicalise. -DSHAPE_NO_FOLLOW runs them
// without `may_follow`. -DSHAPE_DUPLICATE puts the form's class in the state map before they
// start (state 9's), so a canonicalising block finds the class rather than creating it.
//
// THE PROPERTY. Every block that canonicalised published the class key, and the class is state
// 9's under SHAPE_DUPLICATE and otherwise the first claimant's. A block that took its twin holds
// that key and class and the twin's rank 7. A block that waits was handed to `ready` exactly
// once. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_NO_THREADFENCE removes the fences; under SHAPE_DUPLICATE the
// twin's hash is a relaxed store after a found class, with no release between the rank write and
// it, and the checker must report the taken rank racing the twin's write. (The follower stack's
// calibration is keyed_follow_rendezvous's; here it gives no verdict in 300 s.)
// GPUMC-CELL: follow PASS
// GPUMC-CELL: no_follow PASS -DSHAPE_NO_FOLLOW
// GPUMC-CELL: duplicate PASS -DSHAPE_DUPLICATE
// GPUMC-CELL: duplicate_no_threadfence FAIL -DSHAPE_DUPLICATE -DSHAPE_NO_FOLLOW -DHG_GPUMC_CALIBRATE_NO_THREADFENCE

#include "hg_gpumc.hpp"
#include "hg_gpu/keyed.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Map = g::ConcurrentMap<uint64_t, uint32_t>;

constexpr uint32_t kSlots = 4;
constexpr uint32_t kRewrite = 1;
constexpr uint64_t kFormKey = 0x5150;
constexpr uint32_t kRank = 7;
const uint32_t kForm[3] = {1, 2, 3};
#if defined(SHAPE_NO_FOLLOW)
constexpr bool kMayFollow = false;
#else
constexpr bool kMayFollow = true;
#endif

g::DeviceState* g_ds;
g::Edge g_edges[3];
uint32_t g_edges_n = 3;
g::DeviceEvent g_events[2];
uint32_t g_events_n = 2;
g::EdgeId g_consumed[1] = {0};
g::StateEdgeSlice g_slices[3] = {{0, 1}, {1, 1}, {2, 1}};
g::EdgeId g_edge_ids[3] = {0, 1, 2};
uint32_t g_ranks[3];
uint64_t g_hash[3];
uint64_t g_token_sum[3];
uint32_t g_first_new[3];
uint32_t g_follow_head[3] = {g::FOLLOW_EMPTY, g::FOLLOW_EMPTY, g::FOLLOW_EMPTY};
uint32_t g_follow_next[2];
g::MapSlot<uint64_t, uint32_t> g_twins[kSlots], g_states[kSlots];
uint32_t g_forms[64];
uint32_t g_forms_n;
uint32_t g_arena[2][64];
uint64_t g_arena_cursor[2];
uint32_t g_keyed_words[4] = {g::KEYED_INTERNING, 0, 0, 0};
uint32_t g_ready_slots[4];
uint64_t g_ready_seq[4] = {0, 1, 2, 3};
uint64_t g_ready_head, g_ready_tail;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];

g::TwinResult g_result[2];
uint64_t g_h[2];
g::StateId g_rep[2];
bool g_canonicalised[2];

g::RingBuffer<uint32_t>::DeviceView ready() {
    return {g_ready_slots, g_ready_seq, &g_ready_head, &g_ready_tail, 4, 3};
}
Map::DeviceView states() { return {g_states, kSlots, kSlots - 1, nullptr}; }
g::Pool<uint32_t>::DeviceView forms() { return {g_forms, &g_forms_n, 64}; }

void* child(void* arg) {
    const unsigned b = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(b, 0);
    const g::StateId sid = 1 + b;
    uint32_t* slot = nullptr;
    uint64_t slot_words = 0;
    const g::DeviceArena::View arena{g_arena[b], &g_arena_cursor[b], 64};
    uint64_t h = 0;
    g::StateId rep = g::INVALID_ID;
    const g::TwinResult r = g::keyed_take_twin(*g_ds, sid, 0, b,
                                               kRewrite | hgcommon::REWRITE_TWIN_CANDIDATE, arena,
                                               slot, slot_words, /*want_ranks=*/true,
                                               /*want_orbits=*/false, states(), forms(), h, rep,
                                               kMayFollow);
    if (r == g::TwinResult::kNone) {
        // canonicalise_child without a twin: the IR's ranks, the form claim, the hash, the stack.
        g_ranks[g_slices[sid].offset] = kRank;
        const g::StateClaim c = g::state_claim_form(*g_ds, sid, kFormKey, kForm, 3, states(), forms());
        h = c.key;
        rep = c.canonical;
        g::publish_canonical_hash(*g_ds, sid, h);
        g::keyed_close_followers(*g_ds, sid, ready());
        g_canonicalised[b] = true;
    }
    g_result[b] = r;
    g_h[b] = h;
    g_rep[b] = rep;
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[2] = {1, 1};
    hg_gpumc::launch(2, lanes);
    for (uint32_t s = 0; s < kSlots; ++s)
        g_twins[s] = g_states[s] = {0, Map::DeviceView::UNPUBLISHED};
    g_edges[0].creator_event = g::INVALID_ID;
    for (uint32_t e = 0; e < 2; ++e) {
        g_edges[1 + e].creator_event = e;
        g_events[e].id = e;
        g_events[e].rule = 0;
        g_events[e].num_consumed = 1;
        g_events[e].num_produced = 1;
        g_events[e].first_produced = 1 + e;
        g_events[e].consumed_at = 0;
        g_events[e].rewrite_id = kRewrite;   // interned by keyed_after_rewrite
    }
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->edge_pool = {g_edges, &g_edges_n, 3};
    g_ds->event_pool = {g_events, &g_events_n, 2};
    g_ds->event_consumed = g_consumed;
    g_ds->state_edge_slices = g_slices;
    g_ds->state_edge_ids = g_edge_ids;
    g_ds->state_edge_rank = g_ranks;
    g_ds->state_canonical_hash = g_hash;
    g_ds->keyed.state_token_sum = g_token_sum;
    g_ds->keyed.state_first_new_edge = g_first_new;
    g_ds->keyed.follow_head = g_follow_head;
    g_ds->keyed.follow_next = g_follow_next;
    g_ds->keyed.twins = {g_twins, kSlots, kSlots - 1, nullptr};
    g_ds->keyed.words = g_keyed_words;
    g_ds->keyed.enabled = 1;
#if defined(SHAPE_DUPLICATE)
    const g::StateClaim first = g::state_claim_form(*g_ds, 9, kFormKey, kForm, 3, states(), forms());
    assert(first.canonical == 9);
#endif

    pthread_t t[2];
    for (long i = 0; i < 2; ++i) pthread_create(&t[i], nullptr, child, reinterpret_cast<void*>(i));
    for (int i = 0; i < 2; ++i) pthread_join(t[i], nullptr);

    uint32_t canonicalised = 0;
    for (uint32_t b = 0; b < 2; ++b) canonicalised += g_canonicalised[b] ? 1u : 0u;
    assert(canonicalised >= 1 && "neither child canonicalised");
    uint64_t key = 0;
    g::StateId cls = g::INVALID_ID;
    for (uint32_t b = 0; b < 2; ++b) {
        if (!g_canonicalised[b]) continue;
        if (key == 0) { key = g_h[b]; cls = g_rep[b]; }
        assert(g_h[b] == key && g_rep[b] == cls && "two canonicalisations disagree");
        assert(g_hash[1 + b] == key);
    }
#if defined(SHAPE_DUPLICATE)
    assert(cls == 9);
#endif
    for (uint32_t b = 0; b < 2; ++b) {
        uint32_t handed = 0;
        for (uint64_t i = 0; i < g_ready_tail; ++i) handed += g_ready_slots[i & 3] == b ? 1u : 0u;
        if (g_result[b] == g::TwinResult::kTaken) {
            assert(g_h[b] == key && g_rep[b] == cls && "a taken twin's key or class differs");
            assert(g_ranks[g_slices[1 + b].offset] == kRank && "a taken rank was not the twin's");
        }
        if (g_result[b] == g::TwinResult::kFollowing)
            assert(handed == 1 && "a follower was not handed to ready exactly once");
        else
            assert(handed == 0);
    }
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
