// GPUMC harness: a state admitted for expansion by an isomorphic duplicate has its class key
// published before any capture of its matches reads it, so every captured match is recorded
// under the class key.
//
// Runs gpu/include/hg_gpu/exploration.hpp's state_claim_form, gpu/src/persistent.cu's
// publish_class_hash, publish_class_exact_hash and register_child (explore_register_child, then
// DeviceExploreCtx::admit and explore_relax), gpu/include/hg_gpu/work_log.hpp's
// WorkLogView::claim and await on the expand log, and gpu/include/hg_gpu/quotient_expansion.hpp's
// qe_capture_expansion itself, with the device's maps, pools, lists, atomics, fences and warp
// collectives (verification/gpumc/device).
//
// COMPOSED IN THE HARNESS. canonicalise_child (persistent.cu) runs the IR on a warp before its
// claim (D15, which no harness runs), so `identify_and_register` below runs the part of its
// Full-mode leader branch that follows the IR, with the IR's result (the form and first key)
// given: state_claim_form, publish_class_hash and publish_class_exact_hash (the state's keys,
// and the same keys for the class's first state when the state is a duplicate), then
// register_child as k_persistent_evolve calls it. The capturing
// thread takes the expand entry and calls qe_capture_expansion directly: the engine reaches the
// capture through the match ring, a match record and the child's canonicalise_child, each a
// release/acquire hand-off, and the harness keeps only the first (the expand log). The child's
// own hash is published by the capturing thread first, as the child's canonicalise_child does
// on the same warp before the capture.
//
// THE PROTOCOL. A class's first state C claims its class (the map's value exchange) and then
// publishes its key to state_canonical_hash[C]. An isomorphic duplicate D can find C's claim
// before that store, and register_child on D's block can lower C's depth and append C to the
// expand log (release on the entry's published word). A worker that awaits the entry (acquire)
// and captures one of C's matches loads state_canonical_hash[C] as the class key; D stores the
// same key for C before its admit, so the load is ordered after a store of the key.
//
// THE SHAPE. Three blocks of one lane. State 0 is the parent P at depth 0, claimed. Blocks 0 and
// 1 identify states 1 (C) and 2 (D), children of P at step 1 with one IR form and one first key.
// Block 2 claims one expand entry; when it holds state 1, it captures event 0 of state 1 (one
// consumed edge, one produced edge) into state 3 (K), whose class key is kChildKey. The budget
// is 2 steps. The executions where D's claim wins make D the class's first state and C the
// duplicate; block 2 then holds state 2 and does not capture (the same protocol with the roles
// exchanged). Replay is off: the record and its by_from node are the capture's output
// (replay_rendezvous_device runs the match side of the rendezvous from a recorded match).
//   -DSHAPE_COIN   ExplorationProbability is 0.999999, so the admit draws the coin
//                  (explore_admits) on state_canonical_hash of the admitted state. A refused
//                  class has no expand entry.
//
// THE PROPERTY. No error is recorded (kUncomputedStateHash in particular), and when block 2
// captured, the match pool holds one record whose from_hash is C's claimed key (nonzero), linked
// in by_from under qe_expansion_key of that key at depth 1, and state_canonical_hash and
// state_exact_hash of C hold the key. No access races.
//
// CALIBRATION. -DCALIBRATE_NO_DUPLICATE_PUBLISH leaves out the duplicate-side publishes; the
// checker must report the capture refused with kUncomputedStateHash (D admits C, and the capture
// loads state_canonical_hash[C] before C's own store).
// GPUMC-CELL: clean PASS -- --disable-function-inliner
// GPUMC-CELL: coin PASS -DSHAPE_COIN -- --disable-function-inliner
// GPUMC-CELL: no_duplicate_publish FAIL -DCALIBRATE_NO_DUPLICATE_PUBLISH -- --disable-function-inliner

#include "hg_gpumc.hpp"
#include "persistent.cu"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Map = g::DedupMap;
using ChildNode = g::LockFreeList<g::StateId>::Node;
using MatchNode = g::LockFreeList<g::QeMatchRef>::Node;

constexpr g::StateId kP = 0, kC = 1, kD = 2, kK = 3;
constexpr uint32_t kStates = 4;
constexpr uint32_t kBudget = 2;
constexpr uint32_t kStep = 1;   // C and D are created at step 1; C's depth is 1
constexpr uint64_t kFirstKey = 0x1234;
constexpr uint64_t kChildKey = 0x5151;
constexpr uint32_t kSlots = 4;
constexpr uint32_t kLevels = 2;

const uint32_t kForm[3] = {1, 2, 3};

g::DeviceState* g_ds;
g::ExploreView* g_ev;
g::SessionView* g_sess;
g::QeView* g_qe;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];

// State identity: the Full-mode dedup map and the form records.
g::MapSlot<uint64_t, uint32_t> g_dedup_slots[kSlots];
uint32_t g_form_words[64];
uint32_t g_form_words_n;

// Per state: hashes, slices over g_edge_ids (C: edge 0, D: edge 1, K: edge 2), orbits.
uint64_t g_canonical_hash[kStates] = {0x77, 0, 0, 0};
uint64_t g_exact_hash[kStates] = {0x77, 0, 0, 0};
g::StateEdgeSlice g_slices[kStates] = {{0, 0}, {0, 1}, {1, 1}, {2, 1}};
g::EdgeId g_edge_ids[3] = {0, 1, 2};
uint32_t g_edge_orbit[3] = {0, 0, 0};
// Event 0: C's match, consuming edge 0 and producing edge 2.
g::DeviceEvent g_events[1];
uint32_t g_events_n = 1;
g::EdgeId g_event_consumed[1] = {0};

// Exploration: depths, claims, P's child list, the expand log, the walk's frames.
uint32_t g_depth[kStates] = {0, hgcommon::kExploreNoDepth, hgcommon::kExploreNoDepth,
                             hgcommon::kExploreNoDepth};
uint32_t g_claimed[kStates] = {1, 0, 0, 0};
ChildNode g_child_nodes[4];
uint32_t g_child_nodes_n;
uint32_t g_child_heads[kStates];
g::ExpandEntry g_expand[2];
uint32_t g_expand_n, g_expand_cursor, g_expand_done;
uint32_t g_frame_node[3 * kLevels], g_frame_depth[3 * kLevels];

// The capture: match pool, by_from list, class rep and frame maps, slot-word arena, counters.
g::DeviceSlotMatch g_matches[2];
uint32_t g_matches_n;
MatchNode g_match_nodes[2];
uint32_t g_match_nodes_n;
uint32_t g_match_heads[4];
g::MapSlot<uint64_t, uint32_t> g_rep_slots[kSlots];
g::MapSlot<uint64_t, uint64_t> g_frame_slots[kSlots];
uint32_t g_words[8];
uint32_t g_words_cursor;
uint32_t g_next_id;
uint32_t g_class_nmatch[kStates];
uint32_t g_align_moved, g_align_fail;
uint64_t g_surv[4];

uint64_t g_key[kStates];   // the class key each identifying block claimed
bool g_captured;

// canonicalise_child's Full-mode leader branch after the IR, then register_child.
void identify_and_register(g::StateId sid) {
    const g::DeviceState& ds = *g_ds;
    const Map::DeviceView map{g_dedup_slots, kSlots, kSlots - 1, nullptr};
    const g::StateClaim c = g::state_claim_form(ds, sid, kFirstKey & ds.canonical_key_mask, kForm,
                                                3, map, {g_form_words, &g_form_words_n, 64});
    const uint64_t h = c.key;
    const g::StateId canonical = c.canonical;
    g_key[sid] = h;
#if !defined(CALIBRATE_NO_DUPLICATE_PUBLISH)
    g::publish_class_hash(ds, sid, canonical, h);
    g::publish_class_exact_hash(ds, sid, canonical, h);
#else
    g::publish_canonical_hash(ds, sid, h);
    g::publish_exact_hash(ds, sid, h);
#endif
    g::register_child(ds, *g_ev, *g_sess, sid, kP, canonical, c.fresh, kStep, kBudget, true,
                      g::CanonicalizationMode::Full, c.canonical, nullptr);
}

void* identify(void* arg) {
    const g::StateId sid = static_cast<g::StateId>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(sid - kC, 0);
    identify_and_register(sid);
    return nullptr;
}

// A worker taking one expand entry, then the capture of the entry's match.
void* capturer(void*) {
    hg_gpumc::enter(2, 0);
    uint32_t base = 0;
    if (g_ev->expand.claim(1u, base) == 0u) return nullptr;
    const g::ExpandEntry& e = g_ev->expand.await(base);
    if (e.state != kC) return nullptr;
    g::publish_canonical_hash(*g_ds, kK, kChildKey);
    g::qe_capture_expansion(*g_ds, *g_qe, e.state, kK, 0u, 0u, e.depth, 2u, g_surv);
    g_captured = true;
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    for (uint32_t s = 0; s < kSlots; ++s) {
        g_dedup_slots[s] = {0, Map::DeviceView::UNPUBLISHED};
        g_rep_slots[s] = {0, ~uint32_t{0}};
        g_frame_slots[s] = {0, ~uint64_t{0}};
    }
    for (uint32_t s = 0; s < kStates; ++s) g_child_heads[s] = g::INVALID_ID;
    for (uint32_t k = 0; k < 4; ++k) g_match_heads[k] = g::INVALID_ID;
    g_events[0].num_consumed = 1;
    g_events[0].num_produced = 1;
    g_events[0].first_produced = 2;
    g_events[0].consumed_at = 0;
    g_events[0].input_state = kC;
    g_events[0].output_state = kK;
    g_events[0].step = kStep + 1u;

    // On the heap, zeroed first: GPUMC's static allocator places a struct of this size wrongly,
    // and DeviceState{} and QeView{} leave their padding unwritten (QeView is passed by value).
    auto* ds = static_cast<g::DeviceState*>(malloc(sizeof(g::DeviceState)));
    memset(ds, 0, sizeof(g::DeviceState));
    g_ds = new (ds) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->max_states = kStates;
    g_ds->canonical_key_mask = ~uint64_t{0};
    g_ds->event_key_mask = ~uint64_t{0};
#if defined(SHAPE_COIN)
    g_ds->exploration_probability = 0.999999;
#else
    g_ds->exploration_probability = 1.0;
#endif
    g_ds->state_canonical_hash = g_canonical_hash;
    g_ds->state_exact_hash = g_exact_hash;
    g_ds->state_edge_slices = g_slices;
    g_ds->state_edge_ids = g_edge_ids;
    g_ds->state_edge_orbit = g_edge_orbit;
    g_ds->event_pool = {g_events, &g_events_n, 1};
    g_ds->event_consumed = g_event_consumed;

    g_sess = new (malloc(sizeof(g::SessionView))) g::SessionView{};
    g_ev = new (malloc(sizeof(g::ExploreView))) g::ExploreView{};
    g_ev->depth = g_depth;
    g_ev->claimed = g_claimed;
    g_ev->children = {{g_child_nodes, &g_child_nodes_n, 4}, g_child_heads, kStates};
    g_ev->expand.items = {g_expand, &g_expand_n, 2};
    g_ev->expand.cursor = &g_expand_cursor;
    g_ev->expand.done = &g_expand_done;
    g_ev->frame_node = g_frame_node;
    g_ev->frame_depth = g_frame_depth;
    g_ev->frame_levels = kLevels;
    g_ev->max_states = kStates;

    auto* qe = static_cast<g::QeView*>(malloc(sizeof(g::QeView)));
    memset(qe, 0, sizeof(g::QeView));
    g_qe = new (qe) g::QeView{};
    g_qe->enabled = 1;
    g_qe->replay = 0;
    g_qe->max_steps = kBudget;
    g_qe->matches = {g_matches, &g_matches_n, 2};
    g_qe->by_from = {{g_match_nodes, &g_match_nodes_n, 2}, g_match_heads, 4};
    g_qe->rep = {g_rep_slots, kSlots, kSlots - 1, nullptr};
    g_qe->frame = {g_frame_slots, kSlots, kSlots - 1, nullptr};
    g_qe->arr_words = g_words;
    g_qe->arr_cursor = &g_words_cursor;
    g_qe->arr_capacity = 8;
    g_qe->next_id = &g_next_id;
    g_qe->class_nmatch = g_class_nmatch;
    g_qe->class_nmatch_cap = kStates;
    g_qe->align_moved = &g_align_moved;
    g_qe->align_fail = &g_align_fail;

    pthread_t t[3];
    pthread_create(&t[0], nullptr, identify, reinterpret_cast<void*>(static_cast<long>(kC)));
    pthread_create(&t[1], nullptr, identify, reinterpret_cast<void*>(static_cast<long>(kD)));
    pthread_create(&t[2], nullptr, capturer, nullptr);
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    assert(g_errors[static_cast<uint32_t>(g::ErrorKind::kUncomputedStateHash)] == 0u &&
           "a capture loaded the class key of its parent before the key was published");
    assert(g_key[kC] != 0 && g_key[kC] == g_key[kD] && "C and D were not merged into one class");
    assert(g_canonical_hash[kC] == g_key[kC] && g_exact_hash[kC] == g_key[kC]);
#if defined(SHAPE_COIN)
    assert(g_expand_n <= 1u && "the class was logged for expansion twice");
#else
    assert(g_expand_n == 1u && "the class was not logged for expansion exactly once");
#endif
    if (g_captured) {
        assert(g_matches_n == 1u && g_matches[0].from_hash == g_key[kC] &&
               g_matches[0].to_hash == kChildKey && "the match was not recorded under C's class");
        const uint64_t xkey = g::qe_expansion_key(*g_ds, g_key[kC], kStep);
        assert(g_match_nodes_n == 1u && g_match_nodes[0].value.from_key == xkey &&
               g_match_nodes[0].value.record == 0u &&
               "the match is not linked under C's expansion key");
    }
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
