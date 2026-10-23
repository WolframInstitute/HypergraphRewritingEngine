// GPUMC harness: MaxStatesPerStep's selection of a step runs once, after every unit of work that
// can produce one of the step's candidates, keeps the lowest-ranked candidate, and leaves a
// later step's candidate that is still being written to the later step's scan.
//
// Runs gpu/src/persistent.cu's step_release and run_step_selections (select_step) themselves, with
// the device's atomics, fences and barriers (verification/gpumc/device).
//
// THE PROTOCOL. ds.step_pending[s] counts the work that can still produce a candidate of step s.
// A block finishing a unit writes its candidates to the candidate pool, publishes each record, and
// releases the unit with step_release. The release that brings the count to zero runs the
// selection on its block: select_step reads the pool's counter and the published records below
// it from sel.words[1], ranks the step's candidates, books the kept ones on step s + 1, appends
// them to `found`, and records in sel.words[1] where the next step's scan starts.
//
// THE SHAPE. Three blocks of one thread. Step 1 holds two units, one each for blocks 0 and 1, which
// emit candidates from states 1 and 2. Step 2 holds the token of step 1's selection and one unit
// of block 2, which emits a candidate of step 2 from state 3 (a session entry seeded at depth 2,
// matched while step 1 is selected). MaxStatesPerStep is 1. The block that runs step 1's selection
// then applies the kept record and releases its unit on step 2, as the persistent kernel does once
// the record's child is registered; the release that brings step 2 to zero runs step 2's selection.
//
// THE PROPERTY. `found` holds two records: the lower-ranked of states 1 and 2
// (hgcommon::transition_rank over transition_key_device, computed in main after the joins) and
// state 3. Every published flag is set. Step 3's count is 1: the kept record's unit, booked, plus
// step 2's token, released. No access races.
//
// CALIBRATION. -DCALIBRATE_RELEASE_FIRST releases a step-1 unit before its candidate is written:
// the selection can run without that candidate. -DHG_GPUMC_CALIBRATE_NO_THREADFENCE makes every
// __threadfence() a no-op, which removes step_release's ordering. The checker must report each.
// GPUMC-CELL: clean PASS -- --disable-function-inliner
// GPUMC-CELL: release_first FAIL -DCALIBRATE_RELEASE_FIRST -- --disable-function-inliner
// GPUMC-CELL: no_threadfence FAIL -DHG_GPUMC_CALIBRATE_NO_THREADFENCE -- --disable-function-inliner

#include "hg_gpumc.hpp"
#include "persistent.cu"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

constexpr uint32_t kCap = 4;
constexpr uint64_t kSeed = 7;

g::DeviceState* g_ds;
g::MatchRecord g_cand[kCap];
uint32_t g_cand_n = 0;
g::MatchRecord g_found[kCap];
uint32_t g_found_n = 0;
uint32_t g_pending[4] = {0, 2, 2, 1};
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
uint64_t g_exact[4] = {0, 0x1234567u, 0x7654321u, 0x2468ace};
g::StateEdgeSlice g_slices[4];
uint64_t g_rank[kCap];
uint32_t g_idx[kCap];
uint32_t g_words[2] = {0, 0};
g::DeviceRule g_rules[1];

g::Pool<g::MatchRecord>::DeviceView cand() { return {g_cand, &g_cand_n, kCap}; }
g::Pool<g::MatchRecord>::DeviceView found() { return {g_found, &g_found_n, kCap}; }

void select_from(uint32_t step) {
    g::StepSelectScratch sel{g_rank, g_idx, g_words, kCap};
    g::run_step_selections(*g_ds, step, cand(), found(), sel, g_rules, nullptr);
}

// One unit of `step`: a candidate from `state`, then the release.
void unit(uint32_t step, uint32_t state) {
    const g::DeviceState& ds = *g_ds;
#if defined(CALIBRATE_RELEASE_FIRST)
    const bool last = step == 1 && g::step_release(ds, step, 1u);
#endif
    const uint32_t k = cand().claim();
    g::MatchRecord& r = cand().at(k);
    r.rule_id = 0;
    r.state_id = state;
    r.step = step;
    r.num_edges = 0;
    g::publish_match(r);
#if defined(CALIBRATE_RELEASE_FIRST)
    const bool released = step != 1 ? g::step_release(ds, step, 1u) : last;
#else
    const bool released = g::step_release(ds, step, 1u);
#endif
    if (!released) return;
    select_from(step);
    // Step 1's kept record, applied: its unit on step 2 goes.
    if (step == 1 && g::step_release(ds, 2u, 1u)) select_from(2u);
}

void* block(void* arg) {
    const unsigned b = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(b, 0);
    if (b < 2) unit(1u, b + 1u);
    else unit(2u, 3u);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->max_states_per_step = 1;
    g_ds->step_slots = 4;
    g_ds->step_pending = g_pending;
    g_ds->sampling_seed = kSeed;
    g_ds->state_exact_hash = g_exact;
    g_ds->state_edge_slices = g_slices;
    g_ds->errors.counters = g_errors;

    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, block, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    uint64_t rank[3];
    for (uint32_t s = 1; s <= 2; ++s) {
        const uint64_t key = g::transition_key_device(*g_ds, s, 0, nullptr, 0);
        rank[s] = hgcommon::transition_rank(key, kSeed);
    }
    const uint32_t lowest = rank[1] < rank[2] ? 1u : 2u;
    assert(g_found_n == 2 && "the selections kept a number of records other than two");
    bool kept_lowest = false, kept_later = false;
    for (uint32_t i = 0; i < 2; ++i) {
        assert(g_found[i].published == 1u);
        kept_lowest |= g_found[i].state_id == lowest && g_found[i].step == 1u;
        kept_later |= g_found[i].state_id == 3u && g_found[i].step == 2u;
    }
    assert(kept_lowest && "step 1's selection did not keep its lowest-ranked candidate");
    assert(kept_later && "step 2's candidate was not selected");
    assert(g_pending[3] == 1u && "step 3's count is not the kept record plus the token");
    return 0;
}
