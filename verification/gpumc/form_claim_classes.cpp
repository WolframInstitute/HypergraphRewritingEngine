// GPUMC harness: states claiming their canonical form at once are merged exactly when their forms
// are equal, and a claimant compares against a form record its writer finished.
//
// Runs gpu/include/hg_gpu/exploration.hpp's state_claim_form (keyed_claim_device over
// hgcommon::dedup_claim, the form record pool and the device map) itself, with the device's
// atomics and fences (verification/gpumc/device). Full-mode state identity and the exact hash of
// event identity claim this way; Automatic-mode identity (state_claim_content) and the replay's
// run signatures (qe_claim_runsig) run the same keyed_claim_device with another policy.
//
// THE PROTOCOL. A claimant looks its key up; on a hit it fences and compares its form with the
// record the value names. On a miss it writes its form record (claimed from the pool), fences, and
// offers the record's offset; when another offer won, it fences and compares. A record that is
// not its class makes it take the next probe key.
//
// THE SHAPE. Three states in three blocks, all with the same first key: states 0 and 1 have one
// form, state 2 another.
//
// THE PROPERTY. States 0 and 1 have one canonical state and one claimed key, and exactly one of
// them is fresh; state 2 is fresh, its own canonical state, under another key. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the map's exchanges a load and a separate
// store; the checker must report two fresh states of one class.
//
// The fences are not what publishes a record: with -DHG_GPUMC_CALIBRATE_NO_THREADFENCE the
// property holds (696 executions), because the map's value exchange is acq_rel and its lookups
// acquire.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: no_threadfence PASS -DHG_GPUMC_CALIBRATE_NO_THREADFENCE
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/exploration.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Map = g::DedupMap;

constexpr uint32_t kSlots = 4;
constexpr uint64_t kFirstKey = 0x1234;

const uint32_t kFormA[3] = {1, 2, 3};
const uint32_t kFormB[3] = {1, 3, 2};

g::DeviceState* g_ds;
g::MapSlot<uint64_t, uint32_t> g_slots[kSlots];
uint32_t g_words[64];
uint32_t g_words_n;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
g::StateClaim g_claim[3];

void* claimant(void* arg) {
    const unsigned sid = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(sid, 0);
    const uint32_t* form = sid < 2 ? kFormA : kFormB;
    const Map::DeviceView map{g_slots, kSlots, kSlots - 1, nullptr};
    g_claim[sid] = g::state_claim_form(*g_ds, sid, kFirstKey, form, 3, map, {g_words, &g_words_n, 64});
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    for (uint32_t s = 0; s < kSlots; ++s) g_slots[s] = {0, Map::DeviceView::UNPUBLISHED};
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, claimant, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    assert(g_claim[0].canonical == g_claim[1].canonical && g_claim[0].key == g_claim[1].key &&
           "two states with one form were not merged");
    assert(g_claim[0].canonical <= 1 && (g_claim[0].fresh != g_claim[1].fresh) &&
           "the merged class does not have exactly one fresh state");
    assert(g_claim[2].fresh && g_claim[2].canonical == 2 && g_claim[2].key != g_claim[0].key &&
           "a state with another form was merged or shares a key");
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
