// GPUMC harness: the quotient replay's run-signature claims give every application of one event
// class one key, and exactly one of them the claim.
//
// Runs gpu/include/hg_gpu/quotient_expansion.hpp's qe_claim_runsig itself (the match's cached key,
// hgcommon::qr_cached_key and qr_cache_key over the device's cells, and keyed_claim_device on
// canon_seen), with the device's maps and atomics (verification/gpumc/device).
//
// THE PROTOCOL. A match caches its claimed key for one output step: the step is set once by
// compare-exchange and the key stored with release by the setter; a reader takes the key with
// acquire and checks the step. Without a cached key the application computes its signature
// values and claims them in canon_seen; a key hit recomputes the values of the class's first
// application (match record, output step) and compares.
//
// THE SHAPE. Matches 0 and 1 have one signature (same classes, rule and slots). Blocks 0 and 1
// claim match 0 at output step 1; block 2 claims match 1 at output step 1.
//
// THE PROPERTY. The three claims return one key, and exactly one of them won. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the step's compare-exchange and the map's
// exchanges a load and a separate store; the checker must report two wins.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/quotient_expansion.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using SeenMap = g::ConcurrentMap<uint64_t, uint64_t>;

constexpr uint32_t kSlots = 4;

g::DeviceState* g_ds;
g::QeView* g_qe;
g::DeviceSlotMatch g_matches[2];
uint32_t g_matches_n = 2;
uint32_t g_words[8] = {0, 1, 0, 1};   // match 0: consumed {0}, produced {1}; match 1 the same
g::MapSlot<uint64_t, uint64_t> g_seen[kSlots];
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
g::QeRunsigClaim g_claim[3];

void* claimant(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(id, 0);
    const g::DeviceSlotMatch& m = g_matches[id < 2 ? 0 : 1];
    g_claim[id] = g::qe_claim_runsig(*g_ds, *g_qe, g::QeMatchView(m, g_words), m.from_hash, 1u);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    for (uint32_t s = 0; s < kSlots; ++s) g_seen[s] = {0, SeenMap::DeviceView::UNPUBLISHED};
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->event_key_mask = ~uint64_t{0};
    g_qe = new (malloc(sizeof(g::QeView))) g::QeView{};
    g_qe->keys = g::EVENT_SIG_FULL;
    g_qe->matches = {g_matches, &g_matches_n, 2};
    g_qe->arr_words = g_words;
    g_qe->canon_seen = {g_seen, kSlots, kSlots - 1, nullptr};
    for (uint32_t i = 0; i < 2; ++i) {
        g::DeviceSlotMatch& m = g_matches[i];
        m.to_hash = 0xB0B;
        m.from_hash = 0xA0A;
        m.id = i;
        m.rule = 0;
        m.num_consumed = 1;
        m.num_produced = 1;
        m.arr_offset = 2 * i;
    }

    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, claimant, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    assert(g_claim[0].key == g_claim[1].key && g_claim[1].key == g_claim[2].key &&
           "applications of one class received different keys");
    const uint32_t won = (g_claim[0].won ? 1u : 0u) + (g_claim[1].won ? 1u : 0u) +
                         (g_claim[2].won ? 1u : 0u);
    assert(won == 1 && "the class was claimed by no application or by more than one");
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
