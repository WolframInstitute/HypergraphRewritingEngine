// GPUMC harness: the quotient replay applies each (instance, match) pair exactly once, when both
// sides of the rendezvous appended a task for it and when the pair's claim bit lies past the
// instance's first claim block.
//
// Runs gpu/include/hg_gpu/quotient_expansion.hpp's DeviceQrCtx::claim itself (the claim chain:
// hgcommon::qr_claim_chain over the device's block install and bit set, and the applied map for
// an instance with no claim block), with the device's atomics (verification/gpumc/device).
//
// THE PROTOCOL. An instance carries a chain of claim blocks, one bit per class match; a claim sets
// the match's bit with fetch_or and wins when the bit was clear. A match past the chain's end
// installs the next block: the claimant allocates a zeroed block and links it with one
// compare-exchange on its predecessor's next word, and a claimant that loses the link walks to the
// block that won.
//
// THE SHAPE. One instance with a one-word chain (bits 0..63). Blocks 0 and 1 claim match 70 (past
// the first block, so both try to install the next); block 2 claims match 3. -DSHAPE_NO_CHAIN
// gives the instance no chain, so the claims go to the applied map.
//
// THE PROPERTY. Exactly one of blocks 0 and 1 wins match 70; block 2 wins match 3. No access
// races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the link's compare-exchange and the bit's
// fetch_or (and the map's exchanges) a load and a separate store; the checker must report match 70
// applied twice.
// GPUMC-CELL: chain PASS
// GPUMC-CELL: no_chain PASS -DSHAPE_NO_CHAIN
// GPUMC-CELL: chain_split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW
// GPUMC-CELL: no_chain_split_rmw FAIL -DSHAPE_NO_CHAIN -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/quotient_expansion.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Map = g::DedupMap;

constexpr uint32_t kWords = 32;
constexpr uint32_t kSlots = 8;

g::DeviceState* g_ds;
g::QeView* g_qe;
uint32_t g_words[kWords];
uint32_t g_words_cursor;
g::MapSlot<uint64_t, uint32_t> g_applied[kSlots];
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
g::DeviceQcInstance g_inst;
g::DeviceSlotMatch g_match[2];
bool g_won[3];

void* claimant(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(id, 0);
    g::QeChild child;
    g::DeviceQrCtx c{*g_ds, *g_qe, child};
    const g::DeviceSlotMatch& m = g_match[id < 2 ? 0 : 1];
    g_won[id] = c.claim(g_inst, g::QeMatchView(m, g_words));
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    for (uint32_t s = 0; s < kSlots; ++s) g_applied[s] = {0, Map::DeviceView::UNPUBLISHED};
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_qe = new (malloc(sizeof(g::QeView))) g::QeView{};
    g_qe->arr_words = g_words;
    g_qe->arr_cursor = &g_words_cursor;
    g_qe->arr_capacity = kWords;
    g_qe->applied = {g_applied, kSlots, kSlots - 1, nullptr};
    g_inst.id = 5;
#if defined(SHAPE_NO_CHAIN)
    g_inst.claims = g::kQeNoClaims;
#else
    g_inst.claims = g::qe_new_claim_block(*g_ds, *g_qe, 1);
#endif
    g_match[0].id = 9;
    g_match[0].local = 70;
    g_match[1].id = 10;
    g_match[1].local = 3;

    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, claimant, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    assert(g_won[0] != g_won[1] && "match 70 was applied twice or by no block");
    assert(g_won[2] && "match 3 was not applied");
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
