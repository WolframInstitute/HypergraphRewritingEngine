// GPUMC harness: the device's bounded appends hand out each slot or id once and refuse past the
// limit with a report.
//
// Runs gpu/src/persistent.cu's session_frontier_append and gpu/include/hg_gpu/quotient_expansion.hpp's
// DeviceQrCtx::mint_event themselves, with the device's atomics and error counters
// (verification/gpumc/device).
//
// THE PROTOCOLS. session_frontier_append takes a slot with a relaxed fetch-and-add on the
// frontier count and writes the entry when the slot is below the capacity, else records
// kFrontierCapFull; the host reads min(count, capacity) entries after the kernel. mint_event
// reads the next raw event id relaxed and, below replay_id_limit, takes one with a relaxed
// fetch-and-add, refusing (kReplayIdsExhausted, INVALID_ID) an id at or past the limit.
//
// THE SHAPE. Three blocks each append one state to a frontier of capacity 2 and mint one replay
// id under a limit of 2.
//
// THE PROPERTY. The frontier holds two distinct appended states with their steps, and
// kFrontierCapFull was recorded; two blocks hold the distinct ids 0 and 1, the third INVALID_ID,
// and kReplayIdsExhausted was recorded. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the fetch-and-adds a load and a separate
// store; the checker must report a slot or an id handed out twice.
// GPUMC-CELL: clean PASS -- --disable-function-inliner
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW -- --disable-function-inliner

#include "hg_gpumc.hpp"
#include "persistent.cu"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

g::DeviceState* g_ds;
g::SessionView* g_sess;
g::QeView* g_qe;
g::StateId g_frontier[2];
uint32_t g_frontier_step[2];
uint32_t g_frontier_count;
uint32_t g_next_raw_event;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
uint32_t g_id[3];

void* block(void* arg) {
    const unsigned b = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(b, 0);
    g::session_frontier_append(*g_ds, *g_sess, 10 + b, 20 + b);
    g::QeChild child;
    g::DeviceQrCtx c{*g_ds, *g_qe, child};
    g_id[b] = c.mint_event(0);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->replay_id_limit = 2;
    g_sess = new (malloc(sizeof(g::SessionView))) g::SessionView{};
    g_sess->enabled = true;
    g_sess->frontier = g_frontier;
    g_sess->frontier_step = g_frontier_step;
    g_sess->frontier_count = &g_frontier_count;
    g_sess->frontier_cap = 2;
    g_qe = new (malloc(sizeof(g::QeView))) g::QeView{};
    g_qe->next_raw_event = &g_next_raw_event;

    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, block, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    assert(g_frontier[0] != g_frontier[1] && "one frontier slot was handed out twice");
    for (uint32_t i = 0; i < 2; ++i) {
        assert(g_frontier[i] >= 10 && g_frontier[i] <= 12);
        assert(g_frontier_step[i] == g_frontier[i] + 10);
    }
    assert(g_errors[static_cast<uint32_t>(g::ErrorKind::kFrontierCapFull)] != 0u);
    uint32_t refused = 0, seen = 0;
    for (uint32_t b = 0; b < 3; ++b) {
        if (g_id[b] == g::INVALID_ID) { ++refused; continue; }
        assert(g_id[b] < 2u && !(seen & (1u << g_id[b])) && "one replay id was minted twice");
        seen |= 1u << g_id[b];
    }
    assert(refused == 1 && seen == 3u);
    assert(g_errors[static_cast<uint32_t>(g::ErrorKind::kReplayIdsExhausted)] != 0u);
    return 0;
}
