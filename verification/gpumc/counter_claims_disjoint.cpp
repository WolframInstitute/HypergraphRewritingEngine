// GPUMC harness: the device's counter reservations hand out disjoint ranges below the capacity,
// and leave the counter covering every range handed out.
//
// Runs gpu/include/hg_gpu/coalesced.hpp (coalesced_bounded_claim, coalesced_add),
// gpu/include/hg_gpu/atomic_pool.hpp (Pool::DeviceView::claim, claim_n, settle) and
// gpu/include/hg_gpu/device_arena.hpp (DeviceArena::View::claim) themselves, with the device's
// atomics and warp collectives (verification/gpumc/device).
//
// THE PROTOCOLS. apply_one_match (gpu/src/rewrite.cu) reserves its state slot with
// coalesced_bounded_claim and its event, edges and vertex slots with coalesced_add followed by
// Pool::settle; every other pool claim is Pool::claim or claim_n. A full warp at one counter
// reserves for all its lanes with one atomic (the bounded claim by a compare-exchange that takes
// at most what fits, the add by one atomicAdd and a scan of the lanes' shares); any other set of
// threads makes one atomicAdd each, and an add that ends past the capacity is refused and lowers
// the counter (atomicMin) to its own start, when that is below the capacity, or to the capacity.
// The IR arena moves its cursor only by a compare-exchange that fits.
//
// THE SHAPES. Block 0 is a full warp of two lanes; every other block is one lane (its add is the
// per-thread path).
//   default          block 0's lanes ask coalesced_add + settle for 1 and 2 slots, block 1
//                    claim_n for 2 and block 2 claim for 1, from a pool of capacity 4: 6 slots
//                    asked of 4, so some claim is refused and lowers the counter while others add.
//   -DSHAPE_BOUNDED  block 0's lanes and block 1 ask coalesced_bounded_claim for one state slot
//                    below 2.
//   -DSHAPE_ARENA    block 0's lanes and block 1 claim 2, 3 and 3 words (padded to 2, 4, 4) from an
//                    arena of 6 words.
//
// THE PROPERTY. Granted ranges are disjoint and below the capacity; the counter ends at most at
// the capacity and at or past the end of every granted range (so a reader bounded by the counter,
// Pool::size(), sees every granted slot). Under SHAPE_BOUNDED exactly two slots are granted; under
// SHAPE_ARENA the cursor is the sum of the granted claims. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes atomicAdd and atomicCAS a load and a separate
// store; the checker must report a range granted twice.
// GPUMC-CELL: pool PASS
// GPUMC-CELL: bounded PASS -DSHAPE_BOUNDED
// GPUMC-CELL: arena PASS -DSHAPE_ARENA
// GPUMC-CELL: pool_split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW
// GPUMC-CELL: bounded_split_rmw FAIL -DSHAPE_BOUNDED -DHG_GPUMC_CALIBRATE_SPLIT_RMW
// GPUMC-CELL: arena_split_rmw FAIL -DSHAPE_ARENA -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/coalesced.hpp"
#include "hg_gpu/device_arena.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

#if defined(SHAPE_BOUNDED)
constexpr uint32_t kThreads = 3;
constexpr uint32_t kCap = 2;
#elif defined(SHAPE_ARENA)
constexpr uint32_t kThreads = 3;
constexpr uint32_t kCap = 6;
#else
constexpr uint32_t kThreads = 4;
constexpr uint32_t kCap = 4;
#endif

uint32_t g_data[kCap];
uint32_t g_counter;
uint64_t g_cursor;
uint32_t g_first[kThreads];   // granted start, INVALID_ID when refused
uint32_t g_n[kThreads];       // slots taken when granted

void* thread_fn(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    const unsigned block = id < 2 ? 0u : id - 1u;
    hg_gpumc::enter(block, id < 2 ? id : 0u);
#if defined(SHAPE_BOUNDED)
    g_n[id] = 1;
    g_first[id] = g::coalesced_bounded_claim(&g_counter, kCap);
#elif defined(SHAPE_ARENA)
    g::DeviceArena::View arena{g_data, &g_cursor, kCap};
    const uint64_t words = id == 0 ? 2u : 3u;
    g_n[id] = static_cast<uint32_t>((words + 1u) & ~1u);
    uint32_t* at = arena.claim(words);
    g_first[id] = at ? static_cast<uint32_t>(at - g_data) : g::INVALID_ID;
#else
    const g::Pool<uint32_t>::DeviceView pool{g_data, &g_counter, kCap};
    if (block == 0) {
        g_n[id] = id + 1;
        g_first[id] = pool.settle(g::coalesced_add(&g_counter, g_n[id]), g_n[id]);
    } else if (block == 1) {
        g_n[id] = 2;
        g_first[id] = pool.claim_n(2);
    } else {
        g_n[id] = 1;
        g_first[id] = pool.claim();
    }
#endif
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {2, 1, 1};
    hg_gpumc::launch(kThreads - 1, lanes);
    pthread_t t[kThreads];
    for (long i = 0; i < kThreads; ++i)
        pthread_create(&t[i], nullptr, thread_fn, reinterpret_cast<void*>(i));
    for (uint32_t i = 0; i < kThreads; ++i) pthread_join(t[i], nullptr);

#if defined(SHAPE_ARENA)
    const uint64_t end = g_cursor;
#else
    const uint64_t end = g_counter;
#endif
    uint32_t granted = 0;
    uint64_t taken = 0;
    for (uint32_t i = 0; i < kThreads; ++i) {
        if (g_first[i] == g::INVALID_ID) continue;
        ++granted;
        taken += g_n[i];
        assert(g_first[i] + g_n[i] <= kCap && "a granted range passes the capacity");
        assert(g_first[i] + g_n[i] <= end && "the counter ends below a granted range");
        for (uint32_t j = i + 1; j < kThreads; ++j) {
            if (g_first[j] == g::INVALID_ID) continue;
            const bool apart =
                g_first[i] + g_n[i] <= g_first[j] || g_first[j] + g_n[j] <= g_first[i];
            assert(apart && "two threads were granted the same slot");
        }
    }
    assert(end <= kCap && "the counter ends past the capacity");
#if defined(SHAPE_BOUNDED)
    assert(granted == kCap && "the bounded claim refused a slot that fit");
#endif
#if defined(SHAPE_ARENA)
    assert(end == taken && "the arena cursor is not the sum of the granted claims");
#endif
    (void)granted;
    (void)taken;
    return 0;
}
