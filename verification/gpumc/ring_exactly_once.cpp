// GPUMC harness: the device work queue hands every pushed item out exactly once.
//
// Runs gpu/include/hg_gpu/ring_buffer.hpp's RingBuffer::DeviceView::try_push and try_pop
// themselves (hgcommon::ring_claim over the device's atomic_ref storage operations), with the
// device's atomics (verification/gpumc/device). The persistent kernel's match queue and the keyed
// rewrites' `ready` ring are this ring.
//
// WHAT IS BEING PROVED. No item is delivered twice, none is delivered that was never pushed, and
// none is lost: what the threads popped, together with what is left in the ring after they finish,
// is exactly what they pushed. A lost item is a completion that can never be booked, and the
// persistent kernel's termination detector then waits for it.
//
// THE BOUND. Capacity two, two threads in two blocks, each pushing one item and popping once.
// That is the smallest shape in which two reservers contend for one slot and the ring can wrap.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes every compare-exchange a load and a separate
// store, so two reservers can take one position; the checker must report an item delivered twice
// or lost.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/ring_buffer.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

constexpr uint32_t kCap = 2;
constexpr uint32_t kThreads = 2;

uint32_t g_slots[kCap];
uint64_t g_seq[kCap] = {0, 1};   // RingBuffer::clear: seq[i] = i
uint64_t g_head, g_tail;
bool g_pushed[kThreads];
uint32_t g_taken[kThreads];
bool g_got[kThreads];

g::RingBuffer<uint32_t>::DeviceView ring() { return {g_slots, g_seq, &g_head, &g_tail, kCap, kCap - 1}; }

void* worker(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(id, 0);
    // Item ids start at one, so a slot's zero is never an item.
    g::RingBuffer<uint32_t>::DeviceView r = ring();
    g_pushed[id] = r.try_push(id + 1);
    uint32_t got = 0;
    g_got[id] = r.try_pop(got);
    g_taken[id] = got;
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[kThreads] = {1, 1};
    hg_gpumc::launch(kThreads, lanes);
    pthread_t t[kThreads];
    for (long i = 0; i < static_cast<long>(kThreads); ++i)
        pthread_create(&t[i], nullptr, worker, reinterpret_cast<void*>(i));
    for (uint32_t i = 0; i < kThreads; ++i) pthread_join(t[i], nullptr);

    // Every push fits a two-slot ring; what is left is drained here.
    uint32_t seen[kThreads + 1] = {0, 0, 0};
    for (uint32_t i = 0; i < kThreads; ++i) {
        assert(g_pushed[i] && "a push into a ring with room was refused");
        if (!g_got[i]) continue;
        assert(g_taken[i] >= 1 && g_taken[i] <= kThreads && "a pop returned an unpushed slot");
        ++seen[g_taken[i]];
    }
    g::RingBuffer<uint32_t>::DeviceView r = ring();
    for (uint32_t rest; r.try_pop(rest);) {
        assert(rest >= 1 && rest <= kThreads);
        ++seen[rest];
    }
    for (uint32_t item = 1; item <= kThreads; ++item)
        assert(seen[item] == 1 && "an item was handed out twice or lost");
    return 0;
}
