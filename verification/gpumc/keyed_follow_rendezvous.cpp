// GPUMC harness: a keyed-rewrite child whose twin has not published its canonical hash is never
// lost: it either waits on the twin's follower stack and is handed to `ready` when the twin
// closes the stack, or it reads the hash the twin published.
//
// Runs gpu/include/hg_gpu/exploration.hpp's publish_canonical_hash and
// gpu/include/hg_gpu/keyed.hpp's keyed_hash_or_follow (keyed_follow) and keyed_close_followers
// themselves, with the device's ring, atomics and fences (verification/gpumc/device).
//
// THE PROTOCOL. A twin publishes its canonical hash (canonicalise_child, gpu/src/persistent.cu)
// and then closes its follower stack: fence, exchange the head for FOLLOW_CLOSED, fence, push
// every follower on the stack to `ready`. A child that finds the twin's hash unpublished pushes
// its event on the twin's stack by compare-exchange, unless the stack is closed; then it fences
// and reads the hash again.
//
// THE SHAPE. Block 0 is the twin; blocks 1 and 2 are children of events 0 and 1 checking it.
//
// THE PROPERTY. Each child either follows (and its event is in `ready` exactly once) or reads the
// twin's hash. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_NO_THREADFENCE removes the fences; the checker must report a
// child that neither follows nor reads the hash.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: no_threadfence FAIL -DHG_GPUMC_CALIBRATE_NO_THREADFENCE

#include "hg_gpumc.hpp"
#include "hg_gpu/keyed.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

constexpr g::StateId kTwin = 0;
constexpr uint64_t kHash = 0xABCDu;

g::DeviceState* g_ds;
uint64_t g_hash[1];
uint32_t g_follow_head[1] = {g::FOLLOW_EMPTY};
uint32_t g_follow_next[2];
uint32_t g_ready_slots[4];
uint64_t g_ready_seq[4] = {0, 1, 2, 3};
uint64_t g_ready_head, g_ready_tail;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
bool g_following[2];
uint64_t g_seen[2];

g::RingBuffer<uint32_t>::DeviceView ready() {
    return {g_ready_slots, g_ready_seq, &g_ready_head, &g_ready_tail, 4, 3};
}

void* twin(void*) {
    hg_gpumc::enter(0, 0);
    g::publish_canonical_hash(*g_ds, kTwin, kHash);
    g::keyed_close_followers(*g_ds, kTwin, ready());
    return nullptr;
}

void* child(void* arg) {
    const unsigned ev = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(1 + ev, 0);
    g_seen[ev] = g::keyed_hash_or_follow(*g_ds, kTwin, ev, /*may_follow=*/true, g_following[ev]);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->state_canonical_hash = g_hash;
    g_ds->keyed.follow_head = g_follow_head;
    g_ds->keyed.follow_next = g_follow_next;
    g_ds->errors.counters = g_errors;

    pthread_t t[3];
    pthread_create(&t[0], nullptr, twin, nullptr);
    pthread_create(&t[1], nullptr, child, reinterpret_cast<void*>(0L));
    pthread_create(&t[2], nullptr, child, reinterpret_cast<void*>(1L));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    for (uint32_t ev = 0; ev < 2; ++ev) {
        uint32_t handed = 0;
        for (uint64_t i = 0; i < g_ready_tail; ++i) handed += g_ready_slots[i & 3] == ev ? 1u : 0u;
        if (g_following[ev]) {
            assert(handed == 1 && "a follower was not handed to ready exactly once");
        } else {
            assert(g_seen[ev] == kHash && "a child neither followed nor read the twin's hash");
            assert(handed == 0);
        }
    }
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
