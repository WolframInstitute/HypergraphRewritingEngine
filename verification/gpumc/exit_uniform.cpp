// GPUMC harness: every lane of a persistent worker block leaves its loop at the same iteration
// when the detector signals exit, so no lane waits at a barrier its block-mates have left.
//
// Runs gpu/src/persistent.cu's k_persistent_match_rewrite itself, the detector block and a worker
// block, with the device's termination detector, ring, pool and barriers
// (verification/gpumc/device).
//
// THE PROTOCOL. A worker block's loop polls for work on thread 0, synchronises, and when it finds
// none asks term.exit_requested() whether the detector has signalled exit; the block leaves when
// it has, and otherwise thread 0 backs off and the block synchronises again. __syncthreads in a
// loop the block leaves is defined only when every lane leaves at the same iteration (PTX:
// bar.sync is aligned), so every lane must reach the same answer about the exit flag.
//
// THE SHAPE. Block 0 is the detector (one thread); block 1 is a worker warp of two lanes. The
// match queue and the record pool are empty, so the detector signals exit once it has seen the
// counters quiescent twice, and the workers only idle. --unroll=3 bounds the worker loop.
//
// THE PROPERTY. No barrier divergence: GPUMC reports a block whose lanes reach different barriers
// or leave while a block-mate waits.
//
// CALIBRATION. -DCALIBRATE_LANE_READ reads the exit flag on each lane in the harness's own copy
// of the check (exit_uniform_lane_read below) in place of the kernel, and the checker must report
// the divergence.
// GPUMC-CELL: clean PASS -- --disable-function-inliner --unroll=3
// GPUMC-CELL: lane_read FAIL -DCALIBRATE_LANE_READ -- --disable-function-inliner --unroll=3

#include "hg_gpumc.hpp"
#include "persistent.cu"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

g::DeviceState* g_ds;
g::MatchWorkItem g_slots[2];
uint64_t g_seq[2] = {0, 1};
uint64_t g_head, g_tail;
g::MatchRecord g_found[2];
uint32_t g_found_n;
uint32_t g_consume_cursor;
uint64_t g_pushed[g::TerminationDetector::kMaxRoles];
uint64_t g_completed[g::TerminationDetector::kMaxRoles];
uint32_t g_should_exit;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];

g::RingBuffer<g::MatchWorkItem>::DeviceView queue() { return {g_slots, g_seq, &g_head, &g_tail, 2, 1}; }
g::Pool<g::MatchRecord>::DeviceView found() { return {g_found, &g_found_n, 2}; }
g::TerminationDetector::DeviceView term() { return {g_pushed, g_completed, &g_should_exit, 1}; }

#if defined(CALIBRATE_LANE_READ)
// The worker loop's idle path with each lane reading the flag itself.
void exit_uniform_lane_read() {
    const g::TerminationDetector::DeviceView t = term();
    for (;;) {
        __syncthreads();
        if (t.exit_requested()) return;
        __syncthreads();
    }
}
#endif

void* thread_fn(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    const unsigned block = id == 0 ? 0u : 1u;
    hg_gpumc::enter(block, id == 0 ? 0u : static_cast<unsigned>(id - 1));
#if defined(CALIBRATE_LANE_READ)
    if (block == 1) {
        exit_uniform_lane_read();
        return nullptr;
    }
#endif
    g::k_persistent_match_rewrite(*g_ds, nullptr, queue(), found(), &g_consume_cursor, term(), 0);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[2] = {1, 2};
    hg_gpumc::launch(2, lanes);
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, thread_fn, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);
    assert(g_should_exit == 1u);
    return 0;
}
