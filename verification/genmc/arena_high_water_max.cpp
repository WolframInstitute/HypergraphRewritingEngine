// GenMC harness: the arena's byte high-water mark ends at the largest live total.
//
// THE CODE. note_block_grab (hypergraph/src/arena.cpp) adds a block's bytes to a live counter
// with fetch_add and raises the high-water mark to the new total with a compare_exchange_weak
// retry loop:
//     const size_t now = live.fetch_add(bytes, relaxed) + bytes;
//     size_t hw = high_water.load(relaxed);
//     while (now > hw && !high_water.compare_exchange_weak(hw, now, relaxed)) {}
// Every block grab runs it twice (the process total and the arena's own), so it is the weak CAS
// a harness of the allocator meets most often. This harness checks the loop on its own, with
// its spurious failures explored; arena_cursor_vs_shared_disjoint checks the allocator.
//
// THE PROPERTY. Two threads each record one grab of a different size. No block is freed, so the
// largest live total is the sum, and the mark must end equal to it: a raise lost to the other
// thread's raise leaves the smaller total.
//
// CALIBRATION. -DCALIBRATE_LOAD_STORE_MAX raises the mark with a load, a compare and a store,
// the max without the retry; the checker must report the lost raise.
//
// GENMC-ARGS: --disable-estimation --weak-cas-spurious-bound=2
// GENMC-BOUND: weak CAS spurious-failure bound 2: 87 complete executions, K = 32 dropped (bounded, not exhaustive).
// GENMC-EXPECT: pass
// GENMC-CALIBRATE: -DCALIBRATE_LOAD_STORE_MAX
//
// Build/run: verification/genmc/run.sh arena_high_water_max

#define HG_MAX_ARENA_WORKERS 1
#define HG_ARENA_BLOCK_SIZE 512

#include <pthread.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <new>

#include "genmc_support.hpp"

// The same shim as arena_cursor_vs_shared_disjoint: arena.cpp is compiled into this program
// and its over-aligned cursor array needs the aligned allocation functions.
void* operator new(std::size_t n, std::align_val_t) { return ::operator new(n ? n : 1); }
void* operator new[](std::size_t n, std::align_val_t) { return ::operator new(n ? n : 1); }
void operator delete(void* p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::align_val_t) noexcept { ::operator delete(p); }

#include "hypergraph/arena.hpp"
#include "../../hypergraph/src/arena.cpp"

namespace {

std::atomic<size_t> g_live{0};
std::atomic<size_t> g_high_water{0};

void record(size_t bytes) {
#if defined(CALIBRATE_LOAD_STORE_MAX)
    const size_t now = g_live.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    if (now > g_high_water.load(std::memory_order_relaxed))
        g_high_water.store(now, std::memory_order_relaxed);
#else
    hypergraph::note_block_grab(g_live, g_high_water, bytes);
#endif
}

void* worker(void* arg) {
    record(reinterpret_cast<size_t>(arg));
    return nullptr;
}

}  // namespace

int main() {
    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(size_t{100}));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(size_t{200}));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);
    assert(g_high_water.load(std::memory_order_relaxed) == 300);
    return 0;
}
