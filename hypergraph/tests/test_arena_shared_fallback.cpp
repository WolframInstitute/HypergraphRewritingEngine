#include <gtest/gtest.h>

#include "hypergraph/arena.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

// THE TWO ALLOCATION DISCIPLINES MUST NEVER BUMP ONE BLOCK.
//
// A thread holding an arena worker index bumps a private cursor and mirrors the result into
// block->offset with a plain store. A thread past MAX_ARENA_WORKERS gets index -1 and takes the
// shared path, which reserves on current_block_ with a compare-exchange on that same field. If
// the shared path ever bumps a block a cursor owns, the cursor's store erases the reservation
// and both callers are handed the same address.
//
// Reaching index -1 needs MAX_ARENA_WORKERS threads holding indices at once, so this test is
// sized by that constant and skips where the platform will not give it the threads.

using hg::engine::ConcurrentHeterogeneousArena;
using hg::engine::MAX_ARENA_WORKERS;

namespace {

TEST(ArenaSharedFallback, CursorAndSharedAllocationsNeverOverlap) {
    constexpr int kExtra = 8;                       // threads that must fall past the ceiling
    const int total = MAX_ARENA_WORKERS + kExtra;
    // Sized to force BLOCK TURNOVER on the shared path: a block starts at INITIAL_BLOCK_SIZE
    // (64 KB) and the shared threads must exhaust one and call allocate_new_block while cursor
    // workers are grabbing blocks of their own, which is the only moment current_block_ can be
    // set to a block a cursor owns.
    constexpr size_t kAllocs = 512;                 // per thread
    constexpr size_t kSize = 512;                   // bytes per allocation

    ConcurrentHeterogeneousArena arena;

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<bool> spawn_failed{false};
    // Each thread stamps its own byte into every allocation it receives and re-reads them all at
    // the end: an overlap shows as a byte written by another thread.
    std::vector<std::vector<std::pair<unsigned char*, unsigned char>>> owned(total);
    std::vector<std::thread> threads;
    threads.reserve(total);

    for (int t = 0; t < total; ++t) {
        try {
            threads.emplace_back([&, t] {
                const unsigned char stamp = static_cast<unsigned char>(1 + (t % 251));
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire)) {}
                for (size_t i = 0; i < kAllocs; ++i) {
                    void* p = arena.allocate_raw(kSize, alignof(std::max_align_t), nullptr);
                    ASSERT_NE(p, nullptr);
                    std::memset(p, stamp, kSize);
                    owned[t].emplace_back(static_cast<unsigned char*>(p), stamp);
                }
            });
        } catch (const std::system_error&) {
            spawn_failed.store(true, std::memory_order_relaxed);
            break;
        }
    }
    if (spawn_failed.load(std::memory_order_relaxed)) {
        go.store(true, std::memory_order_release);
        for (auto& th : threads) th.join();
        GTEST_SKIP() << "platform would not spawn " << total << " concurrent threads";
    }

    while (ready.load(std::memory_order_acquire) < total) {}
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    size_t corrupted = 0;
    for (int t = 0; t < total; ++t) {
        for (const auto& [p, stamp] : owned[t]) {
            for (size_t b = 0; b < kSize; ++b) {
                if (p[b] != stamp) { ++corrupted; break; }
            }
        }
    }
    EXPECT_EQ(corrupted, 0u)
        << corrupted << " allocations were written by more than one thread, so two callers were "
        << "given the same bytes -- the shared bump path and a worker cursor bumped one block";
}

}  // namespace
