#include <gtest/gtest.h>

#include "hypergraph/arena.hpp"
#include "hgcommon/portable_intrinsics.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <utility>
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

#if defined(__linux__)
// Pages mapped and pages resident, from /proc/self/statm, in bytes.
std::pair<size_t, size_t> mapped_and_resident() {
    long pages = 0, resident = 0;
    if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) pages = resident = 0;
        std::fclose(f);
    }
    return {static_cast<size_t>(pages) * 4096u, static_cast<size_t>(resident) * 4096u};
}
#endif

// A block back in the pool stays mapped until arena_trim_block_pool unmaps it; an arena then
// maps afresh, and its bytes are reported zero. One 23 MB request.
TEST(ArenaPool, TrimUnmapsPooledBlocks) {
    constexpr size_t kBytes = size_t{23} << 20;
    {
        ConcurrentHeterogeneousArena a;
        void* p = a.allocate_raw(kBytes, 64, nullptr);
        std::memset(p, 0xAB, kBytes);
    }
#if defined(__linux__)
    const auto before = mapped_and_resident();
#endif
    EXPECT_GE(hg::engine::arena_trim_block_pool(0), kBytes);
#if defined(__linux__)
    const auto after = mapped_and_resident();
    EXPECT_GE(before.first - std::min(before.first, after.first), kBytes);
    EXPECT_GE(before.second - std::min(before.second, after.second), kBytes - (size_t{1} << 20));
#endif
    ConcurrentHeterogeneousArena b;
    bool zero = false;
    auto* q = static_cast<const unsigned char*>(b.allocate_raw(kBytes, 64, &zero));
    EXPECT_TRUE(zero) << "a fresh mapping was not reported zero";
    size_t nonzero = 0;
    for (size_t i = 0; i < kBytes; ++i) nonzero += q[i] != 0;
    EXPECT_EQ(nonzero, 0u);
}

// hgcommon::atomic_raise never lowers the value, and 16 threads raising one word each to their
// own values leave the largest.
TEST(AtomicRaise, EndsAtTheLargestValue) {
    std::atomic<uint64_t> a{10};
    hgcommon::atomic_raise(a, uint64_t{5});
    EXPECT_EQ(a.load(), 10u);
    hgcommon::atomic_raise(a, uint64_t{12});
    EXPECT_EQ(a.load(), 12u);
    std::atomic<uint64_t> shared{0};
    std::vector<std::thread> ts;
    for (uint64_t t = 0; t < 16; ++t)
        ts.emplace_back([&shared, t] {
            for (uint64_t i = 0; i < 10000; ++i) hgcommon::atomic_raise(shared, t * 10000 + i);
        });
    for (auto& t : ts) t.join();
    EXPECT_EQ(shared.load(), 15u * 10000 + 9999);
}

// An arena's block bytes count every block it grabbed, and the process-wide high-water mark is at
// least that while the arena lives.
TEST(ArenaBlockBytes, ProcessHighWaterCoversALiveArena) {
    ConcurrentHeterogeneousArena a;
    const size_t before = a.block_bytes();
    a.allocate_raw(size_t{3} << 20, 64, nullptr);
    EXPECT_GT(a.block_bytes(), before + (size_t{3} << 20) - 1);
    EXPECT_GE(hg::engine::arena_block_bytes_high_water(), a.block_bytes());
}

}  // namespace
