#include <gtest/gtest.h>

#include "hypergraph/arena.hpp"
#include "hypergraph/segmented_array.hpp"

#include <atomic>
#include <thread>
#include <vector>

// SegmentedArray's capacity is the INDEX TYPE's and not the container's, because segments grow:
// segment k holds segment_size << min(k, GROWTH_STEPS) elements. These pin the two properties a
// caller depends on and that the growth could break -- that an index reads back what was written
// to it across every segment regime, and that the array reaches past what uniform segments could
// address.

using hg::engine::ConcurrentHeterogeneousArena;
using hg::engine::SegmentedArray;

namespace {

// Enough elements to cross the doubling region and land in the constant-size one, so the test
// exercises BOTH arms of the index decomposition rather than only the first.
TEST(SegmentedArrayGrowth, EveryIndexReadsBackWhatWasWrittenAcrossSegmentRegimes) {
    ConcurrentHeterogeneousArena arena;
    SegmentedArray<uint32_t> a;

    const uint32_t n = 3u * 1024u * 1024u;   // past segment_size * (2^(GROWTH_STEPS+1) - 1)
    for (uint32_t i = 0; i < n; ++i) a.emplace_at(i, arena, i * 2654435761u);

    for (uint32_t i = 0; i < n; ++i)
        ASSERT_EQ(a[i], i * 2654435761u) << "index " << i << " did not read back its own value";
    EXPECT_EQ(a.size(), n);
}

// The old uniform layout held MAX_SEGMENTS * segment_size = 4,194,304 elements and raised
// CapacityExhausted past it. Growth removes that ceiling, so an index beyond it is ordinary.
TEST(SegmentedArrayGrowth, AddressesPastTheUniformCeiling) {
    ConcurrentHeterogeneousArena arena;
    SegmentedArray<uint32_t> a;

    const uint32_t beyond = 4u * 1024u * 1024u + 7u;
    a.emplace_at(beyond, arena, 0xABCDEF01u);
    EXPECT_EQ(a[beyond], 0xABCDEF01u);
    EXPECT_GT(a.size(), 4u * 1024u * 1024u);
}

// Concurrent creators of one segment each allocate it and one installs it. The losers give
// their allocation back to their worker cursor, so the arena holds one segment's bytes per
// segment, not one per racing thread. Without the give-back the bytes here grow with the
// thread count for the same array -- the footprint term the release standard requires to be
// zero (measured on bench_cpu_evolve wpp depth 7: 191 MB -> 394 MB of used arena from one
// thread to eight, all of it losing segment allocations).
TEST(SegmentedArrayGrowth, LosingSegmentAllocationsAreGivenBack) {
    ConcurrentHeterogeneousArena arena;
    SegmentedArray<uint32_t> a;
    constexpr int      kThreads  = 8;
    constexpr uint32_t kSegments = 6;

    std::atomic<int> ready{0};
    std::atomic<int> go{0};
    std::vector<std::thread> t;
    for (int w = 0; w < kThreads; ++w) {
        t.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (go.load(std::memory_order_acquire) == 0) {}
            // The first index of each segment, in the same order on every thread, so every
            // segment is raced by every thread.
            for (uint32_t s = 0; s < kSegments; ++s)
                a.get_or_default(static_cast<uint32_t>(a.segment_first_index(s)), arena);
        });
    }
    while (ready.load(std::memory_order_acquire) < kThreads) {}
    go.store(1, std::memory_order_release);
    for (auto& th : t) th.join();

    size_t installed = 0;
    for (uint32_t s = 0; s < kSegments; ++s) installed += a.segment_bytes(s);
    // Alignment slack per allocation is under one cache line; anything beyond that is a
    // losing allocation that was kept.
    EXPECT_LE(arena.bytes_allocated(), installed + kSegments * kThreads * 64);
    EXPECT_GE(arena.bytes_allocated(), installed);
}

}  // namespace
