#include <gtest/gtest.h>

#include "hypergraph/arena.hpp"
#include "hypergraph/quotient_types.hpp"
#include "hypergraph/segmented_array.hpp"
#include "hypergraph/signature.hpp"

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

// SegmentedArray's capacity is the INDEX TYPE's and not the container's, because segments grow:
// segment k holds segment_size << min(k, GROWTH_STEPS) elements. These pin the two properties a
// caller depends on and that the growth could break -- that an index reads back what was written
// to it across every segment regime, and that the array reaches past what uniform segments could
// address.

using hg::engine::ConcurrentHeterogeneousArena;
using hg::engine::SegmentedArray;
using hg::engine::QcEventContent;
using hg::engine::EdgeSignature;
using hg::engine::zero_value_init_v;

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
// allocate_array hands out zero elements: across the first small block, a block boundary, a
// huge-page block, a give-back of written bytes, a block back from the pool with a previous
// life's contents, and -- for a recycling arena -- after a reset over written bytes.
TEST(ArenaZeroInit, FreshAllocationsAreZero) {
    ConcurrentHeterogeneousArena arena;
    for (size_t n : {size_t(100), size_t(20000), size_t(600000)}) {
        uint32_t* a = arena.allocate_array<uint32_t>(n);
        size_t nonzero = 0;
        for (size_t i = 0; i < n; ++i) nonzero += a[i] != 0;
        EXPECT_EQ(nonzero, 0u) << "n=" << n;
    }
    // A give-back of bytes the caller constructed into (a lost segment race builds its
    // elements first): the next request lands on the same bytes and they are zero again.
    uint32_t* g = arena.allocate_array<uint32_t>(1000);
    std::memset(g, 0xFF, 1000 * sizeof(uint32_t));
    ASSERT_TRUE(arena.release_last(g, 1000 * sizeof(uint32_t)));
    uint32_t* h = arena.allocate_array<uint32_t>(1000);
    EXPECT_EQ(h, g);
    for (size_t i = 0; i < 1000; ++i) ASSERT_EQ(h[i], 0u);

    // A block returns to the process-wide pool with its arena and the next arena of the
    // class takes it back (LIFO: the same block, the same bytes) with the old contents.
    uint32_t* first = nullptr;
    {
        ConcurrentHeterogeneousArena a;
        first = a.allocate_array<uint32_t>(1000);
        std::memset(first, 0xFF, 1000 * sizeof(uint32_t));
    }
    {
        ConcurrentHeterogeneousArena b;
        uint32_t* again = b.allocate_array<uint32_t>(1000);
        EXPECT_EQ(again, first);
        for (size_t i = 0; i < 1000; ++i) ASSERT_EQ(again[i], 0u);
    }

    ConcurrentHeterogeneousArena scratch(64 * 1024, /*recycle_blocks=*/true);
    uint32_t* s = scratch.allocate_array<uint32_t>(100);
    std::memset(s, 0xFF, 100 * sizeof(uint32_t));
    scratch.reset();
    uint32_t* t = scratch.allocate_array<uint32_t>(100);
    ASSERT_EQ(t, s);
    for (size_t i = 0; i < 100; ++i) ASSERT_EQ(t[i], 0u);
}

// Every type that opts into zero_value_init_v by specialisation: T() over 0xFF bytes leaves
// only zero bytes, padding included, which is what the skipped fill would have written.
template <typename T>
static bool value_init_is_all_zero() {
    alignas(T) unsigned char buf[sizeof(T)];
    std::memset(buf, 0xFF, sizeof(T));
    new (buf) T();
    for (size_t i = 0; i < sizeof(T); ++i) if (buf[i] != 0) return false;
    return true;
}
TEST(ArenaZeroInit, OptedInTypesAreZeroBytes) {
    static_assert(zero_value_init_v<QcEventContent>);
    static_assert(zero_value_init_v<EdgeSignature>);
    static_assert(zero_value_init_v<uint32_t>);
    static_assert(!zero_value_init_v<std::atomic<uint32_t>>);
    EXPECT_TRUE(value_init_is_all_zero<QcEventContent>());
    EXPECT_TRUE(value_init_is_all_zero<EdgeSignature>());
}

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
