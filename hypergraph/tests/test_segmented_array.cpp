#include <gtest/gtest.h>

#include "hypergraph/arena.hpp"
#include "hypergraph/segmented_array.hpp"

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

}  // namespace
