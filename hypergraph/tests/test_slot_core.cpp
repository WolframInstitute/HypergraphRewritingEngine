// The two access patterns of the frame-slot rule must agree for every edge, on every shape.
// The host fills a state at once; the device reads one edge at a time. Nothing else asserts
// that those are the same function.
#include <gtest/gtest.h>
#include "hgcommon/slot_core.hpp"
#include <random>
#include <vector>

TEST(SlotCore, BulkFormEqualsDefinition) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 400; ++trial) {
        const uint32_t n = 1 + rng() % 64;
        const uint32_t k = 1 + rng() % n;              // orbit count
        std::vector<uint32_t> orbit(n);
        for (uint32_t i = 0; i < n; ++i) orbit[i] = rng() % k;
        uint32_t num_orbits = 0;
        for (uint32_t o : orbit) num_orbits = std::max(num_orbits, o + 1);

        std::vector<uint32_t> bulk(n), counts(num_orbits);
        hgcommon::slots_from_orbits(orbit.data(), n, bulk.data(), counts.data(), num_orbits);

        std::vector<uint32_t> seen(n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            ASSERT_EQ(bulk[i], hgcommon::slot_rank(orbit.data(), n, i))
                << "trial " << trial << " edge " << i;
            ASSERT_LT(bulk[i], n);
            ++seen[bulk[i]];
        }
        // A frame, not a labelling with holes.
        for (uint32_t c : seen) ASSERT_EQ(c, 1u);
    }
}

// Orbit order dominates; ties inside an orbit follow ascending index (== ascending EdgeId,
// because both engines hand edges over in id order).
TEST(SlotCore, OrbitDominatesAndTiesFollowIndex) {
    const uint32_t orbit[] = {1, 0, 1, 0};
    uint32_t out[4], counts[2];
    hgcommon::slots_from_orbits(orbit, 4, out, counts, 2);
    EXPECT_EQ(out[1], 0u);
    EXPECT_EQ(out[3], 1u);
    EXPECT_EQ(out[0], 2u);
    EXPECT_EQ(out[2], 3u);
}
