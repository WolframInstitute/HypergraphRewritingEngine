// The topology family of hgcommon/state_topology_core.hpp (docs/SPEC.md, "StepStatistics
// topology") on graphs with known values.
// reference/verify_state_statistics.wls checks the same values, through "StepStatistics",
// against reference/StateGeometryReference.wl.

#include <gtest/gtest.h>

#include "hgcommon/state_geometry_core.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using hgcommon::SgGeometry;

struct Edges {
    std::vector<std::vector<uint32_t>> e;
    uint32_t count() const { return static_cast<uint32_t>(e.size()); }
    uint32_t arity(uint32_t i) const { return static_cast<uint32_t>(e[i].size()); }
    uint32_t at(uint32_t i, uint32_t k) const { return e[i][k]; }
};

constexpr uint32_t kFamilies =
    hgcommon::SG_TOPOLOGY;

// The geometry under `want`, from a 16-byte buffer grown to what each call reports. `calls`
// receives the number of calls.
SgGeometry measure(std::vector<std::vector<uint32_t>> edges, uint32_t want = kFamilies,
                   uint32_t* calls = nullptr) {
    Edges el{std::move(edges)};
    SgGeometry g;
    std::vector<unsigned char> scratch(16);
    std::vector<double> ball(512);
    size_t needed = 0;
    uint32_t n = 1;
    while (!hgcommon::sg_state_geometry(el, scratch.data(), scratch.size(), g, ball.data(),
                                        static_cast<uint32_t>(ball.size()), needed, want)) {
        EXPECT_GT(needed, scratch.size());
        scratch.resize(needed);
        ++n;
    }
    if (calls) *calls = n;
    return g;
}

std::vector<std::vector<uint32_t>> cycle(uint32_t n) {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < n; ++i) e.push_back({i + 1, (i + 1) % n + 1});
    return e;
}

std::vector<std::vector<uint32_t>> octahedron() {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < 6; ++i)
        for (uint32_t j = i + 1; j < 6; ++j)
            if (j != i + 3) e.push_back({i + 1, j + 1});
    return e;
}

// A w x h torus of squares, with one diagonal per square when `triangular`.
std::vector<std::vector<uint32_t>> torus(uint32_t w, uint32_t h, bool triangular) {
    std::vector<std::vector<uint32_t>> e;
    auto id = [&](uint32_t i, uint32_t j) { return (i % w) * h + (j % h) + 1; };
    for (uint32_t i = 0; i < w; ++i)
        for (uint32_t j = 0; j < h; ++j) {
            e.push_back({id(i, j), id(i + 1, j)});
            e.push_back({id(i, j), id(i, j + 1)});
            if (triangular) e.push_back({id(i, j), id(i + 1, j + 1)});
        }
    return e;
}

std::vector<std::vector<uint32_t>> grid(uint32_t w, uint32_t h) {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < w; ++i)
        for (uint32_t j = 0; j < h; ++j) {
            if (i + 1 < w) e.push_back({i * h + j + 1, (i + 1) * h + j + 1});
            if (j + 1 < h) e.push_back({i * h + j + 1, i * h + j + 2});
        }
    return e;
}

void expect_betti(const SgGeometry& g, double b0, double b1, double b2, double rips1) {
    ASSERT_TRUE(g.defined & hgcommon::SG_TOPOLOGY);
    EXPECT_EQ(g.clique_betti[0], b0);
    EXPECT_EQ(g.clique_betti[1], b1);
    EXPECT_EQ(g.clique_betti[2], b2);
    EXPECT_EQ(g.rips_betti1, rips1);
}

}  // namespace

// The clique complex of K3 is a filled triangle, of a cycle of 4 or more vertices a circle, of
// the octahedron a sphere and of a triangulated torus a torus.
TEST(StateTopology, BettiNumbersOfKnownComplexes) {
    expect_betti(measure(cycle(3)), 1, 0, 0, 0);
    expect_betti(measure(cycle(4)), 1, 1, 0, 0);
    expect_betti(measure(octahedron()), 1, 0, 1, 0);
    expect_betti(measure(torus(6, 6, true)), 1, 2, 1, 0);
    // The square torus has no triangles: b1 = E - V + 1 = 72 - 36 + 1.
    expect_betti(measure(torus(6, 6, false)), 1, 37, 0, 0);
    // The square of C6 is the octahedron; the square of C8 keeps one cycle.
    expect_betti(measure(cycle(6)), 1, 1, 0, 0);
    expect_betti(measure(cycle(8)), 1, 1, 0, 1);
    // Two components; K4 is a solid tetrahedron.
    auto two = cycle(4);
    two.push_back({10, 11});
    expect_betti(measure(two), 2, 1, 0, 0);
    expect_betti(measure({{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}}), 1, 0, 0, 0);
    EXPECT_FALSE(measure({}).defined & hgcommon::SG_TOPOLOGY);
}

// A family not requested is not computed.
TEST(StateTopology, OnlyRequestedFamilies) {
    const SgGeometry g = measure(octahedron(), hgcommon::SG_MEASURES_BASE);
    EXPECT_FALSE(g.defined & hgcommon::SG_TOPOLOGY);
    EXPECT_TRUE(measure(octahedron(), hgcommon::SG_TOPOLOGY).defined & hgcommon::SG_TOPOLOGY);
}

// The device grows a record's scratch at most 3 times (record_state_invariants_device): the
// graph, the base metrics and the families' stage each report their size once.
TEST(StateTopology, ScratchGrowsOncePerStage) {
    for (const auto& e : {octahedron(), torus(6, 6, true), torus(6, 6, false), grid(7, 5),
                          cycle(12)}) {
        uint32_t calls = 0;
        measure(e, hgcommon::SG_MEASURES_ALL, &calls);
        EXPECT_LE(calls, 4u);
    }
}
