// The topology, metric tensor and holonomy families of hgcommon/state_topology_core.hpp
// (docs/SPEC.md, "StepStatistics topology", "StepStatistics metric tensor" and "StepStatistics
// holonomy") on graphs with known values. The holonomy values are those of Infrageometry's tests
// at Infrageometry@ffa5fe4:Tests/InfraFiberBundles/Experimental/InfraLeviCivitaTests.wlt:73-145.
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
    hgcommon::SG_TOPOLOGY | hgcommon::SG_METRIC_TENSOR | hgcommon::SG_HOLONOMY;

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

TEST(StateTopology, DetMathTrigonometryWithinOneUlpScale) {
    double worst = 0.0;
    for (int i = 0; i <= 4096; ++i) {
        const double t = i / 4096.0;
        double s = 0.0, c = 0.0;
        hgcommon::dm_sincos_pi(t, s, c);
        worst = std::fmax(worst, std::fabs(hgcommon::dm_acos01(t) - std::acos(t)));
        worst = std::fmax(worst, std::fabs(s - std::sin(M_PI * t)));
        worst = std::fmax(worst, std::fabs(c - std::cos(M_PI * t)));
    }
    EXPECT_LE(worst, 1e-15);
    EXPECT_EQ(hgcommon::dm_acos01(1.0), 0.0);
}

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

// On the 5 x 5 grid the centre is the middle vertex, R' = 4 and r = 2. On S_2 the tensor is
// 1 - d(v, w)/4 (Infrageometry@ffa5fe4:Tests/RiemannianInfrageometry/InfraMetricTensorTests.wlt:
// "InfraMetricTensor-grid-shell"); the gap is checked here against the midpoint rule on 2^21
// points.
TEST(StateTopology, MetricTensorGapOfTheGrid) {
    const SgGeometry g = measure(grid(5, 5));
    ASSERT_TRUE(g.defined & hgcommon::SG_METRIC_TENSOR);
    // The 8 vertices at distance 2 from (2, 2), and their pairwise grid distances.
    const int pts[8][2] = {{0, 2}, {1, 1}, {1, 3}, {2, 0}, {2, 4}, {3, 1}, {3, 3}, {4, 2}};
    std::vector<double> values;
    for (auto& a : pts)
        for (auto& b : pts)
            values.push_back(1.0 - (std::abs(a[0] - b[0]) + std::abs(a[1] - b[1])) / 4.0);
    const int N = 1 << 21;
    double acc = 0.0;
    for (int i = 0; i < N; ++i) {
        const double t = (i + 0.5) / N;
        double F = 0.0;
        for (double x : values) F += x <= t ? 1.0 : 0.0;
        acc += std::fabs(F / values.size() - (1.0 - std::acos(t) / M_PI));
    }
    EXPECT_NEAR(g.metric_tensor_gap, acc / N, 1e-9);
    // R' < 2: undefined.
    EXPECT_FALSE(measure(cycle(3)).defined & hgcommon::SG_METRIC_TENSOR);
}

// Infrageometry's tests: the octahedron's faces turn the directions by 1 arc radian; the faces
// of the triangular and of the square torus have holonomy 0.
TEST(StateTopology, HolonomyOfKnownFaces) {
    const SgGeometry oct = measure(octahedron());
    ASSERT_TRUE(oct.defined & hgcommon::SG_HOLONOMY);
    EXPECT_EQ(oct.holonomy_triangle, 1.0);
    const SgGeometry tri = measure(torus(5, 5, true));
    ASSERT_TRUE(tri.defined & hgcommon::SG_HOLONOMY);
    EXPECT_EQ(tri.holonomy_triangle, 0.0);
    EXPECT_FALSE(tri.defined & hgcommon::SG_HOLONOMY_SQUARE);
    const SgGeometry sq = measure(torus(6, 6, false));
    EXPECT_FALSE(sq.defined & hgcommon::SG_HOLONOMY);
    ASSERT_TRUE(sq.defined & hgcommon::SG_HOLONOMY_SQUARE);
    EXPECT_EQ(sq.holonomy_square, 0.0);
    // A vertex of degree 7: undefined.
    std::vector<std::vector<uint32_t>> star;
    for (uint32_t i = 2; i <= 8; ++i) star.push_back({1, i});
    star.push_back({2, 3});
    EXPECT_FALSE(measure(star).defined & hgcommon::SG_HOLONOMY);
}

// A family not requested is not computed.
TEST(StateTopology, OnlyRequestedFamilies) {
    const SgGeometry g = measure(octahedron(), hgcommon::SG_MEASURES_BASE);
    EXPECT_FALSE(g.defined & (hgcommon::SG_TOPOLOGY | hgcommon::SG_METRIC_TENSOR |
                              hgcommon::SG_HOLONOMY | hgcommon::SG_HOLONOMY_SQUARE));
    const SgGeometry t = measure(octahedron(), hgcommon::SG_TOPOLOGY);
    EXPECT_TRUE(t.defined & hgcommon::SG_TOPOLOGY);
    EXPECT_FALSE(t.defined & hgcommon::SG_HOLONOMY);
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
