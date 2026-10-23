// The per-state measures of hgcommon/state_geometry_core.hpp defined in docs/SPEC.md
// ("StepStatistics volume growth" and the sections after it), on graphs with known values.
// The known values are those of the Infrageometry paclet's tests at Infrageometry@ffa5fe4
// (InfraGeometry/Tests: VolumeGrowthTests.wlt and the
// others named at each test). reference/verify_state_statistics.wls checks the same functions,
// through "StepStatistics", against reference/StateGeometryReference.wl on random states.

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

// One state's geometry with its per-vertex values: value k of vertex index v (the vertices in
// ascending id order) at vv[k n + v].
struct Measured {
    SgGeometry g;
    std::vector<double> vv;
    std::vector<double> log_geodesic;   // the mean ln sigma at d = 1..D
    uint32_t n = 0;
    double at(uint32_t k, uint32_t v) const { return vv[k * n + v]; }
};

Measured measure(std::vector<std::vector<uint32_t>> edges) {
    Edges el{std::move(edges)};
    Measured r;
    std::vector<unsigned char> scratch(16);
    std::vector<double> ball(256);
    size_t slots = 0, pairs = 0;
    hgcommon::sg_sizes(el, slots, pairs);
    r.vv.assign(hgcommon::SG_VERTEX_VALUES * slots, 0.0);
    r.log_geodesic.assign(slots + 1, 0.0);
    size_t needed = 0;
    while (!hgcommon::sg_state_geometry(el, scratch.data(), scratch.size(), r.g, ball.data(),
                                        static_cast<uint32_t>(ball.size()), needed, ~0u,
                                        hgcommon::IrSerial{}, nullptr, 0, nullptr, nullptr,
                                        r.vv.data(), r.log_geodesic.data(),
                                        static_cast<uint32_t>(r.log_geodesic.size()))) {
        EXPECT_GT(needed, scratch.size());
        scratch.resize(needed);
    }
    r.n = r.g.vertex_count;
    r.vv.resize(hgcommon::SG_VERTEX_VALUES * r.n);
    r.log_geodesic.resize(r.g.geodesic_radii);
    return r;
}

// Vertex ids 1..; a w x h grid, optionally wrapped into a torus.
std::vector<std::vector<uint32_t>> grid(uint32_t w, uint32_t h, bool torus = false) {
    std::vector<std::vector<uint32_t>> e;
    auto id = [&](uint32_t i, uint32_t j) { return (i % w) * h + (j % h) + 1; };
    for (uint32_t i = 0; i < w; ++i)
        for (uint32_t j = 0; j < h; ++j) {
            if (torus || i + 1 < w) e.push_back({id(i, j), id(i + 1, j)});
            if (torus || j + 1 < h) e.push_back({id(i, j), id(i, j + 1)});
        }
    return e;
}

std::vector<std::vector<uint32_t>> cycle(uint32_t n) {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < n; ++i) e.push_back({i + 1, (i + 1) % n + 1});
    return e;
}

std::vector<std::vector<uint32_t>> hypercube(uint32_t d) {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t v = 0; v < (1u << d); ++v)
        for (uint32_t b = 0; b < d; ++b)
            if (!(v & (1u << b))) e.push_back({v + 1, (v | (1u << b)) + 1});
    return e;
}

std::vector<std::vector<uint32_t>> petersen() {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < 5; ++i) {
        e.push_back({i + 1, (i + 1) % 5 + 1});           // outer cycle
        e.push_back({i + 6, (i + 2) % 5 + 6});           // inner pentagram
        e.push_back({i + 1, i + 6});                     // spokes
    }
    return e;
}

// The graph of a state, built once, with a search's per-radius counts from vertex index s, and
// each vertex's path count and log sum from s (0 for a vertex not reached).
struct Profiles {
    std::vector<uint32_t> shell, ball, riem;
    std::vector<double> sigma, acc;
};

Profiles profiles(std::vector<std::vector<uint32_t>> edges, uint32_t s) {
    Edges el{std::move(edges)};
    size_t slots = 0, pairs = 0;
    hgcommon::sg_sizes(el, slots, pairs);
    std::vector<unsigned char> mem(hgcommon::sg_build_bytes(slots, pairs));
    hgcommon::SgArena arena;
    arena.base = mem.data();
    arena.capacity = mem.size();
    hgcommon::SgGraph g;
    EXPECT_TRUE(hgcommon::sg_build(el, arena, g));
    std::vector<int32_t> dist(g.n + 1, -1);
    std::vector<uint32_t> queue(g.n + 1), layer(g.n + 1), inner(g.n + 2);
    Profiles p;
    p.sigma.assign(g.n + 1, 0.0);
    p.acc.assign(g.n + 1, 0.0);
    uint32_t reached = 0;
    const uint32_t e = hgcommon::sg_bfs(g, s, dist.data(), queue.data(), layer.data(), UINT32_MAX,
                                        reached, inner.data(), p.sigma.data(), p.acc.data());
    uint32_t b = 0, w = 0;
    for (uint32_t r = 0; r <= e; ++r) {
        b += layer[r];
        w += inner[r];
        p.shell.push_back(layer[r]);
        p.ball.push_back(b);
        p.riem.push_back(w);
    }
    return p;
}

constexpr uint32_t kBallD = 0, kBallR = 1, kRiemD = 2, kRiemR = 3, kSphD = 4, kSphR = 5;

}  // namespace

// VolumeGrowthTests.wlt "InfraBall-two-measures-Petersen": counting {1, 4, 10}, Riemannian
// {0, 1, 10} about a vertex of the Petersen graph.
TEST(StateMeasures, PetersenTwoMeasures) {
    const Profiles p = profiles(petersen(), 0);
    EXPECT_EQ(p.ball, (std::vector<uint32_t>{1, 4, 10}));
    EXPECT_EQ(p.riem, (std::vector<uint32_t>{0, 1, 10}));
}

// "InfraBall-two-measures-Ehrhart-Z2-Z3": on Z^2 the counting measure of B(v, r) is
// 2r^2 + 2r + 1 and the Riemannian measure that of radius r - 1; on Z^3 4r^3/3 + 2r^2 + 8r/3 + 1.
TEST(StateMeasures, LatticeBallsAreEhrhartPolynomials) {
    const Profiles p = profiles(grid(11, 11), 60);   // the centre of 11 x 11
    for (uint32_t r = 1; r <= 4; ++r) {
        EXPECT_EQ(p.ball[r], 2 * r * r + 2 * r + 1);
        EXPECT_EQ(p.riem[r], 2 * (r - 1) * (r - 1) + 2 * (r - 1) + 1);
    }
    std::vector<std::vector<uint32_t>> cube;
    auto id = [](uint32_t i, uint32_t j, uint32_t k) { return (i * 9 + j) * 9 + k + 1; };
    for (uint32_t i = 0; i < 9; ++i)
        for (uint32_t j = 0; j < 9; ++j)
            for (uint32_t k = 0; k < 9; ++k) {
                if (i + 1 < 9) cube.push_back({id(i, j, k), id(i + 1, j, k)});
                if (j + 1 < 9) cube.push_back({id(i, j, k), id(i, j + 1, k)});
                if (k + 1 < 9) cube.push_back({id(i, j, k), id(i, j, k + 1)});
            }
    const Profiles c = profiles(cube, id(4, 4, 4) - 1);
    auto z3 = [](uint32_t r) { return (4 * r * r * r + 6 * r * r + 8 * r + 3) / 3; };
    for (uint32_t r = 1; r <= 3; ++r) {
        EXPECT_EQ(c.ball[r], z3(r));
        EXPECT_EQ(c.riem[r], z3(r - 1));
    }
}

// "InfraBall-two-measures-triangular-honeycomb": 3r^2 + 3r + 1 and 1 + 3r(r+1)/2, the
// Riemannian measures one radius earlier. The triangular lattice is the grid with one diagonal
// per square; the honeycomb is the brick wall, the grid with every other vertical edge.
TEST(StateMeasures, TriangularAndHoneycombBalls) {
    const uint32_t w = 21;
    auto id = [&](uint32_t i, uint32_t j) { return i * w + j + 1; };
    std::vector<std::vector<uint32_t>> tri, hex;
    for (uint32_t i = 0; i < w; ++i)
        for (uint32_t j = 0; j < w; ++j) {
            if (i + 1 < w) { tri.push_back({id(i, j), id(i + 1, j)}); hex.push_back({id(i, j), id(i + 1, j)}); }
            if (j + 1 < w) {
                tri.push_back({id(i, j), id(i, j + 1)});
                if ((i + j) % 2 == 0) hex.push_back({id(i, j), id(i, j + 1)});
            }
            if (i + 1 < w && j > 0) tri.push_back({id(i, j), id(i + 1, j - 1)});
        }
    const Profiles t = profiles(tri, id(10, 10) - 1);
    const Profiles h = profiles(hex, id(10, 10) - 1);
    for (uint32_t r = 1; r <= 5; ++r) {
        EXPECT_EQ(t.ball[r], 3 * r * r + 3 * r + 1);
        EXPECT_EQ(t.riem[r], 3 * r * r - 3 * r + 1);
        EXPECT_EQ(h.ball[r], 1 + 3 * r * (r + 1) / 2);
        EXPECT_EQ(h.riem[r], 1 + 3 * r * (r - 1) / 2);
    }
}

// "LogDifferenceQuotients-power-law-exponent" and "DimensionCurvatureFit-recovers-exact-line":
// a ball profile r^3 has every quotient 3, so the ball probe reads d = 3 and R = 0; the
// quotients 2 - 0.1 r(r+1), r = 1..10, give d = 2 and R = -3 (2 + 2)(-0.1) = 1.2.
TEST(StateMeasures, FitRecoversExactProfiles) {
    std::vector<uint32_t> shell(21), cube(21);
    for (uint32_t r = 0; r <= 20; ++r) {
        shell[r] = r + 1;   // rising, so the profiles are not cut
        cube[r] = r * r * r;
    }
    std::vector<double> s(hgcommon::sg_fit_doubles(20));
    double out[hgcommon::SG_GROWTH_VALUES];
    hgcommon::sg_growth_of(shell.data(), cube.data(), cube.data(), 20, s.data(), out);
    EXPECT_NEAR(out[kBallD], 3.0, 1e-9);
    EXPECT_NEAR(out[kBallR], 0.0, 1e-8);

    for (uint32_t r = 1; r <= 10; ++r) s[r] = 2.0 - 0.1 * r * (r + 1);
    double d = 0.0, R = 0.0;
    ASSERT_TRUE(hgcommon::sg_growth_fit(s.data(), 10, hgcommon::SG_PROBE_BALL, d, R));
    EXPECT_NEAR(d, 2.0, 1e-9);
    EXPECT_NEAR(R, 1.2, 1e-9);
    EXPECT_FALSE(hgcommon::sg_growth_fit(s.data(), 1, hgcommon::SG_PROBE_BALL, d, R));
}

// "VolumeGrowthObservables-P5-sphere-exact-line": at an end of P5 every shell has one vertex, so
// the sphere probe reads n = 1 and S = 0. The shells of the other vertices peak at r = 1 or 2
// and leave at most one quotient: no fit.
TEST(StateMeasures, PathSphereProbe) {
    const Measured m = measure({{1, 2}, {2, 3}, {3, 4}, {4, 5}});
    EXPECT_NEAR(m.at(kSphD, 0), 1.0, 1e-12);
    EXPECT_NEAR(m.at(kSphR, 0), 0.0, 1e-12);
    for (uint32_t v = 1; v <= 3; ++v) {
        EXPECT_TRUE(std::isnan(m.at(kSphD, v)));
        EXPECT_TRUE(std::isnan(m.at(kBallD, v)));
    }
    ASSERT_TRUE(m.g.defined & hgcommon::SG_VOLUME_GROWTH);
    EXPECT_NEAR(m.g.growth[kSphD], 1.0, 1e-12);
    EXPECT_EQ(m.g.growth[kBallD], m.at(kBallD, 0));
}

// "VolumeGrowthObservables-C40-cycle": the sphere probe reads n = 1 and S = 0, the Riemannian
// ball probe d within 0.15 of 1 and R within 0.01 of 0. "two-measures-bracket-the-dimension":
// the counting ball probe reads d below 1 and the Riemannian one above.
TEST(StateMeasures, CycleOfForty) {
    const Measured m = measure(cycle(40));
    ASSERT_TRUE(m.g.defined & hgcommon::SG_VOLUME_GROWTH);
    EXPECT_NEAR(m.g.growth[kSphD], 1.0, 1e-8);
    EXPECT_NEAR(m.g.growth[kSphR], 0.0, 1e-8);
    EXPECT_LT(std::fabs(m.g.growth[kRiemD] - 1.0), 0.15);
    EXPECT_LT(std::fabs(m.g.growth[kRiemR]), 0.01);
    EXPECT_LT(m.g.growth[kBallD], 1.0);
    EXPECT_GT(m.g.growth[kRiemD], 1.0);
    for (uint32_t v = 0; v < 40; ++v) EXPECT_EQ(m.at(kRiemD, v), m.at(kRiemD, 0));
}

// "VolumeGrowthObservables-grid2D-sphere-flat", "grid2D-ball-flat",
// "ball-sphere-dimension-agreement-flat" and the bracket: the centre of the 15 x 15 grid.
TEST(StateMeasures, FlatGridCentre) {
    const Measured m = measure(grid(15, 15));
    const uint32_t c = 7 * 15 + 7;
    EXPECT_LT(std::fabs(m.at(kSphD, c) - 2.0), 1e-6);
    EXPECT_LT(std::fabs(m.at(kSphR, c)), 1e-6);
    EXPECT_LT(std::fabs(m.at(kRiemR, c)), 0.1);
    EXPECT_LT(std::fabs(m.at(kRiemD, c) - m.at(kSphD, c)), 0.4);
    EXPECT_LT(m.at(kBallD, c), 2.0);
    EXPECT_GT(m.at(kRiemD, c), 2.0);
}

// "VolumeGrowthObservables-torus-window-before-wrap" and "torus-sphere-flat" on the 20 x 20
// torus, every vertex alike: |d - 2| < 0.35 and |R| < 0.1 by the Riemannian ball, d within 0.05
// of 2 and S = 0 by the sphere; the counting ball reads below 2 and the Riemannian above.
TEST(StateMeasures, FlatTorus) {
    const Measured m = measure(grid(20, 20, true));
    EXPECT_LT(std::fabs(m.g.growth[kRiemD] - 2.0), 0.35);
    EXPECT_LT(std::fabs(m.g.growth[kRiemR]), 0.1);
    EXPECT_LT(std::fabs(m.g.growth[kSphD] - 2.0), 0.05);
    EXPECT_LT(std::fabs(m.g.growth[kSphR]), 1e-6);
    EXPECT_LT(m.g.growth[kBallD], 2.0);
    EXPECT_GT(m.g.growth[kRiemD], 2.0);
}

// "VolumeGrowthObservables-Q8-sphere-positive-curvature": the Hamming cube Q8 reads S > 0.
TEST(StateMeasures, HypercubeSpherePositive) {
    const Measured m = measure(hypercube(8));
    ASSERT_TRUE(m.g.defined & hgcommon::SG_VOLUME_GROWTH);
    EXPECT_GT(m.g.growth[kSphR], 0.0);
}

// A state with no vertex of eccentricity 3 or more has no fit.
TEST(StateMeasures, SmallStatesHaveNoGrowthFit) {
    EXPECT_FALSE(measure({{1, 2}, {2, 3}, {3, 1}}).g.defined & hgcommon::SG_VOLUME_GROWTH);
    EXPECT_FALSE(measure({{1, 2}, {2, 3}}).g.defined & hgcommon::SG_VOLUME_GROWTH);
}

// MetricAlgebraTests.wlt "ShortestPathMultiplicityMatrix-small-fixtures": one shortest path
// along P5, two between antipodes of C4, six across the 3 x 3 grid, one along an edge of K5, one
// from a vertex to itself, none across components.
TEST(StateMeasures, ShortestPathCounts) {
    EXPECT_EQ(profiles({{1, 2}, {2, 3}, {3, 4}, {4, 5}}, 0).sigma[4], 1.0);
    const Profiles c4 = profiles(cycle(4), 0);
    EXPECT_EQ(c4.sigma[2], 2.0);
    EXPECT_EQ(profiles(grid(3, 3), 0).sigma[8], 6.0);
    std::vector<std::vector<uint32_t>> k5;
    for (uint32_t a = 1; a <= 5; ++a)
        for (uint32_t b = a + 1; b <= 5; ++b) k5.push_back({a, b});
    EXPECT_EQ(profiles(k5, 0).sigma[1], 1.0);
    EXPECT_EQ(profiles({{1, 2}, {2, 3}, {3, 4}, {4, 5}}, 2).sigma[2], 1.0);
    EXPECT_EQ(profiles({{1, 2}, {3, 4}}, 0).sigma[2], 0.0);
    // The two paths 0-1-2 and 0-3-2 of C4 each have log2 sigma 0, 0, 1 along them.
    EXPECT_EQ(c4.acc[2], 2.0);
}

// C4: 8 ordered pairs at distance 1 with one path and 4 at distance 2 with two. An edge's
// density is 1, 1 (1 bit); an antipodal pair's is 2, 1, 1, 2 over 6.
TEST(StateMeasures, CycleOfFourGeodesics) {
    const Measured m = measure(cycle(4));
    ASSERT_TRUE(m.g.defined & hgcommon::SG_GEODESIC);
    EXPECT_NEAR(m.g.geodetic_fraction, 8.0 / 12.0, 1e-15);
    const double h2 = (2.0 / 3.0) * std::log2(3.0) + (1.0 / 3.0) * std::log2(6.0);
    EXPECT_NEAR(m.g.segment_entropy, (8.0 + 4.0 * h2) / 12.0, 1e-14);
    ASSERT_EQ(m.log_geodesic.size(), 2u);
    EXPECT_EQ(m.log_geodesic[0], 0.0);
    EXPECT_NEAR(m.log_geodesic[1], std::log(2.0), 1e-15);
}

// P3: every pair has one path; the ends' segment has three vertices of density 1.
TEST(StateMeasures, PathOfThreeGeodesics) {
    const Measured m = measure({{1, 2}, {2, 3}});
    EXPECT_EQ(m.g.geodetic_fraction, 1.0);
    EXPECT_NEAR(m.g.segment_entropy, (4.0 + 2.0 * std::log2(3.0)) / 6.0, 1e-15);
}

// No edge, no pair: undefined.
TEST(StateMeasures, NoEdgeHasNoGeodesics) {
    const Measured m = measure({{1}, {2}});
    EXPECT_FALSE(m.g.defined & hgcommon::SG_GEODESIC);
    EXPECT_EQ(m.g.geodesic_radii, 0u);
}

namespace {

// The curvature of each edge of a state's graph, in edge order, with the lazy or the non-lazy walk.
std::vector<double> edge_curvatures(std::vector<std::vector<uint32_t>> edges, bool lazy) {
    Edges el{std::move(edges)};
    size_t slots = 0, pairs = 0;
    hgcommon::sg_sizes(el, slots, pairs);
    std::vector<unsigned char> mem(hgcommon::sg_build_bytes(slots, pairs));
    hgcommon::SgArena arena;
    arena.base = mem.data();
    arena.capacity = mem.size();
    hgcommon::SgGraph g;
    EXPECT_TRUE(hgcommon::sg_build(el, arena, g));
    std::vector<uint64_t> scratch(hgcommon::sg_transport_stride(g.max_degree) / 8 + 1);
    std::vector<double> k;
    for (uint32_t x = 0; x < g.n; ++x)
        for (uint32_t e = g.off[x]; e < g.off[x + 1]; ++e)
            if (g.nbr[e] > x)
                k.push_back(hgcommon::sg_ollivier_edge(
                    g, x, g.nbr[e], reinterpret_cast<unsigned char*>(scratch.data()), lazy));
    return k;
}

std::vector<std::vector<uint32_t>> complete(uint32_t n) {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t a = 1; a <= n; ++a)
        for (uint32_t b = a + 1; b <= n; ++b) e.push_back({a, b});
    return e;
}

}  // namespace

// OllivierCurvatureTests.wlt "OllivierRicciCurvature-K4", "-P5-zero" and "-C6-zero": with
// idleness 0 every edge of K4 has curvature 2/3 and every edge of P5 and C6 has 0.
TEST(StateMeasures, NonLazyOllivierFixtures) {
    for (double k : edge_curvatures(complete(4), false)) EXPECT_NEAR(k, 2.0 / 3.0, 1e-15);
    for (double k : edge_curvatures({{1, 2}, {2, 3}, {3, 4}, {4, 5}}, false)) EXPECT_EQ(k, 0.0);
    for (double k : edge_curvatures(cycle(6), false)) EXPECT_EQ(k, 0.0);
    const Measured m = measure(complete(4));
    ASSERT_TRUE(m.g.defined & hgcommon::SG_NONLAZY_OLLIVIER);
    EXPECT_NEAR(m.g.nonlazy_ollivier, 2.0 / 3.0, 1e-15);
    // The lazy walk on K4 moves 1/3 of the mass from x to y as well: 2/3 on every edge. On P3
    // the lazy walk gives 1/2 and the non-lazy walk 0.
    EXPECT_NEAR(m.g.ollivier_ricci, 2.0 / 3.0, 1e-15);
    const Measured p3 = measure({{1, 2}, {2, 3}});
    EXPECT_NEAR(p3.g.ollivier_ricci, 0.5, 1e-15);
    EXPECT_EQ(p3.g.nonlazy_ollivier, 0.0);
    EXPECT_FALSE(measure({{1}, {2}}).g.defined & hgcommon::SG_NONLAZY_OLLIVIER);
}

namespace {

// The icosahedron: a top vertex, two rings of five and a bottom vertex.
std::vector<std::vector<uint32_t>> icosahedron() {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < 5; ++i) {
        const uint32_t u = 2 + i, un = 2 + (i + 1) % 5, l = 7 + i, ln = 7 + (i + 1) % 5;
        e.push_back({1, u});
        e.push_back({u, un});
        e.push_back({l, ln});
        e.push_back({u, l});
        e.push_back({u, ln});
        e.push_back({l, 12});
    }
    return e;
}

// The wheel: a hub joined to every vertex of a k-cycle.
std::vector<std::vector<uint32_t>> wheel(uint32_t k) {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < k; ++i) {
        e.push_back({1, i + 2});
        e.push_back({i + 2, (i + 1) % k + 2});
    }
    return e;
}

constexpr uint32_t kKappa = hgcommon::SG_GROWTH_VALUES;

}  // namespace

// TessellationGraphsTests.wlt "TessellationGenus-icosahedron-sphere" (chi = 2): every link of
// the icosahedron is a 5-cycle, so each vertex has 1 - 5/6 and the twelve sum to chi = 2. The
// octahedron's links are 4-cycles: six times 1/3 is 2.
TEST(StateMeasures, ClosedSurfacesSatisfyGaussBonnet) {
    const Measured ico = measure(icosahedron());
    ASSERT_EQ(ico.g.edge_count, 30u);
    ASSERT_TRUE(ico.g.defined & hgcommon::SG_EULER);
    EXPECT_EQ(ico.g.euler_characteristic, 2.0);
    ASSERT_TRUE(ico.g.defined & hgcommon::SG_RIM);
    EXPECT_EQ(ico.g.rim_fraction, 0.0);
    ASSERT_TRUE(ico.g.defined & hgcommon::SG_CYCLE_CURVATURE);
    EXPECT_NEAR(ico.g.cycle_curvature_total, 2.0, 1e-14);
    for (uint32_t v = 0; v < 12; ++v) EXPECT_NEAR(ico.at(kKappa, v), 1.0 / 6.0, 1e-15);

    std::vector<std::vector<uint32_t>> octa;
    for (uint32_t a = 1; a <= 6; ++a)
        for (uint32_t b = a + 1; b <= 6; ++b)
            if (!(a % 2 == 1 && b == a + 1)) octa.push_back({a, b});   // K6 less a matching
    const Measured o = measure(octa);
    EXPECT_EQ(o.g.euler_characteristic, 2.0);
    EXPECT_NEAR(o.g.cycle_curvature_total, 2.0, 1e-14);
}

// "TessellationGenus-torus-genus-1": a triangulated 6 x 6 torus has chi = 0, every link a
// 6-cycle and curvature 0 at every vertex.
TEST(StateMeasures, TriangulatedTorusIsFlat) {
    const uint32_t w = 6;
    auto id = [&](uint32_t i, uint32_t j) { return (i % w) * w + (j % w) + 1; };
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < w; ++i)
        for (uint32_t j = 0; j < w; ++j) {
            e.push_back({id(i, j), id(i + 1, j)});
            e.push_back({id(i, j), id(i, j + 1)});
            e.push_back({id(i, j), id(i + 1, j + 1)});
        }
    const Measured m = measure(e);
    EXPECT_EQ(m.g.euler_characteristic, 0.0);
    EXPECT_EQ(m.g.rim_fraction, 0.0);
    EXPECT_EQ(m.g.cycle_curvature_total, 0.0);
}

// "TessellationCurvature-flat-hyperbolic-spherical": a vertex with seven triangles about it has
// curvature below 0, with six 0, with five above 0. The hub of a wheel has a k-cycle as its link
// and each rim vertex a path, so the rim fraction is k / (k + 1).
TEST(StateMeasures, CurvatureSignsAndRim) {
    const double want[3] = {1.0 / 6.0, 0.0, -1.0 / 6.0};
    for (uint32_t k = 5; k <= 7; ++k) {
        const Measured m = measure(wheel(k));
        ASSERT_TRUE(m.g.defined & hgcommon::SG_CYCLE_CURVATURE) << k;
        EXPECT_NEAR(m.g.cycle_curvature_total, want[k - 5], 1e-15) << k;
        EXPECT_NEAR(m.g.rim_fraction, static_cast<double>(k) / (k + 1), 1e-15) << k;
        EXPECT_EQ(m.g.euler_characteristic, 1.0) << k;   // (k + 1) - 2k + k
        EXPECT_TRUE(std::isnan(m.at(kKappa, 1))) << k;
    }
    // A vertex whose link is two triangles lies on the rim; the other six have a triangle as
    // their link and 1 - 3/6 each.
    const Measured two = measure({{1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {2, 3}, {3, 4},
                                  {4, 2}, {5, 6}, {6, 7}, {7, 5}});
    EXPECT_TRUE(std::isnan(two.at(kKappa, 0)));
    EXPECT_NEAR(two.g.cycle_curvature_total, 3.0, 1e-15);
    EXPECT_NEAR(two.g.rim_fraction, 1.0 / 7.0, 1e-15);
    // C5 has no triangle: no rim fraction, and chi = 5 - 5 + 0.
    const Measured c5 = measure(cycle(5));
    EXPECT_FALSE(c5.g.defined & hgcommon::SG_RIM);
    EXPECT_EQ(c5.g.euler_characteristic, 0.0);
}

// hypergraph_viz particle_detection.cpp: the clustering coefficient C_v, the Euler excess of
// B(v, 3) over the planar bound 3V - 6, and the charge log2(deg) (1 - C_v) + 10 x_v. K4 has
// C = 1 and E = 6 = 3V - 6: no excess, charge 0. K6 has E = 15 over the bound 12: excess 1/4,
// charge 2.5. The star K1,3 has C = 0 and its centre the charge log2 3.
TEST(StateMeasures, ParticleMeasures) {
    const Measured k4 = measure(complete(4));
    ASSERT_TRUE(k4.g.defined & hgcommon::SG_PARTICLE);
    EXPECT_EQ(k4.g.clustering, 1.0);
    EXPECT_EQ(k4.g.euler_excess, 0.0);
    EXPECT_EQ(k4.g.charge, 0.0);

    const Measured k6 = measure(complete(6));
    EXPECT_EQ(k6.g.euler_excess, 0.25);
    EXPECT_EQ(k6.g.charge, 2.5);
    EXPECT_EQ(k6.g.charge_max, 2.5);

    const Measured star = measure({{1, 2}, {1, 3}, {1, 4}});
    EXPECT_EQ(star.g.clustering, 0.0);
    EXPECT_EQ(star.g.euler_excess, 0.0);
    EXPECT_NEAR(star.g.charge_max, std::log2(3.0), 1e-15);
    EXPECT_NEAR(star.g.charge, std::log2(3.0) / 4, 1e-15);
    EXPECT_NEAR(star.at(hgcommon::SG_GROWTH_VALUES + 3, 0), std::log2(3.0), 1e-15);
    EXPECT_EQ(star.at(hgcommon::SG_GROWTH_VALUES + 3, 1), 0.0);

    // A wheel's hub sees k triangles among k(k-1)/2 neighbour pairs.
    const Measured w5 = measure(wheel(5));
    EXPECT_NEAR(w5.at(hgcommon::SG_GROWTH_VALUES + 1, 0), 0.5, 1e-15);
    EXPECT_NEAR(w5.at(hgcommon::SG_GROWTH_VALUES + 1, 1), 2.0 / 3.0, 1e-15);
}

// hypergraph_viz curvature_analysis.cpp compute_dimension_gradient_curvature: L_v is the mean
// over v's neighbours of d_u - d_v. On P4 the ends have d = 1 and the middle vertices
// d_mid = (log 3 / log 2 + (log 4 - log 3) / (log 3 - log 2)) / 2, so an end has d_mid - 1 and a
// middle vertex (1 - d_mid) / 2, and the mean is (d_mid - 1) / 4. On a cycle every d_v is equal.
TEST(StateMeasures, DimensionGradientCurvature) {
    const double d_mid = (std::log(3.0) / std::log(2.0) +
                          (std::log(4.0) - std::log(3.0)) / (std::log(3.0) - std::log(2.0))) / 2;
    const Measured p4 = measure({{1, 2}, {2, 3}, {3, 4}});
    ASSERT_TRUE(p4.g.defined & hgcommon::SG_DIMENSION_GRADIENT);
    EXPECT_NEAR(p4.at(hgcommon::SG_GROWTH_VALUES + 4, 0), d_mid - 1.0, 1e-14);
    EXPECT_NEAR(p4.at(hgcommon::SG_GROWTH_VALUES + 4, 1), (1.0 - d_mid) / 2, 1e-14);
    EXPECT_NEAR(p4.g.dimension_gradient, (d_mid - 1.0) / 4, 1e-14);
    EXPECT_EQ(measure(cycle(9)).g.dimension_gradient, 0.0);
    // A disconnected state given no largest component has no G' and no values.
    const Measured two = measure({{1, 2}, {2, 3}, {3, 4}, {7, 8}});
    EXPECT_FALSE(two.g.defined & hgcommon::SG_DIMENSION_GRADIENT);
    EXPECT_TRUE(std::isnan(two.at(hgcommon::SG_GROWTH_VALUES + 4, 4)));
}

// "StatisticsGraph": the hyperedge {1, 2, 3} is the path 1-2-3 under SG_GRAPH_CONSECUTIVE and the
// triangle under SG_GRAPH_ALL_PAIRS. The path has curvature 1/2 on both edges and no triangle; the
// triangle 3/4 on every edge, one triangle, and every link a single edge, so every vertex lies on
// the rim. Both have Euler characteristic 1.
TEST(StateMeasures, StatisticsGraphRules) {
    Edges el{{{1, 2, 3}}};
    auto geometry_of = [&](uint32_t graph) {
        std::vector<unsigned char> scratch(16);
        std::vector<double> ball(8);
        SgGeometry g;
        size_t needed = 0;
        while (!hgcommon::sg_state_geometry(el, scratch.data(), scratch.size(), g, ball.data(), 8,
                                            needed, ~0u, hgcommon::IrSerial{}, nullptr, 0,
                                            nullptr, nullptr, nullptr, nullptr, 0, graph))
            scratch.resize(needed);
        return g;
    };
    const SgGeometry path = geometry_of(hgcommon::SG_GRAPH_CONSECUTIVE);
    const SgGeometry tri = geometry_of(hgcommon::SG_GRAPH_ALL_PAIRS);
    EXPECT_EQ(path.edge_count, 2u);
    EXPECT_EQ(tri.edge_count, 3u);
    EXPECT_NEAR(path.ollivier_ricci, 0.5, 1e-15);
    EXPECT_NEAR(tri.ollivier_ricci, 0.75, 1e-15);
    EXPECT_EQ(path.euler_characteristic, 1.0);
    EXPECT_EQ(tri.euler_characteristic, 1.0);
    EXPECT_FALSE(path.defined & hgcommon::SG_RIM);
    ASSERT_TRUE(tri.defined & hgcommon::SG_RIM);
    EXPECT_EQ(tri.rim_fraction, 1.0);
    EXPECT_NEAR(path.mean_eccentricity, 5.0 / 3.0, 1e-15);
    EXPECT_EQ(tri.mean_eccentricity, 1.0);
    // The default is the consecutive rule.
    const Measured m = measure({{1, 2, 3}});
    EXPECT_EQ(m.g.edge_count, 2u);
}

// docs/SPEC.md "StepStatistics geodesic analysis". P3: the 6 ordered pairs have d = 1, 1, 1, 1,
// 2, 2: mean 4/3, variance 2 - 16/9 = 2/9. An end v has one neighbour, one step closer to every
// target: curvature -1, spread 0. The middle has one neighbour closer and one farther from
// either end: curvature 0, spread 1. Over the pairs: curvature -4/6, spread 2/6.
TEST(StateMeasures, PathOfThreeGeodesicBundles) {
    const Measured m = measure({{1, 2}, {2, 3}});
    ASSERT_TRUE(m.g.defined & hgcommon::SG_GEODESIC_BUNDLE);
    EXPECT_NEAR(m.g.proper_time, 4.0 / 3.0, 1e-15);
    EXPECT_NEAR(m.g.proper_time_variance, 2.0 / 9.0, 1e-15);
    EXPECT_NEAR(m.g.geodesic_curvature, -2.0 / 3.0, 1e-15);
    EXPECT_NEAR(m.g.bundle_spread, 1.0 / 3.0, 1e-15);
}

// C4: toward a neighbour t, v has t (one step closer) and the antipode of t (one step farther):
// curvature 0, spread 1. Toward the antipode both neighbours are one step closer: curvature -1,
// spread 0. Over the 12 pairs: curvature -4/12, spread 8/12, mean d 16/12, variance 2/9.
TEST(StateMeasures, CycleOfFourGeodesicBundles) {
    const Measured m = measure(cycle(4));
    EXPECT_NEAR(m.g.proper_time, 4.0 / 3.0, 1e-15);
    EXPECT_NEAR(m.g.proper_time_variance, 2.0 / 9.0, 1e-15);
    EXPECT_NEAR(m.g.geodesic_curvature, -1.0 / 3.0, 1e-15);
    EXPECT_NEAR(m.g.bundle_spread, 2.0 / 3.0, 1e-15);
}

// The star K1,3. From the centre toward a leaf t: t is one step closer and the other two leaves
// one step farther, s1 = 1, s2 = 3: curvature 1/3, spread sqrt(3 * 3 - 1) / 3. From a leaf toward
// any target the one neighbour is one step closer: curvature -1, spread 0. The 12 pairs: 3 from
// the centre, 9 from a leaf. Mean d = 18/12, variance 30/12 - 9/4 = 1/4.
TEST(StateMeasures, StarGeodesicBundles) {
    const Measured m = measure({{1, 2}, {1, 3}, {1, 4}});
    EXPECT_NEAR(m.g.proper_time, 1.5, 1e-15);
    EXPECT_NEAR(m.g.proper_time_variance, 0.25, 1e-15);
    EXPECT_NEAR(m.g.geodesic_curvature, (1.0 - 9.0) / 12.0, 1e-15);
    EXPECT_NEAR(m.g.bundle_spread, std::sqrt(8.0) / 12.0, 1e-15);
}

// No pair joined by a path: undefined. Two components count the pairs inside each: two disjoint
// edges have 4 pairs at d = 1, each a single neighbour one step closer.
TEST(StateMeasures, GeodesicBundlesWithoutPairs) {
    EXPECT_FALSE(measure({{1}, {2}}).g.defined & hgcommon::SG_GEODESIC_BUNDLE);
    const Measured two = measure({{1, 2}, {3, 4}});
    ASSERT_TRUE(two.g.defined & hgcommon::SG_GEODESIC_BUNDLE);
    EXPECT_EQ(two.g.proper_time, 1.0);
    EXPECT_EQ(two.g.proper_time_variance, 0.0);
    EXPECT_EQ(two.g.geodesic_curvature, -1.0);
    EXPECT_EQ(two.g.bundle_spread, 0.0);
}
