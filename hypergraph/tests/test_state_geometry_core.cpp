// hgcommon/state_geometry_core.hpp on small graphs whose values are worked out by hand.
//
// The values below follow docs/SPEC.md ("StepStatistics geometry"). reference/
// verify_state_statistics.wls checks the same functions, through "StepStatistics", against
// reference/StateGeometryReference.wl and the Wolfram Function Repository on random states.

#include <gtest/gtest.h>

#include "hgcommon/state_geometry_core.hpp"
#include "hgcommon/state_invariants_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

struct Result {
    SgGeometry g;
    std::vector<double> ball;
};

Result geometry(std::vector<std::vector<uint32_t>> edges) {
    Edges el{std::move(edges)};
    Result r;
    std::vector<unsigned char> scratch(16);
    std::vector<double> ball(64);
    size_t needed = 0;
    while (!hgcommon::sg_state_geometry(el, scratch.data(), scratch.size(), r.g, ball.data(),
                                        static_cast<uint32_t>(ball.size()), needed)) {
        EXPECT_GT(needed, scratch.size());
        scratch.resize(needed);
    }
    r.ball.assign(ball.begin(), ball.begin() + r.g.ball_radii);
    return r;
}

constexpr double kTol = 1e-12;

}  // namespace

// P3 = a-b-c. Balls of radius 1: 2, 3, 2; R = 1. Each edge moves 1/4 of mass a distance 2.
TEST(StateGeometryCore, PathOfThree) {
    const Result r = geometry({{1, 2}, {2, 3}});
    ASSERT_TRUE(r.g.defined & hgcommon::SG_HAUSDORFF);
    EXPECT_EQ(r.g.vertex_count, 3u);
    EXPECT_EQ(r.g.edge_count, 2u);
    EXPECT_EQ(r.g.radius, 1);
    EXPECT_NEAR(r.g.mean_eccentricity, 5.0 / 3.0, kTol);
    EXPECT_NEAR(r.g.hausdorff_dimension, (2.0 + std::log2(3.0)) / 3.0, kTol);
    ASSERT_EQ(r.ball.size(), 1u);
    EXPECT_NEAR(r.ball[0], r.g.hausdorff_dimension, kTol);
    EXPECT_NEAR(r.g.ollivier_ricci, 0.5, kTol);
    EXPECT_NEAR(r.g.degree_entropy, -(2.0 / 3) * std::log2(2.0 / 3) - (1.0 / 3) * std::log2(1.0 / 3),
                kTol);
}

// One ternary hyperedge {1, 2, 3} gives the same graph as the path.
TEST(StateGeometryCore, TernaryHyperedgeIsAPath) {
    const Result a = geometry({{1, 2, 3}});
    const Result b = geometry({{7, 8}, {8, 9}});
    EXPECT_EQ(a.g.edge_count, b.g.edge_count);
    EXPECT_NEAR(a.g.hausdorff_dimension, b.g.hausdorff_dimension, kTol);
    EXPECT_NEAR(a.g.ollivier_ricci, b.g.ollivier_ricci, kTol);
    EXPECT_NEAR(a.g.mutual_information, b.g.mutual_information, kTol);
    EXPECT_NEAR(a.g.fisher_information, b.g.fisher_information, kTol);
}

// K3: every ball of radius 1 is the whole graph. Each edge moves 1/4 across itself.
TEST(StateGeometryCore, Triangle) {
    const Result r = geometry({{1, 2}, {2, 3}, {3, 1}});
    EXPECT_EQ(r.g.radius, 1);
    const double d = std::log2(3.0);
    EXPECT_NEAR(r.g.hausdorff_dimension, d, kTol);
    EXPECT_NEAR(r.g.ollivier_ricci, 0.75, kTol);
    EXPECT_NEAR(r.g.degree_entropy, 0.0, kTol);
    EXPECT_NEAR(r.g.local_entropy, 0.0, kTol);
    EXPECT_NEAR(r.g.mutual_information, 0.0, kTol);
    EXPECT_NEAR(r.g.fisher_information, 100.0, 1e-9);
    const double pi = 3.14159265358979323846;
    const double ricci = 6 * (d + 2) * (1 - 3 * std::tgamma(d / 2 + 1) / std::pow(pi, d / 2));
    EXPECT_NEAR(r.g.ricci_scalar, ricci, 1e-12 * std::fabs(ricci));
}

// K1,3: the centre's ball is 4, a leaf's is 2. On a centre-leaf edge the other two leaves' 1/3
// of mass travels distance 2.
TEST(StateGeometryCore, Star) {
    const Result r = geometry({{0, 1}, {0, 2}, {0, 3}});
    EXPECT_EQ(r.g.radius, 1);
    EXPECT_NEAR(r.g.mean_eccentricity, 7.0 / 4.0, kTol);
    EXPECT_NEAR(r.g.hausdorff_dimension, (2.0 + 3.0) / 4.0, kTol);
    EXPECT_NEAR(r.g.ollivier_ricci, 1.0 / 3.0, kTol);
    EXPECT_NEAR(r.g.degree_entropy, -0.25 * std::log2(0.25) - 0.75 * std::log2(0.75), kTol);
}

// Two K1,3 stars with their centres joined. The centre edge moves 3/8 across itself and 3/8
// from leaves to leaves at distance 3: W1 = 3/2, curvature -1/2. Each leaf edge moves 3/8 a
// distance 2: curvature 1/4. Mean over the 7 edges: 1/7.
TEST(StateGeometryCore, JoinedStarsHaveANegativeEdge) {
    const Result r = geometry({{0, 1}, {0, 2}, {0, 3}, {0, 10}, {10, 11}, {10, 12}, {10, 13}});
    EXPECT_EQ(r.g.edge_count, 7u);
    EXPECT_NEAR(r.g.ollivier_ricci, 1.0 / 7.0, kTol);
}

// C4: R = 2, balls 3 then 4 from every vertex; per edge, 1/4 moves across the edge and 1/4
// along the opposite edge.
TEST(StateGeometryCore, Square) {
    const Result r = geometry({{1, 2}, {2, 3}, {3, 4}, {4, 1}});
    EXPECT_EQ(r.g.radius, 2);
    const double t1 = std::log(3.0) / std::log(2.0);
    const double t2 = (std::log(4.0) - std::log(3.0)) / (std::log(3.0) - std::log(2.0));
    ASSERT_EQ(r.ball.size(), 2u);
    EXPECT_NEAR(r.ball[0], t1, kTol);
    EXPECT_NEAR(r.ball[1], t2, kTol);
    EXPECT_NEAR(r.g.hausdorff_dimension, (t1 + t2) / 2, kTol);
    EXPECT_NEAR(r.g.ollivier_ricci, 0.5, kTol);
}

// Two components: no radius and no dimension; each K2 edge has curvature 1.
TEST(StateGeometryCore, DisconnectedHasNoRadius) {
    const Result r = geometry({{1, 2}, {3, 4}});
    EXPECT_FALSE(r.g.defined & hgcommon::SG_RADIUS);
    EXPECT_FALSE(r.g.defined & hgcommon::SG_HAUSDORFF);
    EXPECT_FALSE(r.g.defined & hgcommon::SG_RICCI);
    EXPECT_FALSE(r.g.defined & hgcommon::SG_FISHER);
    ASSERT_TRUE(r.g.defined & hgcommon::SG_OLLIVIER);
    EXPECT_NEAR(r.g.ollivier_ricci, 1.0, kTol);
    EXPECT_TRUE(r.g.defined & hgcommon::SG_DEGREE_ENTROPY);
    EXPECT_EQ(r.ball.size(), 0u);
}

// A self-loop and a unary hyperedge give one isolated vertex: radius 0, nothing to average.
TEST(StateGeometryCore, OneVertex) {
    for (auto edges : {std::vector<std::vector<uint32_t>>{{5, 5}},
                       std::vector<std::vector<uint32_t>>{{5}}}) {
        const Result r = geometry(edges);
        EXPECT_EQ(r.g.vertex_count, 1u);
        EXPECT_EQ(r.g.edge_count, 0u);
        ASSERT_TRUE(r.g.defined & hgcommon::SG_RADIUS);
        EXPECT_EQ(r.g.radius, 0);
        EXPECT_FALSE(r.g.defined & hgcommon::SG_HAUSDORFF);
        EXPECT_FALSE(r.g.defined & hgcommon::SG_OLLIVIER);
        EXPECT_FALSE(r.g.defined & hgcommon::SG_MUTUAL_INFORMATION);
    }
}

// No hyperedges: nothing is defined.
TEST(StateGeometryCore, EmptyState) {
    const Result r = geometry({});
    EXPECT_EQ(r.g.defined, 0u);
}

// Renaming vertices changes nothing: the values are invariants of the isomorphism class.
TEST(StateGeometryCore, IndependentOfVertexNames) {
    const Result a = geometry({{1, 2}, {2, 3}, {3, 1}, {3, 4}, {4, 5, 6}});
    const Result b = geometry({{90, 20}, {20, 7}, {7, 90}, {7, 400}, {400, 3, 11}});
    EXPECT_EQ(a.g.defined, b.g.defined);
    EXPECT_NEAR(a.g.hausdorff_dimension, b.g.hausdorff_dimension, kTol);
    EXPECT_NEAR(a.g.ricci_scalar, b.g.ricci_scalar, 1e-9);
    EXPECT_NEAR(a.g.ollivier_ricci, b.g.ollivier_ricci, kTol);
    EXPECT_NEAR(a.g.local_entropy, b.g.local_entropy, kTol);
    EXPECT_NEAR(a.g.mutual_information, b.g.mutual_information, kTol);
    EXPECT_NEAR(a.g.fisher_information, b.g.fisher_information, 1e-9);
}

// A short buffer computes nothing and names a size; the retried call succeeds.
TEST(StateGeometryCore, ShortScratchReportsWhatItNeeds) {
    Edges el{{{1, 2}, {2, 3}, {3, 4}, {4, 1}}};
    SgGeometry g;
    double ball[8];
    size_t needed = 0;
    std::vector<unsigned char> small(8);
    EXPECT_FALSE(hgcommon::sg_state_geometry(el, small.data(), small.size(), g, ball, 8, needed));
    std::vector<unsigned char> big(needed);
    if (!hgcommon::sg_state_geometry(el, big.data(), big.size(), g, ball, 8, needed)) {
        big.resize(needed);
        ASSERT_TRUE(hgcommon::sg_state_geometry(el, big.data(), big.size(), g, ball, 8, needed));
    }
    EXPECT_EQ(g.radius, 2);
    EXPECT_LE(needed, big.size());
}

namespace {

// hgcommon::state_record of a state given as its edge list, the list standing in for its
// canonical form: the record's geometry with the largest component and the distributions.
struct Recorded {
    hgcommon::SiResult r;
    std::vector<uint64_t> scratch;
};

Recorded recorded(const std::vector<std::vector<uint32_t>>& edges) {
    std::vector<uint32_t> off(1, 0), verts;
    for (const auto& e : edges) {
        verts.insert(verts.end(), e.begin(), e.end());
        off.push_back(static_cast<uint32_t>(verts.size()));
    }
    const uint32_t m = static_cast<uint32_t>(edges.size());
    Recorded out;
    uint64_t bytes = hgcommon::si_record_bytes_hint(off[m], m, 1);
    for (;;) {
        out.scratch.assign((bytes + 7) / 8, 0);
        uint64_t needed = 0;
        if (hgcommon::state_record(off.data(), verts.data(), m, off.data(), verts.data(), m,
                                   reinterpret_cast<unsigned char*>(out.scratch.data()), bytes,
                                   needed, out.r))
            return out;
        bytes = needed > bytes ? needed : 2 * bytes;
    }
}

// P4 = 1-2-3-4, R = 2. The ends have balls 2, 3 and the middle vertices 3, 4, so d_end = 1 and
// d_mid = (log 3 / log 2 + (log 4 - log 3) / (log 3 - log 2)) / 2. The edge curvatures are 1/2,
// 0, 1/2, so k = 1/2, 1/4, 1/4, 1/2.
const double kDEnd = 1.0;
const double kDMid = (std::log(3.0) / std::log(2.0) +
                      (std::log(4.0) - std::log(3.0)) / (std::log(3.0) - std::log(2.0))) / 2;

}  // namespace

// R1 and R3 on P4: the mean, max and population SD of d_v; Moran's I of k is
// (4/3) (-1/64) / (1/16) = -1/3; k falls exactly as the degree rises, correlation -1.
TEST(StateGeometryCore, PathLocalDimensionAndCurvatureCorrelation) {
    const Recorded rec = recorded({{1, 2}, {2, 3}, {3, 4}});
    const SgGeometry& g = rec.r.g;
    ASSERT_TRUE(g.defined & hgcommon::SG_LARGEST_DIMENSION);
    EXPECT_EQ(g.largest_dimension, g.hausdorff_dimension);   // connected: G' = G, same sum
    EXPECT_NEAR(g.largest_dimension, (kDEnd + kDMid) / 2, kTol);
    EXPECT_NEAR(g.local_dimension_max, kDMid, kTol);
    EXPECT_NEAR(g.local_dimension_sd, (kDMid - kDEnd) / 2, kTol);
    ASSERT_TRUE(g.defined & hgcommon::SG_OLLIVIER_MORAN);
    EXPECT_NEAR(g.ollivier_moran_i, -1.0 / 3.0, kTol);
    ASSERT_TRUE(g.defined & hgcommon::SG_OLLIVIER_DEGREE);
    EXPECT_NEAR(g.ollivier_degree_correlation, -1.0, kTol);

    const hgcommon::SgDistribution* d = rec.r.dists;
    const auto& dim = d[hgcommon::SG_DIST_LOCAL_DIMENSION];
    EXPECT_EQ(dim.count, 4u);
    EXPECT_NEAR(dim.sum[0], 2 * (kDEnd + kDMid), kTol);
    EXPECT_NEAR(dim.sum[1], 2 * (kDEnd * kDEnd + kDMid * kDMid), kTol);
    EXPECT_EQ(dim.min, kDEnd);
    EXPECT_EQ(dim.bins[4], 4u);   // [1, 1.25): bin width 1/4 over [0, 8)
    const auto& oll = d[hgcommon::SG_DIST_OLLIVIER];
    EXPECT_EQ(oll.count, 4u);
    EXPECT_NEAR(oll.sum[0], 1.5, kTol);
    EXPECT_EQ(oll.min, 0.25);
    EXPECT_EQ(oll.max, 0.5);
    // Bin width 3/32 over [-2, 1): 1/4 lies in bin 24 = [0.25, 0.34375), 1/2 in bin 26.
    EXPECT_EQ(oll.bins[24], 2u);
    EXPECT_EQ(oll.bins[26], 2u);
    // The mean of K_v over G is the state's Ricci scalar.
    const auto& ric = d[hgcommon::SG_DIST_RICCI];
    EXPECT_EQ(ric.count, 4u);
    EXPECT_NEAR(ric.sum[0] / ric.count, g.ricci_scalar, 1e-12);
}

// C6 is vertex-transitive: every d_v is the same, and k is constant, so Moran's I and the degree
// correlation are undefined.
TEST(StateGeometryCore, CycleHasNoCurvatureCorrelation) {
    const Recorded rec = recorded({{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 1}});
    const SgGeometry& g = rec.r.g;
    ASSERT_TRUE(g.defined & hgcommon::SG_LARGEST_DIMENSION);
    EXPECT_EQ(g.largest_dimension, g.hausdorff_dimension);
    EXPECT_EQ(g.local_dimension_max, rec.r.dists[hgcommon::SG_DIST_LOCAL_DIMENSION].min);
    EXPECT_NEAR(g.local_dimension_sd, 0.0, kTol);
    EXPECT_FALSE(g.defined & hgcommon::SG_OLLIVIER_MORAN);
    EXPECT_FALSE(g.defined & hgcommon::SG_OLLIVIER_DEGREE);
    EXPECT_EQ(rec.r.dists[hgcommon::SG_DIST_OLLIVIER].count, 6u);
}

// A 3x3 grid: G' = G, the K_v average to the Ricci scalar, and the d_v to the dimension.
TEST(StateGeometryCore, GridDistributionsAverageToTheStateValues) {
    std::vector<std::vector<uint32_t>> e;
    for (uint32_t i = 0; i < 3; ++i)
        for (uint32_t j = 0; j < 3; ++j) {
            if (j + 1 < 3) e.push_back({3 * i + j, 3 * i + j + 1});
            if (i + 1 < 3) e.push_back({3 * i + j, 3 * i + j + 3});
        }
    const Recorded rec = recorded(e);
    const SgGeometry& g = rec.r.g;
    const auto* d = rec.r.dists;
    EXPECT_EQ(d[hgcommon::SG_DIST_LOCAL_DIMENSION].count, 9u);
    EXPECT_NEAR(d[hgcommon::SG_DIST_LOCAL_DIMENSION].sum[0] / 9, g.hausdorff_dimension, kTol);
    EXPECT_NEAR(d[hgcommon::SG_DIST_RICCI].sum[0] / 9, g.ricci_scalar, 1e-12);
    // Corner, edge-middle and centre vertices: the curvature is not constant, so both are defined.
    EXPECT_TRUE(g.defined & hgcommon::SG_OLLIVIER_MORAN);
    EXPECT_TRUE(g.defined & hgcommon::SG_OLLIVIER_DEGREE);
    uint32_t binned = 0;
    for (uint32_t b = 0; b < hgcommon::SG_DIST_BINS; ++b)
        binned += d[hgcommon::SG_DIST_RICCI].bins[b];
    EXPECT_EQ(binned, 9u);
}

// Three components, as in the Brill-Lindquist states: P4 (7 incidence nodes), a triangle (6) and
// one edge (3). The whole-state dimension is undefined; the largest component's is P4's. The
// curvature distribution covers every vertex with a neighbour.
TEST(StateGeometryCore, DisconnectedUsesTheLargestComponent) {
    const Recorded rec = recorded({{10, 11}, {5, 6}, {6, 7}, {7, 5}, {1, 2}, {2, 3}, {3, 4}});
    const SgGeometry& g = rec.r.g;
    EXPECT_FALSE(g.defined & hgcommon::SG_HAUSDORFF);
    ASSERT_TRUE(g.defined & hgcommon::SG_LARGEST_DIMENSION);
    EXPECT_NEAR(g.largest_dimension, (kDEnd + kDMid) / 2, kTol);
    EXPECT_NEAR(g.local_dimension_max, kDMid, kTol);
    EXPECT_EQ(rec.r.dists[hgcommon::SG_DIST_LOCAL_DIMENSION].count, 4u);
    EXPECT_EQ(rec.r.dists[hgcommon::SG_DIST_RICCI].count, 4u);
    EXPECT_EQ(rec.r.dists[hgcommon::SG_DIST_OLLIVIER].count, 9u);
    EXPECT_EQ(rec.r.v.components, 3);
}

// Two components of 7 incidence nodes, the star K1,3 and P4: the incidence diameter breaks the
// tie (P4's is 6, the star's 4), whichever comes first.
TEST(StateGeometryCore, LargestComponentTieFollowsTheIncidenceRule) {
    for (const auto& e : {std::vector<std::vector<uint32_t>>{{1, 2}, {1, 3}, {1, 4},
                                                             {5, 6}, {6, 7}, {7, 8}},
                          std::vector<std::vector<uint32_t>>{{1, 2}, {2, 3}, {3, 4},
                                                             {5, 6}, {5, 7}, {5, 8}}}) {
        const Recorded rec = recorded(e);
        ASSERT_TRUE(rec.r.g.defined & hgcommon::SG_LARGEST_DIMENSION);
        EXPECT_NEAR(rec.r.g.largest_dimension, (kDEnd + kDMid) / 2, kTol);
        EXPECT_EQ(rec.r.v.incidence_diameter, 6);
    }
}

// The fixed histogram ranges: a value below the range counts in bin 0, above it in the last.
TEST(StateGeometryCore, DistributionBinsClampToTheRange) {
    EXPECT_EQ(hgcommon::sg_dist_bin(hgcommon::SG_DIST_OLLIVIER, -5.0), 0u);
    EXPECT_EQ(hgcommon::sg_dist_bin(hgcommon::SG_DIST_OLLIVIER, 1.0), hgcommon::SG_DIST_BINS - 1);
    EXPECT_EQ(hgcommon::sg_dist_bin(hgcommon::SG_DIST_RICCI, -24.0), 0u);
    EXPECT_EQ(hgcommon::sg_dist_bin(hgcommon::SG_DIST_RICCI, -23.0), 1u);
    EXPECT_EQ(hgcommon::sg_dist_bin(hgcommon::SG_DIST_LOCAL_DIMENSION, 100.0),
              hgcommon::SG_DIST_BINS - 1);
}

namespace {
// Distance in units in the last place between two finite doubles of the same sign.
int64_t ulps(double a, double b) {
    int64_t ia = 0, ib = 0;
    std::memcpy(&ia, &a, 8);
    std::memcpy(&ib, &b, 8);
    return ia > ib ? ia - ib : ib - ia;
}
}  // namespace

// det_math.hpp against glibc over the arguments the geometry passes: ball sizes and radii to
// log, probabilities and overlap ratios to log2, dimensions to pow and tgamma.
TEST(DetMath, AgreesWithTheLibrary) {
    int64_t worst_log = 0, worst_log2 = 0, worst_exp = 0;
    double worst_gamma = 0.0;
    for (uint32_t i = 1; i <= 200000; ++i) {
        const double x = static_cast<double>(i);
        worst_log = std::max(worst_log, ulps(hgcommon::dm_log(x), std::log(x)));
        const double p = 1.0 / x;
        if (i > 1) worst_log2 = std::max(worst_log2, ulps(hgcommon::dm_log2(p), std::log2(p)));
        const double q = 1.0 + 3.0 / (x + 7.0);
        worst_log2 = std::max(worst_log2, ulps(hgcommon::dm_log2(q), std::log2(q)));
        const double y = -40.0 + 80.0 * x / 200000.0;
        worst_exp = std::max(worst_exp, ulps(hgcommon::dm_exp(y), std::exp(y)));
        const double z = 1.0 + 11.0 * x / 200000.0;
        worst_gamma = std::max(worst_gamma,
                               std::fabs(hgcommon::dm_tgamma(z) / std::tgamma(z) - 1.0));
    }
    EXPECT_LE(worst_log, 2);
    EXPECT_LE(worst_log2, 3);
    EXPECT_LE(worst_exp, 2);
    EXPECT_LE(worst_gamma, 1.5e-15);
    EXPECT_EQ(hgcommon::dm_log(1.0), 0.0);
    EXPECT_EQ(hgcommon::dm_log2(0.25), -2.0);
    EXPECT_EQ(hgcommon::dm_log2(1024.0), 10.0);
    EXPECT_EQ(hgcommon::dm_pow(1.0, 2.7), 1.0);
    EXPECT_EQ(hgcommon::dm_tgamma(1.0), 1.0);
    EXPECT_EQ(hgcommon::dm_tgamma(2.0), 1.0);
    EXPECT_EQ(hgcommon::dm_tgamma(5.0), 24.0);
    std::printf("[ DetMath  ] worst: log %lld ulp, log2 %lld ulp, exp %lld ulp, tgamma %.3g\n",
                static_cast<long long>(worst_log), static_cast<long long>(worst_log2),
                static_cast<long long>(worst_exp), worst_gamma);
}
