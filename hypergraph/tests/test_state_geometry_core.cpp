// hgcommon/state_geometry_core.hpp on small graphs whose values are worked out by hand.
//
// The values below follow docs/SPEC.md ("StepStatistics geometry"). reference/
// verify_state_statistics.wls checks the same functions, through "StepStatistics", against
// reference/StateGeometryReference.wl and the Wolfram Function Repository on random states.

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
