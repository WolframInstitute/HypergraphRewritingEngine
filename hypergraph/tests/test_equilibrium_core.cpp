// hgcommon/equilibrium_core.hpp (docs/SPEC.md, "StepStatistics equilibrium") on series with
// known scores. paclet_source/tests/test_wxf_serialization.cpp checks the per-step key through
// "StepStatistics", and reference/verify_state_statistics.wls against its own computation.

#include <gtest/gtest.h>

#include "hgcommon/equilibrium_core.hpp"

#include <cmath>
#include <vector>

namespace {

std::vector<double> rows_of(const std::vector<double>& x, const std::vector<double>& y,
                            const std::vector<double>& z) {
    std::vector<double> r;
    for (size_t i = 0; i < x.size(); ++i) {
        r.push_back(x[i]);
        r.push_back(y[i]);
        r.push_back(z[i]);
    }
    return r;
}

}  // namespace

// Constant series are stable; one step has stability 1.
TEST(Equilibrium, ConstantSeriesAreStable) {
    const auto rows = rows_of({2, 2, 2, 2}, {1.5, 1.5, 1.5, 1.5}, {10, 10, 10, 10});
    const hgcommon::EqScores s = hgcommon::eq_scores(rows.data(), 4, 60);
    for (double v : s.stability) EXPECT_EQ(v, 1.0);
    EXPECT_EQ(s.overall, 1.0);
    EXPECT_EQ(s.dimension_trend, 0.0);
    EXPECT_EQ(s.window, 4u);
    EXPECT_EQ(s.equilibrated, 1u);
    // Fewer than 50 steps: not in equilibrium.
    EXPECT_EQ(hgcommon::eq_scores(rows.data(), 4, 49).equilibrated, 0u);
    const hgcommon::EqScores one = hgcommon::eq_scores(rows.data(), 1, 1);
    EXPECT_EQ(one.overall, 1.0);
}

// x = 1, 2, 3: mean 2, sample standard deviation 1, stability 1 / (1 + 1/2) = 2/3, slope 1.
// A zero mean gives stability 1.
TEST(Equilibrium, KnownScores) {
    const auto rows = rows_of({1, 2, 3}, {0, 0, 0}, {4, 4, 4});
    const hgcommon::EqScores s = hgcommon::eq_scores(rows.data(), 3, 3);
    EXPECT_NEAR(s.stability[hgcommon::EQ_DIMENSION], 2.0 / 3.0, 1e-15);
    EXPECT_EQ(s.stability[hgcommon::EQ_DEGREE_ENTROPY], 1.0);
    EXPECT_EQ(s.stability[hgcommon::EQ_SIZE], 1.0);
    EXPECT_NEAR(s.overall, (2.0 / 3.0 + 2.0) / 3.0, 1e-15);
    EXPECT_NEAR(s.dimension_trend, 1.0, 1e-15);
    EXPECT_EQ(s.size_trend, 0.0);
    EXPECT_EQ(s.equilibrated, 0u);
}
