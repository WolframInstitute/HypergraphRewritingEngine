#pragma once
#include "hgcommon/namespace.hpp"
// EQUILIBRIUM OVER STEPS (docs/SPEC.md, "StepStatistics equilibrium"), one body for every caller.
// hypergraph_viz's equilibrium detection (blackhole/src/equilibrium_analysis.cpp,
// compute_stability_scores and detect_equilibrium) in double precision. Each step's scores read
// the last EQ_WINDOW steps up to and including it, so a step's scores are final once the steps
// before it have closed, and paclet_support.cpp computes them as each step closes.

#include <math.h>

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

constexpr uint32_t EQ_WINDOW = 20;        // steps per window
constexpr uint32_t EQ_MIN_STEPS = 50;     // steps before a step can be in equilibrium
constexpr double EQ_THRESHOLD = 0.95;     // the least overall stability in equilibrium
// The series, per step: the mean WolframHausdorffDimension, the mean DegreeEntropy and the mean
// VertexCount of the step's states.
constexpr uint32_t EQ_SERIES = 3;
enum : uint32_t { EQ_DIMENSION = 0, EQ_DEGREE_ENTROPY = 1, EQ_SIZE = 2 };

struct EqScores {
    double stability[EQ_SERIES] = {0.0, 0.0, 0.0};
    double overall = 0.0;
    double dimension_trend = 0.0;
    double size_trend = 0.0;
    uint32_t equilibrated = 0;
    uint32_t window = 0;
};

// 1 / (1 + s / |m|) over x[0..n), s the sample standard deviation (0 for n < 2) and m the mean;
// 1 when |m| < 1e-10.
HG_HD inline double eq_stability(const double* x, uint32_t n, uint32_t stride) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) sum += x[i * stride];
    const double mean = sum / n;
    if (::fabs(mean) < 1e-10) return 1.0;
    double var = 0.0;
    if (n >= 2) {
        for (uint32_t i = 0; i < n; ++i) {
            const double d = x[i * stride] - mean;
            var += d * d;
        }
        var /= (n - 1);
    }
    return 1.0 / (1.0 + ::sqrt(var) / ::fabs(mean));
}

// The least-squares slope of x[i] against i over i = 0..n-1; 0 for n < 2.
HG_HD inline double eq_slope(const double* x, uint32_t n, uint32_t stride) {
    if (n < 2) return 0.0;
    double sx = 0.0, sy = 0.0, sxy = 0.0, sxx = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double xi = static_cast<double>(i), y = x[i * stride];
        sx += xi;
        sy += y;
        sxy += xi * y;
        sxx += xi * xi;
    }
    const double nn = static_cast<double>(n);
    return (nn * sxy - sx * sy) / (nn * sxx - sx * sx);
}

// The scores of a step from `rows`, the EQ_SERIES values of the last `n` steps, oldest first,
// n <= EQ_WINDOW; `steps` is the number of steps up to and including this one.
HG_HD inline EqScores eq_scores(const double* rows, uint32_t n, uint32_t steps) {
    EqScores s;
    s.window = n;
    double total = 0.0;
    for (uint32_t k = 0; k < EQ_SERIES; ++k) {
        s.stability[k] = eq_stability(rows + k, n, EQ_SERIES);
        total += s.stability[k];
    }
    s.overall = total / EQ_SERIES;
    s.dimension_trend = eq_slope(rows + EQ_DIMENSION, n, EQ_SERIES);
    s.size_trend = eq_slope(rows + EQ_SIZE, n, EQ_SERIES);
    s.equilibrated = steps >= EQ_MIN_STEPS && s.overall >= EQ_THRESHOLD ? 1u : 0u;
    return s;
}

}  // namespace common
}  // namespace HG_NAMESPACE
