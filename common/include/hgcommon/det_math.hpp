#pragma once
#include "hgcommon/namespace.hpp"
// log, log2, exp, pow and tgamma with one result on the host and the device, for the values
// that both engines report (state_geometry_core.hpp).
//
// The libraries' log, pow and tgamma (glibc, MSVC, CUDA) differ from one another in the last
// bits. These bodies use only IEEE addition, subtraction, multiplication, division and fused
// multiply-add, rounded to nearest, plus frexp, ldexp and floor, which are exact. Every product goes through dm_mul,
// which no compiler fuses into a multiply-add: on the device it is __dmul_rn; under GCC the
// product sits behind __builtin_assoc_barrier. Clang fuses only within one expression under its
// default -ffp-contract=on, and MSVC's /fp:precise does not fuse. So a body that writes every
// product as dm_mul evaluates the same operations in the same order on every target, and the
// result is the same double.
//
// Accuracy, measured against glibc over the arguments the geometry passes (DetMath in
// test_state_geometry_core.cpp): dm_log and dm_exp within 1 ulp, dm_log2 within 3 ulp, dm_tgamma
// within 1.2e-15 relative on [1, 12].

#include <math.h>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

HG_HD inline double dm_mul(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#elif defined(__has_builtin) && !defined(__CUDACC__)
#if __has_builtin(__builtin_assoc_barrier)
    return __builtin_assoc_barrier(a * b);
#else
    return a * b;
#endif
#else
    return a * b;
#endif
}

// p * x + c, the multiply and the add rounded separately.
HG_HD inline double dm_madd(double p, double x, double c) { return dm_mul(p, x) + c; }

// log(m) for m in [sqrt(1/2), sqrt(2)): 2 atanh(f), f = (m - 1) / (m + 1), |f| < 0.1716, as
// 2 f (1 + f^2/3 + f^4/5 + ...) through f^24; the first omitted term is below 2^-56 of the sum.
HG_HD inline double dm_log_reduced(double m) {
    const double f = (m - 1.0) / (m + 1.0);
    const double z = dm_mul(f, f);
    double p = 1.0 / 25.0;
    p = dm_madd(p, z, 1.0 / 23.0);
    p = dm_madd(p, z, 1.0 / 21.0);
    p = dm_madd(p, z, 1.0 / 19.0);
    p = dm_madd(p, z, 1.0 / 17.0);
    p = dm_madd(p, z, 1.0 / 15.0);
    p = dm_madd(p, z, 1.0 / 13.0);
    p = dm_madd(p, z, 1.0 / 11.0);
    p = dm_madd(p, z, 1.0 / 9.0);
    p = dm_madd(p, z, 1.0 / 7.0);
    p = dm_madd(p, z, 1.0 / 5.0);
    p = dm_madd(p, z, 1.0 / 3.0);
    const double t = dm_mul(dm_mul(f, z), p);   // f^3/3 + f^5/5 + ...
    return 2.0 * f + 2.0 * t;                   // doubling is exact
}

// x = m 2^e with m in [sqrt(1/2), sqrt(2)). x is positive and finite.
HG_HD inline double dm_split(double x, int& e) {
    double m = ::frexp(x, &e);   // m in [1/2, 1)
    if (m < 0.70710678118654752440) { m = m + m; --e; }
    return m;
}

// Natural logarithm of a positive finite x. ln 2 is split as ln2_hi + ln2_lo with ln2_hi
// holding 32 significant bits, so e * ln2_hi is exact for every double exponent.
HG_HD inline double dm_log(double x) {
    int e = 0;
    const double m = dm_split(x, e);
    const double ln2_hi = 6.93147180369123816490e-01;
    const double ln2_lo = 1.90821492927058770002e-10;
    const double de = static_cast<double>(e);
    return dm_mul(de, ln2_hi) + (dm_log_reduced(m) + dm_mul(de, ln2_lo));
}

// Base-2 logarithm of a positive finite x: e + log(m) / ln 2. Exact at powers of two.
HG_HD inline double dm_log2(double x) {
    int e = 0;
    const double m = dm_split(x, e);
    const double inv_ln2 = 1.44269504088896338700e+00;
    return static_cast<double>(e) + dm_mul(dm_log_reduced(m), inv_ln2);
}

// e^y for |y| < 700: y = k ln 2 + r with |r| <= ln 2 / 2, e^r by its Taylor series through r^15
// (the first omitted term is below 2^-60), then scaled by 2^k.
HG_HD inline double dm_exp(double y) {
    const double ln2_hi = 6.93147180369123816490e-01;
    const double ln2_lo = 1.90821492927058770002e-10;
    const double inv_ln2 = 1.44269504088896338700e+00;
    const double k = ::floor(dm_mul(y, inv_ln2) + 0.5);
    const double r = (y - dm_mul(k, ln2_hi)) - dm_mul(k, ln2_lo);
    double p = 1.0 / 1307674368000.0;   // 1/15!
    p = dm_madd(p, r, 1.0 / 87178291200.0);
    p = dm_madd(p, r, 1.0 / 6227020800.0);
    p = dm_madd(p, r, 1.0 / 479001600.0);
    p = dm_madd(p, r, 1.0 / 39916800.0);
    p = dm_madd(p, r, 1.0 / 3628800.0);
    p = dm_madd(p, r, 1.0 / 362880.0);
    p = dm_madd(p, r, 1.0 / 40320.0);
    p = dm_madd(p, r, 1.0 / 5040.0);
    p = dm_madd(p, r, 1.0 / 720.0);
    p = dm_madd(p, r, 1.0 / 120.0);
    p = dm_madd(p, r, 1.0 / 24.0);
    p = dm_madd(p, r, 1.0 / 6.0);
    p = dm_madd(p, r, 0.5);
    p = dm_madd(p, r, 1.0);
    p = dm_madd(p, r, 1.0);
    return ::ldexp(p, static_cast<int>(k));
}

// a^y for positive finite a: e^(y log a). Exactly 1 at a = 1 or y = 0.
HG_HD inline double dm_pow(double a, double y) { return dm_exp(dm_mul(y, dm_log(a))); }

// Gamma(z) for z >= 1/2. z = n + x with n = round(z) and x in [-1/2, 1/2]; 1/Gamma(1 + x) is
// the series sum_k c_k x^(k-1) of Abramowitz and Stegun 6.1.34 (Wrench's coefficients), and
// Gamma(z) = Gamma(1 + x) (x + 1) ... (x + n - 1).
HG_HD inline double dm_tgamma(double z) {
    const double n = ::floor(z + 0.5);
    const double x = z - n;
    double p = 0.0000000000000001;
    p = dm_madd(p, x, 0.0000000000000014);
    p = dm_madd(p, x, -0.0000000000000054);
    p = dm_madd(p, x, -0.0000000000000206);
    p = dm_madd(p, x, 0.0000000000005100);
    p = dm_madd(p, x, -0.0000000000036968);
    p = dm_madd(p, x, 0.0000000000077823);
    p = dm_madd(p, x, 0.0000000001043427);
    p = dm_madd(p, x, -0.0000000011812746);
    p = dm_madd(p, x, 0.0000000050020075);
    p = dm_madd(p, x, 0.0000000061160950);
    p = dm_madd(p, x, -0.0000002056338417);
    p = dm_madd(p, x, 0.0000011330272320);
    p = dm_madd(p, x, -0.0000012504934821);
    p = dm_madd(p, x, -0.0000201348547807);
    p = dm_madd(p, x, 0.0001280502823882);
    p = dm_madd(p, x, -0.0002152416741149);
    p = dm_madd(p, x, -0.0011651675918591);
    p = dm_madd(p, x, 0.0072189432466630);
    p = dm_madd(p, x, -0.0096219715278770);
    p = dm_madd(p, x, -0.0421977345555443);
    p = dm_madd(p, x, 0.1665386113822915);
    p = dm_madd(p, x, -0.0420026350340952);
    p = dm_madd(p, x, -0.6558780715202538);
    p = dm_madd(p, x, 0.5772156649015329);
    p = dm_madd(p, x, 1.0);
    double g = 1.0 / p;
    for (double i = 1.0; i < n; i += 1.0) g = dm_mul(g, x + i);
    return g;
}

// DOUBLE-DOUBLE ARITHMETIC for the power sums of the per-vertex distributions
// (state_geometry_core.hpp, sg_dist_add) and the moments pooled from them: a value is hi + lo
// with |lo| at most half an ulp of hi, about 106 bits. The central moments come from power sums
// by a cancellation that multiplies the rounding error by about (mean / standard deviation)^4:
// from double power sums, a kurtosis of values with mean 0.43 and standard deviation 0.0076
// was 1.3e-9 relative from its exact value. dm_fma rounds once on every
// target, so two_prod is exact everywhere and the host and the device give the same pair.
HG_HD inline double dm_fma(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
    return __fma_rn(a, b, c);
#else
    return ::fma(a, b, c);
#endif
}

struct DmDD {
    double hi = 0.0;
    double lo = 0.0;
};

// a + b as hi + lo, exactly (Knuth).
HG_HD inline DmDD dm_two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

// a + b as hi + lo for |a| >= |b|, exactly (Dekker).
HG_HD inline DmDD dm_quick_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

// a * b as hi + lo, exactly.
HG_HD inline DmDD dm_two_prod(double a, double b) {
    const double p = dm_mul(a, b);
    return {p, dm_fma(a, b, -p)};
}

HG_HD inline DmDD dm_dd_add(DmDD x, DmDD y) {
    DmDD s = dm_two_sum(x.hi, y.hi);
    const DmDD t = dm_two_sum(x.lo, y.lo);
    s = dm_quick_two_sum(s.hi, s.lo + t.hi);
    return dm_quick_two_sum(s.hi, s.lo + t.lo);
}

HG_HD inline DmDD dm_dd_neg(DmDD x) { return {-x.hi, -x.lo}; }

HG_HD inline DmDD dm_dd_sub(DmDD x, DmDD y) { return dm_dd_add(x, dm_dd_neg(y)); }

HG_HD inline DmDD dm_dd_mul(DmDD x, DmDD y) {
    const DmDD p = dm_two_prod(x.hi, y.hi);
    return dm_quick_two_sum(p.hi, p.lo + (dm_mul(x.hi, y.lo) + dm_mul(x.lo, y.hi)));
}

HG_HD inline DmDD dm_dd_div(DmDD x, DmDD y) {
    const double q1 = x.hi / y.hi;
    DmDD r = dm_dd_sub(x, dm_dd_mul(y, DmDD{q1, 0.0}));
    const double q2 = r.hi / y.hi;
    r = dm_dd_sub(r, dm_dd_mul(y, DmDD{q2, 0.0}));
    const double q3 = r.hi / y.hi;
    return dm_dd_add(dm_quick_two_sum(q1, q2), DmDD{q3, 0.0});
}

// An unsigned 64-bit integer as hi + lo, exactly.
HG_HD inline DmDD dm_dd_of(uint64_t n) {
    return dm_two_sum(static_cast<double>(n >> 32) * 4294967296.0,
                      static_cast<double>(n & 0xFFFFFFFFu));
}

}  // namespace common
}  // namespace HG_NAMESPACE
