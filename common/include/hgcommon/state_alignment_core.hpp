#pragma once
// BRANCH ALIGNMENT OF A STATE'S GRAPH (docs/SPEC.md, "StepStatistics branch alignment"), one body
// for the host and the device: hypergraph_viz's curvature-weighted spectral embedding and moment
// of inertia (hypergraph_viz@0fb63e5:blackhole/src/branch_alignment.cpp:29-205 and 410-516)
// with the exact eigenvectors of the Laplacian. state_geometry_core.hpp includes this file after
// SgGeometry, and sg_geometry_of calls sa_alignment_of after it has the vertex Ricci scalars.
//
// The Laplacian L = D - A of G' is diagonalised by Householder reduction to a tridiagonal matrix
// and the implicit QL iteration (sa_tridiagonalise, sa_ql), every product through dm_mul, on the
// calling thread: the same operations in the same order on every target. Its cost is
// O(n'^3) time and n'^2 doubles for G' of n' vertices, so the family is defined for
// n' <= SA_MAX_VERTICES.

namespace HG_NAMESPACE {
namespace common {

constexpr uint32_t SA_MAX_VERTICES = 256;
constexpr uint32_t SA_QL_ITERATIONS = 60;   // per eigenvalue; past it the state is undefined
constexpr double SA_GAP = 1e-6;             // the least relative gap between mu4 and mu5

// sqrt(a^2 + b^2) without overflow.
HG_HD inline double sa_hypot(double a, double b) {
    const double x = ::fabs(a), y = ::fabs(b);
    if (x > y) {
        const double t = y / x;
        return dm_mul(x, ::sqrt(1.0 + dm_mul(t, t)));
    }
    if (y == 0.0) return 0.0;
    const double t = x / y;
    return dm_mul(y, ::sqrt(1.0 + dm_mul(t, t)));
}

// Householder reduction of the symmetric n x n matrix a (row-major) to tridiagonal form: d the
// diagonal, e the subdiagonal in e[1..n-1], and a the orthogonal matrix of the reduction.
HG_HD inline void sa_tridiagonalise(double* a, uint32_t n, double* d, double* e) {
    auto A = [&](uint32_t i, uint32_t j) -> double& { return a[static_cast<size_t>(i) * n + j]; };
    for (uint32_t i = n - 1; i > 0; --i) {
        const uint32_t l = i - 1;
        double h = 0.0;
        if (l > 0) {
            double scale = 0.0;
            for (uint32_t k = 0; k <= l; ++k) scale += ::fabs(A(i, k));
            if (scale == 0.0) {
                e[i] = A(i, l);
            } else {
                for (uint32_t k = 0; k <= l; ++k) {
                    A(i, k) /= scale;
                    h += dm_mul(A(i, k), A(i, k));
                }
                double f = A(i, l);
                double g = f >= 0.0 ? -::sqrt(h) : ::sqrt(h);
                e[i] = dm_mul(scale, g);
                h -= dm_mul(f, g);
                A(i, l) = f - g;
                f = 0.0;
                for (uint32_t j = 0; j <= l; ++j) {
                    A(j, i) = A(i, j) / h;
                    g = 0.0;
                    for (uint32_t k = 0; k <= j; ++k) g += dm_mul(A(j, k), A(i, k));
                    for (uint32_t k = j + 1; k <= l; ++k) g += dm_mul(A(k, j), A(i, k));
                    e[j] = g / h;
                    f += dm_mul(e[j], A(i, j));
                }
                const double hh = f / (h + h);
                for (uint32_t j = 0; j <= l; ++j) {
                    f = A(i, j);
                    g = e[j] - dm_mul(hh, f);
                    e[j] = g;
                    for (uint32_t k = 0; k <= j; ++k)
                        A(j, k) -= dm_mul(f, e[k]) + dm_mul(g, A(i, k));
                }
            }
        } else {
            e[i] = A(i, l);
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        if (d[i] != 0.0) {
            for (uint32_t j = 0; j < i; ++j) {
                double g = 0.0;
                for (uint32_t k = 0; k < i; ++k) g += dm_mul(A(i, k), A(k, j));
                for (uint32_t k = 0; k < i; ++k) A(k, j) -= dm_mul(g, A(k, i));
            }
        }
        d[i] = A(i, i);
        A(i, i) = 1.0;
        for (uint32_t j = 0; j < i; ++j) {
            A(j, i) = 0.0;
            A(i, j) = 0.0;
        }
    }
}

// The implicit QL iteration on the tridiagonal (d, e) of sa_tridiagonalise, accumulating the
// rotations into z (n x n, row-major): afterwards d holds the eigenvalues and column i of z the
// eigenvector of d[i]. False when an eigenvalue takes more than SA_QL_ITERATIONS iterations.
HG_HD inline bool sa_ql(double* d, double* e, uint32_t n, double* z) {
    auto Z = [&](uint32_t i, uint32_t j) -> double& { return z[static_cast<size_t>(i) * n + j]; };
    for (uint32_t i = 1; i < n; ++i) e[i - 1] = e[i];
    e[n - 1] = 0.0;
    for (uint32_t l = 0; l < n; ++l) {
        uint32_t iter = 0;
        for (;;) {
            uint32_t m = l;
            for (; m + 1 < n; ++m) {
                const double dd = ::fabs(d[m]) + ::fabs(d[m + 1]);
                if (::fabs(e[m]) <= 2.220446049250313e-16 * dd) break;
            }
            if (m == l) break;
            if (iter++ == SA_QL_ITERATIONS) return false;
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = sa_hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? ::fabs(r) : -::fabs(r)));
            double s = 1.0, c = 1.0, p = 0.0;
            bool underflow = false;
            for (uint32_t i = m; i-- > l;) {
                double f = dm_mul(s, e[i]);
                const double b = dm_mul(c, e[i]);
                r = sa_hypot(f, g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = dm_mul(d[i] - g, s) + dm_mul(2.0 * c, b);
                p = dm_mul(s, r);
                d[i + 1] = g + p;
                g = dm_mul(c, r) - b;
                for (uint32_t k = 0; k < n; ++k) {
                    f = Z(k, i + 1);
                    Z(k, i + 1) = dm_mul(s, Z(k, i)) + dm_mul(c, f);
                    Z(k, i) = dm_mul(c, Z(k, i)) - dm_mul(s, f);
                }
            }
            if (underflow) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    return true;
}

// The eigenvalues of the symmetric 3 x 3 matrix m by cyclic Jacobi rotations, descending.
HG_HD inline void sa_eigenvalues3(double m[3][3], double out[3]) {
    for (int sweep = 0; sweep < 64; ++sweep) {
        const double off = ::fabs(m[0][1]) + ::fabs(m[0][2]) + ::fabs(m[1][2]);
        if (off == 0.0) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (m[p][q] == 0.0) continue;
                const double theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (::fabs(theta) + ::sqrt(dm_mul(theta, theta) + 1.0));
                const double c = 1.0 / ::sqrt(dm_mul(t, t) + 1.0), s = dm_mul(t, c);
                for (int k = 0; k < 3; ++k) {
                    const double mkp = m[k][p], mkq = m[k][q];
                    m[k][p] = dm_mul(c, mkp) - dm_mul(s, mkq);
                    m[k][q] = dm_mul(s, mkp) + dm_mul(c, mkq);
                }
                for (int k = 0; k < 3; ++k) {
                    const double mpk = m[p][k], mqk = m[q][k];
                    m[p][k] = dm_mul(c, mpk) - dm_mul(s, mqk);
                    m[q][k] = dm_mul(s, mpk) + dm_mul(c, mqk);
                }
                m[p][q] = 0.0;
                m[q][p] = 0.0;
            }
    }
    double v[3] = {m[0][0], m[1][1], m[2][2]};
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (v[j] > v[i]) { const double t = v[i]; v[i] = v[j]; v[j] = t; }
    for (int i = 0; i < 3; ++i) out[i] = v[i];
}

// Doubles sa_alignment_of takes for G' of ns vertices.
HG_HD inline size_t sa_alignment_doubles(size_t ns) {
    const size_t k = ns <= SA_MAX_VERTICES ? ns : 0;
    return k * k + 3 * k + 3;
}

// The three principal moments of the curvature-weighted inertia of the spectral embedding of G'
// (the ns vertices vat(0..ns), ascending), each over the total mass, into moments[0..3),
// descending; masses |K_v| + 1/100 from `ricci`. False, with nothing written, when G' has fewer
// than 4 or more than SA_MAX_VERTICES vertices, when the QL iteration does not converge, or when
// the fourth and fifth Laplacian eigenvalues lie within SA_GAP of each other relative to the
// fifth. On the calling thread; `scratch` holds sa_alignment_doubles(ns) doubles.
template <class Vat>
HG_HD bool sa_alignment_of(const SgGraph& g, Vat vat, uint32_t ns, const double* ricci,
                           double* scratch, uint32_t* index, double moments[3]) {
    if (ns < 4 || ns > SA_MAX_VERTICES) return false;
    double* a = scratch;   // the Laplacian, then its eigenvectors as columns
    double* d = a + static_cast<size_t>(ns) * ns;
    double* e = d + ns;
    double* mass = e + ns;
    // index[v] is v's position in G' (ST_NONE outside); the Laplacian over G'.
    for (uint32_t i = 0; i < ns; ++i) index[vat(i)] = i;
    for (size_t k = 0; k < static_cast<size_t>(ns) * ns; ++k) a[k] = 0.0;
    for (uint32_t i = 0; i < ns; ++i) {
        const uint32_t v = vat(i);
        a[static_cast<size_t>(i) * ns + i] = static_cast<double>(g.degree(v));
        for (uint32_t k = g.off[v]; k < g.off[v + 1]; ++k)
            a[static_cast<size_t>(i) * ns + index[g.nbr[k]]] = -1.0;
    }
    for (uint32_t i = 0; i < ns; ++i) index[vat(i)] = ST_NONE;
    sa_tridiagonalise(a, ns, d, e);
    if (!sa_ql(d, e, ns, a)) return false;
    // The eigenvalues ascending by selection, ties to the lower column.
    uint32_t order[5] = {0, 0, 0, 0, 0};
    const uint32_t want = ns < 5 ? ns : 5;
    for (uint32_t r = 0; r < want; ++r) {
        uint32_t best = ST_NONE;
        for (uint32_t i = 0; i < ns; ++i) {
            bool taken = false;
            for (uint32_t q = 0; q < r; ++q) taken = taken || order[q] == i;
            if (!taken && (best == ST_NONE || d[i] < d[best])) best = i;
        }
        order[r] = best;
    }
    if (ns >= 5 && d[order[4]] - d[order[3]] <= SA_GAP * ::fabs(d[order[4]])) return false;
    double total = 0.0;
    for (uint32_t i = 0; i < ns; ++i) {
        mass[i] = ::fabs(ricci[vat(i)]) + 0.01;
        total += mass[i];
    }
    double c[3] = {0.0, 0.0, 0.0};
    for (uint32_t i = 0; i < ns; ++i)
        for (uint32_t k = 0; k < 3; ++k)
            c[k] += dm_mul(mass[i], a[static_cast<size_t>(i) * ns + order[k + 1]]);
    for (uint32_t k = 0; k < 3; ++k) c[k] /= total;
    double m[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (uint32_t i = 0; i < ns; ++i) {
        double x[3];
        for (uint32_t k = 0; k < 3; ++k) x[k] = a[static_cast<size_t>(i) * ns + order[k + 1]] - c[k];
        const double r2 = dm_mul(x[0], x[0]) + dm_mul(x[1], x[1]) + dm_mul(x[2], x[2]);
        for (uint32_t p = 0; p < 3; ++p)
            for (uint32_t q = 0; q < 3; ++q)
                m[p][q] += dm_mul(mass[i], (p == q ? r2 : 0.0) - dm_mul(x[p], x[q]));
    }
    sa_eigenvalues3(m, moments);
    for (uint32_t k = 0; k < 3; ++k) moments[k] /= total;
    return true;
}

}  // namespace common
}  // namespace HG_NAMESPACE
