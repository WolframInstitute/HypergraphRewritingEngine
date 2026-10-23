#pragma once
// TOPOLOGY AND METRIC TENSOR OF A STATE'S GRAPH G, one body for the host and the device.
// state_geometry_core.hpp includes this file after SgGeometry, and sg_geometry_of calls
// st_families_of after its eccentricity pass; this file reads SgGraph, SgArena, sg_bfs,
// sg_bfs_clear, sg_sort, sg_adjacent and sg_short_distance from there. The definitions are in docs/SPEC.md under "StepStatistics
// topology" and "StepStatistics metric tensor";
// reference/StateGeometryReference.wl computes the same values independently.
//
// BETTI NUMBERS over GF(2). For a graph H with c = |E| - |V| + b0 independent cycles, the
// boundaries of the triangles of H lie in the cycle space, and a cycle is fixed by its
// coefficients on the edges outside a spanning forest. So rank d2 is the rank of the triangle
// boundaries restricted to those c edges, found by Gaussian elimination on c-bit rows that
// pivot on their highest bit; b1 = c - rank d2. b2 = T - rank d2 - rank d3, rank d3 the rank of
// the tetrahedron boundaries over the T triangles. Each elimination stops when its rank reaches
// the dimension of the space it spans in, and keeps at most min(dimension, simplices) rows.

namespace HG_NAMESPACE {
namespace common {

constexpr uint32_t ST_NONE = 0xFFFFFFFFu;

// The highest set bit of x != 0.
HG_HD inline uint32_t st_high_bit(uint64_t x) {
    uint32_t b = 0;
    if (x >> 32) { x >>= 32; b += 32; }
    if (x >> 16) { x >>= 16; b += 16; }
    if (x >> 8) { x >>= 8; b += 8; }
    if (x >> 4) { x >>= 4; b += 4; }
    if (x >> 2) { x >>= 2; b += 2; }
    if (x >> 1) b += 1;
    return b;
}

// Reduces `vec` (W words) against the basis rows, each pivoting on its highest bit; a vector
// that stays non-zero becomes row `rank` of `basis`. True when it was independent.
HG_HD inline bool st_reduce(uint64_t* vec, uint32_t W, uint64_t* basis, uint32_t* pivot_row,
                            uint32_t& rank) {
    uint32_t top = W;
    for (;;) {
        while (top > 0 && vec[top - 1] == 0) --top;
        if (top == 0) return false;
        const uint32_t h = (top - 1) * 64 + st_high_bit(vec[top - 1]);
        const uint32_t row = pivot_row[h];
        if (row == ST_NONE) {
            uint64_t* dst = basis + static_cast<size_t>(rank) * W;
            for (uint32_t i = 0; i < top; ++i) dst[i] = vec[i];
            for (uint32_t i = top; i < W; ++i) dst[i] = 0;
            pivot_row[h] = rank++;
            return true;
        }
        const uint64_t* src = basis + static_cast<size_t>(row) * W;
        for (uint32_t i = 0; i < top; ++i) vec[i] ^= src[i];
    }
}

// The slot of b in a's neighbour list; b is a neighbour of a.
HG_HD inline uint32_t st_slot(const SgGraph& g, uint32_t a, uint32_t b) {
    uint32_t lo = g.off[a], hi = g.off[a + 1];
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (g.nbr[mid] < b) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Calls f(u, v, w) for each triangle u < v < w of g, in lexicographic order.
template <class F>
HG_HD void st_for_each_triangle(const SgGraph& g, F&& f) {
    for (uint32_t u = 0; u < g.n; ++u)
        for (uint32_t i = g.off[u]; i < g.off[u + 1]; ++i) {
            const uint32_t v = g.nbr[i];
            if (v <= u) continue;
            for (uint32_t j = i + 1; j < g.off[u + 1]; ++j) {
                const uint32_t w = g.nbr[j];
                if (sg_adjacent(g, v, w)) f(u, v, w);
            }
        }
}

// The index of triangle (a, b, c), a < b < c, in the lexicographic list `tri` of T triangles.
HG_HD inline uint32_t st_triangle_index(const uint32_t* tri, uint32_t T, uint32_t a, uint32_t b,
                                        uint32_t c) {
    uint32_t lo = 0, hi = T;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const uint32_t* t = tri + 3 * static_cast<size_t>(mid);
        const bool less = t[0] != a ? t[0] < a : (t[1] != b ? t[1] < b : t[2] < c);
        if (less) lo = mid + 1; else hi = mid;
    }
    return lo;
}

struct StBetti {
    uint32_t b0 = 0;
    uint32_t has_b2 = 0;
    uint64_t b1 = 0;
    uint64_t b2 = 0;
};

// The Betti numbers b0, b1 and, with `want_b2`, b2 of the clique complex of g over GF(2), on
// the calling thread. False when the arena is short.
HG_HD inline bool st_betti(const SgGraph& g, SgArena& arena, bool want_b2, StBetti& out) {
    const uint32_t n = g.n, m = g.edges;
    out = StBetti{};
    if (n == 0) return true;
    uint32_t* eid = arena.take<uint32_t>(2 * size_t{m} + 1);
    uint32_t* coord = arena.take<uint32_t>(size_t{m} + 1);
    int32_t* dist = arena.take<int32_t>(n);
    uint32_t* queue = arena.take<uint32_t>(n);
    if (arena.failed) return false;
    uint32_t next = 0;
    for (uint32_t u = 0; u < n; ++u)
        for (uint32_t k = g.off[u]; k < g.off[u + 1]; ++k) {
            const uint32_t w = g.nbr[k];
            eid[k] = w > u ? next++ : eid[st_slot(g, w, u)];
        }
    // A breadth-first spanning forest: coord[e] is ST_NONE for a forest edge and the edge's
    // cycle coordinate otherwise.
    for (uint32_t e = 0; e < m; ++e) coord[e] = 0;
    for (uint32_t v = 0; v < n; ++v) dist[v] = -1;
    for (uint32_t s = 0; s < n; ++s) {
        if (dist[s] >= 0) continue;
        ++out.b0;
        uint32_t head = 0, tail = 0;
        queue[tail++] = s;
        dist[s] = 0;
        while (head < tail) {
            const uint32_t u = queue[head++];
            for (uint32_t k = g.off[u]; k < g.off[u + 1]; ++k) {
                const uint32_t w = g.nbr[k];
                if (dist[w] >= 0) continue;
                dist[w] = dist[u] + 1;
                coord[eid[k]] = ST_NONE;
                queue[tail++] = w;
            }
        }
    }
    const uint32_t c = m - n + out.b0;
    uint32_t next_coord = 0;
    for (uint32_t e = 0; e < m; ++e)
        if (coord[e] != ST_NONE) coord[e] = next_coord++;
    uint32_t T = 0;
    st_for_each_triangle(g, [&](uint32_t, uint32_t, uint32_t) { ++T; });
    // rank d2 over the c cycle coordinates.
    const uint32_t W = (c + 63) / 64;
    const uint32_t rows = c < T ? c : T;
    uint64_t* basis = arena.take<uint64_t>(static_cast<size_t>(rows) * W + 1);
    uint32_t* pivot = arena.take<uint32_t>(size_t{c} + 1);
    uint64_t* vec = arena.take<uint64_t>(size_t{W} + 1);
    uint32_t* tri = want_b2 ? arena.take<uint32_t>(3 * size_t{T} + 1) : nullptr;
    if (arena.failed) return false;
    for (uint32_t i = 0; i < c; ++i) pivot[i] = ST_NONE;
    uint32_t rank2 = 0, t = 0;
    auto bit = [&](uint64_t* v, uint32_t a, uint32_t b) {
        const uint32_t x = coord[eid[st_slot(g, a, b)]];
        if (x != ST_NONE) v[x / 64] ^= uint64_t{1} << (x % 64);
    };
    st_for_each_triangle(g, [&](uint32_t u, uint32_t v, uint32_t w) {
        if (tri) { tri[3 * t] = u; tri[3 * t + 1] = v; tri[3 * t + 2] = w; }
        ++t;
        if (rank2 == c) return;
        for (uint32_t i = 0; i < W; ++i) vec[i] = 0;
        bit(vec, u, v);
        bit(vec, u, w);
        bit(vec, v, w);
        st_reduce(vec, W, basis, pivot, rank2);
    });
    out.b1 = c - rank2;
    if (!want_b2) return true;
    // rank d3 over the T triangle coordinates.
    const uint32_t z2 = T - rank2;   // dim ker d2, an upper bound on rank d3
    uint32_t Q = 0;
    auto for_each_tetrahedron = [&](auto&& f) {
        for (uint32_t i = 0; i < T; ++i) {
            const uint32_t a = tri[3 * i], b = tri[3 * i + 1], cc = tri[3 * i + 2];
            for (uint32_t k = g.off[a]; k < g.off[a + 1]; ++k) {
                const uint32_t x = g.nbr[k];
                if (x <= cc) continue;
                if (sg_adjacent(g, b, x) && sg_adjacent(g, cc, x)) f(i, a, b, cc, x);
            }
        }
    };
    for_each_tetrahedron([&](uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { ++Q; });
    const uint32_t W3 = (T + 63) / 64;
    const uint32_t rows3 = z2 < Q ? z2 : Q;
    uint64_t* basis3 = arena.take<uint64_t>(static_cast<size_t>(rows3) * W3 + 1);
    uint32_t* pivot3 = arena.take<uint32_t>(size_t{T} + 1);
    uint64_t* vec3 = arena.take<uint64_t>(size_t{W3} + 1);
    if (arena.failed) return false;
    for (uint32_t i = 0; i < T; ++i) pivot3[i] = ST_NONE;
    uint32_t rank3 = 0;
    for_each_tetrahedron([&](uint32_t i, uint32_t a, uint32_t b, uint32_t cc, uint32_t x) {
        if (rank3 == z2) return;
        for (uint32_t k = 0; k < W3; ++k) vec3[k] = 0;
        const uint32_t f[4] = {i, st_triangle_index(tri, T, a, b, x),
                               st_triangle_index(tri, T, a, cc, x),
                               st_triangle_index(tri, T, b, cc, x)};
        for (uint32_t k = 0; k < 4; ++k) vec3[f[k] / 64] ^= uint64_t{1} << (f[k] % 64);
        st_reduce(vec3, W3, basis3, pivot3, rank3);
    });
    out.b2 = z2 - rank3;
    out.has_b2 = 1;
    return true;
}

// Builds g^2, the graph joining two vertices of g at distance 1 or 2, in the arena. False when
// the arena is short.
HG_HD inline bool st_square_graph(const SgGraph& g, SgArena& arena, SgGraph& h) {
    const uint32_t n = g.n;
    uint32_t* off = arena.take<uint32_t>(size_t{n} + 1);
    uint32_t* mark = arena.take<uint32_t>(size_t{n} + 1);
    if (arena.failed) return false;
    for (uint32_t v = 0; v < n; ++v) mark[v] = ST_NONE;
    auto visit = [&](uint32_t v, auto&& f) {
        mark[v] = v;
        for (uint32_t i = g.off[v]; i < g.off[v + 1]; ++i) {
            const uint32_t u = g.nbr[i];
            if (mark[u] != v) { mark[u] = v; f(u); }
            for (uint32_t j = g.off[u]; j < g.off[u + 1]; ++j) {
                const uint32_t w = g.nbr[j];
                if (mark[w] != v) { mark[w] = v; f(w); }
            }
        }
    };
    off[0] = 0;
    for (uint32_t v = 0; v < n; ++v) {
        uint32_t k = 0;
        visit(v, [&](uint32_t) { ++k; });
        off[v + 1] = off[v] + k;
    }
    uint32_t* nbr = arena.take<uint32_t>(size_t{off[n]} + 1);
    if (arena.failed) return false;
    for (uint32_t v = 0; v < n; ++v) mark[v] = ST_NONE;
    for (uint32_t v = 0; v < n; ++v) {
        uint32_t k = off[v];
        visit(v, [&](uint32_t u) { nbr[k++] = u; });
        sg_sort(nbr + off[v], off[v + 1] - off[v]);
    }
    h.n = n;
    h.edges = off[n] / 2;
    h.off = off;
    h.nbr = nbr;
    h.max_degree = 0;
    for (uint32_t v = 0; v < n; ++v)
        if (h.degree(v) > h.max_degree) h.max_degree = h.degree(v);
    return true;
}

// THE METRIC TENSOR GAP of p at scale r (docs/SPEC.md). `counts` receives, for k = 0..r, the
// number of ordered pairs (v, w) of S_r(p) with T[v, w] = k / r; the gap is
// the integral over [0, 1] of |F_r(t) - (1 - arccos(t) / pi)|, F_r the distribution function of
// the values. F_r is constant on each [k/r, (k+1)/r); with G(t) = t - (t arccos t - sqrt(1 - t^2)
// + 1) / pi the integral of 1 - arccos(t) / pi from 0, each piece is exact up to rounding.
HG_HD inline double st_gap_of(const uint64_t* counts, uint32_t r) {
    uint64_t total = 0;
    for (uint32_t k = 0; k <= r; ++k) total += counts[k];
    const double inv_pi = 1.0 / kDmPi;
    auto F = [&](double t) { return 1.0 - dm_mul(dm_acos01(t), inv_pi); };
    auto G = [&](double t) {
        const double s = ::sqrt(1.0 - dm_mul(t, t));
        return t - dm_mul(dm_mul(t, dm_acos01(t)) - s + 1.0, inv_pi);
    };
    double gap = 0.0;
    uint64_t cum = 0;
    for (uint32_t k = 0; k < r; ++k) {
        cum += counts[k];
        const double c = static_cast<double>(cum) / static_cast<double>(total);
        const double a = static_cast<double>(k) / r, b = static_cast<double>(k + 1) / r;
        const double ga = G(a), gb = G(b);
        double piece = 0.0;
        if (c <= F(a)) {
            piece = (gb - ga) - dm_mul(c, b - a);
        } else if (c >= F(b)) {
            piece = dm_mul(c, b - a) - (gb - ga);
        } else {
            // F(t*) = c at t* = -cos(pi c); arccos t* = pi (1 - c), sqrt(1 - t*^2) = sin(pi c).
            double sn = 0.0, cs = 0.0;
            dm_sincos_pi(c, sn, cs);
            const double ts = -cs;
            const double gs = ts - dm_mul(dm_mul(dm_mul(ts, kDmPi), 1.0 - c) - sn + 1.0, inv_pi);
            piece = dm_mul(c, ts - a) - (gs - ga) + (gb - gs) - dm_mul(c, b - ts);
        }
        gap += piece;
    }
    return gap;
}

// The mean over the centre vertices p of G' (eccentricity R' in G') of the metric tensor gap of
// p at r = floor(R' / 2); `ecc` holds each vertex's eccentricity in its component and G' is the
// sub_n vertices of `sub`, or every vertex when `sub` is null. Every lane calls it; the leader's
// `gap` is the result, `defined` 0 when R' < 2. False when the arena is short.
template <class Par>
HG_HD bool st_metric_tensor_gap(const SgGraph& g, SgArena& arena, const uint32_t* ecc,
                                const uint32_t* sub, uint32_t sub_n, Par par, double& gap,
                                uint32_t& defined) {
    const uint32_t n = g.n, L = par.width(), me = par.rank();
    defined = 0;
    gap = 0.0;
    uint32_t R = ST_NONE;
    for (uint32_t i = 0; i < sub_n; ++i)
        if (ecc[sub ? sub[i] : i] < R) R = ecc[sub ? sub[i] : i];
    if (sub_n < 2 || R < 2) return true;
    const uint32_t r = R / 2;
    const size_t n1 = size_t{n} + 1;
    int32_t* dp = arena.take<int32_t>(n1);
    uint32_t* sphere = arena.take<uint32_t>(n1);
    uint32_t* queue_all = arena.take<uint32_t>(L * n1);
    uint32_t* interval_all = arena.take<uint32_t>(L * n1);
    if (arena.failed) return false;
    // The largest sphere over the centres sizes the distance rows.
    uint32_t s_max = 0;
    if (par.leader()) {
        for (uint32_t v = 0; v < n; ++v) dp[v] = -1;
        for (uint32_t i = 0; i < sub_n; ++i) {
            const uint32_t p = sub ? sub[i] : i;
            if (ecc[p] != R) continue;
            uint32_t reached = 0;
            sg_bfs(g, p, dp, queue_all, nullptr, r, reached);
            uint32_t s = 0;
            for (uint32_t k = 0; k < reached; ++k)
                if (static_cast<uint32_t>(dp[queue_all[k]]) == r) ++s;
            if (s > s_max) s_max = s;
            sg_bfs_clear(dp, queue_all, reached);
        }
    }
    s_max = par.bcast(s_max);
    int32_t* rows = arena.take<int32_t>(static_cast<size_t>(s_max) * n1);
    uint64_t* counts = arena.take<uint64_t>(static_cast<size_t>(s_max) * (r + 1) + 1);
    uint64_t* total = arena.take<uint64_t>(size_t{r} + 1);
    if (arena.failed) return false;
    double sum = 0.0;
    uint32_t centres = 0;
    for (uint32_t i = 0; i < sub_n; ++i) {
        const uint32_t p = sub ? sub[i] : i;
        if (ecc[p] != R) continue;
        uint32_t s = 0;
        if (par.leader()) {
            for (uint32_t v = 0; v < n; ++v) dp[v] = -1;
            uint32_t reached = 0;
            sg_bfs(g, p, dp, queue_all, nullptr, UINT32_MAX, reached);
            for (uint32_t v = 0; v < n; ++v)
                if (dp[v] == static_cast<int32_t>(r)) sphere[s++] = v;
        }
        par.sync();
        s = par.bcast(s);
        // The distances from each sphere vertex, one row each.
        for (uint32_t j = me; j < s; j += L) {
            int32_t* row = rows + static_cast<size_t>(j) * n1;
            for (uint32_t v = 0; v < n; ++v) row[v] = -1;
            uint32_t reached = 0;
            sg_bfs(g, sphere[j], row, queue_all + me * n1, nullptr, UINT32_MAX, reached);
        }
        par.sync();
        // Column w: I(p, w) = {u : d(p, u) + d(u, w) = r}; for each v the least d(p, u) over the
        // u of I(p, w) nearest v.
        for (uint32_t wi = me; wi < s; wi += L) {
            const int32_t* dw = rows + static_cast<size_t>(wi) * n1;
            uint32_t* interval = interval_all + me * n1;
            uint32_t in = 0;
            for (uint32_t u = 0; u < n; ++u)
                if (dp[u] >= 0 && dp[u] + dw[u] == static_cast<int32_t>(r)) interval[in++] = u;
            uint64_t* cnt = counts + static_cast<size_t>(wi) * (r + 1);
            for (uint32_t k = 0; k <= r; ++k) cnt[k] = 0;
            for (uint32_t vi = 0; vi < s; ++vi) {
                const int32_t* dv = rows + static_cast<size_t>(vi) * n1;
                int32_t best = INT32_MAX, foot = INT32_MAX;
                for (uint32_t k = 0; k < in; ++k) {
                    const uint32_t u = interval[k];
                    if (dv[u] < best || (dv[u] == best && dp[u] < foot)) {
                        best = dv[u];
                        foot = dp[u];
                    }
                }
                ++cnt[foot];
            }
        }
        par.sync();
        if (par.leader()) {
            for (uint32_t k = 0; k <= r; ++k) total[k] = 0;
            for (uint32_t wi = 0; wi < s; ++wi)
                for (uint32_t k = 0; k <= r; ++k)
                    total[k] += counts[static_cast<size_t>(wi) * (r + 1) + k];
            sum += st_gap_of(total, r);
            ++centres;
        }
        par.sync();
    }
    if (par.leader()) {
        gap = sum / centres;
        defined = 1;
    }
    return true;
}

// Bytes of arena an array of n elements of `size` bytes takes at most, alignment included.
HG_HD inline size_t st_array(size_t n, size_t size) { return n * size + 8; }

// Bytes st_betti takes for a graph of n vertices, m edges, c independent cycles, T triangles
// and Q tetrahedra (b2 with `want_b2`): the row counts are their upper bounds.
HG_HD inline size_t st_betti_bytes(size_t n, size_t m, size_t c, size_t T, size_t Q,
                                   bool want_b2) {
    const size_t W = (c + 63) / 64, rows = c < T ? c : T;
    size_t b = st_array(2 * m + 1, 4) + st_array(m + 1, 4) + 2 * st_array(n, 4) +
               st_array(rows * W + 1, 8) + st_array(c + 1, 4) + st_array(W + 1, 8);
    if (want_b2) {
        const size_t W3 = (T + 63) / 64, rows3 = T < Q ? T : Q;
        b += st_array(3 * T + 1, 4) + st_array(rows3 * W3 + 1, 8) + st_array(T + 1, 4) +
             st_array(W3 + 1, 8);
    }
    return b;
}

// The bytes the requested families take after the counting scratch (a distance, a queue, a
// mark and a list per vertex, which sg_metric_bytes counts), counted on the calling thread with
// that scratch: dist all -1 and mark all ST_NONE on entry and on return.
HG_HD inline size_t st_families_bytes(const SgGraph& g, uint32_t want, uint32_t lanes,
                                      const uint32_t* ecc, const uint32_t* sub, uint32_t ns,
                                      int32_t* dist, uint32_t* queue, uint32_t* mark,
                                      uint32_t* list) {
    const size_t n = g.n, n1 = n + 1;
    size_t bytes = 0;
    if (n == 0) return 0;
    if (want & SG_TOPOLOGY) {
        // Components, triangles and tetrahedra of G.
        uint32_t b0 = 0;
        for (uint32_t s = 0; s < g.n; ++s) {
            if (dist[s] >= 0) continue;
            ++b0;
            uint32_t reached = 0;
            sg_bfs(g, s, dist, queue, nullptr, UINT32_MAX, reached);
        }
        for (uint32_t v = 0; v < g.n; ++v) dist[v] = -1;
        size_t T = 0, Q = 0;
        st_for_each_triangle(g, [&](uint32_t u, uint32_t v, uint32_t w) {
            ++T;
            for (uint32_t k = g.off[u]; k < g.off[u + 1]; ++k) {
                const uint32_t x = g.nbr[k];
                if (x > w && sg_adjacent(g, v, x) && sg_adjacent(g, w, x)) ++Q;
            }
        });
        const size_t m = g.edges, c = m - n + b0;
        bytes += st_betti_bytes(n, m, c, T, Q, true);
        // G^2: its edges and triangles, two vertices adjacent when at distance at most 2.
        size_t E2 = 0, T2 = 0;
        for (uint32_t u = 0; u < g.n; ++u) {
            uint32_t k = 0;
            mark[u] = u;
            for (uint32_t i = g.off[u]; i < g.off[u + 1]; ++i) {
                const uint32_t a = g.nbr[i];
                if (mark[a] != u) { mark[a] = u; list[k++] = a; }
                for (uint32_t j = g.off[a]; j < g.off[a + 1]; ++j) {
                    const uint32_t b = g.nbr[j];
                    if (mark[b] != u) { mark[b] = u; list[k++] = b; }
                }
            }
            E2 += k;
            for (uint32_t i = 0; i < k; ++i) {
                if (list[i] <= u) continue;
                for (uint32_t j = 0; j < k; ++j)
                    if (list[j] > list[i] && sg_short_distance(g, list[i], list[j]) <= 2) ++T2;
            }
        }
        for (uint32_t v = 0; v < g.n; ++v) mark[v] = ST_NONE;
        E2 /= 2;
        bytes += st_array(n1, 4) * 2 + st_array(2 * E2 + 1, 4) +
                 st_betti_bytes(n, E2, E2 - n + b0, T2, 0, false);
    }
    if (want & SG_METRIC_TENSOR) {
        uint32_t R = ST_NONE;
        for (uint32_t i = 0; i < ns; ++i)
            if (ecc[sub ? sub[i] : i] < R) R = ecc[sub ? sub[i] : i];
        if (ns >= 2 && R >= 2) {
            const uint32_t r = R / 2;
            size_t s_max = 0;
            for (uint32_t i = 0; i < ns; ++i) {
                const uint32_t p = sub ? sub[i] : i;
                if (ecc[p] != R) continue;
                uint32_t reached = 0;
                sg_bfs(g, p, dist, queue, nullptr, r, reached);
                size_t s = 0;
                for (uint32_t k = 0; k < reached; ++k)
                    if (static_cast<uint32_t>(dist[queue[k]]) == r) ++s;
                if (s > s_max) s_max = s;
                sg_bfs_clear(dist, queue, reached);
            }
            bytes += 2 * st_array(n1, 4) + 2 * st_array(lanes * n1, 4) +
                     st_array(s_max * n1, 4) + st_array(s_max * (r + 1) + 1, 8) +
                     st_array(size_t{r} + 1, 8);
        }
    }
    return bytes;
}

// The requested families of `want` (SG_TOPOLOGY, SG_METRIC_TENSOR) into `out`,
// written by the leader, from sg_geometry_of's eccentricities and G' (`sub`, `ns`). Every lane
// calls it and gets the same verdict and arena use. False when the arena is short.
template <class Par>
HG_HD bool st_families_of(const SgGraph& g, SgArena& arena, SgGeometry& out, uint32_t want,
                          Par par, const uint32_t* ecc, const uint32_t* sub, uint32_t ns) {
    // The counting scratch, then every array at once. arena.extra gains the bytes this stage
    // takes, which sg_metric_bytes does not count, so a later stage that fails, or this one,
    // reports a size with room for the whole stage.
    const size_t n1 = size_t{g.n} + 1;
    int32_t* cdist = arena.take<int32_t>(n1);
    uint32_t* cqueue = arena.take<uint32_t>(n1);
    uint32_t* cmark = arena.take<uint32_t>(n1);
    uint32_t* clist = arena.take<uint32_t>(n1);
    if (arena.failed) return false;
    uint64_t need = 0;
    if (par.leader()) {
        for (uint32_t v = 0; v < g.n; ++v) { cdist[v] = -1; cmark[v] = ST_NONE; }
        need = st_families_bytes(g, want, par.width(), ecc, sub, ns, cdist, cqueue, cmark, clist);
    }
    par.sync();
    need = par.bcast64(need);
    arena.extra += static_cast<size_t>(need);
    if (arena.used + need > arena.capacity) {
        arena.failed = true;
        return false;
    }
    if (want & SG_TOPOLOGY) {
        // The eliminations run on the leader; the other lanes take the leader's arena use.
        uint32_t ok = 1;
        uint64_t used = arena.used;
        if (par.leader()) {
            StBetti clique, rips;
            SgGraph square;
            ok = st_betti(g, arena, true, clique) && st_square_graph(g, arena, square) &&
                         st_betti(square, arena, false, rips)
                     ? 1u
                     : 0u;
            used = arena.used;
            if (ok && g.n > 0) {
                out.clique_betti[0] = static_cast<double>(clique.b0);
                out.clique_betti[1] = static_cast<double>(clique.b1);
                out.clique_betti[2] = static_cast<double>(clique.b2);
                out.rips_betti1 = static_cast<double>(rips.b1);
                out.defined |= SG_TOPOLOGY;
            }
        }
        par.sync();
        arena.used = static_cast<size_t>(par.bcast64(used));
        if (!par.bcast(ok)) {
            arena.failed = true;
            return false;
        }
    }
    if (want & SG_METRIC_TENSOR) {
        double gap = 0.0;
        uint32_t defined = 0;
        if (!st_metric_tensor_gap(g, arena, ecc, sub, ns, par, gap, defined)) return false;
        if (par.leader() && defined) {
            out.metric_tensor_gap = gap;
            out.defined |= SG_METRIC_TENSOR;
        }
    }
    par.sync();
    return true;
}

}  // namespace common
}  // namespace HG_NAMESPACE
