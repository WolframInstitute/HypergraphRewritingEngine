#pragma once
#include "hgcommon/namespace.hpp"
// PER-STATE GEOMETRY OF A HYPERGRAPH STATE, one body for the host and the device.
//
// A state's graph G is the undirected simple graph whose vertices are the state's vertices and
// whose edges join the consecutive vertices of each hyperedge: {a, b, c} gives a-b and b-c. A
// self-loop is dropped and a repeated pair is one edge. A vertex that only appears in a unary
// hyperedge, or only next to itself, is an isolated vertex of G. Hypergraph isomorphism keeps
// the order of the vertices inside each hyperedge, so every value here is an invariant of the
// state's isomorphism class.
//
// B(v, r) is the set of vertices at distance at most r from v; |B(v, 0)| = 1. R is the graph
// radius of G (the least eccentricity). The definitions, with the cases where each is undefined,
// are in docs/SPEC.md under "StepStatistics geometry"; reference/StateGeometryReference.wl
// computes the same values independently, and its Wolfram Function Repository checks are
// reference/verify_state_geometry.wls.
//
//   WolframHausdorffDimension  mean over vertices of the mean over r = 1..R of
//                              (log|B(v,r)| - log|B(v,r-1)|) / (log(r+1) - log r)
//   BallGrowthDimension[r]     the same term at one r, averaged over vertices
//   WolframRicciCurvatureScalar  mean over vertices of the mean over r = 1..R of
//                              6(d+2)/r^2 (1 - |B(v,r)| Gamma(d/2+1) / (pi^(d/2) r^d)),
//                              d = the state's WolframHausdorffDimension
//   OllivierRicciCurvature     mean over edges x-y of 1 - W1(m_x, m_y), m_x = 1/2 at x and
//                              1/(2 deg x) at each neighbour; W1 is the exact transport cost
//   GraphRadius, MeanEccentricity
//   DegreeEntropy              Shannon entropy in bits of the degree distribution
//   LocalEntropy               mean over v of the degree entropy over B(v, 2)
//   MutualInformation          mean over vertices with a neighbour of the mean over neighbours
//                              w of max(0, log2(|B2v n B2w| |B2v u B2w| / (|B2v| |B2w|)))
//   FisherInformation          mean over v of (1 + mean_u |d_u - d_v|) / (var_u d_u + 1/100),
//                              u over B(v, 2) minus v, d_u the vertex's own Hausdorff dimension
//
// The logarithms, powers and Gamma function are det_math.hpp's, and every product that feeds a sum
// is dm_mul, so the host and the device compute the same doubles.
//
// NO ALLOCATION. Every array lives in caller memory handed over as a byte buffer. Building G
// needs sg_build_bytes(slots, pairs) and the metrics need sg_metric_bytes(n, max_degree) more;
// sg_state_geometry reports the total it needed when the buffer is short, and computes nothing.
//
// COST. One breadth-first search per vertex for the balls and eccentricities, a second for the
// per-vertex dimensions the Fisher information reads, and radius-2 searches for the local
// measures: O(n (n + m)) per state. Each edge's transport problem is solved exactly by
// successive shortest paths on the (deg x + 1) x (deg y + 1) supports.

#include <math.h>

#include <cstddef>
#include <cstdint>

#include "hgcommon/core.hpp"
#include "hgcommon/det_math.hpp"

namespace HG_NAMESPACE {
namespace common {

// Bump allocation over caller memory. A request that does not fit marks the arena failed and
// returns nullptr; `used` still advances, so after a failed stage it holds what the stage needed.
struct SgArena {
    unsigned char* base = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    bool failed = false;

    template <class T>
    HG_HD T* take(size_t n) {
        const size_t a = (used + alignof(T) - 1) & ~(alignof(T) - 1);
        used = a + n * sizeof(T);
        if (failed || used > capacity) { failed = true; return nullptr; }
        return reinterpret_cast<T*>(base + a);
    }
};

// Heap sort, ascending. The device has no std::sort.
template <class T>
HG_HD void sg_sort(T* a, size_t n) {
    if (n < 2) return;
    auto sift = [&](size_t root, size_t end) {
        while (2 * root + 1 < end) {
            size_t child = 2 * root + 1;
            if (child + 1 < end && a[child] < a[child + 1]) ++child;
            if (!(a[root] < a[child])) return;
            const T t = a[root]; a[root] = a[child]; a[child] = t;
            root = child;
        }
    };
    for (size_t s = n / 2; s-- > 0;) sift(s, n);
    for (size_t end = n - 1; end > 0; --end) {
        const T t = a[0]; a[0] = a[end]; a[end] = t;
        sift(0, end);
    }
}

template <class T>
HG_HD size_t sg_unique(T* a, size_t n) {
    if (n == 0) return 0;
    size_t w = 1;
    for (size_t i = 1; i < n; ++i)
        if (!(a[i] == a[w - 1])) a[w++] = a[i];
    return w;
}

// G in compressed rows: the neighbours of i are nbr[off[i] .. off[i+1]), ascending.
struct SgGraph {
    uint32_t n = 0;
    uint32_t edges = 0;          // undirected edges
    uint32_t max_degree = 0;
    const uint32_t* off = nullptr;
    const uint32_t* nbr = nullptr;
    HG_HD uint32_t degree(uint32_t v) const { return off[v + 1] - off[v]; }
};

HG_HD inline size_t sg_build_bytes(size_t slots, size_t pairs) {
    return 64 + 4 * slots + 8 * 2 * pairs + 4 * (slots + 1) + 4 * 2 * pairs + 32;
}

// An EdgeList supplies   uint32_t count() const;
//                        uint32_t arity(uint32_t i) const;
//                        uint32_t at(uint32_t i, uint32_t k) const;   vertex k of hyperedge i
template <class EdgeList>
HG_HD void sg_sizes(const EdgeList& el, size_t& slots, size_t& pairs) {
    slots = 0;
    pairs = 0;
    const uint32_t m = el.count();
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t k = el.arity(i);
        slots += k;
        if (k > 1) pairs += k - 1;
    }
}

HG_HD inline uint32_t sg_index_of(const uint32_t* sorted, uint32_t n, uint32_t v) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (sorted[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Builds G from a state's hyperedges. False when the arena is short.
template <class EdgeList>
HG_HD bool sg_build(const EdgeList& el, SgArena& arena, SgGraph& g) {
    size_t slots = 0, pairs = 0;
    sg_sizes(el, slots, pairs);
    uint32_t* verts = arena.take<uint32_t>(slots);
    uint64_t* keys = arena.take<uint64_t>(2 * pairs);
    if (arena.failed) return false;
    const uint32_t m = el.count();
    size_t s = 0;
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t k = 0; k < el.arity(i); ++k) verts[s++] = el.at(i, k);
    sg_sort(verts, slots);
    const uint32_t n = static_cast<uint32_t>(sg_unique(verts, slots));
    size_t p = 0;
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t k = el.arity(i);
        for (uint32_t j = 0; j + 1 < k; ++j) {
            const uint64_t a = sg_index_of(verts, n, el.at(i, j));
            const uint64_t b = sg_index_of(verts, n, el.at(i, j + 1));
            if (a == b) continue;
            keys[p++] = (a << 32) | b;
            keys[p++] = (b << 32) | a;
        }
    }
    sg_sort(keys, p);
    const size_t m2 = sg_unique(keys, p);
    uint32_t* off = arena.take<uint32_t>(n + 1);
    uint32_t* nbr = arena.take<uint32_t>(m2);
    if (arena.failed) return false;
    for (uint32_t i = 0; i <= n; ++i) off[i] = 0;
    for (size_t i = 0; i < m2; ++i) ++off[(keys[i] >> 32) + 1];
    for (uint32_t i = 0; i < n; ++i) off[i + 1] += off[i];
    for (size_t i = 0; i < m2; ++i) nbr[i] = static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu);
    g.n = n;
    g.edges = static_cast<uint32_t>(m2 / 2);
    g.off = off;
    g.nbr = nbr;
    g.max_degree = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (g.degree(i) > g.max_degree) g.max_degree = g.degree(i);
    return true;
}

// Breadth-first search from s, stopping after depth `max_depth`. `dist` holds -1 for every
// vertex on entry and again on return. `layer[d]`, when given, receives the number of vertices
// at distance d for d = 0..depth reached. Returns the depth reached (the eccentricity of s when
// max_depth is not reached first); `reached` receives the number of vertices found.
HG_HD inline uint32_t sg_bfs(const SgGraph& g, uint32_t s, int32_t* dist, uint32_t* queue,
                             uint32_t* layer, uint32_t max_depth, uint32_t& reached) {
    uint32_t head = 0, tail = 0;
    queue[tail++] = s;
    dist[s] = 0;
    uint32_t depth = 0;
    if (layer) layer[0] = 0;
    while (head < tail) {
        const uint32_t u = queue[head++];
        const uint32_t du = static_cast<uint32_t>(dist[u]);
        if (du > depth) { depth = du; if (layer) layer[depth] = 0; }
        if (layer) ++layer[du];
        if (du == max_depth) continue;
        for (uint32_t e = g.off[u]; e < g.off[u + 1]; ++e) {
            const uint32_t w = g.nbr[e];
            if (dist[w] < 0) { dist[w] = static_cast<int32_t>(du + 1); queue[tail++] = w; }
        }
    }
    reached = tail;
    return depth;
}

HG_HD inline void sg_bfs_clear(int32_t* dist, const uint32_t* queue, uint32_t reached) {
    for (uint32_t i = 0; i < reached; ++i) dist[queue[i]] = -1;
}

HG_HD inline bool sg_adjacent(const SgGraph& g, uint32_t a, uint32_t b) {
    uint32_t lo = g.off[a], hi = g.off[a + 1];
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (g.nbr[mid] < b) lo = mid + 1; else hi = mid;
    }
    return lo < g.off[a + 1] && g.nbr[lo] == b;
}

HG_HD inline bool sg_common_neighbour(const SgGraph& g, uint32_t a, uint32_t b) {
    uint32_t i = g.off[a], j = g.off[b];
    while (i < g.off[a + 1] && j < g.off[b + 1]) {
        if (g.nbr[i] == g.nbr[j]) return true;
        if (g.nbr[i] < g.nbr[j]) ++i; else ++j;
    }
    return false;
}

// Distance between a vertex within 1 of x and one within 1 of y, for an edge x-y: at most 3.
HG_HD inline uint32_t sg_short_distance(const SgGraph& g, uint32_t a, uint32_t b) {
    if (a == b) return 0;
    if (sg_adjacent(g, a, b)) return 1;
    if (sg_common_neighbour(g, a, b)) return 2;
    return 3;
}

// Scratch for one edge's transport problem with supports of at most k = max_degree + 1 points.
HG_HD inline size_t sg_transport_bytes(size_t k) {
    const size_t nodes = 2 * k + 2;
    return 64 + 4 * k * k + 4 * k * k + 4 * 2 * k + 4 * 5 * (nodes + 1);
}

// The ollivier curvature of edge x-y with the lazy walk (1/2 at the vertex, 1/2 spread over its
// neighbours): 1 - W1, W1 the exact optimal transport cost under the graph distance. The masses
// are scaled to integers by L = 2 deg(x) deg(y) and the transport problem is solved by
// successive shortest paths, so the cost is an exact integer before the one division by L.
HG_HD inline double sg_ollivier_edge(const SgGraph& g, uint32_t x, uint32_t y,
                                     unsigned char* scratch) {
    const uint32_t dx = g.degree(x), dy = g.degree(y);
    const uint32_t a = dx + 1, b = dy + 1;
    // Aligned carve of the scratch sized by sg_transport_bytes(max_degree + 1).
    uint32_t* cost = reinterpret_cast<uint32_t*>(scratch);
    uint32_t* flow = cost + static_cast<size_t>(a) * b;
    uint32_t* supply = flow + static_cast<size_t>(a) * b;   // remaining supply at A points
    uint32_t* demand = supply + a;                          // remaining demand at B points
    const uint32_t nodes = a + b + 2;
    int64_t* dist = reinterpret_cast<int64_t*>(
        (reinterpret_cast<uintptr_t>(demand + b) + 7) & ~static_cast<uintptr_t>(7));
    uint32_t* prev = reinterpret_cast<uint32_t*>(dist + nodes);
    uint32_t* inq = prev + nodes;
    uint32_t* queue = inq + nodes;
    auto pa = [&](uint32_t i) { return i == 0 ? x : g.nbr[g.off[x] + i - 1]; };
    auto pb = [&](uint32_t j) { return j == 0 ? y : g.nbr[g.off[y] + j - 1]; };
    for (uint32_t i = 0; i < a; ++i)
        for (uint32_t j = 0; j < b; ++j) {
            cost[i * b + j] = sg_short_distance(g, pa(i), pb(j));
            flow[i * b + j] = 0;
        }
    for (uint32_t i = 0; i < a; ++i) supply[i] = i == 0 ? dx * dy : dy;
    for (uint32_t j = 0; j < b; ++j) demand[j] = j == 0 ? dx * dy : dx;
    const uint64_t total = 2ull * dx * dy;
    uint64_t moved = 0, total_cost = 0;
    // Node numbering: 0 source, 1..a the A points, a+1..a+b the B points, a+b+1 the sink.
    const uint32_t src = 0, sink = a + b + 1;
    const int64_t inf = INT64_MAX / 4;
    while (moved < total) {
        for (uint32_t v = 0; v < nodes; ++v) { dist[v] = inf; inq[v] = 0; }
        uint32_t head = 0, count = 0;
        auto push = [&](uint32_t v) {
            if (inq[v]) return;
            inq[v] = 1;
            queue[(head + count) % nodes] = v;
            ++count;
        };
        dist[src] = 0;
        push(src);
        while (count) {
            const uint32_t u = queue[head];
            head = (head + 1) % nodes;
            --count;
            inq[u] = 0;
            auto relax = [&](uint32_t v, int64_t w) {
                if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; prev[v] = u; push(v); }
            };
            if (u == src) {
                for (uint32_t i = 0; i < a; ++i) if (supply[i]) relax(1 + i, 0);
            } else if (u <= a) {
                const uint32_t i = u - 1;
                for (uint32_t j = 0; j < b; ++j) relax(1 + a + j, cost[i * b + j]);
            } else if (u < sink) {
                const uint32_t j = u - 1 - a;
                if (demand[j]) relax(sink, 0);
                for (uint32_t i = 0; i < a; ++i)
                    if (flow[i * b + j]) relax(1 + i, -static_cast<int64_t>(cost[i * b + j]));
            }
        }
        if (dist[sink] >= inf) break;   // cannot happen: supply and demand totals are equal
        // Bottleneck along the path.
        uint64_t push_amount = UINT64_MAX;
        for (uint32_t v = sink; v != src; v = prev[v]) {
            const uint32_t u = prev[v];
            uint64_t cap = UINT64_MAX;
            if (v == sink) cap = demand[u - 1 - a];
            else if (u == src) cap = supply[v - 1];
            else if (u > a) cap = flow[(v - 1) * b + (u - 1 - a)];   // B -> A undoes flow
            if (cap < push_amount) push_amount = cap;
        }
        for (uint32_t v = sink; v != src; v = prev[v]) {
            const uint32_t u = prev[v];
            const uint32_t amount = static_cast<uint32_t>(push_amount);
            if (v == sink) demand[u - 1 - a] -= amount;
            else if (u == src) supply[v - 1] -= amount;
            else if (u <= a) flow[(u - 1) * b + (v - 1 - a)] += amount;
            else flow[(v - 1) * b + (u - 1 - a)] -= amount;
        }
        moved += push_amount;
        total_cost += push_amount * static_cast<uint64_t>(dist[sink]);
    }
    // One division of exact integers, so the value is the correctly rounded rational.
    return static_cast<double>(static_cast<int64_t>(total) - static_cast<int64_t>(total_cost)) /
           static_cast<double>(total);
}

// Which fields of SgGeometry hold a value.
enum : uint32_t {
    SG_RADIUS = 1u << 0,                 // radius, mean_eccentricity: G connected, n >= 1
    SG_HAUSDORFF = 1u << 1,              // hausdorff_dimension, ball dimensions: connected, n >= 2
    SG_RICCI = 1u << 2,                  // ricci_scalar: as SG_HAUSDORFF
    SG_OLLIVIER = 1u << 3,               // ollivier_ricci: G has an edge
    SG_DEGREE_ENTROPY = 1u << 4,         // degree_entropy, local_entropy: n >= 1
    SG_MUTUAL_INFORMATION = 1u << 5,     // mutual_information: G has an edge
    SG_FISHER = 1u << 6,                 // fisher_information: as SG_HAUSDORFF
};

struct SgGeometry {
    uint32_t defined = 0;
    uint32_t vertex_count = 0;           // of G
    uint32_t edge_count = 0;             // of G
    int32_t radius = -1;
    double mean_eccentricity = 0.0;
    double hausdorff_dimension = 0.0;
    double ricci_scalar = 0.0;
    double ollivier_ricci = 0.0;
    double degree_entropy = 0.0;
    double local_entropy = 0.0;
    double mutual_information = 0.0;
    double fisher_information = 0.0;
    uint32_t ball_radii = 0;             // R: entries written to the ball-dimension output
};

HG_HD inline size_t sg_metric_bytes(size_t n, size_t max_degree) {
    // dist, dist2, queue, queue2, mark, ecc, layer, degree counts (4 bytes each, n + 1 entries),
    // per-vertex dimension and the per-radius sums (8 bytes each), and one transport problem.
    return 64 + 8 * 4 * (n + 1) + 3 * 8 * (n + 1) + 8 * (n + 1) + 64 +
           sg_transport_bytes(max_degree + 1);
}

// Shannon entropy in bits of a distribution given as counts over `total`.
HG_HD inline double sg_entropy_term(uint32_t count, uint32_t total) {
    const double p = static_cast<double>(count) / static_cast<double>(total);
    return -dm_mul(p, dm_log2(p));
}

// The degree entropy over the vertices queue[0 .. k). `counts` is zero on entry and on return.
HG_HD inline double sg_degree_entropy_of(const SgGraph& g, const uint32_t* vs, uint32_t k,
                                         uint32_t* counts) {
    for (uint32_t i = 0; i < k; ++i) ++counts[g.degree(vs[i])];
    double h = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
        const uint32_t d = g.degree(vs[i]);
        if (counts[d]) { h += sg_entropy_term(counts[d], k); counts[d] = 0; }
    }
    return h;
}

// Computes the geometry of G already built in `arena`. `ball_dimension` receives the mean
// over vertices of the ball-growth term at r = 1..R in entries 0..R-1, up to `ball_capacity`
// entries. `want` is a mask of the SG_* values to compute: the radius and eccentricity are
// always computed, SG_DEGREE_ENTROPY and SG_MUTUAL_INFORMATION each bring both, and SG_RICCI
// and SG_FISHER each bring SG_HAUSDORFF. False when the arena is short.
HG_HD inline bool sg_geometry_of(const SgGraph& g, SgArena& arena, SgGeometry& out,
                                 double* ball_dimension, uint32_t ball_capacity,
                                 uint32_t want = ~0u) {
    const bool want_local = (want & (SG_DEGREE_ENTROPY | SG_MUTUAL_INFORMATION)) != 0;
    const uint32_t n = g.n;
    out = SgGeometry{};
    out.vertex_count = n;
    out.edge_count = g.edges;
    if (n == 0) return true;
    int32_t* dist = arena.take<int32_t>(n + 1);
    int32_t* dist2 = arena.take<int32_t>(n + 1);
    uint32_t* queue = arena.take<uint32_t>(n + 1);
    uint32_t* queue2 = arena.take<uint32_t>(n + 1);
    uint32_t* mark = arena.take<uint32_t>(n + 1);
    uint32_t* ecc = arena.take<uint32_t>(n + 1);
    uint32_t* layer = arena.take<uint32_t>(n + 1);
    uint32_t* counts = arena.take<uint32_t>(n + 1);
    double* dimv = arena.take<double>(n + 1);
    double* log_sum = arena.take<double>(n + 1);       // per r: sum over v of the log ratio
    uint64_t* ball_sum = arena.take<uint64_t>(n + 1);  // per r: sum over v of |B(v, r)|
    unsigned char* transport = arena.take<unsigned char>(sg_transport_bytes(g.max_degree + 1) + 64);
    if (arena.failed) return false;
    transport = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(transport) + 7) & ~static_cast<uintptr_t>(7));
    for (uint32_t i = 0; i <= n; ++i) {
        dist[i] = -1; dist2[i] = -1; mark[i] = 0; counts[i] = 0;
        log_sum[i] = 0.0; ball_sum[i] = 0;
    }

    // PASS 1: every vertex's balls, eccentricity, local entropy and mutual information.
    bool connected = true;
    uint32_t radius = UINT32_MAX;
    double ecc_total = 0.0, local_total = 0.0, mi_total = 0.0;
    uint32_t mi_vertices = 0;
    for (uint32_t v = 0; v < n; ++v) {
        uint32_t reached = 0;
        const uint32_t e = sg_bfs(g, v, dist, queue, layer, UINT32_MAX, reached);
        if (reached != n) connected = false;
        ecc[v] = e;
        ecc_total += e;
        if (e < radius) radius = e;
        uint64_t ball = 1;
        for (uint32_t r = 1; r <= e; ++r) {
            const uint64_t next = ball + layer[r];
            log_sum[r] += dm_log(static_cast<double>(next)) - dm_log(static_cast<double>(ball));
            ball_sum[r] += next;
            ball = next;
        }
        if (!want_local) { sg_bfs_clear(dist, queue, reached); continue; }
        // B(v, 2) is the queue's prefix of vertices at distance at most 2.
        uint32_t b2 = 0;
        while (b2 < reached && dist[queue[b2]] <= 2) ++b2;
        local_total += sg_degree_entropy_of(g, queue, b2, counts);
        for (uint32_t i = 0; i < b2; ++i) mark[queue[i]] = v + 1;
        sg_bfs_clear(dist, queue, reached);
        if (g.degree(v)) {
            double sum = 0.0;
            for (uint32_t k = g.off[v]; k < g.off[v + 1]; ++k) {
                uint32_t reached_w = 0;
                sg_bfs(g, g.nbr[k], dist2, queue2, nullptr, 2, reached_w);
                uint32_t both = 0;
                for (uint32_t i = 0; i < reached_w; ++i) if (mark[queue2[i]] == v + 1) ++both;
                sg_bfs_clear(dist2, queue2, reached_w);
                const double uni = static_cast<double>(b2) + reached_w - both;
                const double pmi = dm_log2(dm_mul(static_cast<double>(both), uni) /
                                           dm_mul(static_cast<double>(b2), reached_w));
                sum += pmi > 0.0 ? pmi : 0.0;
            }
            mi_total += sum / g.degree(v);
            ++mi_vertices;
        }
    }
    {
        uint32_t* all = queue;
        for (uint32_t i = 0; i < n; ++i) all[i] = i;
        out.degree_entropy = sg_degree_entropy_of(g, all, n, counts);
        out.local_entropy = local_total / n;
        if (want_local) out.defined |= SG_DEGREE_ENTROPY;
    }
    if (want_local && mi_vertices) {
        out.mutual_information = mi_total / mi_vertices;
        out.defined |= SG_MUTUAL_INFORMATION;
    }
    if ((want & SG_OLLIVIER) && g.edges) {
        double sum = 0.0;
        for (uint32_t x = 0; x < n; ++x)
            for (uint32_t k = g.off[x]; k < g.off[x + 1]; ++k)
                if (g.nbr[k] > x) sum += sg_ollivier_edge(g, x, g.nbr[k], transport);
        out.ollivier_ricci = sum / g.edges;
        out.defined |= SG_OLLIVIER;
    }
    if (!connected) return true;
    out.radius = static_cast<int32_t>(radius);
    out.mean_eccentricity = ecc_total / n;
    out.defined |= SG_RADIUS;
    if (radius == 0) return true;   // one vertex: no radius to average over
    if (!(want & (SG_HAUSDORFF | SG_RICCI | SG_FISHER))) return true;
    const uint32_t R = radius;

    // PASS 2: each vertex's own dimension over r = 1..R.
    double dim_total = 0.0;
    for (uint32_t v = 0; v < n; ++v) {
        uint32_t reached = 0;
        sg_bfs(g, v, dist, queue, layer, R, reached);
        sg_bfs_clear(dist, queue, reached);
        uint64_t ball = 1;
        double s = 0.0;
        for (uint32_t r = 1; r <= R; ++r) {
            const uint64_t next = ball + layer[r];
            s += (dm_log(static_cast<double>(next)) - dm_log(static_cast<double>(ball))) /
                 (dm_log(r + 1.0) - dm_log(static_cast<double>(r)));
            ball = next;
        }
        dimv[v] = s / R;
        dim_total += dimv[v];
    }
    const double d = dim_total / n;
    out.hausdorff_dimension = d;
    out.defined |= SG_HAUSDORFF;
    out.ball_radii = R;
    for (uint32_t r = 1; r <= R && r <= ball_capacity; ++r)
        ball_dimension[r - 1] =
            log_sum[r] / n / (dm_log(r + 1.0) - dm_log(static_cast<double>(r)));

    // Ricci scalar at dimension d: linear in |B(v, r)|, so the per-radius ball sums suffice.
    const double pi = 3.14159265358979323846;
    const double c = dm_tgamma(d / 2.0 + 1.0) / dm_pow(pi, d / 2.0);
    double ricci = 0.0;
    for (uint32_t r = 1; r <= R; ++r) {
        const double rr = static_cast<double>(r);
        ricci += dm_mul(dm_mul(6.0, d + 2.0) / dm_mul(rr, rr),
                        static_cast<double>(n) -
                            dm_mul(static_cast<double>(ball_sum[r]), c) / dm_pow(rr, d));
    }
    out.ricci_scalar = ricci / R / n;
    out.defined |= SG_RICCI;

    if (!(want & SG_FISHER)) return true;

    // PASS 3: Fisher information from the dimensions within distance 2.
    double fisher_total = 0.0;
    for (uint32_t v = 0; v < n; ++v) {
        uint32_t reached = 0;
        sg_bfs(g, v, dist, queue, nullptr, 2, reached);
        sg_bfs_clear(dist, queue, reached);
        const uint32_t k = reached - 1;   // queue[0] is v; n >= 2 and connected, so k >= 1
        double mean = 0.0, grad = 0.0;
        for (uint32_t i = 1; i < reached; ++i) {
            mean += dimv[queue[i]];
            grad += ::fabs(dimv[queue[i]] - dimv[v]);
        }
        mean /= k;
        grad /= k;
        double var = 0.0;
        for (uint32_t i = 1; i < reached; ++i)
            var += dm_mul(dimv[queue[i]] - mean, dimv[queue[i]] - mean);
        var /= k;
        fisher_total += (1.0 + grad) / (var + 0.01);
    }
    out.fisher_information = fisher_total / n;
    out.defined |= SG_FISHER;
    return true;
}

// Builds G from `el` in `scratch` and computes its geometry. When `capacity` is short, returns
// false with `needed` set to the bytes that would have sufficed for the stage reached; a caller
// that grows its buffer to `needed` and calls again either succeeds or learns the next stage's
// need.
template <class EdgeList>
HG_HD bool sg_state_geometry(const EdgeList& el, unsigned char* scratch, size_t capacity,
                             SgGeometry& out, double* ball_dimension, uint32_t ball_capacity,
                             size_t& needed, uint32_t want = ~0u) {
    SgArena arena;
    arena.base = scratch;
    arena.capacity = capacity;
    SgGraph g;
    size_t slots = 0, pairs = 0;
    sg_sizes(el, slots, pairs);
    if (!sg_build(el, arena, g)) {
        needed = sg_build_bytes(slots, pairs) + sg_metric_bytes(slots, 0);
        return false;
    }
    const size_t built = arena.used;
    if (!sg_geometry_of(g, arena, out, ball_dimension, ball_capacity, want)) {
        needed = built + sg_metric_bytes(g.n, g.max_degree);
        return false;
    }
    needed = arena.used;
    return true;
}

}  // namespace common
}  // namespace HG_NAMESPACE
