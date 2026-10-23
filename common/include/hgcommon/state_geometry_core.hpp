#pragma once
#include "hgcommon/namespace.hpp"
// PER-STATE GEOMETRY OF A HYPERGRAPH STATE, one body for the host and the device.
//
// A state's graph G is the undirected simple graph whose vertices are the state's vertices and
// whose edges join the vertices of each hyperedge by one of two rules ("StatisticsGraph"):
// SG_GRAPH_CONSECUTIVE joins consecutive vertices, {a, b, c} giving a-b and b-c, and
// SG_GRAPH_ALL_PAIRS joins every two vertices, {a, b, c} giving a-b, a-c and b-c. A self-loop
// is dropped and a repeated pair is one edge. A vertex that only appears in a unary hyperedge,
// or only next to itself, is an isolated vertex of G. Hypergraph isomorphism keeps
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
//   NonLazyOllivierRicciCurvature  the same with m_x = 1/deg x at each neighbour, none at x
//   EulerCharacteristic        V - E + T, T the triangles of G
//   RimFraction                the fraction of vertices whose link (the subgraph on their
//                              neighbours) is not a cycle, when every vertex lies on a triangle
//   CombinatorialCurvatureTotal  the sum of 1 - k/6 over the vertices whose link is a k-cycle
//   ClusteringCoefficient, LocalEulerExcess, TopologicalCharge, TopologicalChargeMax
//                              hypergraph_viz's particle measures (docs/SPEC.md)
//   DimensionGradientCurvature  the mean over G' of L_v = the mean over the neighbours u of v
//                              of d_u - d_v
//   GraphRadius, MeanEccentricity
//   DegreeEntropy              Shannon entropy in bits of the degree distribution
//   LocalEntropy               mean over v of the degree entropy over B(v, 2)
//   MutualInformation          mean over vertices with a neighbour of the mean over neighbours
//                              w of max(0, log2(|B2v n B2w| |B2v u B2w| / (|B2v| |B2w|)))
//   FisherInformation          mean over v of (1 + mean_u |d_u - d_v|) / (var_u d_u + 1/100),
//                              u over B(v, 2) minus v, d_u the vertex's own Hausdorff dimension
//
// G' is the largest component of G, given by the caller (state_invariants' largest component;
// G itself when G is connected), and R' its radius. d_v is the mean over r = 1..R' of the
// ball-growth term above, for v in G'.
//
//   LargestComponentDimension  mean of d_v over G'; equal to WolframHausdorffDimension when G
//                              is connected
//   LocalDimensionMax          max of d_v over G'
//   LocalDimensionStandardDeviation  population standard deviation of d_v over G'
//   k_v                        Ollivier curvature of vertex v: mean over the edges v-w of the
//                              edge's curvature, for every v with a neighbour
//   OllivierMoranI             (n1 / |E|) sum over edges x-y of z_x z_y / sum_v z_v^2, z = k - mean
//                              k, n1 the vertices with a neighbour; undefined when k is constant
//   OllivierDegreeCorrelation  Pearson correlation of k_v with deg v over the vertices with a
//                              neighbour; undefined when either is constant
//   K_v                        Wolfram-Ricci scalar of v in G': mean over r = 1..R' of
//                              6(D+2)/r^2 (1 - |B(v,r)| Gamma(D/2+1) / (pi^(D/2) r^D)), D the
//                              LargestComponentDimension
//   growth[0..6)               volume growth (sg_growth_of): the means over the vertices of G'
//                              with a fit of the dimension and scalar curvature fitted to the
//                              ball sizes, to the Riemannian measures of the balls and to the
//                              sphere sizes about the vertex
// The per-vertex values d_v (over G'), k_v (over G), K_v (over G') and the six volume-growth
// values (over the vertices of G' with a fit) are kept as SgDistribution
// summaries: count, min, max, the sums of x, x^2, x^3 and x^4 in double-double (det_math.hpp),
// and a histogram of SG_DIST_BINS
// equal bins over a fixed range per quantity (sg_dist_range); a value outside the range counts in
// the end bin.
//
// The logarithms, powers and Gamma function are det_math.hpp's, and every product that feeds a sum
// is dm_mul, so the host and the device compute the same doubles.
//
// NO ALLOCATION. Every array lives in caller memory handed over as a byte buffer. Building G
// needs sg_build_bytes(slots, pairs) and the metrics need sg_metric_bytes(n, max_degree, edges,
// lanes) more; sg_state_geometry reports the total it needed when the buffer is short, and
// computes nothing.
//
// COST. One breadth-first search per vertex for the balls and eccentricities, a second for the
// per-vertex dimensions the Fisher information reads, a third for the per-vertex Ricci scalars,
// and radius-2 searches for the local measures: O(n (n + m)) per state. Each edge's transport
// problem is solved exactly by successive shortest paths on the (deg x + 1) x (deg y + 1)
// supports. The searches and the
// transport problems are spread over the lanes of the policy (IrSerial on the host, IrTile on
// the device), as state_invariants_core.hpp spreads its distance pass.

#include <math.h>

#include <cstddef>
#include <cstdint>

#include "hgcommon/core.hpp"
#include "hgcommon/det_math.hpp"
#include "hgcommon/ir_core.hpp"

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

// The rule that joins the vertices of a hyperedge in G.
enum : uint32_t {
    SG_GRAPH_CONSECUTIVE = 0,   // vertex j and vertex j + 1
    SG_GRAPH_ALL_PAIRS = 1,     // every two vertices
};

// An EdgeList supplies   uint32_t count() const;
//                        uint32_t arity(uint32_t i) const;
//                        uint32_t at(uint32_t i, uint32_t k) const;   vertex k of hyperedge i
template <class EdgeList>
HG_HD void sg_sizes(const EdgeList& el, size_t& slots, size_t& pairs,
                    uint32_t graph = SG_GRAPH_CONSECUTIVE) {
    slots = 0;
    pairs = 0;
    const uint32_t m = el.count();
    for (uint32_t i = 0; i < m; ++i) {
        const size_t k = el.arity(i);
        slots += k;
        if (k > 1) pairs += graph == SG_GRAPH_ALL_PAIRS ? k * (k - 1) / 2 : k - 1;
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

// Builds G from a state's hyperedges under `graph`. False when the arena is short.
template <class EdgeList>
HG_HD bool sg_build(const EdgeList& el, SgArena& arena, SgGraph& g,
                    uint32_t graph = SG_GRAPH_CONSECUTIVE) {
    size_t slots = 0, pairs = 0;
    sg_sizes(el, slots, pairs, graph);
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
            const uint32_t last = graph == SG_GRAPH_ALL_PAIRS ? k - 1 : j + 1;
            for (uint32_t l = j + 1; l <= last; ++l) {
                const uint64_t a = sg_index_of(verts, n, el.at(i, j));
                const uint64_t b = sg_index_of(verts, n, el.at(i, l));
                if (a == b) continue;
                keys[p++] = (a << 32) | b;
                keys[p++] = (b << 32) | a;
            }
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
//
// `inner`, when given with max_depth UINT32_MAX, receives at index h the number of vertices u
// with h(u) = h, where h(u) is dist(u) + 1 when u has a neighbour at dist(u) + 1 and dist(u)
// otherwise. B(s, r) holds exactly the vertices u with h(u) <= r whose neighbours all lie in
// B(s, r), so the Riemannian measure of B(s, r) is the sum of inner[0..r].
//
// `sigma` and `acc`, when given, receive for each vertex q found sigma[q] = sigma(s, q), the
// number of shortest paths from s to q, and acc[q] = the sum over those paths of the sum over
// their vertices w of log2 sigma(s, w). Both are doubles: sigma is exact below 2^53.
HG_HD inline uint32_t sg_bfs(const SgGraph& g, uint32_t s, int32_t* dist, uint32_t* queue,
                             uint32_t* layer, uint32_t max_depth, uint32_t& reached,
                             uint32_t* inner = nullptr, double* sigma = nullptr,
                             double* acc = nullptr) {
    uint32_t head = 0, tail = 0;
    queue[tail++] = s;
    dist[s] = 0;
    uint32_t depth = 0;
    if (layer) layer[0] = 0;
    if (inner) { inner[0] = 0; inner[1] = 0; }
    if (sigma) { sigma[s] = 1.0; acc[s] = 0.0; }
    while (head < tail) {
        const uint32_t u = queue[head++];
        const uint32_t du = static_cast<uint32_t>(dist[u]);
        if (du > depth) {
            depth = du;
            if (layer) layer[depth] = 0;
            if (inner) inner[depth + 1] = 0;
        }
        if (layer) ++layer[du];
        // sigma[u] is final here: every neighbour of u at distance du - 1 left the queue earlier.
        if (sigma && du > 0) acc[u] += dm_mul(sigma[u], dm_log2(sigma[u]));
        if (du == max_depth) continue;
        uint32_t outward = 0;
        for (uint32_t e = g.off[u]; e < g.off[u + 1]; ++e) {
            const uint32_t w = g.nbr[e];
            if (dist[w] < 0) {
                dist[w] = static_cast<int32_t>(du + 1);
                queue[tail++] = w;
                outward = 1;
                if (sigma) { sigma[w] = sigma[u]; acc[w] = acc[u]; }
            } else if (dist[w] == static_cast<int32_t>(du + 1)) {
                outward = 1;
                if (sigma) { sigma[w] += sigma[u]; acc[w] += acc[u]; }
            }
        }
        if (inner) ++inner[du + outward];
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

// The Ollivier curvature of edge x-y, 1 - W1, W1 the exact optimal transport cost under the
// graph distance. With `lazy` the walk has idleness 1/2: 1/2 at the vertex and 1/2 spread over its
// neighbours, the masses scaled to integers by L = 2 deg(x) deg(y). Without it the walk has
// idleness 0: 1/deg x on each neighbour of x and nothing on x, scaled by L = deg(x) deg(y). The
// transport problem is solved by successive shortest paths, so the cost is an exact integer
// before the one division by L.
HG_HD inline double sg_ollivier_edge(const SgGraph& g, uint32_t x, uint32_t y,
                                     unsigned char* scratch, bool lazy = true) {
    const uint32_t dx = g.degree(x), dy = g.degree(y);
    const uint32_t c = lazy ? 1u : 0u;   // 1 when the vertex itself is a point of its support
    const uint32_t a = dx + c, b = dy + c;
    // Aligned carve of the scratch sized by sg_transport_bytes(max_degree + 1).
    uint32_t* cost = reinterpret_cast<uint32_t*>(scratch);
    uint32_t* flow = cost + static_cast<size_t>(a) * b;
    uint32_t* supply = flow + static_cast<size_t>(a) * b;   // remaining supply at A points
    uint32_t* demand = supply + a;                          // remaining demand at B points
    const uint32_t nodes = a + b + 2;
    int32_t* dist = reinterpret_cast<int32_t*>(demand + b);
    uint32_t* prev = reinterpret_cast<uint32_t*>(dist + nodes);
    uint32_t* inq = prev + nodes;
    uint32_t* queue = inq + nodes;
    auto pa = [&](uint32_t i) { return c && i == 0 ? x : g.nbr[g.off[x] + i - c]; };
    auto pb = [&](uint32_t j) { return c && j == 0 ? y : g.nbr[g.off[y] + j - c]; };
    for (uint32_t i = 0; i < a; ++i)
        for (uint32_t j = 0; j < b; ++j) {
            cost[i * b + j] = sg_short_distance(g, pa(i), pb(j));
            flow[i * b + j] = 0;
        }
    for (uint32_t i = 0; i < a; ++i) supply[i] = c && i == 0 ? dx * dy : dy;
    for (uint32_t j = 0; j < b; ++j) demand[j] = c && j == 0 ? dx * dy : dx;
    const uint64_t total = (lazy ? 2ull : 1ull) * dx * dy;
    uint64_t moved = 0, total_cost = 0;
    // At a vertex in both supports the smaller of its two masses stays in place at cost 0. Some
    // optimal plan leaves it there: under a metric, moving it out and other mass in costs no
    // less. The shortest paths carry the remaining mass.
    for (uint32_t i = 0; i < a; ++i)
        for (uint32_t j = 0; j < b && supply[i]; ++j) {
            if (cost[i * b + j] != 0 || !demand[j]) continue;
            const uint32_t m = supply[i] < demand[j] ? supply[i] : demand[j];
            supply[i] -= m;
            demand[j] -= m;
            flow[i * b + j] += m;
            moved += m;
        }
    // Node numbering: 0 source, 1..a the A points, a+1..a+b the B points, a+b+1 the sink.
    const uint32_t src = 0, sink = a + b + 1;
    const int32_t inf = INT32_MAX / 2;
    while (moved < total) {
        for (uint32_t v = 0; v < nodes; ++v) { dist[v] = inf; inq[v] = 0; }
        uint32_t head = 0, count = 0;
        auto push = [&](uint32_t v) {
            if (inq[v]) return;
            inq[v] = 1;
            uint32_t at = head + count;
            if (at >= nodes) at -= nodes;
            queue[at] = v;
            ++count;
        };
        dist[src] = 0;
        push(src);
        while (count) {
            const uint32_t u = queue[head];
            head = head + 1 == nodes ? 0 : head + 1;
            --count;
            inq[u] = 0;
            auto relax = [&](uint32_t v, int32_t w) {
                if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; prev[v] = u; push(v); }
            };
            if (u == src) {
                for (uint32_t i = 0; i < a; ++i) if (supply[i]) relax(1 + i, 0);
            } else if (u <= a) {
                const uint32_t i = u - 1;
                for (uint32_t j = 0; j < b; ++j) relax(1 + a + j, static_cast<int32_t>(cost[i * b + j]));
            } else if (u < sink) {
                const uint32_t j = u - 1 - a;
                if (demand[j]) relax(sink, 0);
                for (uint32_t i = 0; i < a; ++i)
                    if (flow[i * b + j]) relax(1 + i, -static_cast<int32_t>(cost[i * b + j]));
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
    SG_LARGEST_DIMENSION = 1u << 7,      // largest_dimension, local_dimension_max/_sd: |G'| >= 2
    SG_OLLIVIER_MORAN = 1u << 8,         // ollivier_moran_i: G has an edge, k not constant
    SG_OLLIVIER_DEGREE = 1u << 9,        // ollivier_degree_correlation: k and deg not constant
    SG_VERTEX_DISTRIBUTIONS = 1u << 10,  // `want` only: write the per-vertex distributions
    SG_VOLUME_GROWTH = 1u << 11,         // the six growth_* means: a vertex of G' has a fit
    SG_GEODESIC = 1u << 12,              // geodetic_fraction, segment_entropy: G has an edge
    SG_NONLAZY_OLLIVIER = 1u << 13,      // nonlazy_ollivier: G has an edge
    SG_EULER = 1u << 14,                 // euler_characteristic: n >= 1
    SG_RIM = 1u << 15,                   // rim_fraction: every vertex lies on a triangle
    SG_CYCLE_CURVATURE = 1u << 16,       // cycle_curvature_total: a vertex's link is a cycle
    SG_PARTICLE = 1u << 17,              // clustering, euler_excess, charge, charge_max: n >= 1
    SG_DIMENSION_GRADIENT = 1u << 18,    // dimension_gradient: |G'| >= 2
};

// "StepStatisticsMeasures": the `want` bits of each measure family. SG_MEASURES_BASE is the set
// computed without the option.
constexpr uint32_t SG_MEASURES_BASE = (1u << 11) - 1;
constexpr uint32_t SG_COMBINATORIAL = SG_EULER | SG_RIM | SG_CYCLE_CURVATURE;
constexpr uint32_t SG_MEASURES_ALL = SG_MEASURES_BASE | SG_VOLUME_GROWTH | SG_GEODESIC |
                                     SG_NONLAZY_OLLIVIER | SG_COMBINATORIAL | SG_PARTICLE |
                                     SG_DIMENSION_GRADIENT;

// The per-vertex distributions of a state, in this order.
enum : uint32_t {
    SG_DIST_LOCAL_DIMENSION = 0,         // d_v over G'
    SG_DIST_OLLIVIER = 1,                // k_v over the vertices of G with a neighbour
    SG_DIST_RICCI = 2,                   // K_v over G'
    SG_DIST_BALL_DIMENSION = 3,          // the volume-growth fits of the vertices of G' with one:
    SG_DIST_BALL_CURVATURE = 4,          //   counting ball dimension and scalar curvature,
    SG_DIST_RIEMANNIAN_DIMENSION = 5,    //   Riemannian ball dimension and scalar curvature,
    SG_DIST_RIEMANNIAN_CURVATURE = 6,
    SG_DIST_SPHERE_DIMENSION = 7,        //   sphere dimension and scalar curvature
    SG_DIST_SPHERE_CURVATURE = 8,
    SG_DIST_NONLAZY_OLLIVIER = 9,        // kappa0 over the edges of G
    SG_DIST_CYCLE_CURVATURE = 10,        // 1 - k/6 over the vertices whose link is a k-cycle
    SG_DIST_CLUSTERING = 11,             // C_v over the vertices of G
    SG_DIST_CHARGE = 12,                 // q_v over the vertices of G
    SG_DIST_DIMENSION_GRADIENT = 13,     // L_v over G'
    SG_DISTS = 14,
};
constexpr uint32_t SG_DIST_BINS = 32;

// The number of distributions, in SG_DIST order, a record keeps under `want`: through the last
// one a requested family writes.
HG_HD inline uint32_t sg_dists_for(uint32_t want) {
    if (want & SG_DIMENSION_GRADIENT) return SG_DIST_DIMENSION_GRADIENT + 1;
    if (want & SG_PARTICLE) return SG_DIST_CHARGE + 1;
    if (want & SG_COMBINATORIAL) return SG_DIST_CYCLE_CURVATURE + 1;
    if (want & SG_NONLAZY_OLLIVIER) return SG_DIST_NONLAZY_OLLIVIER + 1;
    if (want & SG_VOLUME_GROWTH) return SG_DIST_SPHERE_CURVATURE + 1;
    return SG_DIST_RICCI + 1;
}

// The volume-growth fits in SG_DIST order from SG_DIST_BALL_DIMENSION: per probe and measure, the
// dimension then the scalar curvature.
constexpr uint32_t SG_GROWTH_VALUES = 6;
// The per-vertex values sg_geometry_of writes to `vertex_values`: the volume-growth fits, then
// the combinatorial curvature, the clustering coefficient, the Euler excess, the charge and the
// dimension-gradient curvature.
constexpr uint32_t SG_VERTEX_VALUES = SG_GROWTH_VALUES + 5;

// Histogram range [lo, hi) of a distribution; the bin width (hi - lo) / SG_DIST_BINS is a power
// of two times 1 or 3, so a value's bin is one exact subtraction and one division.
HG_HD inline void sg_dist_range(uint32_t which, double& lo, double& hi) {
    if (which == SG_DIST_LOCAL_DIMENSION || which == SG_DIST_BALL_DIMENSION ||
        which == SG_DIST_RIEMANNIAN_DIMENSION || which == SG_DIST_SPHERE_DIMENSION) {
        lo = 0.0; hi = 8.0;
    } else if (which == SG_DIST_OLLIVIER || which == SG_DIST_NONLAZY_OLLIVIER) {
        lo = -2.0; hi = 1.0;
    } else if (which == SG_DIST_CYCLE_CURVATURE) {
        lo = -7.0; hi = 1.0;
    } else if (which == SG_DIST_CLUSTERING) {
        lo = 0.0; hi = 1.0;
    } else if (which == SG_DIST_CHARGE) {
        lo = 0.0; hi = 16.0;
    } else if (which == SG_DIST_DIMENSION_GRADIENT) {
        lo = -8.0; hi = 8.0;
    } else {
        lo = -24.0; hi = 8.0;
    }
}

// A per-vertex distribution: the count, extremes, sum[k] + sum_lo[k] = the sum of x^(k+1) in
// vertex order as a double-double (each power exact, each addition dm_dd_add), and the
// histogram.
struct SgDistribution {
    uint32_t count = 0;
    uint32_t reserved = 0;
    double min = 0.0;
    double max = 0.0;
    double sum[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_lo[4] = {0.0, 0.0, 0.0, 0.0};
    uint32_t bins[SG_DIST_BINS] = {};
};

// The bin of x. A value less than 1e-9 bin widths below a bin's lower edge counts in that bin:
// a value whose exact result is an edge, as the dimension 1 of a profile r, may come out a few
// units in the last place below it.
HG_HD inline uint32_t sg_dist_bin(uint32_t which, double x) {
    double lo = 0.0, hi = 0.0;
    sg_dist_range(which, lo, hi);
    if (!(x > lo)) return 0;
    if (x >= hi) return SG_DIST_BINS - 1;
    const double b = ::floor((x - lo) / ((hi - lo) / SG_DIST_BINS) + 1e-9);
    return b >= SG_DIST_BINS - 1 ? SG_DIST_BINS - 1 : static_cast<uint32_t>(b);
}

HG_HD inline void sg_dist_add(SgDistribution& d, uint32_t which, double x) {
    if (d.count == 0 || x < d.min) d.min = x;
    if (d.count == 0 || x > d.max) d.max = x;
    const DmDD x2 = dm_two_prod(x, x);
    const DmDD power[4] = {DmDD{x, 0.0}, x2, dm_dd_mul(x2, DmDD{x, 0.0}), dm_dd_mul(x2, x2)};
    for (uint32_t k = 0; k < 4; ++k) {
        const DmDD s = dm_dd_add(DmDD{d.sum[k], d.sum_lo[k]}, power[k]);
        d.sum[k] = s.hi;
        d.sum_lo[k] = s.lo;
    }
    ++d.bins[sg_dist_bin(which, x)];
    ++d.count;
}

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
    double largest_dimension = 0.0;
    double local_dimension_max = 0.0;
    double local_dimension_sd = 0.0;
    double ollivier_moran_i = 0.0;
    double ollivier_degree_correlation = 0.0;
    // Means over the vertices of G' with a volume-growth fit, in SG_DIST order from
    // SG_DIST_BALL_DIMENSION.
    double growth[SG_GROWTH_VALUES] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // Over the ordered pairs (p, q) of distinct vertices joined by a path: the fraction with one
    // shortest path, and the mean entropy of the shortest paths' vertex density.
    double geodetic_fraction = 0.0;
    double segment_entropy = 0.0;
    uint32_t geodesic_radii = 0;         // D: entries written to the log-geodesic-count output
    uint32_t reserved = 0;
    double nonlazy_ollivier = 0.0;       // the mean over edges of the idleness-0 curvature
    // V - E + T, T the triangles of G; the fraction of vertices whose link is not a cycle; the sum
    // of 1 - k/6 over the vertices whose link is a k-cycle.
    double euler_characteristic = 0.0;
    double rim_fraction = 0.0;
    double cycle_curvature_total = 0.0;
    // The means over the vertices of G of the clustering coefficient C_v, the local Euler excess
    // x_v and the topological charge q_v, and the largest q_v.
    double clustering = 0.0;
    double euler_excess = 0.0;
    double charge = 0.0;
    double charge_max = 0.0;
    double dimension_gradient = 0.0;     // the mean over G' of L_v, the Laplacian of d_v
};

// Doubles of scratch sg_growth_fit takes for k quotients.
HG_HD inline size_t sg_fit_doubles(size_t k) { return 7 * (k + 1); }

// Bytes of one lane's transport scratch, a multiple of 8.
HG_HD inline size_t sg_transport_stride(size_t max_degree) {
    return (sg_transport_bytes(max_degree + 1) + 7) & ~size_t{7};
}

// Lanes that solve Ollivier transport problems at once, out of `lanes`: as many as fit in 1 MB
// of transport scratch, and at least one.
HG_HD inline uint32_t sg_transport_lanes(size_t max_degree, uint32_t lanes) {
    const size_t fit = (size_t{1} << 20) / sg_transport_stride(max_degree);
    return fit == 0 ? 1u : (fit < lanes ? static_cast<uint32_t>(fit) : lanes);
}

// Bytes sg_geometry_of takes for a graph of n vertices, the given maximum degree and edge count,
// on `lanes` lanes, computing `want`. Per lane and per vertex: dist, dist2, queue, queue2, mark,
// layer and counts (4 bytes each) and the log term (8 bytes). Per vertex: the eccentricity (4
// bytes), and the dimension, local entropy, mutual information, Fisher term, log sum, ball sum,
// vertex curvature, vertex Ricci scalar and the two per-radius Ricci coefficients (8 bytes
// each). Per edge: the Ollivier curvature (8 bytes). Then the transport lanes' scratch, and 8
// bytes of alignment per array. SG_VOLUME_GROWTH adds per lane and per vertex the shells and the
// Riemannian counts (4 bytes each) and the fit scratch (56 bytes), and per vertex the six fits.
// SG_NONLAZY_OLLIVIER adds per edge the idleness-0 curvature (8 bytes). SG_EULER, SG_RIM,
// SG_CYCLE_CURVATURE and SG_PARTICLE add per vertex its triangle count (4 bytes) and its
// curvature (8 bytes), and SG_PARTICLE its Euler excess (8 bytes).
// SG_GEODESIC adds per lane and per vertex the path counts, their log sums and the per-distance
// log sums (8 bytes each), and per vertex the one-path count (4 bytes), the entropy sum and the
// pooled per-distance log sums and pair counts (8 bytes each).
HG_HD inline size_t sg_metric_bytes(size_t n, size_t max_degree, size_t edges, uint32_t lanes,
                                    uint32_t want = ~0u) {
    const size_t n1 = n + 1;
    size_t bytes = 64 + size_t{lanes} * n1 * (7 * 4 + 8) + n1 * 4 + 10 * n1 * 8 +
                   8 * (edges + 1) +
                   sg_transport_lanes(max_degree, lanes) * sg_transport_stride(max_degree) + 24 * 8;
    if (want & SG_VOLUME_GROWTH)
        bytes += size_t{lanes} * (n1 * 2 * 4 + sg_fit_doubles(n) * 8) +
                 SG_GROWTH_VALUES * n1 * 8 + 4 * 8;
    if (want & SG_GEODESIC) bytes += size_t{lanes} * n1 * 3 * 8 + n1 * (4 + 3 * 8) + 5 * 8;
    if (want & SG_NONLAZY_OLLIVIER) bytes += 8 * (edges + 1) + 8;
    if (want & (SG_EULER | SG_RIM | SG_CYCLE_CURVATURE | SG_PARTICLE))
        bytes += n1 * (4 + 8) + 2 * 8;
    if (want & SG_PARTICLE) bytes += n1 * 8 + 8;
    return bytes;
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

// THE LINK of v, the subgraph of G on v's neighbours. `twice_triangles` receives the sum over the
// neighbours u of |N(u) n N(v)|, twice the triangles at v. Returns k when the link is a k-cycle
// (k >= 3 vertices, each with two link neighbours, connected), else 0.
HG_HD inline uint32_t sg_link_cycle(const SgGraph& g, uint32_t v, uint32_t& twice_triangles) {
    // The link neighbours of u: the common neighbours of u and v, ascending; f(w) for each.
    auto common = [&](uint32_t u, auto&& f) {
        uint32_t i = g.off[u], j = g.off[v], c = 0;
        while (i < g.off[u + 1] && j < g.off[v + 1]) {
            if (g.nbr[i] == g.nbr[j]) { f(g.nbr[i]); ++c; ++i; ++j; }
            else if (g.nbr[i] < g.nbr[j]) ++i;
            else ++j;
        }
        return c;
    };
    const uint32_t k = g.degree(v);
    twice_triangles = 0;
    bool two = true;
    for (uint32_t e = g.off[v]; e < g.off[v + 1]; ++e) {
        const uint32_t c = common(g.nbr[e], [](uint32_t) {});
        twice_triangles += c;
        if (c != 2) two = false;
    }
    if (!two || k < 3) return 0;
    // Every link vertex has two link neighbours, so the link is disjoint cycles: walk the one
    // through the first neighbour and compare its length with k.
    const uint32_t first = g.nbr[g.off[v]];
    uint32_t prev = first, at = first, steps = 0;
    do {
        uint32_t next = UINT32_MAX;
        common(at, [&](uint32_t w) { if (next == UINT32_MAX && (w != prev || steps == 0)) next = w; });
        prev = at;
        at = next;
        ++steps;
    } while (at != first && steps <= k);
    return steps == k ? k : 0;
}

// VOLUME GROWTH (docs/SPEC.md, "StepStatistics volume growth"). The log-difference quotient of a
// profile f at radius r is q(r) = (log f(r+1) - log f(r)) / (log(r+1) - log r), r = 1..k. A fit
// regresses q on x = r(r+1) by least squares, q = c1 + c2 x, over a window of radii: the ball
// probe gives the dimension c1 and the scalar curvature -3(c1+2)c2, the sphere probe the
// dimension c1 + 1 and the scalar curvature -3(c1+1)c2. For k < 5 the window is 1..k. Otherwise
// e(i, j) is the residual standard error of the regression over i..j, tol = max(2 Q, 1e-10)
// with Q the ceil((k-4)/4)-th smallest e over the k - 4 windows of 5 radii, and the window is
// the longest i..j with e(i, j) <= tol, the leftmost of equal length, or 1..k when none is.
enum : uint32_t { SG_PROBE_BALL = 0, SG_PROBE_SPHERE = 1 };

// The fit of the quotients s[1..k] (s holds sg_fit_doubles(k) doubles); false when k < 2.
HG_HD inline bool sg_growth_fit(double* s, uint32_t k, uint32_t probe, double& dim,
                                double& curv) {
    if (k < 2) return false;
    double* q = s;
    double* px = q + (k + 1);
    double* pxx = px + (k + 1);
    double* pq = pxx + (k + 1);
    double* pqq = pq + (k + 1);
    double* pxq = pqq + (k + 1);
    double* e = pxq + (k + 1);
    px[0] = pxx[0] = pq[0] = pqq[0] = pxq[0] = 0.0;
    for (uint32_t r = 1; r <= k; ++r) {
        const double x = static_cast<double>(r) * static_cast<double>(r + 1);
        px[r] = px[r - 1] + x;
        pxx[r] = pxx[r - 1] + dm_mul(x, x);
        pq[r] = pq[r - 1] + q[r];
        pqq[r] = pqq[r - 1] + dm_mul(q[r], q[r]);
        pxq[r] = pxq[r - 1] + dm_mul(x, q[r]);
    }
    auto rse = [&](uint32_t i, uint32_t j) {
        const double m = static_cast<double>(j - i + 1);
        const double ax = px[j] - px[i - 1], axx = pxx[j] - pxx[i - 1];
        const double aq = pq[j] - pq[i - 1], aqq = pqq[j] - pqq[i - 1], axq = pxq[j] - pxq[i - 1];
        const double b = (dm_mul(m, axq) - dm_mul(ax, aq)) / (dm_mul(m, axx) - dm_mul(ax, ax));
        const double ss = aqq - dm_mul(aq - dm_mul(b, ax), aq) / m - dm_mul(b, axq);
        return ::sqrt((ss > 0.0 ? ss : 0.0) / (m - 2.0));
    };
    uint32_t lo = 1, hi = k;
    if (k >= 5) {
        const uint32_t w = k - 4;
        for (uint32_t i = 1; i <= w; ++i) e[i - 1] = rse(i, i + 4);
        sg_sort(e, w);
        double tol = 2.0 * e[(w + 3) / 4 - 1];
        if (!(tol > 1e-10)) tol = 1e-10;
        bool found = false;
        for (uint32_t len = k; len >= 5 && !found; --len)
            for (uint32_t i = 1; i + len - 1 <= k; ++i)
                if (rse(i, i + len - 1) <= tol) {
                    lo = i;
                    hi = i + len - 1;
                    found = true;
                    break;
                }
    }
    double m = 0.0, sx = 0.0, sxx = 0.0, sq = 0.0, sxq = 0.0;
    for (uint32_t r = lo; r <= hi; ++r) {
        const double x = static_cast<double>(r) * static_cast<double>(r + 1);
        m += 1.0;
        sx += x;
        sxx += dm_mul(x, x);
        sq += q[r];
        sxq += dm_mul(x, q[r]);
    }
    const double den = dm_mul(m, sxx) - dm_mul(sx, sx);
    const double c2 = (dm_mul(m, sxq) - dm_mul(sx, sq)) / den;
    const double c1 = (dm_mul(sxx, sq) - dm_mul(sx, sxq)) / den;
    if (probe == SG_PROBE_BALL) {
        dim = c1;
        curv = dm_mul(dm_mul(-3.0, c1 + 2.0), c2);
    } else {
        dim = c1 + 1.0;
        curv = dm_mul(dm_mul(-3.0, c1 + 1.0), c2);
    }
    return true;
}

// The volume-growth values of a vertex of eccentricity `ecc` into out[0..SG_GROWTH_VALUES), in
// SG_DIST order from SG_DIST_BALL_DIMENSION, NaN when undefined. The profiles, r = 0..ecc, are
// the shells A(r) = |S(v, r)| (`shell`), the balls V(r) = |B(v, r)| (`ball`) and the Riemannian
// measures W(r) (`riem`). Each is cut at p, the largest r at which A(r) is greatest, and fitted
// over its quotients at r = 1..p-1: V and W by the ball probe, A by the sphere probe. Undefined
// when p < 3. `s` holds sg_fit_doubles(ecc) doubles.
HG_HD inline void sg_growth_of(const uint32_t* shell, const uint32_t* ball, const uint32_t* riem,
                               uint32_t ecc, double* s, double* out) {
    uint32_t p = 0;
    for (uint32_t r = 1; r <= ecc; ++r)
        if (shell[r] >= shell[p]) p = r;
    for (uint32_t i = 0; i < SG_GROWTH_VALUES; ++i) out[i] = ::nan("");
    if (p < 3) return;
    const uint32_t k = p - 1;
    const uint32_t* profile[3] = {ball, riem, shell};
    for (uint32_t which = 0; which < 3; ++which) {
        const uint32_t* f = profile[which];
        for (uint32_t r = 1; r <= k; ++r)
            s[r] = (dm_log(static_cast<double>(f[r + 1])) - dm_log(static_cast<double>(f[r]))) /
                   (dm_log(r + 1.0) - dm_log(static_cast<double>(r)));
        sg_growth_fit(s, k, which == 2 ? SG_PROBE_SPHERE : SG_PROBE_BALL, out[2 * which],
                      out[2 * which + 1]);
    }
}

// Computes the geometry of G already built in `arena`. `ball_dimension` receives the mean
// over vertices of the ball-growth term at r = 1..R in entries 0..R-1, up to `ball_capacity`
// entries. `want` is a mask of the SG_* values to compute: the radius and eccentricity are
// always computed, SG_DEGREE_ENTROPY and SG_MUTUAL_INFORMATION each bring both, and SG_RICCI
// and SG_FISHER each bring SG_HAUSDORFF. False when the arena is short.
//
// LANES. Every lane of the policy calls it with the same arguments; `out` and `ball_dimension`
// are written by the leader. Each lane has its own search arrays. The per-vertex searches run
// with the vertices strided over the lanes, and the Ollivier transport problems with the edges
// strided over sg_transport_lanes lanes. Each vertex's and each edge's real values are stored,
// and every sum over vertices or edges is taken in vertex or edge order, so the result is the
// same double for every lane count.
//
// G' is `largest` (largest_n vertex indices of G, ascending) when G is disconnected, and G when
// it is connected; with no `largest` a disconnected G has no G'. `dists`, when given, receives
// the SG_DISTS per-vertex distributions under SG_VERTEX_DISTRIBUTIONS, written by the leader.
// `vertex_curvature`, when given, receives k_v for each vertex index v of G (the vertices in
// ascending id order), NaN for a vertex with no neighbour, written by the leader.
// `vertex_values`, when given, receives SG_VERTEX_VALUES values of each vertex index v of G at
// [k n + v], written by the leader: with SG_VOLUME_GROWTH for k < SG_GROWTH_VALUES the
// volume-growth values in SG_DIST order from SG_DIST_BALL_DIMENSION, NaN for a vertex outside G'
// or without a fit, with SG_EULER, SG_RIM or SG_CYCLE_CURVATURE for k = SG_GROWTH_VALUES the
// combinatorial curvature, NaN for a vertex whose link is not a cycle, and with SG_PARTICLE for
// k = SG_GROWTH_VALUES + 1, + 2 and + 3 the clustering coefficient, the Euler excess and the
// topological charge, and with SG_DIMENSION_GRADIENT for k = SG_GROWTH_VALUES + 4 L_v, NaN for
// a vertex outside G'.
// `log_geodesic`, with SG_GEODESIC, receives in entries 0..D-1, up to `log_geodesic_capacity`
// entries, the mean over the ordered pairs at distance d = 1..D of ln sigma, D the largest
// eccentricity; out.geodesic_radii is D.
template <class Par = IrSerial>
HG_HD inline bool sg_geometry_of(const SgGraph& g, SgArena& arena, SgGeometry& out,
                                 double* ball_dimension, uint32_t ball_capacity,
                                 uint32_t want = ~0u, Par par = Par{},
                                 const uint32_t* largest = nullptr, uint32_t largest_n = 0,
                                 SgDistribution* dists = nullptr,
                                 double* vertex_curvature = nullptr,
                                 double* vertex_values = nullptr,
                                 double* log_geodesic = nullptr,
                                 uint32_t log_geodesic_capacity = 0) {
    const bool want_growth = (want & SG_VOLUME_GROWTH) != 0;
    const bool want_geodesic = (want & SG_GEODESIC) != 0;
    const bool want_nonlazy = (want & SG_NONLAZY_OLLIVIER) != 0;
    const bool want_lazy = (want & SG_OLLIVIER) != 0;
    const bool want_local = (want & (SG_DEGREE_ENTROPY | SG_MUTUAL_INFORMATION)) != 0;
    const bool want_dists = dists != nullptr && (want & SG_VERTEX_DISTRIBUTIONS) != 0;
    const bool want_vertex_ollivier = want_dists || vertex_curvature != nullptr ||
                                      (want & (SG_OLLIVIER_MORAN | SG_OLLIVIER_DEGREE)) != 0;
    const bool want_gradient = (want & SG_DIMENSION_GRADIENT) != 0;
    const bool want_sub = want_dists || (want & SG_LARGEST_DIMENSION) != 0 || want_gradient;
    const uint32_t n = g.n;
    const uint32_t L = par.width(), me = par.rank();
    if (par.leader()) {
        out = SgGeometry{};
        out.vertex_count = n;
        out.edge_count = g.edges;
        if (dists)
            for (uint32_t k = 0; k < SG_DISTS; ++k) dists[k] = SgDistribution{};
        if (vertex_values)
            for (size_t i = 0; i < size_t{SG_VERTEX_VALUES} * n; ++i) vertex_values[i] = ::nan("");
    }
    if (n == 0) return true;
    const size_t n1 = size_t{n} + 1;
    const uint32_t TL = sg_transport_lanes(g.max_degree, L);
    const size_t tstride = sg_transport_stride(g.max_degree);
    int32_t* dist_all = arena.take<int32_t>(L * n1);
    int32_t* dist2_all = arena.take<int32_t>(L * n1);
    uint32_t* queue_all = arena.take<uint32_t>(L * n1);
    uint32_t* queue2_all = arena.take<uint32_t>(L * n1);
    uint32_t* mark_all = arena.take<uint32_t>(L * n1);
    uint32_t* layer_all = arena.take<uint32_t>(L * n1);    // per r: |B(v, r)| after pass 1's search
    uint32_t* counts_all = arena.take<uint32_t>(L * n1);
    double* term_all = arena.take<double>(L * n1);         // per r: log|B(v, r)| - log|B(v, r-1)|
    uint32_t* ecc = arena.take<uint32_t>(n1);
    double* dimv = arena.take<double>(n1);
    double* localv = arena.take<double>(n1);
    double* miv = arena.take<double>(n1);
    double* fishv = arena.take<double>(n1);
    double* log_sum = arena.take<double>(n1);              // per r: sum over v of the log term
    uint64_t* ball_sum = arena.take<uint64_t>(n1);         // per r: sum over v of |B(v, r)|
    double* oll = arena.take<double>(size_t{g.edges} + 1);
    double* ollv = arena.take<double>(n1);                 // per vertex: k_v
    double* ricv = arena.take<double>(n1);                 // per vertex: K_v
    double* ric_a = arena.take<double>(n1);                // per r: 6(D+2)/r^2
    double* ric_p = arena.take<double>(n1);                // per r: r^D
    uint64_t* transport_all = arena.take<uint64_t>(TL * tstride / 8);
    // Volume growth: per lane the shells, the Riemannian counts and the fit scratch; per vertex
    // the fits, value k of vertex v at growth_v[k n1 + v].
    const size_t fit_stride = sg_fit_doubles(n);
    uint32_t* shell_all = want_growth ? arena.take<uint32_t>(L * n1) : nullptr;
    uint32_t* inner_all = want_growth ? arena.take<uint32_t>(L * n1) : nullptr;
    double* fit_all = want_growth ? arena.take<double>(L * fit_stride) : nullptr;
    double* growth_v = want_growth ? arena.take<double>(SG_GROWTH_VALUES * n1) : nullptr;
    // Geodesic multiplicity: per lane the path counts, their log sums and the per-distance sums
    // of ln sigma; per vertex the targets reached by one path and the entropy sum; per distance
    // the pooled sums of ln sigma and the pair counts.
    double* sigma_all = want_geodesic ? arena.take<double>(L * n1) : nullptr;
    double* acc_all = want_geodesic ? arena.take<double>(L * n1) : nullptr;
    double* dsum_all = want_geodesic ? arena.take<double>(L * n1) : nullptr;
    uint32_t* ones_v = want_geodesic ? arena.take<uint32_t>(n1) : nullptr;
    double* ent_v = want_geodesic ? arena.take<double>(n1) : nullptr;
    double* lnsig_sum = want_geodesic ? arena.take<double>(n1) : nullptr;
    uint64_t* pair_sum = want_geodesic ? arena.take<uint64_t>(n1) : nullptr;
    double* oll0 = want_nonlazy ? arena.take<double>(size_t{g.edges} + 1) : nullptr;
    // Combinatorial curvature: per vertex twice its triangles and 1 - k/6 (NaN when the link is
    // not a cycle).
    const bool want_particle = (want & SG_PARTICLE) != 0;
    const bool want_comb = (want & (SG_EULER | SG_RIM | SG_CYCLE_CURVATURE)) != 0 || want_particle;
    uint32_t* tri2_v = want_comb ? arena.take<uint32_t>(n1) : nullptr;
    double* kappa_v = want_comb ? arena.take<double>(n1) : nullptr;
    // Particle measures: per vertex the Euler excess of B(v, 3).
    double* excess_v = want_particle ? arena.take<double>(n1) : nullptr;
    if (arena.failed) return false;
    int32_t* dist = dist_all + me * n1;
    int32_t* dist2 = dist2_all + me * n1;
    uint32_t* queue = queue_all + me * n1;
    uint32_t* queue2 = queue2_all + me * n1;
    uint32_t* mark = mark_all + me * n1;
    uint32_t* layer = layer_all + me * n1;
    uint32_t* counts = counts_all + me * n1;
    double* term = term_all + me * n1;
    for (size_t i = 0; i < n1; ++i) { dist[i] = -1; dist2[i] = -1; mark[i] = 0; counts[i] = 0; }
    par.fan(static_cast<uint32_t>(n1), [&](uint32_t i) {
        log_sum[i] = 0.0;
        ball_sum[i] = 0;
        if (want_geodesic) { lnsig_sum[i] = 0.0; pair_sum[i] = 0; }
    });

    // PASS 1: every vertex's balls, eccentricity, local entropy and mutual information, L
    // vertices per round. After each round the round's terms enter log_sum and ball_sum in
    // vertex order, the radii strided over the lanes.
    uint32_t connected = 1;
    for (uint32_t base = 0; base < n; base += L) {
        const uint32_t v = base + me;
        if (v < n) {
            uint32_t reached = 0;
            uint32_t* shell = want_growth ? shell_all + me * n1 : nullptr;
            uint32_t* inner = want_growth ? inner_all + me * n1 : nullptr;
            double* sigma = want_geodesic ? sigma_all + me * n1 : nullptr;
            double* acc = want_geodesic ? acc_all + me * n1 : nullptr;
            const uint32_t e = sg_bfs(g, v, dist, queue, layer, UINT32_MAX, reached, inner,
                                      sigma, acc);
            if (v == 0) connected = reached == n ? 1u : 0u;
            ecc[v] = e;
            if (want_geodesic) {
                // Over the targets q of v: the count with one shortest path, the sums of
                // ln sigma(v, q) by distance, and the sum of log2 t - 2 acc(v, q) / t with
                // t = (d + 1) sigma(v, q). Summed over every v, the last is the sum of H over
                // the ordered pairs (docs/SPEC.md, "StepStatistics geodesic multiplicity").
                double* dsum = dsum_all + me * n1;
                for (uint32_t r = 1; r <= e; ++r) dsum[r] = 0.0;
                uint32_t ones = 0;
                double ent = 0.0;
                for (uint32_t i = 1; i < reached; ++i) {
                    const uint32_t q = queue[i];
                    const uint32_t d = static_cast<uint32_t>(dist[q]);
                    const double s = sigma[q];
                    if (s == 1.0) ++ones;
                    dsum[d] += dm_log(s);
                    const double t = dm_mul(static_cast<double>(d + 1), s);
                    ent += dm_log2(t) - dm_mul(2.0, acc[q]) / t;
                }
                ones_v[v] = ones;
                ent_v[v] = ent;
            }
            if (want_growth)
                for (uint32_t r = 0; r <= e; ++r) shell[r] = layer[r];
            uint32_t ball = 1;
            for (uint32_t r = 1; r <= e; ++r) {
                const uint32_t next = ball + layer[r];
                term[r] = dm_log(static_cast<double>(next)) - dm_log(static_cast<double>(ball));
                layer[r] = next;
                ball = next;
            }
            if (want_growth) {
                for (uint32_t r = 1; r <= e; ++r) inner[r] += inner[r - 1];
                double fits[SG_GROWTH_VALUES];
                sg_growth_of(shell, layer, inner, e, fit_all + me * fit_stride, fits);
                for (uint32_t k = 0; k < SG_GROWTH_VALUES; ++k) growth_v[k * n1 + v] = fits[k];
            }
            if (want_particle) {
                // B(v, 3), the queue's prefix at distance at most 3, with V vertices and E edges:
                // the excess of E over the planar bound P = 3V - 6 (V(V-1)/2 for V < 3), over P.
                uint32_t b3 = 0;
                uint64_t ends = 0;
                for (; b3 < reached && dist[queue[b3]] <= 3; ++b3) {
                    const uint32_t u = queue[b3];
                    for (uint32_t k = g.off[u]; k < g.off[u + 1]; ++k)
                        if (dist[g.nbr[k]] >= 0 && dist[g.nbr[k]] <= 3) ++ends;
                }
                const int64_t V = b3, E = static_cast<int64_t>(ends / 2);
                const int64_t P = V < 3 ? V * (V - 1) / 2 : 3 * V - 6;
                excess_v[v] = P > 0 && E > P ? static_cast<double>(E - P) / static_cast<double>(P)
                                             : 0.0;
            }
            if (!want_local) {
                sg_bfs_clear(dist, queue, reached);
            } else {
                // B(v, 2) is the queue's prefix of vertices at distance at most 2.
                uint32_t b2 = 0;
                while (b2 < reached && dist[queue[b2]] <= 2) ++b2;
                localv[v] = sg_degree_entropy_of(g, queue, b2, counts);
                for (uint32_t i = 0; i < b2; ++i) mark[queue[i]] = v + 1;
                sg_bfs_clear(dist, queue, reached);
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
                miv[v] = g.degree(v) ? sum / g.degree(v) : 0.0;
            }
        }
        par.sync();
        const uint32_t cnt = n - base < L ? n - base : L;
        uint32_t top = 0;
        for (uint32_t i = 0; i < cnt; ++i) if (ecc[base + i] > top) top = ecc[base + i];
        for (uint32_t r = 1 + me; r <= top; r += L)
            for (uint32_t i = 0; i < cnt; ++i) {
                if (r > ecc[base + i]) continue;
                log_sum[r] += term_all[i * n1 + r];
                ball_sum[r] += layer_all[i * n1 + r];
                if (want_geodesic) {
                    lnsig_sum[r] += dsum_all[i * n1 + r];
                    pair_sum[r] += layer_all[i * n1 + r] - layer_all[i * n1 + r - 1];
                }
            }
        par.sync();
    }
    connected = par.bcast(connected);
    uint32_t radius = UINT32_MAX;
    for (uint32_t v = 0; v < n; ++v) if (ecc[v] < radius) radius = ecc[v];
    if (want_geodesic && par.leader()) {
        uint32_t top = 0;
        for (uint32_t v = 0; v < n; ++v) if (ecc[v] > top) top = ecc[v];
        uint64_t pairs = 0, ones = 0;
        double ent = 0.0;
        for (uint32_t v = 0; v < n; ++v) {
            ones += ones_v[v];
            ent += ent_v[v];
        }
        for (uint32_t d = 1; d <= top; ++d) pairs += pair_sum[d];
        if (pairs) {
            out.geodetic_fraction = static_cast<double>(ones) / static_cast<double>(pairs);
            out.segment_entropy = ent / static_cast<double>(pairs);
            out.geodesic_radii = top;
            out.defined |= SG_GEODESIC;
            for (uint32_t d = 1; d <= top && d <= log_geodesic_capacity; ++d)
                log_geodesic[d - 1] = lnsig_sum[d] / static_cast<double>(pair_sum[d]);
        }
    }
    if (par.leader()) {
        uint32_t* all = queue;
        for (uint32_t i = 0; i < n; ++i) all[i] = i;
        out.degree_entropy = sg_degree_entropy_of(g, all, n, counts);
        if (want_local) {
            double local_total = 0.0, mi_total = 0.0;
            uint32_t mi_vertices = 0;
            for (uint32_t v = 0; v < n; ++v) {
                local_total += localv[v];
                if (g.degree(v)) { mi_total += miv[v]; ++mi_vertices; }
            }
            out.local_entropy = local_total / n;
            out.defined |= SG_DEGREE_ENTROPY;
            if (mi_vertices) {
                out.mutual_information = mi_total / mi_vertices;
                out.defined |= SG_MUTUAL_INFORMATION;
            }
        }
    }
    const bool lazy_edges = want_lazy || want_vertex_ollivier;
    if ((lazy_edges || want_nonlazy) && g.edges) {
        if (me < TL) {
            auto* transport = reinterpret_cast<unsigned char*>(transport_all + me * (tstride / 8));
            uint32_t j = 0;
            for (uint32_t x = 0; x < n; ++x)
                for (uint32_t k = g.off[x]; k < g.off[x + 1]; ++k) {
                    if (g.nbr[k] <= x) continue;
                    if (j % TL == me) {
                        if (lazy_edges) oll[j] = sg_ollivier_edge(g, x, g.nbr[k], transport);
                        if (want_nonlazy)
                            oll0[j] = sg_ollivier_edge(g, x, g.nbr[k], transport, false);
                    }
                    ++j;
                }
        }
        par.sync();
        if (par.leader() && lazy_edges) {
            double sum = 0.0;
            for (uint32_t j = 0; j < g.edges; ++j) sum += oll[j];
            out.ollivier_ricci = sum / g.edges;
            out.defined |= SG_OLLIVIER;
        }
        if (par.leader() && want_nonlazy) {
            double sum = 0.0;
            for (uint32_t j = 0; j < g.edges; ++j) {
                sum += oll0[j];
                if (want_dists)
                    sg_dist_add(dists[SG_DIST_NONLAZY_OLLIVIER], SG_DIST_NONLAZY_OLLIVIER, oll0[j]);
            }
            out.nonlazy_ollivier = sum / g.edges;
            out.defined |= SG_NONLAZY_OLLIVIER;
        }
        if (want_vertex_ollivier && par.leader()) {
            // k_v: each edge's curvature added to both ends in edge order, over the degree.
            for (uint32_t v = 0; v < n; ++v) ollv[v] = 0.0;
            uint32_t j = 0;
            for (uint32_t x = 0; x < n; ++x)
                for (uint32_t k = g.off[x]; k < g.off[x + 1]; ++k) {
                    if (g.nbr[k] <= x) continue;
                    ollv[x] += oll[j];
                    ollv[g.nbr[k]] += oll[j];
                    ++j;
                }
            uint32_t n_with = 0;
            double k_sum = 0.0, d_sum = 0.0, k_lo = 0.0, k_hi = 0.0;
            uint32_t d_lo = 0, d_hi = 0;
            for (uint32_t v = 0; v < n; ++v) {
                const uint32_t dv = g.degree(v);
                if (!dv) continue;
                ollv[v] /= dv;
                if (!n_with || ollv[v] < k_lo) k_lo = ollv[v];
                if (!n_with || ollv[v] > k_hi) k_hi = ollv[v];
                if (!n_with || dv < d_lo) d_lo = dv;
                if (!n_with || dv > d_hi) d_hi = dv;
                k_sum += ollv[v];
                d_sum += dv;
                ++n_with;
            }
            if (k_lo != k_hi) {
                const double k_mean = k_sum / n_with, d_mean = d_sum / n_with;
                double kk = 0.0, kd = 0.0, dd = 0.0, cross = 0.0;
                for (uint32_t v = 0; v < n; ++v) {
                    if (!g.degree(v)) continue;
                    const double zk = ollv[v] - k_mean, zd = g.degree(v) - d_mean;
                    kk += dm_mul(zk, zk);
                    kd += dm_mul(zk, zd);
                    dd += dm_mul(zd, zd);
                }
                for (uint32_t x = 0; x < n; ++x)
                    for (uint32_t k = g.off[x]; k < g.off[x + 1]; ++k)
                        if (g.nbr[k] > x) cross += dm_mul(ollv[x] - k_mean, ollv[g.nbr[k]] - k_mean);
                out.ollivier_moran_i = dm_mul(static_cast<double>(n_with), cross) /
                                       dm_mul(static_cast<double>(g.edges), kk);
                out.defined |= SG_OLLIVIER_MORAN;
                if (d_lo != d_hi) {
                    out.ollivier_degree_correlation = kd / ::sqrt(dm_mul(kk, dd));
                    out.defined |= SG_OLLIVIER_DEGREE;
                }
            }
            if (want_dists)
                for (uint32_t v = 0; v < n; ++v)
                    if (g.degree(v)) sg_dist_add(dists[SG_DIST_OLLIVIER], SG_DIST_OLLIVIER, ollv[v]);
        }
    }
    if (want_comb) {
        for (uint32_t v = me; v < n; v += L) {
            const uint32_t k = sg_link_cycle(g, v, tri2_v[v]);
            kappa_v[v] = k ? 1.0 - static_cast<double>(k) / 6.0 : ::nan("");
        }
        par.sync();
        if (par.leader()) {
            uint64_t tri2 = 0;
            uint32_t rim = 0, cycles = 0;
            bool on_triangle = true;
            double total = 0.0;
            for (uint32_t v = 0; v < n; ++v) {
                tri2 += tri2_v[v];
                if (!tri2_v[v]) on_triangle = false;
                if (kappa_v[v] != kappa_v[v]) { ++rim; continue; }
                total += kappa_v[v];
                ++cycles;
                if (want_dists)
                    sg_dist_add(dists[SG_DIST_CYCLE_CURVATURE], SG_DIST_CYCLE_CURVATURE, kappa_v[v]);
            }
            out.euler_characteristic = static_cast<double>(static_cast<int64_t>(n) -
                                                           static_cast<int64_t>(g.edges) +
                                                           static_cast<int64_t>(tri2 / 6));
            out.defined |= SG_EULER;
            if (on_triangle) {
                out.rim_fraction = static_cast<double>(rim) / static_cast<double>(n);
                out.defined |= SG_RIM;
            }
            if (cycles) {
                out.cycle_curvature_total = total;
                out.defined |= SG_CYCLE_CURVATURE;
            }
            if (vertex_values)
                for (uint32_t v = 0; v < n; ++v)
                    vertex_values[SG_GROWTH_VALUES * n + v] = kappa_v[v];
            if (want_particle) {
                // C_v = t_v / (deg (deg - 1) / 2), 0 below degree 2; q_v = log2(max(1, deg))
                // (1 - C_v) + 10 x_v.
                double c_sum = 0.0, x_sum = 0.0, q_sum = 0.0, q_max = 0.0;
                for (uint32_t v = 0; v < n; ++v) {
                    const uint32_t d = g.degree(v);
                    const double c = d < 2 ? 0.0
                                           : static_cast<double>(tri2_v[v]) /
                                                 (static_cast<double>(d) * static_cast<double>(d - 1));
                    const double q = dm_mul(dm_log2(d > 1 ? static_cast<double>(d) : 1.0), 1.0 - c) +
                                     dm_mul(10.0, excess_v[v]);
                    c_sum += c;
                    x_sum += excess_v[v];
                    q_sum += q;
                    if (v == 0 || q > q_max) q_max = q;
                    if (want_dists) {
                        sg_dist_add(dists[SG_DIST_CLUSTERING], SG_DIST_CLUSTERING, c);
                        sg_dist_add(dists[SG_DIST_CHARGE], SG_DIST_CHARGE, q);
                    }
                    if (vertex_values) {
                        vertex_values[(SG_GROWTH_VALUES + 1) * n + v] = c;
                        vertex_values[(SG_GROWTH_VALUES + 2) * n + v] = excess_v[v];
                        vertex_values[(SG_GROWTH_VALUES + 3) * n + v] = q;
                    }
                }
                out.clustering = c_sum / n;
                out.euler_excess = x_sum / n;
                out.charge = q_sum / n;
                out.charge_max = q_max;
                out.defined |= SG_PARTICLE;
            }
        }
    }
    if (vertex_curvature && par.leader())
        for (uint32_t v = 0; v < n; ++v)
            vertex_curvature[v] = g.edges && g.degree(v) ? ollv[v] : ::nan("");
    if (connected && par.leader()) {
        uint64_t ecc_total = 0;
        for (uint32_t v = 0; v < n; ++v) ecc_total += ecc[v];
        out.radius = static_cast<int32_t>(radius);
        out.mean_eccentricity = static_cast<double>(ecc_total) / n;
        out.defined |= SG_RADIUS;
    }
    // G': every vertex when G is connected, else the caller's list. R' is its least
    // eccentricity; a search from a vertex stays in its component, so ecc[] is that eccentricity.
    const uint32_t* sub = connected ? nullptr : largest;
    const uint32_t ns = connected ? n : (largest ? largest_n : 0u);
    auto vat = [&](uint32_t i) { return sub ? sub[i] : i; };
    if (want_growth && par.leader()) {
        // The six fits over the vertices of G' with one, in vertex order.
        uint32_t fitted = 0;
        for (uint32_t k = 0; k < SG_GROWTH_VALUES; ++k) {
            double sum = 0.0;
            fitted = 0;
            for (uint32_t i = 0; i < ns; ++i) {
                const double x = growth_v[k * n1 + vat(i)];
                if (x != x) continue;
                sum += x;
                ++fitted;
                if (want_dists)
                    sg_dist_add(dists[SG_DIST_BALL_DIMENSION + k], SG_DIST_BALL_DIMENSION + k, x);
            }
            if (fitted) out.growth[k] = sum / fitted;
        }
        if (fitted) out.defined |= SG_VOLUME_GROWTH;
        if (vertex_values)
            for (uint32_t k = 0; k < SG_GROWTH_VALUES; ++k)
                for (uint32_t i = 0; i < ns; ++i)
                    vertex_values[k * n + vat(i)] = growth_v[k * n1 + vat(i)];
    }
    uint32_t Rs = UINT32_MAX;
    for (uint32_t i = 0; i < ns; ++i) if (ecc[vat(i)] < Rs) Rs = ecc[vat(i)];
    const bool whole = connected && radius > 0 && (want & (SG_HAUSDORFF | SG_RICCI | SG_FISHER));
    const bool part = want_sub && ns >= 2 && Rs > 0;
    if (!whole && !part) return true;
    const uint32_t R = Rs;   // the radius of G when G is connected

    // PASS 2: each vertex's own dimension over r = 1..R, for the vertices of G'.
    for (uint32_t i = me; i < ns; i += L) {
        const uint32_t v = vat(i);
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
    }
    par.sync();
    const double pi = 3.14159265358979323846;
    if (par.leader() && whole) {
        double dim_total = 0.0;
        for (uint32_t v = 0; v < n; ++v) dim_total += dimv[v];
        const double d = dim_total / n;
        out.hausdorff_dimension = d;
        out.defined |= SG_HAUSDORFF;
        out.ball_radii = R;
        for (uint32_t r = 1; r <= R && r <= ball_capacity; ++r)
            ball_dimension[r - 1] =
                log_sum[r] / n / (dm_log(r + 1.0) - dm_log(static_cast<double>(r)));

        // Ricci scalar at dimension d: linear in |B(v, r)|, so the per-radius ball sums suffice.
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
    }
    if (par.leader() && part) {
        double total = 0.0, top = 0.0;
        for (uint32_t i = 0; i < ns; ++i) {
            const double x = dimv[vat(i)];
            total += x;
            if (i == 0 || x > top) top = x;
        }
        const double mean = total / ns;
        double sq = 0.0;
        for (uint32_t i = 0; i < ns; ++i) sq += dm_mul(dimv[vat(i)] - mean, dimv[vat(i)] - mean);
        out.largest_dimension = mean;
        out.local_dimension_max = top;
        out.local_dimension_sd = ::sqrt(sq / ns);
        out.defined |= SG_LARGEST_DIMENSION;
        if (want_gradient) {
            // L_v = the mean over the neighbours u of v of d_u - d_v; every vertex of G' has a
            // neighbour, and its neighbours lie in G'.
            double lap_total = 0.0;
            for (uint32_t i = 0; i < ns; ++i) {
                const uint32_t v = vat(i);
                double s = 0.0;
                for (uint32_t k = g.off[v]; k < g.off[v + 1]; ++k) s += dimv[g.nbr[k]] - dimv[v];
                const double lap = s / g.degree(v);
                lap_total += lap;
                if (want_dists)
                    sg_dist_add(dists[SG_DIST_DIMENSION_GRADIENT], SG_DIST_DIMENSION_GRADIENT, lap);
                if (vertex_values) vertex_values[(SG_GROWTH_VALUES + 4) * n + v] = lap;
            }
            out.dimension_gradient = lap_total / ns;
            out.defined |= SG_DIMENSION_GRADIENT;
        }
        if (want_dists) {
            for (uint32_t i = 0; i < ns; ++i)
                sg_dist_add(dists[SG_DIST_LOCAL_DIMENSION], SG_DIST_LOCAL_DIMENSION, dimv[vat(i)]);
            // The per-radius coefficients of K_v at D = the LargestComponentDimension.
            const double c = dm_tgamma(mean / 2.0 + 1.0) / dm_pow(pi, mean / 2.0);
            for (uint32_t r = 1; r <= R; ++r) {
                const double rr = static_cast<double>(r);
                ric_a[r] = dm_mul(6.0, mean + 2.0) / dm_mul(rr, rr);
                ric_p[r] = dm_pow(rr, mean);
            }
            ric_a[0] = c;
        }
    }

    // PASS 3: K_v for the vertices of G', from each vertex's balls to radius R'.
    if (part && want_dists) {
        par.sync();
        const double c = ric_a[0];
        for (uint32_t i = me; i < ns; i += L) {
            const uint32_t v = vat(i);
            uint32_t reached = 0;
            sg_bfs(g, v, dist, queue, layer, R, reached);
            sg_bfs_clear(dist, queue, reached);
            uint64_t ball = 1;
            double s = 0.0;
            for (uint32_t r = 1; r <= R; ++r) {
                ball += layer[r];
                s += dm_mul(ric_a[r], 1.0 - dm_mul(static_cast<double>(ball), c) / ric_p[r]);
            }
            ricv[v] = s / R;
        }
        par.sync();
        if (par.leader())
            for (uint32_t i = 0; i < ns; ++i)
                sg_dist_add(dists[SG_DIST_RICCI], SG_DIST_RICCI, ricv[vat(i)]);
    }

    if (!whole || !(want & SG_FISHER)) return true;

    // PASS 4: Fisher information from the dimensions within distance 2.
    for (uint32_t v = me; v < n; v += L) {
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
        fishv[v] = (1.0 + grad) / (var + 0.01);
    }
    par.sync();
    if (par.leader()) {
        double fisher_total = 0.0;
        for (uint32_t v = 0; v < n; ++v) fisher_total += fishv[v];
        out.fisher_information = fisher_total / n;
        out.defined |= SG_FISHER;
    }
    return true;
}

// Builds G from `el` in `scratch` and computes its geometry. When `capacity` is short, returns
// false with `needed` set to the bytes that would have sufficed for the stage reached; a caller
// that grows its buffer to `needed` and calls again either succeeds or learns the next stage's
// need. Every lane of the policy calls it with the same arguments and gets the same verdict and
// `needed`; the leader builds G and `out` is the leader's. `largest`, `largest_n`, `dists`,
// `vertex_curvature`, `vertex_values`, `log_geodesic` and `log_geodesic_capacity` are
// sg_geometry_of's; `graph` is the rule that builds G (SG_GRAPH_CONSECUTIVE or
// SG_GRAPH_ALL_PAIRS).
template <class EdgeList, class Par = IrSerial>
HG_HD bool sg_state_geometry(const EdgeList& el, unsigned char* scratch, size_t capacity,
                             SgGeometry& out, double* ball_dimension, uint32_t ball_capacity,
                             size_t& needed, uint32_t want = ~0u, Par par = Par{},
                             const uint32_t* largest = nullptr, uint32_t largest_n = 0,
                             SgDistribution* dists = nullptr,
                             double* vertex_curvature = nullptr,
                             double* vertex_values = nullptr, double* log_geodesic = nullptr,
                             uint32_t log_geodesic_capacity = 0,
                             uint32_t graph = SG_GRAPH_CONSECUTIVE) {
    SgArena arena;
    arena.base = scratch;
    arena.capacity = capacity;
    SgGraph g;
    size_t slots = 0, pairs = 0;
    sg_sizes(el, slots, pairs, graph);
    uint32_t built_ok = 1;
    uint64_t off_at = 0, nbr_at = 0;
    if (par.leader()) {
        built_ok = sg_build(el, arena, g, graph) ? 1u : 0u;
        if (built_ok) {
            off_at = static_cast<uint64_t>(reinterpret_cast<const unsigned char*>(g.off) - scratch);
            nbr_at = static_cast<uint64_t>(reinterpret_cast<const unsigned char*>(g.nbr) - scratch);
        }
    }
    par.sync();
    if (!par.bcast(built_ok)) {
        needed = sg_build_bytes(slots, pairs) + sg_metric_bytes(slots, 0, 0, par.width(), want);
        return false;
    }
    g.n = par.bcast(g.n);
    g.edges = par.bcast(g.edges);
    g.max_degree = par.bcast(g.max_degree);
    g.off = reinterpret_cast<const uint32_t*>(scratch + par.bcast64(off_at));
    g.nbr = reinterpret_cast<const uint32_t*>(scratch + par.bcast64(nbr_at));
    arena.used = static_cast<size_t>(par.bcast64(arena.used));
    const size_t built = arena.used;
    if (!sg_geometry_of(g, arena, out, ball_dimension, ball_capacity, want, par, largest,
                        largest_n, dists, vertex_curvature, vertex_values, log_geodesic,
                        log_geodesic_capacity)) {
        needed = built + sg_metric_bytes(g.n, g.max_degree, g.edges, par.width(), want);
        return false;
    }
    needed = arena.used;
    return true;
}

}  // namespace common
}  // namespace HG_NAMESPACE
