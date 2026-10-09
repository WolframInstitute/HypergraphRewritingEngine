#pragma once
#include "hgcommon/namespace.hpp"
// PER-STATE HYPERGRAPH INVARIANTS, one body for host and device. The CPU engine computes them on
// the worker that first claims a class's hash (Hypergraph::record_state_invariants), and
// "StepStatistics" (paclet_source/state_statistics.hpp) summarises them per step. The definitions
// are those of the multiway-statistics probes; reference/verify_state_statistics.wls checks them.
//
// A state is m edges in CSR form: edge i's vertices are verts[off[i] .. off[i + 1]), S = off[m]
// slots in all. Vertex ids are arbitrary. The incidence graph has a node per distinct vertex and
// a node per edge, and a link per distinct (edge, vertex) incidence.
//
// THE LARGEST COMPONENT, whose diameter, mean distance and vertex fraction are reported, is the
// one with the most nodes, then the greatest diameter, then the greatest mean distance, then the
// most vertices. Every value is an isomorphism invariant.
//
// NO ALLOCATION. The caller passes si_scratch_words(S, m, lanes) words of scratch and the two
// output arrays (m arities, S degrees at most). The distance pass is O(k (k + I)) for a
// component of k nodes and I incidences and runs lane-strided over the sources, each lane with
// its own BFS arrays; the rest runs on the leader. The policy is IrSerial on the host and IrTile
// on the device (rank, width, leader, sync).
#include "hgcommon/core.hpp"
#include "hgcommon/ir_core.hpp"

namespace HG_NAMESPACE {
namespace common {

struct StateInvariantValues {
    int64_t vertex_count = 0;
    int64_t edge_count = 0;
    int64_t max_degree = 0;
    int64_t two_section_edge_count = 0;   // distinct unordered vertex pairs sharing an edge
    int64_t components = 0;               // of the incidence graph
    int64_t cycle_rank = 0;               // two_section_edge_count - vertex_count + components
    int64_t incidence_cycle_rank = 0;     // incidences - (vertex_count + edge_count) + components
    int64_t incidence_diameter = 0;       // of the largest component
    double mean_degree = 0.0;             // slots / vertex_count
    double incidence_mean_distance = 0.0; // mean over the largest component's node pairs
    double largest_component_fraction = 0.0;  // that component's vertices / vertex_count
};

// One state's invariants as stored per class: the values, then num_edges arities ascending and
// num_vertices slot degrees descending, as uint32 words directly after the record.
struct StateInvariantRecord {
    StateInvariantValues v;
    uint32_t num_edges = 0;
    uint32_t num_vertices = 0;
    HG_HD const uint32_t* arities() const { return reinterpret_cast<const uint32_t*>(this + 1); }
    HG_HD const uint32_t* degrees() const { return arities() + num_edges; }
    HG_HD static uint64_t bytes(uint32_t m, uint32_t n) {
        return sizeof(StateInvariantRecord) + 4 * (uint64_t{m} + n);
    }
};

// Scratch words for a state of `slots` slots and `m` edges, computed by `lanes` lanes.
HG_HD inline uint64_t si_scratch_words(uint32_t slots, uint32_t m, uint32_t lanes) {
    const uint64_t s = slots, nodes = uint64_t{slots} + m;
    return 8 * s + (m + 1) + (s + 1) + 3 * nodes + 1 + 2 * uint64_t{lanes} * nodes +
           4 * uint64_t{lanes} + 8;
}

namespace si_detail {
HG_HD inline int si_cmp(uint32_t a, uint32_t b) { return (a > b) - (a < b); }
// a[0..n) sorted ascending, values compared directly.
HG_HD inline void si_sort(uint32_t* a, uint32_t n) {
    ir_heapsort_idx(a, n, [](uint32_t x, uint32_t y) { return si_cmp(x, y); });
}
}  // namespace si_detail

// The invariants of one state. `arities_out` receives the m arities ascending, `degrees_out` the
// vertex_count slot degrees descending. Every lane of the policy calls it with the same
// arguments; `out` and the two arrays are written by the leader.
template <class Par = IrSerial>
HG_HD inline void state_invariants(const uint32_t* off, const uint32_t* verts, uint32_t m,
                                   uint32_t* scratch, StateInvariantValues& out,
                                   uint32_t* arities_out, uint32_t* degrees_out,
                                   Par par = Par{}) {
    const uint32_t S = off[m];
    const uint32_t L = par.width();
    const uint32_t N = S + m;   // bound on the incidence graph's nodes
    uint32_t* vs     = scratch;            // S: distinct vertices ascending
    uint32_t* sidx   = vs + S;             // S: each slot's vertex index
    uint32_t* deg    = sidx + S;           // S: slot degree per vertex
    uint32_t* doff   = deg + S;            // m + 1: each edge's distinct vertices in dlist
    uint32_t* dlist  = doff + (m + 1);     // S
    uint32_t* ioff   = dlist + S;          // S + 1: each vertex's edges in ilist
    uint32_t* ilist  = ioff + (S + 1);     // S
    uint32_t* stamp  = ilist + S;          // S
    uint32_t* comp   = stamp + S;          // N: component of each node
    uint32_t* order  = comp + N;           // N: nodes in discovery order, component by component
    uint32_t* cstart = order + N;          // N + 1: each component's run in `order`
    uint32_t* hdr    = cstart + (N + 1);   // 8: n, components, most, incidences
    uint32_t* part   = hdr + 8;            // 4 L: per-lane total (2 words), diameter, spare
    uint32_t* ldist  = part + 4 * L;       // L N
    uint32_t* lq     = ldist + uint64_t{L} * N;   // L N
    constexpr uint32_t kUnset = 0xFFFFFFFFu;

    if (par.leader()) {
        out = StateInvariantValues{};
        // Distinct vertices ascending, and each slot's index among them.
        for (uint32_t j = 0; j < S; ++j) vs[j] = verts[j];
        si_detail::si_sort(vs, S);
        uint32_t n = 0;
        for (uint32_t j = 0; j < S; ++j)
            if (n == 0 || vs[n - 1] != vs[j]) vs[n++] = vs[j];
        for (uint32_t j = 0; j < S; ++j) {
            uint32_t lo = 0, hi = n;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (vs[mid] < verts[j]) lo = mid + 1; else hi = mid;
            }
            sidx[j] = lo;
        }
        out.vertex_count = n;
        out.edge_count = m;

        // Arities ascending; slot degrees descending.
        for (uint32_t i = 0; i < m; ++i) arities_out[i] = off[i + 1] - off[i];
        si_detail::si_sort(arities_out, m);
        for (uint32_t v = 0; v < n; ++v) deg[v] = 0;
        for (uint32_t j = 0; j < S; ++j) ++deg[sidx[j]];
        for (uint32_t v = 0; v < n; ++v) degrees_out[v] = deg[v];
        si_detail::si_sort(degrees_out, n);
        for (uint32_t a = 0, b = n; a + 1 < b; ++a, --b) {
            const uint32_t t = degrees_out[a]; degrees_out[a] = degrees_out[b - 1]; degrees_out[b - 1] = t;
        }
        out.max_degree = n ? degrees_out[0] : 0;
        out.mean_degree = n ? static_cast<double>(S) / static_cast<double>(n) : 0.0;

        // Each edge's distinct vertex indices, ascending.
        uint32_t at = 0;
        for (uint32_t i = 0; i < m; ++i) {
            doff[i] = at;
            const uint32_t b = at;
            for (uint32_t j = off[i]; j < off[i + 1]; ++j) {
                const uint32_t v = sidx[j];
                uint32_t k = at;
                while (k > b && dlist[k - 1] > v) { dlist[k] = dlist[k - 1]; --k; }
                dlist[k] = v;
                ++at;
            }
            uint32_t w = b;
            for (uint32_t k = b; k < at; ++k)
                if (w == b || dlist[w - 1] != dlist[k]) dlist[w++] = dlist[k];
            at = w;
        }
        doff[m] = at;
        const uint32_t incidences = at;

        // Each vertex's edges (ioff/ilist), counted then filled through `stamp` as the cursor.
        for (uint32_t v = 0; v <= n; ++v) ioff[v] = 0;
        for (uint32_t k = 0; k < incidences; ++k) ++ioff[dlist[k] + 1];
        for (uint32_t v = 0; v < n; ++v) ioff[v + 1] += ioff[v];
        for (uint32_t v = 0; v < n; ++v) stamp[v] = ioff[v];
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t k = doff[i]; k < doff[i + 1]; ++k) ilist[stamp[dlist[k]]++] = i;

        // Two-section edges: for each vertex u, the distinct w > u sharing an edge with it.
        int64_t two = 0;
        for (uint32_t v = 0; v < n; ++v) stamp[v] = kUnset;
        for (uint32_t u = 0; u < n; ++u)
            for (uint32_t a = ioff[u]; a < ioff[u + 1]; ++a) {
                const uint32_t e = ilist[a];
                for (uint32_t k = doff[e]; k < doff[e + 1]; ++k) {
                    const uint32_t w = dlist[k];
                    if (w > u && stamp[w] != u) { stamp[w] = u; ++two; }
                }
            }
        out.two_section_edge_count = two;

        // Components of the incidence graph (nodes 0..n-1 vertices, n..n+m-1 edges), by BFS from
        // each unvisited node in ascending order; `order` lists each component contiguously.
        const uint32_t nodes = n + m;
        for (uint32_t x = 0; x < nodes; ++x) comp[x] = kUnset;
        uint32_t c = 0, tail = 0, most = 0;
        for (uint32_t s = 0; s < nodes; ++s) {
            if (comp[s] != kUnset) continue;
            cstart[c] = tail;
            comp[s] = c;
            order[tail++] = s;
            for (uint32_t h = cstart[c]; h < tail; ++h) {
                const uint32_t x = order[h];
                if (x < n) {
                    for (uint32_t a = ioff[x]; a < ioff[x + 1]; ++a) {
                        const uint32_t y = n + ilist[a];
                        if (comp[y] == kUnset) { comp[y] = c; order[tail++] = y; }
                    }
                } else {
                    const uint32_t e = x - n;
                    for (uint32_t k = doff[e]; k < doff[e + 1]; ++k) {
                        const uint32_t y = dlist[k];
                        if (comp[y] == kUnset) { comp[y] = c; order[tail++] = y; }
                    }
                }
            }
            if (tail - cstart[c] > most) most = tail - cstart[c];
            ++c;
        }
        cstart[c] = tail;
        out.components = c;
        out.cycle_rank = out.two_section_edge_count - out.vertex_count + out.components;
        out.incidence_cycle_rank = static_cast<int64_t>(incidences) -
                                   static_cast<int64_t>(nodes) + out.components;
        hdr[0] = n; hdr[1] = c; hdr[2] = most; hdr[3] = 0;
    }
    par.sync();
    const uint32_t n = hdr[0], c = hdr[1], most = hdr[2];

    // The largest component: every component of `most` nodes, in discovery order, its
    // distances by one BFS per source, the sources strided over the lanes.
    uint32_t* dist = ldist + uint64_t{par.rank()} * N;
    uint32_t* q = lq + uint64_t{par.rank()} * N;
    bool chosen = false;
    for (uint32_t ci = 0; ci < c; ++ci) {
        const uint32_t b = cstart[ci], k = cstart[ci + 1] - b;
        if (k != most) continue;
        uint64_t total = 0;
        uint32_t diameter = 0;
        if (k > 1) {
            for (uint32_t si = par.rank(); si < k; si += L) {
                for (uint32_t h = 0; h < k; ++h) dist[order[b + h]] = kUnset;
                const uint32_t s = order[b + si];
                uint32_t tail = 0;
                q[tail++] = s;
                dist[s] = 0;
                for (uint32_t h = 0; h < tail; ++h) {
                    const uint32_t x = q[h], dx = dist[x] + 1;
                    if (x < n) {
                        for (uint32_t a = ioff[x]; a < ioff[x + 1]; ++a) {
                            const uint32_t y = n + ilist[a];
                            if (dist[y] == kUnset) { dist[y] = dx; q[tail++] = y; }
                        }
                    } else {
                        const uint32_t e = x - n;
                        for (uint32_t a = doff[e]; a < doff[e + 1]; ++a) {
                            const uint32_t y = dlist[a];
                            if (dist[y] == kUnset) { dist[y] = dx; q[tail++] = y; }
                        }
                    }
                }
                for (uint32_t h = 0; h < k; ++h) {
                    const uint32_t d = dist[order[b + h]];
                    total += d;
                    if (d > diameter) diameter = d;
                }
            }
        }
        uint32_t* mine = part + 4 * par.rank();
        mine[0] = static_cast<uint32_t>(total);
        mine[1] = static_cast<uint32_t>(total >> 32);
        mine[2] = diameter;
        par.sync();
        if (par.leader()) {
            uint64_t t = 0;
            uint32_t d = 0;
            for (uint32_t r = 0; r < L; ++r) {
                const uint32_t* p = part + 4 * r;
                t += uint64_t{p[0]} | (uint64_t{p[1]} << 32);
                if (p[2] > d) d = p[2];
            }
            const double mean = k > 1 ? static_cast<double>(t) /
                                        (static_cast<double>(k) * static_cast<double>(k - 1))
                                      : 0.0;
            uint32_t vertices_in = 0;
            for (uint32_t h = 0; h < k; ++h) if (order[b + h] < n) ++vertices_in;
            const double fraction =
                n ? static_cast<double>(vertices_in) / static_cast<double>(n) : 0.0;
            const int64_t dd = d;
            if (!chosen || dd > out.incidence_diameter ||
                (dd == out.incidence_diameter && (mean > out.incidence_mean_distance ||
                 (mean == out.incidence_mean_distance &&
                  fraction > out.largest_component_fraction)))) {
                out.incidence_diameter = dd;
                out.incidence_mean_distance = mean;
                out.largest_component_fraction = fraction;
            }
        }
        chosen = true;
        par.sync();
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
