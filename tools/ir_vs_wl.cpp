// Standalone IR (exact) vs WL (approximate) canonicalization comparison.
//
// Two questions:
//   1. SPEED  - per-call time on identical states, low vs high automorphism.
//   2. POWER  - does WL ever collide where IR distinguishes? (1-WL-hard pairs)
//
// IR = IRCanonicalizer::compute_canonical_hash (exact canonical form).
// WL = Hypergraph::compute_canonical_hash with shared-tree on (approximate).
//

#include <hypergraph/hypergraph.hpp>
#include <hypergraph/ir_canonicalization.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

// ---- hashing ----------------------------------------------------------------

static void build_hg(Hypergraph& hg, const Edges& edges, SparseBitset& bs) {
    VertexId maxv = 0;
    for (const auto& e : edges)
        for (VertexId v : e) maxv = std::max(maxv, v);
    for (VertexId i = 0; i <= maxv; ++i) hg.alloc_vertex();
    for (const auto& e : edges) {
        EdgeId id = hg.create_edge(e.data(), static_cast<uint8_t>(e.size()));
        bs.set(id, hg.arena());
    }
}

static uint64_t wl_hash(const Edges& edges) {
    SparseBitset bs;
    Hypergraph hg;
    build_hg(hg, edges, bs);
    hg.enable_wl_hash();
    return hg.compute_canonical_hash(bs);
}

static uint64_t ir_hash(const Edges& edges) {
    IRCanonicalizer ir;
    return ir.compute_canonical_hash(edges);
}

// ---- timing -----------------------------------------------------------------

template <class F>
static double median_us(F f, int iters) {
    std::vector<double> t;
    t.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto a = std::chrono::high_resolution_clock::now();
        volatile uint64_t h = f();
        (void)h;
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// time only the hash call; structures built once outside the loop
static double wl_time(const Edges& edges, int iters) {
    SparseBitset bs;
    Hypergraph hg;
    build_hg(hg, edges, bs);
    hg.enable_wl_hash();
    return median_us([&]() { return hg.compute_canonical_hash(bs); }, iters);
}
static double ir_time(const Edges& edges, int iters) {
    IRCanonicalizer ir;
    return median_us([&]() { return ir.compute_canonical_hash(edges); }, iters);
}

// ---- generators -------------------------------------------------------------

static Edges cycle(int n) {  // n vertices, n edges, 2-regular: high automorphism
    Edges e;
    for (int i = 0; i < n; ++i) e.push_back({(VertexId)i, (VertexId)((i + 1) % n)});
    return e;
}

static Edges random_graph(int nedges, uint32_t seed) {  // low automorphism
    std::mt19937 rng(seed);
    int nverts = std::max(4, nedges);
    std::uniform_int_distribution<int> d(0, nverts - 1);
    Edges e;
    for (int i = 0; i < nedges; ++i) {
        VertexId a = d(rng), b = d(rng);
        e.push_back({a, b});
    }
    return e;
}

static Edges disjoint_cycles(int k, int len) {  // k copies of C_len
    Edges e;
    for (int c = 0; c < k; ++c) {
        int base = c * len;
        for (int i = 0; i < len; ++i)
            e.push_back({(VertexId)(base + i), (VertexId)(base + (i + 1) % len)});
    }
    return e;
}

// ---- main -------------------------------------------------------------------

// Both orientations of every undirected edge.
//
// This engine's states are ORDERED hypergraphs and its WL folds the position within an edge
// into the refinement key, so a lone {u,v} is effectively a DIRECTED edge. Writing every edge
// as (min,max) then leaks the vertex labelling into the structure: two graphs can differ as
// ordered hypergraphs purely in orientation, and both WL and IR would separate them for a
// reason that has nothing to do with the graphs. Emitting both (u,v) and (v,u) makes
// orientation carry no information, so the ordered-hypergraph isomorphism class coincides
// with the undirected one and a 1-WL-hardness claim is actually being tested.
static Edges symmetrize(const Edges& e) {
    Edges out;
    out.reserve(e.size() * 2);
    for (const auto& x : e) { out.push_back({x[0], x[1]}); out.push_back({x[1], x[0]}); }
    return out;
}

// Prism: two triangles joined by a perfect matching. 3-regular, 6 vertices, 9 edges.
static Edges prism() {
    return {{0,1},{1,2},{2,0}, {3,4},{4,5},{5,3}, {0,3},{1,4},{2,5}};
}

// K3,3: complete bipartite. Also 3-regular, 6 vertices, 9 edges, and NOT isomorphic to the
// prism -- it is triangle-free.
static Edges k33() {
    Edges e;
    for (VertexId a = 0; a < 3; ++a)
        for (VertexId b = 3; b < 6; ++b) e.push_back({a, b});
    return e;
}

// Rook's graph on a 4x4 board: vertices are cells, adjacent when they share a row or column.
// Strongly regular with parameters (16,6,2,2).
static Edges rooks4x4() {
    auto id = [](uint32_t r, uint32_t c) { return static_cast<VertexId>(r * 4 + c); };
    Edges e;
    for (uint32_t r = 0; r < 4; ++r)
        for (uint32_t c = 0; c < 4; ++c)
            for (uint32_t r2 = 0; r2 < 4; ++r2)
                for (uint32_t c2 = 0; c2 < 4; ++c2) {
                    if (id(r, c) >= id(r2, c2)) continue;
                    if (r == r2 || c == c2) e.push_back({id(r, c), id(r2, c2)});
                }
    return e;
}

// The Shrikhande graph: the Cayley graph on Z4 x Z4 with connection set
// {+-(1,0), +-(0,1), +-(1,1)}. Same parameters (16,6,2,2) as the rook's graph above and NOT
// isomorphic to it -- the two are the classic witness that those parameters do not determine
// the graph, and therefore that no amount of colour refinement can separate them.
static Edges shrikhande() {
    auto id = [](uint32_t a, uint32_t b) { return static_cast<VertexId>((a % 4) * 4 + (b % 4)); };
    const int dx[3] = {1, 0, 1};
    const int dy[3] = {0, 1, 1};
    std::set<std::pair<VertexId, VertexId>> seen;
    Edges e;
    for (uint32_t a = 0; a < 4; ++a)
        for (uint32_t b = 0; b < 4; ++b)
            for (int k = 0; k < 3; ++k) {
                VertexId u = id(a, b);
                VertexId v = id(a + dx[k], b + dy[k]);
                auto key = std::minmax(u, v);
                if (u != v && seen.insert({key.first, key.second}).second)
                    e.push_back({key.first, key.second});
            }
    return e;
}

int main() {
    printf("=== SPEED: IR vs WL on identical states (median us/call) ===\n");
    printf("%-12s %6s %12s %12s %10s\n", "graph", "edges", "WL_us", "IR_us", "IR/WL");
    struct Cfg { const char* name; Edges e; int iters; };
    std::vector<Cfg> cfgs;
    for (int n : {10, 20, 50, 100, 200}) cfgs.push_back({"random", random_graph(n, 1234u + n), 2000});
    for (int n : {10, 20, 50, 100, 200}) cfgs.push_back({"cycle", cycle(n), 2000});
    for (auto& c : cfgs) {
        int iters = c.e.size() > 100 ? 400 : c.iters;
        double w = wl_time(c.e, iters);
        double r = ir_time(c.e, iters);
        printf("%-12s %6zu %12.3f %12.3f %10.2f\n", c.name, c.e.size(), w, r, r / w);
    }

    // CONSTRUCTIVE worst cases, not sampled ones. No assumption is available about the
    // distribution of states an evolution reaches -- and rewrite rules manufacture regular,
    // highly symmetric structure, so a random corpus would UNDERSTATE collisions exactly
    // where they matter. These families are built, so each one is an existence proof: if WL
    // cannot separate them, a rule set that reaches them is deduplicated wrongly, and no
    // measured rate over some other corpus licenses assuming otherwise.
    printf("\n=== POWER: 1-WL-hard pairs (non-isomorphic, same degree sequence) ===\n");
    printf("%-22s %-10s %-10s %-22s\n", "pair", "WL_equal?", "IR_equal?", "verdict");
    struct Pair { const char* name; Edges a; Edges b; };
    std::vector<Pair> pairs = {
        {"C6 vs 2xC3", cycle(6), disjoint_cycles(2, 3)},
        // Both 3-regular on 6 vertices with 9 edges. Every vertex has degree 3 and every
        // neighbour has degree 3, so colour refinement is stable at one colour immediately
        // and cannot see that one is bipartite and the other has triangles.
        {"prism vs K3,3", symmetrize(prism()), symmetrize(k33())},
        // Both strongly regular with parameters (16,6,2,2): same degree, and every adjacent
        // pair has exactly 2 common neighbours, every non-adjacent pair exactly 2. The
        // canonical 1-WL-indistinguishable pair -- refinement has nothing to split on at any
        // round, for any number of rounds.
        {"rook4x4 vs Shrikhande", symmetrize(rooks4x4()), symmetrize(shrikhande())},
        {"C8 vs 2xC4", cycle(8), disjoint_cycles(2, 4)},
        {"C9 vs 3xC3", cycle(9), disjoint_cycles(3, 3)},
        {"C10 vs 2xC5", cycle(10), disjoint_cycles(2, 5)},
        {"C12 vs 2xC6", cycle(12), disjoint_cycles(2, 6)},
    };
    int wl_collisions = 0;
    for (auto& p : pairs) {
        bool wl_eq = wl_hash(p.a) == wl_hash(p.b);
        bool ir_eq = ir_hash(p.a) == ir_hash(p.b);
        const char* verdict = (wl_eq && !ir_eq) ? "WL COLLISION (IR ok)"
                              : (!wl_eq && !ir_eq) ? "both distinguish"
                              : (wl_eq && ir_eq) ? "BOTH WRONG"
                                                 : "WL>IR?? (impossible)";
        if (wl_eq && !ir_eq) ++wl_collisions;
        printf("%-22s %-10s %-10s %-22s\n", p.name, wl_eq ? "yes" : "no",
               ir_eq ? "yes" : "no", verdict);
    }
    printf("\nWL false-collisions among %zu non-isomorphic pairs: %d\n", pairs.size(), wl_collisions);
    printf("(IR is exact: ir_equal=yes for a non-isomorphic pair would be an IR bug)\n");
    return 0;
}
