// tools/branchial_flux_probe.cpp
//
// Standalone, header-free (NO project headers) test of the OR-via-complexity
// hunch: does the DIVERGENCE between two branchial-neighbour branches track a
// CAUSAL-FLUX "mass" proxy, and is there a re-merge ("collapse") cutoff that
// scales with that mass?
//
// Build:
//   g++ -O2 -std=c++17 tools/branchial_flux_probe.cpp -o /tmp/branchial_flux_probe
//   /tmp/branchial_flux_probe
//
// Setup
// -----
// A "branchial layer" is the set of all single-rewrite children of one parent
// state under a rule. Two children are BRANCHLIKE iff their rewrites consumed
// overlapping edges — they are mutually exclusive, i.e. genuinely "in
// superposition" (only one can actually fire). For each branchlike sibling pair
// we measure:
//
//   divergence (relative complexity of the diff):
//     * edge_diff = |E(Bi) symdiff E(Bj)|   (size of the divergent region)
//     * wl_delta  = fraction of vertices whose WL colour differs between Bi,Bj
//
//   mass proxy (causal-edge flux):
//     * causal_out(r) = number of next-step rewrites that consume an edge r
//       PRODUCED — i.e. the immediate causal out-degree of the event. This is
//       exactly the engine's causal-edge relation (e2 follows e1 when e2
//       consumes an edge e1 produced). flux(pair) = causal_out(i)+causal_out(j).
//
// Hypotheses
//   (a) does divergence correlate with flux?  (Pearson, plus a bucketed table)
//   (b) is there a re-merge cutoff: do high-flux branch pairs reconverge (share
//       a canonical state in their forward multiway cones within K steps) MORE
//       than low-flux ones?  Re-convergence = branchial merge = "collapse".
//
// Rule shapes (2-edge directed path LHS {(x,y),(y,z)}):
//   productive  -> {(x,y),(y,w),(w,z)}   (adds an edge + fresh vertex)
//   reductive   -> {(x,z)}               (removes an edge)
//   idempotent  -> {(z,y),(y,x)}         (reverses; edge count fixed)
//
// Caveat: re-merge uses the WL state hash as an isomorphism proxy (WL can
// collide), so re-merge counts are a lower bound on true reconvergence.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using Vertex = int;
using Edge   = std::vector<Vertex>;
using Graph  = std::vector<Edge>;

static constexpr uint64_t FNV_OFFSET = 1469598103934665603ull;
static inline uint64_t hmix(uint64_t s, uint64_t v) { s ^= v + 0x9e3779b97f4a7c15ull + (s << 6) + (s >> 2); return s; }
static inline uint64_t hlist(const std::vector<uint64_t>& xs) { uint64_t h = FNV_OFFSET; h = hmix(h, xs.size()); for (uint64_t x : xs) h = hmix(h, x); return h; }

struct Occ { int edge; int pos; int arity; };
static std::vector<Vertex> vertices_of(const Graph& g) { std::set<Vertex> s; for (auto& e : g) for (Vertex v : e) s.insert(v); return {s.begin(), s.end()}; }
static std::unordered_map<Vertex, std::vector<Occ>> build_adj(const Graph& g) {
    std::unordered_map<Vertex, std::vector<Occ>> a;
    for (int ei = 0; ei < (int)g.size(); ++ei) { int ar = (int)g[ei].size(); for (int p = 0; p < ar; ++p) a[g[ei][p]].push_back({ei, p, ar}); }
    return a;
}

// ---- WL (per-vertex colours at fixed rounds; commutative-combiner state hash)
static uint64_t wl_init(const std::vector<Occ>& occ) {
    std::vector<uint64_t> t; for (const Occ& o : occ) t.push_back(hmix(hmix(FNV_OFFSET, (uint64_t)o.arity), (uint64_t)o.pos));
    std::sort(t.begin(), t.end()); return hlist(t);
}
static std::unordered_map<Vertex, uint64_t> wl_colors(const Graph& g, int rounds) {
    auto adj = build_adj(g);
    std::unordered_map<Vertex, uint64_t> col; for (auto& kv : adj) col[kv.first] = wl_init(kv.second);
    for (int r = 0; r < rounds; ++r) {
        std::unordered_map<Vertex, uint64_t> nx; nx.reserve(col.size());
        for (auto& kv : adj) {
            Vertex v = kv.first; std::vector<uint64_t> toks;
            for (const Occ& o : kv.second) {
                const Edge& e = g[o.edge]; uint64_t t = hmix(hmix(FNV_OFFSET, (uint64_t)o.arity), (uint64_t)o.pos);
                std::vector<uint64_t> sl; for (int q = 0; q < o.arity; ++q) if (q != o.pos) sl.push_back(hmix((uint64_t)q, col[e[q]]));
                std::sort(sl.begin(), sl.end()); for (uint64_t s : sl) t = hmix(t, s); toks.push_back(t);
            }
            std::sort(toks.begin(), toks.end()); nx[v] = hmix(col[v], hlist(toks));
        }
        col.swap(nx);
    }
    return col;
}
static size_t n_classes(const std::unordered_map<Vertex, uint64_t>& c) { std::set<uint64_t> s; for (auto& kv : c) s.insert(kv.second); return s.size(); }
static int wl_fixpoint_rounds(const Graph& g) {
    int n = (int)vertices_of(g).size(); size_t prev = 0;
    for (int r = 1; r <= n + 1; ++r) { size_t c = n_classes(wl_colors(g, r)); if (c == prev) return r; prev = c; }
    return n + 1;
}
static uint64_t wl_state_hash(const Graph& g) {
    if (g.empty()) return 0;
    auto col = wl_colors(g, wl_fixpoint_rounds(g));
    uint64_t sum = 0; for (auto& kv : col) sum += kv.second;
    return hmix(hmix(hmix(FNV_OFFSET, col.size()), n_classes(col)), sum);
}

// wl_delta between two graphs (vertex ids are comparable: rewrites preserve ids,
// fresh vertices count as added).
static double wl_delta(const Graph& a, const Graph& b) {
    int R = std::max(wl_fixpoint_rounds(a), wl_fixpoint_rounds(b));
    auto ca = wl_colors(a, R), cb = wl_colors(b, R);
    int chg = 0, add = 0, rem = 0;
    for (auto& kv : ca) { auto it = cb.find(kv.first); if (it == cb.end()) rem++; else if (it->second != kv.second) chg++; }
    for (auto& kv : cb) if (!ca.count(kv.first)) add++;
    int denom = std::max(ca.size(), cb.size());
    return denom ? double(chg + add + rem) / denom : 0.0;
}

// edge symmetric-difference over vertex-tuples (size of the divergent region).
static int edge_symdiff(const Graph& a, const Graph& b) {
    std::map<Edge, int> m;
    for (auto& e : a) m[e]++;
    for (auto& e : b) m[e]--;
    int d = 0; for (auto& kv : m) d += std::abs(kv.second);
    return d;
}

// ---- Rule + matcher (2-edge directed path LHS) ----
struct Rule { std::string name; std::vector<std::vector<int>> rhs; };  // slots: 0=x,1=y,2=z, 100+k=fresh
struct Match { int i, j; Vertex x, y, z; };

static int max_vertex(const Graph& g) { int mx = -1; for (auto& e : g) for (Vertex v : e) mx = std::max(mx, v); return mx; }

static std::vector<Match> find_matches(const Graph& A) {
    std::vector<Match> out;
    // group edges by first vertex for the join on the shared middle vertex
    std::unordered_map<Vertex, std::vector<int>> by_first;
    for (int j = 0; j < (int)A.size(); ++j) if (A[j].size() == 2) by_first[A[j][0]].push_back(j);
    for (int i = 0; i < (int)A.size(); ++i) {
        if (A[i].size() != 2) continue;
        auto it = by_first.find(A[i][1]);
        if (it == by_first.end()) continue;
        for (int j : it->second) if (j != i) out.push_back({i, j, A[i][0], A[i][1], A[j][1]});
    }
    return out;
}

// Apply a match; return child and the indices (in the child) of produced edges.
static Graph apply_match(const Graph& A, const Match& m, const Rule& r, std::vector<int>& produced_idx) {
    Graph child; child.reserve(A.size() + r.rhs.size());
    for (int e = 0; e < (int)A.size(); ++e) if (e != m.i && e != m.j) child.push_back(A[e]);
    int base = max_vertex(A) + 1;
    int start = (int)child.size();
    for (auto& tmpl : r.rhs) {
        Edge e;
        for (int slot : tmpl) e.push_back(slot == 0 ? m.x : slot == 1 ? m.y : slot == 2 ? m.z : base + (slot - 100));
        child.push_back(e);
    }
    produced_idx.clear();
    for (int k = start; k < (int)child.size(); ++k) produced_idx.push_back(k);
    return child;
}

// causal out-degree: next-step rewrites in `child` consuming a produced edge.
static int causal_out(const Graph& child, const std::vector<int>& produced_idx) {
    std::set<int> prod(produced_idx.begin(), produced_idx.end());
    int count = 0;
    for (const Match& m : find_matches(child)) if (prod.count(m.i) || prod.count(m.j)) ++count;
    return count;
}

// Depth at which the forward multiway cones of two branches first share a
// canonical state (= branches reconverge / "collapse"). Returns Kmax+1 if they
// do not reconverge within Kmax (censored). Cones are width-capped.
static int remerge_depth(const Graph& bi, const Graph& bj, const Rule& r, int Kmax, int width) {
    std::set<uint64_t> si{wl_state_hash(bi)}, sj{wl_state_hash(bj)};
    for (uint64_t h : si) if (sj.count(h)) return 0;
    std::vector<Graph> fi{bi}, fj{bj};
    auto expand = [&](std::vector<Graph>& f, std::set<uint64_t>& seen) {
        std::vector<Graph> nx;
        for (auto& g : f) for (const Match& m : find_matches(g)) {
            std::vector<int> pid; Graph c = apply_match(g, m, r, pid);
            uint64_t h = wl_state_hash(c);
            if (seen.insert(h).second && (int)nx.size() < width) nx.push_back(std::move(c));
        }
        f.swap(nx);
    };
    for (int d = 1; d <= Kmax; ++d) {
        expand(fi, si); expand(fj, sj);
        for (uint64_t h : si) if (sj.count(h)) return d;
        if (fi.empty() && fj.empty()) break;
    }
    return Kmax + 1;
}

static double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    int n = (int)x.size(); if (n < 2) return 0;
    double mx = 0, my = 0; for (int i = 0; i < n; ++i) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0;
    for (int i = 0; i < n; ++i) { double dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
    return (sxx <= 0 || syy <= 0) ? 0.0 : sxy / std::sqrt(sxx * syy);
}

// ---- generators ----
static Graph gen_cycle(int n) { Graph g; for (int i = 0; i < n; ++i) g.push_back({i, (i + 1) % n}); return g; }
static Graph gen_path(int n) { Graph g; for (int i = 0; i + 1 < n; ++i) g.push_back({i, i + 1}); return g; }
static Graph gen_two_cycles(int len) { Graph g; for (int c = 0; c < 2; ++c) for (int i = 0; i < len; ++i) g.push_back({c * len + i, c * len + (i + 1) % len}); return g; }
static Graph gen_grid(int w, int h) {
    Graph g; auto id = [&](int x, int y) { return y * w + x; };
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) { if (x + 1 < w) g.push_back({id(x, y), id(x + 1, y)}); if (y + 1 < h) g.push_back({id(x, y), id(x, y + 1)}); }
    return g;
}
static Graph gen_random(int n, int m, uint32_t seed) {
    std::mt19937 rng(seed); std::uniform_int_distribution<int> vd(0, n - 1);
    Graph g; for (int i = 0; i < m; ++i) g.push_back({vd(rng), vd(rng)}); return g;
}

// ============================================================================

struct Pair { double flux; double edge_diff; double wl_delta; Graph bi; Graph bj; };

// Collect all branchlike sibling pairs from one parent state under a rule.
static std::vector<Pair> branchlike_pairs(const Graph& A, const Rule& r) {
    std::vector<Match> ms = find_matches(A);
    int M = (int)ms.size();
    std::vector<Graph> child(M); std::vector<std::vector<int>> prod(M); std::vector<int> cout(M);
    for (int a = 0; a < M; ++a) { child[a] = apply_match(A, ms[a], r, prod[a]); cout[a] = causal_out(child[a], prod[a]); }
    std::vector<Pair> out;
    for (int a = 0; a < M; ++a) for (int b = a + 1; b < M; ++b) {
        bool branchlike = (ms[a].i == ms[b].i || ms[a].i == ms[b].j || ms[a].j == ms[b].i || ms[a].j == ms[b].j);
        if (!branchlike) continue;
        Pair p;
        p.flux = cout[a] + cout[b];
        p.edge_diff = edge_symdiff(child[a], child[b]);
        p.wl_delta = wl_delta(child[a], child[b]);
        p.bi = child[a]; p.bj = child[b];
        out.push_back(std::move(p));
    }
    return out;
}

static void hypothesis_a(const Rule& r, const std::vector<std::pair<std::string, Graph>>& states) {
    std::printf("== %s rule : (a) divergence vs causal-flux mass proxy ==\n", r.name.c_str());
    std::vector<double> flux, ediff, wdelta;
    for (auto& s : states) {
        auto pairs = branchlike_pairs(s.second, r);
        for (auto& p : pairs) { flux.push_back(p.flux); ediff.push_back(p.edge_diff); wdelta.push_back(p.wl_delta); }
    }
    int n = (int)flux.size();
    std::printf("   branchlike pairs: %d\n", n);
    if (n < 2) { std::printf("   (too few pairs)\n\n"); return; }
    std::printf("   Pearson(flux, edge_diff) = %+.3f     Pearson(flux, wl_delta) = %+.3f\n",
                pearson(flux, ediff), pearson(flux, wdelta));
    // bucketed table by flux
    double fmax = *std::max_element(flux.begin(), flux.end());
    std::printf("   flux-bucket   n     mean edge_diff   mean wl_delta\n");
    int B = 4;
    for (int b = 0; b < B; ++b) {
        double lo = fmax * b / B, hi = fmax * (b + 1) / B + (b == B - 1 ? 1e-9 : 0);
        double se = 0, sw = 0; int cnt = 0;
        for (int i = 0; i < n; ++i) if (flux[i] >= lo && flux[i] <= hi) { se += ediff[i]; sw += wdelta[i]; ++cnt; }
        if (cnt) std::printf("   [%4.0f,%4.0f]  %-5d %-16.2f %-.3f\n", lo, hi, cnt, se / cnt, sw / cnt);
    }
    std::printf("\n");
}

static void hypothesis_b(const Rule& r, const std::vector<std::pair<std::string, Graph>>& states,
                         int Kmax, int width_cap, int max_pairs_per_state) {
    std::printf("== %s rule : (b) re-merge DEPTH (collapse time) vs flux, Kmax=%d ==\n", r.name.c_str(), Kmax);
    struct R { double flux; int depth; };
    std::vector<R> rows;
    std::vector<double> fx, dp;
    for (auto& s : states) {
        auto pairs = branchlike_pairs(s.second, r);
        int taken = 0;
        for (auto& p : pairs) {
            if (taken++ >= max_pairs_per_state) break;
            int d = remerge_depth(p.bi, p.bj, r, Kmax, width_cap);
            rows.push_back({p.flux, d});
            fx.push_back(p.flux); dp.push_back(d);
        }
    }
    if (rows.size() < 2) { std::printf("   (too few pairs)\n\n"); return; }
    std::printf("   Pearson(flux, remerge_depth) = %+.3f   (negative => more mass => collapses sooner)\n", pearson(fx, dp));
    double fmax = 0; for (auto& x : rows) fmax = std::max(fmax, x.flux);
    std::printf("   flux-bucket   n     mean remerge-depth   censored(>Kmax)\n");
    int B = 4;
    for (int b = 0; b < B; ++b) {
        double lo = fmax * b / B, hi = fmax * (b + 1) / B + (b == B - 1 ? 1e-9 : 0);
        int cnt = 0, cens = 0; double sd = 0;
        for (auto& x : rows) if (x.flux >= lo && x.flux <= hi) { ++cnt; sd += x.depth; if (x.depth > Kmax) ++cens; }
        if (cnt) std::printf("   [%4.0f,%4.0f]  %-5d %-20.2f %.2f\n", lo, hi, cnt, sd / cnt, double(cens) / cnt);
    }
    std::printf("\n");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<std::pair<std::string, Graph>> states = {
        {"cycle-12",     gen_cycle(12)},
        {"cycle-20",     gen_cycle(20)},
        {"path-16",      gen_path(16)},
        {"two-cycles-8", gen_two_cycles(8)},
        {"grid-4x4",     gen_grid(4, 4)},
        {"grid-5x5",     gen_grid(5, 5)},
        {"random-16-20", gen_random(16, 20, 0xA11CE)},
        {"random-16-24", gen_random(16, 24, 0xB0B)},
        {"random-20-28", gen_random(20, 28, 0xC0FFEE)},
    };

    Rule productive{"productive", {{0, 1}, {1, 100}, {100, 2}}};
    Rule reductive {"reductive",  {{0, 2}}};
    Rule idempotent{"idempotent", {{2, 1}, {1, 0}}};

    std::printf("Branchial-diff vs causal-flux probe\n");
    std::printf("flux = causal out-degree sum (mass proxy); edge_diff/wl_delta = divergence.\n\n");

    for (const Rule* r : {&productive, &reductive, &idempotent}) hypothesis_a(*r, states);

    // (b): re-merge is meaningful for size-preserving / shrinking rules; the
    // productive rule grows the graph so cones rarely reconverge within K.
    // Sampled (capped cone width + pairs/state) for tractability — the
    // size-preserving idempotent multiway is otherwise explosive.
    std::vector<std::pair<std::string, Graph>> small = {
        {"cycle-12",     gen_cycle(12)},
        {"two-cycles-8", gen_two_cycles(8)},
        {"grid-4x4",     gen_grid(4, 4)},
        {"path-16",      gen_path(16)},
        {"random-16-20", gen_random(16, 20, 0xA11CE)},
    };
    hypothesis_b(idempotent, small, 4, 10, 6);
    hypothesis_b(reductive, small, 5, 16, 8);

    std::printf("Reading: (a) positive Pearson => more causal flux (mass) => larger divergence.\n"
                "(b) negative Pearson(flux, remerge_depth) => more mass => collapses (reconverges) sooner.\n");
    return 0;
}
