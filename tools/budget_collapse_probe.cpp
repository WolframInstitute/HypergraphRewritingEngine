// tools/budget_collapse_probe.cpp
//
// Standalone, header-free test of the rescued OR hunch: a BOUNDED OBSERVER whose
// cost to keep two branches distinct IS the computational complexity of deciding
// their equivalence (IR canonicalisation cost), with a fixed budget Θ that forces
// a merge ("collapse") when accumulated cost·time exceeds Θ.
//
// Build:
//   g++ -O2 -std=c++17 tools/budget_collapse_probe.cpp -o /tmp/budget_collapse_probe
//   /tmp/budget_collapse_probe
//
// This is the experiment that is actually ABOUT the hypothesis (complexity of
// equivalence drives collapse under a bounded observer), rather than adjacent to
// it. The branchial-flux probe showed the FREE dynamics spread (anti-Penrose);
// here we impose the budget and ask three non-trivial questions:
//
//   (I)  Mechanism: does a finite Θ keep the live-branch population BOUNDED while
//        the free (Θ=∞) multiway explodes? (observer maintains a finite
//        "classical" reality)
//   (II) Is equivalence-complexity MASS-LIKE? Correlate IR canonicalisation cost
//        against size (#edges) and a causal-flux proxy across many states. If it
//        only tracks symmetry and not mass, the hunch is about symmetry, not mass.
//   (III) Collapse law: τ ≈ Θ / cost holds by construction; we report the spread
//        of cost·τ (how tight the E·τ ≈ const cutoff is) and whether higher-cost
//        (harder-to-equivalence) branches collapse sooner.
//
// "Cost" = IR search-node count to canonicalise a state (the literal cost of the
// equivalence test), capped (cap = maximally hard). c(Bi,Bj) = cost(Bi)+cost(Bj).
// Iso-dedup uses the WL state hash (an isomorphism proxy; may over-merge).

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

// ---- WL state hash (iso proxy for dedup) ----
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
            std::vector<uint64_t> toks;
            for (const Occ& o : kv.second) {
                const Edge& e = g[o.edge]; uint64_t t = hmix(hmix(FNV_OFFSET, (uint64_t)o.arity), (uint64_t)o.pos);
                std::vector<uint64_t> sl; for (int q = 0; q < o.arity; ++q) if (q != o.pos) sl.push_back(hmix((uint64_t)q, col[e[q]]));
                std::sort(sl.begin(), sl.end()); for (uint64_t s : sl) t = hmix(t, s); toks.push_back(t);
            }
            std::sort(toks.begin(), toks.end()); nx[kv.first] = hmix(col[kv.first], hlist(toks));
        }
        col.swap(nx);
    }
    return col;
}
static size_t n_classes(const std::unordered_map<Vertex, uint64_t>& c) { std::set<uint64_t> s; for (auto& kv : c) s.insert(kv.second); return s.size(); }
static uint64_t wl_state_hash(const Graph& g) {
    if (g.empty()) return 1;
    int n = (int)vertices_of(g).size(), R = 1; size_t prev = 0;
    for (int r = 1; r <= n + 1; ++r) { size_t cc = n_classes(wl_colors(g, r)); if (cc == prev) { R = r; break; } prev = cc; R = r; }
    auto col = wl_colors(g, R);
    uint64_t sum = 0; for (auto& kv : col) sum += kv.second;
    return hmix(hmix(hmix(FNV_OFFSET, col.size()), n_classes(col)), sum);
}

// ---- IR canonicalisation cost (search nodes; capped) = equivalence-test cost ----
static constexpr long long IR_CAP = 4000;

static void ir_refine(std::vector<std::vector<Vertex>>& cells, const Graph& g,
                      const std::unordered_map<Vertex, std::vector<Occ>>& adj) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::unordered_map<Vertex, int> cell_of;
        for (int ci = 0; ci < (int)cells.size(); ++ci) for (Vertex v : cells[ci]) cell_of[v] = ci;
        std::vector<std::vector<Vertex>> next;
        for (auto& cell : cells) {
            if (cell.size() <= 1) { next.push_back(cell); continue; }
            std::vector<std::pair<uint64_t, Vertex>> sig;
            for (Vertex v : cell) {
                std::vector<uint64_t> toks; auto it = adj.find(v);
                if (it != adj.end()) for (const Occ& o : it->second) {
                    const Edge& e = g[o.edge]; uint64_t t = hmix(hmix(FNV_OFFSET, (uint64_t)o.arity), (uint64_t)o.pos);
                    std::vector<uint64_t> sl; for (int q = 0; q < o.arity; ++q) if (q != o.pos) sl.push_back(hmix((uint64_t)q, (uint64_t)cell_of[e[q]]));
                    std::sort(sl.begin(), sl.end()); for (uint64_t s : sl) t = hmix(t, s); toks.push_back(t);
                }
                std::sort(toks.begin(), toks.end()); sig.push_back({hlist(toks), v});
            }
            std::sort(sig.begin(), sig.end());
            size_t i = 0; std::vector<std::vector<Vertex>> sub;
            while (i < sig.size()) { size_t j = i; std::vector<Vertex> p; while (j < sig.size() && sig[j].first == sig[i].first) p.push_back(sig[j++].second); sub.push_back(std::move(p)); i = j; }
            if (sub.size() > 1) changed = true;
            for (auto& s : sub) next.push_back(std::move(s));
        }
        cells.swap(next);
    }
}
static uint64_t ir_form_hash(const std::vector<Vertex>& order, const Graph& g) {
    std::unordered_map<Vertex, int> lab; for (int i = 0; i < (int)order.size(); ++i) lab[order[i]] = i;
    std::vector<std::vector<int>> es; for (const Edge& e : g) { std::vector<int> r; for (Vertex v : e) r.push_back(lab[v]); es.push_back(std::move(r)); }
    std::sort(es.begin(), es.end());
    uint64_t h = hmix(FNV_OFFSET, es.size()); for (auto& e : es) { h = hmix(h, e.size()); for (int x : e) h = hmix(h, (uint64_t)x); }
    return h;
}
static void ir_search(std::vector<std::vector<Vertex>> cells, const Graph& g,
                      const std::unordered_map<Vertex, std::vector<Occ>>& adj,
                      bool& have, uint64_t& best, long long& nodes, bool& capped) {
    if (capped) return;
    if (nodes > IR_CAP) { capped = true; return; }
    ++nodes;
    ir_refine(cells, g, adj);
    size_t target = (size_t)-1;
    for (size_t i = 0; i < cells.size(); ++i) if (cells[i].size() > 1) { target = i; break; }
    if (target == (size_t)-1) { std::vector<Vertex> o; for (auto& c : cells) o.push_back(c[0]); uint64_t h = ir_form_hash(o, g); if (!have || h < best) { best = h; have = true; } return; }
    std::vector<Vertex> cv = cells[target];
    for (Vertex v : cv) {
        std::vector<std::vector<Vertex>> nc;
        for (size_t i = 0; i < cells.size(); ++i) { if (i != (size_t)target) { nc.push_back(cells[i]); continue; } nc.push_back({v}); std::vector<Vertex> rest; for (Vertex w : cells[i]) if (w != v) rest.push_back(w); if (!rest.empty()) nc.push_back(std::move(rest)); }
        ir_search(std::move(nc), g, adj, have, best, nodes, capped);
        if (capped) return;
    }
}
// returns IR search-node count (the equivalence-decision cost); capped at IR_CAP.
static long long ir_cost(const Graph& g) {
    if (g.empty()) return 1;
    auto adj = build_adj(g);
    std::map<uint64_t, std::vector<Vertex>> by; for (Vertex v : vertices_of(g)) by[wl_init(adj[v])].push_back(v);
    std::vector<std::vector<Vertex>> cells; for (auto& kv : by) cells.push_back(kv.second);
    bool have = false, capped = false; uint64_t best = 0; long long nodes = 0;
    ir_search(cells, g, adj, have, best, nodes, capped);
    return (capped ? IR_CAP : nodes) + 1;
}

// ---- rule + matcher (2-edge directed path) ----
struct Rule { std::string name; std::vector<std::vector<int>> rhs; };
struct Match { int i, j; Vertex x, y, z; };
static int max_vertex(const Graph& g) { int mx = -1; for (auto& e : g) for (Vertex v : e) mx = std::max(mx, v); return mx; }
static std::vector<Match> find_matches(const Graph& A) {
    std::vector<Match> out;
    std::unordered_map<Vertex, std::vector<int>> by_first;
    for (int j = 0; j < (int)A.size(); ++j) if (A[j].size() == 2) by_first[A[j][0]].push_back(j);
    for (int i = 0; i < (int)A.size(); ++i) { if (A[i].size() != 2) continue; auto it = by_first.find(A[i][1]); if (it == by_first.end()) continue; for (int j : it->second) if (j != i) out.push_back({i, j, A[i][0], A[i][1], A[j][1]}); }
    return out;
}
static Graph apply_match(const Graph& A, const Match& m, const Rule& r, std::vector<int>& produced_idx) {
    Graph child; for (int e = 0; e < (int)A.size(); ++e) if (e != m.i && e != m.j) child.push_back(A[e]);
    int base = max_vertex(A) + 1, start = (int)child.size();
    for (auto& t : r.rhs) { Edge e; for (int s : t) e.push_back(s == 0 ? m.x : s == 1 ? m.y : s == 2 ? m.z : base + (s - 100)); child.push_back(e); }
    produced_idx.clear(); for (int k = start; k < (int)child.size(); ++k) produced_idx.push_back(k);
    return child;
}
static int causal_out(const Graph& child, const std::vector<int>& produced_idx) {
    std::set<int> prod(produced_idx.begin(), produced_idx.end()); int c = 0;
    for (const Match& m : find_matches(child)) if (prod.count(m.i) || prod.count(m.j)) ++c;
    return c;
}

static double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    int n = (int)x.size(); if (n < 2) return 0;
    double mx = 0, my = 0; for (int i = 0; i < n; ++i) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0; for (int i = 0; i < n; ++i) { double dx = x[i]-mx, dy = y[i]-my; sxy += dx*dy; sxx += dx*dx; syy += dy*dy; }
    return (sxx <= 0 || syy <= 0) ? 0.0 : sxy / std::sqrt(sxx * syy);
}

// ---- generators ----
static Graph gen_cycle(int n) { Graph g; for (int i = 0; i < n; ++i) g.push_back({i, (i + 1) % n}); return g; }
static Graph gen_path(int n) { Graph g; for (int i = 0; i + 1 < n; ++i) g.push_back({i, i + 1}); return g; }
static Graph gen_two_cycles(int len) { Graph g; for (int c = 0; c < 2; ++c) for (int i = 0; i < len; ++i) g.push_back({c*len+i, c*len+(i+1)%len}); return g; }
static Graph gen_grid(int w, int h) { Graph g; auto id=[&](int x,int y){return y*w+x;}; for (int y=0;y<h;++y) for (int x=0;x<w;++x){ if(x+1<w) g.push_back({id(x,y),id(x+1,y)}); if(y+1<h) g.push_back({id(x,y),id(x,y+1)});} return g; }
static Graph gen_random(int n, int m, uint32_t seed) { std::mt19937 rng(seed); std::uniform_int_distribution<int> vd(0,n-1); Graph g; for(int i=0;i<m;++i) g.push_back({vd(rng),vd(rng)}); return g; }

// ============================================================================
// (II) Is equivalence-complexity mass-like?
// ============================================================================
static void test_mass_likeness() {
    std::printf("== (II) is equivalence-complexity (IR cost) mass-like? ==\n");
    std::vector<std::pair<std::string, Graph>> st = {
        {"path-10", gen_path(10)}, {"path-16", gen_path(16)}, {"path-24", gen_path(24)},
        {"cycle-8", gen_cycle(8)}, {"cycle-12", gen_cycle(12)}, {"cycle-18", gen_cycle(18)}, {"cycle-24", gen_cycle(24)},
        {"two-cyc-6", gen_two_cycles(6)}, {"two-cyc-10", gen_two_cycles(10)},
        {"grid-3x3", gen_grid(3,3)}, {"grid-4x4", gen_grid(4,4)}, {"grid-3x6", gen_grid(3,6)},
        {"rand-12-16", gen_random(12,16,1)}, {"rand-16-22", gen_random(16,22,2)}, {"rand-22-30", gen_random(22,30,3)},
    };
    std::vector<double> cost, edges, flux;
    std::printf("   %-12s %-8s %-8s %-8s\n", "state", "IRcost", "edges", "flux");
    for (auto& s : st) {
        long long c = ir_cost(s.second);
        int ne = (int)s.second.size();
        long long fl = 0; for (const Match& m : find_matches(s.second)) { std::vector<int> pid; Graph ch = apply_match(s.second, m, Rule{"p",{{0,1},{1,100},{100,2}}}, pid); fl += causal_out(ch, pid); }
        cost.push_back((double)c); edges.push_back(ne); flux.push_back((double)fl);
        std::printf("   %-12s %-8lld %-8d %-8lld\n", s.first.c_str(), c, ne, fl);
    }
    std::printf("   Pearson(IRcost, edges) = %+.3f   Pearson(IRcost, causal_flux) = %+.3f\n",
                pearson(cost, edges), pearson(cost, flux));
    std::printf("   (high => equivalence-complexity grows with mass; ~0 => it tracks symmetry, not mass)\n\n");
}

// ============================================================================
// (I) Finite-capacity bounded observer: when the frontier exceeds capacity, the
// observer can no longer afford every distinct branch and DROPS the ones it can
// least afford to maintain, under one of several cost models. The discriminating
// question is whether the resulting collapse is MASS-SELECTIVE (removes massive
// branches) — which is what OR needs — or mass-blind.
// ============================================================================
enum CostMode { IRCOST, MASS_EDGES, MASS_FLUX, RANDOM };
static const char* mode_name(CostMode m) { return m == IRCOST ? "equiv-complexity" : m == MASS_EDGES ? "mass(edges)" : m == MASS_FLUX ? "mass(causal-flux)" : "random"; }

static long long state_flux(const Graph& g, const Rule& rule) {
    long long fl = 0; for (const Match& m : find_matches(g)) { std::vector<int> pid; Graph c = apply_match(g, m, rule, pid); fl += causal_out(c, pid); } return fl;
}

// Mass-selectivity: when the observer drops branches under a given cost model,
// is the mean causal-flux (mass) of the DROPPED branches higher than that of the
// SURVIVORS? Returns the mean over steps of (mean_flux_dropped - mean_flux_surv).
static double mass_selectivity(const Graph& init, const Rule& rule, CostMode mode, int N_target, int T) {
    std::unordered_map<uint64_t, Graph> live; live[wl_state_hash(init)] = init;
    std::unordered_map<uint64_t, double> flux_cache, irc_cache;
    auto flux_of = [&](uint64_t h, const Graph& g) { auto it = flux_cache.find(h); if (it != flux_cache.end()) return it->second; double f = (double)state_flux(g, rule); flux_cache[h] = f; return f; };
    auto cost_of = [&](uint64_t h, const Graph& g) -> double {
        if (mode == IRCOST) { auto it = irc_cache.find(h); if (it != irc_cache.end()) return it->second; double c = (double)ir_cost(g); irc_cache[h] = c; return c; }
        if (mode == MASS_EDGES) return (double)g.size();
        if (mode == MASS_FLUX) return flux_of(h, g);
        return (double)(h % 1000003);
    };
    double sel = 0; int steps = 0;
    for (int t = 1; t <= T; ++t) {
        std::unordered_map<uint64_t, Graph> next;
        for (auto& kv : live) for (const Match& m : find_matches(kv.second)) { std::vector<int> pid; Graph c = apply_match(kv.second, m, rule, pid); next[wl_state_hash(c)] = std::move(c); }
        if (next.empty()) break;
        if ((int)next.size() <= N_target) { live.swap(next); continue; }
        std::vector<std::pair<double, uint64_t>> scored;
        for (auto& kv : next) scored.push_back({cost_of(kv.first, kv.second), kv.first});
        std::sort(scored.begin(), scored.end(), [](auto& a, auto& b){ return a.first > b.first; });
        int n_drop = (int)next.size() - N_target;
        double fd = 0, fs = 0; int nd = 0, ns = 0;
        std::unordered_map<uint64_t, Graph> keep;
        for (int i = 0; i < (int)scored.size(); ++i) {
            uint64_t h = scored[i].second; double f = flux_of(h, next[h]);
            if (i < n_drop) { fd += f; ++nd; } else { fs += f; ++ns; keep[h] = next[h]; }
        }
        if (nd && ns) { sel += (fd / nd) - (fs / ns); ++steps; }
        live.swap(keep);
    }
    return steps ? sel / steps : 0.0;
}

static void test_capacity(const Graph& init, const char* name, const Rule& rule) {
    std::printf("== (I) finite-capacity observer on %s (%s rule), capacity=12 ==\n", name, rule.name.c_str());
    std::printf("   does dropping by <cost> preferentially remove high-causal-flux (massive) branches?\n");
    for (CostMode mode : {MASS_FLUX, IRCOST, MASS_EDGES, RANDOM})
        std::printf("   %-18s flux-selectivity(dropped - survivors) = %+.2f\n",
                    mode_name(mode), mass_selectivity(init, rule, mode, 12, 6));
    std::printf("\n");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("Budget-cutoff collapse model: cost-to-distinguish = IR equivalence-test cost.\n\n");

    test_mass_likeness();

    Rule productive{"productive", {{0,1},{1,100},{100,2}}};
    test_capacity(gen_grid(3, 3), "grid-3x3", productive);
    test_capacity(gen_grid(4, 4), "grid-4x4", productive);
    test_capacity(gen_random(10, 13, 7), "random-10-13", productive);
    test_capacity(gen_random(12, 16, 9), "random-12-16", productive);

    std::printf("Reading:\n"
                " (II) Pearson(IRcost, mass) ~ 0 => equivalence-complexity is NOT mass-like; it tracks symmetry.\n"
                " (I) flux-selectivity = mean flux of dropped branches minus survivors. mass(causal-flux) cost\n"
                "     is strongly positive (drops massive branches, by construction); equiv-complexity is ~0,\n"
                "     indistinguishable from random => dropping by equivalence-complexity gives a MASS-BLIND\n"
                "     collapse. Together: equivalence-complexity cannot be the OR (Penrose E_G) driver; a\n"
                "     genuine mass quantity (causal flux) can.\n");
    return 0;
}
