// tools/higgs_shadow_probe.cpp
//
// Standalone, header-free test of the "Higgs shadow": in the Standard Model mass
// comes from BROKEN symmetry (unbroken symmetry => massless). If, in the rewrite
// substrate, the symmetry channel = automorphism and the mass channel = causal
// flux, then a rewrite that BREAKS a state's automorphism should GENERATE causal
// flux. Prediction: Δ(symmetry) and Δ(mass) anti-correlate — symmetry traded for
// mass, rewrite by rewrite.
//
// Build:
//   g++ -O2 -std=c++17 tools/higgs_shadow_probe.cpp -o /tmp/higgs_shadow_probe
//   /tmp/higgs_shadow_probe
//
// Hardened over the first cut:
//   * EXACT |Aut| via backtracking (WL-colour + incremental edge-consistency
//     pruning) — no IR-leaf cap, so highly symmetric states are not skipped.
//   * larger graphs.
//   * pooled PARTIAL correlation of ΔS vs Δflux controlling for Δedges and
//     Δvertices, so "amount of change" is held fixed across ALL rules (not just
//     within the size-preserving idempotent rule).
//
// Per single rewrite A -> B:
//   symmetry  S(G)    = log2 |Aut(G)|   (0 asymmetric; grows with symmetry)
//   mass      flux(G) = sum over matches of causal out-degree (state energy)
//   ΔS = S(B)-S(A);  Δflux = flux(B)-flux(A);  Δedges, Δverts = size changes.
//
// Higgs prediction: Pearson(ΔS, Δflux) < 0, and it survives controlling for size.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
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
static std::unordered_map<Vertex, uint64_t> wl_fixpoint_colors(const Graph& g) {
    int n = (int)vertices_of(g).size(); size_t prev = 0;
    std::unordered_map<Vertex, uint64_t> last;
    for (int r = 1; r <= n + 1; ++r) { last = wl_colors(g, r); size_t cc = n_classes(last); if (cc == prev) break; prev = cc; }
    return last;
}

// ---- EXACT |Aut| via backtracking (WL-colour + incremental edge-consistency) ----
// log2|Aut|; capped=true only if the safety node cap is hit (shouldn't for the
// structured graph families used here).
static double sym_of(const Graph& g, bool& capped) {
    capped = false;
    std::vector<Vertex> V = vertices_of(g); int n = (int)V.size();
    if (n == 0) return 0.0;
    std::unordered_map<Vertex, int> idx; for (int i = 0; i < n; ++i) idx[V[i]] = i;
    auto col = wl_fixpoint_colors(g);
    std::vector<uint64_t> color(n); for (int i = 0; i < n; ++i) color[i] = col[V[i]];
    std::vector<std::vector<int>> E; for (auto& e : g) { std::vector<int> r; for (Vertex v : e) r.push_back(idx[v]); E.push_back(std::move(r)); }
    std::map<std::vector<int>, int> rem; for (auto& e : E) rem[e]++;
    std::vector<int> arity(E.size()); std::vector<std::vector<int>> vinc(n);
    for (int ei = 0; ei < (int)E.size(); ++ei) { arity[ei] = (int)E[ei].size(); for (int v : E[ei]) vinc[v].push_back(ei); }
    std::map<uint64_t, int> csize; for (int i = 0; i < n; ++i) csize[color[i]]++;
    std::vector<int> order(n); for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (csize[color[a]] != csize[color[b]]) return csize[color[a]] < csize[color[b]];
        if (color[a] != color[b]) return color[a] < color[b];
        return a < b; });
    std::unordered_map<uint64_t, std::vector<int>> bycolor; for (int i = 0; i < n; ++i) bycolor[color[i]].push_back(i);

    std::vector<int> perm(n, -1), pending(E.size());
    for (int ei = 0; ei < (int)E.size(); ++ei) pending[ei] = arity[ei];
    std::vector<char> used(n, 0);
    long long count = 0, nodes = 0; const long long CAP = 500000;

    std::function<void(int)> bt = [&](int pos) {
        if (capped) return;
        if (++nodes > CAP) { capped = true; return; }
        if (pos == n) { ++count; return; }
        int v = order[pos];
        for (int w : bycolor[color[v]]) {
            if (used[w]) continue;
            perm[v] = w; used[w] = 1;
            std::vector<int> dec, mapped; bool ok = true;
            for (int ei : vinc[v]) {
                pending[ei]--; dec.push_back(ei);
                if (pending[ei] == 0) {
                    std::vector<int> me(arity[ei]); for (int s = 0; s < arity[ei]; ++s) me[s] = perm[E[ei][s]];
                    auto it = rem.find(me);
                    if (it == rem.end() || it->second <= 0) { ok = false; break; }
                    it->second--; mapped.push_back(ei);
                }
            }
            if (ok) bt(pos + 1);
            for (int ei : mapped) { std::vector<int> me(arity[ei]); for (int s = 0; s < arity[ei]; ++s) me[s] = perm[E[ei][s]]; rem[me]++; }
            for (int ei : dec) pending[ei]++;
            perm[v] = -1; used[w] = 0;
            if (capped) return;
        }
    };
    bt(0);
    return std::log2((double)(count < 1 ? 1 : count));
}

// ---- rule + matcher (2-edge path) ----
struct Rule { std::string name; std::vector<std::vector<int>> rhs; };
struct Match { int i, j; Vertex x, y, z; };
static int max_vertex(const Graph& g) { int mx = -1; for (auto& e : g) for (Vertex v : e) mx = std::max(mx, v); return mx; }
static std::vector<Match> find_matches(const Graph& A) {
    std::vector<Match> out; std::unordered_map<Vertex, std::vector<int>> bf;
    for (int j = 0; j < (int)A.size(); ++j) if (A[j].size() == 2) bf[A[j][0]].push_back(j);
    for (int i = 0; i < (int)A.size(); ++i) { if (A[i].size() != 2) continue; auto it = bf.find(A[i][1]); if (it == bf.end()) continue; for (int j : it->second) if (j != i) out.push_back({i, j, A[i][0], A[i][1], A[j][1]}); }
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
    for (const Match& m : find_matches(child)) { if (prod.count(m.i) || prod.count(m.j)) ++c; }
    return c;
}
static long long flux_of(const Graph& g, const Rule& r) {
    long long fl = 0; for (const Match& m : find_matches(g)) { std::vector<int> pid; Graph c = apply_match(g, m, r, pid); fl += causal_out(c, pid); } return fl;
}

static double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    int n = (int)x.size(); if (n < 2) return 0;
    double mx = 0, my = 0; for (int i = 0; i < n; ++i) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0; for (int i = 0; i < n; ++i) { double dx = x[i]-mx, dy = y[i]-my; sxy += dx*dy; sxx += dx*dx; syy += dy*dy; }
    return (sxx <= 0 || syy <= 0) ? 0.0 : sxy / std::sqrt(sxx * syy);
}
// residuals of y after OLS on predictors `cols` (+ intercept); small p, Gaussian elim.
static std::vector<double> ols_residuals(const std::vector<double>& y, const std::vector<std::vector<double>>& cols) {
    int n = (int)y.size(), p = (int)cols.size() + 1;
    auto xi = [&](int i, int a) -> double { return a == 0 ? 1.0 : cols[a - 1][i]; };
    std::vector<std::vector<double>> A(p, std::vector<double>(p + 1, 0.0));
    for (int i = 0; i < n; ++i) for (int a = 0; a < p; ++a) { for (int c = 0; c < p; ++c) A[a][c] += xi(i, a) * xi(i, c); A[a][p] += xi(i, a) * y[i]; }
    for (int c = 0; c < p; ++c) { // Gaussian elimination with partial pivot
        int piv = c; for (int r = c + 1; r < p; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        std::swap(A[c], A[piv]);
        if (std::fabs(A[c][c]) < 1e-12) continue;
        for (int r = 0; r < p; ++r) if (r != c) { double f = A[r][c] / A[c][c]; for (int k = c; k <= p; ++k) A[r][k] -= f * A[c][k]; }
    }
    std::vector<double> beta(p, 0.0); for (int c = 0; c < p; ++c) if (std::fabs(A[c][c]) > 1e-12) beta[c] = A[c][p] / A[c][c];
    std::vector<double> res(n); for (int i = 0; i < n; ++i) { double yh = 0; for (int a = 0; a < p; ++a) yh += beta[a] * xi(i, a); res[i] = y[i] - yh; }
    return res;
}

// ---- generators ----
static Graph gen_cycle(int n) { Graph g; for (int i = 0; i < n; ++i) g.push_back({i, (i + 1) % n}); return g; }
static Graph gen_path(int n) { Graph g; for (int i = 0; i + 1 < n; ++i) g.push_back({i, i + 1}); return g; }
static Graph gen_kcycles(int k, int len) { Graph g; for (int c = 0; c < k; ++c) for (int i = 0; i < len; ++i) g.push_back({c*len+i, c*len+(i+1)%len}); return g; }
static Graph gen_grid(int w, int h) { Graph g; auto id=[&](int x,int y){return y*w+x;}; for (int y=0;y<h;++y) for (int x=0;x<w;++x){ if(x+1<w) g.push_back({id(x,y),id(x+1,y)}); if(y+1<h) g.push_back({id(x,y),id(x,y+1)});} return g; }
static Graph gen_random(int n, int m, uint32_t seed) { std::mt19937 rng(seed); std::uniform_int_distribution<int> vd(0,n-1); Graph g; for(int i=0;i<m;++i) g.push_back({vd(rng),vd(rng)}); return g; }

struct Pool { std::vector<double> dS, dF, dE, dV; };

static void higgs_test(const Rule& rule, const std::vector<std::pair<std::string,Graph>>& states, Pool& pool) {
    std::printf("== %s rule : Higgs shadow (symmetry broken vs mass made) ==\n", rule.name.c_str());
    std::vector<double> dS, dF; int skipped = 0;
    for (auto& s : states) {
        bool capA = false; double SA = sym_of(s.second, capA); long long FA = flux_of(s.second, rule);
        int vA = (int)vertices_of(s.second).size(), eA = (int)s.second.size();
        if (capA) { skipped++; continue; }
        auto matches = find_matches(s.second);
        int stride = std::max(1, (int)matches.size() / 20);   // sample <=~20 rewrites/state
        for (int mi = 0; mi < (int)matches.size(); mi += stride) {
            const Match& m = matches[mi];
            std::vector<int> pid; Graph B = apply_match(s.second, m, rule, pid);
            bool capB = false; double SB = sym_of(B, capB);
            if (capB) { skipped++; continue; }
            long long FB = flux_of(B, rule);
            double ds = SB - SA, df = (double)(FB - FA);
            dS.push_back(ds); dF.push_back(df);
            pool.dS.push_back(ds); pool.dF.push_back(df);
            pool.dE.push_back((double)((int)B.size() - eA));
            pool.dV.push_back((double)((int)vertices_of(B).size() - vA));
        }
    }
    int n = (int)dS.size();
    std::printf("   rewrites: %d  (skipped %d)\n", n, skipped);
    if (n >= 2) {
        std::printf("   Pearson(ΔS, Δflux) = %+.3f\n", pearson(dS, dF));
        auto bucket = [&](const char* lab, auto pred) { double sf = 0; int c = 0; for (int i = 0; i < n; ++i) if (pred(dS[i])) { sf += dF[i]; ++c; } if (c) std::printf("   %-22s n=%-4d mean Δflux=%+.2f\n", lab, c, sf / c); };
        bucket("broke symmetry ΔS<0", [](double d){ return d < -1e-9; });
        bucket("symmetry unchanged",  [](double d){ return std::fabs(d) <= 1e-9; });
        bucket("gained symmetry ΔS>0",[](double d){ return d > 1e-9; });
    }
    std::printf("\n");
}

static const Rule R_idem{"idempotent", {{2, 1}, {1, 0}}};
static const Rule R_red {"reductive",  {{0, 2}}};
static const Rule R_prod{"productive", {{0, 1}, {1, 100}, {100, 2}}};

static void run_suite(const char* label, const std::vector<std::pair<std::string, Graph>>& states) {
    std::printf("################  %s  ################\n", label);
    Pool pool;
    higgs_test(R_idem, states, pool);
    higgs_test(R_red, states, pool);
    higgs_test(R_prod, states, pool);
    std::printf("== pooled (all rules), size-controlled ==\n");
    std::printf("   n=%zu   raw Pearson(ΔS,Δflux) = %+.3f\n", pool.dS.size(), pearson(pool.dS, pool.dF));
    std::vector<std::vector<double>> ctrl = {pool.dE, pool.dV};
    auto rS = ols_residuals(pool.dS, ctrl), rF = ols_residuals(pool.dF, ctrl);
    std::printf("   PARTIAL Pearson(ΔS, Δflux | Δedges, Δverts) = %+.3f\n", pearson(rS, rF));
    std::printf("   NOTE: the rules have near-degenerate (Δedges,Δverts) signatures, so this cross-rule\n"
                "   partial conflates size with rule and is NOT rigorous. The trustworthy size-controlled\n"
                "   measure is the idempotent rule alone (Δedges=Δverts=0 for every rewrite).\n\n");
}

// Evolve `rule` from `init` applying a seeded-random match each step, collecting
// the intermediate states (real rule-evolution snapshots, not synthetic).
static void collect_snapshots(const Graph& init, const Rule& rule, int steps, uint32_t seed,
                              int min_e, int max_e, const char* tag,
                              std::vector<std::pair<std::string, Graph>>& out) {
    std::mt19937 rng(seed);
    Graph cur = init;
    for (int t = 0; t < steps; ++t) {
        auto ms = find_matches(cur);
        if (ms.empty()) break;
        std::vector<int> pid; cur = apply_match(cur, ms[rng() % ms.size()], rule, pid);
        if ((int)cur.size() > max_e) break;
        if ((int)cur.size() >= min_e) out.push_back({std::string(tag) + "_t" + std::to_string(t), cur});
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("Higgs-shadow probe (hardened): exact |Aut|, larger graphs, size-controlled.\n\n");

    std::vector<std::pair<std::string, Graph>> synthetic = {
        {"cycle-8", gen_cycle(8)}, {"cycle-12", gen_cycle(12)}, {"cycle-16", gen_cycle(16)}, {"cycle-20", gen_cycle(20)},
        {"two-cyc-5", gen_kcycles(2,5)}, {"two-cyc-8", gen_kcycles(2,8)}, {"two-cyc-10", gen_kcycles(2,10)},
        {"three-cyc-5", gen_kcycles(3,5)}, {"four-cyc-4", gen_kcycles(4,4)},
        {"grid-3x3", gen_grid(3,3)}, {"grid-4x4", gen_grid(4,4)}, {"grid-3x6", gen_grid(3,6)}, {"grid-4x5", gen_grid(4,5)},
        {"path-12", gen_path(12)}, {"path-20", gen_path(20)},
        {"rand-14-18", gen_random(14,18,11)}, {"rand-18-24", gen_random(18,24,22)}, {"rand-22-28", gen_random(22,28,33)},
    };
    run_suite("SYNTHETIC graph families", synthetic);

    // Wolfram-snapshot states: evolve the productive rule from small seeds, keep
    // the intermediate states (the physically-realised, mostly-asymmetric ones).
    std::vector<std::pair<std::string, Graph>> snaps;
    Graph seedA = {{0, 1}, {1, 2}};
    Graph seedB = {{0, 1}, {1, 2}, {2, 0}};
    Graph seedC = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    for (uint32_t s = 1; s <= 3; ++s) collect_snapshots(seedA, R_prod, 12, s, 5, 24, ("A" + std::to_string(s)).c_str(), snaps);
    for (uint32_t s = 1; s <= 3; ++s) collect_snapshots(seedB, R_prod, 12, s, 5, 24, ("B" + std::to_string(s)).c_str(), snaps);
    for (uint32_t s = 1; s <= 2; ++s) collect_snapshots(seedC, R_prod, 12, s, 6, 24, ("C" + std::to_string(s)).c_str(), snaps);
    std::printf("(collected %zu Wolfram-snapshot states)\n", snaps.size());
    run_suite("WOLFRAM SNAPSHOTS (evolved states)", snaps);
    return 0;
}
