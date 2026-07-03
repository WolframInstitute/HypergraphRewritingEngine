// tools/incremental_wl_probe.cpp
//
// Standalone, header-free: a REALIZED incremental WL hash (dirty-frontier
// propagation), VALIDATED bit-identical to full WL, with the actual work saving
// measured and the pay-off regime pinned to a concrete graph property.
//
// Build:
//   g++ -O2 -std=c++17 tools/incremental_wl_probe.cpp -o /tmp/incremental_wl_probe
//   /tmp/incremental_wl_probe
//
// Why this can be correct AND sublinear
// -------------------------------------
// With CONTENT-HASH colours (no per-round renormalisation), a vertex's round-t
// colour is a pure function of its t-ball. After a single rewrite, a vertex whose
// t-ball does not reach the changed edges has the SAME round-t colour in the
// child as in the parent — so it can be REUSED verbatim, and "this vertex's
// colour changed" is exactly comparable to the parent's history. The dirty set
// therefore expands one hop per round from the rewrite; vertices it never reaches
// are never recomputed.
//
// Algorithm (incremental child WL, given parent's per-round colour history):
//   round 0: recompute init colour only for vertices incident to changed edges
//            and for new vertices; everyone else reuses the parent's init colour.
//   round t: recompute a vertex iff it, or a neighbour, CHANGED (differs from the
//            parent's round-(t-1) colour); others reuse the parent's round-t
//            colour. A recomputed vertex that now differs from the parent marks
//            its neighbours dirty for round t+1.
//   Work = number of per-vertex signature recomputations (the dominant cost).
//
// Every case is validated: incremental child colours (at the fixpoint round) must
// equal full-WL child colours exactly. Speedup is reported ONLY over validated
// cases.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
// one vertex's round update from a colour-lookup callable
template <typename ColorOf>
static uint64_t wl_sig(Vertex v, const std::vector<Occ>& occ, const Graph& g, ColorOf&& colorOf) {
    std::vector<uint64_t> toks;
    for (const Occ& o : occ) {
        const Edge& e = g[o.edge]; uint64_t t = hmix(hmix(FNV_OFFSET, (uint64_t)o.arity), (uint64_t)o.pos);
        std::vector<uint64_t> sl; for (int q = 0; q < o.arity; ++q) if (q != o.pos) sl.push_back(hmix((uint64_t)q, colorOf(e[q])));
        std::sort(sl.begin(), sl.end()); for (uint64_t s : sl) t = hmix(t, s); toks.push_back(t);
    }
    std::sort(toks.begin(), toks.end()); return hmix(colorOf(v), hlist(toks));
}
using ColMap = std::unordered_map<Vertex, uint64_t>;

// full WL history rounds 0..R; counts per-vertex signature computations.
static std::vector<ColMap> wl_history(const Graph& g, int R, long long& work) {
    auto adj = build_adj(g);
    std::vector<ColMap> H; ColMap col;
    for (auto& kv : adj) { col[kv.first] = wl_init(kv.second); ++work; }
    H.push_back(col);
    for (int r = 1; r <= R; ++r) {
        ColMap nx;
        for (auto& kv : adj) { nx[kv.first] = wl_sig(kv.first, kv.second, g, [&](Vertex u){ return col[u]; }); ++work; }
        col.swap(nx); H.push_back(col);
    }
    return H;
}
static size_t ndistinct(const ColMap& c) { std::set<uint64_t> s; for (auto& kv : c) s.insert(kv.second); return s.size(); }
static int wl_fixpoint_rounds(const Graph& g) {
    int n = (int)vertices_of(g).size(); size_t prev = 0; long long w = 0;
    for (int r = 1; r <= n + 1; ++r) { auto H = wl_history(g, r, w); size_t cc = ndistinct(H.back()); if (cc == prev) return r; prev = cc; }
    return n + 1;
}
static size_t max_class(const ColMap& c) { std::unordered_map<uint64_t,int> m; for (auto& kv : c) m[kv.second]++; size_t mx = 0; for (auto& kv : m) mx = std::max(mx, (size_t)kv.second); return mx; }

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
// returns child and the set of vertices incident to consumed-or-produced edges (the delta seed)
static Graph apply_match(const Graph& A, const Match& m, const Rule& r, std::set<Vertex>& delta) {
    delta.clear();
    for (Vertex v : A[m.i]) delta.insert(v);
    for (Vertex v : A[m.j]) delta.insert(v);
    Graph child; for (int e = 0; e < (int)A.size(); ++e) if (e != m.i && e != m.j) child.push_back(A[e]);
    int base = max_vertex(A) + 1;
    for (auto& t : r.rhs) { Edge e; for (int s : t) { Vertex vv = s == 0 ? m.x : s == 1 ? m.y : s == 2 ? m.z : base + (s - 100); e.push_back(vv); delta.insert(vv); } child.push_back(e); }
    return child;
}

// ---- the realized incremental WL ----
struct IncrResult { long long work; bool valid; };
static IncrResult incremental_wl(const std::vector<ColMap>& ph,
                                 const Graph& child, const std::vector<ColMap>& cf, int R,
                                 const std::set<Vertex>& delta) {
    auto cadj = build_adj(child);
    std::set<Vertex> pverts; for (auto& kv : ph[0]) pverts.insert(kv.first);
    long long work = 0;
    // overlay[v] = child colour at the CURRENT round for vertices that differ from
    // parent (or are new); others are read from parent history ph[r].
    ColMap overlay;                 // round r-1 overlay
    std::unordered_set<Vertex> changed;   // vertices whose round r-1 colour != parent (or new)
    auto colPrev = [&](int rm1, Vertex u) -> uint64_t {
        auto it = overlay.find(u); if (it != overlay.end()) return it->second;
        return ph[rm1].at(u);       // unaffected => reuse parent (exact for content-hash WL)
    };
    // round 0
    for (Vertex v : delta) { overlay[v] = wl_init(cadj[v]); ++work; }
    for (auto& kv : cadj) if (!pverts.count(kv.first) && !overlay.count(kv.first)) { overlay[kv.first] = wl_init(kv.second); ++work; } // new verts
    for (auto& kv : overlay) { Vertex v = kv.first; bool isnew = !pverts.count(v); if (isnew || kv.second != ph[0].at(v)) changed.insert(v); }

    for (int r = 1; r <= R; ++r) {
        // recompute set = changed ∪ neighbours(changed)
        std::unordered_set<Vertex> recompute = changed;
        for (Vertex v : changed) for (const Occ& o : cadj[v]) for (Vertex w : child[o.edge]) recompute.insert(w);
        ColMap noverlay; std::unordered_set<Vertex> nchanged;
        for (Vertex v : recompute) {
            uint64_t val = wl_sig(v, cadj[v], child, [&](Vertex u){ return colPrev(r - 1, u); }); ++work;
            noverlay[v] = val;
            bool isnew = !pverts.count(v);
            if (isnew || val != ph[r].at(v)) nchanged.insert(v);
        }
        overlay.swap(noverlay); changed.swap(nchanged);
    }
    // validate: child colour at round R (overlay or parent) == full WL child colour
    bool ok = true;
    for (auto& kv : cf[R]) {
        Vertex v = kv.first;
        uint64_t mine = overlay.count(v) ? overlay[v] : (pverts.count(v) ? ph[R].at(v) : 0);
        if (!overlay.count(v) && !pverts.count(v)) { ok = false; break; }   // new vertex must be in overlay
        if (mine != kv.second) { ok = false; break; }
    }
    return {work, ok};
}

// ---- generators + snapshots ----
static Graph gen_cycle(int n) { Graph g; for (int i = 0; i < n; ++i) g.push_back({i, (i + 1) % n}); return g; }
static Graph gen_path(int n) { Graph g; for (int i = 0; i + 1 < n; ++i) g.push_back({i, i + 1}); return g; }
static Graph gen_kcycles(int k, int len) { Graph g; for (int c = 0; c < k; ++c) for (int i = 0; i < len; ++i) g.push_back({c*len+i, c*len+(i+1)%len}); return g; }
static Graph gen_grid(int w, int h) { Graph g; auto id=[&](int x,int y){return y*w+x;}; for (int y=0;y<h;++y) for (int x=0;x<w;++x){ if(x+1<w) g.push_back({id(x,y),id(x+1,y)}); if(y+1<h) g.push_back({id(x,y),id(x,y+1)});} return g; }
static Graph gen_random(int n, int m, uint32_t seed) { std::mt19937 rng(seed); std::uniform_int_distribution<int> vd(0,n-1); Graph g; for(int i=0;i<m;++i) g.push_back({vd(rng),vd(rng)}); return g; }
static const Rule R_prod{"productive", {{0, 1}, {1, 100}, {100, 2}}};
static void collect_snapshots(const Graph& init, int steps, uint32_t seed, int min_e, int max_e, const char* tag, std::vector<std::pair<std::string,Graph>>& out) {
    std::mt19937 rng(seed); Graph cur = init;
    for (int t = 0; t < steps; ++t) { auto ms = find_matches(cur); if (ms.empty()) break; std::set<Vertex> d; cur = apply_match(cur, ms[rng()%ms.size()], R_prod, d); if ((int)cur.size() > max_e) break; if ((int)cur.size() >= min_e) out.push_back({std::string(tag)+"_t"+std::to_string(t), cur}); }
}

static double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    int n = (int)x.size(); if (n < 2) return 0; double mx=0,my=0; for(int i=0;i<n;++i){mx+=x[i];my+=y[i];} mx/=n;my/=n;
    double sxy=0,sxx=0,syy=0; for(int i=0;i<n;++i){double dx=x[i]-mx,dy=y[i]-my; sxy+=dx*dy;sxx+=dx*dx;syy+=dy*dy;}
    return (sxx<=0||syy<=0)?0.0:sxy/std::sqrt(sxx*syy);
}
static double median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(),v.end()); return v[v.size()/2]; }

struct Row { double speedup, deltafrac, maxclassfrac, n; };

static void run_suite(const char* label, const std::vector<std::pair<std::string,Graph>>& states, const Rule& rule) {
    int cases = 0, valid = 0;
    std::vector<Row> rows;
    for (auto& s : states) {
        const Graph& A = s.second;
        auto ms = find_matches(A);
        int stride = std::max(1, (int)ms.size() / 12);
        for (int mi = 0; mi < (int)ms.size(); mi += stride) {
            std::set<Vertex> delta; Graph B = apply_match(A, ms[mi], rule, delta);
            if (B.empty()) continue;
            int R = wl_fixpoint_rounds(B);
            long long wp = 0, wfull = 0;
            auto ph = wl_history(A, R, wp);          // parent history (amortised precompute; not charged)
            auto cf = wl_history(B, R, wfull);       // full child WL = the baseline cost
            IncrResult ir = incremental_wl(ph, B, cf, R, delta);
            ++cases; if (!ir.valid) continue; ++valid;
            // affected fraction at fixpoint, and symmetry concentration of the child
            int nB = (int)cf[R].size(); int aff = 0;
            std::set<Vertex> pv; for (auto& kv : ph[0]) pv.insert(kv.first);
            for (auto& kv : cf[R]) { Vertex v = kv.first; bool isnew = !pv.count(v); if (isnew || kv.second != ph[R].at(v)) ++aff; }
            Row row;
            row.speedup = ir.work > 0 ? (double)wfull / ir.work : 0;
            row.deltafrac = nB ? (double)aff / nB : 0;
            row.maxclassfrac = nB ? (double)max_class(cf[R]) / nB : 0;
            row.n = nB;
            rows.push_back(row);
        }
    }
    std::printf("== %s (rule: %s) ==\n", label, rule.name.c_str());
    std::printf("   cases=%d  validated=%d (%.0f%%)\n", cases, valid, cases ? 100.0*valid/cases : 0);
    std::vector<double> sp, df, mc;
    for (auto& r : rows) { sp.push_back(r.speedup); df.push_back(r.deltafrac); mc.push_back(r.maxclassfrac); }
    if (!sp.empty()) {
        std::printf("   work-speedup (full sigs / incremental sigs): median=%.2fx  mean=%.2fx  min=%.2fx  max=%.2fx\n",
                    median(sp), [&]{double s=0;for(double v:sp)s+=v;return s/sp.size();}(),
                    *std::min_element(sp.begin(),sp.end()), *std::max_element(sp.begin(),sp.end()));
        std::printf("   mean affected-fraction δ/n = %.3f   mean max-class-fraction = %.3f\n",
                    [&]{double s=0;for(double v:df)s+=v;return s/df.size();}(), [&]{double s=0;for(double v:mc)s+=v;return s/mc.size();}());
        std::printf("   regime predictors:  Pearson(speedup, δ/n) = %+.3f   Pearson(speedup, max-class-frac) = %+.3f\n",
                    pearson(sp, df), pearson(sp, mc));
        // bucket speedup by max-class-fraction (the symmetry concentration)
        std::printf("   max-class-frac bucket   n     median speedup   mean δ/n\n");
        double lo[4]={0.0,0.10,0.25,0.50}, hi[4]={0.10,0.25,0.50,1.01};
        for (int b=0;b<4;++b){ std::vector<double> bs,bd; for(auto&r:rows) if(r.maxclassfrac>=lo[b]&&r.maxclassfrac<hi[b]){bs.push_back(r.speedup);bd.push_back(r.deltafrac);}
            if(!bs.empty()) std::printf("   [%.2f,%.2f)             %-5zu %-15.2f %.3f\n", lo[b],hi[b],bs.size(),median(bs),[&]{double s=0;for(double v:bd)s+=v;return s/bd.size();}()); }
    }
    std::printf("\n");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("Realized incremental WL: validated bit-identical to full WL; work saving + regime.\n\n");

    std::vector<std::pair<std::string, Graph>> synthetic = {
        {"path-20", gen_path(20)}, {"path-40", gen_path(40)},
        {"cycle-16", gen_cycle(16)}, {"cycle-24", gen_cycle(24)},
        {"two-cyc-8", gen_kcycles(2,8)}, {"two-cyc-12", gen_kcycles(2,12)},
        {"grid-4x4", gen_grid(4,4)}, {"grid-5x5", gen_grid(5,5)}, {"grid-3x8", gen_grid(3,8)},
        {"rand-20-26", gen_random(20,26,7)}, {"rand-30-40", gen_random(30,40,8)}, {"rand-40-55", gen_random(40,55,9)},
    };
    run_suite("SYNTHETIC", synthetic, R_prod);

    std::vector<std::pair<std::string, Graph>> snaps;
    Graph seedA = {{0,1},{1,2}}, seedB = {{0,1},{1,2},{2,3},{3,0}};
    for (uint32_t s = 1; s <= 5; ++s) collect_snapshots(seedA, 16, s, 8, 40, ("A"+std::to_string(s)).c_str(), snaps);
    for (uint32_t s = 1; s <= 5; ++s) collect_snapshots(seedB, 16, s, 8, 40, ("B"+std::to_string(s)).c_str(), snaps);
    std::printf("(collected %zu Wolfram-snapshot states)\n", snaps.size());
    run_suite("WOLFRAM SNAPSHOTS", snaps, R_prod);

    std::printf("Reading: validated=100%% means the incremental hash is provably equal to full WL on every case.\n"
                "Speedup is full-WL signature computations / incremental ones. The regime: speedup is high when\n"
                "max-class-fraction (symmetry concentration) is LOW (graph near WL-discrete) and δ/n is small;\n"
                "it collapses toward 1x when a large WL colour class exists (a local change re-splits it globally).\n");
    return 0;
}
