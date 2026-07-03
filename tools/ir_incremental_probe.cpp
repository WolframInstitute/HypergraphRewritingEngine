// Measure the amortized cost determinants of an *incremental* IR canonicalizer.
//
// Key reduction: a vertex's equitable-partition color == its WL stable color.
// So the perturbation set L = {shared vertices whose stable color changes
// parent->child} is the work an incremental refinement must redo (identical to
// incremental WL). IR's only extra cost over WL is the individualization search,
// which fires iff the refined partition is non-discrete. We therefore measure,
// per parent->child rewrite:
//   - L / n            (refinement perturbation locality)
//   - whether the child needed individualization (IR-hard?)
//   - IR refine_pops / search_nodes and wall time (cold canonicalization cost)
//
// Scenarios: a growing low-symmetry graph (typical evolution) vs symmetric
// parents (cycle) perturbed by one edge (the "one edge breaks it" worst case).

#include <hypergraph/ir_canonicalization.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

static inline uint64_t mix(uint64_t h, uint64_t x) {
    h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

// WL stable per-vertex colors: iso-invariant and comparable across graphs (same
// k-hop color = same value). Iterate color refinement to convergence.
static std::unordered_map<VertexId, uint64_t> wl_colors(const Edges& edges) {
    struct Occ { uint32_t e; uint8_t pos; uint8_t arity; };
    std::set<VertexId> vs;
    for (const auto& e : edges) for (VertexId v : e) vs.insert(v);
    std::unordered_map<VertexId, std::vector<Occ>> occ;
    for (uint32_t ei = 0; ei < edges.size(); ++ei) {
        uint8_t arity = static_cast<uint8_t>(edges[ei].size());
        for (uint8_t p = 0; p < arity; ++p) occ[edges[ei][p]].push_back({ei, p, arity});
    }
    std::unordered_map<VertexId, uint64_t> col;
    for (VertexId v : vs) {
        std::vector<uint32_t> sig;
        for (const auto& o : occ[v]) sig.push_back((uint32_t(o.arity) << 8) | o.pos);
        std::sort(sig.begin(), sig.end());
        uint64_t h = 1469598103934665603ULL;
        for (uint32_t s : sig) h = mix(h, s);
        col[v] = h;
    }
    auto distinct = [&]() { std::set<uint64_t> s; for (auto& kv : col) s.insert(kv.second); return s.size(); };
    size_t prev = distinct();
    for (size_t it = 0; it < vs.size() + 2; ++it) {
        std::unordered_map<VertexId, uint64_t> nc;
        for (VertexId v : vs) {
            std::vector<uint64_t> sig;
            for (const auto& o : occ[v]) {
                const auto& e = edges[o.e];
                std::vector<std::pair<uint8_t, uint64_t>> others;
                for (uint8_t p = 0; p < e.size(); ++p)
                    if (p != o.pos) others.push_back({p, col[e[p]]});
                std::sort(others.begin(), others.end());
                uint64_t h = mix(uint64_t(o.arity) << 8 | o.pos, 0xABCDEF);
                for (auto& pr : others) h = mix(mix(h, pr.first), pr.second);
                sig.push_back(h);
            }
            std::sort(sig.begin(), sig.end());
            uint64_t h = col[v];
            for (uint64_t s : sig) h = mix(h, s);
            nc[v] = h;
        }
        col.swap(nc);
        size_t d = distinct();
        if (d == prev) break;  // refinement stable
        prev = d;
    }
    return col;
}

// Perturbation set size: shared vertices whose stable color changed.
static size_t perturbation(const Edges& parent, const Edges& child) {
    auto cp = wl_colors(parent), cc = wl_colors(child);
    size_t L = 0;
    for (auto& kv : cp) {
        auto it = cc.find(kv.first);
        if (it != cc.end() && it->second != kv.second) ++L;
    }
    return L;
}

static size_t nverts(const Edges& e) {
    std::set<VertexId> s; for (auto& ed : e) for (VertexId v : ed) s.insert(v);
    return s.size();
}

static double cold_ir_us(const Edges& e, int iters) {
    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        IRCanonicalizer ir;
        auto a = std::chrono::high_resolution_clock::now();
        volatile uint64_t h = ir.compute_canonical_hash(e); (void)h;
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static Edges cycle(int n) {
    Edges e;
    for (int i = 0; i < n; ++i) e.push_back({(VertexId)i, (VertexId)((i + 1) % n)});
    return e;
}

int main() {
    // ---- Scenario 1: growing low-symmetry graph (typical evolution path) ----
    // Repeatedly attach a pendant edge to a random vertex (consume 0/produce 1
    // +1 vertex): a realistic local growth step.
    printf("=== GROWING graph (random local rewrites): perturbation L per step ===\n");
    printf("%6s %6s %6s %8s %6s %8s %10s %10s\n",
           "step", "n", "L", "L/n", "indiv?", "pops", "search", "IR_us");
    {
        std::mt19937 rng(12345);
        Edges cur = {{0, 1}, {1, 2}, {2, 0}};
        VertexId nextv = 3;
        double sumLfrac = 0; size_t maxL = 0; int indiv = 0; int steps = 60; int counted = 0;
        for (int s = 1; s <= steps; ++s) {
            Edges parent = cur;
            // pick a random existing vertex, attach a new pendant
            std::set<VertexId> vs; for (auto& e : parent) for (VertexId v : e) vs.insert(v);
            std::vector<VertexId> vv(vs.begin(), vs.end());
            VertexId a = vv[rng() % vv.size()];
            VertexId w = nextv++;
            cur.push_back({a, w});
            size_t n = nverts(cur);
            size_t L = perturbation(parent, cur);
            IRCanonicalizer ir; ir.compute_canonical_hash(cur);
            auto st = ir.last_stats();
            bool needs_indiv = !st.discrete_after_initial_refine;
            if (needs_indiv) ++indiv;
            if (s % 10 == 0 || s <= 3) {
                printf("%6d %6zu %6zu %8.3f %6s %8zu %10zu %10.2f\n", s, n, L,
                       (double)L / n, needs_indiv ? "yes" : "no",
                       st.refine_pops, st.search_nodes, cold_ir_us(cur, 200));
            }
            sumLfrac += (double)L / n; maxL = std::max(maxL, L); ++counted;
        }
        printf("  SUMMARY growing: mean L/n = %.4f, max L = %zu, individualization needed in %d/%d steps\n",
               sumLfrac / counted, maxL, indiv, counted);
    }

    // ---- Scenario 2: symmetric parent, one-edge perturbation (worst case) ----
    printf("\n=== SYMMETRIC parent (cycle C_n) + one pendant edge: perturbation L ===\n");
    printf("%6s %6s %6s %8s %6s %8s %10s %10s\n",
           "n", "n'", "L", "L/n", "indiv?", "pops", "search", "IR_us");
    for (int n : {20, 50, 100, 200}) {
        Edges parent = cycle(n);
        Edges child = cycle(n);
        child.push_back({(VertexId)0, (VertexId)n});  // pendant breaks vertex-transitivity
        size_t np = nverts(child);
        size_t L = perturbation(parent, child);
        IRCanonicalizer ir; ir.compute_canonical_hash(child);
        auto st = ir.last_stats();
        printf("%6d %6zu %6zu %8.3f %6s %8zu %10zu %10.2f\n", n, np, L, (double)L / np,
               st.discrete_after_initial_refine ? "no" : "yes",
               st.refine_pops, st.search_nodes, cold_ir_us(child, 100));
    }

    // ---- Scenario 3: the pure symmetric states themselves (no perturbation) ----
    printf("\n=== SYMMETRIC states (cycle C_n) cold: individualization cost ===\n");
    printf("%6s %6s %8s %10s %10s\n", "n", "indiv?", "pops", "search", "IR_us");
    for (int n : {20, 50, 100, 200}) {
        Edges e = cycle(n);
        IRCanonicalizer ir; ir.compute_canonical_hash(e);
        auto st = ir.last_stats();
        printf("%6d %6s %8zu %10zu %10.2f\n", n, st.discrete_after_initial_refine ? "no" : "yes",
               st.refine_pops, st.search_nodes, cold_ir_us(e, 100));
    }
    return 0;
}
