// tools/incremental_probe.cpp
//
// Standalone, header-free probe (NO project headers) for the question:
//   "How amenable to incrementalisation is each of our state-equivalencing
//    techniques — uniqueness trees (UT), Weisfeiler-Leman (WL), and
//    individualization-refinement (IR) — when the input changes by a single
//    local rewrite?"
//
// Build:
//   g++ -O2 -std=c++17 tools/incremental_probe.cpp -o /tmp/incremental_probe
//   /tmp/incremental_probe
//
// What it measures
// ----------------
// A single local rewrite touches a few edges. Whether incremental recomputation
// can beat a full recompute is governed by how far the rewrite's effect
// PROPAGATES through each technique's per-vertex value:
//
//   * WL  per-vertex stable colour  (commutative-combiner state hash)
//   * UT  per-vertex uniqueness-tree hash (Gorard trees, corrected for
//         edge multiplicity and self-loops)
//   * IR  refinement is WL-local; the canonical *labelling* is global, so IR's
//         incremental ceiling is reported separately (search cost, not locality)
//
// For WL and UT we compute every vertex's value on the pre-rewrite graph G and
// the post-rewrite graph G', and report the perturbation set
//   delta = { v : value_G(v) != value_G'(v) }  (+ added/removed vertices)
// as delta/n. Small delta/n => incrementalisation can pay; delta/n ~ 1 => a
// local rewrite forces near-global recomputation and incremental buys little.
//
// We sweep graph families spanning the symmetry spectrum (the canonicalisation
// hardness axis) and the three rewrite-rule shapes: productive (adds edges),
// reductive (removes edges), idempotent (rewires, edge count fixed).
//
// The hypergraph model: vertices are ints; an edge is an ORDERED list of
// vertices (directed hyperedge); a vertex may repeat within an edge (self-loop)
// and edges may be duplicated (multiplicity). All three techniques are exact on
// these features (that is the UT correction over Gorard's original).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Core types and hashing
// ============================================================================

using Vertex = int;
using Edge   = std::vector<Vertex>;   // ordered; repeats allowed (self-loops)
using Graph  = std::vector<Edge>;     // duplicates allowed (multiplicity)

static constexpr uint64_t FNV_OFFSET = 1469598103934665603ull;

// Order-sensitive 64-bit mix (boost::hash_combine style). Sort inputs first
// when a multiset/unordered combination is intended.
static inline uint64_t hmix(uint64_t seed, uint64_t v) {
    seed ^= v + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}
static inline uint64_t hlist(const std::vector<uint64_t>& xs) {
    uint64_t h = FNV_OFFSET;
    h = hmix(h, xs.size());
    for (uint64_t x : xs) h = hmix(h, x);
    return h;
}

// Per-vertex occurrence: vertex sits at `pos` in edge `edge` of given `arity`.
struct Occ { int edge; int pos; int arity; };

static std::vector<Vertex> vertices_of(const Graph& g) {
    std::set<Vertex> s;
    for (const auto& e : g) for (Vertex v : e) s.insert(v);
    return std::vector<Vertex>(s.begin(), s.end());
}

static std::unordered_map<Vertex, std::vector<Occ>> build_adj(const Graph& g) {
    std::unordered_map<Vertex, std::vector<Occ>> adj;
    for (int ei = 0; ei < (int)g.size(); ++ei) {
        const Edge& e = g[ei];
        int arity = (int)e.size();
        for (int p = 0; p < arity; ++p) adj[e[p]].push_back({ei, p, arity});
    }
    return adj;
}

// ============================================================================
// Weisfeiler-Leman (hypergraph 1-WL), commutative-combiner state hash
// ============================================================================
//
// A vertex's round-token over each incident occurrence encodes the edge's
// arity, the vertex's position, and the (position, neighbour-colour) of every
// other slot — so directed hyperedges, self-loops and multi-edges are all
// reflected. Tokens are combined as a sorted multiset (label-independent).

static uint64_t wl_init_color(const std::vector<Occ>& occ) {
    std::vector<uint64_t> toks;
    toks.reserve(occ.size());
    for (const Occ& o : occ)
        toks.push_back(hmix(hmix(FNV_OFFSET, (uint64_t)o.arity), (uint64_t)o.pos));
    std::sort(toks.begin(), toks.end());
    return hlist(toks);
}

// Run exactly `rounds` refinement rounds; return per-vertex colour.
static std::unordered_map<Vertex, uint64_t>
wl_colors(const Graph& g, int rounds, long long& ops) {
    auto adj = build_adj(g);
    std::unordered_map<Vertex, uint64_t> col;
    for (auto& kv : adj) col[kv.first] = wl_init_color(kv.second);

    for (int r = 0; r < rounds; ++r) {
        std::unordered_map<Vertex, uint64_t> next;
        next.reserve(col.size());
        for (auto& kv : adj) {
            Vertex v = kv.first;
            const std::vector<Occ>& occ = kv.second;
            std::vector<uint64_t> toks;
            toks.reserve(occ.size());
            for (const Occ& o : occ) {
                const Edge& e = g[o.edge];
                uint64_t t = hmix(FNV_OFFSET, (uint64_t)o.arity);
                t = hmix(t, (uint64_t)o.pos);
                std::vector<uint64_t> slots;
                for (int q = 0; q < o.arity; ++q) {
                    if (q == o.pos) continue;
                    slots.push_back(hmix((uint64_t)q, col[e[q]]));
                }
                std::sort(slots.begin(), slots.end());
                for (uint64_t s : slots) t = hmix(t, s);
                toks.push_back(t);
                ++ops;
            }
            std::sort(toks.begin(), toks.end());
            next[v] = hmix(col[v], hlist(toks));
        }
        col.swap(next);
    }
    return col;
}

// Number of distinct colours (partition size).
static size_t num_classes(const std::unordered_map<Vertex, uint64_t>& col) {
    std::set<uint64_t> s;
    for (auto& kv : col) s.insert(kv.second);
    return s.size();
}

// Rounds until the colour partition stops refining (WL fixpoint).
static int wl_fixpoint_rounds(const Graph& g) {
    long long ops = 0;
    int n = (int)vertices_of(g).size();
    size_t prev = 0;
    for (int r = 1; r <= n + 1; ++r) {
        auto col = wl_colors(g, r, ops);
        size_t c = num_classes(col);
        if (c == prev) return r;
        prev = c;
    }
    return n + 1;
}

// Commutative (sum) combine of fixpoint colours -> iso-invariant state hash.
static uint64_t wl_state_hash(const Graph& g, long long& ops) {
    int R = wl_fixpoint_rounds(g);
    auto col = wl_colors(g, R, ops);
    uint64_t sum = 0;
    for (auto& kv : col) sum += kv.second;       // order-independent
    uint64_t h = hmix(FNV_OFFSET, col.size());
    h = hmix(h, num_classes(col));
    h = hmix(h, sum);
    return h;
}

// ============================================================================
// Uniqueness Trees (Gorard), corrected for multiplicity and self-loops
// ============================================================================
//
// For each root, a DFS tree over the (unvisited) neighbourhood. A vertex's hash
// captures: its level, its own incident positions (sorted (arity<<8)|pos — this
// records degree, arity and self-loops), and the sorted hashes of its children.
// Neighbours reached by more than one occurrence are "non-unique" (multiplicity)
// and contribute a leaf hash carrying all their occurrence positions rather than
// recursing. State hash = sorted combine of all per-root tree hashes.

static constexpr int UT_MAX_DEPTH = 100;

// `visited` is path-scoped (cleared on backtrack), so a root's tree enumerates
// all simple paths from it — exponential on cyclic/dense graphs. That blow-up is
// itself a UT finding; cap the node count and report when hit.
static constexpr long long UT_NODE_CAP = 1500000;

static uint64_t ut_tree_hash(Vertex root, const Graph& g,
                             const std::unordered_map<Vertex, std::vector<Occ>>& adj,
                             long long& ops, bool& capped) {
    std::set<Vertex> visited;
    std::function<uint64_t(Vertex, int)> rec = [&](Vertex v, int level) -> uint64_t {
        ++ops;
        if (capped || ops > UT_NODE_CAP) { capped = true; return hmix(FNV_OFFSET, (uint64_t)level); }
        if (level >= UT_MAX_DEPTH) return hmix(FNV_OFFSET, (uint64_t)level);
        visited.insert(v);

        auto it = adj.find(v);
        const std::vector<Occ>& occ = (it == adj.end()) ? std::vector<Occ>{} : it->second;

        // Own positions: every slot v occupies across its incident edges.
        std::vector<uint64_t> own;
        own.reserve(occ.size());
        for (const Occ& o : occ) own.push_back((uint64_t)o.pos);
        std::sort(own.begin(), own.end());

        // Group unvisited neighbours by vertex id; record (arity<<8)|pos per occ.
        std::map<Vertex, std::vector<uint16_t>> nbr;
        for (const Occ& o : occ) {
            const Edge& e = g[o.edge];
            for (int q = 0; q < o.arity; ++q) {
                if (q == o.pos) continue;
                Vertex w = e[q];
                if (visited.count(w)) continue;
                nbr[w].push_back((uint16_t)(((uint16_t)o.arity << 8) | (uint16_t)q));
            }
        }

        std::vector<uint64_t> child_hashes;
        for (auto& kv : nbr) {
            Vertex w = kv.first;
            std::vector<uint16_t>& positions = kv.second;
            std::sort(positions.begin(), positions.end());
            bool unique = (positions.size() == 1);
            uint64_t ch;
            if (unique) {
                ch = rec(w, level + 1);             // recurse into the subtree
                ch = hmix(ch, 1);                   // is_unique
            } else {
                ch = hmix(FNV_OFFSET, (uint64_t)(level + 1));
                ch = hmix(ch, 0);                   // is_unique = false (multiplicity)
            }
            for (uint16_t p : positions) ch = hmix(ch, p);
            child_hashes.push_back(ch);
        }

        visited.erase(v);  // backtrack: tree per root, vertex reusable elsewhere

        std::sort(child_hashes.begin(), child_hashes.end());
        uint64_t h = hmix(FNV_OFFSET, (uint64_t)level);
        h = hmix(h, child_hashes.size());
        h = hmix(h, own.size());
        for (uint64_t p : own) h = hmix(h, p);
        for (uint64_t c : child_hashes) h = hmix(h, c);
        return h;
    };
    return rec(root, 0);
}

static std::unordered_map<Vertex, uint64_t> ut_vertex_hashes(const Graph& g, long long& ops, bool& capped) {
    auto adj = build_adj(g);
    std::unordered_map<Vertex, uint64_t> out;
    for (Vertex v : vertices_of(g)) {
        out[v] = ut_tree_hash(v, g, adj, ops, capped);
        if (capped) break;
    }
    return out;
}

static uint64_t ut_state_hash(const Graph& g, long long& ops, bool& capped) {
    auto vh = ut_vertex_hashes(g, ops, capped);
    std::vector<uint64_t> hs;
    hs.reserve(vh.size());
    for (auto& kv : vh) hs.push_back(kv.second);
    std::sort(hs.begin(), hs.end());
    return hlist(hs);
}

// ============================================================================
// Individualization-Refinement (McKay-lite), EXACT canonical hash
// ============================================================================
//
// Ordered partition refined to an equitable colouring; if not discrete,
// individualise each vertex of the first non-singleton cell, recurse, and keep
// the lexicographically least canonical form. The canonical form is the relabelled,
// sorted edge list (order within an edge preserved -> directed; duplicates and
// self-loops preserved). Exact (matches brute force); exponential worst case, so
// used only on small graphs here.

struct IRResult { uint64_t hash; long long refine_ops; long long search_nodes; bool discrete_after_initial; bool capped; };

// Bail the IR search after this many tree nodes. The probe's standalone IR has
// no automorphism/orbit pruning (the engine's IR does), so on maximally
// symmetric inputs the canonical-labelling search is super-exponential; hitting
// the cap is itself the finding — IR's labelling cost tracks symmetry.
static constexpr long long IR_NODE_CAP = 200000;

// Refine ordered partition `cells` (vectors of vertex ids) to equitable.
static void ir_refine(std::vector<std::vector<Vertex>>& cells, const Graph& g,
                      const std::unordered_map<Vertex, std::vector<Occ>>& adj,
                      long long& ops) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::unordered_map<Vertex, int> cell_of;
        for (int ci = 0; ci < (int)cells.size(); ++ci)
            for (Vertex v : cells[ci]) cell_of[v] = ci;

        std::vector<std::vector<Vertex>> next;
        for (auto& cell : cells) {
            if (cell.size() <= 1) { next.push_back(cell); continue; }
            // Signature of each vertex = sorted multiset over incident occurrences
            // of (arity, pos, [(q, cell_of(slot))]).
            std::vector<std::pair<uint64_t, Vertex>> sig;
            for (Vertex v : cell) {
                ++ops;
                std::vector<uint64_t> toks;
                auto it = adj.find(v);
                if (it != adj.end()) for (const Occ& o : it->second) {
                    const Edge& e = g[o.edge];
                    uint64_t t = hmix(hmix(FNV_OFFSET, (uint64_t)o.arity), (uint64_t)o.pos);
                    std::vector<uint64_t> slots;
                    for (int q = 0; q < o.arity; ++q) {
                        if (q == o.pos) continue;
                        slots.push_back(hmix((uint64_t)q, (uint64_t)cell_of[e[q]]));
                    }
                    std::sort(slots.begin(), slots.end());
                    for (uint64_t s : slots) t = hmix(t, s);
                    toks.push_back(t);
                }
                std::sort(toks.begin(), toks.end());
                sig.push_back({hlist(toks), v});
            }
            std::sort(sig.begin(), sig.end());
            // Split into sub-cells by equal signature, preserving signature order.
            size_t i = 0;
            std::vector<std::vector<Vertex>> sub;
            while (i < sig.size()) {
                size_t j = i;
                std::vector<Vertex> piece;
                while (j < sig.size() && sig[j].first == sig[i].first) { piece.push_back(sig[j].second); ++j; }
                sub.push_back(std::move(piece));
                i = j;
            }
            if (sub.size() > 1) changed = true;
            for (auto& s : sub) next.push_back(std::move(s));
        }
        cells.swap(next);
    }
}

static uint64_t ir_canonical_form_hash(const std::vector<Vertex>& labeling_order, const Graph& g) {
    // labeling_order[i] = original vertex placed at canonical index i.
    std::unordered_map<Vertex, int> lab;
    for (int i = 0; i < (int)labeling_order.size(); ++i) lab[labeling_order[i]] = i;
    std::vector<std::vector<int>> edges;
    edges.reserve(g.size());
    for (const Edge& e : g) {
        std::vector<int> re;
        re.reserve(e.size());
        for (Vertex v : e) re.push_back(lab[v]);   // order preserved (directed)
        edges.push_back(std::move(re));
    }
    std::sort(edges.begin(), edges.end());          // multiset of edges
    uint64_t h = hmix(FNV_OFFSET, edges.size());
    for (auto& e : edges) { h = hmix(h, e.size()); for (int x : e) h = hmix(h, (uint64_t)x); }
    return h;
}

static void ir_search(std::vector<std::vector<Vertex>> cells, const Graph& g,
                      const std::unordered_map<Vertex, std::vector<Occ>>& adj,
                      bool& have_best, uint64_t& best, long long& refine_ops, long long& nodes,
                      bool& capped) {
    if (capped) return;
    if (nodes > IR_NODE_CAP) { capped = true; return; }
    ++nodes;
    ir_refine(cells, g, adj, refine_ops);
    // discrete?
    size_t target = (size_t)-1;
    for (size_t i = 0; i < cells.size(); ++i) if (cells[i].size() > 1) { target = i; break; }
    if (target == (size_t)-1) {
        std::vector<Vertex> order;
        for (auto& c : cells) order.push_back(c[0]);
        uint64_t h = ir_canonical_form_hash(order, g);
        if (!have_best || h < best) { best = h; have_best = true; }
        return;
    }
    // Individualise each vertex of the target cell in turn.
    std::vector<Vertex> cellv = cells[target];
    for (Vertex v : cellv) {
        std::vector<std::vector<Vertex>> nc;
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i != target) { nc.push_back(cells[i]); continue; }
            nc.push_back({v});
            std::vector<Vertex> rest;
            for (Vertex w : cells[i]) if (w != v) rest.push_back(w);
            if (!rest.empty()) nc.push_back(std::move(rest));
        }
        ir_search(std::move(nc), g, adj, have_best, best, refine_ops, nodes, capped);
        if (capped) return;
    }
}

static IRResult ir_canonical(const Graph& g) {
    IRResult r{0, 0, 0, false, false};
    auto adj = build_adj(g);
    std::vector<Vertex> verts = vertices_of(g);
    if (verts.empty()) return r;
    // Initial partition by WL-style init colour, cells ordered by colour value.
    std::map<uint64_t, std::vector<Vertex>> by_color;
    for (Vertex v : verts) {
        auto it = adj.find(v);
        by_color[wl_init_color(it->second)].push_back(v);
    }
    std::vector<std::vector<Vertex>> cells;
    for (auto& kv : by_color) cells.push_back(kv.second);

    // Probe: is it discrete right after one refine (no individualisation)?
    {
        auto probe = cells;
        long long tmp = 0;
        ir_refine(probe, g, adj, tmp);
        bool disc = true;
        for (auto& c : probe) if (c.size() > 1) { disc = false; break; }
        r.discrete_after_initial = disc;
    }

    bool have_best = false;
    uint64_t best = 0;
    ir_search(cells, g, adj, have_best, best, r.refine_ops, r.search_nodes, r.capped);
    r.hash = best;
    return r;
}

// ============================================================================
// Graph generators across the symmetry spectrum
// ============================================================================

// symmetry_groups distinct edge templates, each repeated -> controlled symmetry.
static Graph gen_symmetric(int num_edges, int symmetry_groups, int arity, uint32_t seed) {
    std::mt19937 rng(seed);
    symmetry_groups = std::max(1, std::min(symmetry_groups, num_edges));
    std::vector<int> copies(symmetry_groups, num_edges / symmetry_groups);
    for (int i = 0; i < num_edges % symmetry_groups; ++i) copies[i]++;
    int total = symmetry_groups * arity;
    std::vector<Vertex> pool;
    for (int i = 1; i <= total; ++i) pool.push_back(i);
    std::shuffle(pool.begin(), pool.end(), rng);
    Graph g; g.reserve(num_edges);
    int idx = 0;
    for (int grp = 0; grp < symmetry_groups; ++grp) {
        Edge tmpl;
        for (int j = 0; j < arity; ++j) tmpl.push_back(pool[idx++]);
        for (int c = 0; c < copies[grp]; ++c) g.push_back(tmpl);
    }
    return g;
}

static Graph gen_cycle(int n) { Graph g; for (int i = 0; i < n; ++i) g.push_back({i, (i + 1) % n}); return g; }
static Graph gen_path(int n)  { Graph g; for (int i = 0; i + 1 < n; ++i) g.push_back({i, i + 1}); return g; }
static Graph gen_star(int n)  { Graph g; for (int i = 1; i < n; ++i) g.push_back({0, i}); return g; }
static Graph gen_complete(int n) { Graph g; for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) if (i != j) g.push_back({i, j}); return g; }
static Graph gen_disjoint_cycles(int k, int len) {
    Graph g; for (int c = 0; c < k; ++c) for (int i = 0; i < len; ++i) g.push_back({c * len + i, c * len + (i + 1) % len}); return g;
}
static Graph gen_grid(int w, int h) {
    Graph g; auto id = [&](int x, int y) { return y * w + x; };
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        if (x + 1 < w) g.push_back({id(x, y), id(x + 1, y)});
        if (y + 1 < h) g.push_back({id(x, y), id(x, y + 1)});
    }
    return g;
}
static Graph gen_random_sparse(int n, int m, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> vd(0, n - 1);
    Graph g; for (int i = 0; i < m; ++i) { int a = vd(rng), b = vd(rng); g.push_back({a, b}); }
    return g;
}

// ============================================================================
// Single local rewrites
// ============================================================================

static int max_vertex(const Graph& g) { int mx = -1; for (auto& e : g) for (Vertex v : e) mx = std::max(mx, v); return mx; }

// Productive: introduce a fresh vertex and two edges around a random edge.
static Graph rewrite_productive(const Graph& g, uint32_t seed) {
    if (g.empty()) return g;
    std::mt19937 rng(seed);
    Graph h = g;
    int w = max_vertex(g) + 1;
    const Edge& e = g[rng() % g.size()];
    Vertex a = e.front(), b = e.back();
    h.push_back({a, w});
    h.push_back({w, b});
    return h;
}

// Reductive: delete a random edge.
static Graph rewrite_reductive(const Graph& g, uint32_t seed) {
    if (g.empty()) return g;
    std::mt19937 rng(seed);
    Graph h = g;
    h.erase(h.begin() + (rng() % h.size()));
    return h;
}

// Idempotent: 2-swap endpoints of two distinct edges (edge count and vertex set
// preserved; structure rewired).
static Graph rewrite_idempotent(const Graph& g, uint32_t seed) {
    if (g.size() < 2) return g;
    std::mt19937 rng(seed);
    Graph h = g;
    int i = rng() % h.size(), j = rng() % h.size();
    int guard = 0;
    while ((j == i || h[i].size() < 2 || h[j].size() < 2) && guard++ < 32) { i = rng() % h.size(); j = rng() % h.size(); }
    if (i == j || h[i].size() < 2 || h[j].size() < 2) return h;
    std::swap(h[i].back(), h[j].back());
    return h;
}

// ============================================================================
// Perturbation (delta/n) measurement
// ============================================================================

struct Delta { int n_before; int n_after; int changed; int added; int removed; };

static Delta delta_of(const std::unordered_map<Vertex, uint64_t>& a,
                      const std::unordered_map<Vertex, uint64_t>& b) {
    Delta d{(int)a.size(), (int)b.size(), 0, 0, 0};
    for (auto& kv : a) {
        auto it = b.find(kv.first);
        if (it == b.end()) d.removed++;
        else if (it->second != kv.second) d.changed++;
    }
    for (auto& kv : b) if (!a.count(kv.first)) d.added++;
    return d;
}

static double frac(const Delta& d) {
    int denom = std::max(d.n_before, d.n_after);
    int affected = d.changed + d.added + d.removed;
    return denom ? (double)affected / denom : 0.0;
}

// Measure WL and UT perturbation for one (graph, rewrite).
static void measure_locality(const char* name, const Graph& g, const Graph& gp) {
    long long ops = 0;
    int R = std::max(wl_fixpoint_rounds(g), wl_fixpoint_rounds(gp));
    auto wl_g  = wl_colors(g,  R, ops);
    auto wl_gp = wl_colors(gp, R, ops);
    Delta dw = delta_of(wl_g, wl_gp);

    bool cap_g = false, cap_gp = false;
    auto ut_g  = ut_vertex_hashes(g,  ops, cap_g);
    auto ut_gp = ut_vertex_hashes(gp, ops, cap_gp);

    if (cap_g || cap_gp) {
        std::printf("  %-26s n=%-4d WL delta/n=%.3f (chg %d add %d rem %d)   UT delta/n=path-exponential (capped)\n",
                    name, dw.n_before, frac(dw), dw.changed, dw.added, dw.removed);
        return;
    }
    Delta du = delta_of(ut_g, ut_gp);
    std::printf("  %-26s n=%-4d WL delta/n=%.3f (chg %d add %d rem %d)   UT delta/n=%.3f (chg %d add %d rem %d)\n",
                name, dw.n_before, frac(dw), dw.changed, dw.added, dw.removed,
                frac(du), du.changed, du.added, du.removed);
}

// ============================================================================
// Self-test: iso-invariance + a WL-incompleteness witness
// ============================================================================

static Graph relabel(const Graph& g, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<Vertex> vs = vertices_of(g);
    std::vector<Vertex> perm = vs;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::unordered_map<Vertex, Vertex> m;
    for (size_t i = 0; i < vs.size(); ++i) m[vs[i]] = perm[i];
    Graph h; for (const Edge& e : g) { Edge ne; for (Vertex v : e) ne.push_back(m[v]); h.push_back(ne); }
    std::shuffle(h.begin(), h.end(), rng);   // also permute edge order
    return h;
}

static void self_test() {
    std::printf("== self-test: isomorphism invariance (relabel must not change hash) ==\n");
    std::vector<std::pair<const char*, Graph>> cases = {
        {"cycle6",          gen_cycle(6)},
        {"complete4",       gen_complete(4)},
        {"grid3x3",         gen_grid(3, 3)},
        {"two-triangles",   gen_disjoint_cycles(2, 3)},
        {"self-loop+multi", Graph{{0,0},{0,1},{0,1},{1,2}}},
    };
    int fails = 0;
    for (auto& c : cases) {
        long long o = 0; bool cap = false;
        Graph r = relabel(c.second, 12345);
        bool wl = wl_state_hash(c.second, o) == wl_state_hash(r, o);
        bool ut = ut_state_hash(c.second, o, cap) == ut_state_hash(r, o, cap);
        bool ir = ir_canonical(c.second).hash == ir_canonical(r).hash;
        if (!(wl && ut && ir)) ++fails;
        std::printf("  %-16s WL %s  UT %s  IR %s\n", c.first,
                    wl ? "ok" : "FAIL", ut ? "ok" : "FAIL", ir ? "ok" : "FAIL");
    }
    // WL-incompleteness witness: C6 vs two triangles are non-isomorphic but
    // 1-WL-equivalent. IR (exact) must separate them; UT trees should too.
    {
        Graph c6 = gen_cycle(6), tt = gen_disjoint_cycles(2, 3);
        long long o = 0; bool cap = false;
        bool wl_sep = wl_state_hash(c6, o) != wl_state_hash(tt, o);
        bool ut_sep = ut_state_hash(c6, o, cap) != ut_state_hash(tt, o, cap);
        bool ir_sep = ir_canonical(c6).hash != ir_canonical(tt).hash;
        std::printf("  C6 vs 2xC3 (1-WL-hard): WL separates=%s  UT separates=%s  IR separates=%s\n",
                    wl_sep ? "yes" : "no", ut_sep ? "yes" : "no", ir_sep ? "yes" : "no");
    }
    std::printf("  self-test %s\n\n", fails == 0 ? "PASSED" : "FAILED");
}

// ============================================================================
// Cost-by-symmetry sweep (reproduces the old "by symmetry" benchmark)
// ============================================================================

template <typename F>
static double time_ms(F&& f) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void cost_by_symmetry() {
    std::printf("== full-recompute cost vs symmetry (24 edges, arity 2) ==\n");
    std::printf("  %-8s %-8s | %-12s %-12s | %-12s %-12s | %-12s %-10s\n",
                "symGrp", "classes", "WL ms", "WL ops", "UT ms", "UT ops", "IR ms", "IR nodes");
    const int num_edges = 24, arity = 2;
    for (int sg : {1, 2, 3, 4, 6, 8, 12, 24}) {
        Graph g = gen_symmetric(num_edges, sg, arity, 0xC0FFEE ^ sg);
        long long wo = 0, uo = 0; bool ut_cap = false;
        double wl_ms = time_ms([&] { wl_state_hash(g, wo); });
        double ut_ms = time_ms([&] { ut_state_hash(g, uo, ut_cap); });
        IRResult ir{}; double ir_ms = time_ms([&] { ir = ir_canonical(g); });
        long long c = 0; { long long o = 0; c = (long long)num_classes(wl_colors(g, wl_fixpoint_rounds(g), o)); }
        char utops[32], irnodes[32];
        std::snprintf(utops, sizeof utops, "%lld%s", uo, ut_cap ? "+CAP" : "");
        std::snprintf(irnodes, sizeof irnodes, "%lld%s", ir.search_nodes, ir.capped ? "+CAP" : "");
        std::printf("  %-8d %-8lld | %-12.3f %-12lld | %-12.3f %-12s | %-12.3f %-10s\n",
                    sg, c, wl_ms, wo, ut_ms, utops, ir_ms, irnodes);
    }
    std::printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: progress visible, survives timeout
    self_test();
    cost_by_symmetry();

    std::printf("== incrementalisation locality: per-vertex perturbation from ONE rewrite ==\n");
    std::printf("   (delta/n near 0 => incremental can pay; near 1 => a local rewrite forces global recompute)\n\n");

    struct Fam { const char* name; Graph g; };
    std::vector<Fam> fams = {
        {"path-40",            gen_path(40)},
        {"cycle-40",           gen_cycle(40)},
        {"two-cycles-20",      gen_disjoint_cycles(2, 20)},
        {"star-40",            gen_star(40)},
        {"grid-7x7",           gen_grid(7, 7)},
        {"complete-8",         gen_complete(8)},
        {"random-sparse-40e",  gen_random_sparse(40, 40, 0xABCDEF)},
        {"sym24-grp4",         gen_symmetric(24, 4, 2, 0x1234)},
        {"sym24-grp12",        gen_symmetric(24, 12, 2, 0x5678)},
    };

    const char* rule_names[3] = {"productive", "reductive", "idempotent"};
    for (auto& f : fams) {
        std::printf(" %s\n", f.name);
        for (int rt = 0; rt < 3; ++rt) {
            Graph gp = (rt == 0) ? rewrite_productive(f.g, 7)
                     : (rt == 1) ? rewrite_reductive(f.g, 7)
                                 : rewrite_idempotent(f.g, 7);
            measure_locality(rule_names[rt], f.g, gp);
        }
    }
    std::printf("\nNote: WL/UT report per-vertex perturbation (an Ω lower bound on incremental work).\n"
                "IR refinement shares WL's locality, but IR's canonical *labelling* is global —\n"
                "its incremental ceiling is the search cost in the cost-by-symmetry table, not delta/n.\n");
    return 0;
}
