// representation_prototype.cpp -- three match-set representations, same evolution, measured.
//
// THE QUESTION. A match is valid in every state whose edge set contains its matched edges
// (the fiber identity: Hom_M(L,S) = { m in Hom_M(L,G_inf) : edges(m) subset of S }). The engine
// MATERIALISES that fiber, one record per (state, match) pair, and tools/unfolding_ratio_probe
// measures 11-50 pairs per match, growing with problem size. Whether a representation that does
// not materialise it wins is a question about SPEED and MEMORY, and it is answered here rather
// than argued.
//
// THREE ARMS, one evolution, identical output required:
//
//   A  MATERIALISED   per state, a flat array of match ids. This is the current design in its
//                     BEST possible form -- 4 bytes per pair, no hash map, no list nodes, no
//                     dedup (single-threaded, so the push/pull race that forces the map does not
//                     arise). If B or C wins against this, it wins against the real thing.
//
//   B  MEMBERSHIP     per state, a copy-on-write chunked bitset over the global match pool.
//                     Inheritance is a chunk-pointer copy; the delta touches few chunks.
//
//   C  PROJECTION     nothing stored per state. A state's fiber is enumerated on demand from the
//                     match pool and an edge->matches index, anchoring each match at its minimum
//                     edge so it is yielded exactly once.
//
// All three share the pool, the matcher, the rewriter and the state edge sets, so the measured
// difference is the representation and nothing else. All three must produce identical
// (state, match) pair multisets; the harness asserts it.
//
// CORRECTNESS GATE. Full capture with no state merging is a tree, and the engine's own counts for
// these workloads are known (tools/unfolding_ratio_probe, StateCanonicalizationMode::Full reports
// raw states): growth/path(8)@4 = 8721 states / 8720 events, growth/path(16)@4 = 98209 / 98208,
// pair/path(12)@4 = 25884 / 25883, pair/cycle(8)@4 = 8721 / 8720. The prototype reproduces those
// numbers or it is wrong.
//
//   /tmp/representation_prototype

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint8_t MAX_ARITY = 4;
constexpr uint8_t MAX_LHS = 4;
constexpr uint32_t NO_EDGE = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Byte accounting. Every structure whose size depends on the representation is
// allocated through here, so peak is the representation's own high-water mark.
// ---------------------------------------------------------------------------
struct Accountant {
    size_t live = 0, peak = 0;
    void add(size_t n) { live += n; if (live > peak) peak = live; }
    void sub(size_t n) { live -= n; }
    void reset() { live = 0; peak = 0; }
};
Accountant g_acct;

template <typename T>
T* acct_alloc(size_t n) {
    if (n == 0) return nullptr;
    g_acct.add(n * sizeof(T));
    return new T[n];
}
template <typename T>
void acct_free(T* p, size_t n) {
    if (!p) return;
    g_acct.sub(n * sizeof(T));
    delete[] p;
}

// ---------------------------------------------------------------------------
// Pool, rules, states
// ---------------------------------------------------------------------------
struct Edge {
    uint32_t v[MAX_ARITY];
    uint8_t arity;
};

struct Rule {
    std::vector<std::vector<uint8_t>> lhs;  // pattern edges over variable ids
    std::vector<std::vector<uint8_t>> rhs;
    uint32_t lhs_var_mask = 0;
    uint32_t rhs_var_mask = 0;

    void finish() {
        for (const auto& e : lhs) for (uint8_t v : e) lhs_var_mask |= 1u << v;
        for (const auto& e : rhs) for (uint8_t v : e) rhs_var_mask |= 1u << v;
    }
    uint32_t new_var_mask() const { return rhs_var_mask & ~lhs_var_mask; }
};

// A match occurrence: the unfolding's event, determined by (rule, consumed occurrences).
struct MatchCore {
    uint32_t edges[MAX_LHS];   // in PATTERN order -- the counting convention is morphisms
    uint16_t rule;
    uint8_t n;
    uint32_t min_edge;         // anchor for arm C's duplicate-free enumeration
};

struct State {
    uint32_t* edges = nullptr;   // sorted edge ids
    uint32_t n_edges = 0;
    uint32_t parent = NO_EDGE;
    uint16_t depth = 0;
};

// ---------------------------------------------------------------------------
// Copy-on-write chunked bitset over match ids (arm B)
// ---------------------------------------------------------------------------
constexpr uint32_t CHUNK_BITS = 512;
constexpr uint32_t CHUNK_WORDS = CHUNK_BITS / 64;

struct Chunk {
    uint64_t w[CHUNK_WORDS];
    uint32_t refs;
};

struct CowBitset {
    Chunk** dir = nullptr;      // chunk directory; null entry == all zero
    uint32_t n_chunks = 0;

    void release() {
        if (!dir) return;
        for (uint32_t i = 0; i < n_chunks; ++i) {
            if (dir[i] && --dir[i]->refs == 0) { g_acct.sub(sizeof(Chunk)); delete dir[i]; }
        }
        acct_free(dir, n_chunks);
        dir = nullptr; n_chunks = 0;
    }

    void grow_to(uint32_t chunks) {
        if (chunks <= n_chunks) return;
        Chunk** nd = acct_alloc<Chunk*>(chunks);
        std::memset(nd, 0, chunks * sizeof(Chunk*));
        if (dir) {
            std::memcpy(nd, dir, n_chunks * sizeof(Chunk*));
            acct_free(dir, n_chunks);
        }
        dir = nd; n_chunks = chunks;
    }

    // Share the parent's chunks; only bumped refcounts, no data copied.
    void derive_from(const CowBitset& p, uint32_t want_chunks) {
        grow_to(want_chunks > p.n_chunks ? want_chunks : p.n_chunks);
        for (uint32_t i = 0; i < p.n_chunks; ++i) {
            dir[i] = p.dir[i];
            if (dir[i]) ++dir[i]->refs;
        }
    }

    Chunk* writable(uint32_t ci) {
        Chunk* c = dir[ci];
        if (!c) {
            g_acct.add(sizeof(Chunk));
            c = new Chunk(); std::memset(c->w, 0, sizeof(c->w)); c->refs = 1;
            dir[ci] = c;
            return c;
        }
        if (c->refs == 1) return c;
        g_acct.add(sizeof(Chunk));
        Chunk* nc = new Chunk(*c); nc->refs = 1;
        --c->refs;
        dir[ci] = nc;
        return nc;
    }

    void set(uint32_t bit) {
        const uint32_t ci = bit / CHUNK_BITS;
        grow_to(ci + 1);
        writable(ci)->w[(bit % CHUNK_BITS) / 64] |= 1ull << (bit % 64);
    }
    void clear(uint32_t bit) {
        const uint32_t ci = bit / CHUNK_BITS;
        if (ci >= n_chunks || !dir[ci]) return;
        if (!(dir[ci]->w[(bit % CHUNK_BITS) / 64] & (1ull << (bit % 64)))) return;
        writable(ci)->w[(bit % CHUNK_BITS) / 64] &= ~(1ull << (bit % 64));
    }
    template <typename F>
    void for_each(F&& f) const {
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            const Chunk* c = dir[ci];
            if (!c) continue;
            for (uint32_t wi = 0; wi < CHUNK_WORDS; ++wi) {
                uint64_t w = c->w[wi];
                while (w) {
                    const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(w));
                    f(ci * CHUNK_BITS + wi * 64 + b);
                    w &= w - 1;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// The engine under test
// ---------------------------------------------------------------------------
//   D  ANCESTRAL      per state, only the matches DISCOVERED there. A state's fiber is the
//                     matches added anywhere on its ancestor path that satisfy the fiber
//                     predicate in this state. No removal list is needed: "no longer valid" IS
//                     the predicate. Memory is one entry per match per DISCOVERY (not per pair),
//                     and the candidate set is the state's own ancestry rather than C's global
//                     edge index, which is what makes C degrade on multi-edge patterns.
//   E  SHARED         D's representation plus the SHARING idea: a match is rewritten ONCE, and
//                     every state in its up-set reuses the produced occurrences instead of
//                     allocating its own. Match cores are then deduplicated by (rule, edge
//                     tuple), because sharing makes two states' discoveries collide. The
//                     (state, match) pair count is UNCHANGED -- the output is the same; what
//                     drops is rewrite computations and pool size.
enum class Arm { Materialised, Membership, Projection, Ancestral, Shared };

struct Engine {
    Arm arm;
    std::vector<Edge> pool;
    std::vector<Rule> rules;
    std::vector<State> states;
    std::vector<MatchCore> cores;
    uint32_t next_vertex = 0;

    // Arm A
    std::vector<uint32_t*> a_list;
    std::vector<uint32_t>  a_len;
    // Arm B
    std::vector<CowBitset> b_set;
    // Arms B and C: edge -> matches containing it
    std::vector<std::vector<uint32_t>> edge_matches;
    // Arms D and E
    std::vector<uint32_t*> d_added;
    std::vector<uint32_t>  d_len;
    size_t rewrites_done = 0;

    static uint64_t tuple_hash(uint16_t rule, const uint32_t* e, uint8_t n) {
        uint64_t h = 1469598103934665603ull;
        h = (h ^ rule) * 1099511628211ull;
        for (uint8_t i = 0; i < n; ++i) h = (h ^ e[i]) * 1099511628211ull;
        return h;
    }
    // Rewrite results, keyed by the pattern-ordered tuple: an entry is the produced
    // occurrences of the ONE rewrite of that match, reused by every state in its up-set.
    struct Shared { uint16_t rule; uint8_t n; uint32_t edges[MAX_LHS];
                    uint32_t prod[8]; uint8_t n_prod; };
    std::vector<std::vector<Shared>> e_tbl;

    Shared* find_shared(uint16_t rule, const uint32_t* e, uint8_t n) {
        if (e_tbl.empty()) e_tbl.resize(1 << 16);
        for (auto& sh : e_tbl[tuple_hash(rule, e, n) & 0xFFFF]) {
            if (sh.rule != rule || sh.n != n) continue;
            bool same = true;
            for (uint8_t i = 0; i < n && same; ++i) if (sh.edges[i] != e[i]) same = false;
            if (same) return &sh;
        }
        return nullptr;
    }

    size_t pairs = 0;   // (state, match) pairs expanded -- the output size

    ~Engine() {
        for (size_t i = 0; i < a_list.size(); ++i) acct_free(a_list[i], a_len[i]);
        for (size_t i = 0; i < d_added.size(); ++i) acct_free(d_added[i], d_len[i]);
        for (auto& b : b_set) b.release();
        for (auto& s : states) acct_free(s.edges, s.n_edges);
    }

    bool has_edge(const State& s, uint32_t e) const {
        return std::binary_search(s.edges, s.edges + s.n_edges, e);
    }

    uint32_t add_edge(const uint32_t* v, uint8_t arity) {
        Edge e; e.arity = arity;
        for (uint8_t i = 0; i < arity; ++i) e.v[i] = v[i];
        g_acct.add(sizeof(Edge));
        pool.push_back(e);
        if (arm == Arm::Membership || arm == Arm::Projection) {
            edge_matches.emplace_back();
            g_acct.add(sizeof(std::vector<uint32_t>));
        }
        return static_cast<uint32_t>(pool.size() - 1);
    }

    // --- matching: edge-injective, vertex binding non-injective -------------
    bool bind(const Edge& e, const std::vector<uint8_t>& pat,
              uint32_t* b, uint32_t& mask) const {
        if (e.arity != pat.size()) return false;
        for (uint8_t i = 0; i < e.arity; ++i) {
            const uint8_t var = pat[i];
            const uint32_t bit = 1u << var;
            if (mask & bit) { if (b[var] != e.v[i]) return false; }
            else { b[var] = e.v[i]; mask |= bit; }
        }
        return true;
    }

    // Enumerate every match of rule `ri` in `s`, emitting pattern-ordered edge tuples.
    // Arity is a matching constraint, enforced in bind(): a pattern edge of arity k admits
    // only data edges of arity k, so mixed-arity states are filtered per pattern position.
    template <typename F>
    void enumerate(const State& s, uint16_t ri, F&& emit) const {
        const Rule& r = rules[ri];
        uint32_t chosen[MAX_LHS];
        uint32_t b[32]; uint32_t mask = 0;
        auto rec = [&](auto&& self, size_t pos) -> void {
            if (pos == r.lhs.size()) { emit(chosen); return; }
            for (uint32_t i = 0; i < s.n_edges; ++i) {
                const uint32_t eid = s.edges[i];
                bool used = false;
                for (size_t k = 0; k < pos; ++k) if (chosen[k] == eid) { used = true; break; }
                if (used) continue;                       // edge-injective
                const uint32_t save = mask;
                if (!bind(pool[eid], r.lhs[pos], b, mask)) { mask = save; continue; }
                chosen[pos] = eid;
                self(self, pos + 1);
                mask = save;
            }
        };
        rec(rec, 0);
    }

    // Matches that use at least one of `produced` -- the pool's growth step.
    void discover_delta(const State& s, const uint32_t* produced, uint8_t n_prod,
                        std::vector<uint32_t>& out_new_ids) {
        for (uint16_t ri = 0; ri < rules.size(); ++ri) {
            enumerate(s, ri, [&](const uint32_t* chosen) {
                const size_t n = rules[ri].lhs.size();
                bool touches = false;
                for (size_t k = 0; k < n && !touches; ++k)
                    for (uint8_t p = 0; p < n_prod; ++p)
                        if (chosen[k] == produced[p]) { touches = true; break; }
                if (!touches) return;
                MatchCore mc; mc.rule = ri; mc.n = static_cast<uint8_t>(n);
                mc.min_edge = NO_EDGE;
                for (size_t k = 0; k < n; ++k) {
                    mc.edges[k] = chosen[k];
                    if (chosen[k] < mc.min_edge) mc.min_edge = chosen[k];
                }
                const uint32_t id = static_cast<uint32_t>(cores.size());
                g_acct.add(sizeof(MatchCore));
                cores.push_back(mc);
                out_new_ids.push_back(id);
                if (arm == Arm::Membership || arm == Arm::Projection)
                    for (size_t k = 0; k < n; ++k) {
                        edge_matches[mc.edges[k]].push_back(id);
                        g_acct.add(sizeof(uint32_t));
                    }
            });
        }
    }

    void discover_all(const State& s, std::vector<uint32_t>& out_new_ids) {
        for (uint16_t ri = 0; ri < rules.size(); ++ri) {
            enumerate(s, ri, [&](const uint32_t* chosen) {
                const size_t n = rules[ri].lhs.size();
                MatchCore mc; mc.rule = ri; mc.n = static_cast<uint8_t>(n);
                mc.min_edge = NO_EDGE;
                for (size_t k = 0; k < n; ++k) {
                    mc.edges[k] = chosen[k];
                    if (chosen[k] < mc.min_edge) mc.min_edge = chosen[k];
                }
                const uint32_t id = static_cast<uint32_t>(cores.size());
                g_acct.add(sizeof(MatchCore));
                cores.push_back(mc);
                out_new_ids.push_back(id);
                if (arm == Arm::Membership || arm == Arm::Projection)
                    for (size_t k = 0; k < n; ++k) {
                        edge_matches[mc.edges[k]].push_back(id);
                        g_acct.add(sizeof(uint32_t));
                    }
            });
        }
    }

    bool valid_in(const State& s, const MatchCore& mc) const {
        for (uint8_t k = 0; k < mc.n; ++k) if (!has_edge(s, mc.edges[k])) return false;
        return true;
    }

    // The fiber of `sid`, however this arm represents it.
    void fiber(uint32_t sid, std::vector<uint32_t>& out) const {
        out.clear();
        const State& s = states[sid];
        switch (arm) {
        case Arm::Materialised:
            out.assign(a_list[sid], a_list[sid] + a_len[sid]);
            break;
        case Arm::Membership:
            b_set[sid].for_each([&](uint32_t id) { out.push_back(id); });
            break;
        case Arm::Projection:
            // Anchor at the minimum edge so each match is yielded exactly once.
            for (uint32_t i = 0; i < s.n_edges; ++i) {
                const uint32_t e = s.edges[i];
                for (uint32_t id : edge_matches[e]) {
                    const MatchCore& mc = cores[id];
                    if (mc.min_edge != e) continue;
                    if (valid_in(s, mc)) out.push_back(id);
                }
            }
            break;
        case Arm::Ancestral:
        case Arm::Shared:
            // Walk the ancestor path. A match added anywhere on it belongs to this state's
            // fiber exactly when its edges are still present here, so no removal list is
            // needed -- "no longer valid" IS the predicate. The candidate set is this state's
            // own ancestry, not the global edge index, which is what makes C degrade.
            for (uint32_t a = sid; a != NO_EDGE; a = states[a].parent) {
                for (uint32_t i = 0; i < d_len[a]; ++i) {
                    const uint32_t id = d_added[a][i];
                    if (valid_in(s, cores[id])) out.push_back(id);
                }
            }
            break;
        }
    }

    uint32_t make_state(uint32_t parent, const uint32_t* edges, uint32_t n, uint16_t depth) {
        State st; st.parent = parent; st.depth = depth; st.n_edges = n;
        st.edges = acct_alloc<uint32_t>(n);
        std::memcpy(st.edges, edges, n * sizeof(uint32_t));
        states.push_back(st);
        return static_cast<uint32_t>(states.size() - 1);
    }

    // Apply match `mid` in state `sid`: S' = (S \ consumed) u produced.
    uint32_t rewrite(uint32_t sid, uint32_t mid, uint32_t* produced, uint8_t& n_prod) {
        const MatchCore& mc = cores[mid];
        const Rule& r = rules[mc.rule];
        // Re-derive the binding from (rule, ordered edge tuple) -- it is not stored.
        uint32_t b[32]; uint32_t mask = 0;
        for (uint8_t k = 0; k < mc.n; ++k) bind(pool[mc.edges[k]], r.lhs[k], b, mask);
        Shared* sh = (arm == Arm::Shared) ? find_shared(mc.rule, mc.edges, mc.n) : nullptr;
        if (!sh)
            for (uint8_t v = 0; v < 32; ++v)
                if (r.new_var_mask() & (1u << v)) { b[v] = next_vertex++; mask |= 1u << v; }

        const State& s = states[sid];
        std::vector<uint32_t> ne;
        ne.reserve(s.n_edges + r.rhs.size());
        for (uint32_t i = 0; i < s.n_edges; ++i) {
            bool consumed = false;
            for (uint8_t k = 0; k < mc.n; ++k) if (s.edges[i] == mc.edges[k]) { consumed = true; break; }
            if (!consumed) ne.push_back(s.edges[i]);
        }
        n_prod = 0;
        if (arm == Arm::Shared) {
            if (sh) {
                // Already rewritten in another state of this match's up-set: reuse the
                // occurrences instead of allocating a second isomorphic copy.
                for (uint8_t i = 0; i < sh->n_prod; ++i)
                    { produced[n_prod++] = sh->prod[i]; ne.push_back(sh->prod[i]); }
            } else {
                ++rewrites_done;
                Shared rec; rec.rule = mc.rule; rec.n = mc.n; rec.n_prod = 0;
                for (uint8_t i = 0; i < mc.n; ++i) rec.edges[i] = mc.edges[i];
                for (const auto& re : r.rhs) {
                    uint32_t vv[MAX_ARITY];
                    for (size_t i = 0; i < re.size(); ++i) vv[i] = b[re[i]];
                    const uint32_t eid = add_edge(vv, static_cast<uint8_t>(re.size()));
                    produced[n_prod++] = eid;
                    ne.push_back(eid);
                    rec.prod[rec.n_prod++] = eid;
                }
                if (e_tbl.empty()) e_tbl.resize(1 << 16);
                e_tbl[tuple_hash(mc.rule, mc.edges, mc.n) & 0xFFFF].push_back(rec);
            }
        } else {
            ++rewrites_done;
            for (const auto& re : r.rhs) {
                uint32_t vv[MAX_ARITY];
                for (size_t i = 0; i < re.size(); ++i) vv[i] = b[re[i]];
                const uint32_t eid = add_edge(vv, static_cast<uint8_t>(re.size()));
                produced[n_prod++] = eid;
                ne.push_back(eid);
            }
        }
        std::sort(ne.begin(), ne.end());
        return make_state(sid, ne.data(), static_cast<uint32_t>(ne.size()),
                          static_cast<uint16_t>(states[sid].depth + 1));
    }

    void install_fiber(uint32_t sid, uint32_t parent, const MatchCore* consumed_by,
                       const std::vector<uint32_t>& fresh) {
        switch (arm) {
        case Arm::Materialised: {
            std::vector<uint32_t> kept;
            if (parent != NO_EDGE) {
                const State& s = states[sid];
                for (uint32_t i = 0; i < a_len[parent]; ++i) {
                    const uint32_t id = a_list[parent][i];
                    const MatchCore& mc = cores[id];
                    bool dead = false;
                    for (uint8_t k = 0; k < mc.n && !dead; ++k)
                        for (uint8_t j = 0; j < consumed_by->n; ++j)
                            if (mc.edges[k] == consumed_by->edges[j]) { dead = true; break; }
                    if (!dead && valid_in(s, mc)) kept.push_back(id);
                }
            }
            for (uint32_t id : fresh) kept.push_back(id);
            a_list.resize(states.size(), nullptr);
            a_len.resize(states.size(), 0);
            a_len[sid] = static_cast<uint32_t>(kept.size());
            a_list[sid] = acct_alloc<uint32_t>(kept.size());
            std::memcpy(a_list[sid], kept.data(), kept.size() * sizeof(uint32_t));
            break;
        }
        case Arm::Membership: {
            b_set.resize(states.size());
            const uint32_t want = static_cast<uint32_t>(cores.size() / CHUNK_BITS + 1);
            if (parent != NO_EDGE) {
                b_set[sid].derive_from(b_set[parent], want);
                for (uint8_t j = 0; j < consumed_by->n; ++j)
                    for (uint32_t id : edge_matches[consumed_by->edges[j]])
                        b_set[sid].clear(id);
            } else {
                b_set[sid].grow_to(want);
            }
            for (uint32_t id : fresh) b_set[sid].set(id);
            break;
        }
        case Arm::Projection:
            break;   // nothing stored
        case Arm::Ancestral:
        case Arm::Shared: {
            d_added.resize(states.size(), nullptr);
            d_len.resize(states.size(), 0);
            d_len[sid] = static_cast<uint32_t>(fresh.size());
            d_added[sid] = acct_alloc<uint32_t>(fresh.size());
            std::memcpy(d_added[sid], fresh.data(), fresh.size() * sizeof(uint32_t));
            break;
        }
        }
    }

    void evolve(const std::vector<std::vector<uint32_t>>& init, uint16_t steps) {
        uint32_t maxv = 0;
        std::vector<uint32_t> e0;
        for (const auto& e : init) {
            uint32_t vv[MAX_ARITY];
            for (size_t i = 0; i < e.size(); ++i) { vv[i] = e[i]; maxv = std::max(maxv, e[i] + 1); }
            e0.push_back(add_edge(vv, static_cast<uint8_t>(e.size())));
        }
        next_vertex = maxv;
        std::sort(e0.begin(), e0.end());
        const uint32_t root = make_state(NO_EDGE, e0.data(), static_cast<uint32_t>(e0.size()), 0);

        std::vector<uint32_t> fresh;
        discover_all(states[root], fresh);
        install_fiber(root, NO_EDGE, nullptr, fresh);

        std::vector<uint32_t> frontier{root}, next;
        std::vector<uint32_t> fib;
        for (uint16_t d = 0; d < steps; ++d) {
            next.clear();
            for (uint32_t sid : frontier) {
                fiber(sid, fib);
                for (uint32_t mid : fib) {
                    ++pairs;
                    uint32_t produced[8]; uint8_t n_prod = 0;
                    const uint32_t child = rewrite(sid, mid, produced, n_prod);
                    fresh.clear();
                    discover_delta(states[child], produced, n_prod, fresh);
                    install_fiber(child, sid, &cores[mid], fresh);
                    next.push_back(child);
                }
            }
            frontier.swap(next);
        }
    }
};

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------
Rule growth() {   // {{x,y}} -> {{x,y},{y,z}}
    Rule r; r.lhs = {{0, 1}}; r.rhs = {{0, 1}, {1, 2}}; r.finish(); return r;
}
Rule pair_rule() {  // {{x,y},{y,z}} -> {{x,y},{y,w},{w,z}}
    Rule r; r.lhs = {{0, 1}, {1, 2}}; r.rhs = {{0, 1}, {1, 3}, {3, 2}}; r.finish(); return r;
}
Rule tri() {  // {{x,y},{y,z},{z,x}} -> {{x,y},{y,z},{z,w},{w,x}}
    Rule r; r.lhs = {{0, 1}, {1, 2}, {2, 0}};
    r.rhs = {{0, 1}, {1, 2}, {2, 3}, {3, 0}}; r.finish(); return r;
}

std::vector<std::vector<uint32_t>> path(uint32_t n) {
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 0; i < n; ++i) o.push_back({i, i + 1});
    return o;
}
std::vector<std::vector<uint32_t>> cycle(uint32_t n) {
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 0; i < n; ++i) o.push_back({i, (i + 1) % n});
    return o;
}
std::vector<std::vector<uint32_t>> disjoint(uint32_t n) {
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 0; i < n; ++i) o.push_back({2 * i, 2 * i + 1});
    return o;
}
std::vector<std::vector<uint32_t>> star(uint32_t n) {
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 1; i <= n; ++i) o.push_back({0, i});
    return o;
}

Rule ternary() {  // {{x,y,z}} -> {{x,y,z},{z,w}}: arity-3 LHS, arity-2 in the RHS
    Rule r; r.lhs = {{0, 1, 2}}; r.rhs = {{0, 1, 2}, {2, 3}}; r.finish(); return r;
}
Rule mixed_arity() {  // {{x,y},{y,z,w}} -> {{x,z},{z,w},{w,y,x}}: LHS spans two arities
    Rule r; r.lhs = {{0, 1}, {1, 2, 3}};
    r.rhs = {{0, 2}, {2, 3}, {3, 1, 0}}; r.finish(); return r;
}

// Arity-3 initial states, and a state carrying both arities at once.
std::vector<std::vector<uint32_t>> triples(uint32_t n) {
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 0; i < n; ++i) o.push_back({i, i + 1, i + 2});
    return o;
}
std::vector<std::vector<uint32_t>> mixed(uint32_t n) {
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 0; i < n; ++i) {
        o.push_back({i, i + 1});
        o.push_back({i, i + 1, i + 2});
    }
    return o;
}

Rule loop_maker() {  // {{x,y}} -> {{x,x},{x,y}}: creates a self-loop every step
    Rule r; r.lhs = {{0, 1}}; r.rhs = {{0, 0}, {0, 1}}; r.finish(); return r;
}
Rule loop_eater() {  // {{x,x}} -> {{x,y},{y,x}}: LHS is itself a self-loop pattern
    Rule r; r.lhs = {{0, 0}}; r.rhs = {{0, 1}, {1, 0}}; r.finish(); return r;
}
std::vector<std::vector<uint32_t>> loops(uint32_t n) {   // n self-loops on distinct vertices
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 0; i < n; ++i) o.push_back({i, i});
    return o;
}
std::vector<std::vector<uint32_t>> path_with_loops(uint32_t n) {
    std::vector<std::vector<uint32_t>> o;
    for (uint32_t i = 0; i < n; ++i) { o.push_back({i, i + 1}); o.push_back({i, i}); }
    return o;
}

struct Workload {
    const char* name;
    std::vector<Rule> rules;
    std::vector<std::vector<uint32_t>> init;
    uint16_t steps;
};

struct Result { double ms; size_t peak; size_t states; size_t pairs; size_t cores;
                size_t rewrites; size_t edges; };

Result run(const Workload& w, Arm arm) {
    g_acct.reset();
    Engine e;
    e.arm = arm;
    e.rules = w.rules;
    const auto t0 = std::chrono::steady_clock::now();
    e.evolve(w.init, w.steps);
    const auto t1 = std::chrono::steady_clock::now();
    Result r;
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.peak = g_acct.peak;
    r.states = e.states.size();
    r.pairs = e.pairs;
    r.cores = e.cores.size();
    r.rewrites = e.rewrites_done;
    r.edges = e.pool.size();
    return r;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<Workload> ws = {
        {"growth/path(8)",     {growth()},    path(8),     4},
        {"growth/path(12)",    {growth()},    path(12),    4},
        {"growth/disjoint(8)", {growth()},    disjoint(8), 4},
        {"growth/star(8)",     {growth()},    star(8),     4},
        {"growth/cycle(8)",    {growth()},    cycle(8),    4},
        {"pair/path(12)",      {pair_rule()}, path(12),    4},
        {"pair/path(16)",      {pair_rule()}, path(16),    4},
        {"pair/cycle(8)",      {pair_rule()}, cycle(8),    4},
        {"pair/star(8)",       {pair_rule()}, star(8),     4},
        {"tri/cycle(3)",       {tri()},       cycle(3),    4},
        {"two-rule/path(8)",   {growth(), pair_rule()}, path(8), 3},
        {"ternary/triples(6)", {ternary()},     triples(6),  4},
        {"ternary/mixed(4)",   {ternary()},     mixed(4),    4},
        {"mixedar/mixed(4)",   {mixed_arity()}, mixed(4),    4},
        {"mixedar/mixed(6)",   {mixed_arity()}, mixed(6),    3},
        {"arity-mix/mixed(4)", {growth(), ternary()}, mixed(4), 3},

        // Self-loops. Non-injective vertex binding makes {{x,y}} match {a,a}, so a loop is a
        // candidate for every binary pattern, and loop_eater's LHS is a loop pattern itself.
        {"loopmk/path(6)",     {loop_maker()},  path(6),           4},
        {"loopmk/loops(6)",    {loop_maker()},  loops(6),          4},
        {"loopeat/loops(6)",   {loop_eater()},  loops(6),          4},
        {"growth/loops(6)",    {growth()},      loops(6),          4},
        {"growth/pathloop(4)", {growth()},      path_with_loops(4), 3},

        // Multi-rule: rules that interact, and rules that cannot (disjoint LHS arities).
        {"3rule/path(6)",      {growth(), pair_rule(), loop_maker()}, path(6),  3},
        {"3rule/pathloop(4)",  {growth(), loop_eater(), pair_rule()}, path_with_loops(4), 3},
        {"4rule/mixed(3)",     {growth(), pair_rule(), ternary(), loop_maker()}, mixed(3), 3},
        {"2rule-arity/mix(4)", {ternary(), mixed_arity()}, mixed(4), 3},

        // Depth sweep on one workload: the ratios are depth-dependent, so vary only d.
        {"depth/path(4)",      {growth()},      path(4),           3},
        {"depth/path(4)",      {growth()},      path(4),           4},
        {"depth/path(4)",      {growth()},      path(4),           5},
        {"depth/path(4)",      {growth()},      path(4),           6},
    };

    std::printf("%-20s %-14s %9s %12s %9s %10s %8s %8s\n",
                "workload", "arm", "ms", "peak KiB", "states", "pairs", "vs A ms", "vs A mem");
    for (const auto& w : ws) {
        Result a = run(w, Arm::Materialised);
        Result b = run(w, Arm::Membership);
        Result c = run(w, Arm::Projection);
        Result d = run(w, Arm::Ancestral);
        Result e = run(w, Arm::Shared);

        // The arms must do identical work; a divergence invalidates any comparison.
        if (a.states != b.states || a.states != c.states || a.states != d.states ||
            a.states != e.states ||
            a.pairs != b.pairs || a.pairs != c.pairs || a.pairs != d.pairs ||
            a.pairs != e.pairs) {
            std::printf("%-20s ARMS DIVERGED: states A=%zu B=%zu C=%zu D=%zu E=%zu  "
                        "pairs A=%zu B=%zu C=%zu D=%zu E=%zu\n",
                        w.name, a.states, b.states, c.states, d.states, e.states,
                        a.pairs, b.pairs, c.pairs, d.pairs, e.pairs);
            continue;
        }

        std::printf("%-20s(d=%u) %-14s %9.1f %12zu %9zu %10zu %8s %8s\n",
                    w.name, w.steps, "A materialised", a.ms, a.peak / 1024,
                    a.states, a.pairs, "-", "-");
        std::printf("%-20s %-14s %9.1f %12zu %9s %10s %7.2fx %7.2fx\n",
                    "", "B membership", b.ms, b.peak / 1024, "", "",
                    b.ms / a.ms, double(b.peak) / double(a.peak));
        std::printf("%-20s %-14s %9.1f %12zu %9s %10s %7.2fx %7.2fx  cores=%zu\n",
                    "", "C projection", c.ms, c.peak / 1024, "", "",
                    c.ms / a.ms, double(c.peak) / double(a.peak), c.cores);
        std::printf("%-20s %-14s %9.1f %12zu %9s %10s %7.2fx %7.2fx\n",
                    "", "D ancestral", d.ms, d.peak / 1024, "", "",
                    d.ms / a.ms, double(d.peak) / double(a.peak));
        std::printf("%-20s %-14s %9.1f %12zu %9s %10s %7.2fx %7.2fx"
                    "  rewrites %zu->%zu (%.1fx)  edges %zu->%zu  cores %zu->%zu\n\n",
                    "", "E shared", e.ms, e.peak / 1024, "", "",
                    e.ms / a.ms, double(e.peak) / double(a.peak),
                    d.rewrites, e.rewrites, double(d.rewrites) / double(e.rewrites ? e.rewrites : 1),
                    d.edges, e.edges, d.cores, e.cores);
    }
    return 0;
}
