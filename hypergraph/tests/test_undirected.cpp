// "UndirectedEdges": two-vertex edges are unordered pairs (hgcommon/undirected_core.hpp).
//
// The canonical hash is checked against a brute-force isomorphism test on small multigraphs. The
// engine is checked against the emulation that expands each rule into the directed rules of
// every orientation of its two-vertex left edges: the multiway trees agree once the emulation's
// duplicate matches are removed, a duplicate being a match on the same edges whose binding
// differs by a symmetry of the rule (computed here by brute force, independently of the engine).
#include <gtest/gtest.h>

#include "hgcommon/undirected_core.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace hypergraph;

namespace {

using EdgeList = std::vector<std::vector<VertexId>>;
using Pattern = std::vector<std::vector<uint8_t>>;

struct RuleSpec {
    Pattern lhs, rhs;
};

// The rules the agreement test runs: Gorard's rule (G3 line 6202), a rule whose two left edges
// are exchanged by its reversal, a one-edge rule whose two orientations differ, and the
// signature rule of the Wolfram model.
const RuleSpec kGorard{{{0, 1}, {1, 2}, {2, 3}, {3, 4}}, {{1, 5}, {5, 4}, {3, 0}, {0, 5}}};
const RuleSpec kPathContract{{{0, 1}, {1, 2}}, {{0, 2}}};
const RuleSpec kGrow{{{0, 1}}, {{0, 1}, {1, 2}}};
const RuleSpec kWolfram{{{0, 1}, {0, 2}}, {{0, 1}, {0, 3}, {1, 3}, {2, 3}}};

RewriteRule build_rule(const Pattern& lhs, const Pattern& rhs, uint16_t index) {
    RuleBuilder b(index);
    for (const auto& e : lhs) b.lhs(e);
    for (const auto& e : rhs) b.rhs(e);
    return b.build();
}

// Every orientation of the two-vertex left edges, first edge most significant.
std::vector<Pattern> orientation_variants(const Pattern& lhs) {
    std::vector<size_t> pairs;
    for (size_t p = 0; p < lhs.size(); ++p)
        if (lhs[p].size() == 2) pairs.push_back(p);
    std::vector<Pattern> out;
    for (uint32_t bits = 0; bits < (1u << pairs.size()); ++bits) {
        Pattern v = lhs;
        for (size_t j = 0; j < pairs.size(); ++j)
            if ((bits >> (pairs.size() - 1 - j)) & 1u) std::swap(v[pairs[j]][0], v[pairs[j]][1]);
        out.push_back(v);
    }
    return out;
}

std::vector<VertexId> sorted_edge(std::vector<VertexId> e) {
    if (e.size() == 2 && e[1] < e[0]) std::swap(e[0], e[1]);
    return e;
}

std::vector<std::vector<uint8_t>> sorted_pattern(const Pattern& p) {
    std::vector<std::vector<uint8_t>> out;
    for (auto e : p) {
        if (e.size() == 2 && e[1] < e[0]) std::swap(e[0], e[1]);
        out.push_back(e);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// A symmetry of the rule: the image of each left variable, and the left position each left
// edge's image occupies.
struct Symmetry {
    std::vector<uint8_t> var;
    std::vector<size_t> pos;
};

// Brute force: every permutation of the left variables that maps the unordered left edges onto
// themselves and, with a permutation of the fresh variables, the unordered right edges onto
// themselves; one Symmetry per assignment of left positions.
std::vector<Symmetry> rule_symmetries(const RuleSpec& r) {
    uint8_t nv = 0;
    for (const auto& e : r.lhs) for (uint8_t v : e) nv = std::max<uint8_t>(nv, v + 1);
    uint8_t nr = nv;
    for (const auto& e : r.rhs) for (uint8_t v : e) nr = std::max<uint8_t>(nr, v + 1);
    std::vector<uint8_t> perm(nv);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<Symmetry> out;
    auto apply = [](const Pattern& p, const std::vector<uint8_t>& m) {
        Pattern q = p;
        for (auto& e : q) for (auto& v : e) v = m[v];
        return q;
    };
    do {
        if (sorted_pattern(apply(r.lhs, perm)) != sorted_pattern(r.lhs)) continue;
        std::vector<uint8_t> fresh(nr - nv);
        std::iota(fresh.begin(), fresh.end(), nv);
        bool rhs_ok = false;
        do {
            std::vector<uint8_t> full(perm);
            full.insert(full.end(), fresh.begin(), fresh.end());
            rhs_ok = sorted_pattern(apply(r.rhs, full)) == sorted_pattern(r.rhs);
        } while (!rhs_ok && std::next_permutation(fresh.begin(), fresh.end()));
        if (!rhs_ok) continue;
        // Every bijection of positions consistent with the map.
        const Pattern img = apply(r.lhs, perm);
        std::vector<size_t> pos(r.lhs.size());
        std::vector<bool> used(r.lhs.size(), false);
        std::function<void(size_t)> place = [&](size_t i) {
            if (i == r.lhs.size()) { out.push_back(Symmetry{perm, pos}); return; }
            for (size_t t = 0; t < r.lhs.size(); ++t) {
                if (used[t]) continue;
                if (sorted_pattern({img[i]}) != sorted_pattern({r.lhs[t]})) continue;
                used[t] = true; pos[i] = t; place(i + 1); used[t] = false;
            }
        };
        place(0);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return out;
}

uint64_t undirected_hash(const EdgeList& edges) {
    return IRCanonicalizer(true).compute_canonical_hash(edges);
}

EdgeList state_edges(const Hypergraph& hg, StateId s) {
    EdgeList out;
    hg.get_state(s).edges.for_each([&](EdgeId e) {
        const Edge& edge = hg.get_edge(e);
        out.emplace_back(edge.vertices, edge.vertices + edge.arity);
    });
    return out;
}

// The multiway tree of a run under None states: each state's undirected hash and its children,
// each child with the key of the match that made it (edges by pattern position and the binding
// of the user rule's variables, least over the rule's symmetries).
struct Node {
    uint64_t hash = 0;
    std::vector<std::pair<std::vector<uint64_t>, StateId>> children;
};

struct Run {
    std::map<StateId, Node> nodes;
    std::vector<StateId> roots;
    size_t events = 0;
};

// The binding of an event of user rule `r`: the one assignment of the left variables that binds
// the consumed edges (pattern order) in some orientation and produces the produced edges (right
// side order). Every rule here shows enough of its left variables on its right side for that to
// be unique; an empty result means it was not.
std::vector<uint64_t> recover_binding(const RuleSpec& r, const Hypergraph& hg, const Event& x) {
    std::set<std::vector<uint64_t>> found;
    const uint32_t masks = 1u << r.lhs.size();
    for (uint32_t mask = 0; mask < masks; ++mask) {
        std::vector<uint64_t> b(32, UINT64_MAX);
        bool ok = x.num_consumed == r.lhs.size() && x.num_produced == r.rhs.size();
        for (size_t p = 0; p < r.lhs.size() && ok; ++p) {
            const Edge& e = hg.get_edge(x.consumed_edges[p]);
            ok = e.arity == r.lhs[p].size();
            const bool rev = ((mask >> p) & 1u) && e.arity == 2;
            for (size_t k = 0; k < r.lhs[p].size() && ok; ++k) {
                const uint64_t v = e.vertices[rev ? 1 - k : k];
                uint64_t& slot = b[r.lhs[p][k]];
                if (slot == UINT64_MAX) slot = v;
                else ok = slot == v;
            }
        }
        const std::vector<uint64_t> lhs_only = b;
        for (size_t j = 0; j < r.rhs.size() && ok; ++j) {
            const Edge& e = hg.get_edge(x.produced_edges[j]);
            ok = e.arity == r.rhs[j].size();
            for (size_t k = 0; k < r.rhs[j].size() && ok; ++k) {
                uint64_t& slot = b[r.rhs[j][k]];
                if (slot == UINT64_MAX) slot = e.vertices[k];
                else ok = slot == e.vertices[k];
            }
        }
        if (ok) found.insert(lhs_only);
    }
    return found.size() == 1 ? *found.begin() : std::vector<uint64_t>{};
}

// `user[t]` is the user rule of an event whose rule index is t.
Run collect(Hypergraph& hg, const std::vector<RuleSpec>& rules, const std::vector<size_t>& user) {
    std::vector<std::vector<Symmetry>> sym;
    for (const auto& r : rules) sym.push_back(rule_symmetries(r));
    Run run;
    for (StateId s = 0; s < hg.num_published_states(); ++s) {
        const State& st = hg.get_state(s);
        if (st.id == INVALID_ID) continue;
        run.nodes[s].hash = undirected_hash(state_edges(hg, s));
    }
    std::set<StateId> derived;
    for (EventId ev = 0; ev < hg.num_published_events(); ++ev) {
        const Event& x = hg.get_event(ev);
        if (x.id == INVALID_ID || hg.is_genesis_event(ev)) continue;
        ++run.events;
        derived.insert(x.output_state);
        const size_t t = x.rule_index;
        const RuleSpec& rule = rules.at(user.at(t));
        const Pattern& lhs = rule.lhs;
        const std::vector<uint64_t> binding = recover_binding(rule, hg, x);
        EXPECT_FALSE(binding.empty()) << "event " << ev << " has no unique binding";
        if (binding.empty()) continue;
        std::vector<uint64_t> best;
        for (const Symmetry& g : sym[user[t]]) {
            std::vector<uint64_t> key;
            for (size_t i = 0; i < lhs.size(); ++i) key.push_back(x.consumed_edges[g.pos[i]]);
            for (size_t v = 0; v < g.var.size(); ++v) key.push_back(binding[g.var[v]]);
            if (best.empty() || key < best) best = key;
        }
        best.insert(best.begin(), user[t]);
        run.nodes[x.input_state].children.emplace_back(best, x.output_state);
    }
    for (const auto& [s, node] : run.nodes)
        if (!derived.count(s)) run.roots.push_back(s);
    return run;
}

// The tree below `s` with duplicate matches removed, as a nested value. `dups` counts the
// children removed and `kept` the events left.
using Shape = std::vector<uint64_t>;
Shape shape(const Run& run, StateId s, size_t& dups, size_t& kept) {
    const Node& n = run.nodes.at(s);
    std::map<std::vector<uint64_t>, StateId> once;
    for (const auto& c : n.children)
        if (!once.emplace(c.first, c.second).second) ++dups;
    kept += once.size();
    std::vector<Shape> kids;
    for (const auto& [key, child] : once) kids.push_back(shape(run, child, dups, kept));
    std::sort(kids.begin(), kids.end());
    Shape out{n.hash, kids.size()};
    for (const auto& k : kids) {
        out.push_back(k.size());
        out.insert(out.end(), k.begin(), k.end());
    }
    return out;
}

struct Agreement {
    Shape native, emulated;
    size_t native_dups = 0, emulated_dups = 0, native_events = 0, emulated_events = 0;
    size_t native_kept = 0, emulated_kept = 0;
};

Agreement compare_with_emulation(const std::vector<RuleSpec>& rules, const EdgeList& init,
                                 size_t steps, size_t threads) {
    Agreement a;
    {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::None);
        ParallelEvolutionEngine e(&hg, threads);
        e.set_undirected(true);
        for (size_t r = 0; r < rules.size(); ++r)
            e.add_rule(build_rule(rules[r].lhs, rules[r].rhs, static_cast<uint16_t>(r)));
        e.evolve(init, steps);
        std::vector<size_t> user(rules.size());
        std::iota(user.begin(), user.end(), 0);
        const Run run = collect(hg, rules, user);
        a.native_events = run.events;
        for (StateId root : run.roots) {
            Shape s = shape(run, root, a.native_dups, a.native_kept);
            a.native.insert(a.native.end(), s.begin(), s.end());
        }
    }
    {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::None);
        ParallelEvolutionEngine e(&hg, threads);
        std::vector<size_t> user;
        for (size_t r = 0; r < rules.size(); ++r)
            for (const Pattern& v : orientation_variants(rules[r].lhs)) {
                e.add_rule(build_rule(v, rules[r].rhs, static_cast<uint16_t>(user.size())));
                user.push_back(r);
            }
        e.evolve(init, steps);
        const Run run = collect(hg, rules, user);
        a.emulated_events = run.events;
        for (StateId root : run.roots) {
            Shape s = shape(run, root, a.emulated_dups, a.emulated_kept);
            a.emulated.insert(a.emulated.end(), s.begin(), s.end());
        }
    }
    return a;
}

EdgeList random_graph(uint32_t n_verts, uint32_t n_edges, uint32_t seed, bool loops) {
    std::mt19937 rng(seed);
    EdgeList out;
    while (out.size() < n_edges) {
        const VertexId a = rng() % n_verts, b = rng() % n_verts;
        if (a == b && !loops) continue;
        out.push_back({a, b});
    }
    return out;
}

bool isomorphic_brute(const EdgeList& a, const EdgeList& b) {
    auto verts = [](const EdgeList& g) {
        std::set<VertexId> s;
        for (const auto& e : g) s.insert(e.begin(), e.end());
        return std::vector<VertexId>(s.begin(), s.end());
    };
    const auto va = verts(a), vb = verts(b);
    if (va.size() != vb.size() || a.size() != b.size()) return false;
    auto sorted_edges = [](EdgeList g) {
        for (auto& e : g) e = sorted_edge(e);
        std::sort(g.begin(), g.end());
        return g;
    };
    const EdgeList target = sorted_edges(b);
    std::vector<size_t> perm(va.size());
    std::iota(perm.begin(), perm.end(), 0);
    do {
        std::map<VertexId, VertexId> m;
        for (size_t i = 0; i < va.size(); ++i) m[va[i]] = vb[perm[i]];
        EdgeList img = a;
        for (auto& e : img) for (auto& v : e) v = m[v];
        if (sorted_edges(img) == target) return true;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return false;
}

}  // namespace

// The symmetries the engine registers are the brute-force ones: Gorard's rule has none, the
// two-edge contraction its reversal, and a doubled edge four.
TEST(UndirectedCore, RuleSymmetriesMatchBruteForce) {
    const std::vector<RuleSpec> cases = {
        kGorard, kPathContract, kGrow, kWolfram,
        RuleSpec{{{0, 1}, {0, 1}}, {}},
        RuleSpec{{{0, 1}}, {}},
        RuleSpec{{{0, 0}, {0, 1}}, {{1, 1}}},
        RuleSpec{{{0}, {0, 1}}, {{1}, {0, 1}}},
    };
    for (const RuleSpec& r : cases) {
        const RewriteRule rule = build_rule(r.lhs, r.rhs, 0);
        hgcommon::UndirectedRuleInfo info;
        ASSERT_EQ(hgcommon::undirected_rule_info(rule.lhs, rule.num_lhs_edges, rule.rhs,
                                                 rule.num_rhs_edges, info),
                  hgcommon::UNDIRECTED_RULE_OK);
        const auto brute = rule_symmetries(r);
        EXPECT_EQ(info.num_aut + 1u, brute.size());
        std::set<std::vector<int>> engine_set, brute_set;
        for (uint32_t a = 0; a < info.num_aut; ++a) {
            std::vector<int> k;
            for (size_t v = 0; v < brute[0].var.size(); ++v) k.push_back(info.aut[a].var[v]);
            for (size_t i = 0; i < r.lhs.size(); ++i) k.push_back(info.aut[a].pos[i]);
            engine_set.insert(k);
        }
        for (const Symmetry& g : brute) {
            std::vector<int> k(g.var.begin(), g.var.end());
            for (size_t p : g.pos) k.push_back(static_cast<int>(p));
            bool identity = true;
            for (size_t v = 0; v < g.var.size(); ++v) identity = identity && g.var[v] == v;
            for (size_t i = 0; i < g.pos.size(); ++i) identity = identity && g.pos[i] == i;
            if (!identity) brute_set.insert(k);
        }
        EXPECT_EQ(engine_set, brute_set);
    }
}

// A graph and its copy with edges reversed and vertices renamed hash equally.
TEST(UndirectedHash, FlipInvariance) {
    std::mt19937 rng(7);
    for (uint32_t trial = 0; trial < 300; ++trial) {
        const uint32_t nv = 2 + rng() % 12;
        const EdgeList g = random_graph(nv, 1 + rng() % 20, trial, trial % 3 == 0);
        std::vector<VertexId> rename(nv);
        std::iota(rename.begin(), rename.end(), 100);
        std::shuffle(rename.begin(), rename.end(), rng);
        EdgeList h;
        for (const auto& e : g) {
            std::vector<VertexId> f{rename[e[0]], rename[e[1]]};
            if (rng() & 1u) std::swap(f[0], f[1]);
            h.push_back(f);
        }
        std::shuffle(h.begin(), h.end(), rng);
        ASSERT_EQ(undirected_hash(g), undirected_hash(h)) << "trial " << trial;
    }
}

// The hash separates the non-isomorphic graphs a brute-force test separates, on every pair of
// a set of small multigraphs with self-loops, and on the named pairs.
TEST(UndirectedHash, NonIsomorphicGraphsStayDistinct) {
    std::vector<EdgeList> graphs = {
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}},             // path of 5 edges
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {3, 5}},             // the other trees on 6 vertices
        {{0, 1}, {1, 2}, {2, 3}, {2, 4}, {2, 5}},
        {{0, 1}, {1, 2}, {1, 3}, {3, 4}, {3, 5}},
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {2, 5}},
        {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}},
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}},             // C5
        {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 3}},             // C3 + C2
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}},     // C6
        {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}},     // 2 C3
        {{0, 3}, {0, 4}, {0, 5}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 5}},   // K3,3
        {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}},   // prism
        {{0, 0}, {0, 1}, {1, 2}},
        {{0, 1}, {1, 1}, {1, 2}},
        {{0, 1}, {0, 1}, {1, 2}},
        {{0, 1}, {1, 2}, {1, 2}},
    };
    for (uint32_t s = 0; s < 120; ++s) graphs.push_back(random_graph(3 + s % 4, 2 + s % 5, 1000 + s, true));
    size_t iso_pairs = 0, distinct_pairs = 0;
    for (size_t i = 0; i < graphs.size(); ++i)
        for (size_t j = i + 1; j < graphs.size(); ++j) {
            const bool iso = isomorphic_brute(graphs[i], graphs[j]);
            const bool same = undirected_hash(graphs[i]) == undirected_hash(graphs[j]);
            ASSERT_EQ(iso, same) << "graphs " << i << " and " << j;
            (iso ? iso_pairs : distinct_pairs) += 1;
        }
    EXPECT_GT(iso_pairs, 0u);
    EXPECT_GT(distinct_pairs, 0u);
    // The directed canonicalisation tells a path and its reversal apart only by orientation;
    // the undirected one does not.
    const EdgeList p{{0, 1}, {1, 2}}, q{{0, 1}, {2, 1}};
    EXPECT_NE(IRCanonicalizer(false).compute_canonical_hash(p),
              IRCanonicalizer(false).compute_canonical_hash(q));
    EXPECT_EQ(undirected_hash(p), undirected_hash(q));
}

// The native run's multiway tree equals the emulation's with duplicate matches removed, and the
// native run has none.
TEST(UndirectedEngine, AgreesWithTheOrientationEmulation) {
    struct Case { std::vector<RuleSpec> rules; EdgeList init; size_t steps; };
    const std::vector<Case> cases = {
        {{kGorard}, {{0, 1}, {2, 1}, {2, 3}, {3, 4}, {0, 4}, {4, 2}, {5, 5}, {5, 0}}, 2},
        {{kGorard}, random_graph(7, 9, 3, true), 2},
        {{kPathContract}, {{0, 1}, {2, 1}, {2, 3}, {3, 3}, {3, 0}, {1, 0}}, 3},
        {{kGrow}, {{0, 1}, {1, 1}}, 4},
        {{kWolfram}, {{0, 1}, {0, 2}, {2, 2}}, 3},
        {{kGrow, kPathContract}, {{0, 1}, {2, 1}, {2, 2}}, 3},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        const Agreement a = compare_with_emulation(cases[c].rules, cases[c].init, cases[c].steps, 4);
        EXPECT_EQ(a.native_dups, 0u) << "case " << c;
        EXPECT_EQ(a.native, a.emulated) << "case " << c;
        EXPECT_GT(a.native_events, 0u) << "case " << c;
        EXPECT_EQ(a.native_kept, a.native_events) << "case " << c;
        EXPECT_EQ(a.emulated_kept, a.native_events) << "case " << c;
        EXPECT_GE(a.emulated_events, a.native_events) << "case " << c;
        std::printf("case %zu: native %zu events, emulation %zu events, %zu repeated matches "
                    "removed\n", c, a.native_events, a.emulated_events, a.emulated_dups);
    }
}

// The emulation counts a path matched by the contraction twice; the native run once.
TEST(UndirectedEngine, ASymmetricLeftSideCountsOnce) {
    const Agreement a = compare_with_emulation({kPathContract}, {{0, 1}, {1, 2}}, 1, 1);
    EXPECT_EQ(a.native_events, 1u);
    EXPECT_EQ(a.emulated_events, 2u);
    // A self-loop binds a two-variable edge once.
    const Agreement b = compare_with_emulation({kGrow}, {{0, 0}}, 1, 1);
    EXPECT_EQ(b.native_events, 1u);
    EXPECT_EQ(b.emulated_events, 2u);
}

namespace {

struct Trace {
    std::vector<std::vector<uint32_t>> events;
    std::vector<uint64_t> hashes;
};

Trace trace(const EdgeList& init, size_t steps, size_t threads, size_t per_step,
            StateCanonicalizationMode mode) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(mode);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_undirected(true);
    e.add_rule(build_rule(kGorard.lhs, kGorard.rhs, 0));
    if (per_step) {
        e.set_matches_per_step(per_step);
        e.set_random_seed(11);
    }
    e.evolve(init, steps);
    Trace t;
    for (EventId ev = 0; ev < hg.num_published_events(); ++ev) {
        const Event& x = hg.get_event(ev);
        if (x.id == INVALID_ID || hg.is_genesis_event(ev)) continue;
        std::vector<uint32_t> row{x.rule_index, hg.get_state(x.input_state).step};
        for (uint8_t i = 0; i < x.num_consumed; ++i) row.push_back(x.consumed_edges[i]);
        for (uint8_t i = 0; i < x.num_produced; ++i) row.push_back(x.produced_edges[i]);
        t.events.push_back(row);
    }
    std::sort(t.events.begin(), t.events.end());
    for (StateId s = 0; s < hg.num_published_states(); ++s)
        if (hg.get_state(s).id != INVALID_ID) t.hashes.push_back(hg.get_or_compute_canonical_hash(s));
    std::sort(t.hashes.begin(), t.hashes.end());
    return t;
}

}  // namespace

// One match per step for 60 steps, and the full multiway for 2 steps: the same events and
// states at 1, 2, 4 and 8 workers.
TEST(UndirectedEngine, IdenticalAcrossThreadCounts) {
    const EdgeList big = random_graph(20, 40, 5, false), small = random_graph(6, 7, 5, true);
    for (const size_t per_step : {size_t{1}, size_t{0}}) {
        const size_t steps = per_step ? 60 : 2;
        const EdgeList& g = per_step ? big : small;
        for (const auto mode : {StateCanonicalizationMode::None, StateCanonicalizationMode::Automatic,
                                StateCanonicalizationMode::Full}) {
            if (per_step && mode == StateCanonicalizationMode::Full) continue;
            const Trace one = trace(g, steps, 1, per_step, mode);
            EXPECT_FALSE(one.events.empty());
            for (const size_t threads : {size_t{2}, size_t{4}, size_t{8}}) {
                const Trace t = trace(g, steps, threads, per_step, mode);
                EXPECT_EQ(t.events.size(), one.events.size()) << threads << " workers";
                EXPECT_EQ(t.hashes, one.hashes) << threads << " workers";
                // One history has one id per edge at every worker count.
                if (per_step && mode == StateCanonicalizationMode::None)
                    EXPECT_EQ(t.events, one.events) << threads << " workers";
            }
        }
    }
}

// Under Full states the classes are the distinct undirected hashes of the None run's states, and
// a state and its copy with an edge reversed are one class.
TEST(UndirectedEngine, FullStatesAreTheUndirectedClasses) {
    for (const auto& rules : std::vector<std::vector<RuleSpec>>{{kGrow}, {kGrow, kPathContract},
                                                               {kWolfram}}) {
        std::set<uint64_t> none_hashes, full_hashes;
        size_t full_states = 0;
        for (const auto mode : {StateCanonicalizationMode::None, StateCanonicalizationMode::Full}) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(mode);
            ParallelEvolutionEngine e(&hg, 4);
            e.set_undirected(true);
            e.set_explore_from_canonical_states_only(mode == StateCanonicalizationMode::Full);
            for (size_t r = 0; r < rules.size(); ++r)
                e.add_rule(build_rule(rules[r].lhs, rules[r].rhs, static_cast<uint16_t>(r)));
            e.evolve(EdgeList{{0, 1}, {2, 1}, {2, 2}}, 3);
            std::set<StateId> classes;
            for (StateId s = 0; s < hg.num_published_states(); ++s) {
                if (hg.get_state(s).id == INVALID_ID) continue;
                const uint64_t h = undirected_hash(state_edges(hg, s));
                if (mode == StateCanonicalizationMode::None) none_hashes.insert(h);
                else { full_hashes.insert(h); classes.insert(hg.get_canonical_state(s)); }
            }
            if (mode == StateCanonicalizationMode::Full) full_states = classes.size();
        }
        EXPECT_EQ(full_hashes, none_hashes);
        EXPECT_EQ(full_states, none_hashes.size());
    }
}

// The matcher's emission: every match is emitted under the variant whose directed left side
// binds it from its edges (the join's reversed positions are the orientation mask), each match
// of a full scan and of a seeded scan from every edge is one of the brute-force bindings, and a
// full scan finds each of them once.
TEST(UndirectedMatcher, EmitsEachBindingOnceUnderItsVariant) {
    Hypergraph hg;
    const EdgeList g = {{0, 1}, {2, 1}, {2, 3}, {3, 3}, {3, 0}, {1, 0}, {4, 2}};
    std::vector<EdgeId> ids;
    for (const auto& e : g) ids.push_back(hg.create_edge({e[0], e[1]}));
    const StateId s = hg.create_state(ids.data(), static_cast<uint32_t>(ids.size()));
    const SparseBitset& edges = hg.get_state(s).edges;
    const AncestryCandidates cands{&hg, s, &edges};
    auto get_edge = [&](EdgeId e) -> const Edge& { return hg.get_edge(e); };
    for (const RuleSpec& spec : {kGorard, kPathContract, kGrow, kWolfram}) {
        const std::vector<RewriteRule> table =
            expand_undirected_rules({build_rule(spec.lhs, spec.rhs, 0)});
        // Every binding: an injective assignment of edges to positions, each in either order.
        std::set<std::vector<uint64_t>> expected;
        const size_t n = spec.lhs.size();
        std::vector<size_t> pick(n, 0);
        std::function<void(size_t, std::vector<uint64_t>&, std::vector<EdgeId>&)> enumerate =
            [&](size_t p, std::vector<uint64_t>& b, std::vector<EdgeId>& used) {
                if (p == n) {
                    std::vector<uint64_t> key(used.begin(), used.end());
                    key.insert(key.end(), b.begin(), b.begin() + 8);
                    expected.insert(key);
                    return;
                }
                for (EdgeId e : ids) {
                    if (std::find(used.begin(), used.end(), e) != used.end()) continue;
                    const Edge& ed = hg.get_edge(e);
                    for (int rev = 0; rev < 2; ++rev) {
                        if (rev && ed.vertices[0] == ed.vertices[1]) continue;
                        std::vector<uint64_t> nb = b;
                        bool ok = true;
                        for (size_t k = 0; k < 2 && ok; ++k) {
                            const uint64_t v = ed.vertices[rev ? 1 - k : k];
                            uint64_t& slot = nb[spec.lhs[p][k]];
                            if (slot == UINT64_MAX) slot = v;
                            else ok = slot == v;
                        }
                        if (!ok) continue;
                        used.push_back(e);
                        enumerate(p + 1, nb, used);
                        used.pop_back();
                    }
                }
            };
        std::vector<uint64_t> b0(32, UINT64_MAX);
        std::vector<EdgeId> used;
        enumerate(0, b0, used);
        auto check = [&](uint16_t rid, const EdgeId* m, uint8_t k, const VariableBinding& b,
                         std::multiset<std::vector<uint64_t>>& seen) {
            ASSERT_LT(rid, table.size());
            const RewriteRule& v = table[rid];
            std::vector<uint64_t> key(m, m + k);
            for (uint8_t p = 0; p < k; ++p)
                for (uint8_t q = 0; q < v.lhs[p].arity; ++q)
                    EXPECT_EQ(b.get(v.lhs[p].vars[q]), hg.get_edge(m[p]).vertices[q]);
            for (uint8_t var = 0; var < 8; ++var)
                key.push_back(b.is_bound(var) ? b.get(var) : UINT64_MAX);
            EXPECT_TRUE(expected.count(key)) << "a match no binding gives";
            seen.insert(key);
        };
        std::multiset<std::vector<uint64_t>> full;
        find_matches(table[0], 0, s, edges, cands, get_edge,
                     [&](uint16_t rid, const EdgeId* m, uint8_t k, const VariableBinding& b,
                         StateId) { check(rid, m, k, b, full); });
        // No rule here has a symmetry but the contraction, whose reversal pairs its bindings.
        const size_t orbit = table[0].und.num_aut + 1u;
        EXPECT_EQ(full.size() * orbit, expected.size());
        EXPECT_EQ(std::set<std::vector<uint64_t>>(full.begin(), full.end()).size(), full.size());
        for (EdgeId e : ids) {
            std::multiset<std::vector<uint64_t>> seeded;
            find_delta_matches(table[0], 0, s, edges, cands, get_edge,
                               [&](EdgeId x) { return EdgeSignature::from_edge(
                                                   hg.get_edge(x).vertices, hg.get_edge(x).arity); },
                               [&](uint16_t rid, const EdgeId* m, uint8_t k,
                                   const VariableBinding& b, StateId) { check(rid, m, k, b, seeded); },
                               &e, 1);
        }
    }
}

// Edges of three vertices are refused.
TEST(UndirectedEngine, RefusesEdgesOfThreeVertices) {
    Hypergraph hg;
    ParallelEvolutionEngine e(&hg, 1);
    e.set_undirected(true);
    e.add_rule(build_rule({{0, 1, 2}}, {{0, 1}}, 0));
    EXPECT_THROW(e.evolve(EdgeList{{0, 1, 2}}, 1), std::invalid_argument);
}
