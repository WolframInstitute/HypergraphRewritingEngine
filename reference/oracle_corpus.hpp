#pragma once
// Shared measurement/verification substrate: a diverse rule corpus plus a
// brute-force isomorphism oracle that is INDEPENDENT of the engine's WL/IR. Used
// by both the oracle gate (a gtest that proves exactness across rule types) and
// the cost harness (tools/cost_matrix.cpp, which proves memory/compute wins). One
// source of truth for "what rules we test and how we check them."
//
// The corpus deliberately spans the rule-type space the engine must handle:
// single- and mixed-arity, varying edge counts and connectivity, and productive /
// idempotent / reductive dynamics, plus self-loops and disconnected LHS.

#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace oracle {

using namespace hypergraph;

// Independent brute-force canonical form of a directed-hyperedge set: relabel by
// every vertex permutation, serialize the sorted edge list, keep the lexicographic
// minimum. O(V! * E log E); returns "" as an over-size sentinel (>8 vertices).
inline std::string brute_canonical(const std::vector<std::vector<uint32_t>>& edges) {
    std::set<uint32_t> vset;
    for (const auto& e : edges) for (uint32_t v : e) vset.insert(v);
    std::vector<uint32_t> verts(vset.begin(), vset.end());
    if (verts.size() > 8) return "";
    std::map<uint32_t, uint32_t> dense;
    for (size_t i = 0; i < verts.size(); ++i) dense[verts[i]] = static_cast<uint32_t>(i);

    std::vector<uint32_t> perm(verts.size());
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<uint32_t>(i);

    std::string best;
    do {
        std::vector<std::vector<uint32_t>> relabeled;
        relabeled.reserve(edges.size());
        for (const auto& e : edges) {
            std::vector<uint32_t> re;
            re.reserve(e.size());
            for (uint32_t v : e) re.push_back(perm[dense[v]]);
            relabeled.push_back(std::move(re));
        }
        std::sort(relabeled.begin(), relabeled.end());
        std::string s;
        for (const auto& e : relabeled) {
            s += '(';
            for (uint32_t v : e) { s += std::to_string(v); s += ','; }
            s += ')';
        }
        if (best.empty() || s < best) best = std::move(s);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

inline std::vector<std::vector<uint32_t>> state_edges(const Hypergraph& hg, StateId sid) {
    std::vector<std::vector<uint32_t>> out;
    const State& st = hg.get_state(sid);
    st.edges.for_each([&](EdgeId e) {
        const Edge& ed = hg.get_edge(e);
        out.emplace_back(ed.vertices, ed.vertices + ed.arity);
    });
    return out;
}

// Engine's exact iso-distinct count via IR (Full mode).
inline size_t engine_full_count(const std::vector<RewriteRule>& rules,
                                const std::vector<std::vector<VertexId>>& initial,
                                int steps, unsigned threads = 4) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, threads);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);
    return engine.num_canonical_states();
}

// Brute-force iso-distinct count of the full raw exploration (None mode).
inline size_t brute_force_iso_count(const std::vector<RewriteRule>& rules,
                                    const std::vector<std::vector<VertexId>>& initial,
                                    int steps, bool* all_small) {
    Hypergraph hg;  // None mode: no dedup -> full raw state set
    ParallelEvolutionEngine engine(&hg, 1);  // single thread -> no wasted states
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);

    std::set<std::string> distinct;
    *all_small = true;
    for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
        auto edges = state_edges(hg, sid);
        if (edges.empty()) continue;
        std::string c = brute_canonical(edges);
        if (c.empty()) { *all_small = false; continue; }
        distinct.insert(std::move(c));
    }
    return distinct.size();
}

// One workload: a named, typed rule set + initial condition, with a small
// oracle-checkable depth and a deeper measurement depth.
struct Case {
    const char* name;
    const char* type;   // dynamics/shape tags
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> init;
    int oracle_steps;   // brute-force cross-check depth (keeps states <= 8 vertices)
    int measure_steps;  // deeper depth for memory/compute measurement
};

inline std::vector<Case> corpus() {
    std::vector<Case> c;
    auto R = [](RewriteRule r) { return std::vector<RewriteRule>{std::move(r)}; };

    // Productive, single arity-2 (the canonical growth rule).
    c.push_back({"binary-growth", "productive/arity2",
                 R(make_rule(0).lhs({0,1}).rhs({0,2}).rhs({1,2}).build()),
                 {{0,1}}, 3, 6});
    // Productive, arity-2, 2->4 (the standard Wolfram rule).
    c.push_back({"wolfram-2to4", "productive/arity2",
                 R(make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()),
                 {{0,1},{0,2}}, 2, 4});
    // Productive, mixed connectivity (path LHS, triangle RHS).
    c.push_back({"triangle", "productive/mixed-conn",
                 R(make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,3}).rhs({3,0}).build()),
                 {{0,1},{1,2}}, 2, 4});
    // Reductive, 2->1 (shrinks edge count).
    c.push_back({"reductive-2to1", "reductive/arity2",
                 R(make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()),
                 {{0,1},{1,2},{2,3},{3,4}}, 3, 6});
    // Idempotent, 2->2 (same edge count).
    c.push_back({"idempotent-2to2", "idempotent/arity2",
                 R(make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build()),
                 {{0,1},{1,2}}, 3, 6});
    // Self-loop LHS.
    c.push_back({"self-loop", "self-loop/arity2",
                 R(make_rule(0).lhs({0,0}).rhs({0,1}).rhs({1,0}).build()),
                 {{0,0}}, 3, 6});
    // Mixed arity: arity-2 LHS producing an arity-3 edge plus a re-matchable arity-2.
    c.push_back({"mixed-arity", "mixed-arity",
                 R(make_rule(0).lhs({0,1}).rhs({0,1,2}).rhs({2,0}).build()),
                 {{0,1}}, 3, 6});
    // Pure arity-3.
    c.push_back({"arity3-growth", "arity3",
                 R(make_rule(0).lhs({0,1,2}).rhs({0,1,3}).rhs({1,2,3}).build()),
                 {{0,1,2}}, 3, 5});
    // Disconnected LHS (two independent edges).
    c.push_back({"disconnected-lhs", "disconnected",
                 R(make_rule(0).lhs({0,1}).lhs({2,3}).rhs({0,2}).rhs({1,3}).build()),
                 {{0,1},{2,3}}, 3, 6});
    // Multi-rule (two productive rules together).
    c.push_back({"multi-rule", "multi-rule",
                 std::vector<RewriteRule>{
                     make_rule(0).lhs({0,1}).rhs({0,2}).rhs({1,2}).build(),
                     make_rule(1).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()},
                 {{0,1},{1,2}}, 2, 4});
    return c;
}

}  // namespace oracle
