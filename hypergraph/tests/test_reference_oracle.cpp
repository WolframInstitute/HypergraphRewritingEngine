#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

// Self-contained differential oracle for the engine's exact (Full-mode) state
// canonicalization — no Wolfram/WL dependency. It cross-checks the fast engine
// against a brute-force reference that is INDEPENDENT of the engine's WL/IR:
//
//   * Full mode  -> the engine deduplicates states by its exact IR canonical hash;
//                   num_canonical_states() is the engine's iso-distinct count.
//   * None mode  -> no deduplication: every rewrite yields a raw state, so the raw
//                   set covers all reachable states (with multiplicity). We then
//                   canonicalize each raw state by BRUTE FORCE (try every vertex
//                   relabelling, take the lexicographically minimal sorted edge
//                   list) and count the distinct forms.
//
// The two counts must agree: isomorphic states have isomorphic successors, so the
// set of reachable iso-classes is identical. A mismatch means the engine's
// canonicalization over/under-merges, or the two modes explore different iso-sets.

using namespace hypergraph;

namespace {

RewriteRule growth_rule() {
    return make_rule(0).lhs({0, 1}).rhs({0, 2}).rhs({1, 2}).build();
}

RewriteRule triangle_rule() {
    return make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).rhs({2, 3}).rhs({3, 0}).build();
}

// Independent brute-force canonical form of a directed-hyperedge set: relabel the
// vertices by every permutation, serialize the sorted edge list, keep the min.
// O(V! * E log E) — bounded to small states (returns "" as an over-size sentinel).
std::string brute_canonical(const std::vector<std::vector<uint32_t>>& edges) {
    std::set<uint32_t> vset;
    for (const auto& e : edges) for (uint32_t v : e) vset.insert(v);
    std::vector<uint32_t> verts(vset.begin(), vset.end());
    if (verts.size() > 8) return "";  // factorial guard; the test asserts this never trips
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

std::vector<std::vector<uint32_t>> state_edges(const Hypergraph& hg, StateId sid) {
    std::vector<std::vector<uint32_t>> out;
    const State& st = hg.get_state(sid);
    st.edges.for_each([&](EdgeId e) {
        const Edge& ed = hg.get_edge(e);
        out.emplace_back(ed.vertices, ed.vertices + ed.arity);
    });
    return out;
}

size_t engine_full_count(const std::vector<RewriteRule>& rules,
                         const std::vector<std::vector<VertexId>>& initial, int steps) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, 4);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);
    return engine.num_canonical_states();
}

// Brute-force iso-distinct count of the full raw exploration (None mode).
size_t brute_force_iso_count(const std::vector<RewriteRule>& rules,
                             const std::vector<std::vector<VertexId>>& initial, int steps,
                             bool* all_small) {
    Hypergraph hg;  // None mode (default): no dedup -> full raw state set
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

void expect_agrees(const std::vector<RewriteRule>& rules,
                   const std::vector<std::vector<VertexId>>& initial, int steps) {
    bool all_small = true;
    size_t brute = brute_force_iso_count(rules, initial, steps, &all_small);
    ASSERT_TRUE(all_small) << "state exceeded brute-force size bound (steps=" << steps << ")";
    size_t full = engine_full_count(rules, initial, steps);
    EXPECT_EQ(full, brute)
        << "engine Full-mode canonical count disagrees with brute-force oracle (steps=" << steps << ")";
}

}  // namespace

TEST(ReferenceOracle, GrowthRuleMatchesBruteForce) {
    for (int steps = 1; steps <= 3; ++steps)
        expect_agrees({growth_rule()}, {{0, 1}}, steps);
}

TEST(ReferenceOracle, TriangleRuleMatchesBruteForce) {
    for (int steps = 1; steps <= 2; ++steps)
        expect_agrees({triangle_rule()}, {{0, 1}, {1, 2}}, steps);
}
