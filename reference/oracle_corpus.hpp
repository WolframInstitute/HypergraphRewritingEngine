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
inline size_t engine_full_count(
    const std::vector<RewriteRule>& rules,
    const std::vector<std::vector<VertexId>>& initial,
    int steps, unsigned threads = 4,
    ParallelEvolutionEngine::ExecutionMode mode = ParallelEvolutionEngine::ExecutionMode::Parallel) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, threads, mode);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);
    return engine.num_canonical_states();
}

// Independent content-ordered canonical form: the edge tuples as they stand, in edge order,
// serialized. No relabelling -- Automatic identifies states by content, not up to isomorphism.
//
// Independent of the engine in the way that matters: it compares the CONTENT, where the engine
// compares a hash of the content. A collision, or a field hashed that should not have been,
// shows up here and cannot show up in a check that hashes the same way.
inline std::string content_canonical(const std::vector<std::vector<uint32_t>>& edges) {
    std::string s;
    for (const auto& e : edges) {
        s += '(';
        for (uint32_t v : e) { s += std::to_string(v); s += ','; }
        s += ')';
    }
    return s;
}

// The refinement lattice, checked rather than assumed.
//
// None keeps every raw state; Automatic merges those with equal content; Full merges those that
// are isomorphic. So quotienting the None-mode state set by an INDEPENDENT content map must
// give exactly Automatic's state count, and by an independent isomorphism map exactly Full's.
// Dedup only ever merges states with identical futures, so pruning re-expansion cannot change
// which classes are reachable -- the counts have to agree.
//
// This is what gives the non-Full cells an oracle at all. The brute-force isomorphism count
// answers only for Full, because isomorphism is what Full MEANS.
struct LatticeCounts {
    size_t raw;            // None-mode states
    size_t by_content;     // distinct under the independent content map
    size_t by_iso;         // distinct under the independent isomorphism map
    bool   all_small;      // false if any state exceeded the brute force's vertex bound
};

inline LatticeCounts brute_force_lattice(const std::vector<RewriteRule>& rules,
                                         const std::vector<std::vector<VertexId>>& initial,
                                         int steps) {
    Hypergraph hg;  // None mode: no dedup -> the full raw state set
    ParallelEvolutionEngine engine(&hg, 1);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);

    LatticeCounts lc{0, 0, 0, true};
    std::set<std::string> content, iso;
    for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
        auto edges = state_edges(hg, sid);
        if (edges.empty()) continue;
        ++lc.raw;
        content.insert(content_canonical(edges));
        std::string c = brute_canonical(edges);
        if (c.empty()) { lc.all_small = false; continue; }
        iso.insert(std::move(c));
    }
    lc.by_content = content.size();
    lc.by_iso = iso.size();
    return lc;
}

// Brute-force iso-distinct count of the full raw exploration (None mode).
inline size_t brute_force_iso_count(
    const std::vector<RewriteRule>& rules,
    const std::vector<std::vector<VertexId>>& initial,
    int steps, bool* all_small,
    ParallelEvolutionEngine::ExecutionMode mode = ParallelEvolutionEngine::ExecutionMode::Parallel) {
    Hypergraph hg;  // None mode: no dedup -> full raw state set
    // One thread -> no wasted states. The MODE is separate: a target without threads must be
    // able to run the oracle too, and a threaded engine there fails in its constructor.
    ParallelEvolutionEngine engine(&hg, 1, mode);
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

// Graph invariants of a full evolution — all independent of event/state id
// assignment, so they must be IDENTICAL across thread counts (determinism) and are
// the quantities every causal/closure optimization must preserve.
struct Counts {
    size_t canonical_states;
    size_t events;
    size_t causal_edges;
    size_t causal_event_pairs;
    size_t branchial_edges;
};

inline Counts engine_counts(const std::vector<RewriteRule>& rules,
                            const std::vector<std::vector<VertexId>>& initial,
                            int steps, unsigned threads,
                            StateCanonicalizationMode mode = StateCanonicalizationMode::Full,
                            bool transitive_reduction = true) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(mode);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_transitive_reduction(transitive_reduction);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);
    Counts c;
    c.canonical_states    = hg.num_canonical_states();
    c.events              = hg.num_events();
    c.causal_edges        = hg.causal_graph().num_causal_edges();
    c.causal_event_pairs  = hg.causal_graph().num_causal_event_pairs();
    c.branchial_edges     = hg.causal_graph().num_branchial_edges();
    return c;
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

    // ---- the arity axis ----------------------------------------------------------------
    //
    // MAX_ARITY is 16 and the cases above reach 3, all with edges of ONE arity per state and
    // per rule side. Arity is not a passive parameter: it sets the width of every signature,
    // the branching of the match join, and the vertex-tuple layout, and a hyperedge of arity 4
    // is not merely a longer arity-2 edge. Rules and initial states that MIX arities are the
    // case least like anything already covered, since a matcher can be right on uniform input
    // by construction and wrong the moment two arities have to be told apart.

    // Pure arity 4: past every corpus case, and past the arity-3 special-casing that a
    // hand-tuned matcher tends to accumulate.
    c.push_back({"arity4-growth", "arity4",
                 R(make_rule(0).lhs({0,1,2,3}).rhs({0,1,2,4}).rhs({1,2,3,4}).build()),
                 {{0,1,2,3}}, 3, 4});

    // LHS mixing arities: the join has to bind across edges of different widths, so a
    // candidate filter keyed on a single arity cannot pass this.
    c.push_back({"mixed-arity-lhs", "mixed-arity/join",
                 R(make_rule(0).lhs({0,1}).lhs({0,1,2}).rhs({0,2}).rhs({1,2,3}).build()),
                 {{0,1},{0,1,2}}, 3, 5});

    // Initial state mixing arities, with a rule that consumes either: state identity and the
    // signature index now have to separate edges by arity as well as by content.
    c.push_back({"mixed-arity-init", "mixed-arity/init",
                 R(make_rule(0).lhs({0,1}).rhs({0,1,2}).build()),
                 {{0,1},{1,2,3},{2,3}}, 3, 5});

    // Arity-REDUCING: 3 -> 2. The reductive cases above shrink the edge count at fixed arity;
    // this shrinks the arity itself, which is the direction that leaves stale wider tuples
    // behind if anything indexes on the old width.
    c.push_back({"arity3-to-2", "arity-reductive",
                 R(make_rule(0).lhs({0,1,2}).rhs({0,1}).rhs({1,2}).build()),
                 {{0,1,2},{1,2,3}}, 3, 6});

    // Unary edges alongside binary: arity 1 is the degenerate end, where an edge has no
    // internal structure to distinguish and everything rests on which vertex it names.
    c.push_back({"arity1-with-binary", "arity1/mixed",
                 R(make_rule(0).lhs({0}).lhs({0,1}).rhs({1}).rhs({0,1}).rhs({0,2}).build()),
                 {{0},{0,1}}, 3, 5});
    // AUTOMORPHISM axis. Every case above starts from a graph whose canonical labelling is a
    // single labelling, which is the case where an edge's canonical RANK is well defined. On a
    // state with a nontrivial automorphism group the canonical labelling is a coset: the
    // canonical form and the state hash stay unique, but interchangeable edges can take each
    // other's positions, so a rank is defined only up to that group. Event identity modes that
    // key on consumed or produced edges read those ranks, which makes this the state class
    // where the event axis and the state axis can disagree about stability.
    //
    // A directed 4-cycle: rotation group of order 4. Directed, so the reflections that would
    // make an undirected cycle's group dihedral are not automorphisms here.
    c.push_back({"cycle4-automorphic", "automorphism",
                 R(make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).build()),
                 {{0,1},{1,2},{2,3},{3,0}}, 3, 6});

    // A state whose automorphism group acts on VERTICES rather than on a cycle: a star, where
    // every leaf is interchangeable. The symmetric group on the leaves is larger than any cycle
    // group, so if orbit size is what governs rank stability this is the harder case.
    c.push_back({"star4-automorphic", "automorphism",
                 R(make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build()),
                 {{0,1},{0,2},{0,3},{0,4}}, 2, 6});

    // Multi-rule (two productive rules together).
    c.push_back({"multi-rule", "multi-rule",
                 std::vector<RewriteRule>{
                     make_rule(0).lhs({0,1}).rhs({0,2}).rhs({1,2}).build(),
                     make_rule(1).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()},
                 {{0,1},{1,2}}, 2, 4});
    return c;
}

}  // namespace oracle
