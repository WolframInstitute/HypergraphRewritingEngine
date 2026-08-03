#pragma once

// What can be decided about a rule set BEFORE running it.
//
// End-to-end evolution time over all rules and all initial states is the target, so a fact
// available at add_rule time for the cost of reading the patterns is worth having. This header
// holds only facts that are DECIDABLE and cheap -- a bounded scan over two finite patterns. It
// deliberately holds nothing about termination, reachable-set finiteness, or global confluence,
// all of which are undecidable for a Turing-complete rewriting system and where cleverness does
// not help.
//
// EVERY PREDICATE HERE IS SOUND IN ONE DIRECTION AND SAYS SO. `can_branch` over-approximates:
// false means branching is structurally impossible, true means it is not ruled out. An
// adaptation may act on the false and must never act on the true, because the true is "unknown".
// Getting that backwards would drop real structure, which is why the direction is stated at every
// predicate rather than left to the caller to remember.

#include <cstdint>
#include <vector>

#include "hypergraph/pattern.hpp"
#include "hypergraph/types.hpp"

namespace hypergraph {

// Facts about ONE rule, each computable from its two patterns.
struct RuleFacts {
    // |RHS| - |LHS| in edges. Positive grows the state, negative shrinks it, zero is neutral.
    // Predicts state-size growth, which is what drives canonicalization cost.
    int edge_delta = 0;
    // Vertices appearing in the RHS and not the LHS: the rule's vertex creation rate. A rule
    // creating none runs inside a bounded vertex set, which permits a denser layout.
    uint32_t new_vertices = 0;
    // Edges in the LHS. A rule whose LHS is a single edge cannot produce two DISTINCT matches
    // that share an edge -- a match IS that edge, so two different matches are two different
    // edges. This is the whole of the branching argument below.
    uint32_t lhs_edges = 0;
    // The arities present in the LHS, one entry per edge, in pattern order.
    std::vector<uint32_t> lhs_arities;
};

inline RuleFacts analyze_rule(const RewriteRule& r) {
    RuleFacts f;
    f.lhs_edges = r.num_lhs_edges;
    f.edge_delta = static_cast<int>(r.num_rhs_edges) - static_cast<int>(r.num_lhs_edges);
    for (uint8_t i = 0; i < r.num_lhs_edges; ++i) f.lhs_arities.push_back(r.lhs[i].arity);
    // The rule already carries this: num_new_vars is variables in the RHS and not the LHS, which
    // is the vertex creation rate. Recomputing it here would be a second implementation of a
    // rule the pattern builder already decides.
    f.new_vertices = r.num_new_vars;
    return f;
}

// CAN TWO DISTINCT MATCHES OF THIS RULE SET SHARE A CONSUMED EDGE?
//
// That question is exactly the branchial relation's: a branchial edge joins two events out of one
// state that consumed a common edge. If no two matches can share an edge then no state has such a
// pair, the branchial relation is empty for every initial condition, and the work of building it
// is work with a provably empty answer.
//
// SOUND IN THE FALSE DIRECTION ONLY. Two cases make sharing possible:
//
//   SAME rule, two distinct matches. They must differ somewhere, so the LHS needs at least two
//   edges: with one, a match is that edge and two different matches are two different edges.
//   With two or more, m1 = {e, f} and m2 = {e, g} is a shape the pattern admits -- whether any
//   reachable state actually contains it is a reachability question, and undecidable, so this
//   answers "not ruled out".
//
//   DIFFERENT rules a and b. One edge can satisfy an edge of each pattern as long as the two have
//   the same arity, and then m_a and m_b share it. Arity is the only structural obstacle: pattern
//   vertices are variables, so any two same-arity edges unify.
//
// Returning true therefore means "not ruled out", never "will happen".
inline bool can_branch(const std::vector<RewriteRule>& rules) {
    for (size_t a = 0; a < rules.size(); ++a) {
        if (rules[a].num_lhs_edges >= 2) return true;   // one rule, two matches, shared edge
        for (size_t b = a + 1; b < rules.size(); ++b) {
            for (uint8_t i = 0; i < rules[a].num_lhs_edges; ++i)
                for (uint8_t j = 0; j < rules[b].num_lhs_edges; ++j)
                    if (rules[a].lhs[i].arity == rules[b].lhs[j].arity)
                        return true;                    // two rules, one edge satisfies both
        }
    }
    return false;
}

// Facts about the SET, which is what a strategy is chosen from.
struct RuleSetFacts {
    // False means the branchial relation is empty for EVERY initial condition. True means it was
    // not ruled out; see can_branch for why the two are not symmetric.
    bool may_branch = true;
    // Every rule shrinks or holds the edge count, so states stay bounded by the initial size.
    bool non_growing = false;
    // No rule introduces a vertex, so the vertex set is bounded by the initial condition.
    bool bounded_vertices = false;
};

inline RuleSetFacts analyze_rules(const std::vector<RewriteRule>& rules) {
    RuleSetFacts s;
    s.may_branch = can_branch(rules);
    s.non_growing = true;
    s.bounded_vertices = true;
    for (const auto& r : rules) {
        const RuleFacts f = analyze_rule(r);
        if (f.edge_delta > 0) s.non_growing = false;
        if (f.new_vertices > 0) s.bounded_vertices = false;
    }
    return s;
}

}  // namespace hypergraph
