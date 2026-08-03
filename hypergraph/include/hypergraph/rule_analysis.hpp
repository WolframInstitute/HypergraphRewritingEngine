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
    // The LHS as a conjunctive query: acyclic admits a join order with no intermediate blow-up.
    bool acyclic = true;
    // Integral edge cover; N^edge_cover bounds the matches one state of N edges can yield.
    uint32_t edge_cover = 0;
};

// IS THE LHS AN ACYCLIC CONJUNCTIVE QUERY? (GYO reduction, Graham-Yu-Ozsoyoglu.)
//
// The LHS is a conjunctive query over the state's edge relation: each pattern edge is an atom and
// each variable a join attribute. Alpha-acyclicity is the property that decides how the join can
// behave -- an acyclic query admits an order in which no intermediate result exceeds the final
// one, which is why join order matters at all and why P5's join-order work paid off where it did.
// A CYCLIC query has no such order, and its intermediate blow-up is a property of the query
// rather than of the schedule.
//
// GYO is the decision procedure and it is exact: repeatedly remove an EAR -- a vertex appearing
// in only one edge, or an edge contained in another -- until nothing changes. Everything gone
// means acyclic. Both operations are confluent, so the order of removals cannot change the
// answer and no search is needed.
inline bool lhs_is_acyclic(const RewriteRule& r) {
    uint32_t edge_vars[MAX_PATTERN_EDGES] = {0};   // bitset of variables per edge
    bool alive[MAX_PATTERN_EDGES] = {false};
    const uint8_t n = r.num_lhs_edges;
    for (uint8_t i = 0; i < n; ++i) {
        alive[i] = true;
        for (uint8_t j = 0; j < r.lhs[i].arity; ++j) {
            const uint8_t v = r.lhs[i].vars[j];
            if (v < 32) edge_vars[i] |= (1u << v);
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;

        // (a) Drop a variable that only one live edge mentions: nothing joins on it.
        for (uint8_t v = 0; v < 32; ++v) {
            const uint32_t bit = 1u << v;
            int holder = -1, count = 0;
            for (uint8_t i = 0; i < n; ++i)
                if (alive[i] && (edge_vars[i] & bit)) { holder = i; if (++count > 1) break; }
            if (count == 1) { edge_vars[holder] &= ~bit; changed = true; }
        }

        // (b) Drop an edge with nothing left to constrain. Stripping (a) empties an edge whose
        // variables were all its own, and a single-edge query reaches exactly that state -- it is
        // trivially acyclic, and a rule requiring ANOTHER edge to contain it would never say so.
        for (uint8_t i = 0; i < n; ++i)
            if (alive[i] && edge_vars[i] == 0u) { alive[i] = false; changed = true; }

        // (c) Drop an edge whose variables another live edge already carries. It constrains
        // nothing the other does not.
        for (uint8_t i = 0; i < n && !changed; ++i) {
            if (!alive[i]) continue;
            for (uint8_t k = 0; k < n; ++k) {
                if (k == i || !alive[k]) continue;
                if ((edge_vars[i] & ~edge_vars[k]) == 0u) {   // i subset of k
                    alive[i] = false; changed = true; break;
                }
            }
        }
    }

    for (uint8_t i = 0; i < n; ++i) if (alive[i]) return false;
    return true;
}

// The smallest set of LHS edges whose variables cover every LHS variable.
//
// The AGM bound says a conjunctive query over relations of size N has at most N^(rho*) results,
// where rho* is the FRACTIONAL edge cover number. This computes the INTEGRAL one, which is >=
// rho*, so N^cover is a valid -- and weaker -- upper bound on the matches one state can yield.
// Exact by subset enumeration: MAX_PATTERN_EDGES is 16, so the search is at most 65536 subsets
// and needs no LP.
inline uint32_t lhs_edge_cover(const RewriteRule& r) {
    const uint8_t n = r.num_lhs_edges;
    if (n == 0) return 0;
    uint32_t edge_vars[MAX_PATTERN_EDGES] = {0}, all = 0;
    for (uint8_t i = 0; i < n; ++i)
        for (uint8_t j = 0; j < r.lhs[i].arity; ++j) {
            const uint8_t v = r.lhs[i].vars[j];
            if (v < 32) { edge_vars[i] |= (1u << v); all |= (1u << v); }
        }
    uint32_t best = n;
    for (uint32_t mask = 1; mask < (1u << n); ++mask) {
        uint32_t covered = 0, used = 0;
        for (uint8_t i = 0; i < n; ++i)
            if (mask & (1u << i)) { covered |= edge_vars[i]; ++used; }
        if (used < best && covered == all) best = used;
    }
    return best;
}

inline RuleFacts analyze_rule(const RewriteRule& r) {
    RuleFacts f;
    f.lhs_edges = r.num_lhs_edges;
    f.edge_delta = static_cast<int>(r.num_rhs_edges) - static_cast<int>(r.num_lhs_edges);
    for (uint8_t i = 0; i < r.num_lhs_edges; ++i) f.lhs_arities.push_back(r.lhs[i].arity);
    // The rule already carries this: num_new_vars is variables in the RHS and not the LHS, which
    // is the vertex creation rate. Recomputing it here would be a second implementation of a
    // rule the pattern builder already decides.
    f.new_vertices = r.num_new_vars;
    f.acyclic = lhs_is_acyclic(r);
    f.edge_cover = lhs_edge_cover(r);
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
