#pragma once
#include "hgcommon/namespace.hpp"

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

namespace HG_NAMESPACE {
namespace engine {

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
    // Whether that bound is TIGHT -- i.e. edge_cover is rho* and not merely above it. True
    // exactly when the query is acyclic; see edge_cover_is_tight.
    bool edge_cover_tight = true;
};

// Does the LHS hold together through shared variables?
//
// The matcher's join order is connected by construction where it can be (compute_match_order
// appends the edge sharing the most variables with the bound prefix). When no edge shares any,
// the step it takes is a CARTESIAN PRODUCT over the state's edges: correct, and quadratic in
// the state size per additional component. The rule set decides that, so it is worth saying so
// once rather than discovering it as a run that does not come back.
//
// Reachability over the pairwise predicate the rule already carries, from edge 0.
// Body in src/rule_analysis.cpp: run once per rule, never per state.
bool lhs_is_connected(const RewriteRule& r);

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
// Body in src/rule_analysis.cpp: run once per rule, never per state.
bool lhs_is_acyclic(const RewriteRule& r);

// The smallest set of LHS edges whose variables cover every LHS variable.
//
// The AGM bound says a conjunctive query over relations of size N has at most N^(rho*) results,
// where rho* is the FRACTIONAL edge cover number. This computes the INTEGRAL one, which is >=
// rho*, so N^cover is a valid -- and weaker -- upper bound on the matches one state can yield.
// Exact by subset enumeration: MAX_PATTERN_EDGES is 16, so the search is at most 65536 subsets
// and needs no LP.
// Body in src/rule_analysis.cpp: run once per rule, never per state.
uint32_t lhs_edge_cover(const RewriteRule& r);

// Body in src/rule_analysis.cpp: run once per rule, never per state.
RuleFacts analyze_rule(const RewriteRule& r);

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
// Body in src/rule_analysis.cpp: run once per rule, never per state.
bool can_branch(const std::vector<RewriteRule>& rules);

// IS THE INTEGRAL COVER ALREADY rho*?
//
// The AGM bound is N^(rho*), where rho* is the FRACTIONAL edge cover number -- the LP relaxation
// of lhs_edge_cover. Solving that LP is only necessary when the two differ, and for an
// ALPHA-ACYCLIC query they do not: an acyclic hypergraph has the integrality property, so its
// fractional cover LP attains its optimum at an integral point and rho* == lhs_edge_cover
// exactly. A cyclic query is where they separate, and the separation is real -- the triangle has
// cover 2 and rho* 3/2 (x = 1/2 on each of the three edges), and a k-cycle has rho* k/2.
//
// So on any acyclic rule the bound this header already computes is TIGHT, and the LP is needed
// only for cyclic ones. Every rule in the shipped corpus is acyclic, which is why no LP is
// implemented here: it would have no input to run on and no way to be checked. When a cyclic rule
// arrives, the closed forms above are the test it must reproduce.
inline bool edge_cover_is_tight(const RewriteRule& r) { return lhs_is_acyclic(r); }

// Facts about the SET, which is what a strategy is chosen from.
struct RuleSetFacts {
    // False means the branchial relation is empty for EVERY initial condition. True means it was
    // not ruled out; see can_branch for why the two are not symmetric.
    bool may_branch = true;
    // Every rule shrinks or holds the edge count, so states stay bounded by the initial size.
    bool non_growing = false;
    // No rule introduces a vertex, so the vertex set is bounded by the initial condition.
    bool bounded_vertices = false;
    // A rule whose LHS has three or more edges AND a cycle: no binary join plan is
    // worst-case-optimal on it (the triangle over N binary edges has at most N^1.5 matches by
    // AGM, while binding two atoms first reaches N^2), and the matcher runs a binary plan.
    bool has_cyclic_multiedge_lhs = false;
    // A rule whose LHS falls into two or more components joined by nothing.
    bool has_disconnected_lhs = false;
    // Some rule's LHS joins two or more edges, so re-matching a child is a join rather than an
    // index scan. That is the property match forwarding has to beat: forwarding replaces the
    // re-match with a walk of the ancestor's records plus the coordination that keeps the walk
    // complete, and against identical event counts it costs +22% / +45% on single-edge rule sets
    // while paying 19% on a multi-edge one.
    bool forwarding_pays = false;
};

// Only the SET facts, and only from what they need.
//
// This runs on the evolve path (configure_identity_and_quotient reads may_branch), so it must not
// pay for facts nobody asked for. analyze_rule additionally runs the GYO reduction and a 2^n
// edge-cover enumeration; calling it here cost 0.036% of the end-to-end corpus total for two
// numbers this function does not use. The three fields below come from counters the rule already
// carries.
// Body in src/rule_analysis.cpp: run once per rule, never per state.
RuleSetFacts analyze_rules(const std::vector<RewriteRule>& rules);

}  // namespace engine
}  // namespace HG_NAMESPACE