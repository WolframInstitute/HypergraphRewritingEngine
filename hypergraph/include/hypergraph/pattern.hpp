#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include <cstring>
#include <vector>
#include <stdexcept>

#include "types.hpp"
#include "signature.hpp"

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// Constants
// =============================================================================

using hgcommon::MAX_PATTERN_EDGES;
// MAX_VARS is defined in types.hpp

// =============================================================================
// PatternEdge
// =============================================================================
// Represents a pattern edge in a rewrite rule.
// All positions are variables (no concrete vertices allowed).
//
// Example patterns:
//   {x, y}     → vars = [0, 1], arity = 2
//   {x, x}     → vars = [0, 0], arity = 2
//   {x, y, x}  → vars = [0, 1, 0], arity = 3

struct PatternEdge {
    uint8_t vars[MAX_ARITY];  // Variable index at each position
    uint8_t arity;

    // Default constructor - empty edge
    PatternEdge();

    // Construct from variable indices
    PatternEdge(std::initializer_list<uint8_t> var_list);

    // Construct from array
    PatternEdge(const uint8_t* var_array, uint8_t n);

    // Get variable at position
    uint8_t var_at(uint8_t pos) const;

    // Compute signature for this pattern edge
    EdgeSignature signature() const;

    // Get mask of variables used in this edge
    uint32_t var_mask() const;

    bool operator==(const PatternEdge& other) const;
    bool operator!=(const PatternEdge& other) const;
};

// =============================================================================
// RewriteRule
// =============================================================================
// Represents a rewrite rule: LHS pattern → RHS pattern
//
// Example: {{x, y}, {y, z}} → {{x, z}, {z, w}}
//   LHS: 2 edges, 3 variables (x, y, z)
//   RHS: 2 edges, 4 variables (x, z, w) - w is new
//
// Variables are numbered 0, 1, 2, ... in order of first appearance in LHS.
// RHS may introduce new variables (numbered after LHS variables).

// 8-byte aligned so that a copy or fill of the whole object (RuleBuilder::build, the rule
// table, the constructor's zeroing) moves 8 bytes at a time; the model checker lowers those
// copies to per-element stores at the object's alignment, and at the 2 bytes `index` alone
// would give, an 18,552-byte rule was 9,276 events per copy.
struct alignas(8) RewriteRule {
    uint16_t index;                        // Rule ID (for identification)
    PatternEdge lhs[MAX_PATTERN_EDGES];    // Left-hand side (pattern to match)
    uint8_t num_lhs_edges;                 // Number of edges in LHS
    PatternEdge rhs[MAX_PATTERN_EDGES];    // Right-hand side (replacement)
    uint8_t num_rhs_edges;                 // Number of edges in RHS
    uint8_t num_lhs_vars;                  // Total variables in LHS
    uint8_t num_rhs_vars;                  // Total variables in RHS (includes new)
    uint8_t num_new_vars;                  // Variables in RHS but not LHS

    // Join order for matching the LHS edges: a permutation of [0, num_lhs_edges).
    // match_order[k] is the original LHS edge index to match at join depth k. Chosen
    // (compute_match_order) so each edge shares a variable with the already-matched
    // prefix (connected join -> bound variables prune candidates, avoiding
    // cartesian-product blowup), seeded from the most self-constrained edge. Defaults
    // to identity (declaration order) until computed. Same matches, better search.
    uint8_t match_order[MAX_PATTERN_EDGES];

    // Per-rule precomputed matching data, indexed by ORIGINAL LHS edge index (the
    // same index space as lhs[]; match_order maps into it). Filled once by
    // compute_var_counts so per-task matching reads these instead of re-running the
    // recursive Bell-number set-partition enumeration in from_pattern per task.
    EdgeSignature lhs_sig[MAX_PATTERN_EDGES];
    CompatibleSignatureCache lhs_cache[MAX_PATTERN_EDGES];

    // Default constructor
    RewriteRule();

    // Get mask of all LHS variables
    uint32_t lhs_var_mask() const;

    // Get mask of all RHS variables
    uint32_t rhs_var_mask() const;

    // Get mask of new variables (in RHS but not LHS)
    uint32_t new_var_mask() const;

    // Check if two LHS edges share any variables (are "connected")
    bool lhs_edges_connected(uint8_t edge1, uint8_t edge2) const;

    // Compute variable counts from edge definitions
    // Body in pattern.cpp: runs once per rule at registration, never per state.
    void compute_var_counts();

    // Static self-constraint score for one LHS edge: higher => fewer candidate data
    // edges expected => better to match earlier. Repeated variables within the edge
    // (self-joins like {x,x}) are strongly constraining; being connected to more
    // other edges is weakly constraining.
    // Body in pattern.cpp: runs once per rule at registration, never per state.
    int edge_constraint_score(uint8_t e) const;

    // Choose a connected, constraint-seeded join order over the LHS edges. Same set
    // of matches as any order; the point is to bind variables early so later edges
    // draw from few candidates instead of the whole state (avoids O(product) blowup
    // for multi-edge rules). Deterministic (ties break to the lower edge index).
    // Body in pattern.cpp: runs once per rule at registration, never per state.
    void compute_match_order();
};

// =============================================================================
// Rule Builder
// =============================================================================
// Fluent interface for building rewrite rules

class RuleBuilder {
    RewriteRule rule_;

public:
    RuleBuilder() = default;

    explicit RuleBuilder(uint16_t index);

    // Add LHS edge (initializer list)
    RuleBuilder& lhs(std::initializer_list<uint8_t> vars);

    // Add LHS edge (vector - for dynamic construction)
    template<typename T>
    RuleBuilder& lhs(const std::vector<T>& vars) {
        if (rule_.num_lhs_edges >= MAX_PATTERN_EDGES) {
            throw std::length_error("RuleBuilder::lhs: exceeds MAX_PATTERN_EDGES");
        }
        if (vars.size() > MAX_ARITY) {
            throw std::length_error("RuleBuilder::lhs: edge arity exceeds MAX_ARITY");
        }
        // A pattern variable is BOTH an index into VariableBinding's MAX_VARS-entry array and
        // a bit position in its 32-bit bound_mask, so an out-of-range one writes out of bounds
        // and shifts past the type's width -- memory corruption, not a wrong answer. The arity
        // and edge-count limits above were enforced; this one was declared and never checked,
        // so every direct C++ caller (tools, tests, embedders) could trip it.
        for (const T& v : vars) {
            if (static_cast<uint64_t>(v) >= MAX_VARS) {
                throw std::length_error(
                    "RuleBuilder::lhs: pattern variable index exceeds MAX_VARS");
            }
        }
        PatternEdge edge;
        edge.arity = static_cast<uint8_t>(vars.size());
        for (uint8_t i = 0; i < edge.arity; ++i) {
            edge.vars[i] = static_cast<uint8_t>(vars[i]);
        }
        rule_.lhs[rule_.num_lhs_edges++] = edge;
        return *this;
    }

    // Add RHS edge (initializer list)
    RuleBuilder& rhs(std::initializer_list<uint8_t> vars);

    // Add RHS edge (vector - for dynamic construction)
    template<typename T>
    RuleBuilder& rhs(const std::vector<T>& vars) {
        if (rule_.num_rhs_edges >= MAX_PATTERN_EDGES) {
            throw std::length_error("RuleBuilder::rhs: exceeds MAX_PATTERN_EDGES");
        }
        if (vars.size() > MAX_ARITY) {
            throw std::length_error("RuleBuilder::rhs: edge arity exceeds MAX_ARITY");
        }
        // A pattern variable is BOTH an index into VariableBinding's MAX_VARS-entry array and
        // a bit position in its 32-bit bound_mask, so an out-of-range one writes out of bounds
        // and shifts past the type's width -- memory corruption, not a wrong answer. The arity
        // and edge-count limits above were enforced; this one was declared and never checked,
        // so every direct C++ caller (tools, tests, embedders) could trip it.
        for (const T& v : vars) {
            if (static_cast<uint64_t>(v) >= MAX_VARS) {
                throw std::length_error(
                    "RuleBuilder::rhs: pattern variable index exceeds MAX_VARS");
            }
        }
        PatternEdge edge;
        edge.arity = static_cast<uint8_t>(vars.size());
        for (uint8_t i = 0; i < edge.arity; ++i) {
            edge.vars[i] = static_cast<uint8_t>(vars[i]);
        }
        rule_.rhs[rule_.num_rhs_edges++] = edge;
        return *this;
    }

    // Build and return the rule
    RewriteRule build();
};

// Convenience function
RuleBuilder make_rule(uint16_t index = 0);

// =============================================================================
// PartialMatch
// =============================================================================
// Represents a partially completed match - some pattern edges have been
// matched, waiting for more edges to complete the match.
//
// Used for incremental matching when new edges are added.

using PartialMatchId = uint32_t;

struct PartialMatch {
    uint32_t id;
    uint16_t rule_index;
    uint8_t num_matched;                     // How many pattern edges matched
    uint8_t num_pattern_edges;               // Total pattern edges to match
    uint8_t match_order[MAX_PATTERN_EDGES];  // Order we're matching in
    EdgeId matched_edges[MAX_PATTERN_EDGES]; // Data edges matched so far
    VariableBinding binding;                  // Current variable bindings
    StateId origin_state;                     // State where this partial started

    PartialMatch();

    // Check if all pattern edges are matched
    bool is_complete() const;

    // Check if complete (with rule parameter for backwards compatibility)
    bool is_complete(const RewriteRule& rule) const;

    // Add an edge match (for simple sequential matching)
    void add_match(uint8_t pattern_idx, EdgeId data_edge, const VariableBinding& new_binding);

    // Create a copy for branching during depth-first expansion.
    PartialMatch branch() const;

    // Check if data edge is already used
    bool contains_edge(EdgeId eid) const;

    // Convert to edges array in pattern order
    void to_pattern_order(EdgeId* out) const;
};

}  // namespace engine
}  // namespace HG_NAMESPACE