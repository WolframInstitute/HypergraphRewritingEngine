#pragma once

#include <cstdint>
#include <cstring>

#include "types.hpp"
#include "signature.hpp"

namespace hypergraph::unified {

// =============================================================================
// Constants
// =============================================================================

constexpr uint8_t MAX_PATTERN_EDGES = 16;
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
    PatternEdge() : arity(0) {
        std::memset(vars, 0, MAX_ARITY);
    }

    // Construct from variable indices
    PatternEdge(std::initializer_list<uint8_t> var_list) : arity(0) {
        std::memset(vars, 0, MAX_ARITY);
        for (uint8_t v : var_list) {
            if (arity < MAX_ARITY) {
                vars[arity++] = v;
            }
        }
    }

    // Construct from array
    PatternEdge(const uint8_t* var_array, uint8_t n) : arity(n) {
        std::memset(vars, 0, MAX_ARITY);
        for (uint8_t i = 0; i < n && i < MAX_ARITY; ++i) {
            vars[i] = var_array[i];
        }
    }

    // Get variable at position
    uint8_t var_at(uint8_t pos) const {
        return vars[pos];
    }

    // Compute signature for this pattern edge
    EdgeSignature signature() const {
        return EdgeSignature::from_pattern(vars, arity);
    }

    // Get mask of variables used in this edge
    uint32_t var_mask() const {
        uint32_t mask = 0;
        for (uint8_t i = 0; i < arity; ++i) {
            mask |= (1u << vars[i]);
        }
        return mask;
    }

    // Count distinct variables
    uint8_t num_distinct_vars() const {
        return static_cast<uint8_t>(__builtin_popcount(var_mask()));
    }

    // Check if variable appears in this edge
    bool contains_var(uint8_t var) const {
        for (uint8_t i = 0; i < arity; ++i) {
            if (vars[i] == var) return true;
        }
        return false;
    }

    // Get positions where a variable appears
    uint8_t get_var_positions(uint8_t var, uint8_t* positions_out) const {
        uint8_t count = 0;
        for (uint8_t i = 0; i < arity; ++i) {
            if (vars[i] == var) {
                positions_out[count++] = i;
            }
        }
        return count;
    }

    bool operator==(const PatternEdge& other) const {
        if (arity != other.arity) return false;
        for (uint8_t i = 0; i < arity; ++i) {
            if (vars[i] != other.vars[i]) return false;
        }
        return true;
    }

    bool operator!=(const PatternEdge& other) const {
        return !(*this == other);
    }
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

struct RewriteRule {
    uint16_t index;                        // Rule ID (for identification)
    PatternEdge lhs[MAX_PATTERN_EDGES];    // Left-hand side (pattern to match)
    uint8_t num_lhs_edges;                 // Number of edges in LHS
    PatternEdge rhs[MAX_PATTERN_EDGES];    // Right-hand side (replacement)
    uint8_t num_rhs_edges;                 // Number of edges in RHS
    uint8_t num_lhs_vars;                  // Total variables in LHS
    uint8_t num_rhs_vars;                  // Total variables in RHS (includes new)
    uint8_t num_new_vars;                  // Variables in RHS but not LHS

    // Default constructor
    RewriteRule()
        : index(0)
        , num_lhs_edges(0)
        , num_rhs_edges(0)
        , num_lhs_vars(0)
        , num_rhs_vars(0)
        , num_new_vars(0)
    {}

    // Get mask of all LHS variables
    uint32_t lhs_var_mask() const {
        uint32_t mask = 0;
        for (uint8_t i = 0; i < num_lhs_edges; ++i) {
            mask |= lhs[i].var_mask();
        }
        return mask;
    }

    // Get mask of all RHS variables
    uint32_t rhs_var_mask() const {
        uint32_t mask = 0;
        for (uint8_t i = 0; i < num_rhs_edges; ++i) {
            mask |= rhs[i].var_mask();
        }
        return mask;
    }

    // Get mask of new variables (in RHS but not LHS)
    uint32_t new_var_mask() const {
        return rhs_var_mask() & ~lhs_var_mask();
    }

    // Get mask of preserved variables (in both LHS and RHS)
    uint32_t preserved_var_mask() const {
        return lhs_var_mask() & rhs_var_mask();
    }

    // Get mask of deleted variables (in LHS but not RHS)
    uint32_t deleted_var_mask() const {
        return lhs_var_mask() & ~rhs_var_mask();
    }

    // Check if two LHS edges share any variables (are "connected")
    bool lhs_edges_connected(uint8_t edge1, uint8_t edge2) const {
        return (lhs[edge1].var_mask() & lhs[edge2].var_mask()) != 0;
    }

    // Compute variable counts from edge definitions
    void compute_var_counts() {
        uint32_t lhs_mask = lhs_var_mask();
        uint32_t rhs_mask = rhs_var_mask();
        uint32_t new_mask = rhs_mask & ~lhs_mask;

        num_lhs_vars = static_cast<uint8_t>(__builtin_popcount(lhs_mask));
        num_rhs_vars = static_cast<uint8_t>(__builtin_popcount(rhs_mask));
        num_new_vars = static_cast<uint8_t>(__builtin_popcount(new_mask));
    }
};

// =============================================================================
// Rule Builder
// =============================================================================
// Fluent interface for building rewrite rules

class RuleBuilder {
    RewriteRule rule_;

public:
    RuleBuilder() = default;

    explicit RuleBuilder(uint16_t index) {
        rule_.index = index;
    }

    // Add LHS edge
    RuleBuilder& lhs(std::initializer_list<uint8_t> vars) {
        if (rule_.num_lhs_edges < MAX_PATTERN_EDGES) {
            rule_.lhs[rule_.num_lhs_edges++] = PatternEdge(vars);
        }
        return *this;
    }

    // Add RHS edge
    RuleBuilder& rhs(std::initializer_list<uint8_t> vars) {
        if (rule_.num_rhs_edges < MAX_PATTERN_EDGES) {
            rule_.rhs[rule_.num_rhs_edges++] = PatternEdge(vars);
        }
        return *this;
    }

    // Build and return the rule
    RewriteRule build() {
        rule_.compute_var_counts();
        return rule_;
    }
};

// Convenience function
inline RuleBuilder make_rule(uint16_t index = 0) {
    return RuleBuilder(index);
}

// =============================================================================
// MatchIdentity
// =============================================================================
// Uniquely identifies a match by its rule and edge mapping.
// Two matches are the same iff they have the same rule and map the same
// pattern edges to the same data edges.
//
// Note: Edges are stored in PATTERN order, not match order.

struct MatchIdentity {
    uint16_t rule_index;
    EdgeId edges[MAX_PATTERN_EDGES];  // In pattern order
    uint8_t num_edges;

    MatchIdentity() : rule_index(0), num_edges(0) {
        std::memset(edges, 0xFF, sizeof(edges));
    }

    MatchIdentity(uint16_t rule, const EdgeId* edge_array, uint8_t n)
        : rule_index(rule), num_edges(n) {
        std::memset(edges, 0xFF, sizeof(edges));
        for (uint8_t i = 0; i < n; ++i) {
            edges[i] = edge_array[i];
        }
    }

    // Hash for ConcurrentMap
    uint64_t hash() const {
        uint64_t h = 14695981039346656037ULL;
        h ^= rule_index;
        h *= 1099511628211ULL;
        for (uint8_t i = 0; i < num_edges; ++i) {
            h ^= edges[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    bool operator==(const MatchIdentity& other) const {
        if (rule_index != other.rule_index) return false;
        if (num_edges != other.num_edges) return false;
        for (uint8_t i = 0; i < num_edges; ++i) {
            if (edges[i] != other.edges[i]) return false;
        }
        return true;
    }

    bool operator!=(const MatchIdentity& other) const {
        return !(*this == other);
    }
};

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

    PartialMatch()
        : id(INVALID_ID)
        , rule_index(0)
        , num_matched(0)
        , num_pattern_edges(0)
        , binding()
        , origin_state(INVALID_ID)
    {
        std::memset(match_order, 0, sizeof(match_order));
        std::memset(matched_edges, 0xFF, sizeof(matched_edges));
    }

    // Check if all pattern edges are matched
    bool is_complete() const {
        return num_matched == num_pattern_edges;
    }

    // Check if complete (with rule parameter for backwards compatibility)
    bool is_complete(const RewriteRule& rule) const {
        return num_matched == rule.num_lhs_edges;
    }

    // Which pattern edge index we need to match next (in match order)
    uint8_t next_match_order_idx() const {
        return num_matched;
    }

    // Which pattern edge (original index) we need to match next
    uint8_t next_pattern_edge_idx() const {
        return match_order[num_matched];
    }

    // Simple sequential next pattern index (ignores match_order)
    uint8_t next_pattern_idx() const {
        // Find first unmatched pattern index
        bool matched_flags[MAX_PATTERN_EDGES] = {};
        for (uint8_t i = 0; i < num_matched; ++i) {
            uint8_t pidx = match_order[i];
            if (pidx < MAX_PATTERN_EDGES) {
                matched_flags[pidx] = true;
            }
        }
        for (uint8_t i = 0; i < num_pattern_edges; ++i) {
            if (!matched_flags[i]) return i;
        }
        return num_pattern_edges;
    }

    // Get signature needed for next pattern edge
    EdgeSignature next_needed_signature(const RewriteRule& rule) const {
        if (num_matched >= rule.num_lhs_edges) {
            return EdgeSignature{};  // No more edges needed
        }
        uint8_t pattern_idx = match_order[num_matched];
        return rule.lhs[pattern_idx].signature();
    }

    // Add an edge match (for simple sequential matching)
    void add_match(uint8_t pattern_idx, EdgeId data_edge, const VariableBinding& new_binding) {
        match_order[num_matched] = pattern_idx;
        matched_edges[num_matched] = data_edge;
        binding = new_binding;
        num_matched++;
    }

    // Create extended partial match with one more edge matched
    PartialMatch extend(EdgeId new_edge, const VariableBinding& new_binding) const {
        PartialMatch extended = *this;
        extended.matched_edges[num_matched] = new_edge;
        extended.num_matched = num_matched + 1;
        extended.binding = new_binding;
        return extended;
    }

    // Create a copy for branching
    PartialMatch branch() const {
        return *this;
    }

    // Check if data edge is already used
    bool contains_edge(EdgeId eid) const {
        for (uint8_t i = 0; i < num_matched; ++i) {
            if (matched_edges[i] == eid) return true;
        }
        return false;
    }

    // Convert to edges array in pattern order
    void to_pattern_order(EdgeId* out) const {
        std::memset(out, 0xFF, MAX_PATTERN_EDGES * sizeof(EdgeId));
        for (uint8_t i = 0; i < num_matched; ++i) {
            uint8_t pattern_idx = match_order[i];
            out[pattern_idx] = matched_edges[i];
        }
    }

    // Convert to MatchIdentity (reorder edges to pattern order)
    MatchIdentity to_identity([[maybe_unused]] const RewriteRule& rule) const {
        MatchIdentity mid;
        mid.rule_index = rule_index;
        mid.num_edges = num_matched;

        // matched_edges is in match_order, convert to pattern order
        for (uint8_t i = 0; i < num_matched; ++i) {
            uint8_t pattern_idx = match_order[i];
            mid.edges[pattern_idx] = matched_edges[i];
        }
        return mid;
    }
};

}  // namespace hypergraph::unified
