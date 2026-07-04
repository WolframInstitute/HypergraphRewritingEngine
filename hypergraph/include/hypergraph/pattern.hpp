#pragma once

#include <cstdint>
#include <cstring>

#include "types.hpp"
#include "signature.hpp"

namespace hypergraph {

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
    PatternEdge() : arity(0) {
        std::memset(vars, 0, MAX_ARITY);
    }

    // Construct from variable indices
    PatternEdge(std::initializer_list<uint8_t> var_list) : arity(0) {
        if (var_list.size() > MAX_ARITY) {
            throw std::length_error("PatternEdge: arity exceeds MAX_ARITY");
        }
        std::memset(vars, 0, MAX_ARITY);
        for (uint8_t v : var_list) {
            vars[arity++] = v;
        }
    }

    // Construct from array
    PatternEdge(const uint8_t* var_array, uint8_t n) : arity(n) {
        if (n > MAX_ARITY) {
            throw std::length_error("PatternEdge: arity exceeds MAX_ARITY");
        }
        std::memset(vars, 0, MAX_ARITY);
        for (uint8_t i = 0; i < n; ++i) {
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
    RewriteRule()
        : index(0)
        , num_lhs_edges(0)
        , num_rhs_edges(0)
        , num_lhs_vars(0)
        , num_rhs_vars(0)
        , num_new_vars(0)
    {
        for (uint8_t i = 0; i < MAX_PATTERN_EDGES; ++i) match_order[i] = i;
    }

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

        compute_match_order();

        // Precompute per-edge signature + compatible-signature cache once, so match
        // tasks never repeat the from_pattern Bell enumeration. Indexed by original
        // LHS edge index (see lhs_sig / lhs_cache).
        for (uint8_t i = 0; i < num_lhs_edges; ++i) {
            lhs_sig[i] = lhs[i].signature();
            lhs_cache[i] = CompatibleSignatureCache::from_pattern(lhs_sig[i]);
        }
    }

    // Static self-constraint score for one LHS edge: higher => fewer candidate data
    // edges expected => better to match earlier. Repeated variables within the edge
    // (self-joins like {x,x}) are strongly constraining; being connected to more
    // other edges is weakly constraining.
    int edge_constraint_score(uint8_t e) const {
        int score = 0;
        for (uint8_t i = 0; i < lhs[e].arity; ++i)
            for (uint8_t j = static_cast<uint8_t>(i + 1); j < lhs[e].arity; ++j)
                if (lhs[e].var_at(i) == lhs[e].var_at(j)) score += 100;
        for (uint8_t o = 0; o < num_lhs_edges; ++o)
            if (o != e && lhs_edges_connected(e, o)) score += 1;
        return score;
    }

    // Choose a connected, constraint-seeded join order over the LHS edges. Same set
    // of matches as any order; the point is to bind variables early so later edges
    // draw from few candidates instead of the whole state (avoids O(product) blowup
    // for multi-edge rules). Deterministic (ties break to the lower edge index).
    void compute_match_order() {
        for (uint8_t i = 0; i < MAX_PATTERN_EDGES; ++i) match_order[i] = i;
        if (num_lhs_edges <= 1) return;

        bool used[MAX_PATTERN_EDGES] = {};

        // Seed with the most self-constrained edge.
        uint8_t first = 0;
        int best = -1;
        for (uint8_t e = 0; e < num_lhs_edges; ++e) {
            int s = edge_constraint_score(e);
            if (s > best) { best = s; first = e; }
        }
        match_order[0] = first;
        used[first] = true;
        uint32_t bound = lhs[first].var_mask();

        // Greedily append the unmatched edge sharing the most variables with the
        // bound prefix; tie-break by self-constraint, then lower index.
        for (uint8_t pos = 1; pos < num_lhs_edges; ++pos) {
            uint8_t pick = 0;
            int pick_shared = -1, pick_self = -1;
            for (uint8_t e = 0; e < num_lhs_edges; ++e) {
                if (used[e]) continue;
                int shared = __builtin_popcount(lhs[e].var_mask() & bound);
                int self = edge_constraint_score(e);
                if (shared > pick_shared || (shared == pick_shared && self > pick_self)) {
                    pick = e; pick_shared = shared; pick_self = self;
                }
            }
            match_order[pos] = pick;
            used[pick] = true;
            bound |= lhs[pick].var_mask();
        }
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

    // Add LHS edge (initializer list)
    RuleBuilder& lhs(std::initializer_list<uint8_t> vars) {
        if (rule_.num_lhs_edges >= MAX_PATTERN_EDGES) {
            throw std::length_error("RuleBuilder::lhs: exceeds MAX_PATTERN_EDGES");
        }
        rule_.lhs[rule_.num_lhs_edges++] = PatternEdge(vars);
        return *this;
    }

    // Add LHS edge (vector - for dynamic construction)
    template<typename T>
    RuleBuilder& lhs(const std::vector<T>& vars) {
        if (rule_.num_lhs_edges >= MAX_PATTERN_EDGES) {
            throw std::length_error("RuleBuilder::lhs: exceeds MAX_PATTERN_EDGES");
        }
        if (vars.size() > MAX_ARITY) {
            throw std::length_error("RuleBuilder::lhs: edge arity exceeds MAX_ARITY");
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
    RuleBuilder& rhs(std::initializer_list<uint8_t> vars) {
        if (rule_.num_rhs_edges >= MAX_PATTERN_EDGES) {
            throw std::length_error("RuleBuilder::rhs: exceeds MAX_PATTERN_EDGES");
        }
        rule_.rhs[rule_.num_rhs_edges++] = PatternEdge(vars);
        return *this;
    }

    // Add RHS edge (vector - for dynamic construction)
    template<typename T>
    RuleBuilder& rhs(const std::vector<T>& vars) {
        if (rule_.num_rhs_edges >= MAX_PATTERN_EDGES) {
            throw std::length_error("RuleBuilder::rhs: exceeds MAX_PATTERN_EDGES");
        }
        if (vars.size() > MAX_ARITY) {
            throw std::length_error("RuleBuilder::rhs: edge arity exceeds MAX_ARITY");
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

    // Add an edge match (for simple sequential matching)
    void add_match(uint8_t pattern_idx, EdgeId data_edge, const VariableBinding& new_binding) {
        match_order[num_matched] = pattern_idx;
        matched_edges[num_matched] = data_edge;
        binding = new_binding;
        num_matched++;
    }

    // Create a copy for branching during depth-first expansion.
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

}  // namespace hypergraph
