#ifndef HYPERGRAPH_REWRITING_HPP
#define HYPERGRAPH_REWRITING_HPP

#include <hypergraph/hypergraph.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <vector>
#include <unordered_set>

namespace hypergraph {

/**
 * Rewriting rule: LHS pattern → RHS pattern
 * When LHS matches in a hypergraph, it gets replaced by RHS with variable substitution.
 */
struct RewritingRule {
    PatternHypergraph lhs;  // Left-hand side pattern to match
    PatternHypergraph rhs;  // Right-hand side pattern to substitute

    RewritingRule(const PatternHypergraph& left, const PatternHypergraph& right)
        : lhs(left), rhs(right) {}

    // Get all variables used in the rule (union of LHS and RHS variables)
    std::unordered_set<VertexId> get_all_variables() const {
        std::unordered_set<VertexId> vars = lhs.variable_vertices();
        const auto& rhs_vars = rhs.variable_vertices();
        vars.insert(rhs_vars.begin(), rhs_vars.end());
        return vars;
    }

    // Check if rule is well-formed (RHS variables can be fresh - they create new vertices)
    bool is_well_formed() const {
        // A rule is well-formed if it has valid structure
        // RHS variables not in LHS represent fresh vertices to be created
        return true;  // All rules with valid patterns are well-formed
    }
};

/**
 * Result of applying a rewriting rule.
 */
struct RewritingResult {
    bool applied;                           // Whether rule was successfully applied
    std::vector<EdgeId> removed_edges;      // Edges removed from hypergraph
    std::vector<EdgeId> added_edges;        // Edges added to hypergraph
    VariableAssignment variable_assignment; // Variable bindings used
    VertexId anchor_vertex;                 // Vertex around which rewriting occurred

    RewritingResult() : applied(false), anchor_vertex(INVALID_VERTEX) {}

    bool was_applied() const { return applied; }
    std::size_t num_changes() const { return removed_edges.size() + added_edges.size(); }
};



/**
 * Debug utilities for rewriting.
 */
namespace debug {
    /**
     * Print rewriting rule in human-readable format.
     */
    std::string rule_to_string(const RewritingRule& rule);

    /**
     * Print rewriting result summary.
     */
    std::string result_to_string(const RewritingResult& result);

}

} // namespace hypergraph

#endif // HYPERGRAPH_REWRITING_HPP