#ifndef HYPERGRAPH_REWRITING_HPP
#define HYPERGRAPH_REWRITING_HPP

#include <hypergraph/hypergraph.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <vector>
#include <unordered_set>
#include <random>
#include <memory>

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
 * Event listener interface for rewriting operations.
 */
class RewritingEventListener {
public:
    virtual ~RewritingEventListener() = default;

    virtual void on_rule_applied(const RewritingResult& result, const Hypergraph& hypergraph) {}
    virtual void on_rule_failed(const RewritingRule& rule, VertexId anchor_vertex) {}
    virtual void on_step_completed(std::size_t step_number, const Hypergraph& hypergraph) {}
};

/**
 * Single-threaded hypergraph rewriting engine.
 */
class RewritingEngine {
private:
    PatternMatcher pattern_matcher_;
    std::vector<std::unique_ptr<RewritingEventListener>> listeners_;
    std::mt19937 rng_;  // For random rule selection

    /**
     * Apply RHS pattern to hypergraph using variable assignment.
     */
    std::vector<EdgeId> apply_rhs_pattern(
        Hypergraph& target,
        const PatternHypergraph& rhs,
        const VariableAssignment& assignment) const;

    /**
     * Create fresh vertices for RHS variables not bound by LHS.
     */
    VariableAssignment extend_assignment_for_rhs(
        Hypergraph& target,
        const VariableAssignment& lhs_assignment,
        const PatternHypergraph& rhs) const;

    /**
     * Remove matched edges from hypergraph.
     */
    void remove_matched_edges(Hypergraph& target, const std::vector<EdgeId>& edge_ids) const;

    /**
     * Notify all listeners of an event.
     */
    void notify_rule_applied(const RewritingResult& result, const Hypergraph& hypergraph) const;
    void notify_rule_failed(const RewritingRule& rule, VertexId anchor_vertex) const;
    void notify_step_completed(std::size_t step_number, const Hypergraph& hypergraph) const;

public:
    RewritingEngine(unsigned int seed = std::random_device{}()) : rng_(seed) {}

    /**
     * Add event listener for rewriting operations.
     */
    void add_listener(std::unique_ptr<RewritingEventListener> listener) {
        listeners_.push_back(std::move(listener));
    }

    /**
     * Apply a single rewriting rule at a specific location.
     */
    RewritingResult apply_rule_at(
        Hypergraph& target,
        const RewritingRule& rule,
        VertexId anchor_vertex,
        std::size_t search_radius = 0) const;

    /**
     * Find and apply the first applicable rule at a location.
     */
    RewritingResult apply_first_rule_at(
        Hypergraph& target,
        const std::vector<RewritingRule>& rules,
        VertexId anchor_vertex,
        std::size_t search_radius = 0) const;

    /**
     * Apply rules randomly across the hypergraph for a number of steps.
     */
    std::vector<RewritingResult> evolve_random(
        Hypergraph& target,
        const std::vector<RewritingRule>& rules,
        std::size_t num_steps,
        std::size_t search_radius = 0);

    /**
     * Apply rules systematically (try every vertex as anchor).
     */
    std::vector<RewritingResult> evolve_systematic(
        Hypergraph& target,
        const std::vector<RewritingRule>& rules,
        std::size_t max_steps,
        std::size_t search_radius = 0) const;

    /**
     * Find all possible rule applications in the hypergraph.
     */
    std::vector<std::pair<RewritingRule, PatternMatch>> find_all_applications(
        const Hypergraph& target,
        const std::vector<RewritingRule>& rules,
        std::size_t search_radius = 0) const;

    /**
     * Check if any rules can be applied.
     */
    bool can_apply_any_rule(
        const Hypergraph& target,
        const std::vector<RewritingRule>& rules,
        std::size_t search_radius = 0) const;

    /**
     * Get multiway evolution states (apply all possible rules).
     */
    std::vector<Hypergraph> get_multiway_states(
        const Hypergraph& initial_state,
        const std::vector<RewritingRule>& rules,
        std::size_t search_radius = 0) const;

    /**
     * Set random seed for deterministic evolution.
     */
    void set_seed(unsigned int seed) {
        rng_.seed(seed);
    }
};

/**
 * Utility functions for creating common rewriting rules.
 */
namespace rules {
    /**
     * Create a simple substitution rule: pattern → replacement
     */
    RewritingRule create_substitution(
        const std::vector<std::vector<PatternVertex>>& lhs_edges,
        const std::vector<std::vector<PatternVertex>>& rhs_edges);

    /**
     * Create edge splitting rule: {A,B} → {A,X}, {X,B}
     */
    RewritingRule create_edge_split();

    /**
     * Create edge merging rule: {A,X}, {X,B} → {A,B} (where X has degree 2)
     */
    RewritingRule create_edge_merge();

    /**
     * Create triangle completion rule: {A,B}, {B,C} → {A,B}, {B,C}, {A,C}
     */
    RewritingRule create_triangle_completion();
}

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

    /**
     * Create a logging event listener.
     */
    std::unique_ptr<RewritingEventListener> create_logging_listener();
}

} // namespace hypergraph

#endif // HYPERGRAPH_REWRITING_HPP