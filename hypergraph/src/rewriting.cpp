#include <hypergraph/rewriting.hpp>
#include <algorithm>
#include <sstream>
#include <iostream>

namespace hypergraph {

std::vector<EdgeId> RewritingEngine::apply_rhs_pattern(
    Hypergraph& target,
    const PatternHypergraph& rhs,
    const VariableAssignment& assignment) const {
    
    std::vector<EdgeId> added_edges;
    
    for (const auto& pattern_edge : rhs.edges()) {
        std::vector<VertexId> concrete_vertices;
        
        // Resolve all pattern vertices to concrete vertices
        for (const auto& pattern_vertex : pattern_edge.vertices) {
            auto resolved = assignment.resolve(pattern_vertex);
            if (!resolved) {
                // This shouldn't happen if assignment is complete
                continue;
            }
            concrete_vertices.push_back(*resolved);
        }
        
        if (!concrete_vertices.empty()) {
            EdgeId new_edge = target.add_edge(concrete_vertices);
            added_edges.push_back(new_edge);
        }
    }
    
    return added_edges;
}

VariableAssignment RewritingEngine::extend_assignment_for_rhs(
    Hypergraph& target,
    const VariableAssignment& lhs_assignment,
    const PatternHypergraph& rhs) const {
    
    VariableAssignment extended = lhs_assignment;
    
    // Find RHS variables not already bound by LHS
    const auto& rhs_vars = rhs.variable_vertices();
    
    for (VertexId rhs_var : rhs_vars) {
        if (extended.variable_to_concrete.find(rhs_var) == extended.variable_to_concrete.end()) {
            // Create fresh vertex for unbound RHS variable
            VertexId fresh_vertex = target.create_vertex();
            extended.assign(rhs_var, fresh_vertex);
        }
    }
    
    return extended;
}

void RewritingEngine::remove_matched_edges(Hypergraph& target, const std::vector<EdgeId>& edge_ids) const {
    // Remove edges in reverse ID order to avoid invalidating IDs
    std::vector<EdgeId> sorted_edges = edge_ids;
    std::sort(sorted_edges.begin(), sorted_edges.end(), std::greater<EdgeId>());
    
    for (EdgeId edge_id : sorted_edges) {
        target.remove_edge(edge_id);
    }
}

void RewritingEngine::notify_rule_applied(const RewritingResult& result, const Hypergraph& hypergraph) const {
    for (const auto& listener : listeners_) {
        listener->on_rule_applied(result, hypergraph);
    }
}

void RewritingEngine::notify_rule_failed(const RewritingRule& rule, VertexId anchor_vertex) const {
    for (const auto& listener : listeners_) {
        listener->on_rule_failed(rule, anchor_vertex);
    }
}

void RewritingEngine::notify_step_completed(std::size_t step_number, const Hypergraph& hypergraph) const {
    for (const auto& listener : listeners_) {
        listener->on_step_completed(step_number, hypergraph);
    }
}

RewritingResult RewritingEngine::apply_rule_at(
    Hypergraph& target,
    const RewritingRule& rule,
    VertexId anchor_vertex,
    std::size_t search_radius) const {
    
    RewritingResult result;
    result.anchor_vertex = anchor_vertex;
    
    if (!rule.is_well_formed()) {
        notify_rule_failed(rule, anchor_vertex);
        return result;  // Rule has unbound RHS variables
    }
    
    // Find LHS matches around anchor vertex using provided radius
    auto matches = pattern_matcher_.find_matches_around(target, rule.lhs, anchor_vertex, search_radius);
    
    if (matches.empty()) {
        notify_rule_failed(rule, anchor_vertex);
        return result;  // No match found
    }
    
    // Use first match (could be randomized)
    const auto& match = matches[0];
    
    // Extend assignment for any RHS-only variables
    VariableAssignment extended_assignment = extend_assignment_for_rhs(target, match.assignment, rule.rhs);
    
    // Remove LHS edges
    remove_matched_edges(target, match.matched_edges);
    result.removed_edges = match.matched_edges;
    
    // Add RHS edges
    result.added_edges = apply_rhs_pattern(target, rule.rhs, extended_assignment);
    result.variable_assignment = extended_assignment;
    result.applied = true;
    
    notify_rule_applied(result, target);
    return result;
}

RewritingResult RewritingEngine::apply_first_rule_at(
    Hypergraph& target,
    const std::vector<RewritingRule>& rules,
    VertexId anchor_vertex,
    std::size_t search_radius) const {
    
    for (const auto& rule : rules) {
        RewritingResult result = apply_rule_at(target, rule, anchor_vertex, search_radius);
        if (result.was_applied()) {
            return result;
        }
    }
    
    // No rule could be applied
    RewritingResult failed_result;
    failed_result.anchor_vertex = anchor_vertex;
    return failed_result;
}

std::vector<RewritingResult> RewritingEngine::evolve_random(
    Hypergraph& target,
    const std::vector<RewritingRule>& rules,
    std::size_t num_steps,
    std::size_t search_radius) {
    
    std::vector<RewritingResult> results;
    results.reserve(num_steps);
    
    for (std::size_t step = 0; step < num_steps; ++step) {
        // Pick random vertex as anchor
        const auto& vertices = target.vertices();
        if (vertices.empty()) {
            break;  // No vertices to work with
        }
        
        std::uniform_int_distribution<std::size_t> vertex_dist(0, vertices.size() - 1);
        auto vertex_it = vertices.begin();
        std::advance(vertex_it, vertex_dist(rng_));
        VertexId anchor_vertex = *vertex_it;
        
        // Pick random rule
        if (rules.empty()) {
            break;
        }
        
        std::uniform_int_distribution<std::size_t> rule_dist(0, rules.size() - 1);
        const auto& rule = rules[rule_dist(rng_)];
        
        // Use provided search_radius parameter
        RewritingResult result = apply_rule_at(target, rule, anchor_vertex, search_radius);
        results.push_back(result);
        
        notify_step_completed(step + 1, target);
        
        // If no rule could be applied and hypergraph is small, might be stuck
        if (!result.was_applied() && target.num_vertices() < 3) {
            break;
        }
    }
    
    return results;
}

std::vector<RewritingResult> RewritingEngine::evolve_systematic(
    Hypergraph& target,
    const std::vector<RewritingRule>& rules,
    std::size_t max_steps,
    std::size_t search_radius) const {
    
    std::vector<RewritingResult> results;
    results.reserve(max_steps);
    
    std::size_t step = 0;
    
    while (step < max_steps) {
        bool any_applied = false;
        
        // Try each vertex as anchor
        auto vertices = target.vertices();  // Copy to avoid iterator invalidation
        for (VertexId anchor_vertex : vertices) {
            if (step >= max_steps) break;
            
            // Use provided search_radius parameter
            RewritingResult result = apply_first_rule_at(target, rules, anchor_vertex, search_radius);
            results.push_back(result);
            
            if (result.was_applied()) {
                any_applied = true;
            }
            
            notify_step_completed(++step, target);
        }
        
        if (!any_applied) {
            break;  // No rules could be applied anywhere
        }
    }
    
    return results;
}

std::vector<std::pair<RewritingRule, PatternMatch>> RewritingEngine::find_all_applications(
    const Hypergraph& target,
    const std::vector<RewritingRule>& rules,
    std::size_t search_radius) const {
    
    std::vector<std::pair<RewritingRule, PatternMatch>> applications;
    
    for (const auto& rule : rules) {
        if (!rule.is_well_formed()) continue;
        
        // Try each vertex as anchor
        // Use provided search_radius parameter
        for (VertexId anchor_vertex : target.vertices()) {
            auto matches = pattern_matcher_.find_matches_around(target, rule.lhs, anchor_vertex, search_radius);
            
            for (const auto& match : matches) {
                applications.emplace_back(rule, match);
            }
        }
    }
    
    return applications;
}

bool RewritingEngine::can_apply_any_rule(
    const Hypergraph& target,
    const std::vector<RewritingRule>& rules,
    std::size_t search_radius) const {
    
    for (const auto& rule : rules) {
        if (!rule.is_well_formed()) continue;
        
        // Use provided search_radius parameter
        for (VertexId anchor_vertex : target.vertices()) {
            auto matches = pattern_matcher_.find_matches_around(target, rule.lhs, anchor_vertex, search_radius);
            if (!matches.empty()) {
                return true;
            }
        }
    }
    
    return false;
}

std::vector<Hypergraph> RewritingEngine::get_multiway_states(
    const Hypergraph& initial_state,
    const std::vector<RewritingRule>& rules,
    std::size_t search_radius) const {
    
    std::vector<Hypergraph> states;
    
    // Use provided search_radius parameter
    auto applications = find_all_applications(initial_state, rules, search_radius);
    
    for (const auto& [rule, match] : applications) {
        Hypergraph state_copy = initial_state;
        
        // Apply this specific rule application
        VariableAssignment extended_assignment = extend_assignment_for_rhs(state_copy, match.assignment, rule.rhs);
        remove_matched_edges(state_copy, match.matched_edges);
        apply_rhs_pattern(state_copy, rule.rhs, extended_assignment);
        
        states.push_back(std::move(state_copy));
    }
    
    return states;
}


namespace debug {

std::string rule_to_string(const RewritingRule& rule) {
    std::ostringstream oss;
    oss << "Rule:\n";
    oss << "  LHS: " << rule.lhs.num_edges() << " edges, ";
    oss << rule.lhs.num_variable_vertices() << " variables, ";
    oss << rule.lhs.num_concrete_vertices() << " concrete vertices\n";
    oss << "  RHS: " << rule.rhs.num_edges() << " edges, ";
    oss << rule.rhs.num_variable_vertices() << " variables, ";
    oss << rule.rhs.num_concrete_vertices() << " concrete vertices\n";
    oss << "  Well-formed: " << (rule.is_well_formed() ? "yes" : "no");
    return oss.str();
}

std::string result_to_string(const RewritingResult& result) {
    std::ostringstream oss;
    oss << "RewritingResult:\n";
    oss << "  Applied: " << (result.applied ? "yes" : "no") << "\n";
    if (result.applied) {
        oss << "  Removed " << result.removed_edges.size() << " edges\n";
        oss << "  Added " << result.added_edges.size() << " edges\n";
        oss << "  Anchor vertex: " << result.anchor_vertex << "\n";
        oss << "  Variable assignments: " << result.variable_assignment.variable_to_concrete.size();
    }
    return oss.str();
}

class LoggingListener : public RewritingEventListener {
public:
    void on_rule_applied(const RewritingResult& result, const Hypergraph& hypergraph) override {
        std::cout << "✓ Applied rule at vertex " << result.anchor_vertex;
        std::cout << " (removed " << result.removed_edges.size() << ", added " << result.added_edges.size() << ")\n";
    }
    
    void on_rule_failed(const RewritingRule& rule, VertexId anchor_vertex) override {
        std::cout << "✗ Failed to apply rule at vertex " << anchor_vertex << "\n";
    }
    
    void on_step_completed(std::size_t step_number, const Hypergraph& hypergraph) override {
        std::cout << "Step " << step_number << ": " << hypergraph.num_vertices() << " vertices, " 
                  << hypergraph.num_edges() << " edges\n";
    }
};

std::unique_ptr<RewritingEventListener> create_logging_listener() {
    return std::make_unique<LoggingListener>();
}

} // namespace debug

} // namespace hypergraph