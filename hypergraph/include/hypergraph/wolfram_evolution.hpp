#ifndef HYPERGRAPH_WOLFRAM_EVOLUTION_HPP
#define HYPERGRAPH_WOLFRAM_EVOLUTION_HPP

#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/pattern_matching_tasks.hpp>
#include <hypergraph/debug_log.hpp>
#include <job_system/job_system.hpp>
#include <memory>
#include <vector>
#include <atomic>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <unordered_map>

namespace hypergraph {

/**
 * Wolfram Physics evolution using task-parallel hypergraph rewriting.
 * Follows HGMatch paper specification: SCAN→EXPAND→SINK→REWRITE pipeline.
 */
class WolframEvolution {
private:
    std::shared_ptr<MultiwayGraph> multiway_graph_;
    std::unique_ptr<job_system::JobSystem<PatternMatchingTaskType>> job_system_;
    std::vector<RewritingRule> rules_;
    std::size_t computed_radius_{0};  // Computed from rule set
    std::atomic<std::size_t> current_step_{0};
    mutable std::atomic<std::size_t> total_matches_found_{0};  // Global counter for all pattern matches across all rules
    std::size_t max_steps_;
    std::size_t num_threads_;

    // Store evolution context objects to keep them alive
    std::vector<std::shared_ptr<Hypergraph>> target_graphs_;
    std::vector<std::shared_ptr<EdgeSignatureIndex>> signature_indices_;
    std::vector<std::shared_ptr<PatternMatchingContext>> contexts_;
    
public:
    WolframEvolution(std::size_t max_steps, std::size_t num_threads = std::thread::hardware_concurrency(),
                     bool canonicalize_states = true, bool full_capture = false,
                     bool canonicalize_events = false, bool deduplicate_events = false,
                     bool transitive_reduction_enabled = true,
                     bool early_termination = false, bool full_capture_non_canonicalised = false,
                     std::size_t max_successor_states_per_parent = 0, std::size_t max_states_per_step = 0,
                     double exploration_probability = 1.0,
                     bool full_event_canonicalization = true);

    ~WolframEvolution();
    
    /**
     * Add rewriting rule.
     */
    void add_rule(const RewritingRule& rule);
    
    /**
     * Initialize radius from the loaded rule set.
     * Should be called after all rules are added and before evolution starts.
     */
    void initialize_radius_config();
    
    /**
     * Get the computed radius for this evolution system.
     */
    std::size_t get_computed_radius() const {
        return computed_radius_ ? computed_radius_ : 3;  // fallback to 3 if not computed
    }
    
    /**
     * Submit initial state and start evolution.
     * Blocks until completion or step limit reached.
     *
     * @param pattern_matching_only If true, only perform pattern matching without rewriting or event relationships.
     *                               Useful for benchmarking pattern matching performance in isolation.
     */
    void evolve(const std::vector<std::vector<GlobalVertexId>>& initial_edges, bool pattern_matching_only = false);
    
    /**
     * Get results.
     */
    const MultiwayGraph& get_multiway_graph() const {
        return *multiway_graph_;
    }

    /**
     * Get mutable multiway graph for configuration.
     */
    MultiwayGraph& get_multiway_graph() {
        return *multiway_graph_;
    }

    void print_summary() const {
        multiway_graph_->print_summary();
    }
    
    std::size_t get_current_step() const {
        return current_step_.load();
    }

    std::size_t increment_step() {
        return current_step_.fetch_add(1);
    }

    std::size_t get_total_matches_found() const {
        return total_matches_found_.load();
    }

    std::size_t increment_total_matches() const {
        return total_matches_found_.fetch_add(1);
    }
    
    /**
     * Get all rewriting rules for applying to states.
     */
    const std::vector<RewritingRule>& get_rules() const {
        return rules_;
    }
    
    /**
     * Apply all rules to a target state (unified method for initial and produced states).
     * State flows through tasks rather than being stored.
     */
    void apply_all_rules_to_state(std::shared_ptr<WolframState> input_state, std::size_t current_step);
    
    
private:
    /**
     * Compute the maximum radius needed for all rules in the rule set
     * This is the maximum distance of non-overlapping edges from any anchor vertex
     * across all rule left-hand sides
     */
    static std::size_t compute_maximum_radius(const std::vector<RewritingRule>& rules);
    
    /**
     * Compute radius for a single rule LHS pattern
     * Returns the maximum distance of non-overlapping edges from any vertex
     */
    static std::size_t compute_rule_radius(const PatternHypergraph& lhs);
    
    /**
     * Compute maximum distance of non-overlapping edges from anchor vertex
     * Uses BFS to find paths of non-overlapping edges
     */
    static std::size_t compute_max_distance_from_vertex(const PatternHypergraph& lhs, VertexId anchor);
};

} // namespace hypergraph

#endif // HYPERGRAPH_WOLFRAM_EVOLUTION_HPP