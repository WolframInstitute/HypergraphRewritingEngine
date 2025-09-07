#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/debug_log.hpp>

namespace hypergraph {

void WolframEvolution::evolve(const std::vector<std::vector<GlobalVertexId>>& initial_edges) {
    DEBUG_LOG("Starting evolution with %zu initial edges", initial_edges.size());
    DEBUG_LOG("Evolution has %zu rules, max steps: %zu", rules_.size(), max_steps_);

    // Create initial state
    StateId initial_state = multiway_graph_->create_initial_state(initial_edges);
    DEBUG_LOG("Created initial state %zu", initial_state);

    // Clear any previous evolution context objects (no longer needed with new approach)
    target_graphs_.clear();
    signature_indices_.clear();
    contexts_.clear();

    // Apply all rules to the initial state using the unified method
    apply_all_rules_to_state(initial_state, 0);

    // Wait for all tasks to complete
    job_system_->wait_for_completion();

    DEBUG_LOG("Evolution completed - task system handled all parallel processing");
}

void WolframEvolution::apply_all_rules_to_state(StateId target_state_id, std::size_t current_step) {
    DEBUG_LOG("Applying %zu rules to state %zu at step %zu", rules_.size(), target_state_id, current_step);
    
    // For each rule, create a task that copies all necessary data by value
    for (std::size_t rule_idx = 0; rule_idx < rules_.size(); ++rule_idx) {
        const auto& rule = rules_[rule_idx];
        
        // Create copyable rule application data
        RuleApplicationData rule_data(rule, target_state_id, current_step, max_steps_, 1000,
                                     multiway_graph_, job_system_.get(), this);
        
        // Access state here where we have friend access
        auto state_opt = multiway_graph_->get_state(target_state_id);
        if (!state_opt) {
            DEBUG_LOG("ERROR: Could not get state %zu for rule %zu", target_state_id, rule_idx);
            continue;
        }
        
        // Create hypergraph (canonical or non-canonical based on settings)
        auto target_hg_data = state_opt->to_canonical_hypergraph();  // This respects canonicalization settings
        
        // Create task that captures rule_data and target_hg_data by value
        auto task = job_system::make_job(
            [rule_data, rule_idx, target_hg_data]() {
                // Create fresh hypergraph and signature index for this rule
                auto target_hg = std::make_shared<Hypergraph>(target_hg_data);
                auto signature_index = std::make_shared<EdgeSignatureIndex>();
                auto label_func = [](VertexId v) -> VertexLabel { return static_cast<VertexLabel>(v); };
                
                // Build signature index
                for (EdgeId edge_id = 0; edge_id < target_hg->num_edges(); ++edge_id) {
                    const auto& edge = target_hg->edges()[edge_id];
                    EdgeSignature sig = EdgeSignature::from_concrete_edge(edge, label_func);
                    signature_index->add_edge(edge_id, sig);
                }
                
                // Create fresh context for this rule
                auto context = std::make_shared<PatternMatchingContext>(
                    target_hg,
                    std::make_shared<const PatternHypergraph>(rule_data.rule.lhs),
                    signature_index,
                    label_func,
                    rule_data.multiway_graph,
                    rule_data.job_system,
                    rule_data.rule,
                    rule_data.wolfram_evolution
                );
                
                context->current_state_id = rule_data.target_state_id;
                context->rule_index = rule_idx;
                context->max_steps = rule_data.max_steps;
                context->current_step = rule_data.current_step;
                context->max_matches_to_find = rule_data.max_matches;
                
                // Spawn SCAN tasks for this rule
                std::size_t num_partitions = rule_data.job_system->get_num_workers();
                pattern_matching_tasks::spawn_scan_tasks(*rule_data.job_system, context, num_partitions);
                
                DEBUG_LOG("Applied rule %zu to state %zu", rule_idx, rule_data.target_state_id);
            },
            PatternMatchingTaskType::REWRITE  // Use REWRITE type for rule application tasks
        );
        
        job_system_->submit(std::move(task), job_system::ScheduleMode::FIFO);
    }
}

} // namespace hypergraph