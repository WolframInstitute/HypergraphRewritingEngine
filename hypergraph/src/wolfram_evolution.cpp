#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/debug_log.hpp>

namespace hypergraph {

void WolframEvolution::evolve(const std::vector<std::vector<GlobalVertexId>>& initial_edges) {
    DEBUG_LOG("Starting evolution with %zu initial edges", initial_edges.size());
    DEBUG_LOG("Evolution has %zu rules, max steps: %zu", rules_.size(), max_steps_);

    // Create initial state
    StateId initial_state = multiway_graph_->create_initial_state(initial_edges);
    DEBUG_LOG("Created initial state %zu", initial_state);

    // Store objects that need to live until tasks complete
    std::vector<std::shared_ptr<Hypergraph>> target_graphs;
    std::vector<std::shared_ptr<EdgeSignatureIndex>> signature_indices;
    std::vector<std::shared_ptr<PatternMatchingContext>> contexts;
    
    // Use the task-based system - let the tasks handle the parallel evolution flow
    for (const auto& rule : rules_) {
        // Create edge signature index for the initial state
        auto state_opt = multiway_graph_->get_state(initial_state);
        if (!state_opt) continue;

        // Create shared objects that will live until tasks complete
        auto target_hg = std::make_shared<Hypergraph>(state_opt->to_canonical_hypergraph());
        auto signature_index = std::make_shared<EdgeSignatureIndex>();
        auto label_func = [](VertexId v) -> VertexLabel { return static_cast<VertexLabel>(v); };

        for (EdgeId edge_id = 0; edge_id < target_hg->num_edges(); ++edge_id) {
            const auto& edge = target_hg->edges()[edge_id];
            EdgeSignature sig = EdgeSignature::from_concrete_edge(edge, label_func);
            signature_index->add_edge(edge_id, sig);
        }

        // Create pattern matching context and let the task system handle the evolution
        auto context = std::make_shared<PatternMatchingContext>(
            target_hg.get(), &rule.lhs, signature_index.get(), label_func, multiway_graph_, job_system_.get(), this);
        context->current_state_id = initial_state;
        context->rewrite_rules = {rule};
        context->max_steps = max_steps_;
        context->current_step = 0;

        // Keep shared objects alive
        target_graphs.push_back(target_hg);
        signature_indices.push_back(signature_index);
        contexts.push_back(context);

        // Submit to task system - let the tasks handle parallel evolution
        pattern_matching_tasks::spawn_scan_tasks(*job_system_, context, num_threads_);
    }

    // Wait for all tasks to complete
    job_system_->wait_for_completion();
    
    DEBUG_LOG("Evolution completed - task system handled all parallel processing");
}

} // namespace hypergraph