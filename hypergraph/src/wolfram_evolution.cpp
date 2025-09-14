#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/debug_log.hpp>
#include <queue>
#include <algorithm>
#include <unordered_map>

namespace hypergraph {

WolframEvolution::WolframEvolution(std::size_t max_steps, std::size_t num_threads,
                 bool canonicalization_enabled, bool full_capture,
                 bool event_deduplication)
    : multiway_graph_(std::make_shared<MultiwayGraph>(full_capture))
    , job_system_(std::make_unique<job_system::JobSystem<PatternMatchingTaskType>>(num_threads))
    , max_steps_(max_steps)
    , num_threads_(num_threads) {
    DEBUG_LOG("WolframEvolution created: %zu steps, %zu threads, canonicalization %s, full_capture %s, event_dedup %s", 
              max_steps, num_threads, canonicalization_enabled ? "enabled" : "disabled", 
              full_capture ? "enabled" : "disabled", event_deduplication ? "enabled" : "disabled");
    multiway_graph_->set_canonicalization_enabled(canonicalization_enabled);
    job_system_->start();
    DEBUG_LOG("Job system started");
}

WolframEvolution::~WolframEvolution() {
    if (job_system_) {
        // Wait for any remaining tasks to complete before shutdown
        job_system_->wait_for_completion();
        job_system_->shutdown();
    }
}

void WolframEvolution::add_rule(const RewritingRule& rule) {
    DEBUG_LOG("Adding rule %zu", rules_.size());
    rules_.push_back(rule);
}

void WolframEvolution::initialize_radius_config() {
    if (!rules_.empty()) {
        computed_radius_ = compute_maximum_radius(rules_);
        DEBUG_LOG("Computed radius %zu from %zu rules", 
                 computed_radius_, rules_.size());
    }
}

void WolframEvolution::evolve(const std::vector<std::vector<GlobalVertexId>>& initial_edges) {
    DEBUG_LOG("Starting evolution with %zu initial edges", initial_edges.size());
    DEBUG_LOG("Evolution has %zu rules, max steps: %zu", rules_.size(), max_steps_);

    // Create initial state (flows through tasks, not stored)
    WolframState initial_state = multiway_graph_->create_initial_state(initial_edges);
    DEBUG_LOG("Created initial state with %zu edges", initial_state.edges().size());

    // Clear any previous evolution context objects (no longer needed with new approach)
    target_graphs_.clear();
    signature_indices_.clear();
    contexts_.clear();

    // Apply all rules to the initial state (state flows through tasks)
    apply_all_rules_to_state(initial_state, 0);

    // Wait for all tasks to complete
    job_system_->wait_for_completion();

    // Compute causal and branchial relationships in post-processing phase
    DEBUG_LOG("Computing event relationships in post-processing phase");
    multiway_graph_->compute_all_event_relationships(job_system_.get());

    DEBUG_LOG("Evolution completed - task system handled all parallel processing");
}

void WolframEvolution::apply_all_rules_to_state(const WolframState& input_state, std::size_t current_step) {
    DEBUG_LOG("Applying %zu rules to state with %zu edges at step %zu",
             rules_.size(), input_state.edges().size(), current_step);

    // For each rule, create a task that copies all necessary data by value
    for (std::size_t rule_idx = 0; rule_idx < rules_.size(); ++rule_idx) {
        const auto& rule = rules_[rule_idx];

        // Create hypergraph from state for pattern matching
        auto target_hg_data = input_state.to_hypergraph();
        auto target_hg = std::make_shared<const Hypergraph>(std::move(target_hg_data));

        // Create mapping from local EdgeIds to GlobalEdgeIds
        std::vector<GlobalEdgeId> edge_mapping;
        const auto& state_edges = input_state.edges();
        edge_mapping.reserve(state_edges.size());
        for (const auto& global_edge : state_edges) {
            edge_mapping.push_back(global_edge.global_id);
        }

        // Create copyable rule application data with state copy, hypergraph, and mapping
        RuleApplicationData rule_data(rule, input_state, target_hg, edge_mapping,
                                     current_step, max_steps_, 1000, multiway_graph_, job_system_.get(), this);
        
        // Create task that captures rule_data by value (includes shared_ptr to hypergraph)
        auto task = job_system::make_job(
            [rule_data, rule_idx]() {
                // Use the hypergraph from rule_data (already a shared_ptr)
                auto target_hg = rule_data.target_hypergraph;
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
                    target_hg,  // Use the shared_ptr from rule_data
                    std::make_shared<const PatternHypergraph>(rule_data.rule.lhs),
                    signature_index,
                    label_func,
                    rule_data.multiway_graph,
                    rule_data.job_system,
                    rule_data.rule,
                    rule_data.wolfram_evolution
                );
                
                context->current_state = rule_data.input_state;  // State flows through task
                context->edge_id_mapping = rule_data.edge_id_mapping;                       // EdgeId to GlobalEdgeId mapping
                context->rule_index = rule_idx;
                context->max_steps = rule_data.max_steps;
                context->current_step = rule_data.current_step;
                context->max_matches_to_find = rule_data.max_matches;
                
                // Spawn SCAN tasks for this rule
                std::size_t num_partitions = rule_data.job_system->get_num_workers();
                pattern_matching_tasks::spawn_scan_tasks(*rule_data.job_system, context, num_partitions);
                
                DEBUG_LOG("Applied rule %zu to state with %zu edges", rule_idx, rule_data.input_state.edges().size());
            },
            PatternMatchingTaskType::REWRITE  // Use REWRITE type for rule application tasks
        );
        
        job_system_->submit(std::move(task), job_system::ScheduleMode::FIFO);
    }
}

std::size_t WolframEvolution::compute_maximum_radius(const std::vector<RewritingRule>& rules) {
    std::size_t max_radius = 0;
    
    for (const auto& rule : rules) {
        std::size_t rule_radius = compute_rule_radius(rule.lhs);
        max_radius = std::max(max_radius, rule_radius);
    }
    
    return max_radius;
}

std::size_t WolframEvolution::compute_rule_radius(const PatternHypergraph& lhs) {
    if (lhs.num_edges() == 0) return 0;
    
    std::size_t max_radius = 0;
    
    // Try each vertex as potential anchor point
    std::unordered_set<VertexId> all_vertices;
    const auto& edges = lhs.edges();
    for (const auto& edge : edges) {
        for (const auto& vertex : edge.vertices) {
            if (vertex.type == PatternVertex::Type::VARIABLE) {
                all_vertices.insert(vertex.id);
            }
        }
    }
    
    // For each potential anchor vertex, compute maximum distance
    for (VertexId anchor : all_vertices) {
        std::size_t radius = compute_max_distance_from_vertex(lhs, anchor);
        max_radius = std::max(max_radius, radius);
    }
    
    return max_radius;
}

std::size_t WolframEvolution::compute_max_distance_from_vertex(const PatternHypergraph& lhs, VertexId anchor) {
    // Build vertex-to-edges mapping
    std::unordered_map<VertexId, std::vector<std::size_t>> vertex_to_edges;
    const auto& edges = lhs.edges();
    for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
        const auto& edge = edges[edge_idx];
        for (const auto& vertex : edge.vertices) {
            if (vertex.type == PatternVertex::Type::VARIABLE) {
                vertex_to_edges[vertex.id].push_back(edge_idx);
            }
        }
    }
    
    // BFS to find maximum distance of non-overlapping edges
    std::queue<std::pair<VertexId, std::size_t>> bfs_queue; // {vertex, distance}
    std::unordered_set<std::size_t> visited_edges;
    std::unordered_set<VertexId> visited_vertices;
    
    bfs_queue.push(std::make_pair(anchor, 0));
    visited_vertices.insert(anchor);
    
    std::size_t max_distance = 0;
    
    while (!bfs_queue.empty()) {
        auto [current_vertex, distance] = bfs_queue.front();
        bfs_queue.pop();
        
        max_distance = std::max(max_distance, distance);
        
        // Explore edges connected to current vertex
        auto edge_it = vertex_to_edges.find(current_vertex);
        if (edge_it == vertex_to_edges.end()) continue;
        
        for (std::size_t edge_idx : edge_it->second) {
            if (visited_edges.count(edge_idx)) continue; // Skip overlapping edges
            
            visited_edges.insert(edge_idx);
            const auto& edge = edges[edge_idx];
            
            // Add all other vertices in this edge to the queue
            for (const auto& vertex : edge.vertices) {
                if (vertex.type == PatternVertex::Type::VARIABLE && 
                    vertex.id != current_vertex &&
                    !visited_vertices.count(vertex.id)) {
                    
                    visited_vertices.insert(vertex.id);
                    bfs_queue.push(std::make_pair(vertex.id, distance + 1));
                }
            }
        }
    }
    
    return max_distance;
}

} // namespace hypergraph