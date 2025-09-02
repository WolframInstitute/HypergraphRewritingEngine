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
    std::size_t max_steps_;
    std::size_t num_threads_;
    
public:
    WolframEvolution(std::size_t max_steps, std::size_t num_threads = std::thread::hardware_concurrency(), 
                     bool canonicalization_enabled = true, bool full_capture = false,
                     bool event_deduplication = true)
        : multiway_graph_(std::make_shared<MultiwayGraph>())
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
    
    ~WolframEvolution() {
        if (job_system_) {
            job_system_->shutdown();
        }
    }
    
    /**
     * Add rewriting rule.
     */
    void add_rule(const RewritingRule& rule) {
        DEBUG_LOG("Adding rule %zu", rules_.size());
        rules_.push_back(rule);
    }
    
    /**
     * Initialize radius from the loaded rule set.
     * Should be called after all rules are added and before evolution starts.
     */
    void initialize_radius_config() {
        if (!rules_.empty()) {
            computed_radius_ = compute_maximum_radius(rules_);
            DEBUG_LOG("Computed radius %zu from %zu rules", 
                     computed_radius_, rules_.size());
        }
    }
    
    /**
     * Get the computed radius for this evolution system.
     */
    std::size_t get_computed_radius() const {
        return computed_radius_ ? computed_radius_ : 3;  // fallback to 3 if not computed
    }
    
    /**
     * Submit initial state and start evolution.
     * Blocks until completion or step limit reached.
     */
    void evolve(const std::vector<std::vector<GlobalVertexId>>& initial_edges);
    
    /**
     * Get results.
     */
    const MultiwayGraph& get_multiway_graph() const {
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
    
private:
    /**
     * Compute the maximum radius needed for all rules in the rule set
     * This is the maximum distance of non-overlapping edges from any anchor vertex
     * across all rule left-hand sides
     */
    static std::size_t compute_maximum_radius(const std::vector<RewritingRule>& rules) {
        std::size_t max_radius = 0;
        
        for (const auto& rule : rules) {
            std::size_t rule_radius = compute_rule_radius(rule.lhs);
            max_radius = std::max(max_radius, rule_radius);
        }
        
        return max_radius;
    }
    
    /**
     * Compute radius for a single rule LHS pattern
     * Returns the maximum distance of non-overlapping edges from any vertex
     */
    static std::size_t compute_rule_radius(const PatternHypergraph& lhs) {
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
    
    /**
     * Compute maximum distance of non-overlapping edges from anchor vertex
     * Uses BFS to find paths of non-overlapping edges
     */
    static std::size_t compute_max_distance_from_vertex(const PatternHypergraph& lhs, VertexId anchor) {
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
};

} // namespace hypergraph

#endif // HYPERGRAPH_WOLFRAM_EVOLUTION_HPP