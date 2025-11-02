#ifndef HYPERGRAPH_PATTERN_MATCHING_TASKS_HPP
#define HYPERGRAPH_PATTERN_MATCHING_TASKS_HPP

#include <hypergraph/pattern_matching.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/rewriting.hpp>
#include <hypergraph/thread_local_pool.hpp>
#include <job_system/job.hpp>
#include <job_system/job_system.hpp>
#include <atomic>
#include <memory>

namespace hypergraph {

// Forward declarations
class WolframEvolution;

/**
 * Task types for the SCAN/EXPAND/SINK pattern matching pipeline.
 * Each type uses different scheduling policies for optimal performance.
 */
enum class PatternMatchingTaskType {
    SCAN,        // Partition signatures and find initial candidates (LIFO - memory efficient)
    EXPAND,      // Expand partial matches with constraints (LIFO - memory efficient)
    SINK,        // Process complete matches (Round-robin - load balancing)
    REWRITE,     // Apply rewriting rules (Round-robin - load balancing)
    RELATE       // Compute event relationships (causal and branchial edges)
};

/**
 * Lightweight copyable data for rule application tasks.
 * Contains only essential data that can be safely copied by value.
 */
struct RuleApplicationData {
    RewritingRule rule;
    std::shared_ptr<WolframState> input_state;        // State flows through tasks
    std::shared_ptr<const Hypergraph> target_hypergraph; // Hypergraph data for pattern matching
    std::vector<GlobalEdgeId> edge_id_mapping;        // Maps local EdgeId to GlobalEdgeId
    std::size_t current_step;
    std::size_t max_steps;
    std::size_t max_matches;

    // Stable pointers that persist for the duration of evolution
    std::shared_ptr<MultiwayGraph> multiway_graph;
    job_system::JobSystem<PatternMatchingTaskType>* job_system;
    WolframEvolution* wolfram_evolution;

    RuleApplicationData(const RewritingRule& r, std::shared_ptr<WolframState> input_state,
                       std::shared_ptr<const Hypergraph> target_hg,
                       const std::vector<GlobalEdgeId>& edge_mapping, std::size_t step,
                       std::size_t max_s, std::size_t max_m,
                       std::shared_ptr<MultiwayGraph> mg,
                       job_system::JobSystem<PatternMatchingTaskType>* js,
                       WolframEvolution* we)
        : rule(r), input_state(input_state),
          target_hypergraph(target_hg), edge_id_mapping(edge_mapping), current_step(step),
          max_steps(max_s), max_matches(max_m),
          multiway_graph(mg), job_system(js), wolfram_evolution(we) {}
};

/**
 * Partial pattern match containing incomplete binding information.
 * Used to track progress through the SCAN → EXPAND → SINK pipeline.
 */
struct PartialMatch {
    std::vector<EdgeId> matched_edges;                    // Edges matched so far
    std::unordered_map<std::size_t, EdgeId> edge_map;     // Pattern edge idx → Data edge ID
    VariableAssignment assignment;                        // Variable bindings
    std::size_t next_pattern_edge_idx;                    // Next pattern edge to match
    VertexId anchor_vertex;                               // Anchor point for search
    std::unordered_set<EdgeId> available_edges;           // Remaining edges to consider

    PartialMatch() : next_pattern_edge_idx(0), anchor_vertex(0) {}

    /**
     * Check if this is a complete match (all pattern edges matched).
     * CRITICAL: Must verify that ALL pattern edge indices from 0 to N-1 are present,
     * because SCAN tasks process pattern edges independently and may only match a subset.
     */
    bool is_complete(std::size_t total_pattern_edges) const {
        // Check that we have the right number of edges
        if (edge_map.size() != total_pattern_edges) {
            return false;
        }

        // Verify that all pattern indices from 0 to total_pattern_edges-1 are present
        for (std::size_t i = 0; i < total_pattern_edges; ++i) {
            if (edge_map.find(i) == edge_map.end()) {
                return false;  // Missing pattern edge i
            }
        }

        return true;
    }

    /**
     * Add an edge to this partial match.
     */
    void add_edge_match(std::size_t pattern_idx, EdgeId data_edge,
                       const VariableAssignment& new_assignment) {
        matched_edges.push_back(data_edge);
        edge_map[pattern_idx] = data_edge;
        assignment = new_assignment;
        next_pattern_edge_idx = pattern_idx + 1;
        available_edges.erase(data_edge);  // Remove used edge
    }

    /**
     * Create a copy for branching to multiple possibilities.
     */
    PartialMatch branch() const {
        return *this;  // Copy constructor
    }
};

/**
 * Context shared across all pattern matching tasks in a session.
 * Provides access to target hypergraph, pattern, and coordination data.
 */
struct PatternMatchingContext {
    std::shared_ptr<const Hypergraph> target_hypergraph;
    std::shared_ptr<const PatternHypergraph> pattern;
    std::shared_ptr<const EdgeSignatureIndex> signature_index;

    // Multiway graph for state and event tracking
    std::shared_ptr<MultiwayGraph> multiway_graph;
    std::shared_ptr<WolframState> current_state;       // State flows through tasks
    std::vector<GlobalEdgeId> edge_id_mapping;         // Maps local EdgeId to GlobalEdgeId
    RewritingRule rewrite_rule;  // Single rule for REWRITE tasks
    std::size_t rule_index{0};  // Index of the rule being processed

    // Job system for spawning tasks
    job_system::JobSystem<PatternMatchingTaskType>* job_system{nullptr};

    // Pointer to WolframEvolution for global step tracking
    WolframEvolution* wolfram_evolution{nullptr};

    // Step tracking for multi-step evolution
    std::size_t max_steps{1};
    std::atomic<std::size_t> current_step{0};
    std::atomic<std::size_t> highest_step_reached{0};

    // Coordination and result collection
    std::atomic<std::size_t> total_matches_found{0};
    std::atomic<std::size_t> max_matches_to_find{1000};  // Limit to prevent explosion
    std::atomic<bool> should_terminate{false};

    // Reference counting for job dependency tracking
    std::atomic<std::size_t> pending_work{0};  // Count of work items that need to complete

    // Statistics
    std::atomic<std::size_t> scan_tasks_spawned{0};
    std::atomic<std::size_t> expand_tasks_spawned{0};
    std::atomic<std::size_t> sink_tasks_spawned{0};
    std::atomic<std::size_t> rewrite_tasks_spawned{0};
    std::atomic<std::size_t> causal_tasks_spawned{0};
    std::atomic<std::size_t> branchial_tasks_spawned{0};

    // Completion counters for debugging
    std::atomic<std::size_t> scan_tasks_completed{0};
    std::atomic<std::size_t> expand_tasks_completed{0};
    std::atomic<std::size_t> sink_tasks_completed{0};
    std::atomic<std::size_t> rewrite_tasks_completed{0};
    std::atomic<std::size_t> causal_tasks_completed{0};
    std::atomic<std::size_t> branchial_tasks_completed{0};

    // Pruning control parameters (passed from MultiwayGraph)
    std::size_t max_successor_states_per_parent{0};
    std::size_t max_states_per_step{0};
    double exploration_probability{1.0};

    // Pruning rejection statistics
    std::atomic<std::size_t> rewrites_rejected_random{0};
    std::atomic<std::size_t> rewrites_rejected_successor{0};
    std::atomic<std::size_t> rewrites_rejected_step{0};

    // Cached pattern edge signatures (optimization #2)
    std::vector<EdgeSignature> cached_pattern_signatures;

    PatternMatchingContext(std::shared_ptr<const Hypergraph> hg, std::shared_ptr<const PatternHypergraph> pat,
                          std::shared_ptr<const EdgeSignatureIndex> idx,
                          std::shared_ptr<MultiwayGraph> mg,
                          job_system::JobSystem<PatternMatchingTaskType>* js,
                          const RewritingRule& rule,
                          WolframEvolution* we = nullptr)
        : target_hypergraph(hg), pattern(pat), signature_index(idx),
          multiway_graph(mg), rewrite_rule(rule),
          job_system(js), wolfram_evolution(we) {
        // Pre-compute pattern edge signatures once (optimization #2)
        cached_pattern_signatures.reserve(pattern->num_edges());
        for (const auto& pattern_edge : pattern->edges()) {
            cached_pattern_signatures.push_back(EdgeSignature::from_pattern_edge(pattern_edge));
        }
    }
};

/**
 * SCAN task: Find initial edge candidates using signature index.
 * Spawns EXPAND tasks for each promising partial match.
 */
class ScanTask : public job_system::Job<PatternMatchingTaskType> {
private:
    std::shared_ptr<PatternMatchingContext> context_;
    std::size_t pattern_edge_idx_;     // Which pattern edge to scan for
    std::size_t partition_start_;      // Partition range to scan
    std::size_t partition_end_;

public:
    ScanTask(std::shared_ptr<PatternMatchingContext> ctx,
             std::size_t pattern_idx, std::size_t start, std::size_t end)
        : context_(std::move(ctx)), pattern_edge_idx_(pattern_idx),
          partition_start_(start), partition_end_(end) {}

    void execute() override;

    PatternMatchingTaskType get_type() const override {
        return PatternMatchingTaskType::SCAN;
    }

};

/**
 * EXPAND task: Extend a partial match by finding compatible edges.
 * May spawn more EXPAND tasks or SINK task if match is complete.
 */
class ExpandTask : public job_system::Job<PatternMatchingTaskType> {
private:
    std::shared_ptr<PatternMatchingContext> context_;
    PartialMatch partial_match_;

public:
    ExpandTask(std::shared_ptr<PatternMatchingContext> ctx, PartialMatch match)
        : context_(std::move(ctx)), partial_match_(std::move(match)) {}

    void execute() override;

    PatternMatchingTaskType get_type() const override {
        return PatternMatchingTaskType::EXPAND;
    }

};

/**
 * SINK task: Process complete pattern matches.
 * Triggers rewrite rule application and creates new states/events.
 */
class SinkTask : public job_system::Job<PatternMatchingTaskType> {
private:
    std::shared_ptr<PatternMatchingContext> context_;
    PatternMatch complete_match_;

public:
    SinkTask(std::shared_ptr<PatternMatchingContext> ctx, PatternMatch match)
        : context_(std::move(ctx)), complete_match_(std::move(match)) {}

    void execute() override;

    PatternMatchingTaskType get_type() const override {
        return PatternMatchingTaskType::SINK;
    }

};

/**
 * REWRITE task: Apply a rewriting rule to create new state.
 * Modifies the multiway graph structure.
 */
class RewriteTask : public job_system::Job<PatternMatchingTaskType> {
private:
    std::shared_ptr<PatternMatchingContext> context_;
    PatternMatch complete_match_;
    // Additional rewrite rule data would go here

public:
    RewriteTask(std::shared_ptr<PatternMatchingContext> ctx, PatternMatch match)
        : context_(std::move(ctx)), complete_match_(std::move(match)) {}

    void execute() override;

    PatternMatchingTaskType get_type() const override {
        return PatternMatchingTaskType::REWRITE;
    }

private:
    /**
     * Spawn patch-based matching around newly added edges.
     * Implements efficient search within a radius around new edges.
     */
    void spawn_patch_based_matching_around_new_edges(
        StateID new_state_id,
        const std::vector<std::vector<GlobalVertexId>>& new_edges,
        std::size_t current_step);


    /**
     * Spawn SCAN tasks limited to a patch of edges.
     */
    void spawn_patch_scan_tasks(
        std::shared_ptr<PatternMatchingContext> patch_context,
        const std::unordered_set<EdgeId>& patch_edges,
        std::size_t rule_idx,
        std::shared_ptr<Hypergraph> hypergraph_keeper = nullptr);
};


// PartialMatchPool uses thread-local storage - defined in thread_local_pool.hpp

/**
 * Factory functions for creating pattern matching tasks.
 */
namespace pattern_matching_tasks {

template<typename JobSystem>
void spawn_scan_tasks(JobSystem& job_system,
                     std::shared_ptr<PatternMatchingContext> context,
                     std::size_t num_partitions = 0) {
    if (num_partitions == 0) {
        num_partitions = job_system.get_num_workers();
    }

    // For each pattern edge, create SCAN tasks across partitions
    for (std::size_t pattern_idx = 0; pattern_idx < context->pattern->num_edges(); ++pattern_idx) {
        std::size_t edges_per_partition = std::max(std::size_t(1), context->target_hypergraph->num_edges() / num_partitions);

        for (std::size_t partition = 0; partition < num_partitions; ++partition) {
            std::size_t start = partition * edges_per_partition;
            std::size_t end = (partition == num_partitions - 1) ?
                             context->target_hypergraph->num_edges() :
                             (partition + 1) * edges_per_partition;

            auto task = job_system::make_job(
                [context, pattern_idx, start, end]() {
                    ScanTask task(context, pattern_idx, start, end);
                    task.execute();
                },
                PatternMatchingTaskType::SCAN
            );

            job_system.submit(std::move(task), job_system::ScheduleMode::LIFO);
            context->scan_tasks_spawned.fetch_add(1);
        }
    }
}

template<typename JobSystem>
void spawn_expand_task(JobSystem& job_system,
                      std::shared_ptr<PatternMatchingContext> context,
                      PartialMatch partial_match) {
    auto task = job_system::make_job(
        [context, partial_match = std::move(partial_match)]() mutable {
            ExpandTask task(context, std::move(partial_match));
            task.execute();
        },
        PatternMatchingTaskType::EXPAND
    );

    job_system.submit(std::move(task), job_system::ScheduleMode::LIFO);
    context->expand_tasks_spawned.fetch_add(1);
}

template<typename JobSystem>
void spawn_sink_task(JobSystem& job_system,
                    std::shared_ptr<PatternMatchingContext> context,
                    PatternMatch complete_match) {
    auto task = job_system::make_job(
        [context, complete_match = std::move(complete_match)]() mutable {
            SinkTask task(context, std::move(complete_match));
            task.execute();
        },
        PatternMatchingTaskType::SINK
    );

    job_system.submit(std::move(task), job_system::ScheduleMode::FIFO);
    context->sink_tasks_spawned.fetch_add(1);
}

} // namespace pattern_matching_tasks

/**
 * Helper functions for pattern matching task execution.
 */

/**
 * Generate variable assignment for matching a pattern edge to a concrete edge.
 * Returns std::nullopt if the edges are incompatible.
 * Optimization #5: Changed from vector to optional to avoid allocation.
 */
std::optional<VariableAssignment> generate_edge_assignment(
    const PatternEdge& pattern_edge,
    const Hyperedge& concrete_edge);

/**
 * Check if two variable assignments are consistent (no conflicting variable bindings).
 */
bool is_assignment_consistent(const VariableAssignment& a1, const VariableAssignment& a2);


} // namespace hypergraph

#endif // HYPERGRAPH_PATTERN_MATCHING_TASKS_HPP