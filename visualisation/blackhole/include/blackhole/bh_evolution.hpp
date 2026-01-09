#pragma once

#include "bh_types.hpp"
#include "bh_initial_condition.hpp"
#include "hausdorff_analysis.hpp"

#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>
#include <job_system/job_system.hpp>

#include <vector>
#include <string>
#include <functional>
#include <atomic>

namespace viz::blackhole {

// =============================================================================
// Evolution Result
// =============================================================================
// Holds the raw output from multiway evolution before analysis.

struct EvolutionResult {
    // States grouped by step: states_by_step[step] = vector of SimpleGraph
    // Used for dimension computation (just needs adjacency structure)
    std::vector<std::vector<SimpleGraph>> states_by_step;

    // State data with edge IDs: state_data_by_step[step] = vector of StateData
    // Used for visualization and frequency counting (preserves edge IDs)
    std::vector<std::vector<StateData>> state_data_by_step;

    // Total counts
    int total_states = 0;
    int total_events = 0;
    int max_step_reached = 0;
};

// =============================================================================
// Analysis Job Type
// =============================================================================
// For parallel Hausdorff analysis

enum class AnalysisJobType {
    STATE_ANALYSIS,    // Analyze a single state (BFS, dimension estimation)
    TIMESTEP_AGGREGATE // Aggregate a timestep's states
};

// =============================================================================
// Evolution Runner
// =============================================================================
// Wraps ParallelEvolutionEngine for black hole analysis workflow.
// Handles:
// - Conversion from BHInitialCondition to hypergraph format
// - Running pruned multiway evolution
// - Extracting states grouped by step
// - Running parallel Hausdorff analysis

class EvolutionRunner {
public:
    // Construct with number of threads (0 = auto-detect)
    explicit EvolutionRunner(size_t num_threads = 0);
    ~EvolutionRunner();

    // Non-copyable
    EvolutionRunner(const EvolutionRunner&) = delete;
    EvolutionRunner& operator=(const EvolutionRunner&) = delete;

    // Run evolution from initial condition
    // Returns states grouped by step as SimpleGraphs for analysis
    EvolutionResult run_evolution(
        const BHInitialCondition& initial,
        const EvolutionConfig& config
    );

    // Run full pipeline: evolution + analysis
    // Uses parallel job system for Hausdorff analysis
    BHAnalysisResult run_full_analysis(
        const BHInitialCondition& initial,
        const EvolutionConfig& config,
        const AnalysisConfig& analysis_config = {}
    );

    // Run analysis only on pre-evolved data (loaded from .bhevo file)
    // Uses parallel job system for Hausdorff analysis
    BHAnalysisResult run_analysis(
        const BHInitialCondition& initial,
        const EvolutionConfig& config,
        const std::vector<std::vector<SimpleGraph>>& states_by_step,
        int total_states,
        int total_events,
        int max_step_reached,
        const AnalysisConfig& analysis_config = {}
    );

    // Progress callback type
    using ProgressCallback = std::function<void(const std::string& stage, float progress)>;

    // Set progress callback
    void set_progress_callback(ProgressCallback callback) {
        progress_callback_ = std::move(callback);
    }

    // Request stop (can be called from another thread)
    void request_stop() {
        should_stop_.store(true, std::memory_order_release);
    }

    // Check if stopped
    bool was_stopped() const {
        return should_stop_.load(std::memory_order_acquire);
    }

    // Get thread count
    size_t num_threads() const { return num_threads_; }

private:
    // Parse rule string in Wolfram format: {{a,b},{b,c}} -> {{a,b},{b,c},{c,d},{d,a}}
    // Returns true if successful
    bool parse_rule(const std::string& rule_string);

    // Convert BHInitialCondition edges to hypergraph format
    std::vector<std::vector<hypergraph::VertexId>> convert_edges(
        const BHInitialCondition& initial
    );

    // Extract states from hypergraph, grouped by step
    EvolutionResult extract_states();

    // Run parallel analysis on extracted states
    // Note: evolution_result is non-const to allow moving data out and clearing after use
    // This significantly reduces peak memory usage
    BHAnalysisResult analyze_parallel(
        const BHInitialCondition& initial,
        const EvolutionConfig& config,
        EvolutionResult& evolution_result,
        const AnalysisConfig& analysis_config
    );

    // Report progress
    void report_progress(const std::string& stage, float progress);

    // Hypergraph and engine (owned)
    std::unique_ptr<hypergraph::Hypergraph> hypergraph_;
    std::unique_ptr<hypergraph::ParallelEvolutionEngine> engine_;

    // Separate job system for analysis (separate from evolution job system)
    std::unique_ptr<job_system::JobSystem<AnalysisJobType>> analysis_job_system_;

    size_t num_threads_;
    std::atomic<bool> should_stop_{false};
    ProgressCallback progress_callback_;
};

// =============================================================================
// Utility: Convert HGE state to SimpleGraph
// =============================================================================

SimpleGraph state_to_simple_graph(
    const hypergraph::Hypergraph& hg,
    hypergraph::StateId state_id
);

} // namespace viz::blackhole
