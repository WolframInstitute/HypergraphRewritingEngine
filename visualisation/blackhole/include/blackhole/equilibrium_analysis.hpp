#pragma once

#include "bh_types.hpp"
#include "branchial_analysis.hpp"
#include <vector>
#include <unordered_map>

namespace viz::blackhole {

// =============================================================================
// Equilibrium Analysis
// =============================================================================
// Tracks macroscopic property stability over time to detect when a system
// (flat space, black hole, etc.) reaches equilibrium.
//
// From Wolfram's discussion: "You quickly reach equilibrium where the
// macroscopic properties that one measures don't change."
//
// Key metrics tracked:
// - Mean dimension: should stabilize in equilibrium
// - Degree distribution entropy: should stabilize
// - Mean sharpness: distribution spread should stabilize
// - Edge/vertex counts: should fluctuate around stable values

// =============================================================================
// Per-Timestep Metrics
// =============================================================================

struct TimestepMetrics {
    uint32_t step = 0;

    // Dimension statistics
    float mean_dimension = 0.0f;
    float dimension_variance = 0.0f;
    float min_dimension = 0.0f;
    float max_dimension = 0.0f;

    // Degree distribution
    float degree_entropy = 0.0f;      // Entropy of degree distribution
    float mean_degree = 0.0f;
    float max_degree = 0.0f;

    // Branchial/sharpness metrics
    float mean_sharpness = 0.0f;      // Mean vertex sharpness across branches
    float sharpness_variance = 0.0f;

    // Graph size
    size_t vertex_count = 0;
    size_t edge_count = 0;
    size_t state_count = 0;           // Number of states at this timestep

    // Hilbert space metrics (if computed)
    float mean_inner_product = 0.0f;
    float mean_mutual_information = 0.0f;
    float vertex_probability_entropy = 0.0f;
};

// =============================================================================
// Equilibrium Detection Result
// =============================================================================

struct EquilibriumAnalysisResult {
    // Per-timestep metrics history
    std::vector<TimestepMetrics> history;

    // Stability scores (0 = unstable, 1 = stable)
    // Computed over a sliding window of recent timesteps
    float dimension_stability = 0.0f;     // 1 - normalized variance of mean_dimension
    float degree_entropy_stability = 0.0f; // 1 - normalized variance of degree_entropy
    float sharpness_stability = 0.0f;      // 1 - normalized variance of mean_sharpness
    float size_stability = 0.0f;           // 1 - normalized variance of vertex_count

    // Combined stability score (average of individual stabilities)
    float overall_stability = 0.0f;

    // Equilibrium detection
    bool is_equilibrated = false;          // True if overall_stability > threshold
    size_t equilibration_step = 0;         // First step where equilibrium was detected
    float equilibrium_threshold = 0.95f;   // Threshold for equilibrium detection

    // Trend analysis
    float dimension_trend = 0.0f;          // Slope of mean_dimension over time (0 = flat)
    float size_trend = 0.0f;               // Slope of vertex_count over time
};

// =============================================================================
// Configuration
// =============================================================================

struct EquilibriumConfig {
    // Sliding window size for stability computation
    int stability_window = 20;

    // Minimum steps before checking equilibrium
    int min_steps_for_equilibrium = 50;

    // Threshold for declaring equilibrium (0-1)
    float equilibrium_threshold = 0.95f;

    // Whether to compute branchial metrics (requires BranchialGraph)
    bool compute_branchial_metrics = true;

    // Whether to compute Hilbert space metrics
    bool compute_hilbert_metrics = true;
};

// =============================================================================
// Functions
// =============================================================================

// Compute metrics for a single timestep from aggregated data
TimestepMetrics compute_timestep_metrics(
    const TimestepAggregation& timestep,
    uint32_t step
);

// Compute metrics for a timestep including branchial analysis
TimestepMetrics compute_timestep_metrics_with_branchial(
    const TimestepAggregation& timestep,
    const BranchialGraph& branchial_graph,
    uint32_t step,
    const EquilibriumConfig& config
);

// Compute stability scores from a history of timestep metrics
// Uses a sliding window over the most recent entries
void compute_stability_scores(
    EquilibriumAnalysisResult& result,
    int window_size
);

// Detect equilibrium and set equilibration_step
void detect_equilibrium(
    EquilibriumAnalysisResult& result,
    float threshold,
    int min_steps
);

// Full equilibrium analysis from a sequence of timestep aggregations
EquilibriumAnalysisResult analyze_equilibrium(
    const std::vector<TimestepAggregation>& timesteps,
    const EquilibriumConfig& config
);

// Full equilibrium analysis with branchial data
EquilibriumAnalysisResult analyze_equilibrium_with_branchial(
    const std::vector<TimestepAggregation>& timesteps,
    const BranchialGraph& branchial_graph,
    const EquilibriumConfig& config
);

// Add a single timestep to existing analysis and update stability
void update_equilibrium_analysis(
    EquilibriumAnalysisResult& result,
    const TimestepMetrics& new_metrics,
    const EquilibriumConfig& config
);

}  // namespace viz::blackhole
