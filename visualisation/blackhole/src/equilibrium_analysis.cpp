#include "blackhole/equilibrium_analysis.hpp"
#include "blackhole/entropy_analysis.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace viz::blackhole {

// =============================================================================
// Utility Functions
// =============================================================================

namespace {

// Compute variance of a vector of floats
float compute_variance(const std::vector<float>& values) {
    if (values.size() < 2) return 0.0f;

    float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    float variance = 0.0f;
    for (float v : values) {
        float diff = v - mean;
        variance += diff * diff;
    }
    return variance / (values.size() - 1);
}

// Compute linear regression slope
float compute_slope(const std::vector<float>& values) {
    if (values.size() < 2) return 0.0f;

    float n = static_cast<float>(values.size());
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;

    for (size_t i = 0; i < values.size(); ++i) {
        float x = static_cast<float>(i);
        float y = values[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    float denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-10f) return 0.0f;

    return (n * sum_xy - sum_x * sum_y) / denom;
}

// Convert variance to stability score (0-1)
// Uses normalized coefficient of variation
float variance_to_stability(float variance, float mean) {
    if (std::abs(mean) < 1e-10f) return 1.0f;  // Constant zero = stable
    float cv = std::sqrt(variance) / std::abs(mean);  // Coefficient of variation
    // Map CV to stability: CV=0 -> 1.0, CV=1 -> 0.5, CV=inf -> 0.0
    return 1.0f / (1.0f + cv);
}

}  // anonymous namespace

// =============================================================================
// Timestep Metrics Computation
// =============================================================================

TimestepMetrics compute_timestep_metrics(
    const TimestepAggregation& timestep,
    uint32_t step
) {
    TimestepMetrics metrics;
    metrics.step = step;
    metrics.vertex_count = timestep.union_vertices.size();
    metrics.edge_count = timestep.union_edges.size();

    // Dimension statistics from mean_dimensions
    if (!timestep.mean_dimensions.empty()) {
        float sum = 0.0f;
        float min_d = std::numeric_limits<float>::max();
        float max_d = std::numeric_limits<float>::lowest();

        for (float d : timestep.mean_dimensions) {
            sum += d;
            min_d = std::min(min_d, d);
            max_d = std::max(max_d, d);
        }

        metrics.mean_dimension = sum / timestep.mean_dimensions.size();
        metrics.min_dimension = min_d;
        metrics.max_dimension = max_d;

        // Compute variance
        float var_sum = 0.0f;
        for (float d : timestep.mean_dimensions) {
            float diff = d - metrics.mean_dimension;
            var_sum += diff * diff;
        }
        metrics.dimension_variance = var_sum / timestep.mean_dimensions.size();
    }

    // Build simple graph for degree entropy computation
    SimpleGraph graph;
    graph.build(timestep.union_vertices, timestep.union_edges);

    // Degree statistics
    if (graph.vertex_count() > 0) {
        metrics.degree_entropy = compute_degree_entropy(graph);

        float degree_sum = 0.0f;
        float max_deg = 0.0f;
        for (VertexId v : graph.vertices()) {
            float deg = static_cast<float>(graph.neighbors(v).size());
            degree_sum += deg;
            max_deg = std::max(max_deg, deg);
        }
        metrics.mean_degree = degree_sum / graph.vertex_count();
        metrics.max_degree = max_deg;
    }

    return metrics;
}

TimestepMetrics compute_timestep_metrics_with_branchial(
    const TimestepAggregation& timestep,
    const BranchialGraph& branchial_graph,
    uint32_t step,
    const EquilibriumConfig& config
) {
    // Start with basic metrics
    TimestepMetrics metrics = compute_timestep_metrics(timestep, step);

    // Get states at this step
    auto it = branchial_graph.step_to_states.find(step);
    if (it != branchial_graph.step_to_states.end()) {
        metrics.state_count = it->second.size();
    }

    // Compute branchial metrics if requested
    if (config.compute_branchial_metrics && metrics.state_count > 0) {
        // Compute sharpness for vertices at this step
        std::vector<float> sharpness_values;
        auto vertices_at_step = get_vertices_at_step(branchial_graph, step);

        for (VertexId v : vertices_at_step) {
            float sharpness = compute_vertex_sharpness(v, branchial_graph);
            sharpness_values.push_back(sharpness);
        }

        if (!sharpness_values.empty()) {
            float sum = std::accumulate(sharpness_values.begin(), sharpness_values.end(), 0.0f);
            metrics.mean_sharpness = sum / sharpness_values.size();

            float var_sum = 0.0f;
            for (float s : sharpness_values) {
                float diff = s - metrics.mean_sharpness;
                var_sum += diff * diff;
            }
            metrics.sharpness_variance = var_sum / sharpness_values.size();
        }
    }

    // Compute Hilbert space metrics if requested
    if (config.compute_hilbert_metrics && metrics.state_count > 1) {
        auto hilbert = analyze_hilbert_space(branchial_graph, step);
        metrics.mean_inner_product = hilbert.mean_inner_product;
        metrics.mean_mutual_information = hilbert.mean_mutual_information;
        metrics.vertex_probability_entropy = hilbert.vertex_probability_entropy;
    }

    return metrics;
}

// =============================================================================
// Stability Score Computation
// =============================================================================

void compute_stability_scores(
    EquilibriumAnalysisResult& result,
    int window_size
) {
    if (result.history.empty()) {
        return;
    }

    // Determine window
    size_t n = result.history.size();
    size_t start = (n > static_cast<size_t>(window_size)) ? n - window_size : 0;

    // Collect values in window
    std::vector<float> dimensions, degree_entropies, sharpnesses, sizes;

    for (size_t i = start; i < n; ++i) {
        const auto& m = result.history[i];
        dimensions.push_back(m.mean_dimension);
        degree_entropies.push_back(m.degree_entropy);
        sharpnesses.push_back(m.mean_sharpness);
        sizes.push_back(static_cast<float>(m.vertex_count));
    }

    // Compute stabilities
    if (!dimensions.empty()) {
        float mean = std::accumulate(dimensions.begin(), dimensions.end(), 0.0f) / dimensions.size();
        float var = compute_variance(dimensions);
        result.dimension_stability = variance_to_stability(var, mean);
        result.dimension_trend = compute_slope(dimensions);
    }

    if (!degree_entropies.empty()) {
        float mean = std::accumulate(degree_entropies.begin(), degree_entropies.end(), 0.0f) / degree_entropies.size();
        float var = compute_variance(degree_entropies);
        result.degree_entropy_stability = variance_to_stability(var, mean);
    }

    if (!sharpnesses.empty()) {
        float mean = std::accumulate(sharpnesses.begin(), sharpnesses.end(), 0.0f) / sharpnesses.size();
        float var = compute_variance(sharpnesses);
        result.sharpness_stability = variance_to_stability(var, mean);
    }

    if (!sizes.empty()) {
        float mean = std::accumulate(sizes.begin(), sizes.end(), 0.0f) / sizes.size();
        float var = compute_variance(sizes);
        result.size_stability = variance_to_stability(var, mean);
        result.size_trend = compute_slope(sizes);
    }

    // Overall stability is average of individual stabilities
    result.overall_stability = (
        result.dimension_stability +
        result.degree_entropy_stability +
        result.sharpness_stability +
        result.size_stability
    ) / 4.0f;
}

// =============================================================================
// Equilibrium Detection
// =============================================================================

void detect_equilibrium(
    EquilibriumAnalysisResult& result,
    float threshold,
    int min_steps
) {
    result.equilibrium_threshold = threshold;
    result.is_equilibrated = false;
    result.equilibration_step = 0;

    if (result.history.size() < static_cast<size_t>(min_steps)) {
        return;
    }

    // Check if we're currently in equilibrium
    if (result.overall_stability >= threshold) {
        result.is_equilibrated = true;

        // Find first step where stability crossed threshold
        // (would need to track historical stability for exact detection)
        // For now, report current step
        result.equilibration_step = result.history.back().step;
    }
}

// =============================================================================
// Full Analysis Functions
// =============================================================================

EquilibriumAnalysisResult analyze_equilibrium(
    const std::vector<TimestepAggregation>& timesteps,
    const EquilibriumConfig& config
) {
    EquilibriumAnalysisResult result;
    result.history.reserve(timesteps.size());

    for (size_t i = 0; i < timesteps.size(); ++i) {
        auto metrics = compute_timestep_metrics(timesteps[i], static_cast<uint32_t>(i));
        result.history.push_back(metrics);
    }

    compute_stability_scores(result, config.stability_window);
    detect_equilibrium(result, config.equilibrium_threshold, config.min_steps_for_equilibrium);

    return result;
}

EquilibriumAnalysisResult analyze_equilibrium_with_branchial(
    const std::vector<TimestepAggregation>& timesteps,
    const BranchialGraph& branchial_graph,
    const EquilibriumConfig& config
) {
    EquilibriumAnalysisResult result;
    result.history.reserve(timesteps.size());

    for (size_t i = 0; i < timesteps.size(); ++i) {
        auto metrics = compute_timestep_metrics_with_branchial(
            timesteps[i], branchial_graph, static_cast<uint32_t>(i), config);
        result.history.push_back(metrics);
    }

    compute_stability_scores(result, config.stability_window);
    detect_equilibrium(result, config.equilibrium_threshold, config.min_steps_for_equilibrium);

    return result;
}

void update_equilibrium_analysis(
    EquilibriumAnalysisResult& result,
    const TimestepMetrics& new_metrics,
    const EquilibriumConfig& config
) {
    result.history.push_back(new_metrics);
    compute_stability_scores(result, config.stability_window);
    detect_equilibrium(result, config.equilibrium_threshold, config.min_steps_for_equilibrium);
}

}  // namespace viz::blackhole
