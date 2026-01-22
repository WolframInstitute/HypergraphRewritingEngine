#include "blackhole/entropy_analysis.hpp"
#include <job_system/job_system.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace viz::blackhole {

// =============================================================================
// Utility Functions
// =============================================================================

float shannon_entropy(const std::vector<float>& probabilities) {
    float entropy = 0.0f;
    for (float p : probabilities) {
        if (p > 0.0f) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

std::vector<float> counts_to_probabilities(const std::vector<int>& counts) {
    int total = std::accumulate(counts.begin(), counts.end(), 0);
    if (total == 0) return {};

    std::vector<float> probs;
    probs.reserve(counts.size());
    for (int c : counts) {
        probs.push_back(static_cast<float>(c) / total);
    }
    return probs;
}

std::vector<int> discretize(
    const std::vector<float>& values,
    int num_bins,
    float min_val,
    float max_val
) {
    std::vector<int> bins(num_bins, 0);
    if (values.empty() || num_bins <= 0) return bins;

    float range = max_val - min_val;
    if (range <= 0) {
        // All values the same, put in first bin
        bins[0] = static_cast<int>(values.size());
        return bins;
    }

    for (float v : values) {
        int bin = static_cast<int>((v - min_val) / range * (num_bins - 1));
        bin = std::clamp(bin, 0, num_bins - 1);
        bins[bin]++;
    }

    return bins;
}

// =============================================================================
// Degree Entropy
// =============================================================================

float compute_degree_entropy(const SimpleGraph& graph) {
    if (graph.vertex_count() == 0) return 0.0f;

    // Count degree frequencies
    std::unordered_map<int, int> degree_counts;
    for (VertexId v : graph.vertices()) {
        int degree = static_cast<int>(graph.neighbors(v).size());
        degree_counts[degree]++;
    }

    // Convert to probability distribution
    std::vector<float> probs;
    float total = static_cast<float>(graph.vertex_count());
    for (const auto& [deg, count] : degree_counts) {
        probs.push_back(count / total);
    }

    return shannon_entropy(probs);
}

// =============================================================================
// Local Entropy
// =============================================================================

float compute_local_entropy(
    const SimpleGraph& graph,
    VertexId vertex,
    int radius
) {
    // Get vertices within radius
    auto distances = graph.distances_from_truncated(vertex, radius);
    const auto& verts = graph.vertices();

    // Collect degrees of vertices in neighborhood
    std::vector<int> neighbor_degrees;
    for (size_t i = 0; i < verts.size() && i < distances.size(); ++i) {
        if (distances[i] >= 0 && distances[i] <= radius) {
            int degree = static_cast<int>(graph.neighbors(verts[i]).size());
            neighbor_degrees.push_back(degree);
        }
    }

    if (neighbor_degrees.empty()) return 0.0f;

    // Compute entropy of degree distribution in neighborhood
    std::unordered_map<int, int> degree_counts;
    for (int d : neighbor_degrees) {
        degree_counts[d]++;
    }

    std::vector<float> probs;
    float total = static_cast<float>(neighbor_degrees.size());
    for (const auto& [deg, count] : degree_counts) {
        probs.push_back(count / total);
    }

    return shannon_entropy(probs);
}

std::unordered_map<VertexId, float> compute_all_local_entropies(
    const SimpleGraph& graph,
    int radius
) {
    std::unordered_map<VertexId, float> result;
    for (VertexId v : graph.vertices()) {
        result[v] = compute_local_entropy(graph, v, radius);
    }
    return result;
}

// =============================================================================
// Mutual Information
// =============================================================================

float compute_mutual_information(
    const SimpleGraph& graph,
    VertexId v1,
    VertexId v2,
    int radius
) {
    // Get neighborhoods
    auto dist1 = graph.distances_from_truncated(v1, radius);
    auto dist2 = graph.distances_from_truncated(v2, radius);
    const auto& verts = graph.vertices();

    std::unordered_set<VertexId> n1, n2;
    for (size_t i = 0; i < verts.size(); ++i) {
        if (i < dist1.size() && dist1[i] >= 0 && dist1[i] <= radius) {
            n1.insert(verts[i]);
        }
        if (i < dist2.size() && dist2[i] >= 0 && dist2[i] <= radius) {
            n2.insert(verts[i]);
        }
    }

    if (n1.empty() || n2.empty()) return 0.0f;

    // Compute intersection and union
    size_t intersection = 0;
    for (VertexId v : n1) {
        if (n2.count(v)) intersection++;
    }

    size_t union_size = n1.size() + n2.size() - intersection;
    if (union_size == 0) return 0.0f;

    // Mutual information approximation using Jaccard-based measure
    // I(X;Y) ≈ log(P(X∩Y) / (P(X)P(Y)))
    // Simplified: use normalized intersection
    float p_intersection = static_cast<float>(intersection) / union_size;
    float p1 = static_cast<float>(n1.size()) / union_size;
    float p2 = static_cast<float>(n2.size()) / union_size;

    if (p1 <= 0 || p2 <= 0 || p_intersection <= 0) return 0.0f;

    // Pointwise mutual information
    float pmi = std::log2(p_intersection / (p1 * p2));
    return std::max(0.0f, pmi);  // Clip negative PMI
}

std::unordered_map<VertexId, float> compute_all_mutual_info(
    const SimpleGraph& graph,
    int radius
) {
    std::unordered_map<VertexId, float> result;

    for (VertexId v : graph.vertices()) {
        float total_mi = 0.0f;
        const auto& neighbors = graph.neighbors(v);

        for (VertexId n : neighbors) {
            total_mi += compute_mutual_information(graph, v, n, radius);
        }

        result[v] = neighbors.empty() ? 0.0f : total_mi / neighbors.size();
    }

    return result;
}

// =============================================================================
// Fisher Information
// =============================================================================

float compute_fisher_information(
    const SimpleGraph& graph,
    VertexId vertex,
    const std::vector<float>& vertex_dimensions,
    int radius
) {
    const auto& verts = graph.vertices();
    auto it = std::find(verts.begin(), verts.end(), vertex);
    if (it == verts.end()) return 0.0f;

    size_t v_idx = std::distance(verts.begin(), it);
    if (v_idx >= vertex_dimensions.size()) return 0.0f;

    float dim_v = vertex_dimensions[v_idx];

    // Get neighborhood dimensions
    auto distances = graph.distances_from_truncated(vertex, radius);
    std::vector<float> neighbor_dims;

    for (size_t i = 0; i < verts.size() && i < distances.size(); ++i) {
        if (distances[i] > 0 && distances[i] <= radius && i < vertex_dimensions.size()) {
            neighbor_dims.push_back(vertex_dimensions[i]);
        }
    }

    if (neighbor_dims.empty()) return 0.0f;

    // Fisher information approximation:
    // I(θ) = E[(d/dθ log p(x|θ))²]
    // For dimension field, approximate as inverse variance of gradient
    // Higher Fisher info = sharper/more consistent dimension locally

    float mean_neighbor = std::accumulate(neighbor_dims.begin(), neighbor_dims.end(), 0.0f) / neighbor_dims.size();

    // Compute variance of dimension in neighborhood
    float variance = 0.0f;
    for (float d : neighbor_dims) {
        variance += (d - mean_neighbor) * (d - mean_neighbor);
    }
    variance /= neighbor_dims.size();

    // Fisher info is inverse variance (regularized to avoid division by zero)
    float fisher = 1.0f / (variance + 0.01f);

    // Also consider gradient magnitude (steepness of dimension change)
    float gradient_sum = 0.0f;
    for (float d : neighbor_dims) {
        gradient_sum += std::abs(d - dim_v);
    }
    float mean_gradient = gradient_sum / neighbor_dims.size();

    // Combined Fisher-like metric: precision × gradient sensitivity
    return fisher * (1.0f + mean_gradient);
}

std::unordered_map<VertexId, float> compute_all_fisher_info(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    int radius
) {
    std::unordered_map<VertexId, float> result;
    for (VertexId v : graph.vertices()) {
        result[v] = compute_fisher_information(graph, v, vertex_dimensions, radius);
    }
    return result;
}

// =============================================================================
// Full Entropy Analysis
// =============================================================================

EntropyAnalysisResult analyze_entropy(
    const SimpleGraph& graph,
    const EntropyConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    EntropyAnalysisResult result;

    if (graph.vertex_count() == 0) {
        return result;
    }

    // Global degree entropy
    result.degree_entropy = compute_degree_entropy(graph);

    // Local entropy
    if (config.compute_local_entropy) {
        result.local_entropy_map = compute_all_local_entropies(graph, config.neighborhood_radius);
    }

    // Mutual information
    if (config.compute_mutual_info) {
        result.mutual_info_map = compute_all_mutual_info(graph, config.neighborhood_radius);
    }

    // Fisher information (requires dimensions)
    if (config.compute_fisher_info && vertex_dimensions && !vertex_dimensions->empty()) {
        result.fisher_info_map = compute_all_fisher_info(graph, *vertex_dimensions, config.fisher_radius);
    }

    // Build per-vertex result
    const auto& verts = graph.vertices();
    result.vertex_entropies.reserve(verts.size());

    std::vector<float> local_values, fisher_values;

    for (VertexId v : verts) {
        VertexEntropy ve;
        ve.vertex = v;
        ve.degree = static_cast<int>(graph.neighbors(v).size());

        if (result.local_entropy_map.count(v)) {
            ve.local_entropy = result.local_entropy_map[v];
            local_values.push_back(ve.local_entropy);
        }

        if (result.mutual_info_map.count(v)) {
            ve.mutual_info = result.mutual_info_map[v];
            result.total_mutual_info += ve.mutual_info;
        }

        if (result.fisher_info_map.count(v)) {
            ve.fisher_info = result.fisher_info_map[v];
            fisher_values.push_back(ve.fisher_info);
            result.total_fisher_info += ve.fisher_info;
        }

        result.vertex_entropies.push_back(ve);
    }

    // Compute statistics
    if (!local_values.empty()) {
        std::sort(local_values.begin(), local_values.end());
        result.min_local_entropy = local_values.front();
        result.max_local_entropy = local_values.back();
        result.mean_local_entropy = std::accumulate(local_values.begin(), local_values.end(), 0.0f) / local_values.size();

        size_t q05_idx = static_cast<size_t>(local_values.size() * 0.05f);
        size_t q95_idx = static_cast<size_t>(local_values.size() * 0.95f);
        result.local_q05 = local_values[std::min(q05_idx, local_values.size() - 1)];
        result.local_q95 = local_values[std::min(q95_idx, local_values.size() - 1)];
    }

    if (!fisher_values.empty()) {
        std::sort(fisher_values.begin(), fisher_values.end());
        result.min_fisher_info = fisher_values.front();
        result.max_fisher_info = fisher_values.back();
        result.mean_fisher_info = std::accumulate(fisher_values.begin(), fisher_values.end(), 0.0f) / fisher_values.size();

        size_t q05_idx = static_cast<size_t>(fisher_values.size() * 0.05f);
        size_t q95_idx = static_cast<size_t>(fisher_values.size() * 0.95f);
        result.fisher_q05 = fisher_values[std::min(q05_idx, fisher_values.size() - 1)];
        result.fisher_q95 = fisher_values[std::min(q95_idx, fisher_values.size() - 1)];
    }

    // Graph entropy = degree entropy + mean local entropy
    result.graph_entropy = result.degree_entropy + result.mean_local_entropy;

    return result;
}

EntropyAnalysisResult analyze_entropy_timestep(
    const TimestepAggregation& timestep,
    const EntropyConfig& config
) {
    SimpleGraph graph;
    graph.build(timestep.union_vertices, timestep.union_edges);

    const std::vector<float>* dims = timestep.mean_dimensions.empty() ? nullptr : &timestep.mean_dimensions;

    return analyze_entropy(graph, config, dims);
}

// =============================================================================
// Parallel Implementations
// =============================================================================

std::unordered_map<VertexId, float> compute_all_local_entropies_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int radius
) {
    const auto& verts = graph.vertices();
    std::vector<float> results(verts.size(), 0.0f);

    for (size_t i = 0; i < verts.size(); ++i) {
        js->submit_function([&graph, &verts, &results, i, radius]() {
            results[i] = compute_local_entropy(graph, verts[i], radius);
        }, 0);
    }
    js->wait_for_completion();

    std::unordered_map<VertexId, float> result;
    for (size_t i = 0; i < verts.size(); ++i) {
        result[verts[i]] = results[i];
    }
    return result;
}

std::unordered_map<VertexId, float> compute_all_mutual_info_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int radius
) {
    const auto& verts = graph.vertices();
    std::vector<float> results(verts.size(), 0.0f);

    for (size_t i = 0; i < verts.size(); ++i) {
        js->submit_function([&graph, &verts, &results, i, radius]() {
            VertexId v = verts[i];
            float total_mi = 0.0f;
            const auto& neighbors = graph.neighbors(v);
            for (VertexId n : neighbors) {
                total_mi += compute_mutual_information(graph, v, n, radius);
            }
            results[i] = neighbors.empty() ? 0.0f : total_mi / neighbors.size();
        }, 0);
    }
    js->wait_for_completion();

    std::unordered_map<VertexId, float> result;
    for (size_t i = 0; i < verts.size(); ++i) {
        result[verts[i]] = results[i];
    }
    return result;
}

std::unordered_map<VertexId, float> compute_all_fisher_info_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const std::vector<float>& vertex_dimensions,
    int radius
) {
    const auto& verts = graph.vertices();
    std::vector<float> results(verts.size(), 0.0f);

    for (size_t i = 0; i < verts.size(); ++i) {
        js->submit_function([&graph, &verts, &results, &vertex_dimensions, i, radius]() {
            results[i] = compute_fisher_information(graph, verts[i], vertex_dimensions, radius);
        }, 0);
    }
    js->wait_for_completion();

    std::unordered_map<VertexId, float> result;
    for (size_t i = 0; i < verts.size(); ++i) {
        result[verts[i]] = results[i];
    }
    return result;
}

EntropyAnalysisResult analyze_entropy_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const EntropyConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    EntropyAnalysisResult result;

    if (graph.vertex_count() == 0) {
        return result;
    }

    // Global degree entropy (fast, no need to parallelize)
    result.degree_entropy = compute_degree_entropy(graph);

    // Local entropy (parallel)
    if (config.compute_local_entropy) {
        result.local_entropy_map = compute_all_local_entropies_parallel(graph, js, config.neighborhood_radius);
    }

    // Mutual information (parallel)
    if (config.compute_mutual_info) {
        result.mutual_info_map = compute_all_mutual_info_parallel(graph, js, config.neighborhood_radius);
    }

    // Fisher information (parallel, requires dimensions)
    if (config.compute_fisher_info && vertex_dimensions && !vertex_dimensions->empty()) {
        result.fisher_info_map = compute_all_fisher_info_parallel(graph, js, *vertex_dimensions, config.fisher_radius);
    }

    // Build per-vertex result (same as serial version)
    const auto& verts = graph.vertices();
    result.vertex_entropies.reserve(verts.size());

    std::vector<float> local_values, fisher_values;

    for (VertexId v : verts) {
        VertexEntropy ve;
        ve.vertex = v;
        ve.degree = static_cast<int>(graph.neighbors(v).size());

        if (result.local_entropy_map.count(v)) {
            ve.local_entropy = result.local_entropy_map[v];
            local_values.push_back(ve.local_entropy);
        }

        if (result.mutual_info_map.count(v)) {
            ve.mutual_info = result.mutual_info_map[v];
            result.total_mutual_info += ve.mutual_info;
        }

        if (result.fisher_info_map.count(v)) {
            ve.fisher_info = result.fisher_info_map[v];
            fisher_values.push_back(ve.fisher_info);
            result.total_fisher_info += ve.fisher_info;
        }

        result.vertex_entropies.push_back(ve);
    }

    // Compute statistics
    if (!local_values.empty()) {
        std::sort(local_values.begin(), local_values.end());
        result.min_local_entropy = local_values.front();
        result.max_local_entropy = local_values.back();
        result.mean_local_entropy = std::accumulate(local_values.begin(), local_values.end(), 0.0f) / local_values.size();

        size_t q05_idx = static_cast<size_t>(local_values.size() * 0.05f);
        size_t q95_idx = static_cast<size_t>(local_values.size() * 0.95f);
        result.local_q05 = local_values[std::min(q05_idx, local_values.size() - 1)];
        result.local_q95 = local_values[std::min(q95_idx, local_values.size() - 1)];
    }

    if (!fisher_values.empty()) {
        std::sort(fisher_values.begin(), fisher_values.end());
        result.min_fisher_info = fisher_values.front();
        result.max_fisher_info = fisher_values.back();
        result.mean_fisher_info = std::accumulate(fisher_values.begin(), fisher_values.end(), 0.0f) / fisher_values.size();

        size_t q05_idx = static_cast<size_t>(fisher_values.size() * 0.05f);
        size_t q95_idx = static_cast<size_t>(fisher_values.size() * 0.95f);
        result.fisher_q05 = fisher_values[std::min(q05_idx, fisher_values.size() - 1)];
        result.fisher_q95 = fisher_values[std::min(q95_idx, fisher_values.size() - 1)];
    }

    // Graph entropy = degree entropy + mean local entropy
    result.graph_entropy = result.degree_entropy + result.mean_local_entropy;

    return result;
}

}  // namespace viz::blackhole
