#include "blackhole/curvature_analysis.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <numeric>
#include <limits>
#include <set>

namespace viz::blackhole {

// =============================================================================
// Wasserstein Distance Implementation
// =============================================================================
// Simplified W1 distance for graph distributions using linear programming relaxation
// For small distributions on graphs, we use a greedy approximation

float wasserstein_distance_w1(
    const SimpleGraph& graph,
    const std::vector<std::pair<VertexId, float>>& mu,
    const std::vector<std::pair<VertexId, float>>& nu
) {
    if (mu.empty() || nu.empty()) return 0.0f;

    // Build distance matrix between support points
    std::vector<VertexId> mu_verts, nu_verts;
    std::vector<float> mu_probs, nu_probs;

    for (const auto& [v, p] : mu) {
        mu_verts.push_back(v);
        mu_probs.push_back(p);
    }
    for (const auto& [v, p] : nu) {
        nu_verts.push_back(v);
        nu_probs.push_back(p);
    }

    // Compute pairwise distances
    std::vector<std::vector<int>> dists(mu_verts.size(), std::vector<int>(nu_verts.size(), 0));
    for (size_t i = 0; i < mu_verts.size(); ++i) {
        auto distances = graph.distances_from(mu_verts[i]);
        for (size_t j = 0; j < nu_verts.size(); ++j) {
            // Find index of nu_verts[j] in graph vertices
            const auto& verts = graph.vertices();
            auto it = std::find(verts.begin(), verts.end(), nu_verts[j]);
            if (it != verts.end()) {
                size_t idx = std::distance(verts.begin(), it);
                dists[i][j] = (idx < distances.size()) ? distances[idx] : std::numeric_limits<int>::max();
            } else {
                dists[i][j] = std::numeric_limits<int>::max();
            }
        }
    }

    // Greedy transport (approximation to optimal transport)
    // Start with remaining mass at each point
    std::vector<float> mu_remain = mu_probs;
    std::vector<float> nu_remain = nu_probs;

    float total_cost = 0.0f;

    // Sort edges by distance and transport greedily
    std::vector<std::tuple<int, size_t, size_t>> edges;  // (distance, mu_idx, nu_idx)
    for (size_t i = 0; i < mu_verts.size(); ++i) {
        for (size_t j = 0; j < nu_verts.size(); ++j) {
            if (dists[i][j] < std::numeric_limits<int>::max()) {
                edges.push_back({dists[i][j], i, j});
            }
        }
    }
    std::sort(edges.begin(), edges.end());

    for (const auto& [d, i, j] : edges) {
        float transport = std::min(mu_remain[i], nu_remain[j]);
        if (transport > 0) {
            total_cost += transport * d;
            mu_remain[i] -= transport;
            nu_remain[j] -= transport;
        }
    }

    return total_cost;
}

// =============================================================================
// Neighbor Distribution
// =============================================================================

std::vector<std::pair<VertexId, float>> neighbor_distribution(
    const SimpleGraph& graph,
    VertexId vertex,
    float alpha
) {
    std::vector<std::pair<VertexId, float>> dist;

    const auto& neighbors = graph.neighbors(vertex);
    if (neighbors.empty()) {
        // Isolated vertex: all mass stays at vertex
        dist.push_back({vertex, 1.0f});
        return dist;
    }

    // Laziness: probability alpha of staying at vertex
    if (alpha > 0) {
        dist.push_back({vertex, alpha});
    }

    // Remaining mass distributed uniformly among neighbors
    float neighbor_prob = (1.0f - alpha) / neighbors.size();
    for (VertexId n : neighbors) {
        dist.push_back({n, neighbor_prob});
    }

    return dist;
}

// =============================================================================
// Ollivier-Ricci Curvature
// =============================================================================

float compute_edge_ollivier_ricci(
    const SimpleGraph& graph,
    VertexId v1,
    VertexId v2,
    float alpha
) {
    // Get neighbor distributions
    auto mu = neighbor_distribution(graph, v1, alpha);
    auto nu = neighbor_distribution(graph, v2, alpha);

    // Compute Wasserstein distance
    float w1 = wasserstein_distance_w1(graph, mu, nu);

    // Edge length in graph is 1 (adjacent vertices)
    float d = 1.0f;

    // Ollivier-Ricci curvature: κ = 1 - W₁/d
    return 1.0f - w1 / d;
}

std::vector<EdgeCurvature> compute_all_edge_curvatures(
    const SimpleGraph& graph,
    float alpha
) {
    std::vector<EdgeCurvature> result;

    // Iterate over all edges (avoid duplicates by only considering v1 < v2)
    std::set<std::pair<VertexId, VertexId>> seen;

    for (VertexId v : graph.vertices()) {
        for (VertexId n : graph.neighbors(v)) {
            VertexId v1 = std::min(v, n);
            VertexId v2 = std::max(v, n);
            if (seen.find({v1, v2}) == seen.end()) {
                seen.insert({v1, v2});

                EdgeCurvature ec;
                ec.v1 = v1;
                ec.v2 = v2;
                ec.ricci = compute_edge_ollivier_ricci(graph, v1, v2, alpha);

                // Also compute Wasserstein distance for reference
                auto mu = neighbor_distribution(graph, v1, alpha);
                auto nu = neighbor_distribution(graph, v2, alpha);
                ec.wasserstein_distance = wasserstein_distance_w1(graph, mu, nu);

                result.push_back(ec);
            }
        }
    }

    return result;
}

std::unordered_map<VertexId, float> compute_vertex_ollivier_ricci(
    const SimpleGraph& graph,
    float alpha
) {
    std::unordered_map<VertexId, float> result;

    // Compute edge curvatures
    auto edge_curvatures = compute_all_edge_curvatures(graph, alpha);

    // Build edge-to-curvature map
    std::unordered_map<VertexId, std::vector<float>> vertex_edge_curvatures;
    for (const auto& ec : edge_curvatures) {
        vertex_edge_curvatures[ec.v1].push_back(ec.ricci);
        vertex_edge_curvatures[ec.v2].push_back(ec.ricci);
    }

    // Average curvature at each vertex
    for (VertexId v : graph.vertices()) {
        const auto& curvatures = vertex_edge_curvatures[v];
        if (curvatures.empty()) {
            result[v] = 0.0f;
        } else {
            float sum = std::accumulate(curvatures.begin(), curvatures.end(), 0.0f);
            result[v] = sum / curvatures.size();
        }
    }

    return result;
}

// =============================================================================
// Dimension Gradient Curvature
// =============================================================================

float local_scalar_curvature(
    const SimpleGraph& graph,
    VertexId vertex,
    const std::vector<float>& dimensions,
    [[maybe_unused]] int radius
) {
    // Get vertex index
    const auto& verts = graph.vertices();
    auto it = std::find(verts.begin(), verts.end(), vertex);
    if (it == verts.end()) return 0.0f;
    size_t v_idx = std::distance(verts.begin(), it);

    if (v_idx >= dimensions.size()) return 0.0f;
    float dim_v = dimensions[v_idx];

    // Compute discrete Laplacian of dimension field
    // Δf(v) = (1/|N(v)|) * Σ_{u ∈ N(v)} (f(u) - f(v))
    const auto& neighbors = graph.neighbors(vertex);
    if (neighbors.empty()) return 0.0f;

    float laplacian = 0.0f;
    int count = 0;

    for (VertexId n : neighbors) {
        auto n_it = std::find(verts.begin(), verts.end(), n);
        if (n_it != verts.end()) {
            size_t n_idx = std::distance(verts.begin(), n_it);
            if (n_idx < dimensions.size()) {
                laplacian += dimensions[n_idx] - dim_v;
                ++count;
            }
        }
    }

    if (count == 0) return 0.0f;
    laplacian /= count;

    // Scalar curvature is related to Laplacian of dimension
    // Higher Laplacian (neighbors have higher dimension) = positive curvature (converging)
    // Lower Laplacian (neighbors have lower dimension) = negative curvature (diverging)
    return laplacian;
}

std::unordered_map<VertexId, float> compute_dimension_gradient_curvature(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    int radius
) {
    std::unordered_map<VertexId, float> result;

    for (VertexId v : graph.vertices()) {
        result[v] = local_scalar_curvature(graph, v, vertex_dimensions, radius);
    }

    return result;
}

// =============================================================================
// Full Curvature Analysis
// =============================================================================

CurvatureAnalysisResult analyze_curvature(
    const SimpleGraph& graph,
    const CurvatureConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    CurvatureAnalysisResult result;

    if (graph.vertex_count() == 0) {
        return result;
    }

    // Compute dimensions if not provided
    std::vector<float> dims;
    if (vertex_dimensions) {
        dims = *vertex_dimensions;
    } else if (config.compute_dimension_gradient) {
        // Compute dimensions using truncated BFS (efficient for local dimension)
        dims = estimate_all_dimensions_truncated(graph, config.max_radius);
    }

    // Compute Ollivier-Ricci
    if (config.compute_ollivier_ricci) {
        result.edge_curvatures = compute_all_edge_curvatures(graph, config.ricci_alpha);
        result.ollivier_ricci_map = compute_vertex_ollivier_ricci(graph, config.ricci_alpha);
    }

    // Compute dimension gradient curvature
    if (config.compute_dimension_gradient && !dims.empty()) {
        result.dimension_gradient_map = compute_dimension_gradient_curvature(graph, dims, config.gradient_radius);
    }

    // Build per-vertex result
    const auto& verts = graph.vertices();
    result.vertex_curvatures.reserve(verts.size());

    std::vector<float> ricci_values, grad_values;

    for (size_t i = 0; i < verts.size(); ++i) {
        VertexCurvature vc;
        vc.vertex = verts[i];

        if (result.ollivier_ricci_map.count(verts[i])) {
            vc.ollivier_ricci = result.ollivier_ricci_map[verts[i]];
            ricci_values.push_back(vc.ollivier_ricci);
        }

        if (result.dimension_gradient_map.count(verts[i])) {
            vc.dimension_gradient = result.dimension_gradient_map[verts[i]];
            grad_values.push_back(vc.dimension_gradient);
        }

        // Combined scalar curvature (average of both methods)
        vc.scalar_curvature = (vc.ollivier_ricci + vc.dimension_gradient) / 2.0f;

        result.vertex_curvatures.push_back(vc);
    }

    // Compute statistics for Ollivier-Ricci
    if (!ricci_values.empty()) {
        std::sort(ricci_values.begin(), ricci_values.end());
        result.min_ollivier_ricci = ricci_values.front();
        result.max_ollivier_ricci = ricci_values.back();
        result.mean_ollivier_ricci = std::accumulate(ricci_values.begin(), ricci_values.end(), 0.0f) / ricci_values.size();

        size_t q05_idx = static_cast<size_t>(ricci_values.size() * 0.05f);
        size_t q95_idx = static_cast<size_t>(ricci_values.size() * 0.95f);
        result.ricci_q05 = ricci_values[std::min(q05_idx, ricci_values.size() - 1)];
        result.ricci_q95 = ricci_values[std::min(q95_idx, ricci_values.size() - 1)];
    }

    // Compute statistics for dimension gradient
    if (!grad_values.empty()) {
        std::sort(grad_values.begin(), grad_values.end());
        result.min_dimension_gradient = grad_values.front();
        result.max_dimension_gradient = grad_values.back();
        result.mean_dimension_gradient = std::accumulate(grad_values.begin(), grad_values.end(), 0.0f) / grad_values.size();

        size_t q05_idx = static_cast<size_t>(grad_values.size() * 0.05f);
        size_t q95_idx = static_cast<size_t>(grad_values.size() * 0.95f);
        result.grad_q05 = grad_values[std::min(q05_idx, grad_values.size() - 1)];
        result.grad_q95 = grad_values[std::min(q95_idx, grad_values.size() - 1)];
    }

    return result;
}

CurvatureAnalysisResult analyze_curvature_timestep(
    const TimestepAggregation& timestep,
    const CurvatureConfig& config
) {
    // Build SimpleGraph from timestep data
    SimpleGraph graph;
    graph.build(timestep.union_vertices, timestep.union_edges);

    // Use mean dimensions if available
    const std::vector<float>* dims = timestep.mean_dimensions.empty() ? nullptr : &timestep.mean_dimensions;

    return analyze_curvature(graph, config, dims);
}

}  // namespace viz::blackhole
