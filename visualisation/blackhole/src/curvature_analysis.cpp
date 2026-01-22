#include "blackhole/curvature_analysis.hpp"
#include <job_system/job_system.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <queue>
#include <numeric>
#include <limits>
#include <set>
#include <unordered_set>

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
// Wolfram-Ricci Curvature (Geodesic Tube Volume Method)
// =============================================================================

int geodesic_ball_volume(
    const SimpleGraph& graph,
    VertexId center,
    int radius
) {
    // BFS to find all vertices within distance r
    std::unordered_set<VertexId> visited;
    std::queue<std::pair<VertexId, int>> q;
    q.push({center, 0});
    visited.insert(center);

    while (!q.empty()) {
        auto [v, dist] = q.front();
        q.pop();

        if (dist < radius) {
            for (VertexId n : graph.neighbors(v)) {
                if (visited.find(n) == visited.end()) {
                    visited.insert(n);
                    q.push({n, dist + 1});
                }
            }
        }
    }

    return static_cast<int>(visited.size());
}

float flat_ball_volume(int radius, float dimension) {
    // Volume of ball in d dimensions: V = C_d * r^d
    // C_d = pi^(d/2) / Gamma(d/2 + 1)
    // For discrete graphs, we use a simplified approximation
    // that matches the expected scaling behavior

    if (radius <= 0) return 1.0f;
    if (dimension <= 0) return 1.0f;

    // Approximate: for integer d, ball of radius r has roughly (2r+1)^d vertices
    // in a regular lattice. We use a continuous approximation.
    float r = static_cast<float>(radius);

    // Use pi^(d/2) / Gamma(d/2 + 1) * r^d
    // Simplified: ~(2*r)^d / d! for small d, or use tgamma
    float half_d = dimension / 2.0f;
    float c_d = std::pow(std::numbers::pi, half_d) / std::tgamma(half_d + 1.0f);

    return c_d * std::pow(r, dimension);
}

float flat_tube_volume(int length, int radius, float dimension) {
    // Volume of geodesic tube in d dimensions (matches WolframRicciCurvatureTensor.wl)
    // V_expected = π^((d-1)/2) × r^(d+1) × L / Γ((d+1)/2)
    // This is the expected volume of a tube of radius r around a geodesic of length L

    if (length <= 0) return 1.0f;
    if (radius <= 0) return static_cast<float>(length);
    if (dimension <= 1) return static_cast<float>(length);

    float L = static_cast<float>(length);
    float r = static_cast<float>(radius);
    float d = dimension;

    float half_d_minus_1 = (d - 1.0f) / 2.0f;
    float half_d_plus_1 = (d + 1.0f) / 2.0f;

    // V = π^((d-1)/2) × r^(d+1) × L / Γ((d+1)/2)
    return std::pow(std::numbers::pi, half_d_minus_1)
         * std::pow(r, d + 1.0f) * L
         / std::tgamma(half_d_plus_1);
}

// =============================================================================
// Wolfram Scalar Curvature (Ball Volume Method - matches ResourceFunction)
// =============================================================================

float wolfram_scalar_at_radius(
    const SimpleGraph& graph,
    VertexId vertex,
    int radius,
    float dimension
) {
    // ResourceFunction formula:
    // K_r = (6(d+2)/r²) × (1 - V_ball × Γ(d/2+1) / (π^(d/2) × r^d))
    //     = (6(d+2)/r²) × (1 - V_ball/V_expected)
    //
    // This formula comes from the relationship between volume deficit
    // and scalar curvature in Riemannian geometry:
    // V(r) = ω_d × r^d × (1 - R/(6(d+2)) × r² + O(r⁴))

    if (radius <= 0) return 0.0f;
    if (dimension <= 0) return 0.0f;

    float r = static_cast<float>(radius);
    float d = dimension;

    // Compute ball volume
    int ball_vol = geodesic_ball_volume(graph, vertex, radius);
    float v_ball = static_cast<float>(ball_vol);

    // Expected volume in flat d-dimensional space
    float v_expected = flat_ball_volume(radius, dimension);

    if (v_expected <= 0.0f) return 0.0f;

    // Volume ratio
    float volume_ratio = v_ball / v_expected;

    // Scaling factor from differential geometry
    // This makes the curvature dimensionally correct
    float scale = 6.0f * (d + 2.0f) / (r * r);

    // Curvature: positive when V_ball < V_expected (volume deficit)
    //            negative when V_ball > V_expected (volume excess)
    return scale * (1.0f - volume_ratio);
}

float compute_vertex_wolfram_scalar(
    const SimpleGraph& graph,
    VertexId vertex,
    int max_radius,
    float dimension,
    RadiusAggregation aggregation
) {
    if (max_radius <= 0) return 0.0f;

    std::vector<float> curvatures;
    curvatures.reserve(max_radius);

    for (int r = 1; r <= max_radius; ++r) {
        float k = wolfram_scalar_at_radius(graph, vertex, r, dimension);
        curvatures.push_back(k);
    }

    if (curvatures.empty()) return 0.0f;

    switch (aggregation) {
        case RadiusAggregation::Mean: {
            float sum = 0.0f;
            for (float k : curvatures) sum += k;
            return sum / static_cast<float>(curvatures.size());
        }
        case RadiusAggregation::Max:
            return *std::max_element(curvatures.begin(), curvatures.end());
        case RadiusAggregation::Min:
            return *std::min_element(curvatures.begin(), curvatures.end());
        default:
            return curvatures[0];
    }
}

std::unordered_map<VertexId, float> compute_all_wolfram_scalar(
    const SimpleGraph& graph,
    int max_radius,
    float dimension,
    RadiusAggregation aggregation,
    const std::vector<float>* local_dimensions
) {
    std::unordered_map<VertexId, float> result;
    const auto& verts = graph.vertices();

    for (size_t i = 0; i < verts.size(); ++i) {
        VertexId v = verts[i];

        // Use local dimension if provided
        float dim = dimension;
        if (local_dimensions && i < local_dimensions->size()) {
            dim = (*local_dimensions)[i];
            if (dim < 1.0f) dim = dimension;  // Fallback for invalid dims
        }

        result[v] = compute_vertex_wolfram_scalar(graph, v, max_radius, dim, aggregation);
    }

    return result;
}

std::unordered_map<VertexId, float> compute_all_wolfram_scalar_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int max_radius,
    float dimension,
    RadiusAggregation aggregation,
    const std::vector<float>* local_dimensions
) {
    const auto& verts = graph.vertices();
    std::vector<float> results(verts.size(), 0.0f);

    for (size_t i = 0; i < verts.size(); ++i) {
        js->submit_function([&graph, &verts, &results, i, max_radius, dimension,
                            aggregation, local_dimensions]() {
            VertexId v = verts[i];

            // Use local dimension if provided
            float dim = dimension;
            if (local_dimensions && i < local_dimensions->size()) {
                dim = (*local_dimensions)[i];
                if (dim < 1.0f) dim = dimension;
            }

            results[i] = compute_vertex_wolfram_scalar(graph, v, max_radius, dim, aggregation);
        }, 0);
    }

    js->wait_for_completion();

    // Convert to map
    std::unordered_map<VertexId, float> result;
    for (size_t i = 0; i < verts.size(); ++i) {
        result[verts[i]] = results[i];
    }

    return result;
}

GeodesicTube compute_geodesic_tube(
    const SimpleGraph& graph,
    VertexId source,
    VertexId target,
    int tube_radius,
    float dimension
) {
    GeodesicTube tube;
    tube.source = source;
    tube.target = target;
    tube.tube_radius = tube_radius;

    // Find shortest path from source to target
    auto distances = graph.distances_from(source);
    const auto& verts = graph.vertices();

    // Find target index
    auto target_it = std::find(verts.begin(), verts.end(), target);
    if (target_it == verts.end()) {
        tube.geodesic_length = 0;
        tube.tube_volume = 0;
        return tube;
    }
    size_t target_idx = std::distance(verts.begin(), target_it);
    if (target_idx >= distances.size() || distances[target_idx] == std::numeric_limits<int>::max()) {
        tube.geodesic_length = 0;
        tube.tube_volume = 0;
        return tube;
    }

    tube.geodesic_length = distances[target_idx];

    // Reconstruct the geodesic path using BFS parent tracking
    // For simplicity, we'll use an alternative: find all vertices within
    // tube_radius of ANY vertex on the shortest path

    // First, find all vertices on shortest paths (there may be multiple)
    // We'll use a simpler approach: vertices v where d(source,v) + d(v,target) = d(source,target)
    auto distances_from_target = graph.distances_from(target);

    std::unordered_set<VertexId> on_geodesic;
    for (size_t i = 0; i < verts.size(); ++i) {
        if (i < distances.size() && i < distances_from_target.size()) {
            int d_from_source = distances[i];
            int d_to_target = distances_from_target[i];
            if (d_from_source != std::numeric_limits<int>::max() &&
                d_to_target != std::numeric_limits<int>::max() &&
                d_from_source + d_to_target == tube.geodesic_length) {
                on_geodesic.insert(verts[i]);
            }
        }
    }

    // Now find all vertices within tube_radius of the geodesic
    std::unordered_set<VertexId> in_tube;
    for (VertexId geo_v : on_geodesic) {
        // BFS from each geodesic vertex
        std::queue<std::pair<VertexId, int>> q;
        q.push({geo_v, 0});
        std::unordered_set<VertexId> local_visited;
        local_visited.insert(geo_v);

        while (!q.empty()) {
            auto [v, dist] = q.front();
            q.pop();
            in_tube.insert(v);

            if (dist < tube_radius) {
                for (VertexId n : graph.neighbors(v)) {
                    if (local_visited.find(n) == local_visited.end()) {
                        local_visited.insert(n);
                        q.push({n, dist + 1});
                    }
                }
            }
        }
    }

    tube.tube_volume = static_cast<int>(in_tube.size());
    tube.expected_flat_volume = flat_tube_volume(tube.geodesic_length, tube_radius, dimension);

    // Wolfram-Ricci: measure deviation from flat space
    // κ = (d-1)/d * (1 - V_graph / V_flat) for d-dimensional Ricci
    // Simplified: we use a normalized version
    if (tube.expected_flat_volume > 0) {
        float volume_ratio = static_cast<float>(tube.tube_volume) / tube.expected_flat_volume;
        // Positive curvature: V_graph < V_flat (volume deficit)
        // Negative curvature: V_graph > V_flat (volume excess)
        tube.wolfram_ricci = 1.0f - volume_ratio;
    } else {
        tube.wolfram_ricci = 0.0f;
    }

    return tube;
}

// Compute the full Wolfram-Ricci tensor at a specific radius
// Formula from WolframRicciCurvatureTensor.wl (lines 614-771):
// R_μν = ScalarCurvature - ((d+1)/((d-1)×r²)) × (1 - TubeVolume/V_expected)
// where V_expected = π^((d-1)/2) × r^(d+1) × TubeLength / Γ((d+1)/2)
float compute_wolfram_ricci_tensor_at_radius(
    const SimpleGraph& graph,
    VertexId source,
    VertexId target,
    int radius,
    float dimension
) {
    if (radius <= 0 || dimension <= 1.0f) return 0.0f;

    // 1. Compute scalar curvature at source using ball volume method
    float scalar = wolfram_scalar_at_radius(graph, source, radius, dimension);

    // 2. Compute geodesic tube
    auto tube = compute_geodesic_tube(graph, source, target, radius, dimension);
    if (tube.geodesic_length == 0) return scalar;  // Disconnected, return scalar only

    // 3. Expected tube volume in flat d-dimensional space
    // V_expected = π^((d-1)/2) × r^(d+1) × L / Γ((d+1)/2)
    float d = dimension;
    float r = static_cast<float>(radius);
    float L = static_cast<float>(tube.geodesic_length);

    float half_d_minus_1 = (d - 1.0f) / 2.0f;
    float half_d_plus_1 = (d + 1.0f) / 2.0f;

    // Guard against division by zero in gamma function for d near 1
    if (half_d_plus_1 <= 0.0f) return scalar;

    float v_expected = std::pow(std::numbers::pi, half_d_minus_1)
                     * std::pow(r, d + 1.0f) * L
                     / std::tgamma(half_d_plus_1);

    if (v_expected <= 0.0f) return scalar;

    // 4. Tube correction term
    // Correction = ((d+1)/((d-1)×r²)) × (1 - TubeVolume/V_expected)
    float tube_ratio = static_cast<float>(tube.tube_volume) / v_expected;

    // Guard against d=1 which causes division by zero
    float d_minus_1 = d - 1.0f;
    if (std::abs(d_minus_1) < 1e-6f) d_minus_1 = 1e-6f;

    float tube_scale = (d + 1.0f) / (d_minus_1 * r * r);
    float tube_correction = tube_scale * (1.0f - tube_ratio);

    // 5. Full tensor: R_μν = Scalar - TubeCorrection
    return scalar - tube_correction;
}

float compute_vertex_wolfram_ricci(
    const SimpleGraph& graph,
    VertexId vertex,
    int max_tube_radius,
    float dimension,
    int num_samples,
    bool use_full_tensor
) {
    const auto& neighbors = graph.neighbors(vertex);
    if (neighbors.empty()) return 0.0f;

    // Compute distances to all vertices (needed for adaptive radius)
    auto distances = graph.distances_from(vertex);
    const auto& verts = graph.vertices();

    // Find vertex index for distance lookup
    std::unordered_map<VertexId, int> vid_to_dist;
    for (size_t i = 0; i < verts.size() && i < distances.size(); ++i) {
        vid_to_dist[verts[i]] = distances[i];
    }

    // Sample targets at VARIOUS distances (not just neighbors)
    // Like ResourceFunction: sample across the graph for diverse geodesics
    std::vector<std::pair<VertexId, int>> targets_with_dist;  // (target, geodesic_length)

    for (size_t i = 0; i < verts.size() && i < distances.size(); ++i) {
        VertexId v = verts[i];
        int dist = distances[i];
        if (v != vertex && dist > 0 && dist < 1000) {  // Valid, reachable target
            targets_with_dist.push_back({v, dist});
        }
    }

    // Sort by distance to get variety, then sample evenly across distances
    std::sort(targets_with_dist.begin(), targets_with_dist.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Sample evenly: take targets at various distance ranges
    std::vector<std::pair<VertexId, int>> sampled_targets;
    if (!targets_with_dist.empty()) {
        int step = std::max(1, static_cast<int>(targets_with_dist.size()) / num_samples);
        for (size_t i = 0; i < targets_with_dist.size() && sampled_targets.size() < static_cast<size_t>(num_samples); i += step) {
            sampled_targets.push_back(targets_with_dist[i]);
        }
        // Also include some far targets if we have room
        if (sampled_targets.size() < static_cast<size_t>(num_samples) && !targets_with_dist.empty()) {
            sampled_targets.push_back(targets_with_dist.back());
        }
    }

    if (sampled_targets.empty()) return 0.0f;

    // Compute average Wolfram-Ricci over sampled geodesics
    // Using ADAPTIVE radius per target (like ResourceFunction)
    float total_ricci = 0.0f;
    int count = 0;

    for (const auto& [target, geodesic_length] : sampled_targets) {
        // Adaptive radius: min(geodesic_length/2, max_tube_radius)
        // This matches WolframRicciCurvatureTensor.wl default behavior
        int adaptive_radius = std::min((geodesic_length + 1) / 2, max_tube_radius);
        if (adaptive_radius < 1) adaptive_radius = 1;

        float ricci;
        if (use_full_tensor) {
            // Use full tensor formula: average across radii 1 to adaptive_radius
            // This makes the result more radius-independent
            float sum = 0.0f;
            int valid_radii = 0;
            for (int r = 1; r <= adaptive_radius; ++r) {
                float tensor_val = compute_wolfram_ricci_tensor_at_radius(
                    graph, vertex, target, r, dimension);
                if (std::isfinite(tensor_val)) {
                    sum += tensor_val;
                    ++valid_radii;
                }
            }
            ricci = (valid_radii > 0) ? sum / valid_radii : 0.0f;
        } else {
            // Legacy simplified method: just tube volume deficit at adaptive radius
            auto tube = compute_geodesic_tube(graph, vertex, target, adaptive_radius, dimension);
            ricci = (tube.geodesic_length > 0) ? tube.wolfram_ricci : 0.0f;
        }

        if (std::isfinite(ricci)) {
            total_ricci += ricci;
            ++count;
        }
    }

    return count > 0 ? total_ricci / count : 0.0f;
}

std::unordered_map<VertexId, float> compute_all_wolfram_ricci(
    const SimpleGraph& graph,
    int tube_radius,
    float dimension,
    int samples_per_vertex,
    const std::vector<float>* local_dimensions,
    bool use_full_tensor
) {
    std::unordered_map<VertexId, float> result;
    const auto& verts = graph.vertices();

    for (size_t i = 0; i < verts.size(); ++i) {
        VertexId v = verts[i];

        // Use local dimension if provided
        float dim = dimension;
        if (local_dimensions && i < local_dimensions->size()) {
            dim = (*local_dimensions)[i];
            if (dim < 1.0f) dim = dimension;  // Fallback for invalid dims
        }

        result[v] = compute_vertex_wolfram_ricci(graph, v, tube_radius, dim, samples_per_vertex, use_full_tensor);
    }

    return result;
}

// =============================================================================
// Full Curvature Analysis
// =============================================================================

// Helper function to compute statistics for a vector of values
static void compute_statistics(
    const std::vector<float>& values,
    float& mean, float& min_val, float& max_val,
    float& q05, float& q95
) {
    if (values.empty()) return;

    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    min_val = sorted.front();
    max_val = sorted.back();
    mean = std::accumulate(sorted.begin(), sorted.end(), 0.0f) / sorted.size();

    size_t q05_idx = static_cast<size_t>(sorted.size() * 0.05f);
    size_t q95_idx = static_cast<size_t>(sorted.size() * 0.95f);
    q05 = sorted[std::min(q05_idx, sorted.size() - 1)];
    q95 = sorted[std::min(q95_idx, sorted.size() - 1)];
}

// Helper function to build per-vertex curvature data and compute statistics
// This is shared between serial and parallel analysis functions
static void finalize_curvature_result(
    CurvatureAnalysisResult& result,
    const SimpleGraph& graph
) {
    const auto& verts = graph.vertices();
    result.vertex_curvatures.reserve(verts.size());

    std::vector<float> ricci_values, wolfram_values, scalar_values, grad_values;

    for (size_t i = 0; i < verts.size(); ++i) {
        VertexCurvature vc;
        vc.vertex = verts[i];

        if (result.ollivier_ricci_map.count(verts[i])) {
            vc.ollivier_ricci = result.ollivier_ricci_map[verts[i]];
            ricci_values.push_back(vc.ollivier_ricci);
        }

        if (result.wolfram_ricci_map.count(verts[i])) {
            vc.wolfram_ricci = result.wolfram_ricci_map[verts[i]];
            wolfram_values.push_back(vc.wolfram_ricci);
        }

        if (result.wolfram_scalar_map.count(verts[i])) {
            vc.wolfram_scalar = result.wolfram_scalar_map[verts[i]];
            scalar_values.push_back(vc.wolfram_scalar);
        }

        if (result.dimension_gradient_map.count(verts[i])) {
            vc.dimension_gradient = result.dimension_gradient_map[verts[i]];
            grad_values.push_back(vc.dimension_gradient);
        }

        // NOTE: We do NOT compute a combined scalar curvature because the different
        // curvature methods have incompatible units:
        // - Ollivier-Ricci: dimensionless, typically in [-1, 1]
        // - Wolfram Scalar: has units of 1/r² (from the 6(d+2)/r² scaling)
        // - Dimension Gradient: dimensionless but different interpretation
        vc.scalar_curvature = 0.0f;  // Deprecated - use specific curvature fields

        result.vertex_curvatures.push_back(vc);
    }

    // Compute statistics for each curvature type
    compute_statistics(ricci_values,
        result.mean_ollivier_ricci, result.min_ollivier_ricci, result.max_ollivier_ricci,
        result.ricci_q05, result.ricci_q95);

    compute_statistics(wolfram_values,
        result.mean_wolfram_ricci, result.min_wolfram_ricci, result.max_wolfram_ricci,
        result.wolfram_q05, result.wolfram_q95);

    compute_statistics(scalar_values,
        result.mean_wolfram_scalar, result.min_wolfram_scalar, result.max_wolfram_scalar,
        result.scalar_q05, result.scalar_q95);

    compute_statistics(grad_values,
        result.mean_dimension_gradient, result.min_dimension_gradient, result.max_dimension_gradient,
        result.grad_q05, result.grad_q95);
}

CurvatureAnalysisResult analyze_curvature(
    const SimpleGraph& graph,
    const CurvatureConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    CurvatureAnalysisResult result;

    if (graph.vertex_count() == 0) {
        return result;
    }

    // Determine effective max radius (use graph_radius if option enabled, else fixed)
    // This matches ResourceFunction default behavior when use_graph_radius=true
    int effective_max_radius = config.max_radius;
    int effective_scalar_max_radius = config.wolfram_scalar_max_radius;
    int effective_tube_max_radius = config.wolfram_tube_radius;

    if (config.use_graph_radius) {
        int gr = graph.graph_radius();
        if (gr > 0) {
            effective_max_radius = gr;
            effective_scalar_max_radius = gr;
            effective_tube_max_radius = gr;
        }
    }

    // Compute dimensions if not provided (needed for Wolfram curvatures and dimension gradient)
    std::vector<float> dims;
    if (vertex_dimensions) {
        dims = *vertex_dimensions;
    } else if (config.compute_dimension_gradient ||
               (config.compute_wolfram_ricci && config.wolfram_use_measured_dim) ||
               (config.compute_wolfram_scalar && config.wolfram_use_measured_dim)) {
        // Compute dimensions using truncated BFS (efficient for local dimension)
        dims = estimate_all_dimensions_truncated(graph, effective_max_radius);
    }

    // Compute Ollivier-Ricci
    if (config.compute_ollivier_ricci) {
        result.edge_curvatures = compute_all_edge_curvatures(graph, config.ricci_alpha);
        result.ollivier_ricci_map = compute_vertex_ollivier_ricci(graph, config.ricci_alpha);
    }

    // Compute Wolfram-Ricci (geodesic tube volume method)
    if (config.compute_wolfram_ricci) {
        const std::vector<float>* dim_ptr = nullptr;
        if (config.wolfram_use_measured_dim && !dims.empty()) {
            dim_ptr = &dims;
        }
        result.wolfram_ricci_map = compute_all_wolfram_ricci(
            graph,
            effective_tube_max_radius,  // Use effective radius
            config.wolfram_dimension,
            config.wolfram_sample_geodesics,
            dim_ptr,
            config.wolfram_ricci_full_tensor
        );
    }

    // Compute Wolfram scalar curvature (ball volume method - matches ResourceFunction)
    if (config.compute_wolfram_scalar) {
        const std::vector<float>* dim_ptr = nullptr;
        if (config.wolfram_use_measured_dim && !dims.empty()) {
            dim_ptr = &dims;
        }
        result.wolfram_scalar_map = compute_all_wolfram_scalar(
            graph,
            effective_scalar_max_radius,  // Use effective radius
            config.wolfram_dimension,
            config.wolfram_scalar_aggregation,
            dim_ptr
        );
    }

    // Compute dimension gradient curvature
    if (config.compute_dimension_gradient && !dims.empty()) {
        result.dimension_gradient_map = compute_dimension_gradient_curvature(graph, dims, config.gradient_radius);
    }

    // Build per-vertex results and compute statistics (shared helper)
    finalize_curvature_result(result, graph);

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

// =============================================================================
// Parallel Curvature Computation
// =============================================================================

std::vector<EdgeCurvature> compute_all_edge_curvatures_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    float alpha
) {
    // First collect all unique edges
    std::vector<std::pair<VertexId, VertexId>> edges;
    std::set<std::pair<VertexId, VertexId>> seen;

    for (VertexId v : graph.vertices()) {
        for (VertexId n : graph.neighbors(v)) {
            VertexId v1 = std::min(v, n);
            VertexId v2 = std::max(v, n);
            if (seen.find({v1, v2}) == seen.end()) {
                seen.insert({v1, v2});
                edges.push_back({v1, v2});
            }
        }
    }

    // Pre-allocate result vector
    std::vector<EdgeCurvature> result(edges.size());

    // Process edges in parallel
    for (size_t i = 0; i < edges.size(); ++i) {
        js->submit_function([&graph, &edges, &result, i, alpha]() {
            auto [v1, v2] = edges[i];

            EdgeCurvature ec;
            ec.v1 = v1;
            ec.v2 = v2;
            ec.ricci = compute_edge_ollivier_ricci(graph, v1, v2, alpha);

            auto mu = neighbor_distribution(graph, v1, alpha);
            auto nu = neighbor_distribution(graph, v2, alpha);
            ec.wasserstein_distance = wasserstein_distance_w1(graph, mu, nu);

            result[i] = ec;
        }, 0);
    }

    js->wait_for_completion();
    return result;
}

std::unordered_map<VertexId, float> compute_vertex_ollivier_ricci_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    float alpha
) {
    // Compute edge curvatures in parallel
    auto edge_curvatures = compute_all_edge_curvatures_parallel(graph, js, alpha);

    // Build vertex averages (fast, keep sequential)
    std::unordered_map<VertexId, std::vector<float>> vertex_edge_curvatures;
    for (const auto& ec : edge_curvatures) {
        vertex_edge_curvatures[ec.v1].push_back(ec.ricci);
        vertex_edge_curvatures[ec.v2].push_back(ec.ricci);
    }

    std::unordered_map<VertexId, float> result;
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

std::unordered_map<VertexId, float> compute_all_wolfram_ricci_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int tube_radius,
    float dimension,
    int samples_per_vertex,
    const std::vector<float>* local_dimensions,
    bool use_full_tensor
) {
    const auto& verts = graph.vertices();
    std::vector<float> results(verts.size(), 0.0f);

    for (size_t i = 0; i < verts.size(); ++i) {
        js->submit_function([&graph, &verts, &results, i, tube_radius, dimension,
                            samples_per_vertex, local_dimensions, use_full_tensor]() {
            VertexId v = verts[i];

            // Use local dimension if provided
            float dim = dimension;
            if (local_dimensions && i < local_dimensions->size()) {
                dim = (*local_dimensions)[i];
                if (dim < 1.0f) dim = dimension;
            }

            results[i] = compute_vertex_wolfram_ricci(graph, v, tube_radius, dim, samples_per_vertex, use_full_tensor);
        }, 0);
    }

    js->wait_for_completion();

    // Convert to map
    std::unordered_map<VertexId, float> result;
    for (size_t i = 0; i < verts.size(); ++i) {
        result[verts[i]] = results[i];
    }

    return result;
}

std::unordered_map<VertexId, float> compute_dimension_gradient_curvature_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const std::vector<float>& vertex_dimensions,
    int radius
) {
    const auto& verts = graph.vertices();
    std::vector<float> results(verts.size(), 0.0f);

    for (size_t i = 0; i < verts.size(); ++i) {
        js->submit_function([&graph, &verts, &results, &vertex_dimensions, i, radius]() {
            results[i] = local_scalar_curvature(graph, verts[i], vertex_dimensions, radius);
        }, 0);
    }

    js->wait_for_completion();

    // Convert to map
    std::unordered_map<VertexId, float> result;
    for (size_t i = 0; i < verts.size(); ++i) {
        result[verts[i]] = results[i];
    }

    return result;
}

CurvatureAnalysisResult analyze_curvature_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const CurvatureConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    CurvatureAnalysisResult result;

    if (graph.vertex_count() == 0) {
        return result;
    }

    // Compute dimensions if not provided (needed for Wolfram curvatures and dimension gradient)
    std::vector<float> dims;
    if (vertex_dimensions) {
        dims = *vertex_dimensions;
    } else if (config.compute_dimension_gradient ||
               (config.compute_wolfram_ricci && config.wolfram_use_measured_dim) ||
               (config.compute_wolfram_scalar && config.wolfram_use_measured_dim)) {
        // Compute dimensions in parallel using truncated BFS
        dims = estimate_all_dimensions_truncated_parallel(graph, js, config.max_radius);
    }

    // Compute Ollivier-Ricci in parallel
    if (config.compute_ollivier_ricci) {
        result.edge_curvatures = compute_all_edge_curvatures_parallel(graph, js, config.ricci_alpha);
        result.ollivier_ricci_map = compute_vertex_ollivier_ricci_parallel(graph, js, config.ricci_alpha);
    }

    // Compute Wolfram-Ricci in parallel (tube method)
    if (config.compute_wolfram_ricci) {
        const std::vector<float>* dim_ptr = nullptr;
        if (config.wolfram_use_measured_dim && !dims.empty()) {
            dim_ptr = &dims;
        }
        result.wolfram_ricci_map = compute_all_wolfram_ricci_parallel(
            graph, js,
            config.wolfram_tube_radius,
            config.wolfram_dimension,
            config.wolfram_sample_geodesics,
            dim_ptr,
            config.wolfram_ricci_full_tensor
        );
    }

    // Compute Wolfram scalar curvature in parallel (ball method - matches ResourceFunction)
    if (config.compute_wolfram_scalar) {
        const std::vector<float>* dim_ptr = nullptr;
        if (config.wolfram_use_measured_dim && !dims.empty()) {
            dim_ptr = &dims;
        }
        result.wolfram_scalar_map = compute_all_wolfram_scalar_parallel(
            graph, js,
            config.wolfram_scalar_max_radius,
            config.wolfram_dimension,
            config.wolfram_scalar_aggregation,
            dim_ptr
        );
    }

    // Compute dimension gradient curvature in parallel
    if (config.compute_dimension_gradient && !dims.empty()) {
        result.dimension_gradient_map = compute_dimension_gradient_curvature_parallel(
            graph, js, dims, config.gradient_radius);
    }

    // Build per-vertex results and compute statistics (shared helper)
    finalize_curvature_result(result, graph);

    return result;
}

}  // namespace viz::blackhole
