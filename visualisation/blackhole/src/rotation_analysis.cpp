#include "blackhole/rotation_analysis.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>

namespace viz::blackhole {

// =============================================================================
// Constants
// =============================================================================

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;

// =============================================================================
// Utility Functions
// =============================================================================

float newtonian_velocity(int radius, float central_mass) {
    if (radius <= 0) return 0.0f;
    // v = √(GM/r), normalized so v(r=1) = √(GM)
    return std::sqrt(central_mass / radius);
}

float inverse_square_velocity(int radius, float central_mass) {
    return newtonian_velocity(radius, central_mass);
}

// =============================================================================
// Center Detection
// =============================================================================

VertexId find_center_vertex(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions
) {
    const auto& verts = graph.vertices();
    if (verts.empty() || vertex_dimensions.empty()) {
        return 0;
    }

    VertexId center = verts[0];
    float max_dim = -1.0f;

    for (size_t i = 0; i < verts.size() && i < vertex_dimensions.size(); ++i) {
        if (vertex_dimensions[i] > max_dim) {
            max_dim = vertex_dimensions[i];
            center = verts[i];
        }
    }

    return center;
}

// =============================================================================
// Radius Shell
// =============================================================================

std::vector<VertexId> vertices_at_radius(
    const SimpleGraph& graph,
    VertexId center,
    int radius
) {
    if (radius <= 0) return {center};

    auto distances = graph.distances_from(center);
    const auto& verts = graph.vertices();

    std::vector<VertexId> shell;
    for (size_t i = 0; i < verts.size() && i < distances.size(); ++i) {
        if (distances[i] == radius) {
            shell.push_back(verts[i]);
        }
    }

    return shell;
}

// =============================================================================
// Orbit Tracing
// =============================================================================

OrbitalPath trace_orbit(
    const SimpleGraph& graph,
    VertexId center,
    int radius,
    const std::vector<float>* dimensions
) {
    OrbitalPath orbit;
    orbit.radius = radius;

    // Get shell of vertices at this radius
    auto shell = vertices_at_radius(graph, center, radius);
    if (shell.empty()) {
        orbit.circumference = 0;
        orbit.velocity = 0;
        return orbit;
    }

    // Try to trace a connected path through the shell
    // This approximates a circular orbit
    std::unordered_set<VertexId> shell_set(shell.begin(), shell.end());
    std::unordered_set<VertexId> visited;

    // Start from first vertex in shell
    VertexId current = shell[0];
    orbit.vertices.push_back(current);
    visited.insert(current);

    // Greedy walk through shell
    bool found = true;
    while (found) {
        found = false;
        const auto& neighbors = graph.neighbors(current);

        for (VertexId n : neighbors) {
            if (shell_set.count(n) && !visited.count(n)) {
                orbit.vertices.push_back(n);
                visited.insert(n);
                current = n;
                found = true;
                break;
            }
        }
    }

    // Circumference = path length (edges traversed)
    orbit.circumference = static_cast<float>(orbit.vertices.size());

    // Compute mean dimension along orbit
    if (dimensions && !dimensions->empty()) {
        const auto& verts = graph.vertices();
        float dim_sum = 0.0f;
        int count = 0;

        for (VertexId v : orbit.vertices) {
            auto it = std::find(verts.begin(), verts.end(), v);
            if (it != verts.end()) {
                size_t idx = std::distance(verts.begin(), it);
                if (idx < dimensions->size()) {
                    dim_sum += (*dimensions)[idx];
                    ++count;
                }
            }
        }

        orbit.mean_dimension = count > 0 ? dim_sum / count : 0.0f;
    }

    // Velocity = actual circumference / theoretical circumference
    // Theoretical: 2πr in flat space
    float theoretical = TWO_PI * radius;
    orbit.velocity = theoretical > 0 ? orbit.circumference / theoretical : 0.0f;

    return orbit;
}

float compute_orbital_velocity(const OrbitalPath& orbit) {
    return orbit.velocity;
}

// =============================================================================
// Power Law Fitting
// =============================================================================

std::tuple<float, float, float> fit_power_law(
    const std::vector<RotationCurvePoint>& curve
) {
    // Fit log(v) = log(A) + n * log(r)
    // Using simple linear regression on log-transformed data

    if (curve.size() < 2) {
        return {1.0f, -0.5f, 0.0f};
    }

    std::vector<float> log_r, log_v;
    for (const auto& pt : curve) {
        if (pt.radius > 0 && pt.orbital_velocity > 0) {
            log_r.push_back(std::log(static_cast<float>(pt.radius)));
            log_v.push_back(std::log(pt.orbital_velocity));
        }
    }

    if (log_r.size() < 2) {
        return {1.0f, -0.5f, 0.0f};
    }

    // Linear regression
    float n = static_cast<float>(log_r.size());
    float sum_x = std::accumulate(log_r.begin(), log_r.end(), 0.0f);
    float sum_y = std::accumulate(log_v.begin(), log_v.end(), 0.0f);
    float sum_xx = 0.0f, sum_xy = 0.0f;

    for (size_t i = 0; i < log_r.size(); ++i) {
        sum_xx += log_r[i] * log_r[i];
        sum_xy += log_r[i] * log_v[i];
    }

    float denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-10f) {
        return {1.0f, -0.5f, 0.0f};
    }

    float exponent = (n * sum_xy - sum_x * sum_y) / denom;
    float log_A = (sum_y - exponent * sum_x) / n;
    float A = std::exp(log_A);

    // Compute residual
    float residual = 0.0f;
    for (size_t i = 0; i < log_r.size(); ++i) {
        float predicted = log_A + exponent * log_r[i];
        float diff = log_v[i] - predicted;
        residual += diff * diff;
    }
    residual = std::sqrt(residual / log_r.size());

    return {A, exponent, residual};
}

// =============================================================================
// Flat Rotation Detection
// =============================================================================

std::pair<float, float> detect_flat_region(
    const std::vector<RotationCurvePoint>& curve,
    float threshold
) {
    if (curve.size() < 3) {
        return {0.0f, 0.0f};
    }

    // Look for region where velocity stops decreasing
    // Compute derivative of velocity
    float flat_start = 0.0f;
    float max_flatness = 0.0f;

    for (size_t i = 1; i < curve.size() - 1; ++i) {
        // Compute local slope
        float dv = curve[i + 1].orbital_velocity - curve[i - 1].orbital_velocity;
        float dr = static_cast<float>(curve[i + 1].radius - curve[i - 1].radius);

        if (dr > 0) {
            float slope = std::abs(dv / dr);

            // Flatness = 1 / (1 + slope)
            float flatness = 1.0f / (1.0f + slope * 10.0f);

            if (flatness > max_flatness) {
                max_flatness = flatness;
                flat_start = static_cast<float>(curve[i].radius);
            }
        }
    }

    return {flat_start, max_flatness};
}

// =============================================================================
// Full Rotation Analysis
// =============================================================================

RotationCurveResult analyze_rotation_curve(
    const SimpleGraph& graph,
    const RotationConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    RotationCurveResult result;

    if (graph.vertex_count() == 0) {
        return result;
    }

    // Get or compute dimensions
    std::vector<float> dims;
    if (vertex_dimensions) {
        dims = *vertex_dimensions;
    } else {
        dims = estimate_all_dimensions_truncated(graph, 5);
    }

    // Find center
    if (config.auto_detect_center) {
        result.center = find_center_vertex(graph, dims);
    } else {
        result.center = config.manual_center;
    }

    // Get center dimension (mass proxy)
    const auto& verts = graph.vertices();
    auto center_it = std::find(verts.begin(), verts.end(), result.center);
    if (center_it != verts.end()) {
        size_t idx = std::distance(verts.begin(), center_it);
        if (idx < dims.size()) {
            result.center_dimension = dims[idx];
        }
    }

    // Trace orbits at each radius
    for (int r = config.min_radius; r <= config.max_radius; ++r) {
        std::vector<float> velocities;

        for (int orbit_idx = 0; orbit_idx < config.orbits_per_radius; ++orbit_idx) {
            auto orbit = trace_orbit(graph, result.center, r, &dims);
            if (orbit.circumference > 0) {
                velocities.push_back(orbit.velocity);
            }
        }

        if (!velocities.empty()) {
            RotationCurvePoint pt;
            pt.radius = r;
            pt.orbital_velocity = std::accumulate(velocities.begin(), velocities.end(), 0.0f) / velocities.size();
            pt.expected_velocity = newtonian_velocity(r, result.center_dimension);
            pt.deviation = pt.expected_velocity > 0 ?
                (pt.orbital_velocity - pt.expected_velocity) / pt.expected_velocity : 0.0f;
            pt.num_orbits = static_cast<int>(velocities.size());

            // Compute variance
            float var = 0.0f;
            for (float v : velocities) {
                var += (v - pt.orbital_velocity) * (v - pt.orbital_velocity);
            }
            pt.velocity_variance = velocities.size() > 1 ? var / (velocities.size() - 1) : 0.0f;

            result.curve.push_back(pt);

            if (pt.orbital_velocity > result.max_velocity) {
                result.max_velocity = pt.orbital_velocity;
            }
        }
    }

    result.max_radius = config.max_radius;

    // Compute mean velocity
    if (!result.curve.empty()) {
        float sum = 0.0f;
        for (const auto& pt : result.curve) {
            sum += pt.orbital_velocity;
        }
        result.mean_velocity = sum / result.curve.size();
    }

    // Fit power law
    if (config.compute_power_law_fit && result.curve.size() >= 2) {
        auto [A, n, residual] = fit_power_law(result.curve);
        result.power_law_exponent = n;
        result.fit_residual = residual;
    }

    // Detect flat rotation
    if (config.detect_flat_rotation && result.curve.size() >= 3) {
        auto [start, score] = detect_flat_region(result.curve, config.flatness_threshold);
        result.flat_region_start = start;
        result.flatness_score = score;
        result.has_flat_rotation = score > config.flatness_threshold;
    }

    return result;
}

RotationCurveResult analyze_rotation_timestep(
    const TimestepAggregation& timestep,
    const RotationConfig& config
) {
    SimpleGraph graph;
    graph.build(timestep.union_vertices, timestep.union_edges);

    const std::vector<float>* dims = timestep.mean_dimensions.empty() ? nullptr : &timestep.mean_dimensions;

    return analyze_rotation_curve(graph, config, dims);
}

}  // namespace viz::blackhole
