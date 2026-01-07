#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include "geodesic_analysis.hpp"
#include <vector>
#include <unordered_map>

namespace viz::blackhole {

// =============================================================================
// Rotation Curve Structures
// =============================================================================
// Rotation curves measure orbital velocity vs. radius from a central mass.
// In GR with Newtonian limit, v ∝ 1/√r (inverse square law for gravity).
// Deviation from this indicates modified gravity or "dark matter" effects.

// Single data point on rotation curve
struct RotationCurvePoint {
    int radius;                       // Graph distance from center
    float orbital_velocity;           // Effective orbital velocity
    float expected_velocity;          // Predicted velocity (inverse square)
    float deviation;                  // (actual - expected) / expected
    int num_orbits;                   // Number of orbits sampled at this radius
    float velocity_variance;          // Variance in velocity at this radius
};

// Full rotation curve result
struct RotationCurveResult {
    VertexId center;                  // Central vertex (highest dimension)
    float center_dimension;           // Dimension at center ("mass proxy")

    std::vector<RotationCurvePoint> curve;  // Points on rotation curve

    // Fit parameters
    float power_law_exponent = 0.0f;  // Best-fit exponent: v ∝ r^exponent
    float expected_exponent = -0.5f;  // Newtonian: -0.5 (v ∝ 1/√r)
    float fit_residual = 0.0f;        // RMS residual of fit

    // Dark matter-like indicators
    float flat_region_start = 0.0f;   // Radius where curve flattens
    float flatness_score = 0.0f;      // How flat the outer curve is (0-1)
    bool has_flat_rotation = false;   // Significant deviation from Keplerian

    // Statistics
    float mean_velocity = 0.0f;
    float max_velocity = 0.0f;
    int max_radius = 0;
};

// =============================================================================
// Orbital Path Structure
// =============================================================================

struct OrbitalPath {
    int radius;                       // Distance from center
    std::vector<VertexId> vertices;   // Vertices on orbit
    float circumference;              // Graph circumference (path length)
    float mean_dimension;             // Mean dimension along orbit
    float velocity;                   // Circumference / (2πr) approximation
};

// =============================================================================
// Configuration
// =============================================================================

struct RotationConfig {
    // Center detection
    bool auto_detect_center = true;   // Use highest dimension vertex as center
    VertexId manual_center = 0;       // Manual center (if auto_detect = false)

    // Orbit tracing
    int max_radius = 20;              // Maximum orbit radius
    int min_radius = 2;               // Minimum orbit radius
    int orbits_per_radius = 4;        // Number of orbits to sample at each radius

    // Analysis
    bool compute_power_law_fit = true;
    bool detect_flat_rotation = true;
    float flatness_threshold = 0.1f;  // Threshold for "flat" detection
};

// =============================================================================
// Rotation Curve Functions
// =============================================================================

// Compute full rotation curve analysis
RotationCurveResult analyze_rotation_curve(
    const SimpleGraph& graph,
    const RotationConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr
);

// Find the center vertex (highest dimension = gravitational center)
VertexId find_center_vertex(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions
);

// Get vertices at exact graph distance r from center
std::vector<VertexId> vertices_at_radius(
    const SimpleGraph& graph,
    VertexId center,
    int radius
);

// Trace an orbital path at given radius
// Returns the closed path approximating a circle at that radius
OrbitalPath trace_orbit(
    const SimpleGraph& graph,
    VertexId center,
    int radius,
    const std::vector<float>* dimensions = nullptr
);

// Compute orbital velocity from path
// v = circumference / theoretical_circumference where theoretical = 2πr
float compute_orbital_velocity(
    const OrbitalPath& orbit
);

// Fit power law v = A * r^n to rotation curve
// Returns (A, n, residual)
std::tuple<float, float, float> fit_power_law(
    const std::vector<RotationCurvePoint>& curve
);

// Detect flat rotation curve region (dark matter signature)
// Returns (start_radius, flatness_score)
std::pair<float, float> detect_flat_region(
    const std::vector<RotationCurvePoint>& curve,
    float threshold = 0.1f
);

// =============================================================================
// Utility Functions
// =============================================================================

// Compute expected Newtonian velocity at radius
// v = √(GM/r) ∝ 1/√r, normalized so v(r=1) = 1
float newtonian_velocity(int radius, float central_mass = 1.0f);

// Inverse square law prediction
float inverse_square_velocity(int radius, float central_mass = 1.0f);

// Run rotation analysis on timestep aggregation
RotationCurveResult analyze_rotation_timestep(
    const TimestepAggregation& timestep,
    const RotationConfig& config = {}
);

}  // namespace viz::blackhole
