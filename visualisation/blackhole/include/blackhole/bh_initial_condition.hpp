#pragma once

#include "bh_types.hpp"
#include <array>

namespace viz::blackhole {

// =============================================================================
// Brill-Lindquist Initial Condition Generator
// =============================================================================
//
// Generates a graph representing discrete spacetime around two black holes
// using the Brill-Lindquist conformal factor.
//
// Vertex sampling:
//   - Rejection sampling with density proportional to (1 + m1/2r1 + m2/2r2)^4
//   - Hard exclusion of horizon interiors (r < m/2 for each BH)
//
// Edge construction:
//   - Connect vertices within distance threshold
//   - Reject edges whose midpoint falls inside either horizon

BHInitialCondition generate_brill_lindquist(
    int n_vertices,
    const BHConfig& config
);

// =============================================================================
// Grid with Holes Initial Condition Generator
// =============================================================================
//
// Generates a regular grid graph with circular "holes" where the black holes are.
// Edges within each horizon region are removed.

BHInitialCondition generate_grid_with_holes(
    int grid_width,
    int grid_height,
    const BHConfig& config
);

// =============================================================================
// Solid Grid Initial Condition Generator
// =============================================================================
//
// Generates a regular grid graph without any holes (ignores black hole positions).
// Useful for testing on uniform topology.

BHInitialCondition generate_solid_grid(
    int grid_width,
    int grid_height,
    const BHConfig& config
);

// =============================================================================
// Utility Functions
// =============================================================================

// Check if a point is inside either black hole horizon
bool inside_horizon(const Vec2& point, const BHConfig& config);

// Compute the Brill-Lindquist conformal factor at a point
float conformal_factor(const Vec2& point, const BHConfig& config);

// Compute the volume element (conformal factor ^ 4)
float volume_element(const Vec2& point, const BHConfig& config);

// =============================================================================
// Topology-Aware Distance Functions
// =============================================================================
// All distances are computed in the internal coordinate system of each topology

// Flat: standard Euclidean distance
float flat_distance(const Vec2& a, const Vec2& b);

// Cylinder: θ wraps around, z is linear
float cylinder_distance(const Vec2& a, const Vec2& b, float radius);

// Torus: both θ and φ wrap around
float torus_distance(const Vec2& a, const Vec2& b, float major_radius, float minor_radius);

// Sphere: great circle distance in (θ, φ) coordinates
float sphere_distance(const Vec2& a, const Vec2& b, float radius);

// Klein bottle: θ wraps with z-flip
float klein_distance(const Vec2& a, const Vec2& b, const TopologyConfig& config);

// Möbius strip: θ wraps with z-flip, finite width
float mobius_distance(const Vec2& a, const Vec2& b, const TopologyConfig& config);

// Generic dispatcher
float topology_distance(const Vec2& a, const Vec2& b, const TopologyConfig& config);

// =============================================================================
// Topology-Aware Sampling Functions
// =============================================================================

// Uniform random sampling on topology
std::vector<Vec2> sample_uniform(int n, const TopologyConfig& config, uint32_t seed = 0);

// Regular grid sampling respecting topology (returns positions in internal coords)
std::vector<Vec2> sample_grid(const TopologyConfig& config);

// Poisson disk sampling with topology-aware distance
std::vector<Vec2> sample_poisson(int target_n, const TopologyConfig& config, uint32_t seed = 0);

// =============================================================================
// Generic Initial Condition Generator
// =============================================================================

// Generate initial condition using topology configuration
// This is the new unified entry point that supports all topologies and sampling methods
BHInitialCondition generate_initial_condition(
    int n_vertices,
    const TopologyConfig& topo_config,
    uint32_t seed = 0
);

// =============================================================================
// Coordinate Conversion for Visualization
// =============================================================================
// Convert from internal topology coordinates to 2D display coordinates

// Flat: identity
// Cylinder: (θ, z) → (R*cos(θ), z) - unrolled view
// Torus: (θ, φ) → ((R+r*cos(φ))*cos(θ), (R+r*cos(φ))*sin(θ)) - flattened
// Sphere: (θ, φ) → stereographic projection or equirectangular
// Klein/Möbius: similar to cylinder with appropriate handling

Vec2 topology_to_display(const Vec2& internal_pos, const TopologyConfig& config);

// Convert from internal topology coordinates to 3D display coordinates
// This provides proper 3D embedding for curved topologies:
// - Flat: (x, y) → (x, y, 0)
// - Cylinder: (θ, z) → (R*cos(θ), R*sin(θ), z)
// - Torus: (θ, φ) → ((R+r*cos(φ))*cos(θ), (R+r*cos(φ))*sin(θ), r*sin(φ))
// - Sphere: (θ, φ) → (R*sin(θ)*cos(φ), R*sin(θ)*sin(φ), R*cos(θ))
// - Klein/Möbius: similar to cylinder
Vec3 topology_to_display_3d(const Vec2& internal_pos, const TopologyConfig& config);

// =============================================================================
// Minkowski Sprinkling
// =============================================================================
// Generate causal set approximations to Minkowski spacetime by random
// "sprinkling" of vertices in spacetime, connecting based on causal structure.
//
// From Gorard's paper: The causal set is constructed by:
// 1. Randomly placing N points in a spacetime region (Poisson process)
// 2. Connecting points x,y if x is in the causal past of y (x ≺ y)
//
// This tests whether the discrete graph structure recovers continuous
// manifold properties (dimension, curvature, geodesics).

struct SprinklingConfig {
    // Spacetime dimensions
    int spatial_dim = 2;              // Number of spatial dimensions (1, 2, or 3)
    float time_extent = 10.0f;        // Time range [0, time_extent]
    float spatial_extent = 10.0f;     // Spatial range [-extent/2, extent/2] per dimension

    // Causal structure
    float lightcone_angle = 1.0f;     // Lightcone angle (c = 1 in natural units)
    bool use_alexandrov_interval = true;  // Connect if in causal diamond (finite interval)
    float alexandrov_cutoff = 5.0f;   // Max proper time separation

    // Sampling
    float density = 0.1f;             // Expected vertices per unit spacetime volume
    uint32_t seed = 0;                // Random seed (0 = random)

    // Edge construction
    bool transitivity_reduction = true;   // Remove redundant edges (keep only direct links)
    int max_edges_per_vertex = 50;        // Limit connectivity (0 = unlimited)
};

// Spacetime point (t, x) or (t, x, y) or (t, x, y, z)
struct SpacetimePoint {
    float t = 0.0f;                   // Time coordinate
    std::array<float, 3> x = {0, 0, 0};  // Spatial coordinates (up to 3)
};

// Result includes causal ordering info
struct SprinklingResult {
    std::vector<SpacetimePoint> points;   // All spacetime points
    std::vector<Edge> causal_edges;       // Directed edges (past → future)
    std::vector<int> time_ordering;       // Points sorted by time
    float mean_causal_density = 0.0f;     // Edges per vertex
    float dimension_estimate = 0.0f;      // Estimated dimension from causet structure
    bool is_faithful = false;             // True if approximates continuum well
};

// Generate Minkowski sprinkling (random vertex placement)
SprinklingResult generate_minkowski_sprinkling(
    int n_vertices,
    const SprinklingConfig& config
);

// Check causal relationship: returns true if a is in the causal past of b
bool causally_precedes(
    const SpacetimePoint& a,
    const SpacetimePoint& b,
    const SprinklingConfig& config
);

// Compute Lorentzian interval: τ² = t² - |x|² (positive for timelike)
float lorentzian_interval(
    const SpacetimePoint& a,
    const SpacetimePoint& b
);

// Estimate dimension from causal set structure
// Uses the Myrheim-Meyer dimension estimator: d = f(n_pairs, n_relations)
float estimate_causet_dimension(
    const SprinklingResult& causet,
    int sample_size = 1000
);

// Test faithfulness: does the causet approximate the continuum?
// Checks geodesic recovery, dimension consistency, etc.
struct FaithfulnessResult {
    float dimension_error = 0.0f;     // |estimated - expected| dimension
    float geodesic_recovery = 0.0f;   // Fraction of geodesics recovered
    float volume_consistency = 0.0f;  // How well volume scaling works
    bool is_faithful = false;         // Overall assessment
};

FaithfulnessResult test_causet_faithfulness(
    const SprinklingResult& causet,
    const SprinklingConfig& config
);

// Convert sprinkling result to BHInitialCondition (for visualization)
BHInitialCondition sprinkling_to_initial_condition(
    const SprinklingResult& causet,
    bool use_time_for_y = true       // Map time coordinate to y-axis
);

} // namespace viz::blackhole
