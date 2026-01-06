#pragma once

#include "bh_types.hpp"

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

} // namespace viz::blackhole
