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

} // namespace viz::blackhole
