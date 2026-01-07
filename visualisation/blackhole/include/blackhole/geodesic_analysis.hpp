#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace viz::blackhole {

// =============================================================================
// Geodesic Path Structures
// =============================================================================

// A single geodesic path through a graph
struct GeodesicPath {
    std::vector<VertexId> vertices;       // Ordered list of vertices in the path
    std::vector<float> proper_time;       // Cumulative graph distance at each vertex
    std::vector<float> local_dimension;   // Local Hausdorff dimension at each vertex
    VertexId source;                      // Starting vertex
    VertexId destination;                 // Ending vertex (may be same as source if no path)
    int length;                           // Number of edges traversed

    bool is_valid() const { return !vertices.empty(); }
};

// A bundle of nearby geodesics (represents wave packet / test particle)
struct GeodesicBundle {
    std::vector<GeodesicPath> paths;      // Collection of paths in bundle
    VertexId source;                      // Common starting region
    float spread;                         // Measure of bundle divergence
    float mean_proper_time;               // Average proper time across paths
    float proper_time_variance;           // Variance in proper time (time dilation indicator)

    size_t num_paths() const { return paths.size(); }
    bool is_valid() const { return !paths.empty(); }
};

// =============================================================================
// Geodesic Tracing Configuration
// =============================================================================

enum class GeodesicDirection {
    Random,              // Random walk (default for exploration)
    DimensionGradient,   // Follow increasing/decreasing dimension
    ShortestPath,        // BFS shortest path to target
    LongestPath          // Explore furthest reachable
};

struct GeodesicConfig {
    GeodesicDirection direction = GeodesicDirection::Random;
    int max_steps = 50;                   // Maximum path length
    bool follow_dimension_ascent = false; // If true, prefer higher dimension neighbors
    bool avoid_revisits = true;           // Don't revisit vertices in same path
    uint32_t seed = 0;                    // Random seed (0 = use time-based)

    // Bundle configuration
    int bundle_width = 5;                 // Number of paths in bundle
    int bundle_neighbor_radius = 2;       // How far to look for bundle sources
};

// =============================================================================
// Geodesic Tracing Functions
// =============================================================================

// Trace a single geodesic from source vertex
// If dimensions provided, can follow dimension gradient
GeodesicPath trace_geodesic(
    const SimpleGraph& graph,
    VertexId source,
    const GeodesicConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr  // Optional: indexed by graph vertex position
);

// Trace geodesic to specific target (shortest path)
GeodesicPath trace_geodesic_to_target(
    const SimpleGraph& graph,
    VertexId source,
    VertexId target
);

// Trace a bundle of geodesics from neighborhood of source
GeodesicBundle trace_geodesic_bundle(
    const SimpleGraph& graph,
    VertexId source,
    const GeodesicConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr
);

// Trace geodesics from multiple sources
std::vector<GeodesicPath> trace_multiple_geodesics(
    const SimpleGraph& graph,
    const std::vector<VertexId>& sources,
    const GeodesicConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr
);

// =============================================================================
// Geodesic Analysis Functions
// =============================================================================

// Compute bundle spread (how much geodesics diverge)
float compute_bundle_spread(
    const GeodesicBundle& bundle,
    const SimpleGraph& graph
);

// Compute proper time along path (cumulative graph distance)
std::vector<float> compute_proper_time(
    const GeodesicPath& path
);

// Compute dimension along path
std::vector<float> compute_path_dimensions(
    const GeodesicPath& path,
    const SimpleGraph& graph,
    int max_radius = 5
);

// Analyze geodesic curvature (deviation from straight line)
float compute_geodesic_curvature(
    const GeodesicPath& path,
    const SimpleGraph& graph
);

// =============================================================================
// Auto-Selection of Geodesic Sources
// =============================================================================

// Select geodesic sources automatically based on graph structure
// Returns vertices near high-dimension regions (test particles near black holes)
std::vector<VertexId> auto_select_geodesic_sources(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    int num_sources = 5,
    float dimension_percentile = 0.9f  // Select sources near high-dimension regions
);

// Select sources uniformly distributed across graph
std::vector<VertexId> select_distributed_sources(
    const SimpleGraph& graph,
    int num_sources,
    int min_separation = 3
);

// =============================================================================
// Gravitational Lensing Analysis
// =============================================================================
// Compute deflection angles for geodesics passing near high-dimension regions
// Compare to GR prediction: δ = 4GM/c²b where b is impact parameter

struct LensingMetrics {
    float deflection_angle = 0.0f;       // Total angular deflection (radians)
    float impact_parameter = 0.0f;       // Closest approach distance to high-dim center
    float expected_deflection = 0.0f;    // GR prediction: 4GM/c²b (normalized)
    float deflection_ratio = 0.0f;       // actual / expected (1.0 = matches GR)
    VertexId closest_vertex = 0;         // Vertex of closest approach
    float closest_dimension = 0.0f;      // Dimension at closest approach
    bool passes_near_center = false;     // True if passes within threshold of high-dim
};

// Compute lensing metrics for a single geodesic path
LensingMetrics compute_lensing_metrics(
    const GeodesicPath& path,
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    VertexId center,                     // High-dimension center vertex
    float center_dimension               // Dimension at center (proxy for mass)
);

// Compute deflection angle from path geometry
// Uses discrete curvature: angle between incoming and outgoing tangent vectors
float compute_deflection_angle(
    const GeodesicPath& path,
    const SimpleGraph& graph
);

// Compute impact parameter (closest approach distance to center)
std::pair<float, VertexId> compute_impact_parameter(
    const GeodesicPath& path,
    const SimpleGraph& graph,
    VertexId center
);

// Expected GR deflection angle: δ = 4GM/c²b ∝ mass/impact_parameter
// Normalized so that δ = center_dimension / impact_parameter
float expected_gr_deflection(
    float center_dimension,
    float impact_parameter
);

// =============================================================================
// Geodesic Result Aggregation
// =============================================================================

struct GeodesicAnalysisResult {
    std::vector<GeodesicPath> paths;
    std::vector<GeodesicBundle> bundles;
    std::vector<VertexId> sources;

    // Statistics
    float mean_path_length = 0;
    float max_path_length = 0;
    float mean_spread = 0;
    float mean_dimension_variance = 0;  // How much dimension changes along paths

    // Lensing results (per-path)
    std::vector<LensingMetrics> lensing;
    float mean_deflection = 0.0f;
    float mean_deflection_ratio = 0.0f;  // How well deflections match GR (1.0 = perfect)
    VertexId lensing_center = 0;         // Center vertex used for lensing analysis

    // For visualization
    std::vector<std::pair<VertexId, VertexId>> path_edges;  // All edges in all paths
};

// Run full geodesic analysis on a graph
GeodesicAnalysisResult analyze_geodesics(
    const SimpleGraph& graph,
    const std::vector<VertexId>& sources,  // Empty = auto-select
    const GeodesicConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr
);

// Run geodesic analysis on aggregated timestep data
GeodesicAnalysisResult analyze_geodesics_timestep(
    const TimestepAggregation& timestep,
    const std::vector<VertexId>& sources,
    const GeodesicConfig& config = {}
);

} // namespace viz::blackhole
