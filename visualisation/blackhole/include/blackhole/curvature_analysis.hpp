#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <vector>
#include <unordered_map>

namespace viz::blackhole {

// =============================================================================
// Curvature Types
// =============================================================================

enum class CurvatureMethod {
    OllivierRicci,      // Discrete Ricci curvature via optimal transport
    WolframRicci,       // Discrete Ricci curvature via geodesic tube volumes
    DimensionGradient   // Scalar curvature from dimension rate of change
};

inline const char* curvature_method_name(CurvatureMethod m) {
    switch (m) {
        case CurvatureMethod::OllivierRicci: return "Ollivier-Ricci";
        case CurvatureMethod::WolframRicci: return "Wolfram-Ricci";
        case CurvatureMethod::DimensionGradient: return "Dimension Gradient";
        default: return "Unknown";
    }
}

// =============================================================================
// Curvature Result Structures
// =============================================================================

// Per-vertex curvature data
struct VertexCurvature {
    VertexId vertex;
    float ollivier_ricci = 0.0f;      // Ollivier-Ricci curvature (optimal transport)
    float wolfram_ricci = 0.0f;       // Wolfram-Ricci curvature (geodesic ball volumes)
    float dimension_gradient = 0.0f;  // Dimension gradient curvature
    float scalar_curvature = 0.0f;    // Combined scalar curvature estimate
};

// Per-edge Ollivier-Ricci curvature (edges have natural Ricci interpretation)
struct EdgeCurvature {
    VertexId v1, v2;
    float ricci = 0.0f;               // Ricci curvature on this edge
    float wasserstein_distance = 0.0f; // W1 distance between neighbor distributions
};

// Geodesic tube data for Wolfram-Ricci computation
struct GeodesicTube {
    VertexId source, target;
    int geodesic_length = 0;          // Length of shortest path
    int tube_radius = 1;              // Radius of tube around geodesic
    int tube_volume = 0;              // Number of vertices in tube
    float expected_flat_volume = 0.0f; // Expected volume in flat d-dimensional space
    float wolfram_ricci = 0.0f;       // Curvature: 1 - V_graph/V_flat (adjusted)
};

// Full curvature analysis result
struct CurvatureAnalysisResult {
    std::vector<VertexCurvature> vertex_curvatures;
    std::vector<EdgeCurvature> edge_curvatures;
    std::vector<GeodesicTube> geodesic_tubes;  // For Wolfram-Ricci
    std::unordered_map<VertexId, float> ollivier_ricci_map;
    std::unordered_map<VertexId, float> wolfram_ricci_map;
    std::unordered_map<VertexId, float> dimension_gradient_map;

    // Ollivier-Ricci statistics
    float mean_ollivier_ricci = 0.0f;
    float min_ollivier_ricci = 0.0f;
    float max_ollivier_ricci = 0.0f;

    // Wolfram-Ricci statistics
    float mean_wolfram_ricci = 0.0f;
    float min_wolfram_ricci = 0.0f;
    float max_wolfram_ricci = 0.0f;

    // Dimension gradient statistics
    float mean_dimension_gradient = 0.0f;
    float min_dimension_gradient = 0.0f;
    float max_dimension_gradient = 0.0f;

    // For color scaling (percentiles)
    float ricci_q05 = 0.0f;  // 5th percentile (Ollivier)
    float ricci_q95 = 0.0f;  // 95th percentile
    float wolfram_q05 = 0.0f;
    float wolfram_q95 = 0.0f;
    float grad_q05 = 0.0f;
    float grad_q95 = 0.0f;
};

// =============================================================================
// Configuration
// =============================================================================

struct CurvatureConfig {
    // Ollivier-Ricci settings
    bool compute_ollivier_ricci = true;
    float ricci_alpha = 0.5f;         // Laziness parameter (0 = pure random walk, 1 = stay put)
    int ricci_neighbor_radius = 1;    // Neighborhood radius for measure computation

    // Wolfram-Ricci settings (geodesic tube volume method)
    bool compute_wolfram_ricci = true;
    int wolfram_tube_radius = 1;      // Radius of tube around geodesics
    int wolfram_sample_geodesics = 10; // Number of geodesics to sample per vertex
    float wolfram_dimension = 3.0f;   // Assumed dimension for flat space reference (or use measured)
    bool wolfram_use_measured_dim = true; // Use locally measured dimension instead of fixed

    // Dimension gradient settings
    bool compute_dimension_gradient = true;
    int gradient_radius = 2;          // Radius for dimension gradient estimation

    // General settings
    int max_radius = 5;               // Max BFS radius for dimension computation
};

// =============================================================================
// Curvature Computation Functions
// =============================================================================

// Compute full curvature analysis on a graph
CurvatureAnalysisResult analyze_curvature(
    const SimpleGraph& graph,
    const CurvatureConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr  // Pre-computed dimensions (optional)
);

// Compute Ollivier-Ricci curvature for a single edge
// Returns curvature value in range [-1, 1] typically
// Negative = hyperbolic (diverging geodesics)
// Positive = spherical (converging geodesics)
// Zero = flat (parallel geodesics)
float compute_edge_ollivier_ricci(
    const SimpleGraph& graph,
    VertexId v1,
    VertexId v2,
    float alpha = 0.5f
);

// Compute Ollivier-Ricci curvature for all edges
std::vector<EdgeCurvature> compute_all_edge_curvatures(
    const SimpleGraph& graph,
    float alpha = 0.5f
);

// Compute per-vertex Ollivier-Ricci (average of incident edge curvatures)
std::unordered_map<VertexId, float> compute_vertex_ollivier_ricci(
    const SimpleGraph& graph,
    float alpha = 0.5f
);

// Compute dimension gradient curvature
// Based on second derivative of dimension field
std::unordered_map<VertexId, float> compute_dimension_gradient_curvature(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    int radius = 2
);

// =============================================================================
// Wolfram-Ricci Curvature (Geodesic Tube Volume Method)
// =============================================================================

// Compute geodesic ball volume (number of vertices within radius r of center)
int geodesic_ball_volume(
    const SimpleGraph& graph,
    VertexId center,
    int radius
);

// Compute geodesic tube volume (vertices within radius r of the geodesic path)
// Returns the tube and populates the GeodesicTube struct
GeodesicTube compute_geodesic_tube(
    const SimpleGraph& graph,
    VertexId source,
    VertexId target,
    int tube_radius,
    float dimension  // For flat space reference
);

// Compute expected volume of a geodesic ball in flat d-dimensional space
// V(r) = C_d * r^d where C_d is the volume of unit ball
float flat_ball_volume(int radius, float dimension);

// Compute expected volume of a geodesic tube in flat d-dimensional space
// Approximation: V(L, r) ≈ L * C_{d-1} * r^{d-1} for thin tubes
float flat_tube_volume(int length, int radius, float dimension);

// Compute Wolfram-Ricci curvature for a single vertex
// Samples geodesics to neighbors and computes average tube volume deficit
float compute_vertex_wolfram_ricci(
    const SimpleGraph& graph,
    VertexId vertex,
    int tube_radius,
    float dimension,
    int num_samples = 10
);

// Compute Wolfram-Ricci curvature for all vertices
std::unordered_map<VertexId, float> compute_all_wolfram_ricci(
    const SimpleGraph& graph,
    int tube_radius = 1,
    float dimension = 3.0f,
    int samples_per_vertex = 10,
    const std::vector<float>* local_dimensions = nullptr  // Use local dim if provided
);

// =============================================================================
// Wasserstein Distance (for Ollivier-Ricci)
// =============================================================================

// Compute W1 (earth mover's) distance between two probability distributions on vertices
// Uses graph shortest paths as ground metric
float wasserstein_distance_w1(
    const SimpleGraph& graph,
    const std::vector<std::pair<VertexId, float>>& mu,  // Source distribution (vertex, probability)
    const std::vector<std::pair<VertexId, float>>& nu   // Target distribution
);

// =============================================================================
// Utility Functions
// =============================================================================

// Get uniform probability distribution on neighbors of vertex
std::vector<std::pair<VertexId, float>> neighbor_distribution(
    const SimpleGraph& graph,
    VertexId vertex,
    float alpha = 0.5f  // Laziness: probability of staying at vertex
);

// Compute scalar curvature from dimension field (Laplacian-based)
float local_scalar_curvature(
    const SimpleGraph& graph,
    VertexId vertex,
    const std::vector<float>& dimensions,
    int radius = 2
);

// Run curvature analysis on timestep aggregation
CurvatureAnalysisResult analyze_curvature_timestep(
    const TimestepAggregation& timestep,
    const CurvatureConfig& config = {}
);

}  // namespace viz::blackhole
