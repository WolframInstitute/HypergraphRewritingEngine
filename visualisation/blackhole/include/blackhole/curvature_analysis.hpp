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
    DimensionGradient   // Scalar curvature from dimension rate of change
};

inline const char* curvature_method_name(CurvatureMethod m) {
    switch (m) {
        case CurvatureMethod::OllivierRicci: return "Ollivier-Ricci";
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
    float ollivier_ricci = 0.0f;      // Ollivier-Ricci curvature
    float dimension_gradient = 0.0f;  // Dimension gradient curvature
    float scalar_curvature = 0.0f;    // Combined scalar curvature estimate
};

// Per-edge Ollivier-Ricci curvature (edges have natural Ricci interpretation)
struct EdgeCurvature {
    VertexId v1, v2;
    float ricci = 0.0f;               // Ricci curvature on this edge
    float wasserstein_distance = 0.0f; // W1 distance between neighbor distributions
};

// Full curvature analysis result
struct CurvatureAnalysisResult {
    std::vector<VertexCurvature> vertex_curvatures;
    std::vector<EdgeCurvature> edge_curvatures;
    std::unordered_map<VertexId, float> ollivier_ricci_map;
    std::unordered_map<VertexId, float> dimension_gradient_map;

    // Statistics
    float mean_ollivier_ricci = 0.0f;
    float min_ollivier_ricci = 0.0f;
    float max_ollivier_ricci = 0.0f;

    float mean_dimension_gradient = 0.0f;
    float min_dimension_gradient = 0.0f;
    float max_dimension_gradient = 0.0f;

    // For color scaling
    float ricci_q05 = 0.0f;  // 5th percentile
    float ricci_q95 = 0.0f;  // 95th percentile
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
