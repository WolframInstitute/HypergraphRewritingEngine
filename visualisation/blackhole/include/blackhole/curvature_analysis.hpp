#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <vector>
#include <unordered_map>

// Forward declaration for job system
namespace job_system {
    template<typename T> class JobSystem;
}

namespace viz::blackhole {

// =============================================================================
// Curvature Types
// =============================================================================

enum class CurvatureMethod {
    OllivierRicci,      // Discrete Ricci curvature via optimal transport
    WolframRicci,       // Discrete Ricci curvature via geodesic tube volumes (legacy)
    WolframScalar,      // Scalar curvature via ball volumes (matches ResourceFunction formula)
    DimensionGradient   // Scalar curvature from dimension rate of change
};

// Method for aggregating curvature across radii
enum class RadiusAggregation {
    Mean,   // Average over all radii (default)
    Max,    // Maximum curvature
    Min     // Minimum curvature
};

inline const char* curvature_method_name(CurvatureMethod m) {
    switch (m) {
        case CurvatureMethod::OllivierRicci: return "Ollivier-Ricci";
        case CurvatureMethod::WolframRicci: return "Wolfram-Ricci (tube)";
        case CurvatureMethod::WolframScalar: return "Wolfram Scalar (ball)";
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
    float wolfram_ricci = 0.0f;       // Wolfram-Ricci curvature (geodesic tube volumes, legacy)
    float wolfram_scalar = 0.0f;      // Wolfram scalar curvature (ball volumes, ResourceFunction formula)
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
    std::unordered_map<VertexId, float> wolfram_ricci_map;      // Tube method (legacy)
    std::unordered_map<VertexId, float> wolfram_scalar_map;     // Ball method (ResourceFunction)
    std::unordered_map<VertexId, float> dimension_gradient_map;

    // Ollivier-Ricci statistics
    float mean_ollivier_ricci = 0.0f;
    float min_ollivier_ricci = 0.0f;
    float max_ollivier_ricci = 0.0f;

    // Wolfram-Ricci statistics (tube method, legacy)
    float mean_wolfram_ricci = 0.0f;
    float min_wolfram_ricci = 0.0f;
    float max_wolfram_ricci = 0.0f;

    // Wolfram scalar statistics (ball method, matches ResourceFunction)
    float mean_wolfram_scalar = 0.0f;
    float min_wolfram_scalar = 0.0f;
    float max_wolfram_scalar = 0.0f;

    // Dimension gradient statistics
    float mean_dimension_gradient = 0.0f;
    float min_dimension_gradient = 0.0f;
    float max_dimension_gradient = 0.0f;

    // For color scaling (percentiles)
    float ricci_q05 = 0.0f;  // 5th percentile (Ollivier)
    float ricci_q95 = 0.0f;  // 95th percentile
    float wolfram_q05 = 0.0f;
    float wolfram_q95 = 0.0f;
    float scalar_q05 = 0.0f;  // Wolfram scalar
    float scalar_q95 = 0.0f;
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
    bool compute_wolfram_ricci = false;  // Disabled by default (use scalar instead)
    int wolfram_tube_radius = 5;      // Max radius cap (actual radius is adaptive: min(geodesic_length/2, this))
    int wolfram_sample_geodesics = 15; // Number of geodesics to sample per vertex (across various distances)
    float wolfram_dimension = 3.0f;   // Assumed dimension for flat space reference (or use measured)
    bool wolfram_use_measured_dim = true; // Use locally measured dimension instead of fixed
    bool wolfram_ricci_full_tensor = true; // Use full tensor formula (Scalar - TubeCorrection) vs simplified

    // Wolfram scalar curvature settings (ball volume method - matches ResourceFunction)
    // Formula: K_r = (6(d+2)/r²) × (1 - V_ball/V_expected)
    bool compute_wolfram_scalar = true;  // Enabled by default
    int wolfram_scalar_max_radius = 5;   // Maximum radius for ball volume computation
    RadiusAggregation wolfram_scalar_aggregation = RadiusAggregation::Mean;  // How to aggregate across radii

    // Dimension gradient settings
    bool compute_dimension_gradient = true;
    int gradient_radius = 2;          // Radius for dimension gradient estimation

    // General settings
    int max_radius = 5;               // Max BFS radius for dimension computation
    bool use_graph_radius = false;    // If true, use GraphRadius instead of fixed max_radius
                                      // (matches ResourceFunction defaults exactly, but expensive for large graphs)
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

// Compute the full Wolfram-Ricci tensor at a specific radius between two vertices
// Formula: R_μν = ScalarCurvature - ((d+1)/((d-1)×r²)) × (1 - TubeVolume/V_expected)
float compute_wolfram_ricci_tensor_at_radius(
    const SimpleGraph& graph,
    VertexId source,
    VertexId target,
    int radius,
    float dimension
);

// Compute Wolfram-Ricci curvature for a single vertex
// Samples geodesics to neighbors and computes average tube volume deficit
// If use_full_tensor=true, uses the complete formula including scalar curvature correction
float compute_vertex_wolfram_ricci(
    const SimpleGraph& graph,
    VertexId vertex,
    int tube_radius,
    float dimension,
    int num_samples = 10,
    bool use_full_tensor = true
);

// Compute Wolfram-Ricci curvature for all vertices
std::unordered_map<VertexId, float> compute_all_wolfram_ricci(
    const SimpleGraph& graph,
    int tube_radius = 1,
    float dimension = 3.0f,
    int samples_per_vertex = 10,
    const std::vector<float>* local_dimensions = nullptr,  // Use local dim if provided
    bool use_full_tensor = true  // Use full tensor formula vs simplified
);

// =============================================================================
// Wolfram Scalar Curvature (Ball Volume Method - ResourceFunction formula)
// =============================================================================

// Compute Wolfram scalar curvature at a single radius for a vertex
// Formula: K_r = (6(d+2)/r²) × (1 - V_ball/V_expected)
// where V_expected = π^(d/2) × r^d / Γ(d/2+1)
float wolfram_scalar_at_radius(
    const SimpleGraph& graph,
    VertexId vertex,
    int radius,
    float dimension
);

// Compute Wolfram scalar curvature for a single vertex
// Aggregates curvature values across radii 1 to max_radius
float compute_vertex_wolfram_scalar(
    const SimpleGraph& graph,
    VertexId vertex,
    int max_radius,
    float dimension,
    RadiusAggregation aggregation = RadiusAggregation::Mean
);

// Compute Wolfram scalar curvature for all vertices
std::unordered_map<VertexId, float> compute_all_wolfram_scalar(
    const SimpleGraph& graph,
    int max_radius = 5,
    float dimension = 3.0f,
    RadiusAggregation aggregation = RadiusAggregation::Mean,
    const std::vector<float>* local_dimensions = nullptr
);

// Parallel version
std::unordered_map<VertexId, float> compute_all_wolfram_scalar_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int max_radius = 5,
    float dimension = 3.0f,
    RadiusAggregation aggregation = RadiusAggregation::Mean,
    const std::vector<float>* local_dimensions = nullptr
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

// =============================================================================
// Parallel Curvature Computation (using job system)
// =============================================================================

// Compute Ollivier-Ricci curvature for all edges in parallel
std::vector<EdgeCurvature> compute_all_edge_curvatures_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    float alpha = 0.5f
);

// Compute per-vertex Ollivier-Ricci in parallel
std::unordered_map<VertexId, float> compute_vertex_ollivier_ricci_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    float alpha = 0.5f
);

// Compute Wolfram-Ricci curvature for all vertices in parallel
std::unordered_map<VertexId, float> compute_all_wolfram_ricci_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int tube_radius = 1,
    float dimension = 3.0f,
    int samples_per_vertex = 10,
    const std::vector<float>* local_dimensions = nullptr,
    bool use_full_tensor = true  // Use full tensor formula vs simplified
);

// Compute dimension gradient curvature in parallel
std::unordered_map<VertexId, float> compute_dimension_gradient_curvature_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const std::vector<float>& vertex_dimensions,
    int radius = 2
);

// Full curvature analysis using job system for parallelization
CurvatureAnalysisResult analyze_curvature_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const CurvatureConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr
);

}  // namespace viz::blackhole
