#pragma once

#include "bh_types.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Forward declaration for job system
namespace job_system {
    template<typename T> class JobSystem;
}

namespace viz::blackhole {

// =============================================================================
// Graph Representation for Analysis
// =============================================================================
// A simple adjacency list graph for computing distances

class SimpleGraph {
public:
    SimpleGraph() = default;

    // Build from edges (assumes vertices are 0..max_vertex_id)
    // Stores both symmetric adjacency and directed (outgoing-only) adjacency
    void build_from_edges(const std::vector<Edge>& edges);

    // Build from explicit vertex list and edges
    void build(const std::vector<VertexId>& vertices, const std::vector<Edge>& edges);

    // Get neighbors of a vertex (symmetric/undirected)
    const std::vector<VertexId>& neighbors(VertexId v) const;

    // Get outgoing neighbors only (directed)
    const std::vector<VertexId>& out_neighbors(VertexId v) const;

    // Get all vertices
    const std::vector<VertexId>& vertices() const { return vertices_; }

    // Number of vertices/edges
    size_t vertex_count() const { return vertices_.size(); }
    size_t edge_count() const { return edge_count_; }

    // Check if vertex exists
    bool has_vertex(VertexId v) const;

    // Compute shortest path distance using BFS
    int distance(VertexId from, VertexId to) const;

    // Compute all-pairs shortest paths (returns distance matrix indexed by vertex position)
    std::vector<std::vector<int>> all_pairs_distances() const;

    // Compute all-pairs using only outgoing edges (directed)
    std::vector<std::vector<int>> all_pairs_distances_directed() const;

    // Compute distances from one vertex to all others
    std::vector<int> distances_from(VertexId source) const;

    // Truncated BFS - only explores up to max_radius (faster for local dimension)
    std::vector<int> distances_from_truncated(VertexId source, int max_radius) const;

    // Compute graph radius (minimum eccentricity) - matches Mathematica's GraphRadius[]
    // This is the eccentricity of the graph's center vertex
    int graph_radius() const;

    // Compute eccentricity of a single vertex (max distance to any other vertex)
    int eccentricity(VertexId v) const;

    // Parallel versions using job system
    std::vector<std::vector<int>> all_pairs_distances_parallel(
        job_system::JobSystem<int>* js) const;

    std::vector<std::vector<int>> all_pairs_distances_directed_parallel(
        job_system::JobSystem<int>* js) const;

private:
    std::vector<VertexId> vertices_;
    std::unordered_map<VertexId, std::vector<VertexId>> adjacency_;  // Symmetric (undirected)
    std::unordered_map<VertexId, std::vector<VertexId>> out_adjacency_;  // Outgoing only (directed)
    std::unordered_map<VertexId, size_t> vertex_to_index_;
    size_t edge_count_ = 0;
    static const std::vector<VertexId> empty_neighbors_;
};

// =============================================================================
// Anchor Selection
// =============================================================================

// Find vertices present in ALL graphs (stable vertices)
std::vector<VertexId> find_stable_vertices(
    const std::vector<SimpleGraph>& graphs
);

// Overload: find stable vertices from nested structure (avoids copying)
std::vector<VertexId> find_stable_vertices_nested(
    const std::vector<std::vector<SimpleGraph>>& states_by_step
);

// Greedy anchor selection: pick k vertices maximizing minimum pairwise distance
// min_separation: reject anchors closer than this to existing anchors
std::vector<VertexId> select_anchors(
    const SimpleGraph& graph,
    const std::vector<VertexId>& candidates,
    int k,
    int min_separation = 3
);

// =============================================================================
// Geodesic Coordinates
// =============================================================================

// Compute geodesic coordinates for all vertices relative to anchors
// Returns map: vertex -> [distance to anchor 0, distance to anchor 1, ...]
std::unordered_map<VertexId, std::vector<int>> compute_geodesic_coordinates(
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors
);

// Version using pre-computed distance matrix (faster for repeated calls)
std::unordered_map<VertexId, std::vector<int>> compute_geodesic_coordinates_from_matrix(
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors,
    const std::vector<std::vector<int>>& dist_matrix
);

// =============================================================================
// Dimension Configuration
// =============================================================================

enum class DimensionFormula {
    LinearRegression,    // log N = d * log r + c (default, more robust)
    DiscreteDerivative   // (log N(r) - log N(r-1)) / (log(r+1) - log(r)) per radius
};

enum class AggregationMethod {
    Mean,
    Min,
    Max
};

struct DimensionConfig {
    DimensionFormula formula = DimensionFormula::LinearRegression;
    float saturation_threshold = 0.5f;  // Skip radii where ball > threshold * total (1.0 = disabled)
    int min_radius = 1;
    int max_radius = 5;
    bool directed = false;  // Use directed edges (VertexOutComponent) vs undirected
    AggregationMethod aggregation = AggregationMethod::Mean;  // For discrete derivative
};

// Full statistics for a set of dimension values
struct DimensionStats {
    float mean = 0;
    float min = 0;
    float max = 0;
    float variance = 0;
    float stddev = 0;
    size_t count = 0;
};

// Compute stats from a vector of values
DimensionStats compute_dimension_stats(const std::vector<float>& values);

// =============================================================================
// Local Hausdorff Dimension
// =============================================================================

// Estimate local Hausdorff dimension at a vertex using ball counting
// N(r) ~ r^d => log N = d * log r + c
// Returns the estimated dimension, or -1 if insufficient data
float estimate_local_dimension(
    const std::vector<int>& distances_from_vertex,  // distances to all other vertices
    int max_radius = 5
);

// Version with full configuration
float estimate_local_dimension(
    const std::vector<int>& distances_from_vertex,
    const DimensionConfig& config
);

// Estimate dimension for all vertices in a graph
// Returns vector indexed same as graph.vertices()
std::vector<float> estimate_all_dimensions(
    const SimpleGraph& graph,
    int max_radius = 5
);

// Version using pre-computed distance matrix (faster for repeated calls)
std::vector<float> estimate_all_dimensions_from_matrix(
    const SimpleGraph& graph,
    const std::vector<std::vector<int>>& dist_matrix,
    int max_radius = 5
);

// Version using truncated BFS - O(V * neighborhood) instead of O(V * V)
// Much faster when max_radius << graph diameter
std::vector<float> estimate_all_dimensions_truncated(
    const SimpleGraph& graph,
    int max_radius = 5
);

// Version with full configuration
std::vector<float> estimate_all_dimensions(
    const SimpleGraph& graph,
    const DimensionConfig& config
);

// Parallel versions using job system
std::vector<float> estimate_all_dimensions_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int max_radius = 5
);

std::vector<float> estimate_all_dimensions_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const DimensionConfig& config
);

std::vector<float> estimate_all_dimensions_truncated_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int max_radius = 5
);

// =============================================================================
// State Analysis
// =============================================================================

// Analyze a single state (graph)
StateAnalysis analyze_state(
    StateId state_id,
    uint32_t step,
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors,
    int max_radius = 5
);

// Lightweight analysis: only computes dimensions and coords (no vertex/edge copy)
LightweightAnalysis analyze_state_lightweight(
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors,
    int max_radius = 5
);

// =============================================================================
// Timestep Aggregation
// =============================================================================

// Aggregate multiple states at the same timestep
// Uses coordinate bucketing to compute mean dimension at each topological location
TimestepAggregation aggregate_timestep(
    uint32_t step,
    const std::vector<StateAnalysis>& states_at_step,
    const std::vector<Vec2>& initial_positions  // From BHInitialCondition
);

// Streaming aggregation: takes graphs and lightweight analyses
// More memory efficient - doesn't require copying vertices/edges into StateAnalysis
// Uses global_anchors for consistent coordinate bucketing across timesteps
TimestepAggregation aggregate_timestep_streaming(
    uint32_t step,
    const std::vector<SimpleGraph>& graphs,
    const std::vector<LightweightAnalysis>& analyses,
    const std::vector<Vec2>& initial_positions,
    const std::vector<VertexId>& global_anchors  // Consistent anchors for all timesteps
);

// =============================================================================
// Layout Computation (only available when BLACKHOLE_WITH_LAYOUT is defined)
// =============================================================================

struct LayoutConfig {
    // MUST match the live viewer's LayoutParams exactly!
    float spring_constant = 0.5f;    // Same as viewer
    float repulsion_constant = 0.25f; // Same as viewer
    float damping = 0.1f;            // Same as viewer
    float gravity = 0.1f;            // Same as viewer
    float max_displacement = 0.01f;  // Same as viewer
    int edge_budget = 2000;          // Same as viewer
    // Convergence settings for pre-computation
    float energy_tolerance = 0.0001f;  // Tight tolerance for full convergence
    int max_iterations = 50000;      // Enough for full convergence
    int min_iterations = 500;        // Minimum before checking convergence
    int energy_window = 50;          // Window for energy averaging
};

#ifdef BLACKHOLE_WITH_LAYOUT
// Compute layout for a single timestep
// prev_positions: positions from previous timestep (or initial positions for step 0)
// Returns new positions for all union_vertices
std::vector<Vec2> compute_timestep_layout(
    const TimestepAggregation& timestep,
    const std::vector<Vec2>& prev_positions,
    const std::unordered_map<VertexId, size_t>& prev_vertex_indices,
    const LayoutConfig& config = {}
);

// Compute layouts for all timesteps
// Updates vertex_positions in each TimestepAggregation
void compute_all_layouts(
    std::vector<TimestepAggregation>& timesteps,
    const std::vector<Vec2>& initial_positions,
    const LayoutConfig& config = {}
);
#endif  // BLACKHOLE_WITH_LAYOUT

// =============================================================================
// Full Pipeline
// =============================================================================

// Forward declarations for analysis configs
struct GeodesicConfig;
struct ParticleDetectionConfig;

struct AnalysisConfig {
    // Core dimension analysis settings
    int num_anchors = 6;
    int anchor_min_separation = 3;
    int max_radius = 5;
    float quantile_low = 0.05f;
    float quantile_high = 0.95f;
    LayoutConfig layout;  // Layout parameters

    // Geodesic analysis (test particle tracing)
    // When enabled, traces geodesics from selected sources through the graph
    bool compute_geodesics = false;
    std::vector<VertexId> geodesic_sources;    // Empty = auto-select from high-dimension regions
    int geodesic_max_steps = 50;               // Max path length
    int geodesic_bundle_width = 5;             // Number of paths in each bundle
    bool geodesic_follow_gradient = false;     // Follow dimension gradient vs random walk
    float geodesic_dimension_percentile = 0.9f; // For auto-selecting sources near high-dim regions

    // Particle detection (topological defects via Robertson-Seymour)
    // When enabled, detects non-planar regions (K5/K3,3 minors) indicating particle-like excitations
    bool detect_particles = false;
    bool detect_k5_minors = true;              // Look for K5 graph minors
    bool detect_k33_minors = true;             // Look for K3,3 bipartite minors
    bool detect_dimension_spikes = true;       // Detect via dimension anomalies
    bool detect_high_degree = true;            // Detect high-degree vertices
    float dimension_spike_threshold = 1.5f;    // Multiplier above mean to flag as spike
    float degree_percentile = 0.95f;           // Top 5% by degree

    // Topological charge computation
    bool compute_topological_charge = false;   // Compute per-vertex charge
    float charge_radius = 3.0f;                // Radius for local charge computation
};

// Run full analysis on evolution result
// states_by_step[step] = list of SimpleGraphs at that step
BHAnalysisResult run_analysis(
    const BHInitialCondition& initial,
    const EvolutionConfig& evolution_config,
    const std::vector<std::vector<SimpleGraph>>& states_by_step,
    const AnalysisConfig& analysis_config = {}
);

// Helper: compute quantiles from a vector of values
float quantile(std::vector<float> values, float q);

// =============================================================================
// Performance Timing Utilities
// =============================================================================

// Reset all analysis timing accumulators (call before starting analysis)
void reset_analysis_timers();

// Print accumulated analysis timing (call after analysis completes)
void print_analysis_timing();

} // namespace viz::blackhole
