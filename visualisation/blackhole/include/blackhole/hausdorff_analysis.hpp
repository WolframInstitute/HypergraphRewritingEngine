#pragma once

#include "bh_types.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace viz::blackhole {

// =============================================================================
// Graph Representation for Analysis
// =============================================================================
// A simple adjacency list graph for computing distances

class SimpleGraph {
public:
    SimpleGraph() = default;

    // Build from edges (assumes vertices are 0..max_vertex_id)
    void build_from_edges(const std::vector<Edge>& edges);

    // Build from explicit vertex list and edges
    void build(const std::vector<VertexId>& vertices, const std::vector<Edge>& edges);

    // Get neighbors of a vertex
    const std::vector<VertexId>& neighbors(VertexId v) const;

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

    // Compute distances from one vertex to all others
    std::vector<int> distances_from(VertexId source) const;

private:
    std::vector<VertexId> vertices_;
    std::unordered_map<VertexId, std::vector<VertexId>> adjacency_;
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

// Estimate dimension for all vertices in a graph
// Returns vector indexed same as graph.vertices()
std::vector<float> estimate_all_dimensions(
    const SimpleGraph& graph,
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
// Layout Computation
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

// =============================================================================
// Full Pipeline
// =============================================================================

struct AnalysisConfig {
    int num_anchors = 6;
    int anchor_min_separation = 3;
    int max_radius = 5;
    float quantile_low = 0.05f;
    float quantile_high = 0.95f;
    LayoutConfig layout;  // Layout parameters
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
