#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace viz::blackhole {

// =============================================================================
// Topological Defect Types (Robertson-Seymour Forbidden Minors)
// =============================================================================
// From Gorard's paper: particles are persistent localized structures
// characterized by forbidden graph minors (Kuratowski/Wagner theorem).
// Non-planarity indicates particle-like excitations.

enum class TopologicalDefectType {
    None,           // No defect / planar region
    K5Minor,        // Complete graph K5 minor (5-vertex clique structure)
    K33Minor,       // Utility graph K3,3 minor (bipartite non-planarity)
    HighDegree,     // High local vertex degree (potential particle core)
    DimensionSpike, // Anomalously high local dimension
    Unknown         // Detected non-planarity of unknown type
};

inline const char* defect_type_name(TopologicalDefectType type) {
    switch (type) {
        case TopologicalDefectType::None: return "None";
        case TopologicalDefectType::K5Minor: return "K5";
        case TopologicalDefectType::K33Minor: return "K3,3";
        case TopologicalDefectType::HighDegree: return "HighDegree";
        case TopologicalDefectType::DimensionSpike: return "DimensionSpike";
        case TopologicalDefectType::Unknown: return "Unknown";
        default: return "Invalid";
    }
}

// =============================================================================
// Topological Defect Structure
// =============================================================================

struct TopologicalDefect {
    TopologicalDefectType type = TopologicalDefectType::None;
    std::vector<VertexId> core_vertices;    // Vertices forming the defect core
    float charge = 0.0f;                    // Topological charge (Euler characteristic contribution)
    Vec2 centroid = {0, 0};                 // Approximate location (for visualization)
    float radius = 0.0f;                    // Approximate size of defect region
    float local_dimension = 0.0f;          // Average dimension in defect region
    int detection_confidence = 0;           // 0-100 confidence in detection

    bool is_valid() const { return type != TopologicalDefectType::None && !core_vertices.empty(); }

    std::string to_string() const;
};

// =============================================================================
// Topological Charge per Vertex
// =============================================================================
// Charge represents local contribution to non-planarity / Euler characteristic.
// High charge indicates potential particle location.

struct VertexCharge {
    VertexId vertex;
    float charge;                    // Local topological charge
    float normalized_charge;         // Charge normalized to [0, 1]
    int degree;                      // Vertex degree
    float clustering_coefficient;    // Local clustering
};

// =============================================================================
// Detection Configuration
// =============================================================================

struct ParticleDetectionConfig {
    // Minor detection
    int max_minor_search_depth = 5;      // BFS depth for minor search
    int min_clique_size = 4;             // Minimum clique to consider as K5 candidate
    bool detect_k5 = true;               // Look for K5 minors
    bool detect_k33 = true;              // Look for K3,3 minors

    // Charge computation
    bool compute_charges = true;         // Compute per-vertex topological charge
    float charge_radius = 3.0f;          // Radius for local charge computation

    // Dimension-based detection
    bool use_dimension_spikes = true;    // Detect via dimension anomalies
    float dimension_spike_threshold = 1.5f;  // Multiplier above mean to flag

    // Degree-based detection
    bool use_high_degree = true;         // Detect high-degree vertices
    float degree_threshold_percentile = 0.95f;  // Top 5% by degree

    // Position computation (requires vertex positions)
    bool compute_centroids = true;
};

// =============================================================================
// Detection Functions
// =============================================================================

// Detect all topological defects in a graph
std::vector<TopologicalDefect> detect_topological_defects(
    const SimpleGraph& graph,
    const ParticleDetectionConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr,
    const std::vector<Vec2>* vertex_positions = nullptr
);

// Detect K5 minors (complete graph on 5 vertices)
std::vector<TopologicalDefect> detect_k5_minors(
    const SimpleGraph& graph,
    int max_search_depth = 5
);

// Detect K3,3 minors (complete bipartite graph)
std::vector<TopologicalDefect> detect_k33_minors(
    const SimpleGraph& graph,
    int max_search_depth = 5
);

// Detect dimension spikes (anomalously high local dimension)
std::vector<TopologicalDefect> detect_dimension_spikes(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    float spike_threshold = 1.5f,
    const std::vector<Vec2>* vertex_positions = nullptr
);

// Detect high-degree vertices (potential particle cores)
std::vector<TopologicalDefect> detect_high_degree_vertices(
    const SimpleGraph& graph,
    float degree_percentile = 0.95f,
    const std::vector<Vec2>* vertex_positions = nullptr
);

// =============================================================================
// Topological Charge Functions
// =============================================================================

// Compute topological charge at each vertex
// Based on local graph structure: degree, clustering, and local non-planarity
std::vector<VertexCharge> compute_topological_charges(
    const SimpleGraph& graph,
    float radius = 3.0f
);

// Get charge map (vertex ID -> charge value)
std::unordered_map<VertexId, float> compute_charge_map(
    const SimpleGraph& graph,
    float radius = 3.0f
);

// Compute local clustering coefficient for a vertex
float compute_clustering_coefficient(
    const SimpleGraph& graph,
    VertexId vertex
);

// Compute local Euler characteristic contribution
// χ = V - E + F (for planar regions)
float compute_local_euler_characteristic(
    const SimpleGraph& graph,
    VertexId center,
    int radius = 2
);

// =============================================================================
// Graph Minor Detection Utilities
// =============================================================================

// Check if graph contains K5 as a minor (approximation)
bool contains_k5_minor_approx(
    const SimpleGraph& graph,
    std::vector<VertexId>* minor_vertices = nullptr
);

// Check if graph contains K3,3 as a minor (approximation)
bool contains_k33_minor_approx(
    const SimpleGraph& graph,
    std::vector<VertexId>* minor_vertices = nullptr
);

// Find cliques of given size (for K5 detection)
std::vector<std::vector<VertexId>> find_cliques(
    const SimpleGraph& graph,
    int min_size,
    int max_cliques = 100
);

// Find dense subgraphs that might contain minors
std::vector<std::vector<VertexId>> find_dense_subgraphs(
    const SimpleGraph& graph,
    float density_threshold = 0.5f,
    int min_size = 5
);

// =============================================================================
// Particle Tracking Across Evolution
// =============================================================================

struct ParticleTrack {
    std::vector<TopologicalDefect> states;  // Defect at each timestep
    std::vector<uint32_t> timesteps;        // Which timesteps
    VertexId initial_vertex;                // Starting vertex
    bool is_persistent;                     // Present in multiple timesteps
    float mean_charge;                      // Average charge over time
};

// Track particles across evolution timesteps
std::vector<ParticleTrack> track_particles(
    const std::vector<std::vector<TopologicalDefect>>& defects_by_step,
    float matching_radius = 3.0f  // Max distance to consider same particle
);

// =============================================================================
// Analysis Result
// =============================================================================

struct ParticleAnalysisResult {
    std::vector<TopologicalDefect> defects;
    std::vector<VertexCharge> charges;
    std::unordered_map<VertexId, float> charge_map;

    // Statistics
    int num_k5 = 0;
    int num_k33 = 0;
    int num_dimension_spikes = 0;
    int num_high_degree = 0;
    float total_charge = 0;
    float mean_charge = 0;
    float max_charge = 0;

    // For visualization
    std::vector<VertexId> high_charge_vertices;  // Vertices above threshold
};

// Run full particle analysis on a graph
ParticleAnalysisResult analyze_particles(
    const SimpleGraph& graph,
    const ParticleDetectionConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr,
    const std::vector<Vec2>* vertex_positions = nullptr
);

// Run particle analysis on timestep aggregation
ParticleAnalysisResult analyze_particles_timestep(
    const TimestepAggregation& timestep,
    const ParticleDetectionConfig& config = {}
);

}  // namespace viz::blackhole
