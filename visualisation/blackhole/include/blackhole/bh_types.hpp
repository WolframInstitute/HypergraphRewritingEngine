#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>

namespace viz::blackhole {

// =============================================================================
// Core Types
// =============================================================================

using VertexId = uint32_t;
using StateId = uint32_t;

struct Vec2 {
    float x, y;

    Vec2() : x(0), y(0) {}
    Vec2(float x_, float y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    Vec2 operator/(float s) const { return {x / s, y / s}; }

    float length() const { return std::sqrt(x * x + y * y); }
    float length_squared() const { return x * x + y * y; }
    Vec2 normalized() const {
        float len = length();
        return len > 0 ? *this / len : Vec2{0, 0};
    }
};

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3(const Vec2& v2, float z_) : x(v2.x), y(v2.y), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    float norm_sq() const { return dot(*this); }
    float norm() const { return std::sqrt(norm_sq()); }
};

struct Vec4 {
    float x, y, z, w;

    Vec4() : x(0), y(0), z(0), w(1) {}
    Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    Vec4(const Vec3& v3, float w_) : x(v3.x), y(v3.y), z(v3.z), w(w_) {}

    Vec4 operator+(const Vec4& o) const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    Vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
};

using EdgeId = uint32_t;
constexpr EdgeId INVALID_EDGE_ID = std::numeric_limits<EdgeId>::max();

struct Edge {
    VertexId v1, v2;
    EdgeId id;  // Unique edge identifier (from hypergraph system)

    Edge() : v1(0), v2(0), id(INVALID_EDGE_ID) {}
    Edge(VertexId a, VertexId b) : v1(a), v2(b), id(INVALID_EDGE_ID) {}
    Edge(VertexId a, VertexId b, EdgeId eid) : v1(a), v2(b), id(eid) {}

    // Compare by ID if both have valid IDs, otherwise by vertices
    bool operator==(const Edge& o) const {
        if (id != INVALID_EDGE_ID && o.id != INVALID_EDGE_ID) {
            return id == o.id;
        }
        return v1 == o.v1 && v2 == o.v2;
    }

    // Less-than for std::set/std::map usage
    bool operator<(const Edge& o) const {
        if (id != INVALID_EDGE_ID && o.id != INVALID_EDGE_ID) {
            return id < o.id;
        }
        if (v1 != o.v1) return v1 < o.v1;
        return v2 < o.v2;
    }
};

// =============================================================================
// Black Hole Initial Condition
// =============================================================================

struct BHConfig {
    float mass1 = 3.0f;
    float mass2 = 3.0f;
    float separation = 10.0f;
    float edge_threshold = 2.0f;
    std::array<float, 2> box_x = {-10.0f, 10.0f};
    std::array<float, 2> box_y = {-10.0f, 10.0f};
};

// =============================================================================
// Topology Configuration (Extended Initial Conditions)
// =============================================================================

enum class TopologyType {
    Flat,           // Current: rectangular 2D domain, no wrapping
    Cylinder,       // Wraps horizontally (left↔right), open vertically
    Torus,          // Wraps both horizontally and vertically
    Sphere,         // S² topology via icosahedral or UV sampling
    KleinBottle,    // Wraps horizontally with twist
    MobiusStrip     // Wraps horizontally with twist, open vertically
};

inline const char* topology_name(TopologyType t) {
    switch (t) {
        case TopologyType::Flat: return "Flat";
        case TopologyType::Cylinder: return "Cylinder";
        case TopologyType::Torus: return "Torus";
        case TopologyType::Sphere: return "Sphere";
        case TopologyType::KleinBottle: return "Klein Bottle";
        case TopologyType::MobiusStrip: return "Möbius Strip";
        default: return "Unknown";
    }
}

enum class SamplingMethod {
    Uniform,        // Equal probability everywhere
    PoissonDisk,    // Blue noise with minimum separation (current default)
    Grid,           // Regular lattice respecting topology
    DensityWeighted // Higher density near defects (generalized Brill-Lindquist)
};

inline const char* sampling_name(SamplingMethod s) {
    switch (s) {
        case SamplingMethod::Uniform: return "Uniform";
        case SamplingMethod::PoissonDisk: return "Poisson Disk";
        case SamplingMethod::Grid: return "Grid";
        case SamplingMethod::DensityWeighted: return "Density Weighted";
        default: return "Unknown";
    }
}

struct DefectConfig {
    int count = 0;                        // Number of defects (0 = none)
    std::vector<Vec2> positions;          // Defect positions (in topology coords)
    std::vector<float> masses;            // Mass/strength of each defect
    float exclusion_radius = 0.5f;        // Remove vertices within this radius
    bool use_conformal_weighting = true;  // Density ∝ ψ⁴ near defects
};

struct TopologyConfig {
    TopologyType type = TopologyType::Flat;
    SamplingMethod sampling = SamplingMethod::PoissonDisk;
    DefectConfig defects;

    // Domain parameters (interpretation depends on topology)
    // Flat: box_x, box_y define rectangular domain
    // Cylinder: theta ∈ [0, 2π), z ∈ [z_min, z_max]
    // Torus: theta ∈ [0, 2π), phi ∈ [0, 2π)
    // Sphere: theta ∈ [0, π], phi ∈ [0, 2π)
    std::array<float, 2> domain_x = {0.0f, 6.28318530718f};  // Default: [0, 2π)
    std::array<float, 2> domain_y = {-10.0f, 10.0f};         // Default: [-10, 10]

    // Topology-specific params
    float major_radius = 10.0f;       // R for torus (major), or radius for cylinder/sphere
    float minor_radius = 3.0f;        // r for torus (minor), or width for Möbius

    // Sampling params
    int grid_resolution = 20;         // For grid sampling
    float poisson_min_distance = 1.0f; // For Poisson disk

    // Edge generation
    float edge_threshold = 0.0f;      // Max distance for edge creation (0 = auto-compute)
};

struct BHInitialCondition {
    BHConfig config;
    std::vector<Vec2> vertex_positions;
    std::vector<float> vertex_z;  // Z coordinates for 3D embedding (empty if 2D)
    std::vector<Edge> edges;

    // Derived
    size_t vertex_count() const { return vertex_positions.size(); }
    size_t edge_count() const { return edges.size(); }
    bool has_3d() const { return !vertex_z.empty(); }

    // Convert to hypergraph edge format for evolution: {{v1, v2}, {v2, v3}, ...}
    std::string to_hge_format() const;
};

// =============================================================================
// Evolution Configuration
// =============================================================================

struct EvolutionConfig {
    std::string rule = "{{a,b},{b,c}} -> {{a,b},{b,c},{c,d},{d,a}}";
    int max_steps = 50;
    int max_states_per_step = 200;
    int max_successors_per_parent = 200;
    float exploration_probability = 1.0f;

    // State canonicalization: deduplicate isomorphic states
    bool canonicalize_states = true;

    // Event canonicalization: deduplicate equivalent events
    bool canonicalize_events = false;

    // Exploration: only explore from canonical state representatives
    bool explore_from_canonical_only = true;

    // Batched matching: collect all matches before spawning REWRITEs
    // (as opposed to eager mode which spawns REWRITEs immediately)
    bool batched_matching = false;

    // Uniform random mode: step-synchronized evolution with random match selection
    // Completes all MATCH tasks at each step, randomly selects matches to apply,
    // then completes all REWRITEs before moving to next step.
    bool uniform_random = false;

    // How many matches to apply per step in uniform random mode (0 = all)
    int matches_per_step = 1;

    // Early termination: stop pattern matching once reservoir is full
    // Trades strict uniform sampling (over ALL matches) for speed
    bool early_terminate_reservoir = false;
};

// =============================================================================
// Per-State Analysis Result
// =============================================================================

constexpr int MAX_ANCHORS = 8;

struct StateAnalysis {
    StateId state_id;
    uint32_t step;

    // Graph structure
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;

    // Dimension analysis per vertex (indexed same as vertices)
    std::vector<float> vertex_dimensions;

    // Geodesic coordinates per vertex (distance to each anchor)
    std::vector<std::array<int, MAX_ANCHORS>> vertex_coords;
    int num_anchors = 0;

    // Stats
    float mean_dimension = 0;
    float min_dimension = 0;
    float max_dimension = 0;

    size_t vertex_count() const { return vertices.size(); }
    size_t edge_count() const { return edges.size(); }
};

// =============================================================================
// Lightweight Analysis Result (for streaming aggregation)
// =============================================================================
// Avoids storing full vertex/edge lists - references original SimpleGraph

struct LightweightAnalysis {
    // Dimension analysis per vertex (indexed same as SimpleGraph.vertices())
    std::vector<float> vertex_dimensions;

    // Geodesic coordinates per vertex
    std::vector<std::array<int, MAX_ANCHORS>> vertex_coords;
    int num_anchors = 0;

    // Per-state curvature (keyed by VertexId for flexible lookup)
    std::unordered_map<VertexId, float> curvature_ollivier;
    std::unordered_map<VertexId, float> curvature_wolfram_scalar;
    std::unordered_map<VertexId, float> curvature_wolfram_ricci;
    std::unordered_map<VertexId, float> curvature_dim_gradient;
};

// =============================================================================
// Aggregated Per-Timestep
// =============================================================================

struct CoordKey {
    std::array<int, MAX_ANCHORS> coords;
    int num_anchors;

    bool operator==(const CoordKey& o) const {
        if (num_anchors != o.num_anchors) return false;
        for (int i = 0; i < num_anchors; ++i) {
            if (coords[i] != o.coords[i]) return false;
        }
        return true;
    }

    std::string to_string() const;
};

struct CoordKeyHash {
    size_t operator()(const CoordKey& k) const {
        size_t h = 0;
        for (int i = 0; i < k.num_anchors; ++i) {
            h ^= std::hash<int>{}(k.coords[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct TimestepAggregation {
    uint32_t step;
    int num_states;  // How many states contributed to this aggregation

    // Union graph across all states at this step
    std::vector<VertexId> union_vertices;
    std::vector<Edge> union_edges;

    // Intersection graph: edges present in ALL states at this step
    std::vector<Edge> intersection_edges;

    // Frequent edges: edges present in MORE than one state (excludes singletons)
    std::vector<Edge> frequent_edges;

    // Vertex positions (from initial condition, geodesic coordinates)
    std::vector<Vec2> vertex_positions;

    // Pre-computed layout positions (force-directed, run to convergence during analysis)
    // Scaled to match vertex_positions bounding sphere for seamless mode switching
    std::vector<Vec2> layout_positions;

    // Mean dimension per vertex (after coordinate bucketing)
    std::vector<float> mean_dimensions;

    // Variance of dimension per vertex (after coordinate bucketing)
    std::vector<float> variance_dimensions;

    // Min/max dimension per vertex across states (for complete branchial stats)
    std::vector<float> min_dimensions;
    std::vector<float> max_dimensions;

    // Raw per-vertex dimensions (before bucketing, averaged across states)
    // Used for "Per-Vertex" aggregation mode (vs "Bucketed" which uses mean_dimensions)
    std::vector<float> raw_vertex_dimensions;

    // Global dimension per vertex (computed on union/timeslice graph, bucketed by coordinate)
    std::vector<float> global_mean_dimensions;
    std::vector<float> global_variance_dimensions;

    // Coordinate to dimension maps (for lookup)
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_to_dim;      // mean
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_to_var;      // variance
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_to_min_dim;  // min
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_to_max_dim;  // max

    // Stats for mean (local/bucketed)
    float pooled_mean = 0;
    float pooled_min = 0;
    float pooled_max = 0;

    // Stats for variance
    float pooled_variance = 0;
    float var_min = 0;
    float var_max = 0;

    // Stats for global mean dimension
    float global_mean_pooled = 0;
    float global_mean_min = 0;
    float global_mean_max = 0;

    // Stats for global variance dimension
    float global_var_pooled = 0;
    float global_var_min = 0;
    float global_var_max = 0;

    // Prefix sums for efficient timeslice aggregation (O(1) range queries)
    // For vertex v: cumulative sum/count of dimension values from step 0 to this step
    // To get average for range [a, b]: (prefix_sum[b][v] - prefix_sum[a-1][v]) / (prefix_count[b][v] - prefix_count[a-1][v])
    std::unordered_map<VertexId, float> dim_prefix_sum;
    std::unordered_map<VertexId, int> dim_prefix_count;

    // Variance prefix sums (for averaging variance across slices)
    std::unordered_map<VertexId, float> var_prefix_sum;
    std::unordered_map<VertexId, int> var_prefix_count;

    // Global dimension prefix sums
    std::unordered_map<VertexId, float> global_dim_prefix_sum;
    std::unordered_map<VertexId, int> global_dim_prefix_count;

    // Global variance prefix sums
    std::unordered_map<VertexId, float> global_var_prefix_sum;
    std::unordered_map<VertexId, int> global_var_prefix_count;

    // =========================================================================
    // Curvature Aggregation (Branchial: mean/variance across states)
    // =========================================================================
    // Bucketed by geodesic coordinate, parallel to mean_dimensions/variance_dimensions

    // Ollivier-Ricci curvature
    std::vector<float> mean_curvature_ollivier;
    std::vector<float> variance_curvature_ollivier;

    // Wolfram scalar curvature (ball volume method)
    std::vector<float> mean_curvature_wolfram_scalar;
    std::vector<float> variance_curvature_wolfram_scalar;

    // Wolfram-Ricci curvature (tube volume method)
    std::vector<float> mean_curvature_wolfram_ricci;
    std::vector<float> variance_curvature_wolfram_ricci;

    // Dimension gradient curvature
    std::vector<float> mean_curvature_dim_gradient;
    std::vector<float> variance_curvature_dim_gradient;

    // =========================================================================
    // Curvature on Union Graph (Foliation: single sample, no variance)
    // =========================================================================
    // Computed on the union graph for this timestep

    std::vector<float> foliation_curvature_ollivier;
    std::vector<float> foliation_curvature_wolfram_scalar;
    std::vector<float> foliation_curvature_wolfram_ricci;
    std::vector<float> foliation_curvature_dim_gradient;

    // =========================================================================
    // Curvature Statistics (for normalization)
    // =========================================================================

    // Branchial mean curvature stats
    float ollivier_mean_min = 0, ollivier_mean_max = 0;
    float wolfram_scalar_mean_min = 0, wolfram_scalar_mean_max = 0;
    float wolfram_ricci_mean_min = 0, wolfram_ricci_mean_max = 0;
    float dim_gradient_mean_min = 0, dim_gradient_mean_max = 0;

    // Branchial variance curvature stats
    float ollivier_var_min = 0, ollivier_var_max = 0;
    float wolfram_scalar_var_min = 0, wolfram_scalar_var_max = 0;
    float wolfram_ricci_var_min = 0, wolfram_ricci_var_max = 0;
    float dim_gradient_var_min = 0, dim_gradient_var_max = 0;

    // Foliation curvature stats
    float foliation_ollivier_min = 0, foliation_ollivier_max = 0;
    float foliation_wolfram_scalar_min = 0, foliation_wolfram_scalar_max = 0;
    float foliation_wolfram_ricci_min = 0, foliation_wolfram_ricci_max = 0;
    float foliation_dim_gradient_min = 0, foliation_dim_gradient_max = 0;

    // =========================================================================
    // Curvature Accessors (indexed by curvature type for extensibility)
    // Index: 0=Ollivier, 1=WolframScalar, 2=WolframRicci, 3=DimGradient
    // =========================================================================
    const std::vector<float>& get_mean_curvature(int idx) const {
        static const std::vector<float> empty;
        switch (idx) {
            case 0: return mean_curvature_ollivier;
            case 1: return mean_curvature_wolfram_scalar;
            case 2: return mean_curvature_wolfram_ricci;
            case 3: return mean_curvature_dim_gradient;
            default: return empty;
        }
    }
    const std::vector<float>& get_variance_curvature(int idx) const {
        static const std::vector<float> empty;
        switch (idx) {
            case 0: return variance_curvature_ollivier;
            case 1: return variance_curvature_wolfram_scalar;
            case 2: return variance_curvature_wolfram_ricci;
            case 3: return variance_curvature_dim_gradient;
            default: return empty;
        }
    }
    const std::vector<float>& get_foliation_curvature(int idx) const {
        static const std::vector<float> empty;
        switch (idx) {
            case 0: return foliation_curvature_ollivier;
            case 1: return foliation_curvature_wolfram_scalar;
            case 2: return foliation_curvature_wolfram_ricci;
            case 3: return foliation_curvature_dim_gradient;
            default: return empty;
        }
    }
    void get_mean_curvature_range(int idx, float& out_min, float& out_max) const {
        switch (idx) {
            case 0: out_min = ollivier_mean_min; out_max = ollivier_mean_max; break;
            case 1: out_min = wolfram_scalar_mean_min; out_max = wolfram_scalar_mean_max; break;
            case 2: out_min = wolfram_ricci_mean_min; out_max = wolfram_ricci_mean_max; break;
            case 3: out_min = dim_gradient_mean_min; out_max = dim_gradient_mean_max; break;
            default: out_min = -1; out_max = 1; break;
        }
    }
    void get_variance_curvature_range(int idx, float& out_min, float& out_max) const {
        switch (idx) {
            case 0: out_min = ollivier_var_min; out_max = ollivier_var_max; break;
            case 1: out_min = wolfram_scalar_var_min; out_max = wolfram_scalar_var_max; break;
            case 2: out_min = wolfram_ricci_var_min; out_max = wolfram_ricci_var_max; break;
            case 3: out_min = dim_gradient_var_min; out_max = dim_gradient_var_max; break;
            default: out_min = 0; out_max = 1; break;
        }
    }
    void get_foliation_curvature_range(int idx, float& out_min, float& out_max) const {
        switch (idx) {
            case 0: out_min = foliation_ollivier_min; out_max = foliation_ollivier_max; break;
            case 1: out_min = foliation_wolfram_scalar_min; out_max = foliation_wolfram_scalar_max; break;
            case 2: out_min = foliation_wolfram_ricci_min; out_max = foliation_wolfram_ricci_max; break;
            case 3: out_min = foliation_dim_gradient_min; out_max = foliation_dim_gradient_max; break;
            default: out_min = -1; out_max = 1; break;
        }
    }
};

// =============================================================================
// Sliding Window Aggregation (optional)
// =============================================================================

struct WindowAggregation {
    uint32_t start_step;
    uint32_t end_step;

    // Same structure as TimestepAggregation but aggregated over window
    std::vector<VertexId> union_vertices;
    std::vector<Edge> union_edges;
    std::vector<Vec2> vertex_positions;
    std::vector<float> mean_dimensions;

    float pooled_mean = 0;
    float pooled_min = 0;
    float pooled_max = 0;
};

// =============================================================================
// Per-State Data (for single-state viewing mode)
// =============================================================================

struct StateData {
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;
};

// =============================================================================
// Per-State Aggregate Values (for scatter plots)
// =============================================================================
// Lightweight struct holding mean values per state for serialization

struct StateAggregate {
    uint32_t state_id;
    uint32_t step;

    // Mean dimension (average across all vertices in this state)
    float mean_dimension = 0;

    // Mean curvatures (average across all vertices in this state)
    float mean_ollivier_ricci = 0;
    float mean_wolfram_scalar = 0;
    float mean_wolfram_ricci = 0;
    float mean_dim_gradient = 0;
};

// =============================================================================
// Complete Analysis Result
// =============================================================================

struct BHAnalysisResult {
    // Input configuration
    BHInitialCondition initial;
    EvolutionConfig evolution_config;

    // Anchor selection
    std::vector<VertexId> anchor_vertices;
    int analysis_max_radius = 5;

    // Per-state analysis (flattened across all steps)
    std::vector<StateAnalysis> per_state;

    // Per-state aggregate values for scatter plots (serialized to .bhdata)
    std::vector<StateAggregate> state_aggregates;

    // Per-timestep aggregation
    std::vector<TimestepAggregation> per_timestep;

    // Optional sliding window aggregations
    std::vector<WindowAggregation> sliding_windows;
    int window_size = 3;  // If used

    // Global dimension range for color scaling (quantile-based)
    float dim_min = 0;
    float dim_max = 3;
    float dim_q05 = 0;  // 5th percentile
    float dim_q95 = 3;  // 95th percentile

    // Global variance range for variance heatmap
    float var_min = 0;
    float var_max = 1;
    float var_q05 = 0;  // 5th percentile
    float var_q95 = 1;  // 95th percentile

    // Global mean dimension (computed on union graph) range for color scaling
    float global_dim_min = 0;
    float global_dim_max = 3;
    float global_dim_q05 = 0;  // 5th percentile
    float global_dim_q95 = 3;  // 95th percentile

    // Global variance dimension range for color scaling
    float global_var_min = 0;
    float global_var_max = 1;
    float global_var_q05 = 0;  // 5th percentile
    float global_var_q95 = 1;  // 95th percentile

    // Evolution stats
    int total_steps = 0;
    int total_states = 0;
    int total_events = 0;

    // Layout info for camera framing
    float layout_bounding_radius = 10.0f;  // Max distance from origin to any laid-out vertex

    // Global vertex/edge union across ALL timesteps (for efficient timeslice view)
    std::vector<VertexId> all_vertices;
    std::vector<Edge> all_edges;

    // Per-state vertices and edges (for single-state viewing mode)
    // states_per_step[step] = vector of StateData for each state at that step
    std::vector<std::vector<StateData>> states_per_step;

    // Mega-union dimension (computed on union across ALL timesteps)
    // Keyed by VertexId, constant across all timesteps
    // Uses geodesic coordinate bucketing for consistency
    std::unordered_map<VertexId, float> mega_dimension;
    float mega_dim_min = 0;
    float mega_dim_max = 0;

    // =========================================================================
    // Geodesic Analysis Results (test particle tracing)
    // =========================================================================
    // Populated when AnalysisConfig::compute_geodesics = true

    bool has_geodesic_analysis = false;

    // Geodesic sources used (either from config or auto-selected)
    std::vector<VertexId> geodesic_sources;

    // Per-timestep geodesic paths: geodesic_paths[step] = paths traced at that step
    // Each path is a sequence of vertex IDs representing the geodesic
    std::vector<std::vector<std::vector<VertexId>>> geodesic_paths;

    // Per-timestep geodesic proper times: geodesic_proper_times[step][path_idx] = times
    // Cumulative graph distance at each vertex along the path
    std::vector<std::vector<std::vector<float>>> geodesic_proper_times;

    // Per-timestep geodesic dimensions: geodesic_dimensions[step][path_idx] = dims
    // Local dimension at each vertex along the path
    std::vector<std::vector<std::vector<float>>> geodesic_dimensions;

    // Geodesic bundle spread per timestep (how much nearby geodesics diverge)
    std::vector<float> geodesic_bundle_spread;

    // Statistics
    float mean_geodesic_length = 0;
    float max_geodesic_length = 0;
    float mean_bundle_spread = 0;

    // =========================================================================
    // Particle Detection Results (topological defects)
    // =========================================================================
    // Populated when AnalysisConfig::detect_particles = true

    bool has_particle_analysis = false;

    // Detected defects per timestep: defects[step] = defects at that step
    // Using simple representation (avoid including full headers here)
    struct DetectedDefect {
        int type;                          // 0=None, 1=K5, 2=K33, 3=HighDegree, 4=DimSpike
        std::vector<VertexId> core_vertices;
        float charge;
        float centroid_x, centroid_y;
        float radius;
        float local_dimension;
        int confidence;
    };
    std::vector<std::vector<DetectedDefect>> detected_defects;

    // Per-vertex topological charge per timestep: charges[step] = map vertex->charge
    std::vector<std::unordered_map<VertexId, float>> vertex_charges;

    // Aggregate charge statistics
    float total_charge = 0;
    float mean_charge = 0;
    float max_charge = 0;

    // Defect counts
    int num_k5_defects = 0;
    int num_k33_defects = 0;
    int num_dimension_spike_defects = 0;
    int num_high_degree_defects = 0;

    // Charge range for visualization
    float charge_min = 0;
    float charge_max = 1;

    // =========================================================================
    // Branch Alignment Results (curvature shape space)
    // =========================================================================
    // Populated when alignment analysis is performed
    // We precompute for both curvature methods so user can toggle without recomputation

    bool has_branch_alignment = false;
    bool has_ollivier_alignment = false;  // Only if --curvature was enabled

    // Per-timestep alignment aggregation (indexed by step)
    struct PerTimestepAlignment {
        std::vector<float> all_pc1;
        std::vector<float> all_pc2;
        std::vector<float> all_pc3;
        std::vector<float> all_curvature;
        std::vector<float> all_rank;
        std::vector<size_t> branch_id;
        std::vector<VertexId> all_vertices;
        std::vector<StateId> state_id;
        std::vector<size_t> branch_sizes;
        size_t total_points = 0;
        size_t num_branches = 0;
    };

    // Wolfram-Ricci alignment (K = 2 - d, always available if dimension computed)
    std::vector<PerTimestepAlignment> alignment_per_timestep;  // Legacy name for compatibility
    float global_pc1_min = 0, global_pc1_max = 0;
    float global_pc2_min = 0, global_pc2_max = 0;
    float global_pc3_min = 0, global_pc3_max = 0;
    float global_curvature_min = 0, global_curvature_max = 0;
    float curvature_abs_max = 0;

    // Ollivier-Ricci alignment (from --curvature analysis)
    std::vector<PerTimestepAlignment> alignment_ollivier;
    float ollivier_pc1_min = 0, ollivier_pc1_max = 0;
    float ollivier_pc2_min = 0, ollivier_pc2_max = 0;
    float ollivier_pc3_min = 0, ollivier_pc3_max = 0;
    float ollivier_curvature_min = 0, ollivier_curvature_max = 0;
    float ollivier_curvature_abs_max = 0;

    // =========================================================================
    // Curvature Analysis Results (for 2D/3D view coloring)
    // =========================================================================
    bool has_curvature_analysis = false;
    std::unordered_map<VertexId, float> curvature_ollivier_ricci;
    std::unordered_map<VertexId, float> curvature_dimension_gradient;
    std::unordered_map<VertexId, float> curvature_wolfram_scalar;  // Ball volume method
    std::unordered_map<VertexId, float> curvature_wolfram_ricci;   // Tube volume method (full tensor)
    float curvature_ollivier_mean = 0, curvature_ollivier_min = 0, curvature_ollivier_max = 0;
    float curvature_dim_grad_mean = 0, curvature_dim_grad_min = 0, curvature_dim_grad_max = 0;
    float curvature_wolfram_scalar_mean = 0, curvature_wolfram_scalar_min = 0, curvature_wolfram_scalar_max = 0;
    float curvature_wolfram_ricci_mean = 0, curvature_wolfram_ricci_min = 0, curvature_wolfram_ricci_max = 0;

    // Global curvature ranges (quantiles across ALL timesteps, for normalization)
    // Branchial mean
    float curv_ollivier_mean_q05 = 0, curv_ollivier_mean_q95 = 0;
    float curv_wolfram_scalar_mean_q05 = 0, curv_wolfram_scalar_mean_q95 = 0;
    float curv_wolfram_ricci_mean_q05 = 0, curv_wolfram_ricci_mean_q95 = 0;
    float curv_dim_gradient_mean_q05 = 0, curv_dim_gradient_mean_q95 = 0;
    // Branchial variance
    float curv_ollivier_var_q05 = 0, curv_ollivier_var_q95 = 0;
    float curv_wolfram_scalar_var_q05 = 0, curv_wolfram_scalar_var_q95 = 0;
    float curv_wolfram_ricci_var_q05 = 0, curv_wolfram_ricci_var_q95 = 0;
    float curv_dim_gradient_var_q05 = 0, curv_dim_gradient_var_q95 = 0;
    // Foliation
    float curv_foliation_ollivier_q05 = 0, curv_foliation_ollivier_q95 = 0;
    float curv_foliation_wolfram_scalar_q05 = 0, curv_foliation_wolfram_scalar_q95 = 0;
    float curv_foliation_wolfram_ricci_q05 = 0, curv_foliation_wolfram_ricci_q95 = 0;
    float curv_foliation_dim_gradient_q05 = 0, curv_foliation_dim_gradient_q95 = 0;

    // =========================================================================
    // Hilbert Space Analysis Results
    // =========================================================================
    bool has_hilbert_analysis = false;
    std::unordered_map<VertexId, float> hilbert_vertex_probabilities;
    size_t hilbert_num_states = 0;
    size_t hilbert_num_vertices = 0;
    float hilbert_mean_inner_product = 0;
    float hilbert_max_inner_product = 0;
    float hilbert_mean_vertex_probability = 0;
    float hilbert_vertex_probability_entropy = 0;

    // =========================================================================
    // Branchial Analysis Results
    // =========================================================================
    bool has_branchial_analysis = false;
    std::unordered_map<VertexId, float> branchial_vertex_sharpness;
    std::unordered_map<VertexId, float> branchial_vertex_entropy;
    float branchial_mean_sharpness = 0;
    float branchial_mean_entropy = 0;

    // Helpers
    const TimestepAggregation* get_timestep(int step) const {
        for (const auto& ts : per_timestep) {
            if (ts.step == static_cast<uint32_t>(step)) return &ts;
        }
        return nullptr;
    }

    // =========================================================================
    // Curvature Accessors (indexed by curvature type for extensibility)
    // Index: 0=Ollivier, 1=WolframScalar, 2=WolframRicci, 3=DimGradient
    // =========================================================================
    const std::unordered_map<VertexId, float>& get_global_curvature_map(int idx) const {
        static const std::unordered_map<VertexId, float> empty;
        switch (idx) {
            case 0: return curvature_ollivier_ricci;
            case 1: return curvature_wolfram_scalar;
            case 2: return curvature_wolfram_ricci;
            case 3: return curvature_dimension_gradient;
            default: return empty;
        }
    }
    void get_global_curvature_range(int idx, float& out_min, float& out_max) const {
        switch (idx) {
            case 0: out_min = curvature_ollivier_min; out_max = curvature_ollivier_max; break;
            case 1: out_min = curvature_wolfram_scalar_min; out_max = curvature_wolfram_scalar_max; break;
            case 2: out_min = curvature_wolfram_ricci_min; out_max = curvature_wolfram_ricci_max; break;
            case 3: out_min = curvature_dim_grad_min; out_max = curvature_dim_grad_max; break;
            default: out_min = -1; out_max = 1; break;
        }
    }
    void get_mean_curvature_quantiles(int idx, float& out_q05, float& out_q95) const {
        switch (idx) {
            case 0: out_q05 = curv_ollivier_mean_q05; out_q95 = curv_ollivier_mean_q95; break;
            case 1: out_q05 = curv_wolfram_scalar_mean_q05; out_q95 = curv_wolfram_scalar_mean_q95; break;
            case 2: out_q05 = curv_wolfram_ricci_mean_q05; out_q95 = curv_wolfram_ricci_mean_q95; break;
            case 3: out_q05 = curv_dim_gradient_mean_q05; out_q95 = curv_dim_gradient_mean_q95; break;
            default: out_q05 = -1; out_q95 = 1; break;
        }
    }
    void get_variance_curvature_quantiles(int idx, float& out_q05, float& out_q95) const {
        switch (idx) {
            case 0: out_q05 = curv_ollivier_var_q05; out_q95 = curv_ollivier_var_q95; break;
            case 1: out_q05 = curv_wolfram_scalar_var_q05; out_q95 = curv_wolfram_scalar_var_q95; break;
            case 2: out_q05 = curv_wolfram_ricci_var_q05; out_q95 = curv_wolfram_ricci_var_q95; break;
            case 3: out_q05 = curv_dim_gradient_var_q05; out_q95 = curv_dim_gradient_var_q95; break;
            default: out_q05 = 0; out_q95 = 1; break;
        }
    }
    void get_foliation_curvature_quantiles(int idx, float& out_q05, float& out_q95) const {
        switch (idx) {
            case 0: out_q05 = curv_foliation_ollivier_q05; out_q95 = curv_foliation_ollivier_q95; break;
            case 1: out_q05 = curv_foliation_wolfram_scalar_q05; out_q95 = curv_foliation_wolfram_scalar_q95; break;
            case 2: out_q05 = curv_foliation_wolfram_ricci_q05; out_q95 = curv_foliation_wolfram_ricci_q95; break;
            case 3: out_q05 = curv_foliation_dim_gradient_q05; out_q95 = curv_foliation_dim_gradient_q95; break;
            default: out_q05 = -1; out_q95 = 1; break;
        }
    }
};

// =============================================================================
// Rendering Vertex (matches existing viz shaders)
// =============================================================================

struct RenderVertex {
    float x, y, z;
    float r, g, b, a;
};

// =============================================================================
// Color Palettes (inspired by Wolfram Language and matplotlib)
// =============================================================================

enum class ColorPalette {
    Temperature,   // Blue -> Cyan -> Green -> Yellow -> Red (default, rainbow-like)
    Viridis,       // Perceptually uniform, colorblind-friendly (matplotlib)
    Plasma,        // Purple -> Pink -> Orange -> Yellow (matplotlib)
    Inferno,       // Black -> Purple -> Red -> Yellow (matplotlib)
    Magma,         // Black -> Purple -> Pink -> White (matplotlib)
    SunsetColors,  // Blue -> Purple -> Red -> Orange -> Yellow (Wolfram)
    Rainbow,       // Full spectrum rainbow
    Grayscale,     // Black -> White
    Turbo,         // Improved rainbow (Google AI)
    COUNT          // Number of palettes
};

inline const char* palette_name(ColorPalette p) {
    switch (p) {
        case ColorPalette::Temperature: return "Temperature";
        case ColorPalette::Viridis: return "Viridis";
        case ColorPalette::Plasma: return "Plasma";
        case ColorPalette::Inferno: return "Inferno";
        case ColorPalette::Magma: return "Magma";
        case ColorPalette::SunsetColors: return "SunsetColors";
        case ColorPalette::Rainbow: return "Rainbow";
        case ColorPalette::Grayscale: return "Grayscale";
        case ColorPalette::Turbo: return "Turbo";
        default: return "Unknown";
    }
}

// Helper: linear interpolation between two colors
inline Vec3 lerp_color(const Vec3& a, const Vec3& b, float t) {
    return Vec3{a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)};
}

// Temperature colormap: blue -> cyan -> green -> yellow -> red
inline Vec3 temperature_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    if (t < 0.25f) {
        float s = t / 0.25f;
        return {0.0f, s, 1.0f};  // blue -> cyan
    } else if (t < 0.5f) {
        float s = (t - 0.25f) / 0.25f;
        return {0.0f, 1.0f, 1.0f - s};  // cyan -> green
    } else if (t < 0.75f) {
        float s = (t - 0.5f) / 0.25f;
        return {s, 1.0f, 0.0f};  // green -> yellow
    } else {
        float s = (t - 0.75f) / 0.25f;
        return {1.0f, 1.0f - s, 0.0f};  // yellow -> red
    }
}

// Viridis: perceptually uniform, colorblind-friendly
inline Vec3 viridis_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    // Approximation of viridis colormap
    const Vec3 c0{0.267f, 0.004f, 0.329f};  // Dark purple
    const Vec3 c1{0.282f, 0.140f, 0.458f};
    const Vec3 c2{0.254f, 0.265f, 0.530f};
    const Vec3 c3{0.207f, 0.372f, 0.553f};
    const Vec3 c4{0.164f, 0.471f, 0.558f};
    const Vec3 c5{0.128f, 0.567f, 0.551f};
    const Vec3 c6{0.135f, 0.659f, 0.518f};
    const Vec3 c7{0.267f, 0.749f, 0.441f};
    const Vec3 c8{0.478f, 0.821f, 0.318f};
    const Vec3 c9{0.741f, 0.873f, 0.150f};
    const Vec3 c10{0.993f, 0.906f, 0.144f};  // Yellow

    const Vec3* colors[] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8, &c9, &c10};
    float pos = t * 10.0f;
    int idx = static_cast<int>(pos);
    if (idx >= 10) return c10;
    float frac = pos - idx;
    return lerp_color(*colors[idx], *colors[idx + 1], frac);
}

// Plasma: purple -> pink -> orange -> yellow
inline Vec3 plasma_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const Vec3 c0{0.050f, 0.030f, 0.528f};
    const Vec3 c1{0.295f, 0.000f, 0.635f};
    const Vec3 c2{0.494f, 0.012f, 0.658f};
    const Vec3 c3{0.665f, 0.138f, 0.618f};
    const Vec3 c4{0.798f, 0.280f, 0.470f};
    const Vec3 c5{0.898f, 0.420f, 0.298f};
    const Vec3 c6{0.957f, 0.592f, 0.117f};
    const Vec3 c7{0.940f, 0.975f, 0.131f};

    const Vec3* colors[] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7};
    float pos = t * 7.0f;
    int idx = static_cast<int>(pos);
    if (idx >= 7) return c7;
    float frac = pos - idx;
    return lerp_color(*colors[idx], *colors[idx + 1], frac);
}

// Inferno: black -> purple -> red -> yellow
inline Vec3 inferno_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const Vec3 c0{0.001f, 0.000f, 0.014f};
    const Vec3 c1{0.135f, 0.039f, 0.329f};
    const Vec3 c2{0.341f, 0.062f, 0.429f};
    const Vec3 c3{0.550f, 0.107f, 0.401f};
    const Vec3 c4{0.735f, 0.216f, 0.330f};
    const Vec3 c5{0.878f, 0.394f, 0.199f};
    const Vec3 c6{0.961f, 0.621f, 0.074f};
    const Vec3 c7{0.988f, 0.998f, 0.645f};

    const Vec3* colors[] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7};
    float pos = t * 7.0f;
    int idx = static_cast<int>(pos);
    if (idx >= 7) return c7;
    float frac = pos - idx;
    return lerp_color(*colors[idx], *colors[idx + 1], frac);
}

// Magma: black -> purple -> pink -> white
inline Vec3 magma_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const Vec3 c0{0.001f, 0.000f, 0.014f};
    const Vec3 c1{0.159f, 0.042f, 0.324f};
    const Vec3 c2{0.373f, 0.074f, 0.432f};
    const Vec3 c3{0.550f, 0.161f, 0.506f};
    const Vec3 c4{0.716f, 0.280f, 0.474f};
    const Vec3 c5{0.868f, 0.468f, 0.468f};
    const Vec3 c6{0.967f, 0.717f, 0.600f};
    const Vec3 c7{0.987f, 0.991f, 0.749f};

    const Vec3* colors[] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7};
    float pos = t * 7.0f;
    int idx = static_cast<int>(pos);
    if (idx >= 7) return c7;
    float frac = pos - idx;
    return lerp_color(*colors[idx], *colors[idx + 1], frac);
}

// SunsetColors (Wolfram): blue -> purple -> red -> orange -> yellow
inline Vec3 sunset_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const Vec3 c0{0.0f, 0.0f, 0.5f};    // Dark blue
    const Vec3 c1{0.3f, 0.0f, 0.6f};    // Purple
    const Vec3 c2{0.6f, 0.0f, 0.4f};    // Magenta
    const Vec3 c3{0.8f, 0.2f, 0.2f};    // Red
    const Vec3 c4{1.0f, 0.5f, 0.0f};    // Orange
    const Vec3 c5{1.0f, 0.9f, 0.2f};    // Yellow

    const Vec3* colors[] = {&c0, &c1, &c2, &c3, &c4, &c5};
    float pos = t * 5.0f;
    int idx = static_cast<int>(pos);
    if (idx >= 5) return c5;
    float frac = pos - idx;
    return lerp_color(*colors[idx], *colors[idx + 1], frac);
}

// Rainbow: full spectrum
inline Vec3 rainbow_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    // HSV to RGB with H = t * 300 degrees (purple to red, skipping magenta)
    float h = t * 300.0f;  // 0 = red, 60 = yellow, 120 = green, 180 = cyan, 240 = blue, 300 = magenta
    h = 270.0f - h;  // Invert so low = blue, high = red
    if (h < 0) h += 360.0f;

    float c = 1.0f;
    float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = 0.0f;

    Vec3 rgb;
    if (h < 60) rgb = {c, x, 0};
    else if (h < 120) rgb = {x, c, 0};
    else if (h < 180) rgb = {0, c, x};
    else if (h < 240) rgb = {0, x, c};
    else if (h < 300) rgb = {x, 0, c};
    else rgb = {c, 0, x};

    return Vec3{rgb.x + m, rgb.y + m, rgb.z + m};
}

// Grayscale
inline Vec3 grayscale_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return Vec3{t, t, t};
}

// Turbo: improved rainbow (Google AI visualization)
inline Vec3 turbo_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const Vec3 c0{0.190f, 0.072f, 0.232f};
    const Vec3 c1{0.256f, 0.291f, 0.798f};
    const Vec3 c2{0.135f, 0.571f, 0.860f};
    const Vec3 c3{0.095f, 0.766f, 0.640f};
    const Vec3 c4{0.282f, 0.899f, 0.352f};
    const Vec3 c5{0.620f, 0.962f, 0.194f};
    const Vec3 c6{0.913f, 0.853f, 0.139f};
    const Vec3 c7{0.993f, 0.603f, 0.033f};
    const Vec3 c8{0.947f, 0.341f, 0.067f};
    const Vec3 c9{0.750f, 0.100f, 0.050f};

    const Vec3* colors[] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8, &c9};
    float pos = t * 9.0f;
    int idx = static_cast<int>(pos);
    if (idx >= 9) return c9;
    float frac = pos - idx;
    return lerp_color(*colors[idx], *colors[idx + 1], frac);
}

// Apply colormap based on palette selection
inline Vec3 apply_palette(float t, ColorPalette palette) {
    switch (palette) {
        case ColorPalette::Temperature: return temperature_color(t);
        case ColorPalette::Viridis: return viridis_color(t);
        case ColorPalette::Plasma: return plasma_color(t);
        case ColorPalette::Inferno: return inferno_color(t);
        case ColorPalette::Magma: return magma_color(t);
        case ColorPalette::SunsetColors: return sunset_color(t);
        case ColorPalette::Rainbow: return rainbow_color(t);
        case ColorPalette::Grayscale: return grayscale_color(t);
        case ColorPalette::Turbo: return turbo_color(t);
        default: return temperature_color(t);
    }
}

// =============================================================================
// Diverging Colormap (for signed values: curvature, etc.)
// =============================================================================

// Apply palette in diverging mode: maps t ∈ [-1, 1] to the palette
// -1 -> low end of palette, 0 -> middle, +1 -> high end
// This allows any sequential palette to work as a diverging colormap
inline Vec3 apply_palette_diverging(float t, ColorPalette palette) {
    // Map from [-1, 1] to [0, 1]
    float mapped = std::clamp((t + 1.0f) * 0.5f, 0.0f, 1.0f);
    return apply_palette(mapped, palette);
}

// Coolwarm diverging colormap: Blue (negative) -> White (zero) -> Red (positive)
// Input t is in [-1, 1] range (already normalized)
// This is a dedicated diverging palette with perceptually balanced endpoints
inline Vec3 diverging_coolwarm_color(float t) {
    // Clamp and transform t from [-1, 1] to [0, 1]
    t = std::clamp((t + 1.0f) * 0.5f, 0.0f, 1.0f);

    const Vec3 blue{0.230f, 0.299f, 0.754f};   // Negative (cool)
    const Vec3 white{0.865f, 0.865f, 0.865f};  // Zero (neutral)
    const Vec3 red{0.706f, 0.016f, 0.150f};    // Positive (warm)

    if (t < 0.5f) {
        return lerp_color(blue, white, t * 2.0f);
    } else {
        return lerp_color(white, red, (t - 0.5f) * 2.0f);
    }
}

// Normalize curvature to [-1, 1] using symmetric absolute max
// Ensures zero maps to center of colormap
inline float normalize_curvature_symmetric(float curvature, float abs_max) {
    if (abs_max < 1e-6f) return 0.0f;
    return std::clamp(curvature / abs_max, -1.0f, 1.0f);
}

// Convert curvature to color using diverging colormap with palette selection
// curvature: the curvature value (can be negative, zero, or positive)
// abs_max: the maximum absolute curvature for normalization
// palette: the color palette to use (applied in diverging mode)
inline Vec4 curvature_to_color(float curvature, float abs_max, ColorPalette palette) {
    if (!std::isfinite(curvature)) {
        return Vec4(0.3f, 0.3f, 0.3f, 1.0f);  // Gray for invalid
    }
    float normalized = normalize_curvature_symmetric(curvature, abs_max);
    Vec3 rgb = apply_palette_diverging(normalized, palette);
    return Vec4(rgb, 1.0f);
}

// Legacy overload using coolwarm (for backward compatibility)
inline Vec4 curvature_to_color(float curvature, float abs_max) {
    if (!std::isfinite(curvature)) {
        return Vec4(0.3f, 0.3f, 0.3f, 1.0f);  // Gray for invalid
    }
    float normalized = normalize_curvature_symmetric(curvature, abs_max);
    Vec3 rgb = diverging_coolwarm_color(normalized);
    return Vec4(rgb, 1.0f);
}

// =============================================================================
// Missing Data Display Mode
// =============================================================================

enum class MissingDataMode {
    Show,       // Show missing vertices as gray (default)
    Hide,       // Hide missing vertices entirely
    Highlight   // Highlight missing vertices in a distinct color (magenta/pink)
};

inline const char* missing_mode_name(MissingDataMode m) {
    switch (m) {
        case MissingDataMode::Show: return "Show (gray)";
        case MissingDataMode::Hide: return "Hide";
        case MissingDataMode::Highlight: return "Highlight";
        default: return "Unknown";
    }
}

// Color for missing/invalid dimension data
inline Vec4 missing_data_color(MissingDataMode mode) {
    switch (mode) {
        case MissingDataMode::Highlight:
            return Vec4(1.0f, 0.0f, 1.0f, 1.0f);  // Bright magenta
        case MissingDataMode::Show:
        default:
            return Vec4(0.3f, 0.3f, 0.3f, 1.0f);  // Gray
    }
}

// Main dimension-to-color function with palette and missing data support
inline Vec4 dimension_to_color(float dim, float dim_min, float dim_max,
                                ColorPalette palette = ColorPalette::Temperature,
                                MissingDataMode missing_mode = MissingDataMode::Show) {
    // Handle invalid dimension
    if (dim < 0 || !std::isfinite(dim)) {
        return missing_data_color(missing_mode);
    }

    // Handle invalid min/max
    if (!std::isfinite(dim_min) || !std::isfinite(dim_max)) {
        return missing_data_color(missing_mode);
    }

    // Ensure min <= max
    if (dim_min > dim_max) {
        std::swap(dim_min, dim_max);
    }

    // Avoid division by zero
    float range = dim_max - dim_min;
    if (range < 0.001f) {
        Vec3 c = apply_palette(0.5f, palette);
        return Vec4(c, 1.0f);
    }

    // Compute normalized value and apply palette
    float t = std::clamp((dim - dim_min) / range, 0.0f, 1.0f);
    Vec3 c = apply_palette(t, palette);
    return Vec4(c, 1.0f);
}

// Legacy overload for backward compatibility
inline Vec4 dimension_to_color(float dim, float dim_min, float dim_max) {
    return dimension_to_color(dim, dim_min, dim_max, ColorPalette::Temperature, MissingDataMode::Show);
}

// =============================================================================
// Quantum Analysis Overlay Configuration
// =============================================================================
// Configuration for overlaying geodesic paths and topological defects on the
// dimension heatmap. Based on Gorard's "Some Quantum Mechanical Properties of
// the Wolfram Model" paper.

struct GeodesicOverlayConfig {
    bool show_geodesics = false;           // Overlay geodesic paths
    bool show_sources = true;              // Highlight geodesic source vertices
    bool show_proper_time = true;          // Color geodesics by proper time

    // Visual style
    Vec4 geodesic_color = {1.0f, 1.0f, 0.0f, 0.8f};      // Yellow (default)
    Vec4 source_color = {0.0f, 1.0f, 0.0f, 1.0f};        // Green for sources
    float geodesic_width = 2.0f;                          // Line width in pixels
    float source_radius_scale = 1.5f;                     // Multiplier for source vertex size

    // Animation (for step-by-step geodesic rendering)
    bool animate_geodesics = false;
    float animation_speed = 1.0f;          // Steps per second
};

struct ParticleOverlayConfig {
    bool show_defects = false;             // Overlay detected topological defects
    bool show_charge_map = false;          // Color vertices by topological charge
    bool show_defect_regions = true;       // Draw circles around defect cores

    // Visual style
    Vec4 k5_color = {1.0f, 0.0f, 0.0f, 0.9f};            // Red for K5 minors
    Vec4 k33_color = {1.0f, 0.5f, 0.0f, 0.9f};           // Orange for K3,3 minors
    Vec4 high_degree_color = {1.0f, 1.0f, 0.0f, 0.9f};   // Yellow for high-degree
    Vec4 dimension_spike_color = {1.0f, 0.0f, 1.0f, 0.9f}; // Magenta for dim spikes
    float defect_marker_scale = 2.0f;                     // Multiplier for defect vertex size
    float region_alpha = 0.3f;                            // Transparency for defect regions

    // Charge visualization (when show_charge_map = true)
    ColorPalette charge_palette = ColorPalette::Plasma;   // Palette for charge coloring
    float charge_min = 0.0f;                              // Min charge for color scale
    float charge_max = 1.0f;                              // Max charge for color scale
};

struct QuantumOverlayConfig {
    GeodesicOverlayConfig geodesics;
    ParticleOverlayConfig particles;

    // Combined overlay settings
    bool overlay_mode = false;              // Enable overlay rendering at all
    float overlay_blend = 0.7f;             // Blend factor: 0 = base only, 1 = overlay only
};

// =============================================================================
// Topological Charge Color Helper
// =============================================================================

// Map topological charge to color using specified palette
inline Vec4 charge_to_color(float charge, float charge_min, float charge_max,
                            ColorPalette palette = ColorPalette::Plasma) {
    if (!std::isfinite(charge)) {
        return Vec4(0.3f, 0.3f, 0.3f, 1.0f);  // Gray for invalid
    }

    float range = charge_max - charge_min;
    if (range < 0.001f) {
        Vec3 c = apply_palette(0.5f, palette);
        return Vec4(c, 1.0f);
    }

    float t = std::clamp((charge - charge_min) / range, 0.0f, 1.0f);
    Vec3 c = apply_palette(t, palette);
    return Vec4(c, 1.0f);
}

} // namespace viz::blackhole
