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

    // Helpers
    const TimestepAggregation* get_timestep(int step) const {
        for (const auto& ts : per_timestep) {
            if (ts.step == static_cast<uint32_t>(step)) return &ts;
        }
        return nullptr;
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

} // namespace viz::blackhole
