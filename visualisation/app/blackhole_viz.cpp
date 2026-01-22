// Black Hole Hausdorff Dimension Visualisation
// Visualizes multiway hypergraph evolution from binary black hole initial conditions
// with Hausdorff dimension coloring

#include <platform/window.hpp>
#include <gal/gal.hpp>
#include <gal/vulkan/vk_loader.hpp>
#include <camera/camera.hpp>
#include <math/types.hpp>

#include <blackhole/bh_types.hpp>
#include <blackhole/bh_serialization.hpp>
#include <blackhole/bh_initial_condition.hpp>
#include <blackhole/bh_evolution.hpp>
#include <blackhole/hausdorff_analysis.hpp>
#include <blackhole/curvature_analysis.hpp>
#include <blackhole/entropy_analysis.hpp>
#include <blackhole/branchial_analysis.hpp>
#include <blackhole/branch_alignment.hpp>
#include <layout/layout_engine.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <csignal>
#include <atomic>
#include <functional>
#include <thread>
#include <queue>

#ifdef _WIN32
#include <windows.h>
#endif

// Global flag for Ctrl-C handling
static std::atomic<bool> g_shutdown_requested{false};

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        if (g_shutdown_requested.load()) {
            // Second Ctrl+C - force exit
            std::cerr << "\nForce exit (second Ctrl+C)\n";
            std::_Exit(1);
        }
        g_shutdown_requested.store(true);
        std::cerr << "\nCtrl+C received, press again to force exit\n";
    }
}

// Mathematical constant
static constexpr float PI = 3.14159265358979323846f;

#ifdef _WIN32
// Windows console control handler
static BOOL WINAPI console_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        if (g_shutdown_requested.load()) {
            std::cerr << "\nForce exit (second Ctrl+C)\n";
            std::_Exit(1);
        }
        g_shutdown_requested.store(true);
        std::cerr << "\nCtrl+C received, press again to force exit\n";
        return TRUE;  // Signal handled
    }
    return FALSE;
}
#endif

using namespace viz;
using namespace viz::blackhole;
using namespace viz::layout;

// Edge display mode (forward declaration for init_layout_from_timestep)
enum class EdgeDisplayMode {
    Union,        // Show all edges present in ANY state (default)
    Frequent,     // Show edges present in MORE than one state (excludes singletons)
    Intersection, // Show only edges present in ALL states
    SingleState   // Show one specific state (for path exploration)
};

// Persistent position cache - maps VertexId to (x, y) position
// This persists across timesteps to prevent layout snapping when looping
using PositionCache = std::unordered_map<VertexId, std::pair<float, float>>;

// Initialize layout graph from a timestep's graph
// If prev_graph and prev_vertices are provided, seed positions from them for existing vertices
// edge_mode controls whether to use union or intersection edges
// global_cache provides persistent positions across all timesteps (prevents snap on loop)
// out_vertices receives the list of vertex IDs in layout order (for cache updates)
void init_layout_from_timestep(
    LayoutGraph& graph,
    const TimestepAggregation& ts,
    const BHInitialCondition& initial,
    EdgeDisplayMode edge_mode,
    PositionCache& global_cache,
    std::vector<VertexId>& out_vertices,
    int edge_freq_threshold = 2,  // Threshold for frequent edge filtering
    const std::vector<StateData>* states = nullptr,  // Per-state data for edge counting
    const LayoutGraph* prev_graph = nullptr,
    const std::vector<VertexId>* prev_vertices = nullptr
) {
    // Build map from previous vertex IDs to their positions (including Z for 3D layout)
    std::unordered_map<VertexId, std::pair<float, float>> prev_positions;
    std::unordered_map<VertexId, float> prev_positions_z;  // Separate Z map for 3D
    if (prev_graph && prev_vertices && prev_graph->vertex_count() == prev_vertices->size()) {
        for (size_t i = 0; i < prev_vertices->size(); ++i) {
            VertexId vid = (*prev_vertices)[i];
            prev_positions[vid] = {
                prev_graph->positions_x[i],
                prev_graph->positions_y[i]
            };
            // Also store Z if available (for 3D layout continuity)
            if (i < prev_graph->positions_z.size()) {
                prev_positions_z[vid] = prev_graph->positions_z[i];
            }
        }
    }

    graph.clear();

    // Helper to compute vertex pair key (for rendering lookup)
    auto vertex_pair_key = [](const Edge& e) -> uint64_t {
        VertexId v1 = std::min(e.v1, e.v2);
        VertexId v2 = std::max(e.v1, e.v2);
        return (static_cast<uint64_t>(v1) << 32) | v2;
    };

    // Select edge list based on display mode
    // Fall back to union if the selected mode has no edges
    // Use SAME filtering logic as build_render_data for consistency
    const std::vector<Edge>* edges_to_use = &ts.union_edges;
    std::vector<Edge> freq_filtered_edges;  // Storage for dynamically filtered edges
    EdgeDisplayMode effective_mode = edge_mode;

    if (edge_mode == EdgeDisplayMode::Intersection) {
        if (!ts.intersection_edges.empty()) {
            edges_to_use = &ts.intersection_edges;
        } else {
            effective_mode = EdgeDisplayMode::Union;
        }
    } else if (edge_mode == EdgeDisplayMode::Frequent) {
        // Use threshold-based filtering (same as build_render_data)
        if (states && !states->empty()) {
            // Compute edge counts by vertex pair (count each pair once per state)
            std::unordered_map<uint64_t, int> edge_counts_by_pair;
            for (const auto& state : *states) {
                std::unordered_set<uint64_t> seen_pairs_this_state;
                for (const auto& e : state.edges) {
                    uint64_t pair_key = vertex_pair_key(e);
                    if (seen_pairs_this_state.insert(pair_key).second) {
                        edge_counts_by_pair[pair_key]++;
                    }
                }
            }
            // Filter edges that appear in >= threshold states
            for (const auto& e : ts.union_edges) {
                auto it = edge_counts_by_pair.find(vertex_pair_key(e));
                if (it != edge_counts_by_pair.end() && it->second >= edge_freq_threshold) {
                    freq_filtered_edges.push_back(e);
                }
            }
            if (!freq_filtered_edges.empty()) {
                edges_to_use = &freq_filtered_edges;
            } else {
                effective_mode = EdgeDisplayMode::Union;  // Fallback if no edges match
            }
        } else if (!ts.frequent_edges.empty()) {
            // Fall back to pre-computed frequent edges if no per-state data
            edges_to_use = &ts.frequent_edges;
        } else {
            effective_mode = EdgeDisplayMode::Union;
        }
    }

    // In non-union modes, only include vertices that have edges
    std::unordered_set<VertexId> active_vertices;
    if (effective_mode != EdgeDisplayMode::Union) {
        for (const auto& e : *edges_to_use) {
            active_vertices.insert(e.v1);
            active_vertices.insert(e.v2);
        }
    } else {
        for (VertexId v : ts.union_vertices) {
            active_vertices.insert(v);
        }
    }

    // Build list of vertices to include (preserving order from union_vertices)
    std::vector<VertexId> vertices_to_use;
    for (VertexId v : ts.union_vertices) {
        if (active_vertices.count(v)) {
            vertices_to_use.push_back(v);
        }
    }

    // Build vertex ID to index map (for the filtered vertices)
    std::unordered_map<VertexId, uint32_t> vid_to_idx;
    for (size_t i = 0; i < vertices_to_use.size(); ++i) {
        vid_to_idx[vertices_to_use[i]] = static_cast<uint32_t>(i);
    }

    // Build adjacency for neighbor-based placement
    std::vector<std::vector<uint32_t>> adjacency(vertices_to_use.size());
    for (const auto& e : *edges_to_use) {
        auto it1 = vid_to_idx.find(e.v1);
        auto it2 = vid_to_idx.find(e.v2);
        if (it1 != vid_to_idx.end() && it2 != vid_to_idx.end()) {
            adjacency[it1->second].push_back(it2->second);
            adjacency[it2->second].push_back(it1->second);
        }
    }

    // Track which vertices have known good positions
    std::vector<bool> has_good_position(vertices_to_use.size(), false);
    std::vector<float> pos_x(vertices_to_use.size(), 0.0f);
    std::vector<float> pos_y(vertices_to_use.size(), 0.0f);
    std::vector<float> pos_z(vertices_to_use.size(), 0.0f);  // Z for 3D embedding

    // Are we seeding from a live layout? (prev_positions is non-empty)
    // If so, we MUST NOT fall back to initial/stored positions - they're at a different scale!
    bool have_live_layout = !prev_positions.empty();

    // First pass: collect positions from caches and initial conditions
    // Priority: 1) prev_positions (current frame), 2) global_cache (persistent), 3) initial/stored
    for (size_t i = 0; i < vertices_to_use.size(); ++i) {
        VertexId vid = vertices_to_use[i];
        bool has_z_from_layout = false;

        // Check previous layout first (live positions have highest priority)
        auto prev_it = prev_positions.find(vid);
        if (prev_it != prev_positions.end()) {
            pos_x[i] = prev_it->second.first;
            pos_y[i] = prev_it->second.second;
            has_good_position[i] = true;
            // Also get Z from previous layout if available (for 3D layout continuity)
            auto prev_z_it = prev_positions_z.find(vid);
            if (prev_z_it != prev_positions_z.end()) {
                pos_z[i] = prev_z_it->second;
                has_z_from_layout = true;
            }
        }

        // Check global cache (persistent positions from previous timesteps)
        if (!has_good_position[i]) {
            auto cache_it = global_cache.find(vid);
            if (cache_it != global_cache.end()) {
                pos_x[i] = cache_it->second.first;
                pos_y[i] = cache_it->second.second;
                has_good_position[i] = true;
                // Global cache is 2D only, Z will come from initial below
            }
        }

        // ONLY use initial/stored positions if we DON'T have a live layout running
        // If we have a live layout, new vertices must be placed via neighbor-based placement
        if (!has_good_position[i] && !have_live_layout) {
            // Check stored LAYOUT positions first (pre-computed, scaled to match geodesic)
            // Find the index in union_vertices
            for (size_t j = 0; j < ts.union_vertices.size(); ++j) {
                if (ts.union_vertices[j] == vid) {
                    // Prefer layout_positions if available (pre-computed force-directed layout)
                    if (j < ts.layout_positions.size()) {
                        float lx = ts.layout_positions[j].x;
                        float ly = ts.layout_positions[j].y;
                        pos_x[i] = lx;
                        pos_y[i] = ly;
                        has_good_position[i] = true;
                        break;
                    }
                    // Fall back to geodesic vertex_positions if no layout positions
                    if (j < ts.vertex_positions.size()) {
                        float sx = ts.vertex_positions[j].x;
                        float sy = ts.vertex_positions[j].y;
                        if (std::abs(sx) > 0.5f || std::abs(sy) > 0.5f) {
                            pos_x[i] = sx;
                            pos_y[i] = sy;
                            has_good_position[i] = true;
                        }
                        break;
                    }
                }
            }

            // If still no position, check initial condition (for original vertices)
            if (!has_good_position[i] && vid < initial.vertex_positions.size()) {
                pos_x[i] = initial.vertex_positions[vid].x;
                pos_y[i] = initial.vertex_positions[vid].y;
                has_good_position[i] = true;
            }
        }

        // Set Z from initial 3D embedding ONLY if we didn't get it from live layout
        // This preserves 3D layout motion while still seeding new vertices with embedding Z
        if (!has_z_from_layout && initial.has_3d() && vid < initial.vertex_z.size()) {
            pos_z[i] = initial.vertex_z[vid];
        }
        // If have_live_layout and vertex not in prev_positions or global_cache, it's NEW
        // It will be placed via neighbor-based placement in the next pass
    }

    // Second pass: place vertices without positions at centroid of their positioned neighbors
    // May need multiple iterations for chains of new vertices
    for (int iteration = 0; iteration < 3; ++iteration) {
        bool any_placed = false;
        for (size_t i = 0; i < vertices_to_use.size(); ++i) {
            if (has_good_position[i]) continue;

            float sum_x = 0, sum_y = 0, sum_z = 0;
            int count = 0;

            for (uint32_t neighbor_idx : adjacency[i]) {
                if (has_good_position[neighbor_idx]) {
                    sum_x += pos_x[neighbor_idx];
                    sum_y += pos_y[neighbor_idx];
                    sum_z += pos_z[neighbor_idx];
                    count++;
                }
            }

            if (count > 0) {
                pos_x[i] = sum_x / count;
                pos_y[i] = sum_y / count;
                pos_z[i] = sum_z / count;
                has_good_position[i] = true;
                any_placed = true;
            }
        }
        if (!any_placed) break;
    }

    // Third pass: any remaining vertices get placed at graph centroid
    float centroid_x = 0, centroid_y = 0, centroid_z = 0;
    int positioned_count = 0;
    for (size_t i = 0; i < vertices_to_use.size(); ++i) {
        if (has_good_position[i]) {
            centroid_x += pos_x[i];
            centroid_y += pos_y[i];
            centroid_z += pos_z[i];
            positioned_count++;
        }
    }
    if (positioned_count > 0) {
        centroid_x /= positioned_count;
        centroid_y /= positioned_count;
        centroid_z /= positioned_count;
    }

    for (size_t i = 0; i < vertices_to_use.size(); ++i) {
        if (!has_good_position[i]) {
            // Last resort: place at graph centroid with unique offset based on vertex ID
            // Use vertex ID (not index) to ensure uniqueness even with many vertices
            VertexId vid = vertices_to_use[i];
            // Golden ratio offset for better distribution
            float phi = 1.618033988749895f;
            float angle = vid * phi * 2.0f * PI;
            float radius = 0.1f + 0.01f * std::sqrt(static_cast<float>(vid % 1000));
            pos_x[i] = centroid_x + radius * std::cos(angle);
            pos_y[i] = centroid_y + radius * std::sin(angle);
            pos_z[i] = centroid_z;  // Keep Z at centroid level
        }
    }

    // Add vertices to layout graph (with Z coordinate for 3D embedding)
    for (size_t i = 0; i < vertices_to_use.size(); ++i) {
        graph.add_vertex(pos_x[i], pos_y[i], pos_z[i], 1.0f, false);
    }

    // Add edges with fixed target rest_length
    // All edges try to reach the same length - springiness handles the rest
    constexpr float target_edge_length = 1.0f;
    for (const auto& e : *edges_to_use) {
        auto it1 = vid_to_idx.find(e.v1);
        auto it2 = vid_to_idx.find(e.v2);
        if (it1 != vid_to_idx.end() && it2 != vid_to_idx.end()) {
            graph.add_edge(it1->second, it2->second, target_edge_length, 1.0f);
        }
    }

    // Output the vertex list for cache updates
    out_vertices = std::move(vertices_to_use);
}

// Helper to update the global position cache from current layout
void update_position_cache(
    PositionCache& cache,
    const LayoutGraph& graph,
    const std::vector<VertexId>& vertices
) {
    if (graph.vertex_count() != vertices.size()) return;
    for (size_t i = 0; i < vertices.size(); ++i) {
        cache[vertices[i]] = {graph.positions_x[i], graph.positions_y[i]};
    }
}

// Vulkan surface creation helpers
namespace viz::gal {
    VkSurfaceKHR create_xcb_surface(VkInstance instance, void* connection, void* window);
    VkSurfaceKHR create_win32_surface(VkInstance instance, void* hinstance, void* hwnd);
    VkInstance get_vk_instance(Device* device);
}

// Load SPIR-V shader
std::vector<uint32_t> load_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader: " << path << std::endl;
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);
    return spirv;
}

// Vertex format matching the basic3d shaders
struct Vertex {
    float x, y, z;
    float r, g, b, a;
};

// Mesh vertex with position and normal (for lit meshes)
struct MeshVertex {
    float px, py, pz;  // Position
    float nx, ny, nz;  // Normal
};

// Index type
using Index = uint16_t;

// Static mesh data
struct Mesh {
    std::vector<MeshVertex> vertices;
    std::vector<Index> indices;
};

// Instance data for edges (cylinders) - matches instance_cylinder.vert
struct CylinderInstance {
    float start_x, start_y, start_z;  // location 2: vec3
    float end_x, end_y, end_z;        // location 3: vec3
    float radius;                      // location 4: float
    float _pad1;
    float r1, g1, b1, a1;             // location 5: vec4 (start color)
    float r2, g2, b2, a2;             // location 6: vec4 (end color)
};

// Instance data for vertices (spheres) - matches instance_sphere.vert
struct SphereInstance {
    float x, y, z;                    // location 2: vec3
    float radius;                     // location 3: float
    float r, g, b, a;                 // location 4: vec4
};

// Generate a unit cylinder mesh (height 1 along Y, radius 1)
// Will be scaled/rotated per-instance
Mesh generate_cylinder_mesh(int segments = 16) {
    Mesh mesh;

    // Generate cap vertices and side vertices
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * PI * i / segments;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        // Bottom cap vertex
        mesh.vertices.push_back({cos_a, 0.0f, sin_a, 0.0f, -1.0f, 0.0f});
        // Top cap vertex
        mesh.vertices.push_back({cos_a, 1.0f, sin_a, 0.0f, 1.0f, 0.0f});
        // Side bottom vertex
        mesh.vertices.push_back({cos_a, 0.0f, sin_a, cos_a, 0.0f, sin_a});
        // Side top vertex
        mesh.vertices.push_back({cos_a, 1.0f, sin_a, cos_a, 0.0f, sin_a});
    }

    // Center vertices for caps
    Index bottom_center = static_cast<Index>(mesh.vertices.size());
    mesh.vertices.push_back({0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f});
    Index top_center = static_cast<Index>(mesh.vertices.size());
    mesh.vertices.push_back({0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f});

    // Generate indices
    for (int i = 0; i < segments; ++i) {
        int base = i * 4;
        int next = ((i + 1) % (segments + 1)) * 4;

        // Bottom cap triangle
        mesh.indices.push_back(bottom_center);
        mesh.indices.push_back(static_cast<Index>(next));
        mesh.indices.push_back(static_cast<Index>(base));

        // Top cap triangle
        mesh.indices.push_back(top_center);
        mesh.indices.push_back(static_cast<Index>(base + 1));
        mesh.indices.push_back(static_cast<Index>(next + 1));

        // Side quad (two triangles)
        mesh.indices.push_back(static_cast<Index>(base + 2));
        mesh.indices.push_back(static_cast<Index>(next + 2));
        mesh.indices.push_back(static_cast<Index>(base + 3));

        mesh.indices.push_back(static_cast<Index>(next + 2));
        mesh.indices.push_back(static_cast<Index>(next + 3));
        mesh.indices.push_back(static_cast<Index>(base + 3));
    }

    return mesh;
}

// Generate a UV sphere mesh (radius 1)
Mesh generate_sphere_mesh(int longitude_segments = 16, int latitude_segments = 8) {
    Mesh mesh;

    // Generate vertices
    for (int lat = 0; lat <= latitude_segments; ++lat) {
        float theta = PI * lat / latitude_segments;
        float sin_theta = std::sin(theta);
        float cos_theta = std::cos(theta);

        for (int lon = 0; lon <= longitude_segments; ++lon) {
            float phi = 2.0f * PI * lon / longitude_segments;
            float sin_phi = std::sin(phi);
            float cos_phi = std::cos(phi);

            float x = sin_theta * cos_phi;
            float y = cos_theta;
            float z = sin_theta * sin_phi;

            // Position and normal are the same for unit sphere
            mesh.vertices.push_back({x, y, z, x, y, z});
        }
    }

    // Generate indices
    for (int lat = 0; lat < latitude_segments; ++lat) {
        for (int lon = 0; lon < longitude_segments; ++lon) {
            int current = lat * (longitude_segments + 1) + lon;
            int next = current + longitude_segments + 1;

            mesh.indices.push_back(static_cast<Index>(current));
            mesh.indices.push_back(static_cast<Index>(next));
            mesh.indices.push_back(static_cast<Index>(current + 1));

            mesh.indices.push_back(static_cast<Index>(current + 1));
            mesh.indices.push_back(static_cast<Index>(next));
            mesh.indices.push_back(static_cast<Index>(next + 1));
        }
    }

    return mesh;
}

// Generate a circle mesh (for 2D mode) - just a flat disc
Mesh generate_circle_mesh(int segments = 16) {
    Mesh mesh;

    // Center vertex
    mesh.vertices.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});

    // Edge vertices
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * PI * i / segments;
        float x = std::cos(angle);
        float y = std::sin(angle);
        mesh.vertices.push_back({x, y, 0.0f, 0.0f, 0.0f, 1.0f});
    }

    // Triangles from center to edge
    for (int i = 0; i < segments; ++i) {
        mesh.indices.push_back(0);
        mesh.indices.push_back(static_cast<Index>(i + 1));
        mesh.indices.push_back(static_cast<Index>(i + 2));
    }

    return mesh;
}

// View mode
enum class ViewMode {
    View2D,       // XY plane with dimension coloring
    View3D,       // 3D with dimension as Z/height
    ShapeSpace2D, // Curvature shape space: PC1 vs PC2
    ShapeSpace3D  // Curvature shape space: PC1/PC2/PC3
};

// =============================================================================
// Value Display System - Four Orthogonal Controls
// =============================================================================
//
// WHAT: ValueType - which quantity to display
//   - WolframHausdorffDimension: Local dimension from ball volume scaling
//   - WolframRicciScalar: Scalar curvature from ball volume deficit
//   - WolframRicciTensor: Directional curvature from geodesic tube volume
//   - OllivierRicciCurvature: Transport-based curvature (Wasserstein distance)
//   - DimensionGradient: Laplacian of the dimension field
//
// HOW: StatisticMode - how to aggregate across multiway states
//   - Mean: Average value
//   - Variance: Variance of values (only meaningful for Branchial scope)
//
// WHERE (spatial): ValueAggregation - spatial aggregation method
//   - PerVertex: Raw values per vertex ID
//   - Bucketed: Values bucketed by geodesic coordinates (smooths noise)
//
// WHERE (graph): GraphScope - which graph topology values come from
//   - Branchial: Per-state computation, averaged across states (dimension only)
//   - Foliation: Per-timestep union graph computation (dimension only)
//   - Global: Mega-union graph computation (curvature always uses this)
//
// APPLICABILITY MATRIX:
// +---------------------------+----------+----------+--------+
// | ValueType                 | Branchial| Foliation| Global |
// +---------------------------+----------+----------+--------+
// | WolframHausdorffDimension | ✓ Mean   | ✓        | ✓      |
// |                           | ✓ Var    |          |        |
// +---------------------------+----------+----------+--------+
// | WolframRicciScalar        |          |          | ✓      |
// | WolframRicciTensor        |          |          | ✓      |
// | OllivierRicciCurvature    |          |          | ✓      |
// | DimensionGradient         |          |          | ✓      |
// +---------------------------+----------+----------+--------+
// Note: Curvature types currently only support Global scope (computed on mega-union).
//       Variance is only meaningful for Branchial scope with dimension.
//
// KEYBINDINGS:
//   V - Cycle ValueType
//   M - Toggle StatisticMode (Mean/Variance)
//   B - Toggle ValueAggregation (PerVertex/Bucketed)
//   G - Cycle GraphScope (Branchial/Foliation/Global)
//   C - Cycle ColorPalette
//   N - Toggle normalization (per-frame/global)
// =============================================================================

// ValueType: WHAT quantity to display
enum class ValueType {
    WolframHausdorffDimension,  // d = Δlog(V)/Δlog(r) - local dimension via ball volume scaling
    WolframRicciScalar,         // K = 6(d+2)/r² × (1 - V/V_expected) - scalar curvature via ball volume
    WolframRicciTensor,         // R_μν = Scalar - TubeCorrection - directional curvature via tube volume
    OllivierRicciCurvature,     // κ = 1 - W₁/d - transport-based curvature
    DimensionGradient,          // Laplacian of dimension field
    COUNT
};

// StatisticMode: HOW to aggregate across multiway states
enum class StatisticMode {
    Mean,     // Average value across states
    Variance  // Variance of values across states (only meaningful for Branchial scope)
};

// ValueAggregation: WHERE values come from spatially
enum class ValueAggregation {
    PerVertex,  // Raw per-vertex values
    Bucketed    // Geodesic-coordinate-bucketed values (default)
};

// Helper to check if value type is curvature (uses diverging colormap)
inline bool is_curvature_type(ValueType vt) {
    return vt != ValueType::WolframHausdorffDimension;
}

// Curvature type index for array-based lookups (avoids switch statement explosion)
// Returns -1 for non-curvature types
inline int curvature_type_index(ValueType vt) {
    switch (vt) {
        case ValueType::OllivierRicciCurvature: return 0;
        case ValueType::WolframRicciScalar: return 1;
        case ValueType::WolframRicciTensor: return 2;
        case ValueType::DimensionGradient: return 3;
        default: return -1;
    }
}
constexpr int NUM_CURVATURE_TYPES = 4;

// Value type display name
inline const char* value_type_name(ValueType vt) {
    switch (vt) {
        case ValueType::WolframHausdorffDimension: return "Dimension (Ball)";
        case ValueType::WolframRicciScalar: return "Curvature (Ball)";
        case ValueType::WolframRicciTensor: return "Curvature (Tube->Scalar)";  // Tensor method, scalar output
        case ValueType::OllivierRicciCurvature: return "Curvature (Transport)";
        case ValueType::DimensionGradient: return "Dimension Gradient";
        case ValueType::COUNT: return "Unknown";
    }
    return "Unknown";
}

// Shape space color mode (what colors curvature points)
enum class ShapeSpaceColorMode {
    Curvature,  // Diverging: blue (-) -> white (0) -> red (+)
    BranchId    // Categorical colors per branch
};

// Shape space display mode
enum class ShapeSpaceDisplayMode {
    Merged,     // All branches combined, same opacity
    PerBranch   // Highlight one branch, dim others
};

// GraphScope: Which graph topology values are computed from
// This affects dimension-based value types. Curvature currently uses Global only.
enum class GraphScope {
    Branchial,  // Values computed per-state, then averaged across multiway states at each timestep
    Foliation,  // Values computed on union graph at each timestep (single graph per timestep)
    Global      // Values computed on mega-union across ALL timesteps (constant per vertex)
};

// EdgeDisplayMode is defined earlier in the file (before init_layout_from_timestep)

// =============================================================================
// Mode Configuration - Centralized compatibility and UI strings
// =============================================================================

// UI string helpers - single source of truth
namespace ui {
    inline const char* name(GraphScope s) {
        switch (s) {
            case GraphScope::Branchial: return "Branchial Aggregation";
            case GraphScope::Foliation: return "Branchial Union";
            case GraphScope::Global: return "All-State Union";
            default: return "Unknown";
        }
    }

    inline const char* name(StatisticMode m) {
        switch (m) {
            case StatisticMode::Mean: return "Mean";
            case StatisticMode::Variance: return "Variance";
            default: return "Unknown";
        }
    }

    inline const char* name(ValueAggregation a) {
        switch (a) {
            case ValueAggregation::PerVertex: return "Per-Vertex";
            case ValueAggregation::Bucketed: return "Bucketed";
            default: return "Unknown";
        }
    }
}

// Centralized mode configuration with compatibility logic
struct ModeConfig {
    ValueType value_type = ValueType::WolframHausdorffDimension;
    StatisticMode statistic_mode = StatisticMode::Mean;
    GraphScope graph_scope = GraphScope::Branchial;
    ValueAggregation value_aggregation = ValueAggregation::Bucketed;

    // Compatibility checks - single source of truth
    bool is_curvature() const {
        return is_curvature_type(value_type);
    }

    bool supports_variance() const {
        // Global: no variance (single mega-union, no per-state or per-bucket statistics)
        if (graph_scope == GraphScope::Global) return false;
        // Foliation + Curvature: no variance (curvature computed once on union graph)
        if (graph_scope == GraphScope::Foliation && is_curvature()) return false;
        // Branchial: always has variance (variance across states within coordinate buckets)
        // Foliation + Dimension: has variance (variance within coordinate buckets on union graph)
        return true;
    }

    bool supports_value_aggregation() const {
        // Value aggregation (PerVertex vs Bucketed) applies to all value types
        // For curvature: Bucketed averages curvature values within geodesic coordinate buckets
        return true;
    }

    bool supports_tube_visualization() const {
        return value_type == ValueType::WolframRicciTensor;
    }

    // State transitions with automatic constraint enforcement
    void set_value_type(ValueType vt) {
        value_type = vt;
        enforce_constraints();
    }

    void set_graph_scope(GraphScope gs) {
        graph_scope = gs;
        enforce_constraints();
    }

    void cycle_value_type() {
        int v = static_cast<int>(value_type);
        v = (v + 1) % 5;  // 5 value types
        set_value_type(static_cast<ValueType>(v));
    }

    void cycle_graph_scope() {
        int s = static_cast<int>(graph_scope);
        s = (s + 1) % 3;  // 3 scopes
        set_graph_scope(static_cast<GraphScope>(s));
    }

    bool toggle_statistic_mode() {
        if (!supports_variance()) return false;
        statistic_mode = (statistic_mode == StatisticMode::Mean)
            ? StatisticMode::Variance
            : StatisticMode::Mean;
        return true;
    }

    bool toggle_value_aggregation() {
        if (!supports_value_aggregation()) return false;
        value_aggregation = (value_aggregation == ValueAggregation::PerVertex)
            ? ValueAggregation::Bucketed
            : ValueAggregation::PerVertex;
        return true;
    }

    // Explanations for why operations are unavailable
    const char* why_no_variance() const {
        if (graph_scope == GraphScope::Global) return "Global mode uses single mega-union graph";
        if (graph_scope == GraphScope::Foliation && is_curvature()) return "Foliation curvature has no variance (single sample)";
        return nullptr;  // Variance IS supported
    }

    const char* why_no_value_aggregation() const {
        return nullptr;  // Aggregation is always supported
    }

    // Track what was auto-corrected for user feedback
    struct ConstraintResult {
        bool statistic_reset = false;
    };

    ConstraintResult last_constraint_result;

private:
    void enforce_constraints() {
        last_constraint_result = {};
        if (!supports_variance() && statistic_mode == StatisticMode::Variance) {
            statistic_mode = StatisticMode::Mean;
            last_constraint_result.statistic_reset = true;
        }
    }
};

// =============================================================================
// Highlight Mode & Shell Colors
// =============================================================================

enum class HighlightMode {
    Tube,  // Geodesic path + surrounding tube (for Curvature Tube->Scalar)
    Ball   // Single vertex + ball neighborhood (for Dimension/Curvature Ball)
};

// Mathematica ColorData[1,...] palette for distance-based shell coloring
static std::tuple<float, float, float> shell_color(int dist) {
    switch (dist) {
        case 0: return {0.37f, 0.51f, 0.71f};  // Blue (center/path)
        case 1: return {0.88f, 0.61f, 0.14f};  // Orange
        case 2: return {0.56f, 0.69f, 0.19f};  // Green
        case 3: return {0.92f, 0.38f, 0.21f};  // Red
        case 4: return {0.53f, 0.38f, 0.64f};  // Purple
        case 5: return {0.55f, 0.34f, 0.29f};  // Brown
        default: return {0.70f, 0.70f, 0.70f}; // Gray for distant
    }
}

// =============================================================================
// Vertex Selection & Histogram State
// =============================================================================

struct VertexSelectionState {
    bool has_selection = false;
    VertexId selected_vertex = 0;
    float panel_ndc_x = 0.0f;      // Panel position in NDC (clamped to screen)
    float panel_ndc_y = 0.0f;
    int click_screen_x = 0;        // Original click position
    int click_screen_y = 0;

    // User-adjustable histogram settings
    int histogram_bin_count = 10;  // Adjustable with +/- keys

    // Cached histogram data (recomputed when selection/timestep/mode changes)
    std::vector<float> dimension_values;  // Raw values (dimension or curvature)
    std::vector<int> bin_counts;          // Histogram bin counts
    int num_bins = 0;
    float bin_min = 0.0f;
    float bin_max = 0.0f;
    float bin_width = 0.0f;
    int max_bin_count = 0;
    float mean_value = 0.0f;
    float std_dev = 0.0f;
    GraphScope cached_graph_scope = GraphScope::Branchial;
    ValueType cached_value_type = ValueType::WolframHausdorffDimension;
    int cached_timestep = -1;
    int cached_bin_count = 10;

    // Distribution mode info (for Foliation/Global/Curvature - shows all vertices)
    bool is_distribution_mode = false;
    float selected_vertex_value = -1.0f;  // For highlighting selected vertex in distribution
    int selected_vertex_bin = -1;         // Which bin contains selected vertex

    // Hit testing bounds (for dismissing when clicking outside)
    float panel_left = 0.0f, panel_right = 0.0f;
    float panel_top = 0.0f, panel_bottom = 0.0f;

    // Highlight mode (Ball for dimension/curvature-ball, Tube for tensor)
    HighlightMode highlight_mode = HighlightMode::Ball;
    bool show_highlight = false;  // Toggle with T key
    float non_highlight_alpha = 0.15f;  // Dim non-highlighted vertices

    // Ball highlight data (single ball around clicked vertex)
    std::unordered_map<VertexId, int> ball_distances;  // Vertex -> distance from center
    int ball_radius = 5;  // Max distance to include in ball

    // Tube highlight data (geodesic path + surrounding tube)
    std::vector<std::unordered_map<VertexId, int>> tube_vertex_distances;  // Vertex -> distance from geodesic path
    std::vector<std::vector<VertexId>> tube_geodesic_paths;  // Central geodesic paths
    std::vector<int> tube_radii;  // Radius of each tube
    int current_tube_index = 0;  // Which tube to display (cycle with T key)
    int max_tube_radius = 1;  // Maximum distance in any tube (for color normalization)

    // Track when highlight was computed (for auto-recompute on timestep change)
    int highlight_timestep = -1;
};

// Picking sphere instance - outputs vertex index instead of color
struct PickingSphereInstance {
    float x, y, z;          // Same as SphereInstance
    float radius;
    uint32_t vertex_index;  // Index into vertices array
    uint32_t _pad[3];       // Padding to 32 bytes for alignment
};

// =============================================================================
// Rendering Data
// =============================================================================

struct RenderData {
    // Legacy vertex data (for timeline bar, horizon circles)
    std::vector<Vertex> vertex_data;   // Points (quads) - legacy
    std::vector<Vertex> edge_data;     // Lines - legacy
    size_t vertex_count = 0;
    size_t edge_count = 0;

    // Instanced data for 3D mode
    std::vector<SphereInstance> sphere_instances;      // Vertices as spheres
    std::vector<CylinderInstance> cylinder_instances;  // Edges as cylinders

    // Frequency data for legend (when edge_color_mode == Frequency)
    int min_freq = 1;   // Minimum frequency (edges and vertices)
    int max_freq = 1;   // Maximum frequency
    bool has_freq_data = false;  // True if frequency was computed
};

// =============================================================================
// Text Rendering - Embedded 8x8 Font
// =============================================================================

// 8x8 bitmap font - CP437-style, 95 printable ASCII chars (32-126)
// Each character is 8 bytes (8 rows of 8 bits)
// Layout: 16 chars per row, 6 rows = 96 chars
// Texture size: 128x48 pixels (8*16 x 8*6)
static const uint8_t FONT_8X8[95 * 8] = {
    // Space (32)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // ! (33)
    0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00,
    // " (34)
    0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00,
    // # (35)
    0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C, 0x6C, 0x00,
    // $ (36)
    0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00,
    // % (37)
    0x00, 0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00,
    // & (38)
    0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00,
    // ' (39)
    0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00,
    // ( (40)
    0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00,
    // ) (41)
    0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00,
    // * (42)
    0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00,
    // + (43)
    0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00,
    // , (44)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30,
    // - (45)
    0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00,
    // . (46)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00,
    // / (47)
    0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00,
    // 0 (48)
    0x7C, 0xCE, 0xDE, 0xF6, 0xE6, 0xC6, 0x7C, 0x00,
    // 1 (49)
    0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00,
    // 2 (50)
    0x7C, 0xC6, 0x06, 0x1C, 0x30, 0x66, 0xFE, 0x00,
    // 3 (51)
    0x7C, 0xC6, 0x06, 0x3C, 0x06, 0xC6, 0x7C, 0x00,
    // 4 (52)
    0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x1E, 0x00,
    // 5 (53)
    0xFE, 0xC0, 0xC0, 0xFC, 0x06, 0xC6, 0x7C, 0x00,
    // 6 (54)
    0x38, 0x60, 0xC0, 0xFC, 0xC6, 0xC6, 0x7C, 0x00,
    // 7 (55)
    0xFE, 0xC6, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00,
    // 8 (56)
    0x7C, 0xC6, 0xC6, 0x7C, 0xC6, 0xC6, 0x7C, 0x00,
    // 9 (57)
    0x7C, 0xC6, 0xC6, 0x7E, 0x06, 0x0C, 0x78, 0x00,
    // : (58)
    0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00,
    // ; (59)
    0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x30,
    // < (60)
    0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00,
    // = (61)
    0x00, 0x00, 0x7E, 0x00, 0x00, 0x7E, 0x00, 0x00,
    // > (62)
    0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00,
    // ? (63)
    0x7C, 0xC6, 0x0C, 0x18, 0x18, 0x00, 0x18, 0x00,
    // @ (64)
    0x7C, 0xC6, 0xDE, 0xDE, 0xDE, 0xC0, 0x78, 0x00,
    // A (65)
    0x38, 0x6C, 0xC6, 0xFE, 0xC6, 0xC6, 0xC6, 0x00,
    // B (66)
    0xFC, 0x66, 0x66, 0x7C, 0x66, 0x66, 0xFC, 0x00,
    // C (67)
    0x3C, 0x66, 0xC0, 0xC0, 0xC0, 0x66, 0x3C, 0x00,
    // D (68)
    0xF8, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0xF8, 0x00,
    // E (69)
    0xFE, 0x62, 0x68, 0x78, 0x68, 0x62, 0xFE, 0x00,
    // F (70)
    0xFE, 0x62, 0x68, 0x78, 0x68, 0x60, 0xF0, 0x00,
    // G (71)
    0x3C, 0x66, 0xC0, 0xC0, 0xCE, 0x66, 0x3A, 0x00,
    // H (72)
    0xC6, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0xC6, 0x00,
    // I (73)
    0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00,
    // J (74)
    0x1E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78, 0x00,
    // K (75)
    0xE6, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0xE6, 0x00,
    // L (76)
    0xF0, 0x60, 0x60, 0x60, 0x62, 0x66, 0xFE, 0x00,
    // M (77)
    0xC6, 0xEE, 0xFE, 0xFE, 0xD6, 0xC6, 0xC6, 0x00,
    // N (78)
    0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6, 0x00,
    // O (79)
    0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00,
    // P (80)
    0xFC, 0x66, 0x66, 0x7C, 0x60, 0x60, 0xF0, 0x00,
    // Q (81)
    0x7C, 0xC6, 0xC6, 0xC6, 0xD6, 0xDE, 0x7C, 0x06,
    // R (82)
    0xFC, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0xE6, 0x00,
    // S (83)
    0x3C, 0x66, 0x30, 0x18, 0x0C, 0x66, 0x3C, 0x00,
    // T (84)
    0x7E, 0x5A, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00,
    // U (85)
    0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00,
    // V (86)
    0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00,
    // W (87)
    0xC6, 0xC6, 0xC6, 0xD6, 0xD6, 0xFE, 0x6C, 0x00,
    // X (88)
    0xC6, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0xC6, 0x00,
    // Y (89)
    0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x3C, 0x00,
    // Z (90)
    0xFE, 0xC6, 0x8C, 0x18, 0x32, 0x66, 0xFE, 0x00,
    // [ (91)
    0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00,
    // \ (92)
    0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00,
    // ] (93)
    0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00,
    // ^ (94)
    0x10, 0x38, 0x6C, 0xC6, 0x00, 0x00, 0x00, 0x00,
    // _ (95)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
    // ` (96)
    0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
    // a (97)
    0x00, 0x00, 0x78, 0x0C, 0x7C, 0xCC, 0x76, 0x00,
    // b (98)
    0xE0, 0x60, 0x7C, 0x66, 0x66, 0x66, 0xDC, 0x00,
    // c (99)
    0x00, 0x00, 0x7C, 0xC6, 0xC0, 0xC6, 0x7C, 0x00,
    // d (100)
    0x1C, 0x0C, 0x7C, 0xCC, 0xCC, 0xCC, 0x76, 0x00,
    // e (101)
    0x00, 0x00, 0x7C, 0xC6, 0xFE, 0xC0, 0x7C, 0x00,
    // f (102)
    0x3C, 0x66, 0x60, 0xF8, 0x60, 0x60, 0xF0, 0x00,
    // g (103)
    0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0xF8,
    // h (104)
    0xE0, 0x60, 0x6C, 0x76, 0x66, 0x66, 0xE6, 0x00,
    // i (105)
    0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00,
    // j (106)
    0x06, 0x00, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C,
    // k (107)
    0xE0, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0xE6, 0x00,
    // l (108)
    0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00,
    // m (109)
    0x00, 0x00, 0xEC, 0xFE, 0xD6, 0xD6, 0xD6, 0x00,
    // n (110)
    0x00, 0x00, 0xDC, 0x66, 0x66, 0x66, 0x66, 0x00,
    // o (111)
    0x00, 0x00, 0x7C, 0xC6, 0xC6, 0xC6, 0x7C, 0x00,
    // p (112)
    0x00, 0x00, 0xDC, 0x66, 0x66, 0x7C, 0x60, 0xF0,
    // q (113)
    0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0x1E,
    // r (114)
    0x00, 0x00, 0xDC, 0x76, 0x60, 0x60, 0xF0, 0x00,
    // s (115)
    0x00, 0x00, 0x7E, 0xC0, 0x7C, 0x06, 0xFC, 0x00,
    // t (116)
    0x30, 0x30, 0xFC, 0x30, 0x30, 0x36, 0x1C, 0x00,
    // u (117)
    0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0x76, 0x00,
    // v (118)
    0x00, 0x00, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00,
    // w (119)
    0x00, 0x00, 0xC6, 0xD6, 0xD6, 0xFE, 0x6C, 0x00,
    // x (120)
    0x00, 0x00, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0x00,
    // y (121)
    0x00, 0x00, 0xC6, 0xC6, 0xC6, 0x7E, 0x06, 0xFC,
    // z (122)
    0x00, 0x00, 0xFE, 0x8C, 0x18, 0x32, 0xFE, 0x00,
    // { (123)
    0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00,
    // | (124)
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00,
    // } (125)
    0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00,
    // ~ (126)
    0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Font atlas dimensions
static const int FONT_CHAR_WIDTH = 8;
static const int FONT_CHAR_HEIGHT = 8;
static const int FONT_ATLAS_COLS = 16;  // 16 chars per row
static const int FONT_ATLAS_ROWS = 6;   // 6 rows
static const int FONT_ATLAS_WIDTH = FONT_ATLAS_COLS * FONT_CHAR_WIDTH;   // 128
static const int FONT_ATLAS_HEIGHT = FONT_ATLAS_ROWS * FONT_CHAR_HEIGHT; // 48
static const int FONT_FIRST_CHAR = 32;  // Space
static const int FONT_LAST_CHAR = 126;  // Tilde

// Generate font atlas texture data (R8 format)
std::vector<uint8_t> generate_font_atlas() {
    std::vector<uint8_t> atlas(FONT_ATLAS_WIDTH * FONT_ATLAS_HEIGHT, 0);

    for (int char_idx = 0; char_idx < 95; ++char_idx) {
        int col = char_idx % FONT_ATLAS_COLS;
        int row = char_idx / FONT_ATLAS_COLS;

        for (int y = 0; y < FONT_CHAR_HEIGHT; ++y) {
            uint8_t line = FONT_8X8[char_idx * 8 + y];
            for (int x = 0; x < FONT_CHAR_WIDTH; ++x) {
                // Bit 7 is leftmost pixel
                bool pixel_set = (line & (0x80 >> x)) != 0;
                int atlas_x = col * FONT_CHAR_WIDTH + x;
                int atlas_y = row * FONT_CHAR_HEIGHT + y;
                atlas[atlas_y * FONT_ATLAS_WIDTH + atlas_x] = pixel_set ? 255 : 0;
            }
        }
    }

    return atlas;
}

// Glyph instance for text rendering
struct GlyphInstance {
    float x, y;           // Screen position (pixels)
    float u_min, v_min;   // UV rect min
    float u_max, v_max;   // UV rect max
    float r, g, b, a;     // Color
};
static_assert(sizeof(GlyphInstance) == 40, "GlyphInstance must be 40 bytes");
static_assert(offsetof(GlyphInstance, x) == 0, "x offset wrong");
static_assert(offsetof(GlyphInstance, u_min) == 8, "u_min offset wrong");
static_assert(offsetof(GlyphInstance, r) == 24, "r offset wrong");

// Text color constants
namespace TextColor {
    const Vec4 White    = {0.9f, 0.9f, 0.9f, 1.0f};
    const Vec4 Gray     = {0.6f, 0.6f, 0.6f, 1.0f};
    const Vec4 Yellow   = {1.0f, 0.9f, 0.2f, 1.0f};
    const Vec4 Cyan     = {0.4f, 0.8f, 1.0f, 1.0f};
    const Vec4 Green    = {0.4f, 1.0f, 0.4f, 1.0f};
    const Vec4 Red      = {1.0f, 0.4f, 0.4f, 1.0f};
    const Vec4 Orange   = {1.0f, 0.6f, 0.2f, 1.0f};
}

// =============================================================================
// Edge Coloring Mode
// =============================================================================

enum class EdgeColorMode {
    Vertex,     // Color edges by vertex dimension (gradient between endpoints)
    Frequency   // Color edges by state frequency (how many states contain this edge)
};

// =============================================================================
// Keybinding Definitions (Unified for stdout and overlay)
// =============================================================================

struct KeyBinding {
    std::string key;
    std::string description;
    std::function<std::string()> get_value;  // nullptr for static bindings
};

struct KeyBindingGroup {
    std::string name;
    std::vector<KeyBinding> bindings;
};

// Build render data for a specific timestep
// If live_positions is provided, use those instead of stored positions
// layout_vertices maps layout indices to vertex IDs (needed for correct position lookup)
RenderData build_render_data(
    const BHAnalysisResult& analysis,
    int timestep,
    ViewMode view_mode,
    EdgeDisplayMode edge_mode = EdgeDisplayMode::Union,  // Which edges to show
    int edge_freq_threshold = 2,  // Edges must appear in >= N states (for Frequent mode)
    int selected_state_index = 0,  // Which state to show in SingleState mode
    ValueType value_type = ValueType::WolframHausdorffDimension,  // What quantity to display
    StatisticMode statistic_mode = StatisticMode::Mean,  // Mean or Variance
    GraphScope graph_scope = GraphScope::Branchial,  // Branchial, Foliation, or Global
    bool per_frame_normalization = false,  // Per-frame (local) vs global normalization
    bool z_mapping_enabled = true,  // Map dimension to Z/height (false = flat)
    MissingDataMode missing_mode = MissingDataMode::Show,  // How to display missing vertices
    ColorPalette palette = ColorPalette::Temperature,  // Color palette for dimension heatmap
    EdgeColorMode edge_color_mode = EdgeColorMode::Vertex,  // How to color edges
    const layout::LayoutGraph* live_layout = nullptr,  // Optional live positions from layout engine
    const std::vector<VertexId>* layout_vertices = nullptr,  // Vertex IDs in layout order
    const PositionCache* position_cache = nullptr,  // Persistent positions from layout engine
    float vertex_radius = 0.04f,  // Smaller spheres for cleaner visualization
    float edge_radius = 0.015f,   // Thinner edges
    float z_scale = 3.0f,  // Z height scale factor
    bool timeslice_enabled = false,  // Aggregate across multiple timesteps
    int timeslice_width = 5,  // Number of timesteps to aggregate
    const std::vector<std::vector<int>>* all_selected_states = nullptr,  // Per-timestep selected state indices (for path selection)
    bool show_geodesics = false,  // Overlay geodesic paths
    bool show_defects = false,  // Overlay topological defect markers
    // Highlight visualization (Ball or Tube mode)
    bool show_highlight = false,
    const std::unordered_map<VertexId, int>* highlight_distances = nullptr,  // vertex -> distance from center/path
    VertexId highlight_source = 0,  // Center vertex (Ball) or source vertex (Tube)
    VertexId highlight_target = 0,  // Target vertex (Tube only, 0 for Ball)
    float non_highlight_alpha = 0.15f,
    // Value aggregation (Per-Vertex vs Bucketed)
    ValueAggregation value_aggregation = ValueAggregation::Bucketed
) {
    RenderData data;

    if (timestep < 0 || timestep >= static_cast<int>(analysis.per_timestep.size())) {
        return data;
    }

    // Determine timestep range for aggregation
    int half_width = timeslice_width / 2;
    int slice_start = timestep;
    int slice_end = timestep;
    if (timeslice_enabled && timeslice_width > 1) {
        slice_start = std::max(0, timestep - half_width);
        slice_end = std::min(static_cast<int>(analysis.per_timestep.size()) - 1, timestep + half_width);
    }

    // Aggregate data from timestep range using pre-computed prefix sums (O(1) range queries)
    std::vector<VertexId> agg_vertices;
    std::vector<Edge> agg_edges;
    std::unordered_map<VertexId, Vec2> agg_positions;  // Position from center timestep
    std::unordered_map<VertexId, float> agg_dimensions;  // Averaged dimension values

    const auto& center_ts = analysis.per_timestep[timestep];
    if (center_ts.union_vertices.empty()) return data;

    // Handle SingleState mode: use vertices/edges from a specific state
    if (edge_mode == EdgeDisplayMode::SingleState &&
        timestep < static_cast<int>(analysis.states_per_step.size()) &&
        !analysis.states_per_step[timestep].empty()) {
        int state_idx = selected_state_index % static_cast<int>(analysis.states_per_step[timestep].size());
        const auto& state = analysis.states_per_step[timestep][state_idx];
        agg_vertices = state.vertices;
        agg_edges = state.edges;
        // Note: dimension values still come from center_ts (aggregated), which is correct
        // since we want to show per-vertex colors using the aggregated dimension data
    } else if (!timeslice_enabled || slice_start == slice_end) {
        // Single timestep - use directly
        agg_vertices = center_ts.union_vertices;
        agg_edges = center_ts.union_edges;
    } else {
        // Use prefix sums for dimension averaging (O(V) instead of O(W*V))
        const auto& end_ts = analysis.per_timestep[slice_end];

        // Select which prefix sums to use based on graph_scope and statistic_mode
        // Foliation/Global use union-graph dimensions, Branchial uses per-state averaged dimensions
        // TODO: True Global mode uses mega-union across all timesteps (constant values, no timeslicing)
        const bool use_foliation_data = (graph_scope == GraphScope::Foliation || graph_scope == GraphScope::Global);
        const auto* end_sum = use_foliation_data
            ? ((statistic_mode == StatisticMode::Variance) ? &end_ts.global_var_prefix_sum : &end_ts.global_dim_prefix_sum)
            : ((statistic_mode == StatisticMode::Variance) ? &end_ts.var_prefix_sum : &end_ts.dim_prefix_sum);
        const auto* end_count = use_foliation_data
            ? ((statistic_mode == StatisticMode::Variance) ? &end_ts.global_var_prefix_count : &end_ts.global_dim_prefix_count)
            : ((statistic_mode == StatisticMode::Variance) ? &end_ts.var_prefix_count : &end_ts.dim_prefix_count);

        // Get start-1 prefix sums (or empty if slice_start == 0)
        const std::unordered_map<VertexId, float>* start_sum = nullptr;
        const std::unordered_map<VertexId, int>* start_count = nullptr;
        if (slice_start > 0) {
            const auto& start_ts = analysis.per_timestep[slice_start - 1];
            start_sum = use_foliation_data
                ? ((statistic_mode == StatisticMode::Variance) ? &start_ts.global_var_prefix_sum : &start_ts.global_dim_prefix_sum)
                : ((statistic_mode == StatisticMode::Variance) ? &start_ts.var_prefix_sum : &start_ts.dim_prefix_sum);
            start_count = use_foliation_data
                ? ((statistic_mode == StatisticMode::Variance) ? &start_ts.global_var_prefix_count : &start_ts.global_dim_prefix_count)
                : ((statistic_mode == StatisticMode::Variance) ? &start_ts.var_prefix_count : &start_ts.dim_prefix_count);
        }

        // Use dim_prefix_count to filter vertices to only those in the slice range
        // (a vertex is in the range if its count increased between start-1 and end)
        const auto& end_presence_count = end_ts.dim_prefix_count;  // Always use dim count for presence check
        const std::unordered_map<VertexId, int>* start_presence_count = nullptr;
        if (slice_start > 0) {
            start_presence_count = &analysis.per_timestep[slice_start - 1].dim_prefix_count;
        }

        // Compute range averages for each vertex, filtering to only vertices in range
        for (VertexId v : analysis.all_vertices) {
            // Check if vertex is present in slice range
            int presence_end = 0, presence_start = 0;
            auto it_pres_end = end_presence_count.find(v);
            if (it_pres_end != end_presence_count.end()) {
                presence_end = it_pres_end->second;
            }
            if (start_presence_count) {
                auto it_pres_start = start_presence_count->find(v);
                if (it_pres_start != start_presence_count->end()) {
                    presence_start = it_pres_start->second;
                }
            }

            // Only include vertex if it appears in the slice range
            if (presence_end - presence_start <= 0) {
                continue;  // Vertex not in slice range
            }

            agg_vertices.push_back(v);

            // Compute dimension average
            float sum_end = 0, sum_start = 0;
            int count_end = 0, count_start = 0;

            auto it_end = end_sum->find(v);
            if (it_end != end_sum->end()) {
                sum_end = it_end->second;
            }
            auto it_count_end = end_count->find(v);
            if (it_count_end != end_count->end()) {
                count_end = it_count_end->second;
            }

            if (start_sum) {
                auto it_start = start_sum->find(v);
                if (it_start != start_sum->end()) {
                    sum_start = it_start->second;
                }
            }
            if (start_count) {
                auto it_count_start = start_count->find(v);
                if (it_count_start != start_count->end()) {
                    count_start = it_count_start->second;
                }
            }

            float range_sum = sum_end - sum_start;
            int range_count = count_end - count_start;

            if (range_count > 0) {
                agg_dimensions[v] = range_sum / static_cast<float>(range_count);
            }
        }

        // Build edge set for vertices in slice range (using all_edges but filtering)
        std::unordered_set<VertexId> vertex_set(agg_vertices.begin(), agg_vertices.end());
        for (const auto& e : analysis.all_edges) {
            if (vertex_set.count(e.v1) > 0 && vertex_set.count(e.v2) > 0) {
                agg_edges.push_back(e);
            }
        }
    }

    // Build position map from timesteps in slice range
    // For timeslice mode, we need positions for vertices in the whole slice, not just center
    // Priority for position sources:
    //   1. position_cache (live layout positions from previous frames - most accurate)
    //   2. layout_positions from timestep data (pre-computed during analysis)
    //   3. vertex_positions from timestep data (geodesic coordinates)
    //   4. initial.vertex_positions (for original vertices)
    //   5. (0, 0) as last resort (should rarely happen with good data)

    // Build vertex-to-index map for step 0 (for fallback lookup when layout is off)
    std::unordered_map<VertexId, size_t> step0_vertex_idx;
    const auto& step0 = analysis.per_timestep[0];
    for (size_t i = 0; i < step0.union_vertices.size(); ++i) {
        step0_vertex_idx[step0.union_vertices[i]] = i;
    }

    // Helper lambda to get best position for a vertex
    auto get_vertex_position = [&](VertexId v, const TimestepAggregation& ts, size_t idx) -> Vec2 {
        // Check position cache first (highest priority - live layout positions)
        if (position_cache) {
            auto cache_it = position_cache->find(v);
            if (cache_it != position_cache->end()) {
                return Vec2{cache_it->second.first, cache_it->second.second};
            }
        }

        // Check layout_positions from timestep (pre-computed layout)
        if (idx < ts.layout_positions.size()) {
            const Vec2& pos = ts.layout_positions[idx];
            // Skip if it's at origin (likely placeholder)
            if (std::abs(pos.x) > 0.001f || std::abs(pos.y) > 0.001f) {
                return pos;
            }
        }

        // Check vertex_positions (geodesic coordinates)
        if (idx < ts.vertex_positions.size()) {
            const Vec2& pos = ts.vertex_positions[idx];
            // Skip if it's at origin (placeholder for new vertices)
            if (std::abs(pos.x) > 0.001f || std::abs(pos.y) > 0.001f) {
                return pos;
            }
        }

        // Check step 0's layout_positions (if vertex exists there and has a layout)
        // This is crucial when layout is off - step 0 has the only pre-computed layout
        auto step0_it = step0_vertex_idx.find(v);
        if (step0_it != step0_vertex_idx.end()) {
            size_t step0_idx = step0_it->second;
            if (step0_idx < step0.layout_positions.size()) {
                const Vec2& pos = step0.layout_positions[step0_idx];
                if (std::abs(pos.x) > 0.001f || std::abs(pos.y) > 0.001f) {
                    return pos;
                }
            }
        }

        // Check initial condition positions
        if (v < analysis.initial.vertex_positions.size()) {
            return analysis.initial.vertex_positions[v];
        }

        // Last resort - will be fixed up later via neighbor interpolation
        return Vec2{0, 0};
    };

    if (timeslice_enabled && slice_start != slice_end) {
        // Build positions from all timesteps in slice range (outer to inner so center wins)
        for (int s = slice_start; s <= slice_end; ++s) {
            if (s == timestep) continue;  // Do center last
            const auto& step_ts = analysis.per_timestep[s];
            for (size_t i = 0; i < step_ts.union_vertices.size(); ++i) {
                VertexId v = step_ts.union_vertices[i];
                if (agg_positions.count(v) > 0) continue;  // Already have position
                agg_positions[v] = get_vertex_position(v, step_ts, i);
            }
        }
    }
    // Always include center timestep positions (these take priority)
    for (size_t i = 0; i < center_ts.union_vertices.size(); ++i) {
        VertexId v = center_ts.union_vertices[i];
        agg_positions[v] = get_vertex_position(v, center_ts, i);
    }

    // Fill in any missing positions from initial condition (for vertices in agg_vertices not in any timestep)
    if (timeslice_enabled && !agg_vertices.empty()) {
        for (VertexId v : agg_vertices) {
            if (agg_positions.count(v) == 0) {
                // Try position cache first
                if (position_cache) {
                    auto cache_it = position_cache->find(v);
                    if (cache_it != position_cache->end()) {
                        agg_positions[v] = Vec2{cache_it->second.first, cache_it->second.second};
                        continue;
                    }
                }
                // Fall back to initial
                if (v < analysis.initial.vertex_positions.size()) {
                    agg_positions[v] = analysis.initial.vertex_positions[v];
                } else {
                    agg_positions[v] = Vec2{0, 0};
                }
            }
        }
    }

    // Second pass: fix up any (0,0) positions using neighbor interpolation
    // This ensures vertices created during evolution get reasonable positions
    // Applied when: timeslice is enabled OR layout is disabled (live_layout == nullptr)
    bool need_interpolation = timeslice_enabled || (live_layout == nullptr);
    if (need_interpolation) {
        // Always use center timestep for interpolation (timeslice only affects dimension averaging)
        const std::vector<VertexId>& verts_for_interp = center_ts.union_vertices;
        const std::vector<Edge>& edges_for_interp = center_ts.union_edges;

        // Ensure all vertices have entries in agg_positions
        for (VertexId v : verts_for_interp) {
            if (agg_positions.count(v) == 0) {
                if (v < analysis.initial.vertex_positions.size()) {
                    agg_positions[v] = analysis.initial.vertex_positions[v];
                } else {
                    agg_positions[v] = Vec2{0, 0};
                }
            }
        }

        // Build adjacency from edges
        std::unordered_map<VertexId, std::vector<VertexId>> adjacency;
        for (const auto& e : edges_for_interp) {
            adjacency[e.v1].push_back(e.v2);
            adjacency[e.v2].push_back(e.v1);
        }

        // Multiple passes to handle chains
        for (int pass = 0; pass < 3; ++pass) {
            bool any_fixed = false;
            for (VertexId v : verts_for_interp) {
                auto pos_it = agg_positions.find(v);
                if (pos_it == agg_positions.end()) continue;

                // Check if position is at origin (needs fixing)
                if (std::abs(pos_it->second.x) < 0.001f && std::abs(pos_it->second.y) < 0.001f) {
                    // Find neighbors with good positions
                    auto adj_it = adjacency.find(v);
                    if (adj_it != adjacency.end()) {
                        float sum_x = 0, sum_y = 0;
                        int count = 0;
                        for (VertexId neighbor : adj_it->second) {
                            auto neighbor_pos = agg_positions.find(neighbor);
                            if (neighbor_pos != agg_positions.end() &&
                                (std::abs(neighbor_pos->second.x) > 0.001f || std::abs(neighbor_pos->second.y) > 0.001f)) {
                                sum_x += neighbor_pos->second.x;
                                sum_y += neighbor_pos->second.y;
                                count++;
                            }
                        }
                        if (count > 0) {
                            pos_it->second.x = sum_x / count;
                            pos_it->second.y = sum_y / count;
                            any_fixed = true;
                        }
                    }
                }
            }
            if (!any_fixed) break;
        }
    }

    // Reference for the rest of the function - use center timestep for stats
    const auto& ts = center_ts;

    // Color scaling - use appropriate range based on value type and dimension mode
    float value_min, value_max;
    float curvature_abs_max = 0.0f;  // For symmetric curvature scaling
    bool using_curvature = is_curvature_type(value_type);

    // Select appropriate curvature map pointer (for Global scope only)
    const std::unordered_map<VertexId, float>* curvature_map = nullptr;
    int curv_idx = curvature_type_index(value_type);
    if (using_curvature && analysis.has_curvature_analysis && graph_scope == GraphScope::Global) {
        curvature_map = &analysis.get_global_curvature_map(curv_idx);
    }

    if (using_curvature) {
        // Get curvature ranges using accessor methods (index-based for extensibility)
        float curv_min, curv_max;
        if (graph_scope == GraphScope::Global) {
            // Global: use mega-union curvature ranges (no per-frame option)
            analysis.get_global_curvature_range(curv_idx, curv_min, curv_max);
        } else if (graph_scope == GraphScope::Foliation) {
            // Foliation: per-timestep or global quantiles
            if (per_frame_normalization) {
                ts.get_foliation_curvature_range(curv_idx, curv_min, curv_max);
            } else {
                analysis.get_foliation_curvature_quantiles(curv_idx, curv_min, curv_max);
            }
        } else {
            // Branchial: mean or variance, per-timestep or global quantiles
            if (statistic_mode == StatisticMode::Variance) {
                if (per_frame_normalization) {
                    ts.get_variance_curvature_range(curv_idx, curv_min, curv_max);
                } else {
                    analysis.get_variance_curvature_quantiles(curv_idx, curv_min, curv_max);
                }
            } else {
                if (per_frame_normalization) {
                    ts.get_mean_curvature_range(curv_idx, curv_min, curv_max);
                } else {
                    analysis.get_mean_curvature_quantiles(curv_idx, curv_min, curv_max);
                }
            }
        }

        if (statistic_mode == StatisticMode::Variance) {
            // Variance mode: use sequential colormap (variance is always >= 0)
            value_min = curv_min;
            value_max = curv_max;
            curvature_abs_max = 0.0f;  // Not used for variance mode
        } else {
            // Mean mode: use symmetric range for diverging colormap
            curvature_abs_max = std::max(std::abs(curv_min), std::abs(curv_max));
            if (curvature_abs_max < 0.001f) curvature_abs_max = 1.0f;  // Avoid division by zero
            value_min = -curvature_abs_max;
            value_max = curvature_abs_max;
        }
    } else if (graph_scope == GraphScope::Global) {
        // Global mode uses mega-union (constant across all timesteps, no per-frame option)
        value_min = analysis.mega_dim_min;
        value_max = analysis.mega_dim_max;
    } else if (graph_scope == GraphScope::Foliation) {
        if (statistic_mode == StatisticMode::Variance) {
            // Foliation dimension variance (within-bucket variance on union graph)
            if (per_frame_normalization) {
                value_min = ts.global_var_min;
                value_max = ts.global_var_max;
            } else {
                value_min = analysis.global_var_q05;
                value_max = analysis.global_var_q95;
            }
        } else {
            // Foliation mean
            if (per_frame_normalization) {
                value_min = ts.global_mean_min;
                value_max = ts.global_mean_max;
            } else {
                value_min = analysis.global_dim_q05;
                value_max = analysis.global_dim_q95;
            }
        }
    } else {
        // Branchial mode uses per-state averaged dimensions
        if (statistic_mode == StatisticMode::Variance) {
            // Branchial variance
            if (per_frame_normalization) {
                value_min = ts.var_min;
                value_max = ts.var_max;
            } else {
                value_min = analysis.var_q05;
                value_max = analysis.var_q95;
            }
        } else {
            // Branchial mean
            if (per_frame_normalization) {
                value_min = ts.pooled_min;
                value_max = ts.pooled_max;
            } else {
                value_min = analysis.dim_q05;
                value_max = analysis.dim_q95;
            }
        }
    }

    // ALWAYS use center timestep vertices for rendering
    // Timeslice only affects dimension value averaging (stored in agg_dimensions), not visibility
    const std::vector<VertexId>& vertices_to_render = ts.union_vertices;

    // Pre-compute bucketed curvature values if needed (curvature + Bucketed mode)
    // Groups vertices by geodesic coordinate and averages curvature within each bucket
    std::vector<float> bucketed_curvature;
    if (using_curvature && curvature_map && value_aggregation == ValueAggregation::Bucketed) {
        // Build map from VertexId to CoordKey using per-state analysis data
        // Find a state at this timestep to get vertex coordinates
        std::unordered_map<VertexId, CoordKey> vertex_to_coord;
        int num_anchors = 0;
        for (const auto& state : analysis.per_state) {
            if (state.step == ts.step) {
                num_anchors = state.num_anchors;
                for (size_t i = 0; i < state.vertices.size() && i < state.vertex_coords.size(); ++i) {
                    VertexId vid = state.vertices[i];
                    if (vertex_to_coord.find(vid) == vertex_to_coord.end()) {
                        CoordKey key;
                        key.num_anchors = num_anchors;
                        for (int a = 0; a < MAX_ANCHORS; ++a) {
                            key.coords[a] = state.vertex_coords[i][a];
                        }
                        vertex_to_coord[vid] = key;
                    }
                }
            }
        }

        if (!vertex_to_coord.empty()) {
            // First pass: accumulate curvature values per coordinate bucket
            std::unordered_map<CoordKey, std::pair<float, int>, CoordKeyHash> coord_to_curv_accum;
            for (VertexId vid : vertices_to_render) {
                auto coord_it = vertex_to_coord.find(vid);
                auto curv_it = curvature_map->find(vid);
                if (coord_it != vertex_to_coord.end() && curv_it != curvature_map->end() &&
                    std::isfinite(curv_it->second)) {
                    auto& accum = coord_to_curv_accum[coord_it->second];
                    accum.first += curv_it->second;
                    accum.second += 1;
                }
            }
            // Compute averages
            std::unordered_map<CoordKey, float, CoordKeyHash> coord_to_curv;
            for (const auto& [key, accum] : coord_to_curv_accum) {
                if (accum.second > 0) {
                    coord_to_curv[key] = accum.first / accum.second;
                }
            }
            // Second pass: build bucketed curvature array for each vertex
            bucketed_curvature.resize(vertices_to_render.size(), 0.0f);
            for (size_t i = 0; i < vertices_to_render.size(); ++i) {
                VertexId vid = vertices_to_render[i];
                auto coord_it = vertex_to_coord.find(vid);
                if (coord_it != vertex_to_coord.end()) {
                    auto curv_it = coord_to_curv.find(coord_it->second);
                    if (curv_it != coord_to_curv.end()) {
                        bucketed_curvature[i] = curv_it->second;
                        continue;
                    }
                }
                // Fall back to per-vertex value
                auto curv_it = curvature_map->find(vid);
                if (curv_it != curvature_map->end()) {
                    bucketed_curvature[i] = curv_it->second;
                }
            }
        }
    }

    // Build vertex index map, positions, and lookup tables
    std::unordered_map<VertexId, size_t> vertex_to_idx;
    std::vector<float> vertex_x, vertex_y, vertex_z;
    std::vector<float> vertex_dims;
    std::vector<Vec4> vertex_colors;
    std::vector<bool> vertex_hidden;  // Track which vertices should be hidden (missing data)

    vertex_x.reserve(vertices_to_render.size());
    vertex_y.reserve(vertices_to_render.size());
    vertex_z.reserve(vertices_to_render.size());
    vertex_dims.reserve(vertices_to_render.size());
    vertex_colors.reserve(vertices_to_render.size());
    vertex_hidden.reserve(vertices_to_render.size());

    // Build reverse mapping: vertex ID -> layout index (if layout_vertices provided)
    std::unordered_map<VertexId, size_t> vid_to_layout_idx;
    if (live_layout && layout_vertices) {
        for (size_t li = 0; li < layout_vertices->size(); ++li) {
            vid_to_layout_idx[(*layout_vertices)[li]] = li;
        }
    }

    for (size_t i = 0; i < vertices_to_render.size(); ++i) {
        VertexId vid = vertices_to_render[i];
        vertex_to_idx[vid] = i;

        // Use live layout positions if provided and this vertex is in the layout
        float px, py, pz = 0;
        auto layout_it = vid_to_layout_idx.find(vid);
        if (live_layout && layout_it != vid_to_layout_idx.end()) {
            size_t layout_idx = layout_it->second;
            px = live_layout->positions_x[layout_idx];
            py = live_layout->positions_y[layout_idx];
            // Get Z from layout (for 3D embedding/layout)
            if (layout_idx < live_layout->positions_z.size()) {
                pz = live_layout->positions_z[layout_idx];
            }
        } else {
            // Use aggregated positions (from center timestep) or fall back
            auto pos_it = agg_positions.find(vid);
            if (pos_it != agg_positions.end() &&
                (std::abs(pos_it->second.x) > 0.001f || std::abs(pos_it->second.y) > 0.001f)) {
                px = pos_it->second.x;
                py = pos_it->second.y;
            } else if (position_cache) {
                // Try position cache (live layout positions from previous frames)
                auto cache_it = position_cache->find(vid);
                if (cache_it != position_cache->end()) {
                    px = cache_it->second.first;
                    py = cache_it->second.second;
                } else if (vid < analysis.initial.vertex_positions.size()) {
                    px = analysis.initial.vertex_positions[vid].x;
                    py = analysis.initial.vertex_positions[vid].y;
                } else {
                    px = 0;
                    py = 0;
                }
            } else if (vid < analysis.initial.vertex_positions.size()) {
                // Fall back to initial condition position
                px = analysis.initial.vertex_positions[vid].x;
                py = analysis.initial.vertex_positions[vid].y;
            } else {
                // Last resort: default position (should rarely happen)
                px = 0;
                py = 0;
            }
            // Try to get Z from initial condition (for 3D embedding)
            if (vid < analysis.initial.vertex_z.size()) {
                pz = analysis.initial.vertex_z[vid];
            }
        }
        vertex_x.push_back(px);
        vertex_y.push_back(py);

        // Store base Z from layout/embedding (will be used or augmented below)
        float base_z = pz;

        // Get value based on value type and graph scope
        float value = -1.0f;
        if (using_curvature) {
            // Find vertex index in union_vertices
            size_t vertex_idx = SIZE_MAX;
            for (size_t j = 0; j < ts.union_vertices.size(); ++j) {
                if (ts.union_vertices[j] == vid) {
                    vertex_idx = j;
                    break;
                }
            }

            if (graph_scope == GraphScope::Global) {
                // Global: use mega-union curvature from BHAnalysisResult
                if (curvature_map) {
                    auto curv_it = curvature_map->find(vid);
                    if (curv_it != curvature_map->end()) {
                        value = curv_it->second;
                    } else {
                        value = 0.0f;
                    }
                }
            } else if (graph_scope == GraphScope::Foliation) {
                // Foliation: use union graph curvature (single sample, no variance)
                if (vertex_idx != SIZE_MAX) {
                    switch (value_type) {
                        case ValueType::OllivierRicciCurvature:
                            value = (vertex_idx < ts.foliation_curvature_ollivier.size()) ? ts.foliation_curvature_ollivier[vertex_idx] : 0.0f;
                            break;
                        case ValueType::WolframRicciScalar:
                            value = (vertex_idx < ts.foliation_curvature_wolfram_scalar.size()) ? ts.foliation_curvature_wolfram_scalar[vertex_idx] : 0.0f;
                            break;
                        case ValueType::WolframRicciTensor:
                            value = (vertex_idx < ts.foliation_curvature_wolfram_ricci.size()) ? ts.foliation_curvature_wolfram_ricci[vertex_idx] : 0.0f;
                            break;
                        case ValueType::DimensionGradient:
                            value = (vertex_idx < ts.foliation_curvature_dim_gradient.size()) ? ts.foliation_curvature_dim_gradient[vertex_idx] : 0.0f;
                            break;
                        default:
                            value = 0.0f;
                            break;
                    }
                }
            } else {
                // Branchial: use aggregated per-state curvature (mean or variance)
                if (vertex_idx != SIZE_MAX) {
                    if (statistic_mode == StatisticMode::Variance) {
                        switch (value_type) {
                            case ValueType::OllivierRicciCurvature:
                                value = (vertex_idx < ts.variance_curvature_ollivier.size()) ? ts.variance_curvature_ollivier[vertex_idx] : 0.0f;
                                break;
                            case ValueType::WolframRicciScalar:
                                value = (vertex_idx < ts.variance_curvature_wolfram_scalar.size()) ? ts.variance_curvature_wolfram_scalar[vertex_idx] : 0.0f;
                                break;
                            case ValueType::WolframRicciTensor:
                                value = (vertex_idx < ts.variance_curvature_wolfram_ricci.size()) ? ts.variance_curvature_wolfram_ricci[vertex_idx] : 0.0f;
                                break;
                            case ValueType::DimensionGradient:
                                value = (vertex_idx < ts.variance_curvature_dim_gradient.size()) ? ts.variance_curvature_dim_gradient[vertex_idx] : 0.0f;
                                break;
                            default:
                                value = 0.0f;
                                break;
                        }
                    } else {
                        // Mean mode
                        switch (value_type) {
                            case ValueType::OllivierRicciCurvature:
                                value = (vertex_idx < ts.mean_curvature_ollivier.size()) ? ts.mean_curvature_ollivier[vertex_idx] : 0.0f;
                                break;
                            case ValueType::WolframRicciScalar:
                                value = (vertex_idx < ts.mean_curvature_wolfram_scalar.size()) ? ts.mean_curvature_wolfram_scalar[vertex_idx] : 0.0f;
                                break;
                            case ValueType::WolframRicciTensor:
                                value = (vertex_idx < ts.mean_curvature_wolfram_ricci.size()) ? ts.mean_curvature_wolfram_ricci[vertex_idx] : 0.0f;
                                break;
                            case ValueType::DimensionGradient:
                                value = (vertex_idx < ts.mean_curvature_dim_gradient.size()) ? ts.mean_curvature_dim_gradient[vertex_idx] : 0.0f;
                                break;
                            default:
                                value = 0.0f;
                                break;
                        }
                    }
                }
            }
        } else if (timeslice_enabled && !agg_dimensions.empty()) {
            // Use pre-computed averaged value from timeslice aggregation (O(1) lookup)
            auto it = agg_dimensions.find(vid);
            if (it != agg_dimensions.end()) {
                value = it->second;
            }
        } else {
            // Find dimension value based on mode
            if (graph_scope == GraphScope::Global) {
                // Global mode: use mega_dimension (constant across all timesteps)
                auto it = analysis.mega_dimension.find(vid);
                if (it != analysis.mega_dimension.end()) {
                    value = it->second;
                }
            } else {
                // Branchial/Foliation: find vertex in center timestep
                for (size_t j = 0; j < ts.union_vertices.size(); ++j) {
                    if (ts.union_vertices[j] == vid) {
                        if (graph_scope == GraphScope::Foliation) {
                            if (statistic_mode == StatisticMode::Variance) {
                                value = (j < ts.global_variance_dimensions.size()) ? ts.global_variance_dimensions[j] : -1.0f;
                            } else {
                                value = (j < ts.global_mean_dimensions.size()) ? ts.global_mean_dimensions[j] : -1.0f;
                            }
                        } else {
                            // Branchial mode - use raw or bucketed based on aggregation setting
                            if (statistic_mode == StatisticMode::Variance) {
                                // Variance is only available for bucketed mode
                                value = (j < ts.variance_dimensions.size()) ? ts.variance_dimensions[j] : -1.0f;
                            } else {
                                // Mean: choose between raw per-vertex and bucketed
                                if (value_aggregation == ValueAggregation::PerVertex &&
                                    j < ts.raw_vertex_dimensions.size()) {
                                    value = ts.raw_vertex_dimensions[j];
                                } else {
                                    value = (j < ts.mean_dimensions.size()) ? ts.mean_dimensions[j] : -1.0f;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
        vertex_dims.push_back(value);

        // Track if this vertex should be hidden (only in Hide mode for missing/invalid data)
        // For curvature mean: only hide if truly missing (NaN/Inf), not if 0
        // For curvature variance: hide if negative (shouldn't happen) or NaN/Inf
        // For dimension: hide if negative or NaN/Inf
        bool is_missing;
        if (using_curvature && statistic_mode == StatisticMode::Mean) {
            is_missing = !std::isfinite(value);
        } else {
            is_missing = (value < 0 || !std::isfinite(value));
        }
        bool is_hidden = (missing_mode == MissingDataMode::Hide) && is_missing;
        vertex_hidden.push_back(is_hidden);

        // Color based on value type
        Vec4 color;
        if (using_curvature && statistic_mode == StatisticMode::Variance) {
            // Curvature + Variance: sequential colormap for dimension variance (always positive)
            color = dimension_to_color(value, value_min, value_max, palette, missing_mode);
        } else if (using_curvature) {
            // Curvature + Mean: diverging colormap for curvature values
            color = curvature_to_color(value, curvature_abs_max, palette);
        } else {
            // Dimension: sequential colormap
            color = dimension_to_color(value, value_min, value_max, palette, missing_mode);
        }
        vertex_colors.push_back(color);

        // 3D position: start with base_z from layout/embedding
        // Optionally add dimension-to-height mapping on top
        float z = base_z;
        // For curvature: any finite value is valid (including negative)
        // For dimension: only non-negative values are valid
        bool valid_value = using_curvature ? std::isfinite(value) : (value >= 0 && std::isfinite(value));
        if (z_mapping_enabled && view_mode == ViewMode::View3D && valid_value) {
            // Add dimension-based height offset to base Z
            float vmin = value_min, vmax = value_max;
            if (std::isfinite(vmin) && std::isfinite(vmax)) {
                if (vmin > vmax) std::swap(vmin, vmax);
                float range = vmax - vmin;
                if (range >= 0.001f) {
                    float t = std::clamp((value - vmin) / range, 0.0f, 1.0f);
                    z = base_z + t * z_scale;  // Add dimension height to base Z
                } else {
                    z = base_z + z_scale * 0.5f;  // Middle height when all values are same
                }
            }
        }
        vertex_z.push_back(z);
    }

    // Debug output for hidden vertices disabled (was spamming stdout)
    // Uncomment to debug missing dimension data:
    // if (missing_mode == MissingDataMode::Hide) {
    //     struct HiddenVertexInfo { VertexId id; float dim; Vec2 pos; };
    //     std::vector<HiddenVertexInfo> hidden_info;
    //     for (size_t i = 0; i < vertex_hidden.size(); ++i) {
    //         if (vertex_hidden[i]) {
    //             VertexId vid = vertices_to_render[i];
    //             Vec2 pos{0, 0};
    //             auto it = agg_positions.find(vid);
    //             if (it != agg_positions.end()) pos = it->second;
    //             hidden_info.push_back({vid, vertex_dims[i], pos});
    //         }
    //     }
    //     if (!hidden_info.empty()) {
    //         std::cout << "[Debug] Hiding " << hidden_info.size() << "/" << vertices_to_render.size()
    //                   << " vertices with missing dimension data at step " << timestep << std::endl;
    //     }
    // }

    // Select edge list based on display mode
    // ALWAYS use center timestep edges for rendering
    // Timeslice only affects dimension value averaging, not edge visibility
    const std::vector<Edge>* edges_to_render;
    EdgeDisplayMode effective_mode = edge_mode;
    std::vector<Edge> freq_filtered_edges;  // Storage for dynamically filtered edges

    // Edge and vertex frequency computation (needed for Frequent mode filtering and Frequency coloring)
    // We maintain two counts:
    // - edge_counts_by_id: counts by edge ID for correct frequency (handles edge multiplicity)
    // - edge_counts_by_pair: counts by vertex pair for rendering lookup (union_edges don't have IDs)
    std::unordered_map<EdgeId, int> edge_counts_by_id;    // Keyed by edge ID
    std::unordered_map<uint64_t, int> edge_counts_by_pair;  // Keyed by vertex pair for rendering
    std::unordered_map<VertexId, int> vertex_counts;  // How many states each vertex appears in
    int max_edge_count = 1;    // For normalizing frequency colors
    int min_edge_count = 1;
    int max_vertex_count = 1;
    int min_vertex_count = 1;

    // Vertex pair key for rendering lookup
    auto vertex_pair_key = [](const Edge& e) -> uint64_t {
        VertexId v1 = std::min(e.v1, e.v2);
        VertexId v2 = std::max(e.v1, e.v2);
        return (static_cast<uint64_t>(v1) << 32) | v2;
    };

    // Compute edge and vertex frequencies if we need them (Frequent mode or Frequency coloring)
    // IMPORTANT: Frequencies are computed at the CENTER timestep only, NOT across the timeslice.
    // Timeslice only affects dimension value averaging, not frequency counts.
    // This ensures max frequency <= num_states at the current timestep.
    bool need_freq_counts = (edge_mode == EdgeDisplayMode::Frequent || edge_color_mode == EdgeColorMode::Frequency);
    if (need_freq_counts) {
        // Count frequencies at the CENTER timestep only
        if (timestep < static_cast<int>(analysis.states_per_step.size())) {
            for (const auto& state : analysis.states_per_step[timestep]) {
                // Track which vertex pairs we've seen in THIS state (for pair counts)
                std::unordered_set<uint64_t> seen_pairs_this_state;

                // Count edges by edge ID (each edge counted once per state it appears in)
                for (const auto& e : state.edges) {
                    // Count by edge ID if available
                    if (e.id != INVALID_EDGE_ID) {
                        int count = ++edge_counts_by_id[e.id];
                        max_edge_count = std::max(max_edge_count, count);
                    }

                    // Also count by vertex pair (for rendering lookup)
                    // Only count each pair once per state even if there's edge multiplicity
                    uint64_t pair_key = vertex_pair_key(e);
                    if (seen_pairs_this_state.insert(pair_key).second) {
                        int count = ++edge_counts_by_pair[pair_key];
                        // Update max from pair counts too
                        max_edge_count = std::max(max_edge_count, count);
                    }
                }
                // Count vertices (each vertex counted once per state it appears in)
                for (VertexId v : state.vertices) {
                    int count = ++vertex_counts[v];
                    max_vertex_count = std::max(max_vertex_count, count);
                }
            }
        }
        // Compute min counts (for legend) - use pair counts for edges
        for (const auto& [k, c] : edge_counts_by_pair) {
            min_edge_count = std::min(min_edge_count, c);
        }
        for (const auto& [v, c] : vertex_counts) {
            min_vertex_count = std::min(min_vertex_count, c);
        }

        // In Frequency color mode, recolor vertices by their frequency
        if (edge_color_mode == EdgeColorMode::Frequency && !vertex_counts.empty()) {
            int freq_range = max_vertex_count - min_vertex_count;
            for (size_t i = 0; i < vertices_to_render.size(); ++i) {
                VertexId vid = vertices_to_render[i];
                auto it = vertex_counts.find(vid);
                int count = (it != vertex_counts.end()) ? it->second : 1;

                // Normalize frequency to 0-1 range
                float t = (freq_range > 0)
                    ? static_cast<float>(count - min_vertex_count) / static_cast<float>(freq_range)
                    : 1.0f;

                Vec3 freq_color = apply_palette(t, palette);
                vertex_colors[i] = Vec4(freq_color, 1.0f);
            }
        }
    }

    // Base edge set is always center timestep (timeslice only affects dimension averaging)
    const std::vector<Edge>* base_edges = &ts.union_edges;
    edges_to_render = base_edges;

    // Apply edge mode filtering (works with both timeslice and single timestep)
    if (edge_mode == EdgeDisplayMode::Intersection) {
        if (!ts.intersection_edges.empty()) {
            if (timeslice_enabled) {
                // Filter agg_edges to only include those in center timestep's intersection
                std::unordered_set<uint64_t> intersection_set;
                for (const auto& e : ts.intersection_edges) {
                    intersection_set.insert(vertex_pair_key(e));
                }
                for (const auto& e : *base_edges) {
                    if (intersection_set.count(vertex_pair_key(e))) {
                        freq_filtered_edges.push_back(e);
                    }
                }
                if (!freq_filtered_edges.empty()) {
                    edges_to_render = &freq_filtered_edges;
                } else {
                    effective_mode = EdgeDisplayMode::Union;  // Fallback
                }
            } else {
                edges_to_render = &ts.intersection_edges;
            }
        } else {
            effective_mode = EdgeDisplayMode::Union;  // Fallback
        }
    } else if (edge_mode == EdgeDisplayMode::Frequent) {
        // Use pre-computed edge counts (by vertex pair) for filtering
        if (!edge_counts_by_pair.empty()) {
            // Filter edges that appear in >= threshold states
            for (const auto& e : *base_edges) {
                auto it = edge_counts_by_pair.find(vertex_pair_key(e));
                if (it != edge_counts_by_pair.end() && it->second >= edge_freq_threshold) {
                    freq_filtered_edges.push_back(e);
                }
            }
            if (!freq_filtered_edges.empty()) {
                edges_to_render = &freq_filtered_edges;
            } else {
                effective_mode = EdgeDisplayMode::Union;  // Fallback if no edges match
            }
        } else if (!ts.frequent_edges.empty()) {
            // Fall back to pre-computed frequent edges if no per-state data
            edges_to_render = &ts.frequent_edges;
        } else {
            effective_mode = EdgeDisplayMode::Union;  // Fallback
        }
    }

    // Apply path selection filter if provided (combined with timeslicing if enabled)
    // Collect edges from selected states across the entire timeslice range
    std::vector<Edge> path_filtered_edges;
    bool has_path_selection = all_selected_states && !all_selected_states->empty();
    if (has_path_selection) {
        std::unordered_set<uint64_t> path_edge_set;
        // Iterate over all timesteps in the timeslice range (or just current if timeslice disabled)
        for (int t = slice_start; t <= slice_end; ++t) {
            if (t >= static_cast<int>(all_selected_states->size()) ||
                t >= static_cast<int>(analysis.states_per_step.size())) continue;
            const auto& selected_for_step = (*all_selected_states)[t];
            for (int state_idx : selected_for_step) {
                if (state_idx >= 0 && state_idx < static_cast<int>(analysis.states_per_step[t].size())) {
                    const auto& state = analysis.states_per_step[t][state_idx];
                    for (const auto& e : state.edges) {
                        VertexId v1 = std::min(e.v1, e.v2);
                        VertexId v2 = std::max(e.v1, e.v2);
                        path_edge_set.insert((static_cast<uint64_t>(v1) << 32) | v2);
                    }
                }
            }
        }
        // Filter edges_to_render to only those in the selected states
        for (const auto& e : *edges_to_render) {
            VertexId v1 = std::min(e.v1, e.v2);
            VertexId v2 = std::max(e.v1, e.v2);
            uint64_t key = (static_cast<uint64_t>(v1) << 32) | v2;
            if (path_edge_set.count(key) > 0) {
                path_filtered_edges.push_back(e);
            }
        }
        if (!path_filtered_edges.empty()) {
            edges_to_render = &path_filtered_edges;
        }
    }

    // In non-union modes OR path selection mode, only show vertices that have edges
    // This prevents showing "orphan" vertices not connected to any visible edge
    bool filter_edgeless = (effective_mode != EdgeDisplayMode::Union) || has_path_selection;
    std::unordered_set<VertexId> vertices_with_edges;
    if (filter_edgeless) {
        for (const auto& e : *edges_to_render) {
            vertices_with_edges.insert(e.v1);
            vertices_with_edges.insert(e.v2);
        }
    }

    // Build sphere instances for vertices
    data.sphere_instances.reserve(vertices_to_render.size());
    for (size_t i = 0; i < vertices_to_render.size(); ++i) {
        // Skip hidden vertices (missing data mode)
        if (vertex_hidden[i]) {
            continue;
        }

        // In filtered modes, skip vertices without edges
        if (filter_edgeless &&
            vertices_with_edges.find(vertices_to_render[i]) == vertices_with_edges.end()) {
            continue;
        }

        SphereInstance inst;
        inst.x = vertex_x[i];
        inst.y = vertex_y[i];
        inst.z = vertex_z[i];
        inst.radius = vertex_radius;
        inst.r = vertex_colors[i].x;
        inst.g = vertex_colors[i].y;
        inst.b = vertex_colors[i].z;
        inst.a = vertex_colors[i].w;
        data.sphere_instances.push_back(inst);
    }

    // Build cylinder instances for edges
    data.cylinder_instances.reserve(edges_to_render->size());
    for (const auto& edge : *edges_to_render) {
        auto it1 = vertex_to_idx.find(edge.v1);
        auto it2 = vertex_to_idx.find(edge.v2);
        if (it1 == vertex_to_idx.end() || it2 == vertex_to_idx.end()) continue;

        size_t idx1 = it1->second;
        size_t idx2 = it2->second;

        // Skip edges where either endpoint is hidden (missing data mode)
        if (vertex_hidden[idx1] || vertex_hidden[idx2]) continue;

        // Determine edge color based on mode
        Vec4 c1, c2;
        if (edge_color_mode == EdgeColorMode::Frequency && !edge_counts_by_pair.empty()) {
            // Color by frequency (uniform color for whole edge)
            auto it = edge_counts_by_pair.find(vertex_pair_key(edge));
            int count = (it != edge_counts_by_pair.end()) ? it->second : min_edge_count;
            // Scale from min to max (not 1 to max!) to use full color range
            int edge_freq_range = max_edge_count - min_edge_count;
            float t = (edge_freq_range > 0)
                ? static_cast<float>(count - min_edge_count) / static_cast<float>(edge_freq_range)
                : 1.0f;
            Vec3 freq_color = apply_palette(t, palette);
            c1 = c2 = Vec4(freq_color, 1.0f);
        } else {
            // Color by vertex dimension (gradient between endpoints)
            c1 = vertex_colors[idx1];
            c2 = vertex_colors[idx2];
        }

        CylinderInstance inst;
        inst.start_x = vertex_x[idx1];
        inst.start_y = vertex_y[idx1];
        inst.start_z = vertex_z[idx1];
        inst.end_x = vertex_x[idx2];
        inst.end_y = vertex_y[idx2];
        inst.end_z = vertex_z[idx2];
        inst.radius = edge_radius;
        inst._pad1 = 0;
        inst.r1 = c1.x;
        inst.g1 = c1.y;
        inst.b1 = c1.z;
        inst.a1 = 0.8f;
        inst.r2 = c2.x;
        inst.g2 = c2.y;
        inst.b2 = c2.z;
        inst.a2 = 0.8f;
        data.cylinder_instances.push_back(inst);
    }

    // Also build legacy line data for 2D mode fallback
    if (view_mode == ViewMode::View2D) {
        data.edge_data.reserve(edges_to_render->size() * 2);
        for (const auto& edge : *edges_to_render) {
            auto it1 = vertex_to_idx.find(edge.v1);
            auto it2 = vertex_to_idx.find(edge.v2);
            if (it1 == vertex_to_idx.end() || it2 == vertex_to_idx.end()) continue;

            size_t idx1 = it1->second;
            size_t idx2 = it2->second;

            // Use same coloring logic as 3D mode
            Vec4 c1, c2;
            if (edge_color_mode == EdgeColorMode::Frequency && !edge_counts_by_pair.empty()) {
                auto it = edge_counts_by_pair.find(vertex_pair_key(edge));
                int count = (it != edge_counts_by_pair.end()) ? it->second : min_edge_count;
                // Scale from min to max (not 1 to max!) to use full color range
                int edge_freq_range = max_edge_count - min_edge_count;
                float t = (edge_freq_range > 0)
                    ? static_cast<float>(count - min_edge_count) / static_cast<float>(edge_freq_range)
                    : 1.0f;
                Vec3 freq_color = apply_palette(t, palette);
                c1 = c2 = Vec4(freq_color, 0.8f);
            } else {
                c1 = vertex_colors[idx1];
                c2 = vertex_colors[idx2];
                c1.w = 0.8f;
                c2.w = 0.8f;
            }

            data.edge_data.push_back({vertex_x[idx1], vertex_y[idx1], 0, c1.x, c1.y, c1.z, c1.w});
            data.edge_data.push_back({vertex_x[idx2], vertex_y[idx2], 0, c2.x, c2.y, c2.z, c2.w});
        }
        data.edge_count = data.edge_data.size();

        // Build quads for 2D vertices
        // In non-union modes, only show vertices that have edges (same as 3D mode)
        data.vertex_data.reserve(ts.union_vertices.size() * 6);
        float s = vertex_radius;
        for (size_t i = 0; i < ts.union_vertices.size(); ++i) {
            // In filtered modes (non-union or path selection), skip vertices without edges
            if (filter_edgeless &&
                vertices_with_edges.find(ts.union_vertices[i]) == vertices_with_edges.end()) {
                continue;
            }
            float x = vertex_x[i], y = vertex_y[i];
            Vec4 c = vertex_colors[i];
            data.vertex_data.push_back({x - s, y - s, 0, c.x, c.y, c.z, c.w});
            data.vertex_data.push_back({x + s, y - s, 0, c.x, c.y, c.z, c.w});
            data.vertex_data.push_back({x + s, y + s, 0, c.x, c.y, c.z, c.w});
            data.vertex_data.push_back({x - s, y - s, 0, c.x, c.y, c.z, c.w});
            data.vertex_data.push_back({x + s, y + s, 0, c.x, c.y, c.z, c.w});
            data.vertex_data.push_back({x - s, y + s, 0, c.x, c.y, c.z, c.w});
        }
        data.vertex_count = data.vertex_data.size();
    }

    // Store frequency data for legend (when in Frequency color mode)
    if (edge_color_mode == EdgeColorMode::Frequency) {
        int overall_min = std::min(min_edge_count, min_vertex_count);
        int overall_max = std::max(max_edge_count, max_vertex_count);
        data.min_freq = overall_min;
        data.max_freq = overall_max;
        data.has_freq_data = true;
    }

    // Render geodesic paths as bright colored lines
    if (show_geodesics && analysis.has_geodesic_analysis &&
        timestep < static_cast<int>(analysis.geodesic_paths.size())) {
        const auto& paths = analysis.geodesic_paths[timestep];

        // Bright cyan for geodesic paths (highly visible against dimension heatmap)
        const Vec4 geodesic_color(0.0f, 1.0f, 1.0f, 1.0f);  // Cyan
        const float geodesic_radius = edge_radius * 2.0f;  // Thicker than regular edges

        for (size_t path_idx = 0; path_idx < paths.size(); ++path_idx) {
            const auto& path = paths[path_idx];
            if (path.size() < 2) continue;

            // Vary color slightly by path index for bundle visualization
            float hue_shift = static_cast<float>(path_idx) / std::max(1.0f, static_cast<float>(paths.size()));
            Vec4 path_color = geodesic_color;
            // Shift from cyan towards magenta based on path index
            path_color.x = hue_shift * 0.8f;  // Add red component

            for (size_t i = 0; i + 1 < path.size(); ++i) {
                VertexId v1 = path[i];
                VertexId v2 = path[i + 1];

                auto it1 = vertex_to_idx.find(v1);
                auto it2 = vertex_to_idx.find(v2);
                if (it1 == vertex_to_idx.end() || it2 == vertex_to_idx.end()) continue;

                size_t idx1 = it1->second;
                size_t idx2 = it2->second;

                // Progress along path for color gradient
                float progress = static_cast<float>(i) / static_cast<float>(path.size() - 1);

                // 3D cylinder instance
                CylinderInstance inst;
                inst.start_x = vertex_x[idx1];
                inst.start_y = vertex_y[idx1];
                inst.start_z = vertex_z[idx1] + 0.01f;  // Slightly above graph edges
                inst.end_x = vertex_x[idx2];
                inst.end_y = vertex_y[idx2];
                inst.end_z = vertex_z[idx2] + 0.01f;
                inst.radius = geodesic_radius;
                inst.r1 = path_color.x;
                inst.g1 = path_color.y * (1.0f - progress * 0.3f);  // Fade slightly along path
                inst.b1 = path_color.z;
                inst.a1 = 1.0f;
                inst.r2 = path_color.x;
                inst.g2 = path_color.y * (1.0f - (progress + 0.1f) * 0.3f);
                inst.b2 = path_color.z;
                inst.a2 = 1.0f;
                data.cylinder_instances.push_back(inst);

                // 2D line data
                if (view_mode == ViewMode::View2D) {
                    data.edge_data.push_back({vertex_x[idx1], vertex_y[idx1], 0.1f,
                                              path_color.x, path_color.y, path_color.z, 1.0f});
                    data.edge_data.push_back({vertex_x[idx2], vertex_y[idx2], 0.1f,
                                              path_color.x, path_color.y, path_color.z, 1.0f});
                }
            }
        }

        // Update edge count for 2D mode
        if (view_mode == ViewMode::View2D) {
            data.edge_count = data.edge_data.size();
        }
    }

    // Render topological defect markers (enlarged, colored vertices at defect locations)
    // Defect types: 0=None, 1=K5, 2=K33, 3=HighDegree, 4=DimSpike
    if (show_defects && analysis.has_particle_analysis &&
        timestep < static_cast<int>(analysis.detected_defects.size())) {
        const auto& defects = analysis.detected_defects[timestep];

        const float defect_radius = vertex_radius * 2.5f;  // Much larger than regular vertices

        for (const auto& defect : defects) {
            // Color by defect type (0=None, 1=K5, 2=K33, 3=HighDegree, 4=DimSpike)
            Vec4 defect_color;
            switch (defect.type) {
                case 1:  // K5
                    defect_color = Vec4(1.0f, 0.2f, 0.2f, 1.0f);  // Red for K5
                    break;
                case 2:  // K33
                    defect_color = Vec4(0.2f, 0.2f, 1.0f, 1.0f);  // Blue for K3,3
                    break;
                case 3:  // HighDegree
                    defect_color = Vec4(1.0f, 0.5f, 0.0f, 1.0f);  // Orange for high degree
                    break;
                case 4:  // DimensionSpike
                    defect_color = Vec4(1.0f, 1.0f, 0.2f, 1.0f);  // Yellow for dimension spike
                    break;
                default:
                    defect_color = Vec4(1.0f, 0.0f, 1.0f, 1.0f);  // Magenta for unknown
                    break;
            }

            // Mark all core vertices of this defect
            for (VertexId v : defect.core_vertices) {
                auto it = vertex_to_idx.find(v);
                if (it == vertex_to_idx.end()) continue;

                size_t idx = it->second;

                // 3D sphere (add as additional sphere with larger radius)
                SphereInstance inst;
                inst.x = vertex_x[idx];
                inst.y = vertex_y[idx];
                inst.z = vertex_z[idx] + 0.02f;  // Slightly above
                inst.radius = defect_radius;
                inst.r = defect_color.x;
                inst.g = defect_color.y;
                inst.b = defect_color.z;
                inst.a = 0.9f;
                data.sphere_instances.push_back(inst);
            }
        }
    }

    // Apply distance-based highlight coloring (works for both Ball and Tube modes)
    // Colors vertices/edges by distance shells (like ResourceFunction HighlightedGraph)
    if (show_highlight && highlight_distances && !highlight_distances->empty()) {

        // Dim vertices not in highlight, color those in highlight by distance shell
        std::vector<SphereInstance> new_spheres;
        new_spheres.reserve(data.sphere_instances.size());

        size_t sphere_idx = 0;
        for (size_t i = 0; i < vertices_to_render.size(); ++i) {
            VertexId vid = vertices_to_render[i];
            if (vertex_hidden[i]) continue;
            if (filter_edgeless && vertices_with_edges.find(vid) == vertices_with_edges.end()) continue;

            if (sphere_idx < data.sphere_instances.size()) {
                SphereInstance inst = data.sphere_instances[sphere_idx];

                auto dist_it = highlight_distances->find(vid);
                if (dist_it != highlight_distances->end()) {
                    // In highlight: color by distance shell
                    auto [r, g, b] = shell_color(dist_it->second);
                    inst.r = r;
                    inst.g = g;
                    inst.b = b;
                    inst.a = 0.9f;
                } else {
                    // Not in highlight: dim significantly
                    inst.a = non_highlight_alpha;
                }

                // Highlight source vertex (white, large)
                if (vid == highlight_source) {
                    inst.radius *= 2.0f;
                    inst.r = 1.0f; inst.g = 1.0f; inst.b = 1.0f; inst.a = 1.0f;
                }

                // Highlight target vertex if specified (magenta, large) - Tube mode only
                if (highlight_target != 0 && vid == highlight_target) {
                    inst.radius *= 2.0f;
                    inst.r = 1.0f; inst.g = 0.0f; inst.b = 1.0f; inst.a = 1.0f;
                }

                new_spheres.push_back(inst);
                ++sphere_idx;
            }
        }

        for (size_t extra_idx = sphere_idx; extra_idx < data.sphere_instances.size(); ++extra_idx) {
            new_spheres.push_back(data.sphere_instances[extra_idx]);
        }
        data.sphere_instances = std::move(new_spheres);

        // COLOR EDGES by distance shell
        std::vector<CylinderInstance> new_cylinders;
        new_cylinders.reserve(data.cylinder_instances.size());

        for (const auto& cyl : data.cylinder_instances) {
            CylinderInstance new_cyl = cyl;

            // Find vertices at cylinder endpoints (approximate by position matching)
            VertexId v1 = 0, v2 = 0;
            float best_dist1 = 1e9f, best_dist2 = 1e9f;
            for (size_t i = 0; i < vertices_to_render.size(); ++i) {
                auto it = vertex_to_idx.find(vertices_to_render[i]);
                if (it == vertex_to_idx.end()) continue;
                size_t idx = it->second;
                float dx1 = vertex_x[idx] - cyl.start_x;
                float dy1 = vertex_y[idx] - cyl.start_y;
                float dz1 = vertex_z[idx] - cyl.start_z;
                float d1 = dx1*dx1 + dy1*dy1 + dz1*dz1;
                if (d1 < best_dist1) { best_dist1 = d1; v1 = vertices_to_render[i]; }

                float dx2 = vertex_x[idx] - cyl.end_x;
                float dy2 = vertex_y[idx] - cyl.end_y;
                float dz2 = vertex_z[idx] - cyl.end_z;
                float d2 = dx2*dx2 + dy2*dy2 + dz2*dz2;
                if (d2 < best_dist2) { best_dist2 = d2; v2 = vertices_to_render[i]; }
            }

            // Get distances for both endpoints
            auto it1 = highlight_distances->find(v1);
            auto it2 = highlight_distances->find(v2);

            if (it1 != highlight_distances->end() && it2 != highlight_distances->end()) {
                // Both endpoints in highlight - color by max distance
                int edge_dist = std::max(it1->second, it2->second);
                auto [r, g, b] = shell_color(edge_dist);
                new_cyl.r1 = new_cyl.r2 = r;
                new_cyl.g1 = new_cyl.g2 = g;
                new_cyl.b1 = new_cyl.b2 = b;
                new_cyl.a1 = new_cyl.a2 = 1.0f;
                new_cyl.radius *= (edge_dist == 0) ? 2.5f : 1.5f;  // Thicker for center/path
            } else {
                // Edge not in highlight - dim it
                new_cyl.a1 = new_cyl.a2 = non_highlight_alpha;
            }

            new_cylinders.push_back(new_cyl);
        }
        data.cylinder_instances = std::move(new_cylinders);
    }

    return data;
}

// Kelly's maximum contrast colors (for branch coloring)
static Vec4 kelly_color(size_t index) {
    static const Vec4 colors[] = {
        {0.902f, 0.624f, 0.000f, 1.0f},  // Vivid Yellow
        {0.502f, 0.243f, 0.459f, 1.0f},  // Strong Purple
        {1.000f, 0.478f, 0.361f, 1.0f},  // Vivid Orange
        {0.663f, 0.800f, 0.929f, 1.0f},  // Very Light Blue
        {0.690f, 0.090f, 0.122f, 1.0f},  // Vivid Red
        {0.776f, 0.667f, 0.055f, 1.0f},  // Grayish Yellow
        {0.502f, 0.502f, 0.502f, 1.0f},  // Medium Gray
        {0.000f, 0.502f, 0.333f, 1.0f},  // Vivid Green
        {0.961f, 0.412f, 0.557f, 1.0f},  // Strong Purplish Pink
        {0.000f, 0.329f, 0.651f, 1.0f},  // Strong Blue
        {1.000f, 0.671f, 0.569f, 1.0f},  // Strong Yellowish Pink
        {0.416f, 0.239f, 0.604f, 1.0f},  // Strong Violet
        {1.000f, 0.612f, 0.416f, 1.0f},  // Vivid Orange Yellow
        {0.651f, 0.329f, 0.329f, 1.0f},  // Strong Purplish Red
        {0.847f, 0.843f, 0.000f, 1.0f},  // Vivid Greenish Yellow
        {0.529f, 0.306f, 0.000f, 1.0f},  // Strong Reddish Brown
    };
    return colors[index % 16];
}

// Build render data for shape space (curvature) visualization
RenderData build_shape_space_render_data(
    const BHAnalysisResult& analysis,
    int timestep,
    ViewMode view_mode,
    ShapeSpaceColorMode color_mode,
    ShapeSpaceDisplayMode display_mode,
    int highlighted_branch,
    bool use_ollivier = false,  // false = Wolfram-Ricci, true = Ollivier-Ricci
    ColorPalette palette = ColorPalette::Temperature  // Color palette for curvature
) {
    RenderData data;

    // Select which alignment to use
    const auto& alignment_data = use_ollivier ? analysis.alignment_ollivier : analysis.alignment_per_timestep;
    bool has_alignment = use_ollivier ? analysis.has_ollivier_alignment : analysis.has_branch_alignment;

    // Check if alignment data is available
    if (!has_alignment) {
        std::cerr << "[shape_space] No " << (use_ollivier ? "Ollivier-Ricci" : "Wolfram-Ricci")
                  << " alignment data" << std::endl;
        return data;
    }
    if (timestep < 0 || timestep >= static_cast<int>(alignment_data.size())) {
        std::cerr << "[shape_space] Invalid timestep: " << timestep
                  << " (size=" << alignment_data.size() << ")" << std::endl;
        return data;
    }

    const auto& agg = alignment_data[timestep];
    if (agg.total_points == 0) {
        std::cerr << "[shape_space] No points at timestep " << timestep << std::endl;
        return data;
    }

    // Get bounds for the selected curvature method
    float pc1_min = use_ollivier ? analysis.ollivier_pc1_min : analysis.global_pc1_min;
    float pc1_max = use_ollivier ? analysis.ollivier_pc1_max : analysis.global_pc1_max;
    float pc2_min = use_ollivier ? analysis.ollivier_pc2_min : analysis.global_pc2_min;
    float pc2_max = use_ollivier ? analysis.ollivier_pc2_max : analysis.global_pc2_max;
    float pc3_min = use_ollivier ? analysis.ollivier_pc3_min : analysis.global_pc3_min;
    float pc3_max = use_ollivier ? analysis.ollivier_pc3_max : analysis.global_pc3_max;
    float curv_abs = use_ollivier ? analysis.ollivier_curvature_abs_max : analysis.curvature_abs_max;

    // Compute adaptive radius based on point density
    float pc_range = std::max({pc1_max - pc1_min, pc2_max - pc2_min, pc3_max - pc3_min});
    float vertex_radius = pc_range / (std::sqrt(static_cast<float>(agg.total_points)) * 4.0f);
    vertex_radius = std::clamp(vertex_radius, 0.002f, 0.05f);

    // Debug: print once per timestep change
    static int last_debug_timestep = -1;
    static bool last_use_ollivier = false;
    if (timestep != last_debug_timestep || use_ollivier != last_use_ollivier) {
        std::cerr << "[shape_space] Rendering timestep " << timestep
                  << " (" << (use_ollivier ? "Ollivier" : "Wolfram") << ")"
                  << ": " << agg.total_points << " points, radius=" << vertex_radius << std::endl;
        last_debug_timestep = timestep;
        last_use_ollivier = use_ollivier;
    }
    if (curv_abs < 1e-6f) curv_abs = 1.0f;  // Avoid division by zero

    // Reserve space for spheres
    data.sphere_instances.reserve(agg.total_points);

    // Process each point
    for (size_t i = 0; i < agg.total_points; ++i) {
        SphereInstance sphere;

        // Position = PC coordinates
        sphere.x = agg.all_pc1[i];
        sphere.y = agg.all_pc2[i];
        sphere.z = (view_mode == ViewMode::ShapeSpace3D) ? agg.all_pc3[i] : 0.0f;
        sphere.radius = vertex_radius;

        // Determine color
        Vec4 color;
        if (color_mode == ShapeSpaceColorMode::Curvature) {
            // Diverging colormap using selected palette
            color = curvature_to_color(agg.all_curvature[i], curv_abs, palette);
        } else {
            // Categorical color by branch ID
            color = kelly_color(agg.branch_id[i]);
        }

        // Dim non-highlighted branches in PerBranch mode
        if (display_mode == ShapeSpaceDisplayMode::PerBranch &&
            highlighted_branch >= 0 &&
            static_cast<int>(agg.branch_id[i]) != highlighted_branch) {
            color.w = 0.15f;  // Very transparent
        }

        sphere.r = color.x;
        sphere.g = color.y;
        sphere.b = color.z;
        sphere.a = color.w;

        data.sphere_instances.push_back(sphere);
    }

    return data;
}

// Build timeline bar (2D overlay)
// Returns vertices for: background bar + position marker + control buttons
// Uses screen coordinates: x in [-1, 1], y at bottom
struct TimelineData {
    std::vector<Vertex> panel_verts;    // Semi-transparent background panels (alpha-blended)
    std::vector<Vertex> bar_verts;      // Background bar (triangles)
    std::vector<Vertex> marker_verts;   // Position marker (triangles)
    std::vector<Vertex> button_verts;   // Control buttons (triangles)

    // Hit regions in NDC for mouse interaction
    float scrubber_left, scrubber_right, scrubber_bottom, scrubber_top;
    float rewind_left, rewind_right;    // Rewind button bounds
    float play_left, play_right;        // Play/pause button bounds
    float skip_end_left, skip_end_right; // Skip to end button bounds
};

// Helper to add a triangle (3 vertices)
static void add_triangle(std::vector<Vertex>& verts,
    float x1, float y1, float x2, float y2, float x3, float y3,
    float z, const Vec4& color) {
    verts.push_back({x1, y1, z, color.x, color.y, color.z, color.w});
    verts.push_back({x2, y2, z, color.x, color.y, color.z, color.w});
    verts.push_back({x3, y3, z, color.x, color.y, color.z, color.w});
}

// Helper to add a quad (2 triangles, 6 vertices)
static void add_quad(std::vector<Vertex>& verts,
    float left, float bottom, float right, float top,
    float z, const Vec4& color) {
    verts.push_back({left, bottom, z, color.x, color.y, color.z, color.w});
    verts.push_back({right, bottom, z, color.x, color.y, color.z, color.w});
    verts.push_back({right, top, z, color.x, color.y, color.z, color.w});
    verts.push_back({left, bottom, z, color.x, color.y, color.z, color.w});
    verts.push_back({right, top, z, color.x, color.y, color.z, color.w});
    verts.push_back({left, top, z, color.x, color.y, color.z, color.w});
}

// Helper to add a filled circle using triangle fan (center + perimeter vertices)
// aspect_ratio = width/height of viewport (for circular circles in NDC space)
static void add_circle(std::vector<Vertex>& verts,
    float cx, float cy, float radius, float z, const Vec4& color,
    float aspect_ratio = 1.0f, int segments = 12) {
    // In NDC space, adjust radius for aspect ratio
    float rx = radius;  // X radius
    float ry = radius * aspect_ratio;  // Y radius scaled for circular appearance

    // Create triangles from center to each edge segment
    for (int i = 0; i < segments; ++i) {
        float angle1 = 2.0f * PI * i / segments;
        float angle2 = 2.0f * PI * (i + 1) / segments;

        float x1 = cx + rx * std::cos(angle1);
        float y1 = cy + ry * std::sin(angle1);
        float x2 = cx + rx * std::cos(angle2);
        float y2 = cy + ry * std::sin(angle2);

        // Triangle: center, point1, point2
        verts.push_back({cx, cy, z, color.x, color.y, color.z, color.w});
        verts.push_back({x1, y1, z, color.x, color.y, color.z, color.w});
        verts.push_back({x2, y2, z, color.x, color.y, color.z, color.w});
    }
}

// Helper to add a rounded rectangle using triangle fans for corners
// corner_segments controls roundness (4-8 is usually enough)
// aspect_ratio = width/height of viewport (needed for circular corners in NDC space)
static void add_rounded_rect(std::vector<Vertex>& verts,
    float left, float top, float right, float bottom,
    float radius, float z, const Vec4& color, float aspect_ratio = 1.0f, int corner_segments = 6) {
    // In NDC space, X and Y both range from -1 to 1, but the viewport is typically wider
    // To get circular corners, we need to scale the X component of the radius
    float radius_x = radius;        // X radius in NDC
    float radius_y = radius * aspect_ratio;  // Y radius scaled by aspect ratio

    // Clamp radii to half the smaller dimension
    float max_radius_x = (right - left) / 2;
    float max_radius_y = (bottom - top) / 2;
    radius_x = std::min(radius_x, max_radius_x);
    radius_y = std::min(radius_y, max_radius_y);

    // Corner centers
    float tl_cx = left + radius_x, tl_cy = top + radius_y;      // Top-left
    float tr_cx = right - radius_x, tr_cy = top + radius_y;     // Top-right
    float bl_cx = left + radius_x, bl_cy = bottom - radius_y;   // Bottom-left
    float br_cx = right - radius_x, br_cy = bottom - radius_y;  // Bottom-right

    // Center rectangle (between all corners)
    add_quad(verts, tl_cx, tl_cy, br_cx, br_cy, z, color);

    // Top edge (between TL and TR corners)
    add_quad(verts, tl_cx, top, tr_cx, tl_cy, z, color);
    // Bottom edge (between BL and BR corners)
    add_quad(verts, bl_cx, bl_cy, br_cx, bottom, z, color);
    // Left edge (between TL and BL corners)
    add_quad(verts, left, tl_cy, tl_cx, bl_cy, z, color);
    // Right edge (between TR and BR corners)
    add_quad(verts, tr_cx, tr_cy, right, br_cy, z, color);

    // Corner triangle fans (elliptical to appear circular after aspect ratio)
    float angle_step = (PI / 2.0f) / corner_segments;

    // Top-left corner (angles from PI to PI/2)
    // Note: In Vulkan NDC, +Y is DOWN, so we SUBTRACT sin to go UP toward the corner
    for (int i = 0; i < corner_segments; i++) {
        float a1 = PI - i * angle_step;
        float a2 = PI - (i + 1) * angle_step;
        float x1 = tl_cx + radius_x * std::cos(a1);
        float y1 = tl_cy - radius_y * std::sin(a1);  // Subtract for Vulkan Y-down
        float x2 = tl_cx + radius_x * std::cos(a2);
        float y2 = tl_cy - radius_y * std::sin(a2);  // Subtract for Vulkan Y-down
        add_triangle(verts, tl_cx, tl_cy, x1, y1, x2, y2, z, color);
    }

    // Top-right corner (angles from PI/2 to 0)
    for (int i = 0; i < corner_segments; i++) {
        float a1 = PI / 2.0f - i * angle_step;
        float a2 = PI / 2.0f - (i + 1) * angle_step;
        float x1 = tr_cx + radius_x * std::cos(a1);
        float y1 = tr_cy - radius_y * std::sin(a1);  // Subtract for Vulkan Y-down
        float x2 = tr_cx + radius_x * std::cos(a2);
        float y2 = tr_cy - radius_y * std::sin(a2);  // Subtract for Vulkan Y-down
        add_triangle(verts, tr_cx, tr_cy, x1, y1, x2, y2, z, color);
    }

    // Bottom-left corner (angles from -PI to -PI/2)
    for (int i = 0; i < corner_segments; i++) {
        float a1 = -PI + i * angle_step;
        float a2 = -PI + (i + 1) * angle_step;
        float x1 = bl_cx + radius_x * std::cos(a1);
        float y1 = bl_cy - radius_y * std::sin(a1);  // Subtract for Vulkan Y-down
        float x2 = bl_cx + radius_x * std::cos(a2);
        float y2 = bl_cy - radius_y * std::sin(a2);  // Subtract for Vulkan Y-down
        add_triangle(verts, bl_cx, bl_cy, x1, y1, x2, y2, z, color);
    }

    // Bottom-right corner (angles from 0 to -PI/2)
    for (int i = 0; i < corner_segments; i++) {
        float a1 = -i * angle_step;
        float a2 = -(i + 1) * angle_step;
        float x1 = br_cx + radius_x * std::cos(a1);
        float y1 = br_cy - radius_y * std::sin(a1);  // Subtract for Vulkan Y-down
        float x2 = br_cx + radius_x * std::cos(a2);
        float y2 = br_cy - radius_y * std::sin(a2);  // Subtract for Vulkan Y-down
        add_triangle(verts, br_cx, br_cy, x1, y1, x2, y2, z, color);
    }
}

TimelineData build_timeline_bar(int current_step, int max_step, float aspect_ratio,
                                 bool is_playing, int playback_direction,
                                 bool timeslice_enabled = false, int timeslice_width = 5) {
    TimelineData data;

    // Layout constants
    float button_size = 0.04f;
    float button_margin = 0.02f;
    float viewport_padding = 0.03f;  // Padding from viewport bottom
    // Z values near 0 to render in front of everything (Vulkan NDC: 0=near, 1=far)
    float panel_z = 0.002f;     // Panels furthest back
    float bar_z = 0.001f;
    float button_z = 0.0005f;
    float marker_z = 0.0001f;

    // Panel styling
    Vec4 panel_color{0.0f, 0.0f, 0.0f, 0.5f};  // Semi-transparent black
    float panel_padding = 0.025f;  // Padding inside panel around content (increased)
    float corner_radius = 0.015f;  // Rounded corner radius (slightly larger)

    // Button positions (left side): rewind | play | skip_end
    float buttons_left = -0.95f;
    float rewind_center = buttons_left + button_size/2;
    float play_center = rewind_center + button_size + button_margin;
    float skip_end_center = play_center + button_size + button_margin;

    // Scrubber bar (to the right of buttons)
    // Vulkan coordinates: y=-1 at top of screen, y=+1 at bottom of screen
    // Bar should be at bottom of screen, so use positive Y values (with padding)
    // Add extra gap to prevent panel overlap (panel_padding on each side + small gap)
    float bar_left = skip_end_center + button_size/2 + panel_padding * 2 + 0.02f;
    float bar_right = 0.95f;
    float bar_y_min = 0.90f - viewport_padding;   // Top edge of bar (smaller Y = higher on screen)
    float bar_y_max = 0.95f - viewport_padding;   // Bottom edge of bar (with padding from bottom)
    float bar_center_y = (bar_y_min + bar_y_max) / 2;

    // Store hit regions (in Vulkan NDC)
    // Extend hit area slightly beyond bar to include the marker at edges (0 and t_max)
    float marker_half_width = 0.008f;  // Same as marker rendering
    data.scrubber_left = bar_left - marker_half_width;
    data.scrubber_right = bar_right + marker_half_width;
    data.scrubber_bottom = bar_y_min - 0.02f;  // Slightly larger hit area (extend upward)
    data.scrubber_top = bar_y_max + 0.02f;     // Extend downward
    data.rewind_left = rewind_center - button_size/2;
    data.rewind_right = rewind_center + button_size/2;
    data.play_left = play_center - button_size/2;
    data.play_right = play_center + button_size/2;
    data.skip_end_left = skip_end_center - button_size/2;
    data.skip_end_right = skip_end_center + button_size/2;

    // === ROUNDED PANEL BACKGROUNDS ===

    // Panel behind all three buttons (one continuous panel)
    float buttons_panel_left = rewind_center - button_size/2 - panel_padding;
    float buttons_panel_right = skip_end_center + button_size/2 + panel_padding;
    float buttons_panel_top = bar_center_y - button_size/2 - panel_padding;
    float buttons_panel_bottom = bar_center_y + button_size/2 + panel_padding;
    add_rounded_rect(data.panel_verts, buttons_panel_left, buttons_panel_top,
                     buttons_panel_right, buttons_panel_bottom, corner_radius, panel_z, panel_color, aspect_ratio);

    // Panel behind scrubber/timeline
    float scrubber_panel_left = bar_left - panel_padding;
    float scrubber_panel_right = bar_right + panel_padding;
    float scrubber_panel_top = bar_y_min - panel_padding;
    float scrubber_panel_bottom = bar_y_max + panel_padding;
    // If timeslice is enabled, extend panel to include the indicator
    if (timeslice_enabled && timeslice_width > 1) {
        scrubber_panel_bottom = bar_y_max + 0.025f + panel_padding;
    }
    add_rounded_rect(data.panel_verts, scrubber_panel_left, scrubber_panel_top,
                     scrubber_panel_right, scrubber_panel_bottom, corner_radius, panel_z, panel_color, aspect_ratio);

    // Background bar (dark gray) - now on top of panel
    Vec4 bg_color{0.2f, 0.2f, 0.2f, 0.8f};
    add_quad(data.bar_verts, bar_left, bar_y_min, bar_right, bar_y_max, bar_z, bg_color);

    // Progress fill (dim cyan)
    float t = (max_step > 0) ? static_cast<float>(current_step) / max_step : 0.0f;
    float progress_right = bar_left + t * (bar_right - bar_left);
    Vec4 progress_color{0.0f, 0.4f, 0.5f, 0.8f};
    add_quad(data.bar_verts, bar_left, bar_y_min, progress_right, bar_y_max, bar_z - 0.01f, progress_color);

    // Position marker (white vertical line)
    float marker_x = progress_right;
    // marker_half_width already defined above for hit regions
    float marker_y_min = bar_y_min - 0.015f;  // Extend above bar
    float marker_y_max = bar_y_max + 0.015f;  // Extend below bar
    Vec4 marker_color{1.0f, 1.0f, 1.0f, 1.0f};
    add_quad(data.marker_verts, marker_x - marker_half_width, marker_y_min,
             marker_x + marker_half_width, marker_y_max, marker_z, marker_color);

    // Timeslice indicator (red bar under the scrubber showing the time window)
    // Only show when timeslice is enabled AND width > 1 (width=1 is same as no timeslice)
    if (timeslice_enabled && max_step > 0 && timeslice_width > 1) {
        // Calculate the timeslice range centered on current step
        int half_width = timeslice_width / 2;
        int slice_start = std::max(0, current_step - half_width);
        int slice_end = std::min(max_step, current_step + half_width);

        // Convert to bar coordinates
        float bar_width = bar_right - bar_left;
        float slice_left_x = bar_left + (static_cast<float>(slice_start) / max_step) * bar_width;
        float slice_right_x = bar_left + (static_cast<float>(slice_end) / max_step) * bar_width;

        // Ensure minimum visual width so the bar is always visible
        float min_visual_width = 0.02f;  // Minimum width in NDC
        float actual_width = slice_right_x - slice_left_x;
        if (actual_width < min_visual_width) {
            float center = (slice_left_x + slice_right_x) / 2.0f;
            slice_left_x = center - min_visual_width / 2.0f;
            slice_right_x = center + min_visual_width / 2.0f;
        }

        // Draw red bar slightly below the main bar
        float slice_y_min = bar_y_max + 0.005f;
        float slice_y_max = bar_y_max + 0.020f;
        Vec4 slice_color{0.9f, 0.2f, 0.2f, 0.9f};  // Red
        add_quad(data.marker_verts, slice_left_x, slice_y_min,
                 slice_right_x, slice_y_max, marker_z, slice_color);
    }

    // Rewind button (double left-pointing triangles)
    Vec4 btn_color{0.7f, 0.7f, 0.7f, 1.0f};
    float tri_h = button_size * 0.6f;
    float tri_w = button_size * 0.4f;
    // First triangle (left)
    add_triangle(data.button_verts,
        rewind_center - tri_w, bar_center_y,  // left point
        rewind_center, bar_center_y - tri_h/2,  // top right
        rewind_center, bar_center_y + tri_h/2,  // bottom right
        button_z, btn_color);
    // Second triangle (right, slightly overlapping)
    add_triangle(data.button_verts,
        rewind_center, bar_center_y,
        rewind_center + tri_w, bar_center_y - tri_h/2,
        rewind_center + tri_w, bar_center_y + tri_h/2,
        button_z, btn_color);

    // Play/Pause button
    if (is_playing) {
        // Pause icon: two vertical bars
        float pause_w = button_size * 0.12f;
        float pause_gap = button_size * 0.1f;
        float pause_h = button_size * 0.5f;
        add_quad(data.button_verts,
            play_center - pause_gap - pause_w, bar_center_y - pause_h/2,
            play_center - pause_gap, bar_center_y + pause_h/2,
            button_z, btn_color);
        add_quad(data.button_verts,
            play_center + pause_gap, bar_center_y - pause_h/2,
            play_center + pause_gap + pause_w, bar_center_y + pause_h/2,
            button_z, btn_color);
    } else {
        // Play icon: right-pointing triangle
        Vec4 play_color = (playback_direction > 0) ? btn_color : Vec4{0.5f, 0.7f, 0.9f, 1.0f};
        add_triangle(data.button_verts,
            play_center - tri_w/2, bar_center_y - tri_h/2,  // top left
            play_center - tri_w/2, bar_center_y + tri_h/2,  // bottom left
            play_center + tri_w/2, bar_center_y,            // right point
            button_z, play_color);
    }

    // Skip to end button (double right-pointing triangles - mirror of rewind)
    // First triangle (left)
    add_triangle(data.button_verts,
        skip_end_center - tri_w, bar_center_y - tri_h/2,  // top left
        skip_end_center - tri_w, bar_center_y + tri_h/2,  // bottom left
        skip_end_center, bar_center_y,                     // right point
        button_z, btn_color);
    // Second triangle (right)
    add_triangle(data.button_verts,
        skip_end_center, bar_center_y - tri_h/2,          // top left
        skip_end_center, bar_center_y + tri_h/2,          // bottom left
        skip_end_center + tri_w, bar_center_y,            // right point
        button_z, btn_color);

    return data;
}

// =============================================================================
// Scatter Plot Overlay
// =============================================================================
// Displays time series scatter plot: each point = one state's mean value
// X-axis = timestep, Y-axis = metric value (dimension or curvature)
// Uses ValueType enum for metric selection (same as vertex coloring)

struct ScatterPlotData {
    std::vector<Vertex> axis_verts;     // Axis lines and tick marks
    std::vector<Vertex> line_verts;     // Mean line, variance bands
    std::vector<Vertex> point_verts;    // Individual scatter points (one per state)
    std::vector<Vertex> marker_verts;   // Current step marker

    // Plot bounds for text positioning
    float plot_left, plot_right, plot_top, plot_bottom;
    float value_min, value_max;
    int max_step;
};

// Build scatter plot overlay showing per-state values across timesteps
ScatterPlotData build_scatter_plot(
    const BHAnalysisResult& analysis,
    ValueType metric,
    int current_step,
    float aspect_ratio
) {
    ScatterPlotData data;
    int max_step = static_cast<int>(analysis.per_timestep.size()) - 1;
    if (max_step < 0 || analysis.state_aggregates.empty()) return data;

    data.max_step = max_step;

    // =========================================================================
    // Layout: Large centered plot (80% of screen)
    // =========================================================================
    float margin = 0.08f;           // Margin from screen edges
    float axis_margin = 0.06f;      // Space for axis labels

    float plot_left = -1.0f + margin + axis_margin;
    float plot_right = 1.0f - margin;
    float plot_top = -1.0f + margin + axis_margin;  // Vulkan Y: -1 = top
    float plot_bottom = 1.0f - margin - 0.15f;      // Leave space for timeline

    data.plot_left = plot_left;
    data.plot_right = plot_right;
    data.plot_top = plot_top;
    data.plot_bottom = plot_bottom;

    // Z-ordering (lower = closer to camera in Vulkan)
    float z_axis = 0.002f;
    float z_band = 0.003f;
    float z_mean = 0.0015f;
    float z_point = 0.001f;
    float z_marker = 0.0005f;

    // =========================================================================
    // Collect per-state values grouped by timestep
    // Each state_aggregate contains the mean value for that state
    // =========================================================================
    std::vector<std::vector<float>> values_per_step(max_step + 1);
    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();

    for (const auto& agg : analysis.state_aggregates) {
        if (agg.step > static_cast<uint32_t>(max_step)) continue;

        float value = 0;
        switch (metric) {
            case ValueType::WolframHausdorffDimension:
                value = agg.mean_dimension;
                break;
            case ValueType::OllivierRicciCurvature:
                value = agg.mean_ollivier_ricci;
                break;
            case ValueType::WolframRicciScalar:
                value = agg.mean_wolfram_scalar;
                break;
            case ValueType::WolframRicciTensor:
                value = agg.mean_wolfram_ricci;
                break;
            case ValueType::DimensionGradient:
                value = agg.mean_dim_gradient;
                break;
            default:
                value = agg.mean_dimension;
                break;
        }

        // For dimension, skip zeros; for curvature, include all values
        bool valid = (metric == ValueType::WolframHausdorffDimension) ? (value > 0) : true;
        if (valid) {
            values_per_step[agg.step].push_back(value);
            global_min = std::min(global_min, value);
            global_max = std::max(global_max, value);
        }
    }

    // Handle edge case of no data or constant values
    if (global_max <= global_min) {
        if (metric == ValueType::WolframHausdorffDimension) {
            global_min = 1.0f;
            global_max = 3.0f;  // Typical dimension range
        } else {
            global_min = -1.0f;
            global_max = 1.0f;  // Typical curvature range
        }
    }

    // Add small padding to range
    float range = global_max - global_min;
    global_min -= range * 0.05f;
    global_max += range * 0.05f;

    data.value_min = global_min;
    data.value_max = global_max;

    // =========================================================================
    // Coordinate mapping
    // =========================================================================
    auto map_x = [&](float step) {
        float t = (max_step > 0) ? step / max_step : 0.5f;
        return plot_left + t * (plot_right - plot_left);
    };
    auto map_y = [&](float val) {
        float t = (val - global_min) / (global_max - global_min);
        return plot_bottom - t * (plot_bottom - plot_top);  // Invert Y for Vulkan
    };

    // =========================================================================
    // Draw axes (white lines)
    // =========================================================================
    Vec4 axis_color{1.0f, 1.0f, 1.0f, 0.8f};
    float axis_thickness = 0.003f;

    // X-axis (bottom)
    add_quad(data.axis_verts, plot_left, plot_bottom - axis_thickness,
             plot_right, plot_bottom + axis_thickness, z_axis, axis_color);

    // Y-axis (left)
    add_quad(data.axis_verts, plot_left - axis_thickness, plot_top,
             plot_left + axis_thickness, plot_bottom, z_axis, axis_color);

    // X-axis tick marks (every 10 steps or so)
    Vec4 tick_color{0.6f, 0.6f, 0.6f, 0.6f};
    float tick_len = 0.015f;
    int x_tick_interval = std::max(1, max_step / 10);
    for (int t = 0; t <= max_step; t += x_tick_interval) {
        float x = map_x(static_cast<float>(t));
        add_quad(data.axis_verts, x - 0.002f, plot_bottom,
                 x + 0.002f, plot_bottom + tick_len, z_axis, tick_color);
    }

    // Y-axis tick marks (5 ticks)
    int num_y_ticks = 5;
    for (int i = 0; i <= num_y_ticks; ++i) {
        float val = global_min + (global_max - global_min) * i / num_y_ticks;
        float y = map_y(val);
        add_quad(data.axis_verts, plot_left - tick_len, y - 0.002f * aspect_ratio,
                 plot_left, y + 0.002f * aspect_ratio, z_axis, tick_color);

        // Faint horizontal grid line
        Vec4 grid_color{0.3f, 0.3f, 0.3f, 0.3f};
        add_quad(data.axis_verts, plot_left, y - 0.001f * aspect_ratio,
                 plot_right, y + 0.001f * aspect_ratio, z_axis + 0.001f, grid_color);
    }

    // =========================================================================
    // Compute per-timestep statistics for mean line and variance bands
    // =========================================================================
    std::vector<float> means(max_step + 1, 0);
    std::vector<float> stddevs(max_step + 1, 0);

    for (int t = 0; t <= max_step; ++t) {
        const auto& vals = values_per_step[t];
        if (vals.empty()) continue;

        // Mean
        float sum = 0;
        for (float v : vals) sum += v;
        float mean = sum / vals.size();
        means[t] = mean;

        // Standard deviation
        float var_sum = 0;
        for (float v : vals) var_sum += (v - mean) * (v - mean);
        stddevs[t] = std::sqrt(var_sum / vals.size());
    }

    // =========================================================================
    // Draw variance band (±1 std dev, semi-transparent)
    // =========================================================================
    Vec4 band_color{0.3f, 0.5f, 0.8f, 0.2f};
    for (int t = 0; t < max_step; ++t) {
        if (values_per_step[t].empty() || values_per_step[t + 1].empty()) continue;

        float x1 = map_x(static_cast<float>(t));
        float x2 = map_x(static_cast<float>(t + 1));

        float y_low1 = map_y(means[t] - stddevs[t]);
        float y_high1 = map_y(means[t] + stddevs[t]);
        float y_low2 = map_y(means[t + 1] - stddevs[t + 1]);
        float y_high2 = map_y(means[t + 1] + stddevs[t + 1]);

        // Two triangles for the band quad
        add_triangle(data.line_verts, x1, y_low1, x2, y_low2, x2, y_high2, z_band, band_color);
        add_triangle(data.line_verts, x1, y_low1, x2, y_high2, x1, y_high1, z_band, band_color);
    }

    // =========================================================================
    // Draw individual scatter points (one per state)
    // =========================================================================
    Vec4 point_color{0.2f, 0.8f, 1.0f, 0.7f};  // Cyan-ish
    float point_radius = 0.003f;

    for (int t = 0; t <= max_step; ++t) {
        float x = map_x(static_cast<float>(t));
        for (float val : values_per_step[t]) {
            float y = map_y(val);
            // Draw small filled circle
            add_circle(data.point_verts, x, y, point_radius, z_point, point_color, aspect_ratio, 8);
        }
    }

    // =========================================================================
    // Draw mean line (yellow/orange, thicker)
    // =========================================================================
    Vec4 mean_color{1.0f, 0.8f, 0.2f, 1.0f};
    float line_thickness = 0.004f;

    for (int t = 0; t < max_step; ++t) {
        if (values_per_step[t].empty() || values_per_step[t + 1].empty()) continue;

        float x1 = map_x(static_cast<float>(t));
        float x2 = map_x(static_cast<float>(t + 1));
        float y1 = map_y(means[t]);
        float y2 = map_y(means[t + 1]);

        // Thick line as quad
        float dx = x2 - x1, dy = y2 - y1;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len > 0) {
            float nx = -dy / len * line_thickness;
            float ny = dx / len * line_thickness * aspect_ratio;
            add_triangle(data.line_verts, x1 + nx, y1 + ny, x2 + nx, y2 + ny, x2 - nx, y2 - ny, z_mean, mean_color);
            add_triangle(data.line_verts, x1 + nx, y1 + ny, x2 - nx, y2 - ny, x1 - nx, y1 - ny, z_mean, mean_color);
        }
    }

    // =========================================================================
    // Draw current step marker (vertical white line)
    // =========================================================================
    Vec4 marker_color{1.0f, 1.0f, 1.0f, 0.9f};
    float marker_x = map_x(static_cast<float>(current_step));
    float marker_half_width = 0.004f;
    add_quad(data.marker_verts, marker_x - marker_half_width, plot_top,
             marker_x + marker_half_width, plot_bottom, z_marker, marker_color);

    return data;
}

// Build color palette legend (vertical gradient bar in top right)
struct LegendData {
    std::vector<Vertex> verts;  // Triangles for gradient
    float label_x;              // X position for labels (right of bar)
    float max_y;                // Y position for max label (top)
    float min_y;                // Y position for min label (bottom)
    float title_x, title_y;     // Position for palette name
    float bar_center_x;         // Center X of gradient bar (for centering text)
};

LegendData build_legend(float aspect_ratio, ColorPalette palette, float value_min, float value_max) {
    LegendData data;

    // Legend position (top right, with padding)
    // NDC: x=-1 left, x=+1 right, y=-1 top, y=+1 bottom (Vulkan)
    float padding = 0.03f;
    float bar_width = 0.025f;
    float bar_height = 0.35f;
    float right_edge = 0.95f;
    float top_edge = -0.85f;  // Near top (negative Y in Vulkan NDC)

    float bar_right = right_edge;
    float bar_left = bar_right - bar_width;
    float bar_top = top_edge;
    float bar_bottom = bar_top + bar_height;

    // Background (dark semi-transparent)
    float bg_padding = 0.01f;
    Vec4 bg_color{0.1f, 0.1f, 0.1f, 0.7f};
    add_quad(data.verts,
        bar_left - bg_padding, bar_top - bg_padding,
        bar_right + bg_padding, bar_bottom + bg_padding,
        0.002f, bg_color);

    // Gradient bar - draw as horizontal strips with interpolated colors
    int num_strips = 32;
    float strip_height = bar_height / num_strips;
    for (int i = 0; i < num_strips; ++i) {
        float t_top = 1.0f - static_cast<float>(i) / num_strips;      // Top = 1.0 (max)
        float t_bot = 1.0f - static_cast<float>(i + 1) / num_strips;  // Bottom = 0.0 (min)
        Vec3 c_top = apply_palette(t_top, palette);
        Vec3 c_bot = apply_palette(t_bot, palette);
        float y_top = bar_top + i * strip_height;
        float y_bot = bar_top + (i + 1) * strip_height;

        // Two triangles with different colors at top and bottom
        // First triangle: top-left, bottom-left, bottom-right
        data.verts.push_back({bar_left, y_top, 0.001f, c_top.x, c_top.y, c_top.z, 1.0f});
        data.verts.push_back({bar_left, y_bot, 0.001f, c_bot.x, c_bot.y, c_bot.z, 1.0f});
        data.verts.push_back({bar_right, y_bot, 0.001f, c_bot.x, c_bot.y, c_bot.z, 1.0f});
        // Second triangle: top-left, bottom-right, top-right
        data.verts.push_back({bar_left, y_top, 0.001f, c_top.x, c_top.y, c_top.z, 1.0f});
        data.verts.push_back({bar_right, y_bot, 0.001f, c_bot.x, c_bot.y, c_bot.z, 1.0f});
        data.verts.push_back({bar_right, y_top, 0.001f, c_top.x, c_top.y, c_top.z, 1.0f});
    }

    // Calculate label positions (for text rendering, need pixel coords not NDC)
    // These will be converted to pixel coords by the caller
    data.label_x = bar_left - bg_padding * 4;  // To the left of bar
    data.max_y = bar_top;
    data.min_y = bar_bottom;
    data.title_x = bar_left;
    data.title_y = bar_top - 0.025f;  // Above the bar
    data.bar_center_x = (bar_left + bar_right) / 2.0f;  // Center of gradient bar

    return data;
}

// Build histogram panel for vertex selection
struct HistogramData {
    std::vector<Vertex> verts;  // Background + bars
    // Label positions in NDC for text rendering
    float title_x, title_y;
    float stats_x, stats_y;
    float axis_min_x, axis_min_y;
    float axis_max_x, axis_max_y;
    // Hit test bounds
    float panel_left, panel_right, panel_top, panel_bottom;
};

HistogramData build_histogram_panel(
    const VertexSelectionState& selection,
    float click_ndc_x, float click_ndc_y,
    float aspect_ratio,
    int screen_width, int screen_height,
    ColorPalette palette,
    GraphScope graph_scope,
    float global_dim_min, float global_dim_max
) {
    HistogramData data;

    if (!selection.has_selection || selection.bin_counts.empty()) {
        return data;
    }

    // Panel dimensions in NDC
    float panel_w = 0.40f;
    float panel_h = 0.32f;

    // Position panel near click, with offset so it doesn't cover the clicked point
    float offset_x = 0.05f;
    float offset_y = -0.05f;

    float panel_left = click_ndc_x + offset_x;
    float panel_top = click_ndc_y + offset_y;

    // Clamp to screen bounds (flip if necessary)
    if (panel_left + panel_w > 0.98f) {
        panel_left = click_ndc_x - offset_x - panel_w;  // Flip to left
    }
    if (panel_left < -0.98f) {
        panel_left = -0.98f;
    }
    if (panel_top + panel_h > 0.98f) {
        panel_top = 0.98f - panel_h;  // Move up
    }
    if (panel_top < -0.98f) {
        panel_top = -0.98f;
    }

    float panel_right = panel_left + panel_w;
    float panel_bottom = panel_top + panel_h;

    data.panel_left = panel_left;
    data.panel_right = panel_right;
    data.panel_top = panel_top;
    data.panel_bottom = panel_bottom;

    // Background (dark semi-transparent)
    Vec4 bg_color{0.08f, 0.08f, 0.12f, 0.92f};
    float corner_radius = 0.012f;
    add_rounded_rect(data.verts, panel_left, panel_top, panel_right, panel_bottom,
                     corner_radius, 0.002f, bg_color);

    // Layout inside panel
    float padding = 0.02f;
    float inner_left = panel_left + padding;
    float inner_right = panel_right - padding;
    float inner_top = panel_top + padding;
    float inner_bottom = panel_bottom - padding;

    // Title area at top
    float title_height = 0.035f;
    data.title_x = inner_left;
    data.title_y = inner_top + 0.01f;

    // Stats area below title
    float stats_y = inner_top + title_height + 0.01f;
    data.stats_x = inner_left;
    data.stats_y = stats_y;

    // Bar chart area
    float chart_top = stats_y + 0.03f;
    float chart_bottom = inner_bottom - 0.025f;  // Room for axis labels
    float chart_left = inner_left + 0.01f;
    float chart_right = inner_right - 0.01f;
    float chart_height = chart_bottom - chart_top;
    float chart_width = chart_right - chart_left;

    // Axis label positions
    data.axis_min_x = chart_left;
    data.axis_min_y = chart_bottom + 0.008f;
    data.axis_max_x = chart_right - 0.05f;
    data.axis_max_y = chart_bottom + 0.008f;

    // Draw bars
    int num_bins = selection.num_bins;
    if (num_bins <= 0) return data;

    float bar_width = chart_width / num_bins;
    float bar_gap = bar_width * 0.1f;
    float actual_bar_width = bar_width - bar_gap;

    // Normalize bar heights
    float max_count = static_cast<float>(selection.max_bin_count);
    if (max_count < 1.0f) max_count = 1.0f;

    for (int i = 0; i < num_bins; ++i) {
        float bar_left_x = chart_left + i * bar_width + bar_gap / 2;
        float bar_right_x = bar_left_x + actual_bar_width;

        // Bar height proportional to count
        float norm_height = selection.bin_counts[i] / max_count;
        float bar_height = norm_height * chart_height;
        float bar_top_y = chart_bottom - bar_height;
        float bar_bottom_y = chart_bottom;

        // Color based on bin center value using palette
        float bin_center = selection.bin_min + (i + 0.5f) * selection.bin_width;
        // Normalize to global range for consistent coloring
        float norm_value = 0.5f;
        if (global_dim_max > global_dim_min) {
            norm_value = (bin_center - global_dim_min) / (global_dim_max - global_dim_min);
            norm_value = std::max(0.0f, std::min(1.0f, norm_value));
        }
        Vec3 c = apply_palette(norm_value, palette);
        Vec4 bar_color{c.x, c.y, c.z, 1.0f};

        // In distribution mode, highlight the bin containing the selected vertex
        bool is_selected_bin = selection.is_distribution_mode && (i == selection.selected_vertex_bin);

        // Skip empty bins (but leave the space)
        if (selection.bin_counts[i] > 0) {
            // Draw bar as two triangles
            float z = 0.001f;
            data.verts.push_back({bar_left_x, bar_top_y, z, bar_color.x, bar_color.y, bar_color.z, bar_color.w});
            data.verts.push_back({bar_left_x, bar_bottom_y, z, bar_color.x, bar_color.y, bar_color.z, bar_color.w});
            data.verts.push_back({bar_right_x, bar_bottom_y, z, bar_color.x, bar_color.y, bar_color.z, bar_color.w});

            data.verts.push_back({bar_left_x, bar_top_y, z, bar_color.x, bar_color.y, bar_color.z, bar_color.w});
            data.verts.push_back({bar_right_x, bar_bottom_y, z, bar_color.x, bar_color.y, bar_color.z, bar_color.w});
            data.verts.push_back({bar_right_x, bar_top_y, z, bar_color.x, bar_color.y, bar_color.z, bar_color.w});

            // Draw highlight border for selected vertex's bin in distribution mode
            if (is_selected_bin) {
                Vec4 highlight_color{1.0f, 1.0f, 1.0f, 1.0f};
                float bw = 0.003f;  // Border width
                float z2 = 0.0005f;
                // Top border
                add_quad(data.verts, bar_left_x, bar_top_y - bw, bar_right_x, bar_top_y, z2, highlight_color);
                // Left border
                add_quad(data.verts, bar_left_x - bw, bar_top_y - bw, bar_left_x, bar_bottom_y, z2, highlight_color);
                // Right border
                add_quad(data.verts, bar_right_x, bar_top_y - bw, bar_right_x + bw, bar_bottom_y, z2, highlight_color);
                // Bottom border
                add_quad(data.verts, bar_left_x - bw, bar_bottom_y, bar_right_x + bw, bar_bottom_y + bw, z2, highlight_color);
            }
        }
    }

    // Draw chart border (thin outline)
    Vec4 border_color{0.4f, 0.4f, 0.5f, 0.8f};
    float border_w = 0.002f;
    // Top border
    add_quad(data.verts, chart_left, chart_top, chart_right, chart_top + border_w, 0.0005f, border_color);
    // Bottom border
    add_quad(data.verts, chart_left, chart_bottom - border_w, chart_right, chart_bottom, 0.0005f, border_color);
    // Left border
    add_quad(data.verts, chart_left, chart_top, chart_left + border_w, chart_bottom, 0.0005f, border_color);
    // Right border
    add_quad(data.verts, chart_right - border_w, chart_top, chart_right, chart_bottom, 0.0005f, border_color);

    return data;
}

// Build Hilbert space stats panel
struct HilbertPanelData {
    std::vector<Vertex> verts;  // Background
    // Text positions (pixel coords)
    float title_x, title_y;
    std::vector<std::pair<float, float>> stat_positions;  // (x, y) for each stat line
    std::vector<std::string> stat_labels;
    std::vector<std::string> stat_values;
};

HilbertPanelData build_hilbert_panel(
    const HilbertSpaceAnalysis& result,
    const BranchialAnalysisResult& branchial,
    int screen_width, int screen_height
) {
    HilbertPanelData data;

    if (result.num_states == 0) {
        return data;
    }

    // Panel position (top-right corner)
    float panel_w = 0.35f;
    float panel_h = 0.28f;

    float panel_right = 0.98f;
    float panel_left = panel_right - panel_w;
    float panel_top = -0.98f;  // Top of screen
    float panel_bottom = panel_top + panel_h;

    // Background (dark semi-transparent)
    Vec4 bg_color{0.08f, 0.08f, 0.15f, 0.92f};
    float corner_radius = 0.012f;
    add_rounded_rect(data.verts, panel_left, panel_top, panel_right, panel_bottom,
                     corner_radius, 0.002f, bg_color);

    // Text positions (convert NDC to pixels)
    auto ndc_to_pixel_x = [&](float ndc) { return (ndc + 1.0f) * 0.5f * screen_width; };
    auto ndc_to_pixel_y = [&](float ndc) { return (ndc + 1.0f) * 0.5f * screen_height; };

    float padding = 0.02f;
    float inner_left = panel_left + padding;
    float inner_top = panel_top + padding;

    data.title_x = ndc_to_pixel_x(inner_left);
    data.title_y = ndc_to_pixel_y(inner_top);

    // Stats to display
    float line_h = 0.028f;
    float y = inner_top + 0.04f;

    auto add_stat = [&](const std::string& label, const std::string& value) {
        data.stat_positions.push_back({ndc_to_pixel_x(inner_left), ndc_to_pixel_y(y)});
        data.stat_labels.push_back(label);
        data.stat_values.push_back(value);
        y += line_h;
    };

    std::ostringstream oss;

    oss.str(""); oss << result.num_states;
    add_stat("States:", oss.str());

    oss.str(""); oss << result.num_vertices;
    add_stat("Vertices:", oss.str());

    oss.str(""); oss << std::fixed << std::setprecision(3) << result.mean_inner_product;
    add_stat("Mean <psi|phi>:", oss.str());

    oss.str(""); oss << std::fixed << std::setprecision(3) << result.max_inner_product;
    add_stat("Max <psi|phi>:", oss.str());

    oss.str(""); oss << std::fixed << std::setprecision(3) << result.mean_vertex_probability;
    add_stat("Mean P(v):", oss.str());

    oss.str(""); oss << std::fixed << std::setprecision(3) << result.vertex_probability_entropy;
    add_stat("Prob entropy:", oss.str());

    oss.str(""); oss << std::fixed << std::setprecision(3) << branchial.mean_sharpness;
    add_stat("Mean sharpness:", oss.str());

    return data;
}

// Build multiway states graph (left panel showing states at each timestep)
struct StatesGraphData {
    std::vector<Vertex> verts;       // Triangles for background, states, highlights
    std::vector<std::pair<float, float>> row_centers;  // NDC (x, y) for each row center (for labels)
    float panel_right;               // Right edge of panel in NDC
    // State positions: [step][state_index] -> (x, y) in NDC
    std::vector<std::vector<std::pair<float, float>>> state_positions;
};

StatesGraphData build_states_graph(
    const BHAnalysisResult& analysis,
    int current_step,
    int max_step,
    bool timeslice_enabled,
    int timeslice_width,
    float scrubber_position,  // 0-1 for interpolating between steps
    bool path_selection_enabled,
    const std::vector<std::vector<int>>& selected_state_indices,  // Per-step selected states
    float aspect_ratio = 1.0f  // Viewport aspect ratio for circular shapes
) {
    StatesGraphData data;

    if (analysis.states_per_step.empty() || max_step <= 0) {
        return data;
    }

    // Panel layout (left side of screen)
    // NDC: x=-1 left, x=+1 right, y=-1 top, y=+1 bottom (Vulkan)
    float panel_left = -0.98f;
    float panel_right = -0.70f;  // Wider panel for edges
    float panel_top = -0.85f;      // Leave room for legend at top
    float panel_bottom = 0.85f;    // Leave room for timeline at bottom
    float panel_width = panel_right - panel_left;
    float panel_height = panel_bottom - panel_top;

    data.panel_right = panel_right;

    // Background panel (always full size)
    Vec4 bg_color{0.05f, 0.05f, 0.1f, 0.85f};
    add_quad(data.verts, panel_left, panel_top, panel_right, panel_bottom, 0.003f, bg_color);

    // Zoom: show ~10 rows at a time for better visibility
    // If timeslice is enabled and wider than 10, zoom out to show entire timeslice
    // If we have fewer than 10 rows, show all
    int num_rows = max_step + 1;
    int base_visible = 10;  // Default zoom level
    int visible_rows = std::min(base_visible, num_rows);

    // If timeslice is enabled, ensure we can see the entire timeslice range
    if (timeslice_enabled && timeslice_width > visible_rows) {
        visible_rows = std::min(timeslice_width + 2, num_rows);  // +2 for context
    }

    float row_height = panel_height / visible_rows;  // Row height based on visible rows, not total
    float row_padding = row_height * 0.1f;
    float usable_row_height = row_height - 2 * row_padding;

    // Calculate viewport offset to center current step
    // viewport_offset is in row units (how many rows to scroll down)
    float center_row = current_step + scrubber_position;
    float half_visible = visible_rows / 2.0f;
    float viewport_offset = center_row - half_visible;
    // Clamp viewport so we don't scroll past ends
    viewport_offset = std::max(0.0f, std::min(viewport_offset, static_cast<float>(num_rows - visible_rows)));
    // If we have fewer rows than visible_rows, no scrolling needed
    if (num_rows <= visible_rows) viewport_offset = 0.0f;

    // Convert viewport_offset to Y coordinate offset
    float y_offset = viewport_offset * row_height;

    // Find max states at any step for normalization
    int max_states = 1;
    for (int step = 0; step <= max_step; ++step) {
        if (step < static_cast<int>(analysis.states_per_step.size())) {
            max_states = std::max(max_states, static_cast<int>(analysis.states_per_step[step].size()));
        }
    }

    // Helper to check if Y position is within visible panel
    auto is_visible = [&](float y) {
        return y >= panel_top - row_height && y <= panel_bottom + row_height;
    };

    // Helper to clamp and draw a quad, clipping to panel bounds
    auto add_clipped_quad = [&](float left, float top, float right, float bottom, float z, Vec4 color) {
        float clipped_top = std::max(top, panel_top);
        float clipped_bottom = std::min(bottom, panel_bottom);
        if (clipped_top < clipped_bottom) {
            add_quad(data.verts, left, clipped_top, right, clipped_bottom, z, color);
        }
    };

    // Draw highlight for current timestep (thin yellow line, ~1.5x row separator thickness)
    {
        float sep_thickness = 0.001f;  // Same as row separator lines
        float highlight_thickness = sep_thickness * 1.5f;
        // Center the highlight line on the current row
        float row_center_y = panel_top + (current_step + scrubber_position + 0.5f) * row_height - y_offset;
        float highlight_top = row_center_y - highlight_thickness / 2;
        float highlight_bottom = row_center_y + highlight_thickness / 2;

        Vec4 highlight_color{1.0f, 0.9f, 0.2f, 0.8f};  // Yellow, brighter since it's thinner
        add_clipped_quad(panel_left, highlight_top, panel_right, highlight_bottom,
                 0.002f, highlight_color);
    }

    // Draw timeslice range highlight if enabled
    if (timeslice_enabled && timeslice_width > 1) {
        int half_width = timeslice_width / 2;
        int slice_start = std::max(0, current_step - half_width);
        int slice_end = std::min(max_step, current_step + half_width);

        float slice_top = panel_top + slice_start * row_height - y_offset;
        float slice_bottom = panel_top + (slice_end + 1) * row_height - y_offset;

        Vec4 slice_color{0.2f, 0.6f, 1.0f, 0.15f};  // Blue, 15% alpha
        add_clipped_quad(panel_left, slice_top, panel_right, slice_bottom, 0.0025f, slice_color);
    }

    // Prepare state positions storage
    data.state_positions.resize(max_step + 1);

    // First pass: compute all state positions
    float dot_radius = std::min(usable_row_height * 0.3f, panel_width * 0.02f);
    float state_margin = panel_width * 0.08f;  // Margin from panel edges

    for (int step = 0; step <= max_step; ++step) {
        int num_states = 0;
        if (step < static_cast<int>(analysis.states_per_step.size())) {
            num_states = static_cast<int>(analysis.states_per_step[step].size());
        }

        float row_y = panel_top + step * row_height + row_height / 2 - y_offset;
        data.row_centers.push_back({panel_left + panel_width / 2, row_y});

        if (num_states == 0) {
            data.state_positions[step].clear();
            continue;
        }

        data.state_positions[step].resize(num_states);

        // Distribute states evenly across the row (with margins)
        float usable_width = panel_width - 2 * state_margin;
        float state_spacing = (num_states > 1) ? usable_width / (num_states - 1) : 0;

        for (int s = 0; s < num_states; ++s) {
            float state_x = panel_left + state_margin + s * state_spacing;
            if (num_states == 1) state_x = panel_left + panel_width / 2;
            data.state_positions[step][s] = {state_x, row_y};
        }
    }

    // Build set of selected states for quick lookup
    std::set<std::pair<int, int>> selected_set;  // (step, state_index)
    if (path_selection_enabled && !selected_state_indices.empty()) {
        for (size_t step = 0; step < selected_state_indices.size(); ++step) {
            for (int idx : selected_state_indices[step]) {
                selected_set.insert({static_cast<int>(step), idx});
            }
        }
    }

    // Second pass: draw edges between adjacent timesteps
    // For each state at step N, draw edges to all states at step N+1
    // (Without parent-child data, we show all possible transitions as faint lines)
    Vec4 edge_color_dim{0.3f, 0.3f, 0.4f, 0.2f};      // Dim gray for all edges
    Vec4 edge_color_selected{0.2f, 0.8f, 1.0f, 0.8f}; // Bright cyan for selected path
    float edge_thickness = 0.001f;  // Thin lines between states

    for (int step = 0; step < max_step; ++step) {
        if (step >= static_cast<int>(data.state_positions.size()) ||
            step + 1 >= static_cast<int>(data.state_positions.size())) continue;

        const auto& from_states = data.state_positions[step];
        const auto& to_states = data.state_positions[step + 1];

        // Skip if both rows are off-screen
        if (!from_states.empty() && !to_states.empty()) {
            float from_y = from_states[0].second;
            float to_y = to_states[0].second;
            if ((from_y < panel_top - row_height && to_y < panel_top - row_height) ||
                (from_y > panel_bottom + row_height && to_y > panel_bottom + row_height)) {
                continue;
            }
        }

        if (from_states.empty() || to_states.empty()) continue;

        // Draw edges: each state connects to "nearby" states in the next step
        // For simplicity, draw edges from each state to states with similar relative position
        for (size_t i = 0; i < from_states.size(); ++i) {
            float from_x = from_states[i].first;
            float from_y = from_states[i].second;

            bool from_selected = selected_set.count({step, static_cast<int>(i)}) > 0;

            // Determine which states in the next step to connect to
            // Simple heuristic: connect to states with similar fractional position ± spread
            float from_frac = (from_states.size() > 1) ? static_cast<float>(i) / (from_states.size() - 1) : 0.5f;

            for (size_t j = 0; j < to_states.size(); ++j) {
                float to_frac = (to_states.size() > 1) ? static_cast<float>(j) / (to_states.size() - 1) : 0.5f;
                float frac_dist = std::abs(to_frac - from_frac);

                // Only draw edges to nearby states (within 30% fractional distance)
                // OR if both states are selected (for path highlighting)
                bool to_selected = selected_set.count({step + 1, static_cast<int>(j)}) > 0;
                bool is_selected_edge = from_selected && to_selected;

                if (frac_dist > 0.35f && !is_selected_edge) continue;

                float to_x = to_states[j].first;
                float to_y = to_states[j].second;

                // Draw edge as a thin quad
                Vec4 edge_col = is_selected_edge ? edge_color_selected : edge_color_dim;
                float z = is_selected_edge ? 0.0018f : 0.0022f;  // Selected edges in front

                // Calculate perpendicular direction for line thickness
                float dx = to_x - from_x;
                float dy = to_y - from_y;
                float len = std::sqrt(dx*dx + dy*dy);
                if (len < 0.001f) continue;

                float nx = -dy / len * edge_thickness;
                float ny = dx / len * edge_thickness;

                // Draw as two triangles
                Vertex v0 = {from_x - nx, from_y - ny, z, edge_col.x, edge_col.y, edge_col.z, edge_col.w};
                Vertex v1 = {from_x + nx, from_y + ny, z, edge_col.x, edge_col.y, edge_col.z, edge_col.w};
                Vertex v2 = {to_x + nx, to_y + ny, z, edge_col.x, edge_col.y, edge_col.z, edge_col.w};
                Vertex v3 = {to_x - nx, to_y - ny, z, edge_col.x, edge_col.y, edge_col.z, edge_col.w};

                data.verts.push_back(v0);
                data.verts.push_back(v1);
                data.verts.push_back(v2);
                data.verts.push_back(v0);
                data.verts.push_back(v2);
                data.verts.push_back(v3);
            }
        }
    }

    // Third pass: draw state dots (on top of edges)
    for (int step = 0; step <= max_step; ++step) {
        if (step >= static_cast<int>(data.state_positions.size())) continue;
        const auto& positions = data.state_positions[step];

        // Skip if row is off-screen
        if (!positions.empty()) {
            float row_y = positions[0].second;
            if (row_y < panel_top - row_height || row_y > panel_bottom + row_height) {
                continue;
            }
        }

        for (size_t s = 0; s < positions.size(); ++s) {
            float state_x = positions[s].first;
            float row_y = positions[s].second;

            bool is_selected = selected_set.count({step, static_cast<int>(s)}) > 0;

            // State dot color
            Vec4 dot_color;
            if (is_selected) {
                dot_color = Vec4{0.2f, 1.0f, 0.8f, 1.0f};  // Bright cyan-green for selected
            } else if (step == current_step) {
                dot_color = Vec4{1.0f, 1.0f, 1.0f, 0.9f};  // White for current step
            } else {
                dot_color = Vec4{0.6f, 0.6f, 0.6f, 0.7f};  // Gray for others
            }

            float r = is_selected ? dot_radius * 1.3f : dot_radius;

            // Draw state as a circle
            add_circle(data.verts, state_x, row_y, r, 0.001f, dot_color, aspect_ratio, 16);
        }
    }

    // Draw row separators (subtle lines) - only for visible rows
    Vec4 sep_color{0.3f, 0.3f, 0.4f, 0.3f};
    float sep_thickness = 0.001f;
    for (int step = 1; step <= max_step; ++step) {
        float sep_y = panel_top + step * row_height - y_offset;
        if (sep_y >= panel_top && sep_y <= panel_bottom) {
            add_quad(data.verts, panel_left, sep_y - sep_thickness/2, panel_right, sep_y + sep_thickness/2,
                     0.0015f, sep_color);
        }
    }

    return data;
}

// Build horizon circles (debug visualization)
std::vector<Vertex> build_horizon_circles(const BHConfig& config, int segments = 64) {
    std::vector<Vertex> verts;

    auto add_circle = [&](Vec2 center, float radius, Vec4 color) {
        for (int i = 0; i < segments; ++i) {
            float a1 = 2.0f * PI * i / segments;
            float a2 = 2.0f * PI * (i + 1) / segments;
            verts.push_back({
                center.x + radius * std::cos(a1),
                center.y + radius * std::sin(a1),
                0,
                color.x, color.y, color.z, color.w
            });
            verts.push_back({
                center.x + radius * std::cos(a2),
                center.y + radius * std::sin(a2),
                0,
                color.x, color.y, color.z, color.w
            });
        }
    };

    // BH1 horizon (positive x)
    Vec2 c1{config.separation / 2.0f, 0};
    float r1 = config.mass1 / 2.0f;
    add_circle(c1, r1, {1, 1, 1, 0.5f});

    // BH2 horizon (negative x)
    Vec2 c2{-config.separation / 2.0f, 0};
    float r2 = config.mass2 / 2.0f;
    add_circle(c2, r2, {1, 1, 1, 0.5f});

    return verts;
}

// =============================================================================
// Main Application
// =============================================================================

void print_usage() {
    std::cout << "Black Hole Hausdorff Dimension Visualisation" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "This tool generates multiway hypergraph evolution from black hole initial" << std::endl;
    std::cout << "conditions and computes Hausdorff dimension estimates for visualization." << std::endl;
    std::cout << std::endl;
    std::cout << "COMMANDS:" << std::endl;
    std::cout << std::endl;
    std::cout << "  run [options]" << std::endl;
    std::cout << "  generate [options]" << std::endl;
    std::cout << "      Full pipeline: generate initial condition, evolve, analyze, save, view." << std::endl;
    std::cout << "      Saves analysis to .bhdata file (default: blackhole_analysis.bhdata)" << std::endl;
    std::cout << "      Use --no-view to skip the viewer." << std::endl;
    std::cout << std::endl;
    std::cout << "  evolve [options]" << std::endl;
    std::cout << "      Generate initial condition, evolve, analyze, and view." << std::endl;
    std::cout << "      Saves .bhevo (evolution) and .bhdata (analysis) files." << std::endl;
    std::cout << "      --no-analyze  Stop after evolution (only save .bhevo)" << std::endl;
    std::cout << "      --no-view     Stop after analysis (save both, skip viewer)" << std::endl;
    std::cout << std::endl;
    std::cout << "  analyze [file.bhevo] [options]" << std::endl;
    std::cout << "      Load cached evolution data and run Hausdorff analysis." << std::endl;
    std::cout << "      Default input: blackhole_evolution.bhevo" << std::endl;
    std::cout << "      Saves analysis to .bhdata file, then optionally launches viewer." << std::endl;
    std::cout << "      Use --no-view to skip the viewer." << std::endl;
    std::cout << std::endl;
    std::cout << "  view [file.bhdata]" << std::endl;
    std::cout << "      Load and view a previously saved analysis file." << std::endl;
    std::cout << "      Default input: blackhole_analysis.bhdata" << std::endl;
    std::cout << std::endl;
    std::cout << "GENERATION OPTIONS (for run, generate, evolve):" << std::endl;
    std::cout << "  --grid WxH        Use grid with holes (e.g., 10x10 or '10 10', default: 30x30)" << std::endl;
    std::cout << "  --solid-grid      Use solid grid without holes (ignores black hole positions)" << std::endl;
    std::cout << "  --brill N         Use Brill-Lindquist sampling with N vertices" << std::endl;
    std::cout << "  --steps N         Max evolution steps (default: 100 if not specified)" << std::endl;
    std::cout << "  --threshold F     Edge connectivity threshold (default: 1.5 brill, 0.8 grid)" << std::endl;
    std::cout << "  --max-states N    Max states per step (default: 0 = unlimited)" << std::endl;
    std::cout << "  --max-children N  Max successors per parent (default: 0 = unlimited)" << std::endl;
    std::cout << "  --rule RULE       Rewrite rule (default: BH 4->4 rule)" << std::endl;
    std::cout << "  --batched         Enable batched matching mode" << std::endl;
    std::cout << "  --uniform-random  Step-synchronized evolution with random match selection" << std::endl;
    std::cout << "  --matches-per-step N  Matches to apply per step in uniform mode (default: 1, 0=all)" << std::endl;
    std::cout << "  --fast-reservoir  Early termination when reservoir is full (faster, less uniform)" << std::endl;
    std::cout << "  --shuffle-edges   Shuffle initial edge order for randomness" << std::endl;
    std::cout << "  --no-canonicalize Disable state canonicalization (all states unique)" << std::endl;
    std::cout << "  --no-explore-dedup Disable explore-from-canonical-only (explore all states)" << std::endl;
    std::cout << "  --no-canonical-orientation  Disable curvature-based canonical orientation (shape space)" << std::endl;
    std::cout << "  --no-scale-normalization    Disable scale normalization (shape space)" << std::endl;
    std::cout << std::endl;
    std::cout << "TOPOLOGY OPTIONS (alternative to --grid/--brill):" << std::endl;
    std::cout << "  --topology TYPE   Topology: flat, cylinder, torus, sphere, klein, mobius" << std::endl;
    std::cout << "  --sampling METHOD Sampling: uniform, poisson, grid, density" << std::endl;
    std::cout << "  --defects N       Number of topological defects (default: 2)" << std::endl;
    std::cout << "  --defect-mass F   Mass/strength of each defect (default: 3.0)" << std::endl;
    std::cout << "  --defect-exclusion F  Exclusion radius around defects (default: 1.5)" << std::endl;
    std::cout << "  --major-radius F  Major radius for curved topologies (default: 10.0)" << std::endl;
    std::cout << "  --minor-radius F  Minor radius for torus (default: 3.0)" << std::endl;
    std::cout << "  --poisson-distance F  Minimum distance for Poisson sampling (default: 1.0)" << std::endl;
    std::cout << "  --embed-3d        Use 3D topological embedding (torus/sphere/cylinder in 3D)" << std::endl;
    std::cout << "  --layout-3d       Use 3D force-directed layout (vs 2D)" << std::endl;
    std::cout << std::endl;
    std::cout << "MINKOWSKI SPRINKLING OPTIONS:" << std::endl;
    std::cout << "  --sprinkling N    Generate N-point Minkowski sprinkling (causal set)" << std::endl;
    std::cout << "  --sprinkling-time T   Time extent (default: 10.0)" << std::endl;
    std::cout << "  --sprinkling-space S  Spatial extent (default: 10.0)" << std::endl;
    std::cout << std::endl;
    std::cout << "BRILL-LINDQUIST BLACK HOLE OPTIONS:" << std::endl;
    std::cout << "  --bh-mass1 M      Mass of first black hole (default: 3.0)" << std::endl;
    std::cout << "  --bh-mass2 M      Mass of second black hole (default: 3.0)" << std::endl;
    std::cout << "  --bh-separation S Distance between black holes (default: 6.0)" << std::endl;
    std::cout << "  --bh-box X1 X2 Y1 Y2  Bounding box (default: -5 5 -4 4)" << std::endl;
    std::cout << std::endl;
    std::cout << "ANALYSIS OPTIONS (for run, generate, analyze):" << std::endl;
    std::cout << "  --max-radius N    Max radius for ball counting (default: 8)" << std::endl;
    std::cout << "  --anchors N       Number of anchor vertices (default: 6)" << std::endl;
    std::cout << "  --geodesics       Enable geodesic analysis (test particle tracing)" << std::endl;
    std::cout << "  --particles       Enable particle detection (Robertson-Seymour defects)" << std::endl;
    std::cout << "  --curvature       Enable curvature analysis (Ollivier-Ricci / dimension gradient)" << std::endl;
    std::cout << "  --entropy         Enable entropy analysis (local entropy, Fisher info)" << std::endl;
    std::cout << "  --hilbert         Enable Hilbert space analysis (branchial structure)" << std::endl;
    std::cout << std::endl;
    std::cout << "OUTPUT OPTIONS:" << std::endl;
    std::cout << "  --output FILE     Output file path (auto-selects extension based on command)" << std::endl;
    std::cout << "  --no-analyze      Skip analysis after evolve (for 'evolve' command)" << std::endl;
    std::cout << "  --no-view         Skip viewer after analysis (for 'evolve' or 'analyze')" << std::endl;
    std::cout << "  --verify          Print verification/debug output (evolution stats per step)" << std::endl;
    std::cout << std::endl;
    std::cout << "VIEWER CONTROLS:" << std::endl;
    std::cout << "  Space        Play/pause timeline" << std::endl;
    std::cout << "  Left/Right   Step backward/forward" << std::endl;
    std::cout << "  Home/End     Jump to start/end" << std::endl;
    std::cout << "  2/3          Toggle 2D/3D view" << std::endl;
    std::cout << "  Z            Toggle Z/depth mapping (flat vs elevated)" << std::endl;
    std::cout << "  < / >        Decrease/increase Z scale" << std::endl;
    std::cout << "  T            Toggle timeslice view" << std::endl;
    std::cout << "  - / =        Decrease/increase timeslice width" << std::endl;
    std::cout << "  H            Toggle horizon circles" << std::endl;
    std::cout << "  J            Toggle geodesic path overlay (requires --geodesics)" << std::endl;
    std::cout << "  K            Toggle defect markers (requires --particles)" << std::endl;
    std::cout << "  D            Cycle curvature mode: OFF/Ollivier-Ricci/Dim Gradient (requires --curvature)" << std::endl;
    std::cout << "  E            Toggle entropy heatmap (requires --entropy)" << std::endl;
    std::cout << "  I            Toggle edge display (union/intersection)" << std::endl;
    std::cout << "  V            Toggle heatmap (mean dimension / variance)" << std::endl;
    std::cout << "  L            Toggle dynamic layout" << std::endl;
    std::cout << "  M            Cycle MSAA antialiasing (OFF -> 2x -> 4x -> ...)" << std::endl;
    std::cout << "  N            Toggle color normalization (per-frame / global)" << std::endl;
    std::cout << "  Mouse drag   Orbit camera (3D) / Pan (2D)" << std::endl;
    std::cout << "  Scroll       Zoom" << std::endl;
    std::cout << "  R            Reset camera" << std::endl;
    std::cout << "  ESC          Exit" << std::endl;
    std::cout << std::endl;
    std::cout << "WORKFLOW EXAMPLES:" << std::endl;
    std::cout << "  # Full pipeline (evolve + analyze + view)" << std::endl;
    std::cout << "  blackhole_viz run --grid 40x40 --steps 50" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Cache evolution for later re-analysis" << std::endl;
    std::cout << "  blackhole_viz evolve --grid 40x40 --steps 50 --output my_evolution.bhevo" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Re-analyze with different parameters (without re-evolving)" << std::endl;
    std::cout << "  blackhole_viz analyze my_evolution.bhevo --max-radius 12 --output analysis_r12.bhdata" << std::endl;
    std::cout << std::endl;
    std::cout << "  # View saved analysis" << std::endl;
    std::cout << "  blackhole_viz view analysis_r12.bhdata" << std::endl;
    std::cout << std::endl;
}

// =============================================================================
// Vertex Selection Helpers
// =============================================================================

// Collect values for histogram display (dimension or curvature)
// For Branchial mode: returns value of selected vertex across all states at timestep
// For Foliation/Global/Curvature: returns values of ALL vertices (distribution view)
// Also returns the selected vertex's value for highlighting
struct HistogramCollectionResult {
    std::vector<float> values;           // All values for histogram
    float selected_vertex_value = -1.0f; // Value of selected vertex (for highlighting)
    bool is_distribution_mode = false;   // True if showing all vertices (Foliation/Global/Curvature)
};

HistogramCollectionResult collect_vertex_values(
    const BHAnalysisResult& analysis,
    VertexId vertex_id,
    int timestep,
    GraphScope graph_scope,
    ValueType value_type
) {
    HistogramCollectionResult result;

    // Curvature mode: show distribution of ALL vertices' curvature values
    if (is_curvature_type(value_type) && analysis.has_curvature_analysis) {
        result.is_distribution_mode = true;

        const std::unordered_map<VertexId, float>* curvature_map = nullptr;
        switch (value_type) {
            case ValueType::WolframRicciScalar:
                curvature_map = &analysis.curvature_wolfram_scalar;
                break;
            case ValueType::WolframRicciTensor:
                curvature_map = &analysis.curvature_wolfram_ricci;
                break;
            case ValueType::OllivierRicciCurvature:
                curvature_map = &analysis.curvature_ollivier_ricci;
                break;
            case ValueType::DimensionGradient:
                curvature_map = &analysis.curvature_dimension_gradient;
                break;
            default:
                break;
        }

        if (curvature_map) {
            for (const auto& [vid, curv] : *curvature_map) {
                if (std::isfinite(curv)) {
                    result.values.push_back(curv);
                    if (vid == vertex_id) {
                        result.selected_vertex_value = curv;
                    }
                }
            }
        }
        return result;
    }

    // Dimension mode: handle per-timestep
    if (timestep < 0 || timestep >= static_cast<int>(analysis.per_timestep.size())) {
        return result;
    }

    const auto& ts = analysis.per_timestep[timestep];

    if (graph_scope == GraphScope::Global) {
        // Global mode: show distribution of ALL vertices' global dimensions
        // Highlight where the selected vertex falls
        result.is_distribution_mode = true;

        for (const auto& [vid, dim] : analysis.mega_dimension) {
            if (dim > 0) {
                result.values.push_back(dim);
                if (vid == vertex_id) {
                    result.selected_vertex_value = dim;
                }
            }
        }
    } else if (graph_scope == GraphScope::Foliation) {
        // Foliation mode: show distribution of ALL vertices' dimensions at this timestep
        result.is_distribution_mode = true;

        for (size_t i = 0; i < ts.union_vertices.size(); ++i) {
            if (i < ts.global_mean_dimensions.size() && ts.global_mean_dimensions[i] > 0) {
                result.values.push_back(ts.global_mean_dimensions[i]);
                if (ts.union_vertices[i] == vertex_id) {
                    result.selected_vertex_value = ts.global_mean_dimensions[i];
                }
            }
        }
    } else {
        // Branchial mode: collect from per_state analyses at this timestep
        // Shows how this vertex's dimension varies across multiway states
        result.is_distribution_mode = false;

        if (!analysis.per_state.empty()) {
            for (const auto& state : analysis.per_state) {
                if (state.step != static_cast<uint32_t>(timestep)) continue;

                // Find vertex in this state
                for (size_t i = 0; i < state.vertices.size(); ++i) {
                    if (state.vertices[i] == vertex_id) {
                        if (i < state.vertex_dimensions.size()) {
                            float dim_val = state.vertex_dimensions[i];
                            if (dim_val > 0) {
                                result.values.push_back(dim_val);
                            }
                        }
                        break;
                    }
                }
            }
        } else {
            // Fallback: use per-timestep averaged dimension (single value)
            for (size_t i = 0; i < ts.union_vertices.size(); ++i) {
                if (ts.union_vertices[i] == vertex_id) {
                    if (i < ts.mean_dimensions.size() && ts.mean_dimensions[i] > 0) {
                        result.values.push_back(ts.mean_dimensions[i]);
                    }
                    break;
                }
            }
        }
    }

    return result;
}

// Compute histogram with fixed number of equally-spaced bins
void compute_histogram(
    const std::vector<float>& values,
    std::vector<int>& bin_counts,
    float& bin_min,
    float& bin_max,
    float& bin_width,
    int& max_count,
    int num_bins = 10
) {
    bin_counts.clear();
    max_count = 0;
    bin_min = 0.0f;
    bin_max = 0.0f;
    bin_width = 0.1f;

    if (values.empty()) return;

    // Find range
    bin_min = *std::min_element(values.begin(), values.end());
    bin_max = *std::max_element(values.begin(), values.end());
    float range = bin_max - bin_min;

    if (range < 0.001f) {
        // All values essentially the same - single bin
        bin_counts.push_back(static_cast<int>(values.size()));
        bin_width = 0.1f;
        max_count = bin_counts[0];
        return;
    }

    // Use fixed number of equally-spaced bins
    num_bins = std::max(1, num_bins);
    bin_width = range / num_bins;
    bin_counts.resize(num_bins, 0);

    // Count values in each bin
    for (float v : values) {
        int bin = static_cast<int>((v - bin_min) / bin_width);
        bin = std::clamp(bin, 0, num_bins - 1);
        bin_counts[bin]++;
        max_count = std::max(max_count, bin_counts[bin]);
    }
}

// Update selection histogram data
void update_selection_histogram(
    VertexSelectionState& selection,
    const BHAnalysisResult& analysis,
    int current_step,
    GraphScope graph_scope,
    ValueType value_type
) {
    // Collect values (dimension or curvature based on value_type)
    auto collection = collect_vertex_values(
        analysis, selection.selected_vertex, current_step, graph_scope, value_type);

    selection.dimension_values = std::move(collection.values);
    selection.is_distribution_mode = collection.is_distribution_mode;
    selection.selected_vertex_value = collection.selected_vertex_value;

    if (!selection.dimension_values.empty()) {
        // Compute statistics
        float sum = 0.0f, sum_sq = 0.0f;
        for (float v : selection.dimension_values) {
            sum += v;
            sum_sq += v * v;
        }
        size_t n = selection.dimension_values.size();
        selection.mean_value = sum / n;
        float variance = sum_sq / n - selection.mean_value * selection.mean_value;
        selection.std_dev = std::sqrt(std::max(0.0f, variance));

        // Compute histogram with user-specified bin count
        compute_histogram(
            selection.dimension_values,
            selection.bin_counts,
            selection.bin_min,
            selection.bin_max,
            selection.bin_width,
            selection.max_bin_count,
            selection.histogram_bin_count
        );
        selection.num_bins = static_cast<int>(selection.bin_counts.size());

        // Find which bin the selected vertex falls into (for highlighting)
        if (selection.is_distribution_mode && selection.selected_vertex_value > 0 &&
            selection.bin_width > 0.0001f) {
            int bin = static_cast<int>((selection.selected_vertex_value - selection.bin_min) / selection.bin_width);
            selection.selected_vertex_bin = std::clamp(bin, 0, selection.num_bins - 1);
        } else {
            selection.selected_vertex_bin = -1;
        }
    } else {
        selection.bin_counts.clear();
        selection.num_bins = 0;
        selection.mean_value = 0.0f;
        selection.std_dev = 0.0f;
        selection.selected_vertex_bin = -1;
    }

    selection.cached_graph_scope = graph_scope;
    selection.cached_value_type = value_type;
    selection.cached_timestep = current_step;
    selection.cached_bin_count = selection.histogram_bin_count;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 0;
    }

    // Check for help flag
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h" || arg == "help") {
            print_usage();
            return 0;
        }
    }

    // Install signal handler for Ctrl-C
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
#endif

    // Parse command line
    std::string mode;  // Must be explicitly set
    std::string data_file;       // Input file for view/analyze
    std::string output_file;     // Output file (default depends on mode)
    // Black hole rule: 4 edges chain -> 4 edges with fresh vertex
    std::string rule = "{{x,y},{y,z},{z,w},{w,v}}->{{y,u},{u,v},{w,x},{x,u}}";
    int grid_width = 30, grid_height = 30;
    int brill_vertices = 0;
    int steps = 100;
    float edge_threshold = -1.0f;  // -1 means use default
    int max_states = 0;     // 0 = unlimited (no pruning)
    int max_children = 0;   // 0 = unlimited (no pruning)
    int max_radius = 8;        // Analysis: max radius for ball counting
    int num_anchors = 6;       // Analysis: number of anchor vertices
    bool no_analyze = false;   // Skip analysis after evolve
    bool no_view = false;      // Skip viewer after analyze
    bool verify = false;       // Enable verification/debug output
    bool batched = false;      // Enable batched matching mode
    bool shuffle_edges = false;    // Shuffle initial edges
    bool no_canonicalize = false;  // Disable state canonicalization
    bool no_explore_dedup = false; // Disable explore-from-canonical-only
    bool canonical_orientation = true;  // Canonical orientation using signed curvature moment
    bool scale_normalization = true;    // Scale by 1/sqrt(eigenvalue) for unit variance
    bool uniform_random = false;   // Step-synchronized uniform random mode
    int matches_per_step = 1;      // Matches to apply per step in uniform mode
    bool fast_reservoir = false;   // Early termination when reservoir full
    bool solid_grid = false;       // Use solid grid (no holes)

    // New topology options
    TopologyType topology = TopologyType::Flat;
    SamplingMethod sampling = SamplingMethod::PoissonDisk;
    int defect_count = 2;          // Default: 2 defects (like current BH behavior)
    float defect_mass = 3.0f;      // Default mass for defects
    float defect_exclusion = 1.5f; // Default exclusion radius
    float major_radius = 10.0f;    // Topology major radius
    float minor_radius = 3.0f;     // Topology minor radius (for torus)
    float poisson_distance = 1.0f; // Poisson disk minimum distance
    bool use_new_topology = false; // Flag to use new topology system
    bool embed_3d = false;         // Use 3D topological embedding for initial positions
    bool layout_3d = false;        // Use 3D force-directed layout
    bool enable_geodesics = false; // Enable geodesic analysis (test particles)
    bool enable_particles = false; // Enable particle detection (Robertson-Seymour)
    bool enable_curvature = false; // Enable curvature analysis (Ollivier-Ricci / dimension gradient)
    bool enable_branch_alignment = false; // Enable branch alignment (curvature shape space)
    bool enable_entropy = false;   // Enable entropy analysis
    bool enable_hilbert = false;   // Enable Hilbert space analysis

    // Minkowski sprinkling options
    bool use_sprinkling = false;      // Use sprinkling instead of grid/brill
    int sprinkling_n = 500;           // Number of spacetime points
    float sprinkling_time = 10.0f;    // Time extent
    float sprinkling_space = 10.0f;   // Spatial extent

    // Brill-Lindquist black hole configuration
    float bh_mass1 = 3.0f;            // Mass of first black hole
    float bh_mass2 = 3.0f;            // Mass of second black hole
    float bh_separation = 6.0f;       // Separation between black holes
    float bh_box_xmin = -5.0f;        // Box x minimum
    float bh_box_xmax = 5.0f;         // Box x maximum
    float bh_box_ymin = -4.0f;        // Box y minimum
    float bh_box_ymax = 4.0f;         // Box y maximum

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "view") {
            mode = "view";
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                data_file = argv[++i];
            }
        } else if (arg == "analyze") {
            mode = "analyze";
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                data_file = argv[++i];
            }
        } else if (arg == "run") {
            mode = "run";
        } else if (arg == "generate") {
            mode = "generate";
        } else if (arg == "evolve") {
            mode = "evolve";
        } else if (arg == "--grid" && i + 1 < argc) {
            std::string dim = argv[++i];
            auto x = dim.find('x');
            if (x != std::string::npos) {
                // Format: WxH (e.g., 10x10)
                grid_width = std::stoi(dim.substr(0, x));
                grid_height = std::stoi(dim.substr(x + 1));
            } else {
                // Format: W H (e.g., 10 10) - width is first arg, height is next
                grid_width = std::stoi(dim);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    grid_height = std::stoi(argv[++i]);
                } else {
                    grid_height = grid_width;  // Square grid if only one number
                }
            }
        } else if (arg == "--brill" && i + 1 < argc) {
            brill_vertices = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            edge_threshold = std::stof(argv[++i]);
        } else if (arg == "--max-states" && i + 1 < argc) {
            max_states = std::stoi(argv[++i]);
        } else if (arg == "--max-children" && i + 1 < argc) {
            max_children = std::stoi(argv[++i]);
        } else if (arg == "--max-radius" && i + 1 < argc) {
            max_radius = std::stoi(argv[++i]);
        } else if (arg == "--anchors" && i + 1 < argc) {
            num_anchors = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--rule" && i + 1 < argc) {
            rule = argv[++i];
        } else if (arg == "--no-analyze") {
            no_analyze = true;
        } else if (arg == "--no-view") {
            no_view = true;
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--batched") {
            batched = true;
        } else if (arg == "--uniform-random") {
            uniform_random = true;
        } else if (arg == "--matches-per-step" && i + 1 < argc) {
            matches_per_step = std::stoi(argv[++i]);
        } else if (arg == "--fast-reservoir") {
            fast_reservoir = true;
        } else if (arg == "--shuffle-edges") {
            shuffle_edges = true;
        } else if (arg == "--no-canonicalize") {
            no_canonicalize = true;
        } else if (arg == "--no-explore-dedup") {
            no_explore_dedup = true;
        } else if (arg == "--no-canonical-orientation") {
            canonical_orientation = false;
        } else if (arg == "--no-scale-normalization") {
            scale_normalization = false;
        } else if (arg == "--solid-grid") {
            solid_grid = true;
        }
        // New topology options
        else if (arg == "--topology" && i + 1 < argc) {
            use_new_topology = true;
            std::string t = argv[++i];
            if (t == "flat") topology = TopologyType::Flat;
            else if (t == "cylinder") topology = TopologyType::Cylinder;
            else if (t == "torus") topology = TopologyType::Torus;
            else if (t == "sphere") topology = TopologyType::Sphere;
            else if (t == "klein") topology = TopologyType::KleinBottle;
            else if (t == "mobius") topology = TopologyType::MobiusStrip;
            else {
                std::cerr << "Unknown topology: " << t << std::endl;
                std::cerr << "Valid options: flat, cylinder, torus, sphere, klein, mobius" << std::endl;
                return 1;
            }
        } else if (arg == "--sampling" && i + 1 < argc) {
            use_new_topology = true;
            std::string s = argv[++i];
            if (s == "uniform") sampling = SamplingMethod::Uniform;
            else if (s == "poisson") sampling = SamplingMethod::PoissonDisk;
            else if (s == "grid") sampling = SamplingMethod::Grid;
            else if (s == "density") sampling = SamplingMethod::DensityWeighted;
            else {
                std::cerr << "Unknown sampling method: " << s << std::endl;
                std::cerr << "Valid options: uniform, poisson, grid, density" << std::endl;
                return 1;
            }
        } else if (arg == "--defects" && i + 1 < argc) {
            defect_count = std::stoi(argv[++i]);
        } else if (arg == "--defect-mass" && i + 1 < argc) {
            defect_mass = std::stof(argv[++i]);
        } else if (arg == "--defect-exclusion" && i + 1 < argc) {
            defect_exclusion = std::stof(argv[++i]);
        } else if (arg == "--major-radius" && i + 1 < argc) {
            major_radius = std::stof(argv[++i]);
        } else if (arg == "--minor-radius" && i + 1 < argc) {
            minor_radius = std::stof(argv[++i]);
        } else if (arg == "--poisson-distance" && i + 1 < argc) {
            poisson_distance = std::stof(argv[++i]);
        } else if (arg == "--embed-3d") {
            embed_3d = true;
        } else if (arg == "--layout-3d") {
            layout_3d = true;
        } else if (arg == "--geodesics") {
            // Enable geodesic analysis (test particle tracing)
            enable_geodesics = true;
        } else if (arg == "--particles") {
            // Enable particle detection (Robertson-Seymour topological defects)
            enable_particles = true;
        } else if (arg == "--curvature") {
            // Enable curvature analysis (Ollivier-Ricci / dimension gradient)
            enable_curvature = true;
        } else if (arg == "--branch-alignment") {
            // Enable branch alignment (curvature shape space) - implies curvature
            enable_branch_alignment = true;
            enable_curvature = true;  // Alignment needs curvature data
        } else if (arg == "--entropy") {
            // Enable entropy analysis
            enable_entropy = true;
        } else if (arg == "--hilbert") {
            // Enable Hilbert space analysis
            enable_hilbert = true;
        }
        // Minkowski sprinkling options
        else if (arg == "--sprinkling" && i + 1 < argc) {
            use_sprinkling = true;
            sprinkling_n = std::stoi(argv[++i]);
        } else if (arg == "--sprinkling-time" && i + 1 < argc) {
            sprinkling_time = std::stof(argv[++i]);
        } else if (arg == "--sprinkling-space" && i + 1 < argc) {
            sprinkling_space = std::stof(argv[++i]);
        }
        // Brill-Lindquist black hole configuration
        else if (arg == "--bh-mass1" && i + 1 < argc) {
            bh_mass1 = std::stof(argv[++i]);
        } else if (arg == "--bh-mass2" && i + 1 < argc) {
            bh_mass2 = std::stof(argv[++i]);
        } else if (arg == "--bh-separation" && i + 1 < argc) {
            bh_separation = std::stof(argv[++i]);
        } else if (arg == "--bh-box" && i + 4 < argc) {
            // Format: --bh-box xmin xmax ymin ymax
            bh_box_xmin = std::stof(argv[++i]);
            bh_box_xmax = std::stof(argv[++i]);
            bh_box_ymin = std::stof(argv[++i]);
            bh_box_ymax = std::stof(argv[++i]);
        }
    }

    // Auto-enable 3D embedding for curved topologies when using 3D layout
    // Without embed_3d, there are no Z coordinates to work with in layout_3d mode
    if (layout_3d && !embed_3d && topology != TopologyType::Flat) {
        embed_3d = true;
        std::cout << "Auto-enabling --embed-3d for curved topology with --layout-3d" << std::endl;
    }

    // Set default input file based on mode
    if (data_file.empty()) {
        if (mode == "analyze") {
            data_file = "blackhole_evolution.bhevo";
        } else if (mode == "view") {
            data_file = "blackhole_analysis.bhdata";
        }
    }

    // Set default output file based on mode
    if (output_file.empty()) {
        if (mode == "evolve") {
            output_file = "blackhole_evolution.bhevo";
        } else {
            output_file = "blackhole_analysis.bhdata";
        }
    }

    // Load or generate analysis data
    BHAnalysisResult analysis;
    bool should_view = true;  // Whether to launch viewer after processing

    if (mode == "view") {
        // Load existing analysis
        std::cout << "Loading analysis from " << data_file << "..." << std::endl;
        if (!read_analysis(data_file, analysis)) {
            std::cerr << "Failed to load analysis file" << std::endl;
            return 1;
        }
        std::cout << "Loaded: " << analysis.total_states << " states, "
                  << analysis.per_timestep.size() << " timesteps" << std::endl;

        // Check if layout positions are present
        if (!analysis.per_timestep.empty()) {
            const auto& ts0 = analysis.per_timestep[0];
            std::cout << "  Step 0: " << ts0.union_vertices.size() << " vertices, "
                      << ts0.layout_positions.size() << " layout positions" << std::endl;
            // Curvature vector sizes
            std::cout << "  Curvature vectors (step 0):" << std::endl;
            std::cout << "    mean_ollivier: " << ts0.mean_curvature_ollivier.size() << std::endl;
            std::cout << "    mean_wolfram_scalar: " << ts0.mean_curvature_wolfram_scalar.size() << std::endl;
            std::cout << "    mean_wolfram_ricci: " << ts0.mean_curvature_wolfram_ricci.size() << std::endl;
            std::cout << "    mean_dim_gradient: " << ts0.mean_curvature_dim_gradient.size() << std::endl;
            std::cout << "    foliation_ollivier: " << ts0.foliation_curvature_ollivier.size() << std::endl;
            if (!ts0.layout_positions.empty()) {
                float max_r = 0;
                for (const auto& p : ts0.layout_positions) {
                    float r = std::sqrt(p.x * p.x + p.y * p.y);
                    if (r > max_r) max_r = r;
                }
                std::cout << "  Layout bounding radius: " << max_r << std::endl;
                // Show first few positions
                std::cout << "  First 5 layout positions: ";
                for (size_t i = 0; i < std::min(size_t(5), ts0.layout_positions.size()); ++i) {
                    std::cout << "(" << ts0.layout_positions[i].x << "," << ts0.layout_positions[i].y << ") ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "  WARNING: No layout positions in analysis file!" << std::endl;
            }
        }

        if (verify) {
            std::cout << "\n=== VERIFICATION OUTPUT ===" << std::endl;
            std::cout << "File: " << data_file << std::endl;
            std::cout << "Total steps: " << analysis.total_steps << std::endl;
            std::cout << "Total states: " << analysis.total_states << std::endl;
            std::cout << "per_timestep.size(): " << analysis.per_timestep.size() << std::endl;
            std::cout << "Dimension range: [" << analysis.dim_min << ", " << analysis.dim_max << "]" << std::endl;
            std::cout << "Initial: " << analysis.initial.vertex_positions.size() << " vertices, "
                      << analysis.initial.edges.size() << " edges" << std::endl;

            std::cout << "\nTimestep summary:" << std::endl;
            for (size_t i = 0; i < analysis.per_timestep.size(); ++i) {
                const auto& ts = analysis.per_timestep[i];
                if (ts.union_vertices.empty()) {
                    std::cout << "  [" << i << "] step=" << ts.step << " num_states=" << ts.num_states
                              << " (EMPTY union)" << std::endl;
                } else {
                    std::cout << "  [" << i << "] step=" << ts.step << " num_states=" << ts.num_states
                              << " V=" << ts.union_vertices.size() << " E=" << ts.union_edges.size()
                              << " pos=" << ts.vertex_positions.size() << " dim=" << ts.mean_dimensions.size()
                              << std::endl;
                }
            }
            std::cout << "=== END VERIFICATION ===" << std::endl << std::endl;
        }

    } else if (mode == "evolve") {
        // Evolution only - no analysis
        std::cout << "Running evolution only (no analysis)..." << std::endl;

        BHInitialCondition initial;

        if (use_sprinkling) {
            // Minkowski sprinkling (causal set approximation to flat spacetime)
            std::cout << "Generating Minkowski sprinkling..." << std::endl;
            SprinklingConfig sconfig;
            sconfig.spatial_dim = 2;
            sconfig.time_extent = sprinkling_time;
            sconfig.spatial_extent = sprinkling_space;
            sconfig.transitivity_reduction = true;
            sconfig.seed = 0;  // Random seed

            auto sprinkling = generate_minkowski_sprinkling(sprinkling_n, sconfig);
            initial = sprinkling_to_initial_condition(sprinkling);

            std::cout << "  Sprinkling: " << sprinkling_n << " points, "
                      << sprinkling.causal_edges.size() << " causal edges" << std::endl;
            std::cout << "  Dimension estimate: " << sprinkling.dimension_estimate << std::endl;
        } else if (use_new_topology) {
            // Use new topology-aware generation
            TopologyConfig topo_config;
            topo_config.type = topology;
            topo_config.sampling = sampling;
            topo_config.major_radius = major_radius;
            topo_config.minor_radius = minor_radius;
            topo_config.grid_resolution = grid_width;
            topo_config.poisson_min_distance = poisson_distance;
            topo_config.edge_threshold = (edge_threshold > 0) ? edge_threshold : 0.0f;  // 0 = auto-compute

            // Set domain based on topology
            if (topology == TopologyType::Flat) {
                topo_config.domain_x = {-5.0f, 5.0f};
                topo_config.domain_y = {-4.0f, 4.0f};
            } else {
                // For curved topologies, domain is in internal coords
                topo_config.domain_y = {-4.0f, 4.0f};  // z range for cylinder/klein/mobius
            }

            // Configure defects
            topo_config.defects.count = defect_count;
            topo_config.defects.exclusion_radius = defect_exclusion;
            if (defect_count > 0) {
                // Place defects at default positions based on topology
                if (topology == TopologyType::Flat) {
                    // Two defects on x-axis like current BH setup
                    float sep = 3.0f;
                    topo_config.defects.positions.push_back({sep / 2, 0});
                    topo_config.defects.positions.push_back({-sep / 2, 0});
                } else if (topology == TopologyType::Sphere) {
                    // Defects at poles
                    topo_config.defects.positions.push_back({0.1f, 0});         // Near north pole
                    if (defect_count > 1)
                        topo_config.defects.positions.push_back({3.04f, 0});    // Near south pole
                } else {
                    // For cylinder/torus etc, spread evenly around θ
                    for (int d = 0; d < defect_count; ++d) {
                        float theta = 6.283f * d / defect_count;
                        topo_config.defects.positions.push_back({theta, 0});
                    }
                }
                for (int d = 0; d < defect_count; ++d) {
                    topo_config.defects.masses.push_back(defect_mass);
                }
            }

            int n_verts = (brill_vertices > 0) ? brill_vertices : (grid_width * grid_height);
            initial = generate_initial_condition(n_verts, topo_config);

            // Convert internal coords to display coords for visualization
            // Store original positions before conversion for 3D embedding
            std::vector<Vec2> internal_positions = initial.vertex_positions;

            if (embed_3d) {
                // Use 3D topological embedding
                initial.vertex_z.resize(initial.vertex_positions.size());
                for (size_t i = 0; i < initial.vertex_positions.size(); ++i) {
                    Vec3 pos3d = topology_to_display_3d(internal_positions[i], topo_config);
                    initial.vertex_positions[i] = {pos3d.x, pos3d.y};
                    initial.vertex_z[i] = pos3d.z;
                }
                std::cout << "Using 3D topological embedding" << std::endl;
            } else {
                // Use 2D flattened display
                for (auto& pos : initial.vertex_positions) {
                    pos = topology_to_display(pos, topo_config);
                }
            }
        } else {
            // Legacy generation
            BHConfig bh_config;
            bh_config.mass1 = bh_mass1;
            bh_config.mass2 = bh_mass2;
            bh_config.separation = bh_separation;
            if (edge_threshold > 0) {
                bh_config.edge_threshold = edge_threshold;
            } else {
                bh_config.edge_threshold = (brill_vertices > 0) ? 1.5f : 0.8f;
            }
            bh_config.box_x = {bh_box_xmin, bh_box_xmax};
            bh_config.box_y = {bh_box_ymin, bh_box_ymax};

            if (brill_vertices > 0) {
                initial = generate_brill_lindquist(brill_vertices, bh_config);
            } else if (solid_grid) {
                initial = generate_solid_grid(grid_width, grid_height, bh_config);
            } else {
                initial = generate_grid_with_holes(grid_width, grid_height, bh_config);
            }
        }

        std::cout << "Initial condition: " << initial.vertex_positions.size() << " vertices, "
                  << initial.edges.size() << " edges" << std::endl;
        if (initial.has_3d()) {
            std::cout << "  (3D embedded)" << std::endl;
        }

        // Shuffle initial edges if requested (helps avoid systematic bias)
        if (shuffle_edges && !initial.edges.empty()) {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::shuffle(initial.edges.begin(), initial.edges.end(), rng);
            std::cout << "  (edges shuffled)" << std::endl;
        }

        EvolutionConfig evo_config;
        evo_config.rule = rule;
        evo_config.max_steps = steps;
        evo_config.max_states_per_step = max_states;
        evo_config.max_successors_per_parent = max_children;
        evo_config.exploration_probability = 1.0f;
        evo_config.canonicalize_states = !no_canonicalize;
        evo_config.explore_from_canonical_only = !no_explore_dedup;
        evo_config.batched_matching = batched;
        evo_config.uniform_random = uniform_random;
        evo_config.matches_per_step = matches_per_step;
        evo_config.early_terminate_reservoir = fast_reservoir;

        EvolutionRunner runner;
        runner.set_progress_callback([](const std::string& stage, float progress) {
            char buf[128];
            snprintf(buf, sizeof(buf), "\r[%s] %d%%\033[K", stage.c_str(), static_cast<int>(progress * 100));
            fputs(buf, stdout);
            fflush(stdout);
        });

        auto evo_result = runner.run_evolution(initial, evo_config);
        std::cout << std::endl;

        if (evo_result.total_states == 0) {
            std::cerr << "Evolution failed (no states generated)" << std::endl;
            return 1;
        }

        std::cout << "Evolution complete: " << evo_result.total_states << " states, "
                  << evo_result.max_step_reached << " steps" << std::endl;

        // Verification output
        if (verify) {
            std::cout << "\n=== VERIFICATION OUTPUT ===" << std::endl;
            std::cout << "Rule: " << rule << std::endl;
            std::cout << "Initial condition: " << initial.vertex_positions.size() << " vertices, "
                      << initial.edges.size() << " edges" << std::endl;
            std::cout << "\nStates per step:" << std::endl;

            for (size_t step = 0; step < evo_result.states_by_step.size(); ++step) {
                const auto& step_states = evo_result.states_by_step[step];
                if (step_states.empty()) {
                    std::cout << "  Step " << step << ": 0 states (EMPTY)" << std::endl;
                    continue;
                }

                // Sample first state at this step
                const auto& sample = step_states[0];
                size_t min_v = sample.vertex_count(), max_v = sample.vertex_count();
                size_t min_e = sample.edge_count(), max_e = sample.edge_count();

                for (const auto& g : step_states) {
                    min_v = std::min(min_v, g.vertex_count());
                    max_v = std::max(max_v, g.vertex_count());
                    min_e = std::min(min_e, g.edge_count());
                    max_e = std::max(max_e, g.edge_count());
                }

                std::cout << "  Step " << step << ": " << step_states.size() << " states, "
                          << "V=[" << min_v << "-" << max_v << "], "
                          << "E=[" << min_e << "-" << max_e << "]" << std::endl;
            }

            // Check if rewrites are happening (compare step 0 vs step 1)
            std::cout << "\nRewrite verification:" << std::endl;
            if (!evo_result.states_by_step.empty() && !evo_result.states_by_step[0].empty()) {
                const auto& step0 = evo_result.states_by_step[0][0];
                std::cout << "  Step 0 (initial): " << step0.vertex_count() << " vertices, "
                          << step0.edge_count() << " edges" << std::endl;
            }

            if (evo_result.states_by_step.size() > 1 && !evo_result.states_by_step[1].empty()) {
                const auto& step1 = evo_result.states_by_step[1][0];
                std::cout << "  Step 1 (first rewrite): " << step1.vertex_count() << " vertices, "
                          << step1.edge_count() << " edges" << std::endl;

                int64_t v_diff = static_cast<int64_t>(step1.vertex_count()) -
                                static_cast<int64_t>(evo_result.states_by_step[0][0].vertex_count());
                int64_t e_diff = static_cast<int64_t>(step1.edge_count()) -
                                static_cast<int64_t>(evo_result.states_by_step[0][0].edge_count());
                std::cout << "  Delta: V=" << (v_diff >= 0 ? "+" : "") << v_diff
                          << ", E=" << (e_diff >= 0 ? "+" : "") << e_diff << std::endl;

                // Rule consumes 4 edges, produces 4 edges, adds 1 vertex
                // So expected: V+1, E+0 (4-4=0)
                if (v_diff == 1 && e_diff == 0) {
                    std::cout << "  (Matches expected BH rule: +1 vertex, +0 edges)" << std::endl;
                }
            } else {
                std::cout << "  Step 1: NO STATES (rule did not match!)" << std::endl;
                std::cout << "  Possible causes:" << std::endl;
                std::cout << "    - Rule pattern not found in initial graph" << std::endl;
                std::cout << "    - Rule parsing error" << std::endl;
                std::cout << "    - Graph connectivity issue" << std::endl;
            }

            std::cout << "=== END VERIFICATION ===" << std::endl << std::endl;
        }

        // Save evolution data
        EvolutionData evo_data;
        evo_data.initial = initial;
        evo_data.config = evo_config;
        evo_data.states_by_step = evo_result.states_by_step;  // Keep copy for analysis
        evo_data.total_states = evo_result.total_states;
        evo_data.total_events = evo_result.total_events;
        evo_data.max_step_reached = evo_result.max_step_reached;

        if (write_evolution(output_file, evo_data)) {
            std::cout << "Saved evolution data to " << output_file << std::endl;
        } else {
            std::cerr << "Failed to save evolution data" << std::endl;
            return 1;
        }

        if (no_analyze) {
            std::cout << "Use 'analyze " << output_file << "' to run Hausdorff analysis." << std::endl;
            return 0;
        }

        // Fall through to analysis
        std::cout << "\nRunning Hausdorff analysis (max_radius=" << max_radius
                  << ", anchors=" << num_anchors << ")..." << std::endl;

        AnalysisConfig ana_config;
        ana_config.num_anchors = num_anchors;
        ana_config.anchor_min_separation = 3;
        ana_config.max_radius = max_radius;
        ana_config.compute_geodesics = enable_geodesics;
        ana_config.detect_particles = enable_particles;
        ana_config.compute_topological_charge = enable_particles;  // Charge if particles enabled

        // Reuse the runner for analysis
        analysis = runner.run_analysis(
            initial,
            evo_config,
            evo_data.states_by_step,
            evo_data.total_states,
            evo_data.total_events,
            evo_data.max_step_reached,
            ana_config
        );
        std::cout << std::endl;

        if (analysis.total_states == 0) {
            std::cerr << "Analysis failed" << std::endl;
            return 1;
        }

        std::cout << "Analysis complete: " << analysis.total_states << " states, "
                  << analysis.per_timestep.size() << " timesteps" << std::endl;
        std::cout << "Dimension range: [" << analysis.dim_min << ", " << analysis.dim_max << "]" << std::endl;

    } else if (mode == "analyze") {
        // Load evolution data and run analysis
        std::cout << "Loading evolution data from " << data_file << "..." << std::endl;

        EvolutionData evo_data;
        if (!read_evolution(data_file, evo_data)) {
            std::cerr << "Failed to load evolution file" << std::endl;
            return 1;
        }

        std::cout << "Loaded: " << evo_data.total_states << " states, "
                  << evo_data.max_step_reached << " steps" << std::endl;

        std::cout << "Running Hausdorff analysis (max_radius=" << max_radius
                  << ", anchors=" << num_anchors << ")..." << std::endl;

        AnalysisConfig ana_config;
        ana_config.num_anchors = num_anchors;
        ana_config.anchor_min_separation = 3;
        ana_config.max_radius = max_radius;
        ana_config.compute_geodesics = enable_geodesics;
        ana_config.detect_particles = enable_particles;
        ana_config.compute_topological_charge = enable_particles;

        EvolutionRunner runner;
        runner.set_progress_callback([](const std::string& stage, float progress) {
            char buf[128];
            snprintf(buf, sizeof(buf), "\r[%s] %d%%\033[K", stage.c_str(), static_cast<int>(progress * 100));
            fputs(buf, stdout);
            fflush(stdout);
        });

        analysis = runner.run_analysis(
            evo_data.initial,
            evo_data.config,
            evo_data.states_by_step,
            evo_data.total_states,
            evo_data.total_events,
            evo_data.max_step_reached,
            ana_config
        );
        std::cout << std::endl;

        if (analysis.total_states == 0) {
            std::cerr << "Analysis failed" << std::endl;
            return 1;
        }

        std::cout << "Analysis complete: " << analysis.total_states << " states, "
                  << analysis.per_timestep.size() << " timesteps" << std::endl;
        std::cout << "Dimension range: [" << analysis.dim_min << ", " << analysis.dim_max << "]" << std::endl;

    } else if (mode == "run" || mode == "generate") {
        // Full pipeline: generate + evolve + analyze
        std::cout << "Generating black hole analysis..." << std::endl;

        BHInitialCondition initial;

        if (use_sprinkling) {
            // Minkowski sprinkling (causal set approximation to flat spacetime)
            std::cout << "Generating Minkowski sprinkling..." << std::endl;
            SprinklingConfig sconfig;
            sconfig.spatial_dim = 2;
            sconfig.time_extent = sprinkling_time;
            sconfig.spatial_extent = sprinkling_space;
            sconfig.transitivity_reduction = true;
            sconfig.seed = 0;  // Random seed

            auto sprinkling = generate_minkowski_sprinkling(sprinkling_n, sconfig);
            initial = sprinkling_to_initial_condition(sprinkling);

            std::cout << "  Sprinkling: " << sprinkling_n << " points, "
                      << sprinkling.causal_edges.size() << " causal edges" << std::endl;
            std::cout << "  Dimension estimate: " << sprinkling.dimension_estimate << std::endl;
        } else if (use_new_topology) {
            // Use new topology-aware generation
            TopologyConfig topo_config;
            topo_config.type = topology;
            topo_config.sampling = sampling;
            topo_config.major_radius = major_radius;
            topo_config.minor_radius = minor_radius;
            topo_config.grid_resolution = grid_width;
            topo_config.poisson_min_distance = poisson_distance;
            topo_config.edge_threshold = (edge_threshold > 0) ? edge_threshold : 0.0f;  // 0 = auto-compute

            // Set domain based on topology
            if (topology == TopologyType::Flat) {
                topo_config.domain_x = {-5.0f, 5.0f};
                topo_config.domain_y = {-4.0f, 4.0f};
            } else {
                // For curved topologies, domain is in internal coords
                topo_config.domain_y = {-4.0f, 4.0f};  // z range for cylinder/klein/mobius
            }

            // Configure defects
            topo_config.defects.count = defect_count;
            topo_config.defects.exclusion_radius = defect_exclusion;
            if (defect_count > 0) {
                // Place defects at default positions based on topology
                if (topology == TopologyType::Flat) {
                    // Two defects on x-axis like current BH setup
                    float sep = 3.0f;
                    topo_config.defects.positions.push_back({sep / 2, 0});
                    topo_config.defects.positions.push_back({-sep / 2, 0});
                } else if (topology == TopologyType::Sphere) {
                    // Defects at poles
                    topo_config.defects.positions.push_back({0.1f, 0});         // Near north pole
                    if (defect_count > 1)
                        topo_config.defects.positions.push_back({3.04f, 0});    // Near south pole
                } else {
                    // For cylinder/torus etc, spread evenly around θ
                    for (int d = 0; d < defect_count; ++d) {
                        float theta = 6.283f * d / defect_count;
                        topo_config.defects.positions.push_back({theta, 0});
                    }
                }
                for (int d = 0; d < defect_count; ++d) {
                    topo_config.defects.masses.push_back(defect_mass);
                }
            }

            int n_verts = (brill_vertices > 0) ? brill_vertices : (grid_width * grid_height);
            initial = generate_initial_condition(n_verts, topo_config);

            // Convert internal coords to display coords for visualization
            // Store original positions before conversion for 3D embedding
            std::vector<Vec2> internal_positions_run = initial.vertex_positions;

            if (embed_3d) {
                // Use 3D topological embedding
                initial.vertex_z.resize(initial.vertex_positions.size());
                for (size_t i = 0; i < initial.vertex_positions.size(); ++i) {
                    Vec3 pos3d = topology_to_display_3d(internal_positions_run[i], topo_config);
                    initial.vertex_positions[i] = {pos3d.x, pos3d.y};
                    initial.vertex_z[i] = pos3d.z;
                }
                std::cout << "Using 3D topological embedding" << std::endl;
            } else {
                // Use 2D flattened display
                for (auto& pos : initial.vertex_positions) {
                    pos = topology_to_display(pos, topo_config);
                }
            }
        } else {
            // Legacy generation
            BHConfig bh_config;
            bh_config.mass1 = bh_mass1;
            bh_config.mass2 = bh_mass2;
            bh_config.separation = bh_separation;
            if (edge_threshold > 0) {
                bh_config.edge_threshold = edge_threshold;
            } else {
                bh_config.edge_threshold = (brill_vertices > 0) ? 1.5f : 0.8f;
            }
            bh_config.box_x = {bh_box_xmin, bh_box_xmax};
            bh_config.box_y = {bh_box_ymin, bh_box_ymax};

            if (brill_vertices > 0) {
                initial = generate_brill_lindquist(brill_vertices, bh_config);
            } else if (solid_grid) {
                initial = generate_solid_grid(grid_width, grid_height, bh_config);
            } else {
                initial = generate_grid_with_holes(grid_width, grid_height, bh_config);
            }
        }

        // Shuffle initial edges if requested (helps avoid systematic bias)
        if (shuffle_edges && !initial.edges.empty()) {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::shuffle(initial.edges.begin(), initial.edges.end(), rng);
            std::cout << "  (edges shuffled)" << std::endl;
        }

        EvolutionConfig evo_config;
        evo_config.rule = rule;
        evo_config.max_steps = steps;
        evo_config.max_states_per_step = max_states;
        evo_config.max_successors_per_parent = max_children;
        evo_config.exploration_probability = 1.0f;
        evo_config.canonicalize_states = !no_canonicalize;
        evo_config.explore_from_canonical_only = !no_explore_dedup;
        evo_config.batched_matching = batched;
        evo_config.uniform_random = uniform_random;
        evo_config.matches_per_step = matches_per_step;
        evo_config.early_terminate_reservoir = fast_reservoir;

        AnalysisConfig ana_config;
        ana_config.num_anchors = num_anchors;
        ana_config.anchor_min_separation = 3;
        ana_config.max_radius = max_radius;
        ana_config.compute_geodesics = enable_geodesics;
        ana_config.detect_particles = enable_particles;
        ana_config.compute_topological_charge = enable_particles;

        EvolutionRunner runner;
        runner.set_progress_callback([](const std::string& stage, float progress) {
            char buf[128];
            snprintf(buf, sizeof(buf), "\r[%s] %d%%\033[K", stage.c_str(), static_cast<int>(progress * 100));
            fputs(buf, stdout);
            fflush(stdout);
        });

        analysis = runner.run_full_analysis(initial, evo_config, ana_config);
        std::cout << std::endl;

        if (analysis.total_states == 0) {
            std::cerr << "Analysis failed (no states generated)" << std::endl;
            return 1;
        }

        std::cout << "Analysis complete: " << analysis.total_states << " states, "
                  << analysis.per_timestep.size() << " timesteps" << std::endl;
        std::cout << "Dimension range: [" << analysis.dim_min << ", " << analysis.dim_max << "]" << std::endl;

    } else {
        // Unknown mode
        std::cerr << "Unknown command: " << mode << std::endl;
        print_usage();
        return 1;
    }

    // ==========================================================================
    // Additional Analysis (curvature, entropy, hilbert, branch alignment)
    // ==========================================================================
    // These analyses are computed during run/generate mode and saved to .bhdata.
    // In view mode, they are loaded from the file (no computation needed).

    CurvatureAnalysisResult curvature_result;
    EntropyAnalysisResult entropy_result;
    HilbertSpaceAnalysis hilbert_result;
    BranchialAnalysisResult branchial_result;
    bool has_curvature_analysis = false;
    bool has_entropy_analysis = false;
    bool has_hilbert_analysis = false;

    // Only compute analyses during run/generate mode (view mode loads from file)
    const bool should_compute_analyses = (mode != "view");

    // In view mode, load analysis flags from the serialized data
    if (!should_compute_analyses) {
        has_curvature_analysis = analysis.has_curvature_analysis;
        has_hilbert_analysis = analysis.has_hilbert_analysis;
        // has_branchial_analysis is stored in analysis but we don't have a local variable for it
        // Branch alignment flags are already in analysis object and used directly
        if (analysis.has_curvature_analysis) {
            // Reconstruct curvature_result from stored data for rendering
            curvature_result.ollivier_ricci_map = analysis.curvature_ollivier_ricci;
            curvature_result.dimension_gradient_map = analysis.curvature_dimension_gradient;
            curvature_result.wolfram_scalar_map = analysis.curvature_wolfram_scalar;
            curvature_result.wolfram_ricci_map = analysis.curvature_wolfram_ricci;
            curvature_result.mean_ollivier_ricci = analysis.curvature_ollivier_mean;
            curvature_result.min_ollivier_ricci = analysis.curvature_ollivier_min;
            curvature_result.max_ollivier_ricci = analysis.curvature_ollivier_max;
            curvature_result.mean_dimension_gradient = analysis.curvature_dim_grad_mean;
            curvature_result.min_dimension_gradient = analysis.curvature_dim_grad_min;
            curvature_result.max_dimension_gradient = analysis.curvature_dim_grad_max;
            curvature_result.mean_wolfram_scalar = analysis.curvature_wolfram_scalar_mean;
            curvature_result.min_wolfram_scalar = analysis.curvature_wolfram_scalar_min;
            curvature_result.max_wolfram_scalar = analysis.curvature_wolfram_scalar_max;
            curvature_result.mean_wolfram_ricci = analysis.curvature_wolfram_ricci_mean;
            curvature_result.min_wolfram_ricci = analysis.curvature_wolfram_ricci_min;
            curvature_result.max_wolfram_ricci = analysis.curvature_wolfram_ricci_max;
        }
        if (analysis.has_hilbert_analysis) {
            hilbert_result.vertex_probabilities = analysis.hilbert_vertex_probabilities;
            hilbert_result.num_states = analysis.hilbert_num_states;
            hilbert_result.num_vertices = analysis.hilbert_num_vertices;
            hilbert_result.mean_inner_product = analysis.hilbert_mean_inner_product;
            hilbert_result.max_inner_product = analysis.hilbert_max_inner_product;
            hilbert_result.mean_vertex_probability = analysis.hilbert_mean_vertex_probability;
            hilbert_result.vertex_probability_entropy = analysis.hilbert_vertex_probability_entropy;
        }
        if (analysis.has_branchial_analysis) {
            branchial_result.vertex_sharpness = analysis.branchial_vertex_sharpness;
            branchial_result.vertex_entropy = analysis.branchial_vertex_entropy;
            branchial_result.mean_sharpness = analysis.branchial_mean_sharpness;
            branchial_result.mean_branch_entropy = analysis.branchial_mean_entropy;
        }
    }

    // Build SimpleGraph from analysis data (for curvature/entropy analysis)
    SimpleGraph analysis_graph;
    std::vector<float> vertex_dimensions_vec;
    bool has_dimension_data = false;
    if (should_compute_analyses && !analysis.all_edges.empty()) {
        analysis_graph.build_from_edges(analysis.all_edges);
        // Build dimension vector indexed by vertex position (if available)
        vertex_dimensions_vec.resize(analysis_graph.vertex_count(), 2.0f);  // Default to 2D
        if (!analysis.mega_dimension.empty()) {
            has_dimension_data = true;
            for (size_t i = 0; i < analysis_graph.vertices().size(); ++i) {
                VertexId vid = analysis_graph.vertices()[i];
                auto it = analysis.mega_dimension.find(vid);
                if (it != analysis.mega_dimension.end()) {
                    vertex_dimensions_vec[i] = it->second;
                }
            }
        }
    }

    // Compute curvature analysis if enabled
    if (enable_curvature && analysis_graph.vertex_count() > 0) {
        std::cout << "\r[Curvature analysis] 0%\033[K" << std::flush;
        auto curv_start = std::chrono::high_resolution_clock::now();

        CurvatureConfig curv_config;
        curv_config.compute_ollivier_ricci = true;
        curv_config.compute_dimension_gradient = has_dimension_data;  // Only if we have dimension data
        curv_config.compute_wolfram_scalar = has_dimension_data;      // Ball volume method
        curv_config.compute_wolfram_ricci = has_dimension_data;       // Tube volume method (full tensor)
        curv_config.wolfram_ricci_full_tensor = true;                 // Use full tensor formula

        // Use parallel version for better performance
        job_system::JobSystem<int> curv_js(std::thread::hardware_concurrency());
        curv_js.start();
        curvature_result = analyze_curvature_parallel(analysis_graph, &curv_js, curv_config, &vertex_dimensions_vec);
        curv_js.shutdown();
        has_curvature_analysis = true;

        // Store results in analysis for serialization
        analysis.curvature_ollivier_ricci = curvature_result.ollivier_ricci_map;
        analysis.curvature_dimension_gradient = curvature_result.dimension_gradient_map;
        analysis.curvature_wolfram_scalar = curvature_result.wolfram_scalar_map;
        analysis.curvature_wolfram_ricci = curvature_result.wolfram_ricci_map;
        analysis.curvature_ollivier_mean = curvature_result.mean_ollivier_ricci;
        analysis.curvature_ollivier_min = curvature_result.min_ollivier_ricci;
        analysis.curvature_ollivier_max = curvature_result.max_ollivier_ricci;
        analysis.curvature_dim_grad_mean = curvature_result.mean_dimension_gradient;
        analysis.curvature_dim_grad_min = curvature_result.min_dimension_gradient;
        analysis.curvature_dim_grad_max = curvature_result.max_dimension_gradient;
        analysis.curvature_wolfram_scalar_mean = curvature_result.mean_wolfram_scalar;
        analysis.curvature_wolfram_scalar_min = curvature_result.min_wolfram_scalar;
        analysis.curvature_wolfram_scalar_max = curvature_result.max_wolfram_scalar;
        analysis.curvature_wolfram_ricci_mean = curvature_result.mean_wolfram_ricci;
        analysis.curvature_wolfram_ricci_min = curvature_result.min_wolfram_ricci;
        analysis.curvature_wolfram_ricci_max = curvature_result.max_wolfram_ricci;
        analysis.has_curvature_analysis = true;

        std::cout << "\r[Curvature analysis] 100%\033[K" << std::endl;
        auto curv_end = std::chrono::high_resolution_clock::now();
        auto curv_ms = std::chrono::duration<double, std::milli>(curv_end - curv_start).count();
        std::cout << "[TIMING] Curvature analysis: " << std::fixed << std::setprecision(1) << curv_ms << " ms" << std::endl;

        std::cout << "  Ollivier-Ricci: mean=" << curvature_result.mean_ollivier_ricci
                  << ", min=" << curvature_result.min_ollivier_ricci
                  << ", max=" << curvature_result.max_ollivier_ricci << std::endl;
        if (has_dimension_data) {
            std::cout << "  Dimension gradient: mean=" << curvature_result.mean_dimension_gradient
                      << ", min=" << curvature_result.min_dimension_gradient
                      << ", max=" << curvature_result.max_dimension_gradient << std::endl;
            std::cout << "  Wolfram scalar (ball): mean=" << curvature_result.mean_wolfram_scalar
                      << ", min=" << curvature_result.min_wolfram_scalar
                      << ", max=" << curvature_result.max_wolfram_scalar << std::endl;
            std::cout << "  Wolfram Ricci (tube): mean=" << curvature_result.mean_wolfram_ricci
                      << ", min=" << curvature_result.min_wolfram_ricci
                      << ", max=" << curvature_result.max_wolfram_ricci << std::endl;
        }
    }

    // Compute entropy analysis if enabled
    if (enable_entropy && analysis_graph.vertex_count() > 0) {
        std::cout << "\r[Entropy analysis] 0%\033[K" << std::flush;
        auto ent_start = std::chrono::high_resolution_clock::now();

        EntropyConfig ent_config;
        ent_config.compute_local_entropy = true;
        ent_config.compute_mutual_info = true;
        ent_config.compute_fisher_info = true;

        // Use parallel version for better performance
        job_system::JobSystem<int> ent_js(std::thread::hardware_concurrency());
        ent_js.start();
        entropy_result = analyze_entropy_parallel(analysis_graph, &ent_js, ent_config, &vertex_dimensions_vec);
        ent_js.shutdown();
        has_entropy_analysis = true;

        std::cout << "\r[Entropy analysis] 100%\033[K" << std::endl;
        auto ent_end = std::chrono::high_resolution_clock::now();
        auto ent_ms = std::chrono::duration<double, std::milli>(ent_end - ent_start).count();
        std::cout << "[TIMING] Entropy analysis: " << std::fixed << std::setprecision(1) << ent_ms << " ms" << std::endl;

        std::cout << "  Graph entropy: " << entropy_result.graph_entropy << std::endl;
        std::cout << "  Degree entropy: " << entropy_result.degree_entropy << std::endl;
        std::cout << "  Total mutual info: " << entropy_result.total_mutual_info << std::endl;
        std::cout << "  Total Fisher info: " << entropy_result.total_fisher_info << std::endl;
    }

    // Compute Hilbert space analysis if enabled
    if (enable_hilbert && should_compute_analyses && analysis.states_per_step.size() > 1) {
        std::cout << "\r[Hilbert space analysis] 0%\033[K" << std::flush;
        auto hilb_start = std::chrono::high_resolution_clock::now();

        // Build BranchState structures from analysis data
        std::vector<BranchState> branch_states;
        uint32_t state_id = 0;
        for (size_t step = 0; step < analysis.states_per_step.size(); ++step) {
            for (size_t branch = 0; branch < analysis.states_per_step[step].size(); ++branch) {
                BranchState bs;
                bs.state_id = state_id++;
                bs.branch_id = static_cast<uint32_t>(branch);
                bs.step = static_cast<uint32_t>(step);

                // Get vertices and edges directly from this state's data
                const auto& state_data = analysis.states_per_step[step][branch];
                bs.vertices = state_data.vertices;
                bs.edges = state_data.edges;

                branch_states.push_back(std::move(bs));
            }
        }

        if (!branch_states.empty()) {
            BranchialConfig config;
            config.compute_sharpness = true;
            config.compute_entropy = true;

            // Use parallel versions for better performance
            job_system::JobSystem<int> bran_js(std::thread::hardware_concurrency());
            bran_js.start();
            branchial_result = analyze_branchial_parallel(branch_states, &bran_js, config);

            auto branchial_graph = build_branchial_graph(branch_states, config);
            hilbert_result = analyze_hilbert_space_full_parallel(branchial_graph, &bran_js);
            bran_js.shutdown();
            has_hilbert_analysis = (hilbert_result.num_states > 0);

            // Store hilbert results in analysis for serialization
            analysis.hilbert_vertex_probabilities = hilbert_result.vertex_probabilities;
            analysis.hilbert_num_states = hilbert_result.num_states;
            analysis.hilbert_num_vertices = hilbert_result.num_vertices;
            analysis.hilbert_mean_inner_product = hilbert_result.mean_inner_product;
            analysis.hilbert_max_inner_product = hilbert_result.max_inner_product;
            analysis.hilbert_mean_vertex_probability = hilbert_result.mean_vertex_probability;
            analysis.hilbert_vertex_probability_entropy = hilbert_result.vertex_probability_entropy;
            analysis.has_hilbert_analysis = true;

            // Store branchial results in analysis for serialization
            analysis.branchial_vertex_sharpness = branchial_result.vertex_sharpness;
            analysis.branchial_vertex_entropy = branchial_result.vertex_entropy;
            analysis.branchial_mean_sharpness = branchial_result.mean_sharpness;
            analysis.branchial_mean_entropy = branchial_result.mean_branch_entropy;
            analysis.has_branchial_analysis = true;

            std::cout << "\r[Hilbert space analysis] 100%\033[K" << std::endl;
            auto hilb_end = std::chrono::high_resolution_clock::now();
            auto hilb_ms = std::chrono::duration<double, std::milli>(hilb_end - hilb_start).count();
            std::cout << "[TIMING] Hilbert space analysis: " << std::fixed << std::setprecision(1) << hilb_ms << " ms" << std::endl;

            std::cout << "  States analyzed: " << hilbert_result.num_states << std::endl;
            std::cout << "  Unique vertices: " << hilbert_result.num_vertices << std::endl;
            std::cout << "  Mean inner product: " << hilbert_result.mean_inner_product << std::endl;
            std::cout << "  Max inner product: " << hilbert_result.max_inner_product << std::endl;
            std::cout << "  Mean vertex probability: " << hilbert_result.mean_vertex_probability << std::endl;
            std::cout << "  Vertex probability entropy: " << hilbert_result.vertex_probability_entropy << std::endl;
        }
    }

    // Compute branch alignment if enabled (curvature shape space visualization)
    // Only during run/generate mode - view mode loads from file
    if (should_compute_analyses && enable_branch_alignment && !analysis.states_per_step.empty()) {
        std::cout << "\r[Branch alignment] 0%\033[K" << std::flush;
        auto align_start = std::chrono::high_resolution_clock::now();

        // Prepare per-state graphs and curvature data
        std::vector<SimpleGraph> all_graphs;
        std::vector<std::unordered_map<VertexId, float>> all_curvatures;
        std::vector<StateId> all_state_ids;
        std::vector<uint32_t> state_to_step;

        StateId state_id = 0;
        for (size_t step = 0; step < analysis.states_per_step.size(); ++step) {
            for (size_t branch = 0; branch < analysis.states_per_step[step].size(); ++branch) {
                const auto& state_data = analysis.states_per_step[step][branch];

                // Build graph for this state
                SimpleGraph graph;
                graph.build_from_edges(state_data.edges);

                // Compute per-vertex curvature using dimension data (Wolfram-Ricci style)
                std::unordered_map<VertexId, float> curvature;
                for (VertexId v : state_data.vertices) {
                    // Get dimension for this vertex from mega_dimension
                    auto dim_it = analysis.mega_dimension.find(v);
                    float dim = (dim_it != analysis.mega_dimension.end()) ? dim_it->second : 2.0f;

                    // Wolfram-Ricci curvature: K = 2 - d (where d is local dimension)
                    // Positive for d < 2, negative for d > 2
                    curvature[v] = 2.0f - dim;
                }

                all_graphs.push_back(std::move(graph));
                all_curvatures.push_back(std::move(curvature));
                all_state_ids.push_back(state_id);
                state_to_step.push_back(static_cast<uint32_t>(step));
                state_id++;
            }
        }

        if (!all_graphs.empty()) {
            // Group states by timestep
            std::map<uint32_t, std::vector<size_t>> step_to_indices;
            for (size_t i = 0; i < all_state_ids.size(); ++i) {
                step_to_indices[state_to_step[i]].push_back(i);
            }

            // Per-timestep alignment with FIXED reference frame (not chained)
            // All timesteps align to the first timestep's frame to prevent drift
            job_system::JobSystem<int> js(std::thread::hardware_concurrency());
            js.start();

            analysis.alignment_per_timestep.resize(analysis.states_per_step.size());
            analysis.global_pc1_min = std::numeric_limits<float>::max();
            analysis.global_pc1_max = std::numeric_limits<float>::lowest();
            analysis.global_pc2_min = std::numeric_limits<float>::max();
            analysis.global_pc2_max = std::numeric_limits<float>::lowest();
            analysis.global_pc3_min = std::numeric_limits<float>::max();
            analysis.global_pc3_max = std::numeric_limits<float>::lowest();
            analysis.global_curvature_min = std::numeric_limits<float>::max();
            analysis.global_curvature_max = std::numeric_limits<float>::lowest();

            // Fixed reference frame from first timestep (prevents drift accumulation)
            AlignmentReferenceFrame fixed_reference_frame;

            size_t steps_done = 0;
            size_t total_steps = step_to_indices.size();
            for (const auto& [step, indices] : step_to_indices) {
                // Collect graphs and curvatures for this timestep only
                std::vector<SimpleGraph> step_graphs;
                std::vector<std::unordered_map<VertexId, float>> step_curvatures;
                std::vector<StateId> step_state_ids;

                for (size_t i : indices) {
                    step_graphs.push_back(all_graphs[i]);
                    step_curvatures.push_back(all_curvatures[i]);
                    step_state_ids.push_back(all_state_ids[i]);
                }

                // Per-branch PCA alignment with canonical orientation
                // Use fixed reference from first timestep (not previous timestep)
                AlignmentReferenceFrame new_frame;
                auto agg = align_branches_per_branch(
                    step_graphs, step_curvatures, step_state_ids, &js,
                    fixed_reference_frame.valid ? &fixed_reference_frame : nullptr,
                    &new_frame, canonical_orientation, scale_normalization);

                // Save first timestep's frame as the fixed reference for all subsequent
                if (!fixed_reference_frame.valid && new_frame.valid) {
                    fixed_reference_frame = new_frame;
                }

                // Copy to PerTimestepAlignment
                auto& pta = analysis.alignment_per_timestep[step];
                pta.all_pc1 = std::move(agg.all_pc1);
                pta.all_pc2 = std::move(agg.all_pc2);
                pta.all_pc3 = std::move(agg.all_pc3);
                pta.all_curvature = std::move(agg.all_curvature);
                pta.all_rank = std::move(agg.all_rank);
                pta.branch_id = std::move(agg.branch_id);
                pta.all_vertices = std::move(agg.all_vertices);
                pta.state_id = std::move(agg.state_id);
                pta.branch_sizes = std::move(agg.branch_sizes);
                pta.total_points = agg.total_points;
                pta.num_branches = agg.num_branches;

                // Update global bounds
                if (agg.total_points > 0) {
                    analysis.global_pc1_min = std::min(analysis.global_pc1_min, agg.pc1_min);
                    analysis.global_pc1_max = std::max(analysis.global_pc1_max, agg.pc1_max);
                    analysis.global_pc2_min = std::min(analysis.global_pc2_min, agg.pc2_min);
                    analysis.global_pc2_max = std::max(analysis.global_pc2_max, agg.pc2_max);
                    analysis.global_pc3_min = std::min(analysis.global_pc3_min, agg.pc3_min);
                    analysis.global_pc3_max = std::max(analysis.global_pc3_max, agg.pc3_max);
                    analysis.global_curvature_min = std::min(analysis.global_curvature_min, agg.curvature_min);
                    analysis.global_curvature_max = std::max(analysis.global_curvature_max, agg.curvature_max);
                }

                // Progress reporting (after work is done)
                steps_done++;
                if (steps_done % 10 == 0 || steps_done == total_steps) {
                    std::cout << "\r[Wolfram-Ricci alignment] " << (100 * steps_done / total_steps) << "%\033[K" << std::flush;
                }
            }
            std::cout << std::endl;  // Complete progress line

            js.shutdown();

            analysis.curvature_abs_max = std::max(
                std::abs(analysis.global_curvature_min),
                std::abs(analysis.global_curvature_max)
            );
            analysis.has_branch_alignment = true;

            auto align_end = std::chrono::high_resolution_clock::now();
            auto align_ms = std::chrono::duration<double, std::milli>(align_end - align_start).count();
            std::cout << "[TIMING] Branch alignment: " << std::fixed << std::setprecision(1) << align_ms << " ms" << std::endl;

            std::cout << "  Wolfram-Ricci aligned " << all_graphs.size() << " states across "
                      << analysis.states_per_step.size() << " timesteps" << std::endl;
            std::cout << "  PC range: [" << analysis.global_pc1_min << ", " << analysis.global_pc1_max << "] x ["
                      << analysis.global_pc2_min << ", " << analysis.global_pc2_max << "] x ["
                      << analysis.global_pc3_min << ", " << analysis.global_pc3_max << "]" << std::endl;
            std::cout << "  Curvature range: [" << analysis.global_curvature_min
                      << ", " << analysis.global_curvature_max << "]" << std::endl;

            // Also compute Ollivier-Ricci alignment if curvature analysis was enabled
            if (has_curvature_analysis) {
                std::cout << "\r[Ollivier-Ricci alignment] 0%\033[K" << std::flush;
                auto ollivier_start = std::chrono::high_resolution_clock::now();

                // Compute Ollivier-Ricci curvature for each state
                std::vector<std::unordered_map<VertexId, float>> all_ollivier_curvatures(all_graphs.size());

                job_system::JobSystem<int> ollivier_js(std::thread::hardware_concurrency());
                ollivier_js.start();

                for (size_t i = 0; i < all_graphs.size(); ++i) {
                    ollivier_js.submit_function([&all_graphs, &all_ollivier_curvatures, i]() {
                        all_ollivier_curvatures[i] = compute_vertex_ollivier_ricci(all_graphs[i], 0.5f);
                    }, 0);
                }
                ollivier_js.wait_for_completion();

                // Run alignment with Ollivier-Ricci curvature
                analysis.alignment_ollivier.resize(analysis.states_per_step.size());
                analysis.ollivier_pc1_min = std::numeric_limits<float>::max();
                analysis.ollivier_pc1_max = std::numeric_limits<float>::lowest();
                analysis.ollivier_pc2_min = std::numeric_limits<float>::max();
                analysis.ollivier_pc2_max = std::numeric_limits<float>::lowest();
                analysis.ollivier_pc3_min = std::numeric_limits<float>::max();
                analysis.ollivier_pc3_max = std::numeric_limits<float>::lowest();
                analysis.ollivier_curvature_min = std::numeric_limits<float>::max();
                analysis.ollivier_curvature_max = std::numeric_limits<float>::lowest();

                AlignmentReferenceFrame ollivier_fixed_ref;

                size_t ollivier_steps_done = 0;
                for (const auto& [step, indices] : step_to_indices) {
                    std::vector<SimpleGraph> step_graphs;
                    std::vector<std::unordered_map<VertexId, float>> step_curvatures;
                    std::vector<StateId> step_state_ids;

                    for (size_t i : indices) {
                        step_graphs.push_back(all_graphs[i]);
                        step_curvatures.push_back(all_ollivier_curvatures[i]);
                        step_state_ids.push_back(all_state_ids[i]);
                    }

                    AlignmentReferenceFrame new_frame;
                    auto agg = align_branches_per_branch(
                        step_graphs, step_curvatures, step_state_ids, &ollivier_js,
                        ollivier_fixed_ref.valid ? &ollivier_fixed_ref : nullptr,
                        &new_frame, canonical_orientation, scale_normalization);

                    if (!ollivier_fixed_ref.valid && new_frame.valid) {
                        ollivier_fixed_ref = new_frame;
                    }

                    auto& pta = analysis.alignment_ollivier[step];
                    pta.all_pc1 = std::move(agg.all_pc1);
                    pta.all_pc2 = std::move(agg.all_pc2);
                    pta.all_pc3 = std::move(agg.all_pc3);
                    pta.all_curvature = std::move(agg.all_curvature);
                    pta.all_rank = std::move(agg.all_rank);
                    pta.branch_id = std::move(agg.branch_id);
                    pta.all_vertices = std::move(agg.all_vertices);
                    pta.state_id = std::move(agg.state_id);
                    pta.branch_sizes = std::move(agg.branch_sizes);
                    pta.total_points = agg.total_points;
                    pta.num_branches = agg.num_branches;

                    if (agg.total_points > 0) {
                        analysis.ollivier_pc1_min = std::min(analysis.ollivier_pc1_min, agg.pc1_min);
                        analysis.ollivier_pc1_max = std::max(analysis.ollivier_pc1_max, agg.pc1_max);
                        analysis.ollivier_pc2_min = std::min(analysis.ollivier_pc2_min, agg.pc2_min);
                        analysis.ollivier_pc2_max = std::max(analysis.ollivier_pc2_max, agg.pc2_max);
                        analysis.ollivier_pc3_min = std::min(analysis.ollivier_pc3_min, agg.pc3_min);
                        analysis.ollivier_pc3_max = std::max(analysis.ollivier_pc3_max, agg.pc3_max);
                        analysis.ollivier_curvature_min = std::min(analysis.ollivier_curvature_min, agg.curvature_min);
                        analysis.ollivier_curvature_max = std::max(analysis.ollivier_curvature_max, agg.curvature_max);
                    }

                    // Progress reporting (after work is done)
                    ollivier_steps_done++;
                    if (ollivier_steps_done % 10 == 0 || ollivier_steps_done == total_steps) {
                        std::cout << "\r[Ollivier-Ricci alignment] " << (100 * ollivier_steps_done / total_steps) << "%\033[K" << std::flush;
                    }
                }
                std::cout << std::endl;  // Complete progress line

                ollivier_js.shutdown();

                analysis.ollivier_curvature_abs_max = std::max(
                    std::abs(analysis.ollivier_curvature_min),
                    std::abs(analysis.ollivier_curvature_max)
                );
                analysis.has_ollivier_alignment = true;

                auto ollivier_end = std::chrono::high_resolution_clock::now();
                auto ollivier_ms = std::chrono::duration<double, std::milli>(ollivier_end - ollivier_start).count();
                std::cout << "[TIMING] Ollivier-Ricci alignment: " << std::fixed << std::setprecision(1) << ollivier_ms << " ms" << std::endl;
                std::cout << "  Ollivier-Ricci aligned " << all_graphs.size() << " states" << std::endl;
                std::cout << "  PC range: [" << analysis.ollivier_pc1_min << ", " << analysis.ollivier_pc1_max << "] x ["
                          << analysis.ollivier_pc2_min << ", " << analysis.ollivier_pc2_max << "] x ["
                          << analysis.ollivier_pc3_min << ", " << analysis.ollivier_pc3_max << "]" << std::endl;
                std::cout << "  Curvature range: [" << analysis.ollivier_curvature_min
                          << ", " << analysis.ollivier_curvature_max << "]" << std::endl;
            }
        }
    }

    // Save analysis after all computations are done (only in run/generate mode)
    if (should_compute_analyses) {
        if (write_analysis(output_file, analysis)) {
            std::cout << "Saved analysis to " << output_file << std::endl;
        }

        if (no_view) {
            std::cout << "Use 'view " << output_file << "' to visualize." << std::endl;
            return 0;
        }
    }

    // Create window
    platform::WindowDesc window_desc;
    window_desc.title = "Black Hole Hausdorff Dimension";
    window_desc.width = 1600;
    window_desc.height = 900;

    auto window = platform::Window::create(window_desc);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        return 1;
    }

    // Initialize GAL
    if (!gal::initialize(gal::Backend::Vulkan)) {
        std::cerr << "Failed to initialize GAL" << std::endl;
        return 1;
    }

    gal::DeviceDesc device_desc;
    device_desc.app_name = "BlackHoleViz";
#ifdef NDEBUG
    device_desc.enable_validation = false;
#else
    device_desc.enable_validation = true;
#endif

    auto device = gal::Device::create(device_desc);
    if (!device) {
        std::cerr << "Failed to create device" << std::endl;
        gal::shutdown();
        return 1;
    }

    std::cout << "Device: " << device->get_info().device_name << std::endl;

    // Create surface
    VkInstance vk_instance = gal::get_vk_instance(device.get());
    VkSurfaceKHR surface = VK_NULL_HANDLE;

#if defined(VIZ_PLATFORM_LINUX)
    surface = gal::create_xcb_surface(vk_instance, window->get_native_display(), window->get_native_window());
#elif defined(VIZ_PLATFORM_WINDOWS)
    surface = gal::create_win32_surface(vk_instance, GetModuleHandle(nullptr), window->get_native_window());
#endif

    if (surface == VK_NULL_HANDLE) {
        std::cerr << "Failed to create surface" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    auto swapchain = device->create_swapchain(
        reinterpret_cast<gal::Handle>(surface),
        window->get_width(), window->get_height());
    if (!swapchain) {
        std::cerr << "Failed to create swapchain" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Load shaders
    auto vert_spirv = load_spirv("./shaders/spirv/basic3d.vert.spv");
    auto frag_spirv = load_spirv("./shaders/spirv/basic3d.frag.spv");
    auto cylinder_vert_spirv = load_spirv("./shaders/spirv/instance_cylinder.vert.spv");
    auto sphere_vert_spirv = load_spirv("./shaders/spirv/instance_sphere.vert.spv");

    // Picking shaders
    auto picking_sphere_vert_spirv = load_spirv("./shaders/spirv/picking_sphere.vert.spv");
    auto picking_sphere_frag_spirv = load_spirv("./shaders/spirv/picking_sphere.frag.spv");
    bool picking_shaders_available = !picking_sphere_vert_spirv.empty() && !picking_sphere_frag_spirv.empty();
    if (!picking_shaders_available) {
        std::cerr << "Warning: Picking shaders not found - vertex selection disabled" << std::endl;
    }

    if (vert_spirv.empty() || frag_spirv.empty()) {
        std::cerr << "Failed to load basic shaders" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    if (cylinder_vert_spirv.empty() || sphere_vert_spirv.empty()) {
        std::cerr << "Failed to load instanced shaders" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    gal::ShaderDesc vert_desc;
    vert_desc.stage = gal::ShaderStage::Vertex;
    vert_desc.spirv_code = vert_spirv.data();
    vert_desc.spirv_size = vert_spirv.size() * sizeof(uint32_t);
    auto vertex_shader = device->create_shader(vert_desc);

    gal::ShaderDesc frag_desc;
    frag_desc.stage = gal::ShaderStage::Fragment;
    frag_desc.spirv_code = frag_spirv.data();
    frag_desc.spirv_size = frag_spirv.size() * sizeof(uint32_t);
    auto fragment_shader = device->create_shader(frag_desc);

    gal::ShaderDesc cyl_vert_desc;
    cyl_vert_desc.stage = gal::ShaderStage::Vertex;
    cyl_vert_desc.spirv_code = cylinder_vert_spirv.data();
    cyl_vert_desc.spirv_size = cylinder_vert_spirv.size() * sizeof(uint32_t);
    auto cylinder_vertex_shader = device->create_shader(cyl_vert_desc);

    gal::ShaderDesc sph_vert_desc;
    sph_vert_desc.stage = gal::ShaderStage::Vertex;
    sph_vert_desc.spirv_code = sphere_vert_spirv.data();
    sph_vert_desc.spirv_size = sphere_vert_spirv.size() * sizeof(uint32_t);
    auto sphere_vertex_shader = device->create_shader(sph_vert_desc);

    // Picking shaders
    std::unique_ptr<gal::Shader> picking_sphere_vert_shader;
    std::unique_ptr<gal::Shader> picking_sphere_frag_shader;
    if (picking_shaders_available) {
        gal::ShaderDesc pick_vert_desc;
        pick_vert_desc.stage = gal::ShaderStage::Vertex;
        pick_vert_desc.spirv_code = picking_sphere_vert_spirv.data();
        pick_vert_desc.spirv_size = picking_sphere_vert_spirv.size() * sizeof(uint32_t);
        picking_sphere_vert_shader = device->create_shader(pick_vert_desc);

        gal::ShaderDesc pick_frag_desc;
        pick_frag_desc.stage = gal::ShaderStage::Fragment;
        pick_frag_desc.spirv_code = picking_sphere_frag_spirv.data();
        pick_frag_desc.spirv_size = picking_sphere_frag_spirv.size() * sizeof(uint32_t);
        picking_sphere_frag_shader = device->create_shader(pick_frag_desc);
    }

    // Create pipelines
    gal::VertexAttribute vertex_attribs[] = {
        {0, gal::Format::RGB32_FLOAT, 0},
        {1, gal::Format::RGBA32_FLOAT, sizeof(float) * 3},
    };

    gal::VertexBufferLayout vertex_layout;
    vertex_layout.stride = sizeof(Vertex);
    vertex_layout.step_mode = gal::VertexStepMode::Vertex;
    vertex_layout.attributes = vertex_attribs;
    vertex_layout.attribute_count = 2;

    gal::Format color_format = swapchain->get_format();
    gal::Format depth_format = gal::Format::D32_FLOAT;

    gal::BlendState opaque_blend;  // defaults to blend_enable = false

    gal::BlendState alpha_blend = gal::BlendState::alpha_blend();

    // Triangle pipeline (opaque)
    gal::RenderPipelineDesc tri_pipeline_desc;
    tri_pipeline_desc.vertex_shader = vertex_shader.get();
    tri_pipeline_desc.fragment_shader = fragment_shader.get();
    tri_pipeline_desc.vertex_layouts = &vertex_layout;
    tri_pipeline_desc.vertex_layout_count = 1;
    tri_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
    tri_pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;
    tri_pipeline_desc.depth_stencil.depth_test_enable = true;
    tri_pipeline_desc.depth_stencil.depth_write_enable = true;
    tri_pipeline_desc.depth_stencil.depth_compare = gal::CompareFunc::Less;
    tri_pipeline_desc.blend_states = &opaque_blend;
    tri_pipeline_desc.blend_state_count = 1;
    tri_pipeline_desc.color_formats = &color_format;
    tri_pipeline_desc.color_format_count = 1;
    tri_pipeline_desc.depth_format = depth_format;
    tri_pipeline_desc.push_constant_size = sizeof(math::mat4);

    auto triangle_pipeline = device->create_render_pipeline(tri_pipeline_desc);

    // Alpha-blended triangle pipeline (for transparent overlays)
    gal::RenderPipelineDesc alpha_tri_desc = tri_pipeline_desc;
    alpha_tri_desc.blend_states = &alpha_blend;
    alpha_tri_desc.depth_stencil.depth_test_enable = false;   // No depth test for 2D overlays
    alpha_tri_desc.depth_stencil.depth_write_enable = false;  // Don't write depth for overlays
    auto alpha_triangle_pipeline = device->create_render_pipeline(alpha_tri_desc);

    // Line pipeline (with alpha)
    gal::RenderPipelineDesc line_pipeline_desc = tri_pipeline_desc;
    line_pipeline_desc.topology = gal::PrimitiveTopology::LineList;
    line_pipeline_desc.blend_states = &alpha_blend;
    auto line_pipeline = device->create_render_pipeline(line_pipeline_desc);

    // ==========================================================================
    // Instanced rendering pipelines for 3D mode
    // ==========================================================================

    // Mesh vertex layout (location 0 = position, location 1 = normal)
    gal::VertexAttribute mesh_attribs[] = {
        {0, gal::Format::RGB32_FLOAT, 0},                    // position
        {1, gal::Format::RGB32_FLOAT, sizeof(float) * 3},    // normal
    };
    gal::VertexBufferLayout mesh_layout;
    mesh_layout.stride = sizeof(MeshVertex);
    mesh_layout.step_mode = gal::VertexStepMode::Vertex;
    mesh_layout.attributes = mesh_attribs;
    mesh_layout.attribute_count = 2;

    // Sphere instance layout (location 2 = center, location 3 = radius, location 4 = color)
    gal::VertexAttribute sphere_inst_attribs[] = {
        {2, gal::Format::RGB32_FLOAT, 0},                                      // center (x,y,z)
        {3, gal::Format::R32_FLOAT, sizeof(float) * 3},                        // radius
        {4, gal::Format::RGBA32_FLOAT, sizeof(float) * 4},                     // color
    };
    gal::VertexBufferLayout sphere_inst_layout;
    sphere_inst_layout.stride = sizeof(SphereInstance);
    sphere_inst_layout.step_mode = gal::VertexStepMode::Instance;
    sphere_inst_layout.attributes = sphere_inst_attribs;
    sphere_inst_layout.attribute_count = 3;

    // Cylinder instance layout (location 2 = start, location 3 = end, location 4 = radius, location 5 = color)
    gal::VertexAttribute cylinder_inst_attribs[] = {
        {2, gal::Format::RGB32_FLOAT, 0},                                      // start (x,y,z)
        {3, gal::Format::RGB32_FLOAT, sizeof(float) * 3},                      // end (x,y,z)
        {4, gal::Format::R32_FLOAT, sizeof(float) * 6},                        // radius
        {5, gal::Format::RGBA32_FLOAT, sizeof(float) * 8},                     // start_color
        {6, gal::Format::RGBA32_FLOAT, sizeof(float) * 12},                    // end_color
    };
    gal::VertexBufferLayout cylinder_inst_layout;
    cylinder_inst_layout.stride = sizeof(CylinderInstance);
    cylinder_inst_layout.step_mode = gal::VertexStepMode::Instance;
    cylinder_inst_layout.attributes = cylinder_inst_attribs;
    cylinder_inst_layout.attribute_count = 5;

    // Sphere pipeline
    gal::VertexBufferLayout sphere_layouts[] = {mesh_layout, sphere_inst_layout};
    gal::RenderPipelineDesc sphere_pipeline_desc;
    sphere_pipeline_desc.vertex_shader = sphere_vertex_shader.get();
    sphere_pipeline_desc.fragment_shader = fragment_shader.get();
    sphere_pipeline_desc.vertex_layouts = sphere_layouts;
    sphere_pipeline_desc.vertex_layout_count = 2;
    sphere_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
    sphere_pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;  // No culling for debugging
    sphere_pipeline_desc.depth_stencil.depth_test_enable = true;
    sphere_pipeline_desc.depth_stencil.depth_write_enable = true;
    sphere_pipeline_desc.depth_stencil.depth_compare = gal::CompareFunc::Less;
    sphere_pipeline_desc.blend_states = &opaque_blend;
    sphere_pipeline_desc.blend_state_count = 1;
    sphere_pipeline_desc.color_formats = &color_format;
    sphere_pipeline_desc.color_format_count = 1;
    sphere_pipeline_desc.depth_format = depth_format;
    sphere_pipeline_desc.push_constant_size = sizeof(math::mat4);
    auto sphere_pipeline = device->create_render_pipeline(sphere_pipeline_desc);

    // Cylinder pipeline
    gal::VertexBufferLayout cylinder_layouts[] = {mesh_layout, cylinder_inst_layout};
    gal::RenderPipelineDesc cylinder_pipeline_desc;
    cylinder_pipeline_desc.vertex_shader = cylinder_vertex_shader.get();
    cylinder_pipeline_desc.fragment_shader = fragment_shader.get();
    cylinder_pipeline_desc.vertex_layouts = cylinder_layouts;
    cylinder_pipeline_desc.vertex_layout_count = 2;
    cylinder_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
    cylinder_pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;  // No culling for debugging
    cylinder_pipeline_desc.depth_stencil.depth_test_enable = true;
    cylinder_pipeline_desc.depth_stencil.depth_write_enable = true;
    cylinder_pipeline_desc.depth_stencil.depth_compare = gal::CompareFunc::Less;
    cylinder_pipeline_desc.blend_states = &alpha_blend;  // Cylinders use alpha for transparency
    cylinder_pipeline_desc.blend_state_count = 1;
    cylinder_pipeline_desc.color_formats = &color_format;
    cylinder_pipeline_desc.color_format_count = 1;
    cylinder_pipeline_desc.depth_format = depth_format;
    cylinder_pipeline_desc.push_constant_size = sizeof(math::mat4);
    auto cylinder_pipeline = device->create_render_pipeline(cylinder_pipeline_desc);

    // Picking pipeline (outputs vertex index to R32_UINT texture)
    std::unique_ptr<gal::RenderPipeline> picking_sphere_pipeline;
    if (picking_shaders_available) {
        // Picking instance layout: center, radius, vertex_index
        gal::VertexAttribute picking_inst_attribs[] = {
            {2, gal::Format::RGB32_FLOAT, 0},                                      // center (x,y,z)
            {3, gal::Format::R32_FLOAT, sizeof(float) * 3},                        // radius
            {4, gal::Format::R32_UINT, sizeof(float) * 4},                         // vertex_index
        };
        gal::VertexBufferLayout picking_inst_layout;
        picking_inst_layout.stride = sizeof(PickingSphereInstance);
        picking_inst_layout.step_mode = gal::VertexStepMode::Instance;
        picking_inst_layout.attributes = picking_inst_attribs;
        picking_inst_layout.attribute_count = 3;

        gal::VertexBufferLayout picking_layouts[] = {mesh_layout, picking_inst_layout};
        gal::Format picking_color_format = gal::Format::R32_UINT;

        gal::RenderPipelineDesc picking_pipeline_desc;
        picking_pipeline_desc.vertex_shader = picking_sphere_vert_shader.get();
        picking_pipeline_desc.fragment_shader = picking_sphere_frag_shader.get();
        picking_pipeline_desc.vertex_layouts = picking_layouts;
        picking_pipeline_desc.vertex_layout_count = 2;
        picking_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
        picking_pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;
        picking_pipeline_desc.depth_stencil.depth_test_enable = true;
        picking_pipeline_desc.depth_stencil.depth_write_enable = true;
        picking_pipeline_desc.depth_stencil.depth_compare = gal::CompareFunc::Less;
        // No blending for integer output
        picking_pipeline_desc.blend_state_count = 0;
        picking_pipeline_desc.color_formats = &picking_color_format;
        picking_pipeline_desc.color_format_count = 1;
        picking_pipeline_desc.depth_format = depth_format;
        picking_pipeline_desc.push_constant_size = sizeof(math::mat4);
        picking_sphere_pipeline = device->create_render_pipeline(picking_pipeline_desc);
    }

    // Generate meshes
    Mesh cylinder_mesh = generate_cylinder_mesh(16);
    Mesh sphere_mesh = generate_sphere_mesh(16, 8);

    // Create depth texture
    auto create_depth_texture = [&](uint32_t w, uint32_t h) {
        gal::TextureDesc depth_desc;
        depth_desc.size = {w, h, 1};
        depth_desc.format = depth_format;
        depth_desc.usage = gal::TextureUsage::DepthStencil;
        depth_desc.sample_count = 1;
        return device->create_texture(depth_desc);
    };

    auto depth_texture = create_depth_texture(window->get_width(), window->get_height());

    // Create buffers (16MB initial size to handle large graphs)
    size_t buffer_size = 16 * 1024 * 1024;
    gal::BufferDesc buffer_desc;
    buffer_desc.size = buffer_size;
    buffer_desc.usage = gal::BufferUsage::Vertex;
    buffer_desc.memory = gal::MemoryLocation::CPU_TO_GPU;

    auto vertex_buffer = device->create_buffer(buffer_desc);
    auto edge_buffer = device->create_buffer(buffer_desc);
    auto horizon_buffer = device->create_buffer(buffer_desc);
    auto timeline_buffer = device->create_buffer(buffer_desc);
    auto overlay_buffer = device->create_buffer(buffer_desc);  // For help overlay background

    // Mesh buffers (static, uploaded once)
    gal::BufferDesc mesh_vb_desc;
    mesh_vb_desc.usage = gal::BufferUsage::Vertex;
    mesh_vb_desc.memory = gal::MemoryLocation::CPU_TO_GPU;

    mesh_vb_desc.size = cylinder_mesh.vertices.size() * sizeof(MeshVertex);
    auto cylinder_vb = device->create_buffer(mesh_vb_desc);
    cylinder_vb->write(cylinder_mesh.vertices.data(), mesh_vb_desc.size);

    mesh_vb_desc.size = sphere_mesh.vertices.size() * sizeof(MeshVertex);
    auto sphere_vb = device->create_buffer(mesh_vb_desc);
    sphere_vb->write(sphere_mesh.vertices.data(), mesh_vb_desc.size);

    gal::BufferDesc mesh_ib_desc;
    mesh_ib_desc.usage = gal::BufferUsage::Index;
    mesh_ib_desc.memory = gal::MemoryLocation::CPU_TO_GPU;

    mesh_ib_desc.size = cylinder_mesh.indices.size() * sizeof(Index);
    auto cylinder_ib = device->create_buffer(mesh_ib_desc);
    cylinder_ib->write(cylinder_mesh.indices.data(), mesh_ib_desc.size);

    mesh_ib_desc.size = sphere_mesh.indices.size() * sizeof(Index);
    auto sphere_ib = device->create_buffer(mesh_ib_desc);
    sphere_ib->write(sphere_mesh.indices.data(), mesh_ib_desc.size);

    // Instance buffers (dynamic, updated per frame)
    gal::BufferDesc instance_buffer_desc;
    instance_buffer_desc.usage = gal::BufferUsage::Vertex;
    instance_buffer_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
    instance_buffer_desc.size = buffer_size;  // Start with 2MB, resize if needed

    auto sphere_instance_buffer = device->create_buffer(instance_buffer_desc);
    auto cylinder_instance_buffer = device->create_buffer(instance_buffer_desc);

    // Picking resources
    std::unique_ptr<gal::Texture> picking_texture;
    std::unique_ptr<gal::Texture> picking_depth_texture;
    std::unique_ptr<gal::Buffer> picking_instance_buffer;
    std::unique_ptr<gal::Buffer> picking_readback_buffer;

    auto create_picking_resources = [&](uint32_t w, uint32_t h) {
        if (!picking_shaders_available) return;

        // Picking color texture (R32_UINT)
        gal::TextureDesc pick_color_desc;
        pick_color_desc.size = {w, h, 1};
        pick_color_desc.format = gal::Format::R32_UINT;
        pick_color_desc.usage = gal::TextureUsage::RenderTarget;
        pick_color_desc.sample_count = 1;
        picking_texture = device->create_texture(pick_color_desc);

        // Picking depth texture
        gal::TextureDesc pick_depth_desc;
        pick_depth_desc.size = {w, h, 1};
        pick_depth_desc.format = depth_format;
        pick_depth_desc.usage = gal::TextureUsage::DepthStencil;
        pick_depth_desc.sample_count = 1;
        picking_depth_texture = device->create_texture(pick_depth_desc);

        // Readback buffer (single pixel)
        gal::BufferDesc readback_desc;
        readback_desc.size = sizeof(uint32_t);
        readback_desc.usage = gal::BufferUsage::TransferDst;
        readback_desc.memory = gal::MemoryLocation::GPU_TO_CPU;
        picking_readback_buffer = device->create_buffer(readback_desc);

        // Instance buffer for picking (reuse same size as sphere instance buffer)
        gal::BufferDesc pick_inst_desc;
        pick_inst_desc.size = buffer_size;
        pick_inst_desc.usage = gal::BufferUsage::Vertex;
        pick_inst_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        picking_instance_buffer = device->create_buffer(pick_inst_desc);
    };

    create_picking_resources(window->get_width(), window->get_height());

    // Picking state
    bool picking_pass_pending = false;
    bool picking_readback_pending = false;  // True after picking pass submitted, waiting for GPU
    int picking_click_x = 0, picking_click_y = 0;
    VertexSelectionState vertex_selection;
    std::vector<VertexId> picking_vertex_ids;  // Maps instance index to vertex ID

    // Camera - compute distance from bounding radius for good initial framing
    // Multiply by ~2.5 to fit the graph nicely in view with some margin
    float base_camera_distance = analysis.layout_bounding_radius * 2.5f;
    float camera_distance_2d = base_camera_distance * 0.8f;  // 2D needs less distance (orthographic)
    float camera_distance_3d = base_camera_distance;          // 3D uses full distance

    // Shape space camera distance (based on PC coordinate range, not physical layout)
    float shape_space_camera_distance = 10.0f;  // Default
    if (analysis.has_branch_alignment) {
        float pc_range = std::max({
            analysis.global_pc1_max - analysis.global_pc1_min,
            analysis.global_pc2_max - analysis.global_pc2_min,
            analysis.global_pc3_max - analysis.global_pc3_min
        });
        shape_space_camera_distance = std::max(1.0f, pc_range * 1.5f);
        std::cout << "Shape space PC range: " << pc_range
                  << ", camera distance: " << shape_space_camera_distance << std::endl;
    }

    camera::PerspectiveCamera cam;
    cam.set_perspective(60.0f, static_cast<float>(window->get_width()) / window->get_height(), 0.1f, base_camera_distance * 10.0f);
    cam.set_target(math::vec3(0, 0, 0));
    cam.set_distance(camera_distance_3d);  // Start in 3D
    cam.orbit(0.5f, -0.8f);  // 3D default angle

    camera::CameraController controller(&cam);

    // Sync objects
    auto image_semaphore = device->create_semaphore();
    auto render_semaphore = device->create_semaphore();
    auto fence = device->create_fence(true);

    // Command buffer for in-flight frame
    std::unique_ptr<gal::CommandEncoder> in_flight_cmd;

    // ==========================================================================
    // MSAA setup
    // ==========================================================================
    uint32_t max_msaa = device->get_info().limits.max_samples;
    std::vector<uint32_t> supported_msaa_levels = {1};  // 1 = off
    for (uint32_t s = 2; s <= max_msaa; s *= 2) {
        supported_msaa_levels.push_back(s);
    }
    // Default to maximum available MSAA
    size_t msaa_level_index = supported_msaa_levels.size() - 1;
    uint32_t msaa_samples = supported_msaa_levels[msaa_level_index];
    bool msaa_enabled = (msaa_samples > 1);
    bool msaa_dirty = true;
    std::cout << "MSAA: " << (msaa_enabled ? std::to_string(msaa_samples) + "x" : "OFF")
              << " (max supported: " << max_msaa << "x)" << std::endl;

    std::unique_ptr<gal::Texture> msaa_color_texture;
    std::unique_ptr<gal::Texture> msaa_depth_texture;
    std::unique_ptr<gal::RenderPipeline> msaa_triangle_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_alpha_triangle_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_line_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_sphere_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_cylinder_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_text_pipeline;

    // Function pointer for creating text pipeline (set after text setup)
    std::function<std::unique_ptr<gal::RenderPipeline>(uint32_t)> create_text_pipeline_fn;

    auto create_msaa_resources = [&](uint32_t width, uint32_t height) {
        // Depth texture (always needed, sample count matches MSAA state)
        gal::TextureDesc depth_desc;
        depth_desc.size = {width, height, 1};
        depth_desc.format = depth_format;
        depth_desc.usage = gal::TextureUsage::DepthStencil;
        depth_desc.sample_count = msaa_enabled ? msaa_samples : 1;
        msaa_depth_texture = device->create_texture(depth_desc);

        if (!msaa_enabled) {
            msaa_color_texture.reset();
            msaa_triangle_pipeline.reset();
            msaa_alpha_triangle_pipeline.reset();
            msaa_line_pipeline.reset();
            msaa_sphere_pipeline.reset();
            msaa_cylinder_pipeline.reset();
            msaa_text_pipeline.reset();
            return;
        }

        // MSAA color texture
        gal::TextureDesc msaa_desc;
        msaa_desc.size = {width, height, 1};
        msaa_desc.format = swapchain->get_format();
        msaa_desc.usage = gal::TextureUsage::RenderTarget;
        msaa_desc.sample_count = msaa_samples;
        msaa_color_texture = device->create_texture(msaa_desc);

        if (!msaa_color_texture) {
            std::cerr << "Failed to create MSAA texture, falling back to no MSAA" << std::endl;
            msaa_enabled = false;
            return;
        }

        // MSAA triangle pipeline
        gal::RenderPipelineDesc msaa_tri_desc = tri_pipeline_desc;
        msaa_tri_desc.multisample.count = msaa_samples;
        msaa_triangle_pipeline = device->create_render_pipeline(msaa_tri_desc);

        // MSAA alpha-blended triangle pipeline (for transparent overlays with MSAA)
        gal::RenderPipelineDesc msaa_alpha_tri_desc = alpha_tri_desc;
        msaa_alpha_tri_desc.multisample.count = msaa_samples;
        msaa_alpha_triangle_pipeline = device->create_render_pipeline(msaa_alpha_tri_desc);

        // MSAA line pipeline
        gal::RenderPipelineDesc msaa_line_desc = line_pipeline_desc;
        msaa_line_desc.multisample.count = msaa_samples;
        msaa_line_pipeline = device->create_render_pipeline(msaa_line_desc);

        // MSAA sphere pipeline
        gal::RenderPipelineDesc msaa_sphere_desc = sphere_pipeline_desc;
        msaa_sphere_desc.multisample.count = msaa_samples;
        msaa_sphere_pipeline = device->create_render_pipeline(msaa_sphere_desc);

        // MSAA cylinder pipeline
        gal::RenderPipelineDesc msaa_cyl_desc = cylinder_pipeline_desc;
        msaa_cyl_desc.multisample.count = msaa_samples;
        msaa_cylinder_pipeline = device->create_render_pipeline(msaa_cyl_desc);

        // MSAA text pipeline (if text rendering is set up)
        if (create_text_pipeline_fn) {
            msaa_text_pipeline = create_text_pipeline_fn(msaa_samples);
        }

        if (!msaa_triangle_pipeline || !msaa_line_pipeline ||
            !msaa_sphere_pipeline || !msaa_cylinder_pipeline) {
            std::cerr << "Failed to create MSAA pipelines, falling back to no MSAA" << std::endl;
            msaa_enabled = false;
            msaa_color_texture.reset();
        }
    };

    // ==========================================================================
    // Text Rendering Setup
    // ==========================================================================

    // Load text shaders
    auto text_vert_spirv = load_spirv("./shaders/spirv/text.vert.spv");
    auto text_frag_spirv = load_spirv("./shaders/spirv/text.frag.spv");
    bool text_rendering_available = !text_vert_spirv.empty() && !text_frag_spirv.empty();
    std::unique_ptr<gal::Shader> text_vertex_shader;
    std::unique_ptr<gal::Shader> text_fragment_shader;
    std::unique_ptr<gal::Texture> font_texture;
    std::unique_ptr<gal::Sampler> font_sampler;
    std::unique_ptr<gal::BindGroupLayout> text_bind_group_layout;
    std::unique_ptr<gal::BindGroup> text_bind_group;
    std::unique_ptr<gal::RenderPipeline> text_pipeline;
    std::unique_ptr<gal::Buffer> text_quad_vb;
    std::unique_ptr<gal::Buffer> text_quad_ib;
    std::unique_ptr<gal::Buffer> text_instance_buffer;
    const size_t MAX_TEXT_GLYPHS = 8192;  // Maximum glyphs per frame

    if (text_rendering_available) {
        // Create text shaders
        gal::ShaderDesc text_vert_desc;
        text_vert_desc.stage = gal::ShaderStage::Vertex;
        text_vert_desc.spirv_code = text_vert_spirv.data();
        text_vert_desc.spirv_size = text_vert_spirv.size() * sizeof(uint32_t);
        text_vertex_shader = device->create_shader(text_vert_desc);

        gal::ShaderDesc text_frag_desc;
        text_frag_desc.stage = gal::ShaderStage::Fragment;
        text_frag_desc.spirv_code = text_frag_spirv.data();
        text_frag_desc.spirv_size = text_frag_spirv.size() * sizeof(uint32_t);
        text_fragment_shader = device->create_shader(text_frag_desc);

        if (!text_vertex_shader || !text_fragment_shader) {
            std::cerr << "Failed to create text shaders" << std::endl;
            text_rendering_available = false;
        }
    }

    if (text_rendering_available) {
        // Generate and create font atlas texture
        auto font_atlas_data = generate_font_atlas();

        gal::TextureDesc font_tex_desc;
        font_tex_desc.size = {FONT_ATLAS_WIDTH, FONT_ATLAS_HEIGHT, 1};
        font_tex_desc.format = gal::Format::R8_UNORM;
        font_tex_desc.usage = gal::TextureUsage::Sampled;
        font_tex_desc.initial_data = font_atlas_data.data();
        font_tex_desc.debug_name = "FontAtlas";
        font_texture = device->create_texture(font_tex_desc);

        // Create font sampler (nearest for crisp text)
        gal::SamplerDesc font_sampler_desc;
        font_sampler_desc.mag_filter = gal::Filter::Nearest;
        font_sampler_desc.min_filter = gal::Filter::Nearest;
        font_sampler_desc.address_u = gal::AddressMode::ClampToEdge;
        font_sampler_desc.address_v = gal::AddressMode::ClampToEdge;
        font_sampler = device->create_sampler(font_sampler_desc);

        if (!font_texture || !font_sampler) {
            std::cerr << "Failed to create font resources" << std::endl;
            text_rendering_available = false;
        }
    }

    if (text_rendering_available) {
        // Create bind group layout for font texture
        gal::BindGroupLayoutEntry layout_entry = {
            0,                                      // binding
            gal::ShaderStage::Fragment,             // visibility
            gal::BindingType::CombinedTextureSampler,  // type
            1                                       // count
        };
        gal::BindGroupLayoutDesc layout_desc;
        layout_desc.entries = &layout_entry;
        layout_desc.entry_count = 1;
        layout_desc.debug_name = "TextBindGroupLayout";
        text_bind_group_layout = device->create_bind_group_layout(layout_desc);

        if (text_bind_group_layout) {
            // Create bind group
            gal::BindGroupEntry bind_entry = {
                0, nullptr, 0, 0, font_texture.get(), font_sampler.get()
            };
            gal::BindGroupDesc bind_desc;
            bind_desc.layout = text_bind_group_layout.get();
            bind_desc.entries = &bind_entry;
            bind_desc.entry_count = 1;
            bind_desc.debug_name = "TextBindGroup";
            text_bind_group = device->create_bind_group(bind_desc);
        }

        if (!text_bind_group_layout || !text_bind_group) {
            std::cerr << "Failed to create text bind group" << std::endl;
            text_rendering_available = false;
        }
    }

    if (text_rendering_available) {
        // Quad vertex data: position (x, y) and UV (u, v)
        // Two triangles forming a unit quad [0,1] x [0,1]
        struct TextQuadVertex {
            float x, y;   // position
            float u, v;   // texture coords
        };
        TextQuadVertex quad_vertices[4] = {
            {0.0f, 0.0f, 0.0f, 0.0f},  // vertex 0: pos(0,0), uv(0,0)
            {1.0f, 0.0f, 1.0f, 0.0f},  // vertex 1: pos(1,0), uv(1,0)
            {1.0f, 1.0f, 1.0f, 1.0f},  // vertex 2: pos(1,1), uv(1,1)
            {0.0f, 1.0f, 0.0f, 1.0f},  // vertex 3: pos(0,1), uv(0,1)
        };
        uint16_t quad_indices[6] = {0, 1, 2, 0, 2, 3};

        gal::BufferDesc quad_vb_desc;
        quad_vb_desc.size = sizeof(quad_vertices);
        quad_vb_desc.usage = gal::BufferUsage::Vertex;
        quad_vb_desc.memory = gal::MemoryLocation::CPU_TO_GPU;  // Need CPU_TO_GPU for initial_data
        quad_vb_desc.initial_data = quad_vertices;
        text_quad_vb = device->create_buffer(quad_vb_desc);

        gal::BufferDesc quad_ib_desc;
        quad_ib_desc.size = sizeof(quad_indices);
        quad_ib_desc.usage = gal::BufferUsage::Index;
        quad_ib_desc.memory = gal::MemoryLocation::CPU_TO_GPU;  // Need CPU_TO_GPU for initial_data
        quad_ib_desc.initial_data = quad_indices;
        text_quad_ib = device->create_buffer(quad_ib_desc);

        gal::BufferDesc instance_desc;
        instance_desc.size = MAX_TEXT_GLYPHS * sizeof(GlyphInstance);
        instance_desc.usage = gal::BufferUsage::Vertex;
        instance_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        text_instance_buffer = device->create_buffer(instance_desc);

        if (!text_quad_vb || !text_quad_ib || !text_instance_buffer) {
            std::cerr << "Failed to create text buffers" << std::endl;
            text_rendering_available = false;
        }
    }

    // Text vertex layouts (defined outside if block for persistence)
    gal::VertexAttribute text_quad_attribs[] = {
        {0, gal::Format::RG32_FLOAT, 0},                    // position
        {1, gal::Format::RG32_FLOAT, sizeof(float) * 2},    // uv
    };
    gal::VertexBufferLayout text_quad_layout;
    text_quad_layout.stride = sizeof(float) * 4;
    text_quad_layout.step_mode = gal::VertexStepMode::Vertex;
    text_quad_layout.attributes = text_quad_attribs;
    text_quad_layout.attribute_count = 2;

    gal::VertexAttribute text_inst_attribs[] = {
        {2, gal::Format::RG32_FLOAT, offsetof(GlyphInstance, x)},       // pos
        {3, gal::Format::RGBA32_FLOAT, offsetof(GlyphInstance, u_min)}, // uv_rect
        {4, gal::Format::RGBA32_FLOAT, offsetof(GlyphInstance, r)},     // color
    };
    gal::VertexBufferLayout text_inst_layout;
    text_inst_layout.stride = sizeof(GlyphInstance);
    text_inst_layout.step_mode = gal::VertexStepMode::Instance;
    text_inst_layout.attributes = text_inst_attribs;
    text_inst_layout.attribute_count = 3;

    gal::VertexBufferLayout text_layouts[] = {text_quad_layout, text_inst_layout};

    // Lambda to create text pipeline with given sample count
    auto create_text_pipeline = [&](uint32_t sample_count) -> std::unique_ptr<gal::RenderPipeline> {
        if (!text_vertex_shader || !text_fragment_shader || !text_bind_group_layout) {
            return nullptr;
        }
        gal::RenderPipelineDesc desc;
        desc.vertex_shader = text_vertex_shader.get();
        desc.fragment_shader = text_fragment_shader.get();
        desc.vertex_layouts = text_layouts;
        desc.vertex_layout_count = 2;
        desc.topology = gal::PrimitiveTopology::TriangleList;
        desc.rasterizer.cull_mode = gal::CullMode::None;
        desc.depth_stencil.depth_test_enable = false;
        desc.depth_stencil.depth_write_enable = false;
        desc.blend_states = &alpha_blend;
        desc.blend_state_count = 1;
        desc.color_formats = &color_format;
        desc.color_format_count = 1;
        desc.depth_format = depth_format;
        desc.push_constant_size = sizeof(float) * 4;
        const gal::BindGroupLayout* bgl = text_bind_group_layout.get();
        desc.bind_group_layouts = &bgl;
        desc.bind_group_layout_count = 1;
        desc.multisample.count = sample_count;
        return device->create_render_pipeline(desc);
    };

    if (text_rendering_available) {
        // Store the lambda for MSAA pipeline recreation
        create_text_pipeline_fn = create_text_pipeline;

        text_pipeline = create_text_pipeline(1);
        if (!text_pipeline) {
            std::cerr << "Failed to create text pipeline" << std::endl;
            text_rendering_available = false;
        }
    }

    if (text_rendering_available) {
        std::cout << "Text rendering: AVAILABLE" << std::endl;
    } else {
        std::cout << "Text rendering: NOT AVAILABLE (overlay will be disabled)" << std::endl;
    }

    // Glyph instance buffer for text overlay (rebuilt each frame)
    std::vector<GlyphInstance> glyph_instances;
    glyph_instances.reserve(MAX_TEXT_GLYPHS);

    // Helper to queue a single glyph
    auto queue_glyph = [&](float x, float y, char c, const Vec4& color) {
        if (c < FONT_FIRST_CHAR || c > FONT_LAST_CHAR) c = '?';
        int char_idx = c - FONT_FIRST_CHAR;
        int col = char_idx % FONT_ATLAS_COLS;
        int row = char_idx / FONT_ATLAS_COLS;

        // Round screen position to whole pixels for pixel-perfect text rendering
        x = std::floor(x);
        y = std::floor(y);

        // UV coordinates - no half-texel inset needed with nearest sampling
        // The fragment shader receives fragments at pixel centers (not edges), so the
        // UV interpolation naturally samples the correct texels.
        // For N screen pixels covering M texels with nearest sampling:
        //   - Fragment at pixel center i+0.5 gets in_uv = (i+0.5)/N
        //   - UV = u_min + in_uv * (u_max - u_min) maps to the appropriate texel
        //   - Nearest sampling selects texel floor(UV * atlas_width)
        // With in_uv ranging from 0.5/N to (N-0.5)/N, we never hit exact texel boundaries.
        float u_min = static_cast<float>(col * FONT_CHAR_WIDTH) / FONT_ATLAS_WIDTH;
        float u_max = static_cast<float>((col + 1) * FONT_CHAR_WIDTH) / FONT_ATLAS_WIDTH;
        float v_top = static_cast<float>(row * FONT_CHAR_HEIGHT) / FONT_ATLAS_HEIGHT;
        float v_bot = static_cast<float>((row + 1) * FONT_CHAR_HEIGHT) / FONT_ATLAS_HEIGHT;

        // The quad has UV (0,0) at position (0,0) which becomes screen top-left after Y-flip.
        // After the Y-flip in NDC, the quad's position (0,0) is at screen TOP.
        // The quad's UV (0,0) samples inst_uv_rect.xy.
        // We want screen TOP to sample atlas TOP (v_top).
        // But the position Y increases DOWNWARD in screen coords, so position (0,1) is at screen BOTTOM.
        // This means in_uv.y=1 should sample atlas BOTTOM (v_bot).
        // So: inst_uv_rect.y = v_top, inst_uv_rect.w = v_bot
        glyph_instances.push_back({x, y, u_min, v_top, u_max, v_bot, color.x, color.y, color.z, color.w});
    };

    // Helper to queue a string of text
    auto queue_text = [&](float x, float y, const std::string& text, const Vec4& color, float scale = 1.0f) {
        float char_w = FONT_CHAR_WIDTH * scale;
        for (size_t i = 0; i < text.size(); ++i) {
            queue_glyph(x + i * char_w, y, text[i], color);
        }
    };

    // Application state
    int current_step = 0;
    int max_step = static_cast<int>(analysis.per_timestep.size()) - 1;
    bool playing = false;
    float playback_speed = 2.0f;  // Steps per second
    int playback_direction = 1;  // 1 = forward, -1 = backward
    bool ping_pong_mode = false; // Reverse direction at boundaries

    // Time-based playback with latch
    // When play starts or direction changes, we record the start time and step
    // Current step is computed as: latch_step + elapsed_time * speed * direction
    std::chrono::high_resolution_clock::time_point play_latch_time;
    int play_latch_step = 0;

    // Helper to reset the playback latch (call when play starts or direction changes)
    auto reset_playback_latch = [&]() {
        play_latch_time = std::chrono::high_resolution_clock::now();
        play_latch_step = current_step;
    };

    // UI interaction state
    bool scrubber_dragging = false;
    TimelineData last_timeline;  // Store for hit testing
    int window_width = static_cast<int>(window->get_width());
    int window_height = static_cast<int>(window->get_height());

    ViewMode view_mode = ViewMode::View3D;
    EdgeDisplayMode edge_mode = EdgeDisplayMode::Union;
    int edge_freq_threshold = 1;   // Edges must appear in >= N states (for Frequent mode, 1 = show all)
    int selected_state_index = 0;  // Which state to show in SingleState mode (cycles with '[' and ']')

    // Shape space (curvature) view state
    ShapeSpaceColorMode shape_space_color_mode = ShapeSpaceColorMode::Curvature;
    ShapeSpaceDisplayMode shape_space_display_mode = ShapeSpaceDisplayMode::Merged;
    int highlighted_branch = -1;  // -1 = all, 0+ = specific branch
    bool use_ollivier_alignment = false;  // false = Wolfram-Ricci (K=2-d), true = Ollivier-Ricci
    // Centralized mode configuration (value type, statistic, scope, aggregation)
    ModeConfig viz_mode;
    bool per_frame_normalization = false;  // Per-frame (local) vs global color normalization (default: global)
    bool show_horizons = false;  // Horizon rings off by default
    bool show_geodesics = false;  // Geodesic path overlay (requires geodesic analysis)
    bool show_defects = false;    // Topological defect markers (requires particle analysis)
    int curvature_display_mode = 0;  // 0=OFF, 1=Ollivier-Ricci, 2=Dimension Gradient
    bool show_entropy = false;       // Entropy heatmap overlay (requires --entropy)
    bool show_hilbert = false;       // Hilbert space stats overlay
    bool timeslice_enabled = false;  // Timeslice view (shows range of timesteps)
    int timeslice_width = 5;         // Number of timesteps in the slice
    bool z_mapping_enabled = true;   // Map bucket dimension values to Z/height
    MissingDataMode missing_mode = MissingDataMode::Show;  // How to display missing dimension data
    ColorPalette current_palette = ColorPalette::Temperature;  // Color palette for heatmap
    EdgeColorMode edge_color_mode = EdgeColorMode::Vertex;  // How to color edges
    bool show_states_graph = false;  // Show multiway states graph panel (toggle with B)
    bool show_multispace = false;     // Show multispace probability overlay (toggle with X)
    bool path_selection_enabled = false;  // Filter to N random paths/states (toggle with A)
    int path_selection_count = 1;         // Number of paths/states to select
    std::vector<std::vector<int>> selected_state_indices;  // Per-timestep selected state indices
    bool show_overlay = false;       // Show help overlay (toggle with / or ?)
    bool show_scatter = false;       // Show scatter plot overlay (toggle with Y)
    ValueType scatter_metric = ValueType::WolframHausdorffDimension;  // Which metric to display
    float z_scale = 3.0f;            // Z height multiplier (adjusted with < / >)
    bool geometry_dirty = true;
    bool layout_enabled = true;  // Live layout on - seeds from pre-computed step 0 positions
    bool layout_use_all_vertices = false;  // Layout ALL union vertices regardless of edge display mode
    int layout_step = -1;        // Which step the layout is currently for
    std::vector<VertexId> layout_vertices;  // Vertex IDs for current layout (for seeding next step)
    PositionCache global_position_cache;    // Persistent positions across all timesteps (prevents snap on loop)

    // ==========================================================================
    // Keybinding Definitions (unified for stdout and overlay)
    // ==========================================================================

    // Helper to convert edge mode to string
    auto edge_mode_str = [&]() -> std::string {
        switch (edge_mode) {
            case EdgeDisplayMode::Union: return "Union";
            case EdgeDisplayMode::Frequent: return "Freq(>=" + std::to_string(edge_freq_threshold) + ")";
            case EdgeDisplayMode::Intersection: return "Intersection";
            case EdgeDisplayMode::SingleState: return "SingleState";
            default: return "?";
        }
    };

    std::vector<KeyBindingGroup> keybindings = {
        {"Navigation", {
            {"Mouse drag", "Rotate camera", nullptr},
            {"Scroll", "Zoom in/out", nullptr},
            {"Arrow keys", "Step through timesteps", nullptr},
            {"Home/End", "Jump to first/last", nullptr},
        }},
        {"Playback", {
            {"Space", "Play/pause", [&]{ return playing ? "PLAYING" : "PAUSED"; }},
            {"[ / ]", "Playback speed", [&]{ return std::to_string(playback_speed).substr(0, 4) + "x"; }},
            {"\\", "Reverse direction", [&]{ return playback_direction > 0 ? "FORWARD" : "REVERSE"; }},
            {"P", "Ping-pong mode", [&]{ return ping_pong_mode ? "ON" : "OFF"; }},
        }},
        {"Dimension Mode", {
            {"G", "Graph scope", [&]{ return ui::name(viz_mode.graph_scope); }},
            {"V", "Value type", [&]{ return value_type_name(viz_mode.value_type); }},
            {"M", "Statistic", [&]{
                if (!viz_mode.supports_variance()) return "(N/A)";
                return ui::name(viz_mode.statistic_mode);
            }},
            {"N", "Normalization", [&]{
                if (viz_mode.graph_scope == GraphScope::Global) return "(N/A)";
                return per_frame_normalization ? "Local" : "Global";
            }},
            {"B", "Value aggregation", [&]{
                if (!viz_mode.supports_value_aggregation()) return "(N/A)";
                return ui::name(viz_mode.value_aggregation);
            }},
        }},
        {"Display", {
            {"2 / 3", "View mode", [&]{ return view_mode == ViewMode::View2D ? "2D" : "3D"; }},
            {"Z", "Z/depth mapping", [&]{ return z_mapping_enabled ? "ON" : "OFF"; }},
            {"< / >", "Z scale", [&]{ return std::to_string(z_scale).substr(0, 4); }},
            {"I", "Edge filter", edge_mode_str},
            {"{ / }", "Freq threshold", [&]{ return ">=" + std::to_string(edge_freq_threshold); }},
            {"O", "Edge coloring", [&]{ return edge_color_mode == EdgeColorMode::Vertex ? "Vertex" : "Frequency"; }},
            {"PgUp/PgDn", "Cycle states", [&]{ return std::to_string(selected_state_index); }},
            {"H", "Horizon rings", [&]{ return show_horizons ? "ON" : "OFF"; }},
            {"J", "Geodesic paths", [&]{ return show_geodesics ? "ON" : "OFF"; }},
            {"K", "Defect markers", [&]{ return show_defects ? "ON" : "OFF"; }},
            {"D", "Curvature mode", [&]{
                if (!has_curvature_analysis) return std::string("(N/A)");
                switch (curvature_display_mode) {
                    case 0: return std::string("OFF");
                    case 1: return std::string("Ollivier-Ricci");
                    case 2: return std::string("Dim Gradient");
                    default: return std::string("?");
                }
            }},
            {"E", "Entropy heatmap", [&]{ return !has_entropy_analysis ? "(N/A)" : (show_entropy ? "ON" : "OFF"); }},
            {"Q", "Hilbert space", [&]{ return !has_hilbert_analysis ? "(N/A)" : (show_hilbert ? "ON" : "OFF"); }},
            {"T", "Timeslice view", [&]{
                if (viz_mode.graph_scope == GraphScope::Global) return std::string("(N/A)");
                return timeslice_enabled ? "ON (w=" + std::to_string(timeslice_width) + ")" : std::string("OFF");
            }},
            {"- / +", "Timeslice width", [&]{
                if (viz_mode.graph_scope == GraphScope::Global) return std::string("(N/A)");
                return std::to_string(timeslice_width);
            }},
            {"U", "Missing data mode", [&]{ return missing_mode_name(missing_mode); }},
            {"C", "Color palette", [&]{ return palette_name(current_palette); }},
            {"L", "Dynamic layout", [&]{ return layout_enabled ? "ON" : "OFF"; }},
            {"Shift+L", "Layout all verts", [&]{ return layout_use_all_vertices ? "ON" : "OFF"; }},
            {"M", "MSAA level", [&]{ return msaa_enabled ? std::to_string(msaa_samples) + "x" : "OFF"; }},
            {"R", "Reset camera", nullptr},
            {"B", "States graph (shape space)", [&]{
                if (view_mode != ViewMode::ShapeSpace2D && view_mode != ViewMode::ShapeSpace3D) return "(N/A)";
                return show_states_graph ? "ON" : "OFF";
            }},
            {"X", "Multispace overlay", [&]{ return show_multispace ? "ON" : "OFF"; }},
            {"Y", "Scatter plot", [&]{
                if (!show_scatter) return std::string("OFF");
                return std::string(value_type_name(scatter_metric));
            }},
            {"Shift+Y", "Cycle scatter metric", nullptr},
            {"A", "Path selection", [&]{
                if (viz_mode.graph_scope != GraphScope::Branchial) return std::string("(N/A)");
                return path_selection_enabled ? "N=" + std::to_string(path_selection_count) : std::string("OFF");
            }},
            {"9 / 0", "Path count -/+", [&]{
                if (viz_mode.graph_scope != GraphScope::Branchial) return std::string("(N/A)");
                return std::to_string(path_selection_count);
            }},
        }},
        {"Help", {
            {"/ or ?", "Toggle this overlay", [&]{ return show_overlay ? "VISIBLE" : "HIDDEN"; }},
            {"ESC", "Exit", nullptr},
        }},
    };

    // Layout engine for dynamic graph layout
    auto layout_engine = create_layout_engine(LayoutBackend::CPU);
    LayoutGraph layout_graph;
    LayoutParams layout_params;
    layout_params.algorithm = LayoutAlgorithm::Direct;
    layout_params.dimension = layout_3d ? LayoutDimension::Layout3D : LayoutDimension::Layout2D;
    if (layout_3d) {
        std::cout << "Using 3D force-directed layout" << std::endl;
    }
    if (embed_3d && analysis.initial.has_3d()) {
        std::cout << "Using 3D topological embedding (" << analysis.initial.vertex_z.size() << " Z coords)" << std::endl;
        // Debug: print some Z values
        if (analysis.initial.vertex_z.size() > 0) {
            float min_z = analysis.initial.vertex_z[0], max_z = analysis.initial.vertex_z[0];
            for (float z : analysis.initial.vertex_z) {
                min_z = std::min(min_z, z);
                max_z = std::max(max_z, z);
            }
            std::cout << "  Z range: [" << min_z << ", " << max_z << "]" << std::endl;
        }
    } else if (embed_3d) {
        std::cout << "WARNING: --embed-3d specified but initial.vertex_z is empty!" << std::endl;
    }
    layout_params.spring_constant = 0.5f;    // Strong springs to maintain edge lengths
    layout_params.repulsion_constant = 0.25f; // Weak repulsion to prevent expansion
    layout_params.damping = 0.1f;            // High damping for stability
    layout_params.gravity = 0.1f;            // Pull toward center to prevent drift
    layout_params.max_displacement = 0.01f;  // Small steps for smooth animation
    layout_params.edge_budget = 2000;        // Limit spring updates for large graphs

    RenderData render_data;
    std::vector<Vertex> horizon_verts;

    // Build horizon visualization
    if (!analysis.initial.vertex_positions.empty()) {
        horizon_verts = build_horizon_circles(analysis.initial.config);
        if (horizon_buffer && !horizon_verts.empty()) {
            horizon_buffer->write(horizon_verts.data(), horizon_verts.size() * sizeof(Vertex));
        }
    }

    // Input state
    bool left_mouse_down = false;
    float last_mouse_x = 0, last_mouse_y = 0;
    bool should_resize = false;
    uint32_t new_width = 0, new_height = 0;

    auto last_frame = std::chrono::high_resolution_clock::now();

    // Unified highlight computation function - called from click handler, V key, and timestep change
    // Having a single function ensures consistent results regardless of when/how highlight is computed
    auto recompute_highlight = [&](VertexId vid, int timestep) {
        if (timestep < 0 || timestep >= static_cast<int>(analysis.per_timestep.size())) {
            vertex_selection.show_highlight = false;
            return;
        }

        const auto& ts = analysis.per_timestep[timestep];
        blackhole::SimpleGraph graph;
        graph.build(ts.union_vertices, ts.union_edges);

        if (!graph.has_vertex(vid)) {
            vertex_selection.show_highlight = false;
            return;
        }

        // Clear all highlight data
        vertex_selection.ball_distances.clear();
        vertex_selection.tube_vertex_distances.clear();
        vertex_selection.tube_geodesic_paths.clear();
        vertex_selection.tube_radii.clear();
        vertex_selection.current_tube_index = 0;
        vertex_selection.max_tube_radius = 1;

        auto distances = graph.distances_from(vid);
        const auto& verts = graph.vertices();

        if (vertex_selection.highlight_mode == HighlightMode::Ball) {
            // Ball mode: simple BFS from clicked vertex
            for (size_t j = 0; j < verts.size() && j < distances.size(); ++j) {
                if (distances[j] >= 0 && distances[j] <= vertex_selection.ball_radius) {
                    vertex_selection.ball_distances[verts[j]] = distances[j];
                }
            }
            vertex_selection.show_highlight = !vertex_selection.ball_distances.empty();

        } else {
            // Tube mode: geodesic tubes
            std::vector<std::pair<VertexId, int>> targets_with_dist;
            for (size_t j = 0; j < verts.size() && j < distances.size(); ++j) {
                VertexId v = verts[j];
                int dist = distances[j];
                if (v != vid && dist >= 3 && dist < 1000) {
                    targets_with_dist.push_back({v, dist});
                }
            }
            std::sort(targets_with_dist.begin(), targets_with_dist.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

            constexpr int NUM_TUBES = 8;
            std::vector<VertexId> targets;
            if (!targets_with_dist.empty()) {
                int step = std::max(1, static_cast<int>(targets_with_dist.size()) / NUM_TUBES);
                for (size_t j = 0; j < targets_with_dist.size() && targets.size() < NUM_TUBES; j += step) {
                    targets.push_back(targets_with_dist[j].first);
                }
                if (targets.size() < NUM_TUBES) {
                    targets.push_back(targets_with_dist.back().first);
                }
            }

            auto find_shortest_path = [&graph](VertexId from, VertexId to) -> std::vector<VertexId> {
                if (from == to) return {from};
                std::unordered_map<VertexId, VertexId> parent;
                std::queue<VertexId> q;
                q.push(from);
                parent[from] = from;
                while (!q.empty()) {
                    VertexId curr = q.front();
                    q.pop();
                    for (VertexId neighbor : graph.neighbors(curr)) {
                        if (parent.find(neighbor) == parent.end()) {
                            parent[neighbor] = curr;
                            if (neighbor == to) {
                                std::vector<VertexId> path;
                                VertexId v = to;
                                while (v != from) {
                                    path.push_back(v);
                                    v = parent[v];
                                }
                                path.push_back(from);
                                std::reverse(path.begin(), path.end());
                                return path;
                            }
                            q.push(neighbor);
                        }
                    }
                }
                return {};
            };

            constexpr int MIN_VIZ_TUBE_RADIUS = 2;
            constexpr int MAX_VIZ_TUBE_RADIUS = 6;
            int global_max_radius = 0;

            for (VertexId target : targets) {
                if (target == vid) continue;
                auto path = find_shortest_path(vid, target);
                if (path.empty() || path.size() < 4) continue;

                int geodesic_length = static_cast<int>(path.size()) - 1;
                int tube_radius = std::clamp((geodesic_length + 1) / 2, MIN_VIZ_TUBE_RADIUS, MAX_VIZ_TUBE_RADIUS);
                if (tube_radius > global_max_radius) global_max_radius = tube_radius;

                std::unordered_map<VertexId, int> tube_distances;
                for (VertexId pv : path) {
                    tube_distances[pv] = 0;
                }
                std::queue<std::pair<VertexId, int>> bfs_q;
                for (VertexId pv : path) {
                    bfs_q.push({pv, 0});
                }
                while (!bfs_q.empty()) {
                    auto [v, dist] = bfs_q.front();
                    bfs_q.pop();
                    if (dist < tube_radius) {
                        for (VertexId n : graph.neighbors(v)) {
                            if (tube_distances.find(n) == tube_distances.end()) {
                                tube_distances[n] = dist + 1;
                                bfs_q.push({n, dist + 1});
                            }
                        }
                    }
                }
                vertex_selection.tube_vertex_distances.push_back(tube_distances);
                vertex_selection.tube_geodesic_paths.push_back(path);
                vertex_selection.tube_radii.push_back(tube_radius);
            }
            vertex_selection.max_tube_radius = global_max_radius;
            vertex_selection.show_highlight = !vertex_selection.tube_geodesic_paths.empty();
        }
        vertex_selection.highlight_timestep = timestep;
    };

    // Window callbacks
    platform::WindowCallbacks callbacks;

    callbacks.on_resize = [&](uint32_t w, uint32_t h) {
        should_resize = true;
        new_width = w;
        new_height = h;
        window_width = static_cast<int>(w);
        window_height = static_cast<int>(h);
        cam.set_aspect_ratio(static_cast<float>(w) / h);
    };

    callbacks.on_key = [&](platform::KeyCode key, bool pressed, platform::Modifiers mods) {
        if (key == platform::KeyCode::LeftShift || key == platform::KeyCode::RightShift) {
            controller.set_shift_held(pressed);
        }

        if (!pressed) return;

        // Helper to regenerate random path selections (used by A, 9, 0 keys)
        auto regenerate_path_selections = [&]() {
            selected_state_indices.clear();
            selected_state_indices.resize(analysis.states_per_step.size());
            std::mt19937 rng(42);  // Fixed seed for reproducibility
            for (size_t step = 0; step < analysis.states_per_step.size(); ++step) {
                int n_states = static_cast<int>(analysis.states_per_step[step].size());
                if (n_states > 0) {
                    int count = std::min(path_selection_count, n_states);
                    std::vector<int> indices(n_states);
                    std::iota(indices.begin(), indices.end(), 0);
                    std::shuffle(indices.begin(), indices.end(), rng);
                    selected_state_indices[step].assign(indices.begin(), indices.begin() + count);
                    std::sort(selected_state_indices[step].begin(), selected_state_indices[step].end());
                }
            }
        };

        switch (key) {
            case platform::KeyCode::Escape:
                window->request_close();
                break;

            case platform::KeyCode::Space:
                playing = !playing;
                if (playing) {
                    reset_playback_latch();  // Start latch from current position
                }
                std::cout << (playing ? "Playing" : "Paused")
                          << " (speed=" << playback_speed << "x, "
                          << (playback_direction > 0 ? "forward" : "backward")
                          << (ping_pong_mode ? ", ping-pong" : "") << ")" << std::endl;
                break;

            case platform::KeyCode::LeftBracket:  // [ = slower, Shift+[ = decrease freq threshold
                if (platform::has_modifier(mods, platform::Modifiers::Shift) && edge_mode == EdgeDisplayMode::Frequent) {
                    // { - Decrease frequency threshold (min 1)
                    if (edge_freq_threshold > 1) {
                        edge_freq_threshold--;
                        geometry_dirty = true;
                        std::cout << "Edge frequency threshold: >= " << edge_freq_threshold << " states" << std::endl;
                    }
                } else {
                    // [ - Decrease playback speed
                    playback_speed = std::max(0.25f, playback_speed * 0.5f);
                    if (playing) reset_playback_latch();  // Reset latch when speed changes
                    std::cout << "Playback speed: " << playback_speed << "x" << std::endl;
                }
                break;

            case platform::KeyCode::RightBracket:  // ] = faster, Shift+] = increase freq threshold
                if (platform::has_modifier(mods, platform::Modifiers::Shift) && edge_mode == EdgeDisplayMode::Frequent) {
                    // } - Increase frequency threshold (max = number of states at this step)
                    int max_states = (current_step < static_cast<int>(analysis.states_per_step.size()))
                        ? static_cast<int>(analysis.states_per_step[current_step].size()) : 1;
                    if (edge_freq_threshold < max_states) {
                        edge_freq_threshold++;
                        geometry_dirty = true;
                        std::cout << "Edge frequency threshold: >= " << edge_freq_threshold << " states"
                                  << " (max=" << max_states << ")" << std::endl;
                    }
                } else {
                    // ] - Increase playback speed
                    playback_speed = std::min(256.0f, playback_speed * 2.0f);
                    if (playing) reset_playback_latch();  // Reset latch when speed changes
                    std::cout << "Playback speed: " << playback_speed << "x" << std::endl;
                }
                break;

            case platform::KeyCode::Backslash:  // \ = Reverse direction
                playback_direction = -playback_direction;
                if (playing) {
                    reset_playback_latch();  // Reset latch when direction changes during playback
                }
                std::cout << "Playback direction: " << (playback_direction > 0 ? "forward" : "backward") << std::endl;
                break;

            case platform::KeyCode::P:  // Ping-pong mode
                ping_pong_mode = !ping_pong_mode;
                std::cout << "Ping-pong mode: " << (ping_pong_mode ? "ON" : "OFF") << std::endl;
                break;

            case platform::KeyCode::Left:
                if (current_step > 0) {
                    current_step--;
                    geometry_dirty = true;
                }
                break;

            case platform::KeyCode::Right:
                if (current_step < max_step) {
                    current_step++;
                    geometry_dirty = true;
                }
                break;

            case platform::KeyCode::Home:
                current_step = 0;
                playing = false;
                geometry_dirty = true;
                global_position_cache.clear();  // Clear cache to prevent memory growth
                break;

            case platform::KeyCode::End:
                current_step = max_step;
                playing = false;
                geometry_dirty = true;
                break;

            case platform::KeyCode::Num2:
                view_mode = ViewMode::View2D;
                cam.set_target(math::vec3(0, 0, 0));
                cam.set_distance(camera_distance_2d);
                cam.set_orbit_angles(0.0f, -1.57f);  // Top-down view
                geometry_dirty = true;
                std::cout << "2D View" << std::endl;
                break;

            case platform::KeyCode::Num3:
                view_mode = ViewMode::View3D;
                cam.set_target(math::vec3(0, 0, 2.5f));
                cam.set_distance(camera_distance_3d);
                cam.set_orbit_angles(0.5f, -0.8f);  // Angled 3D view
                geometry_dirty = true;
                std::cout << "3D View" << std::endl;
                break;

            case platform::KeyCode::Num4:
                if (analysis.has_branch_alignment) {
                    view_mode = ViewMode::ShapeSpace2D;
                    cam.set_target(math::vec3(0, 0, 0));
                    cam.set_distance(shape_space_camera_distance);
                    cam.set_orbit_angles(0.0f, -1.57f);  // Top-down view
                    geometry_dirty = true;
                    std::cout << "Shape Space 2D (PC1 vs PC2)" << std::endl;
                } else {
                    std::cout << "Shape Space: N/A (requires curvature/alignment data)" << std::endl;
                }
                break;

            case platform::KeyCode::Num5:
                if (analysis.has_branch_alignment) {
                    view_mode = ViewMode::ShapeSpace3D;
                    cam.set_target(math::vec3(0, 0, 0));
                    cam.set_distance(shape_space_camera_distance);
                    cam.set_orbit_angles(0.5f, -0.8f);  // Angled 3D view
                    geometry_dirty = true;
                    std::cout << "Shape Space 3D (PC1/PC2/PC3)" << std::endl;
                } else {
                    std::cout << "Shape Space: N/A (requires curvature/alignment data)" << std::endl;
                }
                break;

            case platform::KeyCode::H:
                show_horizons = !show_horizons;
                geometry_dirty = true;
                std::cout << "Horizons: " << (show_horizons ? "ON" : "OFF") << std::endl;
                break;

            case platform::KeyCode::J:
                // Toggle geodesic path overlay
                if (analysis.has_geodesic_analysis) {
                    show_geodesics = !show_geodesics;
                    geometry_dirty = true;
                    std::cout << "Geodesic paths: " << (show_geodesics ? "ON" : "OFF") << std::endl;
                } else {
                    std::cout << "Geodesic paths: N/A (run with --geodesics to enable)" << std::endl;
                }
                break;

            case platform::KeyCode::K:
                // Toggle topological defect markers
                if (analysis.has_particle_analysis) {
                    show_defects = !show_defects;
                    geometry_dirty = true;
                    std::cout << "Defect markers: " << (show_defects ? "ON" : "OFF") << std::endl;
                } else {
                    std::cout << "Defect markers: N/A (run with --particles to enable)" << std::endl;
                }
                break;

            case platform::KeyCode::D:
                // Cycle curvature display mode: OFF -> Ollivier-Ricci -> Dimension Gradient -> OFF
                if (has_curvature_analysis) {
                    curvature_display_mode = (curvature_display_mode + 1) % 3;
                    geometry_dirty = true;
                    const char* mode_names[] = {"OFF", "Ollivier-Ricci", "Dimension Gradient"};
                    std::cout << "Curvature: " << mode_names[curvature_display_mode] << std::endl;
                } else {
                    std::cout << "Curvature: N/A (run with --curvature to enable)" << std::endl;
                }
                break;

            case platform::KeyCode::E:
                // Toggle entropy heatmap
                if (has_entropy_analysis) {
                    show_entropy = !show_entropy;
                    geometry_dirty = true;
                    std::cout << "Entropy heatmap: " << (show_entropy ? "ON" : "OFF") << std::endl;
                } else {
                    std::cout << "Entropy heatmap: N/A (run with --entropy to enable)" << std::endl;
                }
                break;

            case platform::KeyCode::Q:
                // Toggle Hilbert space overlay
                if (has_hilbert_analysis) {
                    show_hilbert = !show_hilbert;
                    geometry_dirty = true;
                    std::cout << "Hilbert space: " << (show_hilbert ? "ON" : "OFF") << std::endl;
                } else {
                    std::cout << "Hilbert space: N/A (need multiple states)" << std::endl;
                }
                break;

            case platform::KeyCode::I:
                // Cycle: Union -> Frequent -> Intersection -> SingleState -> Union
                if (edge_mode == EdgeDisplayMode::Union) {
                    edge_mode = EdgeDisplayMode::Frequent;
                } else if (edge_mode == EdgeDisplayMode::Frequent) {
                    edge_mode = EdgeDisplayMode::Intersection;
                } else if (edge_mode == EdgeDisplayMode::Intersection) {
                    // Check if we have per-state data
                    if (!analysis.states_per_step.empty() &&
                        current_step < static_cast<int>(analysis.states_per_step.size()) &&
                        !analysis.states_per_step[current_step].empty()) {
                        edge_mode = EdgeDisplayMode::SingleState;
                        selected_state_index = 0;  // Reset to first state
                    } else {
                        edge_mode = EdgeDisplayMode::Union;  // Skip SingleState if no data
                    }
                } else {
                    edge_mode = EdgeDisplayMode::Union;
                }
                geometry_dirty = true;
                {
                    std::string mode_name;
                    int n_states = (current_step < static_cast<int>(analysis.states_per_step.size()))
                        ? static_cast<int>(analysis.states_per_step[current_step].size()) : 0;
                    if (edge_mode == EdgeDisplayMode::Union) {
                        mode_name = "Union (all)";
                    } else if (edge_mode == EdgeDisplayMode::Frequent) {
                        mode_name = "Frequent (>=" + std::to_string(edge_freq_threshold) + " states, max="
                                  + std::to_string(n_states) + ") - use { } (Shift+[ ]) to adjust threshold";
                    } else if (edge_mode == EdgeDisplayMode::Intersection) {
                        mode_name = "Intersection (all " + std::to_string(n_states) + " states)";
                    } else {
                        mode_name = "Single State (" + std::to_string(selected_state_index + 1)
                                  + "/" + std::to_string(n_states) + ") - use PgUp/PgDn to cycle";
                    }
                    std::cout << "Edge mode: " << mode_name << std::endl;
                }
                break;

            case platform::KeyCode::O:  // Toggle edge coloring mode
                edge_color_mode = (edge_color_mode == EdgeColorMode::Vertex)
                    ? EdgeColorMode::Frequency
                    : EdgeColorMode::Vertex;
                geometry_dirty = true;
                std::cout << "Edge coloring: " << (edge_color_mode == EdgeColorMode::Vertex ? "Vertex (by dimension)" : "Frequency (by state count)");
                if (edge_color_mode == EdgeColorMode::Frequency && analysis.states_per_step.empty()) {
                    std::cout << " [No per-state data - regenerate file to enable]";
                }
                std::cout << std::endl;
                break;

            case platform::KeyCode::V:
                {
                // V cycles through value types: Dimension -> Scalar -> Tensor -> Ollivier -> Gradient
                viz_mode.cycle_value_type();
                geometry_dirty = true;
                std::cout << "Value type: " << value_type_name(viz_mode.value_type) << std::endl;
                if (viz_mode.last_constraint_result.statistic_reset) {
                    std::cout << "  (Statistic reset to Mean - " << viz_mode.why_no_variance() << ")" << std::endl;
                }

                // Recompute highlight for new mode if vertex is selected
                if (vertex_selection.has_selection && vertex_selection.selected_vertex != 0) {
                    HighlightMode new_mode = viz_mode.supports_tube_visualization()
                        ? HighlightMode::Tube : HighlightMode::Ball;
                    if (new_mode != vertex_selection.highlight_mode) {
                        vertex_selection.highlight_mode = new_mode;
                        recompute_highlight(vertex_selection.selected_vertex, current_step);
                        if (vertex_selection.highlight_mode == HighlightMode::Ball) {
                            std::cout << "  Recomputed ball highlight: " << vertex_selection.ball_distances.size() << " vertices" << std::endl;
                        } else {
                            std::cout << "  Recomputed " << vertex_selection.tube_geodesic_paths.size() << " tube highlights" << std::endl;
                        }
                    }
                }
                }
                break;

            case platform::KeyCode::G:
                // G cycles through graph scopes: Branchial -> Foliation -> Global -> Branchial
                viz_mode.cycle_graph_scope();
                if (viz_mode.last_constraint_result.statistic_reset) {
                    std::cout << "  (Statistic reset to Mean - " << viz_mode.why_no_variance() << ")" << std::endl;
                }
                // Disable inapplicable features based on new mode
                if (viz_mode.graph_scope == GraphScope::Global) {
                    // Global mode: disable timeslice, path selection, per-frame normalization
                    if (timeslice_enabled) {
                        timeslice_enabled = false;
                        std::cout << "  (Timeslice disabled - N/A for Global mode)" << std::endl;
                    }
                    if (path_selection_enabled) {
                        path_selection_enabled = false;
                        std::cout << "  (Path selection disabled - N/A for Global mode)" << std::endl;
                    }
                    if (per_frame_normalization) {
                        per_frame_normalization = false;
                        std::cout << "  (Per-frame normalization disabled - N/A for Global mode)" << std::endl;
                    }
                } else if (viz_mode.graph_scope == GraphScope::Foliation) {
                    // Foliation mode: disable path selection
                    if (path_selection_enabled) {
                        path_selection_enabled = false;
                        std::cout << "  (Path selection disabled - N/A for Foliation mode)" << std::endl;
                    }
                }
                geometry_dirty = true;
                std::cout << "Graph scope: " << ui::name(viz_mode.graph_scope) << std::endl;
                break;

            case platform::KeyCode::L:
                if (platform::has_modifier(mods, platform::Modifiers::Shift)) {
                    // Shift+L - Toggle layout all vertices mode
                    layout_use_all_vertices = !layout_use_all_vertices;
                    std::cout << "Layout all vertices: " << (layout_use_all_vertices ? "ON" : "OFF");
                    if (layout_use_all_vertices) {
                        std::cout << " (layout uses full union graph regardless of edge mode)";
                    }
                    std::cout << std::endl;
                    // Reset layout to force re-initialization with new mode
                    layout_step = -1;
                    geometry_dirty = true;
                } else {
                    // L - Toggle dynamic layout
                    layout_enabled = !layout_enabled;
                    std::cout << "Dynamic layout: " << (layout_enabled ? "ON" : "OFF") << std::endl;
                    if (layout_enabled) {
                        // Reset layout to force re-initialization
                        layout_step = -1;
                    }
                    geometry_dirty = true;  // Always update geometry when toggling layout
                }
                break;

            case platform::KeyCode::R:
                if (view_mode == ViewMode::View2D || view_mode == ViewMode::ShapeSpace2D) {
                    cam.set_target(math::vec3(0, 0, 0));
                    cam.set_distance(camera_distance_2d);
                    cam.set_orbit_angles(0.0f, -1.57f);
                } else {
                    // View3D and ShapeSpace3D
                    cam.set_target(math::vec3(0, 0, 2.5f));
                    cam.set_distance(camera_distance_3d);
                    cam.set_orbit_angles(0.5f, -0.8f);
                }
                std::cout << "Camera reset" << std::endl;
                break;

            case platform::KeyCode::B:
                // In shape space: toggle states graph panel
                // In normal view: toggle value aggregation (Per-Vertex vs Bucketed)
                if (view_mode == ViewMode::ShapeSpace2D || view_mode == ViewMode::ShapeSpace3D) {
                    show_states_graph = !show_states_graph;
                    geometry_dirty = true;
                    std::cout << "States graph: " << (show_states_graph ? "ON" : "OFF");
                    if (show_states_graph && analysis.states_per_step.empty()) {
                        std::cout << " [No per-state data - regenerate analysis file]";
                    } else if (show_states_graph) {
                        std::cout << " (" << analysis.states_per_step.size() << " timesteps)";
                    }
                    std::cout << std::endl;
                } else {
                    // Toggle value aggregation for dimension display
                    if (viz_mode.toggle_value_aggregation()) {
                        geometry_dirty = true;
                        std::cout << "Value aggregation: " << ui::name(viz_mode.value_aggregation) << std::endl;
                    } else {
                        std::cout << "Value aggregation: N/A - " << viz_mode.why_no_value_aggregation() << std::endl;
                    }
                }
                break;

            case platform::KeyCode::A:  // Toggle path selection mode
            {
                // Path selection is only applicable for Branchial mode
                if (viz_mode.graph_scope != GraphScope::Branchial) {
                    std::cout << "Path selection: N/A for " << ui::name(viz_mode.graph_scope) << " mode" << std::endl;
                    break;
                }
                path_selection_enabled = !path_selection_enabled;
                if (path_selection_enabled) {
                    regenerate_path_selections();
                }
                geometry_dirty = true;
                std::cout << "Path selection: " << (path_selection_enabled ? "ON (N=" + std::to_string(path_selection_count) + ")" : "OFF") << std::endl;
                break;
            }

            case platform::KeyCode::Num9:  // ( - Decrease path count
                // Path count is only applicable for Branchial mode
                if (viz_mode.graph_scope != GraphScope::Branchial) {
                    std::cout << "Path count: N/A for " << ui::name(viz_mode.graph_scope) << " mode" << std::endl;
                    break;
                }
                if (path_selection_count > 1) {
                    path_selection_count--;
                    if (path_selection_enabled) {
                        regenerate_path_selections();
                        geometry_dirty = true;
                    }
                    std::cout << "Path count: " << path_selection_count << std::endl;
                }
                break;

            case platform::KeyCode::Num0:  // ) - Increase path count
            {
                // Path count is only applicable for Branchial mode
                if (viz_mode.graph_scope != GraphScope::Branchial) {
                    std::cout << "Path count: N/A for " << ui::name(viz_mode.graph_scope) << " mode" << std::endl;
                    break;
                }
                int max_count = 1;
                for (const auto& states : analysis.states_per_step) {
                    max_count = std::max(max_count, static_cast<int>(states.size()));
                }
                if (path_selection_count < max_count) {
                    path_selection_count++;
                    if (path_selection_enabled) {
                        regenerate_path_selections();
                        geometry_dirty = true;
                    }
                    std::cout << "Path count: " << path_selection_count << std::endl;
                }
                break;
            }

            case platform::KeyCode::M:
                // In shape space mode: cycle branches; otherwise toggle statistic mode
                if (view_mode == ViewMode::ShapeSpace2D || view_mode == ViewMode::ShapeSpace3D) {
                    if (shape_space_display_mode == ShapeSpaceDisplayMode::Merged) {
                        shape_space_display_mode = ShapeSpaceDisplayMode::PerBranch;
                        highlighted_branch = 0;
                        std::cout << "Shape Space: Branch 0 highlighted" << std::endl;
                    } else {
                        int num_branches = 0;
                        if (current_step >= 0 && current_step < static_cast<int>(analysis.alignment_per_timestep.size())) {
                            num_branches = static_cast<int>(analysis.alignment_per_timestep[current_step].num_branches);
                        }
                        if (highlighted_branch < num_branches - 1) {
                            highlighted_branch++;
                            std::cout << "Shape Space: Branch " << highlighted_branch << " highlighted" << std::endl;
                        } else {
                            shape_space_display_mode = ShapeSpaceDisplayMode::Merged;
                            highlighted_branch = -1;
                            std::cout << "Shape Space: All branches merged" << std::endl;
                        }
                    }
                    geometry_dirty = true;
                } else {
                    // Toggle statistic mode (Mean/Variance)
                    if (viz_mode.toggle_statistic_mode()) {
                        geometry_dirty = true;
                        std::cout << "Statistic: " << ui::name(viz_mode.statistic_mode) << std::endl;
                    } else {
                        std::cout << "Variance: N/A - " << viz_mode.why_no_variance() << std::endl;
                    }
                }
                break;

            case platform::KeyCode::N:
                // Per-frame normalization is not applicable for Global mode (values are constant)
                if (viz_mode.graph_scope == GraphScope::Global) {
                    std::cout << "Normalization: N/A for Global mode (values are constant)" << std::endl;
                    break;
                }
                per_frame_normalization = !per_frame_normalization;
                geometry_dirty = true;
                std::cout << "Color normalization: " << (per_frame_normalization ? "Local (per-frame)" : "Global") << std::endl;
                break;

            case platform::KeyCode::T:
                // With selection: toggle/cycle highlight visualization
                // Otherwise: toggle timeslice
                if (vertex_selection.has_selection &&
                    vertex_selection.highlight_mode == HighlightMode::Tube &&
                    !vertex_selection.tube_geodesic_paths.empty()) {
                    // Tube mode: cycle through tubes
                    if (!vertex_selection.show_highlight) {
                        vertex_selection.show_highlight = true;
                        vertex_selection.current_tube_index = 0;
                    } else {
                        vertex_selection.current_tube_index++;
                        if (vertex_selection.current_tube_index >= static_cast<int>(vertex_selection.tube_geodesic_paths.size())) {
                            vertex_selection.show_highlight = false;
                            vertex_selection.current_tube_index = 0;
                        }
                    }
                    geometry_dirty = true;
                    if (vertex_selection.show_highlight) {
                        int idx = vertex_selection.current_tube_index;
                        std::cout << "Tube " << (idx + 1) << "/" << vertex_selection.tube_geodesic_paths.size()
                                  << ": path length=" << (vertex_selection.tube_geodesic_paths[idx].size() - 1)
                                  << ", radius=" << vertex_selection.tube_radii[idx]
                                  << ", vertices=" << vertex_selection.tube_vertex_distances[idx].size() << std::endl;
                    } else {
                        std::cout << "Tube visualization: OFF" << std::endl;
                    }
                } else if (vertex_selection.has_selection &&
                           vertex_selection.highlight_mode == HighlightMode::Ball &&
                           !vertex_selection.ball_distances.empty()) {
                    // Ball mode: simple toggle
                    vertex_selection.show_highlight = !vertex_selection.show_highlight;
                    geometry_dirty = true;
                    std::cout << "Ball visualization: " << (vertex_selection.show_highlight ? "ON" : "OFF") << std::endl;
                } else {
                    // Timeslice is not applicable for Global mode (values are constant)
                    if (viz_mode.graph_scope == GraphScope::Global) {
                        std::cout << "Timeslice: N/A for Global mode (values are constant)" << std::endl;
                        break;
                    }
                    timeslice_enabled = !timeslice_enabled;
                    geometry_dirty = true;
                    std::cout << "Timeslice view: " << (timeslice_enabled ? "ON (width=" + std::to_string(timeslice_width) + ")" : "OFF") << std::endl;
                }
                break;

            case platform::KeyCode::Z:
                if (!z_mapping_enabled && z_scale <= 0.0f) {
                    // Can't enable Z mapping when scale is zero
                    std::cout << "Z/depth mapping: Cannot enable (z_scale is 0, use > to increase)" << std::endl;
                } else {
                    z_mapping_enabled = !z_mapping_enabled;
                    geometry_dirty = true;
                    std::cout << "Z/depth mapping: " << (z_mapping_enabled ? "ON (elevated by dimension)" : "OFF (flat)") << std::endl;
                }
                break;

            case platform::KeyCode::U:  // Cycle missing data display mode: Show -> Hide -> Highlight
            {
                int m = static_cast<int>(missing_mode);
                m = (m + 1) % 3;  // Cycle through Show(0), Hide(1), Highlight(2)
                missing_mode = static_cast<MissingDataMode>(m);
                geometry_dirty = true;
                std::cout << "Missing data mode: " << missing_mode_name(missing_mode) << std::endl;
                break;
            }

            case platform::KeyCode::C:  // In shape space: toggle color mode; otherwise cycle palette
                if (view_mode == ViewMode::ShapeSpace2D || view_mode == ViewMode::ShapeSpace3D) {
                    shape_space_color_mode = (shape_space_color_mode == ShapeSpaceColorMode::Curvature)
                        ? ShapeSpaceColorMode::BranchId : ShapeSpaceColorMode::Curvature;
                    geometry_dirty = true;
                    std::cout << "Shape Space Color: "
                              << (shape_space_color_mode == ShapeSpaceColorMode::Curvature ? "Curvature" : "Branch ID")
                              << std::endl;
                } else {
                    int p = static_cast<int>(current_palette);
                    p = (p + 1) % static_cast<int>(ColorPalette::COUNT);
                    current_palette = static_cast<ColorPalette>(p);
                    geometry_dirty = true;
                    std::cout << "Color palette: " << palette_name(current_palette) << std::endl;
                }
                break;

            case platform::KeyCode::W:  // In shape space: toggle curvature method (Wolfram vs Ollivier)
                if (view_mode == ViewMode::ShapeSpace2D || view_mode == ViewMode::ShapeSpace3D) {
                    if (analysis.has_ollivier_alignment) {
                        use_ollivier_alignment = !use_ollivier_alignment;
                        geometry_dirty = true;
                        std::cout << "Shape Space Curvature: "
                                  << (use_ollivier_alignment ? "Ollivier-Ricci" : "Wolfram-Ricci (K=2-d)")
                                  << std::endl;
                    } else {
                        std::cout << "Ollivier-Ricci alignment not available (use --curvature)" << std::endl;
                    }
                }
                break;

            case platform::KeyCode::Slash:  // / or ? - Toggle help overlay
                show_overlay = !show_overlay;
                std::cout << "Help overlay: " << (show_overlay ? "ON" : "OFF") << std::endl;
                break;

            case platform::KeyCode::X:  // Toggle multispace probability overlay
                show_multispace = !show_multispace;
                std::cout << "Multispace overlay: " << (show_multispace ? "ON" : "OFF");
                if (show_multispace && analysis.states_per_step.empty()) {
                    std::cout << " [Requires Branchial mode data]";
                }
                std::cout << std::endl;
                geometry_dirty = true;
                break;

            case platform::KeyCode::Y:  // Toggle scatter plot or cycle metric
                if (has_modifier(mods, platform::Modifiers::Shift)) {
                    // Shift+Y: cycle through metrics
                    scatter_metric = static_cast<ValueType>(
                        (static_cast<int>(scatter_metric) + 1) % static_cast<int>(ValueType::COUNT));
                    std::cout << "Scatter metric: " << value_type_name(scatter_metric) << std::endl;
                } else {
                    // Y: toggle visibility
                    show_scatter = !show_scatter;
                    std::cout << "Scatter plot (" << value_type_name(scatter_metric) << "): "
                              << (show_scatter ? "ON" : "OFF") << std::endl;
                }
                break;

            case platform::KeyCode::Minus:
                // If histogram is showing, adjust bin count
                if (vertex_selection.has_selection) {
                    if (vertex_selection.histogram_bin_count > 2) {
                        vertex_selection.histogram_bin_count--;
                        update_selection_histogram(vertex_selection, analysis, current_step, viz_mode.graph_scope, viz_mode.value_type);
                        std::cout << "Histogram bins: " << vertex_selection.histogram_bin_count << std::endl;
                    }
                    break;
                }
                // Timeslice width is not applicable for Global mode
                if (viz_mode.graph_scope == GraphScope::Global) {
                    std::cout << "Timeslice width: N/A for Global mode" << std::endl;
                    break;
                }
                if (timeslice_width > 1) {
                    timeslice_width = std::max(1, timeslice_width - 2);
                    geometry_dirty = true;
                    std::cout << "Timeslice width: " << timeslice_width << std::endl;
                }
                break;

            case platform::KeyCode::Equal:
                // If histogram is showing, adjust bin count
                if (vertex_selection.has_selection) {
                    if (vertex_selection.histogram_bin_count < 50) {
                        vertex_selection.histogram_bin_count++;
                        update_selection_histogram(vertex_selection, analysis, current_step, viz_mode.graph_scope, viz_mode.value_type);
                        std::cout << "Histogram bins: " << vertex_selection.histogram_bin_count << std::endl;
                    }
                    break;
                }
                // Timeslice width is not applicable for Global mode
                if (viz_mode.graph_scope == GraphScope::Global) {
                    std::cout << "Timeslice width: N/A for Global mode" << std::endl;
                    break;
                }
                timeslice_width = std::min(max_step, timeslice_width + 2);
                geometry_dirty = true;
                std::cout << "Timeslice width: " << timeslice_width << std::endl;
                break;

            case platform::KeyCode::Comma:  // < key - decrease Z scale
                z_scale = std::max(0.0f, z_scale - 0.5f);
                geometry_dirty = true;
                // Implicitly turn off Z mapping when scale reaches 0
                if (z_scale <= 0.0f && z_mapping_enabled) {
                    z_mapping_enabled = false;
                    std::cout << "Z scale: 0 (Z mapping OFF)" << std::endl;
                } else {
                    std::cout << "Z scale: " << z_scale << std::endl;
                }
                break;

            case platform::KeyCode::Period:  // > key - increase Z scale
                // If Z mapping is off, turn it back on when user increases scale
                if (!z_mapping_enabled) {
                    z_mapping_enabled = true;
                    z_scale = 0.5f;  // Start at minimum visible scale
                    std::cout << "Z scale: " << z_scale << " (Z mapping ON)" << std::endl;
                } else {
                    z_scale = std::min(20.0f, z_scale + 0.5f);
                    std::cout << "Z scale: " << z_scale << std::endl;
                }
                geometry_dirty = true;
                break;

            case platform::KeyCode::PageUp:  // Cycle to previous state in SingleState mode
                if (edge_mode == EdgeDisplayMode::SingleState) {
                    int n_states = (current_step < static_cast<int>(analysis.states_per_step.size()))
                        ? static_cast<int>(analysis.states_per_step[current_step].size()) : 0;
                    if (n_states > 0) {
                        selected_state_index = (selected_state_index - 1 + n_states) % n_states;
                        geometry_dirty = true;
                        std::cout << "State: " << (selected_state_index + 1) << "/" << n_states << std::endl;
                    }
                }
                break;

            case platform::KeyCode::PageDown:  // Cycle to next state in SingleState mode
                if (edge_mode == EdgeDisplayMode::SingleState) {
                    int n_states = (current_step < static_cast<int>(analysis.states_per_step.size()))
                        ? static_cast<int>(analysis.states_per_step[current_step].size()) : 0;
                    if (n_states > 0) {
                        selected_state_index = (selected_state_index + 1) % n_states;
                        geometry_dirty = true;
                        std::cout << "State: " << (selected_state_index + 1) << "/" << n_states << std::endl;
                    }
                }
                break;

            default:
                break;
        }
    };

    // Helper to convert screen coords to NDC (Vulkan coordinates)
    // Screen: y=0 at top, y=height at bottom
    // Vulkan NDC: y=-1 at top, y=+1 at bottom
    auto screen_to_ndc = [&](int x, int y) -> std::pair<float, float> {
        float ndc_x = (2.0f * x / window_width) - 1.0f;
        float ndc_y = (2.0f * y / window_height) - 1.0f;  // Vulkan-style: no flip
        return {ndc_x, ndc_y};
    };

    // Helper to check if point is in timeline UI region
    auto is_in_timeline = [&](float ndc_x, float ndc_y) -> bool {
        // Need valid timeline data (initialized after first render)
        if (last_timeline.scrubber_right <= last_timeline.scrubber_left) return false;

        // Check scrubber region (the entire bar)
        if (ndc_x >= last_timeline.scrubber_left && ndc_x <= last_timeline.scrubber_right &&
            ndc_y >= last_timeline.scrubber_bottom && ndc_y <= last_timeline.scrubber_top) {
            return true;
        }
        // Check button regions
        float btn_bottom = last_timeline.scrubber_bottom;
        float btn_top = last_timeline.scrubber_top;
        if (ndc_y >= btn_bottom && ndc_y <= btn_top) {
            if (ndc_x >= last_timeline.rewind_left && ndc_x <= last_timeline.rewind_right) return true;
            if (ndc_x >= last_timeline.play_left && ndc_x <= last_timeline.play_right) return true;
            if (ndc_x >= last_timeline.skip_end_left && ndc_x <= last_timeline.skip_end_right) return true;
        }
        return false;
    };

    // Helper to check if point is specifically in the scrubber bar area
    auto is_in_scrubber_bar = [&](float ndc_x, float ndc_y) -> bool {
        return ndc_x >= last_timeline.scrubber_left && ndc_x <= last_timeline.scrubber_right &&
               ndc_y >= last_timeline.scrubber_bottom && ndc_y <= last_timeline.scrubber_top;
    };

    // Helper to update step from scrubber position (direct set)
    auto scrubber_set_step = [&](float ndc_x) {
        float t = (ndc_x - last_timeline.scrubber_left) /
                  (last_timeline.scrubber_right - last_timeline.scrubber_left);
        t = std::clamp(t, 0.0f, 1.0f);
        int new_step = static_cast<int>(t * max_step + 0.5f);
        if (new_step != current_step) {
            current_step = new_step;
            geometry_dirty = true;
        }
    };

    // Helper to jump halfway between current position and click position
    auto scrubber_jump_halfway = [&](float ndc_x) {
        float bar_width = last_timeline.scrubber_right - last_timeline.scrubber_left;
        if (bar_width <= 0) return;

        // Current marker position in NDC
        float current_t = (max_step > 0) ? static_cast<float>(current_step) / max_step : 0.0f;
        float current_ndc = last_timeline.scrubber_left + current_t * bar_width;

        // Target is halfway between current and click
        float target_ndc = (current_ndc + ndc_x) / 2.0f;

        // Convert to step
        float t = (target_ndc - last_timeline.scrubber_left) / bar_width;
        t = std::clamp(t, 0.0f, 1.0f);
        int new_step = static_cast<int>(t * max_step + 0.5f);
        if (new_step != current_step) {
            current_step = new_step;
            geometry_dirty = true;
        }
    };

    // Track click start position for picking
    int click_start_x = 0, click_start_y = 0;
    bool is_potential_click = false;

    callbacks.on_mouse_button = [&](platform::MouseButton button, bool pressed, int x, int y, platform::Modifiers mods) {
        if (button == platform::MouseButton::Left) {
            auto [ndc_x, ndc_y] = screen_to_ndc(x, y);

            if (pressed) {
                // Check if clicking on histogram panel (dismiss or keep)
                if (vertex_selection.has_selection) {
                    bool in_panel = ndc_x >= vertex_selection.panel_left &&
                                    ndc_x <= vertex_selection.panel_right &&
                                    ndc_y >= vertex_selection.panel_top &&
                                    ndc_y <= vertex_selection.panel_bottom;
                    if (in_panel) {
                        // Click inside panel - ignore (could add interactivity later)
                        return;
                    }
                }

                // Check if clicking on timeline UI
                if (is_in_timeline(ndc_x, ndc_y)) {
                    // Check buttons first
                    float btn_bottom = last_timeline.scrubber_bottom;
                    float btn_top = last_timeline.scrubber_top;
                    if (ndc_y >= btn_bottom && ndc_y <= btn_top) {
                        if (ndc_x >= last_timeline.rewind_left && ndc_x <= last_timeline.rewind_right) {
                            // Rewind: go to start and stop playback
                            current_step = 0;
                            playing = false;
                            geometry_dirty = true;
                            global_position_cache.clear();  // Clear cache to prevent memory growth
                            return;
                        }
                        if (ndc_x >= last_timeline.play_left && ndc_x <= last_timeline.play_right) {
                            // Play/pause toggle
                            playing = !playing;
                            if (playing) reset_playback_latch();
                            std::cout << (playing ? "Playing" : "Paused") << std::endl;
                            return;
                        }
                        if (ndc_x >= last_timeline.skip_end_left && ndc_x <= last_timeline.skip_end_right) {
                            // Skip to end and stop playback
                            current_step = max_step;
                            playing = false;
                            geometry_dirty = true;
                            return;
                        }
                    }

                    // Clicking on scrubber bar
                    if (is_in_scrubber_bar(ndc_x, ndc_y)) {
                        scrubber_dragging = true;
                        playing = false;  // Stop playback when user drags timeline
                        // Jump halfway to click position (standard behavior for media scrubbers)
                        scrubber_jump_halfway(ndc_x);
                        return;
                    }
                }

                // Not on UI - track for potential click/pick or camera drag
                click_start_x = x;
                click_start_y = y;
                is_potential_click = true;
                left_mouse_down = true;
                controller.set_mouse_captured(true);
            } else {
                // Mouse released
                if (is_potential_click && left_mouse_down) {
                    // Check if this was a click (minimal movement) vs drag
                    int dx = std::abs(x - click_start_x);
                    int dy = std::abs(y - click_start_y);
                    if (dx < 5 && dy < 5) {
                        // This is a click - trigger picking
                        picking_pass_pending = true;
                        picking_click_x = x;
                        picking_click_y = y;
                        vertex_selection.click_screen_x = x;
                        vertex_selection.click_screen_y = y;
                    }
                }
                is_potential_click = false;
                left_mouse_down = false;
                scrubber_dragging = false;
                controller.set_mouse_captured(false);
            }
        }
    };

    callbacks.on_mouse_move = [&](int x, int y) {
        float fx = static_cast<float>(x);
        float fy = static_cast<float>(y);

        if (scrubber_dragging) {
            auto [ndc_x, ndc_y] = screen_to_ndc(x, y);
            scrubber_set_step(ndc_x);
        } else if (left_mouse_down) {
            float dx = fx - last_mouse_x;
            float dy = fy - last_mouse_y;
            if (view_mode == ViewMode::View2D) {
                // In 2D mode, pan in XY plane (layout plane)
                // Scale by zoom level for consistent feel
                float scale = cam.get_distance() * 0.001f;
                math::vec3 target = cam.get_target();
                target.x -= dx * scale;  // Drag right → move target left → show more right
                target.y -= dy * scale;  // Drag down → move target down → show more bottom
                cam.set_target(target);
            } else {
                controller.on_mouse_move(dx, dy);
            }
        }
        last_mouse_x = fx;
        last_mouse_y = fy;
    };

    callbacks.on_scroll = [&](float dx, float dy) {
        controller.on_mouse_scroll(-dy);
    };

    window->set_callbacks(callbacks);

    // Print key map
    // Print help from unified keybinding definitions
    std::cout << "\n=== Black Hole Visualization Controls ===" << std::endl;
    for (const auto& group : keybindings) {
        std::cout << group.name << ":" << std::endl;
        for (const auto& binding : group.bindings) {
            std::cout << "  " << std::left << std::setw(14) << binding.key
                      << " - " << binding.description << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Press / or ? to toggle on-screen help overlay" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;

    // Bring window to front and focus it
    window->focus();

    // Main loop
    while (window->is_open() && !g_shutdown_requested.load()) {
        window->poll_events();

        // Check for Ctrl-C
        if (g_shutdown_requested.load()) {
            std::cout << "\nShutdown requested (Ctrl-C), exiting..." << std::endl;
            break;
        }

        // Skip rendering when minimized (swapchain can't acquire 0x0 images)
        if (window->is_minimized()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Calculate delta time
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_frame).count();
        last_frame = now;

        // Handle resize
        if (should_resize && new_width > 0 && new_height > 0) {
            device->wait_idle();
            swapchain->resize(new_width, new_height);
            depth_texture = create_depth_texture(new_width, new_height);
            msaa_dirty = true;  // Need to recreate MSAA resources at new size
            should_resize = false;
        }

        // Handle MSAA resource creation/recreation
        if (msaa_dirty) {
            // Must wait for GPU to finish before destroying/recreating resources
            device->wait_idle();
            uint32_t w = swapchain->get_texture(0)->get_size().width;
            uint32_t h = swapchain->get_texture(0)->get_size().height;
            create_msaa_resources(w, h);
            msaa_dirty = false;
        }

        // Time-based playback with latch
        if (playing && max_step > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - play_latch_time).count();

            // Compute target step based on elapsed time
            int target_step = play_latch_step + static_cast<int>(elapsed * playback_speed) * playback_direction;

            // Handle boundaries
            bool direction_changed = false;
            if (target_step > max_step) {
                if (ping_pong_mode) {
                    // Bounce back - compute how far past the end we are
                    int overshoot = target_step - max_step;
                    target_step = max_step - overshoot;
                    if (target_step < 0) target_step = 0;
                    playback_direction = -1;
                    direction_changed = true;
                } else {
                    // Loop around
                    target_step = target_step % (max_step + 1);
                }
            } else if (target_step < 0) {
                if (ping_pong_mode) {
                    // Bounce forward
                    int overshoot = -target_step;
                    target_step = overshoot;
                    if (target_step > max_step) target_step = max_step;
                    playback_direction = 1;
                    direction_changed = true;
                } else {
                    // Loop around
                    target_step = max_step + 1 + (target_step % (max_step + 1));
                    if (target_step > max_step) target_step = max_step;
                }
            }

            // Reset latch if direction changed (for ping-pong mode)
            if (direction_changed) {
                play_latch_time = now;
                play_latch_step = target_step;
            }

            // Update current step if changed
            if (target_step != current_step) {
                // Clear position cache when looping back significantly
                // This prevents unbounded memory growth in long-running sessions
                if (target_step < current_step - max_step / 2) {
                    global_position_cache.clear();
                }
                current_step = target_step;
                geometry_dirty = true;
            }
        }

        // Dynamic layout: reinitialize when step, edge mode, OR frequency threshold changes
        // Layout uses current edge mode's edges for forces
        // When layout_use_all_vertices is true, we use Union mode always (ignore edge_mode changes)
        static EdgeDisplayMode layout_edge_mode = EdgeDisplayMode::Union;
        static int layout_freq_threshold = 2;  // Track threshold for re-initialization
        static bool layout_all_verts_tracked = false;  // Track toggle state for re-init detection

        // Check if layout needs re-initialization
        bool layout_needs_reinit = (current_step != layout_step) ||
                                   (layout_use_all_vertices != layout_all_verts_tracked) ||
                                   (!layout_use_all_vertices && edge_mode != layout_edge_mode) ||
                                   (edge_mode == EdgeDisplayMode::Frequent && edge_freq_threshold != layout_freq_threshold);

        if (layout_enabled && layout_needs_reinit) {
            if (current_step >= 0 && current_step < static_cast<int>(analysis.per_timestep.size())) {
                const auto& ts = analysis.per_timestep[current_step];

                // Get per-state data for threshold-based filtering (if available)
                const std::vector<StateData>* states_ptr = nullptr;
                if (current_step < static_cast<int>(analysis.states_per_step.size()) &&
                    !analysis.states_per_step[current_step].empty()) {
                    states_ptr = &analysis.states_per_step[current_step];
                }

                // Initialize layout with global cache for persistent positions across timesteps
                // Seed from previous layout positions if available (preserves positions across mode changes)
                // When layout_use_all_vertices is true, always use Union mode for layout
                // (prevents layout snapping when switching edge display modes)
                EdgeDisplayMode layout_effective_mode = layout_use_all_vertices ? EdgeDisplayMode::Union : edge_mode;
                if (layout_step >= 0 && !layout_vertices.empty()) {
                    init_layout_from_timestep(layout_graph, ts, analysis.initial, layout_effective_mode,
                                              global_position_cache, layout_vertices,
                                              edge_freq_threshold, states_ptr,
                                              &layout_graph, &layout_vertices);
                } else {
                    init_layout_from_timestep(layout_graph, ts, analysis.initial, layout_effective_mode,
                                              global_position_cache, layout_vertices,
                                              edge_freq_threshold, states_ptr);
                }

                layout_edge_mode = edge_mode;
                layout_freq_threshold = edge_freq_threshold;
                layout_all_verts_tracked = layout_use_all_vertices;  // Track toggle state
                layout_engine->upload_graph(layout_graph);
                layout_step = current_step;
                geometry_dirty = true;

                // Diagnostic: print step info (uncomment for GPU layout debugging)
                // std::cout << "[Step " << current_step << "] V=" << ts.union_vertices.size()
                //           << " E_union=" << ts.union_edges.size()
                //           << " E_inter=" << ts.intersection_edges.size()
                //           << " layout_V=" << layout_graph.vertex_count()
                //           << " layout_E=" << layout_graph.edge_count() << std::endl;
            }
        }

        // Run layout iterations per frame when enabled
        static int frame_count = 0;
        if (layout_enabled && layout_engine->has_graph()) {
            // Run a few iterations per frame for smooth animation
            constexpr int iterations_per_frame = 3;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations_per_frame; ++i) {
                layout_engine->iterate(layout_params);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            layout_engine->download_positions(layout_graph);  // Get updated positions
            geometry_dirty = true;  // Always update when layout is running

            // Update global position cache with current positions (for persistence across timesteps)
            update_position_cache(global_position_cache, layout_graph, layout_vertices);

            // Print timing every 60 frames (uncomment for GPU layout debugging)
            frame_count++;
            // if (frame_count % 60 == 0) {
            //     auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            //     std::cout << "  [Layout] " << iterations_per_frame << " iters in "
            //               << std::fixed << std::setprecision(2) << ms << " ms"
            //               << " (V=" << layout_graph.vertex_count()
            //               << ", E=" << layout_graph.edge_count() << ")" << std::endl;
            // }
        }

        // Wait for previous frame BEFORE buffer writes to prevent race conditions
        // (GPU may still be reading buffers from previous frame)
        // NOTE: Only reset fence just before submit, not here. If acquire fails and we
        // continue, we'd leave the fence unsignaled causing next frame's wait to hang.
        fence->wait();
        in_flight_cmd.reset();

        // Process picking readback if pending
        if (picking_readback_pending && picking_readback_buffer) {
            uint32_t picked_index = 0;
            picking_readback_buffer->read(&picked_index, sizeof(uint32_t));

            if (picked_index > 0) {
                // Vertex was clicked (index is 1-based, 0 means background)
                size_t vertex_idx = picked_index - 1;
                if (vertex_idx < picking_vertex_ids.size()) {
                    VertexId clicked_vid = picking_vertex_ids[vertex_idx];

                    // Update selection
                    vertex_selection.has_selection = true;
                    vertex_selection.selected_vertex = clicked_vid;

                    // Position panel near click (convert to NDC)
                    auto [ndc_x, ndc_y] = screen_to_ndc(vertex_selection.click_screen_x,
                                                        vertex_selection.click_screen_y);
                    vertex_selection.panel_ndc_x = ndc_x;
                    vertex_selection.panel_ndc_y = ndc_y;

                    // Compute histogram data
                    update_selection_histogram(vertex_selection, analysis, current_step, viz_mode.graph_scope, viz_mode.value_type);

                    // Set highlight mode based on value type and compute highlight
                    vertex_selection.highlight_mode = viz_mode.supports_tube_visualization()
                        ? HighlightMode::Tube : HighlightMode::Ball;
                    recompute_highlight(clicked_vid, current_step);
                    geometry_dirty = true;

                    // Diagnostic output
                    if (vertex_selection.highlight_mode == HighlightMode::Ball) {
                        std::cout << "  Computed ball highlight: " << vertex_selection.ball_distances.size()
                                  << " vertices within radius " << vertex_selection.ball_radius << std::endl;
                    } else if (!vertex_selection.tube_geodesic_paths.empty()) {
                        std::cout << "  Computed " << vertex_selection.tube_geodesic_paths.size()
                                  << " geodesic tubes from vertex " << clicked_vid
                                  << " (max radius " << vertex_selection.max_tube_radius << "):" << std::endl;
                        for (size_t ti = 0; ti < vertex_selection.tube_geodesic_paths.size(); ++ti) {
                            const auto& path = vertex_selection.tube_geodesic_paths[ti];
                            const auto& tube = vertex_selection.tube_vertex_distances[ti];
                            std::cout << "    Tube " << ti << ": path length=" << (path.size()-1)
                                      << ", tube vertices=" << tube.size() << std::endl;
                        }
                    } else if (vertex_selection.highlight_mode == HighlightMode::Tube) {
                        std::cout << "  No valid tubes found (need paths of length >= 3)" << std::endl;
                    }

                    std::cout << "Selected vertex " << clicked_vid
                              << " (" << vertex_selection.dimension_values.size() << " values)"
                              << std::endl;
                }
            } else {
                // Clicked on empty space - dismiss selection
                vertex_selection.has_selection = false;
                vertex_selection.show_highlight = false;
                vertex_selection.highlight_timestep = -1;
            }

            picking_readback_pending = false;
        }

        // Rebuild geometry if needed
        if (geometry_dirty) {
            auto geom_t0 = std::chrono::high_resolution_clock::now();

            // Recompute highlight if timestep changed and we have a selection
            if (vertex_selection.has_selection && vertex_selection.selected_vertex != 0 &&
                vertex_selection.highlight_timestep != current_step) {
                recompute_highlight(vertex_selection.selected_vertex, current_step);
            }

            // Use live layout positions when layout is enabled
            const layout::LayoutGraph* live_layout = layout_enabled ? &layout_graph : nullptr;
            const std::vector<VertexId>* layout_verts = layout_enabled ? &layout_vertices : nullptr;

            // Get path selection (full vector for timeslice combination)
            const std::vector<std::vector<int>>* path_selection_ptr = nullptr;
            if (path_selection_enabled && !selected_state_indices.empty()) {
                path_selection_ptr = &selected_state_indices;
            }

            // Pass position cache for vertices not in current layout (e.g., during timeslice mode)
            const PositionCache* pos_cache = layout_enabled ? &global_position_cache : nullptr;

            // Use shape space render data if in shape space mode, otherwise use normal render
            if (view_mode == ViewMode::ShapeSpace2D || view_mode == ViewMode::ShapeSpace3D) {
                render_data = build_shape_space_render_data(
                    analysis, current_step, view_mode,
                    shape_space_color_mode, shape_space_display_mode,
                    highlighted_branch, use_ollivier_alignment, current_palette);
            } else {
                // Prepare highlight visualization data (Ball or Tube mode)
                const std::unordered_map<VertexId, int>* highlight_distances = nullptr;
                VertexId highlight_source = vertex_selection.selected_vertex;
                VertexId highlight_target = 0;

                bool show_hl = vertex_selection.show_highlight && vertex_selection.selected_vertex != 0;
                if (show_hl) {
                    if (vertex_selection.highlight_mode == HighlightMode::Ball) {
                        // Ball mode: use ball_distances map
                        if (!vertex_selection.ball_distances.empty()) {
                            highlight_distances = &vertex_selection.ball_distances;
                        }
                    } else {
                        // Tube mode: use current tube's distance map
                        if (!vertex_selection.tube_vertex_distances.empty() &&
                            vertex_selection.current_tube_index < static_cast<int>(vertex_selection.tube_vertex_distances.size())) {
                            highlight_distances = &vertex_selection.tube_vertex_distances[vertex_selection.current_tube_index];
                            // Get target from path
                            if (!vertex_selection.tube_geodesic_paths.empty() &&
                                vertex_selection.current_tube_index < static_cast<int>(vertex_selection.tube_geodesic_paths.size())) {
                                const auto& path = vertex_selection.tube_geodesic_paths[vertex_selection.current_tube_index];
                                if (!path.empty()) {
                                    highlight_target = path.back();
                                }
                            }
                        }
                    }
                }

                render_data = build_render_data(analysis, current_step, view_mode, edge_mode, edge_freq_threshold,
                                                selected_state_index, viz_mode.value_type, viz_mode.statistic_mode, viz_mode.graph_scope, per_frame_normalization,
                                                z_mapping_enabled, missing_mode, current_palette, edge_color_mode,
                                                live_layout, layout_verts, pos_cache, 0.04f, 0.015f, z_scale, timeslice_enabled,
                                                timeslice_width, path_selection_ptr, show_geodesics, show_defects,
                                                show_hl && highlight_distances != nullptr, highlight_distances,
                                                highlight_source, highlight_target, vertex_selection.non_highlight_alpha,
                                                viz_mode.value_aggregation);
            }

            auto geom_t1 = std::chrono::high_resolution_clock::now();
            // Print geometry timing every 60 frames (uncomment for GPU layout debugging)
            // if (frame_count % 60 == 0 && layout_enabled) {
            //     auto ms = std::chrono::duration<double, std::milli>(geom_t1 - geom_t0).count();
            //     std::cout << "  [Geom] build_render_data in " << std::fixed << std::setprecision(2) << ms << " ms"
            //               << " (spheres=" << render_data.sphere_instances.size()
            //               << ", cylinders=" << render_data.cylinder_instances.size() << ")" << std::endl;
            // }
            (void)geom_t0; (void)geom_t1;  // Suppress unused variable warnings

            // Upload legacy vertex data (for 2D mode)
            if (vertex_buffer && render_data.vertex_count > 0) {
                size_t needed = render_data.vertex_data.size() * sizeof(Vertex);
                if (needed > buffer_size) {
                    buffer_size = needed * 2;
                    buffer_desc.size = buffer_size;
                    vertex_buffer = device->create_buffer(buffer_desc);
                }
                vertex_buffer->write(render_data.vertex_data.data(), needed);
            }

            // Upload legacy edge data (for 2D mode)
            if (edge_buffer && render_data.edge_count > 0) {
                size_t needed = render_data.edge_data.size() * sizeof(Vertex);
                if (needed > buffer_size) {
                    buffer_size = needed * 2;
                    buffer_desc.size = buffer_size;
                    edge_buffer = device->create_buffer(buffer_desc);
                }
                edge_buffer->write(render_data.edge_data.data(), needed);
            }

            // Upload sphere instance data (for 3D mode)
            if (sphere_instance_buffer && !render_data.sphere_instances.empty()) {
                size_t needed = render_data.sphere_instances.size() * sizeof(SphereInstance);
                if (needed > instance_buffer_desc.size) {
                    instance_buffer_desc.size = needed * 2;
                    sphere_instance_buffer = device->create_buffer(instance_buffer_desc);
                }
                sphere_instance_buffer->write(render_data.sphere_instances.data(), needed);
            }

            // Upload cylinder instance data (for 3D mode)
            if (cylinder_instance_buffer && !render_data.cylinder_instances.empty()) {
                size_t needed = render_data.cylinder_instances.size() * sizeof(CylinderInstance);
                if (needed > instance_buffer_desc.size) {
                    instance_buffer_desc.size = needed * 2;
                    cylinder_instance_buffer = device->create_buffer(instance_buffer_desc);
                }
                cylinder_instance_buffer->write(render_data.cylinder_instances.data(), needed);
            }

            geometry_dirty = false;
        }

        // Acquire swapchain image
        auto acquire = swapchain->acquire_next_image(image_semaphore.get(), nullptr);
        if (!acquire.success) {
            device->wait_idle();
            swapchain->resize(window->get_width(), window->get_height());
            continue;
        }

        auto* tex = swapchain->get_texture(acquire.image_index);
        uint32_t w = tex->get_size().width;
        uint32_t h = tex->get_size().height;

        math::mat4 vp;
        if (view_mode == ViewMode::View2D) {
            // Orthographic projection for 2D mode
            // Layout is in XY plane, looking down Z axis at XY plane
            // Use camera distance as zoom factor, target as center
            float aspect = static_cast<float>(w) / static_cast<float>(h);
            float zoom = cam.get_distance();  // Distance controls zoom level
            float half_width = zoom * aspect * 0.5f;
            float half_height = zoom * 0.5f;
            math::vec3 target = cam.get_target();

            // Orthographic projection centered on target
            // X is horizontal, Y is vertical (screen layout matches world XY plane)
            math::mat4 ortho_proj = math::mat4::ortho(
                target.x - half_width, target.x + half_width,
                target.y - half_height, target.y + half_height,
                -100.0f, 100.0f  // Near/far for Z clipping
            );
            vp = ortho_proj;  // No view matrix needed - ortho is already world-space
        } else {
            // Perspective projection for 3D mode
            vp = cam.get_view_projection_matrix();
        }

        auto encoder = device->create_command_encoder();

        // Picking render pass (if click pending and in 3D mode with spheres)
        bool picking_result_ready = false;
        if (picking_pass_pending && picking_shaders_available && picking_sphere_pipeline &&
            view_mode == ViewMode::View3D && !render_data.sphere_instances.empty()) {

            // Build picking instance data from sphere instances
            // Use constant picking radius (10x base vertex radius) regardless of display state
            // This prevents selection-dependent sphere size changes from affecting picking
            constexpr float BASE_VERTEX_RADIUS = 0.04f;
            constexpr float PICKING_RADIUS = BASE_VERTEX_RADIUS * 10.0f;

            std::vector<PickingSphereInstance> picking_instances;
            picking_instances.reserve(render_data.sphere_instances.size());

            for (size_t i = 0; i < render_data.sphere_instances.size(); ++i) {
                const auto& si = render_data.sphere_instances[i];
                PickingSphereInstance pi;
                pi.x = si.x;
                pi.y = si.y;
                pi.z = si.z;
                pi.radius = PICKING_RADIUS;  // Constant radius for all vertices
                pi.vertex_index = static_cast<uint32_t>(i);
                pi._pad[0] = 0;
                pi._pad[1] = 0;
                pi._pad[2] = 0;
                picking_instances.push_back(pi);
            }

            // Upload picking instance data
            size_t pick_inst_size = picking_instances.size() * sizeof(PickingSphereInstance);
            picking_instance_buffer->write(picking_instances.data(), pick_inst_size);

            // Picking render pass
            gal::RenderPassColorAttachment pick_color_att;
            pick_color_att.texture = picking_texture.get();
            pick_color_att.load_op = gal::LoadOp::Clear;
            pick_color_att.store_op = gal::StoreOp::Store;
            pick_color_att.clear_color[0] = 0.0f;  // Clear to 0 (no vertex)

            gal::RenderPassDepthAttachment pick_depth_att;
            pick_depth_att.texture = picking_depth_texture.get();
            pick_depth_att.depth_load_op = gal::LoadOp::Clear;
            pick_depth_att.depth_store_op = gal::StoreOp::DontCare;
            pick_depth_att.clear_depth = 1.0f;

            gal::RenderPassBeginInfo pick_rp_info;
            pick_rp_info.pipeline = picking_sphere_pipeline.get();
            pick_rp_info.color_attachments = &pick_color_att;
            pick_rp_info.color_attachment_count = 1;
            pick_rp_info.depth_attachment = pick_depth_att;

            auto pick_rp = encoder->begin_render_pass(pick_rp_info);
            if (pick_rp) {
                pick_rp->set_viewport(0, 0, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f);
                pick_rp->set_scissor(0, 0, w, h);
                pick_rp->set_pipeline(picking_sphere_pipeline.get());
                pick_rp->push_constants(vp.m, sizeof(math::mat4));
                pick_rp->set_vertex_buffer(0, sphere_vb.get());
                pick_rp->set_vertex_buffer(1, picking_instance_buffer.get());
                pick_rp->set_index_buffer(sphere_ib.get(), gal::IndexFormat::Uint16);
                pick_rp->draw_indexed(
                    static_cast<uint32_t>(sphere_mesh.indices.size()),
                    static_cast<uint32_t>(picking_instances.size()),
                    0, 0, 0
                );
                pick_rp->end();

                // Copy the clicked pixel to readback buffer
                // Note: We copy a single pixel at the click location
                gal::BufferTextureCopy copy_region;
                copy_region.buffer_offset = 0;
                copy_region.buffer_row_length = 0;
                copy_region.buffer_image_height = 0;
                copy_region.texture_mip_level = 0;
                copy_region.texture_array_layer = 0;
                copy_region.texture_offset = {
                    static_cast<int32_t>(std::clamp(picking_click_x, 0, static_cast<int>(w) - 1)),
                    static_cast<int32_t>(std::clamp(picking_click_y, 0, static_cast<int>(h) - 1)),
                    0
                };
                copy_region.copy_size = {1, 1, 1};
                encoder->copy_texture_to_buffer(picking_texture.get(), picking_readback_buffer.get(), copy_region);
            }

            // Store vertex IDs for later lookup
            // Sphere instances are built in same order as ts.union_vertices
            if (current_step >= 0 && current_step < static_cast<int>(analysis.per_timestep.size())) {
                picking_vertex_ids = analysis.per_timestep[current_step].union_vertices;
            }

            picking_result_ready = true;
            picking_readback_pending = true;
            picking_pass_pending = false;
        } else if (picking_pass_pending) {
            // 2D mode or no spheres - dismiss and deselect
            vertex_selection.has_selection = false;
            picking_pass_pending = false;
        }

        // Select active pipelines based on MSAA state
        gal::RenderPipeline* active_triangle_pipeline = (msaa_enabled && msaa_triangle_pipeline)
            ? msaa_triangle_pipeline.get() : triangle_pipeline.get();
        gal::RenderPipeline* active_line_pipeline = (msaa_enabled && msaa_line_pipeline)
            ? msaa_line_pipeline.get() : line_pipeline.get();
        gal::RenderPipeline* active_sphere_pipeline = (msaa_enabled && msaa_sphere_pipeline)
            ? msaa_sphere_pipeline.get() : sphere_pipeline.get();
        gal::RenderPipeline* active_cylinder_pipeline = (msaa_enabled && msaa_cylinder_pipeline)
            ? msaa_cylinder_pipeline.get() : cylinder_pipeline.get();

        // Render pass setup
        gal::RenderPassColorAttachment color_att;
        if (msaa_enabled && msaa_color_texture) {
            // MSAA: render to MSAA texture, resolve to swapchain
            color_att.texture = msaa_color_texture.get();
            color_att.resolve_texture = tex;
        } else {
            // Non-MSAA: render directly to swapchain
            color_att.texture = tex;
            color_att.resolve_texture = nullptr;
        }
        color_att.load_op = gal::LoadOp::Clear;
        color_att.store_op = gal::StoreOp::Store;
        color_att.clear_color[0] = 0.05f;
        color_att.clear_color[1] = 0.05f;
        color_att.clear_color[2] = 0.08f;
        color_att.clear_color[3] = 1.0f;

        gal::RenderPassDepthAttachment depth_att;
        depth_att.texture = (msaa_enabled && msaa_depth_texture) ? msaa_depth_texture.get() : depth_texture.get();
        depth_att.depth_load_op = gal::LoadOp::Clear;
        depth_att.depth_store_op = gal::StoreOp::DontCare;
        depth_att.clear_depth = 1.0f;

        gal::RenderPassBeginInfo rp_info;
        rp_info.pipeline = active_triangle_pipeline;
        rp_info.color_attachments = &color_att;
        rp_info.color_attachment_count = 1;
        rp_info.depth_attachment = depth_att;

        auto rp = encoder->begin_render_pass(rp_info);
        if (rp) {
            rp->set_viewport(0, 0, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f);
            rp->set_scissor(0, 0, w, h);

            if (view_mode == ViewMode::View3D ||
                view_mode == ViewMode::ShapeSpace2D ||
                view_mode == ViewMode::ShapeSpace3D) {
                // =====================================================
                // 3D Mode / Shape Space: Instanced rendering with cylinders and spheres
                // =====================================================

                // Draw edges as cylinders (instanced)
                if (active_cylinder_pipeline && !render_data.cylinder_instances.empty()) {
                    rp->set_pipeline(active_cylinder_pipeline);
                    rp->push_constants(vp.m, sizeof(math::mat4));
                    rp->set_vertex_buffer(0, cylinder_vb.get());
                    rp->set_vertex_buffer(1, cylinder_instance_buffer.get());
                    rp->set_index_buffer(cylinder_ib.get(), gal::IndexFormat::Uint16);
                    rp->draw_indexed(
                        static_cast<uint32_t>(cylinder_mesh.indices.size()),
                        static_cast<uint32_t>(render_data.cylinder_instances.size()),
                        0, 0, 0
                    );
                }

                // Draw vertices as spheres (instanced)
                if (active_sphere_pipeline && !render_data.sphere_instances.empty()) {
                    rp->set_pipeline(active_sphere_pipeline);
                    rp->push_constants(vp.m, sizeof(math::mat4));
                    rp->set_vertex_buffer(0, sphere_vb.get());
                    rp->set_vertex_buffer(1, sphere_instance_buffer.get());
                    rp->set_index_buffer(sphere_ib.get(), gal::IndexFormat::Uint16);
                    rp->draw_indexed(
                        static_cast<uint32_t>(sphere_mesh.indices.size()),
                        static_cast<uint32_t>(render_data.sphere_instances.size()),
                        0, 0, 0
                    );
                }
            } else {
                // =====================================================
                // 2D Mode: Legacy line and quad rendering
                // =====================================================

                // Draw edges (lines)
                if (line_pipeline && edge_buffer && render_data.edge_count > 0) {
                    rp->set_pipeline(active_line_pipeline);
                    rp->push_constants(vp.m, sizeof(math::mat4));
                    rp->set_vertex_buffer(0, edge_buffer.get());
                    rp->draw(static_cast<uint32_t>(render_data.edge_count), 1, 0, 0);
                }

                // Draw vertices (quads as triangles)
                if (triangle_pipeline && vertex_buffer && render_data.vertex_count > 0) {
                    rp->set_pipeline(active_triangle_pipeline);
                    rp->push_constants(vp.m, sizeof(math::mat4));
                    rp->set_vertex_buffer(0, vertex_buffer.get());
                    rp->draw(static_cast<uint32_t>(render_data.vertex_count), 1, 0, 0);
                }
            }

            // Draw horizon circles (both modes)
            if (show_horizons && line_pipeline && horizon_buffer && !horizon_verts.empty()) {
                rp->set_pipeline(active_line_pipeline);
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, horizon_buffer.get());
                rp->draw(static_cast<uint32_t>(horizon_verts.size()), 1, 0, 0);
            }

            // Legend data - declared outside timeline block so it persists for text rendering
            LegendData legend;
            float legend_min = 0, legend_max = 3;

            // Draw timeline bar (2D overlay with identity matrix)
            if (triangle_pipeline && timeline_buffer && max_step > 0) {
                // Update window size for hit testing
                window_width = w;
                window_height = h;

                // Build timeline bar geometry (includes buttons)
                last_timeline = build_timeline_bar(current_step, max_step, static_cast<float>(w) / h,
                                                   playing, playback_direction,
                                                   timeslice_enabled, timeslice_width);
                std::vector<Vertex> timeline_verts;
                timeline_verts.insert(timeline_verts.end(), last_timeline.bar_verts.begin(), last_timeline.bar_verts.end());
                timeline_verts.insert(timeline_verts.end(), last_timeline.marker_verts.begin(), last_timeline.marker_verts.end());
                timeline_verts.insert(timeline_verts.end(), last_timeline.button_verts.begin(), last_timeline.button_verts.end());

                // Build and add color legend (always visible)
                if (edge_color_mode == EdgeColorMode::Frequency && render_data.has_freq_data) {
                    // Frequency mode: show frequency range (integer counts)
                    legend_min = static_cast<float>(render_data.min_freq);
                    legend_max = static_cast<float>(render_data.max_freq);
                } else if (viz_mode.is_curvature() && analysis.has_curvature_analysis) {
                    // Curvature mode: symmetric range around zero
                    switch (viz_mode.value_type) {
                        case ValueType::WolframRicciScalar:
                            legend_min = -std::max(std::abs(analysis.curvature_wolfram_scalar_min),
                                                   std::abs(analysis.curvature_wolfram_scalar_max));
                            legend_max = -legend_min;
                            break;
                        case ValueType::WolframRicciTensor:
                            legend_min = -std::max(std::abs(analysis.curvature_wolfram_ricci_min),
                                                   std::abs(analysis.curvature_wolfram_ricci_max));
                            legend_max = -legend_min;
                            break;
                        case ValueType::OllivierRicciCurvature:
                            legend_min = -std::max(std::abs(analysis.curvature_ollivier_min),
                                                   std::abs(analysis.curvature_ollivier_max));
                            legend_max = -legend_min;
                            break;
                        case ValueType::DimensionGradient:
                            legend_min = -std::max(std::abs(analysis.curvature_dim_grad_min),
                                                   std::abs(analysis.curvature_dim_grad_max));
                            legend_max = -legend_min;
                            break;
                        default:
                            legend_min = -1.0f;
                            legend_max = 1.0f;
                            break;
                    }
                } else if (current_step < static_cast<int>(analysis.per_timestep.size())) {
                    const auto& ts = analysis.per_timestep[current_step];
                    if (viz_mode.graph_scope == GraphScope::Global) {
                        // Global mode: constant range from mega-union
                        legend_min = analysis.mega_dim_min;
                        legend_max = analysis.mega_dim_max;
                    } else if (viz_mode.graph_scope == GraphScope::Foliation) {
                        legend_min = viz_mode.statistic_mode == StatisticMode::Variance ? ts.global_var_min : ts.global_mean_min;
                        legend_max = viz_mode.statistic_mode == StatisticMode::Variance ? ts.global_var_max : ts.global_mean_max;
                    } else {
                        // Branchial mode
                        legend_min = viz_mode.statistic_mode == StatisticMode::Variance ? ts.var_min : ts.pooled_min;
                        legend_max = viz_mode.statistic_mode == StatisticMode::Variance ? ts.var_max : ts.pooled_max;
                    }
                }
                legend = build_legend(static_cast<float>(w) / h, current_palette, legend_min, legend_max);
                timeline_verts.insert(timeline_verts.end(), legend.verts.begin(), legend.verts.end());

                // Build and add states graph panel if enabled
                StatesGraphData states_graph;
                if (show_states_graph && !analysis.states_per_step.empty()) {
                    float states_aspect = static_cast<float>(w) / static_cast<float>(h);
                    states_graph = build_states_graph(analysis, current_step, max_step,
                                                       timeslice_enabled, timeslice_width, 0.0f,
                                                       path_selection_enabled, selected_state_indices,
                                                       states_aspect);
                    timeline_verts.insert(timeline_verts.end(), states_graph.verts.begin(), states_graph.verts.end());
                }

                math::mat4 identity = math::mat4::identity();

                // =================================================================
                // ACCUMULATION-BASED RENDERING
                //
                // ARCHITECTURE: To avoid buffer overwrites before GPU execution,
                // we accumulate ALL overlay geometry into ONE buffer and draw ONCE.
                // Same for text - all text uses uniform scale, one buffer, one draw.
                // =================================================================

                gal::RenderPipeline* alpha_pipe = (msaa_enabled && msaa_alpha_triangle_pipeline)
                    ? msaa_alpha_triangle_pipeline.get() : alpha_triangle_pipeline.get();

                // --- STEP 1: Accumulate ALL overlay geometry ---
                std::vector<Vertex> all_overlay_verts;

                // 1a. Add panel backgrounds (drawn first, furthest back in painter's order)
                all_overlay_verts.insert(all_overlay_verts.end(),
                    last_timeline.panel_verts.begin(), last_timeline.panel_verts.end());

                // 1b. Add help overlay background (if visible)
                float overlay_scale = 1.0f;  // Will be set if overlay is shown
                if (show_overlay) {
                    // Use integer multiple of font size for pixel-perfect nearest sampling
                    // With nearest filtering, non-integer scales cause sampling artifacts
                    // 2x scale = 16x16 pixels (from 8x8 base)
                    constexpr float TEXT_SCALE_FACTOR = 2.0f;
                    float target_char_w = FONT_CHAR_WIDTH * TEXT_SCALE_FACTOR;   // 16 pixels
                    float target_char_h = FONT_CHAR_HEIGHT * TEXT_SCALE_FACTOR;  // 16 pixels
                    overlay_scale = TEXT_SCALE_FACTOR;

                    float overlay_char_w = target_char_w;
                    float overlay_char_h = target_char_h;
                    float line_h = overlay_char_h + 3;
                    float margin = 12.0f;

                    float indent = overlay_char_w * 2;
                    float val_col = margin + indent + 42 * overlay_char_w;

                    // Calculate height based on actual text content (matching text rendering below)
                    // Start with top margin
                    float overlay_height = margin;
                    for (const auto& group : keybindings) {
                        overlay_height += line_h;  // Group header
                        overlay_height += group.bindings.size() * line_h;  // Bindings
                        overlay_height += line_h * 0.5f;  // Spacing after group
                    }
                    overlay_height += line_h;  // Frame info line
                    overlay_height += margin;  // Bottom margin (same as top for equal padding)
                    float overlay_width = val_col + 15 * overlay_char_w + margin;

                    float bg_left = (margin - 8) / w * 2.0f - 1.0f;
                    float bg_right = (overlay_width + 8) / w * 2.0f - 1.0f;
                    float bg_top = (margin - 8) / h * 2.0f - 1.0f;
                    float bg_bottom = (overlay_height + 8) / h * 2.0f - 1.0f;
                    Vec4 bg_color{0.0f, 0.0f, 0.0f, 0.5f};
                    float corner_radius = 0.02f;
                    float help_aspect = static_cast<float>(w) / static_cast<float>(h);
                    add_rounded_rect(all_overlay_verts, bg_left, bg_top, bg_right, bg_bottom,
                                     corner_radius, 0.002f, bg_color, help_aspect);
                }

                // 1c. Add timeline UI (bar, markers, buttons, legend, states graph)
                // These go LAST in the buffer so they're drawn on top (painter's algorithm)
                all_overlay_verts.insert(all_overlay_verts.end(),
                    timeline_verts.begin(), timeline_verts.end());

                // 1d. Add histogram panel (if vertex selected)
                HistogramData histogram_data;
                if (vertex_selection.has_selection) {
                    // Check if we need to recompute histogram (timestep, mode, or bin count changed)
                    if (vertex_selection.cached_timestep != current_step ||
                        vertex_selection.cached_graph_scope != viz_mode.graph_scope ||
                        vertex_selection.cached_value_type != viz_mode.value_type ||
                        vertex_selection.cached_bin_count != vertex_selection.histogram_bin_count) {
                        update_selection_histogram(vertex_selection, analysis, current_step, viz_mode.graph_scope, viz_mode.value_type);
                    }

                    histogram_data = build_histogram_panel(
                        vertex_selection,
                        vertex_selection.panel_ndc_x, vertex_selection.panel_ndc_y,
                        static_cast<float>(w) / static_cast<float>(h),
                        w, h,
                        current_palette, viz_mode.graph_scope,
                        legend_min, legend_max
                    );

                    // Update panel bounds for hit testing
                    vertex_selection.panel_left = histogram_data.panel_left;
                    vertex_selection.panel_right = histogram_data.panel_right;
                    vertex_selection.panel_top = histogram_data.panel_top;
                    vertex_selection.panel_bottom = histogram_data.panel_bottom;

                    all_overlay_verts.insert(all_overlay_verts.end(),
                        histogram_data.verts.begin(), histogram_data.verts.end());
                }

                // 1e. Add Hilbert space panel (if enabled and data available)
                HilbertPanelData hilbert_panel_data;
                if (show_hilbert && has_hilbert_analysis) {
                    hilbert_panel_data = build_hilbert_panel(hilbert_result, branchial_result, w, h);
                    all_overlay_verts.insert(all_overlay_verts.end(),
                        hilbert_panel_data.verts.begin(), hilbert_panel_data.verts.end());
                }

                // 1f. Add scatter plot overlay (if enabled)
                ScatterPlotData scatter_data;
                if (show_scatter && !analysis.state_aggregates.empty()) {
                    scatter_data = build_scatter_plot(analysis, scatter_metric, current_step,
                                                       static_cast<float>(w) / static_cast<float>(h));
                    // Axes first (furthest back)
                    all_overlay_verts.insert(all_overlay_verts.end(),
                        scatter_data.axis_verts.begin(), scatter_data.axis_verts.end());
                    // Lines (variance band and mean line)
                    all_overlay_verts.insert(all_overlay_verts.end(),
                        scatter_data.line_verts.begin(), scatter_data.line_verts.end());
                    // Scatter points (one per state at each timestep)
                    all_overlay_verts.insert(all_overlay_verts.end(),
                        scatter_data.point_verts.begin(), scatter_data.point_verts.end());
                    // Current step marker on top
                    all_overlay_verts.insert(all_overlay_verts.end(),
                        scatter_data.marker_verts.begin(), scatter_data.marker_verts.end());
                }

                // --- STEP 2: Single draw call for ALL overlay geometry ---
                if (!all_overlay_verts.empty()) {
                    overlay_buffer->write(all_overlay_verts.data(),
                                          all_overlay_verts.size() * sizeof(Vertex));
                    rp->set_pipeline(alpha_pipe);
                    rp->push_constants(identity.m, sizeof(math::mat4));
                    rp->set_vertex_buffer(0, overlay_buffer.get());
                    rp->draw(static_cast<uint32_t>(all_overlay_verts.size()), 1, 0, 0);
                }

                // --- STEP 3: Accumulate ALL text with UNIFORM scale ---
                // Use integer multiple of font size for pixel-perfect nearest sampling
                // 2x scale = 16x16 pixels (from 8x8 base)
                glyph_instances.clear();
                constexpr float TEXT_SCALE_FACTOR = 2.0f;
                float text_char_w = FONT_CHAR_WIDTH * TEXT_SCALE_FACTOR;   // 16 pixels
                float text_char_h = FONT_CHAR_HEIGHT * TEXT_SCALE_FACTOR;  // 16 pixels
                float text_scale = TEXT_SCALE_FACTOR;

                // 3a. Queue help overlay text (if visible)
                if (show_overlay && text_rendering_available) {
                    float margin = 12.0f;
                    float line_h = text_char_h + 3;
                    float x = margin;
                    float y = margin;

                    float indent = text_char_w * 2;
                    float key_col = x + indent;
                    float desc_col = x + 16 * text_char_w;
                    float val_col = x + 42 * text_char_w;

                    for (const auto& group : keybindings) {
                        queue_text(x, y, group.name, TextColor::Yellow, text_scale);
                        y += line_h;
                        for (const auto& binding : group.bindings) {
                            queue_text(key_col, y, binding.key, TextColor::Cyan, text_scale);
                            queue_text(desc_col, y, binding.description, TextColor::White, text_scale);
                            if (binding.get_value) {
                                std::string val = binding.get_value();
                                Vec4 val_color = TextColor::White;
                                if (val == "ON" || val == "PLAYING" || val == "FORWARD" || val == "VISIBLE") {
                                    val_color = TextColor::Green;
                                } else if (val == "OFF" || val == "PAUSED" || val == "REVERSE" || val == "HIDDEN") {
                                    val_color = TextColor::Red;
                                }
                                queue_text(val_col, y, val, val_color, text_scale);
                            }
                            y += line_h;
                        }
                        y += line_h * 0.5f;
                    }
                    y += line_h;
                    queue_text(x, y, "Frame: " + std::to_string(current_step) + " / " + std::to_string(max_step),
                               TextColor::Orange, text_scale);
                }

                // 3b. Queue legend text
                if (legend.bar_center_x != 0.0f) {
                    auto ndc_to_pixel_x = [&](float ndc) { return (ndc + 1.0f) * 0.5f * w; };
                    auto ndc_to_pixel_y = [&](float ndc) { return (ndc + 1.0f) * 0.5f * h; };

                    std::string desc_line1, desc_line2;
                    if (edge_color_mode == EdgeColorMode::Frequency) {
                        desc_line1 = "State";
                        desc_line2 = "Frequency";
                    } else if (viz_mode.is_curvature()) {
                        desc_line1 = "Curvature";
                        switch (viz_mode.value_type) {
                            case ValueType::WolframRicciScalar: desc_line2 = "Scalar"; break;
                            case ValueType::WolframRicciTensor: desc_line2 = "Ricci"; break;
                            case ValueType::OllivierRicciCurvature: desc_line2 = "Ollivier"; break;
                            case ValueType::DimensionGradient: desc_line2 = "Gradient"; break;
                            default: desc_line2 = "Unknown"; break;
                        }
                    } else {
                        desc_line1 = ui::name(viz_mode.graph_scope);
                        desc_line2 = (viz_mode.statistic_mode == StatisticMode::Variance) ? "Variance" : "Dimension";
                    }

                    std::ostringstream max_ss, min_ss;
                    if (edge_color_mode == EdgeColorMode::Frequency && render_data.has_freq_data) {
                        // Integer values for frequency counts
                        max_ss << static_cast<int>(legend_max);
                        min_ss << static_cast<int>(legend_min);
                    } else {
                        // Floating point values for dimension
                        max_ss << std::fixed << std::setprecision(2) << legend_max;
                        min_ss << std::fixed << std::setprecision(2) << legend_min;
                    }

                    float line_spacing = text_char_h * 1.4f;
                    float bar_center_px = ndc_to_pixel_x(legend.bar_center_x);
                    float desc_y1 = ndc_to_pixel_y(legend.title_y) - line_spacing * 3.5f;
                    float desc_y2 = ndc_to_pixel_y(legend.title_y) - line_spacing * 2.5f;
                    float desc_x1 = bar_center_px - (desc_line1.length() * text_char_w) / 2.0f;
                    float desc_x2 = bar_center_px - (desc_line2.length() * text_char_w) / 2.0f;
                    queue_text(desc_x1, desc_y1, desc_line1, TextColor::White, text_scale);
                    queue_text(desc_x2, desc_y2, desc_line2, TextColor::White, text_scale);

                    float max_x = ndc_to_pixel_x(legend.label_x) - max_ss.str().length() * text_char_w;
                    float max_y = ndc_to_pixel_y(legend.max_y);
                    queue_text(max_x, max_y, max_ss.str(), TextColor::White, text_scale);

                    float min_x = ndc_to_pixel_x(legend.label_x) - min_ss.str().length() * text_char_w;
                    float min_y = ndc_to_pixel_y(legend.min_y) - text_char_h;
                    queue_text(min_x, min_y, min_ss.str(), TextColor::White, text_scale);
                }

                // 3c. Queue histogram text (if visible)
                if (vertex_selection.has_selection && !histogram_data.verts.empty()) {
                    auto ndc_to_pixel_x = [&](float ndc) { return (ndc + 1.0f) * 0.5f * w; };
                    auto ndc_to_pixel_y = [&](float ndc) { return (ndc + 1.0f) * 0.5f * h; };

                    // Title depends on mode
                    std::string title;
                    if (vertex_selection.is_distribution_mode) {
                        // Curvature: distribution of ALL vertices
                        title = "Curvature (" + std::string(value_type_name(viz_mode.value_type)) + ")";
                    } else {
                        // Dimension: values for selected vertex across timesteps
                        title = "Vertex " + std::to_string(vertex_selection.selected_vertex) +
                                " (" + std::string(ui::name(viz_mode.graph_scope)) + ")";
                    }
                    float title_px_x = ndc_to_pixel_x(histogram_data.title_x);
                    float title_px_y = ndc_to_pixel_y(histogram_data.title_y);
                    queue_text(title_px_x, title_px_y, title, TextColor::Cyan, text_scale);

                    // Stats: n=X, mean=Y, std=Z
                    std::ostringstream stats_ss;
                    stats_ss << "n=" << vertex_selection.dimension_values.size()
                             << " mean=" << std::fixed << std::setprecision(2) << vertex_selection.mean_value
                             << " std=" << std::fixed << std::setprecision(2) << vertex_selection.std_dev;
                    float stats_px_x = ndc_to_pixel_x(histogram_data.stats_x);
                    float stats_px_y = ndc_to_pixel_y(histogram_data.stats_y);
                    queue_text(stats_px_x, stats_px_y, stats_ss.str(), TextColor::White, text_scale);

                    // Axis min/max labels
                    std::ostringstream min_ss, max_ss;
                    min_ss << std::fixed << std::setprecision(2) << vertex_selection.bin_min;
                    max_ss << std::fixed << std::setprecision(2) <<
                           (vertex_selection.bin_min + vertex_selection.num_bins * vertex_selection.bin_width);
                    float axis_min_px_x = ndc_to_pixel_x(histogram_data.axis_min_x);
                    float axis_min_px_y = ndc_to_pixel_y(histogram_data.axis_min_y);
                    float axis_max_px_x = ndc_to_pixel_x(histogram_data.axis_max_x);
                    float axis_max_px_y = ndc_to_pixel_y(histogram_data.axis_max_y);
                    queue_text(axis_min_px_x, axis_min_px_y, min_ss.str(), TextColor::White, text_scale);
                    queue_text(axis_max_px_x, axis_max_px_y, max_ss.str(), TextColor::White, text_scale);
                }

                // 3d. Queue Hilbert space panel text (if visible)
                if (show_hilbert && has_hilbert_analysis && !hilbert_panel_data.verts.empty()) {
                    // Title
                    queue_text(hilbert_panel_data.title_x, hilbert_panel_data.title_y,
                               "Hilbert Space Analysis", TextColor::Cyan, text_scale);

                    // Stat lines
                    for (size_t i = 0; i < hilbert_panel_data.stat_labels.size(); ++i) {
                        auto [sx, sy] = hilbert_panel_data.stat_positions[i];
                        queue_text(sx, sy, hilbert_panel_data.stat_labels[i], TextColor::White, text_scale);
                        queue_text(sx + 18 * text_char_w, sy, hilbert_panel_data.stat_values[i],
                                   TextColor::Green, text_scale);
                    }
                }

                // --- STEP 4: Single draw call for ALL text ---
                if (text_rendering_available && text_pipeline &&
                    !glyph_instances.empty() && glyph_instances.size() <= MAX_TEXT_GLYPHS) {
                    text_instance_buffer->write(glyph_instances.data(),
                                                glyph_instances.size() * sizeof(GlyphInstance));
                    gal::RenderPipeline* text_pipe = (msaa_enabled && msaa_text_pipeline)
                        ? msaa_text_pipeline.get() : text_pipeline.get();
                    rp->set_pipeline(text_pipe);
                    float screen_data[4] = {static_cast<float>(w), static_cast<float>(h),
                                            text_char_w, text_char_h};
                    rp->push_constants(screen_data, sizeof(screen_data));
                    rp->set_bind_group(0, text_bind_group.get());
                    rp->set_vertex_buffer(0, text_quad_vb.get());
                    rp->set_vertex_buffer(1, text_instance_buffer.get());
                    rp->set_index_buffer(text_quad_ib.get(), gal::IndexFormat::Uint16);
                    rp->draw_indexed(6, static_cast<uint32_t>(glyph_instances.size()), 0, 0, 0);
                }
            }

            rp->end();
        }

        auto cmd = encoder->finish();
        fence->reset();  // Reset fence just before submit (not earlier, in case acquire fails)
        device->submit(cmd.get(), image_semaphore.get(), render_semaphore.get(), fence.get());
        in_flight_cmd = std::move(encoder);

        swapchain->present(render_semaphore.get());
    }

    // Cleanup
    device->wait_idle();
    device.reset();
    gal::shutdown();

    return 0;
}
