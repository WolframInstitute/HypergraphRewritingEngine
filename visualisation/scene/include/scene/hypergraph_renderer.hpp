// Hypergraph renderer - converts hypergraph data to renderable geometry
// Handles: vertices as spheres, edges as lines/tubes, hyperedges, self-loops, convex hulls

#pragma once

#include "hypergraph_data.hpp"
#include "color_palette.hpp"
#include <math/types.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <numbers>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <iostream>

namespace viz::scene {

// Vertex format for rendering (position + color)
struct RenderVertex {
    float x, y, z;
    float r, g, b, a;
};

// ==========================================================================
// INSTANCED RENDERING STRUCTURES
// ==========================================================================

// Vertex for unit cone mesh (position + normal)
struct ConeVertex {
    float x, y, z;      // Position
    float nx, ny, nz;   // Normal (for future lighting)
};

// Instance data for instanced cone rendering
// Matches layout locations 2-5 in instance_cone.vert
struct ConeInstance {
    float tip_x, tip_y, tip_z;        // Cone tip position (world space)
    float dir_x, dir_y, dir_z;        // Direction (from base to tip, normalized)
    float length;                      // Cone length
    float radius;                      // Cone base radius
    float r, g, b, a;                  // RGBA color
};

// Pre-built unit cone mesh (shared by all instances)
// Unit cone: tip at origin, base at z = -1, radius 1
struct UnitConeMesh {
    std::vector<ConeVertex> vertices;  // Triangle list

    // Generate unit cone mesh with given number of segments
    void generate(int segments = 8) {
        vertices.clear();
        vertices.reserve(segments * 6);  // 2 triangles per segment (side + base)

        for (int i = 0; i < segments; ++i) {
            float angle1 = math::TAU * i / segments;
            float angle2 = math::TAU * (i + 1) / segments;

            float cos1 = std::cos(angle1);
            float sin1 = std::sin(angle1);
            float cos2 = std::cos(angle2);
            float sin2 = std::sin(angle2);

            // Base vertices at z = -1
            // p1 = (cos1, sin1, -1), p2 = (cos2, sin2, -1)
            // Tip at origin (0, 0, 0)

            // Side triangle normal: cross(p2-tip, p1-tip) = cross(p2, p1)
            // For unit cone slant, normal points outward
            // Simplified: approximate normal as pointing radially outward with some Z component
            float slant_z = 1.0f / std::sqrt(2.0f);  // 45 degree slope approx

            // Side triangle: tip, p2, p1 (CCW winding from outside)
            ConeVertex tip = {0, 0, 0, 0, 0, 1};  // Tip normal points up (+Z)
            ConeVertex s1 = {cos1, sin1, -1, cos1 * slant_z, sin1 * slant_z, slant_z};
            ConeVertex s2 = {cos2, sin2, -1, cos2 * slant_z, sin2 * slant_z, slant_z};

            vertices.push_back(tip);
            vertices.push_back(s2);
            vertices.push_back(s1);

            // Base triangle: center at (0,0,-1), vertices p1, p2
            // CCW winding from below (looking at base from -Z direction)
            ConeVertex base_center = {0, 0, -1, 0, 0, -1};  // Normal points -Z
            ConeVertex b1 = {cos1, sin1, -1, 0, 0, -1};
            ConeVertex b2 = {cos2, sin2, -1, 0, 0, -1};

            vertices.push_back(base_center);
            vertices.push_back(b1);
            vertices.push_back(b2);
        }
    }

    size_t vertex_count() const { return vertices.size(); }
    size_t byte_size() const { return vertices.size() * sizeof(ConeVertex); }
    const ConeVertex* data() const { return vertices.data(); }
};

// Vertex for unit sphere mesh (position + normal)
struct SphereVertex {
    float x, y, z;      // Position
    float nx, ny, nz;   // Normal (for lighting)
};

// Instance data for instanced sphere rendering
// Matches layout locations 2-4 in instance_sphere.vert
struct SphereInstance {
    float center_x, center_y, center_z;  // Sphere center (world space)
    float radius;                         // Sphere radius
    float r, g, b, a;                    // RGBA color
};

// Pre-built unit sphere mesh (shared by all instances)
// Unit sphere: centered at origin, radius 1
struct UnitSphereMesh {
    std::vector<SphereVertex> vertices;  // Triangle list

    // Generate unit sphere mesh using UV sphere method
    void generate(int segments = 12, int rings = 8) {
        vertices.clear();
        vertices.reserve(segments * rings * 6);  // 2 triangles per quad

        for (int ring = 0; ring < rings; ++ring) {
            float theta1 = math::PI * ring / rings;
            float theta2 = math::PI * (ring + 1) / rings;

            for (int seg = 0; seg < segments; ++seg) {
                float phi1 = math::TAU * seg / segments;
                float phi2 = math::TAU * (seg + 1) / segments;

                // Four corners of quad on unit sphere
                // For unit sphere, position == normal
                float sin_t1 = std::sin(theta1), cos_t1 = std::cos(theta1);
                float sin_t2 = std::sin(theta2), cos_t2 = std::cos(theta2);
                float sin_p1 = std::sin(phi1), cos_p1 = std::cos(phi1);
                float sin_p2 = std::sin(phi2), cos_p2 = std::cos(phi2);

                // p1 = top-left, p2 = top-right, p3 = bottom-right, p4 = bottom-left
                SphereVertex p1 = {
                    sin_t1 * cos_p1, cos_t1, sin_t1 * sin_p1,  // position
                    sin_t1 * cos_p1, cos_t1, sin_t1 * sin_p1   // normal (same as pos for unit sphere)
                };
                SphereVertex p2 = {
                    sin_t1 * cos_p2, cos_t1, sin_t1 * sin_p2,
                    sin_t1 * cos_p2, cos_t1, sin_t1 * sin_p2
                };
                SphereVertex p3 = {
                    sin_t2 * cos_p2, cos_t2, sin_t2 * sin_p2,
                    sin_t2 * cos_p2, cos_t2, sin_t2 * sin_p2
                };
                SphereVertex p4 = {
                    sin_t2 * cos_p1, cos_t2, sin_t2 * sin_p1,
                    sin_t2 * cos_p1, cos_t2, sin_t2 * sin_p1
                };

                // Two triangles per quad (CCW winding from outside)
                // p1=top-left, p2=top-right, p3=bottom-right, p4=bottom-left
                // CCW from outside: p1 -> p4 -> p3 and p1 -> p3 -> p2
                vertices.push_back(p1);
                vertices.push_back(p4);
                vertices.push_back(p3);

                vertices.push_back(p1);
                vertices.push_back(p3);
                vertices.push_back(p2);
            }
        }
    }

    size_t vertex_count() const { return vertices.size(); }
    size_t byte_size() const { return vertices.size() * sizeof(SphereVertex); }
    const SphereVertex* data() const { return vertices.data(); }
};

// Rendering configuration
// Colors sourced from color_palette.hpp - edit that file to change colors globally
struct HypergraphRenderConfig {
    // Vertex (node) rendering
    float vertex_radius = 0.15f;
    int sphere_segments = 12;
    int sphere_rings = 8;
    math::vec4 default_vertex_color = colors::VERTEX_SPHERE;

    // Edge rendering
    float edge_thickness = 0.02f;  // For tube rendering (future)
    math::vec4 default_edge_color = colors::HYPEREDGE_LINE;

    // Arrowhead rendering (cones at edge endpoints)
    float arrow_length = 0.12f;
    float arrow_radius = 0.05f;
    int arrow_segments = 8;
    math::vec4 arrow_color = colors::HYPEREDGE_ARROW;

    // Hyperedge bubble rendering
    float bubble_extrusion = 0.2f;  // Minkowski sum radius
    math::vec4 bubble_color = colors::HYPEREDGE_BUBBLE;

    // Self-loop rendering
    float self_loop_offset = 0.25f;  // Virtual vertex offset from line
    math::vec4 self_loop_color = colors::SELF_LOOP;

    // State container rendering
    float state_size = 1.5f;       // Cube size (larger for visibility)
    float state_face_alpha = colors::STATE_CUBE_FACE_ALPHA;
    float state_padding = 0.5f;    // Padding around contained hypergraph
    math::vec4 state_color = colors::STATE_CUBE_NORMAL;
    math::vec4 state_initial_color = colors::STATE_CUBE_INITIAL;
    math::vec4 state_canonical_color = colors::STATE_CUBE_CANONICAL;

    // Evolution edge colors by type
    math::vec4 event_edge_color = colors::EVENT_EDGE;
    float event_arrow_length = 0.2f;    // Length of event edge arrowheads
    float event_arrow_radius = 0.08f;   // Radius of event edge arrowheads
    math::vec4 causal_edge_color = colors::CAUSAL_EDGE;
    math::vec4 branchial_edge_color = colors::BRANCHIAL_EDGE;

    // Performance options
    bool render_internal_hypergraphs = true;  // Render hypergraph inside each state cube
    bool use_instanced_rendering = true;      // Use instanced rendering for spheres/cones (skip legacy triangles)
};

// Generated geometry for a hypergraph
struct HypergraphGeometry {
    // Vertex spheres (triangles)
    // LEGACY: Still populated for backwards compatibility, but prefer sphere_instances
    std::vector<RenderVertex> vertex_triangles;

    // Vertex sphere instances (for instanced rendering)
    // Use this with UnitSphereMesh for efficient rendering
    std::vector<SphereInstance> sphere_instances;

    // Edge lines (line segments)
    std::vector<RenderVertex> edge_lines;

    // Arrowhead cones (triangles) - rendered with triangles for directionality
    // LEGACY: Still populated for backwards compatibility, but prefer cone_instances
    std::vector<RenderVertex> arrow_triangles;

    // Arrowhead cone instances (for instanced rendering)
    // Use this with UnitConeMesh for efficient rendering
    std::vector<ConeInstance> cone_instances;

    // Hyperedge bubbles (triangles, translucent)
    std::vector<RenderVertex> bubble_triangles;

    // Self-loop arcs (line segments to virtual vertices)
    std::vector<RenderVertex> self_loop_lines;

    void clear() {
        vertex_triangles.clear();
        sphere_instances.clear();
        edge_lines.clear();
        arrow_triangles.clear();
        cone_instances.clear();
        bubble_triangles.clear();
        self_loop_lines.clear();
    }

    bool empty() const {
        return vertex_triangles.empty() && sphere_instances.empty() &&
               edge_lines.empty() &&
               arrow_triangles.empty() && cone_instances.empty() &&
               bubble_triangles.empty() && self_loop_lines.empty();
    }

    // Add a sphere instance (for instanced rendering)
    void add_sphere_instance(const math::vec3& center, float radius, const math::vec4& color) {
        sphere_instances.push_back({
            center.x, center.y, center.z,
            radius,
            color.x, color.y, color.z, color.w
        });
    }

    // Add a cone instance (for instanced rendering)
    void add_cone_instance(const math::vec3& tip, const math::vec3& direction,
                           float length, float radius, const math::vec4& color) {
        math::vec3 dir = math::normalize(direction);
        cone_instances.push_back({
            tip.x, tip.y, tip.z,
            dir.x, dir.y, dir.z,
            length, radius,
            color.x, color.y, color.z, color.w
        });
    }
};

// Layout positions for a hypergraph
struct HypergraphLayout {
    std::vector<math::vec3> vertex_positions;

    // Virtual vertex positions for self-loops (per hyperedge, per loop)
    // Key: edge_id * 1000 + loop_index
    std::vector<std::pair<uint32_t, math::vec3>> virtual_positions;

    math::vec3 get_vertex_pos(VertexId v) const {
        return v < vertex_positions.size() ? vertex_positions[v] : math::vec3(0, 0, 0);
    }

    // Compute bounding box
    void get_bounds(math::vec3& min_bound, math::vec3& max_bound) const {
        if (vertex_positions.empty()) {
            min_bound = max_bound = math::vec3(0, 0, 0);
            return;
        }

        min_bound = max_bound = vertex_positions[0];
        for (const auto& p : vertex_positions) {
            min_bound.x = std::min(min_bound.x, p.x);
            min_bound.y = std::min(min_bound.y, p.y);
            min_bound.z = std::min(min_bound.z, p.z);
            max_bound.x = std::max(max_bound.x, p.x);
            max_bound.y = std::max(max_bound.y, p.y);
            max_bound.z = std::max(max_bound.z, p.z);
        }
    }

    math::vec3 get_center() const {
        math::vec3 min_b, max_b;
        get_bounds(min_b, max_b);
        return (min_b + max_b) * 0.5f;
    }
};

// Hypergraph geometry generator
class HypergraphRenderer {
public:
    HypergraphRenderer() = default;

    // Set rendering configuration
    void set_config(const HypergraphRenderConfig& config) { config_ = config; }
    const HypergraphRenderConfig& get_config() const { return config_; }

    // Generate geometry for a hypergraph with given layout
    HypergraphGeometry generate(const Hypergraph& hg, const HypergraphLayout& layout) {
        HypergraphGeometry geo;

        // Generate vertex spheres only for actually used vertices
        for (VertexId v : hg.get_vertices()) {
            math::vec3 pos = layout.get_vertex_pos(v);
            if (config_.use_instanced_rendering) {
                // Instanced rendering: just store instance data (center, radius, color)
                geo.add_sphere_instance(pos, config_.vertex_radius, config_.default_vertex_color);
            } else {
                // Legacy rendering: generate full triangulated sphere
                generate_sphere(geo.vertex_triangles, pos, config_.vertex_radius,
                               config_.default_vertex_color);
            }
        }

        // Pre-compute all loop virtual positions using optimal spherical distribution
        auto loop_virtuals = compute_all_loop_virtuals(hg, layout);

        // Build lookup: (hyperedge_idx, loop_idx_in_edge) -> virtual_pos
        std::map<std::pair<size_t, size_t>, math::vec3> loop_virtual_map;
        for (const auto& occ : loop_virtuals) {
            loop_virtual_map[{occ.hyperedge_idx, occ.loop_idx_in_edge}] = occ.virtual_pos;
        }

        // Generate edges from hyperedges (near-line-graph representation)
        for (size_t e = 0; e < hg.edges.size(); ++e) {
            generate_hyperedge_with_virtuals(geo, hg.edges[e], layout, e, loop_virtual_map);
        }

        return geo;
    }

    // Generate geometry for a single vertex sphere
    void generate_sphere(std::vector<RenderVertex>& out,
                        const math::vec3& center, float radius,
                        const math::vec4& color) {
        int segments = config_.sphere_segments;
        int rings = config_.sphere_rings;

        for (int ring = 0; ring < rings; ++ring) {
            float theta1 = math::PI * ring / rings;
            float theta2 = math::PI * (ring + 1) / rings;

            for (int seg = 0; seg < segments; ++seg) {
                float phi1 = math::TAU * seg / segments;
                float phi2 = math::TAU * (seg + 1) / segments;

                // Four corners of quad
                math::vec3 p1 = sphere_point(center, radius, theta1, phi1);
                math::vec3 p2 = sphere_point(center, radius, theta1, phi2);
                math::vec3 p3 = sphere_point(center, radius, theta2, phi2);
                math::vec3 p4 = sphere_point(center, radius, theta2, phi1);

                // Two triangles
                out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
                out.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
                out.push_back({p3.x, p3.y, p3.z, color.x, color.y, color.z, color.w});

                out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
                out.push_back({p3.x, p3.y, p3.z, color.x, color.y, color.z, color.w});
                out.push_back({p4.x, p4.y, p4.z, color.x, color.y, color.z, color.w});
            }
        }
    }

    // Generate a line segment
    void generate_line(std::vector<RenderVertex>& out,
                      const math::vec3& p1, const math::vec3& p2,
                      const math::vec4& color) {
        out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
        out.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
    }

    // Generate a cone (arrowhead) pointing in direction from base to tip
    // Legacy version - outputs triangles only
    void generate_cone(std::vector<RenderVertex>& out,
                      const math::vec3& tip, const math::vec3& direction,
                      float length, float radius,
                      const math::vec4& color) {
        // Cone base center
        math::vec3 dir = math::normalize(direction);
        math::vec3 base_center = tip - dir * length;

        // Find perpendicular vectors for the base circle
        math::vec3 perp1 = math::cross(dir, math::vec3(0, 1, 0));
        if (math::length(perp1) < 0.001f) {
            perp1 = math::cross(dir, math::vec3(1, 0, 0));
        }
        perp1 = math::normalize(perp1);
        math::vec3 perp2 = math::cross(dir, perp1);

        int segments = config_.arrow_segments;
        for (int i = 0; i < segments; ++i) {
            float angle1 = math::TAU * i / segments;
            float angle2 = math::TAU * (i + 1) / segments;

            math::vec3 p1 = base_center + (perp1 * std::cos(angle1) + perp2 * std::sin(angle1)) * radius;
            math::vec3 p2 = base_center + (perp1 * std::cos(angle2) + perp2 * std::sin(angle2)) * radius;

            // Side triangle: tip, p2, p1 (CCW winding when viewed from outside)
            // p1→p2 goes CCW around base, so tip→p2→p1 is CCW from outside
            out.push_back({tip.x, tip.y, tip.z, color.x, color.y, color.z, color.w});
            out.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
            out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});

            // Base triangle: base_center, p1, p2 (CCW winding when viewed from below)
            out.push_back({base_center.x, base_center.y, base_center.z, color.x, color.y, color.z, color.w});
            out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
            out.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
        }
    }

    // Generate a cone with instanced rendering support
    // Based on config, outputs either legacy triangles OR instance data (not both)
    void generate_cone_instanced(HypergraphGeometry& geo,
                                 const math::vec3& tip, const math::vec3& direction,
                                 float length, float radius,
                                 const math::vec4& color) {
        if (config_.use_instanced_rendering) {
            // Instanced rendering: just store instance data
            geo.add_cone_instance(tip, direction, length, radius, color);
        } else {
            // Legacy rendering: generate full triangulated cone
            generate_cone(geo.arrow_triangles, tip, direction, length, radius, color);
        }
    }

    // Optimal spherical point distributions for N points
    // Returns unit vectors for optimal placement on a sphere
    std::vector<math::vec3> get_spherical_distribution(size_t n) {
        std::vector<math::vec3> dirs;
        if (n == 0) return dirs;

        if (n == 1) {
            dirs.push_back({1, 0, 0});
        }
        else if (n == 2) {
            // Antipodal
            dirs.push_back({1, 0, 0});
            dirs.push_back({-1, 0, 0});
        }
        else if (n == 3) {
            // Equilateral triangle on equator
            for (size_t i = 0; i < 3; ++i) {
                float angle = math::TAU * i / 3.0f;
                dirs.push_back({std::cos(angle), 0, std::sin(angle)});
            }
        }
        else if (n == 4) {
            // Tetrahedron vertices
            // One vertex at top, three forming base
            float h = 1.0f / 3.0f;  // height of base below center
            float r = std::sqrt(8.0f / 9.0f);  // radius of base
            dirs.push_back({0, 1, 0});  // top
            for (size_t i = 0; i < 3; ++i) {
                float angle = math::TAU * i / 3.0f;
                dirs.push_back({r * std::cos(angle), -h, r * std::sin(angle)});
            }
            // Normalize
            for (auto& d : dirs) d = math::normalize(d);
        }
        else if (n == 5) {
            // Triangular bipyramid (two poles + equatorial triangle)
            dirs.push_back({0, 1, 0});   // top
            dirs.push_back({0, -1, 0});  // bottom
            for (size_t i = 0; i < 3; ++i) {
                float angle = math::TAU * i / 3.0f;
                dirs.push_back({std::cos(angle), 0, std::sin(angle)});
            }
        }
        else if (n == 6) {
            // Octahedron vertices
            dirs.push_back({1, 0, 0});
            dirs.push_back({-1, 0, 0});
            dirs.push_back({0, 1, 0});
            dirs.push_back({0, -1, 0});
            dirs.push_back({0, 0, 1});
            dirs.push_back({0, 0, -1});
        }
        else {
            // Fibonacci sphere for N > 6
            float golden_ratio = (1.0f + std::sqrt(5.0f)) / 2.0f;
            for (size_t i = 0; i < n; ++i) {
                float theta = math::TAU * i / golden_ratio;
                float phi = std::acos(1.0f - 2.0f * (i + 0.5f) / n);
                dirs.push_back({
                    std::sin(phi) * std::cos(theta),
                    std::cos(phi),
                    std::sin(phi) * std::sin(theta)
                });
            }
        }
        return dirs;
    }

    // Structure to track loops across all hyperedges for a vertex
    struct VertexLoopInfo {
        VertexId vertex;
        math::vec3 position;
        size_t total_loops = 0;
        std::vector<math::vec3> virtual_directions;  // Unit vectors for each loop
    };

    // Loop info for a single loop occurrence
    struct LoopOccurrence {
        VertexId vertex;
        size_t hyperedge_idx;
        size_t loop_idx_in_edge;  // Which loop within the hyperedge
        math::vec3 virtual_pos;   // Computed virtual vertex position
    };

    // Collect all loops from all hyperedges, compute optimal virtual positions
    // Takes into account edge directions to avoid placing loops where edges already go
    std::vector<LoopOccurrence> compute_all_loop_virtuals(
            const Hypergraph& hg,
            const HypergraphLayout& layout) {

        // First pass: count loops per vertex AND collect edge directions
        std::map<VertexId, size_t> loop_counts;
        std::map<VertexId, std::vector<math::vec3>> edge_directions;  // Normalized dirs to neighbors

        for (const auto& edge : hg.edges) {
            for (size_t i = 0; i + 1 < edge.vertices.size(); ++i) {
                VertexId v1 = edge.vertices[i];
                VertexId v2 = edge.vertices[i + 1];

                if (v1 == v2) {
                    // Self-loop
                    loop_counts[v1]++;
                } else {
                    // Normal edge - record direction from v1 to v2 and v2 to v1
                    math::vec3 p1 = layout.get_vertex_pos(v1);
                    math::vec3 p2 = layout.get_vertex_pos(v2);
                    math::vec3 dir = p2 - p1;
                    if (math::length(dir) > 0.001f) {
                        dir = math::normalize(dir);
                        edge_directions[v1].push_back(dir);
                        edge_directions[v2].push_back(dir * -1.0f);
                    }
                }
            }
        }

        // For each vertex with loops, compute optimal loop directions
        // avoiding the edge directions
        std::map<VertexId, std::vector<math::vec3>> vertex_loop_dirs;

        for (const auto& [v, loop_count] : loop_counts) {
            // Get edge directions at this vertex
            std::vector<math::vec3> edges = edge_directions[v];

            // Compute "centroid" of edge directions (average direction of edges)
            math::vec3 edge_centroid(0, 0, 0);
            for (const auto& ed : edges) {
                edge_centroid = edge_centroid + ed;
            }

            // Get base spherical distribution for loops
            auto base_dirs = get_spherical_distribution(loop_count);

            // If there are edges, rotate the distribution away from them
            if (edges.size() > 0 && math::length(edge_centroid) > 0.001f) {
                edge_centroid = math::normalize(edge_centroid);

                // We want loops in the opposite hemisphere from the edge centroid
                // Compute rotation to align base distribution away from edge_centroid
                math::vec3 away = edge_centroid * -1.0f;

                // Find rotation from (1,0,0) to 'away' direction
                // Then apply to all base directions
                math::vec3 from(1, 0, 0);
                math::vec3 axis = math::cross(from, away);
                float axis_len = math::length(axis);

                if (axis_len > 0.001f) {
                    axis = axis / axis_len;
                    float angle = std::acos(std::max(-1.0f, std::min(1.0f, math::dot(from, away))));

                    // Rodrigues rotation for each direction
                    for (auto& dir : base_dirs) {
                        float cos_a = std::cos(angle);
                        float sin_a = std::sin(angle);
                        math::vec3 rotated = dir * cos_a +
                                            math::cross(axis, dir) * sin_a +
                                            axis * math::dot(axis, dir) * (1.0f - cos_a);
                        dir = math::normalize(rotated);
                    }
                }
            }

            vertex_loop_dirs[v] = base_dirs;
        }

        // Second pass: assign virtual positions to each loop
        std::map<VertexId, size_t> vertex_next_idx;
        for (const auto& [v, _] : loop_counts) {
            vertex_next_idx[v] = 0;
        }

        std::vector<LoopOccurrence> occurrences;
        float offset = config_.self_loop_offset;

        for (size_t e = 0; e < hg.edges.size(); ++e) {
            const auto& edge = hg.edges[e];
            size_t loop_in_edge = 0;

            for (size_t i = 0; i + 1 < edge.vertices.size(); ++i) {
                if (edge.vertices[i] == edge.vertices[i + 1]) {
                    VertexId v = edge.vertices[i];
                    math::vec3 center = layout.get_vertex_pos(v);

                    // Get next available direction for this vertex
                    size_t dir_idx = vertex_next_idx[v]++;
                    math::vec3 dir = vertex_loop_dirs[v][dir_idx];

                    LoopOccurrence occ;
                    occ.vertex = v;
                    occ.hyperedge_idx = e;
                    occ.loop_idx_in_edge = loop_in_edge++;
                    occ.virtual_pos = center + dir * offset;
                    occurrences.push_back(occ);
                }
            }
        }

        return occurrences;
    }

    // Generate geometry for a hyperedge using pre-computed virtual positions
    void generate_hyperedge_with_virtuals(
            HypergraphGeometry& geo,
            const Hyperedge& edge,
            const HypergraphLayout& layout,
            size_t hyperedge_idx,
            const std::map<std::pair<size_t, size_t>, math::vec3>& loop_virtual_map) {

        if (edge.vertices.size() < 2) return;

        // Get positions
        std::vector<math::vec3> positions;
        for (auto v : edge.vertices) {
            positions.push_back(layout.get_vertex_pos(v));
        }

        // Process loops using pre-computed virtual positions
        // Only generate film for arity 3+ edges (2-edge self-loops like {x,x} get no film)
        bool should_fill_loop = (edge.vertices.size() >= 3);

        size_t loop_idx = 0;
        for (size_t i = 0; i + 1 < edge.vertices.size(); ++i) {
            if (edge.vertices[i] == edge.vertices[i + 1]) {
                // Self-loop - look up pre-computed virtual position
                auto it = loop_virtual_map.find({hyperedge_idx, loop_idx});
                if (it != loop_virtual_map.end()) {
                    math::vec3 center = positions[i];
                    math::vec3 virtual_pos = it->second;

                    // Generate smooth arc with optional film
                    // Film is only generated for arity 3+ edges
                    generate_smooth_loop_arc(geo, center, virtual_pos,
                                            config_.self_loop_color,
                                            should_fill_loop);
                }
                loop_idx++;
            }
        }

        // Generate normal edges between consecutive vertices (non-loops)
        for (size_t i = 0; i + 1 < edge.vertices.size(); ++i) {
            VertexId v1 = edge.vertices[i];
            VertexId v2 = edge.vertices[i + 1];

            if (v1 != v2) {
                // Normal edge with line
                math::vec3 p1 = positions[i];
                math::vec3 p2 = positions[i + 1];
                generate_line(geo.edge_lines, p1, p2, config_.default_edge_color);

                // Add arrowhead pointing toward p2 (destination)
                math::vec3 dir = p2 - p1;
                float dist = math::length(dir);
                if (dist > config_.arrow_length * 2) {
                    math::vec3 arrow_tip = p2 - math::normalize(dir) * config_.vertex_radius;
                    generate_cone_instanced(geo, arrow_tip, dir,
                                           config_.arrow_length, config_.arrow_radius,
                                           config_.arrow_color);
                }
            }
        }

        // Generate convex hull bubble for arity 3+ hyperedges
        // For self-loops, include virtual positions in the bubble calculation
        // so that {x, x, x} generates a bubble around the loop arcs
        std::vector<math::vec3> bubble_positions;

        // First, collect unique vertex positions
        for (size_t i = 0; i < positions.size(); ++i) {
            bool duplicate = false;
            for (const auto& p : bubble_positions) {
                if (math::length(p - positions[i]) < 0.001f) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) bubble_positions.push_back(positions[i]);
        }

        // For arity 3+ edges, also include loop virtual positions in bubble
        // This ensures self-loop hyperedges like {x,x,x} get a bubble
        if (edge.vertices.size() >= 3) {
            size_t loop_idx = 0;
            for (size_t i = 0; i + 1 < edge.vertices.size(); ++i) {
                if (edge.vertices[i] == edge.vertices[i + 1]) {
                    auto it = loop_virtual_map.find({hyperedge_idx, loop_idx});
                    if (it != loop_virtual_map.end()) {
                        bubble_positions.push_back(it->second);
                    }
                    loop_idx++;
                }
            }
        }

        // Generate bubble if we have enough points
        if (bubble_positions.size() >= 3) {
            generate_bubble_for_arity(geo.bubble_triangles, bubble_positions,
                                      config_.bubble_color);
        }
    }

    // Generate smooth self-loop that circles AROUND the virtual vertex
    // Also generates an arrowhead to show direction
    // generate_film: if true, fills the loop with triangles; if false, only draws outline
    //                For arity-2 self-loops ({x,x}), we don't want film - just the arc
    //                For higher arity edges with self-loops, we want film to show the loop region
    void generate_smooth_loop_arc(HypergraphGeometry& geo,
                                  const math::vec3& center,
                                  const math::vec3& virtual_pos,
                                  const math::vec4& color,
                                  bool generate_film = true) {
        // Create an elliptical loop that:
        // - Starts at center vertex
        // - Loops around the virtual vertex
        // - Returns to center vertex

        // The loop is an ellipse centered at the midpoint between center and virtual
        // with the major axis along center→virtual direction

        math::vec3 to_virtual = virtual_pos - center;
        float dist = math::length(to_virtual);
        if (dist < 0.001f) return;

        math::vec3 dir = to_virtual / dist;

        // Find perpendicular direction for the loop width
        math::vec3 perp = math::cross(dir, math::vec3(0, 1, 0));
        if (math::length(perp) < 0.001f) {
            perp = math::cross(dir, math::vec3(1, 0, 0));
        }
        perp = math::normalize(perp);

        // Loop parameters:
        // The loop should START at center, go AROUND the virtual vertex, and RETURN to center
        // At angle=π, we should be PAST the virtual vertex (further from center)
        float loop_width = dist * 0.5f;   // How wide the loop is (perpendicular)
        float overshoot = dist * 0.3f;    // How far past virtual the loop extends

        const int segments = 24;  // Smooth curve
        std::vector<math::vec3> loop_points;

        // Generate loop that encircles the virtual vertex
        // At angle=0,2π: at center vertex
        // At angle=π: past virtual vertex by 'overshoot' amount
        for (int i = 0; i <= segments; ++i) {
            float t = static_cast<float>(i) / segments;
            float angle = t * math::TAU;  // Full circle

            // Distance from center along the dir axis:
            // - At angle=0: r=0 (at center)
            // - At angle=π: r=dist+overshoot (past virtual)
            float r = (dist + overshoot) * (1.0f - std::cos(angle)) * 0.5f;

            // Lateral offset (perpendicular to dir):
            // Creates the width of the loop
            float lateral = loop_width * std::sin(angle);

            math::vec3 pt = center + dir * r + perp * lateral;
            loop_points.push_back(pt);
        }

        // Generate line segments for the loop outline
        for (size_t i = 0; i + 1 < loop_points.size(); ++i) {
            generate_line(geo.self_loop_lines, loop_points[i], loop_points[i + 1], color);
        }

        // Generate arrowhead at the "return" part of the loop (around 3/4 of the way)
        // to show the direction of the loop
        size_t arrow_idx = segments * 3 / 4;  // 75% around the loop (heading back to center)
        if (arrow_idx > 0 && arrow_idx < loop_points.size() - 1) {
            math::vec3 arrow_pos = loop_points[arrow_idx];
            math::vec3 arrow_dir = loop_points[arrow_idx + 1] - loop_points[arrow_idx - 1];
            if (math::length(arrow_dir) > 0.001f) {
                arrow_dir = math::normalize(arrow_dir);
                // Darker arrow color
                math::vec4 arrow_color = {color.x * 0.7f, color.y * 0.7f, color.z * 0.7f, 1.0f};
                float arrow_len = config_.arrow_length * 0.8f;
                float arrow_rad = config_.arrow_radius * 0.8f;
                // Use instanced cones when available, fallback to triangles
                if (config_.use_instanced_rendering) {
                    geo.add_cone_instance(arrow_pos, arrow_dir, arrow_len, arrow_rad, arrow_color);
                } else {
                    generate_cone(geo.arrow_triangles, arrow_pos, arrow_dir, arrow_len, arrow_rad, arrow_color);
                }
            }
        }

        // Generate film (fan triangles from center to loop) - only if requested
        // For arity-2 self-loops, we skip the film to show just the arc
        if (generate_film) {
            math::vec4 film_color = {color.x, color.y, color.z, color.w * 0.5f};
            for (size_t i = 0; i + 1 < loop_points.size(); ++i) {
                // Triangle: center, loop[i], loop[i+1]
                geo.bubble_triangles.push_back({center.x, center.y, center.z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                geo.bubble_triangles.push_back({loop_points[i].x, loop_points[i].y, loop_points[i].z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                geo.bubble_triangles.push_back({loop_points[i+1].x, loop_points[i+1].y, loop_points[i+1].z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                // Back face
                geo.bubble_triangles.push_back({center.x, center.y, center.z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                geo.bubble_triangles.push_back({loop_points[i+1].x, loop_points[i+1].y, loop_points[i+1].z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                geo.bubble_triangles.push_back({loop_points[i].x, loop_points[i].y, loop_points[i].z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
            }
        }
    }

    // Generate bubble based on arity (number of unique vertices)
    // This is the Minkowski sum of the convex hull with a sphere
    // Only for arity 3+: uses unified algorithm for all cases
    // Arity 1-2: No bubble (edges are shown as lines only)
    void generate_bubble_for_arity(std::vector<RenderVertex>& out,
                                   const std::vector<math::vec3>& points,
                                   const math::vec4& color) {
        // No bubbles for arity 1 or 2 - only lines
        if (points.size() < 3) return;

        float radius = config_.bubble_extrusion;

        // Unified Minkowski sum for all arities (3+)
        generate_minkowski_sum_fast(out, points, radius, color);
    }

    // Generate a capsule between two points (cylinder + hemispherical caps)
    void generate_capsule(std::vector<RenderVertex>& out,
                         const math::vec3& p1, const math::vec3& p2,
                         float radius, const math::vec4& color) {
        math::vec3 axis = p2 - p1;
        float length = math::length(axis);
        if (length < 0.001f) {
            generate_sphere(out, p1, radius, color);
            return;
        }
        math::vec3 dir = axis / length;

        // Find perpendicular vectors
        math::vec3 perp1 = math::cross(dir, math::vec3(0, 1, 0));
        if (math::length(perp1) < 0.001f) {
            perp1 = math::cross(dir, math::vec3(1, 0, 0));
        }
        perp1 = math::normalize(perp1);
        math::vec3 perp2 = math::cross(dir, perp1);

        int segments = 12;
        int rings = 6;

        // Cylinder body
        for (int i = 0; i < segments; ++i) {
            float angle1 = math::TAU * i / segments;
            float angle2 = math::TAU * (i + 1) / segments;

            math::vec3 offset1 = (perp1 * std::cos(angle1) + perp2 * std::sin(angle1)) * radius;
            math::vec3 offset2 = (perp1 * std::cos(angle2) + perp2 * std::sin(angle2)) * radius;

            math::vec3 v1 = p1 + offset1;
            math::vec3 v2 = p1 + offset2;
            math::vec3 v3 = p2 + offset2;
            math::vec3 v4 = p2 + offset1;

            // Two triangles for quad
            out.push_back({v1.x, v1.y, v1.z, color.x, color.y, color.z, color.w});
            out.push_back({v2.x, v2.y, v2.z, color.x, color.y, color.z, color.w});
            out.push_back({v3.x, v3.y, v3.z, color.x, color.y, color.z, color.w});

            out.push_back({v1.x, v1.y, v1.z, color.x, color.y, color.z, color.w});
            out.push_back({v3.x, v3.y, v3.z, color.x, color.y, color.z, color.w});
            out.push_back({v4.x, v4.y, v4.z, color.x, color.y, color.z, color.w});
        }

        // Hemisphere at p1 (pointing away from p2)
        generate_hemisphere(out, p1, -dir, radius, color, segments, rings);

        // Hemisphere at p2 (pointing away from p1)
        generate_hemisphere(out, p2, dir, radius, color, segments, rings);
    }

    // Generate a hemisphere
    void generate_hemisphere(std::vector<RenderVertex>& out,
                            const math::vec3& center, const math::vec3& pole_dir,
                            float radius, const math::vec4& color,
                            int segments, int rings) {
        math::vec3 dir = math::normalize(pole_dir);

        // Find perpendicular vectors
        math::vec3 perp1 = math::cross(dir, math::vec3(0, 1, 0));
        if (math::length(perp1) < 0.001f) {
            perp1 = math::cross(dir, math::vec3(1, 0, 0));
        }
        perp1 = math::normalize(perp1);
        math::vec3 perp2 = math::cross(dir, perp1);

        for (int ring = 0; ring < rings; ++ring) {
            float theta1 = (math::PI / 2.0f) * ring / rings;
            float theta2 = (math::PI / 2.0f) * (ring + 1) / rings;

            float r1 = std::cos(theta1) * radius;
            float r2 = std::cos(theta2) * radius;
            float h1 = std::sin(theta1) * radius;
            float h2 = std::sin(theta2) * radius;

            for (int seg = 0; seg < segments; ++seg) {
                float phi1 = math::TAU * seg / segments;
                float phi2 = math::TAU * (seg + 1) / segments;

                math::vec3 q1 = center + dir * h1 + (perp1 * std::cos(phi1) + perp2 * std::sin(phi1)) * r1;
                math::vec3 q2 = center + dir * h1 + (perp1 * std::cos(phi2) + perp2 * std::sin(phi2)) * r1;
                math::vec3 q3 = center + dir * h2 + (perp1 * std::cos(phi2) + perp2 * std::sin(phi2)) * r2;
                math::vec3 q4 = center + dir * h2 + (perp1 * std::cos(phi1) + perp2 * std::sin(phi1)) * r2;

                out.push_back({q1.x, q1.y, q1.z, color.x, color.y, color.z, color.w});
                out.push_back({q2.x, q2.y, q2.z, color.x, color.y, color.z, color.w});
                out.push_back({q3.x, q3.y, q3.z, color.x, color.y, color.z, color.w});

                out.push_back({q1.x, q1.y, q1.z, color.x, color.y, color.z, color.w});
                out.push_back({q3.x, q3.y, q3.z, color.x, color.y, color.z, color.w});
                out.push_back({q4.x, q4.y, q4.z, color.x, color.y, color.z, color.w});
            }
        }
    }

    // Generate a cylinder between two points (no caps - for edge rounding)
    void generate_cylinder(std::vector<RenderVertex>& out,
                          const math::vec3& p1, const math::vec3& p2,
                          float radius, const math::vec4& color, int segments = 8) {
        math::vec3 axis = p2 - p1;
        float length = math::length(axis);
        if (length < 0.001f) return;
        math::vec3 dir = axis / length;

        math::vec3 perp1 = math::cross(dir, math::vec3(0, 1, 0));
        if (math::length(perp1) < 0.001f) {
            perp1 = math::cross(dir, math::vec3(1, 0, 0));
        }
        perp1 = math::normalize(perp1);
        math::vec3 perp2 = math::cross(dir, perp1);

        for (int i = 0; i < segments; ++i) {
            float angle1 = math::TAU * i / segments;
            float angle2 = math::TAU * (i + 1) / segments;

            math::vec3 offset1 = (perp1 * std::cos(angle1) + perp2 * std::sin(angle1)) * radius;
            math::vec3 offset2 = (perp1 * std::cos(angle2) + perp2 * std::sin(angle2)) * radius;

            math::vec3 v1 = p1 + offset1;
            math::vec3 v2 = p1 + offset2;
            math::vec3 v3 = p2 + offset2;
            math::vec3 v4 = p2 + offset1;

            out.push_back({v1.x, v1.y, v1.z, color.x, color.y, color.z, color.w});
            out.push_back({v2.x, v2.y, v2.z, color.x, color.y, color.z, color.w});
            out.push_back({v3.x, v3.y, v3.z, color.x, color.y, color.z, color.w});

            out.push_back({v1.x, v1.y, v1.z, color.x, color.y, color.z, color.w});
            out.push_back({v3.x, v3.y, v3.z, color.x, color.y, color.z, color.w});
            out.push_back({v4.x, v4.y, v4.z, color.x, color.y, color.z, color.w});
        }
    }

    // Generate a triangle face (both sides)
    void generate_triangle_face(std::vector<RenderVertex>& out,
                               const math::vec3& a, const math::vec3& b, const math::vec3& c,
                               const math::vec4& color) {
        // Front face
        out.push_back({a.x, a.y, a.z, color.x, color.y, color.z, color.w});
        out.push_back({b.x, b.y, b.z, color.x, color.y, color.z, color.w});
        out.push_back({c.x, c.y, c.z, color.x, color.y, color.z, color.w});

        // Back face (reversed winding)
        out.push_back({a.x, a.y, a.z, color.x, color.y, color.z, color.w});
        out.push_back({c.x, c.y, c.z, color.x, color.y, color.z, color.w});
        out.push_back({b.x, b.y, b.z, color.x, color.y, color.z, color.w});
    }

    // ==========================================================================
    // Timing accumulators for Minkowski sum (static for persistence across calls)
    struct MinkowskiTiming {
        double icosphere_ms;
        double sum_points_ms;
        double hull_ms;
        double output_ms;
        int call_count;

        MinkowskiTiming() : icosphere_ms(0), sum_points_ms(0), hull_ms(0), output_ms(0), call_count(0) {}

        void print_and_reset() {
            if (call_count > 0) {
                std::cout << "[Minkowski Breakdown] calls=" << call_count
                          << " icosphere=" << icosphere_ms << "ms"
                          << " sum=" << sum_points_ms << "ms"
                          << " hull=" << hull_ms << "ms"
                          << " output=" << output_ms << "ms" << std::endl;
            }
            icosphere_ms = sum_points_ms = hull_ms = output_ms = 0;
            call_count = 0;
        }
    };

    inline static MinkowskiTiming mink_timing_;

    // Call this after geometry generation to print timing breakdown
    void print_minkowski_timing() {
        mink_timing_.print_and_reset();
    }

    // Static version for access from other classes
    static void print_minkowski_timing_static() {
        mink_timing_.print_and_reset();
    }

    // =========================================================================
    // EFFICIENT MINKOWSKI SUM: Convex Hull ⊕ Sphere
    // =========================================================================
    //
    // The boundary of hull(P) ⊕ sphere(r) consists of three types of patches:
    //
    // 1. FACE PATCHES: For each triangular face with outward normal n,
    //    output the same triangle offset by r*n.
    //
    // 2. EDGE PATCHES: For each edge shared by faces with normals n1, n2,
    //    output a cylindrical strip. The strip sweeps from n1 to n2 around
    //    the edge, covering the angular range where the sphere "rolls" along
    //    the edge.
    //
    // 3. VERTEX PATCHES: For each vertex where faces with normals n1..nk meet,
    //    output a spherical cap. The cap covers directions not covered by any
    //    adjacent face's normal.
    //
    // =========================================================================

    // Helper: emit a triangle with consistent winding (outward from centroid)
    void emit_triangle(std::vector<RenderVertex>& out,
                       const math::vec3& a, const math::vec3& b, const math::vec3& c,
                       const math::vec3& centroid, const math::vec4& color) {
        math::vec3 face_center = (a + b + c) / 3.0f;
        math::vec3 normal = math::cross(b - a, c - a);

        // Ensure outward-facing
        if (math::dot(normal, face_center - centroid) < 0) {
            out.push_back({a.x, a.y, a.z, color.x, color.y, color.z, color.w});
            out.push_back({c.x, c.y, c.z, color.x, color.y, color.z, color.w});
            out.push_back({b.x, b.y, b.z, color.x, color.y, color.z, color.w});
        } else {
            out.push_back({a.x, a.y, a.z, color.x, color.y, color.z, color.w});
            out.push_back({b.x, b.y, b.z, color.x, color.y, color.z, color.w});
            out.push_back({c.x, c.y, c.z, color.x, color.y, color.z, color.w});
        }
    }

    // =========================================================================
    // CYLINDER STRIP: Connects two faces along a shared edge
    // =========================================================================
    // Given an edge from e1 to e2, and two adjacent face normals n1 and n2,
    // we generate a cylindrical strip that sweeps from n1 to n2 around the edge.
    //
    // The sweep goes the "short way" around - the exterior of the hull.
    // For a planar polygon, n1 and n2 are opposite (+N and -N), so we sweep 180°.
    //
    // KEY INSIGHT: The cylinder surface is at distance r from the edge LINE.
    // At each angle θ, the offset direction is:
    //   d(θ) = n1 * cos(θ) + (edge × n1) * sin(θ)
    // where θ goes from 0 to the angle between n1 and n2.
    // =========================================================================
    void emit_cylinder_strip(std::vector<RenderVertex>& out,
                             const math::vec3& e1, const math::vec3& e2,
                             const math::vec3& n1, const math::vec3& n2,
                             float radius, int segments,
                             const math::vec3& centroid, const math::vec4& color) {
        math::vec3 edge_dir = math::normalize(e2 - e1);
        math::vec3 edge_mid = (e1 + e2) * 0.5f;

        // Angle between normals
        float dot_n = math::dot(n1, n2);
        dot_n = std::max(-1.0f, std::min(1.0f, dot_n));
        float small_angle = std::acos(dot_n);  // This is the SMALL angle

        // If normals are nearly identical, no cylinder needed
        if (small_angle < 0.001f) return;

        // The large angle is (2*PI - small_angle).
        // We want to sweep the way that goes OUTWARD (away from centroid).
        //
        // Key insight: The direction (n1 + n2) / 2 points toward where the
        // FACES are (between the two face normals).
        // The OUTWARD direction for the cylinder is OPPOSITE to this:
        // it's where the edge is exposed, away from the faces.
        //
        // So we sweep via the hemisphere OPPOSITE to (n1 + n2).

        math::vec3 face_avg = math::normalize(n1 + n2);
        // The outward direction at the edge is OPPOSITE to where the faces point
        // (the edge cylinder covers the "outside" of the corner)

        // There are two ways to rotate from n1 to n2 around edge_dir:
        // 1. The short way (angle = small_angle)
        // 2. The long way (angle = 2*PI - small_angle)
        //
        // Test which way goes outward: check midpoint of each sweep
        // The correct sweep has its midpoint pointing AWAY from centroid.

        // Compute midpoint direction for SHORT sweep (positive rotation)
        float half_small = small_angle * 0.5f;
        float cos_h = std::cos(half_small);
        float sin_h = std::sin(half_small);
        math::vec3 mid_short_pos = n1 * cos_h
                                 + math::cross(edge_dir, n1) * sin_h
                                 + edge_dir * math::dot(edge_dir, n1) * (1.0f - cos_h);
        mid_short_pos = math::normalize(mid_short_pos);

        // Compute midpoint direction for LONG sweep (the other way)
        // Long sweep goes the other direction: negate the rotation
        math::vec3 mid_long_pos = n1 * cos_h
                                - math::cross(edge_dir, n1) * sin_h
                                + edge_dir * math::dot(edge_dir, n1) * (1.0f - cos_h);
        // But wait, that's the same angle... we need the OPPOSITE semicircle
        // The midpoint of the long way is at angle = PI (opposite to short midpoint)
        float half_long = std::numbers::pi_v<float>;  // midpoint of long arc is at 180 degrees
        float cos_l = std::cos(half_long);
        float sin_l = std::sin(half_long);
        math::vec3 mid_long = n1 * cos_l
                            + math::cross(edge_dir, n1) * sin_l
                            + edge_dir * math::dot(edge_dir, n1) * (1.0f - cos_l);
        mid_long = math::normalize(mid_long);

        // Test which midpoint is more outward (farther from centroid when projected from edge)
        // Actually, we want the direction that points AWAY from centroid
        math::vec3 to_centroid = centroid - edge_mid;
        float short_dot = math::dot(mid_short_pos, to_centroid);
        float long_dot = math::dot(mid_long, to_centroid);

        // We want the sweep whose midpoint has NEGATIVE dot with to_centroid
        // (i.e., points away from centroid)
        bool use_short_way = (short_dot < long_dot);

        // If using short way: sweep angle goes from 0 to small_angle
        // If using long way: sweep angle goes from 0 to (2*PI - small_angle)
        // Direction of rotation: determined by cross(n1, n2) vs edge_dir
        math::vec3 cross_n = math::cross(n1, n2);
        float cross_dot_edge = math::dot(cross_n, edge_dir);
        float base_sign = (cross_dot_edge >= 0) ? 1.0f : -1.0f;

        float sweep_angle;
        float sign;
        if (use_short_way) {
            sweep_angle = small_angle;
            sign = base_sign;
        } else {
            sweep_angle = 2.0f * std::numbers::pi_v<float> - small_angle;
            sign = -base_sign;  // Go the other way
        }

        // Generate the strip by rotating n1 around edge_dir
        math::vec3 prev_dir = n1;
        math::vec3 prev1 = e1 + prev_dir * radius;
        math::vec3 prev2 = e2 + prev_dir * radius;

        for (int i = 1; i <= segments; ++i) {
            float t = static_cast<float>(i) / segments;
            float angle = t * sweep_angle * sign;

            // Rodrigues rotation of n1 around edge_dir by angle
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            math::vec3 curr_dir = n1 * cos_a
                                + math::cross(edge_dir, n1) * sin_a
                                + edge_dir * math::dot(edge_dir, n1) * (1.0f - cos_a);
            curr_dir = math::normalize(curr_dir);

            math::vec3 curr1 = e1 + curr_dir * radius;
            math::vec3 curr2 = e2 + curr_dir * radius;

            // Emit quad as two triangles
            emit_triangle(out, prev1, prev2, curr1, centroid, color);
            emit_triangle(out, curr1, prev2, curr2, centroid, color);

            prev_dir = curr_dir;
            prev1 = curr1;
            prev2 = curr2;
        }
    }

    // =========================================================================
    // SPHERICAL CAP: Fills the gap at a vertex where multiple edges meet
    // =========================================================================
    // The spherical cap covers the solid angle at the vertex not covered by
    // any face. For volumetric hulls, this is the spherical polygon bounded
    // by the face normals. For planar polygons, this is a semicircle.
    //
    // GEOMETRY: At vertex V with adjacent face normals {n1, n2, ...}, the cap
    // is the region of the unit sphere "outside" all the faces - the directions
    // from which V is visible from outside the hull.
    // =========================================================================
    void emit_spherical_cap(std::vector<RenderVertex>& out,
                            const math::vec3& vertex,
                            const std::vector<math::vec3>& normals,
                            const math::vec3& edge_dir_hint, // direction along one adjacent edge
                            float radius, int segments,
                            const math::vec3& centroid, const math::vec4& color) {
        if (normals.size() < 2) return;

        // Compute the "outward" direction - the center of the spherical cap
        // For volumetric: average of normals (they all point outward)
        // For planar: normals are opposite, use edge direction as outward
        math::vec3 avg_normal(0, 0, 0);
        for (const auto& n : normals) avg_normal = avg_normal + n;
        float avg_len = math::length(avg_normal);

        math::vec3 outward_dir;
        bool is_planar = (avg_len < 0.001f);

        if (is_planar) {
            // Planar case: normals cancel out (+N and -N)
            // The outward direction is along the edge (pointing away from centroid)
            math::vec3 v_to_centroid = centroid - vertex;
            // Project out the face-normal component to get edge direction
            math::vec3 face_normal = normals[0];  // Either +N or -N
            math::vec3 edge_component = edge_dir_hint - face_normal * math::dot(edge_dir_hint, face_normal);
            if (math::length(edge_component) > 0.001f) {
                outward_dir = math::normalize(edge_component);
            } else {
                outward_dir = math::normalize(edge_dir_hint);
            }
            // Make sure outward points away from centroid
            if (math::dot(outward_dir, v_to_centroid) > 0) {
                outward_dir = outward_dir * -1.0f;
            }
        } else {
            // Volumetric case: outward is the average of normals
            outward_dir = math::normalize(avg_normal);
        }

        // For planar case with exactly 2 opposite normals:
        // The cap is a semicircle from +N to -N going through outward_dir
        if (is_planar && normals.size() == 2) {
            math::vec3 n1 = normals[0];
            math::vec3 n2 = normals[1];

            // The semicircle goes from n1 through outward_dir to n2
            // This is a 180-degree arc with outward_dir at the middle
            // We rotate n1 around the axis perpendicular to the plane containing n1, n2, outward

            // The rotation axis is face_normal (perpendicular to the edge plane)
            math::vec3 axis = math::cross(n1, outward_dir);
            if (math::length(axis) < 0.001f) {
                // n1 is parallel to outward_dir, use n2 to find axis
                axis = math::cross(n2, outward_dir);
            }
            axis = math::normalize(axis);

            // Generate the semicircle fan from outward_dir
            math::vec3 cap_center = vertex + outward_dir * radius;

            // Generate fan: cap_center -> arc from n1 to n2 via outward_dir
            // First half: n1 to outward_dir (90 degrees)
            // Second half: outward_dir to n2 (90 degrees)
            int half_segs = segments;
            math::vec3 prev_on_sphere = vertex + n1 * radius;

            for (int s = 1; s <= 2 * half_segs; ++s) {
                float t = static_cast<float>(s) / (2 * half_segs);
                float angle = t * std::numbers::pi_v<float>;  // 0 to PI

                // Rotate n1 around axis by angle (Rodrigues)
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                math::vec3 curr_dir = n1 * cos_a
                                    + math::cross(axis, n1) * sin_a
                                    + axis * math::dot(axis, n1) * (1.0f - cos_a);
                curr_dir = math::normalize(curr_dir);

                math::vec3 curr_on_sphere = vertex + curr_dir * radius;

                // Use VERTEX as winding reference (sphere center), NOT polygon centroid
                emit_triangle(out, cap_center, prev_on_sphere, curr_on_sphere, vertex, color);

                prev_on_sphere = curr_on_sphere;
            }
            return;
        }

        // Volumetric case: sort normals by angle around outward direction
        math::vec3 ref = normals[0];
        math::vec3 tangent = math::cross(outward_dir, ref);
        if (math::length(tangent) < 0.001f) {
            ref = (std::abs(outward_dir.x) < 0.9f) ? math::vec3(1,0,0) : math::vec3(0,1,0);
            tangent = math::cross(outward_dir, ref);
        }
        tangent = math::normalize(tangent);
        math::vec3 bitangent = math::cross(outward_dir, tangent);

        std::vector<std::pair<float, math::vec3>> sorted_normals;
        for (const auto& n : normals) {
            float angle = std::atan2(math::dot(n, bitangent), math::dot(n, tangent));
            sorted_normals.push_back({angle, n});
        }
        std::sort(sorted_normals.begin(), sorted_normals.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        // The cap center on the sphere
        math::vec3 cap_center = vertex + outward_dir * radius;

        // For each pair of adjacent normals, create a spherical triangle fan
        // from the cap_center to the arc between normals
        for (size_t i = 0; i < sorted_normals.size(); ++i) {
            size_t j = (i + 1) % sorted_normals.size();
            math::vec3 n1 = sorted_normals[i].second;
            math::vec3 n2 = sorted_normals[j].second;

            // Arc from n1 to n2 via slerp
            math::vec3 prev_on_sphere = vertex + n1 * radius;

            for (int s = 1; s <= segments; ++s) {
                float t = static_cast<float>(s) / segments;

                // Slerp from n1 to n2
                float dot = math::dot(n1, n2);
                dot = std::max(-1.0f, std::min(1.0f, dot));
                float omega = std::acos(dot);

                math::vec3 curr_dir;
                if (omega < 0.001f) {
                    curr_dir = n1;
                } else {
                    float sin_omega = std::sin(omega);
                    curr_dir = n1 * (std::sin((1-t)*omega)/sin_omega)
                             + n2 * (std::sin(t*omega)/sin_omega);
                    curr_dir = math::normalize(curr_dir);
                }

                math::vec3 curr_on_sphere = vertex + curr_dir * radius;

                // Use VERTEX as winding reference (sphere center), NOT polygon centroid
                emit_triangle(out, cap_center, prev_on_sphere, curr_on_sphere, vertex, color);

                prev_on_sphere = curr_on_sphere;
            }
        }
    }

    // Fast Minkowski sum using OFFSET SURFACE approach
    // Instead of computing hull of (n × 162) points, we:
    // 1. Compute hull of original n points (fast for small n)
    // 2. Offset each face outward by radius
    // 3. Add cylinder strips for edges
    // 4. Add spherical caps for vertices
    // This is O(n) instead of O((n × sphere_verts)²)
    void generate_minkowski_sum_fast(std::vector<RenderVertex>& out,
                                     const std::vector<math::vec3>& points,
                                     float radius, const math::vec4& color) {
        if (points.size() < 2) return;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Compute centroid for winding orientation
        math::vec3 centroid(0, 0, 0);
        for (const auto& p : points) centroid = centroid + p;
        centroid = centroid * (1.0f / points.size());

        const int SEGMENTS = 8;  // Arc segments for cylinders and caps

        // Special case: 2 points = capsule
        if (points.size() == 2) {
            generate_capsule_fast(out, points[0], points[1], radius, SEGMENTS, centroid, color);
            auto t1 = std::chrono::high_resolution_clock::now();
            mink_timing_.hull_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            mink_timing_.call_count++;
            return;
        }

        // Check if points are coplanar
        math::vec3 plane_normal;
        bool is_coplanar = check_coplanar(points, plane_normal);

        auto t1 = std::chrono::high_resolution_clock::now();

        if (is_coplanar) {
            // 2D case: rounded polygon
            generate_minkowski_2d(out, points, plane_normal, radius, SEGMENTS, centroid, color);
        } else {
            // 3D case: offset polyhedron
            generate_minkowski_3d(out, points, radius, SEGMENTS, centroid, color);
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        mink_timing_.hull_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        mink_timing_.output_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        mink_timing_.call_count++;
    }

    // Check if points are coplanar (within tolerance)
    bool check_coplanar(const std::vector<math::vec3>& points, math::vec3& plane_normal) {
        if (points.size() < 3) return true;

        // Compute plane from first 3 points
        math::vec3 v1 = points[1] - points[0];
        math::vec3 v2 = points[2] - points[0];
        plane_normal = math::cross(v1, v2);
        float len = math::length(plane_normal);
        if (len < 1e-6f) {
            // First 3 points are collinear, try to find non-collinear point
            for (size_t i = 3; i < points.size(); ++i) {
                v2 = points[i] - points[0];
                plane_normal = math::cross(v1, v2);
                len = math::length(plane_normal);
                if (len >= 1e-6f) break;
            }
        }
        if (len < 1e-6f) return true;  // All collinear
        plane_normal = plane_normal / len;

        // Check all points against plane
        for (size_t i = 3; i < points.size(); ++i) {
            float dist = std::abs(math::dot(plane_normal, points[i] - points[0]));
            if (dist > 1e-4f) return false;  // Not coplanar
        }
        return true;
    }

    // Generate capsule (2 hemispheres + cylinder)
    void generate_capsule_fast(std::vector<RenderVertex>& out,
                               const math::vec3& p1, const math::vec3& p2,
                               float radius, int segments,
                               const math::vec3& centroid, const math::vec4& color) {
        math::vec3 axis = p2 - p1;
        float length = math::length(axis);
        if (length < 1e-6f) {
            // Degenerate: just a sphere
            generate_sphere_triangles(out, p1, radius, segments, color);
            return;
        }
        axis = axis / length;

        // Find perpendicular vectors
        math::vec3 perp1 = (std::abs(axis.x) < 0.9f) ? math::vec3(1,0,0) : math::vec3(0,1,0);
        perp1 = math::normalize(math::cross(axis, perp1));
        math::vec3 perp2 = math::cross(axis, perp1);

        // Cylinder body
        for (int i = 0; i < segments; ++i) {
            float a1 = math::TAU * i / segments;
            float a2 = math::TAU * (i + 1) / segments;

            math::vec3 d1 = perp1 * std::cos(a1) + perp2 * std::sin(a1);
            math::vec3 d2 = perp1 * std::cos(a2) + perp2 * std::sin(a2);

            math::vec3 v1 = p1 + d1 * radius;
            math::vec3 v2 = p1 + d2 * radius;
            math::vec3 v3 = p2 + d1 * radius;
            math::vec3 v4 = p2 + d2 * radius;

            emit_triangle(out, v1, v3, v2, centroid, color);
            emit_triangle(out, v2, v3, v4, centroid, color);
        }

        // Hemisphere at p1 (pointing away from p2)
        generate_hemisphere(out, p1, axis * -1.0f, perp1, perp2, radius, segments, centroid, color);

        // Hemisphere at p2 (pointing away from p1)
        generate_hemisphere(out, p2, axis, perp1, perp2, radius, segments, centroid, color);
    }

    // Generate hemisphere
    void generate_hemisphere(std::vector<RenderVertex>& out,
                             const math::vec3& center, const math::vec3& pole,
                             const math::vec3& perp1, const math::vec3& perp2,
                             float radius, int segments,
                             const math::vec3& centroid, const math::vec4& color) {
        int rings = segments / 2;
        for (int ring = 0; ring < rings; ++ring) {
            float theta1 = math::PI * 0.5f * ring / rings;
            float theta2 = math::PI * 0.5f * (ring + 1) / rings;

            float r1 = std::cos(theta1);
            float z1 = std::sin(theta1);
            float r2 = std::cos(theta2);
            float z2 = std::sin(theta2);

            for (int seg = 0; seg < segments; ++seg) {
                float phi1 = math::TAU * seg / segments;
                float phi2 = math::TAU * (seg + 1) / segments;

                auto make_point = [&](float r, float z, float phi) {
                    math::vec3 dir = perp1 * (r * std::cos(phi)) + perp2 * (r * std::sin(phi)) + pole * z;
                    return center + dir * radius;
                };

                math::vec3 v1 = make_point(r1, z1, phi1);
                math::vec3 v2 = make_point(r1, z1, phi2);
                math::vec3 v3 = make_point(r2, z2, phi1);
                math::vec3 v4 = make_point(r2, z2, phi2);

                // Use CENTER as winding reference (sphere center), NOT polygon centroid
                emit_triangle(out, v1, v3, v2, center, color);
                emit_triangle(out, v2, v3, v4, center, color);
            }
        }
    }

    // Generate sphere triangles (for degenerate capsule)
    void generate_sphere_triangles(std::vector<RenderVertex>& out,
                                   const math::vec3& center, float radius,
                                   int segments, const math::vec4& color) {
        for (int ring = 0; ring < segments; ++ring) {
            float theta1 = math::PI * ring / segments;
            float theta2 = math::PI * (ring + 1) / segments;

            for (int seg = 0; seg < segments; ++seg) {
                float phi1 = math::TAU * seg / segments;
                float phi2 = math::TAU * (seg + 1) / segments;

                auto make_point = [&](float theta, float phi) {
                    float st = std::sin(theta), ct = std::cos(theta);
                    float sp = std::sin(phi), cp = std::cos(phi);
                    return center + math::vec3(st * cp, ct, st * sp) * radius;
                };

                math::vec3 v1 = make_point(theta1, phi1);
                math::vec3 v2 = make_point(theta1, phi2);
                math::vec3 v3 = make_point(theta2, phi1);
                math::vec3 v4 = make_point(theta2, phi2);

                out.push_back({v1.x, v1.y, v1.z, color.x, color.y, color.z, color.w});
                out.push_back({v3.x, v3.y, v3.z, color.x, color.y, color.z, color.w});
                out.push_back({v2.x, v2.y, v2.z, color.x, color.y, color.z, color.w});

                out.push_back({v2.x, v2.y, v2.z, color.x, color.y, color.z, color.w});
                out.push_back({v3.x, v3.y, v3.z, color.x, color.y, color.z, color.w});
                out.push_back({v4.x, v4.y, v4.z, color.x, color.y, color.z, color.w});
            }
        }
    }

    // 2D Minkowski sum: rounded polygon (coplanar points)
    void generate_minkowski_2d(std::vector<RenderVertex>& out,
                               const std::vector<math::vec3>& points,
                               const math::vec3& plane_normal_in,
                               float radius, int segments,
                               const math::vec3& centroid, const math::vec4& color) {
        [[maybe_unused]] size_t start_size = out.size();

        // Compute 2D convex hull of points projected onto plane
        std::vector<size_t> hull_indices = convex_hull_2d(points, plane_normal_in);
        if (hull_indices.size() < 3) return;

        size_t n = hull_indices.size();

        // Ensure hull is CCW when viewed from plane_normal
        // Compute signed area to check winding
        math::vec3 n_up = plane_normal_in;

        // Project hull to 2D and compute signed area
        math::vec3 u = (std::abs(n_up.x) < 0.9f) ? math::vec3(1,0,0) : math::vec3(0,1,0);
        u = math::normalize(math::cross(n_up, u));
        math::vec3 v = math::cross(n_up, u);

        float signed_area = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            float x1 = math::dot(points[hull_indices[i]], u);
            float y1 = math::dot(points[hull_indices[i]], v);
            float x2 = math::dot(points[hull_indices[j]], u);
            float y2 = math::dot(points[hull_indices[j]], v);
            signed_area += (x2 - x1) * (y2 + y1);
        }

        // If signed_area > 0, hull is CW → flip normal to make it CCW
        if (signed_area > 0) {
            n_up = n_up * -1.0f;
        }

        math::vec3 n_down = n_up * -1.0f;

        // 1. Top face (offset by +radius along normal)
        for (size_t i = 1; i + 1 < n; ++i) {
            math::vec3 a = points[hull_indices[0]] + n_up * radius;
            math::vec3 b = points[hull_indices[i]] + n_up * radius;
            math::vec3 c = points[hull_indices[i + 1]] + n_up * radius;
            emit_triangle(out, a, b, c, centroid, color);
        }

        // 2. Bottom face (offset by -radius along normal)
        for (size_t i = 1; i + 1 < n; ++i) {
            math::vec3 a = points[hull_indices[0]] + n_down * radius;
            math::vec3 b = points[hull_indices[i + 1]] + n_down * radius;
            math::vec3 c = points[hull_indices[i]] + n_down * radius;
            emit_triangle(out, a, b, c, centroid, color);
        }

        // 3. Edge cylinders (half-cylinders, 180 degrees from +normal to -normal)
        // For 2D polygons, we compute the outward direction directly instead of
        // relying on emit_cylinder_strip's complex short/long way logic
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            math::vec3 e1 = points[hull_indices[i]];
            math::vec3 e2 = points[hull_indices[j]];

            // Edge direction and outward perpendicular (in-plane, away from centroid)
            math::vec3 edge_dir = math::normalize(e2 - e1);
            math::vec3 edge_mid = (e1 + e2) * 0.5f;

            // Outward direction: perpendicular to edge, in the plane
            // For CCW polygon viewed from +n_up, right-hand perp is cross(edge_dir, n_up)
            math::vec3 outward = math::cross(edge_dir, n_up);

            // Make sure it points away from centroid
            if (math::dot(outward, edge_mid - centroid) < 0) {
                outward = outward * -1.0f;
            }

            // Generate half-cylinder: sweep from n_up through outward to n_down
            math::vec3 prev_dir = n_up;
            math::vec3 prev1 = e1 + prev_dir * radius;
            math::vec3 prev2 = e2 + prev_dir * radius;

            for (int s = 1; s <= segments; ++s) {
                float t = static_cast<float>(s) / segments;
                float angle = t * math::PI;  // 0 to PI

                // Interpolate: n_up -> outward -> n_down
                // At t=0: n_up, at t=0.5: outward, at t=1: n_down
                math::vec3 curr_dir = n_up * std::cos(angle) + outward * std::sin(angle);
                curr_dir = math::normalize(curr_dir);

                math::vec3 curr1 = e1 + curr_dir * radius;
                math::vec3 curr2 = e2 + curr_dir * radius;

                emit_triangle(out, prev1, prev2, curr1, centroid, color);
                emit_triangle(out, curr1, prev2, curr2, centroid, color);

                prev_dir = curr_dir;
                prev1 = curr1;
                prev2 = curr2;
            }
        }

        // 4. Vertex caps (spherical wedges at corners)
        // Each cap fills the gap between two adjacent edge cylinders
        for (size_t i = 0; i < n; ++i) {
            size_t prev_idx = (i + n - 1) % n;
            size_t next_idx = (i + 1) % n;

            math::vec3 v = points[hull_indices[i]];
            math::vec3 v_prev = points[hull_indices[prev_idx]];
            math::vec3 v_next = points[hull_indices[next_idx]];

            // Edge directions (along the edges, not toward vertices)
            math::vec3 edge_in = math::normalize(v - v_prev);   // Edge coming IN to v
            math::vec3 edge_out = math::normalize(v_next - v);  // Edge going OUT from v

            // Perpendiculars in-plane, pointing outward from polygon
            // For CCW polygon viewed from +n_up: outward is cross(edge_dir, n_up)
            math::vec3 perp_in = math::cross(edge_in, n_up);
            math::vec3 perp_out = math::cross(edge_out, n_up);

            // Normalize (should already be unit, but be safe)
            perp_in = math::normalize(perp_in);
            perp_out = math::normalize(perp_out);

            // Ensure both point outward (away from centroid) - this is the ground truth
            if (math::dot(perp_in, v - centroid) < 0) perp_in = perp_in * -1.0f;
            if (math::dot(perp_out, v - centroid) < 0) perp_out = perp_out * -1.0f;

            emit_vertex_cap_2d(out, v, n_up, perp_in, perp_out, radius, segments, centroid, color);
        }

        // Debug assertions
        assert(out.size() > start_size && "generate_minkowski_2d produced no geometry");
        #ifndef NDEBUG
        for (size_t i = start_size; i < out.size(); ++i) {
            assert(!std::isnan(out[i].x) && !std::isnan(out[i].y) && !std::isnan(out[i].z) &&
                   "NaN detected in generated geometry");
        }
        #endif
    }

    // Emit spherical wedge for 2D polygon vertex
    // Spans from perp_start to perp_end (in-plane) × from +normal to -normal (out-of-plane)
    // This covers the exterior of the convex polygon vertex.
    void emit_vertex_cap_2d(std::vector<RenderVertex>& out,
                            const math::vec3& vertex,
                            const math::vec3& plane_normal,
                            const math::vec3& perp_start,
                            const math::vec3& perp_end,
                            float radius, int segments,
                            [[maybe_unused]] const math::vec3& centroid, const math::vec4& color) {
        // For a convex polygon, the exterior angle at a vertex is always < 180°
        // We use SLERP to interpolate between perp_start and perp_end

        // Clamp dot product to avoid acos domain errors
        float dot_p = math::dot(perp_start, perp_end);
        dot_p = std::max(-0.9999f, std::min(0.9999f, dot_p));

        // Angle between the perpendiculars
        float angle_between = std::acos(dot_p);

        // Determine rotation direction: we want to go the EXTERIOR way
        // For CCW polygon viewed from +normal, exterior is CCW from perp_prev to perp_next
        math::vec3 cross_p = math::cross(perp_start, perp_end);
        float cross_sign = math::dot(cross_p, plane_normal);

        // If cross_sign > 0, CCW from start to end is the short way (< 180°)
        // If cross_sign < 0, CCW from start to end is the long way (> 180°)
        // For convex polygon exterior, we always want the short way
        bool go_ccw = (cross_sign > 0);

        // If the "short way" going CCW is > 180°, something is wrong with inputs
        // Just use the direct angle
        float angle_span = angle_between;
        if (!go_ccw) {
            // Need to go CW, so negative angles
            angle_span = -angle_between;
        }

        // Segments for horizontal (in-plane) sweep
        int h_segs = std::max(2, (int)(segments * std::abs(angle_span) / math::PI) + 1);
        int v_segs = segments;

        // Generate the spherical patch using explicit slerp
        for (int h = 0; h < h_segs; ++h) {
            float t1 = (float)h / h_segs;
            float t2 = (float)(h + 1) / h_segs;

            // Slerp between perp_start and perp_end
            auto slerp_perp = [&](float t) -> math::vec3 {
                if (std::abs(angle_between) < 0.001f) {
                    return perp_start;  // Nearly parallel, just use start
                }
                float omega = angle_between;
                float sin_omega = std::sin(omega);
                float a = std::sin((1.0f - t) * omega) / sin_omega;
                float b = std::sin(t * omega) / sin_omega;
                math::vec3 result = perp_start * a + perp_end * b;
                float len = math::length(result);
                return (len > 0.001f) ? result / len : perp_start;
            };

            // If going CW, flip the interpolation direction
            math::vec3 h_dir1 = go_ccw ? slerp_perp(t1) : slerp_perp(1.0f - t1);
            math::vec3 h_dir2 = go_ccw ? slerp_perp(t2) : slerp_perp(1.0f - t2);

            // If going CW, we started from perp_end, so swap
            if (!go_ccw) {
                std::swap(h_dir1, h_dir2);
            }

            for (int v = 0; v < v_segs; ++v) {
                float p1 = (float)v / v_segs;
                float p2 = (float)(v + 1) / v_segs;

                // Vertical angle: 0 = +normal, PI = -normal
                auto make_dir = [&](const math::vec3& h_dir, float p) {
                    float angle = p * math::PI;
                    return plane_normal * std::cos(angle) + h_dir * std::sin(angle);
                };

                math::vec3 d11 = make_dir(h_dir1, p1);
                math::vec3 d12 = make_dir(h_dir1, p2);
                math::vec3 d21 = make_dir(h_dir2, p1);
                math::vec3 d22 = make_dir(h_dir2, p2);

                math::vec3 p11 = vertex + d11 * radius;
                math::vec3 p12 = vertex + d12 * radius;
                math::vec3 p21 = vertex + d21 * radius;
                math::vec3 p22 = vertex + d22 * radius;

                // Use VERTEX as winding reference (sphere center)
                emit_triangle(out, p11, p21, p12, vertex, color);
                emit_triangle(out, p12, p21, p22, vertex, color);
            }
        }
    }

    // Simple 2D convex hull using gift wrapping (for small point sets)
    std::vector<size_t> convex_hull_2d(const std::vector<math::vec3>& points,
                                        const math::vec3& plane_normal) {
        size_t n = points.size();
        if (n < 3) {
            std::vector<size_t> result;
            for (size_t i = 0; i < n; ++i) result.push_back(i);
            return result;
        }

        // Find leftmost point (using arbitrary 2D projection)
        math::vec3 u = (std::abs(plane_normal.x) < 0.9f) ? math::vec3(1,0,0) : math::vec3(0,1,0);
        u = math::normalize(math::cross(plane_normal, u));
        math::vec3 v = math::cross(plane_normal, u);

        auto project = [&](const math::vec3& p) -> std::pair<float, float> {
            return {math::dot(p, u), math::dot(p, v)};
        };

        size_t start = 0;
        auto start_2d = project(points[0]);
        for (size_t i = 1; i < n; ++i) {
            auto p = project(points[i]);
            if (p.first < start_2d.first ||
                (p.first == start_2d.first && p.second < start_2d.second)) {
                start = i;
                start_2d = p;
            }
        }

        std::vector<size_t> hull;
        size_t current = start;
        size_t iterations = 0;
        const size_t max_iterations = n + 2;

        do {
            hull.push_back(current);
            size_t next = (current + 1) % n;

            for (size_t i = 0; i < n; ++i) {
                if (i == current) continue;
                auto curr_2d = project(points[current]);
                auto next_2d = project(points[next]);
                auto test_2d = project(points[i]);

                // Cross product to determine turn direction
                float cross = (next_2d.first - curr_2d.first) * (test_2d.second - curr_2d.second)
                            - (next_2d.second - curr_2d.second) * (test_2d.first - curr_2d.first);

                if (cross < 0 || (cross == 0 &&
                    // Collinear: take farther point
                    ((test_2d.first - curr_2d.first) * (test_2d.first - curr_2d.first) +
                     (test_2d.second - curr_2d.second) * (test_2d.second - curr_2d.second)) >
                    ((next_2d.first - curr_2d.first) * (next_2d.first - curr_2d.first) +
                     (next_2d.second - curr_2d.second) * (next_2d.second - curr_2d.second)))) {
                    next = i;
                }
            }

            current = next;
            iterations++;
        } while (current != start && hull.size() < n && iterations < max_iterations);

        // Debug assertions
        assert(iterations < max_iterations && "convex_hull_2d hit iteration limit");
        assert(!(hull.size() < 3 && n >= 3) && "convex_hull_2d produced degenerate hull");

        return hull;
    }

    // 3D Minkowski sum: offset polyhedron (non-coplanar points)
    void generate_minkowski_3d(std::vector<RenderVertex>& out,
                               const std::vector<math::vec3>& points,
                               float radius, int segments,
                               const math::vec3& centroid, const math::vec4& color) {
        // Compute 3D convex hull of original points
        auto hull_faces = compute_convex_hull_fast(points);
        if (hull_faces.empty()) return;

        // Build edge and vertex adjacency info
        std::map<std::pair<size_t, size_t>, std::vector<math::vec3>> edge_normals;
        std::map<size_t, std::vector<math::vec3>> vertex_normals;
        std::map<size_t, std::vector<size_t>> vertex_neighbors;

        for (const auto& face : hull_faces) {
            math::vec3 a = points[face[0]];
            math::vec3 b = points[face[1]];
            math::vec3 c = points[face[2]];

            math::vec3 normal = math::normalize(math::cross(b - a, c - a));

            // Make sure normal points outward
            math::vec3 face_center = (a + b + c) / 3.0f;
            if (math::dot(normal, face_center - centroid) < 0) {
                normal = normal * -1.0f;
            }

            // 1. Emit offset face
            math::vec3 offset = normal * radius;
            emit_triangle(out, a + offset, b + offset, c + offset, centroid, color);

            // Track edges and vertices
            for (int e = 0; e < 3; ++e) {
                size_t v1 = face[e];
                size_t v2 = face[(e + 1) % 3];
                auto edge = std::make_pair(std::min(v1, v2), std::max(v1, v2));
                edge_normals[edge].push_back(normal);

                vertex_normals[face[e]].push_back(normal);
                vertex_neighbors[face[e]].push_back(face[(e + 1) % 3]);
            }
        }

        // 2. Emit cylinder strips for edges
        for (const auto& [edge, normals] : edge_normals) {
            if (normals.size() != 2) continue;  // Internal edge (shouldn't happen for convex hull)

            math::vec3 e1 = points[edge.first];
            math::vec3 e2 = points[edge.second];
            emit_cylinder_strip(out, e1, e2, normals[0], normals[1], radius, segments, centroid, color);
        }

        // 3. Emit spherical caps for vertices
        for (const auto& [vi, normals] : vertex_normals) {
            if (normals.size() < 2) continue;

            math::vec3 vertex = points[vi];

            // Edge direction hint: direction to first neighbor
            math::vec3 edge_hint(1, 0, 0);
            if (!vertex_neighbors[vi].empty()) {
                edge_hint = math::normalize(points[vertex_neighbors[vi][0]] - vertex);
            }

            emit_spherical_cap(out, vertex, normals, edge_hint, radius, segments, centroid, color);
        }
    }

    // ==========================================================================
    // 3D QUICKHULL - O(n log n) expected time convex hull algorithm
    // ==========================================================================
    std::vector<std::array<size_t, 3>> compute_convex_hull_fast(const std::vector<math::vec3>& points) {
        const float EPS = 1e-6f;
        const size_t n = points.size();
        if (n < 4) return {};

        // Find extreme points along each axis
        size_t minX = 0, maxX = 0, minY = 0, maxY = 0, minZ = 0, maxZ = 0;
        for (size_t i = 1; i < n; ++i) {
            if (points[i].x < points[minX].x) minX = i;
            if (points[i].x > points[maxX].x) maxX = i;
            if (points[i].y < points[minY].y) minY = i;
            if (points[i].y > points[maxY].y) maxY = i;
            if (points[i].z < points[minZ].z) minZ = i;
            if (points[i].z > points[maxZ].z) maxZ = i;
        }

        // Find the most distant pair of extreme points
        size_t extremes[6] = {minX, maxX, minY, maxY, minZ, maxZ};
        size_t p0 = 0, p1 = 1;
        float maxDistSq = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i + 1; j < 6; ++j) {
                math::vec3 diff = points[extremes[j]] - points[extremes[i]];
                float distSq = math::dot(diff, diff);
                if (distSq > maxDistSq) {
                    maxDistSq = distSq;
                    p0 = extremes[i];
                    p1 = extremes[j];
                }
            }
        }
        if (maxDistSq < EPS * EPS) return {};

        // Find point farthest from line p0-p1
        math::vec3 lineDir = points[p1] - points[p0];
        float lineLenSq = math::dot(lineDir, lineDir);
        size_t p2 = p0;
        float maxDist = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i == p0 || i == p1) continue;
            math::vec3 toPoint = points[i] - points[p0];
            float t = math::dot(toPoint, lineDir) / lineLenSq;
            math::vec3 closest = points[p0] + lineDir * t;
            math::vec3 diff = points[i] - closest;
            float distSq = math::dot(diff, diff);
            if (distSq > maxDist) { maxDist = distSq; p2 = i; }
        }
        if (maxDist < EPS * EPS) return {};

        // Find point farthest from plane p0-p1-p2
        math::vec3 planeNormal = math::cross(points[p1] - points[p0], points[p2] - points[p0]);
        float planeNormalLen = math::length(planeNormal);
        if (planeNormalLen < EPS) return {};
        planeNormal = planeNormal / planeNormalLen;

        size_t p3 = p0;
        maxDist = 0;
        float p3Sign = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i == p0 || i == p1 || i == p2) continue;
            float dist = math::dot(planeNormal, points[i] - points[p0]);
            if (std::abs(dist) > maxDist) { maxDist = std::abs(dist); p3 = i; p3Sign = dist; }
        }
        if (maxDist < EPS) return {};
        if (p3Sign < 0) std::swap(p1, p2);

        // Face structure with adjacency info for BFS visibility search
        struct Face {
            size_t v[3];
            math::vec3 normal;
            float dist;
            std::vector<size_t> outsideSet;
            size_t neighbors[3];  // Adjacent face indices for each edge
            bool active;
        };

        std::vector<Face> faces;
        faces.reserve(n * 2);

        // Edge to face mapping for building adjacency
        std::unordered_map<uint64_t, std::pair<size_t, int>> edgeToFace;

        auto edgeKey = [](size_t a, size_t b) -> uint64_t {
            if (a > b) std::swap(a, b);
            return (uint64_t(a) << 32) | uint64_t(b);
        };

        math::vec3 interior = (points[p0] + points[p1] + points[p2] + points[p3]) * 0.25f;

        auto createFace = [&](size_t a, size_t b, size_t c) -> size_t {
            Face f;
            f.v[0] = a; f.v[1] = b; f.v[2] = c;
            f.neighbors[0] = f.neighbors[1] = f.neighbors[2] = SIZE_MAX;
            f.active = true;

            math::vec3 ab = points[b] - points[a];
            math::vec3 ac = points[c] - points[a];
            f.normal = math::cross(ab, ac);
            float len = math::length(f.normal);
            if (len > EPS) f.normal = f.normal / len;
            else f.normal = {0, 0, 1};
            f.dist = math::dot(f.normal, points[a]);

            if (math::dot(f.normal, interior) - f.dist > EPS) {
                std::swap(f.v[1], f.v[2]);
                f.normal = f.normal * -1.0f;
                f.dist = -f.dist;
            }

            size_t idx = faces.size();
            faces.push_back(std::move(f));

            // Register edges and link to neighbors
            Face& newFace = faces[idx];
            for (int e = 0; e < 3; ++e) {
                size_t ea = newFace.v[e], eb = newFace.v[(e + 1) % 3];
                uint64_t key = edgeKey(ea, eb);
                auto it = edgeToFace.find(key);
                if (it != edgeToFace.end()) {
                    size_t otherIdx = it->second.first;
                    int otherEdge = it->second.second;
                    newFace.neighbors[e] = otherIdx;
                    faces[otherIdx].neighbors[otherEdge] = idx;
                    edgeToFace.erase(it);
                } else {
                    edgeToFace[key] = {idx, e};
                }
            }

            return idx;
        };

        createFace(p0, p1, p2);
        createFace(p0, p2, p3);
        createFace(p0, p3, p1);
        createFace(p1, p3, p2);

        // Assign points to faces (only need to scan initial 4 faces)
        std::vector<bool> assigned(n, false);
        assigned[p0] = assigned[p1] = assigned[p2] = assigned[p3] = true;

        for (size_t i = 0; i < n; ++i) {
            if (assigned[i]) continue;
            float bestDist = EPS;
            size_t bestFace = SIZE_MAX;
            for (size_t fi = 0; fi < 4; ++fi) {
                float d = math::dot(faces[fi].normal, points[i]) - faces[fi].dist;
                if (d > bestDist) { bestDist = d; bestFace = fi; }
            }
            if (bestFace != SIZE_MAX) {
                faces[bestFace].outsideSet.push_back(i);
                assigned[i] = true;
            }
        }

        // Process faces with outside points
        std::vector<size_t> faceStack;
        for (size_t fi = 0; fi < 4; ++fi) {
            if (!faces[fi].outsideSet.empty()) faceStack.push_back(fi);
        }

        std::vector<size_t> visibleFaces;
        std::vector<size_t> newFaceIndices;
        std::vector<bool> visited;
        visited.reserve(n * 2);

        while (!faceStack.empty()) {
            size_t faceIdx = faceStack.back();
            faceStack.pop_back();

            if (!faces[faceIdx].active || faces[faceIdx].outsideSet.empty()) continue;

            Face& currentFace = faces[faceIdx];

            // Find farthest point
            size_t eyeIdx = currentFace.outsideSet[0];
            float maxEyeDist = math::dot(currentFace.normal, points[eyeIdx]) - currentFace.dist;
            for (size_t pi : currentFace.outsideSet) {
                float d = math::dot(currentFace.normal, points[pi]) - currentFace.dist;
                if (d > maxEyeDist) { maxEyeDist = d; eyeIdx = pi; }
            }
            const math::vec3& eye = points[eyeIdx];

            // Find visible faces using BFS from current face (instead of scanning all)
            visibleFaces.clear();
            visited.assign(faces.size(), false);

            std::vector<size_t> bfsQueue;
            bfsQueue.push_back(faceIdx);
            visited[faceIdx] = true;

            while (!bfsQueue.empty()) {
                size_t fi = bfsQueue.back();
                bfsQueue.pop_back();

                if (!faces[fi].active) continue;

                float d = math::dot(faces[fi].normal, eye) - faces[fi].dist;
                if (d > EPS) {
                    visibleFaces.push_back(fi);
                    // Check neighbors
                    for (int e = 0; e < 3; ++e) {
                        size_t ni = faces[fi].neighbors[e];
                        if (ni != SIZE_MAX && !visited[ni]) {
                            visited[ni] = true;
                            bfsQueue.push_back(ni);
                        }
                    }
                }
            }

            if (visibleFaces.empty()) {
                auto& os = currentFace.outsideSet;
                os.erase(std::remove(os.begin(), os.end(), eyeIdx), os.end());
                if (!os.empty()) faceStack.push_back(faceIdx);
                continue;
            }

            // Find horizon edges - edges shared with non-visible faces
            std::vector<std::pair<size_t, size_t>> horizonEdges;
            std::unordered_set<size_t> visibleSet(visibleFaces.begin(), visibleFaces.end());

            for (size_t vi : visibleFaces) {
                const Face& vf = faces[vi];
                for (int e = 0; e < 3; ++e) {
                    size_t neighborIdx = vf.neighbors[e];
                    // Horizon edge: neighbor is not visible (either inactive or not in visibleSet)
                    if (neighborIdx == SIZE_MAX ||
                        !faces[neighborIdx].active ||
                        visibleSet.find(neighborIdx) == visibleSet.end()) {
                        size_t a = vf.v[e], b = vf.v[(e + 1) % 3];
                        horizonEdges.push_back({b, a});  // Reverse winding for new face
                    }
                }
            }

            // Collect orphaned points
            std::vector<size_t> orphanedPoints;
            for (size_t vi : visibleFaces) {
                for (size_t pi : faces[vi].outsideSet) {
                    if (pi != eyeIdx) orphanedPoints.push_back(pi);
                }
            }

            // Deactivate visible faces and unregister their edges
            for (size_t vi : visibleFaces) {
                faces[vi].active = false;
                faces[vi].outsideSet.clear();
                // Unlink from neighbors
                for (int e = 0; e < 3; ++e) {
                    size_t ni = faces[vi].neighbors[e];
                    if (ni != SIZE_MAX) {
                        // Find which edge of neighbor points to us
                        for (int ne = 0; ne < 3; ++ne) {
                            if (faces[ni].neighbors[ne] == vi) {
                                faces[ni].neighbors[ne] = SIZE_MAX;
                                break;
                            }
                        }
                    }
                }
            }

            // Create new faces
            newFaceIndices.clear();
            for (const auto& edge : horizonEdges) {
                size_t newIdx = createFace(edge.first, edge.second, eyeIdx);
                newFaceIndices.push_back(newIdx);
            }

            // Redistribute orphaned points to new faces only
            for (size_t pi : orphanedPoints) {
                float bestDist = EPS;
                size_t bestFace = SIZE_MAX;
                for (size_t ni : newFaceIndices) {
                    float d = math::dot(faces[ni].normal, points[pi]) - faces[ni].dist;
                    if (d > bestDist) { bestDist = d; bestFace = ni; }
                }
                if (bestFace != SIZE_MAX) {
                    faces[bestFace].outsideSet.push_back(pi);
                }
            }

            // Add new faces with outside points to stack
            for (size_t ni : newFaceIndices) {
                if (!faces[ni].outsideSet.empty()) faceStack.push_back(ni);
            }
        }

        // Collect result from active faces
        std::vector<std::array<size_t, 3>> result;
        result.reserve(faces.size());
        for (size_t fi = 0; fi < faces.size(); ++fi) {
            if (faces[fi].active) {
                result.push_back({faces[fi].v[0], faces[fi].v[1], faces[fi].v[2]});
            }
        }
        return result;
    }

private:
    HypergraphRenderConfig config_;

    math::vec3 sphere_point(const math::vec3& center, float radius, float theta, float phi) {
        return center + math::vec3(
            std::sin(theta) * std::cos(phi),
            std::cos(theta),
            std::sin(theta) * std::sin(phi)
        ) * radius;
    }
};

// Evolution graph renderer (states + events + causal/branchial edges)
class EvolutionRenderer {
public:
    EvolutionRenderer() = default;

    void set_config(const HypergraphRenderConfig& config) { config_ = config; }

    // Forward timing print to HypergraphRenderer's static timing
    void print_minkowski_timing() {
        HypergraphRenderer::print_minkowski_timing_static();
    }

    // Layout for evolution graph (state positions)
    struct EvolutionLayout {
        std::vector<math::vec3> state_positions;
        std::vector<math::vec3> event_positions;  // For EvolutionGraph mode
        std::vector<bool> state_visible;  // Which states to render (reachable from roots)

        math::vec3 get_state_pos(StateId s) const {
            return s < state_positions.size() ? state_positions[s] : math::vec3(0, 0, 0);
        }

        bool is_state_visible(StateId s) const {
            return s < state_visible.size() ? state_visible[s] : false;
        }
    };

    // Generated geometry for evolution graph
    struct EvolutionGeometry {
        // State containers - wireframe edges (12 lines per cube)
        std::vector<RenderVertex> state_wireframe;

        // State containers - translucent faces (very faint)
        std::vector<RenderVertex> state_faces;

        // Internal hypergraph geometry (inside each state cube)
        // When use_instanced_rendering=true: only instance vectors are populated
        // When use_instanced_rendering=false: only legacy triangle vectors are populated
        std::vector<RenderVertex> internal_vertex_spheres;  // Opaque vertex spheres (legacy triangles)
        std::vector<SphereInstance> internal_sphere_instances;  // Instanced spheres (preferred)
        std::vector<RenderVertex> internal_edge_lines;      // Edge lines
        std::vector<RenderVertex> internal_arrows;          // Arrow cones (legacy triangles)
        std::vector<ConeInstance> internal_cone_instances;  // Instanced cones (preferred)
        std::vector<RenderVertex> internal_bubbles;         // Translucent bubbles

        // Event edges (between states)
        std::vector<RenderVertex> event_lines;
        std::vector<RenderVertex> event_arrows;  // Arrowheads for event edges (legacy)
        std::vector<ConeInstance> event_cone_instances;  // Instanced cones (preferred)

        // Causal edges
        std::vector<RenderVertex> causal_lines;
        std::vector<RenderVertex> causal_arrows;  // Arrowheads for causal edges (legacy)
        std::vector<ConeInstance> causal_cone_instances;  // Instanced cones (preferred)

        // Branchial edges
        std::vector<RenderVertex> branchial_lines;

        void clear() {
            state_wireframe.clear();
            state_faces.clear();
            internal_vertex_spheres.clear();
            internal_sphere_instances.clear();
            internal_edge_lines.clear();
            internal_arrows.clear();
            internal_cone_instances.clear();
            internal_bubbles.clear();
            event_lines.clear();
            event_arrows.clear();
            event_cone_instances.clear();
            causal_lines.clear();
            causal_arrows.clear();
            causal_cone_instances.clear();
            branchial_lines.clear();
        }
    };

    // Generate geometry for states graph view
    EvolutionGeometry generate_states_graph(const Evolution& evo,
                                            const EvolutionLayout& layout) {
        EvolutionGeometry geo;

        // We'll use a HypergraphRenderer for internal hypergraphs
        HypergraphRenderer hg_renderer;
        // Scale down internal elements to fit in cube
        float internal_scale = config_.state_size * 0.35f;  // Leave margin inside cube
        HypergraphRenderConfig internal_config;
        internal_config.vertex_radius = 0.08f * internal_scale;
        internal_config.edge_thickness = 0.02f * internal_scale;
        internal_config.arrow_length = 0.15f * internal_scale;
        internal_config.arrow_radius = 0.06f * internal_scale;
        hg_renderer.set_config(internal_config);

        // Generate state containers and internal hypergraphs
        // Only render states that are visible (reachable from roots)
        for (const auto& state : evo.states) {
            // Skip non-visible states (orphaned states waiting for edges)
            if (!layout.is_state_visible(state.id)) {
                continue;
            }

            math::vec3 cube_center = layout.get_state_pos(state.id);

            // Color based on state type
            math::vec4 color;
            if (state.is_initial) {
                color = config_.state_initial_color;  // Yellow for initial
            } else if (state.canonical_id != state.id) {
                color = config_.state_canonical_color;  // Blue for non-canonical (maps to other)
            } else {
                color = config_.state_color;  // Green for normal/canonical
            }

            // Generate wireframe cube (12 edges as lines)
            generate_cube_wireframe(geo.state_wireframe, cube_center, config_.state_size, color);

            // Generate translucent faces (very faint)
            math::vec4 face_color = {color.x, color.y, color.z, config_.state_face_alpha};
            generate_cube_faces(geo.state_faces, cube_center, config_.state_size, face_color);

            // Generate internal hypergraph if non-empty
            const auto& hg = state.hypergraph;
            auto used_verts = hg.get_vertices();
            if (!used_verts.empty()) {
                // Simple layout for internal hypergraph: spread vertices in a sphere
                // Layout positions need to accommodate vertex_count (max ID + 1)
                // but we only position actually used vertices
                HypergraphLayout internal_layout;
                internal_layout.vertex_positions.resize(hg.vertex_count);

                // Use golden spiral for uniform distribution of used vertices
                float golden_angle = math::PI * (3.0f - std::sqrt(5.0f));
                uint32_t num_used = static_cast<uint32_t>(used_verts.size());
                for (uint32_t i = 0; i < num_used; ++i) {
                    VertexId v = used_verts[i];
                    float y = 1.0f - (i / float(num_used - 1 + 0.001f)) * 2.0f;  // -1 to 1
                    float radius_at_y = std::sqrt(1.0f - y * y);
                    float theta = golden_angle * i;

                    // Scale to fit inside cube (use ~70% of half-size)
                    float scale = config_.state_size * 0.35f;
                    internal_layout.vertex_positions[v] = cube_center + math::vec3(
                        std::cos(theta) * radius_at_y * scale,
                        y * scale,
                        std::sin(theta) * radius_at_y * scale
                    );
                }

                // Generate internal hypergraph geometry
                auto internal_geo = hg_renderer.generate(hg, internal_layout);

                // Append to evolution geometry
                // Based on use_instanced_rendering flag, only one set will be populated
                geo.internal_sphere_instances.insert(geo.internal_sphere_instances.end(),
                    internal_geo.sphere_instances.begin(), internal_geo.sphere_instances.end());
                geo.internal_vertex_spheres.insert(geo.internal_vertex_spheres.end(),
                    internal_geo.vertex_triangles.begin(), internal_geo.vertex_triangles.end());
                geo.internal_edge_lines.insert(geo.internal_edge_lines.end(),
                    internal_geo.edge_lines.begin(), internal_geo.edge_lines.end());
                // Also include self-loop lines (arcs for {x,x} type edges)
                geo.internal_edge_lines.insert(geo.internal_edge_lines.end(),
                    internal_geo.self_loop_lines.begin(), internal_geo.self_loop_lines.end());
                geo.internal_cone_instances.insert(geo.internal_cone_instances.end(),
                    internal_geo.cone_instances.begin(), internal_geo.cone_instances.end());
                geo.internal_arrows.insert(geo.internal_arrows.end(),
                    internal_geo.arrow_triangles.begin(), internal_geo.arrow_triangles.end());
                geo.internal_bubbles.insert(geo.internal_bubbles.end(),
                    internal_geo.bubble_triangles.begin(), internal_geo.bubble_triangles.end());
            }
        }

        // Generate edges
        float half_size = config_.state_size * 0.5f;

        // Bundle event edges by (source, target) to count multiplicity
        // Key: (min(src,tgt), max(src,tgt)) for undirected key, but store actual source for direction
        std::map<std::pair<StateId, StateId>, std::vector<const EvolutionEdge*>> event_bundles;
        std::map<std::pair<StateId, StateId>, std::vector<const EvolutionEdge*>> branchial_bundles;

        for (const auto& edge : evo.evolution_edges) {
            // Validate state IDs are within bounds before bundling
            if (edge.source >= evo.states.size() || edge.target >= evo.states.size()) {
                continue;  // Skip edges with invalid state references
            }

            if (edge.type == EvolutionEdgeType::Event) {
                // Use ordered pair (source, target) as key
                auto key = std::make_pair(edge.source, edge.target);
                event_bundles[key].push_back(&edge);
            } else if (edge.type == EvolutionEdgeType::Branchial) {
                // Branchial are undirected, use canonical ordering
                auto key = edge.source < edge.target
                    ? std::make_pair(edge.source, edge.target)
                    : std::make_pair(edge.target, edge.source);
                branchial_bundles[key].push_back(&edge);
            }
        }

        // Generate bundled event edges with curves when multiplicity > 1
        for (const auto& [key, edges] : event_bundles) {
            StateId source = key.first;
            StateId target = key.second;

            // Skip edges with invalid state IDs (out of bounds)
            if (source >= layout.state_positions.size() || target >= layout.state_positions.size()) {
                continue;
            }

            // Skip edges involving invisible states (safety check)
            if (!layout.is_state_visible(source) || !layout.is_state_visible(target)) {
                continue;
            }

            uint32_t multiplicity = static_cast<uint32_t>(edges.size());

            math::vec3 center1 = layout.get_state_pos(source);
            math::vec3 center2 = layout.get_state_pos(target);

            // Skip degenerate edges
            if (math::length(center2 - center1) < 0.001f) continue;

            if (multiplicity == 1) {
                // Single edge: straight line with arrow (special case for efficiency)
                math::vec3 dir = math::normalize(center2 - center1);
                math::vec3 p1 = ray_cube_exit(center1, dir, half_size);
                math::vec3 p2 = ray_cube_enter(center2, dir * -1.0f, half_size);
                float arrow_len = config_.event_arrow_length;
                math::vec3 arrow_base = p2 - dir * arrow_len;
                generate_line(geo.event_lines, p1, arrow_base, config_.event_edge_color);
                add_cone_instance(geo.event_cone_instances, p2, dir,
                                 arrow_len, config_.event_arrow_radius, config_.event_edge_color);
            } else {
                // Multiple edges: use bundled curved rendering
                EdgeBundleParams params = compute_edge_bundle_params(
                    center1, center2, half_size, math::vec3(0, 0, 1),
                    config_.event_edge_color, multiplicity,
                    0.4f, 0.15f,  // spread factors
                    true, config_.event_arrow_length, config_.event_arrow_radius
                );
                render_edge_bundle(geo.event_lines, &geo.event_cone_instances, params);
            }
        }

        // Generate branchial edges (always curved for visual distinction, no arrows)
        for (const auto& [key, edges] : branchial_bundles) {
            StateId source = key.first;
            StateId target = key.second;

            // Skip edges with invalid state IDs (out of bounds)
            if (source >= layout.state_positions.size() || target >= layout.state_positions.size()) {
                continue;
            }

            // Skip edges involving invisible states
            if (!layout.is_state_visible(source) || !layout.is_state_visible(target)) {
                continue;
            }

            uint32_t multiplicity = static_cast<uint32_t>(edges.size());

            math::vec3 center1 = layout.get_state_pos(source);
            math::vec3 center2 = layout.get_state_pos(target);

            // Skip degenerate edges
            if (math::length(center2 - center1) < 0.001f) continue;

            EdgeBundleParams params = compute_edge_bundle_params(
                center1, center2, half_size, math::vec3(0, 0, 1),
                config_.branchial_edge_color, multiplicity,
                0.3f, 0.08f,  // spread factors
                false, 0.0f, 0.0f  // no arrows
            );
            render_edge_bundle(geo.branchial_lines, nullptr, params);
        }

        // Generate causal edges (connect producer's output state to consumer's input state)
        // Causal edges show the flow of causality between events
        std::map<std::pair<StateId, StateId>, std::vector<const EvolutionEdge*>> causal_bundles;

        for (const auto& edge : evo.evolution_edges) {
            if (edge.type == EvolutionEdgeType::Causal) {
                // edge.source and edge.target are EventIds
                const Event* producer = evo.get_event(static_cast<EventId>(edge.source));
                const Event* consumer = evo.get_event(static_cast<EventId>(edge.target));

                if (producer && consumer) {
                    // Causal edge: producer's output state → consumer's output state
                    StateId src_state = producer->output_state;
                    StateId tgt_state = consumer->output_state;

                    // Validate state IDs are within bounds
                    if (src_state < evo.states.size() && tgt_state < evo.states.size()) {
                        auto key = std::make_pair(src_state, tgt_state);
                        causal_bundles[key].push_back(&edge);
                    }
                }
            }
        }

        // Causal edges use smaller arrows to distinguish from event edges
        float causal_arrow_len = config_.event_arrow_length * 0.7f;
        float causal_arrow_rad = config_.event_arrow_radius * 0.7f;

        for (const auto& [key, edges] : causal_bundles) {
            StateId source = key.first;
            StateId target = key.second;

            // Skip edges with invalid state IDs (out of bounds)
            if (source >= layout.state_positions.size() || target >= layout.state_positions.size()) {
                continue;
            }

            // Skip edges involving invisible states
            if (!layout.is_state_visible(source) || !layout.is_state_visible(target)) {
                continue;
            }

            uint32_t multiplicity = static_cast<uint32_t>(edges.size());

            math::vec3 center1 = layout.get_state_pos(source);
            math::vec3 center2 = layout.get_state_pos(target);

            // Skip degenerate edges
            if (math::length(center2 - center1) < 0.001f) continue;

            // Causal edges use different up_hint (0,1,0) than branchial (0,0,1) to distinguish
            EdgeBundleParams params = compute_edge_bundle_params(
                center1, center2, half_size, math::vec3(0, 1, 0),
                config_.causal_edge_color, multiplicity,
                0.25f, 0.06f,  // spread factors
                true, causal_arrow_len, causal_arrow_rad
            );
            render_edge_bundle(geo.causal_lines, &geo.causal_cone_instances, params);
        }

        return geo;
    }

    // Parameters for rendering an edge bundle (common to all edge types)
    struct EdgeBundleParams {
        math::vec3 p1;              // Start surface point
        math::vec3 p2;              // End surface point
        math::vec3 dir;             // Normalized direction from p1 to p2
        math::vec3 perp;            // Perpendicular direction for spreading
        math::vec4 color;           // Edge color
        uint32_t multiplicity;      // Number of edges in bundle
        float spread;               // Max spread at control point (state_size multiplier)
        float endpoint_spread;      // Spread at endpoints (state_size multiplier)
        bool has_arrow;             // Whether to draw arrows
        float arrow_length;         // Arrow length (0 if no arrow)
        float arrow_radius;         // Arrow radius (0 if no arrow)
    };

    // Compute common edge bundle geometry from source/target positions
    EdgeBundleParams compute_edge_bundle_params(
        const math::vec3& center1, const math::vec3& center2,
        float half_size, const math::vec3& up_hint,
        const math::vec4& color, uint32_t multiplicity,
        float spread_factor, float endpoint_spread_factor,
        bool has_arrow, float arrow_length, float arrow_radius
    ) {
        EdgeBundleParams params;
        params.color = color;
        params.multiplicity = multiplicity;
        params.spread = config_.state_size * spread_factor;
        params.endpoint_spread = config_.state_size * endpoint_spread_factor;
        params.has_arrow = has_arrow;
        params.arrow_length = arrow_length;
        params.arrow_radius = arrow_radius;

        params.dir = center2 - center1;
        float len = math::length(params.dir);
        if (len < 0.001f) {
            params.dir = math::vec3(1, 0, 0);
        } else {
            params.dir = params.dir / len;
        }

        params.p1 = ray_cube_exit(center1, params.dir, half_size);
        params.p2 = ray_cube_enter(center2, params.dir * -1.0f, half_size);
        params.perp = compute_consistent_perpendicular(params.p1, params.p2, up_hint);

        return params;
    }

    // Render a single edge within a bundle at the given t value [-1, +1]
    // t=0 is center line, negative/positive spread to either side
    void render_bundle_edge(
        std::vector<RenderVertex>& lines_out,
        std::vector<ConeInstance>* cones_out,  // nullptr if no arrows
        const EdgeBundleParams& params,
        float t  // Position in bundle: -1 to +1, 0 = center
    ) {
        float control_offset = t * params.spread;
        float endpoint_offset = t * params.endpoint_spread;

        math::vec3 p1_offset = params.p1 + params.perp * endpoint_offset;
        math::vec3 p2_offset = params.p2 + params.perp * endpoint_offset;
        math::vec3 midpoint = (p1_offset + p2_offset) * 0.5f;
        math::vec3 control = midpoint + params.perp * control_offset;

        if (params.has_arrow && cones_out) {
            generate_quadratic_bezier(lines_out, p1_offset, control, p2_offset,
                                     params.arrow_length, params.color);

            math::vec3 final_tangent = p2_offset - control;
            if (math::length(final_tangent) > 0.001f) {
                final_tangent = math::normalize(final_tangent);
            } else {
                final_tangent = params.dir;
            }
            add_cone_instance(*cones_out, p2_offset, final_tangent,
                             params.arrow_length, params.arrow_radius, params.color);
        } else {
            generate_quadratic_bezier_full(lines_out, p1_offset, control, p2_offset, params.color);
        }
    }

    // Render an entire edge bundle with all its edges spread perpendicular
    void render_edge_bundle(
        std::vector<RenderVertex>& lines_out,
        std::vector<ConeInstance>* cones_out,
        const EdgeBundleParams& params
    ) {
        for (uint32_t i = 0; i < params.multiplicity; ++i) {
            // t ranges from -1 to +1, equally spaced
            // For single edge: t = 0 (center line)
            float t = (params.multiplicity == 1) ? 0.0f :
                (static_cast<float>(i) / (params.multiplicity - 1) - 0.5f) * 2.0f;
            render_bundle_edge(lines_out, cones_out, params, t);
        }
    }

    // Generate quadratic bezier curve as line segments
    // Leaves space for arrow at end
    void generate_quadratic_bezier(std::vector<RenderVertex>& out,
                                   const math::vec3& p0, const math::vec3& control, const math::vec3& p2,
                                   float arrow_len, const math::vec4& color, int segments = 16) {
        // Shorten the curve to leave room for arrow
        math::vec3 final_dir = p2 - control;
        float final_len = math::length(final_dir);
        math::vec3 actual_end = p2;
        if (final_len > arrow_len) {
            actual_end = p2 - math::normalize(final_dir) * arrow_len;
        }

        math::vec3 prev = p0;
        for (int i = 1; i <= segments; ++i) {
            float t = static_cast<float>(i) / segments;
            // Quadratic bezier: B(t) = (1-t)^2 * P0 + 2(1-t)t * Control + t^2 * P2
            // But we're shortening P2, so interpolate toward actual_end
            float u = 1.0f - t;
            math::vec3 end_pt = (t < 1.0f) ? p2 : actual_end;
            // For intermediate points, use full curve; for final segment, use shortened
            math::vec3 pt;
            if (i < segments) {
                pt = p0 * (u * u) + control * (2.0f * u * t) + p2 * (t * t);
            } else {
                pt = actual_end;
            }
            generate_line(out, prev, pt, color);
            prev = pt;
        }
    }

    // Generate full quadratic bezier curve (no arrow space)
    void generate_quadratic_bezier_full(std::vector<RenderVertex>& out,
                                        const math::vec3& p0, const math::vec3& control, const math::vec3& p2,
                                        const math::vec4& color, int segments = 16) {
        math::vec3 prev = p0;
        for (int i = 1; i <= segments; ++i) {
            float t = static_cast<float>(i) / segments;
            float u = 1.0f - t;
            math::vec3 pt = p0 * (u * u) + control * (2.0f * u * t) + p2 * (t * t);
            generate_line(out, prev, pt, color);
            prev = pt;
        }
    }

    // Find where ray from cube center exits the cube surface
    math::vec3 ray_cube_exit(const math::vec3& center, const math::vec3& dir, float half_size) {
        // Find t where ray exits cube (max of all axis crossings)
        float t = std::numeric_limits<float>::max();
        for (int axis = 0; axis < 3; ++axis) {
            float d = (axis == 0) ? dir.x : (axis == 1) ? dir.y : dir.z;
            if (std::abs(d) > 0.0001f) {
                float boundary = (d > 0) ? half_size : -half_size;
                float t_axis = boundary / d;
                if (t_axis > 0 && t_axis < t) t = t_axis;
            }
        }
        return center + dir * t;
    }

    // Find where ray entering cube hits the surface (same as exit from opposite direction)
    math::vec3 ray_cube_enter(const math::vec3& center, const math::vec3& dir, float half_size) {
        return ray_cube_exit(center, dir, half_size);
    }

    // Generate arrow cone
    void generate_arrow_cone(std::vector<RenderVertex>& out,
                            const math::vec3& base, const math::vec3& tip,
                            float radius, const math::vec4& color) {
        math::vec3 dir = tip - base;
        float height = dir.length();
        if (height < 0.001f) return;
        dir = dir / height;

        // Find perpendicular vectors
        math::vec3 up = (std::abs(dir.y) < 0.9f) ? math::vec3(0, 1, 0) : math::vec3(1, 0, 0);
        math::vec3 right = math::cross(dir, up).normalized();
        math::vec3 forward = math::cross(right, dir).normalized();

        const int segments = 8;
        for (int i = 0; i < segments; ++i) {
            float a1 = math::TAU * i / segments;
            float a2 = math::TAU * (i + 1) / segments;

            math::vec3 p1 = base + (right * std::cos(a1) + forward * std::sin(a1)) * radius;
            math::vec3 p2 = base + (right * std::cos(a2) + forward * std::sin(a2)) * radius;

            // Side triangle (CCW winding from outside)
            out.push_back({tip.x, tip.y, tip.z, color.x, color.y, color.z, color.w});
            out.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
            out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});

            // Base triangle (CCW winding from below)
            out.push_back({base.x, base.y, base.z, color.x, color.y, color.z, color.w});
            out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
            out.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
        }
    }

private:
    HypergraphRenderConfig config_;

    // Compute a perpendicular direction for edge bundling that is symmetric.
    // The key insight is that we always order the endpoints canonically first,
    // so the direction vector is always computed the same way regardless of
    // which endpoint is "source" vs "target".
    math::vec3 compute_consistent_perpendicular(const math::vec3& p1, const math::vec3& p2,
                                                 const math::vec3& up_hint) {
        // Canonically order endpoints to ensure consistent direction computation
        // Order by: x first, then y, then z
        math::vec3 lo = p1, hi = p2;
        if (p1.x > p2.x || (p1.x == p2.x && p1.y > p2.y) ||
            (p1.x == p2.x && p1.y == p2.y && p1.z > p2.z)) {
            lo = p2;
            hi = p1;
        }

        math::vec3 dir = hi - lo;
        float len = math::length(dir);
        if (len < 0.001f) return math::vec3(1, 0, 0);
        dir = dir / len;

        // Compute perpendicular using cross product with up hint
        math::vec3 perp = math::cross(dir, up_hint);
        if (math::length(perp) < 0.001f) {
            // dir is parallel to up_hint, use alternative
            perp = math::cross(dir, math::vec3(1, 0, 0));
            if (math::length(perp) < 0.001f) {
                perp = math::cross(dir, math::vec3(0, 0, 1));
            }
        }
        perp = math::normalize(perp);

        return perp;
    }

    void generate_line(std::vector<RenderVertex>& out,
                      const math::vec3& p1, const math::vec3& p2,
                      const math::vec4& color) {
        out.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
        out.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
    }

    // Add cone instance (for instanced rendering)
    void add_cone_instance(std::vector<ConeInstance>& out,
                          const math::vec3& tip, const math::vec3& direction,
                          float length, float radius, const math::vec4& color) {
        math::vec3 dir = math::normalize(direction);
        out.push_back({
            tip.x, tip.y, tip.z,
            dir.x, dir.y, dir.z,
            length, radius,
            color.x, color.y, color.z, color.w
        });
    }

    // Generate wireframe cube - 12 edges as line segments
    void generate_cube_wireframe(std::vector<RenderVertex>& out,
                                 const math::vec3& center, float size,
                                 const math::vec4& color) {
        float s = size * 0.5f;

        // 8 corners of cube
        math::vec3 corners[8] = {
            center + math::vec3(-s, -s, -s),  // 0: back-bottom-left
            center + math::vec3( s, -s, -s),  // 1: back-bottom-right
            center + math::vec3( s,  s, -s),  // 2: back-top-right
            center + math::vec3(-s,  s, -s),  // 3: back-top-left
            center + math::vec3(-s, -s,  s),  // 4: front-bottom-left
            center + math::vec3( s, -s,  s),  // 5: front-bottom-right
            center + math::vec3( s,  s,  s),  // 6: front-top-right
            center + math::vec3(-s,  s,  s),  // 7: front-top-left
        };

        // 12 edges of cube (pairs of corner indices)
        int edges[12][2] = {
            // Back face edges
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            // Front face edges
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            // Connecting edges (front to back)
            {0, 4}, {1, 5}, {2, 6}, {3, 7},
        };

        for (int e = 0; e < 12; ++e) {
            const auto& c0 = corners[edges[e][0]];
            const auto& c1 = corners[edges[e][1]];
            out.push_back({c0.x, c0.y, c0.z, color.x, color.y, color.z, color.w});
            out.push_back({c1.x, c1.y, c1.z, color.x, color.y, color.z, color.w});
        }
    }

    // Generate cube faces as triangles (for translucent fill)
    void generate_cube_faces(std::vector<RenderVertex>& out,
                             const math::vec3& center, float size,
                             const math::vec4& color) {
        float s = size * 0.5f;

        // 8 corners of cube
        math::vec3 corners[8] = {
            center + math::vec3(-s, -s, -s),
            center + math::vec3( s, -s, -s),
            center + math::vec3( s,  s, -s),
            center + math::vec3(-s,  s, -s),
            center + math::vec3(-s, -s,  s),
            center + math::vec3( s, -s,  s),
            center + math::vec3( s,  s,  s),
            center + math::vec3(-s,  s,  s),
        };

        // Face indices (2 triangles per face)
        int faces[6][4] = {
            {0, 1, 2, 3},  // Back
            {5, 4, 7, 6},  // Front
            {4, 0, 3, 7},  // Left
            {1, 5, 6, 2},  // Right
            {3, 2, 6, 7},  // Top
            {4, 5, 1, 0},  // Bottom
        };

        for (int f = 0; f < 6; ++f) {
            const auto& c = corners;
            int* idx = faces[f];

            // Triangle 1
            out.push_back({c[idx[0]].x, c[idx[0]].y, c[idx[0]].z, color.x, color.y, color.z, color.w});
            out.push_back({c[idx[1]].x, c[idx[1]].y, c[idx[1]].z, color.x, color.y, color.z, color.w});
            out.push_back({c[idx[2]].x, c[idx[2]].y, c[idx[2]].z, color.x, color.y, color.z, color.w});

            // Triangle 2
            out.push_back({c[idx[0]].x, c[idx[0]].y, c[idx[0]].z, color.x, color.y, color.z, color.w});
            out.push_back({c[idx[2]].x, c[idx[2]].y, c[idx[2]].z, color.x, color.y, color.z, color.w});
            out.push_back({c[idx[3]].x, c[idx[3]].y, c[idx[3]].z, color.x, color.y, color.z, color.w});
        }
    }
};

} // namespace viz::scene
