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
#include <map>

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
                vertices.push_back(p1);
                vertices.push_back(p2);
                vertices.push_back(p3);

                vertices.push_back(p1);
                vertices.push_back(p3);
                vertices.push_back(p4);
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
                    generate_smooth_loop_arc(geo.self_loop_lines, geo.bubble_triangles,
                                            geo.arrow_triangles,
                                            center, virtual_pos,
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
    void generate_smooth_loop_arc(std::vector<RenderVertex>& line_out,
                                  std::vector<RenderVertex>& film_out,
                                  std::vector<RenderVertex>& arrow_out,
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
            generate_line(line_out, loop_points[i], loop_points[i + 1], color);
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
                generate_cone(arrow_out, arrow_pos, arrow_dir,
                             config_.arrow_length * 0.8f, config_.arrow_radius * 0.8f,
                             arrow_color);
            }
        }

        // Generate film (fan triangles from center to loop) - only if requested
        // For arity-2 self-loops, we skip the film to show just the arc
        if (generate_film) {
            math::vec4 film_color = {color.x, color.y, color.z, color.w * 0.5f};
            for (size_t i = 0; i + 1 < loop_points.size(); ++i) {
                // Triangle: center, loop[i], loop[i+1]
                film_out.push_back({center.x, center.y, center.z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                film_out.push_back({loop_points[i].x, loop_points[i].y, loop_points[i].z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                film_out.push_back({loop_points[i+1].x, loop_points[i+1].y, loop_points[i+1].z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                // Back face
                film_out.push_back({center.x, center.y, center.z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                film_out.push_back({loop_points[i+1].x, loop_points[i+1].y, loop_points[i+1].z,
                                   film_color.x, film_color.y, film_color.z, film_color.w});
                film_out.push_back({loop_points[i].x, loop_points[i].y, loop_points[i].z,
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
        generate_minkowski_sum(out, points, radius, color);
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
    // MINKOWSKI SUM - Proper algorithm for two convex polyhedra
    // P ⊕ Q = convex_hull({p + q : p ∈ vertices(P), q ∈ vertices(Q)})
    // ==========================================================================

    // Generate icosphere vertices (unit sphere approximation)
    std::vector<math::vec3> generate_icosphere_vertices(int subdivisions = 1) {
        // Start with icosahedron vertices
        const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;  // Golden ratio
        std::vector<math::vec3> verts = {
            math::normalize(math::vec3(-1,  t,  0)),
            math::normalize(math::vec3( 1,  t,  0)),
            math::normalize(math::vec3(-1, -t,  0)),
            math::normalize(math::vec3( 1, -t,  0)),
            math::normalize(math::vec3( 0, -1,  t)),
            math::normalize(math::vec3( 0,  1,  t)),
            math::normalize(math::vec3( 0, -1, -t)),
            math::normalize(math::vec3( 0,  1, -t)),
            math::normalize(math::vec3( t,  0, -1)),
            math::normalize(math::vec3( t,  0,  1)),
            math::normalize(math::vec3(-t,  0, -1)),
            math::normalize(math::vec3(-t,  0,  1)),
        };

        // Icosahedron faces
        std::vector<std::array<size_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
        };

        // Subdivide faces
        for (int s = 0; s < subdivisions; ++s) {
            std::vector<std::array<size_t, 3>> new_faces;
            std::map<std::pair<size_t, size_t>, size_t> edge_midpoints;

            auto get_midpoint = [&](size_t i1, size_t i2) -> size_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = edge_midpoints.find(key);
                if (it != edge_midpoints.end()) return it->second;
                size_t idx = verts.size();
                verts.push_back(math::normalize((verts[i1] + verts[i2]) * 0.5f));
                edge_midpoints[key] = idx;
                return idx;
            };

            for (const auto& f : faces) {
                size_t a = f[0], b = f[1], c = f[2];
                size_t ab = get_midpoint(a, b);
                size_t bc = get_midpoint(b, c);
                size_t ca = get_midpoint(c, a);
                new_faces.push_back({a, ab, ca});
                new_faces.push_back({b, bc, ab});
                new_faces.push_back({c, ca, bc});
                new_faces.push_back({ab, bc, ca});
            }
            faces = std::move(new_faces);
        }

        return verts;
    }

    // Generate Minkowski sum of convex point set with sphere
    // Uses the mathematically correct algorithm: hull(P + Q)
    void generate_minkowski_sum(std::vector<RenderVertex>& out,
                                const std::vector<math::vec3>& points,
                                float radius, const math::vec4& color) {
        if (points.size() < 3) return;

        // Get icosphere vertices (sphere approximation)
        // 1 subdivision = 42 vertices, 2 = 162, good enough for smooth appearance
        std::vector<math::vec3> sphere_verts = generate_icosphere_vertices(1);

        // Scale sphere vertices by radius
        for (auto& v : sphere_verts) {
            v = v * radius;
        }

        // Generate all pairwise sums: {p + q : p ∈ points, q ∈ sphere_verts}
        std::vector<math::vec3> sum_points;
        sum_points.reserve(points.size() * sphere_verts.size());
        for (const auto& p : points) {
            for (const auto& q : sphere_verts) {
                sum_points.push_back(p + q);
            }
        }

        // Compute convex hull of the sum points
        auto hull_faces = compute_convex_hull(sum_points);
        if (hull_faces.empty()) return;

        // Compute centroid for normal orientation
        math::vec3 centroid(0, 0, 0);
        for (const auto& p : sum_points) centroid = centroid + p;
        centroid = centroid * (1.0f / sum_points.size());

        // Output triangles with correct winding
        for (const auto& face : hull_faces) {
            math::vec3 a = sum_points[face[0]];
            math::vec3 b = sum_points[face[1]];
            math::vec3 c = sum_points[face[2]];

            // Check and fix winding for outward-facing normal
            math::vec3 face_center = (a + b + c) / 3.0f;
            math::vec3 normal = math::cross(b - a, c - a);
            if (math::dot(normal, face_center - centroid) < 0) {
                std::swap(b, c);  // Flip winding
            }

            out.push_back({a.x, a.y, a.z, color.x, color.y, color.z, color.w});
            out.push_back({b.x, b.y, b.z, color.x, color.y, color.z, color.w});
            out.push_back({c.x, c.y, c.z, color.x, color.y, color.z, color.w});
        }
    }

    // Legacy function - now calls unified Minkowski sum
    void generate_rounded_triangle(std::vector<RenderVertex>& out,
                                   const math::vec3& a, const math::vec3& b, const math::vec3& c,
                                   float radius, const math::vec4& color) {
        generate_minkowski_sum(out, {a, b, c}, radius, color);
    }

    // Legacy function - now calls unified Minkowski sum
    void generate_rounded_tetrahedron(std::vector<RenderVertex>& out,
                                      const math::vec3& a, const math::vec3& b,
                                      const math::vec3& c, const math::vec3& d,
                                      float radius, const math::vec4& color) {
        generate_minkowski_sum(out, {a, b, c, d}, radius, color);
    }

    // Legacy function - now calls unified Minkowski sum
    void generate_rounded_convex_hull(std::vector<RenderVertex>& out,
                                      const std::vector<math::vec3>& points,
                                      float radius, const math::vec4& color) {
        generate_minkowski_sum(out, points, radius, color);
    }

    // REMOVED: Old generate_rounded_triangle implementation (replaced by unified)
    // REMOVED: Old generate_rounded_tetrahedron implementation (replaced by unified)
    // REMOVED: Old generate_rounded_convex_hull implementation (replaced by unified)

    // Simple convex hull computation (quickhull-like)
    // Returns list of triangular faces as indices into points array
    std::vector<std::array<size_t, 3>> compute_convex_hull(const std::vector<math::vec3>& points) {
        std::vector<std::array<size_t, 3>> faces;
        if (points.size() < 4) return faces;

        // Find initial tetrahedron (4 non-coplanar points)
        size_t p0 = 0, p1 = 1, p2 = 2, p3 = 3;

        // Find most distant pair for first edge
        float max_dist = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = i + 1; j < points.size(); ++j) {
                float d = math::length(points[j] - points[i]);
                if (d > max_dist) {
                    max_dist = d;
                    p0 = i; p1 = j;
                }
            }
        }

        // Find point most distant from line p0-p1
        max_dist = 0;
        math::vec3 line_dir = math::normalize(points[p1] - points[p0]);
        for (size_t i = 0; i < points.size(); ++i) {
            if (i == p0 || i == p1) continue;
            math::vec3 v = points[i] - points[p0];
            math::vec3 proj = line_dir * math::dot(v, line_dir);
            float d = math::length(v - proj);
            if (d > max_dist) {
                max_dist = d;
                p2 = i;
            }
        }

        // Find point most distant from plane p0-p1-p2
        math::vec3 normal = math::normalize(math::cross(points[p1] - points[p0], points[p2] - points[p0]));
        max_dist = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            if (i == p0 || i == p1 || i == p2) continue;
            float d = std::abs(math::dot(points[i] - points[p0], normal));
            if (d > max_dist) {
                max_dist = d;
                p3 = i;
            }
        }

        // Initial tetrahedron faces (ensure consistent winding)
        math::vec3 centroid = (points[p0] + points[p1] + points[p2] + points[p3]) * 0.25f;

        auto add_face_if_outward = [&](size_t a, size_t b, size_t c) {
            math::vec3 face_center = (points[a] + points[b] + points[c]) / 3.0f;
            math::vec3 n = math::cross(points[b] - points[a], points[c] - points[a]);
            if (math::dot(n, face_center - centroid) < 0) {
                std::swap(b, c);
            }
            faces.push_back({a, b, c});
        };

        add_face_if_outward(p0, p1, p2);
        add_face_if_outward(p0, p1, p3);
        add_face_if_outward(p0, p2, p3);
        add_face_if_outward(p1, p2, p3);

        // For each remaining point, check if outside any face and expand hull
        for (size_t i = 0; i < points.size(); ++i) {
            if (i == p0 || i == p1 || i == p2 || i == p3) continue;

            // Find faces visible from this point
            std::vector<size_t> visible;
            for (size_t f = 0; f < faces.size(); ++f) {
                math::vec3 a = points[faces[f][0]];
                math::vec3 b = points[faces[f][1]];
                math::vec3 c = points[faces[f][2]];
                math::vec3 n = math::normalize(math::cross(b - a, c - a));
                math::vec3 face_center = (a + b + c) / 3.0f;

                // Point is outside if it's on the positive side of the face
                if (math::dot(points[i] - face_center, n) > 0.0001f) {
                    visible.push_back(f);
                }
            }

            if (visible.empty()) continue;  // Point is inside hull

            // Find horizon edges (edges of visible faces that border non-visible faces)
            std::vector<std::pair<size_t, size_t>> horizon;
            for (size_t vi : visible) {
                for (int e = 0; e < 3; ++e) {
                    size_t v1 = faces[vi][e];
                    size_t v2 = faces[vi][(e + 1) % 3];

                    // Check if this edge is shared with a non-visible face
                    bool shared = false;
                    for (size_t f = 0; f < faces.size(); ++f) {
                        bool is_visible = false;
                        for (size_t vf : visible) if (vf == f) { is_visible = true; break; }
                        if (is_visible) continue;

                        for (int e2 = 0; e2 < 3; ++e2) {
                            size_t u1 = faces[f][e2];
                            size_t u2 = faces[f][(e2 + 1) % 3];
                            if ((v1 == u1 && v2 == u2) || (v1 == u2 && v2 == u1)) {
                                shared = true;
                                break;
                            }
                        }
                        if (shared) break;
                    }
                    if (shared) {
                        horizon.push_back({v1, v2});
                    }
                }
            }

            // Remove visible faces
            std::vector<std::array<size_t, 3>> new_faces;
            for (size_t f = 0; f < faces.size(); ++f) {
                bool is_visible = false;
                for (size_t vi : visible) if (vi == f) { is_visible = true; break; }
                if (!is_visible) new_faces.push_back(faces[f]);
            }
            faces = new_faces;

            // Add new faces from horizon edges to the new point
            for (const auto& edge : horizon) {
                // Ensure consistent winding (outward facing)
                math::vec3 a = points[edge.first];
                math::vec3 b = points[edge.second];
                math::vec3 c = points[i];
                math::vec3 face_center = (a + b + c) / 3.0f;
                math::vec3 n = math::cross(b - a, c - a);

                // Recompute centroid with new point
                math::vec3 new_centroid = centroid;  // Approximation
                if (math::dot(n, face_center - new_centroid) < 0) {
                    faces.push_back({edge.first, i, edge.second});
                } else {
                    faces.push_back({edge.first, edge.second, i});
                }
            }
        }

        return faces;
    }

    // Legacy function for backward compatibility
    void generate_convex_hull_bubble(std::vector<RenderVertex>& out,
                                     const std::vector<math::vec3>& points,
                                     const math::vec4& color) {
        generate_bubble_for_arity(out, points, color);
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
        std::vector<RenderVertex> event_arrows;  // Arrowheads for event edges

        // Causal edges
        std::vector<RenderVertex> causal_lines;

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
            causal_lines.clear();
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
        internal_config.bubble_color.w = 0.15f;  // Low alpha for internal bubbles
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
            uint32_t multiplicity = static_cast<uint32_t>(edges.size());

            math::vec3 center1 = layout.get_state_pos(source);
            math::vec3 center2 = layout.get_state_pos(target);
            math::vec4 color = config_.event_edge_color;

            math::vec3 dir = center2 - center1;
            float len = math::length(dir);
            if (len < 0.001f) continue;
            dir = dir / len;

            // Surface points
            math::vec3 p1 = ray_cube_exit(center1, dir, half_size);
            math::vec3 p2 = ray_cube_enter(center2, dir * -1.0f, half_size);

            // Find perpendicular direction for spreading (consistent regardless of edge direction)
            math::vec3 perp = compute_consistent_perpendicular(p1, p2, math::vec3(0, 0, 1));

            if (multiplicity == 1) {
                // Single edge: straight line with arrow
                float arrow_len = config_.event_arrow_length;
                math::vec3 arrow_base = p2 - dir * arrow_len;
                generate_line(geo.event_lines, p1, arrow_base, color);
                generate_arrow_cone(geo.event_arrows, arrow_base, p2,
                                   config_.event_arrow_radius, color);
            } else {
                // Multiple edges: draw curved/bowed edges spread perpendicular
                // Spread edges symmetrically around the center line
                float spread = config_.state_size * 0.4f;  // Max spread at control point
                float endpoint_spread = config_.state_size * 0.15f;  // Spread at endpoints

                for (uint32_t i = 0; i < multiplicity; ++i) {
                    // Compute offset: centered around 0
                    // For n edges: positions at -spread, ..., 0, ..., +spread
                    float t = (static_cast<float>(i) / (multiplicity - 1) - 0.5f) * 2.0f;
                    float control_offset = t * spread;
                    float endpoint_offset = t * endpoint_spread;

                    // Offset endpoints perpendicular to edge direction
                    math::vec3 p1_offset = p1 + perp * endpoint_offset;
                    math::vec3 p2_offset = p2 + perp * endpoint_offset;

                    // Control point for quadratic bezier is at midpoint + perpendicular offset
                    math::vec3 midpoint = (p1_offset + p2_offset) * 0.5f;
                    math::vec3 control = midpoint + perp * control_offset;

                    // Generate curved edge as line segments
                    generate_quadratic_bezier(geo.event_lines, p1_offset, control, p2_offset,
                                             config_.event_arrow_length, color);

                    // Arrow at end of curve (pointing along final tangent)
                    math::vec3 final_tangent = p2_offset - control;
                    if (math::length(final_tangent) > 0.001f) {
                        final_tangent = math::normalize(final_tangent);
                    } else {
                        final_tangent = dir;
                    }
                    math::vec3 arrow_base = p2_offset - final_tangent * config_.event_arrow_length;
                    generate_arrow_cone(geo.event_arrows, arrow_base, p2_offset,
                                       config_.event_arrow_radius, color);
                }
            }
        }

        // Generate branchial edges (always curved for visual distinction)
        for (const auto& [key, edges] : branchial_bundles) {
            StateId source = key.first;
            StateId target = key.second;
            uint32_t multiplicity = static_cast<uint32_t>(edges.size());

            math::vec3 center1 = layout.get_state_pos(source);
            math::vec3 center2 = layout.get_state_pos(target);
            math::vec4 color = config_.branchial_edge_color;

            math::vec3 dir = center2 - center1;
            float len = math::length(dir);
            if (len < 0.001f) continue;
            dir = dir / len;

            math::vec3 p1 = ray_cube_exit(center1, dir, half_size);
            math::vec3 p2 = ray_cube_enter(center2, dir * -1.0f, half_size);

            // Branchial edges are always curved (bowed) for visual distinction
            // Use consistent perpendicular to ensure symmetric bowing
            math::vec3 perp = compute_consistent_perpendicular(p1, p2, math::vec3(0, 0, 1));

            // Equal spacing: edges spread from -spread to +spread
            // For single edge: center line (no bowing)
            // For multiple edges: equally spaced from -spread to +spread
            float spread = config_.state_size * 0.3f;  // Total spread from center
            float endpoint_spread = config_.state_size * 0.08f;

            for (uint32_t i = 0; i < multiplicity; ++i) {
                // t ranges from -1 to +1, equally spaced
                float t = (multiplicity == 1) ? 0.0f :
                    (static_cast<float>(i) / (multiplicity - 1) - 0.5f) * 2.0f;
                // Simple linear offset for equal spacing
                float control_offset = t * spread;
                float endpoint_offset = t * endpoint_spread;

                // Offset endpoints perpendicular to edge direction
                math::vec3 p1_offset = p1 + perp * endpoint_offset;
                math::vec3 p2_offset = p2 + perp * endpoint_offset;

                math::vec3 midpoint = (p1_offset + p2_offset) * 0.5f;
                math::vec3 control = midpoint + perp * control_offset;

                // No arrow for branchial (undirected)
                generate_quadratic_bezier_full(geo.branchial_lines, p1_offset, control, p2_offset, color);
            }
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
                    // This shows that an edge produced by the producer event was
                    // consumed by the consumer event to create a new state
                    StateId src_state = producer->output_state;
                    StateId tgt_state = consumer->output_state;

                    auto key = std::make_pair(src_state, tgt_state);
                    causal_bundles[key].push_back(&edge);
                }
            }
        }

        for (const auto& [key, edges] : causal_bundles) {
            StateId source = key.first;
            StateId target = key.second;
            uint32_t multiplicity = static_cast<uint32_t>(edges.size());

            math::vec3 center1 = layout.get_state_pos(source);
            math::vec3 center2 = layout.get_state_pos(target);
            math::vec4 color = config_.causal_edge_color;

            math::vec3 dir = center2 - center1;
            float len = math::length(dir);
            if (len < 0.001f) continue;
            dir = dir / len;

            math::vec3 p1 = ray_cube_exit(center1, dir, half_size);
            math::vec3 p2 = ray_cube_enter(center2, dir * -1.0f, half_size);

            // Causal edges are always curved (bowed) for visual distinction
            // Use consistent perpendicular with different up_hint than branchial to distinguish
            math::vec3 perp = compute_consistent_perpendicular(p1, p2, math::vec3(0, 1, 0));

            // Equal spacing: edges spread from -spread to +spread
            // For single edge: center line (no bowing)
            // For multiple edges: equally spaced from -spread to +spread
            float spread = config_.state_size * 0.25f;  // Total spread from center
            float endpoint_spread = config_.state_size * 0.06f;

            for (uint32_t i = 0; i < multiplicity; ++i) {
                // t ranges from -1 to +1, equally spaced
                float t = (multiplicity == 1) ? 0.0f :
                    (static_cast<float>(i) / (multiplicity - 1) - 0.5f) * 2.0f;
                // Simple linear offset for equal spacing
                float control_offset = t * spread;
                float endpoint_offset = t * endpoint_spread;

                // Offset endpoints perpendicular to edge direction
                math::vec3 p1_offset = p1 + perp * endpoint_offset;
                math::vec3 p2_offset = p2 + perp * endpoint_offset;

                math::vec3 midpoint = (p1_offset + p2_offset) * 0.5f;
                math::vec3 control = midpoint + perp * control_offset;

                // Causal edges have direction: draw with arrow
                generate_quadratic_bezier(geo.causal_lines, p1_offset, control, p2_offset,
                                         config_.event_arrow_length, color);

                // Arrow at end of curve (pointing along final tangent)
                math::vec3 final_tangent = p2_offset - control;
                if (math::length(final_tangent) > 0.001f) {
                    final_tangent = math::normalize(final_tangent);
                } else {
                    final_tangent = dir;
                }
                // Use smaller arrow for causal edges
                float causal_arrow_len = config_.event_arrow_length * 0.7f;
                float causal_arrow_rad = config_.event_arrow_radius * 0.7f;
                math::vec3 arrow_base = p2_offset - final_tangent * causal_arrow_len;
                generate_arrow_cone(geo.causal_lines, arrow_base, p2_offset,
                                   causal_arrow_rad, color);
            }
        }

        return geo;
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
