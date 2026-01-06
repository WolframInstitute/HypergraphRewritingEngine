#include "blackhole/particle_detection.hpp"
#include <algorithm>
#include <queue>
#include <cmath>
#include <sstream>
#include <numeric>

namespace viz::blackhole {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Find vertex index in graph's vertex list
size_t find_vertex_index(const SimpleGraph& graph, VertexId v) {
    const auto& vertices = graph.vertices();
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (vertices[i] == v) return i;
    }
    return SIZE_MAX;
}

// Compute centroid from vertex positions
Vec2 compute_centroid(
    const std::vector<VertexId>& vertices,
    const std::vector<Vec2>* positions,
    const SimpleGraph& graph
) {
    if (!positions || positions->empty()) {
        return Vec2{0, 0};
    }

    float sum_x = 0, sum_y = 0;
    int count = 0;

    for (VertexId v : vertices) {
        size_t idx = find_vertex_index(graph, v);
        if (idx < positions->size()) {
            sum_x += (*positions)[idx].x;
            sum_y += (*positions)[idx].y;
            count++;
        }
    }

    if (count == 0) return Vec2{0, 0};
    return Vec2{sum_x / count, sum_y / count};
}

// Compute radius of vertex set from centroid
float compute_radius(
    const std::vector<VertexId>& vertices,
    const Vec2& centroid,
    const std::vector<Vec2>* positions,
    const SimpleGraph& graph
) {
    if (!positions || positions->empty()) return 0.0f;

    float max_dist = 0;

    for (VertexId v : vertices) {
        size_t idx = find_vertex_index(graph, v);
        if (idx < positions->size()) {
            Vec2 diff = (*positions)[idx] - centroid;
            float dist = diff.length();
            if (dist > max_dist) max_dist = dist;
        }
    }

    return max_dist;
}

// Get local subgraph within radius of center vertex
std::vector<VertexId> get_local_vertices(
    const SimpleGraph& graph,
    VertexId center,
    int radius
) {
    std::vector<VertexId> result;
    auto distances = graph.distances_from(center);
    const auto& vertices = graph.vertices();

    for (size_t i = 0; i < vertices.size(); ++i) {
        if (distances[i] >= 0 && distances[i] <= radius) {
            result.push_back(vertices[i]);
        }
    }

    return result;
}

// Count edges within a vertex set
int count_edges_in_set(
    const SimpleGraph& graph,
    const std::vector<VertexId>& vertex_set
) {
    std::unordered_set<VertexId> set_lookup(vertex_set.begin(), vertex_set.end());
    int count = 0;

    for (VertexId v : vertex_set) {
        for (VertexId n : graph.neighbors(v)) {
            if (set_lookup.count(n) && v < n) {  // Count each edge once
                count++;
            }
        }
    }

    return count;
}

// Compute density of induced subgraph
float compute_subgraph_density(
    const SimpleGraph& graph,
    const std::vector<VertexId>& vertex_set
) {
    if (vertex_set.size() < 2) return 0.0f;

    int edges = count_edges_in_set(graph, vertex_set);
    int max_edges = static_cast<int>(vertex_set.size() * (vertex_set.size() - 1) / 2);

    return max_edges > 0 ? static_cast<float>(edges) / max_edges : 0.0f;
}

}  // anonymous namespace

// =============================================================================
// TopologicalDefect Methods
// =============================================================================

std::string TopologicalDefect::to_string() const {
    std::ostringstream ss;
    ss << "Defect{type=" << defect_type_name(type)
       << ", vertices=" << core_vertices.size()
       << ", charge=" << charge
       << ", dim=" << local_dimension
       << ", conf=" << detection_confidence << "%}";
    return ss.str();
}

// =============================================================================
// Clustering Coefficient
// =============================================================================

float compute_clustering_coefficient(
    const SimpleGraph& graph,
    VertexId vertex
) {
    const auto& neighbors = graph.neighbors(vertex);
    if (neighbors.size() < 2) return 0.0f;

    // Count edges between neighbors
    std::unordered_set<VertexId> neighbor_set(neighbors.begin(), neighbors.end());
    int triangles = 0;

    for (VertexId n1 : neighbors) {
        for (VertexId n2 : graph.neighbors(n1)) {
            if (neighbor_set.count(n2) && n2 > n1) {
                triangles++;
            }
        }
    }

    int possible = static_cast<int>(neighbors.size() * (neighbors.size() - 1) / 2);
    return possible > 0 ? static_cast<float>(triangles) / possible : 0.0f;
}

// =============================================================================
// Local Euler Characteristic
// =============================================================================

float compute_local_euler_characteristic(
    const SimpleGraph& graph,
    VertexId center,
    int radius
) {
    // Get local subgraph
    auto local_vertices = get_local_vertices(graph, center, radius);
    if (local_vertices.empty()) return 0.0f;

    int V = static_cast<int>(local_vertices.size());
    int E = count_edges_in_set(graph, local_vertices);

    // For planar graphs embedded in R^2: V - E + F = 2
    // Estimate F from Euler's formula for planar graph
    // If non-planar, χ will deviate from expected value

    // Simple estimate: assume roughly planar with triangular faces
    // F ≈ 2E/3 for triangular mesh
    // χ = V - E + F ≈ V - E + 2E/3 = V - E/3

    // Deviation from planarity is indicated by edge density exceeding 3V - 6
    int max_planar_edges = 3 * V - 6;
    if (V < 3) max_planar_edges = V * (V - 1) / 2;

    // Return normalized measure: 0 = planar, >0 = non-planar
    if (max_planar_edges <= 0) return 0.0f;

    float excess = static_cast<float>(E - max_planar_edges) / max_planar_edges;
    return std::max(0.0f, excess);
}

// =============================================================================
// Topological Charge Computation
// =============================================================================

std::vector<VertexCharge> compute_topological_charges(
    const SimpleGraph& graph,
    float radius
) {
    std::vector<VertexCharge> charges;
    const auto& vertices = graph.vertices();
    charges.reserve(vertices.size());

    // First pass: compute raw charges
    float max_charge = 0;

    for (VertexId v : vertices) {
        VertexCharge vc;
        vc.vertex = v;
        vc.degree = static_cast<int>(graph.neighbors(v).size());
        vc.clustering_coefficient = compute_clustering_coefficient(graph, v);

        // Charge based on:
        // 1. Degree (higher = more potential for non-planarity)
        // 2. Inverse clustering (low clustering with high degree = potential tangle)
        // 3. Local Euler deviation

        float euler_deviation = compute_local_euler_characteristic(graph, v, static_cast<int>(radius));

        // Charge formula: combines multiple factors
        float degree_factor = std::log2(std::max(1.0f, static_cast<float>(vc.degree)));
        float clustering_factor = 1.0f - vc.clustering_coefficient;
        float euler_factor = euler_deviation * 10.0f;  // Scale up Euler deviation

        vc.charge = degree_factor * clustering_factor + euler_factor;

        if (vc.charge > max_charge) max_charge = vc.charge;

        charges.push_back(vc);
    }

    // Normalize charges
    if (max_charge > 0) {
        for (auto& vc : charges) {
            vc.normalized_charge = vc.charge / max_charge;
        }
    }

    return charges;
}

std::unordered_map<VertexId, float> compute_charge_map(
    const SimpleGraph& graph,
    float radius
) {
    auto charges = compute_topological_charges(graph, radius);

    std::unordered_map<VertexId, float> result;
    for (const auto& vc : charges) {
        result[vc.vertex] = vc.charge;
    }

    return result;
}

// =============================================================================
// Clique Finding (for K5 detection)
// =============================================================================

std::vector<std::vector<VertexId>> find_cliques(
    const SimpleGraph& graph,
    int min_size,
    int max_cliques
) {
    std::vector<std::vector<VertexId>> cliques;
    const auto& vertices = graph.vertices();

    if (vertices.size() < static_cast<size_t>(min_size)) {
        return cliques;
    }

    // Simple greedy clique finding
    // For each vertex, try to extend clique greedily
    std::unordered_set<VertexId> used_starts;

    for (VertexId start : vertices) {
        if (static_cast<int>(cliques.size()) >= max_cliques) break;
        if (used_starts.count(start)) continue;

        std::vector<VertexId> clique;
        clique.push_back(start);

        std::unordered_set<VertexId> clique_set;
        clique_set.insert(start);

        // Get neighbors of start
        const auto& start_neighbors = graph.neighbors(start);
        std::unordered_set<VertexId> candidates(start_neighbors.begin(), start_neighbors.end());

        // Greedily add vertices that are connected to all clique members
        while (!candidates.empty()) {
            VertexId best = *candidates.begin();
            bool found = false;

            for (VertexId c : candidates) {
                // Check if c is connected to all clique members
                bool connected_to_all = true;
                for (VertexId m : clique) {
                    bool is_neighbor = false;
                    for (VertexId n : graph.neighbors(c)) {
                        if (n == m) {
                            is_neighbor = true;
                            break;
                        }
                    }
                    if (!is_neighbor) {
                        connected_to_all = false;
                        break;
                    }
                }

                if (connected_to_all) {
                    best = c;
                    found = true;
                    break;
                }
            }

            if (!found) break;

            clique.push_back(best);
            clique_set.insert(best);
            candidates.erase(best);

            // Intersect candidates with neighbors of best
            std::unordered_set<VertexId> new_candidates;
            for (VertexId n : graph.neighbors(best)) {
                if (candidates.count(n)) {
                    new_candidates.insert(n);
                }
            }
            candidates = new_candidates;
        }

        if (static_cast<int>(clique.size()) >= min_size) {
            cliques.push_back(clique);
            for (VertexId v : clique) {
                used_starts.insert(v);
            }
        }
    }

    return cliques;
}

// =============================================================================
// Dense Subgraph Finding
// =============================================================================

std::vector<std::vector<VertexId>> find_dense_subgraphs(
    const SimpleGraph& graph,
    float density_threshold,
    int min_size
) {
    std::vector<std::vector<VertexId>> dense_subgraphs;
    const auto& vertices = graph.vertices();

    // BFS from each vertex to find local dense regions
    std::unordered_set<VertexId> processed;

    for (VertexId start : vertices) {
        if (processed.count(start)) continue;

        // Grow region while density stays high
        std::vector<VertexId> region;
        region.push_back(start);

        std::queue<VertexId> frontier;
        for (VertexId n : graph.neighbors(start)) {
            frontier.push(n);
        }

        std::unordered_set<VertexId> in_region;
        in_region.insert(start);

        while (!frontier.empty() && region.size() < 50) {  // Cap size
            VertexId v = frontier.front();
            frontier.pop();

            if (in_region.count(v)) continue;

            // Check if adding v maintains density
            region.push_back(v);
            float new_density = compute_subgraph_density(graph, region);

            if (new_density >= density_threshold) {
                in_region.insert(v);
                for (VertexId n : graph.neighbors(v)) {
                    if (!in_region.count(n)) {
                        frontier.push(n);
                    }
                }
            } else {
                region.pop_back();
            }
        }

        if (static_cast<int>(region.size()) >= min_size) {
            float final_density = compute_subgraph_density(graph, region);
            if (final_density >= density_threshold) {
                dense_subgraphs.push_back(region);
                for (VertexId v : region) {
                    processed.insert(v);
                }
            }
        }
    }

    return dense_subgraphs;
}

// =============================================================================
// K5 Minor Detection
// =============================================================================

bool contains_k5_minor_approx(
    const SimpleGraph& graph,
    std::vector<VertexId>* minor_vertices
) {
    // Approximate K5 detection:
    // Look for 5+ vertices with high interconnection
    // True K5 minor detection is NP-hard, so we use heuristics

    auto cliques = find_cliques(graph, 4, 10);

    for (const auto& clique : cliques) {
        if (clique.size() >= 5) {
            if (minor_vertices) {
                *minor_vertices = std::vector<VertexId>(clique.begin(), clique.begin() + 5);
            }
            return true;
        }

        // For 4-cliques, check if there's a 5th vertex with high connectivity
        if (clique.size() == 4) {
            std::unordered_set<VertexId> clique_set(clique.begin(), clique.end());

            for (VertexId v : graph.vertices()) {
                if (clique_set.count(v)) continue;

                int connections = 0;
                for (VertexId c : clique) {
                    for (VertexId n : graph.neighbors(v)) {
                        if (n == c) {
                            connections++;
                            break;
                        }
                    }
                }

                // Need at least 3 connections to form K5-like structure
                if (connections >= 3) {
                    if (minor_vertices) {
                        *minor_vertices = clique;
                        minor_vertices->push_back(v);
                    }
                    return true;
                }
            }
        }
    }

    return false;
}

// =============================================================================
// K3,3 Minor Detection
// =============================================================================

bool contains_k33_minor_approx(
    const SimpleGraph& graph,
    std::vector<VertexId>* minor_vertices
) {
    // Approximate K3,3 detection:
    // Look for bipartite-like structure with 3+3 vertices heavily interconnected

    const auto& vertices = graph.vertices();
    if (vertices.size() < 6) return false;

    // For each pair of non-adjacent vertices, check if they could be K3,3 "poles"
    for (size_t i = 0; i < std::min(vertices.size(), size_t(100)); ++i) {
        for (size_t j = i + 1; j < std::min(vertices.size(), size_t(100)); ++j) {
            VertexId v1 = vertices[i];
            VertexId v2 = vertices[j];

            // Skip if adjacent
            bool adjacent = false;
            for (VertexId n : graph.neighbors(v1)) {
                if (n == v2) {
                    adjacent = true;
                    break;
                }
            }
            if (adjacent) continue;

            // Get neighbors of both
            const auto& n1 = graph.neighbors(v1);
            const auto& n2 = graph.neighbors(v2);

            // Find common neighbors (potential middle layer of K3,3)
            std::unordered_set<VertexId> n1_set(n1.begin(), n1.end());
            std::vector<VertexId> common;

            for (VertexId n : n2) {
                if (n1_set.count(n)) {
                    common.push_back(n);
                }
            }

            // K3,3 structure: 3 common neighbors connecting two "sides"
            if (common.size() >= 3) {
                // Check for additional structure
                // Need another pair of vertices on each side
                std::vector<VertexId> side1, side2;

                for (VertexId c : common) {
                    // Is c connected to other common neighbors?
                    int conn = 0;
                    for (VertexId c2 : common) {
                        if (c == c2) continue;
                        for (VertexId n : graph.neighbors(c)) {
                            if (n == c2) {
                                conn++;
                                break;
                            }
                        }
                    }
                    // Low interconnection among common = more K3,3-like
                    if (conn <= 1) {
                        if (side1.size() < 3) side1.push_back(c);
                    }
                }

                if (side1.size() >= 3) {
                    side2 = {v1, v2};
                    // Find third vertex for side2
                    for (VertexId v : vertices) {
                        if (v == v1 || v == v2) continue;
                        bool in_side1 = false;
                        for (VertexId s : side1) {
                            if (v == s) {
                                in_side1 = true;
                                break;
                            }
                        }
                        if (in_side1) continue;

                        // Check connections to side1
                        int conn = 0;
                        for (VertexId s : side1) {
                            for (VertexId n : graph.neighbors(v)) {
                                if (n == s) {
                                    conn++;
                                    break;
                                }
                            }
                        }
                        if (conn >= 2) {
                            side2.push_back(v);
                            break;
                        }
                    }

                    if (side2.size() >= 3) {
                        if (minor_vertices) {
                            minor_vertices->clear();
                            for (VertexId s : side1) minor_vertices->push_back(s);
                            for (VertexId s : side2) minor_vertices->push_back(s);
                        }
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

// =============================================================================
// K5 Minors Detection
// =============================================================================

std::vector<TopologicalDefect> detect_k5_minors(
    const SimpleGraph& graph,
    int max_search_depth
) {
    std::vector<TopologicalDefect> defects;

    // Find cliques >= 4 vertices
    auto cliques = find_cliques(graph, 4, 50);

    for (const auto& clique : cliques) {
        if (clique.size() >= 5) {
            TopologicalDefect defect;
            defect.type = TopologicalDefectType::K5Minor;
            defect.core_vertices = std::vector<VertexId>(clique.begin(), clique.begin() + std::min(size_t(5), clique.size()));
            defect.charge = static_cast<float>(clique.size()) / 5.0f;  // Normalized to K5
            defect.detection_confidence = 90;
            defects.push_back(defect);
        } else if (clique.size() == 4) {
            // Check for near-K5 (4-clique with highly connected 5th vertex)
            std::vector<VertexId> minor_verts;
            if (contains_k5_minor_approx(graph, &minor_verts)) {
                TopologicalDefect defect;
                defect.type = TopologicalDefectType::K5Minor;
                defect.core_vertices = minor_verts;
                defect.charge = 0.8f;  // Slightly lower confidence
                defect.detection_confidence = 70;
                defects.push_back(defect);
            }
        }
    }

    return defects;
}

// =============================================================================
// K3,3 Minors Detection
// =============================================================================

std::vector<TopologicalDefect> detect_k33_minors(
    const SimpleGraph& graph,
    int max_search_depth
) {
    std::vector<TopologicalDefect> defects;

    std::vector<VertexId> minor_verts;
    if (contains_k33_minor_approx(graph, &minor_verts)) {
        TopologicalDefect defect;
        defect.type = TopologicalDefectType::K33Minor;
        defect.core_vertices = minor_verts;
        defect.charge = 1.0f;
        defect.detection_confidence = 75;
        defects.push_back(defect);
    }

    return defects;
}

// =============================================================================
// Dimension Spike Detection
// =============================================================================

std::vector<TopologicalDefect> detect_dimension_spikes(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    float spike_threshold,
    const std::vector<Vec2>* vertex_positions
) {
    std::vector<TopologicalDefect> defects;
    const auto& vertices = graph.vertices();

    if (vertex_dimensions.empty()) return defects;

    // Compute mean dimension
    float sum = 0;
    int count = 0;
    for (float d : vertex_dimensions) {
        if (d > 0 && std::isfinite(d)) {
            sum += d;
            count++;
        }
    }
    if (count == 0) return defects;

    float mean_dim = sum / count;
    float threshold = mean_dim * spike_threshold;

    // Find vertices above threshold
    for (size_t i = 0; i < vertices.size() && i < vertex_dimensions.size(); ++i) {
        if (vertex_dimensions[i] > threshold) {
            TopologicalDefect defect;
            defect.type = TopologicalDefectType::DimensionSpike;
            defect.core_vertices.push_back(vertices[i]);
            defect.charge = (vertex_dimensions[i] - mean_dim) / mean_dim;
            defect.local_dimension = vertex_dimensions[i];
            defect.detection_confidence = 85;

            if (vertex_positions && i < vertex_positions->size()) {
                defect.centroid = (*vertex_positions)[i];
            }

            defects.push_back(defect);
        }
    }

    return defects;
}

// =============================================================================
// High Degree Detection
// =============================================================================

std::vector<TopologicalDefect> detect_high_degree_vertices(
    const SimpleGraph& graph,
    float degree_percentile,
    const std::vector<Vec2>* vertex_positions
) {
    std::vector<TopologicalDefect> defects;
    const auto& vertices = graph.vertices();

    // Compute degree distribution
    std::vector<int> degrees;
    degrees.reserve(vertices.size());

    for (VertexId v : vertices) {
        degrees.push_back(static_cast<int>(graph.neighbors(v).size()));
    }

    if (degrees.empty()) return defects;

    // Find percentile threshold
    std::vector<int> sorted_degrees = degrees;
    std::sort(sorted_degrees.begin(), sorted_degrees.end());

    size_t idx = static_cast<size_t>(degree_percentile * sorted_degrees.size());
    if (idx >= sorted_degrees.size()) idx = sorted_degrees.size() - 1;

    int degree_threshold = sorted_degrees[idx];

    // Find vertices above threshold
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (degrees[i] >= degree_threshold) {
            TopologicalDefect defect;
            defect.type = TopologicalDefectType::HighDegree;
            defect.core_vertices.push_back(vertices[i]);
            defect.charge = static_cast<float>(degrees[i]) / static_cast<float>(sorted_degrees.back());
            defect.detection_confidence = 80;

            if (vertex_positions && i < vertex_positions->size()) {
                defect.centroid = (*vertex_positions)[i];
            }

            defects.push_back(defect);
        }
    }

    return defects;
}

// =============================================================================
// Main Detection Function
// =============================================================================

std::vector<TopologicalDefect> detect_topological_defects(
    const SimpleGraph& graph,
    const ParticleDetectionConfig& config,
    const std::vector<float>* vertex_dimensions,
    const std::vector<Vec2>* vertex_positions
) {
    std::vector<TopologicalDefect> all_defects;

    // K5 detection
    if (config.detect_k5) {
        auto k5_defects = detect_k5_minors(graph, config.max_minor_search_depth);
        for (auto& d : k5_defects) {
            if (config.compute_centroids && vertex_positions) {
                d.centroid = compute_centroid(d.core_vertices, vertex_positions, graph);
                d.radius = compute_radius(d.core_vertices, d.centroid, vertex_positions, graph);
            }
            all_defects.push_back(std::move(d));
        }
    }

    // K3,3 detection
    if (config.detect_k33) {
        auto k33_defects = detect_k33_minors(graph, config.max_minor_search_depth);
        for (auto& d : k33_defects) {
            if (config.compute_centroids && vertex_positions) {
                d.centroid = compute_centroid(d.core_vertices, vertex_positions, graph);
                d.radius = compute_radius(d.core_vertices, d.centroid, vertex_positions, graph);
            }
            all_defects.push_back(std::move(d));
        }
    }

    // Dimension spike detection
    if (config.use_dimension_spikes && vertex_dimensions) {
        auto spike_defects = detect_dimension_spikes(
            graph, *vertex_dimensions, config.dimension_spike_threshold, vertex_positions
        );
        for (auto& d : spike_defects) {
            all_defects.push_back(std::move(d));
        }
    }

    // High degree detection
    if (config.use_high_degree) {
        auto degree_defects = detect_high_degree_vertices(
            graph, config.degree_threshold_percentile, vertex_positions
        );
        for (auto& d : degree_defects) {
            all_defects.push_back(std::move(d));
        }
    }

    return all_defects;
}

// =============================================================================
// Particle Tracking
// =============================================================================

std::vector<ParticleTrack> track_particles(
    const std::vector<std::vector<TopologicalDefect>>& defects_by_step,
    float matching_radius
) {
    std::vector<ParticleTrack> tracks;

    if (defects_by_step.empty()) return tracks;

    // Initialize tracks from first timestep
    for (const auto& defect : defects_by_step[0]) {
        ParticleTrack track;
        track.states.push_back(defect);
        track.timesteps.push_back(0);
        if (!defect.core_vertices.empty()) {
            track.initial_vertex = defect.core_vertices[0];
        }
        tracks.push_back(track);
    }

    // Match defects across timesteps
    for (size_t step = 1; step < defects_by_step.size(); ++step) {
        const auto& current_defects = defects_by_step[step];

        std::vector<bool> matched(current_defects.size(), false);

        // Try to match each track to a defect
        for (auto& track : tracks) {
            if (track.states.empty()) continue;

            const auto& last_defect = track.states.back();
            float best_dist = matching_radius;
            int best_idx = -1;

            for (size_t i = 0; i < current_defects.size(); ++i) {
                if (matched[i]) continue;

                // Check type match
                if (current_defects[i].type != last_defect.type) continue;

                // Compute distance between centroids
                Vec2 diff = current_defects[i].centroid - last_defect.centroid;
                float dist = diff.length();

                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = static_cast<int>(i);
                }
            }

            if (best_idx >= 0) {
                track.states.push_back(current_defects[best_idx]);
                track.timesteps.push_back(static_cast<uint32_t>(step));
                matched[best_idx] = true;
            }
        }

        // Create new tracks for unmatched defects
        for (size_t i = 0; i < current_defects.size(); ++i) {
            if (!matched[i]) {
                ParticleTrack track;
                track.states.push_back(current_defects[i]);
                track.timesteps.push_back(static_cast<uint32_t>(step));
                if (!current_defects[i].core_vertices.empty()) {
                    track.initial_vertex = current_defects[i].core_vertices[0];
                }
                tracks.push_back(track);
            }
        }
    }

    // Compute track statistics
    for (auto& track : tracks) {
        track.is_persistent = track.timesteps.size() > 1;

        float total_charge = 0;
        for (const auto& state : track.states) {
            total_charge += state.charge;
        }
        track.mean_charge = track.states.empty() ? 0 : total_charge / track.states.size();
    }

    return tracks;
}

// =============================================================================
// Full Analysis
// =============================================================================

ParticleAnalysisResult analyze_particles(
    const SimpleGraph& graph,
    const ParticleDetectionConfig& config,
    const std::vector<float>* vertex_dimensions,
    const std::vector<Vec2>* vertex_positions
) {
    ParticleAnalysisResult result;

    // Detect defects
    result.defects = detect_topological_defects(graph, config, vertex_dimensions, vertex_positions);

    // Count by type
    for (const auto& d : result.defects) {
        switch (d.type) {
            case TopologicalDefectType::K5Minor: result.num_k5++; break;
            case TopologicalDefectType::K33Minor: result.num_k33++; break;
            case TopologicalDefectType::DimensionSpike: result.num_dimension_spikes++; break;
            case TopologicalDefectType::HighDegree: result.num_high_degree++; break;
            default: break;
        }
    }

    // Compute charges
    if (config.compute_charges) {
        result.charges = compute_topological_charges(graph, config.charge_radius);

        for (const auto& vc : result.charges) {
            result.charge_map[vc.vertex] = vc.charge;
            result.total_charge += vc.charge;
            if (vc.charge > result.max_charge) {
                result.max_charge = vc.charge;
            }
        }

        if (!result.charges.empty()) {
            result.mean_charge = result.total_charge / result.charges.size();

            // Find high-charge vertices (above mean + stddev)
            float sum_sq = 0;
            for (const auto& vc : result.charges) {
                float diff = vc.charge - result.mean_charge;
                sum_sq += diff * diff;
            }
            float stddev = std::sqrt(sum_sq / result.charges.size());
            float threshold = result.mean_charge + stddev;

            for (const auto& vc : result.charges) {
                if (vc.charge > threshold) {
                    result.high_charge_vertices.push_back(vc.vertex);
                }
            }
        }
    }

    return result;
}

ParticleAnalysisResult analyze_particles_timestep(
    const TimestepAggregation& timestep,
    const ParticleDetectionConfig& config
) {
    // Build SimpleGraph from timestep union
    SimpleGraph graph;
    graph.build(timestep.union_vertices, timestep.union_edges);

    return analyze_particles(
        graph, config,
        &timestep.mean_dimensions,
        &timestep.vertex_positions
    );
}

}  // namespace viz::blackhole
