#include <blackhole/hausdorff_analysis.hpp>
#include <blackhole/curvature_analysis.hpp>
#include <job_system/job_system.hpp>
#ifdef BLACKHOLE_WITH_LAYOUT
#include <layout/layout_engine.hpp>
#endif
#include <queue>
#include <algorithm>
#include <set>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <execution>
#include <thread>
#include <chrono>
#include <atomic>

namespace viz::blackhole {

// =============================================================================
// Performance Timing Accumulators (thread-safe)
// =============================================================================
namespace {
    std::atomic<uint64_t> g_geodesic_time_us{0};
    std::atomic<uint64_t> g_dimension_time_us{0};
    std::atomic<size_t> g_dimension_calls{0};
    std::atomic<size_t> g_total_vertices_processed{0};

    // Aggregation phase timing
    std::atomic<uint64_t> g_union_build_time_us{0};
    std::atomic<uint64_t> g_lcc_time_us{0};
    std::atomic<uint64_t> g_bucketing_time_us{0};
}

void reset_analysis_timers() {
    g_geodesic_time_us = 0;
    g_dimension_time_us = 0;
    g_dimension_calls = 0;
    g_total_vertices_processed = 0;
    g_union_build_time_us = 0;
    g_lcc_time_us = 0;
    g_bucketing_time_us = 0;
}

void print_analysis_timing() {
    std::cout << "[TIMING] Geodesic coordinates: " << std::fixed << std::setprecision(1)
              << (g_geodesic_time_us.load() / 1000.0) << " ms" << std::endl;
    std::cout << "[TIMING] Dimension estimation: " << std::fixed << std::setprecision(1)
              << (g_dimension_time_us.load() / 1000.0) << " ms"
              << " (" << g_dimension_calls.load() << " calls, "
              << g_total_vertices_processed.load() << " total vertices)" << std::endl;
    std::cout << "[TIMING] Aggregation breakdown:" << std::endl;
    std::cout << "  - Union graph building: " << (g_union_build_time_us.load() / 1000.0) << " ms" << std::endl;
    std::cout << "  - LCC computation: " << (g_lcc_time_us.load() / 1000.0) << " ms" << std::endl;
    std::cout << "  - Coordinate bucketing: " << (g_bucketing_time_us.load() / 1000.0) << " ms" << std::endl;
}

// =============================================================================
// SimpleGraph Implementation
// =============================================================================

const std::vector<VertexId> SimpleGraph::empty_neighbors_{};

void SimpleGraph::build_from_edges(const std::vector<Edge>& edges) {
    adjacency_.clear();
    out_adjacency_.clear();
    vertex_to_index_.clear();

    // Deduplicate edges first using canonical ordering (for undirected)
    std::set<std::pair<VertexId, VertexId>> unique_edges;
    std::unordered_set<VertexId> vertex_set;
    for (const auto& e : edges) {
        vertex_set.insert(e.v1);
        vertex_set.insert(e.v2);
        VertexId v1 = std::min(e.v1, e.v2);
        VertexId v2 = std::max(e.v1, e.v2);
        unique_edges.insert({v1, v2});
    }

    edge_count_ = unique_edges.size();

    vertices_.assign(vertex_set.begin(), vertex_set.end());
    std::sort(vertices_.begin(), vertices_.end());

    for (size_t i = 0; i < vertices_.size(); ++i) {
        vertex_to_index_[vertices_[i]] = i;
        adjacency_[vertices_[i]] = {};  // Initialize empty adjacency
        out_adjacency_[vertices_[i]] = {};  // Initialize empty outgoing adjacency
    }

    // Build symmetric adjacency from deduplicated edges
    for (const auto& [v1, v2] : unique_edges) {
        adjacency_[v1].push_back(v2);
        adjacency_[v2].push_back(v1);
    }

    // Build directed (outgoing) adjacency from original edges
    // Deduplicate directed edges
    std::set<std::pair<VertexId, VertexId>> directed_edges;
    for (const auto& e : edges) {
        directed_edges.insert({e.v1, e.v2});
    }
    for (const auto& [v1, v2] : directed_edges) {
        out_adjacency_[v1].push_back(v2);
    }
}

void SimpleGraph::build(const std::vector<VertexId>& vertices, const std::vector<Edge>& edges) {
    vertices_ = vertices;
    adjacency_.clear();
    out_adjacency_.clear();
    vertex_to_index_.clear();

    for (size_t i = 0; i < vertices_.size(); ++i) {
        vertex_to_index_[vertices_[i]] = i;
        adjacency_[vertices_[i]] = {};  // Initialize empty adjacency
        out_adjacency_[vertices_[i]] = {};  // Initialize empty outgoing adjacency
    }

    // Use set to deduplicate edges, then build symmetric adjacency
    std::set<std::pair<VertexId, VertexId>> unique_edges;
    for (const auto& e : edges) {
        if (vertex_to_index_.count(e.v1) && vertex_to_index_.count(e.v2)) {
            VertexId v1 = std::min(e.v1, e.v2);
            VertexId v2 = std::max(e.v1, e.v2);
            unique_edges.insert({v1, v2});
        }
    }

    edge_count_ = unique_edges.size();
    for (const auto& [v1, v2] : unique_edges) {
        adjacency_[v1].push_back(v2);
        adjacency_[v2].push_back(v1);
    }

    // Build directed (outgoing) adjacency from original edges
    std::set<std::pair<VertexId, VertexId>> directed_edges;
    for (const auto& e : edges) {
        if (vertex_to_index_.count(e.v1) && vertex_to_index_.count(e.v2)) {
            directed_edges.insert({e.v1, e.v2});
        }
    }
    for (const auto& [v1, v2] : directed_edges) {
        out_adjacency_[v1].push_back(v2);
    }
}

const std::vector<VertexId>& SimpleGraph::neighbors(VertexId v) const {
    auto it = adjacency_.find(v);
    if (it != adjacency_.end()) {
        return it->second;
    }
    return empty_neighbors_;
}

const std::vector<VertexId>& SimpleGraph::out_neighbors(VertexId v) const {
    auto it = out_adjacency_.find(v);
    if (it != out_adjacency_.end()) {
        return it->second;
    }
    return empty_neighbors_;
}

bool SimpleGraph::has_vertex(VertexId v) const {
    return vertex_to_index_.count(v) > 0;
}

int SimpleGraph::distance(VertexId from, VertexId to) const {
    if (from == to) return 0;
    if (!has_vertex(from) || !has_vertex(to)) return -1;

    std::unordered_map<VertexId, int> dist;
    std::queue<VertexId> q;

    dist[from] = 0;
    q.push(from);

    while (!q.empty()) {
        VertexId curr = q.front();
        q.pop();

        for (VertexId next : neighbors(curr)) {
            if (dist.find(next) == dist.end()) {
                dist[next] = dist[curr] + 1;
                if (next == to) return dist[next];
                q.push(next);
            }
        }
    }

    return -1;  // Not reachable
}

std::vector<int> SimpleGraph::distances_from(VertexId source) const {
    std::vector<int> result(vertices_.size(), -1);

    if (!has_vertex(source)) return result;

    std::unordered_map<VertexId, int> dist;
    std::queue<VertexId> q;

    dist[source] = 0;
    q.push(source);

    while (!q.empty()) {
        VertexId curr = q.front();
        q.pop();

        for (VertexId next : neighbors(curr)) {
            if (dist.find(next) == dist.end()) {
                dist[next] = dist[curr] + 1;
                q.push(next);
            }
        }
    }

    // Map to result vector
    for (const auto& [v, d] : dist) {
        auto it = vertex_to_index_.find(v);
        if (it != vertex_to_index_.end()) {
            result[it->second] = d;
        }
    }

    return result;
}

// Truncated BFS - only explores up to max_radius, much faster for local dimension computation
std::vector<int> SimpleGraph::distances_from_truncated(VertexId source, int max_radius) const {
    std::vector<int> result(vertices_.size(), -1);

    if (!has_vertex(source)) return result;

    std::unordered_map<VertexId, int> dist;
    std::queue<VertexId> q;

    dist[source] = 0;
    q.push(source);

    while (!q.empty()) {
        VertexId curr = q.front();
        q.pop();

        int curr_dist = dist[curr];
        if (curr_dist >= max_radius) continue;  // Don't explore beyond max_radius

        for (VertexId next : neighbors(curr)) {
            if (dist.find(next) == dist.end()) {
                dist[next] = curr_dist + 1;
                q.push(next);
            }
        }
    }

    // Map to result vector
    for (const auto& [v, d] : dist) {
        auto it = vertex_to_index_.find(v);
        if (it != vertex_to_index_.end()) {
            result[it->second] = d;
        }
    }

    return result;
}

int SimpleGraph::eccentricity(VertexId v) const {
    // Eccentricity = max distance from v to any other vertex
    auto distances = distances_from(v);
    int max_dist = 0;
    for (int d : distances) {
        if (d > max_dist) max_dist = d;
    }
    return max_dist;
}

int SimpleGraph::graph_radius() const {
    // GraphRadius = minimum eccentricity over all vertices
    // This matches Mathematica's GraphRadius[graph]
    if (vertices_.empty()) return 0;

    int min_ecc = std::numeric_limits<int>::max();
    for (VertexId v : vertices_) {
        int ecc = eccentricity(v);
        if (ecc < min_ecc) min_ecc = ecc;
    }
    return min_ecc;
}

std::vector<std::vector<int>> SimpleGraph::all_pairs_distances() const {
    std::vector<std::vector<int>> result(vertices_.size());

    for (size_t i = 0; i < vertices_.size(); ++i) {
        result[i] = distances_from(vertices_[i]);
    }

    return result;
}

std::vector<std::vector<int>> SimpleGraph::all_pairs_distances_directed() const {
    std::vector<std::vector<int>> result(vertices_.size());

    // BFS from each vertex using only outgoing edges
    for (size_t src_idx = 0; src_idx < vertices_.size(); ++src_idx) {
        VertexId source = vertices_[src_idx];
        result[src_idx].assign(vertices_.size(), -1);

        std::unordered_map<VertexId, int> dist;
        std::queue<VertexId> q;

        dist[source] = 0;
        q.push(source);

        while (!q.empty()) {
            VertexId curr = q.front();
            q.pop();

            // Use outgoing neighbors only
            for (VertexId next : out_neighbors(curr)) {
                if (dist.find(next) == dist.end()) {
                    dist[next] = dist[curr] + 1;
                    q.push(next);
                }
            }
        }

        // Map to result vector
        for (const auto& [v, d] : dist) {
            auto it = vertex_to_index_.find(v);
            if (it != vertex_to_index_.end()) {
                result[src_idx][it->second] = d;
            }
        }
    }

    return result;
}

std::vector<std::vector<int>> SimpleGraph::all_pairs_distances_parallel(
    job_system::JobSystem<int>* js) const
{
    std::vector<std::vector<int>> result(vertices_.size());

    for (size_t i = 0; i < vertices_.size(); ++i) {
        js->submit_function([this, i, &result]() {
            result[i] = distances_from(vertices_[i]);
        }, 0);
    }

    js->wait_for_completion();
    return result;
}

std::vector<std::vector<int>> SimpleGraph::all_pairs_distances_directed_parallel(
    job_system::JobSystem<int>* js) const
{
    std::vector<std::vector<int>> result(vertices_.size());

    for (size_t src_idx = 0; src_idx < vertices_.size(); ++src_idx) {
        js->submit_function([this, src_idx, &result]() {
            VertexId source = vertices_[src_idx];
            result[src_idx].assign(vertices_.size(), -1);

            std::unordered_map<VertexId, int> dist;
            std::queue<VertexId> q;

            dist[source] = 0;
            q.push(source);

            while (!q.empty()) {
                VertexId curr = q.front();
                q.pop();

                for (VertexId next : out_neighbors(curr)) {
                    if (dist.find(next) == dist.end()) {
                        dist[next] = dist[curr] + 1;
                        q.push(next);
                    }
                }
            }

            for (const auto& [v, d] : dist) {
                auto it = vertex_to_index_.find(v);
                if (it != vertex_to_index_.end()) {
                    result[src_idx][it->second] = d;
                }
            }
        }, 0);
    }

    js->wait_for_completion();
    return result;
}

// =============================================================================
// Anchor Selection
// =============================================================================

std::vector<VertexId> find_stable_vertices(const std::vector<SimpleGraph>& graphs) {
    if (graphs.empty()) return {};

    // Start with vertices from first graph
    std::unordered_set<VertexId> stable(
        graphs[0].vertices().begin(),
        graphs[0].vertices().end()
    );

    // Intersect with all other graphs
    for (size_t i = 1; i < graphs.size(); ++i) {
        std::unordered_set<VertexId> current(
            graphs[i].vertices().begin(),
            graphs[i].vertices().end()
        );

        std::unordered_set<VertexId> intersection;
        for (VertexId v : stable) {
            if (current.count(v)) {
                intersection.insert(v);
            }
        }
        stable = std::move(intersection);
    }

    std::vector<VertexId> result(stable.begin(), stable.end());
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<VertexId> find_stable_vertices_nested(
    const std::vector<std::vector<SimpleGraph>>& states_by_step
) {
    // Find first non-empty graph to initialize
    const SimpleGraph* first_graph = nullptr;
    for (const auto& step_graphs : states_by_step) {
        if (!step_graphs.empty()) {
            first_graph = &step_graphs[0];
            break;
        }
    }
    if (!first_graph) return {};

    // Start with vertices from first graph
    std::unordered_set<VertexId> stable(
        first_graph->vertices().begin(),
        first_graph->vertices().end()
    );

    // Intersect with all graphs (iterate without copying)
    for (const auto& step_graphs : states_by_step) {
        for (const auto& g : step_graphs) {
            if (&g == first_graph) continue;  // Skip first graph

            std::unordered_set<VertexId> intersection;
            for (VertexId v : stable) {
                if (g.has_vertex(v)) {
                    intersection.insert(v);
                }
            }
            stable = std::move(intersection);

            if (stable.empty()) break;  // Early exit
        }
        if (stable.empty()) break;
    }

    std::vector<VertexId> result(stable.begin(), stable.end());
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<VertexId> select_anchors(
    const SimpleGraph& graph,
    const std::vector<VertexId>& candidates,
    int k,
    int min_separation
) {
    if (candidates.empty()) return {};
    if (static_cast<int>(candidates.size()) <= k) return candidates;

    // Precompute distances between all candidates
    std::unordered_map<std::pair<VertexId, VertexId>, int,
        decltype([](const std::pair<VertexId, VertexId>& p) {
            return std::hash<VertexId>{}(p.first) ^ (std::hash<VertexId>{}(p.second) << 1);
        })> dist_cache;

    auto get_dist = [&](VertexId a, VertexId b) -> int {
        if (a > b) std::swap(a, b);
        auto key = std::make_pair(a, b);
        auto it = dist_cache.find(key);
        if (it != dist_cache.end()) return it->second;

        int d = graph.distance(a, b);
        dist_cache[key] = d;
        return d;
    };

    // Precompute some distances for finding a good starting vertex
    // Pick vertex with highest total distance to others (peripheral)
    VertexId best_start = candidates[0];
    int best_total = 0;

    for (VertexId v : candidates) {
        int total = 0;
        for (VertexId u : candidates) {
            if (u != v) {
                int d = get_dist(v, u);
                if (d > 0) total += d;
            }
        }
        if (total > best_total) {
            best_total = total;
            best_start = v;
        }
    }

    std::vector<VertexId> selected = {best_start};
    std::unordered_set<VertexId> remaining(candidates.begin(), candidates.end());
    remaining.erase(best_start);

    // Greedily add vertices that maximize minimum distance to selected set
    while (static_cast<int>(selected.size()) < k && !remaining.empty()) {
        VertexId best_next = 0;
        int best_min_dist = -1;

        for (VertexId v : remaining) {
            int min_dist = std::numeric_limits<int>::max();
            for (VertexId s : selected) {
                int d = get_dist(v, s);
                if (d >= 0 && d < min_dist) {
                    min_dist = d;
                }
            }

            if (min_dist >= min_separation && min_dist > best_min_dist) {
                best_min_dist = min_dist;
                best_next = v;
            }
        }

        if (best_min_dist < 0) break;  // No valid candidate found

        selected.push_back(best_next);
        remaining.erase(best_next);
    }

    return selected;
}

// =============================================================================
// Geodesic Coordinates
// =============================================================================

// Version that uses pre-computed distance matrix (O(V*A) lookups, no BFS)
std::unordered_map<VertexId, std::vector<int>> compute_geodesic_coordinates_from_matrix(
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors,
    const std::vector<std::vector<int>>& dist_matrix
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::unordered_map<VertexId, std::vector<int>> result;
    const auto& verts = graph.vertices();

    // Build anchor index lookup
    std::vector<size_t> anchor_indices(anchors.size());
    for (size_t a = 0; a < anchors.size(); ++a) {
        // Find anchor's index in vertex list
        for (size_t i = 0; i < verts.size(); ++i) {
            if (verts[i] == anchors[a]) {
                anchor_indices[a] = i;
                break;
            }
        }
    }

    // Build coordinate vectors for each vertex using matrix lookups
    for (size_t i = 0; i < verts.size(); ++i) {
        VertexId v = verts[i];
        std::vector<int> coords(anchors.size(), -1);
        bool valid = true;

        for (size_t a = 0; a < anchors.size(); ++a) {
            int dist = dist_matrix[i][anchor_indices[a]];
            if (dist >= 0) {
                coords[a] = dist;
            } else {
                valid = false;
                break;
            }
        }

        if (valid) {
            result[v] = std::move(coords);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_geodesic_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return result;
}

// Original version that computes BFS per anchor (for backward compatibility)
std::unordered_map<VertexId, std::vector<int>> compute_geodesic_coordinates(
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::unordered_map<VertexId, std::vector<int>> result;

    // For each anchor, compute distances to all vertices
    std::vector<std::unordered_map<VertexId, int>> anchor_distances(anchors.size());

    for (size_t a = 0; a < anchors.size(); ++a) {
        auto dists = graph.distances_from(anchors[a]);
        const auto& verts = graph.vertices();
        for (size_t i = 0; i < verts.size(); ++i) {
            if (dists[i] >= 0) {
                anchor_distances[a][verts[i]] = dists[i];
            }
        }
    }

    // Build coordinate vectors for each vertex
    for (VertexId v : graph.vertices()) {
        std::vector<int> coords(anchors.size(), -1);
        bool valid = true;

        for (size_t a = 0; a < anchors.size(); ++a) {
            auto it = anchor_distances[a].find(v);
            if (it != anchor_distances[a].end()) {
                coords[a] = it->second;
            } else {
                valid = false;
                break;
            }
        }

        if (valid) {
            result[v] = std::move(coords);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_geodesic_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return result;
}

// =============================================================================
// Local Hausdorff Dimension
// =============================================================================

// Debug: print ball growth for first few calls (currently unused)
// static int debug_dimension_calls = 0;
// static constexpr int DEBUG_DIMENSION_LIMIT = 5;

float estimate_local_dimension(
    const std::vector<int>& distances_from_vertex,
    int max_radius
) {
    // Count vertices at each distance (cumulative ball size)
    // counts[r] = |B(v, r)| = number of vertices within distance r
    std::vector<int> counts(max_radius + 1, 0);
    for (int d : distances_from_vertex) {
        if (d >= 0 && d <= max_radius) {
            for (int r = d; r <= max_radius; ++r) {
                counts[r]++;
            }
        }
    }

    // Use ResourceFunction's discrete derivative formula:
    // d_r = (Log[V(r)] - Log[V(r-1)]) / (Log[r+1] - Log[r])
    // This measures the local growth rate at each radius.
    std::vector<float> dimensions;
    dimensions.reserve(max_radius);

    for (int r = 1; r <= max_radius; ++r) {
        if (counts[r] > 0 && counts[r-1] > 0) {
            float log_v_r = std::log(static_cast<float>(counts[r]));
            float log_v_r_minus_1 = std::log(static_cast<float>(counts[r-1]));
            float log_r_plus_1 = std::log(static_cast<float>(r + 1));
            float log_r = std::log(static_cast<float>(r));

            float denom = log_r_plus_1 - log_r;
            if (std::abs(denom) > 1e-10f) {
                float d_r = (log_v_r - log_v_r_minus_1) / denom;
                if (std::isfinite(d_r)) {
                    dimensions.push_back(d_r);
                }
            }
        }
    }

    if (dimensions.empty()) {
        return -1.0f;  // Insufficient data
    }

    // Return mean dimension (matching ResourceFunction's default DimensionMethod -> Mean)
    float sum = 0.0f;
    for (float d : dimensions) {
        sum += d;
    }
    return sum / static_cast<float>(dimensions.size());
}

// Version that uses pre-computed distance matrix (O(V * max_radius) scans, no BFS)
std::vector<float> estimate_all_dimensions_from_matrix(
    const SimpleGraph& graph,
    const std::vector<std::vector<int>>& dist_matrix,
    int max_radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<float> dimensions(graph.vertex_count());

    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        dimensions[i] = estimate_local_dimension(dist_matrix[i], max_radius);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_dimension_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    g_dimension_calls++;

    return dimensions;
}

// Version that uses truncated BFS per vertex - O(V * neighborhood) instead of O(V * V)
// Much faster when max_radius << graph diameter
std::vector<float> estimate_all_dimensions_truncated(
    const SimpleGraph& graph,
    int max_radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<float> dimensions(graph.vertex_count());
    const auto& vertices = graph.vertices();

    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        auto truncated_dists = graph.distances_from_truncated(vertices[i], max_radius);
        dimensions[i] = estimate_local_dimension(truncated_dists, max_radius);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_dimension_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    g_dimension_calls++;
    g_total_vertices_processed += graph.vertex_count();

    return dimensions;
}

// Original version that computes BFS per vertex (for backward compatibility)
std::vector<float> estimate_all_dimensions(
    const SimpleGraph& graph,
    int max_radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute distance matrix once - O(V * (V+E)) total
    auto dist_matrix = graph.all_pairs_distances();

    std::vector<float> dimensions(graph.vertex_count());
    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        dimensions[i] = estimate_local_dimension(dist_matrix[i], max_radius);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_dimension_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    g_dimension_calls++;
    g_total_vertices_processed += graph.vertex_count();

    return dimensions;
}

// Parallel version using job system - computes distance matrix in parallel
std::vector<float> estimate_all_dimensions_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int max_radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute distance matrix in parallel
    auto dist_matrix = graph.all_pairs_distances_parallel(js);

    // Dimension estimation from matrix is fast (O(V * max_radius)), can stay sequential
    std::vector<float> dimensions(graph.vertex_count());
    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        dimensions[i] = estimate_local_dimension(dist_matrix[i], max_radius);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_dimension_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    g_dimension_calls++;
    g_total_vertices_processed += graph.vertex_count();

    return dimensions;
}

// Parallel version using truncated BFS - each vertex's BFS runs in parallel
std::vector<float> estimate_all_dimensions_truncated_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int max_radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<float> dimensions(graph.vertex_count());
    const auto& vertices = graph.vertices();

    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        js->submit_function([&graph, &vertices, &dimensions, i, max_radius]() {
            auto truncated_dists = graph.distances_from_truncated(vertices[i], max_radius);
            dimensions[i] = estimate_local_dimension(truncated_dists, max_radius);
        }, 0);
    }

    js->wait_for_completion();

    auto end_time = std::chrono::high_resolution_clock::now();
    g_dimension_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    g_dimension_calls++;
    g_total_vertices_processed += graph.vertex_count();

    return dimensions;
}

// Parallel version with full config - supports directed graphs and different formulas
std::vector<float> estimate_all_dimensions_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const DimensionConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use directed or undirected distances based on config (computed in parallel)
    auto dist_matrix = config.directed
        ? graph.all_pairs_distances_directed_parallel(js)
        : graph.all_pairs_distances_parallel(js);

    // Dimension estimation from matrix is fast, stays sequential
    std::vector<float> dimensions(graph.vertex_count());
    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        dimensions[i] = estimate_local_dimension(dist_matrix[i], config);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_dimension_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    g_dimension_calls++;
    g_total_vertices_processed += graph.vertex_count();

    return dimensions;
}

// Compute statistics from a vector of dimension values
DimensionStats compute_dimension_stats(const std::vector<float>& values) {
    DimensionStats stats;
    if (values.empty()) return stats;

    // Filter valid values
    std::vector<float> valid;
    valid.reserve(values.size());
    for (float v : values) {
        if (v > 0 && std::isfinite(v)) {
            valid.push_back(v);
        }
    }

    if (valid.empty()) return stats;

    stats.count = valid.size();
    stats.min = *std::min_element(valid.begin(), valid.end());
    stats.max = *std::max_element(valid.begin(), valid.end());

    // Mean
    float sum = 0;
    for (float v : valid) sum += v;
    stats.mean = sum / static_cast<float>(valid.size());

    // Variance and stddev
    float sum_sq = 0;
    for (float v : valid) {
        float diff = v - stats.mean;
        sum_sq += diff * diff;
    }
    stats.variance = sum_sq / static_cast<float>(valid.size());
    stats.stddev = std::sqrt(stats.variance);

    return stats;
}

// Estimate local dimension using discrete derivative formula
// (log N(r) - log N(r-1)) / (log(r+1) - log(r))
static float estimate_local_dimension_discrete_derivative(
    const std::vector<int>& distances_from_vertex,
    const DimensionConfig& config
) {
    // Count total reachable vertices
    int total_reachable = 0;
    for (int d : distances_from_vertex) {
        if (d >= 0) total_reachable++;
    }

    // Count vertices at each distance (cumulative ball size)
    int actual_max_radius = std::min(config.max_radius, static_cast<int>(distances_from_vertex.size()));
    std::vector<int> counts(actual_max_radius + 1, 0);
    for (int d : distances_from_vertex) {
        if (d >= 0 && d <= actual_max_radius) {
            for (int r = d; r <= actual_max_radius; ++r) {
                counts[r]++;
            }
        }
    }

    // Compute dimension at each radius using discrete derivative
    float saturation_threshold = config.saturation_threshold * total_reachable;
    std::vector<float> per_radius_dims;

    for (int r = config.min_radius; r <= actual_max_radius; ++r) {
        // Skip if saturated
        if (counts[r] >= saturation_threshold) continue;

        // Need N(r) and N(r-1) both > 0
        if (counts[r] <= 0 || counts[r-1] <= 0) continue;

        // Discrete derivative: (log N(r) - log N(r-1)) / (log(r+1) - log(r))
        float log_N_r = std::log(static_cast<float>(counts[r]));
        float log_N_r1 = std::log(static_cast<float>(counts[r-1]));
        float log_r_plus_1 = std::log(static_cast<float>(r + 1));
        float log_r = std::log(static_cast<float>(r));

        float denom = log_r_plus_1 - log_r;
        if (std::abs(denom) < 1e-10f) continue;

        float dim = (log_N_r - log_N_r1) / denom;
        if (std::isfinite(dim) && dim > 0) {
            per_radius_dims.push_back(dim);
        }
    }

    if (per_radius_dims.empty()) return -1.0f;

    // Aggregate based on method
    switch (config.aggregation) {
        case AggregationMethod::Min:
            return *std::min_element(per_radius_dims.begin(), per_radius_dims.end());
        case AggregationMethod::Max:
            return *std::max_element(per_radius_dims.begin(), per_radius_dims.end());
        case AggregationMethod::Mean:
        default: {
            float sum = 0;
            for (float d : per_radius_dims) sum += d;
            return sum / static_cast<float>(per_radius_dims.size());
        }
    }
}

// Estimate local dimension with full configuration
float estimate_local_dimension(
    const std::vector<int>& distances_from_vertex,
    const DimensionConfig& config
) {
    if (config.formula == DimensionFormula::DiscreteDerivative) {
        return estimate_local_dimension_discrete_derivative(distances_from_vertex, config);
    }

    // Linear regression (default) - same as original but with configurable saturation
    int total_reachable = 0;
    for (int d : distances_from_vertex) {
        if (d >= 0) total_reachable++;
    }

    int actual_max_radius = std::min(config.max_radius, static_cast<int>(distances_from_vertex.size()));
    std::vector<int> counts(actual_max_radius + 1, 0);
    for (int d : distances_from_vertex) {
        if (d >= 0 && d <= actual_max_radius) {
            for (int r = d; r <= actual_max_radius; ++r) {
                counts[r]++;
            }
        }
    }

    // Build (log r, log N) pairs for regression
    float saturation_threshold = config.saturation_threshold * total_reachable;
    std::vector<std::pair<float, float>> valid_pairs;
    for (int r = config.min_radius; r <= actual_max_radius; ++r) {
        if (counts[r] > 1 && counts[r] < saturation_threshold) {
            valid_pairs.emplace_back(std::log(static_cast<float>(r)),
                                     std::log(static_cast<float>(counts[r])));
        }
    }

    if (valid_pairs.size() < 2) {
        return -1.0f;
    }

    // Linear regression
    float n = static_cast<float>(valid_pairs.size());
    float sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;

    for (const auto& [x, y] : valid_pairs) {
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    float denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-10f) {
        return -1.0f;
    }

    float slope = (n * sum_xy - sum_x * sum_y) / denom;
    return std::isfinite(slope) ? slope : -1.0f;
}

// Estimate all dimensions with full configuration
std::vector<float> estimate_all_dimensions(
    const SimpleGraph& graph,
    const DimensionConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use directed or undirected distances based on config
    auto dist_matrix = config.directed
        ? graph.all_pairs_distances_directed()
        : graph.all_pairs_distances();

    std::vector<float> dimensions(graph.vertex_count());
    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        dimensions[i] = estimate_local_dimension(dist_matrix[i], config);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    g_dimension_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    g_dimension_calls++;
    g_total_vertices_processed += graph.vertex_count();

    return dimensions;
}

// =============================================================================
// State Analysis
// =============================================================================

LightweightAnalysis analyze_state_lightweight(
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors,
    int max_radius
) {
    LightweightAnalysis result;
    result.num_anchors = static_cast<int>(anchors.size());

    // Compute distance matrix ONCE - O(V × (V+E)) BFS traversals
    // This is reused for coords, dimensions, AND curvature (avoids redundant BFS)
    auto dist_matrix = graph.all_pairs_distances();

    // Compute geodesic coordinates from matrix - O(V × A) lookups
    auto coords = compute_geodesic_coordinates_from_matrix(graph, anchors, dist_matrix);

    // Compute dimensions from matrix - O(V × max_radius) scans
    auto dimensions = estimate_all_dimensions_from_matrix(graph, dist_matrix, max_radius);

    // Build vertex data arrays
    const auto& verts = graph.vertices();
    result.vertex_dimensions.resize(graph.vertex_count());
    result.vertex_coords.resize(graph.vertex_count());

    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        VertexId v = verts[i];
        result.vertex_dimensions[i] = dimensions[i];

        // Geodesic coords
        auto it = coords.find(v);
        if (it != coords.end()) {
            for (size_t a = 0; a < it->second.size() && a < MAX_ANCHORS; ++a) {
                result.vertex_coords[i][a] = it->second[a];
            }
        } else {
            std::fill(result.vertex_coords[i].begin(), result.vertex_coords[i].end(), -1);
        }
    }

    // Compute all curvature types per-state for Branchial aggregation
    CurvatureConfig curv_config;
    curv_config.compute_ollivier_ricci = true;
    curv_config.compute_wolfram_ricci = true;
    curv_config.compute_wolfram_scalar = true;
    curv_config.compute_dimension_gradient = true;
    curv_config.max_radius = max_radius;

    CurvatureAnalysisResult curv_result = analyze_curvature(graph, curv_config, &dimensions);

    // Store curvature maps in result
    result.curvature_ollivier = std::move(curv_result.ollivier_ricci_map);
    result.curvature_wolfram_scalar = std::move(curv_result.wolfram_scalar_map);
    result.curvature_wolfram_ricci = std::move(curv_result.wolfram_ricci_map);
    result.curvature_dim_gradient = std::move(curv_result.dimension_gradient_map);

    return result;
}

StateAnalysis analyze_state(
    StateId state_id,
    uint32_t step,
    const SimpleGraph& graph,
    const std::vector<VertexId>& anchors,
    int max_radius
) {
    StateAnalysis result;
    result.state_id = state_id;
    result.step = step;
    result.vertices = graph.vertices();
    result.num_anchors = static_cast<int>(anchors.size());

    // Build edges list (from adjacency)
    std::unordered_set<uint64_t> seen_edges;
    for (VertexId v : graph.vertices()) {
        for (VertexId u : graph.neighbors(v)) {
            uint64_t key = (static_cast<uint64_t>(std::min(v, u)) << 32) | std::max(v, u);
            if (seen_edges.find(key) == seen_edges.end()) {
                seen_edges.insert(key);
                result.edges.push_back({std::min(v, u), std::max(v, u)});
            }
        }
    }

    // Compute distance matrix ONCE - reused for both geodesic coords and dimensions
    auto dist_matrix = graph.all_pairs_distances();

    // Compute geodesic coordinates from matrix
    auto coords = compute_geodesic_coordinates_from_matrix(graph, anchors, dist_matrix);

    // Compute dimensions from matrix
    auto dimensions = estimate_all_dimensions_from_matrix(graph, dist_matrix, max_radius);

    // Build vertex data arrays
    result.vertex_dimensions.resize(graph.vertex_count());
    result.vertex_coords.resize(graph.vertex_count());

    float sum_dim = 0;
    int count_valid = 0;
    result.min_dimension = std::numeric_limits<float>::max();
    result.max_dimension = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < graph.vertex_count(); ++i) {
        VertexId v = result.vertices[i];

        result.vertex_dimensions[i] = dimensions[i];

        if (dimensions[i] > 0) {
            sum_dim += dimensions[i];
            count_valid++;
            result.min_dimension = std::min(result.min_dimension, dimensions[i]);
            result.max_dimension = std::max(result.max_dimension, dimensions[i]);
        }

        // Geodesic coords
        auto it = coords.find(v);
        if (it != coords.end()) {
            for (size_t a = 0; a < it->second.size() && a < MAX_ANCHORS; ++a) {
                result.vertex_coords[i][a] = it->second[a];
            }
        } else {
            std::fill(result.vertex_coords[i].begin(), result.vertex_coords[i].end(), -1);
        }
    }

    result.mean_dimension = count_valid > 0 ? sum_dim / count_valid : 0;

    return result;
}

// =============================================================================
// Largest Connected Component
// =============================================================================

// Find the largest connected component in a graph defined by vertices and edges
// Returns the set of vertices in the largest component
static std::unordered_set<VertexId> find_largest_connected_component(
    const std::vector<VertexId>& vertices,
    const std::vector<Edge>& edges
) {
    if (vertices.empty()) return {};

    // Build adjacency list
    std::unordered_map<VertexId, std::vector<VertexId>> adj;
    for (VertexId v : vertices) {
        adj[v] = {};  // Ensure all vertices are in the map
    }
    for (const auto& e : edges) {
        adj[e.v1].push_back(e.v2);
        adj[e.v2].push_back(e.v1);
    }

    // BFS to find connected components
    std::unordered_set<VertexId> visited;
    std::unordered_set<VertexId> largest_component;

    for (VertexId start : vertices) {
        if (visited.count(start)) continue;

        // BFS from this vertex
        std::unordered_set<VertexId> component;
        std::queue<VertexId> q;
        q.push(start);
        visited.insert(start);
        component.insert(start);

        while (!q.empty()) {
            VertexId v = q.front();
            q.pop();

            for (VertexId neighbor : adj[v]) {
                if (!visited.count(neighbor)) {
                    visited.insert(neighbor);
                    component.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }

        if (component.size() > largest_component.size()) {
            largest_component = std::move(component);
        }
    }

    return largest_component;
}

// =============================================================================
// Timestep Aggregation
// =============================================================================

TimestepAggregation aggregate_timestep(
    uint32_t step,
    const std::vector<StateAnalysis>& states_at_step,
    const std::vector<Vec2>& initial_positions
) {
    TimestepAggregation result;
    result.step = step;
    result.num_states = static_cast<int>(states_at_step.size());

    if (states_at_step.empty()) return result;

    // Union of all vertices and edges, plus count for intersection
    std::unordered_set<VertexId> union_vertex_set;
    std::unordered_map<uint64_t, int> edge_counts;  // Count how many states contain each edge (by vertex pair)

    for (const auto& state : states_at_step) {
        for (VertexId v : state.vertices) {
            union_vertex_set.insert(v);
        }
        // Deduplicate edges by vertex pair within each state
        // This ensures edge multiplicity doesn't inflate the frequency count
        std::unordered_set<uint64_t> seen_pairs_this_state;
        for (const auto& e : state.edges) {
            uint64_t key = (static_cast<uint64_t>(std::min(e.v1, e.v2)) << 32) | std::max(e.v1, e.v2);
            if (seen_pairs_this_state.insert(key).second) {
                edge_counts[key]++;
            }
        }
    }

    // Union = all edges that appear in at least one state
    std::unordered_set<uint64_t> union_edge_set;
    for (const auto& [key, count] : edge_counts) {
        union_edge_set.insert(key);
    }

    // Convert to vectors temporarily
    std::vector<VertexId> all_vertices(union_vertex_set.begin(), union_vertex_set.end());
    std::vector<Edge> all_edges;
    for (uint64_t key : union_edge_set) {
        VertexId v1 = static_cast<VertexId>(key >> 32);
        VertexId v2 = static_cast<VertexId>(key & 0xFFFFFFFF);
        all_edges.push_back({v1, v2});
    }

    // Filter to largest connected component
    auto lcc = find_largest_connected_component(all_vertices, all_edges);

    // Keep only vertices in LCC
    for (VertexId v : all_vertices) {
        if (lcc.count(v)) {
            result.union_vertices.push_back(v);
        }
    }
    std::sort(result.union_vertices.begin(), result.union_vertices.end());

    // Keep only edges where both endpoints are in LCC
    for (const auto& e : all_edges) {
        if (lcc.count(e.v1) && lcc.count(e.v2)) {
            result.union_edges.push_back(e);
        }
    }

    // Intersection = edges present in ALL states at this step
    // Frequent = edges present in MORE than one state (excludes singletons)
    int num_states = static_cast<int>(states_at_step.size());
    for (const auto& [key, count] : edge_counts) {
        VertexId v1 = static_cast<VertexId>(key >> 32);
        VertexId v2 = static_cast<VertexId>(key & 0xFFFFFFFF);
        bool in_lcc = lcc.count(v1) && lcc.count(v2);

        if (count == num_states && in_lcc) {
            result.intersection_edges.push_back({v1, v2});
        }

        if (count > 1 && in_lcc) {
            result.frequent_edges.push_back({v1, v2});
        }
    }

    // Build vertex positions from initial condition
    result.vertex_positions.resize(result.union_vertices.size());
    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];
        if (v < initial_positions.size()) {
            result.vertex_positions[i] = initial_positions[v];
        } else {
            // For new vertices created during evolution, place at origin
            // (in real usage, we'd need a layout algorithm)
            result.vertex_positions[i] = {0, 0};
        }
    }

    // Build SimpleGraph from union (LCC) for coordinate computation
    SimpleGraph union_graph;
    union_graph.build(result.union_vertices, result.union_edges);

    // Compute distance matrix for union graph
    auto union_dist_matrix = union_graph.all_pairs_distances();

    // Select anchors from union graph (well-separated vertices)
    auto anchors = select_anchors(union_graph, result.union_vertices, 6, 3);
    int num_anchors = static_cast<int>(anchors.size());

    // Compute geodesic coordinates from pre-computed matrix
    auto union_coords = compute_geodesic_coordinates_from_matrix(union_graph, anchors, union_dist_matrix);

    // Build vertex index map for union
    std::unordered_map<VertexId, size_t> union_vertex_idx;
    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        union_vertex_idx[result.union_vertices[i]] = i;
    }

    // Aggregate dimensions by coordinate bucket
    // Use coordinates from union graph, dimensions from individual states
    std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> coord_dims;

    for (const auto& state : states_at_step) {
        for (size_t i = 0; i < state.vertices.size(); ++i) {
            VertexId v = state.vertices[i];

            // Only consider vertices in the LCC
            if (!lcc.count(v)) continue;
            if (state.vertex_dimensions[i] <= 0) continue;

            // Get coordinate from union graph (always valid for LCC vertices)
            auto coord_it = union_coords.find(v);
            if (coord_it == union_coords.end()) continue;

            CoordKey key;
            key.num_anchors = num_anchors;
            for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
                key.coords[a] = coord_it->second[a];
            }

            coord_dims[key].push_back(state.vertex_dimensions[i]);
        }
    }

    // Compute mean dimension per coordinate
    for (auto& [key, dims] : coord_dims) {
        float mean = std::accumulate(dims.begin(), dims.end(), 0.0f) / dims.size();
        result.coord_to_dim[key] = mean;
    }

    // Assign dimensions to union vertices via coordinate lookup
    result.mean_dimensions.resize(result.union_vertices.size(), -1);

    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];

        auto coord_it = union_coords.find(v);
        if (coord_it == union_coords.end()) continue;

        CoordKey key;
        key.num_anchors = num_anchors;
        for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
            key.coords[a] = coord_it->second[a];
        }

        auto dim_it = result.coord_to_dim.find(key);
        if (dim_it != result.coord_to_dim.end()) {
            result.mean_dimensions[i] = dim_it->second;
        }
    }

    // Compute pooled stats
    std::vector<float> all_dims;
    for (const auto& [_, mean] : result.coord_to_dim) {
        all_dims.push_back(mean);
    }

    if (!all_dims.empty()) {
        result.pooled_mean = std::accumulate(all_dims.begin(), all_dims.end(), 0.0f) / all_dims.size();
        result.pooled_min = *std::min_element(all_dims.begin(), all_dims.end());
        result.pooled_max = *std::max_element(all_dims.begin(), all_dims.end());
    }

    return result;
}

TimestepAggregation aggregate_timestep_streaming(
    uint32_t step,
    const std::vector<SimpleGraph>& graphs,
    const std::vector<LightweightAnalysis>& analyses,
    const std::vector<Vec2>& initial_positions,
    const std::vector<VertexId>& global_anchors
) {
    TimestepAggregation result;
    result.step = step;
    result.num_states = static_cast<int>(graphs.size());

    if (graphs.empty() || graphs.size() != analyses.size()) return result;

#ifdef BH_DEBUG_AGGREGATION
    // Debug: track edge counts and vertex counts per graph
    std::cout << "[DEBUG step " << step << "] num_states=" << graphs.size() << std::endl;
    for (size_t i = 0; i < graphs.size(); ++i) {
        std::cout << "  graph[" << i << "]: V=" << graphs[i].vertex_count()
                  << ", E=" << graphs[i].edge_count() << std::endl;
    }
#endif

    // =========================================================================
    // Phase 1: Union graph building
    // =========================================================================
    auto union_build_start = std::chrono::high_resolution_clock::now();

    // Union of all vertices and edges, plus count for intersection
    std::unordered_set<VertexId> union_vertex_set;
    std::unordered_map<uint64_t, int> edge_counts;  // Count how many states contain each edge

    for (size_t g_idx = 0; g_idx < graphs.size(); ++g_idx) {
        const auto& graph = graphs[g_idx];
        const auto& verts = graph.vertices();

        for (VertexId v : verts) {
            union_vertex_set.insert(v);
        }

        // Build edges from adjacency (same logic as analyze_state)
        std::unordered_set<uint64_t> seen_edges;
        for (VertexId v : verts) {
            for (VertexId u : graph.neighbors(v)) {
                uint64_t key = (static_cast<uint64_t>(std::min(v, u)) << 32) | std::max(v, u);
                if (seen_edges.find(key) == seen_edges.end()) {
                    seen_edges.insert(key);
                    edge_counts[key]++;
                }
            }
        }
    }

    // Union = all edges that appear in at least one state
    std::unordered_set<uint64_t> union_edge_set;
    for (const auto& [key, count] : edge_counts) {
        union_edge_set.insert(key);
    }

    // Convert to vectors temporarily
    std::vector<VertexId> all_vertices(union_vertex_set.begin(), union_vertex_set.end());
    std::vector<Edge> all_edges;
    for (uint64_t key : union_edge_set) {
        VertexId v1 = static_cast<VertexId>(key >> 32);
        VertexId v2 = static_cast<VertexId>(key & 0xFFFFFFFF);
        all_edges.push_back({v1, v2});
    }

    auto union_build_end = std::chrono::high_resolution_clock::now();
    g_union_build_time_us += std::chrono::duration_cast<std::chrono::microseconds>(union_build_end - union_build_start).count();

    // =========================================================================
    // Phase 2: LCC computation
    // =========================================================================
    auto lcc_start = std::chrono::high_resolution_clock::now();

    // Filter to largest connected component
    auto lcc = find_largest_connected_component(all_vertices, all_edges);

    auto lcc_end = std::chrono::high_resolution_clock::now();
    g_lcc_time_us += std::chrono::duration_cast<std::chrono::microseconds>(lcc_end - lcc_start).count();

    // =========================================================================
    // Phase 3: Build result structures
    // =========================================================================

    // Keep only vertices in LCC
    for (VertexId v : all_vertices) {
        if (lcc.count(v)) {
            result.union_vertices.push_back(v);
        }
    }
    std::sort(result.union_vertices.begin(), result.union_vertices.end());

    // Keep only edges where both endpoints are in LCC
    for (const auto& e : all_edges) {
        if (lcc.count(e.v1) && lcc.count(e.v2)) {
            result.union_edges.push_back(e);
        }
    }

    // Intersection = edges present in ALL states at this step
    // Frequent = edges present in MORE than one state (excludes singletons)
    int num_states = static_cast<int>(graphs.size());
    size_t edges_in_all_states = 0;

    // =========================================================================
    // Phase 4: Bucketing (starts after basic edge processing)
    // =========================================================================
    auto bucketing_start = std::chrono::high_resolution_clock::now();
    size_t edges_excluded_by_lcc = 0;
    for (const auto& [key, count] : edge_counts) {
        VertexId v1 = static_cast<VertexId>(key >> 32);
        VertexId v2 = static_cast<VertexId>(key & 0xFFFFFFFF);
        bool in_lcc = lcc.count(v1) && lcc.count(v2);

        if (count == num_states) {
            edges_in_all_states++;
            if (in_lcc) {
                result.intersection_edges.push_back({v1, v2});
            } else {
                edges_excluded_by_lcc++;
            }
        }

        // Frequent: more than one state (excludes edges unique to a single state)
        if (count > 1 && in_lcc) {
            result.frequent_edges.push_back({v1, v2});
        }
    }

#ifdef BH_DEBUG_AGGREGATION
    // Debug only: count how many connected components exist (expensive BFS)
    size_t num_components = 0;
    {
        std::unordered_set<VertexId> remaining(all_vertices.begin(), all_vertices.end());
        std::unordered_map<VertexId, std::vector<VertexId>> temp_adj;
        for (const auto& e : all_edges) {
            temp_adj[e.v1].push_back(e.v2);
            temp_adj[e.v2].push_back(e.v1);
        }
        while (!remaining.empty()) {
            num_components++;
            std::queue<VertexId> q;
            q.push(*remaining.begin());
            remaining.erase(remaining.begin());
            while (!q.empty()) {
                VertexId v = q.front();
                q.pop();
                for (VertexId n : temp_adj[v]) {
                    if (remaining.count(n)) {
                        remaining.erase(n);
                        q.push(n);
                    }
                }
            }
        }
    }
    std::cout << "[DEBUG step " << step << "] total_unique_edges=" << edge_counts.size()
              << ", LCC_size=" << lcc.size()
              << ", all_vertices=" << all_vertices.size()
              << ", num_components=" << num_components
              << ", edges_in_all_states=" << edges_in_all_states
              << ", excluded_by_lcc=" << edges_excluded_by_lcc
              << ", intersection_edges=" << result.intersection_edges.size()
              << ", union_edges=" << result.union_edges.size() << std::endl;
#endif

    // Build vertex positions from initial condition
    result.vertex_positions.resize(result.union_vertices.size());
    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];
        if (v < initial_positions.size()) {
            result.vertex_positions[i] = initial_positions[v];
        } else {
            result.vertex_positions[i] = {0, 0};
        }
    }

    // Build SimpleGraph from union (LCC) for coordinate computation
    SimpleGraph union_graph;
    union_graph.build(result.union_vertices, result.union_edges);

    // NOTE: We no longer compute all_pairs_distances() here!
    // - Geodesic coords: use compute_geodesic_coordinates() - O(A × (V+E)) where A=6 anchors
    // - Global dims: use estimate_all_dimensions_truncated() - O(V × neighborhood) for radius=5

    // Use global anchors (filtered to those present in this union graph)
    // This ensures consistent coordinate bucketing across all timesteps
    std::vector<VertexId> valid_anchors;
    for (VertexId a : global_anchors) {
        if (union_graph.has_vertex(a)) {
            valid_anchors.push_back(a);
        }
    }

    // Minimum anchors needed for reasonable coordinate system
    constexpr int MIN_ANCHORS = 3;
    int original_valid_count = static_cast<int>(valid_anchors.size());

    if (original_valid_count < MIN_ANCHORS) {
        // Not enough global anchors survive in this timestep - fall back to local anchors
        // Properties are basis-invariant so this is mathematically sound
        valid_anchors = select_anchors(
            union_graph,
            result.union_vertices,
            static_cast<int>(global_anchors.size()),  // Request same number as global
            3  // Minimum separation
        );
        std::cout << "  [Step " << step << "] WARNING: Only "
                  << original_valid_count << " of " << global_anchors.size()
                  << " global anchors valid, using " << valid_anchors.size()
                  << " local anchors instead" << std::endl;
    }

    int num_anchors = static_cast<int>(valid_anchors.size());

#ifdef BH_DEBUG_AGGREGATION
    std::cout << "[DEBUG step " << step << "] global_anchors=" << global_anchors.size()
              << ", valid_in_LCC=" << valid_anchors.size()
              << (using_local_anchors ? " (LOCAL)" : " (GLOBAL)") << std::endl;
#endif

    // Compute geodesic coordinates directly - O(A × (V+E)) where A = num_anchors (typically 6)
    auto union_coords = compute_geodesic_coordinates(union_graph, valid_anchors);

    // ==========================================================================
    // Aggregate dimensions by GEODESIC COORDINATE (not vertex ID!)
    // Multiple vertices with the same coordinate bucket together
    // ==========================================================================
    std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> coord_dims;

    for (size_t g_idx = 0; g_idx < graphs.size(); ++g_idx) {
        const auto& graph = graphs[g_idx];
        const auto& analysis = analyses[g_idx];
        const auto& verts = graph.vertices();

        for (size_t i = 0; i < verts.size(); ++i) {
            VertexId v = verts[i];

            if (!lcc.count(v)) continue;
            if (i >= analysis.vertex_dimensions.size()) continue;
            if (analysis.vertex_dimensions[i] <= 0) continue;

            // Get coordinate from union graph
            auto coord_it = union_coords.find(v);
            if (coord_it == union_coords.end()) continue;

            CoordKey key;
            key.num_anchors = num_anchors;
            for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
                key.coords[a] = coord_it->second[a];
            }

            // Bucket by coordinate, not vertex ID!
            coord_dims[key].push_back(analysis.vertex_dimensions[i]);
        }
    }

    // Compute mean, variance, min, max per coordinate bucket
    for (auto& [key, dims] : coord_dims) {
        if (dims.empty()) continue;

        float mean = std::accumulate(dims.begin(), dims.end(), 0.0f) / dims.size();
        result.coord_to_dim[key] = mean;

        float min_val = *std::min_element(dims.begin(), dims.end());
        float max_val = *std::max_element(dims.begin(), dims.end());
        result.coord_to_min_dim[key] = min_val;
        result.coord_to_max_dim[key] = max_val;

        float variance = 0.0f;
        if (dims.size() > 1) {
            for (float d : dims) {
                float diff = d - mean;
                variance += diff * diff;
            }
            variance /= dims.size();
        }
        result.coord_to_var[key] = variance;
    }

    // Assign bucketed dimensions to union vertices via coordinate lookup
    result.mean_dimensions.resize(result.union_vertices.size(), -1);
    result.variance_dimensions.resize(result.union_vertices.size(), -1);
    result.min_dimensions.resize(result.union_vertices.size(), -1);
    result.max_dimensions.resize(result.union_vertices.size(), -1);
    result.raw_vertex_dimensions.resize(result.union_vertices.size(), -1);

    // Compute raw per-vertex dimensions (average across states by vertex ID, no bucketing)
    // Map from vertex ID -> sum and count for averaging
    std::unordered_map<VertexId, std::pair<float, int>> vertex_dim_accum;
    for (size_t g_idx = 0; g_idx < graphs.size(); ++g_idx) {
        const auto& graph = graphs[g_idx];
        const auto& analysis = analyses[g_idx];
        const auto& verts = graph.vertices();

        for (size_t i = 0; i < verts.size(); ++i) {
            VertexId v = verts[i];
            if (!lcc.count(v)) continue;
            if (i >= analysis.vertex_dimensions.size()) continue;
            if (analysis.vertex_dimensions[i] <= 0) continue;

            auto& [sum, count] = vertex_dim_accum[v];
            sum += analysis.vertex_dimensions[i];
            count++;
        }
    }

    // Build vertex index map for raw dimensions
    std::unordered_map<VertexId, size_t> union_vertex_idx;
    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        union_vertex_idx[result.union_vertices[i]] = i;
    }

    // Assign raw per-vertex average
    for (const auto& [vid, sum_count] : vertex_dim_accum) {
        auto idx_it = union_vertex_idx.find(vid);
        if (idx_it != union_vertex_idx.end() && sum_count.second > 0) {
            result.raw_vertex_dimensions[idx_it->second] = sum_count.first / static_cast<float>(sum_count.second);
        }
    }

    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];

        auto coord_it = union_coords.find(v);
        if (coord_it == union_coords.end()) continue;

        CoordKey key;
        key.num_anchors = num_anchors;
        for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
            key.coords[a] = coord_it->second[a];
        }

        auto dim_it = result.coord_to_dim.find(key);
        if (dim_it != result.coord_to_dim.end()) {
            result.mean_dimensions[i] = dim_it->second;
        }
        auto var_it = result.coord_to_var.find(key);
        if (var_it != result.coord_to_var.end()) {
            result.variance_dimensions[i] = var_it->second;
        }
        auto min_it = result.coord_to_min_dim.find(key);
        if (min_it != result.coord_to_min_dim.end()) {
            result.min_dimensions[i] = min_it->second;
        }
        auto max_it = result.coord_to_max_dim.find(key);
        if (max_it != result.coord_to_max_dim.end()) {
            result.max_dimensions[i] = max_it->second;
        }
    }

    // Compute pooled stats from coordinate buckets (not from vertex array)
    std::vector<float> all_dims;
    for (const auto& [_, mean] : result.coord_to_dim) {
        all_dims.push_back(mean);
    }

    if (!all_dims.empty()) {
        result.pooled_mean = std::accumulate(all_dims.begin(), all_dims.end(), 0.0f) / all_dims.size();
        result.pooled_min = *std::min_element(all_dims.begin(), all_dims.end());
        result.pooled_max = *std::max_element(all_dims.begin(), all_dims.end());
    }

    // Compute pooled stats for variance from coordinate buckets
    std::vector<float> all_vars;
    for (const auto& [_, var] : result.coord_to_var) {
        all_vars.push_back(var);
    }

    if (!all_vars.empty()) {
        result.pooled_variance = std::accumulate(all_vars.begin(), all_vars.end(), 0.0f) / all_vars.size();
        result.var_min = *std::min_element(all_vars.begin(), all_vars.end());
        result.var_max = *std::max_element(all_vars.begin(), all_vars.end());
    }

    // ==========================================================================
    // GLOBAL dimension computation (ball growth on union graph, bucketed by coordinate)
    // ==========================================================================
    // 1. Compute dimension per vertex using ball growth on UNION graph edges
    // 2. Bucket by geodesic coordinate (same as local mode)
    // 3. Compute mean/variance per bucket
    // 4. Assign bucket values back to vertices

    constexpr int global_max_radius = 5;
    // Use truncated BFS - O(V × neighborhood) instead of O(V × V)
    auto global_raw_dims = estimate_all_dimensions_truncated(union_graph, global_max_radius);

    // Bucket global dimensions by geodesic coordinate
    std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> global_coord_buckets;

    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];
        if (i >= global_raw_dims.size()) continue;
        float dim = global_raw_dims[i];
        if (dim <= 0) continue;

        auto coord_it = union_coords.find(v);
        if (coord_it == union_coords.end()) continue;

        CoordKey key;
        key.num_anchors = num_anchors;
        for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
            key.coords[a] = coord_it->second[a];
        }
        global_coord_buckets[key].push_back(dim);
    }

    // Compute mean/variance per bucket for global dimensions
    std::unordered_map<CoordKey, float, CoordKeyHash> global_coord_to_mean;
    std::unordered_map<CoordKey, float, CoordKeyHash> global_coord_to_var;

    for (auto& [key, dims] : global_coord_buckets) {
        if (dims.empty()) continue;

        float mean = std::accumulate(dims.begin(), dims.end(), 0.0f) / dims.size();
        global_coord_to_mean[key] = mean;

        float variance = 0.0f;
        if (dims.size() > 1) {
            for (float d : dims) {
                float diff = d - mean;
                variance += diff * diff;
            }
            variance /= dims.size();
        }
        global_coord_to_var[key] = variance;
    }

    // Assign global mean/variance to vertices based on their coordinate bucket
    result.global_mean_dimensions.resize(result.union_vertices.size(), -1);
    result.global_variance_dimensions.resize(result.union_vertices.size(), -1);

    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];
        auto coord_it = union_coords.find(v);
        if (coord_it == union_coords.end()) continue;

        CoordKey key;
        key.num_anchors = num_anchors;
        for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
            key.coords[a] = coord_it->second[a];
        }

        auto mean_it = global_coord_to_mean.find(key);
        if (mean_it != global_coord_to_mean.end()) {
            result.global_mean_dimensions[i] = mean_it->second;
        }

        auto var_it = global_coord_to_var.find(key);
        if (var_it != global_coord_to_var.end()) {
            result.global_variance_dimensions[i] = var_it->second;
        }
    }

    // Compute pooled stats for global mean
    std::vector<float> all_global_means;
    for (float d : result.global_mean_dimensions) {
        if (d >= 0) all_global_means.push_back(d);
    }
    if (!all_global_means.empty()) {
        result.global_mean_pooled = std::accumulate(all_global_means.begin(), all_global_means.end(), 0.0f) / all_global_means.size();
        result.global_mean_min = *std::min_element(all_global_means.begin(), all_global_means.end());
        result.global_mean_max = *std::max_element(all_global_means.begin(), all_global_means.end());
    }

    // Compute pooled stats for global variance
    std::vector<float> all_global_vars;
    for (float v : result.global_variance_dimensions) {
        if (v >= 0) all_global_vars.push_back(v);
    }
    if (!all_global_vars.empty()) {
        result.global_var_pooled = std::accumulate(all_global_vars.begin(), all_global_vars.end(), 0.0f) / all_global_vars.size();
        result.global_var_min = *std::min_element(all_global_vars.begin(), all_global_vars.end());
        result.global_var_max = *std::max_element(all_global_vars.begin(), all_global_vars.end());
    }

    // ==========================================================================
    // CURVATURE aggregation (Branchial: mean/variance across states)
    // ==========================================================================
    // Same pattern as dimension: bucket by geodesic coordinate, compute mean/variance

    // Buckets: coord -> list of curvature values from all states
    std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> curv_ollivier_buckets;
    std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> curv_wolfram_scalar_buckets;
    std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> curv_wolfram_ricci_buckets;
    std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> curv_dim_gradient_buckets;

    for (size_t g_idx = 0; g_idx < graphs.size(); ++g_idx) {
        const auto& graph = graphs[g_idx];
        const auto& analysis = analyses[g_idx];
        const auto& verts = graph.vertices();

        for (size_t i = 0; i < verts.size(); ++i) {
            VertexId v = verts[i];
            if (!lcc.count(v)) continue;

            // Get coordinate from union graph
            auto coord_it = union_coords.find(v);
            if (coord_it == union_coords.end()) continue;

            CoordKey key;
            key.num_anchors = num_anchors;
            for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
                key.coords[a] = coord_it->second[a];
            }

            // Collect curvature values by coordinate bucket
            auto oll_it = analysis.curvature_ollivier.find(v);
            if (oll_it != analysis.curvature_ollivier.end() && std::isfinite(oll_it->second)) {
                curv_ollivier_buckets[key].push_back(oll_it->second);
            }

            auto ws_it = analysis.curvature_wolfram_scalar.find(v);
            if (ws_it != analysis.curvature_wolfram_scalar.end() && std::isfinite(ws_it->second)) {
                curv_wolfram_scalar_buckets[key].push_back(ws_it->second);
            }

            auto wr_it = analysis.curvature_wolfram_ricci.find(v);
            if (wr_it != analysis.curvature_wolfram_ricci.end() && std::isfinite(wr_it->second)) {
                curv_wolfram_ricci_buckets[key].push_back(wr_it->second);
            }

            auto dg_it = analysis.curvature_dim_gradient.find(v);
            if (dg_it != analysis.curvature_dim_gradient.end() && std::isfinite(dg_it->second)) {
                curv_dim_gradient_buckets[key].push_back(dg_it->second);
            }
        }
    }

    // Helper lambda to compute mean/variance from bucket
    auto compute_bucket_stats = [](const std::vector<float>& vals, float& mean_out, float& var_out) {
        if (vals.empty()) {
            mean_out = 0;
            var_out = 0;
            return;
        }
        mean_out = std::accumulate(vals.begin(), vals.end(), 0.0f) / vals.size();
        var_out = 0;
        if (vals.size() > 1) {
            for (float v : vals) {
                float diff = v - mean_out;
                var_out += diff * diff;
            }
            var_out /= vals.size();
        }
    };

    // Compute per-bucket mean/variance for each curvature type
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_ollivier_mean, coord_ollivier_var;
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_wolfram_scalar_mean, coord_wolfram_scalar_var;
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_wolfram_ricci_mean, coord_wolfram_ricci_var;
    std::unordered_map<CoordKey, float, CoordKeyHash> coord_dim_gradient_mean, coord_dim_gradient_var;

    for (auto& [key, vals] : curv_ollivier_buckets) {
        float mean, var;
        compute_bucket_stats(vals, mean, var);
        coord_ollivier_mean[key] = mean;
        coord_ollivier_var[key] = var;
    }
    for (auto& [key, vals] : curv_wolfram_scalar_buckets) {
        float mean, var;
        compute_bucket_stats(vals, mean, var);
        coord_wolfram_scalar_mean[key] = mean;
        coord_wolfram_scalar_var[key] = var;
    }
    for (auto& [key, vals] : curv_wolfram_ricci_buckets) {
        float mean, var;
        compute_bucket_stats(vals, mean, var);
        coord_wolfram_ricci_mean[key] = mean;
        coord_wolfram_ricci_var[key] = var;
    }
    for (auto& [key, vals] : curv_dim_gradient_buckets) {
        float mean, var;
        compute_bucket_stats(vals, mean, var);
        coord_dim_gradient_mean[key] = mean;
        coord_dim_gradient_var[key] = var;
    }

    // Resize result vectors
    size_t n_verts = result.union_vertices.size();
    result.mean_curvature_ollivier.resize(n_verts, 0);
    result.variance_curvature_ollivier.resize(n_verts, 0);
    result.mean_curvature_wolfram_scalar.resize(n_verts, 0);
    result.variance_curvature_wolfram_scalar.resize(n_verts, 0);
    result.mean_curvature_wolfram_ricci.resize(n_verts, 0);
    result.variance_curvature_wolfram_ricci.resize(n_verts, 0);
    result.mean_curvature_dim_gradient.resize(n_verts, 0);
    result.variance_curvature_dim_gradient.resize(n_verts, 0);

    // Assign bucketed curvatures to union vertices via coordinate lookup
    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];
        auto coord_it = union_coords.find(v);
        if (coord_it == union_coords.end()) continue;

        CoordKey key;
        key.num_anchors = num_anchors;
        for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
            key.coords[a] = coord_it->second[a];
        }

        auto it_om = coord_ollivier_mean.find(key);
        if (it_om != coord_ollivier_mean.end()) result.mean_curvature_ollivier[i] = it_om->second;
        auto it_ov = coord_ollivier_var.find(key);
        if (it_ov != coord_ollivier_var.end()) result.variance_curvature_ollivier[i] = it_ov->second;

        auto it_wsm = coord_wolfram_scalar_mean.find(key);
        if (it_wsm != coord_wolfram_scalar_mean.end()) result.mean_curvature_wolfram_scalar[i] = it_wsm->second;
        auto it_wsv = coord_wolfram_scalar_var.find(key);
        if (it_wsv != coord_wolfram_scalar_var.end()) result.variance_curvature_wolfram_scalar[i] = it_wsv->second;

        auto it_wrm = coord_wolfram_ricci_mean.find(key);
        if (it_wrm != coord_wolfram_ricci_mean.end()) result.mean_curvature_wolfram_ricci[i] = it_wrm->second;
        auto it_wrv = coord_wolfram_ricci_var.find(key);
        if (it_wrv != coord_wolfram_ricci_var.end()) result.variance_curvature_wolfram_ricci[i] = it_wrv->second;

        auto it_dgm = coord_dim_gradient_mean.find(key);
        if (it_dgm != coord_dim_gradient_mean.end()) result.mean_curvature_dim_gradient[i] = it_dgm->second;
        auto it_dgv = coord_dim_gradient_var.find(key);
        if (it_dgv != coord_dim_gradient_var.end()) result.variance_curvature_dim_gradient[i] = it_dgv->second;
    }

    // Compute curvature stats (min/max for normalization)
    auto compute_range = [](const std::vector<float>& v, float& min_out, float& max_out) {
        min_out = 0; max_out = 0;
        bool first = true;
        for (float val : v) {
            if (std::isfinite(val)) {
                if (first) { min_out = max_out = val; first = false; }
                else { min_out = std::min(min_out, val); max_out = std::max(max_out, val); }
            }
        }
    };

    compute_range(result.mean_curvature_ollivier, result.ollivier_mean_min, result.ollivier_mean_max);
    compute_range(result.variance_curvature_ollivier, result.ollivier_var_min, result.ollivier_var_max);
    compute_range(result.mean_curvature_wolfram_scalar, result.wolfram_scalar_mean_min, result.wolfram_scalar_mean_max);
    compute_range(result.variance_curvature_wolfram_scalar, result.wolfram_scalar_var_min, result.wolfram_scalar_var_max);
    compute_range(result.mean_curvature_wolfram_ricci, result.wolfram_ricci_mean_min, result.wolfram_ricci_mean_max);
    compute_range(result.variance_curvature_wolfram_ricci, result.wolfram_ricci_var_min, result.wolfram_ricci_var_max);
    compute_range(result.mean_curvature_dim_gradient, result.dim_gradient_mean_min, result.dim_gradient_mean_max);
    compute_range(result.variance_curvature_dim_gradient, result.dim_gradient_var_min, result.dim_gradient_var_max);

    // ==========================================================================
    // FOLIATION curvature (computed on union graph, single sample)
    // ==========================================================================
    CurvatureConfig foliation_curv_config;
    foliation_curv_config.compute_ollivier_ricci = true;
    foliation_curv_config.compute_wolfram_scalar = true;
    foliation_curv_config.compute_wolfram_ricci = true;
    foliation_curv_config.compute_dimension_gradient = true;
    foliation_curv_config.max_radius = global_max_radius;

    CurvatureAnalysisResult foliation_curv = analyze_curvature(union_graph, foliation_curv_config, &global_raw_dims);

    // Resize foliation curvature vectors
    result.foliation_curvature_ollivier.resize(n_verts, 0);
    result.foliation_curvature_wolfram_scalar.resize(n_verts, 0);
    result.foliation_curvature_wolfram_ricci.resize(n_verts, 0);
    result.foliation_curvature_dim_gradient.resize(n_verts, 0);

    // Assign foliation curvature to vertices
    for (size_t i = 0; i < result.union_vertices.size(); ++i) {
        VertexId v = result.union_vertices[i];

        auto it_o = foliation_curv.ollivier_ricci_map.find(v);
        if (it_o != foliation_curv.ollivier_ricci_map.end()) {
            result.foliation_curvature_ollivier[i] = it_o->second;
        }

        auto it_ws = foliation_curv.wolfram_scalar_map.find(v);
        if (it_ws != foliation_curv.wolfram_scalar_map.end()) {
            result.foliation_curvature_wolfram_scalar[i] = it_ws->second;
        }

        auto it_wr = foliation_curv.wolfram_ricci_map.find(v);
        if (it_wr != foliation_curv.wolfram_ricci_map.end()) {
            result.foliation_curvature_wolfram_ricci[i] = it_wr->second;
        }

        auto it_dg = foliation_curv.dimension_gradient_map.find(v);
        if (it_dg != foliation_curv.dimension_gradient_map.end()) {
            result.foliation_curvature_dim_gradient[i] = it_dg->second;
        }
    }

    // Compute foliation curvature stats
    compute_range(result.foliation_curvature_ollivier, result.foliation_ollivier_min, result.foliation_ollivier_max);
    compute_range(result.foliation_curvature_wolfram_scalar, result.foliation_wolfram_scalar_min, result.foliation_wolfram_scalar_max);
    compute_range(result.foliation_curvature_wolfram_ricci, result.foliation_wolfram_ricci_min, result.foliation_wolfram_ricci_max);
    compute_range(result.foliation_curvature_dim_gradient, result.foliation_dim_gradient_min, result.foliation_dim_gradient_max);

    auto bucketing_end = std::chrono::high_resolution_clock::now();
    g_bucketing_time_us += std::chrono::duration_cast<std::chrono::microseconds>(bucketing_end - bucketing_start).count();

    return result;
}

// =============================================================================
// Full Pipeline
// =============================================================================

float quantile(std::vector<float> values, float q) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());

    float idx = q * (values.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, values.size() - 1);
    float frac = idx - lo;

    return values[lo] * (1 - frac) + values[hi] * frac;
}

BHAnalysisResult run_analysis(
    const BHInitialCondition& initial,
    const EvolutionConfig& evolution_config,
    const std::vector<std::vector<SimpleGraph>>& states_by_step,
    const AnalysisConfig& analysis_config
) {
    BHAnalysisResult result;
    result.initial = initial;
    result.evolution_config = evolution_config;
    result.analysis_max_radius = analysis_config.max_radius;

    // Flatten all graphs to find stable vertices
    std::vector<SimpleGraph> all_graphs;
    for (const auto& step_graphs : states_by_step) {
        for (const auto& g : step_graphs) {
            all_graphs.push_back(g);
        }
    }

    // Find stable vertices and select anchors
    auto stable = find_stable_vertices(all_graphs);

    if (!states_by_step.empty() && !states_by_step[0].empty()) {
        result.anchor_vertices = select_anchors(
            states_by_step[0][0],
            stable,
            analysis_config.num_anchors,
            analysis_config.anchor_min_separation
        );
    }

    // Analyze each state
    std::vector<float> all_dimensions;

    StateId state_counter = 0;
    for (uint32_t step = 0; step < states_by_step.size(); ++step) {
        for (const auto& graph : states_by_step[step]) {
            auto analysis = analyze_state(
                state_counter++,
                step,
                graph,
                result.anchor_vertices,
                analysis_config.max_radius
            );

            // Collect dimensions for quantile computation
            for (float d : analysis.vertex_dimensions) {
                if (d > 0) all_dimensions.push_back(d);
            }

            result.per_state.push_back(std::move(analysis));
        }
    }

    result.total_states = static_cast<int>(result.per_state.size());
    result.total_steps = static_cast<int>(states_by_step.size());

    // Compute dimension quantiles for color scaling
    if (!all_dimensions.empty()) {
        result.dim_q05 = quantile(all_dimensions, analysis_config.quantile_low);
        result.dim_q95 = quantile(all_dimensions, analysis_config.quantile_high);
        result.dim_min = *std::min_element(all_dimensions.begin(), all_dimensions.end());
        result.dim_max = *std::max_element(all_dimensions.begin(), all_dimensions.end());
    }

    // Aggregate by timestep and store per-state data for single-state viewing
    result.states_per_step.resize(states_by_step.size());
    for (uint32_t step = 0; step < states_by_step.size(); ++step) {
        std::vector<StateAnalysis> states_at_step;
        for (const auto& state : result.per_state) {
            if (state.step == step) {
                states_at_step.push_back(state);
            }
        }

        auto agg = aggregate_timestep(step, states_at_step, initial.vertex_positions);
        result.per_timestep.push_back(std::move(agg));

        // Store per-state vertices and edges for single-state viewing
        for (const auto& graph : states_by_step[step]) {
            StateData sd;
            sd.vertices = graph.vertices();
            // Reconstruct edges from adjacency - use set to deduplicate
            std::set<std::pair<VertexId, VertexId>> edge_set;
            for (VertexId v : graph.vertices()) {
                for (VertexId u : graph.neighbors(v)) {
                    VertexId v1 = std::min(v, u);
                    VertexId v2 = std::max(v, u);
                    edge_set.insert({v1, v2});
                }
            }
            for (const auto& [v1, v2] : edge_set) {
                sd.edges.push_back({v1, v2});
            }
            result.states_per_step[step].push_back(std::move(sd));
        }
    }

#ifdef BLACKHOLE_WITH_LAYOUT
    // Compute layouts for all timesteps
    compute_all_layouts(result.per_timestep, initial.vertex_positions, analysis_config.layout);
#endif

    // Compute prefix sums for efficient timeslice aggregation
    std::cout << "  Computing prefix sums for timeslice aggregation..." << std::endl;
    {
        std::unordered_set<VertexId> all_vertex_set;
        std::set<std::pair<VertexId, VertexId>> all_edge_set;

        for (size_t t = 0; t < result.per_timestep.size(); ++t) {
            auto& ts = result.per_timestep[t];

            // Build global vertex/edge sets
            for (VertexId v : ts.union_vertices) {
                all_vertex_set.insert(v);
            }
            for (const Edge& e : ts.union_edges) {
                VertexId v1 = std::min(e.v1, e.v2);
                VertexId v2 = std::max(e.v1, e.v2);
                all_edge_set.insert({v1, v2});
            }

            // Copy prefix sums from previous timestep
            if (t > 0) {
                const auto& prev = result.per_timestep[t - 1];
                ts.dim_prefix_sum = prev.dim_prefix_sum;
                ts.dim_prefix_count = prev.dim_prefix_count;
                ts.var_prefix_sum = prev.var_prefix_sum;
                ts.var_prefix_count = prev.var_prefix_count;
                ts.global_dim_prefix_sum = prev.global_dim_prefix_sum;
                ts.global_dim_prefix_count = prev.global_dim_prefix_count;
                ts.global_var_prefix_sum = prev.global_var_prefix_sum;
                ts.global_var_prefix_count = prev.global_var_prefix_count;
            }

            // Add current timestep values to prefix sums
            for (size_t i = 0; i < ts.union_vertices.size(); ++i) {
                VertexId v = ts.union_vertices[i];

                // Local mean dimension
                if (i < ts.mean_dimensions.size()) {
                    float dim = ts.mean_dimensions[i];
                    if (dim >= 0 && std::isfinite(dim)) {
                        ts.dim_prefix_sum[v] += dim;
                        ts.dim_prefix_count[v] += 1;
                    }
                }

                // Local variance
                if (i < ts.variance_dimensions.size()) {
                    float var = ts.variance_dimensions[i];
                    if (var >= 0 && std::isfinite(var)) {
                        ts.var_prefix_sum[v] += var;
                        ts.var_prefix_count[v] += 1;
                    }
                }

                // Global mean dimension
                if (i < ts.global_mean_dimensions.size()) {
                    float gdim = ts.global_mean_dimensions[i];
                    if (gdim >= 0 && std::isfinite(gdim)) {
                        ts.global_dim_prefix_sum[v] += gdim;
                        ts.global_dim_prefix_count[v] += 1;
                    }
                }

                // Global variance
                if (i < ts.global_variance_dimensions.size()) {
                    float gvar = ts.global_variance_dimensions[i];
                    if (gvar >= 0 && std::isfinite(gvar)) {
                        ts.global_var_prefix_sum[v] += gvar;
                        ts.global_var_prefix_count[v] += 1;
                    }
                }
            }
        }

        // Store global vertex/edge union
        result.all_vertices.assign(all_vertex_set.begin(), all_vertex_set.end());
        std::sort(result.all_vertices.begin(), result.all_vertices.end());

        for (const auto& ep : all_edge_set) {
            result.all_edges.push_back({ep.first, ep.second});
        }

        std::cout << "    Total unique vertices: " << result.all_vertices.size()
                  << ", edges: " << result.all_edges.size() << std::endl;

        // ==========================================================================
        // Mega-union dimension computation (across ALL timesteps)
        // ==========================================================================
        std::cout << "  Computing mega-union dimension..." << std::endl;

        // Build mega-union graph
        SimpleGraph mega_union;
        mega_union.build(result.all_vertices, result.all_edges);

        // Compute dimension on mega-union
        constexpr int mega_max_radius = 5;
        auto mega_dims = estimate_all_dimensions(mega_union, mega_max_radius);

        // Get geodesic coordinates on mega-union using stored anchor vertices
        std::vector<VertexId> valid_mega_anchors;
        for (VertexId a : result.anchor_vertices) {
            if (mega_union.has_vertex(a)) {
                valid_mega_anchors.push_back(a);
            }
        }
        auto mega_coords = compute_geodesic_coordinates(mega_union, valid_mega_anchors);
        int mega_num_anchors = static_cast<int>(valid_mega_anchors.size());

        // Bucket by geodesic coordinate
        std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> mega_coord_buckets;
        for (size_t i = 0; i < result.all_vertices.size(); ++i) {
            VertexId v = result.all_vertices[i];
            if (i >= mega_dims.size()) continue;
            float dim = mega_dims[i];
            if (dim <= 0) continue;

            auto coord_it = mega_coords.find(v);
            if (coord_it == mega_coords.end()) continue;

            CoordKey key;
            key.num_anchors = mega_num_anchors;
            for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
                key.coords[a] = coord_it->second[a];
            }
            mega_coord_buckets[key].push_back(dim);
        }

        // Compute mean per bucket
        std::unordered_map<CoordKey, float, CoordKeyHash> mega_coord_to_dim;
        for (auto& [key, dims] : mega_coord_buckets) {
            if (!dims.empty()) {
                float mean = std::accumulate(dims.begin(), dims.end(), 0.0f) / dims.size();
                mega_coord_to_dim[key] = mean;
            }
        }

        // Assign to vertices and compute min/max
        result.mega_dim_min = std::numeric_limits<float>::max();
        result.mega_dim_max = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < result.all_vertices.size(); ++i) {
            VertexId v = result.all_vertices[i];

            auto coord_it = mega_coords.find(v);
            if (coord_it == mega_coords.end()) continue;

            CoordKey key;
            key.num_anchors = mega_num_anchors;
            for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
                key.coords[a] = coord_it->second[a];
            }

            auto dim_it = mega_coord_to_dim.find(key);
            if (dim_it != mega_coord_to_dim.end()) {
                result.mega_dimension[v] = dim_it->second;
                result.mega_dim_min = std::min(result.mega_dim_min, dim_it->second);
                result.mega_dim_max = std::max(result.mega_dim_max, dim_it->second);
            }
        }

        if (result.mega_dim_min > result.mega_dim_max) {
            result.mega_dim_min = 0;
            result.mega_dim_max = 0;
        }

        std::cout << "    Mega-union dimension range: [" << result.mega_dim_min
                  << ", " << result.mega_dim_max << "] for "
                  << result.mega_dimension.size() << " vertices" << std::endl;
    }

    // Compute overall bounding radius from laid-out positions (for camera framing)
    float max_radius = 0.0f;
    for (const auto& ts : result.per_timestep) {
        for (const auto& p : ts.layout_positions) {
            float r = std::sqrt(p.x * p.x + p.y * p.y);
            if (r > max_radius) max_radius = r;
        }
    }
    result.layout_bounding_radius = std::max(1.0f, max_radius);

    return result;
}

#ifdef BLACKHOLE_WITH_LAYOUT
// =============================================================================
// Layout Computation
// =============================================================================

std::vector<Vec2> compute_timestep_layout(
    const TimestepAggregation& timestep,
    const std::vector<Vec2>& prev_positions,
    const std::unordered_map<VertexId, size_t>& prev_vertex_indices,
    const LayoutConfig& config
) {
    using namespace viz::layout;

    // Create layout graph
    LayoutGraph graph;

    // Map from timestep vertex index to layout vertex index
    std::unordered_map<VertexId, uint32_t> vertex_to_layout_idx;

    // Random number generator for new vertex placement (thread_local for safety)
    thread_local std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise_dist(-0.1f, 0.1f);

    // Track new vertices
    size_t new_vertex_count = 0;
    size_t found_in_prev = 0;

    // Add vertices
    for (size_t i = 0; i < timestep.union_vertices.size(); ++i) {
        VertexId vid = timestep.union_vertices[i];
        float x = 0, y = 0, z = 0;

        // Check if this vertex existed in previous timestep
        auto prev_it = prev_vertex_indices.find(vid);
        if (prev_it != prev_vertex_indices.end() && prev_it->second < prev_positions.size()) {
            // Use previous position
            x = prev_positions[prev_it->second].x;
            y = prev_positions[prev_it->second].y;
            found_in_prev++;
        } else {
            // New vertex: place near neighbors (will be computed after edges are added)
            x = noise_dist(rng);
            y = noise_dist(rng);
            new_vertex_count++;
        }

        uint32_t layout_idx = graph.add_vertex(x, y, z);
        vertex_to_layout_idx[vid] = layout_idx;
    }

    // Add edges
    for (const auto& edge : timestep.union_edges) {
        auto it1 = vertex_to_layout_idx.find(edge.v1);
        auto it2 = vertex_to_layout_idx.find(edge.v2);
        if (it1 != vertex_to_layout_idx.end() && it2 != vertex_to_layout_idx.end()) {
            graph.add_edge(it1->second, it2->second, 1.0f, 1.0f);
        }
    }

    // For new vertices without previous positions, place them near their neighbors
    for (size_t i = 0; i < timestep.union_vertices.size(); ++i) {
        VertexId vid = timestep.union_vertices[i];
        auto prev_it = prev_vertex_indices.find(vid);

        if (prev_it == prev_vertex_indices.end() || prev_it->second >= prev_positions.size()) {
            // This is a new vertex - place it at centroid of neighbors
            uint32_t layout_idx = vertex_to_layout_idx[vid];
            float sum_x = 0, sum_y = 0;
            int neighbor_count = 0;
            int total_neighbors = 0;

            // Find neighbors from edges
            for (const auto& edge : timestep.union_edges) {
                VertexId neighbor;
                bool found_neighbor = false;
                if (edge.v1 == vid) {
                    neighbor = edge.v2;
                    found_neighbor = true;
                } else if (edge.v2 == vid) {
                    neighbor = edge.v1;
                    found_neighbor = true;
                }
                if (found_neighbor) {
                    total_neighbors++;
                    auto neighbor_prev_it = prev_vertex_indices.find(neighbor);
                    if (neighbor_prev_it != prev_vertex_indices.end() &&
                        neighbor_prev_it->second < prev_positions.size()) {
                        sum_x += prev_positions[neighbor_prev_it->second].x;
                        sum_y += prev_positions[neighbor_prev_it->second].y;
                        neighbor_count++;
                    }
                }
            }

            if (neighbor_count > 0) {
                float new_x = sum_x / neighbor_count + noise_dist(rng);
                float new_y = sum_y / neighbor_count + noise_dist(rng);
                graph.positions_x[layout_idx] = new_x;
                graph.positions_y[layout_idx] = new_y;
            }
            new_vertex_count++;
        }
    }

    // Run force-directed layout with energy-based convergence
    // This is critical for getting a proper visual layout
    auto engine = create_layout_engine(LayoutBackend::CPU);
    if (engine) {
        engine->upload_graph(graph);

        // Configure layout parameters - MUST match live viewer exactly!
        LayoutParams params;
        params.algorithm = LayoutAlgorithm::Direct;
        params.dimension = LayoutDimension::Layout2D;
        params.spring_constant = config.spring_constant;
        params.repulsion_constant = config.repulsion_constant;
        params.damping = config.damping;
        params.gravity = config.gravity;
        params.max_displacement = config.max_displacement;
        params.edge_budget = static_cast<uint32_t>(config.edge_budget);
        params.max_iterations = 1;  // Run one iteration at a time for energy tracking

        // Lambda to compute system energy
        auto compute_energy = [&]() -> double {
            engine->download_positions(graph);
            double energy = 0.0;

            // Spring energy: sum of (k/2) * (dist - rest)^2
            for (size_t e = 0; e < graph.edge_sources.size(); ++e) {
                uint32_t src = graph.edge_sources[e];
                uint32_t dst = graph.edge_targets[e];
                float dx = graph.positions_x[dst] - graph.positions_x[src];
                float dy = graph.positions_y[dst] - graph.positions_y[src];
                float dist = std::sqrt(dx * dx + dy * dy);
                float rest = graph.edge_rest_lengths[e];
                if (rest <= 0) rest = 1.0f;
                float stretch = dist - rest;
                energy += 0.5 * config.spring_constant * stretch * stretch;
            }

            // Repulsion energy: sum of repulsion_constant / dist (for all pairs)
            // Use sampling for large graphs to avoid O(n^2)
            uint32_t n = graph.vertex_count();
            if (n <= 200) {
                for (uint32_t i = 0; i < n; ++i) {
                    for (uint32_t j = i + 1; j < n; ++j) {
                        float dx = graph.positions_x[i] - graph.positions_x[j];
                        float dy = graph.positions_y[i] - graph.positions_y[j];
                        float dist = std::sqrt(dx * dx + dy * dy + 0.01f);
                        energy += config.repulsion_constant / dist;
                    }
                }
            } else {
                // Sample-based estimation for large graphs
                std::mt19937 rng(42);
                std::uniform_int_distribution<uint32_t> dist(0, n - 1);
                int samples = std::min(5000, static_cast<int>(n * 10));
                double sample_energy = 0;
                for (int s = 0; s < samples; ++s) {
                    uint32_t i = dist(rng);
                    uint32_t j = dist(rng);
                    if (i == j) continue;
                    float dx = graph.positions_x[i] - graph.positions_x[j];
                    float dy = graph.positions_y[i] - graph.positions_y[j];
                    float d = std::sqrt(dx * dx + dy * dy + 0.01f);
                    sample_energy += config.repulsion_constant / d;
                }
                // Scale to estimate full energy
                double pair_count = static_cast<double>(n) * (n - 1) / 2;
                energy += sample_energy * pair_count / samples;
            }

            return energy;
        };

        // Energy-based convergence loop
        int max_iters = config.max_iterations > 0 ? config.max_iterations : 5000;
        int min_iters = config.min_iterations > 0 ? config.min_iterations : 100;
        int window = config.energy_window > 0 ? config.energy_window : 20;
        float tolerance = config.energy_tolerance > 0 ? config.energy_tolerance : 0.001f;

        std::vector<double> energy_history;
        energy_history.reserve(max_iters / 10 + 1);

        int iter = 0;
        bool converged = false;
        double last_energy = compute_energy();
        energy_history.push_back(last_energy);

        while (iter < max_iters && !converged) {
            // Run a batch of iterations
            int batch_size = 10;
            for (int b = 0; b < batch_size && iter < max_iters; ++b, ++iter) {
                engine->iterate(params);
            }

            // Compute energy every batch
            double current_energy = compute_energy();
            energy_history.push_back(current_energy);

            // Check convergence after minimum iterations
            if (iter >= min_iters && energy_history.size() >= static_cast<size_t>(window)) {
                // Compare average of last window vs previous window
                size_t n_hist = energy_history.size();
                double recent_avg = 0, older_avg = 0;
                for (int i = 0; i < window / 2; ++i) {
                    recent_avg += energy_history[n_hist - 1 - i];
                    older_avg += energy_history[n_hist - 1 - window / 2 - i];
                }
                recent_avg /= (window / 2);
                older_avg /= (window / 2);

                // Relative change
                double rel_change = std::abs(recent_avg - older_avg) / (std::abs(older_avg) + 1e-10);
                if (rel_change < tolerance) {
                    converged = true;
                }
            }
        }

        // Final download
        engine->download_positions(graph);

        // Debug: print layout result for first timestep
        static bool first_layout = true;
        if (first_layout) {
            std::cout << "    Layout: " << iter << " iterations, "
                      << (converged ? "converged" : "max reached")
                      << ", energy=" << std::fixed << std::setprecision(1) << last_energy
                      << " -> " << energy_history.back() << std::endl;
            std::cout << "    Vertices: " << found_in_prev << " from prev, "
                      << new_vertex_count << " new (total " << timestep.union_vertices.size() << ")" << std::endl;
            first_layout = false;
        }
    } else {
        std::cerr << "    WARNING: Failed to create layout engine!" << std::endl;
    }

    // Convert back to Vec2 vector
    std::vector<Vec2> result(timestep.union_vertices.size());
    for (size_t i = 0; i < timestep.union_vertices.size(); ++i) {
        VertexId vid = timestep.union_vertices[i];
        uint32_t layout_idx = vertex_to_layout_idx[vid];
        result[i] = Vec2{graph.positions_x[layout_idx], graph.positions_y[layout_idx]};
    }

    return result;
}

// Helper: compute bounding radius (max distance from origin)
static float compute_bounding_radius(const std::vector<Vec2>& positions) {
    float max_r_sq = 0.0f;
    for (const auto& p : positions) {
        float r_sq = p.x * p.x + p.y * p.y;
        if (r_sq > max_r_sq) max_r_sq = r_sq;
    }
    return std::sqrt(max_r_sq);
}

// Helper: scale positions to match target bounding radius
static void scale_positions_to_radius(std::vector<Vec2>& positions, float target_radius) {
    float current_radius = compute_bounding_radius(positions);
    if (current_radius < 0.001f) return;  // Avoid division by near-zero

    float scale = target_radius / current_radius;
    for (auto& p : positions) {
        p.x *= scale;
        p.y *= scale;
    }
}

void compute_all_layouts(
    std::vector<TimestepAggregation>& timesteps,
    const std::vector<Vec2>& initial_positions,
    const LayoutConfig& config
) {
    if (timesteps.empty()) {
        std::cerr << "  WARNING: compute_all_layouts called with empty timesteps!" << std::endl;
        return;
    }

    // Only compute layout for step 0 - the live viewer will handle subsequent timesteps
    std::cout << "  Computing force-directed layout for step 0 only..." << std::endl;

    auto& ts0 = timesteps[0];
    std::cout << "    Step 0: " << ts0.union_vertices.size() << " vertices, "
              << ts0.union_edges.size() << " edges" << std::endl;

    // Build initial vertex index map
    std::unordered_map<VertexId, size_t> prev_vertex_indices;
    for (size_t i = 0; i < initial_positions.size(); ++i) {
        prev_vertex_indices[static_cast<VertexId>(i)] = i;
    }

    // Compute layout for step 0 only - using same params as live viewer
    std::vector<Vec2> layout = compute_timestep_layout(ts0, initial_positions, prev_vertex_indices, config);

    // Store in layout_positions (NO scaling - must match live viewer behavior)
    ts0.layout_positions = std::move(layout);

    // Print final layout info
    float final_radius = compute_bounding_radius(ts0.layout_positions);
    std::cout << "    Layout complete. Bounding radius: " << final_radius
              << " (" << ts0.layout_positions.size() << " vertices)" << std::endl;
}
#endif  // BLACKHOLE_WITH_LAYOUT

} // namespace viz::blackhole
