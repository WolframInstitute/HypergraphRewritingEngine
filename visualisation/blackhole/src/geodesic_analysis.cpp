#include "blackhole/geodesic_analysis.hpp"
#include <queue>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace viz::blackhole {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Get random engine with proper seeding
std::mt19937& get_rng(uint32_t seed) {
    static thread_local std::mt19937 rng;
    if (seed == 0) {
        seed = static_cast<uint32_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
    }
    rng.seed(seed);
    return rng;
}

// Find vertex index in graph's vertex list
size_t find_vertex_index(const SimpleGraph& graph, VertexId v) {
    const auto& vertices = graph.vertices();
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (vertices[i] == v) return i;
    }
    return SIZE_MAX;
}

// Get dimension at vertex (from pre-computed array or compute on-the-fly)
float get_vertex_dimension(
    const SimpleGraph& graph,
    VertexId v,
    const std::vector<float>* vertex_dimensions,
    int max_radius = 5
) {
    if (vertex_dimensions) {
        size_t idx = find_vertex_index(graph, v);
        if (idx < vertex_dimensions->size()) {
            return (*vertex_dimensions)[idx];
        }
    }
    // Fallback: compute on-the-fly (expensive)
    auto distances = graph.distances_from(v);
    return estimate_local_dimension(distances, max_radius);
}

// Select neighbor based on direction strategy
VertexId select_next_vertex(
    const SimpleGraph& graph,
    VertexId current,
    const std::unordered_set<VertexId>& visited,
    const GeodesicConfig& config,
    const std::vector<float>* vertex_dimensions,
    std::mt19937& rng
) {
    const auto& neighbors = graph.neighbors(current);
    if (neighbors.empty()) return current;

    // Filter out visited if configured
    std::vector<VertexId> candidates;
    for (VertexId n : neighbors) {
        if (!config.avoid_revisits || visited.find(n) == visited.end()) {
            candidates.push_back(n);
        }
    }

    if (candidates.empty()) return current;  // Stuck

    switch (config.direction) {
        case GeodesicDirection::Random: {
            std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
            return candidates[dist(rng)];
        }

        case GeodesicDirection::DimensionGradient: {
            if (!vertex_dimensions) {
                // Fallback to random if no dimensions
                std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
                return candidates[dist(rng)];
            }

            // Find neighbor with highest/lowest dimension
            VertexId best = candidates[0];
            float best_dim = get_vertex_dimension(graph, best, vertex_dimensions);

            for (size_t i = 1; i < candidates.size(); ++i) {
                float dim = get_vertex_dimension(graph, candidates[i], vertex_dimensions);
                if (config.follow_dimension_ascent) {
                    if (dim > best_dim) {
                        best_dim = dim;
                        best = candidates[i];
                    }
                } else {
                    if (dim < best_dim) {
                        best_dim = dim;
                        best = candidates[i];
                    }
                }
            }
            return best;
        }

        case GeodesicDirection::LongestPath: {
            // Heuristic: choose neighbor with most unvisited neighbors
            VertexId best = candidates[0];
            int best_count = 0;

            for (VertexId c : candidates) {
                int count = 0;
                for (VertexId n : graph.neighbors(c)) {
                    if (visited.find(n) == visited.end()) count++;
                }
                if (count > best_count) {
                    best_count = count;
                    best = c;
                }
            }
            return best;
        }

        default:
            return candidates[0];
    }
}

}  // anonymous namespace

// =============================================================================
// Single Geodesic Tracing
// =============================================================================

GeodesicPath trace_geodesic(
    const SimpleGraph& graph,
    VertexId source,
    const GeodesicConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    GeodesicPath path;
    path.source = source;

    if (!graph.has_vertex(source)) {
        return path;  // Invalid source
    }

    auto& rng = get_rng(config.seed);
    std::unordered_set<VertexId> visited;

    VertexId current = source;
    float cumulative_time = 0.0f;

    for (int step = 0; step < config.max_steps; ++step) {
        path.vertices.push_back(current);
        path.proper_time.push_back(cumulative_time);

        // Get dimension at current vertex
        float dim = get_vertex_dimension(graph, current, vertex_dimensions);
        path.local_dimension.push_back(dim);

        visited.insert(current);

        // Select next vertex
        VertexId next = select_next_vertex(graph, current, visited, config, vertex_dimensions, rng);

        if (next == current) {
            // Stuck - can't continue
            break;
        }

        // Each edge contributes 1 to proper time
        cumulative_time += 1.0f;
        current = next;
    }

    path.destination = path.vertices.back();
    path.length = static_cast<int>(path.vertices.size()) - 1;

    return path;
}

// =============================================================================
// Shortest Path Geodesic
// =============================================================================

GeodesicPath trace_geodesic_to_target(
    const SimpleGraph& graph,
    VertexId source,
    VertexId target
) {
    GeodesicPath path;
    path.source = source;
    path.destination = target;

    if (!graph.has_vertex(source) || !graph.has_vertex(target)) {
        return path;
    }

    if (source == target) {
        path.vertices.push_back(source);
        path.proper_time.push_back(0.0f);
        path.length = 0;
        return path;
    }

    // BFS to find shortest path
    std::queue<VertexId> queue;
    std::unordered_map<VertexId, VertexId> parent;
    std::unordered_set<VertexId> visited;

    queue.push(source);
    visited.insert(source);
    parent[source] = source;

    bool found = false;
    while (!queue.empty() && !found) {
        VertexId current = queue.front();
        queue.pop();

        for (VertexId neighbor : graph.neighbors(current)) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                parent[neighbor] = current;
                queue.push(neighbor);

                if (neighbor == target) {
                    found = true;
                    break;
                }
            }
        }
    }

    if (!found) {
        return path;  // No path exists
    }

    // Reconstruct path
    std::vector<VertexId> reversed_path;
    VertexId current = target;
    while (current != source) {
        reversed_path.push_back(current);
        current = parent[current];
    }
    reversed_path.push_back(source);

    // Reverse to get source -> target order
    for (auto it = reversed_path.rbegin(); it != reversed_path.rend(); ++it) {
        path.vertices.push_back(*it);
    }

    // Compute proper time
    for (size_t i = 0; i < path.vertices.size(); ++i) {
        path.proper_time.push_back(static_cast<float>(i));
    }

    path.length = static_cast<int>(path.vertices.size()) - 1;
    return path;
}

// =============================================================================
// Geodesic Bundle Tracing
// =============================================================================

GeodesicBundle trace_geodesic_bundle(
    const SimpleGraph& graph,
    VertexId source,
    const GeodesicConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    GeodesicBundle bundle;
    bundle.source = source;

    if (!graph.has_vertex(source)) {
        return bundle;
    }

    // Find bundle sources: source vertex and nearby neighbors
    std::vector<VertexId> bundle_sources;
    bundle_sources.push_back(source);

    // Add neighbors within radius
    auto distances = graph.distances_from(source);
    const auto& vertices = graph.vertices();

    for (size_t i = 0; i < vertices.size() && bundle_sources.size() < static_cast<size_t>(config.bundle_width); ++i) {
        if (distances[i] > 0 && distances[i] <= config.bundle_neighbor_radius) {
            bundle_sources.push_back(vertices[i]);
        }
    }

    // Trace geodesic from each bundle source
    GeodesicConfig path_config = config;
    for (size_t i = 0; i < bundle_sources.size(); ++i) {
        // Vary seed for each path
        path_config.seed = config.seed + static_cast<uint32_t>(i);
        GeodesicPath path = trace_geodesic(graph, bundle_sources[i], path_config, vertex_dimensions);
        if (path.is_valid()) {
            bundle.paths.push_back(std::move(path));
        }
    }

    // Compute bundle statistics
    if (!bundle.paths.empty()) {
        bundle.spread = compute_bundle_spread(bundle, graph);

        float total_time = 0;
        for (const auto& path : bundle.paths) {
            total_time += path.proper_time.back();
        }
        bundle.mean_proper_time = total_time / bundle.paths.size();

        // Variance in proper time
        float variance = 0;
        for (const auto& path : bundle.paths) {
            float diff = path.proper_time.back() - bundle.mean_proper_time;
            variance += diff * diff;
        }
        bundle.proper_time_variance = variance / bundle.paths.size();
    }

    return bundle;
}

// =============================================================================
// Multiple Geodesics
// =============================================================================

std::vector<GeodesicPath> trace_multiple_geodesics(
    const SimpleGraph& graph,
    const std::vector<VertexId>& sources,
    const GeodesicConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    std::vector<GeodesicPath> paths;
    paths.reserve(sources.size());

    GeodesicConfig path_config = config;
    for (size_t i = 0; i < sources.size(); ++i) {
        path_config.seed = config.seed + static_cast<uint32_t>(i);
        GeodesicPath path = trace_geodesic(graph, sources[i], path_config, vertex_dimensions);
        if (path.is_valid()) {
            paths.push_back(std::move(path));
        }
    }

    return paths;
}

// =============================================================================
// Geodesic Analysis Functions
// =============================================================================

float compute_bundle_spread(
    const GeodesicBundle& bundle,
    const SimpleGraph& graph
) {
    if (bundle.paths.size() < 2) return 0.0f;

    // Compute spread as average pairwise distance between endpoints
    float total_distance = 0;
    int count = 0;

    for (size_t i = 0; i < bundle.paths.size(); ++i) {
        for (size_t j = i + 1; j < bundle.paths.size(); ++j) {
            VertexId end_i = bundle.paths[i].destination;
            VertexId end_j = bundle.paths[j].destination;

            int dist = graph.distance(end_i, end_j);
            if (dist >= 0) {
                total_distance += dist;
                count++;
            }
        }
    }

    return count > 0 ? total_distance / count : 0.0f;
}

std::vector<float> compute_proper_time(const GeodesicPath& path) {
    return path.proper_time;  // Already computed during tracing
}

std::vector<float> compute_path_dimensions(
    const GeodesicPath& path,
    const SimpleGraph& graph,
    int max_radius
) {
    std::vector<float> dimensions;
    dimensions.reserve(path.vertices.size());

    for (VertexId v : path.vertices) {
        auto distances = graph.distances_from(v);
        float dim = estimate_local_dimension(distances, max_radius);
        dimensions.push_back(dim);
    }

    return dimensions;
}

float compute_geodesic_curvature(
    const GeodesicPath& path,
    const SimpleGraph& graph
) {
    if (path.vertices.size() < 3) return 0.0f;

    // Curvature: compare actual path length to shortest path
    int actual_length = path.length;
    int shortest = graph.distance(path.source, path.destination);

    if (shortest <= 0) return 0.0f;

    // Curvature = (actual - shortest) / shortest
    return static_cast<float>(actual_length - shortest) / shortest;
}

// =============================================================================
// Source Selection
// =============================================================================

std::vector<VertexId> auto_select_geodesic_sources(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    int num_sources,
    float dimension_percentile
) {
    const auto& vertices = graph.vertices();
    if (vertices.empty() || vertex_dimensions.empty()) {
        return {};
    }

    // Sort vertices by dimension
    std::vector<std::pair<float, VertexId>> dim_vertex;
    for (size_t i = 0; i < vertices.size() && i < vertex_dimensions.size(); ++i) {
        if (vertex_dimensions[i] > 0 && std::isfinite(vertex_dimensions[i])) {
            dim_vertex.push_back({vertex_dimensions[i], vertices[i]});
        }
    }

    if (dim_vertex.empty()) {
        return select_distributed_sources(graph, num_sources, 3);
    }

    std::sort(dim_vertex.begin(), dim_vertex.end());

    // Select sources near high-dimension threshold
    size_t percentile_idx = static_cast<size_t>(dimension_percentile * dim_vertex.size());
    if (percentile_idx >= dim_vertex.size()) percentile_idx = dim_vertex.size() - 1;

    float threshold = dim_vertex[percentile_idx].first;

    // Collect vertices near threshold (within 10%)
    std::vector<VertexId> candidates;
    for (const auto& [dim, v] : dim_vertex) {
        if (dim >= threshold * 0.9f) {
            candidates.push_back(v);
        }
    }

    // Select distributed subset
    if (static_cast<int>(candidates.size()) <= num_sources) {
        return candidates;
    }

    // Greedy selection for distribution
    std::vector<VertexId> selected;
    selected.push_back(candidates[0]);

    while (static_cast<int>(selected.size()) < num_sources) {
        VertexId best = candidates[0];
        int best_min_dist = -1;

        for (VertexId c : candidates) {
            bool already_selected = false;
            for (VertexId s : selected) {
                if (c == s) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) continue;

            // Find minimum distance to already selected
            int min_dist = INT_MAX;
            for (VertexId s : selected) {
                int d = graph.distance(c, s);
                if (d >= 0 && d < min_dist) min_dist = d;
            }

            if (min_dist > best_min_dist) {
                best_min_dist = min_dist;
                best = c;
            }
        }

        if (best_min_dist < 0) break;  // No more candidates
        selected.push_back(best);
    }

    return selected;
}

std::vector<VertexId> select_distributed_sources(
    const SimpleGraph& graph,
    int num_sources,
    int min_separation
) {
    const auto& vertices = graph.vertices();
    if (vertices.empty()) return {};

    std::vector<VertexId> selected;

    // Start with arbitrary vertex
    selected.push_back(vertices[0]);

    while (static_cast<int>(selected.size()) < num_sources && selected.size() < vertices.size()) {
        VertexId best = vertices[0];
        int best_min_dist = -1;

        for (VertexId v : vertices) {
            bool already_selected = std::find(selected.begin(), selected.end(), v) != selected.end();
            if (already_selected) continue;

            // Find minimum distance to already selected
            int min_dist = INT_MAX;
            for (VertexId s : selected) {
                int d = graph.distance(v, s);
                if (d >= 0 && d < min_dist) min_dist = d;
            }

            if (min_dist >= min_separation && min_dist > best_min_dist) {
                best_min_dist = min_dist;
                best = v;
            }
        }

        if (best_min_dist < min_separation) break;  // Can't find well-separated vertex
        selected.push_back(best);
    }

    return selected;
}

// =============================================================================
// Full Analysis
// =============================================================================

GeodesicAnalysisResult analyze_geodesics(
    const SimpleGraph& graph,
    const std::vector<VertexId>& sources,
    const GeodesicConfig& config,
    const std::vector<float>* vertex_dimensions
) {
    GeodesicAnalysisResult result;

    // Auto-select sources if not provided
    if (sources.empty()) {
        if (vertex_dimensions && !vertex_dimensions->empty()) {
            result.sources = auto_select_geodesic_sources(graph, *vertex_dimensions, 5);
        } else {
            result.sources = select_distributed_sources(graph, 5, 3);
        }
    } else {
        result.sources = sources;
    }

    // Trace geodesics from each source
    result.paths = trace_multiple_geodesics(graph, result.sources, config, vertex_dimensions);

    // Trace bundles from each source
    for (VertexId source : result.sources) {
        GeodesicBundle bundle = trace_geodesic_bundle(graph, source, config, vertex_dimensions);
        if (bundle.is_valid()) {
            result.bundles.push_back(std::move(bundle));
        }
    }

    // Compute statistics
    if (!result.paths.empty()) {
        float total_length = 0;
        result.max_path_length = 0;

        for (const auto& path : result.paths) {
            total_length += path.length;
            if (path.length > result.max_path_length) {
                result.max_path_length = static_cast<float>(path.length);
            }
        }
        result.mean_path_length = total_length / result.paths.size();
    }

    if (!result.bundles.empty()) {
        float total_spread = 0;
        for (const auto& bundle : result.bundles) {
            total_spread += bundle.spread;
        }
        result.mean_spread = total_spread / result.bundles.size();
    }

    // Collect all path edges for visualization
    for (const auto& path : result.paths) {
        for (size_t i = 0; i + 1 < path.vertices.size(); ++i) {
            result.path_edges.push_back({path.vertices[i], path.vertices[i + 1]});
        }
    }

    // Compute mean dimension variance along paths
    if (vertex_dimensions && !result.paths.empty()) {
        float total_var = 0;
        int count = 0;

        for (const auto& path : result.paths) {
            if (path.local_dimension.size() < 2) continue;

            float mean = 0;
            for (float d : path.local_dimension) mean += d;
            mean /= path.local_dimension.size();

            float var = 0;
            for (float d : path.local_dimension) {
                float diff = d - mean;
                var += diff * diff;
            }
            var /= path.local_dimension.size();

            total_var += var;
            count++;
        }

        if (count > 0) {
            result.mean_dimension_variance = total_var / count;
        }
    }

    return result;
}

GeodesicAnalysisResult analyze_geodesics_timestep(
    const TimestepAggregation& timestep,
    const std::vector<VertexId>& sources,
    const GeodesicConfig& config
) {
    // Build SimpleGraph from timestep union
    SimpleGraph graph;
    graph.build(timestep.union_vertices, timestep.union_edges);

    // Use mean dimensions from timestep
    return analyze_geodesics(graph, sources, config, &timestep.mean_dimensions);
}

}  // namespace viz::blackhole
