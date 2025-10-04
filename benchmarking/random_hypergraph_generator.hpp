#ifndef RANDOM_HYPERGRAPH_GENERATOR_HPP
#define RANDOM_HYPERGRAPH_GENERATOR_HPP

#include <hypergraph/hypergraph.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <random>
#include <functional>
#include <string>

namespace benchmark {

/**
 * Deterministic random hypergraph generator
 * Seed is computed from input parameters for reproducibility
 */
class RandomHypergraphGenerator {
public:
    /**
     * Compute deterministic seed from parameters using FNV-1a hash
     */
    static uint32_t compute_seed(const std::string& benchmark_name,
                                 int num_vertices, int num_edges,
                                 int avg_degree, int max_arity) {
        // FNV-1a hash for better mixing
        std::hash<std::string> hasher;
        uint32_t seed = static_cast<uint32_t>(hasher(benchmark_name));

        // FNV-1a constants
        const uint32_t FNV_PRIME = 16777619u;
        const uint32_t FNV_OFFSET_BASIS = 2166136261u;

        seed ^= FNV_OFFSET_BASIS;
        seed = (seed ^ static_cast<uint32_t>(num_vertices)) * FNV_PRIME;
        seed = (seed ^ static_cast<uint32_t>(num_edges)) * FNV_PRIME;
        seed = (seed ^ static_cast<uint32_t>(avg_degree)) * FNV_PRIME;
        seed = (seed ^ static_cast<uint32_t>(max_arity)) * FNV_PRIME;

        return seed;
    }

    /**
     * Generate random hypergraph with specified properties
     * Creates connected graphs with overlapping vertices (realistic structure)
     *
     * @param num_vertices Number of vertices in the vertex pool
     * @param num_edges Number of edges to generate
     * @param max_arity Maximum edge arity (vertices per edge)
     * @param seed Random seed for reproducibility
     * @return Hypergraph with random edges
     */
    static hypergraph::Hypergraph generate(int num_vertices, int num_edges, int max_arity, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> arity_dist(2, max_arity);
        std::uniform_int_distribution<hypergraph::VertexId> vertex_dist(1, num_vertices);

        hypergraph::Hypergraph hg;

        for (int i = 0; i < num_edges; ++i) {
            int arity = arity_dist(rng);
            std::vector<hypergraph::VertexId> edge;
            edge.reserve(arity);

            // Pick random vertices (allow repeats for self-loops)
            for (int j = 0; j < arity; ++j) {
                edge.push_back(vertex_dist(rng));
            }

            hg.add_edge(edge);
        }

        return hg;
    }

    /**
     * Generate random hypergraph edges (as vector of vectors)
     * Creates connected graphs with overlapping vertices
     */
    static std::vector<std::vector<hypergraph::GlobalVertexId>>
    generate_edges(int num_vertices, int num_edges, int max_arity, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> arity_dist(2, max_arity);
        std::uniform_int_distribution<hypergraph::GlobalVertexId> vertex_dist(1, num_vertices);

        std::vector<std::vector<hypergraph::GlobalVertexId>> edges;
        edges.reserve(num_edges);

        for (int i = 0; i < num_edges; ++i) {
            int arity = arity_dist(rng);
            std::vector<hypergraph::GlobalVertexId> edge;
            edge.reserve(arity);

            for (int j = 0; j < arity; ++j) {
                edge.push_back(vertex_dist(rng));
            }

            edges.push_back(std::move(edge));
        }

        return edges;
    }

    /**
     * Generate random pattern hypergraph for pattern matching benchmarks
     * Creates patterns with shared variables for realistic matching
     */
    static hypergraph::PatternHypergraph generate_pattern(int num_edges, int max_arity, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> arity_dist(2, max_arity);

        hypergraph::PatternHypergraph pattern;

        // Use limited variable count to ensure overlap
        // More edges with fewer variables = more constraints = more interesting patterns
        double avg_arity = (2 + max_arity) / 2.0;
        int num_vars = std::max(2, static_cast<int>(num_edges * avg_arity * 0.6));
        std::uniform_int_distribution<int> var_dist(1, num_vars);

        for (int i = 0; i < num_edges; ++i) {
            int arity = arity_dist(rng);
            std::vector<hypergraph::PatternVertex> edge_vertices;
            edge_vertices.reserve(arity);

            // Pick random variables (allow repeats for self-loop patterns)
            for (int j = 0; j < arity; ++j) {
                edge_vertices.push_back(hypergraph::PatternVertex::variable(var_dist(rng)));
            }

            pattern.add_edge(hypergraph::PatternEdge(edge_vertices));
        }

        return pattern;
    }

    /**
     * Extract a pattern from an existing hypergraph by selecting a subset of edges
     * This guarantees the pattern will match the graph (at least once)
     */
    static hypergraph::PatternHypergraph extract_pattern(const hypergraph::Hypergraph& hg,
                                                         int num_edges, uint32_t seed) {
        std::mt19937 rng(seed);

        if (num_edges > static_cast<int>(hg.num_edges())) {
            num_edges = hg.num_edges();
        }

        // Select random edges from the graph
        std::vector<int> edge_indices;
        for (int i = 0; i < static_cast<int>(hg.num_edges()); ++i) {
            edge_indices.push_back(i);
        }
        std::shuffle(edge_indices.begin(), edge_indices.end(), rng);

        // Build pattern from selected edges
        hypergraph::PatternHypergraph pattern;
        std::unordered_map<hypergraph::VertexId, int> vertex_to_var;
        int next_var = 1;

        for (int i = 0; i < num_edges; ++i) {
            const auto& edge = hg.edges()[edge_indices[i]];
            std::vector<hypergraph::PatternVertex> pattern_vertices;

            for (auto vertex_id : edge.vertices()) {
                if (vertex_to_var.find(vertex_id) == vertex_to_var.end()) {
                    vertex_to_var[vertex_id] = next_var++;
                }
                pattern_vertices.push_back(hypergraph::PatternVertex::variable(vertex_to_var[vertex_id]));
            }

            pattern.add_edge(hypergraph::PatternEdge(pattern_vertices));
        }

        return pattern;
    }

    /**
     * Generate connected hypergraph with specified average degree
     * (More realistic for benchmarking)
     */
    static hypergraph::Hypergraph generate_connected(int num_vertices, double avg_degree,
                                                     int max_arity, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> arity_dist(2, max_arity);
        std::uniform_int_distribution<hypergraph::VertexId> vertex_dist(1, num_vertices);

        hypergraph::Hypergraph hg;

        // Calculate approximate number of edges needed for desired average degree
        // avg_degree ≈ (num_edges * avg_arity) / num_vertices
        double avg_arity = (max_arity + 2.0) / 2.0;  // Midpoint of arity range
        int num_edges = static_cast<int>((avg_degree * num_vertices) / avg_arity);

        for (int i = 0; i < num_edges; ++i) {
            int arity = arity_dist(rng);
            std::vector<hypergraph::VertexId> edge;
            edge.reserve(arity);

            // Pick random vertices (may repeat within edge for simplicity)
            for (int j = 0; j < arity; ++j) {
                edge.push_back(vertex_dist(rng));
            }

            hg.add_edge(edge);
        }

        return hg;
    }

    /**
     * Generate hypergraph with controlled symmetry for canonicalization benchmarks
     *
     * The canonicalization algorithm's runtime is dominated by the number of
     * distinct edge orderings it must try. This is controlled by:
     * 1. num_edges: Total graph size
     * 2. symmetry_groups: Number of groups of identical edges
     *    - symmetry_groups = 1: All edges identical → very fast (early return)
     *    - symmetry_groups = num_edges: All edges unique → asymmetric, moderately fast
     *    - symmetry_groups in middle: Maximum complexity (many permutations to try)
     *
     * @param num_edges Total number of edges to generate
     * @param symmetry_groups Number of distinct edge types (1 to num_edges)
     * @param arity Fixed arity for all edges (controls vertex count)
     * @param seed Random seed for reproducibility
     * @return Hypergraph with controlled symmetry structure
     */
    static hypergraph::Hypergraph generate_symmetric(int num_edges, int symmetry_groups,
                                                     int arity, uint32_t seed) {
        std::mt19937 rng(seed);

        // Clamp symmetry_groups to valid range
        symmetry_groups = std::max(1, std::min(symmetry_groups, num_edges));

        // Calculate how many copies of each edge type
        // Distribute edges evenly across groups
        std::vector<int> copies_per_group(symmetry_groups, num_edges / symmetry_groups);
        int remainder = num_edges % symmetry_groups;
        for (int i = 0; i < remainder; ++i) {
            copies_per_group[i]++;
        }

        hypergraph::Hypergraph hg;

        // Generate symmetry_groups different edge templates
        // Create a pool of vertices and shuffle them for variation
        int total_vertices = symmetry_groups * arity;
        std::vector<hypergraph::VertexId> vertex_pool;
        for (int i = 1; i <= total_vertices; ++i) {
            vertex_pool.push_back(i);
        }
        std::shuffle(vertex_pool.begin(), vertex_pool.end(), rng);

        int vertex_idx = 0;
        for (int group = 0; group < symmetry_groups; ++group) {
            // Create template edge for this group using shuffled vertices
            std::vector<hypergraph::VertexId> template_edge;
            template_edge.reserve(arity);

            for (int j = 0; j < arity; ++j) {
                template_edge.push_back(vertex_pool[vertex_idx++]);
            }

            // Add copies_per_group[group] identical copies
            for (int copy = 0; copy < copies_per_group[group]; ++copy) {
                hg.add_edge(template_edge);
            }
        }

        return hg;
    }
};

} // namespace benchmark

#endif // RANDOM_HYPERGRAPH_GENERATOR_HPP
