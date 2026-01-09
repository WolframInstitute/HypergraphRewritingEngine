#ifndef RANDOM_HYPERGRAPH_GENERATOR_HPP
#define RANDOM_HYPERGRAPH_GENERATOR_HPP

#include <hypergraph/types.hpp>
#include <random>
#include <functional>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Deterministic random hypergraph generator
 * Seed is computed from input parameters for reproducibility
 *
 * Generates hypergraph edges as vector<vector<VertexId>> for use with
 * the API (canonicalize_edges, evolve, etc.)
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
     * Generate random hypergraph edges (as vector of vectors)
     * Creates connected graphs with overlapping vertices
     */
    static std::vector<std::vector<hypergraph::VertexId>>
    generate_edges(int num_vertices, int num_edges, int max_arity, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> arity_dist(2, max_arity);
        std::uniform_int_distribution<hypergraph::VertexId> vertex_dist(1, num_vertices);

        std::vector<std::vector<hypergraph::VertexId>> edges;
        edges.reserve(num_edges);

        for (int i = 0; i < num_edges; ++i) {
            int arity = arity_dist(rng);
            std::vector<hypergraph::VertexId> edge;
            edge.reserve(arity);

            for (int j = 0; j < arity; ++j) {
                edge.push_back(vertex_dist(rng));
            }

            edges.push_back(std::move(edge));
        }

        return edges;
    }

    /**
     * Generate connected hypergraph with specified average degree
     * (More realistic for benchmarking)
     */
    static std::vector<std::vector<hypergraph::VertexId>>
    generate_connected_edges(int num_vertices, double avg_degree,
                             int max_arity, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> arity_dist(2, max_arity);
        std::uniform_int_distribution<hypergraph::VertexId> vertex_dist(1, num_vertices);

        // Calculate approximate number of edges needed for desired average degree
        // avg_degree ≈ (num_edges * avg_arity) / num_vertices
        double avg_arity = (max_arity + 2.0) / 2.0;  // Midpoint of arity range
        int num_edges = static_cast<int>((avg_degree * num_vertices) / avg_arity);

        std::vector<std::vector<hypergraph::VertexId>> edges;
        edges.reserve(num_edges);

        for (int i = 0; i < num_edges; ++i) {
            int arity = arity_dist(rng);
            std::vector<hypergraph::VertexId> edge;
            edge.reserve(arity);

            // Pick random vertices (may repeat within edge for simplicity)
            for (int j = 0; j < arity; ++j) {
                edge.push_back(vertex_dist(rng));
            }

            edges.push_back(std::move(edge));
        }

        return edges;
    }

    /**
     * Generate hypergraph with controlled symmetry for canonicalization benchmarks
     *
     * The canonicalization algorithm's runtime is dominated by the number of
     * distinct edge orderings it must try. This is controlled by:
     * 1. num_edges: Total graph size
     * 2. symmetry_groups: Number of groups of identical edges
     *    - symmetry_groups = 1: All edges identical -> very fast (early return)
     *    - symmetry_groups = num_edges: All edges unique -> asymmetric, moderately fast
     *    - symmetry_groups in middle: Maximum complexity (many permutations to try)
     *
     * @param num_edges Total number of edges to generate
     * @param symmetry_groups Number of distinct edge types (1 to num_edges)
     * @param arity Fixed arity for all edges (controls vertex count)
     * @param seed Random seed for reproducibility
     * @return Edge vectors with controlled symmetry structure
     */
    static std::vector<std::vector<hypergraph::VertexId>>
    generate_symmetric_edges(int num_edges, int symmetry_groups,
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

        std::vector<std::vector<hypergraph::VertexId>> edges;
        edges.reserve(num_edges);

        // Generate symmetry_groups different edge templates
        // Create a pool of vertices and shuffle them for variation
        int total_vertices = symmetry_groups * arity;
        std::vector<hypergraph::VertexId> vertex_pool;
        for (int i = 1; i <= total_vertices; ++i) {
            vertex_pool.push_back(static_cast<hypergraph::VertexId>(i));
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
                edges.push_back(template_edge);
            }
        }

        return edges;
    }
};

} // namespace benchmark

#endif // RANDOM_HYPERGRAPH_GENERATOR_HPP
