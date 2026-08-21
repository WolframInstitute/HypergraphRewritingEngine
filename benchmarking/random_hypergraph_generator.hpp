#pragma once

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
                                 int avg_degree, int max_arity);

    /**
     * Generate random hypergraph edges (as vector of vectors)
     * Creates connected graphs with overlapping vertices
     */
    static std::vector<std::vector<hypergraph::VertexId>>
    generate_edges(int num_vertices, int num_edges, int max_arity, uint32_t seed);

    /**
     * Generate connected hypergraph with specified average degree
     * (More realistic for benchmarking)
     */
    static std::vector<std::vector<hypergraph::VertexId>>
    generate_connected_edges(int num_vertices, double avg_degree,
                             int max_arity, uint32_t seed);

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
                             int arity, uint32_t seed);
};

} // namespace benchmark

