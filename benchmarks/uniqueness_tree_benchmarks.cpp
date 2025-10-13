// BENCHMARK_CATEGORY: Uniqueness Trees

#include "benchmark_framework.hpp"
#include "random_hypergraph_generator.hpp"
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/uniqueness_tree.hpp>
#include <hypergraph/wolfram_states.hpp>

using namespace hypergraph;
using namespace benchmark;

// Helper to convert Hypergraph to GlobalHyperedge format for uniqueness trees
std::vector<GlobalHyperedge> hypergraph_to_global_edges(const Hypergraph& hg) {
    std::vector<GlobalHyperedge> edges;
    GlobalEdgeId edge_id = 0;

    for (const auto& edge : hg.edges()) {
        std::vector<GlobalVertexId> vertices;
        for (auto vertex : edge.vertices()) {
            vertices.push_back(static_cast<GlobalVertexId>(vertex));
        }
        edges.emplace_back(edge_id++, vertices);
    }

    return edges;
}

// =============================================================================
// Category: Uniqueness Tree Benchmarks (Single-threaded)
// =============================================================================
// These benchmarks use the same controlled-symmetry graphs as canonicalization
// to allow direct performance comparison

BENCHMARK(uniqueness_tree_by_edge_count, "Measures uniqueness tree performance as graph size increases") {
    for (int edges : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}) {
        BENCHMARK_PARAM("edges", edges);

        // Fixed arity=3, symmetry_groups = edges/2 for moderate complexity
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("uniqueness_tree_by_edge_count", 0, edges, symmetry_groups, 3);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);
        auto global_edges = hypergraph_to_global_edges(hg);

        BENCHMARK_CODE([&]() {
            UniquenessTreeSet tree_set(global_edges);
            tree_set.canonical_hash();
        }, 50);
    }
}

BENCHMARK(uniqueness_tree_by_symmetry, "Shows how graph symmetry affects uniqueness tree time") {
    const int num_edges = 12;
    for (int symmetry_groups : {1, 2, 3, 4, 6, 12}) {
        BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("uniqueness_tree_by_symmetry", 0, num_edges, symmetry_groups, 3);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(num_edges, symmetry_groups, 2, seed);
        auto global_edges = hypergraph_to_global_edges(hg);

        BENCHMARK_CODE([&]() {
            UniquenessTreeSet tree_set(global_edges);
            tree_set.canonical_hash();
        }, 50);
    }
}

BENCHMARK(uniqueness_tree_2d_sweep, "2D parameter sweep: edges vs symmetry_groups for surface plots") {
    for (int edges : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}) {
        for (int symmetry_groups : {1, 2, 3, 4, 5, 6}) {
            // Skip invalid combinations
            if (symmetry_groups > edges) continue;

            BENCHMARK_PARAM("edges", edges);
            BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

            // Generate single graph
            uint32_t seed = RandomHypergraphGenerator::compute_seed("uniqueness_tree_2d_sweep", 0, edges, symmetry_groups, 0);
            Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);
            auto global_edges = hypergraph_to_global_edges(hg);

            BENCHMARK_CODE([&]() {
                UniquenessTreeSet tree_set(global_edges);
                tree_set.canonical_hash();
            }, 50);
        }
    }
}

BENCHMARK(uniqueness_tree_by_vertex_count, "Measures performance as vertex count increases") {
    for (int vertices : {5, 10, 15, 20, 25, 30, 35, 40}) {
        BENCHMARK_PARAM("vertices", vertices);

        // Generate graph with ~2 edges per vertex
        int num_edges = vertices * 2;
        int symmetry_groups = std::max(1, vertices / 5);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("uniqueness_tree_by_vertex_count", 0, num_edges, symmetry_groups, 3);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(num_edges, symmetry_groups, 2, seed);
        auto global_edges = hypergraph_to_global_edges(hg);

        BENCHMARK_CODE([&]() {
            UniquenessTreeSet tree_set(global_edges);
            tree_set.canonical_hash();
        }, 50);
    }
}

BENCHMARK(uniqueness_tree_by_arity, "Tests impact of hyperedge arity on performance") {
    const int num_edges = 20;
    for (int arity : {2, 3, 4, 5, 6, 8}) {
        BENCHMARK_PARAM("arity", arity);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("uniqueness_tree_by_arity", 0, num_edges, 4, arity);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(num_edges, 4, arity, seed);
        auto global_edges = hypergraph_to_global_edges(hg);

        BENCHMARK_CODE([&]() {
            UniquenessTreeSet tree_set(global_edges);
            tree_set.canonical_hash();
        }, 50);
    }
}
