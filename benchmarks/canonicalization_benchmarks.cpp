// BENCHMARK_CATEGORY: Canonicalization

#include "benchmark_framework.hpp"
#include "random_hypergraph_generator.hpp"
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/canonicalization.hpp>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category A: Canonicalization Benchmarks (Single-threaded)
// =============================================================================
// These benchmarks use controlled-symmetry graphs to get predictable results

BENCHMARK(canonicalization_by_edge_count, "Measures canonicalization performance as graph size increases (arity=2)") {
    for (int edges : {2, 4, 6}) {
        BENCHMARK_PARAM("edges", edges);

        // Fixed arity=2, symmetry_groups = edges/2 for moderate complexity
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_by_edge_count", 0, edges, symmetry_groups, 3);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);
        Canonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            canonicalizer.canonicalize(hg);
        });
    }
}

BENCHMARK(canonicalization_by_edge_count_arity3, "Measures canonicalization performance as graph size increases (arity=3, higher complexity)") {
    for (int edges : {2, 4, 6}) {
        BENCHMARK_PARAM("edges", edges);

        // Fixed arity=3, symmetry_groups = edges/2 for moderate complexity
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_by_edge_count_arity3", 0, edges, symmetry_groups, 3);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 3, seed);
        Canonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            canonicalizer.canonicalize(hg);
        });
    }
}

BENCHMARK(canonicalization_by_symmetry, "Shows how graph symmetry affects canonicalization time") {
    const int num_edges = 12;
    for (int symmetry_groups : {1, 2, 3, 4, 6}) {
        BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_by_symmetry", 0, num_edges, symmetry_groups, 3);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(num_edges, symmetry_groups, 2, seed);
        Canonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            canonicalizer.canonicalize(hg);
        });
    }
}

BENCHMARK(canonicalization_2d_sweep, "2D parameter sweep: edges vs symmetry_groups for surface plots") {
    for (int edges : {2, 3, 4, 5, 6}) {
        for (int symmetry_groups : {1, 2, 3, 4, 5, 6}) {
            // Skip invalid combinations
            if (symmetry_groups > edges) continue;

            BENCHMARK_PARAM("edges", edges);
            BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

            // Generate single graph - capture by value to avoid dangling reference
            uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_2d_sweep", 0, edges, symmetry_groups, 0);
            Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);

            BENCHMARK_CODE([&]() {
                Canonicalizer canonicalizer;
                canonicalizer.canonicalize(hg);
            });
        }
    }
}
