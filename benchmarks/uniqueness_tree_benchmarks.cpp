// BENCHMARK_CATEGORY: Uniqueness Trees

#include "benchmark_framework.hpp"
#include "random_hypergraph_generator.hpp"
#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category: Uniqueness Tree / State Hashing Benchmarks (Single-threaded)
// =============================================================================
// These benchmarks measure the performance of state canonicalization (WL hashing)
// which is the API equivalent of uniqueness tree operations.

BENCHMARK(state_hashing_by_edge_count, "Measures state hashing performance as graph size increases (arity=2)") {
    for (int edges : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 75, 100}) {
        BENCHMARK_PARAM("edges", edges);

        // Fixed arity=2, symmetry_groups = edges/2 for moderate complexity
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("state_hashing_by_edge_count", 0, edges, symmetry_groups, 3);
        auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(edges, symmetry_groups, 2, seed);

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine engine(&hg, 1);

            // Identity rule to create initial state and trigger hashing
            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .build();
            engine.add_rule(rule);

            engine.evolve(edge_list, 0);  // 0 steps just creates initial state
        });
    }
}

BENCHMARK(state_hashing_by_edge_count_arity3, "Measures state hashing performance as graph size increases (arity=3, higher complexity)") {
    for (int edges : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 75, 100}) {
        BENCHMARK_PARAM("edges", edges);

        // Fixed arity=3, symmetry_groups = edges/2 for moderate complexity
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("state_hashing_by_edge_count_arity3", 0, edges, symmetry_groups, 3);
        auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(edges, symmetry_groups, 3, seed);

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine engine(&hg, 1);

            auto rule = make_rule(0)
                .lhs({0, 1, 2})
                .rhs({0, 1, 2})
                .build();
            engine.add_rule(rule);

            engine.evolve(edge_list, 0);
        });
    }
}

BENCHMARK(state_hashing_by_symmetry, "Shows how graph symmetry affects state hashing time") {
    const int num_edges = 12;
    for (int symmetry_groups : {1, 2, 3, 4, 6, 12}) {
        BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("state_hashing_by_symmetry", 0, num_edges, symmetry_groups, 3);
        auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(num_edges, symmetry_groups, 2, seed);

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine engine(&hg, 1);

            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .build();
            engine.add_rule(rule);

            engine.evolve(edge_list, 0);
        });
    }
}

BENCHMARK(state_hashing_2d_sweep, "2D parameter sweep: edges vs symmetry_groups for surface plots") {
    for (int edges : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}) {
        for (int symmetry_groups : {1, 2, 3, 4, 5, 6}) {
            // Skip invalid combinations
            if (symmetry_groups > edges) continue;

            BENCHMARK_PARAM("edges", edges);
            BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

            // Generate single graph
            uint32_t seed = RandomHypergraphGenerator::compute_seed("state_hashing_2d_sweep", 0, edges, symmetry_groups, 0);
            auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(edges, symmetry_groups, 2, seed);

            BENCHMARK_CODE([&]() {
                Hypergraph hg;
                hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
                ParallelEvolutionEngine engine(&hg, 1);

                auto rule = make_rule(0)
                    .lhs({0, 1})
                    .rhs({0, 1})
                    .build();
                engine.add_rule(rule);

                engine.evolve(edge_list, 0);
            });
        }
    }
}

BENCHMARK(state_hashing_by_arity, "Tests impact of hyperedge arity on state hashing performance") {
    const int num_edges = 20;
    for (int arity : {2, 3, 4, 5, 6}) {
        BENCHMARK_PARAM("arity", arity);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("state_hashing_by_arity", 0, num_edges, 4, arity);
        auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(num_edges, 4, arity, seed);

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine engine(&hg, 1);

            // Build rule with appropriate arity
            RuleBuilder builder = make_rule(0);
            std::vector<uint8_t> lhs_edge;
            for (int i = 0; i < arity; ++i) {
                lhs_edge.push_back(static_cast<uint8_t>(i));
            }
            builder.lhs(lhs_edge);
            builder.rhs(lhs_edge);

            auto rule = builder.build();
            engine.add_rule(rule);

            engine.evolve(edge_list, 0);
        });
    }
}

BENCHMARK(state_hashing_modes_comparison, "Compares different state canonicalization modes") {
    const int num_edges = 30;
    int symmetry_groups = 5;
    uint32_t seed = RandomHypergraphGenerator::compute_seed("state_hashing_modes", 0, num_edges, symmetry_groups, 0);
    auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(num_edges, symmetry_groups, 2, seed);

    for (auto mode : {StateCanonicalizationMode::None, StateCanonicalizationMode::Automatic, StateCanonicalizationMode::Full}) {
        const char* mode_name = (mode == StateCanonicalizationMode::None) ? "none" :
                                (mode == StateCanonicalizationMode::Automatic) ? "automatic" : "full";
        BENCHMARK_PARAM("mode", mode_name);

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(mode);
            ParallelEvolutionEngine engine(&hg, 1);

            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .build();
            engine.add_rule(rule);

            engine.evolve(edge_list, 0);
        });
    }
}
