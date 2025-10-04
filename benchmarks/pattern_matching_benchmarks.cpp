// BENCHMARK_CATEGORY: Pattern Matching

#include "benchmark_framework.hpp"
#include "random_hypergraph_generator.hpp"
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/rewriting.hpp>
#include <hypergraph/wolfram_evolution.hpp>
#include <thread>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category B: Pattern Matching Benchmarks (Parallelized via job system)
// =============================================================================

BENCHMARK(pattern_matching_by_pattern_size, "2D parameter sweep: pattern matching time vs pattern complexity (1-5 edges) and graph size (5-15 edges)") {
    for (int pattern_edges : {1, 2, 3, 4, 5, 6, 7}) {
        for (int graph_edges : {5, 10, 15, 20, 25, 30}) {
            BENCHMARK_PARAM("pattern_edges", pattern_edges);
            BENCHMARK_PARAM("graph_edges", graph_edges);

            // Pre-generate all samples
            constexpr int num_samples = 50;
            std::vector<Hypergraph> graphs;
            std::vector<PatternHypergraph> patterns;
            graphs.reserve(num_samples);
            patterns.reserve(num_samples);

            int num_vertices = 10 + graph_edges;
            for (int i = 0; i < num_samples; ++i) {
                uint32_t graph_seed = RandomHypergraphGenerator::compute_seed("pattern_matching_by_pattern_size", num_vertices, graph_edges, i, 3);
                uint32_t pattern_seed = RandomHypergraphGenerator::compute_seed("pattern_matching_by_pattern_size_pattern", 0, pattern_edges, i, 2);

                Hypergraph hg = RandomHypergraphGenerator::generate(num_vertices, graph_edges, 3, graph_seed);
                PatternHypergraph pattern = RandomHypergraphGenerator::extract_pattern(hg, pattern_edges, pattern_seed);

                graphs.push_back(std::move(hg));
                patterns.push_back(std::move(pattern));
            }

            int sample_index = 0;

            BENCHMARK_CODE([&]() {
                const auto& hg = graphs[sample_index];
                const auto& pattern = patterns[sample_index];

                WolframEvolution evolution(0, std::thread::hardware_concurrency(), false, false);
                RewritingRule dummy_rule(pattern, pattern);
                evolution.add_rule(dummy_rule);

                std::vector<std::vector<GlobalVertexId>> initial;
                for (const auto& edge : hg.edges()) {
                    initial.push_back(edge.vertices());
                }
                evolution.evolve(initial, true);  // pattern_matching_only=true

                // Verify pattern matching found at least one match
                if (evolution.get_total_matches_found() == 0) {
                    throw std::runtime_error("Pattern matching found no matches - benchmark invalid!");
                }

                sample_index++;
            }, num_samples);
        }
    }
}

BENCHMARK(pattern_matching_by_graph_size, "Evaluates pattern matching scalability as target graph size increases from 5 to 15 edges") {
    for (int graph_edges : {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}) {
        BENCHMARK_PARAM("graph_edges", graph_edges);

        // Pre-generate all samples
        constexpr int num_samples = 50;
        std::vector<Hypergraph> graphs;
        std::vector<PatternHypergraph> patterns;
        graphs.reserve(num_samples);
        patterns.reserve(num_samples);

        int num_vertices = 10 + graph_edges;
        for (int i = 0; i < num_samples; ++i) {
            uint32_t graph_seed = RandomHypergraphGenerator::compute_seed("pattern_matching_by_graph_size", num_vertices, graph_edges, i, 3);
            uint32_t pattern_seed = RandomHypergraphGenerator::compute_seed("pattern_matching_by_graph_size_pattern", 0, 2, i, 2);

            Hypergraph hg = RandomHypergraphGenerator::generate(num_vertices, graph_edges, 3, graph_seed);
            PatternHypergraph pattern = RandomHypergraphGenerator::extract_pattern(hg, 2, pattern_seed);

            graphs.push_back(std::move(hg));
            patterns.push_back(std::move(pattern));
        }

        int sample_index = 0;

        BENCHMARK_CODE([&]() {
            const auto& hg = graphs[sample_index];
            const auto& pattern = patterns[sample_index];

            WolframEvolution evolution(0, std::thread::hardware_concurrency(), false, false);
            RewritingRule dummy_rule(pattern, pattern);
            evolution.add_rule(dummy_rule);

            std::vector<std::vector<GlobalVertexId>> initial;
            for (const auto& edge : hg.edges()) {
                initial.push_back(edge.vertices());
            }
            evolution.evolve(initial, true);  // pattern_matching_only=true

            // Verify pattern matching found at least one match
            if (evolution.get_total_matches_found() == 0) {
                throw std::runtime_error("Pattern matching found no matches - benchmark invalid!");
            }

            sample_index++;
        }, num_samples);
    }
}

BENCHMARK(pattern_matching_2d_sweep_threads_size, "2D parameter sweep of pattern matching across thread count (1-32) and graph size (5-100 edges) for parallel scalability analysis") {
    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts;
    for (size_t i = 1; i <= max_threads; ++i) {
        thread_counts.push_back(i);
    }

    for (size_t num_threads : thread_counts) {
        for (int graph_edges : {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100}) {
            BENCHMARK_PARAM("num_threads", num_threads);
            BENCHMARK_PARAM("graph_edges", graph_edges);

            // Pre-generate all samples
            constexpr int num_samples = 50;
            std::vector<Hypergraph> graphs;
            std::vector<PatternHypergraph> patterns;
            graphs.reserve(num_samples);
            patterns.reserve(num_samples);

            int num_vertices = 10 + graph_edges;
            for (int i = 0; i < num_samples; ++i) {
                uint32_t graph_seed = RandomHypergraphGenerator::compute_seed("pattern_matching_2d_sweep", num_vertices, graph_edges, i, 3);
                uint32_t pattern_seed = RandomHypergraphGenerator::compute_seed("pattern_matching_2d_sweep_pattern", 0, 2, i, 2);

                Hypergraph hg = RandomHypergraphGenerator::generate(num_vertices, graph_edges, 3, graph_seed);
                PatternHypergraph pattern = RandomHypergraphGenerator::extract_pattern(hg, 2, pattern_seed);

                graphs.push_back(std::move(hg));
                patterns.push_back(std::move(pattern));
            }

            int sample_index = 0;

            BENCHMARK_CODE([&]() {
                const auto& hg = graphs[sample_index];
                const auto& pattern = patterns[sample_index];

                // 0 steps evolution to isolate pattern matching
                WolframEvolution evolution(0, num_threads, false, false);
                RewritingRule dummy_rule(pattern, pattern);
                evolution.add_rule(dummy_rule);

                std::vector<std::vector<GlobalVertexId>> initial;
                for (const auto& edge : hg.edges()) {
                    initial.push_back(edge.vertices());
                }
                evolution.evolve(initial, true);  // pattern_matching_only=true

                // Verify pattern matching found at least one match
                if (evolution.get_total_matches_found() == 0) {
                    throw std::runtime_error("Pattern matching found no matches - benchmark invalid!");
                }

                sample_index++;
            }, num_samples);
        }
    }
}
