// BENCHMARK_CATEGORY: Pattern Matching

#include "benchmark_framework.hpp"
#include "random_hypergraph_generator.hpp"
#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/unified_hypergraph.hpp>
#include <thread>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category B: Pattern Matching Benchmarks (Parallelized via job system)
// =============================================================================

BENCHMARK(pattern_matching_by_lhs_size, "Tests evolution performance with increasing LHS pattern complexity (1-4 edges in LHS)") {
    size_t num_threads = std::thread::hardware_concurrency();

    for (int lhs_edges : {1, 2, 3, 4}) {
        BENCHMARK_PARAM("lhs_edges", lhs_edges);

        std::vector<std::vector<VertexId>> initial;
        // Create initial state with chain of edges
        for (int i = 1; i <= 10; ++i) {
            initial.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
        }

        BENCHMARK_CODE([&]() {
            UnifiedHypergraph hg;
            ParallelEvolutionEngine engine(&hg, num_threads);

            // Build rule with lhs_edges edges in LHS
            RuleBuilder builder = make_rule(0);

            // LHS: chain of edges {0,1}, {1,2}, {2,3}, ...
            for (int i = 0; i < lhs_edges; ++i) {
                builder.lhs({static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1)});
            }

            // RHS: same edges plus one new edge
            for (int i = 0; i < lhs_edges; ++i) {
                builder.rhs({static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1)});
            }
            builder.rhs({static_cast<uint8_t>(lhs_edges), static_cast<uint8_t>(lhs_edges + 1)});

            auto rule = builder.build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, 1);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(pattern_matching_by_graph_size, "Evaluates pattern matching scalability as target graph size increases") {
    size_t num_threads = std::thread::hardware_concurrency();

    for (int graph_edges : {10, 20, 30, 40, 50}) {
        BENCHMARK_PARAM("graph_edges", graph_edges);

        std::vector<std::vector<VertexId>> initial;
        // Create initial state with edges
        for (int i = 1; i <= graph_edges; ++i) {
            initial.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
        }

        BENCHMARK_CODE([&]() {
            UnifiedHypergraph hg;
            ParallelEvolutionEngine engine(&hg, num_threads);

            // Simple 2-edge pattern
            auto rule = make_rule(0)
                .lhs({0, 1})
                .lhs({1, 2})
                .rhs({0, 1})
                .rhs({1, 2})
                .rhs({2, 3})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, 1);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(pattern_matching_2d_sweep_threads_size, "2D parameter sweep of evolution across thread count and graph size") {
    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts;
    for (size_t i = 1; i <= max_threads; ++i) {
        thread_counts.push_back(i);
    }

    for (size_t num_threads : thread_counts) {
        for (int graph_edges : {10, 20, 30, 40, 50}) {
            BENCHMARK_PARAM("num_threads", static_cast<int>(num_threads));
            BENCHMARK_PARAM("graph_edges", graph_edges);

            std::vector<std::vector<VertexId>> initial;
            for (int i = 1; i <= graph_edges; ++i) {
                initial.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
            }

            BENCHMARK_CODE([&]() {
                UnifiedHypergraph hg;
                ParallelEvolutionEngine engine(&hg, num_threads);

                auto rule = make_rule(0)
                    .lhs({0, 1})
                    .rhs({0, 1})
                    .rhs({1, 2})
                    .build();
                engine.add_rule(rule);

                BENCHMARK_BEGIN();
                engine.evolve(initial, 1);
                BENCHMARK_END();
            });
        }
    }
}

BENCHMARK(pattern_matching_by_arity, "Tests pattern matching with different edge arities") {
    size_t num_threads = std::thread::hardware_concurrency();

    for (int arity : {2, 3, 4, 5}) {
        BENCHMARK_PARAM("arity", arity);

        std::vector<std::vector<VertexId>> initial;
        // Create initial state with edges of specified arity
        for (int i = 0; i < 20; ++i) {
            std::vector<VertexId> edge;
            for (int j = 0; j < arity; ++j) {
                edge.push_back(static_cast<VertexId>(i * arity + j + 1));
            }
            initial.push_back(edge);
        }

        // Also add some edges with shared vertices for pattern matching
        for (int i = 0; i < 10; ++i) {
            std::vector<VertexId> edge;
            for (int j = 0; j < arity; ++j) {
                edge.push_back(static_cast<VertexId>(j + 1));  // All use vertices 1..arity
            }
            initial.push_back(edge);
        }

        BENCHMARK_CODE([&]() {
            UnifiedHypergraph hg;
            ParallelEvolutionEngine engine(&hg, num_threads);

            // Build LHS pattern with specified arity
            RuleBuilder builder = make_rule(0);
            std::vector<uint8_t> lhs_edge;
            for (int j = 0; j < arity; ++j) {
                lhs_edge.push_back(static_cast<uint8_t>(j));
            }
            builder.lhs(lhs_edge);

            // RHS: same edge plus new edge
            builder.rhs(lhs_edge);
            std::vector<uint8_t> new_edge;
            for (int j = 0; j < arity; ++j) {
                new_edge.push_back(static_cast<uint8_t>(arity + j));
            }
            builder.rhs(new_edge);

            auto rule = builder.build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, 1);
            BENCHMARK_END();
        });
    }
}
