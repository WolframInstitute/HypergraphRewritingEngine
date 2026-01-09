// BENCHMARK_CATEGORY: Evolution

#include "benchmark_framework.hpp"
#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>
#include <thread>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category E: Multi-Rule and Multi-Threaded Evolution Benchmarks
// =============================================================================

BENCHMARK(evolution_multi_rule_by_rule_count, "Tests evolution performance with increasing rule complexity (1-3 rules with mixed arities)") {
    size_t num_threads = std::thread::hardware_concurrency();

    for (int num_rules : {1, 2, 3}) {
        BENCHMARK_PARAM("num_rules", num_rules);

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            ParallelEvolutionEngine engine(&hg, num_threads);

            // Rule 1: {x,y} -> {x,z},{z,y}
            auto rule1 = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 2})
                .rhs({2, 1})
                .build();
            engine.add_rule(rule1);

            if (num_rules >= 2) {
                // Rule 2: {x,y,z} -> {x,y,w},{w,z}
                auto rule2 = make_rule(1)
                    .lhs({0, 1, 2})
                    .rhs({0, 1, 3})
                    .rhs({3, 2})
                    .build();
                engine.add_rule(rule2);
            }

            if (num_rules >= 3) {
                // Rule 3: {x,y} -> {x,z,w},{w,y}
                auto rule3 = make_rule(2)
                    .lhs({0, 1})
                    .rhs({0, 2, 3})
                    .rhs({3, 1})
                    .build();
                engine.add_rule(rule3);
            }

            // Initial state with both arity-2 and arity-3 edges
            std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3, 4}};

            BENCHMARK_BEGIN();
            engine.evolve(initial, 2);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(evolution_thread_scaling, "Evaluates parallel speedup from 1 thread up to full hardware concurrency (3-step evolution)") {
    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts;
    for (size_t i = 1; i <= max_threads; ++i) {
        thread_counts.push_back(i);
    }

    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}};

    for (size_t num_threads : thread_counts) {
        BENCHMARK_PARAM("num_threads", static_cast<int>(num_threads));

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            ParallelEvolutionEngine engine(&hg, num_threads);

            // Rule: {x,y} -> {x,y},{y,z}
            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .rhs({1, 2})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, 3);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(evolution_2d_sweep_threads_steps, "2D sweep: evolution with rule {{x,y},{y,z}} -> {{z,y},{y,x},{x,w}} across thread count and steps") {
    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts;
    for (size_t i = 1; i <= max_threads; ++i) {
        thread_counts.push_back(i);
    }

    std::vector<std::vector<VertexId>> initial = {{1, 1}, {1, 1}};

    for (size_t num_threads : thread_counts) {
        for (int steps : {1, 2, 3, 4}) {
            BENCHMARK_PARAM("num_threads", static_cast<int>(num_threads));
            BENCHMARK_PARAM("steps", steps);

            BENCHMARK_CODE([&]() {
                Hypergraph hg;
                ParallelEvolutionEngine engine(&hg, num_threads);

                // Rule: {{x,y},{y,z}} -> {{z,y},{y,x},{x,w}}
                auto rule = make_rule(0)
                    .lhs({0, 1})
                    .lhs({1, 2})
                    .rhs({2, 1})
                    .rhs({1, 0})
                    .rhs({0, 3})
                    .build();
                engine.add_rule(rule);

                BENCHMARK_BEGIN();
                engine.evolve(initial, steps);
                BENCHMARK_END();
            });
        }
    }
}

BENCHMARK(evolution_with_self_loops, "Tests evolution performance on hypergraphs containing self-loop edges") {
    size_t num_threads = std::thread::hardware_concurrency();

    for (int steps : {1, 2}) {
        BENCHMARK_PARAM("steps", steps);

        std::vector<std::vector<VertexId>> initial = {{1, 1, 1}, {1, 1}};

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            ParallelEvolutionEngine engine(&hg, num_threads);

            // Rule: {x,x,x} -> {x,x},{x,y}
            auto rule = make_rule(0)
                .lhs({0, 0, 0})
                .rhs({0, 0})
                .rhs({0, 1})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, steps);
            BENCHMARK_END();
        });
    }
}
