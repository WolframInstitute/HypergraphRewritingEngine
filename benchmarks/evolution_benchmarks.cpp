// BENCHMARK_CATEGORY: Evolution

#include "benchmark_framework.hpp"
#include <hypergraph/rewriting.hpp>
#include <hypergraph/wolfram_evolution.hpp>
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

        std::vector<RewritingRule> rules;

        // Create rules with different arities for interesting behavior
        // Rule 1: arity-2 edges
        PatternHypergraph lhs1, rhs1;
        lhs1.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        rhs1.add_edge({PatternVertex::variable(1), PatternVertex::variable(3)});
        rhs1.add_edge({PatternVertex::variable(3), PatternVertex::variable(2)});
        rules.emplace_back(lhs1, rhs1);

        if (num_rules >= 2) {
            // Rule 2: arity-3 edges
            PatternHypergraph lhs2, rhs2;
            lhs2.add_edge({PatternVertex::variable(1), PatternVertex::variable(2), PatternVertex::variable(3)});
            rhs2.add_edge({PatternVertex::variable(1), PatternVertex::variable(2), PatternVertex::variable(4)});
            rhs2.add_edge({PatternVertex::variable(4), PatternVertex::variable(3)});
            rules.emplace_back(lhs2, rhs2);
        }

        if (num_rules >= 3) {
            // Rule 3: mixed arity (both arity-2 and arity-3 in RHS)
            PatternHypergraph lhs3, rhs3;
            lhs3.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
            rhs3.add_edge({PatternVertex::variable(1), PatternVertex::variable(3), PatternVertex::variable(4)});
            rhs3.add_edge({PatternVertex::variable(4), PatternVertex::variable(2)});
            rules.emplace_back(lhs3, rhs3);
        }

        // Initial state with both arity-2 and arity-3 edges
        std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3, 4}};

        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(2, num_threads, true, true);
            for (const auto& rule : rules) {
                evolution.add_rule(rule);
            }
            BENCHMARK_BEGIN();
            evolution.evolve(initial);
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

    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}};

    for (size_t num_threads : thread_counts) {
        BENCHMARK_PARAM("num_threads", static_cast<int>(num_threads));

        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(3, num_threads, true, true);
            evolution.add_rule(rule);
            BENCHMARK_BEGIN();
            evolution.evolve(initial);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(evolution_2d_sweep_threads_steps, "2D sweep: evolution with rule {{1,2},{2,3}} -> {{3,2},{2,1},{1,4}} on init {{1,1},{1,1}} across thread count and steps") {
    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts;
    for (size_t i = 1; i <= max_threads; ++i) {
        thread_counts.push_back(i);
    }

    // Rule: {{1,2},{2,3}} -> {{3,2},{2,1},{1,4}}
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    lhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    rhs.add_edge({PatternVertex::variable(3), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(1)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(4)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 1}, {1, 1}};

    for (size_t num_threads : thread_counts) {
        for (int steps : {1, 2, 3, 4}) {
            BENCHMARK_PARAM("num_threads", static_cast<int>(num_threads));
            BENCHMARK_PARAM("steps", steps);

            BENCHMARK_CODE([&]() {
                WolframEvolution evolution(steps, num_threads, true, true);
                evolution.add_rule(rule);
                BENCHMARK_BEGIN();
                evolution.evolve(initial);
                BENCHMARK_END();
            });
        }
    }
}

BENCHMARK(evolution_with_self_loops, "Tests evolution performance on hypergraphs containing self-loop edges") {
    size_t num_threads = std::thread::hardware_concurrency();

    for (int steps : {1, 2}) {
        BENCHMARK_PARAM("steps", steps);

        PatternHypergraph lhs, rhs;
        lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(1), PatternVertex::variable(1)});
        rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(1)});
        rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        RewritingRule rule(lhs, rhs);

        // Initial state with self-loops
        std::vector<std::vector<GlobalVertexId>> initial = {{1, 1, 1}, {1, 1}};

        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(steps, num_threads, true, true);
            evolution.add_rule(rule);
            BENCHMARK_BEGIN();
            evolution.evolve(initial);
            BENCHMARK_END();
        });
    }
}
