// BENCHMARK_CATEGORY: Event Relationships

#include "benchmark_framework.hpp"
#include <hypergraph/rewriting.hpp>
#include <hypergraph/wolfram_evolution.hpp>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category D: Event Relationships Benchmarks
// =============================================================================

BENCHMARK(causal_edges_overhead, "Measures the overhead of computing causal edges during evolution (1-3 steps)") {
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}};

    for (int steps : {1, 2, 3}) {
        for (bool causal_edges : {false, true}) {
            BENCHMARK_PARAM("steps", steps);
            BENCHMARK_PARAM("causal_edges", causal_edges ? "true" : "false");

            BENCHMARK_CODE([&]() {
                WolframEvolution evolution(steps, 1, true, true, causal_edges);
                evolution.add_rule(rule);
                BENCHMARK_BEGIN();
                evolution.evolve(initial);
                BENCHMARK_END();
            });
        }
    }
}

BENCHMARK(transitive_reduction_overhead, "Isolates transitive reduction overhead by comparing evolution with it enabled vs disabled") {
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}, {3, 4}};

    for (int steps : {1, 2, 3}) {
        for (bool transitive_reduction : {false, true}) {
            BENCHMARK_PARAM("steps", steps);
            BENCHMARK_PARAM("transitive_reduction", transitive_reduction ? "true" : "false");

            BENCHMARK_CODE([&]() {
                WolframEvolution evolution(steps, 1, true, true, true, transitive_reduction);
                evolution.add_rule(rule);
                BENCHMARK_BEGIN();
                evolution.evolve(initial);
                BENCHMARK_END();
            });
        }
    }
}
