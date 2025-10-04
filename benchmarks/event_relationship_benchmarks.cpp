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
    for (int steps : {1, 2, 3}) {
        BENCHMARK_PARAM("steps", steps);

        PatternHypergraph lhs, rhs;
        lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
        RewritingRule rule(lhs, rhs);

        std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}};

        BENCHMARK_PARAM("causal_edges", "false");
        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(steps, 1, true, true, false);  // No causal edges
            evolution.add_rule(rule);
            evolution.evolve(initial);
        }, 3);

        BENCHMARK_PARAM("causal_edges", "true");
        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(steps, 1, true, true, true);  // With causal edges
            evolution.add_rule(rule);
            evolution.evolve(initial);
        }, 3);
    }
}

BENCHMARK(transitive_reduction_overhead, "Isolates transitive reduction overhead by comparing evolution with it enabled vs disabled") {
    for (int steps : {1, 2, 3}) {
        BENCHMARK_PARAM("steps", steps);

        PatternHypergraph lhs, rhs;
        lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
        RewritingRule rule(lhs, rhs);

        std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}, {3, 4}};

        BENCHMARK_PARAM("transitive_reduction", "false");
        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(steps, 1, true, true, true, false);
            evolution.add_rule(rule);
            evolution.evolve(initial);
        }, 3);

        BENCHMARK_PARAM("transitive_reduction", "true");
        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(steps, 1, true, true, true, true);
            evolution.add_rule(rule);
            evolution.evolve(initial);
        }, 3);
    }
}
