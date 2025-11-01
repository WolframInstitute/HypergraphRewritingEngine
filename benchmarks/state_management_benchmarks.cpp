// BENCHMARK_CATEGORY: State Management

#include "benchmark_framework.hpp"
#include <hypergraph/rewriting.hpp>
#include <hypergraph/wolfram_evolution.hpp>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category C: State Management Benchmarks
// =============================================================================

BENCHMARK(state_storage_by_steps, "Measures state storage and retrieval overhead as evolution progresses from 1 to 3 steps") {
    for (int steps : {1, 2, 3}) {
        BENCHMARK_PARAM("steps", steps);

        PatternHypergraph lhs, rhs;
        lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
        rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
        RewritingRule rule(lhs, rhs);

        std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}};

        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(steps, 1, true, true);
            evolution.add_rule(rule);
            BENCHMARK_BEGIN();
            evolution.evolve(initial);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(full_capture_overhead, "Compares evolution performance with and without full state capture enabled") {
    PatternHypergraph lhs, rhs;
    lhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(1), PatternVertex::variable(2)});
    rhs.add_edge({PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}};

    for (bool full_capture : {false, true}) {
        BENCHMARK_PARAM("full_capture", full_capture ? "true" : "false");

        BENCHMARK_CODE([&]() {
            WolframEvolution evolution(2, 1, true, full_capture);
            evolution.add_rule(rule);
            BENCHMARK_BEGIN();
            evolution.evolve(initial);
            BENCHMARK_END();
        });
    }
}
