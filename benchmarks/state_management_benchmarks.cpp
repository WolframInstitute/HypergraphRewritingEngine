// BENCHMARK_CATEGORY: State Management

#include "benchmark_framework.hpp"
#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category C: State Management Benchmarks
// =============================================================================

BENCHMARK(state_storage_by_steps, "Measures state storage and retrieval overhead as evolution progresses from 1 to 3 steps") {
    for (int steps : {1, 2, 3}) {
        BENCHMARK_PARAM("steps", steps);

        std::vector<std::vector<VertexId>> initial = {{1, 2}};

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            ParallelEvolutionEngine engine(&hg, 1);

            // Rule: {x,y} -> {x,y},{y,z}
            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .rhs({1, 2})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, steps);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(state_canonicalization_modes, "Compares evolution performance with different state canonicalization modes") {
    std::vector<std::vector<VertexId>> initial = {{1, 2}};

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
                .rhs({1, 2})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, 2);
            BENCHMARK_END();
        });
    }
}

BENCHMARK(state_count_by_steps, "Tracks how state count grows with evolution steps") {
    for (int steps : {1, 2, 3, 4}) {
        BENCHMARK_PARAM("steps", steps);

        std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}};

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            ParallelEvolutionEngine engine(&hg, 4);

            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .rhs({1, 2})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, steps);
            BENCHMARK_END();
        });
    }
}
