// BENCHMARK_CATEGORY: Event Relationships

#include "benchmark_framework.hpp"
#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/unified_hypergraph.hpp>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category D: Event Relationships Benchmarks
// =============================================================================

BENCHMARK(causal_edges_by_steps, "Measures causal edge count growth as evolution progresses (1-4 steps)") {
    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}};

    for (int steps : {1, 2, 3, 4}) {
        BENCHMARK_PARAM("steps", steps);

        BENCHMARK_CODE([&]() {
            UnifiedHypergraph hg;
            ParallelEvolutionEngine engine(&hg, 4);

            // Rule: {x,y} -> {x,y},{y,z}
            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .rhs({1, 2})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, steps);
            auto causal_edges = hg.causal_graph().get_causal_edges();
            BENCHMARK_END();
        });
    }
}

BENCHMARK(branchial_edges_by_steps, "Measures branchial edge count growth as evolution progresses (1-4 steps)") {
    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}};

    for (int steps : {1, 2, 3, 4}) {
        BENCHMARK_PARAM("steps", steps);

        BENCHMARK_CODE([&]() {
            UnifiedHypergraph hg;
            ParallelEvolutionEngine engine(&hg, 4);

            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .rhs({1, 2})
                .build();
            engine.add_rule(rule);

            BENCHMARK_BEGIN();
            engine.evolve(initial, steps);
            auto branchial_edges = hg.causal_graph().get_branchial_edges();
            BENCHMARK_END();
        });
    }
}

BENCHMARK(causal_graph_retrieval, "Measures causal graph edge retrieval performance") {
    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}, {3, 4}};

    for (int steps : {1, 2, 3}) {
        BENCHMARK_PARAM("steps", steps);

        BENCHMARK_CODE([&]() {
            UnifiedHypergraph hg;
            ParallelEvolutionEngine engine(&hg, 4);

            auto rule = make_rule(0)
                .lhs({0, 1})
                .rhs({0, 1})
                .rhs({1, 2})
                .build();
            engine.add_rule(rule);

            engine.evolve(initial, steps);

            BENCHMARK_BEGIN();
            auto causal = hg.causal_graph().get_causal_edges();
            auto branchial = hg.causal_graph().get_branchial_edges();
            BENCHMARK_END();
        });
    }
}

BENCHMARK(event_signature_modes, "Compares event signature computation modes") {
    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}};

    for (auto mode : {EVENT_SIG_NONE, EVENT_SIG_AUTOMATIC, EVENT_SIG_FULL}) {
        const char* mode_name = (mode == EVENT_SIG_NONE) ? "none" :
                                (mode == EVENT_SIG_AUTOMATIC) ? "automatic" : "full";
        BENCHMARK_PARAM("mode", mode_name);

        BENCHMARK_CODE([&]() {
            UnifiedHypergraph hg;
            hg.set_event_signature_keys(mode);
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
