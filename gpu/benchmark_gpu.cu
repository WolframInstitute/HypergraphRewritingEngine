// GPU vs CPU Performance Benchmark
// Compares GPU evolution engine against CPU unified ParallelEvolutionEngine

#include <chrono>
#include <cstdio>
#include <vector>
#include <iomanip>
#include <random>

#include "evolution.cuh"

using namespace hypergraph::gpu;

// =============================================================================
// Benchmark Configuration
// =============================================================================

struct BenchmarkConfig {
    const char* name;
    std::vector<std::vector<uint8_t>> lhs;
    std::vector<std::vector<uint8_t>> rhs;
    uint8_t first_fresh;
    std::vector<std::vector<uint32_t>> initial;
    std::vector<size_t> steps_to_test;
};

// =============================================================================
// Benchmark Runner
// =============================================================================

struct BenchmarkResult {
    size_t steps;
    size_t states;
    size_t canonical_states;
    size_t events;
    double time_ms;
};

BenchmarkResult run_gpu_benchmark(
    const BenchmarkConfig& config,
    size_t steps,
    int warmup_runs = 2,
    int benchmark_runs = 5
) {
    BenchmarkResult result;
    result.steps = steps;

    // Warmup runs (not timed)
    for (int i = 0; i < warmup_runs; ++i) {
        GPUEvolutionEngine engine;
        engine.add_rule(config.lhs, config.rhs, config.first_fresh);
        engine.evolve(config.initial, steps);
    }

    // Benchmark runs
    double total_time = 0.0;
    EvolutionResults last_results;

    for (int i = 0; i < benchmark_runs; ++i) {
        GPUEvolutionEngine engine;
        engine.add_rule(config.lhs, config.rhs, config.first_fresh);

        auto start = std::chrono::high_resolution_clock::now();
        engine.evolve(config.initial, steps);
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::milli>(end - start).count();
        last_results = engine.get_results();
    }

    result.time_ms = total_time / benchmark_runs;
    result.states = last_results.num_states;
    result.canonical_states = last_results.num_canonical_states;
    result.events = last_results.num_events;

    return result;
}

// =============================================================================
// Generate Initial Graphs
// =============================================================================

std::vector<std::vector<uint32_t>> generate_chain(size_t num_edges) {
    std::vector<std::vector<uint32_t>> edges;
    for (uint32_t i = 0; i < num_edges; ++i) {
        edges.push_back({i, i + 1});
    }
    return edges;
}

std::vector<std::vector<uint32_t>> generate_random_graph(size_t num_edges, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::vector<uint32_t>> edges;

    // Start with a chain to ensure connectivity
    for (size_t i = 0; i < num_edges; ++i) {
        edges.push_back({static_cast<uint32_t>(i), static_cast<uint32_t>(i + 1)});
    }

    // Add some random cross-links
    std::uniform_int_distribution<uint32_t> vertex_dist(0, num_edges);
    for (size_t i = 0; i < num_edges / 4; ++i) {
        uint32_t v1 = vertex_dist(rng);
        uint32_t v2 = vertex_dist(rng);
        if (v1 != v2) {
            size_t idx = rng() % edges.size();
            edges[idx] = {v1, v2};
        }
    }

    return edges;
}

// =============================================================================
// Benchmark Configurations
// =============================================================================

std::vector<BenchmarkConfig> get_benchmark_configs() {
    return {
        // Simple chain rule: {x,y} -> {y,z}
        {
            "SimpleChain",
            {{0, 1}},
            {{1, 2}},
            2,
            {{0, 1}},
            {3, 4, 5, 6}
        },

        // Two-edge rule: {x,y},{y,z} -> {x,z},{z,w}
        {
            "TwoEdgeChain",
            {{0, 1}, {1, 2}},
            {{0, 2}, {2, 3}},
            3,
            {{0, 1}, {1, 2}, {2, 3}},
            {1, 2}
        },

        // Simple rule on larger initial graph
        {
            "SimpleRule_LargeInit",
            {{0, 1}},
            {{1, 2}},
            2,
            generate_chain(50),
            {1, 2}
        },
    };
}

// =============================================================================
// Main Benchmark
// =============================================================================

void print_header() {
    printf("\n");
    printf("================================================================================\n");
    printf("                     GPU Hypergraph Evolution Benchmark\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Configuration:\n");
    printf("  - Warmup runs: 2\n");
    printf("  - Benchmark runs: 5 (average time reported)\n");
    printf("\n");
}

void print_benchmark_results(const char* config_name, const std::vector<BenchmarkResult>& results) {
    printf("\n--- %s ---\n", config_name);
    printf("%6s %12s %12s %12s %12s\n",
           "Steps", "States", "Canonical", "Events", "Time (ms)");
    printf("--------------------------------------------------------------\n");

    for (const auto& r : results) {
        printf("%6zu %12zu %12zu %12zu %12.2f\n",
               r.steps, r.states, r.canonical_states, r.events, r.time_ms);
    }
}

void run_scaling_benchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("                         Scaling Benchmark\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Rule: {x,y} -> {y,z}  (linear chain growth)\n");
    printf("\n");

    BenchmarkConfig config = {
        "ScalingTest",
        {{0, 1}},
        {{1, 2}},
        2,
        {{0, 1}},
        {}  // Will be filled dynamically
    };

    printf("%6s %12s %12s %12s %12s %12s\n",
           "Steps", "States", "Canonical", "Events", "Time (ms)", "Events/ms");
    printf("------------------------------------------------------------------------\n");

    // Test increasing steps until it takes too long
    for (size_t steps = 3; steps <= 8; ++steps) {
        auto result = run_gpu_benchmark(config, steps, 1, 3);

        double events_per_ms = result.events / result.time_ms;

        printf("%6zu %12zu %12zu %12zu %12.2f %12.1f\n",
               result.steps, result.states, result.canonical_states,
               result.events, result.time_ms, events_per_ms);

        // Stop if taking too long (> 10 seconds)
        if (result.time_ms > 10000) {
            printf("(Stopping - time exceeded 10 seconds)\n");
            break;
        }
    }
}

void run_initial_size_benchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("                    Initial Graph Size Benchmark\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Rule: {x,y} -> {y,z}  (simple rewrite, 1 step)\n");
    printf("\n");

    printf("%12s %12s %12s %12s %12s\n",
           "InitEdges", "States", "Events", "Time (ms)", "Events/ms");
    printf("------------------------------------------------------------\n");

    for (size_t init_size : {10, 25, 50, 100, 200, 500}) {
        BenchmarkConfig config = {
            "InitSizeTest",
            {{0, 1}},
            {{1, 2}},
            2,
            generate_chain(init_size),
            {}
        };

        auto result = run_gpu_benchmark(config, 1, 2, 5);
        double events_per_ms = result.time_ms > 0 ? result.events / result.time_ms : 0;

        printf("%12zu %12zu %12zu %12.2f %12.1f\n",
               init_size, result.states, result.events, result.time_ms, events_per_ms);
    }
}

void run_pattern_complexity_benchmark() {
    printf("\n");
    printf("================================================================================\n");
    printf("                    Pattern Complexity Benchmark\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Different LHS pattern sizes on 50-edge random graph, 1 step\n");
    printf("\n");

    auto initial = generate_random_graph(50, 42);

    std::vector<BenchmarkConfig> configs = {
        {"1-edge LHS", {{0, 1}}, {{1, 2}}, 2, initial, {1}},
        {"2-edge LHS", {{0, 1}, {1, 2}}, {{0, 2}, {2, 3}}, 3, initial, {1}},
        {"3-edge LHS", {{0, 1}, {1, 2}, {2, 3}}, {{0, 3}, {3, 4}}, 4, initial, {1}},
        {"4-edge LHS", {{0, 1}, {1, 2}, {2, 3}, {3, 4}}, {{0, 4}, {4, 5}}, 5, initial, {1}},
    };

    printf("%12s %12s %12s %12s %12s\n",
           "Pattern", "States", "Events", "Time (ms)", "Events/ms");
    printf("------------------------------------------------------------\n");

    for (const auto& config : configs) {
        auto result = run_gpu_benchmark(config, 1, 2, 5);
        double events_per_ms = result.time_ms > 0 ? result.events / result.time_ms : 0;

        printf("%12s %12zu %12zu %12.2f %12.1f\n",
               config.name, result.states, result.events, result.time_ms, events_per_ms);
    }
}

int main(int argc, char** argv) {
    print_header();

    // Check for specific benchmark
    bool run_all = true;
    bool run_scaling = false;
    bool run_init_size = false;
    bool run_pattern = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--scaling") == 0) {
            run_all = false;
            run_scaling = true;
        } else if (strcmp(argv[i], "--init-size") == 0) {
            run_all = false;
            run_init_size = true;
        } else if (strcmp(argv[i], "--pattern") == 0) {
            run_all = false;
            run_pattern = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --scaling     Run step scaling benchmark\n");
            printf("  --init-size   Run initial graph size benchmark\n");
            printf("  --pattern     Run pattern complexity benchmark\n");
            printf("  (no args)     Run all benchmarks\n");
            return 0;
        }
    }

    // Run standard benchmarks
    if (run_all) {
        auto configs = get_benchmark_configs();

        for (const auto& config : configs) {
            std::vector<BenchmarkResult> results;
            for (size_t steps : config.steps_to_test) {
                results.push_back(run_gpu_benchmark(config, steps));
            }
            print_benchmark_results(config.name, results);
        }
    }

    // Run specialized benchmarks
    if (run_all || run_scaling) {
        run_scaling_benchmark();
    }

    if (run_all || run_init_size) {
        run_initial_size_benchmark();
    }

    if (run_all || run_pattern) {
        run_pattern_complexity_benchmark();
    }

    printf("\n");
    printf("================================================================================\n");
    printf("                           Benchmark Complete\n");
    printf("================================================================================\n");

    return 0;
}
