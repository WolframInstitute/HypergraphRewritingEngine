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

// =============================================================================
// Grid-with-holes evolution: 322 edges, 182 vertices, 4-edge blackhole rule.
// Previously lived in hypergraph/tests/test_grid_performance.cpp as a test;
// relocated here because the repeated-runs-at-increasing-depth shape is a
// benchmark, not a correctness test. Parameterised by step depth so the suite
// can track scaling across commits.
// =============================================================================
namespace {
std::vector<std::vector<VertexId>> grid_with_holes_edges() {
    return {
        {1, 2}, {1, 11}, {2, 3}, {2, 12}, {3, 4}, {3, 13}, {4, 5}, {4, 14},
        {5, 6}, {5, 15}, {6, 7}, {6, 16}, {7, 8}, {7, 17}, {8, 9}, {8, 18},
        {9, 10}, {9, 19}, {10, 20}, {11, 12}, {11, 21}, {12, 13}, {12, 22},
        {13, 14}, {13, 23}, {14, 15}, {14, 24}, {15, 16}, {15, 25}, {16, 17},
        {16, 26}, {17, 18}, {17, 27}, {18, 19}, {18, 28}, {19, 20}, {19, 29},
        {20, 30}, {21, 22}, {21, 31}, {22, 23}, {22, 32}, {23, 24}, {23, 33},
        {24, 25}, {24, 34}, {25, 26}, {25, 35}, {26, 27}, {26, 36}, {27, 28},
        {27, 37}, {28, 29}, {28, 38}, {29, 30}, {29, 39}, {30, 40}, {31, 32},
        {31, 41}, {32, 33}, {32, 42}, {33, 34}, {33, 43}, {34, 35}, {34, 44},
        {35, 36}, {35, 45}, {36, 37}, {36, 46}, {37, 38}, {37, 47}, {38, 39},
        {38, 48}, {39, 40}, {39, 49}, {40, 50}, {41, 42}, {41, 51}, {42, 43},
        {42, 52}, {43, 44}, {43, 53}, {44, 45}, {45, 46}, {46, 47}, {47, 48},
        {47, 57}, {48, 49}, {48, 58}, {49, 50}, {49, 59}, {50, 60}, {51, 52},
        {51, 61}, {52, 53}, {52, 62}, {53, 63}, {57, 58}, {57, 67}, {58, 59},
        {58, 68}, {59, 60}, {59, 69}, {60, 70}, {61, 62}, {61, 71}, {62, 63},
        {62, 72}, {63, 73}, {67, 68}, {67, 77}, {68, 69}, {68, 78}, {69, 70},
        {69, 79}, {70, 80}, {71, 72}, {71, 81}, {72, 73}, {72, 82}, {73, 83},
        {77, 78}, {77, 87}, {78, 79}, {78, 88}, {79, 80}, {79, 89}, {80, 90},
        {81, 82}, {81, 91}, {82, 83}, {82, 92}, {83, 84}, {83, 93}, {84, 85},
        {84, 94}, {85, 86}, {85, 95}, {86, 87}, {86, 96}, {87, 88}, {87, 97},
        {88, 89}, {88, 98}, {89, 90}, {89, 99}, {90, 100}, {91, 92}, {91, 101},
        {92, 93}, {92, 102}, {93, 94}, {93, 103}, {94, 95}, {94, 104}, {95, 96},
        {95, 105}, {96, 97}, {96, 106}, {97, 98}, {97, 107}, {98, 99}, {98, 108},
        {99, 100}, {99, 109}, {100, 110}, {101, 102}, {101, 111}, {102, 103},
        {102, 112}, {103, 104}, {103, 113}, {104, 105}, {104, 114}, {105, 106},
        {105, 115}, {106, 107}, {106, 116}, {107, 108}, {107, 117}, {108, 109},
        {108, 118}, {109, 110}, {109, 119}, {110, 120}, {111, 112}, {111, 121},
        {112, 113}, {112, 122}, {113, 114}, {113, 123}, {114, 115}, {115, 116},
        {116, 117}, {117, 118}, {117, 127}, {118, 119}, {118, 128}, {119, 120},
        {119, 129}, {120, 130}, {121, 122}, {121, 131}, {122, 123}, {122, 132},
        {123, 133}, {127, 128}, {127, 137}, {128, 129}, {128, 138}, {129, 130},
        {129, 139}, {130, 140}, {131, 132}, {131, 141}, {132, 133}, {132, 142},
        {133, 143}, {137, 138}, {137, 147}, {138, 139}, {138, 148}, {139, 140},
        {139, 149}, {140, 150}, {141, 142}, {141, 151}, {142, 143}, {142, 152},
        {143, 153}, {147, 148}, {147, 157}, {148, 149}, {148, 158}, {149, 150},
        {149, 159}, {150, 160}, {151, 152}, {151, 161}, {152, 153}, {152, 162},
        {153, 154}, {153, 163}, {154, 155}, {154, 164}, {155, 156}, {155, 165},
        {156, 157}, {156, 166}, {157, 158}, {157, 167}, {158, 159}, {158, 168},
        {159, 160}, {159, 169}, {160, 170}, {161, 162}, {161, 171}, {162, 163},
        {162, 172}, {163, 164}, {163, 173}, {164, 165}, {164, 174}, {165, 166},
        {165, 175}, {166, 167}, {166, 176}, {167, 168}, {167, 177}, {168, 169},
        {168, 178}, {169, 170}, {169, 179}, {170, 180}, {171, 172}, {171, 181},
        {172, 173}, {172, 182}, {173, 174}, {173, 183}, {174, 175}, {174, 184},
        {175, 176}, {175, 185}, {176, 177}, {176, 186}, {177, 178}, {177, 187},
        {178, 179}, {178, 188}, {179, 180}, {179, 189}, {180, 190}, {181, 182},
        {181, 191}, {182, 183}, {182, 192}, {183, 184}, {183, 193}, {184, 185},
        {184, 194}, {185, 186}, {185, 195}, {186, 187}, {186, 196}, {187, 188},
        {187, 197}, {188, 189}, {188, 198}, {189, 190}, {189, 199}, {190, 200},
        {191, 192}, {192, 193}, {193, 194}, {194, 195}, {195, 196}, {196, 197},
        {197, 198}, {198, 199}, {199, 200}
    };
}

RewriteRule grid_blackhole_rule() {
    return make_rule(0)
        .lhs({0, 1}).lhs({1, 2}).lhs({2, 3}).lhs({3, 4})
        .rhs({1, 5}).rhs({5, 4}).rhs({3, 0}).rhs({0, 5})
        .build();
}
}  // namespace

BENCHMARK(evolution_grid_with_holes_blackhole_rule,
          "Evolves a 322-edge grid-with-holes under a 4-edge blackhole rule at increasing depth; capped successor fanout.") {
    size_t num_threads = std::thread::hardware_concurrency();
    auto edges = grid_with_holes_edges();

    for (int steps : {4, 8, 16}) {
        BENCHMARK_PARAM("steps", steps);

        BENCHMARK_CODE([&]() {
            Hypergraph hg;
            hg.set_hash_strategy(HashStrategy::WL);
            ParallelEvolutionEngine engine(&hg, num_threads);
            engine.add_rule(grid_blackhole_rule());
            engine.set_max_states_per_step(80);
            engine.set_max_successor_states_per_parent(80);

            BENCHMARK_BEGIN();
            engine.evolve(edges, steps);
            BENCHMARK_END();
        });
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
