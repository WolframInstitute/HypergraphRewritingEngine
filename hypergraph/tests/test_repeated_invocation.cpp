#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"

using namespace hypergraph;

// Get current process memory usage (RSS) in KB
size_t get_memory_usage_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            size_t kb = 0;
            sscanf(line.c_str(), "VmRSS: %zu", &kb);
            return kb;
        }
    }
    return 0;
}

// Grid graph with holes: 322 edges, 182 vertices
static std::vector<std::vector<VertexId>> create_grid_with_holes() {
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

// Blackhole rule: {{0,1},{1,2},{2,3},{3,4}} -> {{1,5},{5,4},{3,0},{0,5}}
static RewriteRule blackhole_rule() {
    return make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .lhs({2, 3})  // {z, w}
        .lhs({3, 4})  // {w, v}
        .rhs({1, 5})  // {y, u}
        .rhs({5, 4})  // {u, v}
        .rhs({3, 0})  // {w, x}
        .rhs({0, 5})  // {x, u}
        .build();
}

// Mimics exactly what performRewriting does with the user's exact parameters
void run_grid_evolution(HashStrategy strategy, int steps, size_t max_per_parent, size_t max_per_step) {
    size_t mem_before = get_memory_usage_kb();
    {
    // Create hypergraph (local, destroyed at end)
    Hypergraph hg;
    hg.set_hash_strategy(strategy);
    // CanonicalizeEvents -> False means no event canonicalization
    hg.set_event_signature_keys(EVENT_SIG_NONE);

    // Create parallel evolution engine
    ParallelEvolutionEngine engine(&hg, std::thread::hardware_concurrency());
    engine.set_max_steps(steps);
    engine.set_transitive_reduction(true);
    engine.set_max_successor_states_per_parent(max_per_parent);
    engine.set_max_states_per_step(max_per_step);

    engine.add_rule(blackhole_rule());

    auto edges = create_grid_with_holes();

    // Run evolution
    engine.evolve(edges, steps);

    // Access results (like FFI does)
    uint32_t num_states = hg.num_states();
    uint32_t num_events = hg.num_events();

    // Debug: show arena memory usage
    size_t arena_bytes = hg.arena().bytes_allocated();
    std::cout << "  States: " << num_states << ", Events: " << num_events
              << ", Arena: " << (arena_bytes / 1024) << " KB" << std::endl;
    } // hg and engine destroyed here

    size_t mem_after = get_memory_usage_kb();
    std::cout << "  After destruction: " << mem_after << " KB (retained: "
              << (int64_t)(mem_after - mem_before) << " KB)" << std::endl;
}

TEST(RepeatedGridInvocation, FiveTimesWL) {
    std::cout << "\n=== Running grid evolution 5 times with WL ===" << std::endl;
    size_t initial_mem = get_memory_usage_kb();
    std::cout << "Initial memory: " << initial_mem << " KB" << std::endl;

    for (int i = 0; i < 5; ++i) {
        std::cout << "Iteration " << (i + 1) << ":" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        run_grid_evolution(HashStrategy::WL, 10, 100, 100);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        size_t current_mem = get_memory_usage_kb();
        std::cout << "  Time: " << ms << " ms, Memory: " << current_mem
                  << " KB (delta: " << (int64_t)(current_mem - initial_mem) << " KB)" << std::endl;
    }

    size_t final_mem = get_memory_usage_kb();
    std::cout << "Final memory delta: " << (int64_t)(final_mem - initial_mem) << " KB" << std::endl;
}

TEST(RepeatedGridInvocation, FiveTimesUT) {
    std::cout << "\n=== Running grid evolution 5 times with UT ===" << std::endl;
    size_t initial_mem = get_memory_usage_kb();
    std::cout << "Initial memory: " << initial_mem << " KB" << std::endl;

    for (int i = 0; i < 5; ++i) {
        std::cout << "Iteration " << (i + 1) << ":" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        run_grid_evolution(HashStrategy::UniquenessTree, 10, 100, 100);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        size_t current_mem = get_memory_usage_kb();
        std::cout << "  Time: " << ms << " ms, Memory: " << current_mem
                  << " KB (delta: " << (int64_t)(current_mem - initial_mem) << " KB)" << std::endl;
    }

    size_t final_mem = get_memory_usage_kb();
    std::cout << "Final memory delta: " << (int64_t)(final_mem - initial_mem) << " KB" << std::endl;
}

TEST(RepeatedGridInvocation, TwentyTimesWL) {
    std::cout << "\n=== Running grid evolution 20 times with WL ===" << std::endl;
    size_t initial_mem = get_memory_usage_kb();

    for (int i = 0; i < 20; ++i) {
        std::cout << "Iteration " << (i + 1) << ": ";
        run_grid_evolution(HashStrategy::WL, 10, 100, 100);
    }

    size_t final_mem = get_memory_usage_kb();
    std::cout << "Final memory: " << final_mem << " KB (delta: "
              << (int64_t)(final_mem - initial_mem) << " KB)" << std::endl;

    // Memory should stabilize - allow 500MB which is ~1 arena worth of fragmentation
    EXPECT_LT(final_mem - initial_mem, 500000);
}

// Test abort functionality
void run_grid_evolution_with_abort(HashStrategy strategy, int steps, size_t max_per_parent, size_t max_per_step, bool abort_early) {
    Hypergraph hg;
    hg.set_hash_strategy(strategy);
    hg.set_event_signature_keys(EVENT_SIG_NONE);

    ParallelEvolutionEngine engine(&hg, std::thread::hardware_concurrency());
    engine.set_max_steps(steps);
    engine.set_transitive_reduction(true);
    engine.set_max_successor_states_per_parent(max_per_parent);
    engine.set_max_states_per_step(max_per_step);

    engine.add_rule(blackhole_rule());
    auto edges = create_grid_with_holes();

    std::atomic<bool> should_abort{false};
    std::atomic<int> check_count{0};

    // Abort after ~500ms if abort_early is true
    auto start = std::chrono::steady_clock::now();

    bool was_aborted = engine.evolve_with_abort(edges, steps, [&]() {
        check_count.fetch_add(1);
        if (abort_early) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > std::chrono::milliseconds(500)) {
                return true;
            }
        }
        return false;
    });

    std::cout << "  States: " << hg.num_states() << ", Events: " << hg.num_events()
              << ", Aborted: " << (was_aborted ? "yes" : "no")
              << ", Checks: " << check_count.load() << std::endl;
}

TEST(RepeatedGridInvocation, AbortTestWL) {
    std::cout << "\n=== Testing abort functionality (WL) ===" << std::endl;

    std::cout << "Run 1 (no abort):" << std::endl;
    run_grid_evolution_with_abort(HashStrategy::WL, 10, 100, 100, false);

    std::cout << "Run 2 (abort after 500ms):" << std::endl;
    run_grid_evolution_with_abort(HashStrategy::WL, 10, 100, 100, true);

    std::cout << "Run 3 (no abort again):" << std::endl;
    run_grid_evolution_with_abort(HashStrategy::WL, 10, 100, 100, false);

    std::cout << "Run 4 (abort after 500ms):" << std::endl;
    run_grid_evolution_with_abort(HashStrategy::WL, 10, 100, 100, true);
}

TEST(RepeatedGridInvocation, AbortTestUT) {
    std::cout << "\n=== Testing abort functionality (UniquenessTree) ===" << std::endl;

    std::cout << "Run 1 (no abort):" << std::endl;
    run_grid_evolution_with_abort(HashStrategy::UniquenessTree, 10, 100, 100, false);

    std::cout << "Run 2 (abort after 500ms):" << std::endl;
    run_grid_evolution_with_abort(HashStrategy::UniquenessTree, 10, 100, 100, true);

    std::cout << "Run 3 (no abort again):" << std::endl;
    run_grid_evolution_with_abort(HashStrategy::UniquenessTree, 10, 100, 100, false);
}
