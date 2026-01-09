#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <memory>

#include "hypergraph/arena.hpp"
#include "hypergraph/types.hpp"
#include "hypergraph/unified_hypergraph.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/parallel_evolution.hpp"

namespace v2 = hypergraph;

// =============================================================================
// Black Hole Simulation - Idempotent Rule Tests
// =============================================================================
// Tests idempotent rules (4 edges -> 4 edges) on varying initial state sizes.
// Compares UT, UT-Inc, WL hash strategies with ByState and ByStateAndEdges modes.
// All methods should produce identical state/event/causal/branchial counts.

class BlackHoleIdempotentTest : public ::testing::Test {
protected:
    struct RunResult {
        size_t num_states;
        size_t num_events;
        size_t num_causal;
        size_t num_branchial;
        double time_ms;
        size_t reused;
        size_t recomputed;
    };

    // Generate a random connected hypergraph with the given number of 2-edges
    std::vector<std::vector<v2::VertexId>> generate_random_graph(
        size_t num_edges,
        uint32_t seed
    ) {
        std::mt19937 rng(seed);
        std::vector<std::vector<v2::VertexId>> edges;

        // Start with a chain to ensure connectivity
        for (size_t i = 0; i < num_edges; ++i) {
            v2::VertexId v1 = static_cast<v2::VertexId>(i);
            v2::VertexId v2 = static_cast<v2::VertexId>(i + 1);
            edges.push_back({v1, v2});
        }

        // Add some random cross-links to make it more interesting
        std::uniform_int_distribution<v2::VertexId> vertex_dist(0, num_edges);
        for (size_t i = 0; i < num_edges / 4; ++i) {
            v2::VertexId v1 = vertex_dist(rng);
            v2::VertexId v2 = vertex_dist(rng);
            if (v1 != v2) {
                // Replace a random edge with a cross-link
                size_t idx = rng() % edges.size();
                edges[idx] = {v1, v2};
            }
        }

        return edges;
    }

    RunResult run_evolution(
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        v2::HashStrategy hash_strategy,
        v2::EventSignatureKeys event_keys
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_hash_strategy(hash_strategy);
        hg->set_event_signature_keys(event_keys);
        hg->reset_incremental_tree_stats();

        v2::ParallelEvolutionEngine engine(hg.get(), 0);  // Use all threads
        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }

        auto start = std::chrono::high_resolution_clock::now();
        engine.evolve(initial, steps);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        auto [reused, recomputed] = hg->incremental_tree_stats();

        return {
            engine.num_canonical_states(),
            engine.num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges(),
            time_ms,
            reused,
            recomputed
        };
    }

    const char* hash_strategy_name(v2::HashStrategy s) {
        switch (s) {
            case v2::HashStrategy::UniquenessTree: return "UT";
            case v2::HashStrategy::IncrementalUniquenessTree: return "UT-Inc";
            case v2::HashStrategy::WL: return "WL";
            case v2::HashStrategy::IncrementalWL: return "WL-Inc";
            default: return "?";
        }
    }

    const char* event_keys_name(v2::EventSignatureKeys k) {
        if (k == v2::EVENT_SIG_NONE) return "None";
        if (k == v2::EVENT_SIG_FULL) return "Full";
        if (k == v2::EVENT_SIG_AUTOMATIC) return "Automatic";
        return "?";
    }
};

TEST_F(BlackHoleIdempotentTest, BlackHoleRule_4to4_VaryingSize) {
    // Black hole rule: 4 edges -> 4 edges with fresh vertex
    // Pattern: {{x,y},{y,z},{z,w},{w,v}} -> {{y,u},{u,v},{w,x},{x,u}}
    // LHS: chain x-y-z-w-v (vertices 0,1,2,3,4)
    // RHS: introduces fresh vertex u (vertex 5)
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .lhs({2, 3})  // {z, w}
        .lhs({3, 4})  // {w, v}
        .rhs({1, 5})  // {y, u}
        .rhs({5, 4})  // {u, v}
        .rhs({3, 0})  // {w, x}
        .rhs({0, 5})  // {x, u}
        .build();

    std::vector<size_t> edge_counts = {25, 50, 100};
    std::vector<v2::HashStrategy> strategies = {
        v2::HashStrategy::UniquenessTree,
        v2::HashStrategy::IncrementalUniquenessTree,
        v2::HashStrategy::WL,
        v2::HashStrategy::IncrementalWL
    };
    std::vector<v2::EventSignatureKeys> modes = {
        v2::EVENT_SIG_FULL,
        v2::EVENT_SIG_AUTOMATIC
    };

    const size_t steps = 1;
    const uint32_t seed = 42;  // Fixed seed for reproducibility

    std::cout << "\n=== Black Hole Rule Test (4->4 with fresh vertex) ===\n";
    std::cout << "Rule: {{x,y},{y,z},{z,w},{w,v}} -> {{y,u},{u,v},{w,x},{x,u}}\n";
    std::cout << "Steps: " << steps << "\n\n";

    for (size_t num_edges : edge_counts) {
        auto initial = generate_random_graph(num_edges, seed);

        std::cout << "=== Initial state: " << num_edges << " edges ===\n";
        std::cout << std::setw(10) << "Mode"
                  << std::setw(8) << "Hash"
                  << std::setw(10) << "States"
                  << std::setw(10) << "Events"
                  << std::setw(10) << "Causal"
                  << std::setw(12) << "Branchial"
                  << std::setw(12) << "Time(ms)"
                  << std::setw(10) << "Reuse%"
                  << "\n";
        std::cout << std::string(82, '-') << "\n";

        // Check consistency within each event mode (hash strategies should match)
        bool all_match = true;

        for (auto mode : modes) {
            RunResult mode_reference;
            bool have_mode_reference = false;

            for (auto strategy : strategies) {
                auto result = run_evolution({rule}, initial, steps, strategy, mode);

                double reuse_pct = (result.reused + result.recomputed > 0)
                    ? 100.0 * result.reused / (result.reused + result.recomputed)
                    : 0.0;

                std::cout << std::fixed << std::setprecision(2);
                std::cout << std::setw(10) << event_keys_name(mode)
                          << std::setw(8) << hash_strategy_name(strategy)
                          << std::setw(10) << result.num_states
                          << std::setw(10) << result.num_events
                          << std::setw(10) << result.num_causal
                          << std::setw(12) << result.num_branchial
                          << std::setw(12) << result.time_ms
                          << std::setw(9) << std::setprecision(1) << reuse_pct << "%"
                          << "\n";

                // Check consistency within this event mode
                if (!have_mode_reference) {
                    mode_reference = result;
                    have_mode_reference = true;
                } else {
                    if (result.num_states != mode_reference.num_states ||
                        result.num_events != mode_reference.num_events ||
                        result.num_causal != mode_reference.num_causal ||
                        result.num_branchial != mode_reference.num_branchial) {
                        std::cout << "  ^ MISMATCH with " << hash_strategy_name(strategies[0]) << "\n";
                        all_match = false;
                    }
                }
            }
        }

        std::cout << "\n";

        // Verify all hash strategies produce same results within each event mode
        EXPECT_TRUE(all_match)
            << "Hash strategies produce different results for " << num_edges << " edges";
    }
}

TEST_F(BlackHoleIdempotentTest, BlackHoleRule_3to3_VaryingSize) {
    // 3 edges -> 3 edges with fresh vertex
    // Pattern: {{x,y},{y,z},{z,w}} -> {{y,u},{u,w},{x,u}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .lhs({2, 3})  // {z, w}
        .rhs({1, 4})  // {y, u}
        .rhs({4, 3})  // {u, w}
        .rhs({0, 4})  // {x, u}
        .build();

    std::vector<size_t> edge_counts = {25, 50, 100};
    std::vector<v2::HashStrategy> strategies = {
        v2::HashStrategy::UniquenessTree,
        v2::HashStrategy::IncrementalUniquenessTree,
        v2::HashStrategy::WL,
        v2::HashStrategy::IncrementalWL
    };
    std::vector<v2::EventSignatureKeys> modes = {
        v2::EVENT_SIG_FULL,
        v2::EVENT_SIG_AUTOMATIC
    };

    const size_t steps = 1;
    const uint32_t seed = 42;

    std::cout << "\n=== Black Hole Rule Test (3->3 with fresh vertex) ===\n";
    std::cout << "Rule: {{x,y},{y,z},{z,w}} -> {{y,u},{u,w},{x,u}}\n";
    std::cout << "Steps: " << steps << "\n\n";

    for (size_t num_edges : edge_counts) {
        auto initial = generate_random_graph(num_edges, seed);

        std::cout << "=== Initial state: " << num_edges << " edges ===\n";
        std::cout << std::setw(10) << "Mode"
                  << std::setw(8) << "Hash"
                  << std::setw(10) << "States"
                  << std::setw(10) << "Events"
                  << std::setw(10) << "Causal"
                  << std::setw(12) << "Branchial"
                  << std::setw(12) << "Time(ms)"
                  << std::setw(10) << "Reuse%"
                  << "\n";
        std::cout << std::string(82, '-') << "\n";

        bool all_match = true;

        for (auto mode : modes) {
            RunResult mode_reference;
            bool have_mode_reference = false;

            for (auto strategy : strategies) {
                auto result = run_evolution({rule}, initial, steps, strategy, mode);

                double reuse_pct = (result.reused + result.recomputed > 0)
                    ? 100.0 * result.reused / (result.reused + result.recomputed)
                    : 0.0;

                std::cout << std::fixed << std::setprecision(2);
                std::cout << std::setw(10) << event_keys_name(mode)
                          << std::setw(8) << hash_strategy_name(strategy)
                          << std::setw(10) << result.num_states
                          << std::setw(10) << result.num_events
                          << std::setw(10) << result.num_causal
                          << std::setw(12) << result.num_branchial
                          << std::setw(12) << result.time_ms
                          << std::setw(9) << std::setprecision(1) << reuse_pct << "%"
                          << "\n";

                if (!have_mode_reference) {
                    mode_reference = result;
                    have_mode_reference = true;
                } else {
                    if (result.num_states != mode_reference.num_states ||
                        result.num_events != mode_reference.num_events ||
                        result.num_causal != mode_reference.num_causal ||
                        result.num_branchial != mode_reference.num_branchial) {
                        std::cout << "  ^ MISMATCH with " << hash_strategy_name(strategies[0]) << "\n";
                        all_match = false;
                    }
                }
            }
        }

        std::cout << "\n";

        EXPECT_TRUE(all_match)
            << "Hash strategies produce different results for " << num_edges << " edges";
    }
}

// Detailed timing comparison for a specific size
TEST_F(BlackHoleIdempotentTest, DetailedTiming_100Edges) {
    // Black hole rule with fresh vertex
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .lhs({2, 3})  // {z, w}
        .lhs({3, 4})  // {w, v}
        .rhs({1, 5})  // {y, u}
        .rhs({5, 4})  // {u, v}
        .rhs({3, 0})  // {w, x}
        .rhs({0, 5})  // {x, u}
        .build();

    const size_t num_edges = 100;
    const size_t steps = 1;
    const uint32_t seed = 42;
    const int num_runs = 5;

    auto initial = generate_random_graph(num_edges, seed);

    std::vector<v2::HashStrategy> strategies = {
        v2::HashStrategy::UniquenessTree,
        v2::HashStrategy::IncrementalUniquenessTree,
        v2::HashStrategy::WL,
        v2::HashStrategy::IncrementalWL
    };

    std::cout << "\n=== Detailed Timing: 100 edges, black hole rule ===\n";
    std::cout << "Rule: {{x,y},{y,z},{z,w},{w,v}} -> {{y,u},{u,v},{w,x},{x,u}}\n";
    std::cout << "Averaging over " << num_runs << " runs\n\n";

    for (auto mode : {v2::EVENT_SIG_FULL,
                      v2::EVENT_SIG_AUTOMATIC}) {
        std::cout << "Event mode: " << event_keys_name(mode) << "\n";
        std::cout << std::setw(10) << "Strategy"
                  << std::setw(12) << "Avg(ms)"
                  << std::setw(12) << "Min(ms)"
                  << std::setw(12) << "Max(ms)"
                  << std::setw(10) << "States"
                  << std::setw(10) << "Events"
                  << "\n";
        std::cout << std::string(66, '-') << "\n";

        for (auto strategy : strategies) {
            double total = 0, min_t = 1e9, max_t = 0;
            RunResult last;

            for (int i = 0; i < num_runs; ++i) {
                last = run_evolution({rule}, initial, steps, strategy, mode);
                total += last.time_ms;
                min_t = std::min(min_t, last.time_ms);
                max_t = std::max(max_t, last.time_ms);
            }

            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(10) << hash_strategy_name(strategy)
                      << std::setw(12) << (total / num_runs)
                      << std::setw(12) << min_t
                      << std::setw(12) << max_t
                      << std::setw(10) << last.num_states
                      << std::setw(10) << last.num_events
                      << "\n";
        }
        std::cout << "\n";
    }
}

// WL-only test with 2 steps for black hole simulation
TEST_F(BlackHoleIdempotentTest, WL_Only_2Steps) {
    // Black hole rule: 4 edges -> 4 edges with fresh vertex
    // Pattern: {{x,y},{y,z},{z,w},{w,v}} -> {{y,u},{u,v},{w,x},{x,u}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .lhs({2, 3})  // {z, w}
        .lhs({3, 4})  // {w, v}
        .rhs({1, 5})  // {y, u}
        .rhs({5, 4})  // {u, v}
        .rhs({3, 0})  // {w, x}
        .rhs({0, 5})  // {x, u}
        .build();

    std::vector<size_t> edge_counts = {25, 50, 100};
    std::vector<v2::EventSignatureKeys> modes = {
        v2::EVENT_SIG_FULL,
        v2::EVENT_SIG_AUTOMATIC
    };

    const size_t steps = 2;
    const uint32_t seed = 42;

    std::cout << "\n=== WL-Only Black Hole Test (2 steps) ===\n";
    std::cout << "Rule: {{x,y},{y,z},{z,w},{w,v}} -> {{y,u},{u,v},{w,x},{x,u}}\n";
    std::cout << "Steps: " << steps << "\n\n";

    std::cout << std::setw(8) << "Edges"
              << std::setw(18) << "Event Mode"
              << std::setw(10) << "States"
              << std::setw(10) << "Events"
              << std::setw(10) << "Causal"
              << std::setw(12) << "Branchial"
              << std::setw(12) << "Time(ms)"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t num_edges : edge_counts) {
        auto initial = generate_random_graph(num_edges, seed);

        for (auto mode : modes) {
            auto result = run_evolution({rule}, initial, steps, v2::HashStrategy::WL, mode);

            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(8) << num_edges
                      << std::setw(18) << event_keys_name(mode)
                      << std::setw(10) << result.num_states
                      << std::setw(10) << result.num_events
                      << std::setw(10) << result.num_causal
                      << std::setw(12) << result.num_branchial
                      << std::setw(12) << result.time_ms
                      << "\n";
        }
    }
    std::cout << "\n";
}

