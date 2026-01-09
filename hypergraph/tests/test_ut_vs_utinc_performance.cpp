#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>

#include "hypergraph/arena.hpp"
#include "hypergraph/types.hpp"
#include "hypergraph/unified_hypergraph.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/parallel_evolution.hpp"

namespace v2 = hypergraph;

// =============================================================================
// UT vs UT-Inc Performance Comparison Tests
// =============================================================================
// These tests compare UniquenessTree (UT) and IncrementalUniquenessTree (UT-Inc)
// performance to identify where UT-Inc overhead comes from and how to optimize.

class UTvsUTIncPerformanceTest : public ::testing::Test {
protected:
    struct PerfResult {
        double time_ms;
        size_t num_states;
        size_t num_events;
        size_t reused;
        size_t recomputed;
        size_t stored_caches;
    };

    PerfResult run_with_strategy(
        v2::HashStrategy strategy,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        size_t num_threads = 0
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_hash_strategy(strategy);
        hg->reset_incremental_tree_stats();

        v2::ParallelEvolutionEngine engine(hg.get(), num_threads);
        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }

        auto start = std::chrono::high_resolution_clock::now();
        engine.evolve(initial, steps);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto [reused, recomputed] = hg->incremental_tree_stats();

        return {
            time_ms,
            engine.num_canonical_states(),
            engine.num_events(),
            reused,
            recomputed,
            hg->num_stored_caches()
        };
    }

    void compare_strategies(
        const std::string& test_name,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        int num_runs = 5
    ) {
        std::cout << "\n=== " << test_name << " (steps=" << steps << ") ===\n";

        // Warmup
        run_with_strategy(v2::HashStrategy::UniquenessTree, rules, initial, steps);
        run_with_strategy(v2::HashStrategy::IncrementalUniquenessTree, rules, initial, steps);

        // Collect results
        double ut_total = 0, utinc_total = 0;
        double ut_min = 1e9, ut_max = 0;
        double utinc_min = 1e9, utinc_max = 0;
        PerfResult ut_result, utinc_result;

        for (int i = 0; i < num_runs; ++i) {
            ut_result = run_with_strategy(v2::HashStrategy::UniquenessTree, rules, initial, steps);
            utinc_result = run_with_strategy(v2::HashStrategy::IncrementalUniquenessTree, rules, initial, steps);

            ut_total += ut_result.time_ms;
            utinc_total += utinc_result.time_ms;
            ut_min = std::min(ut_min, ut_result.time_ms);
            ut_max = std::max(ut_max, ut_result.time_ms);
            utinc_min = std::min(utinc_min, utinc_result.time_ms);
            utinc_max = std::max(utinc_max, utinc_result.time_ms);
        }

        double ut_avg = ut_total / num_runs;
        double utinc_avg = utinc_total / num_runs;
        double ratio = utinc_avg / ut_avg;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "States: " << ut_result.num_states << ", Events: " << ut_result.num_events << "\n";
        std::cout << "UT:     avg=" << ut_avg << "ms, min=" << ut_min << "ms, max=" << ut_max << "ms\n";
        std::cout << "UT-Inc: avg=" << utinc_avg << "ms, min=" << utinc_min << "ms, max=" << utinc_max << "ms\n";

        if (ratio > 1.0) {
            std::cout << "Result: UT is " << ratio << "x FASTER\n";
        } else {
            std::cout << "Result: UT-Inc is " << (1.0/ratio) << "x FASTER\n";
        }

        // UT-Inc specific stats
        double reuse_ratio = (utinc_result.reused + utinc_result.recomputed > 0)
            ? 100.0 * utinc_result.reused / (utinc_result.reused + utinc_result.recomputed)
            : 0.0;
        std::cout << "UT-Inc stats: reused=" << utinc_result.reused
                  << ", recomputed=" << utinc_result.recomputed
                  << ", reuse_ratio=" << std::setprecision(1) << reuse_ratio << "%"
                  << ", stored_caches=" << utinc_result.stored_caches << "\n";

        // Per-hash cost analysis
        size_t total_hashes = utinc_result.reused + utinc_result.recomputed;
        if (total_hashes > 0) {
            double ut_per_state = ut_avg / ut_result.num_states;
            double utinc_per_state = utinc_avg / utinc_result.num_states;
            double utinc_per_hash = utinc_avg / total_hashes;
            std::cout << "Cost per state: UT=" << std::setprecision(3) << ut_per_state
                      << "ms, UT-Inc=" << utinc_per_state << "ms\n";
            std::cout << "UT-Inc cost per vertex hash: " << (utinc_avg * 1000.0 / total_hashes)
                      << " us\n";
        }
    }

    // Detailed profiling version that measures individual phases
    void profile_utinc_phases(
        const std::string& test_name,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps
    ) {
        std::cout << "\n=== " << test_name << " Phase Analysis ===\n";

        // Run with UT to get baseline
        auto ut_result = run_with_strategy(v2::HashStrategy::UniquenessTree, rules, initial, steps);

        // Run with UT-Inc
        auto utinc_result = run_with_strategy(v2::HashStrategy::IncrementalUniquenessTree, rules, initial, steps);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "UT time: " << ut_result.time_ms << "ms\n";
        std::cout << "UT-Inc time: " << utinc_result.time_ms << "ms\n";
        std::cout << "Overhead: " << (utinc_result.time_ms - ut_result.time_ms) << "ms ("
                  << ((utinc_result.time_ms / ut_result.time_ms - 1.0) * 100.0) << "%)\n";

        // Calculate theoretical savings from reuse
        size_t total = utinc_result.reused + utinc_result.recomputed;
        if (total > 0 && utinc_result.recomputed > 0) {
            // Estimate time per recompute (assuming reuse is ~free)
            double time_per_recompute = utinc_result.time_ms / utinc_result.recomputed;
            double theoretical_full_time = time_per_recompute * total;
            double actual_savings = theoretical_full_time - utinc_result.time_ms;

            std::cout << "\nReuse analysis:\n";
            std::cout << "  Vertex hashes: " << total << " total, "
                      << utinc_result.reused << " reused, "
                      << utinc_result.recomputed << " recomputed\n";
            std::cout << "  Reuse ratio: " << std::setprecision(1)
                      << (100.0 * utinc_result.reused / total) << "%\n";

            // Compare with UT's vertex hash count (approximation)
            std::cout << "\nPotential issues to investigate:\n";
            if (utinc_result.time_ms > ut_result.time_ms) {
                double overhead_per_state = (utinc_result.time_ms - ut_result.time_ms) / utinc_result.num_states;
                std::cout << "  Overhead per state: " << (overhead_per_state * 1000.0) << " us\n";
                std::cout << "  Possible causes:\n";
                std::cout << "    - Cache lookup overhead (hash map access)\n";
                std::cout << "    - Bloom filter checking overhead\n";
                std::cout << "    - Per-state adjacency building (vs global adjacency)\n";
                std::cout << "    - Cache storage overhead\n";
            }
        }
    }
};

// =============================================================================
// Test Cases - Simple Rules
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, SimpleRule_Steps1to5) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    for (size_t steps = 1; steps <= 5; ++steps) {
        compare_strategies("SimpleRule", {rule}, initial, steps);
    }
}

TEST_F(UTvsUTIncPerformanceTest, SimpleRule_PhaseAnalysis) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    profile_utinc_phases("SimpleRule_Step4", {rule}, initial, 4);
}

// =============================================================================
// Test Cases - Two-Edge Rules
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, TwoEdgeRule_Steps1to4) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}};

    for (size_t steps = 1; steps <= 4; ++steps) {
        compare_strategies("TwoEdgeRule_Chain", {rule}, initial, steps);
    }
}

TEST_F(UTvsUTIncPerformanceTest, TwoEdgeRule_Triangle) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    for (size_t steps = 1; steps <= 3; ++steps) {
        compare_strategies("TwoEdgeRule_Triangle", {rule}, initial, steps);
    }
}

// =============================================================================
// Test Cases - Sparse Graphs (High Reuse Potential)
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, SparseGraph_DisconnectedComponents) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    // Many disconnected components - high reuse potential
    std::vector<std::vector<v2::VertexId>> initial;
    for (v2::VertexId v = 0; v < 60; v += 2) {
        initial.push_back({v, static_cast<v2::VertexId>(v + 1)});
    }

    compare_strategies("SparseGraph_30Components", {rule}, initial, 1);
    profile_utinc_phases("SparseGraph_30Components", {rule}, initial, 1);
}

TEST_F(UTvsUTIncPerformanceTest, SparseGraph_LargeDisconnected) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    // 100 disconnected components
    std::vector<std::vector<v2::VertexId>> initial;
    for (v2::VertexId v = 0; v < 200; v += 2) {
        initial.push_back({v, static_cast<v2::VertexId>(v + 1)});
    }

    compare_strategies("SparseGraph_100Components", {rule}, initial, 1);
}

// =============================================================================
// Test Cases - Connected Graphs (Low Reuse Potential)
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, ConnectedGraph_Chain) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    // Long chain - all vertices connected, low reuse potential
    std::vector<std::vector<v2::VertexId>> initial;
    for (v2::VertexId v = 0; v < 20; ++v) {
        initial.push_back({v, static_cast<v2::VertexId>(v + 1)});
    }

    compare_strategies("ConnectedGraph_Chain20", {rule}, initial, 1);
    profile_utinc_phases("ConnectedGraph_Chain20", {rule}, initial, 1);
}

// =============================================================================
// Test Cases - Hyperedges
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, HyperedgeRule_Steps1to4) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    for (size_t steps = 1; steps <= 4; ++steps) {
        compare_strategies("HyperedgeRule", {rule}, initial, steps);
    }
}

// =============================================================================
// Test Cases - Multi-Rule Systems
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, MultiRule_Steps1to3) {
    v2::RewriteRule rule1 = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    v2::RewriteRule rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({0, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    for (size_t steps = 1; steps <= 3; ++steps) {
        compare_strategies("MultiRule", {rule1, rule2}, initial, steps);
    }
}

// =============================================================================
// Scaling Tests
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, Scaling_ByStateCount) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    std::cout << "\n=== Scaling by State Count ===\n";
    std::cout << std::setw(8) << "Steps" << std::setw(10) << "States"
              << std::setw(12) << "UT(ms)" << std::setw(12) << "UT-Inc(ms)"
              << std::setw(10) << "Ratio" << std::setw(12) << "Reuse%" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (size_t steps = 1; steps <= 6; ++steps) {
        auto ut = run_with_strategy(v2::HashStrategy::UniquenessTree, {rule}, initial, steps);
        auto utinc = run_with_strategy(v2::HashStrategy::IncrementalUniquenessTree, {rule}, initial, steps);

        double ratio = utinc.time_ms / ut.time_ms;
        double reuse_pct = (utinc.reused + utinc.recomputed > 0)
            ? 100.0 * utinc.reused / (utinc.reused + utinc.recomputed)
            : 0.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(8) << steps
                  << std::setw(10) << ut.num_states
                  << std::setw(12) << ut.time_ms
                  << std::setw(12) << utinc.time_ms
                  << std::setw(10) << ratio
                  << std::setw(11) << std::setprecision(1) << reuse_pct << "%"
                  << "\n";
    }
}

TEST_F(UTvsUTIncPerformanceTest, Scaling_ByGraphSize) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::cout << "\n=== Scaling by Initial Graph Size (Disconnected Components) ===\n";
    std::cout << std::setw(10) << "Edges" << std::setw(12) << "UT(ms)"
              << std::setw(12) << "UT-Inc(ms)" << std::setw(10) << "Ratio"
              << std::setw(12) << "Reuse%" << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (size_t num_edges : {10, 20, 50, 100, 200}) {
        std::vector<std::vector<v2::VertexId>> initial;
        for (v2::VertexId v = 0; v < num_edges * 2; v += 2) {
            initial.push_back({v, static_cast<v2::VertexId>(v + 1)});
        }

        auto ut = run_with_strategy(v2::HashStrategy::UniquenessTree, {rule}, initial, 1);
        auto utinc = run_with_strategy(v2::HashStrategy::IncrementalUniquenessTree, {rule}, initial, 1);

        double ratio = utinc.time_ms / ut.time_ms;
        double reuse_pct = (utinc.reused + utinc.recomputed > 0)
            ? 100.0 * utinc.reused / (utinc.reused + utinc.recomputed)
            : 0.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(10) << num_edges
                  << std::setw(12) << ut.time_ms
                  << std::setw(12) << utinc.time_ms
                  << std::setw(10) << ratio
                  << std::setw(11) << std::setprecision(1) << reuse_pct << "%"
                  << "\n";
    }
}

// =============================================================================
// Cost Breakdown Analysis
// =============================================================================

TEST_F(UTvsUTIncPerformanceTest, CostBreakdown_OverheadSources) {
    std::cout << "\n=== UT-Inc Overhead Source Analysis ===\n";
    std::cout << "Comparing overhead on connected vs disconnected graphs\n";
    std::cout << "to isolate different cost components.\n\n";

    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    // Test 1: Disconnected components (high reuse, should show bloom filter benefit)
    {
        std::vector<std::vector<v2::VertexId>> initial;
        for (v2::VertexId v = 0; v < 100; v += 2) {
            initial.push_back({v, static_cast<v2::VertexId>(v + 1)});
        }

        auto ut = run_with_strategy(v2::HashStrategy::UniquenessTree, {rule}, initial, 1);
        auto utinc = run_with_strategy(v2::HashStrategy::IncrementalUniquenessTree, {rule}, initial, 1);

        std::cout << "Disconnected (50 components, high reuse):\n";
        std::cout << "  UT: " << std::fixed << std::setprecision(2) << ut.time_ms << "ms\n";
        std::cout << "  UT-Inc: " << utinc.time_ms << "ms (reuse="
                  << std::setprecision(1) << (100.0 * utinc.reused / (utinc.reused + utinc.recomputed)) << "%)\n";
        std::cout << "  Ratio: " << std::setprecision(2) << (utinc.time_ms / ut.time_ms) << "x\n\n";
    }

    // Test 2: Connected chain (low reuse, shows base overhead)
    {
        std::vector<std::vector<v2::VertexId>> initial;
        for (v2::VertexId v = 0; v < 50; ++v) {
            initial.push_back({v, static_cast<v2::VertexId>(v + 1)});
        }

        auto ut = run_with_strategy(v2::HashStrategy::UniquenessTree, {rule}, initial, 1);
        auto utinc = run_with_strategy(v2::HashStrategy::IncrementalUniquenessTree, {rule}, initial, 1);

        std::cout << "Connected chain (50 edges, low reuse):\n";
        std::cout << "  UT: " << std::fixed << std::setprecision(2) << ut.time_ms << "ms\n";
        std::cout << "  UT-Inc: " << utinc.time_ms << "ms (reuse="
                  << std::setprecision(1) << (100.0 * utinc.reused / (utinc.reused + utinc.recomputed)) << "%)\n";
        std::cout << "  Ratio: " << std::setprecision(2) << (utinc.time_ms / ut.time_ms) << "x\n\n";

        // This shows the base overhead when reuse doesn't help
        double overhead = utinc.time_ms - ut.time_ms;
        double overhead_per_state = overhead / utinc.num_states;
        std::cout << "  Base overhead: " << overhead << "ms total, "
                  << (overhead_per_state * 1000.0) << " us/state\n";
    }

    std::cout << "\nConclusion: If ratio is high on connected graphs, the overhead is from:\n";
    std::cout << "  1. Per-state cache lookup (checking parent cache validity)\n";
    std::cout << "  2. Parent cache index building (O(1) lookup map construction)\n";
    std::cout << "  3. Bloom filter checks (O(affected_vertices) per vertex)\n";
    std::cout << "  4. Per-state adjacency building (lazy, only when recomputing)\n";
    std::cout << "  5. Cache storage overhead (atomic CAS, memory allocation)\n";
}

