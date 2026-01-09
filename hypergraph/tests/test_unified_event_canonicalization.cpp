#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <memory>
#include <iomanip>

// v2 (unified) includes
#include "hypergraph/arena.hpp"
#include "hypergraph/types.hpp"
#include "hypergraph/bitset.hpp"
#include "hypergraph/unified_hypergraph.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/parallel_evolution.hpp"

namespace v2 = hypergraph::unified;

// =============================================================================
// Event Canonicalization Tests (ByStateAndEdges mode)
// =============================================================================
// These tests verify v2's ByStateAndEdges mode against expected counts from
// v1's disabled "Automatic mode" tests in test_determinism_fuzzing.cpp.
//
// v1 API:
//   - canonicalize_events=true, full_event_canonicalization=false -> "Automatic mode" (DISABLED in v1)
//   - canonicalize_events=true, full_event_canonicalization=true  -> "Full mode" (ByState)
//
// v2 API:
//   - EventCanonicalizationMode::ByStateAndEdges -> should match v1's Automatic mode
//   - EventCanonicalizationMode::ByState -> matches v1's Full mode
//
// Expected counts come from v1's disabled test_determinism_fuzzing.cpp tests.
// These expected values may need verification with wolframscript.

class V2_EventCanonByStateAndEdgesTest : public ::testing::Test {
protected:
    struct EvolutionCounts {
        size_t num_states;
        size_t num_events;
        size_t num_causal;
        size_t num_branchial;
    };

    // Run unified evolution with ByStateAndEdges mode
    EvolutionCounts run_unified(
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_event_signature_keys(v2::EVENT_SIG_AUTOMATIC);
        v2::ParallelEvolutionEngine engine(hg.get(), 1);  // single thread for determinism

        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }

        engine.evolve(initial, steps);

        return {
            hg->num_canonical_states(),
            engine.num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges()
        };
    }

    // Compare counts against expected and print results
    void verify(
        const std::string& test_name,
        size_t step,
        const EvolutionCounts& actual,
        const EvolutionCounts& expected
    ) {
        std::cout << test_name << " (step " << step << "):\n";
        std::cout << "  expected: states=" << expected.num_states
                  << ", events=" << expected.num_events
                  << ", causal=" << expected.num_causal
                  << ", branchial=" << expected.num_branchial << "\n";
        std::cout << "  actual:   states=" << actual.num_states
                  << ", events=" << actual.num_events
                  << ", causal=" << actual.num_causal
                  << ", branchial=" << actual.num_branchial << "\n";

        EXPECT_EQ(expected.num_states, actual.num_states)
            << "State count mismatch at step " << step;
        EXPECT_EQ(expected.num_events, actual.num_events)
            << "Event count mismatch at step " << step;
        // Causal/branchial may need separate verification
    }
};

// =============================================================================
// TestCase1: SimpleRule {1,2} -> {1,2}, {2,3}
// From v1's disabled TestCase1_SimpleRule_EventCanonicalization_Automatic
// Expected (step -> {states, events, causal, branchial}):
//   1 -> {2, 1, 0, 0}
//   2 -> {4, 3, 0, 2}
//   3 -> {8, 9, 0, 8}
//   4 -> {17, 26, 0, 32}
// =============================================================================

TEST_F(V2_EventCanonByStateAndEdgesTest, SimpleRule_Step1) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    auto actual = run_unified({rule}, initial, 1);
    verify("SimpleRule", 1, actual, {2, 1, 0, 0});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, SimpleRule_Step2) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    auto actual = run_unified({rule}, initial, 2);
    verify("SimpleRule", 2, actual, {4, 3, 0, 2});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, SimpleRule_Step3) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    auto actual = run_unified({rule}, initial, 3);
    verify("SimpleRule", 3, actual, {8, 9, 0, 8});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, SimpleRule_Step4) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    auto actual = run_unified({rule}, initial, 4);
    verify("SimpleRule", 4, actual, {17, 26, 0, 32});
}

// =============================================================================
// TestCase2: TwoEdgeRule {1,2},{2,3} -> {1,2},{2,3},{2,4}
// Initial: triangle {{1,2}, {2,3}, {3,1}}
// From v1's disabled TestCase2_TwoEdgeRule_EventCanonicalization_Automatic
// Expected:
//   1 -> {2, 3, 3, 0}
//   2 -> {4, 7, 18, 12}
// =============================================================================

TEST_F(V2_EventCanonByStateAndEdgesTest, TwoEdgeRule_Triangle_Step1) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    // Triangle: 0-1-2-0
    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 1);
    verify("TwoEdgeRule_Triangle", 1, actual, {2, 3, 3, 0});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, TwoEdgeRule_Triangle_Step2) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 2);
    verify("TwoEdgeRule_Triangle", 2, actual, {4, 7, 18, 12});
}

// =============================================================================
// TestCase3: HyperedgeRule {1,2,3} -> {1,2},{1,3},{1,4}
// Initial: single ternary edge {{1,2,3}}
// From v1's disabled TestCase3_HyperedgeRule_EventCanonicalization_Automatic
// Expected (all steps give same result - no more matches after step 1):
//   1-4 -> {2, 1, 0, 0}
// =============================================================================

TEST_F(V2_EventCanonByStateAndEdgesTest, HyperedgeRule_Step1) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    auto actual = run_unified({rule}, initial, 1);
    verify("HyperedgeRule", 1, actual, {2, 1, 0, 0});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, HyperedgeRule_Step2) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    auto actual = run_unified({rule}, initial, 2);
    // No more ternary edges to match, so no new events
    verify("HyperedgeRule", 2, actual, {2, 1, 0, 0});
}

// =============================================================================
// TestCase6: TwoEdgeRuleVariant {1,2},{2,3} -> {1,4},{4,2},{2,3}
// Initial: triangle {{1,2}, {2,3}, {3,1}}
// From v1's disabled TestCase6_TwoEdgeRuleVariant_EventCanonicalization_Automatic
// Expected:
//   1 -> {2, 3, 3, 0}
//   2 -> {4, 9, 18, 6}
// =============================================================================

TEST_F(V2_EventCanonByStateAndEdgesTest, TwoEdgeRuleVariant_Triangle_Step1) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 1);
    verify("TwoEdgeRuleVariant_Triangle", 1, actual, {2, 3, 3, 0});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, TwoEdgeRuleVariant_Triangle_Step2) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 2);
    verify("TwoEdgeRuleVariant_Triangle", 2, actual, {4, 9, 18, 6});
}

// =============================================================================
// TestCase7: TwoEdgeRuleWithSelfLoops {1,2},{2,3} -> {1,4},{4,2},{2,2}
// Initial: triangle {{1,2}, {2,3}, {3,1}}
// From v1's disabled TestCase7_TwoEdgeRuleWithSelfLoops_EventCanonicalization_Automatic
// Expected:
//   1 -> {2, 3, 3, 0}
//   2 -> {4, 9, 18, 6}
// =============================================================================

TEST_F(V2_EventCanonByStateAndEdgesTest, TwoEdgeRuleWithSelfLoops_Step1) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 1})  // self-loop
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 1);
    verify("TwoEdgeRuleWithSelfLoops", 1, actual, {2, 3, 3, 0});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, TwoEdgeRuleWithSelfLoops_Step2) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 1})  // self-loop
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 2);
    verify("TwoEdgeRuleWithSelfLoops", 2, actual, {4, 9, 18, 6});
}

// =============================================================================
// TestCase8: ComplexTwoEdgeRule {1,2},{2,3} -> {1,4},{4,2},{2,3},{3,5}
// Initial: triangle {{1,2}, {2,3}, {3,1}}
// From v1's disabled TestCase8_ComplexTwoEdgeRule_EventCanonicalization_Automatic
// Expected:
//   1 -> {2, 3, 3, 0}
//   2 -> {4, 15, 18, 12}
// =============================================================================

TEST_F(V2_EventCanonByStateAndEdgesTest, ComplexTwoEdgeRule_Step1) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 2})
        .rhs({2, 4})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 1);
    verify("ComplexTwoEdgeRule", 1, actual, {2, 3, 3, 0});
}

TEST_F(V2_EventCanonByStateAndEdgesTest, ComplexTwoEdgeRule_Step2) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 2})
        .rhs({2, 4})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    auto actual = run_unified({rule}, initial, 2);
    verify("ComplexTwoEdgeRule", 2, actual, {4, 15, 18, 12});
}

// =============================================================================
// Hash Strategy Comparison Tests: WL vs UniquenessTree vs IncrementalUniquenessTree
// =============================================================================
// Compare all three hash strategies across all event canonicalization modes.
// All strategies should produce identical counts.

#include <chrono>

class HashStrategyComparisonTest : public ::testing::Test {
protected:
    struct EvolutionResult {
        size_t num_states;
        size_t num_events;
        size_t num_causal;
        size_t num_branchial;
        double runtime_ms;

        bool counts_equal(const EvolutionResult& o) const {
            return num_states == o.num_states && num_events == o.num_events &&
                   num_causal == o.num_causal && num_branchial == o.num_branchial;
        }
    };

    EvolutionResult run_with_strategy(
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        v2::HashStrategy strategy,
        v2::EventSignatureKeys mode
    ) {
        auto start = std::chrono::high_resolution_clock::now();

        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_hash_strategy(strategy);
        hg->set_event_signature_keys(mode);
        v2::ParallelEvolutionEngine engine(hg.get(), 1);

        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }
        engine.evolve(initial, steps);

        auto end = std::chrono::high_resolution_clock::now();
        double runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return {
            hg->num_canonical_states(),
            hg->num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges(),
            runtime_ms
        };
    }

    static const char* mode_name(v2::EventSignatureKeys mode) {
        if (mode == v2::EVENT_SIG_NONE) return "None";
        if (mode == v2::EVENT_SIG_FULL) return "ByState";
        if (mode == v2::EVENT_SIG_AUTOMATIC) return "ByStateAndEdges";
        return "Unknown";
    }

    static const char* strategy_name(v2::HashStrategy strategy) {
        switch (strategy) {
            case v2::HashStrategy::UniquenessTree: return "UT";
            case v2::HashStrategy::IncrementalUniquenessTree: return "UT-Inc";
            case v2::HashStrategy::WL: return "WL";
            case v2::HashStrategy::IncrementalWL: return "WL-Inc";
        }
        return "Unknown";
    }

    void compare_all_modes(
        const std::string& test_name,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps
    ) {
        std::cout << "\n" << test_name << " (step " << steps << "):\n";
        std::cout << std::setw(20) << "Mode"
                  << std::setw(10) << "Strategy"
                  << std::setw(8) << "States"
                  << std::setw(8) << "Events"
                  << std::setw(8) << "Causal"
                  << std::setw(10) << "Branchial"
                  << std::setw(12) << "Time (ms)" << "\n";
        std::cout << std::string(78, '-') << "\n";

        std::vector<v2::HashStrategy> strategies = {
            v2::HashStrategy::UniquenessTree,
            v2::HashStrategy::IncrementalUniquenessTree,
            v2::HashStrategy::WL,
            v2::HashStrategy::IncrementalWL
        };

        for (auto mode : {v2::EVENT_SIG_NONE,
                          v2::EVENT_SIG_FULL,
                          v2::EVENT_SIG_AUTOMATIC}) {

            EvolutionResult results[4];
            for (size_t i = 0; i < strategies.size(); ++i) {
                results[i] = run_with_strategy(rules, initial, steps, strategies[i], mode);
            }

            // Print results for each strategy
            for (size_t i = 0; i < strategies.size(); ++i) {
                const auto& r = results[i];
                std::cout << std::setw(20) << (i == 0 ? mode_name(mode) : "")
                          << std::setw(10) << strategy_name(strategies[i])
                          << std::setw(8) << r.num_states
                          << std::setw(8) << r.num_events
                          << std::setw(8) << r.num_causal
                          << std::setw(10) << r.num_branchial
                          << std::setw(12) << std::fixed << std::setprecision(2) << r.runtime_ms;

                // Show timing comparison vs UT (baseline)
                if (i > 0 && results[0].counts_equal(r)) {
                    double speedup = results[0].runtime_ms / r.runtime_ms;
                    if (speedup > 1.1) {
                        std::cout << "  " << strategy_name(strategies[i]) << " "
                                  << std::fixed << std::setprecision(1) << speedup << "x faster";
                    } else if (speedup < 0.9) {
                        std::cout << "  UT " << std::fixed << std::setprecision(1) << (1.0/speedup) << "x faster";
                    } else {
                        std::cout << "  ~same";
                    }
                }
                std::cout << "\n";
            }

            // Verify all strategies produce same counts
            bool all_match = results[0].counts_equal(results[1]) &&
                             results[0].counts_equal(results[2]) &&
                             results[0].counts_equal(results[3]);
            if (!all_match) {
                std::cout << "  *** MISMATCH between strategies! ***\n";
            }

            EXPECT_TRUE(results[0].counts_equal(results[1]))
                << "UT vs UT-Inc mismatch in " << mode_name(mode);
            EXPECT_TRUE(results[0].counts_equal(results[2]))
                << "UT vs WL mismatch in " << mode_name(mode);
            EXPECT_TRUE(results[0].counts_equal(results[3]))
                << "UT vs WL-Inc mismatch in " << mode_name(mode);
        }
    }

    // Run with stats for UT-Inc to show cache effectiveness
    void run_with_stats(
        const std::string& test_name,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_hash_strategy(v2::HashStrategy::IncrementalUniquenessTree);
        hg->set_event_signature_keys(v2::EVENT_SIG_FULL);
        hg->reset_incremental_tree_stats();

        v2::ParallelEvolutionEngine engine(hg.get(), 1);
        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }
        engine.evolve(initial, steps);

        auto [reused, recomputed] = hg->incremental_tree_stats();
        double ratio = (reused + recomputed > 0)
            ? 100.0 * reused / (reused + recomputed)
            : 0.0;

        std::cout << test_name << " (step " << steps << ") UT-Inc stats:\n";
        std::cout << "  Reused: " << reused << ", Recomputed: " << recomputed
                  << ", Reuse ratio: " << std::fixed << std::setprecision(1) << ratio << "%\n";
        std::cout << "  Stored caches: " << hg->num_stored_caches()
                  << ", States: " << hg->num_states() << "\n";
    }
};

TEST_F(HashStrategyComparisonTest, SimpleRule_Step5) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    compare_all_modes("SimpleRule", {rule}, {{0, 1}}, 5);
}

TEST_F(HashStrategyComparisonTest, TwoEdgeRule_Triangle_Step3) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 3})
        .rhs({3, 2})
        .build();

    compare_all_modes("TwoEdgeRule_Triangle", {rule}, {{0, 1}, {1, 2}, {2, 0}}, 3);
}

TEST_F(HashStrategyComparisonTest, HyperedgeRule_Step4) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1, 2})
        .rhs({1, 2, 3})
        .build();

    compare_all_modes("HyperedgeRule", {rule}, {{0, 1, 2}}, 4);
}

TEST_F(HashStrategyComparisonTest, TwoEdgeRuleWithSelfLoops_Step3) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({2, 0})
        .rhs({0, 0})  // self-loop
        .build();

    compare_all_modes("TwoEdgeRuleWithSelfLoops", {rule}, {{0, 1}, {1, 2}, {2, 0}}, 3);
}

// Test incremental stats - check what ratio of vertex hashes are being reused
TEST_F(HashStrategyComparisonTest, IncrementalStats_SimpleRule) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    run_with_stats("SimpleRule", {rule}, {{0, 1}}, 6);
}

// Larger test to see if incremental helps with bigger states
TEST_F(HashStrategyComparisonTest, LargerGraph_Step4) {
    // Rule that grows the graph more significantly
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 2})
        .rhs({2, 1})
        .rhs({1, 3})
        .build();

    // Start with a small cycle
    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

    compare_all_modes("LargerGraph", {rule}, initial, 4);
    run_with_stats("LargerGraph", {rule}, initial, 4);
}

// Large initial state with 2-edge rule (consumes 2, produces 3)
// Single step to measure hashing overhead on large states
TEST_F(HashStrategyComparisonTest, LargeInitial_TwoEdgeRule) {
    // Classic Wolfram rule: {x,y},{y,z} -> {x,y},{y,w},{w,z}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 3})
        .rhs({3, 2})
        .build();

    // Large initial state: chain of 20 edges (many matches)
    std::vector<std::vector<v2::VertexId>> initial;
    for (v2::VertexId i = 0; i < 20; ++i) {
        initial.push_back({i, i + 1});
    }

    std::cout << "\nLarge chain: " << initial.size() << " edges\n";
    compare_all_modes("LargeInitial_TwoEdge", {rule}, initial, 1);
    run_with_stats("LargeInitial_TwoEdge", {rule}, initial, 1);
}

// Large initial state with 3-edge rule (consumes 3, produces 4)
TEST_F(HashStrategyComparisonTest, LargeInitial_ThreeEdgeRule) {
    // 3-edge rule: {x,y},{y,z},{z,w} -> {x,y},{y,z},{z,w},{w,v}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .lhs({2, 3})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({2, 3})
        .rhs({3, 4})
        .build();

    // Large initial state: chain of 25 edges
    std::vector<std::vector<v2::VertexId>> initial;
    for (v2::VertexId i = 0; i < 25; ++i) {
        initial.push_back({i, i + 1});
    }

    std::cout << "\nLarge chain (3-edge rule): " << initial.size() << " edges\n";
    compare_all_modes("LargeInitial_ThreeEdge", {rule}, initial, 1);
    run_with_stats("LargeInitial_ThreeEdge", {rule}, initial, 1);
}

// Large sparse graph - disconnected components
// This is where incremental SHOULD shine - local rewrites don't affect distant vertices
TEST_F(HashStrategyComparisonTest, LargeSparse_TwoEdgeRule) {
    // Simple growth rule
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 3})
        .rhs({3, 2})
        .build();

    // Large sparse initial state: 30 disconnected edge pairs
    // Each pair is {v, v+1}, {v+1, v+2} - matches the rule
    std::vector<std::vector<v2::VertexId>> initial;
    v2::VertexId v = 0;

    for (int i = 0; i < 30; ++i) {
        initial.push_back({v, v + 1});
        initial.push_back({v + 1, v + 2});
        v += 4;  // Gap between components
    }

    std::cout << "\nLarge sparse state: " << initial.size() << " edges, "
              << v << " vertices (disconnected components)\n";
    compare_all_modes("LargeSparse", {rule}, initial, 1);
    run_with_stats("LargeSparse", {rule}, initial, 1);
}
