#include <gtest/gtest.h>
#include <vector>
#include <set>
#include <iostream>
#include <memory>

#include "hypergraph/unified/arena.hpp"
#include "hypergraph/unified/types.hpp"
#include "hypergraph/unified/bitset.hpp"
#include "hypergraph/unified/unified_hypergraph.hpp"
#include "hypergraph/unified/pattern.hpp"
#include "hypergraph/unified/rewriter.hpp"
#include "hypergraph/unified/parallel_evolution.hpp"

namespace v2 = hypergraph::unified;

// =============================================================================
// Unified Determinism Fuzzing Tests
// =============================================================================
// Comprehensive determinism testing modeled after v1 DeterminismFuzzing tests.
// Runs evolution multiple times to detect any non-determinism in the system.

class Unified_DeterminismFuzzingTest : public ::testing::Test {
protected:
    struct TestResult {
        size_t num_states;
        size_t num_events;
        size_t num_causal;
        size_t num_branchial;
        uint32_t final_vertex_id;
        uint32_t final_edge_id;
        uint32_t final_event_id;

        bool operator==(const TestResult& other) const {
            return num_states == other.num_states &&
                   num_events == other.num_events &&
                   num_causal == other.num_causal &&
                   num_branchial == other.num_branchial &&
                   final_vertex_id == other.final_vertex_id &&
                   final_edge_id == other.final_edge_id &&
                   final_event_id == other.final_event_id;
        }

        bool operator<(const TestResult& other) const {
            if (num_states != other.num_states) return num_states < other.num_states;
            if (num_events != other.num_events) return num_events < other.num_events;
            if (num_causal != other.num_causal) return num_causal < other.num_causal;
            if (num_branchial != other.num_branchial) return num_branchial < other.num_branchial;
            if (final_vertex_id != other.final_vertex_id) return final_vertex_id < other.final_vertex_id;
            if (final_edge_id != other.final_edge_id) return final_edge_id < other.final_edge_id;
            return final_event_id < other.final_event_id;
        }
    };

    struct DeterminismResults {
        std::set<size_t> unique_states;
        std::set<size_t> unique_events;
        std::set<size_t> unique_causal;
        std::set<size_t> unique_branchial;
        std::set<uint32_t> unique_final_vertex_ids;
        std::set<uint32_t> unique_final_edge_ids;
        std::set<uint32_t> unique_final_event_ids;
        std::vector<TestResult> all_results;
    };

    TestResult run_single_evolution(
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        size_t num_threads = 0  // 0 = use hardware_concurrency
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), num_threads);

        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }

        engine.evolve(initial, steps);

        // Use canonical states (not raw states) for determinism checking
        // Raw state count includes "wasted" states from parallel race conditions
        return {
            engine.num_canonical_states(),
            engine.num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges(),
            hg->num_vertices(),
            hg->num_edges(),
            static_cast<uint32_t>(engine.num_events())
        };
    }

    DeterminismResults fuzz_test_rules(
        const std::string& test_name,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        int num_runs = 50
    ) {
        DeterminismResults results;

        for (int i = 0; i < num_runs; ++i) {
            TestResult result = run_single_evolution(rules, initial, steps);
            results.all_results.push_back(result);
            results.unique_states.insert(result.num_states);
            results.unique_events.insert(result.num_events);
            results.unique_causal.insert(result.num_causal);
            results.unique_branchial.insert(result.num_branchial);
            results.unique_final_vertex_ids.insert(result.final_vertex_id);
            results.unique_final_edge_ids.insert(result.final_edge_id);
            results.unique_final_event_ids.insert(result.final_event_id);
        }

        // Report detailed results
        std::cout << test_name << " fuzz test results after " << num_runs << " runs:\n";
        std::cout << "  States deterministic: " << (results.unique_states.size() == 1 ? "YES" : "NO")
                  << " (" << results.unique_states.size() << " unique values)\n";
        std::cout << "  Events deterministic: " << (results.unique_events.size() == 1 ? "YES" : "NO")
                  << " (" << results.unique_events.size() << " unique values)\n";
        std::cout << "  Causal deterministic: " << (results.unique_causal.size() == 1 ? "YES" : "NO")
                  << " (" << results.unique_causal.size() << " unique values)\n";
        std::cout << "  Branchial deterministic: " << (results.unique_branchial.size() == 1 ? "YES" : "NO")
                  << " (" << results.unique_branchial.size() << " unique values)\n";
        std::cout << "  Final Vertex ID deterministic: " << (results.unique_final_vertex_ids.size() == 1 ? "YES" : "NO")
                  << " (" << results.unique_final_vertex_ids.size() << " unique values)\n";
        std::cout << "  Final Edge ID deterministic: " << (results.unique_final_edge_ids.size() == 1 ? "YES" : "NO")
                  << " (" << results.unique_final_edge_ids.size() << " unique values)\n";
        std::cout << "  Final Event ID deterministic: " << (results.unique_final_event_ids.size() == 1 ? "YES" : "NO")
                  << " (" << results.unique_final_event_ids.size() << " unique values)\n";

        if (results.unique_states.size() == 1 && results.unique_events.size() == 1) {
            const auto& r = results.all_results[0];
            std::cout << "  Result: States=" << r.num_states
                      << ", Events=" << r.num_events
                      << ", Causal=" << r.num_causal
                      << ", Branchial=" << r.num_branchial
                      << ", FinalVertexID=" << r.final_vertex_id
                      << ", FinalEdgeID=" << r.final_edge_id
                      << ", FinalEventID=" << r.final_event_id << "\n";
        }

        // Assertions - check counts are deterministic
        // Note: With parallel execution, exact IDs (vertex, edge, event) may vary
        // but the counts should always be the same
        EXPECT_EQ(results.unique_states.size(), 1u) << "Canonical state count non-deterministic";
        EXPECT_EQ(results.unique_events.size(), 1u) << "Event count non-deterministic";
        EXPECT_EQ(results.unique_causal.size(), 1u) << "Causal edge count non-deterministic";
        EXPECT_EQ(results.unique_branchial.size(), 1u) << "Branchial edge count non-deterministic";
        // Don't check exact IDs - they depend on thread scheduling order
        // EXPECT_EQ(results.unique_final_vertex_ids.size(), 1u) << "Final vertex IDs non-deterministic";
        // EXPECT_EQ(results.unique_final_edge_ids.size(), 1u) << "Final edge IDs non-deterministic";
        // EXPECT_EQ(results.unique_final_event_ids.size(), 1u) << "Final event IDs non-deterministic";

        return results;
    }

    void validate_against_expected(
        const std::string& test_name,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        const std::vector<std::pair<size_t, TestResult>>& expected
    ) {
        std::cout << "\n=== Validating Expected Results for " << test_name << " ===\n";

        for (const auto& [steps, expected_result] : expected) {
            TestResult actual = run_single_evolution(rules, initial, steps);

            std::cout << "Step " << steps << " - Expected: {"
                      << expected_result.num_states << ", "
                      << expected_result.num_events << ", "
                      << expected_result.num_branchial << ", "
                      << expected_result.num_causal << "}\n";
            std::cout << "Step " << steps << " - Actual:   {"
                      << actual.num_states << ", "
                      << actual.num_events << ", "
                      << actual.num_branchial << ", "
                      << actual.num_causal << "}\n";

            EXPECT_EQ(actual.num_states, expected_result.num_states) << "States mismatch at step " << steps;
            EXPECT_EQ(actual.num_events, expected_result.num_events) << "Events mismatch at step " << steps;
        }
    }
};

// =============================================================================
// TestCase1: Simple Rule {{x,y}} -> {{x,y},{y,z}}
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase1_SimpleRule_Fuzz) {
    // Rule: {{x,y}} -> {{x,y},{y,z}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    fuzz_test_rules("SimpleRule", {rule}, initial, 4, 50);
}

TEST_F(Unified_DeterminismFuzzingTest, TestCase1_SimpleRule_Steps) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    // Test at different step counts
    for (size_t steps = 1; steps <= 5; ++steps) {
        auto result = run_single_evolution({rule}, initial, steps);
        std::cout << "SimpleRule step " << steps
                  << ": states=" << result.num_states
                  << ", events=" << result.num_events
                  << ", causal=" << result.num_causal << "\n";
    }
}

// =============================================================================
// TestCase2: Two-Edge Rule {{x,y},{y,z}} -> {{x,y},{y,z},{y,w}}
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase2_TwoEdgeRule_Fuzz) {
    // Rule: {{x,y},{y,z}} -> {{x,y},{y,z},{y,w}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    // Triangle initial state
    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    fuzz_test_rules("TwoEdgeRule_Triangle", {rule}, initial, 3, 50);
}

TEST_F(Unified_DeterminismFuzzingTest, TestCase2_TwoEdgeRule_Chain) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    // Chain initial state
    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}};

    fuzz_test_rules("TwoEdgeRule_Chain", {rule}, initial, 4, 50);
}

// =============================================================================
// TestCase3: Hyperedge Rule {{x,y,z}} -> {{x,y},{x,z},{x,w}}
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase3_HyperedgeRule_Fuzz) {
    // Rule: {{x,y,z}} -> {{x,y},{x,z},{x,w}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    fuzz_test_rules("HyperedgeRule", {rule}, initial, 4, 50);
}

// =============================================================================
// TestCase4: Multi-Rule System
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase4_MultiRule_Fuzz) {
    // Rule 1: {{x,y,z}} -> {{x,y},{x,z},{x,w}}
    v2::RewriteRule rule1 = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    // Rule 2: {{x,y}} -> {{x,y},{x,z}}
    v2::RewriteRule rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({0, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    fuzz_test_rules("MultiRule", {rule1, rule2}, initial, 4, 50);
}

// =============================================================================
// TestCase5: Complex Two-Rule System
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase5_ComplexTwoRuleSystem_Fuzz) {
    // Rule 1: {{x,y,z}} -> {{x,y},{x,z},{x,w}}
    v2::RewriteRule rule1 = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    // Rule 2: {{x,y},{x,z}} -> {{x,y},{x,z},{y,w}}
    v2::RewriteRule rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .lhs({0, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    fuzz_test_rules("ComplexTwoRuleSystem", {rule1, rule2}, initial, 3, 50);
}

// =============================================================================
// TestCase6: Two-Edge Rule Variant with Chain Initial
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase6_TwoEdgeRuleVariant_Fuzz) {
    // Rule: {{x,y},{y,z}} -> {{x,y},{y,z},{y,w}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}};

    fuzz_test_rules("TwoEdgeRuleVariant", {rule}, initial, 4, 50);
}

// =============================================================================
// TestCase7: Self-Loop Initial State
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase7_SelfLoops_Fuzz) {
    // Rule: {{x,y},{y,z}} -> {{x,y},{y,z},{y,w}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    // Self-loop initial state
    std::vector<std::vector<v2::VertexId>> initial = {{0, 0}, {0, 0}};

    fuzz_test_rules("SelfLoops", {rule}, initial, 3, 50);
}

// =============================================================================
// TestCase8: Complex Two-Edge Rule
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase8_ComplexTwoEdgeRule_Fuzz) {
    // Rule: {{x,y},{y,z}} -> {{w,x},{x,w},{y,z},{w,z}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({3, 0})
        .rhs({0, 3})
        .rhs({1, 2})
        .rhs({3, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}};

    fuzz_test_rules("ComplexTwoEdgeRule", {rule}, initial, 3, 50);
}

// =============================================================================
// TestCase9: Complex Two-Edge Rule with Self-Loops
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase9_ComplexTwoEdgeRuleSelfLoops_Fuzz) {
    // Rule: {{x,y},{y,z}} -> {{w,x},{x,w},{y,z},{w,z}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({3, 0})
        .rhs({0, 3})
        .rhs({1, 2})
        .rhs({3, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 0}, {0, 0}};

    fuzz_test_rules("ComplexTwoEdgeRuleSelfLoops", {rule}, initial, 2, 50);
}

// =============================================================================
// TestCase10: Another Two-Edge Rule
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase10_AnotherTwoEdgeRule_Fuzz) {
    // Rule: {{x,y},{y,z}} -> {{x,z},{y,z},{z,w}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 2})
        .rhs({1, 2})
        .rhs({2, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}};

    fuzz_test_rules("AnotherTwoEdgeRule", {rule}, initial, 4, 50);
}

// =============================================================================
// TestCase11: Another Two-Edge Rule with Self-Loops
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase11_AnotherTwoEdgeRuleSelfLoops_Fuzz) {
    // Rule: {{x,y},{y,z}} -> {{x,z},{y,z},{z,w}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 2})
        .rhs({1, 2})
        .rhs({2, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 0}, {0, 0}};

    fuzz_test_rules("AnotherTwoEdgeRuleSelfLoops", {rule}, initial, 3, 50);
}

// =============================================================================
// TestCase12: Complex Three-Edge Rule (Ternary Edge)
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, TestCase12_ComplexThreeEdgeRule_Fuzz) {
    // Rule: {{x,y,z},{w,x}} -> {{x,w,u},{z,y},{z,w}}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1, 2})
        .lhs({3, 0})
        .rhs({0, 3, 4})
        .rhs({2, 1})
        .rhs({2, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 0, 0}, {0, 0}};

    fuzz_test_rules("ComplexThreeEdgeRule", {rule}, initial, 4, 50);
}

// =============================================================================
// Match Forwarding Validation Tests
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, MatchForwarding_SimpleRule) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    // Run multiple times and check match forwarding stats are consistent
    std::set<size_t> unique_forwarded;
    std::set<size_t> unique_invalidated;
    std::set<size_t> unique_new_discovered;

    for (int i = 0; i < 20; ++i) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), 4);
        engine.add_rule(rule);
        engine.evolve(initial, 3);

        const auto& stats = engine.stats();
        unique_forwarded.insert(stats.matches_forwarded.load());
        unique_invalidated.insert(stats.matches_invalidated.load());
        unique_new_discovered.insert(stats.new_matches_discovered.load());
    }

    std::cout << "Match forwarding stats over 20 runs:\n";
    std::cout << "  Forwarded: " << unique_forwarded.size() << " unique values\n";
    std::cout << "  Invalidated: " << unique_invalidated.size() << " unique values\n";
    std::cout << "  New discovered: " << unique_new_discovered.size() << " unique values\n";

    EXPECT_EQ(unique_forwarded.size(), 1u) << "Match forwarding non-deterministic";
    EXPECT_EQ(unique_invalidated.size(), 1u) << "Match invalidation non-deterministic";
    EXPECT_EQ(unique_new_discovered.size(), 1u) << "New match discovery non-deterministic";
}

// =============================================================================
// Extended Fuzz Tests (longer evolution)
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, Extended_SimpleRule_5Steps) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    fuzz_test_rules("SimpleRule_5Steps", {rule}, initial, 5, 30);
}

TEST_F(Unified_DeterminismFuzzingTest, Extended_TwoEdgeRule_4Steps) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 0}};

    fuzz_test_rules("TwoEdgeRule_4Steps", {rule}, initial, 4, 30);
}

// =============================================================================
// Stress Test: Many Runs
// =============================================================================

TEST_F(Unified_DeterminismFuzzingTest, StressTest_100Runs) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};

    fuzz_test_rules("StressTest_100Runs", {rule}, initial, 3, 100);
}

