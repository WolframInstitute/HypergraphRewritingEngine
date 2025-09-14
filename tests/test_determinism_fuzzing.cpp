#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <vector>
#include <thread>
#include "test_helpers.hpp"

using namespace hypergraph;

class DeterminismFuzzingTest : public ::testing::Test {
protected:
    struct TestResult {
        std::size_t num_states;
        std::size_t num_events;
        std::size_t num_branchial;
        std::size_t num_causal;
        GlobalVertexId final_vertex_id;
        GlobalEdgeId final_edge_id;
        EventId final_event_id;

        bool operator==(const TestResult& other) const {
            return num_states == other.num_states &&
                   num_events == other.num_events &&
                   num_branchial == other.num_branchial &&
                   num_causal == other.num_causal &&
                   final_vertex_id == other.final_vertex_id &&
                   final_edge_id == other.final_edge_id &&
                   final_event_id == other.final_event_id;
        }

        bool operator<(const TestResult& other) const {
            if (num_states != other.num_states) return num_states < other.num_states;
            if (num_events != other.num_events) return num_events < other.num_events;
            if (num_branchial != other.num_branchial) return num_branchial < other.num_branchial;
            if (num_causal != other.num_causal) return num_causal < other.num_causal;
            if (final_vertex_id != other.final_vertex_id) return final_vertex_id < other.final_vertex_id;
            if (final_edge_id != other.final_edge_id) return final_edge_id < other.final_edge_id;
            return final_event_id < other.final_event_id;
        }
    };

    struct DeterminismResults {
        std::set<std::size_t> unique_states;
        std::set<std::size_t> unique_events;
        std::set<std::size_t> unique_causal;
        std::set<std::size_t> unique_branchial;
        std::set<GlobalVertexId> unique_final_vertex_ids;
        std::set<GlobalEdgeId> unique_final_edge_ids;
        std::set<EventId> unique_final_event_ids;
        std::vector<TestResult> all_results;
    };

    DeterminismResults fuzz_test_rules(const std::string& test_name,
                                      const std::vector<RewritingRule>& rules,
                                      const std::vector<std::vector<GlobalVertexId>>& initial,
                                      std::size_t steps,
                                      int num_runs = 50) {
        DeterminismResults results;

        // Run the evolution many times to test for non-determinism
        for (int i = 0; i < num_runs; ++i) {
            // Create clean evolution system
            WolframEvolution evolution(steps, std::thread::hardware_concurrency(), true, false);
            for (const auto& rule : rules) {
                evolution.add_rule(rule);
            }
            evolution.evolve(initial);

            const auto& graph = evolution.get_multiway_graph();
            TestResult result = {
                graph.num_states(),
                graph.num_events(),
                graph.get_branchial_edge_count(),
                graph.get_causal_edge_count(),
                graph.get_final_vertex_id(),
                graph.get_final_edge_id(),
                graph.get_final_event_id()
            };

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
        std::cout << test_name << " fuzz test results after " << num_runs << " runs:" << std::endl;
        std::cout << "  States deterministic: " << (results.unique_states.size() == 1 ? "✓" : "✗")
                  << " (" << results.unique_states.size() << " unique values)" << std::endl;
        std::cout << "  Events deterministic: " << (results.unique_events.size() == 1 ? "✓" : "✗")
                  << " (" << results.unique_events.size() << " unique values)" << std::endl;
        std::cout << "  Causal deterministic: " << (results.unique_causal.size() == 1 ? "✓" : "✗")
                  << " (" << results.unique_causal.size() << " unique values)" << std::endl;
        std::cout << "  Branchial deterministic: " << (results.unique_branchial.size() == 1 ? "✓" : "✗")
                  << " (" << results.unique_branchial.size() << " unique values)" << std::endl;
        std::cout << "  Final Vertex ID deterministic: " << (results.unique_final_vertex_ids.size() == 1 ? "✓" : "✗")
                  << " (" << results.unique_final_vertex_ids.size() << " unique values)" << std::endl;
        std::cout << "  Final Edge ID deterministic: " << (results.unique_final_edge_ids.size() == 1 ? "✓" : "✗")
                  << " (" << results.unique_final_edge_ids.size() << " unique values)" << std::endl;
        std::cout << "  Final Event ID deterministic: " << (results.unique_final_event_ids.size() == 1 ? "✓" : "✗")
                  << " (" << results.unique_final_event_ids.size() << " unique values)" << std::endl;

        if (results.unique_states.size() == 1 && results.unique_events.size() == 1 &&
            results.unique_causal.size() == 1 && results.unique_branchial.size() == 1 &&
            results.unique_final_vertex_ids.size() == 1 && results.unique_final_edge_ids.size() == 1 &&
            results.unique_final_event_ids.size() == 1) {
            const auto& result = results.all_results[0];
            std::cout << "  All metrics deterministic: States=" << result.num_states
                      << ", Events=" << result.num_events << ", Causal=" << result.num_causal
                      << ", Branchial=" << result.num_branchial << ", FinalVertexID=" << result.final_vertex_id
                      << ", FinalEdgeID=" << result.final_edge_id << ", FinalEventID=" << result.final_event_id << std::endl;
        }

        // Add EXPECT_EQ assertions for determinism
        EXPECT_EQ(results.unique_states.size(), 1) << "States non-deterministic";
        EXPECT_EQ(results.unique_events.size(), 1) << "Events non-deterministic";
        EXPECT_EQ(results.unique_causal.size(), 1) << "Causal edges non-deterministic";
        EXPECT_EQ(results.unique_branchial.size(), 1) << "Branchial edges non-deterministic";
        EXPECT_EQ(results.unique_final_vertex_ids.size(), 1) << "Final vertex IDs non-deterministic";
        EXPECT_EQ(results.unique_final_edge_ids.size(), 1) << "Final edge IDs non-deterministic";
        EXPECT_EQ(results.unique_final_event_ids.size(), 1) << "Final event IDs non-deterministic";

        return results;
    }

    void validate_expected_results(const std::string& test_name,
                                  const std::vector<RewritingRule>& rules,
                                  const std::vector<std::vector<GlobalVertexId>>& initial,
                                  const std::vector<std::pair<int, TestResult>>& expected) {
        std::cout << "\n=== Validating Expected Results for " << test_name << " ===" << std::endl;
        for (const auto& [steps, expected_result] : expected) {
            WolframEvolution evolution(steps, std::thread::hardware_concurrency(), true, false);
            for (const auto& rule : rules) {
                evolution.add_rule(rule);
            }
            evolution.evolve(initial);
            const auto& graph = evolution.get_multiway_graph();

            TestResult actual = {
                graph.num_states(),
                graph.num_events(),
                graph.get_branchial_edge_count(),
                graph.get_causal_edge_count()
            };

            std::cout << "Step " << steps << " - Expected: {" << expected_result.num_states
                      << ", " << expected_result.num_events << ", " << expected_result.num_branchial
                      << ", " << expected_result.num_causal << "}" << std::endl;
            std::cout << "Step " << steps << " - Actual:   {" << actual.num_states
                      << ", " << actual.num_events << ", " << actual.num_branchial
                      << ", " << actual.num_causal << "}" << std::endl;

            EXPECT_EQ(actual.num_states, expected_result.num_states) << "States mismatch at step " << steps;
            EXPECT_EQ(actual.num_events, expected_result.num_events) << "Events mismatch at step " << steps;
            EXPECT_EQ(actual.num_branchial, expected_result.num_branchial) << "Branchial edges mismatch at step " << steps;
            EXPECT_EQ(actual.num_causal, expected_result.num_causal) << "Causal edges mismatch at step " << steps;
        }
    }

};

TEST_F(DeterminismFuzzingTest, TestCase1_SimpleRule) {
    // Create rule: {{1,2}} -> {{1,2}, {2,3}}

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 1, 0, 0}},
        //{2, {4, 3, 0, 2}},
        //{3, {8, 9, 0, 8}},
        {4, {17, 33, 0, 32}}
    };

    validate_expected_results("SimpleRule", {rule}, initial, expected);

    //fuzz_test_rules("SimpleRule", {rule}, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase2_TwoEdgeRule) {
    // Create rule: {{1,2},{2,3}} -> {{1,2},{2,3},{2,4}}

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(4)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}, {3, 1}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 3, 3, 0}},
        //{2, {4, 15, 18, 12}},
        //{3, {8, 75, 108, 72}},
        {4, {13, 435, 738, 444}}
    };

    validate_expected_results("TwoEdgeRule", {rule}, initial, expected);

    //fuzz_test_rules("TwoEdgeRule", {rule}, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase3_HyperedgeRule) {
    // Create rule: {{1,2,3}} -> {{1,2},{1,3},{1,4}}

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(4)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2, 3}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        // {1, {2, 1, 0, 0}},
        // {2, {2, 1, 0, 0}},
        // {3, {2, 1, 0, 0}},
        {4, {2, 1, 0, 0}}
    };

    validate_expected_results("HyperedgeRule", {rule}, initial, expected);

    //fuzz_test_rules("HyperedgeRule", {rule}, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase4_MultiRule) {
    // Create Rule 1: {{1,2,3}} -> {{1,2},{1,3},{1,4}}
    PatternHypergraph lhs1;

    lhs1.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(2),
        PatternVertex::variable(3)
    });

    PatternHypergraph rhs1;

    rhs1.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(2)
    });

    rhs1.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(3)
    });

    rhs1.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(4)
    });

    RewritingRule rule1(lhs1, rhs1);

    // Create Rule 2: {{1,2}} -> {{1,2},{1,3}}
    PatternHypergraph lhs2;

    lhs2.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(2)
    });

    PatternHypergraph rhs2;

    rhs2.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(2)
    });

    rhs2.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(3)
    });

    RewritingRule rule2(lhs2, rhs2);

    std::vector<RewritingRule> rules = {rule1, rule2};

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2, 3}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 1, 0, 0}},
        //{2, {3, 4, 0, 3}},
        //{3, {4, 16, 0, 15}},
        {4, {5, 76, 0, 75}}
    };

    validate_expected_results("MultiRule", rules, initial, expected);

    //fuzz_test_rules("MultiRule", rules, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase5_ComplexTwoRuleSystem) {
    // Create rules: {{1,2,3}} -> {{1,2},{1,3},{1,4}} and {{1,2},{1,3}} -> {{1,2},{1,3},{2,4}}

    // Rule 1: {{1,2,3}} -> {{1,2},{1,3},{1,4}}
    PatternHypergraph lhs1;

    lhs1.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs1;

    rhs1.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    rhs1.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(3)
    });

    rhs1.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(4)
    });

    // Rule 2: {{1,2},{1,3}} -> {{1,2},{1,3},{2,4}}
    PatternHypergraph lhs2;

    lhs2.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs2.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(3)
    });

    PatternHypergraph rhs2;

    rhs2.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    rhs2.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(3)
    });

    rhs2.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(4)
    });

    RewritingRule rule1(lhs1, rhs1);
    RewritingRule rule2(lhs2, rhs2);

    std::vector<RewritingRule> rules = {rule1, rule2};

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2, 3}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 1, 0, 0}},
        //{2, {3, 7, 15, 6}},
        //{3, {5, 43, 105, 42}},
        {4, {9, 283, 657, 282}}
    };

    validate_expected_results("ComplexTwoRuleSystem", rules, initial, expected);

    //fuzz_test_rules("ComplexTwoRuleSystem", rules, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase6_TwoEdgeRuleVariant) {
    // Create rule: {{1,2},{2,3}} -> {{1,2},{2,3},{2,4}}

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(4)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 1, 0, 0}},
        //{2, {3, 3, 1, 2}},
        //{3, {4, 9, 7, 8}},
        {4, {5, 33, 43, 32}}
    };

    validate_expected_results("TwoEdgeRuleVariant", {rule}, initial, expected);

    // fuzz_test_rules("TwoEdgeRuleVariant", {rule}, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase7_TwoEdgeRuleWithSelfLoops) {
    // Create rule: {{1,2},{2,3}} -> {{1,2},{2,3},{2,4}}
    // Initial state: {{1,1},{1,1}} (self-loops)

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(4)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 1}, {1, 1}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        // {1, {2, 2, 1, 0}},
        // {2, {3, 10, 13, 8}},
        // {3, {4, 58, 117, 56}},
        {4, {5, 442, 1173, 440}}
    };

    validate_expected_results("TwoEdgeRuleWithSelfLoops", {rule}, initial, expected);

    // fuzz_test_rules("TwoEdgeRuleWithSelfLoops", {rule}, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase8_ComplexTwoEdgeRule) {
    // Create rule: {{1,2},{2,3}} -> {{4,1},{1,4},{2,3},{4,3}}

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(4), PatternVertex::variable(1)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(4)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(4), PatternVertex::variable(3)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 1, 0, 0}},
        //{2, {5, 4, 3, 3}},
        //{3, {17, 21, 37, 20}},
        {4, {83, 182, 423, 181}}
    };

    validate_expected_results("ComplexTwoEdgeRule", {rule}, initial, expected);

    // fuzz_test_rules("ComplexTwoEdgeRule", {rule}, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase9_ComplexTwoEdgeRuleWithSelfLoops) {
    // Create rule: {{1,2},{2,3}} -> {{4,1},{1,4},{2,3},{4,3}}
    // Initial state: {{1,1},{1,1}} (self-loops)

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(4), PatternVertex::variable(1)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(4)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(4), PatternVertex::variable(3)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 1}, {1, 1}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 2, 1, 0}},
        //{2, {6, 16, 35, 14}},
        {3, {33, 172, 489, 170}}
    };

    validate_expected_results("ComplexTwoEdgeRuleWithSelfLoops", {rule}, initial, expected);

    // fuzz_test_rules("ComplexTwoEdgeRuleWithSelfLoops", {rule}, initial, 3);
}

TEST_F(DeterminismFuzzingTest, TestCase10_AnotherTwoEdgeRule) {
    // Create rule: {{1,2},{2,3}} -> {{1,3},{2,3},{3,4}}

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(3), PatternVertex::variable(4)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}, {2, 3}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 1, 0, 0}},
        //{2, {3, 3, 1, 2}},
        //{3, {6, 9, 5, 8}},
        //{4, {10, 37, 35, 36}},
        {5, {21, 203, 231, 210}}
    };

    validate_expected_results("AnotherTwoEdgeRule", {rule}, initial, expected);

    //fuzz_test_rules("AnotherTwoEdgeRule", {rule}, initial, 4);
}

TEST_F(DeterminismFuzzingTest, TestCase11_AnotherTwoEdgeRuleWithSelfLoops) {
    // Create rule: {{1,2},{2,3}} -> {{1,3},{2,3},{3,4}}
    // Initial state: {{1,1},{1,1}} (self-loops)

    PatternHypergraph lhs;

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });

    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    PatternHypergraph rhs;

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    rhs.add_edge(PatternEdge{
        PatternVertex::variable(3), PatternVertex::variable(4)
    });

    RewritingRule rule(lhs, rhs);

    std::vector<std::vector<GlobalVertexId>> initial = {{1, 1}, {1, 1}};

    // Expected results for different step counts
    // Format: {steps, {states, events, branchial_edges, causal_edges}}
    std::vector<std::pair<int, TestResult>> expected = {
        //{1, {2, 2, 1, 0}},
        //{2, {4, 10, 13, 8}},
        //{3, {8, 50, 81, 48}},
        {4, {18, 282, 465, 280}}
    };

    validate_expected_results("AnotherTwoEdgeRuleWithSelfLoops", {rule}, initial, expected);

    // fuzz_test_rules("AnotherTwoEdgeRuleWithSelfLoops", {rule}, initial, 4);
}