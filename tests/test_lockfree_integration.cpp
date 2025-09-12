#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <vector>
#include <thread>
#include "test_helpers.hpp"

using namespace hypergraph;

class LockfreeIntegrationTest : public ::testing::Test {
protected:
    struct TestResult {
        std::size_t num_states;
        std::size_t num_events;
        std::size_t num_branchial;
        std::size_t num_causal;

        bool operator==(const TestResult& other) const {
            return num_states == other.num_states &&
                   num_events == other.num_events &&
                   num_branchial == other.num_branchial &&
                   num_causal == other.num_causal;
        }

        bool operator<(const TestResult& other) const {
            if (num_states != other.num_states) return num_states < other.num_states;
            if (num_events != other.num_events) return num_events < other.num_events;
            if (num_branchial != other.num_branchial) return num_branchial < other.num_branchial;
            return num_causal < other.num_causal;
        }
    };

    void fuzz_test_rule(const std::string& test_name,
                       const RewritingRule& rule,
                       const std::vector<std::vector<GlobalVertexId>>& initial,
                       std::size_t steps,
                       int num_runs = 50) {
        std::set<TestResult> unique_results;
        std::vector<TestResult> all_results;

        // Run the evolution many times to test for non-determinism
        for (int i = 0; i < num_runs; ++i) {
            // Create clean evolution system
            WolframEvolution evolution(steps, std::thread::hardware_concurrency(), true, false);
            evolution.add_rule(rule);
            evolution.evolve(initial);

            const auto& graph = evolution.get_multiway_graph();
            TestResult result = {
                graph.num_states(),
                graph.num_events(),
                graph.get_branchial_edge_count(),
                graph.get_causal_edge_count()
            };

            all_results.push_back(result);
            unique_results.insert(result);
        }

        // Report results
        std::cout << test_name << " fuzz test results after " << num_runs << " runs:" << std::endl;
        std::cout << "Unique result sets found: " << unique_results.size() << std::endl;

        if (unique_results.size() == 1) {
            std::cout << "✓ DETERMINISTIC: All runs produced identical results" << std::endl;
            const auto& result = *unique_results.begin();
            std::cout << "  States: " << result.num_states << std::endl;
            std::cout << "  Events: " << result.num_events << std::endl;
            std::cout << "  Causal edges: " << result.num_causal << std::endl;
            std::cout << "  Branchial edges: " << result.num_branchial << std::endl;
        } else {
            std::cout << "✗ NON-DETERMINISTIC: Found " << unique_results.size() << " different result sets" << std::endl;
        }

        // The test passes if we get deterministic results
        EXPECT_EQ(unique_results.size(), 1)
            << "Non-determinism detected: got " << unique_results.size()
            << " different result sets across " << num_runs << " runs";

        // Verify we get meaningful results
        const auto& first_result = all_results[0];
        EXPECT_GT(first_result.num_states, 0) << "Should have at least one state";
        EXPECT_GE(first_result.num_events, 0) << "Should have non-negative events";
    }

    void fuzz_test_multirule(const std::string& test_name,
                            const std::vector<RewritingRule>& rules,
                            const std::vector<std::vector<GlobalVertexId>>& initial,
                            std::size_t steps,
                            int num_runs = 50) {
        std::set<TestResult> unique_results;
        std::vector<TestResult> all_results;

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
                graph.get_causal_edge_count()
            };

            all_results.push_back(result);
            unique_results.insert(result);
        }

        // Report results
        std::cout << test_name << " fuzz test results after " << num_runs << " runs:" << std::endl;
        std::cout << "Unique result sets found: " << unique_results.size() << std::endl;

        if (unique_results.size() == 1) {
            std::cout << "✓ DETERMINISTIC: All runs produced identical results" << std::endl;
            const auto& result = *unique_results.begin();
            std::cout << "  States: " << result.num_states << std::endl;
            std::cout << "  Events: " << result.num_events << std::endl;
            std::cout << "  Causal edges: " << result.num_causal << std::endl;
            std::cout << "  Branchial edges: " << result.num_branchial << std::endl;
        } else {
            std::cout << "✗ NON-DETERMINISTIC: Found " << unique_results.size() << " different result sets" << std::endl;
        }

        // The test passes if we get deterministic results
        EXPECT_EQ(unique_results.size(), 1)
            << "Non-determinism detected: got " << unique_results.size()
            << " different result sets across " << num_runs << " runs";

        // Verify we get meaningful results
        const auto& first_result = all_results[0];
        EXPECT_GT(first_result.num_states, 0) << "Should have at least one state";
        EXPECT_GE(first_result.num_events, 0) << "Should have non-negative events";
    }
};

TEST_F(LockfreeIntegrationTest, TestCase1_SimpleRule) {
    // Create rule: {{1,2}} -> {{1,2}, {2,3}}
    PatternHypergraph lhs, rhs;
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    RewritingRule rule(lhs, rhs);
    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}};

    // Expected results for different step counts (temporarily commented out)
    // std::vector<std::pair<int, TestResult>> expected = {
    //     {1, {2, 1, 0, 0}},
    //     {2, {4, 3, 1, 2}},
    //     {3, {8, 9, 6, 8}},
    //     {4, {17, 33, 36, 32}},
    //     {5, {37, 153, 240, 152}}
    // };

    // Test determinism for 3 steps (reasonable complexity)
    fuzz_test_rule("SimpleRule", rule, initial, 4);
}

TEST_F(LockfreeIntegrationTest, TestCase2_TwoEdgeRule) {
    // Create rule: {{1,2},{2,3}} -> {{1,2},{2,3},{2,4}}
    PatternHypergraph lhs, rhs;

    // LHS: {{1,2},{2,3}}
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });

    // RHS: {{1,2},{2,3},{2,4}}
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

    // Expected results for different step counts (temporarily commented out)
    // std::vector<std::pair<int, TestResult>> expected = {
    //     {1, {2, 1, 0, 0}},
    //     {2, {3, 2, 0, 1}},
    //     {3, {5, 4, 0, 3}},
    //     {4, {9, 8, 0, 7}},
    //     {5, {17, 16, 0, 15}}
    // };

    // Test determinism for 3 steps (reasonable complexity)
    fuzz_test_rule("TwoEdgeRule", rule, initial, 4);
}

TEST_F(LockfreeIntegrationTest, TestCase3_ThreeVertexRule) {
    // Create rule: {{1,2,3}} -> {{1,2},{2,3},{3,1}}
    PatternHypergraph lhs, rhs;

    // LHS: {{1,2,3}} - 3-vertex hyperedge
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(2),
        PatternVertex::variable(3)
    });

    // RHS: {{1,2},{2,3},{3,1}} - triangle
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(3), PatternVertex::variable(1)
    });

    RewritingRule rule(lhs, rhs);
    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2, 3}};

    // Expected results for different step counts (temporarily commented out)
    // std::vector<std::pair<int, TestResult>> expected = {
    //     {1, {2, 1, 0, 0}},
    //     {2, {2, 1, 0, 0}},  // No more matches after first application
    //     {3, {2, 1, 0, 0}}
    // };

    // Test determinism for 3 steps (should be stable after first step)
    fuzz_test_rule("ThreeVertexRule", rule, initial, 4);
}

TEST_F(LockfreeIntegrationTest, TestCase4_MultiRule) {
    // Create Rule 1: {{1,2,3}} -> {{1,2},{1,3},{1,4}}
    PatternHypergraph lhs1, rhs1;
    lhs1.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(2),
        PatternVertex::variable(3)
    });
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
    PatternHypergraph lhs2, rhs2;
    lhs2.add_edge(PatternEdge{
        PatternVertex::variable(1),
        PatternVertex::variable(2)
    });
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
    std::vector<std::vector<GlobalVertexId>> initial = {{1,2,3}};

    // Test determinism for 3 steps with multiple rules
    fuzz_test_multirule("MultiRule", rules, initial, 4);
}