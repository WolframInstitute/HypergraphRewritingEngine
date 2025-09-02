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
    };

    TestResult run_evolution(const RewritingRule& rule, 
                           const std::vector<std::vector<GlobalVertexId>>& initial,
                           std::size_t steps) {
        // Create clean evolution system
        WolframEvolution evolution(steps, std::thread::hardware_concurrency(), 
                                  true, false);
        evolution.add_rule(rule);
        
        // Run evolution
        evolution.evolve(initial);
        
        // Get results
        const auto& graph = evolution.get_multiway_graph();
        
        return {
            graph.num_states(),
            graph.num_events(),
            graph.get_branchial_edge_count(),
            graph.get_causal_edge_count()
        };
    }
    
    void verify_determinism(const RewritingRule& rule,
                           const std::vector<std::vector<GlobalVertexId>>& initial,
                           std::size_t steps) {
        TestResult first_run = run_evolution(rule, initial, steps);
        
        // Verify determinism with 3 more runs
        for (int i = 1; i <= 3; ++i) {
            TestResult repeat_run = run_evolution(rule, initial, steps);
            EXPECT_EQ(repeat_run, first_run) << "Determinism failed on run " << (i + 1);
        }
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
    
    // Expected results for different step counts
    std::vector<std::pair<int, TestResult>> expected = {
        {1, {2, 1, 0, 0}},
        {2, {4, 3, 1, 2}},
        {3, {8, 9, 6, 8}},
        {4, {17, 33, 36, 32}},
        {5, {37, 153, 240, 152}}
    };
    
    for (const auto& [steps, expected_result] : expected) {
        TestResult actual = run_evolution(rule, initial, steps);
        
        EXPECT_EQ(actual.num_states, expected_result.num_states) 
            << "States mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_events, expected_result.num_events)
            << "Events mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_branchial, expected_result.num_branchial)
            << "Branchial edges mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_causal, expected_result.num_causal)
            << "Causal edges mismatch at " << steps << " steps";
        
        // Verify determinism for this step count
        verify_determinism(rule, initial, steps);
    }
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
    
    // Expected results for different step counts
    std::vector<std::pair<int, TestResult>> expected = {
        {1, {2, 1, 0, 0}},
        {2, {3, 2, 0, 1}},
        {3, {5, 4, 0, 3}},
        {4, {9, 8, 0, 7}},
        {5, {17, 16, 0, 15}}
    };
    
    for (const auto& [steps, expected_result] : expected) {
        TestResult actual = run_evolution(rule, initial, steps);
        
        EXPECT_EQ(actual.num_states, expected_result.num_states)
            << "States mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_events, expected_result.num_events)
            << "Events mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_branchial, expected_result.num_branchial)
            << "Branchial edges mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_causal, expected_result.num_causal)
            << "Causal edges mismatch at " << steps << " steps";
        
        // Verify determinism for this step count
        verify_determinism(rule, initial, steps);
    }
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
    
    // Expected results for different step counts
    std::vector<std::pair<int, TestResult>> expected = {
        {1, {2, 1, 0, 0}},
        {2, {2, 1, 0, 0}},  // No more matches after first application
        {3, {2, 1, 0, 0}}
    };
    
    for (const auto& [steps, expected_result] : expected) {
        TestResult actual = run_evolution(rule, initial, steps);
        
        EXPECT_EQ(actual.num_states, expected_result.num_states)
            << "States mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_events, expected_result.num_events)
            << "Events mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_branchial, expected_result.num_branchial)
            << "Branchial edges mismatch at " << steps << " steps";
        EXPECT_EQ(actual.num_causal, expected_result.num_causal)
            << "Causal edges mismatch at " << steps << " steps";
        
        // Verify determinism for this step count
        verify_determinism(rule, initial, steps);
    }
}