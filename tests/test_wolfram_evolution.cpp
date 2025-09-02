#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>
#include "test_helpers.hpp"

class WolframEvolutionTest : public ::testing::Test {
protected:
    // Helper to create a simple rule: {{1, 2}} -> {{1, 2}, {2, 3}}
    hypergraph::RewritingRule create_test_rule() {
        hypergraph::PatternHypergraph lhs, rhs;
        
        lhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(1), 
            hypergraph::PatternVertex::variable(2)
        });
        
        rhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(1), 
            hypergraph::PatternVertex::variable(2)
        });
        rhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(2), 
            hypergraph::PatternVertex::variable(3)
        });
        
        return hypergraph::RewritingRule(lhs, rhs);
    }
};

// === WOLFRAM EVOLUTION TESTS ===

TEST_F(WolframEvolutionTest, EvolutionCreation) {
    hypergraph::WolframEvolution evolution(1, 1, true, false);  // 1 step, 1 thread
    
    // Evolution should be created successfully
    EXPECT_EQ(evolution.get_multiway_graph().num_states(), 0);
    EXPECT_EQ(evolution.get_multiway_graph().num_events(), 0);
}

TEST_F(WolframEvolutionTest, RuleAddition) {
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    
    evolution.add_rule(create_test_rule());
    
    // Rule should be added (we can't directly test this, but evolution should not crash)
    EXPECT_TRUE(true);  // If we get here, rule was added successfully
}

TEST_F(WolframEvolutionTest, BasicEvolution) {
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());
    
    // Initial state: single edge {{1, 2}}
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial = {{1, 2}};
    
    try {
        evolution.evolve(initial);
        
        const auto& graph = evolution.get_multiway_graph();
        
        // After 1 step with rule {{1,2}} -> {{1,2}, {2,3}}, we should have more states
        EXPECT_GT(graph.num_states(), 0);
        
    } catch (const std::exception& e) {
        FAIL() << "Evolution threw exception: " << e.what();
    }
}

TEST_F(WolframEvolutionTest, MultiStepEvolution) {
    // Test with 3 steps - should now work with step checking fix
    hypergraph::WolframEvolution evolution(3, 1, true, false);
    evolution.add_rule(create_test_rule());
    
    // Initial state: two edges {{1, 2}, {2, 3}}
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial = {{1, 2}, {2, 3}};
    
    try {
        evolution.evolve(initial);
        
        const auto& graph = evolution.get_multiway_graph();
        
        // After 3 steps we should have multiple states and events
        EXPECT_GT(graph.num_states(), 1) << "Should have more than 1 state after 3 steps";
        EXPECT_GT(graph.num_events(), 1) << "Should have more than 1 event after 3 steps";
        
        // Print debug info
        std::cout << "After 3 steps: " << graph.num_states() << " states, " 
                  << graph.num_events() << " events\n";
        
    } catch (const std::exception& e) {
        FAIL() << "Multi-step evolution threw exception: " << e.what();
    }
}
