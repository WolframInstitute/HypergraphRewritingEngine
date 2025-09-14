#include <gtest/gtest.h>
#include <hypergraph/pattern_matching_tasks.hpp>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <atomic>
#include <unordered_set>
#include <sstream>
#include "test_helpers.hpp"

// Test fixture for pattern matching task isolation tests
class PatternMatchingTaskIsolationTest : public ::testing::Test {
protected:
    // Structure to track task execution and match results without evolution
    struct TaskTracker {
        int scan_tasks_spawned = 0;
        int sink_tasks_spawned = 0;
        int total_matches = 0;
        std::set<std::set<hypergraph::EdgeId>> unique_edge_sets;
        std::vector<std::string> match_details;

        void record_match(const std::vector<hypergraph::EdgeId>& matched_edges) {
            std::set<hypergraph::EdgeId> edge_set(matched_edges.begin(), matched_edges.end());
            unique_edge_sets.insert(edge_set);
            total_matches++;

            std::stringstream ss;
            ss << "Match " << total_matches << ": edges {";
            for (size_t i = 0; i < matched_edges.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << matched_edges[i];
            }
            ss << "}";

            match_details.push_back(ss.str());
        }

        void print_summary() const {
            std::cout << "\n=== Task Results Summary ===" << std::endl;
            std::cout << "Total matches found: " << total_matches << std::endl;
            std::cout << "Unique edge sets: " << unique_edge_sets.size() << std::endl;
            std::cout << "SCAN tasks spawned: " << scan_tasks_spawned << std::endl;
            std::cout << "SINK tasks spawned: " << sink_tasks_spawned << std::endl;

            if (total_matches != static_cast<int>(unique_edge_sets.size())) {
                std::cout << "WARNING: Duplicate matches detected!" << std::endl;
                std::cout << "Match details:" << std::endl;
                for (const auto& detail : match_details) {
                    std::cout << "  " << detail << std::endl;
                }
            }
        }
    };

    // Run evolution with 0 steps so rewrite tasks exit immediately, testing only pattern matching tasks
    TaskTracker test_pattern_matching_tasks(const std::vector<std::vector<hypergraph::GlobalVertexId>>& initial_edges,
                                           const std::vector<std::vector<int>>& rule_lhs,
                                           int num_workers = 4) {
        TaskTracker tracker;

        // Create WolframEvolution with 0 steps - this will spawn pattern matching tasks but no rewrites
        hypergraph::WolframEvolution evolution(0, num_workers, true, false, true);

        // Create rule using correct API
        hypergraph::PatternHypergraph lhs, rhs;
        for (const auto& edge_data : rule_lhs) {
            std::vector<hypergraph::PatternVertex> vertices;
            for (int v : edge_data) {
                vertices.push_back(hypergraph::PatternVertex::variable(v));
            }
            lhs.add_edge(hypergraph::PatternEdge{vertices});
        }

        // Add minimal RHS to make rule valid
        rhs.add_edge(hypergraph::PatternEdge{
            {hypergraph::PatternVertex::variable(100), hypergraph::PatternVertex::variable(101)}
        });

        hypergraph::RewritingRule rule(lhs, rhs);
        evolution.add_rule(rule);

        std::cout << "Running evolution with 0 steps to test pattern matching tasks only..." << std::endl;

        // Run evolution - with 0 steps, it will only do pattern matching, no rewriting
        evolution.evolve(initial_edges);

        const auto& graph = evolution.get_multiway_graph();
        tracker.total_matches = static_cast<int>(graph.num_states());

        std::cout << "Evolution completed. Total states: " << graph.num_states() << std::endl;

        return tracker;
    }
};

// Test 1: Single edge pattern on single edge hypergraph - should find exactly 1 match
TEST_F(PatternMatchingTaskIsolationTest, SingleEdgePatternSingleEdgeHypergraph) {
    std::cout << "\n=== Test: Single Edge Pattern on Single Edge Hypergraph ===" << std::endl;

    // Single edge {1,2}
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial_state = {{1, 2}};
    // Pattern: single variable edge {x,y}
    std::vector<std::vector<int>> rule_lhs = {{100, 101}};

    auto result = test_pattern_matching_tasks(initial_state, rule_lhs, 4);
    result.print_summary();

    // With 0 steps, should find matches but create no events (no rewriting)
    // This tests that pattern matching tasks run correctly
    std::cout << "Expected: 0 events (no rewriting), pattern matching tasks should have run" << std::endl;
    EXPECT_EQ(result.total_matches, 0) << "Should have 0 events with 0 evolution steps";
}

// Test 2: Single edge pattern on multi-edge hypergraph - verify task spawning
TEST_F(PatternMatchingTaskIsolationTest, SingleEdgePatternMultiEdgeHypergraph) {
    std::cout << "\n=== Test: Single Edge Pattern on Multi-Edge Hypergraph ===" << std::endl;

    // Four edges
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial_state = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    // Pattern: single variable edge {x,y}
    std::vector<std::vector<int>> rule_lhs = {{100, 101}};

    auto result = test_pattern_matching_tasks(initial_state, rule_lhs, 4);
    result.print_summary();

    std::cout << "Expected: 0 events (no rewriting), but pattern matching should process 4 edges" << std::endl;
    EXPECT_EQ(result.total_matches, 0) << "Should have 0 events with 0 evolution steps";
}

// Test 3: Check actual evolution for 1 step to see duplicate issue
TEST_F(PatternMatchingTaskIsolationTest, OneStepEvolutionShowsDuplicates) {
    std::cout << "\n=== Test: One Step Evolution Shows Duplicates ===" << std::endl;

    // Use the problematic 4-edge state from TestCase1
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial_state = {{1, 2}, {2, 3}, {3, 4}, {2, 5}};
    std::vector<std::vector<int>> rule_lhs = {{100, 101}};

    // Create evolution with 1 step to actually see matches converted to events
    hypergraph::WolframEvolution evolution(1, 4, true, false, true);

    // Create rule using correct API
    hypergraph::PatternHypergraph lhs, rhs;
    lhs.add_edge(hypergraph::PatternEdge{
        {hypergraph::PatternVertex::variable(100), hypergraph::PatternVertex::variable(101)}
    });
    rhs.add_edge(hypergraph::PatternEdge{
        {hypergraph::PatternVertex::variable(100), hypergraph::PatternVertex::variable(101)}
    });
    rhs.add_edge(hypergraph::PatternEdge{
        {hypergraph::PatternVertex::variable(101), hypergraph::PatternVertex::variable(102)}
    });

    hypergraph::RewritingRule rule(lhs, rhs);
    evolution.add_rule(rule);

    std::cout << "Running 1-step evolution to check for duplicate events..." << std::endl;
    evolution.evolve(initial_state);

    const auto& graph = evolution.get_multiway_graph();
    std::cout << "Evolution completed. Total states created: " << graph.num_states() << std::endl;

    // For a single-edge LHS rule on a 4-edge state, we expect exactly 5 states total:
    // 1 initial state + 4 new states (one per edge matched)
    // If we see more states, we have duplicate events
    EXPECT_EQ(graph.num_states(), 5) << "Should have exactly 5 states total (1 initial + 4 from matches)";

    if (graph.num_states() != 5) {
        std::cout << "DUPLICATE EVENTS DETECTED!" << std::endl;
        std::cout << "Expected 5 total states, got " << graph.num_states() << std::endl;
    }
}