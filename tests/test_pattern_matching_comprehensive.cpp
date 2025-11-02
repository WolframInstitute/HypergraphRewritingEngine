#include <gtest/gtest.h>
#include <hypergraph/pattern_matching.hpp>
#include <hypergraph/pattern_matching_tasks.hpp>
#include <hypergraph/hypergraph.hpp>
#include "test_helpers.hpp"

class PatternMatchingComprehensiveTest : public ::testing::Test {
protected:
    hypergraph::PatternMatcher matcher;
    
    // Helper to create pattern from edge descriptions
    hypergraph::PatternHypergraph create_pattern(const std::vector<std::vector<int>>& pattern_edges) {
        hypergraph::PatternHypergraph pattern;
        for (const auto& edge_data : pattern_edges) {
            std::vector<hypergraph::PatternVertex> vertices;
            for (int v : edge_data) {
                if (v < 100) {
                    vertices.push_back(hypergraph::PatternVertex::concrete(v));
                } else {
                    vertices.push_back(hypergraph::PatternVertex::variable(v));
                }
            }
            pattern.add_edge(hypergraph::PatternEdge{vertices});
        }
        return pattern;
    }
};

// === PATTERN VERTEX TESTS ===

TEST_F(PatternMatchingComprehensiveTest, PatternVertexTypes) {
    auto concrete = hypergraph::PatternVertex::concrete(5);
    auto variable = hypergraph::PatternVertex::variable(100);
    
    EXPECT_TRUE(concrete.is_concrete());
    EXPECT_FALSE(concrete.is_variable());
    EXPECT_EQ(concrete.id, 5);
    
    EXPECT_FALSE(variable.is_concrete());
    EXPECT_TRUE(variable.is_variable());
    EXPECT_EQ(variable.id, 100);
    
    EXPECT_NE(concrete, variable);
    EXPECT_EQ(concrete, hypergraph::PatternVertex::concrete(5));
}

// === VARIABLE ASSIGNMENT TESTS ===

TEST_F(PatternMatchingComprehensiveTest, VariableAssignmentBasics) {
    hypergraph::VariableAssignment assignment;
    
    // Test assignment
    EXPECT_TRUE(assignment.assign(100, 1));
    EXPECT_TRUE(assignment.assign(101, 2));
    
    // Test consistency - can't reassign to different value
    EXPECT_FALSE(assignment.assign(100, 3));
    EXPECT_TRUE(assignment.assign(100, 1));  // Same value is ok
    
    // Test resolution
    auto concrete_vertex = hypergraph::PatternVertex::concrete(5);
    auto variable_vertex = hypergraph::PatternVertex::variable(100);
    
    auto resolved_concrete = assignment.resolve(concrete_vertex);
    auto resolved_variable = assignment.resolve(variable_vertex);
    
    EXPECT_TRUE(resolved_concrete.has_value());
    EXPECT_EQ(*resolved_concrete, 5);
    
    EXPECT_TRUE(resolved_variable.has_value());
    EXPECT_EQ(*resolved_variable, 1);
    
    // Unassigned variable
    auto unassigned = hypergraph::PatternVertex::variable(999);
    auto resolved_unassigned = assignment.resolve(unassigned);
    EXPECT_FALSE(resolved_unassigned.has_value());
}

TEST_F(PatternMatchingComprehensiveTest, VariableAssignmentCompleteness) {
    hypergraph::VariableAssignment assignment;
    assignment.assign(100, 1);
    assignment.assign(101, 2);
    
    std::unordered_set<hypergraph::VertexId> required_vars = {100, 101, 102};
    EXPECT_FALSE(assignment.is_complete(required_vars));
    
    assignment.assign(102, 3);
    EXPECT_TRUE(assignment.is_complete(required_vars));
}

// === PATTERN EDGE TESTS ===

TEST_F(PatternMatchingComprehensiveTest, PatternEdgeConstruction) {
    hypergraph::PatternEdge edge{{
        hypergraph::PatternVertex::concrete(1),
        hypergraph::PatternVertex::variable(100),
        hypergraph::PatternVertex::concrete(2)
    }};
    
    EXPECT_EQ(edge.arity(), 3);
    EXPECT_TRUE(edge.vertices[0].is_concrete());
    EXPECT_TRUE(edge.vertices[1].is_variable());
    EXPECT_TRUE(edge.vertices[2].is_concrete());
}

// === BASIC PATTERN MATCHING TESTS ===

TEST_F(PatternMatchingComprehensiveTest, ExactConcreteMatch) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}});
    auto pattern = create_pattern({{1, 2}});  // Exact concrete match
    
    auto matches = matcher.find_matches_around(target, pattern, 1, 2);
    EXPECT_EQ(matches.size(), 1);
    
    auto& match = matches[0];
    EXPECT_EQ(match.matched_edges.size(), 1);
    EXPECT_TRUE(match.assignment.variable_to_concrete.empty());  // No variables
}

TEST_F(PatternMatchingComprehensiveTest, VariablePatternMatch) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}});
    auto pattern = create_pattern({{100, 101}});  // Variable pattern
    
    auto matches = matcher.find_matches_around(target, pattern, 2, 2);
    EXPECT_EQ(matches.size(), 2);  // Should match both edges
    
    for (const auto& match : matches) {
        EXPECT_EQ(match.matched_edges.size(), 1);
        EXPECT_EQ(match.assignment.variable_to_concrete.size(), 2);
        EXPECT_TRUE(match.is_valid());
    }
}

TEST_F(PatternMatchingComprehensiveTest, MixedConcreteVariableMatch) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {1, 3}});
    auto pattern = create_pattern({{1, 100}});  // Concrete vertex 1, variable 100
    
    auto matches = matcher.find_matches_around(target, pattern, 1, 2);
    EXPECT_EQ(matches.size(), 2);  // Should match both edges containing vertex 1
    
    for (const auto& match : matches) {
        EXPECT_EQ(match.matched_edges.size(), 1);
        EXPECT_EQ(match.assignment.variable_to_concrete.size(), 1);
        EXPECT_TRUE(match.assignment.variable_to_concrete.count(100));
    }
}

// === MULTI-EDGE PATTERN TESTS ===

TEST_F(PatternMatchingComprehensiveTest, TwoEdgeConnectedPattern) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}, {3, 4}});
    auto pattern = create_pattern({{100, 101}, {101, 102}});  // Connected chain
    
    auto matches = matcher.find_matches_around(target, pattern, 2, 2);
    EXPECT_GE(matches.size(), 1);
    
    auto& match = matches[0];
    EXPECT_EQ(match.matched_edges.size(), 2);
    EXPECT_EQ(match.assignment.variable_to_concrete.size(), 3);
    
    // Verify connectivity is preserved
    auto var100 = match.assignment.variable_to_concrete[100];
    auto var101 = match.assignment.variable_to_concrete[101];
    auto var102 = match.assignment.variable_to_concrete[102];
    
    EXPECT_NE(var100, var101);
    EXPECT_NE(var101, var102);
}

TEST_F(PatternMatchingComprehensiveTest, VariableConsistencyConstraint) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 1}});
    auto pattern = create_pattern({{100, 101}, {101, 100}});  // Symmetric pattern
    
    auto matches = matcher.find_matches_around(target, pattern, 1, 2);
    EXPECT_EQ(matches.size(), 2);  // Should find two valid assignments (symmetric pattern)
    
    auto& match = matches[0];
    EXPECT_EQ(match.assignment.variable_to_concrete.size(), 2);
    
    auto var100 = match.assignment.variable_to_concrete[100];
    auto var101 = match.assignment.variable_to_concrete[101];
    EXPECT_NE(var100, var101);
}

// === SELF-LOOP AND SPECIAL CASES ===

TEST_F(PatternMatchingComprehensiveTest, SelfLoopMatching) {
    auto target = test_utils::create_test_hypergraph({{1, 1}, {1, 2}});
    auto pattern = create_pattern({{100, 100}});  // Variable self-loop
    
    auto matches = matcher.find_matches_around(target, pattern, 1, 2);
    EXPECT_EQ(matches.size(), 1);  // Should match the self-loop
    
    auto& match = matches[0];
    EXPECT_EQ(match.assignment.variable_to_concrete.size(), 1);
    EXPECT_EQ(match.assignment.variable_to_concrete[100], 1);
}

TEST_F(PatternMatchingComprehensiveTest, NoMatchDifferentArity) {
    auto target = test_utils::create_test_hypergraph({{1, 2}});
    auto pattern = create_pattern({{100, 101, 102}});  // 3-ary pattern vs 2-ary edge

    auto matches = matcher.find_matches_around(target, pattern, 1, 2);
    EXPECT_TRUE(matches.empty());
}

TEST_F(PatternMatchingComprehensiveTest, BugMultiEdgePatternMatchesSingleEdge) {
    // BUG REPORT: Pattern {{1,2,3},{2,3,4}} incorrectly matches single edge {{1,1,1}}
    // A pattern with 2 edges should NEVER match a graph with only 1 edge
    auto target = test_utils::create_test_hypergraph({{1, 1, 1}});
    auto pattern = create_pattern({{1, 2, 3}, {2, 3, 4}});  // 2-edge concrete pattern

    auto matches = matcher.find_matches_around(target, pattern, 1, 10);

    EXPECT_TRUE(matches.empty())
        << "BUG: Pattern with 2 edges {{1,2,3},{2,3,4}} should NOT match single edge {{1,1,1}}, "
        << "but found " << matches.size() << " match(es)";

    EXPECT_EQ(pattern.num_edges(), 2) << "Pattern should have 2 edges";
    EXPECT_EQ(target.num_edges(), 1) << "Target should have only 1 edge";
}

TEST_F(PatternMatchingComprehensiveTest, BugAllVariablePatternTreatedAsVariables) {
    // BUG: FFI code treats ALL vertices as variables, not concrete
    // This test simulates what happens when FFI parses {{1,2,3},{2,3,4}}
    // and incorrectly treats all numbers as variable IDs
    auto target = test_utils::create_test_hypergraph({{5, 5, 5}});

    // Create a pattern where 1,2,3,4 are all VARIABLES (simulating the FFI bug)
    hypergraph::PatternHypergraph buggy_pattern;
    buggy_pattern.add_edge({
        hypergraph::PatternVertex::variable(1),
        hypergraph::PatternVertex::variable(2),
        hypergraph::PatternVertex::variable(3)
    });
    buggy_pattern.add_edge({
        hypergraph::PatternVertex::variable(2),
        hypergraph::PatternVertex::variable(3),
        hypergraph::PatternVertex::variable(4)
    });

    auto matches = matcher.find_matches_around(target, buggy_pattern, 5, 10);

    // With the bug, this MIGHT incorrectly match because all are variables
    // The correct behavior is: 2-edge pattern should NEVER match 1-edge graph
    EXPECT_TRUE(matches.empty())
        << "CRITICAL BUG: 2-edge pattern (even with all variables) should NOT match 1-edge graph! "
        << "Found " << matches.size() << " match(es)";

    EXPECT_EQ(buggy_pattern.num_edges(), 2);
    EXPECT_EQ(target.num_edges(), 1);
}

// === RADIUS CONSTRAINT TESTS ===

TEST_F(PatternMatchingComprehensiveTest, RadiusConstraintEnforcement) {
    // Create linear chain: 1-2-3-4-5
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}, {3, 4}, {4, 5}});
    auto pattern = create_pattern({{100, 101}});
    
    // Search with radius 1 from vertex 3
    auto matches_r1 = matcher.find_matches_around(target, pattern, 3, 1);
    EXPECT_EQ(matches_r1.size(), 2);  // Should find {2,3} and {3,4}
    
    // Search with radius 2 from vertex 3  
    auto matches_r2 = matcher.find_matches_around(target, pattern, 3, 2);
    EXPECT_EQ(matches_r2.size(), 4);  // Should find all edges
    
    // Search with radius 0 from vertex 3 (no edges incident to 3 in isolation)
    auto matches_r0 = matcher.find_matches_around(target, pattern, 3, 0);
    EXPECT_TRUE(matches_r0.empty());
}

// === EDGE MAPPING TESTS ===

TEST_F(PatternMatchingComprehensiveTest, EdgeMappingCorrectness) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}});
    auto pattern = create_pattern({{100, 101}, {101, 102}});
    
    auto matches = matcher.find_matches_around(target, pattern, 2, 2);
    EXPECT_GE(matches.size(), 1);
    
    auto& match = matches[0];
    EXPECT_EQ(match.edge_map.size(), 2);  // Two pattern edges mapped
    
    // Verify edge mapping contains valid edge IDs
    for (const auto& [pattern_idx, target_edge_id] : match.edge_map) {
        EXPECT_LT(pattern_idx, pattern.num_edges());
        EXPECT_LT(target_edge_id, target.num_edges());
    }
}

// === PERFORMANCE TESTS ===

TEST_F(PatternMatchingComprehensiveTest, LargeGraphPerformance) {
    // Create larger graph for performance testing
    std::vector<std::vector<hypergraph::VertexId>> large_edges;
    for (int i = 0; i < 100; ++i) {
        large_edges.push_back({static_cast<hypergraph::VertexId>(i), 
                              static_cast<hypergraph::VertexId>((i + 1) % 100)});
    }
    auto large_target = test_utils::create_test_hypergraph(large_edges);
    auto simple_pattern = create_pattern({{100, 101}});
    
    test_utils::PerfTimer timer;
    auto matches = matcher.find_matches_around(large_target, simple_pattern, 50, 5);
    double elapsed = timer.elapsed_ms();
    
    EXPECT_FALSE(matches.empty());
    EXPECT_LT(elapsed, 100.0);  // Should complete within 100ms
}

// === FIRST MATCH OPTIMIZATION ===

TEST_F(PatternMatchingComprehensiveTest, FirstMatchOptimization) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}, {3, 4}});
    auto pattern = create_pattern({{100, 101}});
    
    test_utils::PerfTimer timer1;
    auto first_match = matcher.find_first_match_around(target, pattern, 2, 2);
    double time1 = timer1.elapsed_ms();
    
    test_utils::PerfTimer timer2;
    auto all_matches = matcher.find_matches_around(target, pattern, 2, 2);
    double time2 = timer2.elapsed_ms();
    
    EXPECT_TRUE(first_match.has_value());
    EXPECT_FALSE(all_matches.empty());
    // First match should be faster or roughly equal (allowing for timing variations)
    EXPECT_LE(time1, time2 + 0.05);  // Allow 0.05ms tolerance for timing variations
    
    // Results should be consistent
    EXPECT_TRUE(first_match->is_valid());
    EXPECT_EQ(first_match->matched_edges.size(), all_matches[0].matched_edges.size());
}