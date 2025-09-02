#include <gtest/gtest.h>
#include <hypergraph/rewriting.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <hypergraph/hypergraph.hpp>
#include "test_helpers.hpp"

class RewritingEngineTest : public ::testing::Test {
protected:
    hypergraph::RewritingEngine engine;
    
    // Helper to create rules
    hypergraph::RewritingRule create_simple_rule() {
        // Rule: {x, y} → {{x, z}, {z, y}} where z is fresh
        hypergraph::PatternHypergraph lhs;
        lhs.add_edge(hypergraph::PatternEdge{{
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        }});
        
        hypergraph::PatternHypergraph rhs;
        rhs.add_edge(hypergraph::PatternEdge{{
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(3)
        }});
        rhs.add_edge(hypergraph::PatternEdge{{
            hypergraph::PatternVertex::variable(3),
            hypergraph::PatternVertex::variable(2)
        }});
        
        return hypergraph::RewritingRule{lhs, rhs};
    }
    
    hypergraph::RewritingRule create_identity_rule() {
        // Rule: {x, y} → {x, y} (no change)
        hypergraph::PatternHypergraph pattern;
        pattern.add_edge(hypergraph::PatternEdge{{
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        }});
        
        return hypergraph::RewritingRule{pattern, pattern};
    }
    
    hypergraph::RewritingRule create_deletion_rule() {
        // Rule: {x, y} → {} (delete edge)
        hypergraph::PatternHypergraph lhs;
        lhs.add_edge(hypergraph::PatternEdge{{
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        }});
        
        hypergraph::PatternHypergraph rhs;  // Empty RHS
        
        return hypergraph::RewritingRule{lhs, rhs};
    }
};

// === BASIC REWRITING TESTS ===

TEST_F(RewritingEngineTest, BasicRuleApplication) {
    auto target = test_utils::create_test_hypergraph({{1, 2}});
    auto rule = create_simple_rule();
    
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result.applied);
    EXPECT_FALSE(result.added_edges.empty());
    EXPECT_FALSE(result.removed_edges.empty());
}

TEST_F(RewritingEngineTest, IdentityRuleApplication) {
    auto target = test_utils::create_test_hypergraph({{1, 2}});
    auto rule = create_identity_rule();
    
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.added_edges.size(), 1);
    EXPECT_EQ(result.removed_edges.size(), 1);
    
    // Note: apply_rule_at modifies the graph in place, so test is different
    EXPECT_GT(target.num_edges(), 0);
}

TEST_F(RewritingEngineTest, DeletionRuleApplication) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}});
    auto rule = create_deletion_rule();
    
    std::size_t initial_edges = target.num_edges();
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.added_edges.empty());  // No edges added
    EXPECT_FALSE(result.removed_edges.empty());  // One edge removed
    
    // Rule applied in place - check edge count decreased
    EXPECT_LT(target.num_edges(), initial_edges);
}

// === RULE APPLICATION WITH MULTIPLE MATCHES ===

TEST_F(RewritingEngineTest, MultipleMatchesSingleRule) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {3, 4}});
    auto rule = create_simple_rule();
    
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result.applied);  // Should find at least one match
    EXPECT_EQ(result.removed_edges.size(), 1);
    EXPECT_EQ(result.added_edges.size(), 2);
}

TEST_F(RewritingEngineTest, RuleApplicationWithAnchor) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}, {3, 4}});
    auto rule = create_simple_rule();
    
    // Apply rule anchored at vertex 2
    auto result_anchor2 = engine.apply_rule_at(target, rule, 2, 2);
    EXPECT_TRUE(result_anchor2.applied);
    
    // Create fresh copy for second test
    auto target2 = test_utils::create_test_hypergraph({{1, 2}, {2, 3}, {3, 4}});
    auto result_anchor1 = engine.apply_rule_at(target2, rule, 1, 2);
    EXPECT_TRUE(result_anchor1.applied);
}

// === FRESH VERTEX GENERATION ===

TEST_F(RewritingEngineTest, FreshVertexGeneration) {
    auto target = test_utils::create_test_hypergraph({{1, 2}});
    auto rule = create_simple_rule();  // Introduces variable 3 (fresh)
    
    std::size_t initial_vertices = target.num_vertices();
    std::size_t initial_edges = target.num_edges();
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result.applied);
    
    // Should have introduced new vertices and edges
    EXPECT_GT(target.num_vertices(), initial_vertices);
    EXPECT_GT(target.num_edges(), initial_edges);
}

// === EDGE ADDITION AND REMOVAL ===

TEST_F(RewritingEngineTest, EdgeRemovalCorrectness) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}});
    auto rule = create_deletion_rule();
    
    std::size_t initial_edges = target.num_edges();
    auto result = engine.apply_rule_at(target, rule, 2, 2);
    EXPECT_TRUE(result.applied);
    
    // Rule applied in place - edge count should decrease
    EXPECT_EQ(target.num_edges(), initial_edges - 1);
}

TEST_F(RewritingEngineTest, EdgeAdditionCorrectness) {
    auto target = test_utils::create_test_hypergraph({{1, 2}});
    auto rule = create_simple_rule();  // Adds 2 edges, removes 1
    
    std::size_t initial_edges = target.num_edges();
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result.applied);
    
    EXPECT_EQ(target.num_edges(), initial_edges + 1);  // Net +1 edge
}

// === RULE VALIDATION ===

TEST_F(RewritingEngineTest, InvalidRuleHandling) {
    auto target = test_utils::create_test_hypergraph({{1, 2}});
    
    // Create rule with concrete vertices that don't exist
    hypergraph::PatternHypergraph lhs;
    lhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::concrete(999),  // Non-existent vertex
        hypergraph::PatternVertex::concrete(1000)
    }});
    
    hypergraph::PatternHypergraph rhs;
    rhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::concrete(999),
        hypergraph::PatternVertex::concrete(1000)
    }});
    
    hypergraph::RewritingRule invalid_rule{lhs, rhs};
    
    auto result = engine.apply_rule_at(target, invalid_rule, 1);
    EXPECT_FALSE(result.applied);  // No matches possible
}

// === MULTI-STEP REWRITING ===

TEST_F(RewritingEngineTest, MultiStepRewriting) {
    auto target = test_utils::create_test_hypergraph({{1, 2}});
    auto rule = create_simple_rule();
    
    // First application
    std::size_t initial_edges = target.num_edges();
    auto result1 = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result1.applied);
    
    std::size_t step1_edges = target.num_edges();
    EXPECT_GT(step1_edges, initial_edges);
    
    // Second application on result
    auto result2 = engine.apply_rule_at(target, rule, 1, 2);
    EXPECT_TRUE(result2.applied);  // Should be able to apply again
    
    // Should continue growing
    EXPECT_GT(target.num_edges(), step1_edges);
}

// === PERFORMANCE TESTS ===

TEST_F(RewritingEngineTest, RewritingPerformance) {
    // Create moderately sized hypergraph
    std::vector<std::vector<hypergraph::VertexId>> edges;
    for (int i = 0; i < 50; ++i) {
        edges.push_back({static_cast<hypergraph::VertexId>(i), 
                        static_cast<hypergraph::VertexId>((i + 1) % 50)});
    }
    auto target = test_utils::create_test_hypergraph(edges);
    auto rule = create_simple_rule();
    
    test_utils::PerfTimer timer;
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    double elapsed = timer.elapsed_ms();
    
    EXPECT_TRUE(result.applied);
    EXPECT_LT(elapsed, 50.0);  // Should complete within 50ms
}

// === LISTENERS AND EVENTS ===

class TestRewritingListener : public hypergraph::RewritingEventListener {
public:
    int rules_applied = 0;
    int rules_failed = 0;
    int steps_completed = 0;
    
    void on_rule_applied(const hypergraph::RewritingResult& result, 
                        const hypergraph::Hypergraph& hypergraph) override {
        rules_applied++;
    }
    
    void on_rule_failed(const hypergraph::RewritingRule& rule, 
                       hypergraph::VertexId anchor_vertex) override {
        rules_failed++;
    }
    
    void on_step_completed(std::size_t step_number, 
                          const hypergraph::Hypergraph& hypergraph) override {
        steps_completed++;
    }
};

TEST_F(RewritingEngineTest, EventListenerIntegration) {
    auto target = test_utils::create_test_hypergraph({{1, 2}, {2, 3}});
    auto rule = create_simple_rule();
    
    auto listener_ptr = std::make_unique<TestRewritingListener>();
    auto& listener_ref = *listener_ptr;
    engine.add_listener(std::move(listener_ptr));
    
    auto result = engine.apply_rule_at(target, rule, 1, 2);
    
    if (result.applied) {
        EXPECT_GT(listener_ref.rules_applied, 0);
    }
    EXPECT_EQ(listener_ref.rules_failed, 0);  // No failures expected
}

// === COMPLEX RULE PATTERNS ===

TEST_F(RewritingEngineTest, ComplexRulePattern) {
    // Rule: triangle {x,y}, {y,z}, {z,x} → star {x,w}, {y,w}, {z,w}
    hypergraph::PatternHypergraph triangle_lhs;
    triangle_lhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::variable(1),
        hypergraph::PatternVertex::variable(2)
    }});
    triangle_lhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::variable(2),
        hypergraph::PatternVertex::variable(3)
    }});
    triangle_lhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::variable(3),
        hypergraph::PatternVertex::variable(1)
    }});
    
    hypergraph::PatternHypergraph star_rhs;
    star_rhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::variable(1),
        hypergraph::PatternVertex::variable(4)  // Fresh center vertex
    }});
    star_rhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::variable(2),
        hypergraph::PatternVertex::variable(4)
    }});
    star_rhs.add_edge(hypergraph::PatternEdge{{
        hypergraph::PatternVertex::variable(3),
        hypergraph::PatternVertex::variable(4)
    }});
    
    hypergraph::RewritingRule triangle_to_star{triangle_lhs, star_rhs};
    
    // Create triangle: 1-2, 2-3, 3-1
    auto triangle = test_utils::create_test_hypergraph({{1, 2}, {2, 3}, {3, 1}});
    
    auto result = engine.apply_rule_at(triangle, triangle_to_star, 1, 3);
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.removed_edges.size(), 3);  // Remove triangle
    EXPECT_EQ(result.added_edges.size(), 3);    // Add star
}