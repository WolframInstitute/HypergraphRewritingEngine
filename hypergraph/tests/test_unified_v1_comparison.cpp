#include <gtest/gtest.h>
#include <vector>
#include <set>
#include <iostream>

// v1 includes
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>

// v2 (unified) includes
#include "hypergraph/unified/arena.hpp"
#include "hypergraph/unified/types.hpp"
#include "hypergraph/unified/bitset.hpp"
#include "hypergraph/unified/unified_hypergraph.hpp"
#include "hypergraph/unified/pattern.hpp"
#include "hypergraph/unified/pattern_matcher.hpp"
#include "hypergraph/unified/rewriter.hpp"

// Use namespaces with prefixes to avoid conflicts
namespace v1 = hypergraph;
namespace v2 = hypergraph::unified;

// =============================================================================
// V1 vs Unified Comparison Test Framework
// =============================================================================
//
// These tests verify that the unified architecture produces identical results
// to v1 for the same inputs. The comparison is based on:
// - Number of unique states (canonicalized)
// - Number of events
// - (Optional) Causal/branchial edge counts
//
// Note: The unified architecture is a redesigned system - we don't expect
// identical internal structures, but we DO expect identical observable behavior.

// =============================================================================
// Test Helpers
// =============================================================================

struct V1Result {
    size_t num_states;
    size_t num_events;
    size_t num_causal_edges;
    size_t num_branchial_edges;
};

struct UnifiedResult {
    size_t num_states;
    size_t num_events;
    size_t num_causal_edges;
    size_t num_branchial_edges;
};

// Run v1 evolution and return results
V1Result run_v1_evolution(
    const std::vector<v1::RewritingRule>& rules,
    const std::vector<std::vector<v1::GlobalVertexId>>& initial_edges,
    int steps
) {
    v1::WolframEvolution evolution(steps, 1, true, false);  // 1 thread, full capture

    for (const auto& rule : rules) {
        evolution.add_rule(rule);
    }

    evolution.evolve(initial_edges);

    const auto& graph = evolution.get_multiway_graph();

    V1Result result;
    result.num_states = graph.num_states();
    result.num_events = graph.num_events();

    // Count causal and branchial edges from event_edges
    auto event_edges = graph.get_event_edges();
    result.num_causal_edges = 0;
    result.num_branchial_edges = 0;
    for (const auto& ee : event_edges) {
        if (ee.type == v1::EventRelationType::CAUSAL) {
            result.num_causal_edges++;
        } else if (ee.type == v1::EventRelationType::BRANCHIAL) {
            result.num_branchial_edges++;
        }
    }

    return result;
}

// TODO: run_unified_evolution requires full EvolutionEngine integration
// For now, we test unified components individually and compare v1 behavior
// The full evolution comparison will be added when EvolutionEngine is complete

// =============================================================================
// Helper to create matching rules for v1 and v2.3
// =============================================================================

// Create v1 rule: {{x, y}} -> {{y, z}}
v1::RewritingRule create_v1_simple_rule() {
    v1::PatternHypergraph lhs, rhs;

    lhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(1),
        v1::PatternVertex::variable(2)
    });

    rhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(2),
        v1::PatternVertex::variable(3)  // Fresh vertex
    });

    return v1::RewritingRule(lhs, rhs);
}

// Create unified rule: {{x, y}} -> {{y, z}}
v2::RewriteRule create_unified_simple_rule() {
    return v2::make_rule(0)
        .lhs({0, 1})    // {x, y}
        .rhs({1, 2})    // {y, z} - z is fresh
        .build();
}

// Create v1 rule: {{x, y}, {y, z}} -> {{x, z}}
v1::RewritingRule create_v1_two_edge_rule() {
    v1::PatternHypergraph lhs, rhs;

    lhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(1),
        v1::PatternVertex::variable(2)
    });
    lhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(2),
        v1::PatternVertex::variable(3)
    });

    rhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(1),
        v1::PatternVertex::variable(3)
    });

    return v1::RewritingRule(lhs, rhs);
}

// Create unified rule: {{x, y}, {y, z}} -> {{x, z}}
v2::RewriteRule create_unified_two_edge_rule() {
    return v2::make_rule(0)
        .lhs({0, 1})    // {x, y}
        .lhs({1, 2})    // {y, z}
        .rhs({0, 2})    // {x, z}
        .build();
}

// Create v1 rule: {{x, y}} -> {{x, y}, {y, z}}
v1::RewritingRule create_v1_growth_rule() {
    v1::PatternHypergraph lhs, rhs;

    lhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(1),
        v1::PatternVertex::variable(2)
    });

    rhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(1),
        v1::PatternVertex::variable(2)
    });
    rhs.add_edge(v1::PatternEdge{
        v1::PatternVertex::variable(2),
        v1::PatternVertex::variable(3)
    });

    return v1::RewritingRule(lhs, rhs);
}

// Create unified rule: {{x, y}} -> {{x, y}, {y, z}}
v2::RewriteRule create_unified_growth_rule() {
    return v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();
}

// =============================================================================
// Comparison Tests
// =============================================================================

TEST(V1_Unified_Comparison, SimpleRule_SingleEdge_OneStep) {
    // v1 execution
    V1Result v1_result = run_v1_evolution(
        {create_v1_simple_rule()},
        {{1, 2}},  // Initial: single edge
        1          // 1 step
    );

    std::cout << "v1: " << v1_result.num_states << " states, " << v1_result.num_events << " events\n";

    // unified execution - manual for now since full evolution engine isn't wired
    v2::UnifiedHypergraph hg;

    v2::VertexId v0 = hg.alloc_vertex();
    v2::VertexId v1_vert = hg.alloc_vertex();
    v2::EdgeId e0 = hg.create_edge({v0, v1_vert});

    v2::SparseBitset edges;
    edges.set(e0, hg.arena());

    uint64_t hash = hg.compute_canonical_hash(edges);
    v2::StateId s0 = hg.create_state(std::move(edges), 0, hash, v2::INVALID_ID);

    // Apply simple rule once
    v2::RewriteRule rule = create_unified_simple_rule();
    v2::Rewriter rewriter(&hg);

    v2::VariableBinding binding;
    binding.bind(0, v0);
    binding.bind(1, v1_vert);

    v2::EdgeId matched[] = {e0};
    v2::RewriteResult result = rewriter.apply(rule, s0, matched, 1, binding, 1);

    EXPECT_TRUE(result.success);

    // After 1 step with rule {{x,y}} -> {{y,z}}, we get 1 new state
    // v1 should report 2 states (initial + 1 new) and 1 event
    EXPECT_GE(v1_result.num_states, 1u);
    EXPECT_GE(v1_result.num_events, 1u);

    // unified should have 2 states (initial + new) and 1 event
    EXPECT_EQ(hg.num_states(), 2u);
    EXPECT_EQ(hg.num_events(), 1u);
}

TEST(V1_Unified_Comparison, GrowthRule_Triangle_TwoSteps) {
    // v1 execution
    V1Result v1 = run_v1_evolution(
        {create_v1_growth_rule()},
        {{0, 1}, {1, 2}, {2, 0}},  // Initial: triangle
        2  // 2 steps
    );

    std::cout << "v1 (triangle, 2 steps): " << v1.num_states << " states, "
              << v1.num_events << " events\n";

    // For now, just verify v1 produces non-trivial results
    EXPECT_GT(v1.num_states, 1u);
    EXPECT_GT(v1.num_events, 0u);
}

TEST(V1_Unified_Comparison, NoMatchRule_SingleEdge) {
    // Rule requires 2 edges, initial state has 1 edge
    // Both v1 and unified should produce 0 events, 1 state

    V1Result v1 = run_v1_evolution(
        {create_v1_two_edge_rule()},
        {{1, 1, 1}},  // Single edge (self-loop)
        2             // 2 steps
    );

    std::cout << "v1 (no match): " << v1.num_states << " states, "
              << v1.num_events << " events\n";

    // Should have exactly 1 state (initial) and 0 events
    // (rule requires 2 edges, state has 1)
    EXPECT_EQ(v1.num_events, 0u);
    EXPECT_EQ(v1.num_states, 1u);
}

TEST(V1_Unified_Comparison, MatchTest_TwoEdgeChain) {
    // Initial: {{0,1}, {1,2}} forms a chain
    // Rule: {{x,y}, {y,z}} -> {{x,z}} should match once

    V1Result v1 = run_v1_evolution(
        {create_v1_two_edge_rule()},
        {{0, 1}, {1, 2}},  // Chain
        1                   // 1 step
    );

    std::cout << "v1 (chain, 1 step): " << v1.num_states << " states, "
              << v1.num_events << " events\n";

    // Should produce 1 event (the match) and 2 states (initial + result)
    EXPECT_EQ(v1.num_events, 1u);
    EXPECT_EQ(v1.num_states, 2u);
}

// =============================================================================
// Causal/Branchial Comparison
// =============================================================================

TEST(V1_Unified_Comparison, CausalEdges_SimpleChain) {
    // Create a chain of rewrites to verify causal edge tracking
    V1Result v1 = run_v1_evolution(
        {create_v1_simple_rule()},
        {{1, 2}},
        3  // 3 steps to create a causal chain
    );

    std::cout << "v1 (3 steps): " << v1.num_states << " states, "
              << v1.num_events << " events, "
              << v1.num_causal_edges << " causal, "
              << v1.num_branchial_edges << " branchial\n";

    // v1 should have causal edges forming a chain
    // (Each event produces edge consumed by next event)
    EXPECT_GT(v1.num_events, 1u);
    // Causal edges should exist (at least events-1 for a chain)
    if (v1.num_events > 1) {
        EXPECT_GT(v1.num_causal_edges, 0u);
    }
}

TEST(V1_Unified_Comparison, BranchialEdges_MultipleMatches) {
    // Triangle with growth rule - multiple matches from same state
    // Should create branchial edges between events
    V1Result v1 = run_v1_evolution(
        {create_v1_growth_rule()},
        {{0, 1}, {1, 2}, {2, 0}},  // Triangle
        1  // 1 step - should match 3 times
    );

    std::cout << "v1 (triangle multiway): " << v1.num_states << " states, "
              << v1.num_events << " events, "
              << v1.num_causal_edges << " causal, "
              << v1.num_branchial_edges << " branchial\n";

    // Should have multiple events from same initial state
    EXPECT_EQ(v1.num_events, 3u);  // 3 edges in triangle, 3 matches
}
