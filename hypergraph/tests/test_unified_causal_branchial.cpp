#include <gtest/gtest.h>
#include "hypergraph/unified/arena.hpp"
#include "hypergraph/unified/types.hpp"
#include "hypergraph/unified/bitset.hpp"
#include "hypergraph/unified/unified_hypergraph.hpp"
#include "hypergraph/unified/rewriter.hpp"
#include "hypergraph/unified/causal_graph.hpp"
#include <vector>
#include <set>
#include <algorithm>

using namespace hypergraph::unified;

// =============================================================================
// Test Helpers
// =============================================================================

// Create a simple rule: {{x, y}} -> {{y, z}} (consumes edge, creates new with fresh vertex)
RewriteRule create_simple_rule() {
    // x=0, y=1, z=2 (fresh)
    return make_rule(0)
        .lhs({0, 1})        // {x, y}
        .rhs({1, 2})        // {y, z}
        .build();
}

// Create a rule that produces two edges: {{x, y}} -> {{y, z}, {z, x}}
RewriteRule create_branching_rule() {
    // x=0, y=1, z=2 (fresh)
    return make_rule(0)
        .lhs({0, 1})        // {x, y}
        .rhs({1, 2})        // {y, z}
        .rhs({2, 0})        // {z, x}
        .build();
}

// Create a two-edge rule: {{x, y}, {y, z}} -> {{x, z}}
RewriteRule create_two_edge_rule() {
    // x=0, y=1, z=2
    return make_rule(0)
        .lhs({0, 1})        // {x, y}
        .lhs({1, 2})        // {y, z}
        .rhs({0, 2})        // {x, z}
        .build();
}

// =============================================================================
// CausalGraph Basic Tests
// =============================================================================

TEST(Unified_CausalGraph, BasicConstruction) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    EXPECT_EQ(cg.num_causal_edges(), 0u);
    EXPECT_EQ(cg.num_branchial_edges(), 0u);
}

TEST(Unified_CausalGraph, SetEdgeProducer) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Set producer for edge 0
    EXPECT_TRUE(cg.set_edge_producer(0, 42));  // First set succeeds
    EXPECT_EQ(cg.get_edge_producer(0), 42u);

    // Second set should fail (already set)
    EXPECT_FALSE(cg.set_edge_producer(0, 99));
    EXPECT_EQ(cg.get_edge_producer(0), 42u);  // Still 42
}

TEST(Unified_CausalGraph, ProducerConsumerCausalEdge) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Event 0 produces edge 5
    cg.set_edge_producer(5, 0);

    // Event 1 consumes edge 5
    cg.add_edge_consumer(5, 1);

    // Should create causal edge 0 -> 1
    EXPECT_EQ(cg.num_causal_edges(), 1u);

    auto edges = cg.get_causal_edges();
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0].producer, 0u);
    EXPECT_EQ(edges[0].consumer, 1u);
    EXPECT_EQ(edges[0].edge, 5u);
}

TEST(Unified_CausalGraph, ConsumerBeforeProducer_Rendezvous) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Event 1 consumes edge 5 BEFORE producer is set
    cg.add_edge_consumer(5, 1);

    // No causal edge yet (no producer)
    EXPECT_EQ(cg.num_causal_edges(), 0u);

    // Event 0 produces edge 5 (arrives later)
    cg.set_edge_producer(5, 0);

    // Now causal edge should exist (rendezvous pattern)
    EXPECT_EQ(cg.num_causal_edges(), 1u);

    auto edges = cg.get_causal_edges();
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0].producer, 0u);
    EXPECT_EQ(edges[0].consumer, 1u);
}

TEST(Unified_CausalGraph, MultipleConsumers) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Event 0 produces edge 10
    cg.set_edge_producer(10, 0);

    // Events 1, 2, 3 all consume edge 10
    cg.add_edge_consumer(10, 1);
    cg.add_edge_consumer(10, 2);
    cg.add_edge_consumer(10, 3);

    // Should create 3 causal edges: 0->1, 0->2, 0->3
    EXPECT_EQ(cg.num_causal_edges(), 3u);

    auto edges = cg.get_causal_edges();
    EXPECT_EQ(edges.size(), 3u);

    // All should have producer 0
    for (const auto& e : edges) {
        EXPECT_EQ(e.producer, 0u);
        EXPECT_TRUE(e.consumer == 1 || e.consumer == 2 || e.consumer == 3);
    }
}

TEST(Unified_CausalGraph, BranchialEdgeManual) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Manually add branchial edge
    cg.add_branchial_edge(1, 2, 5);

    EXPECT_EQ(cg.num_branchial_edges(), 1u);

    auto edges = cg.get_branchial_edges();
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0].event1, 1u);  // Always smaller
    EXPECT_EQ(edges[0].event2, 2u);
    EXPECT_EQ(edges[0].shared_edge, 5u);
}

// =============================================================================
// Rewriter Integration Tests (Online Causal/Branchial)
// =============================================================================

TEST(Unified_Rewriter, SingleRewrite_CausalTracking) {
    UnifiedHypergraph hg;

    // Create initial state with edge {0, 1}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    StateId s0 = hg.create_state({e0}, 0, 0, INVALID_ID);

    // Create rule: {{x, y}} -> {{y, z}}
    RewriteRule rule = create_simple_rule();

    // Create binding: x -> v0, y -> v1
    VariableBinding binding;
    binding.bind(0, v0);
    binding.bind(1, v1);

    // Apply rewrite
    Rewriter rewriter(&hg);
    EdgeId matched[] = {e0};
    RewriteResult result = rewriter.apply(rule, s0, matched, 1, binding, 1);

    EXPECT_TRUE(result.success);
    EXPECT_NE(result.event, INVALID_ID);
    EXPECT_NE(result.new_state, INVALID_ID);
    EXPECT_EQ(result.num_produced, 1u);

    // Check that produced edge has producer set
    EdgeId produced = result.produced_edges[0];
    EXPECT_EQ(hg.causal_graph().get_edge_producer(produced), result.event);

    // Initial edge e0 has no producer (it's an initial edge)
    // But it was consumed, so this event is registered as consumer
    // No causal edge expected since e0 has no producer
    EXPECT_EQ(hg.num_causal_edges(), 0u);
}

TEST(Unified_Rewriter, ChainedRewrites_CausalChain) {
    UnifiedHypergraph hg;

    // Create initial edge {0, 1}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    StateId s0 = hg.create_state({e0}, 0, 0, INVALID_ID);

    RewriteRule rule = create_simple_rule();
    Rewriter rewriter(&hg);

    // First rewrite: consumes e0 (initial), produces e1
    VariableBinding bind1;
    bind1.bind(0, v0);
    bind1.bind(1, v1);

    EdgeId matched1[] = {e0};
    RewriteResult result1 = rewriter.apply(rule, s0, matched1, 1, bind1, 1);
    ASSERT_TRUE(result1.success);

    EdgeId e1 = result1.produced_edges[0];
    EventId event1 = result1.event;
    StateId s1 = result1.new_state;

    // No causal edges yet (e0 was initial, no producer)
    EXPECT_EQ(hg.num_causal_edges(), 0u);

    // Second rewrite: consumes e1 (produced by event1), produces e2
    // e1 is {v1, fresh_vertex}
    const Edge& edge1 = hg.get_edge(e1);
    VertexId v2 = edge1.vertices[1];  // The fresh vertex

    VariableBinding bind2;
    bind2.bind(0, v1);  // x -> v1
    bind2.bind(1, v2);  // y -> v2

    EdgeId matched2[] = {e1};
    RewriteResult result2 = rewriter.apply(rule, s1, matched2, 1, bind2, 2);
    ASSERT_TRUE(result2.success);

    EventId event2 = result2.event;

    // Now we should have causal edge: event1 -> event2
    EXPECT_EQ(hg.num_causal_edges(), 1u);

    auto causal_edges = hg.causal_graph().get_causal_edges();
    ASSERT_EQ(causal_edges.size(), 1u);
    EXPECT_EQ(causal_edges[0].producer, event1);
    EXPECT_EQ(causal_edges[0].consumer, event2);
    EXPECT_EQ(causal_edges[0].edge, e1);
}

TEST(Unified_Rewriter, ParallelRewrites_BranchialEdges) {
    UnifiedHypergraph hg;

    // Create initial state with two edges: {0, 1}, {1, 2}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});

    StateId s0 = hg.create_state({e0, e1}, 0, 0, INVALID_ID);

    RewriteRule rule = create_simple_rule();
    Rewriter rewriter(&hg);

    // First rewrite from s0: apply to e0
    VariableBinding bind1;
    bind1.bind(0, v0);
    bind1.bind(1, v1);

    EdgeId matched1[] = {e0};
    RewriteResult result1 = rewriter.apply(rule, s0, matched1, 1, bind1, 1);
    ASSERT_TRUE(result1.success);

    // Second rewrite from SAME state s0: apply to e1
    // This should create branchial edge with event1
    VariableBinding bind2;
    bind2.bind(0, v1);
    bind2.bind(1, v2);

    EdgeId matched2[] = {e1};
    RewriteResult result2 = rewriter.apply(rule, s0, matched2, 1, bind2, 1);
    ASSERT_TRUE(result2.success);

    // Both events consume different edges from same state
    // No overlapping consumed edges -> NO branchial edge
    EXPECT_EQ(hg.num_branchial_edges(), 0u);
}

TEST(Unified_Rewriter, OverlappingRewrites_BranchialEdges) {
    UnifiedHypergraph hg;

    // Create initial state with one edge: {0, 1}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});

    StateId s0 = hg.create_state({e0}, 0, 0, INVALID_ID);

    RewriteRule rule = create_simple_rule();
    Rewriter rewriter(&hg);

    // First rewrite from s0: apply to e0
    VariableBinding bind1;
    bind1.bind(0, v0);
    bind1.bind(1, v1);

    EdgeId matched1[] = {e0};
    RewriteResult result1 = rewriter.apply(rule, s0, matched1, 1, bind1, 1);
    ASSERT_TRUE(result1.success);
    EventId event1 = result1.event;

    // Second rewrite from SAME state s0: ALSO apply to e0 (different binding direction)
    // In practice this would be x->v1, y->v0 but same edge consumed
    VariableBinding bind2;
    bind2.bind(0, v1);
    bind2.bind(1, v0);

    EdgeId matched2[] = {e0};  // Same edge!
    RewriteResult result2 = rewriter.apply(rule, s0, matched2, 1, bind2, 1);
    ASSERT_TRUE(result2.success);
    EventId event2 = result2.event;

    // Both events consume e0 from same state -> branchial edge
    EXPECT_EQ(hg.num_branchial_edges(), 1u);

    auto branchial_edges = hg.causal_graph().get_branchial_edges();
    ASSERT_EQ(branchial_edges.size(), 1u);

    // Check that it connects the two events
    EventId e1 = branchial_edges[0].event1;
    EventId e2 = branchial_edges[0].event2;
    EXPECT_TRUE((e1 == event1 && e2 == event2) || (e1 == event2 && e2 == event1));
    EXPECT_EQ(branchial_edges[0].shared_edge, e0);
}

TEST(Unified_Rewriter, TwoEdgeRule_CausalTracking) {
    UnifiedHypergraph hg;

    // Create initial state with edges forming a path: {0, 1}, {1, 2}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});

    StateId s0 = hg.create_state({e0, e1}, 0, 0, INVALID_ID);

    // Rule: {{x, y}, {y, z}} -> {{x, z}}
    RewriteRule rule = create_two_edge_rule();
    Rewriter rewriter(&hg);

    VariableBinding binding;
    binding.bind(0, v0);  // x -> v0
    binding.bind(1, v1);  // y -> v1
    binding.bind(2, v2);  // z -> v2

    EdgeId matched[] = {e0, e1};
    RewriteResult result = rewriter.apply(rule, s0, matched, 2, binding, 1);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.num_produced, 1u);

    // Produced edge should be {v0, v2}
    EdgeId produced = result.produced_edges[0];
    const Edge& pedge = hg.get_edge(produced);
    EXPECT_EQ(pedge.arity, 2u);
    EXPECT_EQ(pedge.vertices[0], v0);
    EXPECT_EQ(pedge.vertices[1], v2);

    // Producer should be set
    EXPECT_EQ(hg.causal_graph().get_edge_producer(produced), result.event);
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST(Unified_CausalGraph, ManyEdges) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    const int NUM_EDGES = 1000;
    const int NUM_EVENTS = 100;

    // Create chain: event i produces edge i, event i+1 consumes edge i
    for (int i = 0; i < NUM_EVENTS - 1; ++i) {
        cg.set_edge_producer(i, i);
        cg.add_edge_consumer(i, i + 1);
    }

    // Should have NUM_EVENTS - 1 causal edges
    EXPECT_EQ(cg.num_causal_edges(), static_cast<size_t>(NUM_EVENTS - 1));

    auto edges = cg.get_causal_edges();
    EXPECT_EQ(edges.size(), static_cast<size_t>(NUM_EVENTS - 1));
}

TEST(Unified_CausalGraph, MultipleProducersMultipleConsumers) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // 3 producer events, each produces 2 edges
    // 4 consumer events, each consumes all 6 edges
    for (int p = 0; p < 3; ++p) {
        cg.set_edge_producer(p * 2, p);
        cg.set_edge_producer(p * 2 + 1, p);
    }

    for (int c = 0; c < 4; ++c) {
        for (int e = 0; e < 6; ++e) {
            cg.add_edge_consumer(e, 10 + c);  // Consumer events start at ID 10
        }
    }

    // Each of 6 edges has 4 consumers = 24 causal edges
    EXPECT_EQ(cg.num_causal_edges(), 24u);
}

// =============================================================================
// UniquenessTree Integration Tests
// =============================================================================

TEST(Unified_UniquenessTree, CanonicalHash_EmptyState) {
    UnifiedHypergraph hg;
    SparseBitset empty_edges;

    uint64_t hash = hg.compute_canonical_hash(empty_edges);
    EXPECT_EQ(hash, 0u);
}

TEST(Unified_UniquenessTree, CanonicalHash_SingleEdge) {
    UnifiedHypergraph hg;

    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    SparseBitset edges;
    edges.set(e0, hg.arena());

    uint64_t hash = hg.compute_canonical_hash(edges);
    EXPECT_NE(hash, 0u);  // Should produce non-zero hash
}

TEST(Unified_UniquenessTree, CanonicalHash_IsomorphicStates) {
    UnifiedHypergraph hg;

    // State 1: Triangle with vertices 0, 1, 2
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});
    EdgeId e2 = hg.create_edge({v2, v0});

    SparseBitset state1_edges;
    state1_edges.set(e0, hg.arena());
    state1_edges.set(e1, hg.arena());
    state1_edges.set(e2, hg.arena());

    uint64_t hash1 = hg.compute_canonical_hash(state1_edges);

    // State 2: Triangle with vertices 3, 4, 5 (isomorphic to state 1)
    VertexId v3 = hg.alloc_vertex();
    VertexId v4 = hg.alloc_vertex();
    VertexId v5 = hg.alloc_vertex();

    EdgeId e3 = hg.create_edge({v3, v4});
    EdgeId e4 = hg.create_edge({v4, v5});
    EdgeId e5 = hg.create_edge({v5, v3});

    SparseBitset state2_edges;
    state2_edges.set(e3, hg.arena());
    state2_edges.set(e4, hg.arena());
    state2_edges.set(e5, hg.arena());

    uint64_t hash2 = hg.compute_canonical_hash(state2_edges);

    // Isomorphic states should have same canonical hash
    EXPECT_EQ(hash1, hash2);
}

TEST(Unified_UniquenessTree, CanonicalHash_NonIsomorphicStates) {
    UnifiedHypergraph hg;

    // State 1: Triangle with vertices 0, 1, 2
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});
    EdgeId e2 = hg.create_edge({v2, v0});

    SparseBitset triangle_edges;
    triangle_edges.set(e0, hg.arena());
    triangle_edges.set(e1, hg.arena());
    triangle_edges.set(e2, hg.arena());

    uint64_t triangle_hash = hg.compute_canonical_hash(triangle_edges);

    // State 2: Path with vertices 3, 4, 5, 6
    VertexId v3 = hg.alloc_vertex();
    VertexId v4 = hg.alloc_vertex();
    VertexId v5 = hg.alloc_vertex();
    VertexId v6 = hg.alloc_vertex();

    EdgeId e3 = hg.create_edge({v3, v4});
    EdgeId e4 = hg.create_edge({v4, v5});
    EdgeId e5 = hg.create_edge({v5, v6});

    SparseBitset path_edges;
    path_edges.set(e3, hg.arena());
    path_edges.set(e4, hg.arena());
    path_edges.set(e5, hg.arena());

    uint64_t path_hash = hg.compute_canonical_hash(path_edges);

    // Non-isomorphic states should have different canonical hashes
    EXPECT_NE(triangle_hash, path_hash);
}

TEST(Unified_UniquenessTree, CanonicalHash_SelfLoop) {
    UnifiedHypergraph hg;

    // State 1: Self-loop at vertex 0
    VertexId v0 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v0});

    SparseBitset state1_edges;
    state1_edges.set(e0, hg.arena());

    uint64_t hash1 = hg.compute_canonical_hash(state1_edges);

    // State 2: Self-loop at vertex 1 (isomorphic)
    VertexId v1 = hg.alloc_vertex();
    EdgeId e1 = hg.create_edge({v1, v1});

    SparseBitset state2_edges;
    state2_edges.set(e1, hg.arena());

    uint64_t hash2 = hg.compute_canonical_hash(state2_edges);

    EXPECT_EQ(hash1, hash2);
}

TEST(Unified_UniquenessTree, CanonicalInfo_VertexClasses) {
    UnifiedHypergraph hg;

    // Create a star graph: center vertex 0 connected to 1, 2, 3
    VertexId v0 = hg.alloc_vertex();  // Center
    VertexId v1 = hg.alloc_vertex();  // Leaf
    VertexId v2 = hg.alloc_vertex();  // Leaf
    VertexId v3 = hg.alloc_vertex();  // Leaf

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v0, v2});
    EdgeId e2 = hg.create_edge({v0, v3});

    SparseBitset edges;
    edges.set(e0, hg.arena());
    edges.set(e1, hg.arena());
    edges.set(e2, hg.arena());

    StateCanonicalInfo info = hg.compute_canonical_info(edges);

    EXPECT_NE(info.canonical_hash, 0u);
    EXPECT_EQ(info.num_vertices, 4u);

    // Should have 2 equivalence classes: 1 for center (degree 3), 1 for leaves (degree 1)
    EXPECT_EQ(info.num_classes, 2u);

    // One class should have 1 vertex (center), other should have 3 (leaves)
    bool found_center = false;
    bool found_leaves = false;
    for (uint32_t i = 0; i < info.num_classes; ++i) {
        if (info.equiv_classes[i].count == 1) found_center = true;
        if (info.equiv_classes[i].count == 3) found_leaves = true;
    }
    EXPECT_TRUE(found_center);
    EXPECT_TRUE(found_leaves);
}

// =============================================================================
// Level 2 Edge Correspondence Tests
// =============================================================================

TEST(Unified_Level2, EdgeCorrespondence_IsomorphicStates) {
    UnifiedHypergraph hg;
    hg.enable_level2();

    // Create first triangle: vertices 0, 1, 2
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});
    EdgeId e2 = hg.create_edge({v2, v0});

    SparseBitset state1_edges;
    state1_edges.set(e0, hg.arena());
    state1_edges.set(e1, hg.arena());
    state1_edges.set(e2, hg.arena());

    uint64_t hash1 = hg.compute_canonical_hash(state1_edges);

    // Create state 1
    auto [state1, raw1, was_new1] = hg.create_or_get_canonical_state(
        std::move(state1_edges), hash1, 0, INVALID_ID);
    EXPECT_TRUE(was_new1);

    // Create second triangle: vertices 3, 4, 5 (isomorphic to first)
    VertexId v3 = hg.alloc_vertex();
    VertexId v4 = hg.alloc_vertex();
    VertexId v5 = hg.alloc_vertex();

    EdgeId e3 = hg.create_edge({v3, v4});
    EdgeId e4 = hg.create_edge({v4, v5});
    EdgeId e5 = hg.create_edge({v5, v3});

    SparseBitset state2_edges;
    state2_edges.set(e3, hg.arena());
    state2_edges.set(e4, hg.arena());
    state2_edges.set(e5, hg.arena());

    uint64_t hash2 = hg.compute_canonical_hash(state2_edges);
    EXPECT_EQ(hash1, hash2);  // Should be isomorphic

    // Try to create state 2 - should detect duplicate and compute correspondence
    auto [state2, raw2, was_new2] = hg.create_or_get_canonical_state(
        std::move(state2_edges), hash2, 0, INVALID_ID);
    EXPECT_FALSE(was_new2);  // Should find existing state
    EXPECT_EQ(state2, state1);  // Should return existing canonical state

    // Check that edges are now in equivalence classes
    // e0 and e3 should correspond (both are first edge of triangle)
    // e1 and e4 should correspond (both are second edge)
    // e2 and e5 should correspond (both are third edge closing triangle)
    EXPECT_TRUE(hg.edge_equiv_manager().are_equivalent(e0, e3));
    EXPECT_TRUE(hg.edge_equiv_manager().are_equivalent(e1, e4));
    EXPECT_TRUE(hg.edge_equiv_manager().are_equivalent(e2, e5));

    // Edges from different positions should NOT be equivalent
    EXPECT_FALSE(hg.edge_equiv_manager().are_equivalent(e0, e4));
    EXPECT_FALSE(hg.edge_equiv_manager().are_equivalent(e1, e5));
}

TEST(Unified_Level2, EdgeCorrespondence_Disabled) {
    UnifiedHypergraph hg;
    // Level 2 disabled by default

    // Create first triangle
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});
    EdgeId e2 = hg.create_edge({v2, v0});

    SparseBitset state1_edges;
    state1_edges.set(e0, hg.arena());
    state1_edges.set(e1, hg.arena());
    state1_edges.set(e2, hg.arena());

    uint64_t hash1 = hg.compute_canonical_hash(state1_edges);
    auto [state1, raw1, was_new1] = hg.create_or_get_canonical_state(
        std::move(state1_edges), hash1, 0, INVALID_ID);
    EXPECT_TRUE(was_new1);

    // Create second triangle (isomorphic)
    VertexId v3 = hg.alloc_vertex();
    VertexId v4 = hg.alloc_vertex();
    VertexId v5 = hg.alloc_vertex();

    EdgeId e3 = hg.create_edge({v3, v4});
    EdgeId e4 = hg.create_edge({v4, v5});
    EdgeId e5 = hg.create_edge({v5, v3});

    SparseBitset state2_edges;
    state2_edges.set(e3, hg.arena());
    state2_edges.set(e4, hg.arena());
    state2_edges.set(e5, hg.arena());

    uint64_t hash2 = hg.compute_canonical_hash(state2_edges);
    auto [state2, raw2, was_new2] = hg.create_or_get_canonical_state(
        std::move(state2_edges), hash2, 0, INVALID_ID);
    EXPECT_FALSE(was_new2);

    // With Level 2 disabled, edges should NOT be in equivalence classes
    // Each edge is its own representative
    EXPECT_FALSE(hg.edge_equiv_manager().are_equivalent(e0, e3));
    EXPECT_FALSE(hg.edge_equiv_manager().are_equivalent(e1, e4));
    EXPECT_FALSE(hg.edge_equiv_manager().are_equivalent(e2, e5));
}

TEST(Unified_Level2, EventCanonicalization_EquivalentEvents) {
    UnifiedHypergraph hg;
    hg.enable_level2();

    // Create first triangle with edge correspondence
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});
    EdgeId e2 = hg.create_edge({v2, v0});

    SparseBitset state1_edges;
    state1_edges.set(e0, hg.arena());
    state1_edges.set(e1, hg.arena());
    state1_edges.set(e2, hg.arena());

    uint64_t hash1 = hg.compute_canonical_hash(state1_edges);
    auto [state1, raw1, was_new1] = hg.create_or_get_canonical_state(
        std::move(state1_edges), hash1, 0, INVALID_ID);

    // Create second triangle (isomorphic)
    VertexId v3 = hg.alloc_vertex();
    VertexId v4 = hg.alloc_vertex();
    VertexId v5 = hg.alloc_vertex();

    EdgeId e3 = hg.create_edge({v3, v4});
    EdgeId e4 = hg.create_edge({v4, v5});
    EdgeId e5 = hg.create_edge({v5, v3});

    SparseBitset state2_edges;
    state2_edges.set(e3, hg.arena());
    state2_edges.set(e4, hg.arena());
    state2_edges.set(e5, hg.arena());

    uint64_t hash2 = hg.compute_canonical_hash(state2_edges);
    auto [state2, raw2, was_new2] = hg.create_or_get_canonical_state(
        std::move(state2_edges), hash2, 0, INVALID_ID);

    // Edges should now be in equivalence classes
    EXPECT_TRUE(hg.edge_equiv_manager().are_equivalent(e0, e3));
    EXPECT_TRUE(hg.edge_equiv_manager().are_equivalent(e1, e4));
    EXPECT_TRUE(hg.edge_equiv_manager().are_equivalent(e2, e5));

    // Compute canonical hash for two hypothetical events that consume corresponding edges
    // Event 1: rule 0, consumes e0 from state1
    // Event 2: rule 0, consumes e3 from state1 (equivalent to e0)

    uint64_t event_hash_1 = EventCanonicalizer::compute_canonical_hash(
        0,       // rule_index
        hash1,   // input_state_hash
        &e0, 1,  // consumed_edges
        hash1,   // output_state_hash (hypothetical)
        hg.edge_equiv_manager()
    );

    uint64_t event_hash_2 = EventCanonicalizer::compute_canonical_hash(
        0,       // same rule
        hash1,   // same input state hash
        &e3, 1,  // corresponding edge
        hash1,   // same output state hash
        hg.edge_equiv_manager()
    );

    // Events consuming corresponding edges should have same canonical hash
    EXPECT_EQ(event_hash_1, event_hash_2);

    // Event with different edge should have different hash
    uint64_t event_hash_3 = EventCanonicalizer::compute_canonical_hash(
        0,
        hash1,
        &e1, 1,  // Different edge (not equivalent to e0)
        hash1,
        hg.edge_equiv_manager()
    );

    EXPECT_NE(event_hash_1, event_hash_3);
}

TEST(Unified_Level2, EventCanonicalization_DifferentRules) {
    UnifiedHypergraph hg;
    hg.enable_level2();

    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    SparseBitset state_edges;
    state_edges.set(e0, hg.arena());
    uint64_t hash = hg.compute_canonical_hash(state_edges);

    // Same edge, different rules -> different hash
    uint64_t event_hash_1 = EventCanonicalizer::compute_canonical_hash(
        0, hash, &e0, 1, hash, hg.edge_equiv_manager());

    uint64_t event_hash_2 = EventCanonicalizer::compute_canonical_hash(
        1, hash, &e0, 1, hash, hg.edge_equiv_manager());

    EXPECT_NE(event_hash_1, event_hash_2);
}
