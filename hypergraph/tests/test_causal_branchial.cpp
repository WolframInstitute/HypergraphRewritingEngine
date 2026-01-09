#include <gtest/gtest.h>
#include "hypergraph/arena.hpp"
#include "hypergraph/types.hpp"
#include "hypergraph/bitset.hpp"
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/causal_graph.hpp"
#include <vector>
#include <set>
#include <algorithm>

using namespace hypergraph;

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
    Hypergraph hg;

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
    Hypergraph hg;

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
    Hypergraph hg;

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
    Hypergraph hg;

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
    Hypergraph hg;

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
    Hypergraph hg;
    SparseBitset empty_edges;

    uint64_t hash = hg.compute_canonical_hash(empty_edges);
    EXPECT_EQ(hash, 0u);
}

TEST(Unified_UniquenessTree, CanonicalHash_SingleEdge) {
    Hypergraph hg;

    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    SparseBitset edges;
    edges.set(e0, hg.arena());

    uint64_t hash = hg.compute_canonical_hash(edges);
    EXPECT_NE(hash, 0u);  // Should produce non-zero hash
}

TEST(Unified_UniquenessTree, CanonicalHash_IsomorphicStates) {
    Hypergraph hg;

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
    Hypergraph hg;

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
    Hypergraph hg;

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
    Hypergraph hg;

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

    // Verify canonical hash is computed
    uint64_t hash = hg.compute_canonical_hash(edges);
    EXPECT_NE(hash, 0u);

    // Create an isomorphic star graph with different vertex IDs
    VertexId u0 = hg.alloc_vertex();  // Center
    VertexId u1 = hg.alloc_vertex();  // Leaf
    VertexId u2 = hg.alloc_vertex();  // Leaf
    VertexId u3 = hg.alloc_vertex();  // Leaf

    EdgeId e3 = hg.create_edge({u0, u1});
    EdgeId e4 = hg.create_edge({u0, u2});
    EdgeId e5 = hg.create_edge({u0, u3});

    SparseBitset edges2;
    edges2.set(e3, hg.arena());
    edges2.set(e4, hg.arena());
    edges2.set(e5, hg.arena());

    uint64_t hash2 = hg.compute_canonical_hash(edges2);

    // Isomorphic graphs should have the same canonical hash
    EXPECT_EQ(hash, hash2);
}

// =============================================================================
// Event Canonicalization with Edge Correspondence Tests
// =============================================================================
// These tests verify that event canonicalization correctly identifies equivalent
// events using on-the-fly edge correspondence computation.

TEST(Unified_EventCanonicalization, CorrespondingEdges_SameCanonicalEvent) {
    // Test that events from isomorphic states consuming corresponding edges
    // are identified as the same canonical event.
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);  // Enable state canonicalization
    hg.set_event_signature_keys(EVENT_SIG_FULL);  // Enable full event canonicalization

    // Create first state: single edge {0, 1}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    SparseBitset state1_edges;
    state1_edges.set(e0, hg.arena());
    uint64_t hash1 = hg.compute_canonical_hash(state1_edges);

    auto [canonical1, raw1, was_new1] = hg.create_or_get_canonical_state(
        std::move(state1_edges), hash1, 0, INVALID_ID);
    EXPECT_TRUE(was_new1);

    // Create second state: isomorphic single edge {2, 3}
    VertexId v2 = hg.alloc_vertex();
    VertexId v3 = hg.alloc_vertex();
    EdgeId e1 = hg.create_edge({v2, v3});

    SparseBitset state2_edges;
    state2_edges.set(e1, hg.arena());
    uint64_t hash2 = hg.compute_canonical_hash(state2_edges);

    EXPECT_EQ(hash1, hash2);  // States should be isomorphic

    auto [canonical2, raw2, was_new2] = hg.create_or_get_canonical_state(
        std::move(state2_edges), hash2, 0, INVALID_ID);
    EXPECT_FALSE(was_new2);  // Should find existing canonical state
    EXPECT_EQ(canonical1, canonical2);

    // Create output states (just single edges for simplicity)
    VertexId v4 = hg.alloc_vertex();
    EdgeId out_e0 = hg.create_edge({v1, v4});
    SparseBitset out1_edges;
    out1_edges.set(out_e0, hg.arena());
    uint64_t out_hash1 = hg.compute_canonical_hash(out1_edges);
    auto [out_canonical1, out_raw1, out_new1] = hg.create_or_get_canonical_state(
        std::move(out1_edges), out_hash1, 1, canonical1);

    VertexId v5 = hg.alloc_vertex();
    EdgeId out_e1 = hg.create_edge({v3, v5});
    SparseBitset out2_edges;
    out2_edges.set(out_e1, hg.arena());
    uint64_t out_hash2 = hg.compute_canonical_hash(out2_edges);
    auto [out_canonical2, out_raw2, out_new2] = hg.create_or_get_canonical_state(
        std::move(out2_edges), out_hash2, 1, canonical2);

    EXPECT_EQ(out_canonical1, out_canonical2);  // Output states should also be canonical equivalent

    // Create events from both raw states consuming corresponding edges
    // Event 1: from raw1, consumes e0, produces out_e0
    VariableBinding empty_binding;
    auto result1 = hg.create_event(
        raw1, out_raw1,  // raw input -> raw output
        0,               // rule_index
        &e0, 1,          // consumed edges
        &out_e0, 1,      // produced edges
        empty_binding    // binding
    );

    // Event 2: from raw2, consumes e1 (corresponds to e0), produces out_e1
    auto result2 = hg.create_event(
        raw2, out_raw2,
        0,
        &e1, 1,
        &out_e1, 1,
        empty_binding
    );

    // Both events should have the same canonical event
    // The canonical_event_id should match for both (either both canonical to same, or one points to other)
    EXPECT_EQ(result1.canonical_event_id, result2.canonical_event_id);
}

TEST(Unified_EventCanonicalization, DifferentEdges_DifferentCanonicalEvent) {
    // Test that events consuming non-corresponding edges have different canonical events.
    Hypergraph hg;
    hg.set_event_signature_keys(EVENT_SIG_FULL);

    // Create state with two edges
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});

    SparseBitset state_edges;
    state_edges.set(e0, hg.arena());
    state_edges.set(e1, hg.arena());
    uint64_t hash = hg.compute_canonical_hash(state_edges);

    auto [canonical, raw, was_new] = hg.create_or_get_canonical_state(
        std::move(state_edges), hash, 0, INVALID_ID);

    // Create output states
    VertexId v3 = hg.alloc_vertex();
    EdgeId out_e0 = hg.create_edge({v1, v3});
    SparseBitset out1_edges;
    out1_edges.set(out_e0, hg.arena());
    out1_edges.set(e1, hg.arena());  // Keep e1
    uint64_t out_hash1 = hg.compute_canonical_hash(out1_edges);
    auto [out1, out_raw1, _1] = hg.create_or_get_canonical_state(
        std::move(out1_edges), out_hash1, 1, canonical);

    VertexId v4 = hg.alloc_vertex();
    EdgeId out_e1 = hg.create_edge({v2, v4});
    SparseBitset out2_edges;
    out2_edges.set(e0, hg.arena());  // Keep e0
    out2_edges.set(out_e1, hg.arena());
    uint64_t out_hash2 = hg.compute_canonical_hash(out2_edges);
    auto [out2, out_raw2, _2] = hg.create_or_get_canonical_state(
        std::move(out2_edges), out_hash2, 1, canonical);

    // Event 1: consumes e0
    VariableBinding empty_binding;
    auto result1 = hg.create_event(raw, out_raw1, 0, &e0, 1, &out_e0, 1, empty_binding);

    // Event 2: consumes e1 (NOT corresponding to e0)
    auto result2 = hg.create_event(raw, out_raw2, 0, &e1, 1, &out_e1, 1, empty_binding);

    // Both events should be canonical (different signatures - consuming different edges)
    // So they should NOT share the same canonical event
    EXPECT_NE(result1.canonical_event_id, result2.canonical_event_id);
    EXPECT_TRUE(result1.is_canonical);
    EXPECT_TRUE(result2.is_canonical);
}

TEST(Unified_EventCanonicalization, DifferentRules_DifferentCanonicalEvent) {
    // Test that events with different rules have different canonical events.
    Hypergraph hg;
    // Include Rule in signature so different rules produce different canonical events
    hg.set_event_signature_keys(EVENT_SIG_FULL | EventKey_Rule);

    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    SparseBitset state_edges;
    state_edges.set(e0, hg.arena());
    uint64_t hash = hg.compute_canonical_hash(state_edges);
    auto [canonical, raw, _] = hg.create_or_get_canonical_state(
        std::move(state_edges), hash, 0, INVALID_ID);

    // Create output state
    VertexId v2 = hg.alloc_vertex();
    EdgeId out_e = hg.create_edge({v1, v2});
    SparseBitset out_edges;
    out_edges.set(out_e, hg.arena());
    uint64_t out_hash = hg.compute_canonical_hash(out_edges);
    auto [out_canonical, out_raw, __] = hg.create_or_get_canonical_state(
        std::move(out_edges), out_hash, 1, canonical);

    // Event 1: rule 0
    VariableBinding empty_binding;
    auto result1 = hg.create_event(raw, out_raw, 0, &e0, 1, &out_e, 1, empty_binding);

    // Event 2: rule 1 (same edges, different rule)
    auto result2 = hg.create_event(raw, out_raw, 1, &e0, 1, &out_e, 1, empty_binding);

    // Both should be canonical (different rule indices)
    EXPECT_TRUE(result1.is_canonical);
    EXPECT_TRUE(result2.is_canonical);
}

TEST(Unified_EventCanonicalization, NoSignatureKeys_AllCanonical) {
    // Test that with EVENT_SIG_NONE, all events are canonical.
    Hypergraph hg;
    hg.set_event_signature_keys(EVENT_SIG_NONE);  // No canonicalization

    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});

    SparseBitset state_edges;
    state_edges.set(e0, hg.arena());
    uint64_t hash = hg.compute_canonical_hash(state_edges);
    auto [canonical, raw, _] = hg.create_or_get_canonical_state(
        std::move(state_edges), hash, 0, INVALID_ID);

    VertexId v2 = hg.alloc_vertex();
    EdgeId out_e = hg.create_edge({v1, v2});
    SparseBitset out_edges;
    out_edges.set(out_e, hg.arena());
    uint64_t out_hash = hg.compute_canonical_hash(out_edges);
    auto [out_canonical, out_raw, __] = hg.create_or_get_canonical_state(
        std::move(out_edges), out_hash, 1, canonical);

    // Create two identical events
    VariableBinding empty_binding;
    auto result1 = hg.create_event(raw, out_raw, 0, &e0, 1, &out_e, 1, empty_binding);
    auto result2 = hg.create_event(raw, out_raw, 0, &e0, 1, &out_e, 1, empty_binding);

    // Both should be canonical since canonicalization is disabled
    EXPECT_TRUE(result1.is_canonical);
    EXPECT_TRUE(result2.is_canonical);
}

// =============================================================================
// Online Transitive Reduction Tests (Goranci Algorithm)
// =============================================================================

TEST(Unified_CausalGraph, OnlineTransitiveReduction_Basic) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Without TR enabled, all edges should be stored
    cg.set_transitive_reduction(false);

    // Create a chain: event 0 -> event 1 -> event 2
    // Edge 0 from event 0 to event 1
    cg.set_edge_producer(0, 0);  // Event 0 produces edge 0
    cg.add_edge_consumer(0, 1);  // Event 1 consumes edge 0

    // Edge 1 from event 1 to event 2
    cg.set_edge_producer(1, 1);  // Event 1 produces edge 1
    cg.add_edge_consumer(1, 2);  // Event 2 consumes edge 1

    // Without TR: 2 causal edges (0->1, 1->2)
    EXPECT_EQ(cg.num_causal_edges(), 2u);
    EXPECT_EQ(cg.num_redundant_edges_skipped(), 0u);
}

TEST(Unified_CausalGraph, OnlineTransitiveReduction_SkipsRedundant) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Enable TR
    cg.set_transitive_reduction(true);

    // Create a chain: event 0 -> event 1 -> event 2
    // Edge 0 from event 0 to event 1
    cg.set_edge_producer(0, 0);
    cg.add_edge_consumer(0, 1);

    // Edge 1 from event 1 to event 2
    cg.set_edge_producer(1, 1);
    cg.add_edge_consumer(1, 2);

    // Now we have: 0 -> 1 -> 2
    EXPECT_EQ(cg.num_causal_edges(), 2u);

    // Try to add redundant edge: 0 -> 2 (already reachable via 0 -> 1 -> 2)
    cg.add_causal_edge(0, 2, 100);  // Edge 100 is just a unique identifier

    // The redundant edge should be skipped
    EXPECT_EQ(cg.num_causal_edges(), 2u);  // Still 2
    EXPECT_EQ(cg.num_redundant_edges_skipped(), 1u);  // 1 skipped
}

TEST(Unified_CausalGraph, OnlineTransitiveReduction_DirectEdgeStillAdded) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Enable TR
    cg.set_transitive_reduction(true);

    // First add direct edge 0 -> 2
    cg.add_causal_edge(0, 2, 0);
    EXPECT_EQ(cg.num_causal_edges(), 1u);

    // Then add edges that form a path 0 -> 1 -> 2
    cg.add_causal_edge(0, 1, 1);
    cg.add_causal_edge(1, 2, 2);

    // All 3 edges should be stored because they were added before the path existed
    // TR only skips edges where the target is ALREADY reachable
    EXPECT_EQ(cg.num_causal_edges(), 3u);
}

TEST(Unified_CausalGraph, OnlineTransitiveReduction_LongerPath) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Enable TR
    cg.set_transitive_reduction(true);

    // Create a longer chain: 0 -> 1 -> 2 -> 3 -> 4
    for (EventId i = 0; i < 4; ++i) {
        cg.add_causal_edge(i, i + 1, i);
    }
    EXPECT_EQ(cg.num_causal_edges(), 4u);

    // Try to add skip edges: 0 -> 2, 0 -> 3, 0 -> 4, 1 -> 3, 1 -> 4, 2 -> 4
    size_t skipped_before = cg.num_redundant_edges_skipped();

    cg.add_causal_edge(0, 2, 100);
    cg.add_causal_edge(0, 3, 101);
    cg.add_causal_edge(0, 4, 102);
    cg.add_causal_edge(1, 3, 103);
    cg.add_causal_edge(1, 4, 104);
    cg.add_causal_edge(2, 4, 105);

    // All should be skipped as redundant
    EXPECT_EQ(cg.num_causal_edges(), 4u);  // Still 4
    EXPECT_EQ(cg.num_redundant_edges_skipped() - skipped_before, 6u);
}

TEST(Unified_CausalGraph, OnlineTransitiveReduction_DiamondPattern) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // Enable TR
    cg.set_transitive_reduction(true);

    // Create diamond: 0 -> {1, 2} -> 3
    //      0
    //     / \
    //    1   2
    //     \ /
    //      3
    cg.add_causal_edge(0, 1, 0);
    cg.add_causal_edge(0, 2, 1);
    cg.add_causal_edge(1, 3, 2);
    cg.add_causal_edge(2, 3, 3);

    EXPECT_EQ(cg.num_causal_edges(), 4u);

    // Try to add 0 -> 3 (redundant via either path)
    cg.add_causal_edge(0, 3, 100);
    EXPECT_EQ(cg.num_causal_edges(), 4u);  // Still 4
    EXPECT_EQ(cg.num_redundant_edges_skipped(), 1u);
}

TEST(Unified_CausalGraph, OnlineTransitiveReduction_DisabledByDefault) {
    ConcurrentHeterogeneousArena arena;
    CausalGraph cg(&arena);

    // TR should be disabled by default
    EXPECT_FALSE(cg.transitive_reduction_enabled());

    // Create chain and redundant edge
    cg.add_causal_edge(0, 1, 0);
    cg.add_causal_edge(1, 2, 1);
    cg.add_causal_edge(0, 2, 2);  // Would be redundant with TR

    // Without TR, all 3 edges stored
    EXPECT_EQ(cg.num_causal_edges(), 3u);
    EXPECT_EQ(cg.num_redundant_edges_skipped(), 0u);
}
