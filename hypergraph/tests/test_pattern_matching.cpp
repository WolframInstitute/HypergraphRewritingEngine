#include <gtest/gtest.h>
#include "hypergraph/arena.hpp"
#include "hypergraph/types.hpp"
#include "hypergraph/bitset.hpp"
#include "hypergraph/signature.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/ancestry.hpp"
#include "hypergraph/pattern_matcher.hpp"
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/rewriter.hpp"
#include <vector>
#include <set>

using namespace hypergraph;

// =============================================================================
// Test Helpers
// =============================================================================

// A hypergraph whose edges all belong to one root state, with the state's edge set and its
// ancestry candidate source for the matcher.
class TestHypergraph {
public:
    Hypergraph hg;
    std::vector<EdgeId> ids;

    EdgeId add_edge(std::initializer_list<VertexId> verts) {
        EdgeId eid = hg.create_edge(verts);
        ids.push_back(eid);
        return eid;
    }

    // The root state over every edge added so far; created once, on first use.
    StateId state() {
        if (state_ == INVALID_ID) state_ = hg.create_state(ids.data(), static_cast<uint32_t>(ids.size()));
        return state_;
    }
    const SparseBitset& edges() { return hg.get_state(state()).edges; }
    AncestryCandidates candidates() { return AncestryCandidates{&hg, state(), &edges()}; }
    const Edge& edge(EdgeId eid) const { return hg.get_edge(eid); }

    auto get_edge_accessor() {
        return [this](EdgeId eid) -> const Edge& { return hg.get_edge(eid); };
    }

private:
    StateId state_ = INVALID_ID;
};

// =============================================================================
// Signature Tests
// =============================================================================

TEST(Signature, EdgeSignature_AllDifferent) {
    VertexId verts[] = {5, 6, 8};
    EdgeSignature sig = EdgeSignature::from_edge(verts, 3);
    EXPECT_EQ(sig.arity, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 1);
    EXPECT_EQ(sig.pattern[2], 2);
}

TEST(Signature, EdgeSignature_FirstTwoSame) {
    VertexId verts[] = {3, 3, 4};
    EdgeSignature sig = EdgeSignature::from_edge(verts, 3);
    EXPECT_EQ(sig.arity, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 0);
    EXPECT_EQ(sig.pattern[2], 1);
}

TEST(Signature, EdgeSignature_AllSame) {
    VertexId verts[] = {1, 1, 1};
    EdgeSignature sig = EdgeSignature::from_edge(verts, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 0);
    EXPECT_EQ(sig.pattern[2], 0);
}

TEST(Signature, EdgeSignature_FirstLastSame) {
    VertexId verts[] = {7, 8, 7};
    EdgeSignature sig = EdgeSignature::from_edge(verts, 3);
    EXPECT_EQ(sig.pattern[0], 0);
    EXPECT_EQ(sig.pattern[1], 1);
    EXPECT_EQ(sig.pattern[2], 0);
}

TEST(Signature, Compatibility_DistinctVarsMatch) {
    // Pattern [0,1] (distinct vars) matches both [0,0] and [0,1]
    uint8_t pattern_vars[] = {0, 1};
    EdgeSignature pattern_sig = EdgeSignature::from_pattern(pattern_vars, 2);

    VertexId data1[] = {5, 5};  // [0,0]
    EdgeSignature data_sig1 = EdgeSignature::from_edge(data1, 2);
    EXPECT_TRUE(signature_compatible(data_sig1, pattern_sig));

    VertexId data2[] = {5, 6};  // [0,1]
    EdgeSignature data_sig2 = EdgeSignature::from_edge(data2, 2);
    EXPECT_TRUE(signature_compatible(data_sig2, pattern_sig));
}

TEST(Signature, Compatibility_SameVarConstraint) {
    // Pattern [0,0] (same var) only matches [0,0]
    uint8_t pattern_vars[] = {0, 0};
    EdgeSignature pattern_sig = EdgeSignature::from_pattern(pattern_vars, 2);

    VertexId data1[] = {5, 5};  // [0,0]
    EdgeSignature data_sig1 = EdgeSignature::from_edge(data1, 2);
    EXPECT_TRUE(signature_compatible(data_sig1, pattern_sig));

    VertexId data2[] = {5, 6};  // [0,1]
    EdgeSignature data_sig2 = EdgeSignature::from_edge(data2, 2);
    EXPECT_FALSE(signature_compatible(data_sig2, pattern_sig));
}

// =============================================================================
// Pattern Matching Tests
// =============================================================================

TEST(PatternMatching, SingleEdgePattern) {
    TestHypergraph hg;

    // Add edges: {0,1}, {1,2}, {2,0}
    hg.add_edge({0, 1});
    hg.add_edge({1, 2});
    hg.add_edge({2, 0});

    // Pattern: {x, y} - should match all 3 edges
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})  // {x, y}
        .rhs({1, 0})  // {y, x}
        .build();

    std::vector<std::pair<EdgeId, VariableBinding>> matches;

    find_matches(
        rule, 0, 0,
        hg.edges(),
        hg.candidates(),
        hg.get_edge_accessor(),
        [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding& binding, StateId) {
            matches.push_back({edges[0], binding});
        }
    );

    EXPECT_EQ(matches.size(), 3);

    // Check bindings
    for (const auto& [eid, binding] : matches) {
        const auto& edge = hg.edge(eid);
        EXPECT_EQ(binding.get(0), edge.vertices[0]);
        EXPECT_EQ(binding.get(1), edge.vertices[1]);
    }
}

TEST(PatternMatching, SelfLoopPattern) {
    TestHypergraph hg;

    // Add edges: {0,0}, {1,1}, {0,1}
    hg.add_edge({0, 0});  // Self-loop
    hg.add_edge({1, 1});  // Self-loop
    hg.add_edge({0, 1});  // Not a self-loop

    // Pattern: {x, x} - should match only self-loops
    RewriteRule rule = make_rule(0)
        .lhs({0, 0})  // {x, x}
        .rhs({0})     // {x}
        .build();

    std::vector<EdgeId> matches;

    find_matches(
        rule, 0, 0,
        hg.edges(),
        hg.candidates(),
        hg.get_edge_accessor(),
        [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding&, StateId) {
            matches.push_back(edges[0]);
        }
    );

    EXPECT_EQ(matches.size(), 2);

    // Verify they are the self-loops
    for (EdgeId eid : matches) {
        const auto& edge = hg.edge(eid);
        EXPECT_EQ(edge.vertices[0], edge.vertices[1]);
    }
}

TEST(PatternMatching, TwoEdgePattern) {
    TestHypergraph hg;

    // Triangle: {0,1}, {1,2}, {2,0}
    hg.add_edge({0, 1});
    hg.add_edge({1, 2});
    hg.add_edge({2, 0});

    // Pattern: {x, y}, {y, z} - find connected pairs
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .rhs({0, 2})  // {x, z}
        .build();

    std::vector<std::pair<EdgeId, EdgeId>> matches;

    find_matches(
        rule, 0, 0,
        hg.edges(),
        hg.candidates(),
        hg.get_edge_accessor(),
        [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding&, StateId) {
            matches.push_back({edges[0], edges[1]});
        }
    );

    // Each edge can be first, second edge must share a vertex
    // {0,1} can pair with {1,2} (y=1) → match
    // {1,2} can pair with {2,0} (y=2) → match
    // {2,0} can pair with {0,1} (y=0) → match
    EXPECT_EQ(matches.size(), 3);

    // Verify connectivity
    for (const auto& [e1, e2] : matches) {
        const auto& edge1 = hg.edge(e1);
        const auto& edge2 = hg.edge(e2);
        // y is at position 1 of first edge and position 0 of second edge
        EXPECT_EQ(edge1.vertices[1], edge2.vertices[0]);
    }
}

TEST(PatternMatching, NoDuplicateEdgeUse) {
    TestHypergraph hg;

    // Single edge: {0, 1}
    hg.add_edge({0, 1});

    // Pattern: {x, y}, {y, z} - needs TWO edges
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .rhs({0, 2})  // {x, z}
        .build();

    size_t match_count = 0;

    find_matches(
        rule, 0, 0,
        hg.edges(),
        hg.candidates(),
        hg.get_edge_accessor(),
        [&](uint16_t, const EdgeId*, uint8_t, const VariableBinding&, StateId) {
            match_count++;
        }
    );

    // Should find NO matches - can't use same edge twice
    EXPECT_EQ(match_count, 0);
}

TEST(PatternMatching, VariableBindingConsistency) {
    TestHypergraph hg;

    // Edges: {0,1}, {1,0}, {2,3}
    hg.add_edge({0, 1});
    hg.add_edge({1, 0});
    hg.add_edge({2, 3});

    // Pattern: {x, y}, {y, x} - edges must have swapped endpoints
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 0})  // {y, x}
        .rhs({0})     // {x}
        .build();

    std::vector<std::pair<EdgeId, EdgeId>> matches;

    find_matches(
        rule, 0, 0,
        hg.edges(),
        hg.candidates(),
        hg.get_edge_accessor(),
        [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding&, StateId) {
            matches.push_back({edges[0], edges[1]});
        }
    );

    // {0,1} and {1,0} form a match (x=0, y=1)
    // {1,0} and {0,1} form a match (x=1, y=0)
    // {2,3} doesn't pair with anything
    EXPECT_EQ(matches.size(), 2);

    // Verify binding consistency
    for (const auto& [e1, e2] : matches) {
        const auto& edge1 = hg.edge(e1);
        const auto& edge2 = hg.edge(e2);
        // First edge: {x, y}, Second edge: {y, x}
        EXPECT_EQ(edge1.vertices[0], edge2.vertices[1]);  // x
        EXPECT_EQ(edge1.vertices[1], edge2.vertices[0]);  // y
    }
}

TEST(PatternMatching, TernaryEdges) {
    TestHypergraph hg;

    // Ternary edges: {0,1,2}, {1,2,3}, {0,0,0}
    hg.add_edge({0, 1, 2});
    hg.add_edge({1, 2, 3});
    hg.add_edge({0, 0, 0});

    // Pattern: {x, y, z} - all different vars
    RewriteRule rule = make_rule(0)
        .lhs({0, 1, 2})  // {x, y, z}
        .rhs({2, 1, 0})  // {z, y, x}
        .build();

    size_t match_count = 0;

    find_matches(
        rule, 0, 0,
        hg.edges(),
        hg.candidates(),
        hg.get_edge_accessor(),
        [&](uint16_t, const EdgeId*, uint8_t, const VariableBinding&, StateId) {
            match_count++;
        }
    );

    // All 3 edges match {x,y,z} (non-distinct vars)
    EXPECT_EQ(match_count, 3);
}

// =============================================================================
// Candidate Tests
// =============================================================================

// The candidate source is the state's ancestry: a root state's own edges through its vertex
// index, and every derived state's edges through its chain of parent events (ancestry.hpp).
TEST(AncestryCandidates, RootEdgesBySignature) {
    Hypergraph hg;
    EdgeId e0 = hg.create_edge({0, 1});   // [0,1]
    EdgeId e1 = hg.create_edge({2, 2});   // [0,0]
    EdgeId e2 = hg.create_edge({3, 4});   // [0,1]
    EdgeId e3 = hg.create_edge({5, 5});   // [0,0]
    StateId s0 = hg.create_state({e0, e1, e2, e3});
    const State& st = hg.get_state(s0);
    AncestryCandidates cands{&hg, s0, &st.edges};
    auto get_signature = [&](EdgeId eid) { return hg.edge_signature(eid); };

    // [0,0] is compatible with the self-loops only; [0,1] with every edge.
    uint8_t same[] = {0, 0};
    std::set<EdgeId> found;
    cands.for_each_edge_compatible(EdgeSignature::from_pattern(same, 2), get_signature,
                                   [&](EdgeId eid) { found.insert(eid); });
    EXPECT_EQ(found, (std::set<EdgeId>{e1, e3}));

    uint8_t distinct[] = {0, 1};
    found.clear();
    cands.for_each_edge_compatible(EdgeSignature::from_pattern(distinct, 2), get_signature,
                                   [&](EdgeId eid) { found.insert(eid); });
    EXPECT_EQ(found, (std::set<EdgeId>{e0, e1, e2, e3}));
}

TEST(AncestryCandidates, RootEdgesByVertex) {
    Hypergraph hg;
    EdgeId e0 = hg.create_edge({0, 1});
    EdgeId e1 = hg.create_edge({1, 2});
    EdgeId e2 = hg.create_edge({2, 0});
    EdgeId e3 = hg.create_edge({3, 4});
    StateId s0 = hg.create_state({e0, e1, e2, e3});
    const State& st = hg.get_state(s0);
    AncestryCandidates cands{&hg, s0, &st.edges};

    std::set<EdgeId> at1;
    cands.for_each_edge_at(1, [&](EdgeId eid, const Edge&) { at1.insert(eid); });
    EXPECT_EQ(at1, (std::set<EdgeId>{e0, e1}));

    VertexId both[] = {0, 1};
    std::vector<EdgeId> containing;
    cands.for_each_edge_containing_all(both, 2, [&](EdgeId eid, const Edge&) { containing.push_back(eid); });
    ASSERT_EQ(containing.size(), 1u);
    EXPECT_EQ(containing[0], e0);

    std::set<EdgeId> at9;
    cands.for_each_edge_at(9, [&](EdgeId eid, const Edge&) { at9.insert(eid); });
    EXPECT_TRUE(at9.empty());
}

// A derived state reaches its parent's surviving edges and its own produced edges through
// the chain, and an edge its event consumed is not yielded.
TEST(AncestryCandidates, DerivedStateWalksItsChain) {
    Hypergraph hg;
    EdgeId e0 = hg.create_edge({0, 1});
    EdgeId e1 = hg.create_edge({1, 2});
    StateId s0 = hg.create_state({e0, e1});

    // The event consumes e0 and produces e2 = (0,3), e3 = (3,1).
    EdgeId e2 = hg.create_edge({0, 3});
    EdgeId e3 = hg.create_edge({3, 1});
    EdgeId consumed[] = {e0};
    EdgeId produced[] = {e2, e3};
    SparseBitset child_edges;
    for (EdgeId e : {e1, e2, e3}) child_edges.set(e, hg.arena());
    StateId s1 = hg.create_state(std::move(child_edges), 1, 0, INVALID_ID, s0, produced, 2);
    hg.create_event(s0, s1, 0, consumed, 1, produced, 2);
    const State& st1 = hg.get_state(s1);
    AncestryCandidates cands{&hg, s1, &st1.edges};

    std::set<EdgeId> at1;
    cands.for_each_edge_at(1, [&](EdgeId eid, const Edge&) { at1.insert(eid); });
    EXPECT_EQ(at1, (std::set<EdgeId>{e1, e3}));   // e0 was consumed

    std::set<EdgeId> at3;
    cands.for_each_edge_at(3, [&](EdgeId eid, const Edge&) { at3.insert(eid); });
    EXPECT_EQ(at3, (std::set<EdgeId>{e2, e3}));

    std::set<EdgeId> at0;
    cands.for_each_edge_at(0, [&](EdgeId eid, const Edge&) { at0.insert(eid); });
    EXPECT_EQ(at0, (std::set<EdgeId>{e2}));
}

// =============================================================================
// Hypergraph Tests
// =============================================================================

TEST(HypergraphCore, CreateEdgesAndState) {
    Hypergraph hg;

    // Create edges
    EdgeId e0 = hg.create_edge({0, 1});
    EdgeId e1 = hg.create_edge({1, 2});
    EdgeId e2 = hg.create_edge({2, 0});

    EXPECT_EQ(e0, 0);
    EXPECT_EQ(e1, 1);
    EXPECT_EQ(e2, 2);

    // Verify edge data
    const Edge& edge0 = hg.get_edge(e0);
    EXPECT_EQ(edge0.arity, 2);
    EXPECT_EQ(edge0.vertices[0], 0);
    EXPECT_EQ(edge0.vertices[1], 1);

    // Create state
    StateId s0 = hg.create_state({e0, e1, e2});
    EXPECT_EQ(s0, 0);

    // Verify state
    const State& state = hg.get_state(s0);
    EXPECT_TRUE(state.edges.contains(e0));
    EXPECT_TRUE(state.edges.contains(e1));
    EXPECT_TRUE(state.edges.contains(e2));

    // Verify counts
    EXPECT_EQ(hg.num_edges(), 3);
    EXPECT_EQ(hg.num_states(), 1);
}

TEST(HypergraphCore, VertexAllocation) {
    Hypergraph hg;

    // Allocate vertices
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();

    EXPECT_EQ(v0, 0);
    EXPECT_EQ(v1, 1);
    EXPECT_EQ(v2, 2);

    // Allocate batch
    VertexId batch_start = hg.alloc_vertices(5);
    EXPECT_EQ(batch_start, 3);
    EXPECT_EQ(hg.num_vertices(), 8);
}

TEST(HypergraphCore, PatternMatchingOnHypergraph) {
    Hypergraph hg;

    // Create triangle: {0,1}, {1,2}, {2,0}
    EdgeId e0 = hg.create_edge({0, 1});
    EdgeId e1 = hg.create_edge({1, 2});
    EdgeId e2 = hg.create_edge({2, 0});

    StateId s0 = hg.create_state({e0, e1, e2});

    // Pattern: {x, y} - should match all 3 edges
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})
        .rhs({1, 0})
        .build();

    std::vector<EdgeId> matches;
    const State& state = hg.get_state(s0);

    find_matches(
        rule, 0, s0,
        state.edges,
        AncestryCandidates{&hg, s0, &state.edges},
        hg.edge_accessor(),
        [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding&, StateId) {
            matches.push_back(edges[0]);
        }
    );

    EXPECT_EQ(matches.size(), 3);
}

// =============================================================================
// Rewriter Tests
// =============================================================================

TEST(Rewriter, SimpleRewrite) {
    Hypergraph hg;

    // Create initial state: {0, 1}, {1, 2}
    EdgeId e0 = hg.create_edge({0, 1});
    EdgeId e1 = hg.create_edge({1, 2});
    StateId s0 = hg.create_state({e0, e1});

    // Rule: {x, y}, {y, z} → {x, z}
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})  // {x, y}
        .lhs({1, 2})  // {y, z}
        .rhs({0, 2})  // {x, z}
        .build();

    // Create binding: x=0, y=1, z=2
    VariableBinding binding;
    binding.bind(0, 0);  // x=0
    binding.bind(1, 1);  // y=1
    binding.bind(2, 2);  // z=2

    // Apply rewrite
    EdgeId matched_edges[] = {e0, e1};
    RewriteResult result = apply_rewrite(hg, rule, s0, matched_edges, 2, binding, 1);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.num_produced, 1);

    // Verify new state
    const State& new_state = hg.get_state(result.new_state);

    // New state should NOT contain consumed edges
    EXPECT_FALSE(new_state.edges.contains(e0));
    EXPECT_FALSE(new_state.edges.contains(e1));

    // New state should contain produced edge
    EXPECT_TRUE(new_state.edges.contains(result.produced_edges[0]));

    // Verify produced edge is {0, 2}
    const Edge& produced = hg.get_edge(result.produced_edges[0]);
    EXPECT_EQ(produced.arity, 2);
    EXPECT_EQ(produced.vertices[0], 0);
    EXPECT_EQ(produced.vertices[1], 2);
}

TEST(Rewriter, RewriteWithFreshVertex) {
    Hypergraph hg;

    // Reserve vertices 0, 1, 2
    hg.reserve_vertices(2);

    // Create initial state: {0, 1}
    EdgeId e0 = hg.create_edge({0, 1});
    StateId s0 = hg.create_state({e0});

    // Rule: {x, y} → {x, y}, {y, w}
    // w is a fresh variable (not in LHS)
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})     // {x, y}
        .rhs({0, 1})     // {x, y}
        .rhs({1, 2})     // {y, w} - w is var 2, fresh
        .build();

    // Create binding: x=0, y=1
    VariableBinding binding;
    binding.bind(0, 0);  // x=0
    binding.bind(1, 1);  // y=1

    // Apply rewrite
    EdgeId matched_edges[] = {e0};
    RewriteResult result = apply_rewrite(hg, rule, s0, matched_edges, 1, binding, 1);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.num_produced, 2);

    // Verify produced edges
    const Edge& prod0 = hg.get_edge(result.produced_edges[0]);
    const Edge& prod1 = hg.get_edge(result.produced_edges[1]);

    // First produced edge should be {0, 1} (same as consumed)
    EXPECT_EQ(prod0.vertices[0], 0);
    EXPECT_EQ(prod0.vertices[1], 1);

    // Second produced edge should be {1, w} where w is fresh (>=3)
    EXPECT_EQ(prod1.vertices[0], 1);
    EXPECT_GE(prod1.vertices[1], 3);  // Fresh vertex
}

TEST(Rewriter, EventCreation) {
    Hypergraph hg;

    // Create initial state
    EdgeId e0 = hg.create_edge({0, 1});
    EdgeId e1 = hg.create_edge({1, 2});
    StateId s0 = hg.create_state({e0, e1});

    // Rule: {x, y} → {y, x}
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})
        .rhs({1, 0})
        .build();

    VariableBinding binding;
    binding.bind(0, 0);
    binding.bind(1, 1);

    EdgeId matched_edges[] = {e0};
    RewriteResult result = apply_rewrite(hg, rule, s0, matched_edges, 1, binding, 1);

    EXPECT_TRUE(result.success);
    EXPECT_NE(result.event, INVALID_ID);

    // Verify event
    const Event& event = hg.get_event(result.event);
    EXPECT_EQ(event.input_state, s0);
    EXPECT_EQ(event.output_state, result.new_state);
    EXPECT_EQ(event.rule_index, 0);
    EXPECT_EQ(event.num_consumed, 1);
    EXPECT_EQ(event.num_produced, 1);
    EXPECT_EQ(event.consumed_edges[0], e0);
    EXPECT_EQ(event.produced_edges[0], result.produced_edges[0]);
}

// =============================================================================
// End-to-End Integration Tests
// =============================================================================

TEST(EngineIntegration, SingleStepEvolution) {
    Hypergraph hg;
    // Enable canonicalization for this test
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_event_signature_keys(EVENT_SIG_FULL);

    // Initial state: triangle {0,1}, {1,2}, {2,0}
    EdgeId e0 = hg.create_edge({0, 1});
    EdgeId e1 = hg.create_edge({1, 2});
    EdgeId e2 = hg.create_edge({2, 0});

    // Create initial state and register in canonical map
    SparseBitset initial_edges;
    initial_edges.set(e0, hg.arena());
    initial_edges.set(e1, hg.arena());
    initial_edges.set(e2, hg.arena());
    uint64_t initial_hash = hg.compute_canonical_hash(initial_edges);
    auto [initial, raw_initial, was_new] = hg.create_or_get_canonical_state(
        std::move(initial_edges), 0, INVALID_ID);

    // Rule: {x, y}, {y, z} → {x, z}
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 2})
        .build();

    Rewriter rewriter(&hg);
    const State& state = hg.get_state(initial);

    // Find all matches
    std::vector<std::tuple<EdgeId, EdgeId, VariableBinding>> matches;
    find_matches(
        rule, 0, initial,
        state.edges,
        AncestryCandidates{&hg, initial, &state.edges},
        hg.edge_accessor(),
        [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding& binding, StateId) {
            matches.push_back({edges[0], edges[1], binding});
        }
    );

    // Should find 3 matches (one for each starting edge)
    EXPECT_EQ(matches.size(), 3);

    // Apply all matches to create new states
    std::set<StateId> new_states;
    size_t events_created = 0;
    for (const auto& [matched_a, matched_b, binding] : matches) {
        EdgeId matched[] = {matched_a, matched_b};
        RewriteResult result = rewriter.apply(rule, initial, matched, 2, binding, 1);
        EXPECT_TRUE(result.success);
        new_states.insert(result.new_state);
        events_created++;
    }

    // With canonical state deduplication, all 3 matches produce the same canonical state:
    // Each match contracts an edge of the triangle, resulting in a 2-edge state
    // that is isomorphic to all other contracted states (two directed edges between two vertices)
    EXPECT_EQ(new_states.size(), 1);  // All 3 results are canonically equivalent

    // Each new state should have 2 edges (removed 2, added 1)
    for (StateId sid : new_states) {
        uint32_t edge_count = hg.count_state_edges(sid);
        EXPECT_EQ(edge_count, 2);
    }

    // Total canonical states: 1 initial + 1 unique evolved state (canonical deduplication)
    // Note: num_states() returns ALL states including "wasted" raw states
    // num_canonical_states() returns only unique canonical representatives
    EXPECT_EQ(hg.num_canonical_states(), 2);
    // With ByState event canonicalization (default), events from same canonical input
    // to same canonical output are deduplicated. All 3 matches go from initial state
    // to the same canonical output state, so only 1 canonical event is created.
    EXPECT_EQ(hg.num_events(), 1);
}

TEST(EngineIntegration, MultiStepEvolution) {
    Hypergraph hg;

    // Initial state: single edge {0, 1}
    EdgeId e0 = hg.create_edge({0, 1});
    StateId s0 = hg.create_state({e0});

    // Growth rule: {x, y} → {x, y}, {y, z} (z is fresh)
    RewriteRule rule = make_rule(0)
        .lhs({0, 1})     // {x, y}
        .rhs({0, 1})     // {x, y}
        .rhs({1, 2})     // {y, z}
        .build();

    Rewriter rewriter(&hg);

    // Track states at each step
    std::vector<StateId> current_states = {s0};
    std::vector<StateId> all_states = {s0};

    // Run 3 steps of evolution
    for (int step = 0; step < 3; ++step) {
        std::vector<StateId> next_states;

        for (StateId sid : current_states) {
            const State& state = hg.get_state(sid);

            // Find matches
            std::vector<std::pair<EdgeId, VariableBinding>> matches;
            find_matches(
                rule, 0, sid,
                state.edges,
                AncestryCandidates{&hg, sid, &state.edges},
                hg.edge_accessor(),
                [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding& binding, StateId) {
                    matches.push_back({edges[0], binding});
                }
            );

            // Apply each match
            for (const auto& [matched_edge, binding] : matches) {
                EdgeId matched[] = {matched_edge};
                RewriteResult result = rewriter.apply(rule, sid, matched, 1, binding, step + 1);
                if (result.success) {
                    next_states.push_back(result.new_state);
                    all_states.push_back(result.new_state);
                }
            }
        }

        current_states = std::move(next_states);
    }

    // After step 0: 1 edge → rule applies → 2 edges. 1 state created.
    // After step 1: 2 edges → rule applies twice → 2 states, each with 3 edges
    // After step 2: each of 2 states has 3 edges → 3 matches each → 6 states

    // Verify growth
    EXPECT_GT(hg.num_states(), 1);
    EXPECT_GT(hg.num_events(), 0);

    // Verify edges grew
    // Final states should have more edges than initial
    for (StateId sid : current_states) {
        uint32_t edge_count = hg.count_state_edges(sid);
        EXPECT_GT(edge_count, 1);  // More than initial
    }
}

TEST(EngineIntegration, SelfLoopEvolution) {
    Hypergraph hg;

    // Initial state with self-loop and regular edge: {0,0}, {0,1}
    EdgeId e0 = hg.create_edge({0, 0});  // Self-loop
    EdgeId e1 = hg.create_edge({0, 1});
    StateId s0 = hg.create_state({e0, e1});

    // Rule: {x, x} → {x} (remove self-loops, create unary edge)
    RewriteRule rule = make_rule(0)
        .lhs({0, 0})  // {x, x}
        .rhs({0})     // {x}
        .build();

    // Find matches
    const State& state = hg.get_state(s0);
    std::vector<std::pair<EdgeId, VariableBinding>> matches;
    find_matches(
        rule, 0, s0,
        state.edges,
        AncestryCandidates{&hg, s0, &state.edges},
        hg.edge_accessor(),
        [&](uint16_t, const EdgeId* edges, uint8_t, const VariableBinding& binding, StateId) {
            matches.push_back({edges[0], binding});
        }
    );

    // Should find only 1 match (the self-loop)
    EXPECT_EQ(matches.size(), 1);

    // Apply the match
    Rewriter rewriter(&hg);
    EdgeId matched[] = {matches[0].first};
    RewriteResult result = rewriter.apply(rule, s0, matched, 1, matches[0].second, 1);

    EXPECT_TRUE(result.success);

    // Verify new state
    const State& new_state = hg.get_state(result.new_state);

    // Should not contain the self-loop
    EXPECT_FALSE(new_state.edges.contains(e0));

    // Should still contain the regular edge
    EXPECT_TRUE(new_state.edges.contains(e1));

    // Should contain the new unary edge
    EXPECT_TRUE(new_state.edges.contains(result.produced_edges[0]));

    // Verify unary edge
    const Edge& unary = hg.get_edge(result.produced_edges[0]);
    EXPECT_EQ(unary.arity, 1);
    EXPECT_EQ(unary.vertices[0], 0);
}

// A high-arity all-distinct pattern edge has more compatible data signatures (Bell(arity)) than
// the CompatibleSignatureCache holds (64). Candidate generation tests each edge's signature
// directly, so every one of the Bell(6) = 203 signature classes is found.
TEST(PatternMatchingBellOverflow, EveryCompatibleSignatureIsFound) {
    const uint8_t arity = 6;
    uint8_t pattern_vars[arity] = {0, 1, 2, 3, 4, 5};
    EdgeSignature pattern_sig = EdgeSignature::from_pattern(pattern_vars, arity);
    CompatibleSignatureCache cache = CompatibleSignatureCache::from_pattern(pattern_sig);
    ASSERT_TRUE(cache.overflowed) << "arity-6 all-distinct should overflow the signature cache";

    std::vector<EdgeSignature> sigs;
    enumerate_compatible_signatures(
        pattern_sig,
        [](const EdgeSignature& s, void* ud) {
            static_cast<std::vector<EdgeSignature>*>(ud)->push_back(s);
        },
        &sigs);
    ASSERT_GT(sigs.size(), static_cast<size_t>(CompatibleSignatureCache::MAX_CACHED_SIGS));

    // One data edge per compatible signature, all in one root state.
    Hypergraph hg;
    std::vector<EdgeId> ids;
    for (size_t i = 0; i < sigs.size(); ++i) {
        VertexId verts[arity];
        VertexId base = static_cast<VertexId>(i) * arity + 1000;
        for (uint8_t p = 0; p < arity; ++p) verts[p] = base + sigs[i].pattern[p];
        ids.push_back(hg.create_edge(verts, arity));
    }
    StateId s0 = hg.create_state(ids.data(), static_cast<uint32_t>(ids.size()));
    const State& st = hg.get_state(s0);
    AncestryCandidates cands{&hg, s0, &st.edges};
    std::set<EdgeId> found;
    cands.for_each_edge_compatible(pattern_sig, [&](EdgeId eid) { return hg.edge_signature(eid); },
                                   [&](EdgeId e) { found.insert(e); });
    EXPECT_EQ(found.size(), sigs.size());
}
