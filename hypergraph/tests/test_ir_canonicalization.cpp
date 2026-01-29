#include <gtest/gtest.h>
#include <hypergraph/ir_canonicalization.hpp>
#include <hypergraph/hypergraph.hpp>
#include <vector>
#include <set>
#include <algorithm>

using namespace hypergraph;

class IRCanonicalizationTest : public ::testing::Test {
protected:
    IRCanonicalizer ir;
};

// =============================================================================
// Basic correctness
// =============================================================================

TEST_F(IRCanonicalizationTest, EmptyHypergraph) {
    std::vector<std::vector<VertexId>> edges;
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 0);
    EXPECT_EQ(result.canonical_form.vertex_count, 0);
}

TEST_F(IRCanonicalizationTest, SingleVertex) {
    std::vector<std::vector<VertexId>> edges = {{5}};
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 1);
    EXPECT_EQ(result.canonical_form.edges[0], std::vector<VertexId>({0}));
    EXPECT_EQ(result.canonical_form.vertex_count, 1);
}

TEST_F(IRCanonicalizationTest, SingleEdge) {
    std::vector<std::vector<VertexId>> edges = {{10, 20}};
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 1);
    EXPECT_EQ(result.canonical_form.vertex_count, 2);
    // Canonical form should have vertices 0 and 1
    ASSERT_EQ(result.canonical_form.edges[0].size(), 2);
    std::set<VertexId> verts(result.canonical_form.edges[0].begin(),
                              result.canonical_form.edges[0].end());
    EXPECT_EQ(verts, (std::set<VertexId>{0, 1}));
}

// =============================================================================
// Isomorphism detection
// =============================================================================

TEST_F(IRCanonicalizationTest, IsomorphicSimple) {
    // {1,2,3} and {4,5,6} are isomorphic (same structure, different labels)
    std::vector<std::vector<VertexId>> edges1 = {{1, 2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{4, 5, 6}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, IsomorphicMultiEdge) {
    // {{1,2},{2,3}} and {{10,20},{20,30}} are isomorphic
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{10, 20}, {20, 30}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, IsomorphicPermuted) {
    // {{1,2},{2,3},{3,1}} and {{5,6},{6,4},{4,5}} (triangle)
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}, {3, 1}};
    std::vector<std::vector<VertexId>> edges2 = {{5, 6}, {6, 4}, {4, 5}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, NonIsomorphicDifferentStructure) {
    // Path vs star
    std::vector<std::vector<VertexId>> path = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> star = {{1, 2}, {1, 3}};
    auto r1 = ir.canonicalize_edges(path);
    auto r2 = ir.canonicalize_edges(star);
    EXPECT_NE(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, DirectedEdgeOrderPreserved) {
    // {{1,2,3}} ≠ {{1,3,2}} since position matters in directed hypergraphs
    std::vector<std::vector<VertexId>> edges1 = {{1, 2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{1, 3, 2}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    // Both have 1 edge with 3 vertices, but internal order differs
    // The position encoding should distinguish them
    // Actually, {1,2,3} and {1,3,2} ARE isomorphic as directed hyperedges
    // because there exists a vertex renaming (swap 2<->3) that maps one to the other.
    // The position encoding ensures position matters within an edge, but
    // swapping vertex labels can swap positions.
    // So {1,2,3} with mapping 1->0,2->1,3->2 = {0,1,2}
    // And {1,3,2} with mapping 1->0,3->1,2->2 = {0,1,2}
    // They ARE isomorphic.
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, DirectedEdgeOrderNonIsomorphic) {
    // {{1,2},{2,1}} vs {{1,2},{1,2}} - first has two edges with opposite directions
    // These should differ
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 1}};
    std::vector<std::vector<VertexId>> edges2 = {{1, 2}, {1, 2}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_NE(r1.canonical_form, r2.canonical_form);
}

// =============================================================================
// Hash consistency
// =============================================================================

TEST_F(IRCanonicalizationTest, HashConsistency) {
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{10, 20}, {20, 30}};
    uint64_t h1 = ir.compute_canonical_hash(edges1);
    uint64_t h2 = ir.compute_canonical_hash(edges2);
    EXPECT_EQ(h1, h2);
    EXPECT_NE(h1, 0u);
}

TEST_F(IRCanonicalizationTest, HashDiffers) {
    std::vector<std::vector<VertexId>> path = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> star = {{1, 2}, {1, 3}};
    uint64_t h1 = ir.compute_canonical_hash(path);
    uint64_t h2 = ir.compute_canonical_hash(star);
    EXPECT_NE(h1, h2);
}

// =============================================================================
// Isomorphism detection on permuted graphs
// =============================================================================

TEST_F(IRCanonicalizationTest, AgreesOnPermutedGraphs) {
    // Each pair should be isomorphic - verify IR detects this
    std::vector<std::pair<std::vector<std::vector<VertexId>>,
                          std::vector<std::vector<VertexId>>>> pairs = {
        {{{1, 2}, {2, 3}, {3, 1}}, {{5, 6}, {6, 4}, {4, 5}}},  // triangle
        {{{1, 2, 3}, {3, 4, 5}}, {{10, 20, 30}, {30, 40, 50}}},  // shared vertex
        {{{1, 2}, {1, 3}, {1, 4}}, {{7, 8}, {7, 9}, {7, 10}}},  // star
    };
    for (size_t i = 0; i < pairs.size(); ++i) {
        auto ir_a = ir.canonicalize_edges(pairs[i].first);
        auto ir_b = ir.canonicalize_edges(pairs[i].second);
        EXPECT_EQ(ir_a.canonical_form, ir_b.canonical_form)
            << "IR failed isomorphism on pair " << i;
    }
}

TEST_F(IRCanonicalizationTest, CanonicalFormConsistency) {
    // Same graph, different vertex labels → same canonical form
    std::vector<std::vector<VertexId>> a = {{3, 7, 1}};
    std::vector<std::vector<VertexId>> b = {{100, 200, 300}};
    auto ra = ir.canonicalize_edges(a);
    auto rb = ir.canonicalize_edges(b);
    EXPECT_EQ(ra.canonical_form, rb.canonical_form);
}

// =============================================================================
// Vertex mapping correctness
// =============================================================================

TEST_F(IRCanonicalizationTest, VertexMappingCorrect) {
    std::vector<std::vector<VertexId>> edges = {{10, 20}, {20, 30}};
    auto result = ir.canonicalize_edges(edges);

    // Verify mapping is bijective for original vertices
    EXPECT_EQ(result.vertex_mapping.original_to_canonical.size(), 3);
    EXPECT_EQ(result.vertex_mapping.canonical_to_original.size(), 3);

    std::set<VertexId> canonical_ids;
    for (auto& [orig, canon] : result.vertex_mapping.original_to_canonical) {
        canonical_ids.insert(canon);
    }
    EXPECT_EQ(canonical_ids, (std::set<VertexId>{0, 1, 2}));
}

// =============================================================================
// Performance: IR handles larger graphs than brute-force
// =============================================================================

TEST_F(IRCanonicalizationTest, HandlesLargerGraphs) {
    // Create a chain of 20 edges (brute-force can't handle 20 vertices)
    std::vector<std::vector<VertexId>> edges;
    for (VertexId i = 0; i < 20; ++i) {
        edges.push_back({i, i + 1});
    }
    // Should complete without timeout
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 20);
    EXPECT_EQ(result.canonical_form.vertex_count, 21);
}

// =============================================================================
// Integration: IR verification in Hypergraph
// =============================================================================

TEST_F(IRCanonicalizationTest, IRVerificationExactEdgeCorrespondence) {
    Hypergraph hg;
    hg.set_ir_verification(true);
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);

    // Create two isomorphic states: {v0,v1},{v1,v2} and {v3,v4},{v4,v5}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});
    StateId s1 = hg.create_state({e0, e1});

    VertexId v3 = hg.alloc_vertex();
    VertexId v4 = hg.alloc_vertex();
    VertexId v5 = hg.alloc_vertex();
    EdgeId e2 = hg.create_edge({v3, v4});
    EdgeId e3 = hg.create_edge({v4, v5});
    StateId s2 = hg.create_state({e2, e3});

    // IR verification should produce valid edge correspondence
    auto corr = hg.find_edge_correspondence_dispatch(
        hg.get_state_edges(s1), hg.get_state_edges(s2));
    EXPECT_TRUE(corr.valid);
    EXPECT_EQ(corr.count, 2u);
}

TEST_F(IRCanonicalizationTest, IRVerificationNonIsomorphicNoCorrespondence) {
    Hypergraph hg;
    hg.set_ir_verification(true);

    // Path: {v0,v1},{v1,v2} vs Star: {v3,v4},{v3,v5}
    VertexId v0 = hg.alloc_vertex();
    VertexId v1 = hg.alloc_vertex();
    VertexId v2 = hg.alloc_vertex();
    EdgeId e0 = hg.create_edge({v0, v1});
    EdgeId e1 = hg.create_edge({v1, v2});
    StateId s1 = hg.create_state({e0, e1});

    VertexId v3 = hg.alloc_vertex();
    VertexId v4 = hg.alloc_vertex();
    VertexId v5 = hg.alloc_vertex();
    EdgeId e2 = hg.create_edge({v3, v4});
    EdgeId e3 = hg.create_edge({v3, v5});
    StateId s2 = hg.create_state({e2, e3});

    auto corr = hg.find_edge_correspondence_dispatch(
        hg.get_state_edges(s1), hg.get_state_edges(s2));
    EXPECT_FALSE(corr.valid);
}

TEST_F(IRCanonicalizationTest, AreIsomorphicMethod) {
    std::vector<std::vector<VertexId>> path = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> path2 = {{10, 20}, {20, 30}};
    std::vector<std::vector<VertexId>> star = {{1, 2}, {1, 3}};

    EXPECT_TRUE(ir.are_isomorphic(path, path2));
    EXPECT_FALSE(ir.are_isomorphic(path, star));
    EXPECT_TRUE(ir.are_isomorphic({}, {}));
}
