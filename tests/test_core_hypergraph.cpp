#include <gtest/gtest.h>
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/hyperedge.hpp>
#include <hypergraph/vertex.hpp>
#include "test_helpers.hpp"

class CoreHypergraphTest : public ::testing::Test {
protected:
    hypergraph::Hypergraph graph;
};

// === VERTEX OPERATIONS ===

TEST_F(CoreHypergraphTest, EmptyHypergraph) {
    EXPECT_EQ(graph.num_vertices(), 0);
    EXPECT_EQ(graph.num_edges(), 0);
    EXPECT_TRUE(graph.vertices().empty());
    EXPECT_TRUE(graph.edges().empty());
}

TEST_F(CoreHypergraphTest, VertexManagement) {
    // Vertices are created implicitly when adding edges
    auto edge_id = graph.add_edge({1, 2});
    
    EXPECT_EQ(graph.num_vertices(), 2);
    EXPECT_TRUE(graph.has_vertex(1));
    EXPECT_TRUE(graph.has_vertex(2));
    EXPECT_TRUE(graph.has_edge(edge_id));
}

// === EDGE OPERATIONS ===

TEST_F(CoreHypergraphTest, BasicEdgeCreation) {
    auto edge_id = graph.add_edge({1, 2});
    
    EXPECT_EQ(graph.num_edges(), 1);
    EXPECT_TRUE(graph.has_edge(edge_id));
    
    const auto* edge = graph.get_edge(edge_id);
    ASSERT_NE(edge, nullptr);
    EXPECT_EQ(edge->arity(), 2);
    EXPECT_EQ(edge->vertices()[0], 1);
    EXPECT_EQ(edge->vertices()[1], 2);
}

TEST_F(CoreHypergraphTest, SelfLoopEdge) {
    auto edge_id = graph.add_edge({1, 1});
    
    EXPECT_EQ(graph.num_edges(), 1);
    const auto* edge = graph.get_edge(edge_id);
    ASSERT_NE(edge, nullptr);
    EXPECT_EQ(edge->arity(), 2);
    EXPECT_EQ(edge->vertices()[0], 1);
    EXPECT_EQ(edge->vertices()[1], 1);
}

TEST_F(CoreHypergraphTest, HyperedgeCreation) {
    auto edge_id = graph.add_edge({1, 2, 3});  // 3-ary hyperedge
    
    EXPECT_EQ(graph.num_edges(), 1);
    const auto* edge = graph.get_edge(edge_id);
    ASSERT_NE(edge, nullptr);
    EXPECT_EQ(edge->arity(), 3);
}

// === VERTEX-EDGE RELATIONSHIPS ===

TEST_F(CoreHypergraphTest, VertexToEdgesMapping) {
    auto e1 = graph.add_edge({1, 2});
    auto e2 = graph.add_edge({2, 3});
    auto e3 = graph.add_edge({1, 3});
    
    auto v1_edges = graph.edges_containing(1);
    auto v2_edges = graph.edges_containing(2);
    auto v3_edges = graph.edges_containing(3);
    
    EXPECT_EQ(v1_edges.size(), 2);  // e1, e3
    EXPECT_EQ(v2_edges.size(), 2);  // e1, e2
    EXPECT_EQ(v3_edges.size(), 2);  // e2, e3
    
    EXPECT_TRUE(std::find(v1_edges.begin(), v1_edges.end(), e1) != v1_edges.end());
    EXPECT_TRUE(std::find(v1_edges.begin(), v1_edges.end(), e3) != v1_edges.end());
}

// === EDGE REMOVAL ===

TEST_F(CoreHypergraphTest, EdgeRemoval) {
    auto e1 = graph.add_edge({1, 2});
    auto e2 = graph.add_edge({2, 3});
    
    EXPECT_EQ(graph.num_edges(), 2);
    EXPECT_EQ(graph.num_vertices(), 3);
    
    graph.remove_edge(e1);
    
    EXPECT_EQ(graph.num_edges(), 1);
    EXPECT_FALSE(graph.has_edge(e1));
    EXPECT_TRUE(graph.has_edge(e2));
    
    // Vertices might be cleaned up or remain - depends on implementation
    EXPECT_TRUE(graph.has_vertex(2));
    EXPECT_TRUE(graph.has_vertex(3));
}

TEST_F(CoreHypergraphTest, IsolatedVertexCleanup) {
    auto e1 = graph.add_edge({1, 2});
    
    EXPECT_EQ(graph.num_vertices(), 2);
    EXPECT_EQ(graph.num_edges(), 1);
    
    graph.remove_edge(e1);
    
    // After removing the only edge, check vertex cleanup behavior
    EXPECT_EQ(graph.num_edges(), 0);
    // Vertices may or may not be cleaned up automatically
}

// === VERTEX DEGREE ===

TEST_F(CoreHypergraphTest, VertexDegree) {
    // No add_vertex method - vertices created implicitly
    // No get_vertex_degree method - test using edges_containing instead
    
    graph.add_edge({1, 2});
    EXPECT_EQ(graph.edges_containing(1).size(), 1);
    EXPECT_EQ(graph.edges_containing(2).size(), 1);
    EXPECT_EQ(graph.edges_containing(3).size(), 0);
    
    graph.add_edge({1, 3});
    EXPECT_EQ(graph.edges_containing(1).size(), 2);  // Connected to 2 edges
    EXPECT_EQ(graph.edges_containing(2).size(), 1);
    EXPECT_EQ(graph.edges_containing(3).size(), 1);
}

// === NEIGHBORHOOD QUERIES ===

TEST_F(CoreHypergraphTest, VertexNeighbors) {
    graph.add_edge({1, 2});
    graph.add_edge({1, 3});
    graph.add_edge({2, 4});
    
    // No get_vertex_neighbors method - test edges_containing instead
    auto v1_edges = graph.edges_containing(1);
    EXPECT_EQ(v1_edges.size(), 2);  // Connected to 2 edges
    
    auto v2_edges = graph.edges_containing(2);
    EXPECT_EQ(v2_edges.size(), 2);  // Connected to 2 edges
    
    auto v4_edges = graph.edges_containing(4);
    EXPECT_EQ(v4_edges.size(), 1);  // Connected to 1 edge
}

// === RADIUS QUERIES ===

TEST_F(CoreHypergraphTest, EdgesWithinRadius) {
    // Create a linear chain: 1 - 2 - 3 - 4
    graph.add_edge({1, 2});
    graph.add_edge({2, 3});
    graph.add_edge({3, 4});
    
    auto radius0 = graph.edges_within_radius(2, 0);
    EXPECT_EQ(radius0.size(), 0);  // Radius 0 returns empty
    
    auto radius1 = graph.edges_within_radius(2, 1);
    EXPECT_EQ(radius1.size(), 2);  // Edges {1,2} and {2,3}
    
    auto radius2 = graph.edges_within_radius(2, 2);
    EXPECT_EQ(radius2.size(), 3);  // All 3 edges
}

// === ERROR HANDLING ===

TEST_F(CoreHypergraphTest, InvalidVertexAccess) {
    hypergraph::VertexId invalid_vertex = 999;
    EXPECT_FALSE(graph.has_vertex(invalid_vertex));
    EXPECT_TRUE(graph.edges_containing(invalid_vertex).empty());
}

TEST_F(CoreHypergraphTest, InvalidEdgeAccess) {
    hypergraph::EdgeId invalid_edge = 999;
    EXPECT_FALSE(graph.has_edge(invalid_edge));
    
    // get_edge with invalid ID should return nullptr
    EXPECT_EQ(graph.get_edge(invalid_edge), nullptr);
}

// === PERFORMANCE TESTS ===

TEST_F(CoreHypergraphTest, LargeGraphConstruction) {
    test_utils::PerfTimer timer;
    
    const int num_edges = 1000;
    
    // Create edges (linear chain) - vertices created implicitly
    for (int i = 0; i < num_edges; ++i) {
        graph.add_edge({static_cast<hypergraph::VertexId>(i), 
                       static_cast<hypergraph::VertexId>(i + 1)});
    }
    
    double elapsed = timer.elapsed_ms();
    
    EXPECT_EQ(graph.num_edges(), num_edges);
    EXPECT_EQ(graph.num_vertices(), num_edges + 1);  // Chain has n+1 vertices
    EXPECT_LT(elapsed, 100.0);  // Should be fast
}