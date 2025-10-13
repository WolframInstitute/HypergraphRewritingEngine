#include <gtest/gtest.h>
#include <hypergraph/uniqueness_tree.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/hash_strategy.hpp>

using namespace hypergraph;

class UniquenessTreeTest : public ::testing::Test {
protected:
    // Helper to create edges for testing
    std::vector<GlobalHyperedge> create_edges(
        const std::vector<std::vector<GlobalVertexId>>& edge_structures) {
        std::vector<GlobalHyperedge> edges;
        GlobalEdgeId edge_id = 0;
        for (const auto& vertices : edge_structures) {
            edges.emplace_back(edge_id++, vertices);
        }
        return edges;
    }
};

// Test basic uniqueness tree construction
TEST_F(UniquenessTreeTest, BasicConstruction) {
    auto edges = create_edges({{1, 2}, {2, 3}});

    UniquenessTree tree1(1, edges);
    EXPECT_EQ(tree1.root_vertex(), 1);
    EXPECT_GT(tree1.hash(), 0);
}

// Test uniqueness tree set
TEST_F(UniquenessTreeTest, TreeSetConstruction) {
    auto edges = create_edges({{1, 2}, {2, 3}});

    UniquenessTreeSet tree_set(edges);
    auto hash = tree_set.canonical_hash();
    EXPECT_GT(hash, 0);
}

// Test edge multiplicity with non-isomorphic graphs
TEST_F(UniquenessTreeTest, EdgeMultiplicityNonIsomorphic) {
    auto edges1 = create_edges({{1, 2}});
    auto edges2 = create_edges({{1, 2}, {1, 2}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_NE(hash1, hash2) << "Edge multiplicity should produce different hashes";
}

// Test edge multiplicity with isomorphic graphs
TEST_F(UniquenessTreeTest, EdgeMultiplicityIsomorphic) {
    auto edges1 = create_edges({{1, 2}, {1, 2}});
    auto edges2 = create_edges({{10, 20}, {10, 20}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Isomorphic graphs with edge multiplicity should have same hash";
}

// Test self-loops: {1,1}
TEST_F(UniquenessTreeTest, SelfLoops) {
    auto edges = create_edges({{1, 1}});

    UniquenessTree tree(1, edges);
    EXPECT_GT(tree.hash(), 0);

    UniquenessTreeSet tree_set(edges);
    EXPECT_GT(tree_set.canonical_hash(), 0);
}

// Test multiple self-loops: {1,1,1}
TEST_F(UniquenessTreeTest, MultipleSelfLoops) {
    auto edges = create_edges({{1, 1, 1}});

    UniquenessTree tree(1, edges);
    EXPECT_GT(tree.hash(), 0);

    UniquenessTreeSet tree_set(edges);
    EXPECT_GT(tree_set.canonical_hash(), 0);
}

// Test self-loops with different multiplicities
TEST_F(UniquenessTreeTest, SelfLoopMultiplicity) {
    auto edges1 = create_edges({{1, 1}});
    auto edges2 = create_edges({{1, 1}, {1, 1}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_NE(hash1, hash2) << "Multiple self-loops should be distinguishable";
}

// Test self-loops with vertex relabeling
TEST_F(UniquenessTreeTest, SelfLoopIsomorphism) {
    auto edges1 = create_edges({{1, 1}, {2, 2}});
    auto edges2 = create_edges({{10, 10}, {20, 20}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Isomorphic self-loop patterns should have same hash";
}

// Test that edge position matters: {1,1,2} is ordered
TEST_F(UniquenessTreeTest, EdgePositionMatters) {
    auto edges1 = create_edges({{1, 1, 2}});
    auto edges2 = create_edges({{1, 2, 1}});
    auto edges3 = create_edges({{2, 1, 1}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);
    UniquenessTreeSet tree_set3(edges3);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();
    auto hash3 = tree_set3.canonical_hash();

    // All three should be different because vertex order matters in hyperedges
    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash2, hash3);
    EXPECT_NE(hash1, hash3);
}

// Test mixed edge multiplicity
TEST_F(UniquenessTreeTest, MixedMultiplicity) {
    auto edges1 = create_edges({{1, 2}, {1, 2}, {2, 3}});
    auto edges2 = create_edges({{1, 2}, {2, 3}, {2, 3}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_NE(hash1, hash2) << "Different multiplicity patterns should have different hashes";
}

// Test different multiplicity counts
TEST_F(UniquenessTreeTest, DifferentMultiplicityCounts) {
    auto edges1 = create_edges({{1, 2}});
    auto edges2 = create_edges({{1, 2}, {1, 2}});
    auto edges3 = create_edges({{1, 2}, {1, 2}, {1, 2}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);
    UniquenessTreeSet tree_set3(edges3);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();
    auto hash3 = tree_set3.canonical_hash();

    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash2, hash3);
    EXPECT_NE(hash1, hash3);
}

// Test isomorphism: graphs with same structure but different labels should have same hash
TEST_F(UniquenessTreeTest, IsomorphismInvariance) {
    auto edges1 = create_edges({{1, 2}, {2, 3}});
    auto edges2 = create_edges({{10, 20}, {20, 30}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Isomorphic graphs should have same hash";
}

// Test non-isomorphic graphs have different hashes
TEST_F(UniquenessTreeTest, NonIsomorphicDifferentHashes) {
    auto edges1 = create_edges({{1, 2}, {3, 4}});  // Disconnected
    auto edges2 = create_edges({{1, 2}, {2, 3}});  // Connected

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_NE(hash1, hash2) << "Non-isomorphic graphs should have different hashes";
}

// Test edge order independence: same edges in different order should give same hash
TEST_F(UniquenessTreeTest, EdgeOrderIndependence) {
    auto edges1 = create_edges({{1, 2}, {2, 3}, {3, 4}});
    auto edges2 = create_edges({{3, 4}, {1, 2}, {2, 3}});
    auto edges3 = create_edges({{2, 3}, {3, 4}, {1, 2}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);
    UniquenessTreeSet tree_set3(edges3);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();
    auto hash3 = tree_set3.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Edge order should not affect hash";
    EXPECT_EQ(hash2, hash3) << "Edge order should not affect hash";
}

// Test disconnected components with edge multiplicity
TEST_F(UniquenessTreeTest, DisconnectedComponentsWithMultiplicity) {
    auto edges1 = create_edges({{1, 2}, {1, 2}, {3, 4}, {3, 4}});
    auto edges2 = create_edges({{10, 20}, {10, 20}, {30, 40}, {30, 40}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Isomorphic disconnected components should have same hash";
}

// Test large arity with repeated vertices
TEST_F(UniquenessTreeTest, LargeArityWithRepeats) {
    auto edges1 = create_edges({{1, 1, 2, 2, 3, 3}});
    auto edges2 = create_edges({{10, 10, 20, 20, 30, 30}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Isomorphic hyperedges with repeated vertices should have same hash";
}

// Test complex multiplicity pattern
TEST_F(UniquenessTreeTest, ComplexMultiplicityPattern) {
    auto edges1 = create_edges({{1, 2}, {1, 2}, {2, 3}, {3, 4}});
    auto edges2 = create_edges({{10, 20}, {10, 20}, {20, 30}, {30, 40}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Complex isomorphic patterns should have same hash";
}

// Test hash strategy with edge multiplicity
TEST_F(UniquenessTreeTest, HashStrategyEdgeMultiplicity) {
    auto edges1 = create_edges({{1, 2}});
    auto edges2 = create_edges({{1, 2}, {1, 2}});

    UniquenessTreeHashStrategy strategy;

    auto hash1 = strategy.compute_hash(edges1);
    auto hash2 = strategy.compute_hash(edges2);

    EXPECT_NE(hash1, hash2) << "Hash strategy should distinguish edge multiplicity";
}

// Test hash strategy with self-loops
TEST_F(UniquenessTreeTest, HashStrategySelfLoops) {
    auto edges1 = create_edges({{1, 1}});
    auto edges2 = create_edges({{1, 2}});

    UniquenessTreeHashStrategy strategy;

    auto hash1 = strategy.compute_hash(edges1);
    auto hash2 = strategy.compute_hash(edges2);

    EXPECT_NE(hash1, hash2) << "Self-loops should produce different hash than regular edges";
}

// Test hash strategy isomorphism invariance
TEST_F(UniquenessTreeTest, HashStrategyIsomorphismInvariance) {
    auto edges1 = create_edges({{1, 2}, {2, 3}, {3, 1}});
    auto edges2 = create_edges({{10, 20}, {20, 30}, {30, 10}});

    UniquenessTreeHashStrategy strategy;

    auto hash1 = strategy.compute_hash(edges1);
    auto hash2 = strategy.compute_hash(edges2);

    EXPECT_EQ(hash1, hash2) << "Isomorphic triangles should have same hash";
}

// Test complex hypergraph with mixed arities
TEST_F(UniquenessTreeTest, MixedArities) {
    auto edges = create_edges({{1, 2}, {2, 3, 4}, {4, 5, 6, 7}});

    UniquenessTreeSet tree_set(edges);
    auto hash = tree_set.canonical_hash();
    EXPECT_GT(hash, 0);
}

// Test empty hypergraph
TEST_F(UniquenessTreeTest, EmptyHypergraph) {
    std::vector<GlobalHyperedge> edges;

    UniquenessTreeSet tree_set(edges);
    auto hash = tree_set.canonical_hash();
    EXPECT_EQ(hash, 0) << "Empty hypergraph should have zero hash";
}

// Test triangle vs path (same number of edges and vertices, different structure)
TEST_F(UniquenessTreeTest, TriangleVsPath) {
    auto triangle = create_edges({{1, 2}, {2, 3}, {3, 1}});
    auto path = create_edges({{1, 2}, {2, 3}, {3, 4}});

    UniquenessTreeSet tree_set1(triangle);
    UniquenessTreeSet tree_set2(path);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_NE(hash1, hash2) << "Triangle and path should have different hashes";
}

// Test symmetric vs asymmetric structures
TEST_F(UniquenessTreeTest, SymmetricVsAsymmetric) {
    auto symmetric = create_edges({{1, 2}, {1, 3}, {2, 3}});  // Complete triangle
    auto asymmetric = create_edges({{1, 2}, {1, 3}, {1, 4}});  // Star

    UniquenessTreeSet tree_set1(symmetric);
    UniquenessTreeSet tree_set2(asymmetric);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_NE(hash1, hash2) << "Symmetric and asymmetric structures should differ";
}

// Test hyperedge with all vertices the same
TEST_F(UniquenessTreeTest, AllVerticesSame) {
    auto edges1 = create_edges({{1, 1, 1, 1}});
    auto edges2 = create_edges({{10, 10, 10, 10}});

    UniquenessTreeSet tree_set1(edges1);
    UniquenessTreeSet tree_set2(edges2);

    auto hash1 = tree_set1.canonical_hash();
    auto hash2 = tree_set2.canonical_hash();

    EXPECT_EQ(hash1, hash2) << "Isomorphic hyperedges with all same vertex should match";
}
