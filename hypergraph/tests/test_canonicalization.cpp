#include <gtest/gtest.h>
#include <hypergraph/canonicalization.hpp>
#include <vector>
#include <set>
#include <algorithm>

using namespace hypergraph;

class CanonicalizationTest : public ::testing::Test {
protected:
    Canonicalizer canonicalizer;
    
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper to check if two hypergraphs should be isomorphic
    bool should_be_isomorphic(const std::vector<std::vector<VertexId>>& edges1,
                             const std::vector<std::vector<VertexId>>& edges2) {
        auto result1 = canonicalizer.canonicalize_edges(edges1);
        auto result2 = canonicalizer.canonicalize_edges(edges2);
        return result1.canonical_form == result2.canonical_form;
    }
    
    // Helper to check if two hypergraphs should NOT be isomorphic
    bool should_not_be_isomorphic(const std::vector<std::vector<VertexId>>& edges1,
                                 const std::vector<std::vector<VertexId>>& edges2) {
        return !should_be_isomorphic(edges1, edges2);
    }
};

TEST_F(CanonicalizationTest, EmptyHypergraph) {
    std::vector<std::vector<VertexId>> empty_edges;
    auto result = canonicalizer.canonicalize_edges(empty_edges);
    
    EXPECT_EQ(result.canonical_form.edges.size(), 0);
    EXPECT_EQ(result.canonical_form.vertex_count, 0);
}

TEST_F(CanonicalizationTest, SingleVertex) {
    std::vector<std::vector<VertexId>> edges = {{1}};
    auto result = canonicalizer.canonicalize_edges(edges);
    
    EXPECT_EQ(result.canonical_form.edges.size(), 1);
    EXPECT_EQ(result.canonical_form.edges[0], std::vector<VertexId>({0}));
    EXPECT_EQ(result.canonical_form.vertex_count, 1);
}

TEST_F(CanonicalizationTest, SingleEdge) {
    std::vector<std::vector<VertexId>> edges = {{1, 2}};
    auto result = canonicalizer.canonicalize_edges(edges);
    
    EXPECT_EQ(result.canonical_form.edges.size(), 1);
    EXPECT_EQ(result.canonical_form.edges[0], std::vector<VertexId>({0, 1}));
    EXPECT_EQ(result.canonical_form.vertex_count, 2);
}

TEST_F(CanonicalizationTest, VertexRelabeling) {
    // Different vertex labels, same structure
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{10, 20}, {20, 30}};
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
}

TEST_F(CanonicalizationTest, EdgeOrdering) {
    // Same edges, different order
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}, {3, 4}};
    std::vector<std::vector<VertexId>> edges2 = {{3, 4}, {1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> edges3 = {{2, 3}, {3, 4}, {1, 2}};
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
    EXPECT_TRUE(should_be_isomorphic(edges1, edges3));
    EXPECT_TRUE(should_be_isomorphic(edges2, edges3));
}

TEST_F(CanonicalizationTest, TriangleIsomorphism) {
    // All should be the same triangle
    std::vector<std::vector<VertexId>> triangle1 = {{1, 2}, {2, 3}, {3, 1}};
    std::vector<std::vector<VertexId>> triangle2 = {{2, 3}, {3, 1}, {1, 2}};
    std::vector<std::vector<VertexId>> triangle3 = {{3, 1}, {1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> triangle4 = {{5, 6}, {6, 7}, {7, 5}};
    
    EXPECT_TRUE(should_be_isomorphic(triangle1, triangle2));
    EXPECT_TRUE(should_be_isomorphic(triangle1, triangle3));
    EXPECT_TRUE(should_be_isomorphic(triangle1, triangle4));
    EXPECT_TRUE(should_be_isomorphic(triangle2, triangle3));
    EXPECT_TRUE(should_be_isomorphic(triangle2, triangle4));
    EXPECT_TRUE(should_be_isomorphic(triangle3, triangle4));
}

TEST_F(CanonicalizationTest, DirectedEdgesMatter) {
    // For 2-arity edges, {1,2} and {2,1} ARE isomorphic via vertex relabeling
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}};
    std::vector<std::vector<VertexId>> edges2 = {{2, 1}};
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
    
    // But for higher arity, vertex order within edges matters
    std::vector<std::vector<VertexId>> edges3 = {{1, 3, 2}, {3, 4}};
    std::vector<std::vector<VertexId>> edges4 = {{1, 2, 3}, {3, 4}};
    
    EXPECT_TRUE(should_not_be_isomorphic(edges3, edges4));
}

TEST_F(CanonicalizationTest, HyperedgeArityMatters) {
    // Different arity edges should not be equivalent
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}};
    std::vector<std::vector<VertexId>> edges2 = {{1, 2, 3}};
    
    EXPECT_TRUE(should_not_be_isomorphic(edges1, edges2));
}

TEST_F(CanonicalizationTest, MultipleArityGroups) {
    // Mix of unary, binary, and ternary edges
    std::vector<std::vector<VertexId>> edges1 = {{1}, {1, 2}, {2, 3, 4}};
    std::vector<std::vector<VertexId>> edges2 = {{5}, {5, 6}, {6, 7, 8}};
    std::vector<std::vector<VertexId>> edges3 = {{2, 3, 4}, {1, 2}, {1}}; // Different order
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
    EXPECT_TRUE(should_be_isomorphic(edges1, edges3));
}

TEST_F(CanonicalizationTest, SelfLoops) {
    // Self-loops should be preserved
    std::vector<std::vector<VertexId>> edges1 = {{1, 1}};
    std::vector<std::vector<VertexId>> edges2 = {{2, 2}};
    std::vector<std::vector<VertexId>> edges3 = {{1, 2}}; // Different structure
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
    EXPECT_TRUE(should_not_be_isomorphic(edges1, edges3));
}

TEST_F(CanonicalizationTest, MultipleEdges) {
    // Multiple edges between same vertices
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {1, 2}};
    std::vector<std::vector<VertexId>> edges2 = {{3, 4}, {3, 4}};
    std::vector<std::vector<VertexId>> edges3 = {{1, 2}}; // Single edge
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
    EXPECT_TRUE(should_not_be_isomorphic(edges1, edges3));
}

TEST_F(CanonicalizationTest, ComplexConnectivity) {
    // More complex connectivity patterns
    std::vector<std::vector<VertexId>> star1 = {{1, 2}, {1, 3}, {1, 4}};
    std::vector<std::vector<VertexId>> star2 = {{5, 6}, {5, 7}, {5, 8}};
    std::vector<std::vector<VertexId>> chain = {{1, 2}, {2, 3}, {3, 4}};
    
    EXPECT_TRUE(should_be_isomorphic(star1, star2));
    EXPECT_TRUE(should_not_be_isomorphic(star1, chain));
}

TEST_F(CanonicalizationTest, ProblemCaseFromDebugging) {
    // Test truly non-isomorphic cases with different structural patterns
    
    // Case 1: Triangle with one additional edge
    std::vector<std::vector<VertexId>> triangle_plus = {{1, 2}, {2, 3}, {3, 1}, {1, 4}};
    
    // Case 2: Path of length 3 with one additional edge  
    std::vector<std::vector<VertexId>> path_plus = {{1, 2}, {2, 3}, {3, 4}, {1, 4}};
    
    // These have different structural connectivity and should NOT be isomorphic
    EXPECT_TRUE(should_not_be_isomorphic(triangle_plus, path_plus));
    
    // Test isomorphic versions (same structure, different labels)
    std::vector<std::vector<VertexId>> triangle_plus_iso = {{5, 6}, {6, 7}, {7, 5}, {5, 8}};
    EXPECT_TRUE(should_be_isomorphic(triangle_plus, triangle_plus_iso));
}

TEST_F(CanonicalizationTest, LargerHyperedges) {
    // Test hyperedges with more than 2 vertices
    std::vector<std::vector<VertexId>> edges1 = {{1, 2, 3, 4}};
    std::vector<std::vector<VertexId>> edges2 = {{5, 6, 7, 8}};
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
    
    // Test truly non-isomorphic case: different connectivity pattern
    std::vector<std::vector<VertexId>> edges3 = {{1, 2, 3}, {3, 4}};   // 3-edge connected to 2-edge via vertex 3
    std::vector<std::vector<VertexId>> edges4 = {{1, 2, 4}, {3, 4}};   // 3-edge connected to 2-edge via vertex 4
    
    EXPECT_TRUE(should_not_be_isomorphic(edges3, edges4)); // Different connectivity
}

TEST_F(CanonicalizationTest, MixedArityWithConnectivity) {
    // Complex case: triangle + 3-edge + 4-edge
    std::vector<std::vector<VertexId>> edges1 = {
        {1, 2}, {2, 3}, {3, 1},  // Triangle
        {1, 4, 5},               // 3-edge from triangle
        {2, 6, 7, 8}             // 4-edge from triangle
    };
    
    std::vector<std::vector<VertexId>> edges2 = {
        {10, 20}, {20, 30}, {30, 10},  // Triangle (different labels)
        {10, 40, 50},                   // 3-edge from vertex 10 (maps to 1)
        {20, 60, 70, 80}                // 4-edge from vertex 20 (maps to 2)
    };
    
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));
}

TEST_F(CanonicalizationTest, CanonicalFormConsistency) {
    // Same input should always produce same canonical form
    std::vector<std::vector<VertexId>> edges = {{1, 2}, {2, 3}, {3, 1}};
    
    auto result1 = canonicalizer.canonicalize_edges(edges);
    auto result2 = canonicalizer.canonicalize_edges(edges);
    auto result3 = canonicalizer.canonicalize_edges(edges);
    
    EXPECT_EQ(result1.canonical_form, result2.canonical_form);
    EXPECT_EQ(result2.canonical_form, result3.canonical_form);
}

TEST_F(CanonicalizationTest, VertexMappingCorrectness) {
    std::vector<std::vector<VertexId>> edges = {{10, 20}, {20, 30}};
    auto result = canonicalizer.canonicalize_edges(edges);
    
    // Check mapping is bijective
    EXPECT_EQ(result.vertex_mapping.canonical_to_original.size(), 3);
    EXPECT_EQ(result.vertex_mapping.original_to_canonical.size(), 3);
    
    // Check reverse mapping works
    for (size_t i = 0; i < result.vertex_mapping.canonical_to_original.size(); ++i) {
        VertexId original = result.vertex_mapping.canonical_to_original[i];
        VertexId canonical = result.vertex_mapping.original_to_canonical[original];
        EXPECT_EQ(canonical, i);
    }
}

TEST_F(CanonicalizationTest, DuplicateEdgesHandling) {
    // Test canonicalization correctly handles duplicate edges
    std::vector<std::vector<VertexId>> edges1 = {{1,2}, {1,2}, {2,3}};
    std::vector<std::vector<VertexId>> edges2 = {{3,4}, {3,4}, {4,5}};
    std::vector<std::vector<VertexId>> edges3 = {{1,2}, {2,3}}; // No duplicates

    // Graphs with same structure and duplicates should be isomorphic
    EXPECT_TRUE(should_be_isomorphic(edges1, edges2));

    // Graph with duplicates should NOT be isomorphic to graph without duplicates
    EXPECT_TRUE(should_not_be_isomorphic(edges1, edges3));
}

TEST_F(CanonicalizationTest, WolframLanguageExamples) {
    // Test against known examples from FindCanonicalHypergraphExamples.wl

    // Example 1: {{10, 20}, {20, 30}} -> {{1, 2}, {2, 3}}
    std::vector<std::vector<VertexId>> input1 = {{10, 20}, {20, 30}};
    auto result1 = canonicalizer.canonicalize_edges(input1);
    std::vector<std::vector<VertexId>> expected1 = {{1, 2}, {2, 3}};

    // Convert expected to 0-based indexing for our implementation
    std::vector<std::vector<VertexId>> expected1_0based = {{0, 1}, {1, 2}};

    std::cout << "WL Example 1 - Input: {{10,20}, {20,30}}" << std::endl;
    std::cout << "Expected (WL): {{1,2}, {2,3}}" << std::endl;
    std::cout << "Expected (0-based): {{0,1}, {1,2}}" << std::endl;
    std::cout << "Actual: ";
    for (const auto& edge : result1.canonical_form.edges) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i < edge.size() - 1) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    EXPECT_EQ(result1.canonical_form.edges, expected1_0based);
}

TEST_F(CanonicalizationTest, TestCase1ProblematicStates) {
    // Test the specific states from TestCase1 that are incorrectly being treated as equivalent
    // These states appeared in debug output and should NOT be isomorphic

    // State A: {2,3} {1,2} {2,5} {5,9}
    std::vector<std::vector<VertexId>> stateA = {{2,3}, {1,2}, {2,5}, {5,9}};

    // State B: {1,2} {2,5} {2,3} {3,11}
    std::vector<std::vector<VertexId>> stateB = {{1,2}, {2,5}, {2,3}, {3,11}};

    // Debug: Print the original inputs
    std::cout << "State A input: ";
    for (const auto& edge : stateA) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i < edge.size() - 1) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    std::cout << "State B input: ";
    for (const auto& edge : stateB) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i < edge.size() - 1) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    // Debug: Print the reverse sorted order first
    auto stateA_copy = stateA;
    auto stateB_copy = stateB;

    std::sort(stateA_copy.rbegin(), stateA_copy.rend());
    std::sort(stateB_copy.rbegin(), stateB_copy.rend());

    std::cout << "State A reverse sorted: ";
    for (const auto& edge : stateA_copy) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i < edge.size() - 1) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    std::cout << "State B reverse sorted: ";
    for (const auto& edge : stateB_copy) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i < edge.size() - 1) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    // Debug: Print the canonical forms
    auto resultA = canonicalizer.canonicalize_edges(stateA);
    auto resultB = canonicalizer.canonicalize_edges(stateB);

    std::cout << "State A canonical: ";
    for (const auto& edge : resultA.canonical_form.edges) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i < edge.size() - 1) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    std::cout << "State B canonical: ";
    for (const auto& edge : resultB.canonical_form.edges) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i < edge.size() - 1) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    // These ARE actually isomorphic - mapping: A{1,2,3,5,9} -> B{5,2,1,3,11}
    // The canonicalization is working correctly
    EXPECT_TRUE(should_be_isomorphic(stateA, stateB));
}

TEST_F(CanonicalizationTest, SimpleVertexRelabeling) {
    // {1,2} and {2,1} should be isomorphic via vertex relabeling
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}};
    std::vector<std::vector<VertexId>> edges2 = {{2, 1}};

    Canonicalizer canonicalizer;
    auto result1 = canonicalizer.canonicalize_edges(edges1);
    auto result2 = canonicalizer.canonicalize_edges(edges2);

    std::cout << "Input 1: {" << edges1[0][0] << "," << edges1[0][1] << "}" << std::endl;
    std::cout << "Input 2: {" << edges2[0][0] << "," << edges2[0][1] << "}" << std::endl;

    std::cout << "Canonical 1: ";
    for (const auto& edge : result1.canonical_form.edges) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i + 1 < edge.size()) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    std::cout << "Canonical 2: ";
    for (const auto& edge : result2.canonical_form.edges) {
        std::cout << "{";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i + 1 < edge.size()) std::cout << ",";
        }
        std::cout << "} ";
    }
    std::cout << std::endl;

    EXPECT_EQ(result1.canonical_form.edges, result2.canonical_form.edges);
}

TEST_F(CanonicalizationTest, RandomGraphFromFuzzing) {
    // The failing case from our fuzzing test
    std::vector<std::vector<VertexId>> edges1 = {
        {8, 7, 6, 10},
        {2, 9, 5, 10},
        {5, 1},
        {6, 9, 2},
        {10, 8, 1},
        {3, 6, 10, 6}
    };

    std::vector<std::vector<VertexId>> edges2 = {
        {1, 10, 8, 5},
        {2, 3, 9, 5},
        {9, 6},
        {8, 3, 2},
        {5, 1, 6},
        {7, 8, 5, 8}
    };

    Canonicalizer canonicalizer;
    auto result1 = canonicalizer.canonicalize_edges(edges1);
    auto result2 = canonicalizer.canonicalize_edges(edges2);

    std::cout << "Canonical form 1:" << std::endl;
    for (const auto& edge : result1.canonical_form.edges) {
        std::cout << "  {";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i + 1 < edge.size()) std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }

    std::cout << "Canonical form 2:" << std::endl;
    for (const auto& edge : result2.canonical_form.edges) {
        std::cout << "  {";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i + 1 < edge.size()) std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }

    EXPECT_EQ(result1.canonical_form.edges, result2.canonical_form.edges);
}

