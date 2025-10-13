#include <gtest/gtest.h>
#include <hypergraph/hash_strategy.hpp>
#include <hypergraph/wolfram_states.hpp>
#include "../benchmarking/random_hypergraph_generator.hpp"
#include <iostream>
#include <algorithm>

using namespace hypergraph;

namespace {

// Helper to convert edge vectors to GlobalHyperedge format
std::vector<GlobalHyperedge> edges_to_global(const std::vector<std::vector<GlobalVertexId>>& edge_vecs) {
    std::vector<GlobalHyperedge> edges;
    for (std::size_t i = 0; i < edge_vecs.size(); ++i) {
        edges.emplace_back(i, edge_vecs[i]);
    }
    return edges;
}

// Helper to relabel vertices in edge vectors
std::vector<std::vector<GlobalVertexId>> relabel_edges(
    const std::vector<std::vector<GlobalVertexId>>& edges,
    const std::unordered_map<GlobalVertexId, GlobalVertexId>& mapping) {

    std::vector<std::vector<GlobalVertexId>> relabeled;
    for (const auto& edge : edges) {
        std::vector<GlobalVertexId> new_edge;
        for (auto v : edge) {
            auto it = mapping.find(v);
            new_edge.push_back(it != mapping.end() ? it->second : v);
        }
        relabeled.push_back(new_edge);
    }
    return relabeled;
}

// Brute force isomorphism check
// Try all possible vertex mappings to see if any preserves the edge structure
bool brute_force_isomorphism_check(
    const std::vector<std::vector<GlobalVertexId>>& edges1,
    const std::vector<std::vector<GlobalVertexId>>& edges2) {

    if (edges1.size() != edges2.size()) {
        return false;
    }

    // Get all vertices in both graphs
    std::set<GlobalVertexId> verts1, verts2;
    for (const auto& edge : edges1) {
        for (auto v : edge) {
            verts1.insert(v);
        }
    }
    for (const auto& edge : edges2) {
        for (auto v : edge) {
            verts2.insert(v);
        }
    }

    if (verts1.size() != verts2.size()) {
        return false;
    }

    std::vector<GlobalVertexId> v1_list(verts1.begin(), verts1.end());
    std::vector<GlobalVertexId> v2_list(verts2.begin(), verts2.end());

    // Try all permutations of v2_list
    std::sort(v2_list.begin(), v2_list.end());

    do {
        // Build mapping v1[i] -> v2[i]
        std::unordered_map<GlobalVertexId, GlobalVertexId> mapping;
        for (size_t i = 0; i < v1_list.size(); ++i) {
            mapping[v1_list[i]] = v2_list[i];
        }

        // Apply mapping to edges1
        auto remapped = relabel_edges(edges1, mapping);

        // Sort both edge lists for comparison
        auto sorted_remapped = remapped;
        auto sorted_edges2 = edges2;
        std::sort(sorted_remapped.begin(), sorted_remapped.end());
        std::sort(sorted_edges2.begin(), sorted_edges2.end());

        if (sorted_remapped == sorted_edges2) {
            return true;
        }
    } while (std::next_permutation(v2_list.begin(), v2_list.end()));

    return false;
}

void print_edges(const std::vector<std::vector<GlobalVertexId>>& edges, const std::string& label) {
    std::cout << label << ":" << std::endl;
    for (const auto& edge : edges) {
        std::cout << "  {";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if (i + 1 < edge.size()) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
    }
}

} // namespace

TEST(BruteForceIsomorphism, FailingCaseFromFuzzing) {
    // The failing case from our fuzzing test (test_case = 2)
    int test_case = 2;
    uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
        "fuzzing_relabeling", test_case, 0, 0, 0);

    std::cout << "Seed: " << seed << std::endl;

    // Generate random hypergraph
    auto edge_vecs = benchmark::RandomHypergraphGenerator::generate_edges(10, 6, 4, seed);

    // Find all vertices
    std::unordered_set<GlobalVertexId> vertex_set;
    for (const auto& edge : edge_vecs) {
        for (auto v : edge) {
            vertex_set.insert(v);
        }
    }
    std::vector<GlobalVertexId> vertices(vertex_set.begin(), vertex_set.end());
    std::sort(vertices.begin(), vertices.end());

    // Create random permutation
    std::mt19937 rng(seed + 1000);
    std::vector<GlobalVertexId> shuffled_vertices = vertices;
    std::shuffle(shuffled_vertices.begin(), shuffled_vertices.end(), rng);

    // Build relabeling map
    std::unordered_map<GlobalVertexId, GlobalVertexId> mapping;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        mapping[vertices[i]] = shuffled_vertices[i];
    }

    // Apply relabeling
    auto relabeled_vecs = relabel_edges(edge_vecs, mapping);

    print_edges(edge_vecs, "Original edges");
    print_edges(relabeled_vecs, "Relabeled edges");

    std::cout << "\nVertex mapping:" << std::endl;
    for (size_t i = 0; i < vertices.size(); ++i) {
        std::cout << "  " << vertices[i] << " -> " << shuffled_vertices[i] << std::endl;
    }

    // Test with all three methods
    CanonicalizationHashStrategy canon_strategy;
    UniquenessTreeHashStrategy tree_strategy;

    bool canon_says_iso = canon_strategy.are_isomorphic(edges_to_global(edge_vecs), edges_to_global(relabeled_vecs));
    bool tree_says_iso = tree_strategy.compute_hash(edges_to_global(edge_vecs)) ==
                         tree_strategy.compute_hash(edges_to_global(relabeled_vecs));
    bool brute_force_says_iso = brute_force_isomorphism_check(edge_vecs, relabeled_vecs);

    std::cout << "\nIsomorphism check results:" << std::endl;
    std::cout << "  Canonicalization:  " << (canon_says_iso ? "ISOMORPHIC" : "NON-ISOMORPHIC") << std::endl;
    std::cout << "  Uniqueness trees:  " << (tree_says_iso ? "ISOMORPHIC" : "NON-ISOMORPHIC") << std::endl;
    std::cout << "  Brute force:       " << (brute_force_says_iso ? "ISOMORPHIC" : "NON-ISOMORPHIC") << std::endl;

    // By construction (vertex relabeling), these SHOULD be isomorphic
    EXPECT_TRUE(brute_force_says_iso) << "Brute force should confirm vertex relabeling creates isomorphic graph";

    // Check which method is correct
    EXPECT_EQ(tree_says_iso, brute_force_says_iso) << "Uniqueness trees should match brute force";
    EXPECT_EQ(canon_says_iso, brute_force_says_iso) << "Canonicalization should match brute force";
}

TEST(BruteForceIsomorphism, SimpleCase) {
    // Simple case that should work
    std::vector<std::vector<GlobalVertexId>> edges1 = {{1, 2}, {2, 3}};
    std::vector<std::vector<GlobalVertexId>> edges2 = {{2, 1}, {1, 3}};

    print_edges(edges1, "Edges 1");
    print_edges(edges2, "Edges 2");

    bool brute_force = brute_force_isomorphism_check(edges1, edges2);
    std::cout << "Brute force says: " << (brute_force ? "ISOMORPHIC" : "NON-ISOMORPHIC") << std::endl;

    CanonicalizationHashStrategy canon_strategy;
    bool canon = canon_strategy.are_isomorphic(edges_to_global(edges1), edges_to_global(edges2));
    std::cout << "Canonicalization says: " << (canon ? "ISOMORPHIC" : "NON-ISOMORPHIC") << std::endl;

    UniquenessTreeHashStrategy tree_strategy;
    bool tree = tree_strategy.compute_hash(edges_to_global(edges1)) ==
                tree_strategy.compute_hash(edges_to_global(edges2));
    std::cout << "Uniqueness trees say: " << (tree ? "ISOMORPHIC" : "NON-ISOMORPHIC") << std::endl;

    EXPECT_EQ(canon, brute_force);
    EXPECT_EQ(tree, brute_force);
}
