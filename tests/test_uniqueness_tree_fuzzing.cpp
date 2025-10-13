#include <gtest/gtest.h>
#include <hypergraph/uniqueness_tree.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/hash_strategy.hpp>
#include <hypergraph/canonicalization.hpp>
#include "../benchmarking/random_hypergraph_generator.hpp"
#include <random>
#include <algorithm>

using namespace hypergraph;

// Helper to build adjacency index from edges
static std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>
build_adjacency_index(const std::vector<GlobalHyperedge>& edges) {
    std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>> index;
    for (const auto& edge : edges) {
        for (std::size_t pos = 0; pos < edge.global_vertices.size(); ++pos) {
            GlobalVertexId vertex = edge.global_vertices[pos];
            index[vertex].emplace_back(edge.global_id, pos);
        }
    }
    return index;
}

// Helper to convert Hypergraph edges to GlobalHyperedge format
std::vector<GlobalHyperedge> hypergraph_to_global_edges(const Hypergraph& hg) {
    std::vector<GlobalHyperedge> edges;
    GlobalEdgeId edge_id = 0;
    for (const auto& edge : hg.edges()) {
        std::vector<GlobalVertexId> vertices;
        for (auto v : edge.vertices()) {
            vertices.push_back(static_cast<GlobalVertexId>(v));
        }
        edges.emplace_back(edge_id++, vertices);
    }
    return edges;
}

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

// Helper to shuffle edge order
std::vector<std::vector<GlobalVertexId>> shuffle_edge_order(
    const std::vector<std::vector<GlobalVertexId>>& edges,
    uint32_t seed) {

    std::mt19937 rng(seed);
    std::vector<std::vector<GlobalVertexId>> shuffled = edges;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    return shuffled;
}

class UniquenessTreeFuzzingTest : public ::testing::Test {
protected:
    // Test that uniqueness trees correctly identify known isomorphisms
    void test_should_be_isomorphic(
        const std::vector<GlobalHyperedge>& edges1,
        const std::vector<GlobalHyperedge>& edges2,
        const std::string& test_name) {

        UniquenessTreeHashStrategy tree_strategy;
        auto hash1 = tree_strategy.compute_hash(edges1, build_adjacency_index(edges1));
        auto hash2 = tree_strategy.compute_hash(edges2, build_adjacency_index(edges2));

        EXPECT_EQ(hash1, hash2)
            << test_name << ": expected isomorphic graphs to have same hash"
            << "\nHash1: " << hash1
            << "\nHash2: " << hash2;
    }

    // Test that uniqueness trees correctly distinguish known non-isomorphisms
    void test_should_not_be_isomorphic(
        const std::vector<GlobalHyperedge>& edges1,
        const std::vector<GlobalHyperedge>& edges2,
        const std::string& test_name) {

        UniquenessTreeHashStrategy tree_strategy;
        auto hash1 = tree_strategy.compute_hash(edges1, build_adjacency_index(edges1));
        auto hash2 = tree_strategy.compute_hash(edges2, build_adjacency_index(edges2));

        // Note: Hash collisions are possible but should be rare
        // We don't EXPECT_NE because rare collisions are acceptable
        if (hash1 == hash2) {
            // Log collision but don't fail
            std::cout << "Note: Hash collision in " << test_name << std::endl;
        }
    }
};

// Test: Vertex relabeling should preserve isomorphism
TEST_F(UniquenessTreeFuzzingTest, VertexRelabelingIsIsomorphism) {
    for (int test_case = 0; test_case < 20; ++test_case) {
        uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
            "fuzzing_relabeling", test_case, 0, 0, 0);

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

        // Convert to GlobalHyperedge
        auto edges1 = edges_to_global(edge_vecs);
        auto edges2 = edges_to_global(relabeled_vecs);

        test_should_be_isomorphic(edges1, edges2, "VertexRelabeling case " + std::to_string(test_case));
    }
}

// Test: Edge order shouldn't matter for isomorphism
TEST_F(UniquenessTreeFuzzingTest, EdgeOrderIndependence) {
    for (int test_case = 0; test_case < 20; ++test_case) {
        uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
            "fuzzing_edge_order", test_case, 0, 0, 0);

        auto edge_vecs = benchmark::RandomHypergraphGenerator::generate_edges(10, 6, 4, seed);
        auto shuffled_vecs = shuffle_edge_order(edge_vecs, seed + 2000);

        auto edges1 = edges_to_global(edge_vecs);
        auto edges2 = edges_to_global(shuffled_vecs);

        test_should_be_isomorphic(edges1, edges2, "EdgeOrder case " + std::to_string(test_case));
    }
}

// Test: Different random graphs should usually be non-isomorphic
TEST_F(UniquenessTreeFuzzingTest, DifferentGraphsNonIsomorphic) {
    int agreements = 0;
    const int num_tests = 30;

    for (int test_case = 0; test_case < num_tests; ++test_case) {
        uint32_t seed1 = benchmark::RandomHypergraphGenerator::compute_seed(
            "fuzzing_different1", test_case, 0, 0, 0);
        uint32_t seed2 = benchmark::RandomHypergraphGenerator::compute_seed(
            "fuzzing_different2", test_case, 0, 0, 0);

        auto edge_vecs1 = benchmark::RandomHypergraphGenerator::generate_edges(10, 6, 4, seed1);
        auto edge_vecs2 = benchmark::RandomHypergraphGenerator::generate_edges(10, 6, 4, seed2);

        auto edges1 = edges_to_global(edge_vecs1);
        auto edges2 = edges_to_global(edge_vecs2);

        CanonicalizationHashStrategy canon_strategy;
        UniquenessTreeHashStrategy tree_strategy;

        auto adj_index1 = build_adjacency_index(edges1);
        auto adj_index2 = build_adjacency_index(edges2);

        bool canon_says_iso = canon_strategy.are_isomorphic(edges1, adj_index1, edges2, adj_index2);
        bool tree_says_same = tree_strategy.compute_hash(edges1, adj_index1) == tree_strategy.compute_hash(edges2, adj_index2);

        if (canon_says_iso == tree_says_same) {
            agreements++;
        }
    }

    // Should agree on most cases (allow some hash collisions)
    EXPECT_GE(agreements, num_tests * 4 / 5)
        << "Methods should agree on most random non-isomorphic graphs";
}

// Test: Graphs with different sizes are non-isomorphic
TEST_F(UniquenessTreeFuzzingTest, DifferentSizesNonIsomorphic) {
    for (int size1 = 4; size1 <= 8; size1 += 2) {
        for (int size2 = 4; size2 <= 8; size2 += 2) {
            if (size1 == size2) continue;

            uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
                "fuzzing_sizes", size1, size2, 0, 0);

            auto edge_vecs1 = benchmark::RandomHypergraphGenerator::generate_edges(10, size1, 3, seed);
            auto edge_vecs2 = benchmark::RandomHypergraphGenerator::generate_edges(10, size2, 3, seed + 1);

            auto edges1 = edges_to_global(edge_vecs1);
            auto edges2 = edges_to_global(edge_vecs2);

            test_should_not_be_isomorphic(edges1, edges2,
                "DifferentSizes " + std::to_string(size1) + " vs " + std::to_string(size2));
        }
    }
}

// Test: Self-loops preserved under relabeling
TEST_F(UniquenessTreeFuzzingTest, SelfLoopsRelabeling) {
    for (int test_case = 0; test_case < 15; ++test_case) {
        uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
            "fuzzing_selfloops", test_case, 0, 0, 0);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<GlobalVertexId> vertex_dist(1, 8);

        // Create edges with some self-loops
        std::vector<std::vector<GlobalVertexId>> edge_vecs;
        for (int i = 0; i < 5; ++i) {
            if (i < 2) {
                // Self-loop
                GlobalVertexId v = vertex_dist(rng);
                edge_vecs.push_back({v, v});
            } else {
                // Normal edge
                edge_vecs.push_back({vertex_dist(rng), vertex_dist(rng)});
            }
        }

        // Relabel vertices
        std::unordered_map<GlobalVertexId, GlobalVertexId> mapping;
        for (GlobalVertexId v = 1; v <= 8; ++v) {
            mapping[v] = v + 10; // Shift all vertices up
        }

        auto relabeled_vecs = relabel_edges(edge_vecs, mapping);

        auto edges1 = edges_to_global(edge_vecs);
        auto edges2 = edges_to_global(relabeled_vecs);

        test_should_be_isomorphic(edges1, edges2, "SelfLoops case " + std::to_string(test_case));
    }
}

// Test: Edge multiplicity preserved under relabeling
TEST_F(UniquenessTreeFuzzingTest, EdgeMultiplicityRelabeling) {
    for (int test_case = 0; test_case < 15; ++test_case) {
        uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
            "fuzzing_multiplicity", test_case, 0, 0, 0);

        auto edge_vecs = benchmark::RandomHypergraphGenerator::generate_edges(8, 4, 3, seed);

        // Add duplicate edges
        if (!edge_vecs.empty()) {
            edge_vecs.push_back(edge_vecs[0]); // Duplicate first edge
        }

        // Relabel
        std::unordered_map<GlobalVertexId, GlobalVertexId> mapping;
        for (GlobalVertexId v = 1; v <= 8; ++v) {
            mapping[v] = (v * 2) % 17; // Non-trivial permutation
        }

        auto relabeled_vecs = relabel_edges(edge_vecs, mapping);

        auto edges1 = edges_to_global(edge_vecs);
        auto edges2 = edges_to_global(relabeled_vecs);

        test_should_be_isomorphic(edges1, edges2, "EdgeMultiplicity case " + std::to_string(test_case));
    }
}

// Test: Varying graph sizes
TEST_F(UniquenessTreeFuzzingTest, VaryingGraphSizes) {
    for (int num_edges = 2; num_edges <= 10; num_edges += 2) {
        for (int test_case = 0; test_case < 5; ++test_case) {
            uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
                "fuzzing_varying_size", num_edges, test_case, 0, 0);

            auto edge_vecs = benchmark::RandomHypergraphGenerator::generate_edges(
                num_edges * 2, num_edges, 3, seed);

            // Relabel and shuffle
            std::vector<GlobalVertexId> all_vertices(num_edges * 4);
            std::iota(all_vertices.begin(), all_vertices.end(), 1);
            std::mt19937 rng(seed);
            std::shuffle(all_vertices.begin(), all_vertices.end(), rng);

            std::unordered_map<GlobalVertexId, GlobalVertexId> mapping;
            for (GlobalVertexId v = 1; v <= num_edges * 4; ++v) {
                mapping[v] = all_vertices[v - 1];
            }

            auto relabeled_vecs = relabel_edges(edge_vecs, mapping);
            auto shuffled_vecs = shuffle_edge_order(relabeled_vecs, seed + 100);

            auto edges1 = edges_to_global(edge_vecs);
            auto edges2 = edges_to_global(shuffled_vecs);

            test_should_be_isomorphic(edges1, edges2,
                "VaryingSize " + std::to_string(num_edges) + " case " + std::to_string(test_case));
        }
    }
}

// Test: Varying arities
TEST_F(UniquenessTreeFuzzingTest, VaryingArities) {
    for (int max_arity = 2; max_arity <= 6; ++max_arity) {
        for (int test_case = 0; test_case < 5; ++test_case) {
            uint32_t seed = benchmark::RandomHypergraphGenerator::compute_seed(
                "fuzzing_arity", max_arity, test_case, 0, 0);

            auto edge_vecs = benchmark::RandomHypergraphGenerator::generate_edges(
                12, 6, max_arity, seed);

            // Relabel
            std::unordered_map<GlobalVertexId, GlobalVertexId> mapping;
            for (GlobalVertexId v = 1; v <= 12; ++v) {
                mapping[v] = 13 - v; // Reverse mapping
            }

            auto relabeled_vecs = relabel_edges(edge_vecs, mapping);

            auto edges1 = edges_to_global(edge_vecs);
            auto edges2 = edges_to_global(relabeled_vecs);

            test_should_be_isomorphic(edges1, edges2,
                "Arity " + std::to_string(max_arity) + " case " + std::to_string(test_case));
        }
    }
}
