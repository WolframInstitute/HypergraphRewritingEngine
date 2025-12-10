// BENCHMARK_CATEGORY: v1 vs Unified Comparison
//
// Compares v1 canonicalization with unified uniqueness tree hashing
// to measure the performance improvement from the new architecture.

#include "benchmark_framework.hpp"
#include "random_hypergraph_generator.hpp"
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/canonicalization.hpp>
#include <hypergraph/uniqueness_tree.hpp>
#include <hypergraph/wolfram_states.hpp>

// v2 (unified) includes
#include <hypergraph/unified/arena.hpp>
#include <hypergraph/unified/types.hpp>
#include <hypergraph/unified/bitset.hpp>
#include <hypergraph/unified/unified_hypergraph.hpp>
#include <hypergraph/unified/wl_hash.hpp>

using namespace hypergraph;
using namespace benchmark;
namespace v2 = hypergraph::unified;

// =============================================================================
// Helper functions
// =============================================================================

// Convert v1 Hypergraph to GlobalHyperedge format for v1 uniqueness trees
static std::vector<GlobalHyperedge> convert_to_global_edges(const Hypergraph& hg) {
    std::vector<GlobalHyperedge> edges;
    GlobalEdgeId edge_id = 0;

    for (const auto& edge : hg.edges()) {
        std::vector<GlobalVertexId> vertices;
        for (auto vertex : edge.vertices()) {
            vertices.push_back(static_cast<GlobalVertexId>(vertex));
        }
        edges.emplace_back(edge_id++, vertices);
    }

    return edges;
}

// Build adjacency index for v1 uniqueness trees
static std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>
make_adjacency_index(const std::vector<GlobalHyperedge>& edges) {
    std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>> index;
    for (const auto& edge : edges) {
        for (std::size_t pos = 0; pos < edge.global_vertices.size(); ++pos) {
            GlobalVertexId vertex = edge.global_vertices[pos];
            index[vertex].emplace_back(edge.global_id, pos);
        }
    }
    return index;
}

// Helper to setup unified hypergraph from v1 hypergraph
void setup_unified_hypergraph(const Hypergraph& hg, v2::UnifiedHypergraph& v2_hg) {
    std::unordered_map<VertexId, v2::VertexId> vertex_map;
    for (const auto& edge : hg.edges()) {
        std::vector<v2::VertexId> v2_vertices;
        for (auto vertex : edge.vertices()) {
            if (vertex_map.find(vertex) == vertex_map.end()) {
                vertex_map[vertex] = v2_hg.alloc_vertex();
            }
            v2_vertices.push_back(vertex_map[vertex]);
        }
        // Use pointer + size version of create_edge
        v2_hg.create_edge(v2_vertices.data(), static_cast<uint8_t>(v2_vertices.size()));
    }
}

// =============================================================================
// Category: v1 vs Unified Canonicalization Comparison
// =============================================================================

BENCHMARK(v1_uniqueness_tree_hash_by_edges, "v1 UniquenessTree hash computation by edge count") {
    for (int edges : {2, 4, 6, 8, 10, 15, 20, 30, 40, 50}) {
        BENCHMARK_PARAM("edges", edges);

        // Generate symmetric hypergraph (same as existing benchmarks)
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("v1_uniqueness_tree_hash", 0, edges, symmetry_groups, 2);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);

        // Setup for v1
        auto global_edges = convert_to_global_edges(hg);
        auto adjacency_index = make_adjacency_index(global_edges);

        BENCHMARK_CODE([&]() {
            UniquenessTreeSet tree_set(global_edges, adjacency_index);
            volatile uint64_t hash = tree_set.canonical_hash();
            (void)hash;
        });
    }
}

BENCHMARK(unified_uniqueness_tree_hash_by_edges, "Unified UniquenessTree hash computation by edge count") {
    for (int edges : {2, 4, 6, 8, 10, 15, 20, 30, 40, 50}) {
        BENCHMARK_PARAM("edges", edges);

        // Generate symmetric hypergraph
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("unified_uniqueness_tree_hash", 0, edges, symmetry_groups, 2);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);

        // Setup for unified
        v2::UnifiedHypergraph v2_hg;
        setup_unified_hypergraph(hg, v2_hg);

        // Create state with all edges
        v2::SparseBitset state_edges;
        for (v2::EdgeId e = 0; e < static_cast<v2::EdgeId>(edges); ++e) {
            state_edges.set(e, v2_hg.arena());
        }

        BENCHMARK_CODE([&]() {
            volatile uint64_t hash = v2_hg.compute_canonical_hash(state_edges);
            (void)hash;
        });
    }
}

BENCHMARK(v1_full_canonicalization_by_edges, "v1 full Canonicalizer by edge count (factorial worst case)") {
    // Only small graphs - v1 canonicalization is O(V!) worst case
    for (int edges : {2, 3, 4, 5, 6}) {
        BENCHMARK_PARAM("edges", edges);

        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("v1_full_canon", 0, edges, symmetry_groups, 2);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);

        Canonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            canonicalizer.canonicalize(hg);
        });
    }
}

BENCHMARK(unified_hash_scaling, "Unified hash computation scaling (should be polynomial)") {
    for (int edges : {5, 10, 20, 40, 80, 160, 320}) {
        BENCHMARK_PARAM("edges", edges);

        int symmetry_groups = std::max(1, edges / 4);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("unified_hash_scaling", 0, edges, symmetry_groups, 2);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed);

        v2::UnifiedHypergraph v2_hg;
        setup_unified_hypergraph(hg, v2_hg);

        v2::SparseBitset state_edges;
        for (v2::EdgeId e = 0; e < static_cast<v2::EdgeId>(edges); ++e) {
            state_edges.set(e, v2_hg.arena());
        }

        BENCHMARK_CODE([&]() {
            volatile uint64_t hash = v2_hg.compute_canonical_hash(state_edges);
            (void)hash;
        });
    }
}

BENCHMARK(v1_unified_isomorphism_check, "Compares time to check if two states are isomorphic") {
    for (int edges : {4, 6, 8, 10, 12}) {
        BENCHMARK_PARAM("edges", edges);

        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed1 = RandomHypergraphGenerator::compute_seed("v1_unified_iso_check1", 0, edges, symmetry_groups, 2);
        uint32_t seed2 = RandomHypergraphGenerator::compute_seed("v1_unified_iso_check2", 1, edges, symmetry_groups, 2);

        Hypergraph hg1 = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed1);
        Hypergraph hg2 = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed2);

        // unified setup
        v2::UnifiedHypergraph v2_hg1, v2_hg2;
        setup_unified_hypergraph(hg1, v2_hg1);
        setup_unified_hypergraph(hg2, v2_hg2);

        v2::SparseBitset state_edges1, state_edges2;
        for (v2::EdgeId e = 0; e < static_cast<v2::EdgeId>(edges); ++e) {
            state_edges1.set(e, v2_hg1.arena());
            state_edges2.set(e, v2_hg2.arena());
        }

        // unified: Just compare hashes (the benchmark we want to highlight)
        BENCHMARK_CODE([&]() {
            uint64_t h1 = v2_hg1.compute_canonical_hash(state_edges1);
            uint64_t h2 = v2_hg2.compute_canonical_hash(state_edges2);
            volatile bool eq = (h1 == h2);
            (void)eq;
        });
    }
}

BENCHMARK(v1_isomorphism_check_full_canonicalization, "v1 isomorphism check via full canonicalization") {
    // v1: Full canonicalization is expensive
    for (int edges : {2, 3, 4, 5, 6}) {
        BENCHMARK_PARAM("edges", edges);

        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed1 = RandomHypergraphGenerator::compute_seed("v1_iso_full1", 0, edges, symmetry_groups, 2);
        uint32_t seed2 = RandomHypergraphGenerator::compute_seed("v1_iso_full2", 1, edges, symmetry_groups, 2);

        Hypergraph hg1 = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed1);
        Hypergraph hg2 = RandomHypergraphGenerator::generate_symmetric(edges, symmetry_groups, 2, seed2);

        Canonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            auto c1 = canonicalizer.canonicalize(hg1);
            auto c2 = canonicalizer.canonicalize(hg2);
            // Compare canonical hashes (from CanonicalizationResult)
            volatile bool eq = (c1.canonical_form == c2.canonical_form);
            (void)eq;
        });
    }
}

BENCHMARK(v1_uniqueness_tree_by_symmetry, "v1 UniquenessTree performance by symmetry groups") {
    const int num_edges = 20;
    for (int symmetry_groups : {1, 2, 4, 5, 10, 20}) {
        BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("v1_ut_symmetry", 0, num_edges, symmetry_groups, 2);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(num_edges, symmetry_groups, 2, seed);
        auto global_edges = convert_to_global_edges(hg);
        auto adjacency_index = make_adjacency_index(global_edges);

        BENCHMARK_CODE([&]() {
            UniquenessTreeSet tree_set(global_edges, adjacency_index);
            volatile uint64_t hash = tree_set.canonical_hash();
            (void)hash;
        });
    }
}

BENCHMARK(unified_uniqueness_tree_by_symmetry, "Unified UniquenessTree performance by symmetry groups") {
    const int num_edges = 20;
    for (int symmetry_groups : {1, 2, 4, 5, 10, 20}) {
        BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("unified_ut_symmetry", 0, num_edges, symmetry_groups, 2);
        Hypergraph hg = RandomHypergraphGenerator::generate_symmetric(num_edges, symmetry_groups, 2, seed);

        v2::UnifiedHypergraph v2_hg;
        setup_unified_hypergraph(hg, v2_hg);

        v2::SparseBitset state_edges;
        for (v2::EdgeId e = 0; e < static_cast<v2::EdgeId>(num_edges); ++e) {
            state_edges.set(e, v2_hg.arena());
        }

        BENCHMARK_CODE([&]() {
            volatile uint64_t hash = v2_hg.compute_canonical_hash(state_edges);
            (void)hash;
        });
    }
}
