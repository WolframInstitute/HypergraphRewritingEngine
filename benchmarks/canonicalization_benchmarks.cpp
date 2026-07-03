// BENCHMARK_CATEGORY: Canonicalization

#include "benchmark_framework.hpp"
#include "random_hypergraph_generator.hpp"
#include <hypergraph/ir_canonicalization.hpp>
#include <hypergraph/hypergraph.hpp>
#include <algorithm>

using namespace hypergraph;
using namespace benchmark;

// =============================================================================
// Category A: Canonicalization Benchmarks (Single-threaded)
// =============================================================================
// These benchmarks use controlled-symmetry graphs to get predictable results

BENCHMARK(canonicalization_by_edge_count, "Measures canonicalization performance as graph size increases (arity=2)") {
    for (int edges : {2, 4, 6}) {
        BENCHMARK_PARAM("edges", edges);

        // Fixed arity=2, symmetry_groups = edges/2 for moderate complexity
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_by_edge_count", 0, edges, symmetry_groups, 3);
        auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(edges, symmetry_groups, 2, seed);
        IRCanonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            canonicalizer.canonicalize_edges(edge_list);
        });
    }
}

BENCHMARK(canonicalization_by_edge_count_arity3, "Measures canonicalization performance as graph size increases (arity=3, higher complexity)") {
    for (int edges : {2, 4, 6}) {
        BENCHMARK_PARAM("edges", edges);

        // Fixed arity=3, symmetry_groups = edges/2 for moderate complexity
        int symmetry_groups = std::max(1, edges / 2);
        uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_by_edge_count_arity3", 0, edges, symmetry_groups, 3);
        auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(edges, symmetry_groups, 3, seed);
        IRCanonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            canonicalizer.canonicalize_edges(edge_list);
        });
    }
}

BENCHMARK(canonicalization_by_symmetry, "Shows how graph symmetry affects canonicalization time") {
    const int num_edges = 12;
    for (int symmetry_groups : {1, 2, 3, 4, 6}) {
        BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

        uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_by_symmetry", 0, num_edges, symmetry_groups, 3);
        auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(num_edges, symmetry_groups, 2, seed);
        IRCanonicalizer canonicalizer;

        BENCHMARK_CODE([&]() {
            canonicalizer.canonicalize_edges(edge_list);
        });
    }
}

// =============================================================================
// IR exact vs WL approximate: same states, head-to-head
// =============================================================================
// Times Hypergraph::compute_canonical_hash on identical states with the WL
// approximate hash (shared tree on) and IR exact canonicalization (shared tree
// off). "low" symmetry = each edge its own orbit (refinement discretizes in one
// pass, so IR does no individualization); "high" symmetry = few orbits (WL
// refinement stabilizes non-discrete, forcing IR to individualize and branch).
// This exposes where the two algorithms diverge in cost, and drives the decision
// on whether WL is worth keeping alongside IR.
BENCHMARK(canon_ir_vs_wl, "IR exact vs WL approximate canonical hash on identical states") {
    for (int edges : {10, 20, 50, 100}) {
        for (const char* sym : {"low", "high"}) {
            int symmetry_groups = (std::string(sym) == "low") ? edges : 2;
            uint32_t seed = RandomHypergraphGenerator::compute_seed("canon_ir_vs_wl", 0, edges, symmetry_groups, 2);
            auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(edges, symmetry_groups, 2, seed);

            // Build the hypergraph + edge bitset once, outside the timed region.
            Hypergraph hg;
            VertexId maxv = 0;
            for (const auto& e : edge_list)
                for (VertexId v : e) maxv = std::max(maxv, v);
            for (VertexId i = 0; i <= maxv; ++i) hg.alloc_vertex();
            SparseBitset bitset;
            for (const auto& e : edge_list) {
                EdgeId eid = hg.create_edge(e.data(), static_cast<uint8_t>(e.size()));
                bitset.set(eid, hg.arena());
            }

            BENCHMARK_PARAM("edges", edges);
            BENCHMARK_PARAM("symmetry", sym);
            BENCHMARK_PARAM("engine", "wl");
            hg.enable_shared_tree();
            BENCHMARK_CODE([&]() {
                volatile uint64_t h = hg.compute_canonical_hash(bitset);
                (void)h;
            });

            BENCHMARK_PARAM("edges", edges);
            BENCHMARK_PARAM("symmetry", sym);
            BENCHMARK_PARAM("engine", "ir");
            hg.disable_shared_tree();
            BENCHMARK_CODE([&]() {
                volatile uint64_t h = hg.compute_canonical_hash(bitset);
                (void)h;
            });
        }
    }
}

BENCHMARK(canonicalization_2d_sweep, "2D parameter sweep: edges vs symmetry_groups for surface plots") {
    for (int edges : {2, 3, 4, 5, 6}) {
        for (int symmetry_groups : {1, 2, 3, 4, 5, 6}) {
            // Skip invalid combinations
            if (symmetry_groups > edges) continue;

            BENCHMARK_PARAM("edges", edges);
            BENCHMARK_PARAM("symmetry_groups", symmetry_groups);

            // Generate single graph - capture by value to avoid dangling reference
            uint32_t seed = RandomHypergraphGenerator::compute_seed("canonicalization_2d_sweep", 0, edges, symmetry_groups, 0);
            auto edge_list = RandomHypergraphGenerator::generate_symmetric_edges(edges, symmetry_groups, 2, seed);

            BENCHMARK_CODE([&]() {
                IRCanonicalizer canonicalizer;
                canonicalizer.canonicalize_edges(edge_list);
            });
        }
    }
}
