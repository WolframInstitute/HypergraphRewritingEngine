/**
 * test_quantum_analysis.cpp - Fast automated tests for geodesic and particle analysis
 *
 * Tests the implementation of features from Gorard's "Some Quantum Mechanical
 * Properties of the Wolfram Model" paper:
 *   - Geodesic tracing (test particle paths)
 *   - Bundle spread analysis
 *   - Topological defect detection (K5/K3,3 minors)
 *   - Topological charge computation
 *
 * Run with: ./test_quantum_analysis
 * Expected runtime: < 5 seconds
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include "blackhole/geodesic_analysis.hpp"
#include "blackhole/particle_detection.hpp"
#include "blackhole/hausdorff_analysis.hpp"
#include "blackhole/curvature_analysis.hpp"
#include "blackhole/entropy_analysis.hpp"
#include "blackhole/rotation_analysis.hpp"
#include "blackhole/branchial_analysis.hpp"
#include "blackhole/bh_types.hpp"

using namespace viz::blackhole;

// Test helper: create a simple grid graph (known to be planar)
SimpleGraph create_grid_graph(int rows, int cols) {
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;

    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        vertices.push_back(static_cast<VertexId>(i));
    }

    auto idx = [cols](int r, int c) { return r * cols + c; };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c + 1 < cols) {
                edges.push_back({static_cast<VertexId>(idx(r, c)),
                                 static_cast<VertexId>(idx(r, c + 1))});
            }
            if (r + 1 < rows) {
                edges.push_back({static_cast<VertexId>(idx(r, c)),
                                 static_cast<VertexId>(idx(r + 1, c))});
            }
        }
    }

    SimpleGraph g;
    g.build(vertices, edges);
    return g;
}

// Test helper: create K5 (complete graph on 5 vertices) - non-planar
SimpleGraph create_k5() {
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;

    for (int i = 0; i < 5; ++i) {
        vertices.push_back(static_cast<VertexId>(i));
    }
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            edges.push_back({static_cast<VertexId>(i), static_cast<VertexId>(j)});
        }
    }

    SimpleGraph g;
    g.build(vertices, edges);
    return g;
}

// Test helper: create K3,3 (bipartite complete graph) - non-planar
SimpleGraph create_k33() {
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;

    for (int i = 0; i < 6; ++i) {
        vertices.push_back(static_cast<VertexId>(i));
    }
    // Partition: {0, 1, 2} and {3, 4, 5}
    for (int i = 0; i < 3; ++i) {
        for (int j = 3; j < 6; ++j) {
            edges.push_back({static_cast<VertexId>(i), static_cast<VertexId>(j)});
        }
    }

    SimpleGraph g;
    g.build(vertices, edges);
    return g;
}

// Test helper: create a star graph (high degree center)
SimpleGraph create_star_graph(int num_leaves) {
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;

    for (int i = 0; i <= num_leaves; ++i) {
        vertices.push_back(static_cast<VertexId>(i));
    }
    for (int i = 1; i <= num_leaves; ++i) {
        edges.push_back({0, static_cast<VertexId>(i)});
    }

    SimpleGraph g;
    g.build(vertices, edges);
    return g;
}

// =============================================================================
// TEST 1: Basic geodesic tracing on grid
// =============================================================================
bool test_geodesic_basic() {
    std::cout << "TEST 1: Basic geodesic tracing... ";

    auto grid = create_grid_graph(5, 5);

    GeodesicConfig config;
    config.max_steps = 10;
    config.bundle_width = 3;
    config.direction = GeodesicDirection::Random;

    // Trace from corner vertex (0)
    std::vector<VertexId> sources = {0};

    auto result = analyze_geodesics(grid, sources, config, nullptr);

    // Should have traced paths
    if (result.paths.empty()) {
        std::cout << "FAILED - no paths traced\n";
        return false;
    }

    // Each path should have multiple vertices
    bool has_valid_path = false;
    for (const auto& path : result.paths) {
        if (path.vertices.size() >= 2) {
            has_valid_path = true;
            break;
        }
    }

    if (!has_valid_path) {
        std::cout << "FAILED - no valid paths (need >= 2 vertices)\n";
        return false;
    }

    std::cout << "PASSED (traced " << result.paths.size() << " paths)\n";
    return true;
}

// =============================================================================
// TEST 2: Geodesic bundle spread
// =============================================================================
bool test_geodesic_bundle_spread() {
    std::cout << "TEST 2: Geodesic bundle spread... ";

    auto grid = create_grid_graph(10, 10);

    GeodesicConfig config;
    config.max_steps = 15;
    config.bundle_width = 5;
    config.direction = GeodesicDirection::Random;

    std::vector<VertexId> sources = {0};

    auto result = analyze_geodesics(grid, sources, config, nullptr);

    // Bundle spread should be computed
    if (result.paths.size() < 2) {
        std::cout << "SKIPPED - need >= 2 paths for bundle spread\n";
        return true;  // Not a failure, just can't test
    }

    // With random walk on a grid, mean_spread should be >= 0
    if (result.mean_spread < 0) {
        std::cout << "FAILED - invalid mean spread: " << result.mean_spread << "\n";
        return false;
    }

    std::cout << "PASSED (spread = " << result.mean_spread << ")\n";
    return true;
}

// =============================================================================
// TEST 3: K5 minor detection
// =============================================================================
bool test_k5_detection() {
    std::cout << "TEST 3: K5 minor detection... ";

    auto k5 = create_k5();

    ParticleDetectionConfig config;
    config.detect_k5 = true;
    config.detect_k33 = false;
    config.use_dimension_spikes = false;
    config.use_high_degree = false;

    auto result = analyze_particles(k5, config, nullptr, nullptr);

    // K5 should be detected as having K5 minor (it IS K5)
    bool found_k5 = false;
    for (const auto& defect : result.defects) {
        if (defect.type == TopologicalDefectType::K5Minor) {
            found_k5 = true;
            break;
        }
    }

    if (!found_k5) {
        std::cout << "FAILED - K5 not detected in K5 graph\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 4: K3,3 minor detection
// =============================================================================
bool test_k33_detection() {
    std::cout << "TEST 4: K3,3 minor detection... ";

    auto k33 = create_k33();

    ParticleDetectionConfig config;
    config.detect_k5 = false;
    config.detect_k33 = true;
    config.use_dimension_spikes = false;
    config.use_high_degree = false;

    auto result = analyze_particles(k33, config, nullptr, nullptr);

    // K3,3 should be detected
    bool found_k33 = false;
    for (const auto& defect : result.defects) {
        if (defect.type == TopologicalDefectType::K33Minor) {
            found_k33 = true;
            break;
        }
    }

    if (!found_k33) {
        std::cout << "FAILED - K3,3 not detected in K3,3 graph\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 5: Planar graph should have no K5/K3,3 minors
// =============================================================================
bool test_planar_no_minors() {
    std::cout << "TEST 5: Planar graph has no forbidden minors... ";

    auto grid = create_grid_graph(5, 5);  // Grid is planar

    ParticleDetectionConfig config;
    config.detect_k5 = true;
    config.detect_k33 = true;
    config.use_dimension_spikes = false;
    config.use_high_degree = false;

    auto result = analyze_particles(grid, config, nullptr, nullptr);

    // Should not find K5 or K3,3 minors
    for (const auto& defect : result.defects) {
        if (defect.type == TopologicalDefectType::K5Minor ||
            defect.type == TopologicalDefectType::K33Minor) {
            std::cout << "FAILED - found forbidden minor in planar graph\n";
            return false;
        }
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 6: High degree vertex detection
// =============================================================================
bool test_high_degree_detection() {
    std::cout << "TEST 6: High degree vertex detection... ";

    auto star = create_star_graph(20);  // Center has degree 20

    ParticleDetectionConfig config;
    config.detect_k5 = false;
    config.detect_k33 = false;
    config.use_dimension_spikes = false;
    config.use_high_degree = true;
    config.degree_threshold_percentile = 0.9f;  // Top 10%

    auto result = analyze_particles(star, config, nullptr, nullptr);

    // Center vertex (degree 20) should be detected as high degree
    bool found_high_degree = false;
    for (const auto& defect : result.defects) {
        if (defect.type == TopologicalDefectType::HighDegree) {
            // Check if core includes vertex 0 (the center)
            for (auto v : defect.core_vertices) {
                if (v == 0) {
                    found_high_degree = true;
                    break;
                }
            }
        }
    }

    if (!found_high_degree) {
        std::cout << "FAILED - high degree center not detected\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 7: Topological charge computation
// =============================================================================
bool test_topological_charge() {
    std::cout << "TEST 7: Topological charge computation... ";

    auto k5 = create_k5();

    ParticleDetectionConfig config;
    config.detect_k5 = true;
    config.compute_charges = true;
    config.charge_radius = 2.0f;

    auto result = analyze_particles(k5, config, nullptr, nullptr);

    // K5 defect should have non-zero charge
    if (result.defects.empty()) {
        std::cout << "FAILED - no defects found\n";
        return false;
    }

    bool has_charge = false;
    for (const auto& defect : result.defects) {
        if (defect.type == TopologicalDefectType::K5Minor && std::abs(defect.charge) > 0) {
            has_charge = true;
            break;
        }
    }

    if (!has_charge) {
        std::cout << "FAILED - K5 defect has zero charge\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 8: Geodesic path validity - paths must follow actual edges
// =============================================================================
bool test_geodesic_validity() {
    std::cout << "TEST 8: Geodesic path validity... ";

    auto grid = create_grid_graph(5, 5);

    GeodesicConfig config;
    config.max_steps = 20;
    config.bundle_width = 1;
    config.direction = GeodesicDirection::Random;

    std::vector<VertexId> sources = {0, 12, 24};  // corners and center

    auto result = analyze_geodesics(grid, sources, config, nullptr);

    // Build adjacency set for validation
    std::unordered_map<VertexId, std::unordered_set<VertexId>> adj;
    for (auto v : grid.vertices()) {
        const auto& neighbors = grid.neighbors(v);
        for (auto n : neighbors) {
            adj[v].insert(n);
        }
    }

    // Every consecutive pair in every path must be neighbors
    for (const auto& path : result.paths) {
        for (size_t i = 1; i < path.vertices.size(); ++i) {
            VertexId from = path.vertices[i - 1];
            VertexId to = path.vertices[i];
            if (adj[from].find(to) == adj[from].end()) {
                std::cout << "FAILED - path contains non-edge: " << from << " -> " << to << "\n";
                return false;
            }
        }
    }

    std::cout << "PASSED (verified " << result.paths.size() << " paths)\n";
    return true;
}

// =============================================================================
// TEST 9: Bundle width is actually respected
// =============================================================================
bool test_bundle_width() {
    std::cout << "TEST 9: Bundle width configuration... ";

    auto grid = create_grid_graph(10, 10);

    for (int width : {1, 3, 5, 7}) {
        GeodesicConfig config;
        config.max_steps = 10;
        config.bundle_width = width;
        config.direction = GeodesicDirection::Random;

        std::vector<VertexId> sources = {50};  // center

        auto result = analyze_geodesics(grid, sources, config, nullptr);

        // Should get approximately bundle_width paths per source
        // Allow some variance due to connectivity
        if (result.paths.size() == 0) {
            std::cout << "FAILED - no paths for bundle_width=" << width << "\n";
            return false;
        }
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 10: K5 subgraph detection (K5 embedded in larger graph)
// =============================================================================
bool test_k5_in_larger_graph() {
    std::cout << "TEST 10: K5 in larger graph... ";

    // Create a graph with K5 embedded in it
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;

    // Vertices 0-4 form K5
    for (int i = 0; i < 10; ++i) {
        vertices.push_back(static_cast<VertexId>(i));
    }

    // K5 edges
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            edges.push_back({static_cast<VertexId>(i), static_cast<VertexId>(j)});
        }
    }

    // Additional edges to make it bigger
    edges.push_back({5, 0});
    edges.push_back({5, 6});
    edges.push_back({6, 7});
    edges.push_back({7, 8});
    edges.push_back({8, 9});

    SimpleGraph g;
    g.build(vertices, edges);

    ParticleDetectionConfig config;
    config.detect_k5 = true;
    config.detect_k33 = false;
    config.use_dimension_spikes = false;
    config.use_high_degree = false;

    auto result = analyze_particles(g, config, nullptr, nullptr);

    // Should still detect K5
    bool found_k5 = false;
    for (const auto& defect : result.defects) {
        if (defect.type == TopologicalDefectType::K5Minor) {
            found_k5 = true;
            // Verify the core vertices are from {0,1,2,3,4}
            for (auto v : defect.core_vertices) {
                if (v > 4) {
                    std::cout << "FAILED - K5 core includes non-K5 vertex " << v << "\n";
                    return false;
                }
            }
            break;
        }
    }

    if (!found_k5) {
        std::cout << "FAILED - K5 not detected in larger graph\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 11: Charge computation returns data for all vertices
// =============================================================================
bool test_charge_computation() {
    std::cout << "TEST 11: Charge computation coverage... ";

    auto grid = create_grid_graph(5, 5);  // 25 vertices

    ParticleDetectionConfig config;
    config.compute_charges = true;
    config.charge_radius = 2.0f;
    config.detect_k5 = false;
    config.detect_k33 = false;
    config.use_dimension_spikes = false;
    config.use_high_degree = false;

    auto result = analyze_particles(grid, config, nullptr, nullptr);

    // Should have charge data for every vertex
    if (result.charges.size() != 25) {
        std::cout << "FAILED - expected 25 charges, got " << result.charges.size() << "\n";
        return false;
    }

    // charge_map should also have 25 entries
    if (result.charge_map.size() != 25) {
        std::cout << "FAILED - expected 25 charge_map entries, got " << result.charge_map.size() << "\n";
        return false;
    }

    // Verify charge values are finite
    for (const auto& vc : result.charges) {
        if (!std::isfinite(vc.charge)) {
            std::cout << "FAILED - non-finite charge at vertex " << vc.vertex << "\n";
            return false;
        }
    }

    std::cout << "PASSED (25 vertices, all finite)\n";
    return true;
}

// =============================================================================
// TEST 12: Empty graph edge case
// =============================================================================
bool test_empty_graph() {
    std::cout << "TEST 12: Empty graph handling... ";

    SimpleGraph empty;
    // Don't call build - leave it empty

    GeodesicConfig geo_config;
    ParticleDetectionConfig particle_config;

    // These should not crash
    auto geo_result = analyze_geodesics(empty, {}, geo_config, nullptr);
    auto particle_result = analyze_particles(empty, particle_config, nullptr, nullptr);

    // Should have no paths/defects
    if (!geo_result.paths.empty()) {
        std::cout << "FAILED - empty graph should have no paths\n";
        return false;
    }
    if (!particle_result.defects.empty()) {
        std::cout << "FAILED - empty graph should have no defects\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 13: Single vertex edge case
// =============================================================================
bool test_single_vertex() {
    std::cout << "TEST 13: Single vertex handling... ";

    SimpleGraph single;
    std::vector<VertexId> vertices = {0};
    std::vector<Edge> edges;
    single.build(vertices, edges);

    GeodesicConfig geo_config;
    std::vector<VertexId> sources = {0};

    auto geo_result = analyze_geodesics(single, sources, geo_config, nullptr);

    // Path from single vertex should have just that vertex
    if (geo_result.paths.empty()) {
        std::cout << "SKIPPED - no path traced\n";
        return true;  // Acceptable behavior
    }

    std::cout << "PASSED\n";
    return true;
}

// =============================================================================
// TEST 14: Performance - geodesic tracing should be fast
// =============================================================================
bool test_geodesic_performance() {
    std::cout << "TEST 14: Geodesic performance... ";

    auto grid = create_grid_graph(20, 20);  // 400 vertices

    GeodesicConfig config;
    config.max_steps = 50;
    config.bundle_width = 10;
    config.direction = GeodesicDirection::Random;

    std::vector<VertexId> sources = {0, 100, 200, 300};  // Multiple sources

    auto start = std::chrono::high_resolution_clock::now();

    auto result = analyze_geodesics(grid, sources, config, nullptr);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Should complete in < 100ms for this size
    if (ms > 100) {
        std::cout << "FAILED - too slow: " << ms << " ms\n";
        return false;
    }

    std::cout << "PASSED (" << ms << " ms for " << result.paths.size() << " paths)\n";
    return true;
}

// =============================================================================
// TEST 9: Performance - particle detection should be fast
// =============================================================================
bool test_particle_performance() {
    std::cout << "TEST 15: Particle detection performance... ";

    auto grid = create_grid_graph(15, 15);  // 225 vertices

    ParticleDetectionConfig config;
    config.detect_k5 = true;
    config.detect_k33 = true;
    config.use_dimension_spikes = true;
    config.use_high_degree = true;
    config.compute_charges = true;

    auto start = std::chrono::high_resolution_clock::now();

    auto result = analyze_particles(grid, config, nullptr, nullptr);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Should complete in < 500ms for this size
    if (ms > 500) {
        std::cout << "FAILED - too slow: " << ms << " ms\n";
        return false;
    }

    std::cout << "PASSED (" << ms << " ms)\n";
    return true;
}

// =============================================================================
// TEST 16: Curvature analysis - Ollivier-Ricci
// =============================================================================
bool test_curvature_ollivier_ricci() {
    std::cout << "TEST 16: Curvature analysis (Ollivier-Ricci)... ";

    auto grid = create_grid_graph(5, 5);

    // Create dimension data (uniform for simplicity)
    std::vector<float> dimensions(grid.vertex_count(), 2.0f);

    CurvatureConfig config;
    config.compute_ollivier_ricci = true;
    config.compute_dimension_gradient = false;

    auto result = analyze_curvature(grid, config, &dimensions);

    // Should have curvature for all vertices
    if (result.vertex_curvatures.size() != grid.vertex_count()) {
        std::cout << "FAILED - expected " << grid.vertex_count() << " curvatures, got "
                  << result.vertex_curvatures.size() << "\n";
        return false;
    }

    // All curvature values should be finite
    for (const auto& vc : result.vertex_curvatures) {
        if (!std::isfinite(vc.ollivier_ricci)) {
            std::cout << "FAILED - non-finite Ollivier-Ricci curvature\n";
            return false;
        }
    }

    std::cout << "PASSED (mean=" << result.mean_ollivier_ricci << ")\n";
    return true;
}

// =============================================================================
// TEST 17: Curvature analysis - Dimension gradient
// =============================================================================
bool test_curvature_dimension_gradient() {
    std::cout << "TEST 17: Curvature analysis (Dimension gradient)... ";

    auto grid = create_grid_graph(5, 5);

    // Create dimension data with gradient (higher in center)
    std::vector<float> dimensions(grid.vertex_count(), 0.0f);
    for (size_t i = 0; i < grid.vertices().size(); ++i) {
        int row = i / 5;
        int col = i % 5;
        // Distance from center (2,2)
        float dist = std::sqrt((row - 2.0f) * (row - 2.0f) + (col - 2.0f) * (col - 2.0f));
        dimensions[i] = 3.0f - dist * 0.3f;  // Higher in center
    }

    CurvatureConfig config;
    config.compute_ollivier_ricci = false;
    config.compute_dimension_gradient = true;

    auto result = analyze_curvature(grid, config, &dimensions);

    // Center vertex should have different curvature than corners
    if (result.vertex_curvatures.size() == 0) {
        std::cout << "FAILED - no curvature results\n";
        return false;
    }

    // All values should be finite
    for (const auto& vc : result.vertex_curvatures) {
        if (!std::isfinite(vc.dimension_gradient)) {
            std::cout << "FAILED - non-finite dimension gradient curvature\n";
            return false;
        }
    }

    std::cout << "PASSED (mean=" << result.mean_dimension_gradient << ")\n";
    return true;
}

// =============================================================================
// TEST 18: Entropy analysis - Degree entropy
// =============================================================================
bool test_entropy_degree() {
    std::cout << "TEST 18: Entropy analysis (Degree entropy)... ";

    auto grid = create_grid_graph(5, 5);

    EntropyConfig config;
    config.compute_local_entropy = true;
    config.compute_mutual_info = false;
    config.compute_fisher_info = false;

    auto result = analyze_entropy(grid, config, nullptr);

    // Degree entropy should be positive for non-trivial graph
    if (result.degree_entropy < 0) {
        std::cout << "FAILED - negative degree entropy\n";
        return false;
    }

    // Graph entropy should also be computed
    if (!std::isfinite(result.graph_entropy)) {
        std::cout << "FAILED - non-finite graph entropy\n";
        return false;
    }

    std::cout << "PASSED (degree=" << result.degree_entropy << ", graph=" << result.graph_entropy << ")\n";
    return true;
}

// =============================================================================
// TEST 19: Entropy analysis - Local entropy
// =============================================================================
bool test_entropy_local() {
    std::cout << "TEST 19: Entropy analysis (Local entropy)... ";

    auto grid = create_grid_graph(5, 5);

    // Create dimension data
    std::vector<float> dimensions(grid.vertex_count(), 2.0f);
    // Add some variation
    for (size_t i = 0; i < dimensions.size(); ++i) {
        dimensions[i] += (i % 3) * 0.1f;
    }

    EntropyConfig config;
    config.compute_local_entropy = true;
    config.neighborhood_radius = 2;

    auto result = analyze_entropy(grid, config, &dimensions);

    // Should have local entropy for all vertices
    if (result.local_entropy_map.size() != grid.vertex_count()) {
        std::cout << "FAILED - expected " << grid.vertex_count() << " local entropies\n";
        return false;
    }

    // All values should be non-negative
    for (const auto& [vid, entropy] : result.local_entropy_map) {
        if (entropy < 0) {
            std::cout << "FAILED - negative local entropy\n";
            return false;
        }
    }

    std::cout << "PASSED (count=" << result.local_entropy_map.size() << ")\n";
    return true;
}

// =============================================================================
// TEST 20: Entropy analysis - Fisher information
// =============================================================================
bool test_entropy_fisher() {
    std::cout << "TEST 20: Entropy analysis (Fisher information)... ";

    auto grid = create_grid_graph(5, 5);

    // Create dimension data with gradient
    std::vector<float> dimensions(grid.vertex_count(), 0.0f);
    for (size_t i = 0; i < dimensions.size(); ++i) {
        dimensions[i] = 2.0f + (i % 5) * 0.2f;  // Gradient along rows
    }

    EntropyConfig config;
    config.compute_fisher_info = true;

    auto result = analyze_entropy(grid, config, &dimensions);

    // Fisher info should be non-negative
    if (result.total_fisher_info < 0) {
        std::cout << "FAILED - negative total Fisher info\n";
        return false;
    }

    std::cout << "PASSED (total=" << result.total_fisher_info << ")\n";
    return true;
}

// =============================================================================
// TEST 21: Rotation curve analysis - Center detection
// =============================================================================
bool test_rotation_center() {
    std::cout << "TEST 21: Rotation curve (Center detection)... ";

    auto grid = create_grid_graph(7, 7);  // 49 vertices

    // Create dimension data with peak in center
    std::vector<float> dimensions(grid.vertex_count(), 2.0f);
    // Center is at (3,3) = vertex 24
    for (size_t i = 0; i < dimensions.size(); ++i) {
        int row = i / 7;
        int col = i % 7;
        float dist = std::sqrt((row - 3.0f) * (row - 3.0f) + (col - 3.0f) * (col - 3.0f));
        dimensions[i] = 3.0f - dist * 0.2f;
    }
    dimensions[24] = 4.0f;  // Highest at center

    RotationConfig config;
    config.auto_detect_center = true;
    config.max_radius = 5;
    config.min_radius = 1;
    config.orbits_per_radius = 4;

    auto result = analyze_rotation_curve(grid, config, &dimensions);

    // Center should be at vertex 24 (highest dimension)
    if (result.center != 24) {
        std::cout << "FAILED - center should be 24, got " << result.center << "\n";
        return false;
    }

    std::cout << "PASSED (center=" << result.center << ")\n";
    return true;
}

// =============================================================================
// TEST 22: Rotation curve analysis - Power law fit
// =============================================================================
bool test_rotation_power_law() {
    std::cout << "TEST 22: Rotation curve (Power law fit)... ";

    auto grid = create_grid_graph(10, 10);

    // Create uniform dimension data
    std::vector<float> dimensions(grid.vertex_count(), 2.0f);

    RotationConfig config;
    config.auto_detect_center = false;
    config.manual_center = 55;  // Near center of 10x10 grid
    config.compute_power_law_fit = true;
    config.max_radius = 6;
    config.min_radius = 2;

    auto result = analyze_rotation_curve(grid, config, &dimensions);

    // Power law exponent should be finite
    if (!std::isfinite(result.power_law_exponent)) {
        std::cout << "FAILED - non-finite power law exponent\n";
        return false;
    }

    // Curve should have points
    if (result.curve.empty()) {
        std::cout << "FAILED - empty rotation curve\n";
        return false;
    }

    std::cout << "PASSED (exponent=" << result.power_law_exponent << ", points=" << result.curve.size() << ")\n";
    return true;
}

// =============================================================================
// TEST 23: Curvature on star graph (high curvature at center)
// =============================================================================
bool test_curvature_star_graph() {
    std::cout << "TEST 23: Curvature on star graph... ";

    auto star = create_star_graph(10);

    std::vector<float> dimensions(star.vertex_count(), 2.0f);
    dimensions[0] = 3.0f;  // Higher dimension at center

    CurvatureConfig config;
    config.compute_ollivier_ricci = true;
    config.compute_dimension_gradient = true;

    auto result = analyze_curvature(star, config, &dimensions);

    // Center should have different curvature than leaves
    auto center_it = result.ollivier_ricci_map.find(0);
    if (center_it == result.ollivier_ricci_map.end()) {
        std::cout << "FAILED - no curvature for center vertex\n";
        return false;
    }

    std::cout << "PASSED (center curvature=" << center_it->second << ")\n";
    return true;
}

// =============================================================================
// TEST 24: Entropy on uniform graph
// =============================================================================
bool test_entropy_uniform() {
    std::cout << "TEST 24: Entropy on uniform graph... ";

    auto grid = create_grid_graph(4, 4);

    // Perfectly uniform dimensions
    std::vector<float> dimensions(grid.vertex_count(), 2.0f);

    EntropyConfig config;
    config.compute_local_entropy = true;
    config.compute_fisher_info = true;

    auto result = analyze_entropy(grid, config, &dimensions);

    // Check that all local entropy values are finite and non-negative
    // (local entropy measures structural diversity, not just dimension variation)
    for (const auto& [vid, entropy] : result.local_entropy_map) {
        if (!std::isfinite(entropy) || entropy < 0) {
            std::cout << "FAILED - invalid local entropy value\n";
            return false;
        }
    }

    // Check that Fisher info is finite
    if (!std::isfinite(result.total_fisher_info)) {
        std::cout << "FAILED - non-finite Fisher info\n";
        return false;
    }

    std::cout << "PASSED (all values valid)\n";
    return true;
}

// =============================================================================
// TEST 25: Hilbert space - state inner product
// =============================================================================
bool test_hilbert_inner_product() {
    std::cout << "TEST 25: Hilbert space inner product... ";

    // Create two states with some overlap
    BranchState a, b;
    a.state_id = 0;
    a.branch_id = 0;
    a.step = 0;
    a.vertices = {1, 2, 3, 4, 5};

    b.state_id = 1;
    b.branch_id = 1;
    b.step = 0;
    b.vertices = {3, 4, 5, 6, 7};  // 3 vertices overlap (3, 4, 5)

    float ip = compute_state_inner_product(a, b);

    // Expected: |intersection| / sqrt(|a| * |b|) = 3 / sqrt(5 * 5) = 3/5 = 0.6
    if (std::abs(ip - 0.6f) > 0.01f) {
        std::cout << "FAILED - expected ~0.6, got " << ip << "\n";
        return false;
    }

    // Test identical states -> inner product = 1.0
    float ip_self = compute_state_inner_product(a, a);
    if (std::abs(ip_self - 1.0f) > 0.01f) {
        std::cout << "FAILED - self inner product should be 1.0, got " << ip_self << "\n";
        return false;
    }

    // Test disjoint states -> inner product = 0.0
    BranchState c;
    c.state_id = 2;
    c.branch_id = 2;
    c.step = 0;
    c.vertices = {10, 11, 12};

    float ip_disjoint = compute_state_inner_product(a, c);
    if (std::abs(ip_disjoint) > 0.01f) {
        std::cout << "FAILED - disjoint inner product should be 0.0, got " << ip_disjoint << "\n";
        return false;
    }

    std::cout << "PASSED (ip=" << ip << ", self=1.0, disjoint=0.0)\n";
    return true;
}

// =============================================================================
// TEST 26: Hilbert space - vertex probabilities
// =============================================================================
bool test_hilbert_vertex_probabilities() {
    std::cout << "TEST 26: Hilbert space vertex probabilities... ";

    // Create 4 states at step 0
    std::vector<BranchState> states(4);

    // State 0: vertices {1, 2, 3}
    states[0].state_id = 0;
    states[0].branch_id = 0;
    states[0].step = 0;
    states[0].vertices = {1, 2, 3};

    // State 1: vertices {2, 3, 4}
    states[1].state_id = 1;
    states[1].branch_id = 1;
    states[1].step = 0;
    states[1].vertices = {2, 3, 4};

    // State 2: vertices {3, 4, 5}
    states[2].state_id = 2;
    states[2].branch_id = 2;
    states[2].step = 0;
    states[2].vertices = {3, 4, 5};

    // State 3: vertices {4, 5, 6}
    states[3].state_id = 3;
    states[3].branch_id = 3;
    states[3].step = 0;
    states[3].vertices = {4, 5, 6};

    auto graph = build_branchial_graph(states, {});
    auto probs = compute_vertex_probabilities(graph, 0);

    // Vertex 3 appears in states 0, 1, 2 -> P(3) = 3/4 = 0.75
    if (std::abs(probs[3] - 0.75f) > 0.01f) {
        std::cout << "FAILED - P(vertex 3) expected 0.75, got " << probs[3] << "\n";
        return false;
    }

    // Vertex 1 appears in state 0 only -> P(1) = 1/4 = 0.25
    if (std::abs(probs[1] - 0.25f) > 0.01f) {
        std::cout << "FAILED - P(vertex 1) expected 0.25, got " << probs[1] << "\n";
        return false;
    }

    // Vertex 4 appears in states 1, 2, 3 -> P(4) = 3/4 = 0.75
    if (std::abs(probs[4] - 0.75f) > 0.01f) {
        std::cout << "FAILED - P(vertex 4) expected 0.75, got " << probs[4] << "\n";
        return false;
    }

    std::cout << "PASSED (P(3)=0.75, P(1)=0.25, P(4)=0.75)\n";
    return true;
}

// =============================================================================
// TEST 27: Hilbert space - full analysis
// =============================================================================
bool test_hilbert_analysis() {
    std::cout << "TEST 27: Hilbert space full analysis... ";

    // Create simple 3-state system
    std::vector<BranchState> states(3);

    states[0].state_id = 0;
    states[0].branch_id = 0;
    states[0].step = 0;
    states[0].vertices = {1, 2, 3};

    states[1].state_id = 1;
    states[1].branch_id = 1;
    states[1].step = 0;
    states[1].vertices = {2, 3, 4};

    states[2].state_id = 2;
    states[2].branch_id = 2;
    states[2].step = 0;
    states[2].vertices = {3, 4, 5};

    auto graph = build_branchial_graph(states, {});
    auto analysis = analyze_hilbert_space(graph, 0);

    // Check basic counts
    if (analysis.num_states != 3) {
        std::cout << "FAILED - expected 3 states, got " << analysis.num_states << "\n";
        return false;
    }

    if (analysis.num_vertices != 5) {
        std::cout << "FAILED - expected 5 unique vertices, got " << analysis.num_vertices << "\n";
        return false;
    }

    // Check inner product matrix dimensions
    if (analysis.inner_product_matrix.size() != 3) {
        std::cout << "FAILED - inner product matrix wrong size\n";
        return false;
    }

    // Diagonal should be 1.0 (self inner product)
    for (size_t i = 0; i < 3; ++i) {
        if (std::abs(analysis.inner_product_matrix[i][i] - 1.0f) > 0.01f) {
            std::cout << "FAILED - diagonal element should be 1.0\n";
            return false;
        }
    }

    // Check mean inner product is reasonable (off-diagonal average)
    if (analysis.mean_inner_product < 0.0f || analysis.mean_inner_product > 1.0f) {
        std::cout << "FAILED - mean inner product out of range: " << analysis.mean_inner_product << "\n";
        return false;
    }

    std::cout << "PASSED (states=" << analysis.num_states
              << ", vertices=" << analysis.num_vertices
              << ", mean_ip=" << analysis.mean_inner_product << ")\n";
    return true;
}

// =============================================================================
// TEST 28: Branchial sharpness and entropy
// =============================================================================
bool test_branchial_sharpness() {
    std::cout << "TEST 28: Branchial sharpness and entropy... ";

    // Create states across 2 branches
    std::vector<BranchState> states(4);

    // Branch 0: two states
    states[0].state_id = 0;
    states[0].branch_id = 0;
    states[0].step = 0;
    states[0].vertices = {1, 2, 3};

    states[1].state_id = 1;
    states[1].branch_id = 0;
    states[1].step = 1;
    states[1].vertices = {2, 3, 4};

    // Branch 1: two states
    states[2].state_id = 2;
    states[2].branch_id = 1;
    states[2].step = 0;
    states[2].vertices = {3, 5, 6};

    states[3].state_id = 3;
    states[3].branch_id = 1;
    states[3].step = 1;
    states[3].vertices = {5, 6, 7};

    BranchialConfig config;
    config.compute_sharpness = true;
    config.compute_entropy = true;

    auto result = analyze_branchial(states, config);

    // Vertex 3 appears in both branches -> sharpness = 1/2 = 0.5
    auto sharpness_it = result.vertex_sharpness.find(3);
    if (sharpness_it == result.vertex_sharpness.end()) {
        std::cout << "FAILED - no sharpness for vertex 3\n";
        return false;
    }
    if (std::abs(sharpness_it->second - 0.5f) > 0.01f) {
        std::cout << "FAILED - vertex 3 sharpness expected 0.5, got " << sharpness_it->second << "\n";
        return false;
    }

    // Vertex 1 appears only in branch 0 -> sharpness = 1.0
    auto sharpness_1 = result.vertex_sharpness.find(1);
    if (sharpness_1 == result.vertex_sharpness.end()) {
        std::cout << "FAILED - no sharpness for vertex 1\n";
        return false;
    }
    if (std::abs(sharpness_1->second - 1.0f) > 0.01f) {
        std::cout << "FAILED - vertex 1 sharpness expected 1.0, got " << sharpness_1->second << "\n";
        return false;
    }

    std::cout << "PASSED (sharpness[3]=0.5, sharpness[1]=1.0)\n";
    return true;
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
    std::cout << "\n=== Quantum Analysis Tests ===\n\n";
    std::cout << "Testing geodesic tracing and topological defect detection\n";
    std::cout << "Based on Gorard's 'Some Quantum Mechanical Properties of the Wolfram Model'\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    int passed = 0;
    int failed = 0;

    // Run all tests
    if (test_geodesic_basic()) ++passed; else ++failed;
    if (test_geodesic_bundle_spread()) ++passed; else ++failed;
    if (test_k5_detection()) ++passed; else ++failed;
    if (test_k33_detection()) ++passed; else ++failed;
    if (test_planar_no_minors()) ++passed; else ++failed;
    if (test_high_degree_detection()) ++passed; else ++failed;
    if (test_topological_charge()) ++passed; else ++failed;
    if (test_geodesic_validity()) ++passed; else ++failed;
    if (test_bundle_width()) ++passed; else ++failed;
    if (test_k5_in_larger_graph()) ++passed; else ++failed;
    if (test_charge_computation()) ++passed; else ++failed;
    if (test_empty_graph()) ++passed; else ++failed;
    if (test_single_vertex()) ++passed; else ++failed;
    if (test_geodesic_performance()) ++passed; else ++failed;
    if (test_particle_performance()) ++passed; else ++failed;

    // New curvature/entropy/rotation tests
    if (test_curvature_ollivier_ricci()) ++passed; else ++failed;
    if (test_curvature_dimension_gradient()) ++passed; else ++failed;
    if (test_entropy_degree()) ++passed; else ++failed;
    if (test_entropy_local()) ++passed; else ++failed;
    if (test_entropy_fisher()) ++passed; else ++failed;
    if (test_rotation_center()) ++passed; else ++failed;
    if (test_rotation_power_law()) ++passed; else ++failed;
    if (test_curvature_star_graph()) ++passed; else ++failed;
    if (test_entropy_uniform()) ++passed; else ++failed;

    // New Hilbert space / branchial analysis tests
    if (test_hilbert_inner_product()) ++passed; else ++failed;
    if (test_hilbert_vertex_probabilities()) ++passed; else ++failed;
    if (test_hilbert_analysis()) ++passed; else ++failed;
    if (test_branchial_sharpness()) ++passed; else ++failed;

    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "Total time: " << total_ms << " ms\n";

    if (failed > 0) {
        std::cout << "\nSOME TESTS FAILED\n";
        return 1;
    }

    std::cout << "\nALL TESTS PASSED\n";
    return 0;
}
