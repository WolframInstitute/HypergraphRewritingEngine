#include <blackhole/bh_initial_condition.hpp>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace viz::blackhole {

// =============================================================================
// Utility Functions
// =============================================================================

Vec2 bh1_center(const BHConfig& config) {
    return {config.separation / 2.0f, 0.0f};
}

Vec2 bh2_center(const BHConfig& config) {
    return {-config.separation / 2.0f, 0.0f};
}

float bh1_horizon_radius(const BHConfig& config) {
    return config.mass1 / 2.0f;
}

float bh2_horizon_radius(const BHConfig& config) {
    return config.mass2 / 2.0f;
}

bool inside_horizon(const Vec2& point, const BHConfig& config) {
    Vec2 c1 = bh1_center(config);
    Vec2 c2 = bh2_center(config);

    float r1 = (point - c1).length();
    float r2 = (point - c2).length();

    return r1 < bh1_horizon_radius(config) || r2 < bh2_horizon_radius(config);
}

float conformal_factor(const Vec2& point, const BHConfig& config) {
    Vec2 c1 = bh1_center(config);
    Vec2 c2 = bh2_center(config);

    float r1 = (point - c1).length();
    float r2 = (point - c2).length();

    // Avoid division by zero
    if (r1 < 1e-6f) r1 = 1e-6f;
    if (r2 < 1e-6f) r2 = 1e-6f;

    // Brill-Lindquist conformal factor: 1 + m1/(2*r1) + m2/(2*r2)
    return 1.0f + config.mass1 / (2.0f * r1) + config.mass2 / (2.0f * r2);
}

float volume_element(const Vec2& point, const BHConfig& config) {
    float psi = conformal_factor(point, config);
    return std::pow(psi, 4);
}

// =============================================================================
// Brill-Lindquist Generator
// =============================================================================

BHInitialCondition generate_brill_lindquist(
    int n_vertices,
    const BHConfig& config
) {
    BHInitialCondition result;
    result.config = config;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_x(config.box_x[0], config.box_x[1]);
    std::uniform_real_distribution<float> dist_y(config.box_y[0], config.box_y[1]);
    std::uniform_real_distribution<float> dist_01(0.0f, 1.0f);

    // Maximum volume element for rejection sampling (overestimate is fine)
    float max_vol = 50.0f;

    std::cout << "Generating Brill-Lindquist initial condition..." << std::endl;
    std::cout << "  BH1: center = (" << config.separation / 2 << ", 0), horizon radius = "
              << config.mass1 / 2 << std::endl;
    std::cout << "  BH2: center = (" << -config.separation / 2 << ", 0), horizon radius = "
              << config.mass2 / 2 << std::endl;

    // Rejection sampling
    int attempts = 0;
    int max_attempts = n_vertices * 1000;

    while (static_cast<int>(result.vertex_positions.size()) < n_vertices && attempts < max_attempts) {
        ++attempts;

        Vec2 pt{dist_x(gen), dist_y(gen)};

        // Hard exclusion of horizon interiors
        if (inside_horizon(pt, config)) {
            continue;
        }

        // Accept with probability proportional to volume element
        float vol = volume_element(pt, config);
        if (dist_01(gen) < vol / max_vol) {
            result.vertex_positions.push_back(pt);
        }
    }

    if (static_cast<int>(result.vertex_positions.size()) < n_vertices) {
        std::cerr << "Warning: only generated " << result.vertex_positions.size()
                  << " vertices (requested " << n_vertices << ")" << std::endl;
    }

    // Build edges: connect vertices within threshold, reject if crossing horizon
    // Randomly flip edge directions for unbiased exploration
    std::uniform_int_distribution<int> flip_dist(0, 1);

    int n = static_cast<int>(result.vertex_positions.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Vec2 pi = result.vertex_positions[i];
            Vec2 pj = result.vertex_positions[j];

            float dist = (pi - pj).length();
            if (dist < config.edge_threshold) {
                // Check if midpoint is inside horizon
                Vec2 mid = (pi + pj) * 0.5f;
                if (!inside_horizon(mid, config)) {
                    // Randomly flip edge direction
                    if (flip_dist(gen)) {
                        result.edges.push_back({static_cast<uint32_t>(j), static_cast<uint32_t>(i)});
                    } else {
                        result.edges.push_back({static_cast<uint32_t>(i), static_cast<uint32_t>(j)});
                    }
                }
            }
        }
    }

    std::cout << "  Generated " << result.vertex_positions.size() << " vertices, "
              << result.edges.size() << " edges" << std::endl;

    return result;
}

// =============================================================================
// Grid with Holes Generator
// =============================================================================

BHInitialCondition generate_grid_with_holes(
    int grid_width,
    int grid_height,
    const BHConfig& config
) {
    BHInitialCondition result;
    result.config = config;

    std::cout << "Generating grid with holes initial condition..." << std::endl;
    std::cout << "  Grid: " << grid_width << " x " << grid_height << std::endl;
    std::cout << "  BH1: center = (" << config.separation / 2 << ", 0), horizon radius = "
              << config.mass1 / 2 << std::endl;
    std::cout << "  BH2: center = (" << -config.separation / 2 << ", 0), horizon radius = "
              << config.mass2 / 2 << std::endl;

    // Compute grid spacing to fit the box
    float width = config.box_x[1] - config.box_x[0];
    float height = config.box_y[1] - config.box_y[0];
    float dx = width / (grid_width - 1);
    float dy = height / (grid_height - 1);

    // Map from grid position to vertex index (for vertices that exist)
    std::vector<std::vector<int>> grid_to_vertex(grid_width, std::vector<int>(grid_height, -1));

    // Create vertices (skip those inside horizons)
    for (int i = 0; i < grid_width; ++i) {
        for (int j = 0; j < grid_height; ++j) {
            Vec2 pt{
                config.box_x[0] + i * dx,
                config.box_y[0] + j * dy
            };

            if (!inside_horizon(pt, config)) {
                grid_to_vertex[i][j] = static_cast<int>(result.vertex_positions.size());
                result.vertex_positions.push_back(pt);
            }
        }
    }

    // Random generator for edge direction flipping
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> flip_dist(0, 1);

    // Create edges (only between adjacent existing vertices)
    auto add_edge_if_valid = [&](int i1, int j1, int i2, int j2) {
        if (i2 < 0 || i2 >= grid_width || j2 < 0 || j2 >= grid_height) return;

        int v1 = grid_to_vertex[i1][j1];
        int v2 = grid_to_vertex[i2][j2];

        if (v1 >= 0 && v2 >= 0) {
            // Also check that edge midpoint doesn't cross horizon
            Vec2 p1 = result.vertex_positions[v1];
            Vec2 p2 = result.vertex_positions[v2];
            Vec2 mid = (p1 + p2) * 0.5f;

            if (!inside_horizon(mid, config)) {
                // Randomly flip edge direction for unbiased exploration
                if (flip_dist(gen)) {
                    result.edges.push_back({static_cast<uint32_t>(v2), static_cast<uint32_t>(v1)});
                } else {
                    result.edges.push_back({static_cast<uint32_t>(v1), static_cast<uint32_t>(v2)});
                }
            }
        }
    };

    for (int i = 0; i < grid_width; ++i) {
        for (int j = 0; j < grid_height; ++j) {
            if (grid_to_vertex[i][j] >= 0) {
                // Right neighbor
                add_edge_if_valid(i, j, i + 1, j);
                // Up neighbor
                add_edge_if_valid(i, j, i, j + 1);
                // Optionally add diagonals for more connectivity
                // add_edge_if_valid(i, j, i + 1, j + 1);
                // add_edge_if_valid(i, j, i + 1, j - 1);
            }
        }
    }

    std::cout << "  Generated " << result.vertex_positions.size() << " vertices, "
              << result.edges.size() << " edges" << std::endl;

    return result;
}

} // namespace viz::blackhole
