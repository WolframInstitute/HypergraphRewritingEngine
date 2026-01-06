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

// =============================================================================
// Solid Grid Generator (no holes)
// =============================================================================

BHInitialCondition generate_solid_grid(
    int grid_width,
    int grid_height,
    const BHConfig& config
) {
    BHInitialCondition result;
    result.config = config;

    std::cout << "Generating solid grid initial condition..." << std::endl;
    std::cout << "  Grid: " << grid_width << " x " << grid_height << std::endl;

    // Compute grid spacing to fit the box
    float width = config.box_x[1] - config.box_x[0];
    float height = config.box_y[1] - config.box_y[0];
    float dx = width / (grid_width - 1);
    float dy = height / (grid_height - 1);

    // Map from grid position to vertex index
    std::vector<std::vector<int>> grid_to_vertex(grid_width, std::vector<int>(grid_height, -1));

    // Create all vertices (no exclusion)
    for (int i = 0; i < grid_width; ++i) {
        for (int j = 0; j < grid_height; ++j) {
            Vec2 pt{
                config.box_x[0] + i * dx,
                config.box_y[0] + j * dy
            };
            grid_to_vertex[i][j] = static_cast<int>(result.vertex_positions.size());
            result.vertex_positions.push_back(pt);
        }
    }

    // Random generator for edge direction flipping
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> flip_dist(0, 1);

    // Create edges between adjacent vertices
    auto add_edge = [&](int i1, int j1, int i2, int j2) {
        if (i2 < 0 || i2 >= grid_width || j2 < 0 || j2 >= grid_height) return;

        int v1 = grid_to_vertex[i1][j1];
        int v2 = grid_to_vertex[i2][j2];

        // Randomly flip edge direction for unbiased exploration
        if (flip_dist(gen)) {
            result.edges.push_back({static_cast<uint32_t>(v2), static_cast<uint32_t>(v1)});
        } else {
            result.edges.push_back({static_cast<uint32_t>(v1), static_cast<uint32_t>(v2)});
        }
    };

    for (int i = 0; i < grid_width; ++i) {
        for (int j = 0; j < grid_height; ++j) {
            // Right neighbor
            add_edge(i, j, i + 1, j);
            // Up neighbor
            add_edge(i, j, i, j + 1);
        }
    }

    std::cout << "  Generated " << result.vertex_positions.size() << " vertices, "
              << result.edges.size() << " edges" << std::endl;

    return result;
}

// =============================================================================
// Topology-Aware Distance Functions
// =============================================================================

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;

float flat_distance(const Vec2& a, const Vec2& b) {
    return (a - b).length();
}

float cylinder_distance(const Vec2& a, const Vec2& b, float radius) {
    // a, b are (theta, z) coordinates
    // theta ∈ [0, 2π), z ∈ ℝ
    float dtheta = std::abs(a.x - b.x);
    dtheta = std::min(dtheta, TWO_PI - dtheta);  // Handle wraparound
    float dz = a.y - b.y;
    // Arc length on cylinder surface
    return std::sqrt(radius * radius * dtheta * dtheta + dz * dz);
}

float torus_distance(const Vec2& a, const Vec2& b, float major_radius, float minor_radius) {
    // a, b are (theta, phi) coordinates
    // Both wrap around [0, 2π)
    float dtheta = std::abs(a.x - b.x);
    dtheta = std::min(dtheta, TWO_PI - dtheta);
    float dphi = std::abs(a.y - b.y);
    dphi = std::min(dphi, TWO_PI - dphi);
    // Flat torus metric (intrinsic distance approximation)
    return std::sqrt(major_radius * major_radius * dtheta * dtheta +
                     minor_radius * minor_radius * dphi * dphi);
}

float sphere_distance(const Vec2& a, const Vec2& b, float radius) {
    // a, b are (theta, phi) coordinates (colatitude, azimuth)
    // theta ∈ [0, π], phi ∈ [0, 2π)

    // Convert to 3D Cartesian for great circle distance
    float sin_a = std::sin(a.x), cos_a = std::cos(a.x);
    float sin_b = std::sin(b.x), cos_b = std::cos(b.x);

    float xa = sin_a * std::cos(a.y);
    float ya = sin_a * std::sin(a.y);
    float za = cos_a;

    float xb = sin_b * std::cos(b.y);
    float yb = sin_b * std::sin(b.y);
    float zb = cos_b;

    // Dot product gives cos of angle between points
    float dot = xa * xb + ya * yb + za * zb;
    dot = std::clamp(dot, -1.0f, 1.0f);

    // Great circle distance
    return radius * std::acos(dot);
}

float klein_distance(const Vec2& a, const Vec2& b, const TopologyConfig& config) {
    // Klein bottle: when θ wraps around, z is flipped
    // Consider both direct path and wrapped path

    float z_min = config.domain_y[0];
    float z_max = config.domain_y[1];
    float z_range = z_max - z_min;

    // Direct distance
    float dtheta_direct = std::abs(a.x - b.x);
    float dz_direct = std::abs(a.y - b.y);

    // Wrapped distance: go around θ with z-flip
    float dtheta_wrap = TWO_PI - dtheta_direct;
    // When wrapping, b.y becomes (z_max + z_min) - b.y (reflection)
    float b_z_flipped = (z_max + z_min) - b.y;
    float dz_wrap = std::abs(a.y - b_z_flipped);

    float R = config.major_radius;

    float dist_direct = std::sqrt(R * R * dtheta_direct * dtheta_direct + dz_direct * dz_direct);
    float dist_wrap = std::sqrt(R * R * dtheta_wrap * dtheta_wrap + dz_wrap * dz_wrap);

    return std::min(dist_direct, dist_wrap);
}

float mobius_distance(const Vec2& a, const Vec2& b, const TopologyConfig& config) {
    // Möbius strip: θ wraps with z-flip, finite width
    // Similar to Klein but with open edges in z direction

    float z_min = config.domain_y[0];
    float z_max = config.domain_y[1];

    // Direct distance
    float dtheta_direct = std::abs(a.x - b.x);
    float dz_direct = std::abs(a.y - b.y);

    // Wrapped distance
    float dtheta_wrap = TWO_PI - dtheta_direct;
    float b_z_flipped = (z_max + z_min) - b.y;
    float dz_wrap = std::abs(a.y - b_z_flipped);

    float R = config.major_radius;

    float dist_direct = std::sqrt(R * R * dtheta_direct * dtheta_direct + dz_direct * dz_direct);
    float dist_wrap = std::sqrt(R * R * dtheta_wrap * dtheta_wrap + dz_wrap * dz_wrap);

    return std::min(dist_direct, dist_wrap);
}

float topology_distance(const Vec2& a, const Vec2& b, const TopologyConfig& config) {
    switch (config.type) {
        case TopologyType::Flat:
            return flat_distance(a, b);
        case TopologyType::Cylinder:
            return cylinder_distance(a, b, config.major_radius);
        case TopologyType::Torus:
            return torus_distance(a, b, config.major_radius, config.minor_radius);
        case TopologyType::Sphere:
            return sphere_distance(a, b, config.major_radius);
        case TopologyType::KleinBottle:
            return klein_distance(a, b, config);
        case TopologyType::MobiusStrip:
            return mobius_distance(a, b, config);
        default:
            return flat_distance(a, b);
    }
}

// =============================================================================
// Topology-Aware Sampling Functions
// =============================================================================

std::vector<Vec2> sample_uniform(int n, const TopologyConfig& config, uint32_t seed) {
    std::vector<Vec2> positions;
    positions.reserve(n);

    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);

    switch (config.type) {
        case TopologyType::Flat: {
            std::uniform_real_distribution<float> dist_x(config.domain_x[0], config.domain_x[1]);
            std::uniform_real_distribution<float> dist_y(config.domain_y[0], config.domain_y[1]);
            for (int i = 0; i < n; ++i) {
                positions.push_back({dist_x(gen), dist_y(gen)});
            }
            break;
        }

        case TopologyType::Cylinder:
        case TopologyType::KleinBottle:
        case TopologyType::MobiusStrip: {
            // θ ∈ [0, 2π), z ∈ [z_min, z_max]
            std::uniform_real_distribution<float> dist_theta(0.0f, TWO_PI);
            std::uniform_real_distribution<float> dist_z(config.domain_y[0], config.domain_y[1]);
            for (int i = 0; i < n; ++i) {
                positions.push_back({dist_theta(gen), dist_z(gen)});
            }
            break;
        }

        case TopologyType::Torus: {
            // θ ∈ [0, 2π), φ ∈ [0, 2π)
            std::uniform_real_distribution<float> dist_angle(0.0f, TWO_PI);
            for (int i = 0; i < n; ++i) {
                positions.push_back({dist_angle(gen), dist_angle(gen)});
            }
            break;
        }

        case TopologyType::Sphere: {
            // Use Fibonacci lattice for near-uniform distribution
            float golden = (1.0f + std::sqrt(5.0f)) / 2.0f;
            for (int i = 0; i < n; ++i) {
                // Theta (colatitude): use arccos for uniform distribution on sphere
                float theta = std::acos(1.0f - 2.0f * (i + 0.5f) / n);
                // Phi (azimuth): golden angle for even spacing
                float phi = TWO_PI * i / golden;
                phi = std::fmod(phi, TWO_PI);
                positions.push_back({theta, phi});
            }
            break;
        }
    }

    return positions;
}

std::vector<Vec2> sample_grid(const TopologyConfig& config) {
    std::vector<Vec2> positions;
    int res = config.grid_resolution;

    switch (config.type) {
        case TopologyType::Flat: {
            float dx = (config.domain_x[1] - config.domain_x[0]) / (res - 1);
            float dy = (config.domain_y[1] - config.domain_y[0]) / (res - 1);
            for (int i = 0; i < res; ++i) {
                for (int j = 0; j < res; ++j) {
                    positions.push_back({
                        config.domain_x[0] + i * dx,
                        config.domain_y[0] + j * dy
                    });
                }
            }
            break;
        }

        case TopologyType::Cylinder:
        case TopologyType::KleinBottle:
        case TopologyType::MobiusStrip: {
            // Note: theta grid wraps, so we use res points covering [0, 2π)
            float dtheta = TWO_PI / res;  // NOT res-1, because it wraps
            float dz = (config.domain_y[1] - config.domain_y[0]) / (res - 1);
            for (int i = 0; i < res; ++i) {
                for (int j = 0; j < res; ++j) {
                    positions.push_back({
                        i * dtheta,
                        config.domain_y[0] + j * dz
                    });
                }
            }
            break;
        }

        case TopologyType::Torus: {
            // Both dimensions wrap
            float dtheta = TWO_PI / res;
            float dphi = TWO_PI / res;
            std::cout << "  [Torus grid] res=" << res << " dtheta=" << dtheta << " dphi=" << dphi << std::endl;
            for (int i = 0; i < res; ++i) {
                for (int j = 0; j < res; ++j) {
                    positions.push_back({i * dtheta, j * dphi});
                }
            }
            std::cout << "  [Torus grid] Generated " << positions.size() << " positions" << std::endl;
            break;
        }

        case TopologyType::Sphere: {
            // Use UV grid, but account for pole singularities
            // Skip the exact poles and distribute evenly
            int n_lat = res;      // Number of latitude bands
            int n_lon = res * 2;  // More longitude points (sphere is 2:1 aspect in UV)

            for (int i = 0; i < n_lat; ++i) {
                // Theta from small epsilon to π - epsilon (avoid exact poles)
                float theta = PI * (i + 0.5f) / n_lat;

                // Number of longitude points scales with sin(theta) for uniformity
                int lon_count = std::max(1, static_cast<int>(n_lon * std::sin(theta)));

                for (int j = 0; j < lon_count; ++j) {
                    float phi = TWO_PI * j / lon_count;
                    positions.push_back({theta, phi});
                }
            }
            break;
        }
    }

    return positions;
}

std::vector<Vec2> sample_poisson(int target_n, const TopologyConfig& config, uint32_t seed) {
    // Simple dart-throwing Poisson disk sampling with topology-aware distance
    std::vector<Vec2> positions;
    positions.reserve(target_n);

    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    float min_dist = config.poisson_min_distance;

    // Sample candidates
    int max_attempts = target_n * 100;
    int attempts = 0;

    auto generate_candidate = [&]() -> Vec2 {
        switch (config.type) {
            case TopologyType::Flat: {
                std::uniform_real_distribution<float> dx(config.domain_x[0], config.domain_x[1]);
                std::uniform_real_distribution<float> dy(config.domain_y[0], config.domain_y[1]);
                return {dx(gen), dy(gen)};
            }
            case TopologyType::Cylinder:
            case TopologyType::KleinBottle:
            case TopologyType::MobiusStrip: {
                std::uniform_real_distribution<float> dtheta(0.0f, TWO_PI);
                std::uniform_real_distribution<float> dz(config.domain_y[0], config.domain_y[1]);
                return {dtheta(gen), dz(gen)};
            }
            case TopologyType::Torus: {
                std::uniform_real_distribution<float> da(0.0f, TWO_PI);
                return {da(gen), da(gen)};
            }
            case TopologyType::Sphere: {
                // Uniform on sphere
                std::uniform_real_distribution<float> u(0.0f, 1.0f);
                float theta = std::acos(1.0f - 2.0f * u(gen));
                float phi = TWO_PI * u(gen);
                return {theta, phi};
            }
            default:
                return {0, 0};
        }
    };

    while (static_cast<int>(positions.size()) < target_n && attempts < max_attempts) {
        ++attempts;

        Vec2 candidate = generate_candidate();

        // Check distance to all existing points
        bool valid = true;
        for (const auto& p : positions) {
            if (topology_distance(candidate, p, config) < min_dist) {
                valid = false;
                break;
            }
        }

        if (valid) {
            positions.push_back(candidate);
        }
    }

    return positions;
}

// =============================================================================
// Helper: Compute appropriate edge threshold for topology
// =============================================================================

float compute_default_edge_threshold(const TopologyConfig& config, int n_vertices) {
    // Estimate expected spacing between vertices based on topology and count
    switch (config.type) {
        case TopologyType::Flat: {
            float width = config.domain_x[1] - config.domain_x[0];
            float height = config.domain_y[1] - config.domain_y[0];
            float area = width * height;
            float spacing = std::sqrt(area / n_vertices);
            return spacing * 1.5f;  // Connect to ~6 neighbors
        }
        case TopologyType::Cylinder:
        case TopologyType::KleinBottle:
        case TopologyType::MobiusStrip: {
            // θ wraps [0, 2π), z linear
            float z_range = config.domain_y[1] - config.domain_y[0];
            float theta_len = TWO_PI * config.major_radius;  // Circumference
            float area = theta_len * z_range;
            float spacing = std::sqrt(area / n_vertices);
            return spacing * 1.5f;
        }
        case TopologyType::Torus: {
            // For a grid, neighbors are spaced by 2π/sqrt(n) in each angle dimension
            // Arc distances: R * dθ in theta direction, r * dφ in phi direction
            // These can be very different when R >> r or R << r
            // Threshold must exceed the LARGER of the two to connect all neighbors
            float res = std::sqrt(static_cast<float>(n_vertices));
            float dtheta = TWO_PI / res;
            float dphi = TWO_PI / res;
            float theta_dist = config.major_radius * dtheta;  // Arc length between theta neighbors
            float phi_dist = config.minor_radius * dphi;      // Arc length between phi neighbors
            float max_neighbor_dist = std::max(theta_dist, phi_dist);
            return max_neighbor_dist * 1.5f;  // 1.5x to ensure connectivity
        }
        case TopologyType::Sphere: {
            // Surface area = 4πR²
            float area = 4.0f * PI * config.major_radius * config.major_radius;
            float spacing = std::sqrt(area / n_vertices);
            return spacing * 1.5f;
        }
        default:
            return 2.0f;
    }
}

// =============================================================================
// Generic Initial Condition Generator
// =============================================================================

BHInitialCondition generate_initial_condition(
    int n_vertices,
    const TopologyConfig& topo_config,
    uint32_t seed
) {
    BHInitialCondition result;

    // Auto-compute edge threshold if not explicitly set (threshold <= 0 means auto)
    float effective_threshold = topo_config.edge_threshold;
    if (effective_threshold <= 0) {
        effective_threshold = compute_default_edge_threshold(topo_config, n_vertices);
    }

    // Set a default BHConfig (for compatibility)
    result.config.edge_threshold = effective_threshold;
    result.config.box_x = topo_config.domain_x;
    result.config.box_y = topo_config.domain_y;

    std::cout << "Generating initial condition:" << std::endl;
    std::cout << "  Topology: " << topology_name(topo_config.type) << std::endl;
    std::cout << "  Sampling: " << sampling_name(topo_config.sampling) << std::endl;
    std::cout << "  Grid resolution: " << topo_config.grid_resolution << std::endl;
    std::cout << "  Major radius: " << topo_config.major_radius << std::endl;
    std::cout << "  Minor radius: " << topo_config.minor_radius << std::endl;
    std::cout << "  Edge threshold: " << effective_threshold << std::endl;

    // 1. Sample vertices
    switch (topo_config.sampling) {
        case SamplingMethod::Uniform:
            result.vertex_positions = sample_uniform(n_vertices, topo_config, seed);
            break;
        case SamplingMethod::Grid:
            result.vertex_positions = sample_grid(topo_config);
            break;
        case SamplingMethod::PoissonDisk:
            result.vertex_positions = sample_poisson(n_vertices, topo_config, seed);
            break;
        case SamplingMethod::DensityWeighted:
            // TODO: Implement density-weighted sampling with defects
            // For now, fall back to uniform
            result.vertex_positions = sample_uniform(n_vertices, topo_config, seed);
            break;
    }

    std::cout << "  Sampled " << result.vertex_positions.size() << " vertices" << std::endl;

    // 2. Filter out vertices inside defect exclusion zones
    if (topo_config.defects.count > 0) {
        std::vector<Vec2> filtered;
        for (const auto& pos : result.vertex_positions) {
            bool inside_defect = false;
            for (size_t i = 0; i < topo_config.defects.positions.size(); ++i) {
                float d = topology_distance(pos, topo_config.defects.positions[i], topo_config);
                if (d < topo_config.defects.exclusion_radius) {
                    inside_defect = true;
                    break;
                }
            }
            if (!inside_defect) {
                filtered.push_back(pos);
            }
        }
        result.vertex_positions = std::move(filtered);
        std::cout << "  After defect filtering: " << result.vertex_positions.size() << " vertices" << std::endl;
    }

    // 3. Generate edges using topology-aware distance
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<int> flip_dist(0, 1);

    size_t n = result.vertex_positions.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float d = topology_distance(result.vertex_positions[i], result.vertex_positions[j], topo_config);
            if (d < effective_threshold) {
                // Check if edge crosses defect (midpoint inside exclusion zone)
                bool crosses_defect = false;
                if (topo_config.defects.count > 0) {
                    // Simple midpoint check (not exact for curved topologies, but good enough)
                    Vec2 mid{
                        (result.vertex_positions[i].x + result.vertex_positions[j].x) * 0.5f,
                        (result.vertex_positions[i].y + result.vertex_positions[j].y) * 0.5f
                    };
                    for (size_t k = 0; k < topo_config.defects.positions.size(); ++k) {
                        float dm = topology_distance(mid, topo_config.defects.positions[k], topo_config);
                        if (dm < topo_config.defects.exclusion_radius) {
                            crosses_defect = true;
                            break;
                        }
                    }
                }

                if (!crosses_defect) {
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

    std::cout << "  Generated " << result.edges.size() << " edges" << std::endl;

    return result;
}

// =============================================================================
// Coordinate Conversion for Visualization
// =============================================================================

Vec2 topology_to_display(const Vec2& internal_pos, const TopologyConfig& config) {
    switch (config.type) {
        case TopologyType::Flat:
            // Identity
            return internal_pos;

        case TopologyType::Cylinder: {
            // Unroll: (θ, z) → (R * θ - π*R, z) centered at origin
            float x = config.major_radius * internal_pos.x - PI * config.major_radius;
            float y = internal_pos.y;
            return {x, y};
        }

        case TopologyType::Torus: {
            // Flatten to rectangle: (θ, φ) → (R * θ, r * φ)
            float x = config.major_radius * internal_pos.x - PI * config.major_radius;
            float y = config.minor_radius * internal_pos.y - PI * config.minor_radius;
            return {x, y};
        }

        case TopologyType::Sphere: {
            // Equirectangular projection: (θ, φ) → (φ - π, π/2 - θ) scaled
            float x = (internal_pos.y - PI) * config.major_radius;
            float y = (PI / 2.0f - internal_pos.x) * config.major_radius;
            return {x, y};
        }

        case TopologyType::KleinBottle:
        case TopologyType::MobiusStrip: {
            // Same as cylinder for display (topology is in the edges)
            float x = config.major_radius * internal_pos.x - PI * config.major_radius;
            float y = internal_pos.y;
            return {x, y};
        }

        default:
            return internal_pos;
    }
}

Vec3 topology_to_display_3d(const Vec2& internal_pos, const TopologyConfig& config) {
    switch (config.type) {
        case TopologyType::Flat:
            // Identity with z=0
            return {internal_pos.x, internal_pos.y, 0.0f};

        case TopologyType::Cylinder: {
            // (θ, z) → (R*cos(θ), R*sin(θ), z)
            float theta = internal_pos.x;
            float z = internal_pos.y;
            float R = config.major_radius;
            return {R * std::cos(theta), R * std::sin(theta), z};
        }

        case TopologyType::Torus: {
            // (θ, φ) → ((R + r*cos(φ))*cos(θ), (R + r*cos(φ))*sin(θ), r*sin(φ))
            float theta = internal_pos.x;  // Major angle (around the donut)
            float phi = internal_pos.y;    // Minor angle (around the tube)
            float R = config.major_radius;
            float r = config.minor_radius;
            float rho = R + r * std::cos(phi);  // Distance from central axis
            Vec3 result = {
                rho * std::cos(theta),
                rho * std::sin(theta),
                r * std::sin(phi)
            };
            return result;
        }

        case TopologyType::Sphere: {
            // (θ, φ) → (R*sin(θ)*cos(φ), R*sin(θ)*sin(φ), R*cos(θ))
            // θ = colatitude (0 at north pole, π at south pole)
            // φ = longitude (0 to 2π)
            float theta = internal_pos.x;
            float phi = internal_pos.y;
            float R = config.major_radius;
            float sin_theta = std::sin(theta);
            return {
                R * sin_theta * std::cos(phi),
                R * sin_theta * std::sin(phi),
                R * std::cos(theta)
            };
        }

        case TopologyType::KleinBottle:
        case TopologyType::MobiusStrip: {
            // Use cylinder embedding for display (topology is in the edges)
            float theta = internal_pos.x;
            float z = internal_pos.y;
            float R = config.major_radius;
            return {R * std::cos(theta), R * std::sin(theta), z};
        }

        default:
            return {internal_pos.x, internal_pos.y, 0.0f};
    }
}

} // namespace viz::blackhole
