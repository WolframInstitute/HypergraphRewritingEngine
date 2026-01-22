#include "blackhole/branch_alignment.hpp"
#include <job_system/job_system.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <random>

namespace viz::blackhole {

// =============================================================================
// Spectral Embedding Implementation
// =============================================================================

// Deflate matrix to find next eigenvector
static void deflate_matrix(
    std::vector<std::vector<float>>& matrix,
    const std::vector<float>& eigenvector,
    float eigenvalue
) {
    size_t n = matrix.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
}

std::vector<Vec3> compute_spectral_embedding(
    const SimpleGraph& graph,
    int num_dimensions,
    int num_iterations,
    LaplacianType laplacian_type
) {
    const auto& vertices = graph.vertices();
    size_t n = vertices.size();

    if (n < 4) {
        // Too few vertices, return simple positions
        std::vector<Vec3> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = Vec3(static_cast<float>(i), 0.0f, 0.0f);
        }
        return result;
    }

    // Build vertex index map
    std::unordered_map<VertexId, size_t> vertex_to_idx;
    for (size_t i = 0; i < n; ++i) {
        vertex_to_idx[vertices[i]] = i;
    }

    // Build adjacency matrix and degree vector
    std::vector<std::vector<float>> adj(n, std::vector<float>(n, 0.0f));
    std::vector<float> degree(n, 0.0f);
    float max_degree = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        VertexId v = vertices[i];
        for (VertexId neighbor : graph.neighbors(v)) {
            auto it = vertex_to_idx.find(neighbor);
            if (it != vertex_to_idx.end()) {
                size_t j = it->second;
                adj[i][j] = 1.0f;
                degree[i] += 1.0f;
            }
        }
        max_degree = std::max(max_degree, degree[i]);
    }

    // Build Laplacian matrix based on type
    std::vector<std::vector<float>> laplacian(n, std::vector<float>(n, 0.0f));
    float shift_value = 0.0f;

    if (laplacian_type == LaplacianType::Unnormalized) {
        // Unnormalized Laplacian: L = D - A (matches Mathematica's SpectralEmbedding)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    laplacian[i][j] = degree[i];
                } else {
                    laplacian[i][j] = -adj[i][j];
                }
            }
        }
        // Max eigenvalue of unnormalized Laplacian is bounded by 2 * max_degree
        shift_value = 2.0f * max_degree + 1.0f;
    } else {
        // Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    laplacian[i][j] = 1.0f;
                }
                if (degree[i] > 0 && degree[j] > 0) {
                    float d_inv_sqrt_i = 1.0f / std::sqrt(degree[i]);
                    float d_inv_sqrt_j = 1.0f / std::sqrt(degree[j]);
                    laplacian[i][j] -= d_inv_sqrt_i * adj[i][j] * d_inv_sqrt_j;
                }
            }
        }
        // Max eigenvalue of normalized Laplacian is bounded by 2
        shift_value = 2.0f;
    }

    // Use shifted matrix: M = shift*I - L (so smallest eigenvectors of L become largest of M)
    std::vector<std::vector<float>> shifted(n, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                shifted[i][j] = shift_value - laplacian[i][j];
            } else {
                shifted[i][j] = -laplacian[i][j];
            }
        }
    }

    // Find eigenvectors using power iteration with deflation
    std::vector<std::vector<float>> eigenvectors;
    std::vector<float> eigenvalues;

    auto working_matrix = shifted;

    // Skip first eigenvector (constant vector for Laplacian)
    // Find num_dimensions + 1 eigenvectors and discard the first
    for (int d = 0; d < num_dimensions + 1; ++d) {
        // Power iteration
        std::vector<float> v(n);
        std::mt19937 rng(42 + d);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < n; ++i) {
            v[i] = dist(rng);
        }

        // Normalize
        float norm = 0.0f;
        for (size_t i = 0; i < n; ++i) norm += v[i] * v[i];
        norm = std::sqrt(norm);
        for (size_t i = 0; i < n; ++i) v[i] /= norm;

        // Orthogonalize against previous eigenvectors
        auto orthogonalize = [&]() {
            for (const auto& prev : eigenvectors) {
                float dot = 0.0f;
                for (size_t i = 0; i < n; ++i) dot += v[i] * prev[i];
                for (size_t i = 0; i < n; ++i) v[i] -= dot * prev[i];
            }
            float norm = 0.0f;
            for (size_t i = 0; i < n; ++i) norm += v[i] * v[i];
            norm = std::sqrt(norm);
            if (norm > 1e-10f) {
                for (size_t i = 0; i < n; ++i) v[i] /= norm;
            }
        };

        for (int iter = 0; iter < num_iterations; ++iter) {
            // Matrix-vector multiply
            std::vector<float> v_new(n, 0.0f);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    v_new[i] += working_matrix[i][j] * v[j];
                }
            }

            // Normalize
            float norm = 0.0f;
            for (size_t i = 0; i < n; ++i) norm += v_new[i] * v_new[i];
            norm = std::sqrt(norm);
            if (norm < 1e-10f) break;
            for (size_t i = 0; i < n; ++i) v_new[i] /= norm;

            v = std::move(v_new);
            orthogonalize();
        }

        // Compute eigenvalue (Rayleigh quotient)
        float eigenvalue = 0.0f;
        std::vector<float> Mv(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                Mv[i] += working_matrix[i][j] * v[j];
            }
        }
        for (size_t i = 0; i < n; ++i) {
            eigenvalue += v[i] * Mv[i];
        }

        eigenvectors.push_back(v);
        eigenvalues.push_back(eigenvalue);

        // Deflate
        deflate_matrix(working_matrix, v, eigenvalue);
    }

    // Build positions from eigenvectors (skip first which is constant)
    // Eigenvectors are unit normalized - the scale is arbitrary but structure is preserved
    std::vector<Vec3> positions(n);
    for (size_t i = 0; i < n; ++i) {
        if (eigenvectors.size() > 1) positions[i].x = eigenvectors[1][i];
        if (eigenvectors.size() > 2) positions[i].y = eigenvectors[2][i];
        if (eigenvectors.size() > 3) positions[i].z = eigenvectors[3][i];
    }

    return positions;
}

// =============================================================================
// Moment of Inertia and PCA
// =============================================================================

Vec3 compute_weighted_centroid(
    const std::vector<Vec3>& positions,
    const std::vector<float>& masses
) {
    Vec3 centroid;
    float total_mass = 0.0f;

    for (size_t i = 0; i < positions.size(); ++i) {
        float m = (i < masses.size()) ? masses[i] : 1.0f;
        centroid += positions[i] * m;
        total_mass += m;
    }

    if (total_mass > 1e-10f) {
        centroid = centroid / total_mass;
    }

    return centroid;
}

Mat3 compute_inertia_tensor(
    const std::vector<Vec3>& positions,
    const std::vector<float>& masses
) {
    Mat3 inertia;

    for (size_t i = 0; i < positions.size(); ++i) {
        float m = (i < masses.size()) ? masses[i] : 1.0f;
        const Vec3& r = positions[i];

        float r_sq = r.norm_sq();

        // I_ij = sum_k m_k * (|r_k|^2 * delta_ij - r_k,i * r_k,j)
        Mat3 term = Mat3::identity() * (r_sq * m) - Mat3::outer(r, r) * m;
        inertia += term;
    }

    return inertia;
}

// Jacobi eigenvalue algorithm for 3x3 symmetric matrix
void eigen_decomposition_3x3(
    const Mat3& matrix,
    std::array<float, 3>& eigenvalues,
    Mat3& eigenvectors
) {
    // Copy matrix
    Mat3 A = matrix;
    eigenvectors = Mat3::identity();

    const int max_iterations = 50;
    const float tolerance = 1e-10f;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        float max_val = std::abs(A.m[0][1]);
        if (std::abs(A.m[0][2]) > max_val) { p = 0; q = 2; max_val = std::abs(A.m[0][2]); }
        if (std::abs(A.m[1][2]) > max_val) { p = 1; q = 2; max_val = std::abs(A.m[1][2]); }

        if (max_val < tolerance) break;

        // Compute rotation angle
        float theta;
        float diff = A.m[q][q] - A.m[p][p];
        if (std::abs(diff) < 1e-10f) {
            theta = (A.m[p][q] > 0) ? std::numbers::pi_v<float> / 4.0f : -std::numbers::pi_v<float> / 4.0f;
        } else {
            theta = 0.5f * std::atan2(2.0f * A.m[p][q], diff);
        }

        float c = std::cos(theta);
        float s = std::sin(theta);

        // Apply rotation to A
        Mat3 A_new = A;
        A_new.m[p][p] = c * c * A.m[p][p] - 2 * s * c * A.m[p][q] + s * s * A.m[q][q];
        A_new.m[q][q] = s * s * A.m[p][p] + 2 * s * c * A.m[p][q] + c * c * A.m[q][q];
        A_new.m[p][q] = A_new.m[q][p] = 0;

        int r = 3 - p - q;  // The third index
        A_new.m[p][r] = A_new.m[r][p] = c * A.m[p][r] - s * A.m[q][r];
        A_new.m[q][r] = A_new.m[r][q] = s * A.m[p][r] + c * A.m[q][r];

        A = A_new;

        // Apply rotation to eigenvectors
        for (int i = 0; i < 3; ++i) {
            float vip = eigenvectors.m[i][p];
            float viq = eigenvectors.m[i][q];
            eigenvectors.m[i][p] = c * vip - s * viq;
            eigenvectors.m[i][q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues
    eigenvalues[0] = A.m[0][0];
    eigenvalues[1] = A.m[1][1];
    eigenvalues[2] = A.m[2][2];

    // Sort by descending eigenvalue
    std::array<int, 3> order = {0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return eigenvalues[a] > eigenvalues[b];
    });

    std::array<float, 3> sorted_eigenvalues;
    Mat3 sorted_eigenvectors;
    for (int i = 0; i < 3; ++i) {
        sorted_eigenvalues[i] = eigenvalues[order[i]];
        for (int j = 0; j < 3; ++j) {
            sorted_eigenvectors.m[j][i] = eigenvectors.m[j][order[i]];
        }
    }

    eigenvalues = sorted_eigenvalues;
    eigenvectors = sorted_eigenvectors;
}

// =============================================================================
// Canonical Orientation and Scale Normalization
// =============================================================================

// Canonical orientation using CURVATURE-WEIGHTED moment
// For each axis: flip if the curvature-weighted moment is negative
// This orients axes so positive curvature → positive direction
// This is INTRINSIC to the curvature distribution (Wolfram's moment of inertia method)
static void canonicalize_orientation_by_curvature(
    Mat3& eigenvectors,
    std::vector<Vec3>& projected,
    const std::vector<float>& signed_curvature
) {
    size_t n = projected.size();
    if (n == 0 || signed_curvature.size() != n) return;

    // Compute curvature-weighted moment for each axis
    // Positive curvature vertices pull toward positive direction
    // Negative curvature vertices pull toward negative direction
    float moment[3] = {0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        float kappa = signed_curvature[i];
        moment[0] += kappa * projected[i].x;
        moment[1] += kappa * projected[i].y;
        moment[2] += kappa * projected[i].z;
    }

    // Flip axis if curvature-weighted moment is negative
    // This ensures positive curvature tends toward positive coordinates
    float signs[3];
    for (int i = 0; i < 3; ++i) {
        signs[i] = (moment[i] >= 0) ? 1.0f : -1.0f;
    }

    // Apply flips to eigenvectors and projected positions
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            eigenvectors.m[row][col] *= signs[col];
        }
    }
    for (size_t i = 0; i < n; ++i) {
        projected[i].x *= signs[0];
        projected[i].y *= signs[1];
        projected[i].z *= signs[2];
    }
}

// Legacy interface - use curvature-based orientation
static void canonicalize_orientation(
    Mat3& eigenvectors,
    std::vector<Vec3>& projected,
    const std::vector<float>& curvature
) {
    canonicalize_orientation_by_curvature(eigenvectors, projected, curvature);
}

// Apply scale normalization: divide by sqrt(eigenvalue) for unit variance
static void apply_scale_normalization(
    std::vector<Vec3>& projected,
    const std::array<float, 3>& eigenvalues
) {
    // Compute scale factors (1/sqrt(eigenvalue)), with safeguard for near-zero
    float scales[3];
    for (int i = 0; i < 3; ++i) {
        float ev = std::max(eigenvalues[i], 1e-6f);
        scales[i] = 1.0f / std::sqrt(ev);
    }

    // Apply scaling
    for (auto& p : projected) {
        p.x *= scales[0];
        p.y *= scales[1];
        p.z *= scales[2];
    }
}

// =============================================================================
// Branch Alignment
// =============================================================================

BranchAlignmentResult align_branch(
    const SimpleGraph& graph,
    const std::unordered_map<VertexId, float>& curvature,
    bool use_canonical_orientation,
    bool use_scale_normalization
) {
    BranchAlignmentResult result;

    const auto& vertices = graph.vertices();
    size_t n = vertices.size();

    if (n < 4) {
        result.error = "Too few vertices";
        return result;
    }

    // Step 1: Spectral embedding
    auto positions = compute_spectral_embedding(graph, 3, 100);

    if (positions.size() != n) {
        result.error = "Embedding failed";
        return result;
    }

    // Step 2: Get curvature values as masses
    std::vector<float> masses(n);
    std::vector<float> curv_values(n);
    for (size_t i = 0; i < n; ++i) {
        auto it = curvature.find(vertices[i]);
        float c = (it != curvature.end()) ? it->second : 0.0f;
        curv_values[i] = c;
        masses[i] = std::abs(c) + 0.01f;  // Small offset to avoid zero mass
    }

    // Step 3: Compute curvature-weighted centroid
    result.centroid = compute_weighted_centroid(positions, masses);

    // Step 4: Center the positions
    std::vector<Vec3> centered(n);
    for (size_t i = 0; i < n; ++i) {
        centered[i] = positions[i] - result.centroid;
    }

    // Step 5: Compute moment of inertia tensor
    Mat3 inertia = compute_inertia_tensor(centered, masses);

    // Step 6: Eigendecomposition
    eigen_decomposition_3x3(inertia, result.eigenvalues, result.eigenvectors);

    // Step 7: Project onto principal axes
    std::vector<Vec3> aligned(n);
    Mat3 evT = result.eigenvectors.transpose();
    for (size_t i = 0; i < n; ++i) {
        aligned[i] = evT * centered[i];
    }

    // Step 7.5: Canonical orientation using CURVATURE
    // Orient axes so positive curvature → positive direction
    // This is intrinsic to the curvature distribution (Wolfram's moment of inertia method)
    if (use_canonical_orientation) {
        canonicalize_orientation_by_curvature(result.eigenvectors, aligned, curv_values);
    }

    // Step 7.6: Scale normalization for unit variance
    if (use_scale_normalization) {
        apply_scale_normalization(aligned, result.eigenvalues);
    }

    // Step 8: Sort by PC1 to get canonical vertex ordering
    std::vector<size_t> ordering(n);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::sort(ordering.begin(), ordering.end(), [&](size_t a, size_t b) {
        return aligned[a].x < aligned[b].x;
    });

    // Build result in canonical order
    result.vertices.resize(n);
    result.pc1.resize(n);
    result.pc2.resize(n);
    result.pc3.resize(n);
    result.curvature.resize(n);
    result.rank.resize(n);

    float curv_sum = 0.0f;
    result.curvature_min = std::numeric_limits<float>::max();
    result.curvature_max = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < n; ++i) {
        size_t idx = ordering[i];
        result.vertices[i] = vertices[idx];
        result.pc1[i] = aligned[idx].x;
        result.pc2[i] = aligned[idx].y;
        result.pc3[i] = aligned[idx].z;
        result.curvature[i] = curv_values[idx];
        result.rank[i] = (n > 1) ? static_cast<float>(i) / (n - 1) : 0.0f;

        curv_sum += curv_values[idx];
        result.curvature_min = std::min(result.curvature_min, curv_values[idx]);
        result.curvature_max = std::max(result.curvature_max, curv_values[idx]);
    }

    result.curvature_mean = curv_sum / n;
    result.num_vertices = n;
    result.valid = true;

    return result;
}

// Overload for backwards compatibility (defaults to canonical orientation + scale normalization)
BranchAlignmentResult align_branch(
    const SimpleGraph& graph,
    const std::unordered_map<VertexId, float>& curvature
) {
    return align_branch(graph, curvature, true, true);
}

std::vector<BranchAlignmentResult> align_branches_parallel(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    job_system::JobSystem<int>* js,
    bool use_canonical_orientation,
    bool use_scale_normalization
) {
    size_t num_branches = std::min(graphs.size(), curvatures.size());
    std::vector<BranchAlignmentResult> results(num_branches);

    for (size_t i = 0; i < num_branches; ++i) {
        js->submit_function([&graphs, &curvatures, &results, i,
                             use_canonical_orientation, use_scale_normalization]() {
            results[i] = align_branch(graphs[i], curvatures[i],
                                       use_canonical_orientation, use_scale_normalization);
        }, 0);
    }

    js->wait_for_completion();

    return results;
}

// Backwards-compatible overload
std::vector<BranchAlignmentResult> align_branches_parallel(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    job_system::JobSystem<int>* js
) {
    return align_branches_parallel(graphs, curvatures, js, true, true);
}

// =============================================================================
// Per-Branch Alignment with Canonical Orientation (Recommended)
// =============================================================================

AlignmentAggregation align_branches_per_branch(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    const std::vector<StateId>& state_ids,
    job_system::JobSystem<int>* js,
    const AlignmentReferenceFrame* /* reference_frame */,
    AlignmentReferenceFrame* /* output_frame */,
    bool use_canonical_orientation,
    bool use_scale_normalization
) {
    // WOLFRAM'S MOMENT OF INERTIA METHOD:
    // Each branch's coordinate frame is INTRINSIC to its curvature distribution.
    // - Principal axes from moment of inertia tensor (|curvature| as mass)
    // - Canonical orientation: positive curvature → positive direction
    // NO vertex-based alignment (Kabsch) needed - the frame is determined by
    // the distribution itself, making it comparable across branches and timesteps.

    AlignmentAggregation agg;
    size_t num_branches = std::min({graphs.size(), curvatures.size(), state_ids.size()});

    if (num_branches == 0) return agg;

    // Align each branch using curvature-weighted moment of inertia
    // ALWAYS use canonical orientation - it's what makes frames comparable
    auto branch_results = align_branches_parallel(graphs, curvatures, js,
                                                   use_canonical_orientation, use_scale_normalization);

    // Build aggregation directly from branch results (no Kabsch rotation needed)
    size_t total_points = 0;
    for (const auto& br : branch_results) {
        if (br.valid) total_points += br.num_vertices;
    }

    agg.all_pc1.reserve(total_points);
    agg.all_pc2.reserve(total_points);
    agg.all_pc3.reserve(total_points);
    agg.all_curvature.reserve(total_points);
    agg.all_rank.reserve(total_points);
    agg.branch_id.reserve(total_points);
    agg.all_vertices.reserve(total_points);
    agg.state_id.reserve(total_points);

    agg.curvature_min = std::numeric_limits<float>::max();
    agg.curvature_max = std::numeric_limits<float>::lowest();
    agg.pc1_min = std::numeric_limits<float>::max();
    agg.pc1_max = std::numeric_limits<float>::lowest();
    agg.pc2_min = std::numeric_limits<float>::max();
    agg.pc2_max = std::numeric_limits<float>::lowest();
    agg.pc3_min = std::numeric_limits<float>::max();
    agg.pc3_max = std::numeric_limits<float>::lowest();

    // Group by branch for metadata
    std::vector<float> branch_pc1_sum(num_branches, 0);
    std::vector<float> branch_pc2_sum(num_branches, 0);
    std::vector<float> branch_pc3_sum(num_branches, 0);
    std::vector<size_t> branch_point_counts(num_branches, 0);

    for (size_t b = 0; b < branch_results.size(); ++b) {
        const auto& br = branch_results[b];
        if (!br.valid) continue;

        for (size_t i = 0; i < br.num_vertices; ++i) {
            float px = br.pc1[i];
            float py = br.pc2[i];
            float pz = br.pc3[i];

            agg.all_pc1.push_back(px);
            agg.all_pc2.push_back(py);
            agg.all_pc3.push_back(pz);
            agg.all_curvature.push_back(br.curvature[i]);
            agg.all_rank.push_back(br.rank[i]);
            agg.branch_id.push_back(b);
            agg.all_vertices.push_back(br.vertices[i]);
            agg.state_id.push_back((b < state_ids.size()) ? state_ids[b] : static_cast<StateId>(b));

            branch_pc1_sum[b] += px;
            branch_pc2_sum[b] += py;
            branch_pc3_sum[b] += pz;
            branch_point_counts[b]++;

            agg.pc1_min = std::min(agg.pc1_min, px);
            agg.pc1_max = std::max(agg.pc1_max, px);
            agg.pc2_min = std::min(agg.pc2_min, py);
            agg.pc2_max = std::max(agg.pc2_max, py);
            agg.pc3_min = std::min(agg.pc3_min, pz);
            agg.pc3_max = std::max(agg.pc3_max, pz);
            agg.curvature_min = std::min(agg.curvature_min, br.curvature[i]);
            agg.curvature_max = std::max(agg.curvature_max, br.curvature[i]);
        }
    }

    // Build branch metadata
    size_t current_start = 0;
    for (size_t b = 0; b < num_branches; ++b) {
        const auto& br = branch_results[b];

        AlignmentAggregation::BranchMetadata meta;
        meta.state_id = (b < state_ids.size()) ? state_ids[b] : static_cast<StateId>(b);
        meta.start_index = current_start;

        if (!br.valid || branch_point_counts[b] == 0) {
            meta.point_count = 0;
            agg.branch_sizes.push_back(0);
            agg.branches.push_back(meta);
            continue;
        }

        meta.point_count = branch_point_counts[b];
        meta.mean_curvature = br.curvature_mean;
        meta.centroid_pc = Vec3{
            branch_pc1_sum[b] / branch_point_counts[b],
            branch_pc2_sum[b] / branch_point_counts[b],
            branch_pc3_sum[b] / branch_point_counts[b]
        };

        agg.branch_sizes.push_back(branch_point_counts[b]);
        agg.branches.push_back(meta);
        current_start += branch_point_counts[b];
    }

    agg.total_points = agg.all_pc1.size();
    agg.num_branches = num_branches;

    return agg;
}

// Backwards-compatible overload
AlignmentAggregation align_branches_per_branch(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    const std::vector<StateId>& state_ids,
    job_system::JobSystem<int>* js,
    const AlignmentReferenceFrame* reference_frame,
    AlignmentReferenceFrame* output_frame
) {
    return align_branches_per_branch(graphs, curvatures, state_ids, js,
                                      reference_frame, output_frame, true, true);
}

// =============================================================================
// Global Branch Alignment (Alternative)
// =============================================================================

AlignmentAggregation align_branches_globally(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    const std::vector<StateId>& state_ids,
    job_system::JobSystem<int>* js,
    const AlignmentReferenceFrame* reference_frame,
    AlignmentReferenceFrame* output_frame
) {
    AlignmentAggregation agg;
    size_t num_branches = std::min({graphs.size(), curvatures.size(), state_ids.size()});

    if (num_branches == 0) return agg;

    // Step 1: Compute spectral embeddings for all branches in parallel
    struct BranchData {
        std::vector<Vec3> positions;
        std::vector<VertexId> vertices;
        std::vector<float> curv_values;
        std::vector<float> masses;
        bool valid = false;
    };
    std::vector<BranchData> branch_data(num_branches);

    for (size_t b = 0; b < num_branches; ++b) {
        js->submit_function([&graphs, &curvatures, &branch_data, b]() {
            const auto& graph = graphs[b];
            const auto& curv_map = curvatures[b];
            const auto& vertices = graph.vertices();
            size_t n = vertices.size();

            if (n < 4) return;

            // Compute spectral embedding
            auto positions = compute_spectral_embedding(graph, 3, 100);
            if (positions.size() != n) return;

            // Extract curvature values
            std::vector<float> curv_values(n);
            std::vector<float> masses(n);
            for (size_t i = 0; i < n; ++i) {
                auto it = curv_map.find(vertices[i]);
                float c = (it != curv_map.end()) ? it->second : 0.0f;
                curv_values[i] = c;
                masses[i] = std::abs(c) + 0.01f;  // Small offset to avoid zero mass
            }

            branch_data[b].positions = std::move(positions);
            branch_data[b].vertices = vertices;
            branch_data[b].curv_values = std::move(curv_values);
            branch_data[b].masses = std::move(masses);
            branch_data[b].valid = true;
        }, 0);
    }
    js->wait_for_completion();

    // Step 2: Collect all points and compute global statistics
    size_t total_points = 0;
    for (const auto& bd : branch_data) {
        if (bd.valid) total_points += bd.positions.size();
    }

    if (total_points == 0) return agg;

    // Collect all positions and masses
    std::vector<Vec3> all_positions;
    std::vector<float> all_masses;
    std::vector<float> all_curvatures_tmp;
    std::vector<VertexId> all_vertices_tmp;
    std::vector<size_t> all_branch_ids;
    std::vector<StateId> all_state_ids;

    all_positions.reserve(total_points);
    all_masses.reserve(total_points);
    all_curvatures_tmp.reserve(total_points);
    all_vertices_tmp.reserve(total_points);
    all_branch_ids.reserve(total_points);
    all_state_ids.reserve(total_points);

    for (size_t b = 0; b < num_branches; ++b) {
        const auto& bd = branch_data[b];
        if (!bd.valid) continue;

        for (size_t i = 0; i < bd.positions.size(); ++i) {
            all_positions.push_back(bd.positions[i]);
            all_masses.push_back(bd.masses[i]);
            all_curvatures_tmp.push_back(bd.curv_values[i]);
            all_vertices_tmp.push_back(bd.vertices[i]);
            all_branch_ids.push_back(b);
            all_state_ids.push_back(state_ids[b]);
        }
    }

    // Step 3: Compute GLOBAL curvature-weighted centroid
    Vec3 global_centroid = compute_weighted_centroid(all_positions, all_masses);

    // Step 4: Center all positions
    std::vector<Vec3> centered(total_points);
    for (size_t i = 0; i < total_points; ++i) {
        centered[i] = all_positions[i] - global_centroid;
    }

    // Step 5: Compute GLOBAL moment of inertia tensor
    Mat3 global_inertia = compute_inertia_tensor(centered, all_masses);

    // Step 6: Global eigendecomposition
    std::array<float, 3> eigenvalues;
    Mat3 eigenvectors;
    eigen_decomposition_3x3(global_inertia, eigenvalues, eigenvectors);

    // Step 6.5: Align to reference frame if provided (for temporal consistency)
    if (reference_frame && reference_frame->valid) {
        const Mat3& ref = reference_frame->eigenvectors;
        Mat3 aligned_evecs;

        // For each axis, find the best matching reference axis and align sign
        for (int i = 0; i < 3; ++i) {
            // Current eigenvector (column i)
            Vec3 curr{eigenvectors.m[0][i], eigenvectors.m[1][i], eigenvectors.m[2][i]};

            // Find best matching reference axis by dot product
            float best_dot = 0.0f;
            int best_axis = i;
            for (int j = 0; j < 3; ++j) {
                Vec3 ref_axis{ref.m[0][j], ref.m[1][j], ref.m[2][j]};
                float dot = curr.x * ref_axis.x + curr.y * ref_axis.y + curr.z * ref_axis.z;
                if (std::abs(dot) > std::abs(best_dot)) {
                    best_dot = dot;
                    best_axis = j;
                }
            }

            // Use reference axis direction, flip sign if needed
            Vec3 ref_axis{ref.m[0][best_axis], ref.m[1][best_axis], ref.m[2][best_axis]};
            float dot = curr.x * ref_axis.x + curr.y * ref_axis.y + curr.z * ref_axis.z;
            float sign = (dot >= 0) ? 1.0f : -1.0f;

            // Store aligned eigenvector
            aligned_evecs.m[0][i] = curr.x * sign;
            aligned_evecs.m[1][i] = curr.y * sign;
            aligned_evecs.m[2][i] = curr.z * sign;
        }

        eigenvectors = aligned_evecs;
    }

    // Output the reference frame for the next timestep
    if (output_frame) {
        output_frame->eigenvectors = eigenvectors;
        output_frame->valid = true;
    }

    // Step 7: Project ALL points onto global principal axes
    Mat3 evT = eigenvectors.transpose();
    std::vector<Vec3> aligned(total_points);
    for (size_t i = 0; i < total_points; ++i) {
        aligned[i] = evT * centered[i];
    }

    // Step 8: Build aggregation result
    agg.all_pc1.resize(total_points);
    agg.all_pc2.resize(total_points);
    agg.all_pc3.resize(total_points);
    agg.all_curvature = std::move(all_curvatures_tmp);
    agg.all_vertices = std::move(all_vertices_tmp);
    agg.branch_id = std::move(all_branch_ids);
    agg.state_id = std::move(all_state_ids);
    agg.all_rank.resize(total_points);

    agg.curvature_min = std::numeric_limits<float>::max();
    agg.curvature_max = std::numeric_limits<float>::lowest();
    agg.pc1_min = std::numeric_limits<float>::max();
    agg.pc1_max = std::numeric_limits<float>::lowest();
    agg.pc2_min = std::numeric_limits<float>::max();
    agg.pc2_max = std::numeric_limits<float>::lowest();
    agg.pc3_min = std::numeric_limits<float>::max();
    agg.pc3_max = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < total_points; ++i) {
        agg.all_pc1[i] = aligned[i].x;
        agg.all_pc2[i] = aligned[i].y;
        agg.all_pc3[i] = aligned[i].z;

        agg.pc1_min = std::min(agg.pc1_min, aligned[i].x);
        agg.pc1_max = std::max(agg.pc1_max, aligned[i].x);
        agg.pc2_min = std::min(agg.pc2_min, aligned[i].y);
        agg.pc2_max = std::max(agg.pc2_max, aligned[i].y);
        agg.pc3_min = std::min(agg.pc3_min, aligned[i].z);
        agg.pc3_max = std::max(agg.pc3_max, aligned[i].z);
        agg.curvature_min = std::min(agg.curvature_min, agg.all_curvature[i]);
        agg.curvature_max = std::max(agg.curvature_max, agg.all_curvature[i]);
    }

    // Compute per-branch rank (within each branch, sort by PC1)
    // Also build branch metadata
    size_t current_idx = 0;
    for (size_t b = 0; b < num_branches; ++b) {
        const auto& bd = branch_data[b];

        AlignmentAggregation::BranchMetadata meta;
        meta.state_id = state_ids[b];
        meta.start_index = current_idx;

        if (!bd.valid) {
            meta.point_count = 0;
            agg.branch_sizes.push_back(0);
            agg.branches.push_back(meta);
            continue;
        }

        size_t n = bd.positions.size();
        meta.point_count = n;

        // Compute mean curvature and centroid for this branch
        float curv_sum = 0;
        float pc1_sum = 0, pc2_sum = 0, pc3_sum = 0;
        for (size_t i = current_idx; i < current_idx + n; ++i) {
            curv_sum += agg.all_curvature[i];
            pc1_sum += agg.all_pc1[i];
            pc2_sum += agg.all_pc2[i];
            pc3_sum += agg.all_pc3[i];
        }
        meta.mean_curvature = curv_sum / n;
        meta.centroid_pc = Vec3{pc1_sum / n, pc2_sum / n, pc3_sum / n};

        // Sort indices by PC1 within this branch to compute rank
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return agg.all_pc1[current_idx + a] < agg.all_pc1[current_idx + b];
        });

        for (size_t r = 0; r < n; ++r) {
            agg.all_rank[current_idx + order[r]] = (n > 1) ? static_cast<float>(r) / (n - 1) : 0.0f;
        }

        agg.branch_sizes.push_back(n);
        agg.branches.push_back(meta);
        current_idx += n;
    }

    agg.total_points = total_points;
    agg.num_branches = num_branches;

    return agg;
}

// =============================================================================
// Global PCA Across All Timesteps (Recommended for Animation)
// =============================================================================

std::vector<AlignmentAggregation> align_all_timesteps_global_pca(
    const std::vector<SimpleGraph>& all_graphs,
    const std::vector<std::unordered_map<VertexId, float>>& all_curvatures,
    const std::vector<StateId>& all_state_ids,
    const std::vector<uint32_t>& state_to_step,
    size_t num_timesteps,
    job_system::JobSystem<int>* js,
    bool use_canonical_orientation,
    bool use_scale_normalization
) {
    std::vector<AlignmentAggregation> results(num_timesteps);
    size_t num_states = std::min({all_graphs.size(), all_curvatures.size(),
                                  all_state_ids.size(), state_to_step.size()});

    if (num_states == 0 || num_timesteps == 0) return results;

    // Step 1: Compute spectral embeddings for ALL states in parallel
    struct StateData {
        std::vector<Vec3> positions;
        std::vector<VertexId> vertices;
        std::vector<float> curv_values;
        std::vector<float> masses;
        bool valid = false;
    };
    std::vector<StateData> state_data(num_states);

    for (size_t s = 0; s < num_states; ++s) {
        js->submit_function([&all_graphs, &all_curvatures, &state_data, s]() {
            const auto& graph = all_graphs[s];
            const auto& curv_map = all_curvatures[s];
            const auto& vertices = graph.vertices();
            size_t n = vertices.size();

            if (n < 4) return;

            auto positions = compute_spectral_embedding(graph, 3, 100);
            if (positions.size() != n) return;

            std::vector<float> curv_values(n);
            std::vector<float> masses(n);
            for (size_t i = 0; i < n; ++i) {
                auto it = curv_map.find(vertices[i]);
                float c = (it != curv_map.end()) ? it->second : 0.0f;
                curv_values[i] = c;
                masses[i] = std::abs(c) + 0.01f;
            }

            state_data[s].positions = std::move(positions);
            state_data[s].vertices = vertices;
            state_data[s].curv_values = std::move(curv_values);
            state_data[s].masses = std::move(masses);
            state_data[s].valid = true;
        }, 0);
    }
    js->wait_for_completion();

    // Step 2: Pool ALL embedded points into one global point cloud
    size_t total_points = 0;
    for (const auto& sd : state_data) {
        if (sd.valid) total_points += sd.positions.size();
    }

    if (total_points == 0) return results;

    std::vector<Vec3> all_positions;
    std::vector<float> all_masses;
    std::vector<float> all_curv_tmp;
    std::vector<size_t> point_to_state;  // Map each point to its state index

    all_positions.reserve(total_points);
    all_masses.reserve(total_points);
    all_curv_tmp.reserve(total_points);
    point_to_state.reserve(total_points);

    for (size_t s = 0; s < num_states; ++s) {
        const auto& sd = state_data[s];
        if (!sd.valid) continue;

        for (size_t i = 0; i < sd.positions.size(); ++i) {
            all_positions.push_back(sd.positions[i]);
            all_masses.push_back(sd.masses[i]);
            all_curv_tmp.push_back(sd.curv_values[i]);
            point_to_state.push_back(s);
        }
    }

    // Step 3: Compute ONE global curvature-weighted centroid
    Vec3 global_centroid = compute_weighted_centroid(all_positions, all_masses);

    // Step 4: Center all positions
    std::vector<Vec3> centered(total_points);
    for (size_t i = 0; i < total_points; ++i) {
        centered[i] = all_positions[i] - global_centroid;
    }

    // Step 5: Compute ONE global moment of inertia tensor
    Mat3 global_inertia = compute_inertia_tensor(centered, all_masses);

    // Step 6: Global eigendecomposition
    std::array<float, 3> eigenvalues;
    Mat3 eigenvectors;
    eigen_decomposition_3x3(global_inertia, eigenvalues, eigenvectors);

    // Step 7: Apply canonical orientation (positive curvature → positive direction)
    // Project all points first
    Mat3 evT = eigenvectors.transpose();
    std::vector<Vec3> projected(total_points);
    for (size_t i = 0; i < total_points; ++i) {
        projected[i] = evT * centered[i];
    }

    if (use_canonical_orientation) {
        canonicalize_orientation(eigenvectors, projected, all_curv_tmp);
    }

    // Step 8: Apply scale normalization
    if (use_scale_normalization) {
        apply_scale_normalization(projected, eigenvalues);
    }

    // Step 9: Compute global bounds for all projected points
    float global_pc1_min = std::numeric_limits<float>::max();
    float global_pc1_max = std::numeric_limits<float>::lowest();
    float global_pc2_min = std::numeric_limits<float>::max();
    float global_pc2_max = std::numeric_limits<float>::lowest();
    float global_pc3_min = std::numeric_limits<float>::max();
    float global_pc3_max = std::numeric_limits<float>::lowest();
    float global_curv_min = std::numeric_limits<float>::max();
    float global_curv_max = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < total_points; ++i) {
        global_pc1_min = std::min(global_pc1_min, projected[i].x);
        global_pc1_max = std::max(global_pc1_max, projected[i].x);
        global_pc2_min = std::min(global_pc2_min, projected[i].y);
        global_pc2_max = std::max(global_pc2_max, projected[i].y);
        global_pc3_min = std::min(global_pc3_min, projected[i].z);
        global_pc3_max = std::max(global_pc3_max, projected[i].z);
        global_curv_min = std::min(global_curv_min, all_curv_tmp[i]);
        global_curv_max = std::max(global_curv_max, all_curv_tmp[i]);
    }

    // Step 10: Group states by timestep
    std::vector<std::vector<size_t>> step_to_states(num_timesteps);
    for (size_t s = 0; s < num_states; ++s) {
        uint32_t step = state_to_step[s];
        if (step < num_timesteps) {
            step_to_states[step].push_back(s);
        }
    }

    // Step 11: Build per-timestep aggregations
    // First, build a map from point index to projected coordinates
    // (already have projected vector)

    // For each timestep, collect points from all states in that timestep
    for (size_t step = 0; step < num_timesteps; ++step) {
        const auto& states_in_step = step_to_states[step];
        AlignmentAggregation& agg = results[step];

        // Count points for this timestep
        size_t step_points = 0;
        for (size_t s : states_in_step) {
            if (state_data[s].valid) {
                step_points += state_data[s].positions.size();
            }
        }

        if (step_points == 0) continue;

        // Reserve
        agg.all_pc1.reserve(step_points);
        agg.all_pc2.reserve(step_points);
        agg.all_pc3.reserve(step_points);
        agg.all_curvature.reserve(step_points);
        agg.all_rank.reserve(step_points);
        agg.branch_id.reserve(step_points);
        agg.all_vertices.reserve(step_points);
        agg.state_id.reserve(step_points);

        // Use global bounds for consistent visualization across timesteps
        agg.pc1_min = global_pc1_min;
        agg.pc1_max = global_pc1_max;
        agg.pc2_min = global_pc2_min;
        agg.pc2_max = global_pc2_max;
        agg.pc3_min = global_pc3_min;
        agg.pc3_max = global_pc3_max;
        agg.curvature_min = global_curv_min;
        agg.curvature_max = global_curv_max;

        // Find where each state's points start in the global arrays
        // Build a state_start map
        std::vector<size_t> state_start(num_states, 0);
        size_t running_idx = 0;
        for (size_t s = 0; s < num_states; ++s) {
            state_start[s] = running_idx;
            if (state_data[s].valid) {
                running_idx += state_data[s].positions.size();
            }
        }

        // Collect points for this timestep
        size_t branch_idx = 0;
        for (size_t s : states_in_step) {
            const auto& sd = state_data[s];

            AlignmentAggregation::BranchMetadata meta;
            meta.state_id = all_state_ids[s];
            meta.start_index = agg.all_pc1.size();

            if (!sd.valid) {
                meta.point_count = 0;
                agg.branch_sizes.push_back(0);
                agg.branches.push_back(meta);
                ++branch_idx;
                continue;
            }

            size_t n = sd.positions.size();
            meta.point_count = n;

            size_t global_start = state_start[s];
            float pc1_sum = 0, pc2_sum = 0, pc3_sum = 0, curv_sum = 0;

            for (size_t i = 0; i < n; ++i) {
                size_t global_idx = global_start + i;
                const Vec3& p = projected[global_idx];

                agg.all_pc1.push_back(p.x);
                agg.all_pc2.push_back(p.y);
                agg.all_pc3.push_back(p.z);
                agg.all_curvature.push_back(all_curv_tmp[global_idx]);
                agg.all_rank.push_back(0.0f);  // Will compute below
                agg.branch_id.push_back(branch_idx);
                agg.all_vertices.push_back(sd.vertices[i]);
                agg.state_id.push_back(all_state_ids[s]);

                pc1_sum += p.x;
                pc2_sum += p.y;
                pc3_sum += p.z;
                curv_sum += all_curv_tmp[global_idx];
            }

            meta.mean_curvature = curv_sum / n;
            meta.centroid_pc = Vec3{pc1_sum / n, pc2_sum / n, pc3_sum / n};

            // Compute rank within this branch (sort by PC1)
            std::vector<size_t> order(n);
            std::iota(order.begin(), order.end(), 0);
            size_t branch_start = meta.start_index;
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return agg.all_pc1[branch_start + a] < agg.all_pc1[branch_start + b];
            });
            for (size_t r = 0; r < n; ++r) {
                agg.all_rank[branch_start + order[r]] = (n > 1) ? static_cast<float>(r) / (n - 1) : 0.0f;
            }

            agg.branch_sizes.push_back(n);
            agg.branches.push_back(meta);
            ++branch_idx;
        }

        agg.total_points = agg.all_pc1.size();
        agg.num_branches = states_in_step.size();
    }

    return results;
}

// =============================================================================
// Aggregation (Legacy - uses independently-aligned branches)
// =============================================================================

AlignmentAggregation aggregate_alignments(
    const std::vector<BranchAlignmentResult>& alignments
) {
    // Default: no state IDs provided, use branch index as state ID
    std::vector<StateId> state_ids(alignments.size());
    for (size_t i = 0; i < alignments.size(); ++i) {
        state_ids[i] = static_cast<StateId>(i);
    }
    return aggregate_alignments(alignments, state_ids);
}

AlignmentAggregation aggregate_alignments(
    const std::vector<BranchAlignmentResult>& alignments,
    const std::vector<StateId>& state_ids
) {
    AlignmentAggregation agg;

    // Count total points and build branch metadata
    size_t total = 0;
    size_t current_index = 0;

    for (size_t b = 0; b < alignments.size(); ++b) {
        const auto& a = alignments[b];

        AlignmentAggregation::BranchMetadata meta;
        meta.state_id = (b < state_ids.size()) ? state_ids[b] : static_cast<StateId>(b);
        meta.start_index = current_index;

        if (a.valid) {
            meta.point_count = a.num_vertices;
            meta.mean_curvature = a.curvature_mean;

            // Compute centroid in PC space
            if (a.num_vertices > 0) {
                float pc1_sum = 0, pc2_sum = 0, pc3_sum = 0;
                for (size_t i = 0; i < a.num_vertices; ++i) {
                    pc1_sum += a.pc1[i];
                    pc2_sum += a.pc2[i];
                    pc3_sum += a.pc3[i];
                }
                meta.centroid_pc = Vec3{
                    pc1_sum / a.num_vertices,
                    pc2_sum / a.num_vertices,
                    pc3_sum / a.num_vertices
                };
            }

            total += a.num_vertices;
            current_index += a.num_vertices;
            agg.branch_sizes.push_back(a.num_vertices);
        } else {
            meta.point_count = 0;
            agg.branch_sizes.push_back(0);
        }

        agg.branches.push_back(meta);
    }

    if (total == 0) return agg;

    // Reserve space
    agg.all_pc1.reserve(total);
    agg.all_pc2.reserve(total);
    agg.all_pc3.reserve(total);
    agg.all_curvature.reserve(total);
    agg.all_rank.reserve(total);
    agg.branch_id.reserve(total);
    agg.all_vertices.reserve(total);
    agg.state_id.reserve(total);

    // Initialize bounds
    agg.curvature_min = std::numeric_limits<float>::max();
    agg.curvature_max = std::numeric_limits<float>::lowest();
    agg.pc1_min = std::numeric_limits<float>::max();
    agg.pc1_max = std::numeric_limits<float>::lowest();
    agg.pc2_min = std::numeric_limits<float>::max();
    agg.pc2_max = std::numeric_limits<float>::lowest();
    agg.pc3_min = std::numeric_limits<float>::max();
    agg.pc3_max = std::numeric_limits<float>::lowest();

    // Collect all points
    for (size_t b = 0; b < alignments.size(); ++b) {
        const auto& a = alignments[b];
        if (!a.valid) continue;

        StateId sid = (b < state_ids.size()) ? state_ids[b] : static_cast<StateId>(b);

        for (size_t i = 0; i < a.num_vertices; ++i) {
            agg.all_pc1.push_back(a.pc1[i]);
            agg.all_pc2.push_back(a.pc2[i]);
            agg.all_pc3.push_back(a.pc3[i]);
            agg.all_curvature.push_back(a.curvature[i]);
            agg.all_rank.push_back(a.rank[i]);
            agg.branch_id.push_back(b);
            agg.all_vertices.push_back(a.vertices[i]);
            agg.state_id.push_back(sid);

            agg.curvature_min = std::min(agg.curvature_min, a.curvature[i]);
            agg.curvature_max = std::max(agg.curvature_max, a.curvature[i]);
            agg.pc1_min = std::min(agg.pc1_min, a.pc1[i]);
            agg.pc1_max = std::max(agg.pc1_max, a.pc1[i]);
            agg.pc2_min = std::min(agg.pc2_min, a.pc2[i]);
            agg.pc2_max = std::max(agg.pc2_max, a.pc2[i]);
            agg.pc3_min = std::min(agg.pc3_min, a.pc3[i]);
            agg.pc3_max = std::max(agg.pc3_max, a.pc3[i]);
        }
    }

    agg.total_points = total;
    agg.num_branches = alignments.size();

    return agg;
}

}  // namespace viz::blackhole
