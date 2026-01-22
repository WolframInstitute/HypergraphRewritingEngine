#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <vector>
#include <unordered_map>
#include <array>

// Forward declaration for job system
namespace job_system {
    template<typename T> class JobSystem;
}

namespace viz::blackhole {

// =============================================================================
// 3x3 Matrix Type (for PCA / moment of inertia)
// =============================================================================

struct Mat3 {
    float m[3][3] = {{0}};

    Mat3() = default;

    static Mat3 identity() {
        Mat3 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = 1;
        return r;
    }

    // Helper to access Vec3 component by index
    static float vec3_component(const Vec3& v, int i) {
        return i == 0 ? v.x : (i == 1 ? v.y : v.z);
    }

    static Mat3 outer(const Vec3& a, const Vec3& b) {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = vec3_component(a, i) * vec3_component(b, j);
        return r;
    }

    Mat3 operator+(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] + o.m[i][j];
        return r;
    }

    Mat3 operator-(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] - o.m[i][j];
        return r;
    }

    Mat3 operator*(float s) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] * s;
        return r;
    }

    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    r.m[i][j] += m[i][k] * o.m[k][j];
        return r;
    }

    Mat3& operator+=(const Mat3& o) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] += o.m[i][j];
        return *this;
    }

    Vec3 operator*(const Vec3& v) const {
        return {
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        };
    }

    Mat3 transpose() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[j][i];
        return r;
    }
};

// =============================================================================
// Branch Alignment Result
// =============================================================================

struct BranchAlignmentResult {
    // Vertices in canonical order (sorted by PC1)
    std::vector<VertexId> vertices;

    // Principal component positions for each vertex (in canonical order)
    std::vector<float> pc1;
    std::vector<float> pc2;
    std::vector<float> pc3;

    // Curvature values (in canonical order)
    std::vector<float> curvature;

    // Normalized rank [0, 1] for each vertex
    std::vector<float> rank;

    // Eigenvalues of inertia tensor (principal moments)
    std::array<float, 3> eigenvalues;

    // Eigenvectors (principal axes) as columns
    Mat3 eigenvectors;

    // Curvature-weighted centroid
    Vec3 centroid;

    // Statistics
    size_t num_vertices = 0;
    float curvature_min = 0;
    float curvature_max = 0;
    float curvature_mean = 0;

    bool valid = false;
    std::string error;
};

// =============================================================================
// Spectral Embedding
// =============================================================================

// Laplacian type for spectral embedding
enum class LaplacianType {
    Unnormalized,  // L = D - A (matches Mathematica's SpectralEmbedding)
    Normalized     // L = I - D^{-1/2} A D^{-1/2}
};

// Compute spectral embedding of graph vertices into R^3
// Uses smallest non-trivial eigenvectors of graph Laplacian
// Returns positions for each vertex (indexed same as graph.vertices())
std::vector<Vec3> compute_spectral_embedding(
    const SimpleGraph& graph,
    int num_dimensions = 3,
    int num_iterations = 100,  // Power iteration iterations
    LaplacianType laplacian_type = LaplacianType::Unnormalized  // Default matches Mathematica
);

// =============================================================================
// Moment of Inertia and PCA
// =============================================================================

// Compute moment of inertia tensor for point masses
// masses[i] corresponds to positions[i]
Mat3 compute_inertia_tensor(
    const std::vector<Vec3>& positions,
    const std::vector<float>& masses
);

// Compute curvature-weighted centroid
Vec3 compute_weighted_centroid(
    const std::vector<Vec3>& positions,
    const std::vector<float>& masses
);

// Eigendecomposition of 3x3 symmetric matrix
// Returns eigenvalues (sorted descending) and eigenvectors as columns
void eigen_decomposition_3x3(
    const Mat3& matrix,
    std::array<float, 3>& eigenvalues,
    Mat3& eigenvectors
);

// =============================================================================
// Branch Alignment
// =============================================================================

// Align a single branch using curvature-weighted moment of inertia
// graph: the hypergraph for this state
// curvature: per-vertex curvature values (from Ollivier-Ricci or Wolfram-Ricci)
// use_canonical_orientation: orient axes so positive curvature → positive direction
// use_scale_normalization: divide by sqrt(eigenvalue) for unit variance
BranchAlignmentResult align_branch(
    const SimpleGraph& graph,
    const std::unordered_map<VertexId, float>& curvature,
    bool use_canonical_orientation,
    bool use_scale_normalization
);

// Backwards-compatible overload (defaults to canonical orientation + scale normalization)
BranchAlignmentResult align_branch(
    const SimpleGraph& graph,
    const std::unordered_map<VertexId, float>& curvature
);

// Align multiple branches in parallel (each branch gets its own canonical PCA)
// graphs: one graph per branch
// curvatures: one curvature map per branch
// use_canonical_orientation: orient axes so positive curvature → positive direction
// use_scale_normalization: divide by sqrt(eigenvalue) for unit variance
std::vector<BranchAlignmentResult> align_branches_parallel(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    job_system::JobSystem<int>* js,
    bool use_canonical_orientation,
    bool use_scale_normalization
);

// Backwards-compatible overload (defaults to canonical orientation + scale normalization)
std::vector<BranchAlignmentResult> align_branches_parallel(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    job_system::JobSystem<int>* js
);

// =============================================================================
// Aggregated Analysis
// =============================================================================

struct AlignmentAggregation {
    // All points from all branches combined
    std::vector<float> all_pc1;
    std::vector<float> all_pc2;
    std::vector<float> all_pc3;
    std::vector<float> all_curvature;
    std::vector<float> all_rank;
    std::vector<size_t> branch_id;  // Which branch each point came from

    // Visualization metadata
    std::vector<VertexId> all_vertices;
    std::vector<StateId> state_id;  // Which state each point came from

    struct BranchMetadata {
        StateId state_id = 0;
        size_t start_index = 0;
        size_t point_count = 0;
        float mean_curvature = 0;
        Vec3 centroid_pc{0, 0, 0};
    };
    std::vector<BranchMetadata> branches;

    // Per-branch statistics
    std::vector<size_t> branch_sizes;

    // Global statistics
    float curvature_min = 0;
    float curvature_max = 0;
    float pc1_min = 0, pc1_max = 0;
    float pc2_min = 0, pc2_max = 0;
    float pc3_min = 0, pc3_max = 0;

    size_t total_points = 0;
    size_t num_branches = 0;
};

// Aggregate alignment results from multiple branches
AlignmentAggregation aggregate_alignments(
    const std::vector<BranchAlignmentResult>& alignments
);

// Aggregate with explicit state IDs (for visualization metadata)
AlignmentAggregation aggregate_alignments(
    const std::vector<BranchAlignmentResult>& alignments,
    const std::vector<StateId>& state_ids
);

// =============================================================================
// Per-Branch Alignment with Cross-Branch Matching (Recommended)
// =============================================================================

// Reference frame for cross-branch and cross-timestep alignment
// Stores vertex positions for Kabsch alignment using shared vertices
struct AlignmentReferenceFrame {
    Mat3 eigenvectors;  // Principal axes (columns) - legacy, kept for compatibility

    // For Kabsch alignment: store aligned positions of all vertices
    std::vector<Vec3> vertex_positions;  // Aligned PC coordinates
    std::vector<VertexId> vertex_ids;    // Corresponding vertex IDs

    bool valid = false;
};

// Align multiple branches using PER-BRANCH canonical PCA
// Each branch is aligned to its own canonical form based on intrinsic curvature properties:
// 1. Compute spectral embedding for each branch
// 2. Compute curvature-weighted PCA for EACH branch separately
// 3. Canonical orientation: orient axes so positive curvature → positive direction
// 4. Scale normalization: divide by sqrt(eigenvalue) for unit variance
// 5. Temporal smoothing: find optimal rotation from previous timestep's frame
//
// This gives comparable coordinates within a timestep and smooth animation across timesteps.
AlignmentAggregation align_branches_per_branch(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    const std::vector<StateId>& state_ids,
    job_system::JobSystem<int>* js,
    const AlignmentReferenceFrame* reference_frame,
    AlignmentReferenceFrame* output_frame,
    bool use_canonical_orientation,
    bool use_scale_normalization
);

// Backwards-compatible overload (defaults to canonical orientation + scale normalization)
AlignmentAggregation align_branches_per_branch(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    const std::vector<StateId>& state_ids,
    job_system::JobSystem<int>* js,
    const AlignmentReferenceFrame* reference_frame = nullptr,
    AlignmentReferenceFrame* output_frame = nullptr
);

// =============================================================================
// Global Branch Alignment (Alternative - combines all branches)
// =============================================================================

// Align multiple branches using GLOBAL PCA (alternative approach)
// All branches are embedded into the same coordinate system:
// 1. Compute spectral embedding for each branch independently
// 2. Collect ALL points from all branches
// 3. Compute ONE global curvature-weighted PCA
// 4. Project all points onto global principal axes
// This ensures consistent axes across branches for animation/comparison.
//
// If reference_frame is provided and valid, the resulting axes are aligned
// to match the reference (handles sign/axis ambiguity for smooth animation).
// The function also returns the new reference frame for use in the next timestep.
AlignmentAggregation align_branches_globally(
    const std::vector<SimpleGraph>& graphs,
    const std::vector<std::unordered_map<VertexId, float>>& curvatures,
    const std::vector<StateId>& state_ids,
    job_system::JobSystem<int>* js,
    const AlignmentReferenceFrame* reference_frame = nullptr,
    AlignmentReferenceFrame* output_frame = nullptr
);

// =============================================================================
// Global PCA Across All Timesteps (Recommended for Animation)
// =============================================================================

// Align ALL states across ALL timesteps using a single global PCA.
// This completely eliminates frame-to-frame jitter by computing ONE set of
// principal axes from ALL embedded points across the entire evolution.
//
// Algorithm:
// 1. Compute spectral embedding for each state (parallel)
// 2. Pool ALL embedded points from ALL states into one point cloud
// 3. Compute ONE global curvature-weighted PCA
// 4. Apply canonical orientation once (positive curvature → positive direction)
// 5. Project each state's points onto the global axes
// 6. Group results by timestep
//
// Returns: Vector of AlignmentAggregation indexed by timestep
//
// Parameters:
// - all_graphs: All state graphs (flat list across all timesteps)
// - all_curvatures: Per-vertex curvature for each state
// - all_state_ids: State ID for each graph
// - state_to_step: Maps each state index to its timestep
// - num_timesteps: Total number of timesteps
// - js: Job system for parallelization
// - use_canonical_orientation: Orient axes so positive curvature → positive direction
// - use_scale_normalization: Divide by sqrt(eigenvalue) for unit variance
std::vector<AlignmentAggregation> align_all_timesteps_global_pca(
    const std::vector<SimpleGraph>& all_graphs,
    const std::vector<std::unordered_map<VertexId, float>>& all_curvatures,
    const std::vector<StateId>& all_state_ids,
    const std::vector<uint32_t>& state_to_step,
    size_t num_timesteps,
    job_system::JobSystem<int>* js,
    bool use_canonical_orientation = true,
    bool use_scale_normalization = true
);

}  // namespace viz::blackhole
