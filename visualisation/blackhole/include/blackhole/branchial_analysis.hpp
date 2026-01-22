#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>

// Forward declaration for job system
namespace job_system {
    template<typename T> class JobSystem;
}

namespace viz::blackhole {

// =============================================================================
// Branchial Space Visualization
// =============================================================================
// Branchial space represents the "space of branches" in a multiway evolution.
// Each branch corresponds to a different choice of rule application.
//
// Key concepts from Gorard's paper:
// - States are "stacked" when they contain the same vertex across branches
// - Branchial edges connect states that share common vertices
// - Distribution sharpness measures how localized a vertex is across branches
//
// This module provides tools for:
// 1. Building branchial graph structure
// 2. Computing per-vertex branch distribution
// 3. Measuring distribution sharpness (quantum-like uncertainty)
// 4. Visualizing "stack of nodes" across branches

// =============================================================================
// Branch State Structures
// =============================================================================

// A single state in one branch at one timestep
struct BranchState {
    uint32_t state_id;                    // Unique state identifier
    uint32_t branch_id;                   // Which branch this belongs to
    uint32_t step;                        // Evolution step (time)
    std::vector<VertexId> vertices;       // Vertices in this state
    std::vector<Edge> edges;              // Edges in this state
};

// Vertex occurrence across branches
struct VertexBranchInfo {
    VertexId vertex;
    std::vector<uint32_t> branches;       // Branches containing this vertex
    std::vector<uint32_t> states;         // States containing this vertex
    float distribution_sharpness;         // How localized (1.0 = single branch)
    float branch_entropy;                 // Entropy of branch distribution
};

// Branchial edge (connects states that share vertices)
struct BranchialEdge {
    uint32_t state1;
    uint32_t state2;
    int shared_vertex_count;              // Number of vertices in common
    float overlap_fraction;               // shared / union
};

// =============================================================================
// Branchial Graph Structure
// =============================================================================

struct BranchialGraph {
    std::vector<BranchState> states;
    std::vector<BranchialEdge> edges;

    // Index: vertex -> states containing it
    std::unordered_map<VertexId, std::vector<uint32_t>> vertex_to_states;

    // Index: branch -> states in that branch
    std::unordered_map<uint32_t, std::vector<uint32_t>> branch_to_states;

    // Index: step -> states at that step
    std::unordered_map<uint32_t, std::vector<uint32_t>> step_to_states;

    size_t num_branches() const { return branch_to_states.size(); }
    size_t num_steps() const { return step_to_states.size(); }
};

// =============================================================================
// Branchial Analysis Configuration
// =============================================================================

struct BranchialConfig {
    // Edge construction
    float min_overlap_fraction = 0.1f;    // Minimum overlap to create edge
    int min_shared_vertices = 1;          // Minimum vertices in common

    // Distribution analysis
    bool compute_sharpness = true;
    bool compute_entropy = true;

    // Visualization
    int stack_separation = 5;             // Visual separation between stacked states
    bool group_by_step = true;            // Group states by evolution step
};

// =============================================================================
// Branchial Analysis Results
// =============================================================================

struct BranchialAnalysisResult {
    BranchialGraph graph;

    // Per-vertex branch distribution
    std::vector<VertexBranchInfo> vertex_info;
    std::unordered_map<VertexId, float> vertex_sharpness;
    std::unordered_map<VertexId, float> vertex_entropy;

    // Global statistics
    float mean_sharpness = 0.0f;          // Average distribution sharpness
    float mean_branch_entropy = 0.0f;     // Average entropy
    float max_branches_per_vertex = 0;    // Max branches any vertex spans
    int num_unique_vertices = 0;

    // For visualization
    std::vector<std::pair<uint32_t, uint32_t>> stack_edges;  // Edges connecting stacked states
};

// =============================================================================
// Branchial Analysis Functions
// =============================================================================

// Build branchial graph from list of states
BranchialGraph build_branchial_graph(
    const std::vector<BranchState>& states,
    const BranchialConfig& config = {}
);

// Analyze branchial structure
BranchialAnalysisResult analyze_branchial(
    const std::vector<BranchState>& states,
    const BranchialConfig& config = {}
);

// Compute distribution sharpness for a vertex
// Sharpness = 1 / num_branches (1.0 = localized to single branch)
float compute_vertex_sharpness(
    VertexId vertex,
    const BranchialGraph& graph
);

// Compute branch entropy for a vertex
// H = -Σ p_b log(p_b) where p_b = fraction of appearances in branch b
float compute_vertex_branch_entropy(
    VertexId vertex,
    const BranchialGraph& graph
);

// Get vertices that appear in multiple branches ("delocalized" vertices)
std::vector<VertexId> get_delocalized_vertices(
    const BranchialGraph& graph,
    int min_branches = 2
);

// Get "stacked" states - states at same step that share vertices
std::vector<std::pair<uint32_t, uint32_t>> get_stacked_state_pairs(
    const BranchialGraph& graph,
    int at_step = -1  // -1 = all steps
);

// =============================================================================
// Branchial Embedding Functions
// =============================================================================
// Convert branchial structure to 2D/3D positions for visualization

struct BranchialEmbedding {
    std::vector<Vec2> state_positions_2d;
    std::vector<Vec3> state_positions_3d;
    std::vector<float> vertex_sharpness_colors;  // For heatmap visualization
};

// Embed branchial graph in 2D
// X-axis: step (time), Y-axis: branch separation
BranchialEmbedding embed_branchial_2d(
    const BranchialGraph& graph,
    float step_spacing = 10.0f,
    float branch_spacing = 5.0f
);

// Embed branchial graph in 3D
// X-axis: step, Y-axis: branch, Z-axis: "sharpness" (delocalized vertices elevated)
BranchialEmbedding embed_branchial_3d(
    const BranchialGraph& graph,
    const BranchialAnalysisResult& analysis,
    float step_spacing = 10.0f,
    float branch_spacing = 5.0f,
    float sharpness_scale = 3.0f
);

// =============================================================================
// Utility Functions
// =============================================================================

// Compute overlap between two states (shared vertices / union)
float compute_state_overlap(
    const BranchState& a,
    const BranchState& b
);

// Count shared vertices between two states
int count_shared_vertices(
    const BranchState& a,
    const BranchState& b
);

// Get all vertices across all branches at a given step
std::unordered_set<VertexId> get_vertices_at_step(
    const BranchialGraph& graph,
    uint32_t step
);

// =============================================================================
// Hilbert Space / Bitvector Analysis
// =============================================================================
// Treating branchial space as a discrete Hilbert space where:
// - Each state is a basis vector (bitvector of vertex membership)
// - Inner product ⟨ψ|φ⟩ = normalized count of shared vertices
// - Per-vertex probability = fraction of states containing that vertex
//
// This captures the quantum-like structure of multiway systems.

// Result structure for Hilbert space analysis
struct HilbertSpaceAnalysis {
    // Per-vertex probability: P(vertex exists) across all states at a timestep
    // This is the "probability to be turned on of each bit"
    std::unordered_map<VertexId, float> vertex_probabilities;

    // Inner product matrix between states: inner_products[i][j] = ⟨state_i|state_j⟩
    // Normalized: ⟨ψ|φ⟩ = |ψ ∩ φ| / sqrt(|ψ| * |φ|)
    std::vector<std::vector<float>> inner_product_matrix;

    // Mutual information matrix between states (vertex-level): mi[i][j] = I(state_i; state_j)
    // Computed from joint distribution of vertex membership
    std::vector<std::vector<float>> mutual_information_matrix;

    // Edge-level mutual information matrix: edge_mi[i][j] = I(state_i; state_j)
    // Computed from joint distribution of edge membership
    std::vector<std::vector<float>> edge_mutual_information_matrix;

    // State indices corresponding to matrix rows/columns
    std::vector<uint32_t> state_indices;

    // Statistics
    float mean_inner_product = 0.0f;      // Average off-diagonal inner product
    float max_inner_product = 0.0f;       // Maximum off-diagonal inner product
    float mean_vertex_probability = 0.0f; // Average P(vertex exists)
    float vertex_probability_entropy = 0.0f; // Entropy of vertex probability distribution

    // Mutual information statistics (vertex-level)
    float mean_mutual_information = 0.0f; // Average off-diagonal MI
    float max_mutual_information = 0.0f;  // Maximum off-diagonal MI

    // Edge-level mutual information statistics
    float mean_edge_mutual_information = 0.0f;
    float max_edge_mutual_information = 0.0f;

    // Number of states analyzed
    size_t num_states = 0;
    size_t num_vertices = 0;
};

// Compute inner product between two states (normalized dot product of bitvectors)
// ⟨ψ|φ⟩ = |ψ ∩ φ| / sqrt(|ψ| * |φ|)
// Returns 1.0 for identical states, 0.0 for disjoint states
float compute_state_inner_product(
    const BranchState& a,
    const BranchState& b
);

// Compute mutual information between two states (vertex-level)
// I(A;B) = H(A) + H(B) - H(A,B) where H is entropy of vertex membership
// Each state is treated as a binary vector over the universe of vertices
// Returns bits of shared information between the two states
float compute_state_mutual_information(
    const BranchState& a,
    const BranchState& b,
    const std::unordered_set<VertexId>& universe  // All vertices at this timestep
);

// Compute edge-level mutual information between two states
// I(A;B) where membership is over the edge universe
// Uses edge representation as {v1, v2} pairs
float compute_state_edge_mutual_information(
    const BranchState& a,
    const BranchState& b,
    const std::set<Edge>& edge_universe  // All edges at this timestep
);

// Compute per-vertex probability across all states at a given timestep
// P(v) = (number of states containing v) / (total states at step)
std::unordered_map<VertexId, float> compute_vertex_probabilities(
    const BranchialGraph& graph,
    uint32_t step
);

// Compute full Hilbert space analysis for a timestep
HilbertSpaceAnalysis analyze_hilbert_space(
    const BranchialGraph& graph,
    uint32_t step
);

// Compute Hilbert space analysis for all states (regardless of step)
HilbertSpaceAnalysis analyze_hilbert_space_full(
    const BranchialGraph& graph
);

// =============================================================================
// Parallel Versions (using job system)
// =============================================================================

// Full branchial analysis using job system for parallelization
BranchialAnalysisResult analyze_branchial_parallel(
    const std::vector<BranchState>& states,
    job_system::JobSystem<int>* js,
    const BranchialConfig& config = {}
);

// Hilbert space analysis with parallel inner product computation
HilbertSpaceAnalysis analyze_hilbert_space_parallel(
    const BranchialGraph& graph,
    job_system::JobSystem<int>* js,
    uint32_t step
);

// Full Hilbert space analysis (all states) with parallel inner product computation
HilbertSpaceAnalysis analyze_hilbert_space_full_parallel(
    const BranchialGraph& graph,
    job_system::JobSystem<int>* js
);

}  // namespace viz::blackhole
