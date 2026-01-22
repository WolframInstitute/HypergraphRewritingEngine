#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <vector>
#include <unordered_map>

// Forward declaration for job system
namespace job_system {
    template<typename T> class JobSystem;
}

namespace viz::blackhole {

// =============================================================================
// Entropy Types
// =============================================================================

enum class EntropyMethod {
    Degree,           // Entropy of degree distribution
    Local,            // Local structural entropy per vertex
    MutualInfo,       // Mutual information between neighborhoods
    Fisher            // Fisher information from dimension gradients
};

inline const char* entropy_method_name(EntropyMethod m) {
    switch (m) {
        case EntropyMethod::Degree: return "Degree";
        case EntropyMethod::Local: return "Local";
        case EntropyMethod::MutualInfo: return "Mutual Info";
        case EntropyMethod::Fisher: return "Fisher";
        default: return "Unknown";
    }
}

// =============================================================================
// Entropy Result Structures
// =============================================================================

// Per-vertex entropy data
struct VertexEntropy {
    VertexId vertex;
    float local_entropy = 0.0f;       // Local structural entropy
    float mutual_info = 0.0f;         // Mutual info with neighbors
    float fisher_info = 0.0f;         // Fisher information at vertex
    int degree = 0;                   // Vertex degree
};

// Full entropy analysis result
struct EntropyAnalysisResult {
    std::vector<VertexEntropy> vertex_entropies;
    std::unordered_map<VertexId, float> local_entropy_map;
    std::unordered_map<VertexId, float> mutual_info_map;
    std::unordered_map<VertexId, float> fisher_info_map;

    // Global entropy measures
    float degree_entropy = 0.0f;      // Entropy of global degree distribution
    float graph_entropy = 0.0f;       // Overall graph structural entropy
    float total_mutual_info = 0.0f;   // Sum of pairwise mutual info
    float total_fisher_info = 0.0f;   // Sum of Fisher information

    // Statistics for local entropy
    float mean_local_entropy = 0.0f;
    float min_local_entropy = 0.0f;
    float max_local_entropy = 0.0f;

    // Statistics for Fisher info
    float mean_fisher_info = 0.0f;
    float min_fisher_info = 0.0f;
    float max_fisher_info = 0.0f;

    // Quantiles for color scaling
    float local_q05 = 0.0f;
    float local_q95 = 0.0f;
    float fisher_q05 = 0.0f;
    float fisher_q95 = 0.0f;
};

// =============================================================================
// Configuration
// =============================================================================

struct EntropyConfig {
    bool compute_local_entropy = true;
    bool compute_mutual_info = true;
    bool compute_fisher_info = true;

    int neighborhood_radius = 2;      // Radius for local entropy computation
    int fisher_radius = 2;            // Radius for Fisher info computation

    // Binning for entropy computation
    int num_bins = 10;                // Number of bins for discretization
};

// =============================================================================
// Entropy Computation Functions
// =============================================================================

// Compute full entropy analysis on a graph
EntropyAnalysisResult analyze_entropy(
    const SimpleGraph& graph,
    const EntropyConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr  // For Fisher info
);

// Compute entropy of degree distribution
// H(D) = -Σ p(d) log p(d) where p(d) = fraction of vertices with degree d
float compute_degree_entropy(const SimpleGraph& graph);

// Compute local structural entropy at a vertex
// Based on diversity of neighborhood structure
float compute_local_entropy(
    const SimpleGraph& graph,
    VertexId vertex,
    int radius = 2
);

// Compute all local entropies
std::unordered_map<VertexId, float> compute_all_local_entropies(
    const SimpleGraph& graph,
    int radius = 2
);

// Compute mutual information between two vertices' neighborhoods
// I(X;Y) = H(X) + H(Y) - H(X,Y)
float compute_mutual_information(
    const SimpleGraph& graph,
    VertexId v1,
    VertexId v2,
    int radius = 1
);

// Compute mutual info for each vertex (sum of pairwise with neighbors)
std::unordered_map<VertexId, float> compute_all_mutual_info(
    const SimpleGraph& graph,
    int radius = 1
);

// =============================================================================
// Fisher Information
// =============================================================================
// Fisher information measures the "sharpness" of the dimension distribution
// at each vertex. High Fisher info = dimension is well-defined locally.

// Compute Fisher information at a vertex from dimension gradient
float compute_fisher_information(
    const SimpleGraph& graph,
    VertexId vertex,
    const std::vector<float>& vertex_dimensions,
    int radius = 2
);

// Compute Fisher info for all vertices
std::unordered_map<VertexId, float> compute_all_fisher_info(
    const SimpleGraph& graph,
    const std::vector<float>& vertex_dimensions,
    int radius = 2
);

// =============================================================================
// Utility Functions
// =============================================================================

// Shannon entropy of a probability distribution
// H(p) = -Σ p_i log(p_i)
float shannon_entropy(const std::vector<float>& probabilities);

// Normalize counts to probabilities
std::vector<float> counts_to_probabilities(const std::vector<int>& counts);

// Discretize continuous values into bins
std::vector<int> discretize(
    const std::vector<float>& values,
    int num_bins,
    float min_val,
    float max_val
);

// Run entropy analysis on timestep aggregation
EntropyAnalysisResult analyze_entropy_timestep(
    const TimestepAggregation& timestep,
    const EntropyConfig& config = {}
);

// =============================================================================
// Parallel Versions (using job system)
// =============================================================================

// Compute all local entropies in parallel
std::unordered_map<VertexId, float> compute_all_local_entropies_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int radius = 2
);

// Compute all mutual info in parallel
std::unordered_map<VertexId, float> compute_all_mutual_info_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    int radius = 1
);

// Compute all Fisher info in parallel
std::unordered_map<VertexId, float> compute_all_fisher_info_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const std::vector<float>& vertex_dimensions,
    int radius = 2
);

// Full entropy analysis using job system for parallelization
EntropyAnalysisResult analyze_entropy_parallel(
    const SimpleGraph& graph,
    job_system::JobSystem<int>* js,
    const EntropyConfig& config = {},
    const std::vector<float>* vertex_dimensions = nullptr
);

}  // namespace viz::blackhole
