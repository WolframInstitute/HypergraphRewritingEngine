#include "blackhole/branchial_analysis.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace viz::blackhole {

// =============================================================================
// Utility Functions
// =============================================================================

int count_shared_vertices(
    const BranchState& a,
    const BranchState& b
) {
    std::unordered_set<VertexId> set_a(a.vertices.begin(), a.vertices.end());
    int count = 0;
    for (VertexId v : b.vertices) {
        if (set_a.count(v)) {
            ++count;
        }
    }
    return count;
}

float compute_state_overlap(
    const BranchState& a,
    const BranchState& b
) {
    std::unordered_set<VertexId> set_a(a.vertices.begin(), a.vertices.end());
    std::unordered_set<VertexId> set_b(b.vertices.begin(), b.vertices.end());

    int intersection = 0;
    for (VertexId v : set_b) {
        if (set_a.count(v)) {
            ++intersection;
        }
    }

    // Union = |A| + |B| - |A ∩ B|
    int union_size = static_cast<int>(set_a.size() + set_b.size()) - intersection;

    if (union_size == 0) return 0.0f;
    return static_cast<float>(intersection) / union_size;
}

std::unordered_set<VertexId> get_vertices_at_step(
    const BranchialGraph& graph,
    uint32_t step
) {
    std::unordered_set<VertexId> vertices;

    auto it = graph.step_to_states.find(step);
    if (it != graph.step_to_states.end()) {
        for (uint32_t sid : it->second) {
            if (sid < graph.states.size()) {
                for (VertexId v : graph.states[sid].vertices) {
                    vertices.insert(v);
                }
            }
        }
    }

    return vertices;
}

// =============================================================================
// Branchial Graph Construction
// =============================================================================

BranchialGraph build_branchial_graph(
    const std::vector<BranchState>& states,
    const BranchialConfig& config
) {
    BranchialGraph graph;
    graph.states = states;

    // Build indices
    for (size_t i = 0; i < states.size(); ++i) {
        const auto& state = states[i];

        // Vertex to states
        for (VertexId v : state.vertices) {
            graph.vertex_to_states[v].push_back(static_cast<uint32_t>(i));
        }

        // Branch to states
        graph.branch_to_states[state.branch_id].push_back(static_cast<uint32_t>(i));

        // Step to states
        graph.step_to_states[state.step].push_back(static_cast<uint32_t>(i));
    }

    // Build branchial edges (between states that share vertices)
    for (size_t i = 0; i < states.size(); ++i) {
        for (size_t j = i + 1; j < states.size(); ++j) {
            // Only connect states at the same step (or different branches at same step)
            if (states[i].step != states[j].step) continue;
            if (states[i].branch_id == states[j].branch_id) continue;

            int shared = count_shared_vertices(states[i], states[j]);
            if (shared < config.min_shared_vertices) continue;

            float overlap = compute_state_overlap(states[i], states[j]);
            if (overlap < config.min_overlap_fraction) continue;

            BranchialEdge edge;
            edge.state1 = static_cast<uint32_t>(i);
            edge.state2 = static_cast<uint32_t>(j);
            edge.shared_vertex_count = shared;
            edge.overlap_fraction = overlap;
            graph.edges.push_back(edge);
        }
    }

    return graph;
}

// =============================================================================
// Distribution Analysis
// =============================================================================

float compute_vertex_sharpness(
    VertexId vertex,
    const BranchialGraph& graph
) {
    auto it = graph.vertex_to_states.find(vertex);
    if (it == graph.vertex_to_states.end() || it->second.empty()) {
        return 1.0f;  // Not found = maximally sharp (localized)
    }

    // Count unique branches containing this vertex
    std::unordered_set<uint32_t> branches;
    for (uint32_t sid : it->second) {
        if (sid < graph.states.size()) {
            branches.insert(graph.states[sid].branch_id);
        }
    }

    if (branches.empty()) return 1.0f;

    // Sharpness = 1 / num_branches
    return 1.0f / branches.size();
}

float compute_vertex_branch_entropy(
    VertexId vertex,
    const BranchialGraph& graph
) {
    auto it = graph.vertex_to_states.find(vertex);
    if (it == graph.vertex_to_states.end() || it->second.empty()) {
        return 0.0f;  // No entropy for missing vertex
    }

    // Count occurrences per branch
    std::unordered_map<uint32_t, int> branch_counts;
    int total = 0;

    for (uint32_t sid : it->second) {
        if (sid < graph.states.size()) {
            branch_counts[graph.states[sid].branch_id]++;
            ++total;
        }
    }

    if (total == 0 || branch_counts.size() <= 1) return 0.0f;

    // Compute entropy: H = -Σ p log(p)
    float entropy = 0.0f;
    for (const auto& [branch, count] : branch_counts) {
        float p = static_cast<float>(count) / total;
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

std::vector<VertexId> get_delocalized_vertices(
    const BranchialGraph& graph,
    int min_branches
) {
    std::vector<VertexId> result;

    for (const auto& [vertex, state_ids] : graph.vertex_to_states) {
        std::unordered_set<uint32_t> branches;
        for (uint32_t sid : state_ids) {
            if (sid < graph.states.size()) {
                branches.insert(graph.states[sid].branch_id);
            }
        }

        if (static_cast<int>(branches.size()) >= min_branches) {
            result.push_back(vertex);
        }
    }

    return result;
}

std::vector<std::pair<uint32_t, uint32_t>> get_stacked_state_pairs(
    const BranchialGraph& graph,
    int at_step
) {
    std::vector<std::pair<uint32_t, uint32_t>> pairs;

    auto add_pairs_for_step = [&](uint32_t step) {
        auto it = graph.step_to_states.find(step);
        if (it == graph.step_to_states.end()) return;

        const auto& states_at_step = it->second;
        for (size_t i = 0; i < states_at_step.size(); ++i) {
            for (size_t j = i + 1; j < states_at_step.size(); ++j) {
                uint32_t si = states_at_step[i];
                uint32_t sj = states_at_step[j];

                // Check if they share vertices
                if (si < graph.states.size() && sj < graph.states.size()) {
                    if (count_shared_vertices(graph.states[si], graph.states[sj]) > 0) {
                        pairs.push_back({si, sj});
                    }
                }
            }
        }
    };

    if (at_step >= 0) {
        add_pairs_for_step(static_cast<uint32_t>(at_step));
    } else {
        for (const auto& [step, _] : graph.step_to_states) {
            add_pairs_for_step(step);
        }
    }

    return pairs;
}

// =============================================================================
// Full Analysis
// =============================================================================

BranchialAnalysisResult analyze_branchial(
    const std::vector<BranchState>& states,
    const BranchialConfig& config
) {
    BranchialAnalysisResult result;

    if (states.empty()) {
        return result;
    }

    // Build graph
    result.graph = build_branchial_graph(states, config);

    // Collect all unique vertices
    std::unordered_set<VertexId> all_vertices;
    for (const auto& state : states) {
        for (VertexId v : state.vertices) {
            all_vertices.insert(v);
        }
    }
    result.num_unique_vertices = static_cast<int>(all_vertices.size());

    // Analyze each vertex
    result.vertex_info.reserve(all_vertices.size());
    float total_sharpness = 0.0f;
    float total_entropy = 0.0f;
    int max_branches = 0;

    for (VertexId v : all_vertices) {
        VertexBranchInfo info;
        info.vertex = v;

        // Get branches and states containing this vertex
        auto it = result.graph.vertex_to_states.find(v);
        if (it != result.graph.vertex_to_states.end()) {
            info.states = it->second;

            std::unordered_set<uint32_t> branches;
            for (uint32_t sid : it->second) {
                if (sid < result.graph.states.size()) {
                    branches.insert(result.graph.states[sid].branch_id);
                }
            }
            info.branches.assign(branches.begin(), branches.end());
        }

        // Compute metrics
        if (config.compute_sharpness) {
            info.distribution_sharpness = compute_vertex_sharpness(v, result.graph);
            result.vertex_sharpness[v] = info.distribution_sharpness;
            total_sharpness += info.distribution_sharpness;
        }

        if (config.compute_entropy) {
            info.branch_entropy = compute_vertex_branch_entropy(v, result.graph);
            result.vertex_entropy[v] = info.branch_entropy;
            total_entropy += info.branch_entropy;
        }

        if (static_cast<int>(info.branches.size()) > max_branches) {
            max_branches = static_cast<int>(info.branches.size());
        }

        result.vertex_info.push_back(std::move(info));
    }

    // Compute global statistics
    if (!all_vertices.empty()) {
        result.mean_sharpness = total_sharpness / all_vertices.size();
        result.mean_branch_entropy = total_entropy / all_vertices.size();
    }
    result.max_branches_per_vertex = static_cast<float>(max_branches);

    // Get stack edges for visualization
    result.stack_edges = get_stacked_state_pairs(result.graph, -1);

    return result;
}

// =============================================================================
// Embedding Functions
// =============================================================================

BranchialEmbedding embed_branchial_2d(
    const BranchialGraph& graph,
    float step_spacing,
    float branch_spacing
) {
    BranchialEmbedding embedding;
    embedding.state_positions_2d.resize(graph.states.size());

    // Assign positions: x = step * spacing, y = branch * spacing
    for (size_t i = 0; i < graph.states.size(); ++i) {
        const auto& state = graph.states[i];
        embedding.state_positions_2d[i] = {
            state.step * step_spacing,
            state.branch_id * branch_spacing
        };
    }

    return embedding;
}

BranchialEmbedding embed_branchial_3d(
    const BranchialGraph& graph,
    const BranchialAnalysisResult& analysis,
    float step_spacing,
    float branch_spacing,
    float sharpness_scale
) {
    BranchialEmbedding embedding;
    embedding.state_positions_3d.resize(graph.states.size());

    for (size_t i = 0; i < graph.states.size(); ++i) {
        const auto& state = graph.states[i];

        // Compute mean sharpness for this state's vertices
        float mean_sharpness = 0.0f;
        int count = 0;
        for (VertexId v : state.vertices) {
            auto it = analysis.vertex_sharpness.find(v);
            if (it != analysis.vertex_sharpness.end()) {
                mean_sharpness += it->second;
                ++count;
            }
        }
        if (count > 0) mean_sharpness /= count;

        // Z-coordinate: inverted sharpness (delocalized states are elevated)
        float z = (1.0f - mean_sharpness) * sharpness_scale;

        embedding.state_positions_3d[i] = {
            state.step * step_spacing,
            state.branch_id * branch_spacing,
            z
        };
    }

    return embedding;
}

// =============================================================================
// Hilbert Space / Bitvector Analysis
// =============================================================================

float compute_state_inner_product(
    const BranchState& a,
    const BranchState& b
) {
    if (a.vertices.empty() || b.vertices.empty()) {
        return 0.0f;
    }

    // Build set for fast lookup
    std::unordered_set<VertexId> set_a(a.vertices.begin(), a.vertices.end());

    // Count intersection
    int intersection = 0;
    for (VertexId v : b.vertices) {
        if (set_a.count(v)) {
            ++intersection;
        }
    }

    // Normalized inner product: |ψ ∩ φ| / sqrt(|ψ| * |φ|)
    float norm = std::sqrt(static_cast<float>(a.vertices.size() * b.vertices.size()));
    return static_cast<float>(intersection) / norm;
}

std::unordered_map<VertexId, float> compute_vertex_probabilities(
    const BranchialGraph& graph,
    uint32_t step
) {
    std::unordered_map<VertexId, float> probabilities;

    auto it = graph.step_to_states.find(step);
    if (it == graph.step_to_states.end() || it->second.empty()) {
        return probabilities;
    }

    const auto& state_indices = it->second;
    float num_states = static_cast<float>(state_indices.size());

    // Count occurrences of each vertex across states at this step
    std::unordered_map<VertexId, int> vertex_counts;

    for (uint32_t sid : state_indices) {
        if (sid < graph.states.size()) {
            for (VertexId v : graph.states[sid].vertices) {
                vertex_counts[v]++;
            }
        }
    }

    // Convert counts to probabilities
    for (const auto& [vertex, count] : vertex_counts) {
        probabilities[vertex] = static_cast<float>(count) / num_states;
    }

    return probabilities;
}

HilbertSpaceAnalysis analyze_hilbert_space(
    const BranchialGraph& graph,
    uint32_t step
) {
    HilbertSpaceAnalysis result;

    auto it = graph.step_to_states.find(step);
    if (it == graph.step_to_states.end() || it->second.empty()) {
        return result;
    }

    result.state_indices = it->second;
    result.num_states = result.state_indices.size();

    // Compute vertex probabilities
    result.vertex_probabilities = compute_vertex_probabilities(graph, step);
    result.num_vertices = result.vertex_probabilities.size();

    // Compute inner product matrix
    size_t n = result.num_states;
    result.inner_product_matrix.resize(n, std::vector<float>(n, 0.0f));

    float sum_off_diagonal = 0.0f;
    int off_diagonal_count = 0;

    for (size_t i = 0; i < n; ++i) {
        uint32_t si = result.state_indices[i];
        if (si >= graph.states.size()) continue;

        for (size_t j = i; j < n; ++j) {
            uint32_t sj = result.state_indices[j];
            if (sj >= graph.states.size()) continue;

            float ip = compute_state_inner_product(graph.states[si], graph.states[sj]);
            result.inner_product_matrix[i][j] = ip;
            result.inner_product_matrix[j][i] = ip;

            if (i != j) {
                sum_off_diagonal += ip;
                ++off_diagonal_count;
                if (ip > result.max_inner_product) {
                    result.max_inner_product = ip;
                }
            }
        }
    }

    if (off_diagonal_count > 0) {
        result.mean_inner_product = sum_off_diagonal / off_diagonal_count;
    }

    // Compute mean vertex probability and entropy
    if (!result.vertex_probabilities.empty()) {
        float sum_prob = 0.0f;
        float entropy = 0.0f;

        for (const auto& [vertex, prob] : result.vertex_probabilities) {
            sum_prob += prob;
            if (prob > 0.0f && prob < 1.0f) {
                // Binary entropy contribution
                entropy -= prob * std::log2(prob);
            }
        }

        result.mean_vertex_probability = sum_prob / result.vertex_probabilities.size();
        result.vertex_probability_entropy = entropy;
    }

    return result;
}

HilbertSpaceAnalysis analyze_hilbert_space_full(
    const BranchialGraph& graph
) {
    HilbertSpaceAnalysis result;

    if (graph.states.empty()) {
        return result;
    }

    // Use all states
    result.state_indices.reserve(graph.states.size());
    for (size_t i = 0; i < graph.states.size(); ++i) {
        result.state_indices.push_back(static_cast<uint32_t>(i));
    }
    result.num_states = result.state_indices.size();

    // Compute vertex probabilities across all states
    std::unordered_map<VertexId, int> vertex_counts;
    for (const auto& state : graph.states) {
        for (VertexId v : state.vertices) {
            vertex_counts[v]++;
        }
    }

    float num_states = static_cast<float>(graph.states.size());
    for (const auto& [vertex, count] : vertex_counts) {
        result.vertex_probabilities[vertex] = static_cast<float>(count) / num_states;
    }
    result.num_vertices = result.vertex_probabilities.size();

    // Compute inner product matrix
    size_t n = result.num_states;
    result.inner_product_matrix.resize(n, std::vector<float>(n, 0.0f));

    float sum_off_diagonal = 0.0f;
    int off_diagonal_count = 0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            float ip = compute_state_inner_product(graph.states[i], graph.states[j]);
            result.inner_product_matrix[i][j] = ip;
            result.inner_product_matrix[j][i] = ip;

            if (i != j) {
                sum_off_diagonal += ip;
                ++off_diagonal_count;
                if (ip > result.max_inner_product) {
                    result.max_inner_product = ip;
                }
            }
        }
    }

    if (off_diagonal_count > 0) {
        result.mean_inner_product = sum_off_diagonal / off_diagonal_count;
    }

    // Compute mean vertex probability and entropy
    if (!result.vertex_probabilities.empty()) {
        float sum_prob = 0.0f;
        float entropy = 0.0f;

        for (const auto& [vertex, prob] : result.vertex_probabilities) {
            sum_prob += prob;
            if (prob > 0.0f && prob < 1.0f) {
                entropy -= prob * std::log2(prob);
            }
        }

        result.mean_vertex_probability = sum_prob / result.vertex_probabilities.size();
        result.vertex_probability_entropy = entropy;
    }

    return result;
}

}  // namespace viz::blackhole
