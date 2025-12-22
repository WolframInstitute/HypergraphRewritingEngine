// Layout manager implementation
// Handles backend selection and provides factory function

#include <layout/layout_engine.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <random>

namespace viz::layout {

// Forward declarations for backend-specific factory functions
#ifdef VIZ_HAS_CUDA
std::unique_ptr<ILayoutEngine> create_cuda_layout_engine();
bool is_cuda_available();
#endif

// =============================================================================
// Barnes-Hut Quadtree for O(n log n) repulsion computation
// =============================================================================

struct QuadTreeNode {
    // Bounding box
    float x_min, y_min, x_max, y_max;

    // Center of mass and total mass
    float mass_cx = 0, mass_cy = 0;
    float total_mass = 0;

    // Size (for theta comparison)
    float size = 0;

    // Children (nullptr if leaf)
    std::unique_ptr<QuadTreeNode> nw, ne, sw, se;

    // Leaf data: single vertex index (-1 if internal or empty)
    int vertex_idx = -1;

    bool is_leaf() const { return !nw && !ne && !sw && !se; }
    bool is_empty() const { return total_mass == 0; }
};

class QuadTree {
public:
    void build(const std::vector<float>& pos_x,
               const std::vector<float>& pos_y,
               const std::vector<float>& masses) {
        if (pos_x.empty()) {
            root_.reset();
            return;
        }

        // Find bounding box
        float x_min = pos_x[0], x_max = pos_x[0];
        float y_min = pos_y[0], y_max = pos_y[0];
        for (size_t i = 1; i < pos_x.size(); ++i) {
            x_min = std::min(x_min, pos_x[i]);
            x_max = std::max(x_max, pos_x[i]);
            y_min = std::min(y_min, pos_y[i]);
            y_max = std::max(y_max, pos_y[i]);
        }

        // Add small padding to avoid edge cases
        float pad = 0.01f * std::max(x_max - x_min, y_max - y_min) + 0.1f;
        x_min -= pad; x_max += pad;
        y_min -= pad; y_max += pad;

        // Store positions for insertion
        pos_x_ = &pos_x;
        pos_y_ = &pos_y;
        masses_ = &masses;

        root_ = std::make_unique<QuadTreeNode>();
        root_->x_min = x_min; root_->x_max = x_max;
        root_->y_min = y_min; root_->y_max = y_max;
        root_->size = std::sqrt((x_max - x_min) * (x_max - x_min) +
                                (y_max - y_min) * (y_max - y_min));

        for (size_t i = 0; i < pos_x.size(); ++i) {
            insert(root_.get(), static_cast<int>(i));
        }
    }

    // Build tree with only vertices where include_mask[i] is true
    // Used for ghost layout: build separate trees for visible and hidden vertices
    void build_masked(const std::vector<float>& pos_x,
                      const std::vector<float>& pos_y,
                      const std::vector<float>& masses,
                      const std::vector<bool>& include_mask) {
        if (pos_x.empty() || include_mask.empty()) {
            root_.reset();
            return;
        }

        // Collect indices of included vertices
        std::vector<int> included_indices;
        for (size_t i = 0; i < pos_x.size() && i < include_mask.size(); ++i) {
            if (include_mask[i]) {
                included_indices.push_back(static_cast<int>(i));
            }
        }

        if (included_indices.empty()) {
            root_.reset();
            return;
        }

        // Find bounding box of included vertices only
        float x_min = pos_x[included_indices[0]], x_max = pos_x[included_indices[0]];
        float y_min = pos_y[included_indices[0]], y_max = pos_y[included_indices[0]];
        for (size_t i = 1; i < included_indices.size(); ++i) {
            int idx = included_indices[i];
            x_min = std::min(x_min, pos_x[idx]);
            x_max = std::max(x_max, pos_x[idx]);
            y_min = std::min(y_min, pos_y[idx]);
            y_max = std::max(y_max, pos_y[idx]);
        }

        // Add small padding to avoid edge cases
        float pad = 0.01f * std::max(x_max - x_min, y_max - y_min) + 0.1f;
        x_min -= pad; x_max += pad;
        y_min -= pad; y_max += pad;

        // Store positions for insertion
        pos_x_ = &pos_x;
        pos_y_ = &pos_y;
        masses_ = &masses;

        root_ = std::make_unique<QuadTreeNode>();
        root_->x_min = x_min; root_->x_max = x_max;
        root_->y_min = y_min; root_->y_max = y_max;
        root_->size = std::sqrt((x_max - x_min) * (x_max - x_min) +
                                (y_max - y_min) * (y_max - y_min));

        // Insert only included vertices
        for (int idx : included_indices) {
            insert(root_.get(), idx);
        }
    }

    // Compute repulsion force on vertex i using Barnes-Hut approximation
    void compute_repulsion(int i, float theta, float repulsion_k,
                           float& force_x, float& force_y) const {
        if (!root_) return;
        compute_repulsion_recursive(root_.get(), i, theta, repulsion_k, force_x, force_y);
    }

private:
    void insert(QuadTreeNode* node, int idx, int depth = 0) {
        // Safety: prevent infinite recursion from coincident points
        // At depth 30, cell size is ~1e-9 of original - treat as coincident
        constexpr int MAX_DEPTH = 30;

        if (node->is_empty() && node->is_leaf()) {
            // Empty leaf - just store this vertex
            node->vertex_idx = idx;
            node->mass_cx = (*pos_x_)[idx];
            node->mass_cy = (*pos_y_)[idx];
            node->total_mass = (*masses_)[idx];
            return;
        }

        // Update center of mass
        float new_mass = node->total_mass + (*masses_)[idx];
        node->mass_cx = (node->mass_cx * node->total_mass + (*pos_x_)[idx] * (*masses_)[idx]) / new_mass;
        node->mass_cy = (node->mass_cy * node->total_mass + (*pos_y_)[idx] * (*masses_)[idx]) / new_mass;
        node->total_mass = new_mass;

        if (node->is_leaf() && node->vertex_idx >= 0) {
            // At max depth, just accumulate mass without subdividing (coincident points)
            if (depth >= MAX_DEPTH) {
                return;
            }
            // Subdivide and reinsert existing vertex
            subdivide(node);
            int old_idx = node->vertex_idx;
            node->vertex_idx = -1;
            insert_into_child(node, old_idx, depth + 1);
        }

        if (!node->is_leaf()) {
            insert_into_child(node, idx, depth + 1);
        }
    }

    void subdivide(QuadTreeNode* node) {
        float mid_x = (node->x_min + node->x_max) / 2;
        float mid_y = (node->y_min + node->y_max) / 2;
        float child_size = node->size / 2;

        node->nw = std::make_unique<QuadTreeNode>();
        node->nw->x_min = node->x_min; node->nw->x_max = mid_x;
        node->nw->y_min = mid_y; node->nw->y_max = node->y_max;
        node->nw->size = child_size;

        node->ne = std::make_unique<QuadTreeNode>();
        node->ne->x_min = mid_x; node->ne->x_max = node->x_max;
        node->ne->y_min = mid_y; node->ne->y_max = node->y_max;
        node->ne->size = child_size;

        node->sw = std::make_unique<QuadTreeNode>();
        node->sw->x_min = node->x_min; node->sw->x_max = mid_x;
        node->sw->y_min = node->y_min; node->sw->y_max = mid_y;
        node->sw->size = child_size;

        node->se = std::make_unique<QuadTreeNode>();
        node->se->x_min = mid_x; node->se->x_max = node->x_max;
        node->se->y_min = node->y_min; node->se->y_max = mid_y;
        node->se->size = child_size;
    }

    void insert_into_child(QuadTreeNode* node, int idx, int depth) {
        float x = (*pos_x_)[idx];
        float y = (*pos_y_)[idx];
        float mid_x = (node->x_min + node->x_max) / 2;
        float mid_y = (node->y_min + node->y_max) / 2;

        QuadTreeNode* child;
        if (x < mid_x) {
            child = (y >= mid_y) ? node->nw.get() : node->sw.get();
        } else {
            child = (y >= mid_y) ? node->ne.get() : node->se.get();
        }
        insert(child, idx, depth);
    }

    void compute_repulsion_recursive(const QuadTreeNode* node, int i, float theta,
                                      float repulsion_k, float& fx, float& fy) const {
        if (node->is_empty()) return;

        float dx = (*pos_x_)[i] - node->mass_cx;
        float dy = (*pos_y_)[i] - node->mass_cy;
        float dist_sq = dx * dx + dy * dy + 0.0001f;
        float dist = std::sqrt(dist_sq);

        // Barnes-Hut criterion: if node is far enough, treat as single body
        if (node->is_leaf() || (node->size / dist < theta)) {
            // Skip self-interaction
            if (node->is_leaf() && node->vertex_idx == i) return;

            // Apply repulsion from this node's center of mass
            float f = repulsion_k * (*masses_)[i] * node->total_mass / dist_sq;
            fx += f * dx / dist;
            fy += f * dy / dist;
        } else {
            // Recurse into children
            if (node->nw) compute_repulsion_recursive(node->nw.get(), i, theta, repulsion_k, fx, fy);
            if (node->ne) compute_repulsion_recursive(node->ne.get(), i, theta, repulsion_k, fx, fy);
            if (node->sw) compute_repulsion_recursive(node->sw.get(), i, theta, repulsion_k, fx, fy);
            if (node->se) compute_repulsion_recursive(node->se.get(), i, theta, repulsion_k, fx, fy);
        }
    }

    std::unique_ptr<QuadTreeNode> root_;
    const std::vector<float>* pos_x_ = nullptr;
    const std::vector<float>* pos_y_ = nullptr;
    const std::vector<float>* masses_ = nullptr;
};

// =============================================================================
// CPU Layout Engine with Barnes-Hut optimization
// =============================================================================

// CPU fallback implementation (for debugging and WebGPU builds without compute)
class CPULayoutEngine : public ILayoutEngine {
public:
    LayoutBackend get_backend() const override { return LayoutBackend::CPU; }

    void upload_graph(const LayoutGraph& graph) override {
        graph_ = graph;
        uint32_t n = graph_.vertex_count();

        // Initialize/resize persistent buffers (no allocation during iterate)
        velocities_x_.resize(n, 0.0f);
        velocities_y_.resize(n, 0.0f);
        velocities_z_.resize(n, 0.0f);
        force_x_.resize(n);
        force_y_.resize(n);
        force_z_.resize(n);

        // Per-vertex stability tracking for oscillation prevention
        settled_count_.resize(n, 0);      // Frames at low velocity
        prev_force_x_.resize(n, 0.0f);    // Previous force direction
        prev_force_y_.resize(n, 0.0f);
        oscillation_count_.resize(n, 0);  // Force direction reversals

        // Reset velocities on new graph
        std::fill(velocities_x_.begin(), velocities_x_.end(), 0.0f);
        std::fill(velocities_y_.begin(), velocities_y_.end(), 0.0f);
        std::fill(velocities_z_.begin(), velocities_z_.end(), 0.0f);
        std::fill(settled_count_.begin(), settled_count_.end(), 0);
        std::fill(prev_force_x_.begin(), prev_force_x_.end(), 0.0f);
        std::fill(prev_force_y_.begin(), prev_force_y_.end(), 0.0f);
        std::fill(oscillation_count_.begin(), oscillation_count_.end(), 0);

        // Build edge index array for shuffling
        edge_indices_.resize(graph_.edge_count());
        for (uint32_t i = 0; i < graph_.edge_count(); i++) {
            edge_indices_[i] = i;
        }

        iteration_count_ = 0;
    }

    void download_positions(LayoutGraph& graph) override {
        graph.positions_x = graph_.positions_x;
        graph.positions_y = graph_.positions_y;
        graph.positions_z = graph_.positions_z;
    }

    LayoutResult iterate(const LayoutParams& params) override {
        // Barnes-Hut O(n log n) implementation with optional edge budget
        // Supports ghost layout: hidden vertices receive forces but don't influence visible ones
        uint32_t n = graph_.vertex_count();
        if (n == 0) {
            return {true, 0.0f, 0.0f, 0, 0.0};
        }

        // Zero force accumulators (no allocation, just memset)
        std::fill(force_x_.begin(), force_x_.end(), 0.0f);
        std::fill(force_y_.begin(), force_y_.end(), 0.0f);
        std::fill(force_z_.begin(), force_z_.end(), 0.0f);

        // Ghost layout model: hidden vertices receive forces but don't influence visible ones
        // Force from j to i applies if: visible[j] || !visible[i]
        // Skip if: !visible[j] && visible[i] (hidden shouldn't push visible)
        const bool has_visibility = !graph_.visible.empty();
        auto should_apply_force = [&](uint32_t source, uint32_t target) -> bool {
            if (!has_visibility) return true;  // No mask = all visible
            bool source_visible = graph_.visible[source];
            bool target_visible = graph_.visible[target];
            return source_visible || !target_visible;
        };

        // Build quadtree for Barnes-Hut approximation (2D only for now)
        // Theta controls accuracy vs speed: higher = faster but less accurate
        // Typical values: 0.5 (accurate) to 1.5 (fast)
        const float theta = 1.2f;

        // 3D requires O(n²) direct computation (octree not implemented)
        // 2D can use Barnes-Hut O(n log n)
        bool use_direct = (params.dimension == LayoutDimension::Layout3D);

        if (!use_direct) {
            if (has_visibility) {
                // Two-tree Barnes-Hut for ghost layout
                // Build separate trees for visible and hidden vertices
                // Visible vertices: only affected by visible tree (hidden→visible blocked)
                // Hidden vertices: affected by both trees (visible→hidden ✓, hidden→hidden ✓)

                // Build inverted mask for hidden vertices
                hidden_mask_.resize(n);
                for (uint32_t i = 0; i < n; i++) {
                    hidden_mask_[i] = !graph_.visible[i];
                }

                visible_quadtree_.build_masked(graph_.positions_x, graph_.positions_y,
                                               graph_.masses, graph_.visible);
                hidden_quadtree_.build_masked(graph_.positions_x, graph_.positions_y,
                                              graph_.masses, hidden_mask_);

                for (uint32_t i = 0; i < n; i++) {
                    if (graph_.visible[i]) {
                        // Visible vertex: only query visible tree (hidden→visible blocked)
                        visible_quadtree_.compute_repulsion(static_cast<int>(i), theta,
                                                           params.repulsion_constant,
                                                           force_x_[i], force_y_[i]);
                    } else {
                        // Hidden vertex: query BOTH trees (visible→hidden ✓, hidden→hidden ✓)
                        visible_quadtree_.compute_repulsion(static_cast<int>(i), theta,
                                                           params.repulsion_constant,
                                                           force_x_[i], force_y_[i]);
                        hidden_quadtree_.compute_repulsion(static_cast<int>(i), theta,
                                                          params.repulsion_constant,
                                                          force_x_[i], force_y_[i]);
                    }
                }
            } else {
                // Single-tree Barnes-Hut (no visibility mask)
                quadtree_.build(graph_.positions_x, graph_.positions_y, graph_.masses);
                for (uint32_t i = 0; i < n; i++) {
                    quadtree_.compute_repulsion(static_cast<int>(i), theta,
                                               params.repulsion_constant,
                                               force_x_[i], force_y_[i]);
                }
            }
        } else {
            // O(n^2) direct computation (3D only)
            for (uint32_t i = 0; i < n; i++) {
                for (uint32_t j = i + 1; j < n; j++) {
                    float dx = graph_.positions_x[i] - graph_.positions_x[j];
                    float dy = graph_.positions_y[i] - graph_.positions_y[j];
                    float dz = graph_.positions_z[i] - graph_.positions_z[j];

                    float dist_sq = dx * dx + dy * dy + dz * dz + 0.0001f;
                    float dist = std::sqrt(dist_sq);
                    float f = params.repulsion_constant * graph_.masses[i] * graph_.masses[j] / dist_sq;

                    float fx = f * dx / dist;
                    float fy = f * dy / dist;
                    float fz = f * dz / dist;

                    // Ghost model: only apply force if source can influence target
                    // Force from j to i: j pushes i away
                    if (should_apply_force(j, i)) {
                        force_x_[i] += fx;
                        force_y_[i] += fy;
                        force_z_[i] += fz;
                    }
                    // Force from i to j: i pushes j away
                    if (should_apply_force(i, j)) {
                        force_x_[j] -= fx;
                        force_y_[j] -= fy;
                        force_z_[j] -= fz;
                    }
                }
            }
        }

        // Gravity (pull toward origin)
        if (params.gravity > 0.0f) {
            for (uint32_t i = 0; i < n; i++) {
                float dx = -graph_.positions_x[i];
                float dy = -graph_.positions_y[i];
                float dz = -graph_.positions_z[i];
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 0.0001f);
                float f = params.gravity * graph_.masses[i];
                force_x_[i] += f * dx / dist;
                force_y_[i] += f * dy / dist;
                force_z_[i] += f * dz / dist;
            }
        }

        // Attractive forces (springs) - with optional edge budget
        uint32_t edge_count = graph_.edge_count();
        uint32_t edges_to_process = edge_count;
        float edge_scale = 1.0f;

        if (params.edge_budget > 0 && edge_count > params.edge_budget) {
            // Stochastic edge sampling: shuffle and take first edge_budget edges
            // Scale forces to compensate for sampling (unbiased estimator)
            edges_to_process = params.edge_budget;
            edge_scale = static_cast<float>(edge_count) / static_cast<float>(edges_to_process);

            // Fisher-Yates partial shuffle: only shuffle first edge_budget elements
            for (uint32_t i = 0; i < edges_to_process; i++) {
                std::uniform_int_distribution<uint32_t> dist(i, edge_count - 1);
                uint32_t j = dist(rng_);
                std::swap(edge_indices_[i], edge_indices_[j]);
            }
        }

        for (uint32_t i = 0; i < edges_to_process; i++) {
            uint32_t e = edge_indices_[i];
            uint32_t src = graph_.edge_sources[e];
            uint32_t dst = graph_.edge_targets[e];

            float dx = graph_.positions_x[dst] - graph_.positions_x[src];
            float dy = graph_.positions_y[dst] - graph_.positions_y[src];
            float dz = graph_.positions_z[dst] - graph_.positions_z[src];

            float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 0.0001f);
            float rest = graph_.edge_rest_lengths[e];
            if (rest <= 0.0f) rest = 1.0f;

            // Scale by edge_scale to compensate for sampling
            float f = edge_scale * params.spring_constant * graph_.edge_strengths[e] * (dist - rest) / dist;

            float fx = f * dx;
            float fy = f * dy;
            float fz = f * dz;

            // Ghost model: only apply spring force if source can influence target
            // Spring from dst to src: dst pulls src toward it
            if (should_apply_force(dst, src)) {
                force_x_[src] += fx;
                force_y_[src] += fy;
                force_z_[src] += fz;
            }
            // Spring from src to dst: src pulls dst toward it
            if (should_apply_force(src, dst)) {
                force_x_[dst] -= fx;
                force_y_[dst] -= fy;
                force_z_[dst] -= fz;
            }
        }

        // Zero Z forces in 2D mode
        if (params.dimension == LayoutDimension::Layout2D) {
            for (uint32_t i = 0; i < n; i++) {
                force_z_[i] = 0.0f;
            }
        }

        // Velocity-free gradient descent integration
        // This approach CANNOT oscillate because there's no stored momentum.
        // Instead of: velocity += force; velocity *= damping; position += velocity
        // We use: position += force * step_size (capped to max_displacement)
        //
        // The step_size (damping parameter) controls how quickly we move toward equilibrium.
        // Smaller step = smoother but slower convergence.
        // Larger step = faster but potential overshooting (but still no oscillation).
        float max_disp = 0.0f;
        float sum_disp = 0.0f;

        // Step size: damping parameter now acts as step multiplier (0 to 1 range)
        // Lower values = smaller steps = smoother animation but slower convergence
        const float step_size = params.damping;
        const float stop_threshold = params.max_displacement * 0.01f;  // Below this, vertex is settled

        for (uint32_t i = 0; i < n; i++) {
            if (graph_.pinned[i]) continue;

            // Compute force magnitude
            float force_mag_sq = force_x_[i] * force_x_[i] +
                                 force_y_[i] * force_y_[i] +
                                 force_z_[i] * force_z_[i];

            if (force_mag_sq < 1e-20f) {
                // Negligible force - vertex is at equilibrium
                continue;
            }

            // Compute displacement directly from force (no velocity accumulation)
            float dx = force_x_[i] * step_size;
            float dy = force_y_[i] * step_size;
            float dz = force_z_[i] * step_size;

            // Cap displacement to max_displacement
            float disp_sq = dx * dx + dy * dy + dz * dz;
            if (disp_sq > params.max_displacement * params.max_displacement) {
                float scale = params.max_displacement / std::sqrt(disp_sq);
                dx *= scale;
                dy *= scale;
                dz *= scale;
                disp_sq = params.max_displacement * params.max_displacement;
            }

            float disp = std::sqrt(disp_sq);

            // Skip tiny movements to prevent numerical noise
            if (disp < stop_threshold) {
                continue;
            }

            // Apply position update
            graph_.positions_x[i] += dx;
            graph_.positions_y[i] += dy;
            graph_.positions_z[i] += dz;

            max_disp = std::max(max_disp, disp);
            sum_disp += disp;
        }

        // Clear velocities (not used in gradient descent, but kept for interface compatibility)
        std::fill(velocities_x_.begin(), velocities_x_.end(), 0.0f);
        std::fill(velocities_y_.begin(), velocities_y_.end(), 0.0f);
        std::fill(velocities_z_.begin(), velocities_z_.end(), 0.0f);

        iteration_count_++;

        LayoutResult result;
        result.converged = (sum_disp / n) < params.convergence_threshold;
        result.max_displacement = max_disp;
        result.average_displacement = sum_disp / n;
        result.iteration_count = iteration_count_;
        result.compute_time_ms = 0.0;  // Not measured for CPU

        return result;
    }

    LayoutResult run_until_converged(const LayoutParams& params,
                                      LayoutProgressCallback progress) override {
        LayoutResult result;
        for (uint32_t i = 0; i < params.max_iterations; i++) {
            result = iterate(params);
            if (progress) {
                progress(result.iteration_count, result.average_displacement);
            }
            if (result.converged) break;
        }
        return result;
    }

    bool has_graph() const override { return graph_.vertex_count() > 0; }
    uint32_t vertex_count() const override { return graph_.vertex_count(); }

    void seed_from(const LayoutGraph& parent,
                   const std::vector<uint32_t>& vertex_mapping,
                   const std::vector<float>& seed_x,
                   const std::vector<float>& seed_y,
                   const std::vector<float>& seed_z) override {
        for (size_t i = 0; i < vertex_mapping.size() && i < graph_.vertex_count(); i++) {
            if (vertex_mapping[i] < parent.positions_x.size()) {
                graph_.positions_x[i] = parent.positions_x[vertex_mapping[i]];
                graph_.positions_y[i] = parent.positions_y[vertex_mapping[i]];
                graph_.positions_z[i] = parent.positions_z[vertex_mapping[i]];
            } else {
                graph_.positions_x[i] = seed_x[i];
                graph_.positions_y[i] = seed_y[i];
                graph_.positions_z[i] = seed_z[i];
            }
        }
    }

private:
    LayoutGraph graph_;

    // Persistent buffers (allocated in upload_graph, reused in iterate)
    std::vector<float> velocities_x_;
    std::vector<float> velocities_y_;
    std::vector<float> velocities_z_;
    std::vector<float> force_x_;
    std::vector<float> force_y_;
    std::vector<float> force_z_;
    std::vector<uint32_t> edge_indices_;  // For stochastic edge sampling

    // Per-vertex stability tracking for oscillation prevention
    std::vector<int> settled_count_;      // Frames vertex has been at low velocity
    std::vector<float> prev_force_x_;     // Previous force for direction reversal detection
    std::vector<float> prev_force_y_;
    std::vector<int> oscillation_count_;  // Force direction reversals count

    uint32_t iteration_count_ = 0;
    QuadTree quadtree_;         // Barnes-Hut quadtree for O(n log n) repulsion
    QuadTree visible_quadtree_; // Quadtree for visible vertices only (ghost layout)
    QuadTree hidden_quadtree_;  // Quadtree for hidden vertices only (ghost layout)
    std::vector<bool> hidden_mask_;  // Inverted visibility mask (cached)
    std::mt19937 rng_{std::random_device{}()};  // RNG for edge sampling
};

// Factory function
std::unique_ptr<ILayoutEngine> create_layout_engine(LayoutBackend backend) {
    if (backend == LayoutBackend::Auto) {
        // Try backends in order of preference
#ifdef VIZ_HAS_CUDA
        if (is_cuda_available()) {
            std::cout << "Layout: Using CUDA backend" << std::endl;
            return create_cuda_layout_engine();
        }
#endif
        // TODO: Try Vulkan compute
        // TODO: Try WebGPU compute

        std::cout << "Layout: Using CPU backend (fallback)" << std::endl;
        return std::make_unique<CPULayoutEngine>();
    }

    switch (backend) {
#ifdef VIZ_HAS_CUDA
        case LayoutBackend::CUDA:
            if (is_cuda_available()) {
                return create_cuda_layout_engine();
            }
            std::cerr << "Layout: CUDA requested but not available" << std::endl;
            return nullptr;
#endif

        case LayoutBackend::CPU:
            return std::make_unique<CPULayoutEngine>();

        case LayoutBackend::VulkanCompute:
        case LayoutBackend::WebGPUCompute:
            std::cerr << "Layout: Compute shader backends not yet implemented" << std::endl;
            return nullptr;

        default:
            return nullptr;
    }
}

bool is_backend_available(LayoutBackend backend) {
    switch (backend) {
#ifdef VIZ_HAS_CUDA
        case LayoutBackend::CUDA:
            return is_cuda_available();
#endif
        case LayoutBackend::CPU:
            return true;

        case LayoutBackend::VulkanCompute:
        case LayoutBackend::WebGPUCompute:
            return false;  // Not yet implemented

        case LayoutBackend::Auto:
            return true;  // Always something available

        default:
            return false;
    }
}

const char* backend_name(LayoutBackend backend) {
    switch (backend) {
        case LayoutBackend::Auto: return "Auto";
        case LayoutBackend::CUDA: return "CUDA";
        case LayoutBackend::VulkanCompute: return "Vulkan Compute";
        case LayoutBackend::WebGPUCompute: return "WebGPU Compute";
        case LayoutBackend::CPU: return "CPU";
        default: return "Unknown";
    }
}

} // namespace viz::layout
