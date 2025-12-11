// Layout manager implementation
// Handles backend selection and provides factory function

#include <layout/layout_engine.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace viz::layout {

// Forward declarations for backend-specific factory functions
#ifdef VIZ_HAS_CUDA
std::unique_ptr<ILayoutEngine> create_cuda_layout_engine();
bool is_cuda_available();
#endif

// CPU fallback implementation (for debugging and WebGPU builds without compute)
class CPULayoutEngine : public ILayoutEngine {
public:
    LayoutBackend get_backend() const override { return LayoutBackend::CPU; }

    void upload_graph(const LayoutGraph& graph) override {
        graph_ = graph;
        // Initialize velocities
        velocities_x_.resize(graph_.vertex_count(), 0.0f);
        velocities_y_.resize(graph_.vertex_count(), 0.0f);
        velocities_z_.resize(graph_.vertex_count(), 0.0f);
        iteration_count_ = 0;
    }

    void download_positions(LayoutGraph& graph) override {
        graph.positions_x = graph_.positions_x;
        graph.positions_y = graph_.positions_y;
        graph.positions_z = graph_.positions_z;
    }

    LayoutResult iterate(const LayoutParams& params) override {
        // Simple O(n^2) implementation for debugging
        uint32_t n = graph_.vertex_count();
        if (n == 0) {
            return {true, 0.0f, 0.0f, 0, 0.0};
        }

        std::vector<float> force_x(n, 0.0f);
        std::vector<float> force_y(n, 0.0f);
        std::vector<float> force_z(n, 0.0f);

        // Repulsive forces (all pairs)
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

                force_x[i] += fx;
                force_y[i] += fy;
                force_z[i] += fz;
                force_x[j] -= fx;
                force_y[j] -= fy;
                force_z[j] -= fz;
            }
        }

        // Attractive forces (springs)
        for (uint32_t e = 0; e < graph_.edge_count(); e++) {
            uint32_t src = graph_.edge_sources[e];
            uint32_t dst = graph_.edge_targets[e];

            float dx = graph_.positions_x[dst] - graph_.positions_x[src];
            float dy = graph_.positions_y[dst] - graph_.positions_y[src];
            float dz = graph_.positions_z[dst] - graph_.positions_z[src];

            float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 0.0001f);
            float rest = graph_.edge_rest_lengths[e];
            if (rest <= 0.0f) rest = 1.0f;

            float f = params.spring_constant * graph_.edge_strengths[e] * (dist - rest) / dist;

            float fx = f * dx;
            float fy = f * dy;
            float fz = f * dz;

            force_x[src] += fx;
            force_y[src] += fy;
            force_z[src] += fz;
            force_x[dst] -= fx;
            force_y[dst] -= fy;
            force_z[dst] -= fz;
        }

        // Integration
        float max_disp = 0.0f;
        float sum_disp = 0.0f;

        for (uint32_t i = 0; i < n; i++) {
            if (graph_.pinned[i]) continue;

            velocities_x_[i] = (velocities_x_[i] + force_x[i]) * params.damping;
            velocities_y_[i] = (velocities_y_[i] + force_y[i]) * params.damping;
            velocities_z_[i] = (velocities_z_[i] + force_z[i]) * params.damping;

            float disp = std::sqrt(velocities_x_[i] * velocities_x_[i] +
                                   velocities_y_[i] * velocities_y_[i] +
                                   velocities_z_[i] * velocities_z_[i]);

            if (disp > params.max_displacement) {
                float scale = params.max_displacement / disp;
                velocities_x_[i] *= scale;
                velocities_y_[i] *= scale;
                velocities_z_[i] *= scale;
                disp = params.max_displacement;
            }

            graph_.positions_x[i] += velocities_x_[i];
            graph_.positions_y[i] += velocities_y_[i];
            graph_.positions_z[i] += velocities_z_[i];

            max_disp = std::max(max_disp, disp);
            sum_disp += disp;
        }

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
    std::vector<float> velocities_x_;
    std::vector<float> velocities_y_;
    std::vector<float> velocities_z_;
    uint32_t iteration_count_ = 0;
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
