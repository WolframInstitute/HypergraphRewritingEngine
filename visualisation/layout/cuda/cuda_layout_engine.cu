// CUDA Layout Engine implementation
// Orchestrates Barnes-Hut kernels for GPU-accelerated graph layout

#include <layout/layout_engine.hpp>
#include "barnes_hut_kernels.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

namespace viz::layout {

class CUDALayoutEngine : public ILayoutEngine {
public:
    CUDALayoutEngine();
    ~CUDALayoutEngine() override;

    LayoutBackend get_backend() const override { return LayoutBackend::CUDA; }

    void upload_graph(const LayoutGraph& graph) override;
    void download_positions(LayoutGraph& graph) override;
    LayoutResult iterate(const LayoutParams& params) override;
    LayoutResult run_until_converged(const LayoutParams& params,
                                      LayoutProgressCallback progress) override;

    bool has_graph() const override { return nbodies_ > 0; }
    uint32_t vertex_count() const override { return nbodies_; }

    void seed_from(const LayoutGraph& parent,
                   const std::vector<uint32_t>& vertex_mapping,
                   const std::vector<float>& seed_x,
                   const std::vector<float>& seed_y,
                   const std::vector<float>& seed_z) override;

private:
    void allocate_buffers(int nbodies, int nedges);
    void free_buffers();
    void clear_forces();
    int compute_grid_blocks(int threads, int n);

    // Device buffers
    float* d_body_pos_x_ = nullptr;
    float* d_body_pos_y_ = nullptr;
    float* d_body_pos_z_ = nullptr;
    float* d_body_mass_ = nullptr;
    float* d_vel_x_ = nullptr;
    float* d_vel_y_ = nullptr;
    float* d_vel_z_ = nullptr;
    float* d_force_x_ = nullptr;
    float* d_force_y_ = nullptr;
    float* d_force_z_ = nullptr;
    bool* d_pinned_ = nullptr;
    float* d_displacement_ = nullptr;

    // Tree buffers
    float* d_node_pos_x_ = nullptr;
    float* d_node_pos_y_ = nullptr;
    float* d_node_pos_z_ = nullptr;
    float* d_node_mass_ = nullptr;
    int* d_children_ = nullptr;
    int* d_count_ = nullptr;
    int* d_start_ = nullptr;
    int* d_sorted_ = nullptr;

    // Edge buffers
    uint32_t* d_edge_src_ = nullptr;
    uint32_t* d_edge_dst_ = nullptr;
    float* d_edge_rest_len_ = nullptr;
    float* d_edge_strength_ = nullptr;

    // Host buffer for displacement reduction
    float* h_displacement_ = nullptr;

    int nbodies_ = 0;
    int nedges_ = 0;
    int nnodes_ = 0;  // Total tree nodes (bodies + internal)

    cudaStream_t stream_ = 0;
    uint32_t iteration_count_ = 0;
};

CUDALayoutEngine::CUDALayoutEngine() {
    cudaStreamCreate(&stream_);
}

CUDALayoutEngine::~CUDALayoutEngine() {
    free_buffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CUDALayoutEngine::allocate_buffers(int nbodies, int nedges) {
    if (nbodies_ == nbodies && nedges_ == nedges) {
        return;  // Already allocated
    }

    free_buffers();

    nbodies_ = nbodies;
    nedges_ = nedges;
    nnodes_ = cuda::compute_tree_size(nbodies);

    // Body buffers
    cudaMalloc(&d_body_pos_x_, nbodies * sizeof(float));
    cudaMalloc(&d_body_pos_y_, nbodies * sizeof(float));
    cudaMalloc(&d_body_pos_z_, nbodies * sizeof(float));
    cudaMalloc(&d_body_mass_, nbodies * sizeof(float));
    cudaMalloc(&d_vel_x_, nbodies * sizeof(float));
    cudaMalloc(&d_vel_y_, nbodies * sizeof(float));
    cudaMalloc(&d_vel_z_, nbodies * sizeof(float));
    cudaMalloc(&d_force_x_, nbodies * sizeof(float));
    cudaMalloc(&d_force_y_, nbodies * sizeof(float));
    cudaMalloc(&d_force_z_, nbodies * sizeof(float));
    cudaMalloc(&d_pinned_, nbodies * sizeof(bool));
    cudaMalloc(&d_displacement_, nbodies * sizeof(float));

    // Tree buffers (internal nodes)
    int node_count = nnodes_ + 1;  // +1 for root
    cudaMalloc(&d_node_pos_x_, node_count * sizeof(float));
    cudaMalloc(&d_node_pos_y_, node_count * sizeof(float));
    cudaMalloc(&d_node_pos_z_, node_count * sizeof(float));
    cudaMalloc(&d_node_mass_, node_count * sizeof(float));
    cudaMalloc(&d_children_, node_count * TREE_CHILDREN * sizeof(int));
    cudaMalloc(&d_count_, node_count * sizeof(int));
    cudaMalloc(&d_start_, node_count * sizeof(int));
    cudaMalloc(&d_sorted_, nbodies * sizeof(int));

    // Edge buffers
    if (nedges > 0) {
        cudaMalloc(&d_edge_src_, nedges * sizeof(uint32_t));
        cudaMalloc(&d_edge_dst_, nedges * sizeof(uint32_t));
        cudaMalloc(&d_edge_rest_len_, nedges * sizeof(float));
        cudaMalloc(&d_edge_strength_, nedges * sizeof(float));
    }

    // Host buffer for reduction
    h_displacement_ = new float[nbodies];

    // Initialize velocities and forces to zero
    cudaMemsetAsync(d_vel_x_, 0, nbodies * sizeof(float), stream_);
    cudaMemsetAsync(d_vel_y_, 0, nbodies * sizeof(float), stream_);
    cudaMemsetAsync(d_vel_z_, 0, nbodies * sizeof(float), stream_);
    cudaMemsetAsync(d_force_x_, 0, nbodies * sizeof(float), stream_);
    cudaMemsetAsync(d_force_y_, 0, nbodies * sizeof(float), stream_);
    cudaMemsetAsync(d_force_z_, 0, nbodies * sizeof(float), stream_);
}

void CUDALayoutEngine::free_buffers() {
    cudaFree(d_body_pos_x_);
    cudaFree(d_body_pos_y_);
    cudaFree(d_body_pos_z_);
    cudaFree(d_body_mass_);
    cudaFree(d_vel_x_);
    cudaFree(d_vel_y_);
    cudaFree(d_vel_z_);
    cudaFree(d_force_x_);
    cudaFree(d_force_y_);
    cudaFree(d_force_z_);
    cudaFree(d_pinned_);
    cudaFree(d_displacement_);

    cudaFree(d_node_pos_x_);
    cudaFree(d_node_pos_y_);
    cudaFree(d_node_pos_z_);
    cudaFree(d_node_mass_);
    cudaFree(d_children_);
    cudaFree(d_count_);
    cudaFree(d_start_);
    cudaFree(d_sorted_);

    cudaFree(d_edge_src_);
    cudaFree(d_edge_dst_);
    cudaFree(d_edge_rest_len_);
    cudaFree(d_edge_strength_);

    delete[] h_displacement_;

    d_body_pos_x_ = nullptr;
    nbodies_ = 0;
    nedges_ = 0;
}

void CUDALayoutEngine::clear_forces() {
    cudaMemsetAsync(d_force_x_, 0, nbodies_ * sizeof(float), stream_);
    cudaMemsetAsync(d_force_y_, 0, nbodies_ * sizeof(float), stream_);
    cudaMemsetAsync(d_force_z_, 0, nbodies_ * sizeof(float), stream_);
}

int CUDALayoutEngine::compute_grid_blocks(int threads, int n) {
    return (n + threads - 1) / threads;
}

void CUDALayoutEngine::upload_graph(const LayoutGraph& graph) {
    int nbodies = graph.vertex_count();
    int nedges = graph.edge_count();

    allocate_buffers(nbodies, nedges);

    // Upload body data
    cudaMemcpyAsync(d_body_pos_x_, graph.positions_x.data(), nbodies * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_body_pos_y_, graph.positions_y.data(), nbodies * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_body_pos_z_, graph.positions_z.data(), nbodies * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_body_mass_, graph.masses.data(), nbodies * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_pinned_, graph.pinned.data(), nbodies * sizeof(bool),
                    cudaMemcpyHostToDevice, stream_);

    // Upload edge data
    if (nedges > 0) {
        cudaMemcpyAsync(d_edge_src_, graph.edge_sources.data(), nedges * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_edge_dst_, graph.edge_targets.data(), nedges * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_edge_rest_len_, graph.edge_rest_lengths.data(), nedges * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_edge_strength_, graph.edge_strengths.data(), nedges * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }

    // Reset velocities
    cudaMemsetAsync(d_vel_x_, 0, nbodies * sizeof(float), stream_);
    cudaMemsetAsync(d_vel_y_, 0, nbodies * sizeof(float), stream_);
    cudaMemsetAsync(d_vel_z_, 0, nbodies * sizeof(float), stream_);

    iteration_count_ = 0;
    cudaStreamSynchronize(stream_);
}

void CUDALayoutEngine::download_positions(LayoutGraph& graph) {
    if (nbodies_ == 0) return;

    graph.positions_x.resize(nbodies_);
    graph.positions_y.resize(nbodies_);
    graph.positions_z.resize(nbodies_);

    cudaMemcpyAsync(graph.positions_x.data(), d_body_pos_x_, nbodies_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(graph.positions_y.data(), d_body_pos_y_, nbodies_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(graph.positions_z.data(), d_body_pos_z_, nbodies_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}

LayoutResult CUDALayoutEngine::iterate(const LayoutParams& params) {
    if (nbodies_ == 0) {
        return {true, 0.0f, 0.0f, 0, 0.0};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Reset tree state
    cuda::reset_device_variables(nnodes_, stream_);

    int blocks_bbox = compute_grid_blocks(THREADS_BBOX, nbodies_);
    int blocks_tree = compute_grid_blocks(THREADS_TREE, nbodies_);
    int blocks_summary = compute_grid_blocks(THREADS_SUMMARY, nnodes_);
    int blocks_sort = compute_grid_blocks(THREADS_SORT, nnodes_);
    int blocks_force = compute_grid_blocks(THREADS_FORCE, nbodies_);
    int blocks_integrate = compute_grid_blocks(THREADS_INTEGRATE, nbodies_);

    // 1. Compute bounding box
    cuda::BoundingBoxKernel<<<blocks_bbox, THREADS_BBOX, 0, stream_>>>(
        nbodies_,
        d_body_pos_x_, d_body_pos_y_, d_body_pos_z_,
        d_node_pos_x_, d_node_pos_y_, d_node_pos_z_
    );

    // 2. Clear tree structure
    cuda::ClearKernel1<<<blocks_tree, THREADS_TREE, 0, stream_>>>(
        nnodes_, nbodies_, d_children_
    );

    // 3. Build octree
    cuda::TreeBuildingKernel<<<blocks_tree, THREADS_TREE, 0, stream_>>>(
        nnodes_, nbodies_, d_children_,
        d_body_pos_x_, d_body_pos_y_, d_body_pos_z_,
        d_node_pos_x_, d_node_pos_y_, d_node_pos_z_
    );

    // 4. Clear for summarization
    cuda::ClearKernel2<<<blocks_summary, THREADS_SUMMARY, 0, stream_>>>(
        nnodes_, d_start_, d_node_mass_
    );

    // 5. Summarize tree (center of mass)
    cuda::SummarizationKernel<<<blocks_summary, THREADS_SUMMARY, 0, stream_>>>(
        nnodes_, nbodies_, d_count_, d_children_,
        d_body_mass_, d_node_mass_,
        d_body_pos_x_, d_body_pos_y_, d_body_pos_z_,
        d_node_pos_x_, d_node_pos_y_, d_node_pos_z_
    );

    // 6. Sort bodies
    cuda::SortKernel<<<blocks_sort, THREADS_SORT, 0, stream_>>>(
        nnodes_, nbodies_, d_sorted_, d_count_, d_start_, d_children_
    );

    // 7. Clear forces
    clear_forces();

    // 8. Compute repulsive forces (Barnes-Hut)
    float theta_sq = params.theta * params.theta;
    cuda::ForceCalculationKernel<<<blocks_force, THREADS_FORCE, 0, stream_>>>(
        nnodes_, nbodies_, theta_sq, params.repulsion_constant,
        d_sorted_, d_children_,
        d_body_mass_, d_node_mass_,
        d_body_pos_x_, d_body_pos_y_, d_body_pos_z_,
        d_node_pos_x_, d_node_pos_y_, d_node_pos_z_,
        d_force_x_, d_force_y_, d_force_z_
    );

    // 9. Compute attractive forces (springs)
    if (nedges_ > 0) {
        int blocks_edges = compute_grid_blocks(256, nedges_);
        cuda::AttractiveForceKernel<<<blocks_edges, 256, 0, stream_>>>(
            nedges_, params.spring_constant,
            d_edge_src_, d_edge_dst_, d_edge_rest_len_, d_edge_strength_,
            d_body_pos_x_, d_body_pos_y_, d_body_pos_z_,
            d_force_x_, d_force_y_, d_force_z_
        );
    }

    // 10. Optional gravity
    if (params.gravity > 0.0f) {
        int blocks_gravity = compute_grid_blocks(256, nbodies_);
        cuda::GravityKernel<<<blocks_gravity, 256, 0, stream_>>>(
            nbodies_, params.gravity,
            0.0f, 0.0f, 0.0f,  // Center at origin
            d_body_pos_x_, d_body_pos_y_, d_body_pos_z_,
            d_body_mass_,
            d_force_x_, d_force_y_, d_force_z_
        );
    }

    // 11. Integrate
    cuda::IntegrationKernel<<<blocks_integrate, THREADS_INTEGRATE, 0, stream_>>>(
        nbodies_, params.damping, params.max_displacement,
        d_pinned_,
        d_body_pos_x_, d_body_pos_y_, d_body_pos_z_,
        d_vel_x_, d_vel_y_, d_vel_z_,
        d_force_x_, d_force_y_, d_force_z_,
        d_displacement_
    );

    // Download displacement for convergence check
    cudaMemcpyAsync(h_displacement_, d_displacement_, nbodies_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Compute max and average displacement
    float max_disp = 0.0f;
    float sum_disp = 0.0f;
    for (int i = 0; i < nbodies_; i++) {
        max_disp = std::max(max_disp, h_displacement_[i]);
        sum_disp += h_displacement_[i];
    }
    float avg_disp = sum_disp / nbodies_;

    auto end_time = std::chrono::high_resolution_clock::now();
    double compute_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    iteration_count_++;

    LayoutResult result;
    result.converged = avg_disp < params.convergence_threshold;
    result.max_displacement = max_disp;
    result.average_displacement = avg_disp;
    result.iteration_count = iteration_count_;
    result.compute_time_ms = compute_time;

    return result;
}

LayoutResult CUDALayoutEngine::run_until_converged(const LayoutParams& params,
                                                    LayoutProgressCallback progress) {
    LayoutResult result;

    for (uint32_t i = 0; i < params.max_iterations; i++) {
        result = iterate(params);

        if (progress) {
            progress(result.iteration_count, result.average_displacement);
        }

        if (result.converged) {
            break;
        }
    }

    return result;
}

void CUDALayoutEngine::seed_from(const LayoutGraph& parent,
                                  const std::vector<uint32_t>& vertex_mapping,
                                  const std::vector<float>& seed_x,
                                  const std::vector<float>& seed_y,
                                  const std::vector<float>& seed_z) {
    // For new vertices, use seed positions
    // For mapped vertices, copy from parent

    std::vector<float> new_pos_x(seed_x.size());
    std::vector<float> new_pos_y(seed_y.size());
    std::vector<float> new_pos_z(seed_z.size());

    for (size_t i = 0; i < vertex_mapping.size(); i++) {
        if (vertex_mapping[i] < parent.positions_x.size()) {
            // Mapped from parent
            new_pos_x[i] = parent.positions_x[vertex_mapping[i]];
            new_pos_y[i] = parent.positions_y[vertex_mapping[i]];
            new_pos_z[i] = parent.positions_z[vertex_mapping[i]];
        } else {
            // New vertex, use seed
            new_pos_x[i] = seed_x[i];
            new_pos_y[i] = seed_y[i];
            new_pos_z[i] = seed_z[i];
        }
    }

    // Upload seeded positions
    cudaMemcpyAsync(d_body_pos_x_, new_pos_x.data(), new_pos_x.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_body_pos_y_, new_pos_y.data(), new_pos_y.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_body_pos_z_, new_pos_z.data(), new_pos_z.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
}

// Factory function implementation (CUDA version)
#ifdef VIZ_HAS_CUDA
std::unique_ptr<ILayoutEngine> create_cuda_layout_engine() {
    return std::make_unique<CUDALayoutEngine>();
}

bool is_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}
#endif

} // namespace viz::layout
