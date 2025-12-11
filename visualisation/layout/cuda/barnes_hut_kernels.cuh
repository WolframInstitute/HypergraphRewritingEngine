// CUDA Barnes-Hut kernels for 3D graph layout
// Adapted from GPUGraphLayout (Martin Burtscher, Texas State University)
// Extended to 3D (octree) with optional 2D mode (quadtree)

#pragma once

#include <layout/layout_constants.h>
#include <cuda_runtime.h>

namespace viz::layout::cuda {

// Device variables (global state)
extern __device__ int bottomd;          // Bottom of tree (decremented during build)
extern __device__ int maxdepthd;        // Maximum tree depth reached
extern __device__ float radiusd;        // Root cell radius
extern __device__ int errd;             // Error flag

// Bounding box results (written by BoundingBoxKernel)
extern __device__ float minxd, minyd, minzd;
extern __device__ float maxxd, maxyd, maxzd;

// Kernel declarations with launch bounds

// Compute bounding box of all bodies
// Output: minx/y/z, maxx/y/z, radius
__global__ __launch_bounds__(THREADS_BBOX, FACTOR_BBOX)
void BoundingBoxKernel(
    int nbodies,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,  // Root position written to last node
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z
);

// Clear tree structure (child pointers)
__global__ __launch_bounds__(THREADS_TREE, FACTOR_TREE)
void ClearKernel1(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ children
);

// Build octree by inserting bodies
__global__ __launch_bounds__(THREADS_TREE, FACTOR_TREE)
void TreeBuildingKernel(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ children,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z
);

// Clear mass and count for summarization
__global__ __launch_bounds__(THREADS_SUMMARY, FACTOR_SUMMARY)
void ClearKernel2(
    int nnodes,
    volatile int* __restrict__ start,
    volatile float* __restrict__ node_mass
);

// Compute center of mass for each cell (bottom-up)
__global__ __launch_bounds__(THREADS_SUMMARY, FACTOR_SUMMARY)
void SummarizationKernel(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ count,
    volatile int* __restrict__ children,
    volatile float* __restrict__ body_mass,
    volatile float* __restrict__ node_mass,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z
);

// Sort bodies into depth-first order for cache efficiency
__global__ __launch_bounds__(THREADS_SORT, FACTOR_SORT)
void SortKernel(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ sorted,
    volatile int* __restrict__ count,
    volatile int* __restrict__ start,
    volatile int* __restrict__ children
);

// Compute repulsive forces using Barnes-Hut approximation
__global__ __launch_bounds__(THREADS_FORCE, FACTOR_FORCE)
void ForceCalculationKernel(
    int nnodes,
    int nbodies,
    float theta_sq,             // BH_THETA^2
    float repulsion_k,          // Repulsion constant
    volatile int* __restrict__ sorted,
    volatile int* __restrict__ children,
    volatile float* __restrict__ body_mass,
    volatile float* __restrict__ node_mass,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z
);

// Compute attractive forces from springs (edges)
__global__
void AttractiveForceKernel(
    int nedges,
    float spring_k,             // Spring constant
    const uint32_t* __restrict__ edge_src,
    const uint32_t* __restrict__ edge_dst,
    const float* __restrict__ edge_rest_len,
    const float* __restrict__ edge_strength,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z
);

// Optional gravity toward center
__global__
void GravityKernel(
    int nbodies,
    float gravity_k,
    float center_x,
    float center_y,
    float center_z,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ body_mass,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z
);

// Integrate forces to update positions (Verlet-style with damping)
// Returns max displacement for convergence check
__global__ __launch_bounds__(THREADS_INTEGRATE, FACTOR_INTEGRATE)
void IntegrationKernel(
    int nbodies,
    float damping,
    float max_displacement,
    const bool* __restrict__ pinned,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ vel_x,
    volatile float* __restrict__ vel_y,
    volatile float* __restrict__ vel_z,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z,
    volatile float* __restrict__ displacement_out  // Per-body displacement for reduction
);

// Host-side helper to compute required tree size
inline int compute_tree_size(int nbodies) {
    // Heuristic: need at most 2*nbodies internal nodes for balanced tree
    // Add extra for worst-case unbalanced insertions
    return nbodies * 2 + 12000;
}

// Host-side helper to reset device variables before iteration
void reset_device_variables(int nnodes, cudaStream_t stream = 0);

// Host-side helper to check for errors after kernels
bool check_kernel_errors(cudaStream_t stream = 0);

} // namespace viz::layout::cuda
