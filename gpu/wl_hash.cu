#include "wl_hash.cuh"
#include "types.cuh"
#include <cub/cub.cuh>

namespace hypergraph::gpu {

// =============================================================================
// Device Helper Functions
// =============================================================================

// Check if edge is in state bitmap
__device__ __forceinline__ bool edge_in_state(EdgeId eid, const uint64_t* bitmap) {
    uint32_t word = eid / 64;
    uint32_t bit = eid % 64;
    return (bitmap[word] & (1ULL << bit)) != 0;
}

// Compute initial vertex color from structural signature
__device__ uint64_t compute_initial_color(
    VertexId vertex,
    const DeviceAdjacency* adj,
    const uint64_t* state_bitmap
) {
    uint64_t color = FNV_OFFSET_BASIS;

    // Count degree in this state
    uint32_t degree_in_state = 0;
    uint32_t start = adj->row_offsets[vertex];
    uint32_t end = adj->row_offsets[vertex + 1];

    for (uint32_t i = start; i < end; ++i) {
        EdgeId eid = adj->edge_ids[i];
        if (edge_in_state(eid, state_bitmap)) {
            ++degree_in_state;
            // Include position and arity in signature
            uint8_t pos = adj->positions[i];
            uint8_t arity = adj->edge_arities[i];
            color = fnv_hash_combine(color, (uint64_t(arity) << 8) | pos);
        }
    }

    // Mix in degree
    color = fnv_hash_combine(color, degree_in_state);

    return color;
}

// Compute next color by aggregating neighbor colors
__device__ uint64_t compute_next_color(
    VertexId vertex,
    const uint64_t* current_colors,
    const DeviceAdjacency* adj,
    const DeviceEdges* edges,
    const uint64_t* state_bitmap
) {
    // Start with current color
    uint64_t color = current_colors[vertex];

    // Collect neighbor colors for each edge this vertex participates in
    uint32_t start = adj->row_offsets[vertex];
    uint32_t end = adj->row_offsets[vertex + 1];

    // Local buffer for sorting neighbor colors (small, fits in registers)
    uint64_t neighbor_colors[MAX_EDGE_ARITY * 8];
    uint32_t num_neighbors = 0;

    for (uint32_t i = start; i < end; ++i) {
        EdgeId eid = adj->edge_ids[i];
        if (!edge_in_state(eid, state_bitmap)) continue;

        uint8_t my_pos = adj->positions[i];
        uint8_t arity = edges->arities[eid];

        // Get colors of all vertices in this edge (except self)
        for (uint8_t p = 0; p < arity; ++p) {
            if (p == my_pos) continue;
            VertexId neighbor = edges->get_vertex(eid, p);
            if (num_neighbors < sizeof(neighbor_colors)/sizeof(neighbor_colors[0])) {
                // Include position information in the neighbor color
                uint64_t nc = current_colors[neighbor];
                nc = fnv_hash_combine(nc, p);  // Position in edge
                nc = fnv_hash_combine(nc, arity);  // Edge arity
                neighbor_colors[num_neighbors++] = nc;
            }
        }
    }

    // Sort neighbor colors for canonical ordering
    // Simple insertion sort (small arrays)
    for (uint32_t i = 1; i < num_neighbors; ++i) {
        uint64_t key = neighbor_colors[i];
        int32_t j = i - 1;
        while (j >= 0 && neighbor_colors[j] > key) {
            neighbor_colors[j + 1] = neighbor_colors[j];
            --j;
        }
        neighbor_colors[j + 1] = key;
    }

    // Combine sorted neighbor colors into new color
    for (uint32_t i = 0; i < num_neighbors; ++i) {
        color = fnv_hash_combine(color, neighbor_colors[i]);
    }

    return color;
}

// =============================================================================
// Kernel Implementations
// =============================================================================

__global__ void wl_init_colors_kernel(
    uint64_t* colors,
    DeviceAdjacency adj,            // BY VALUE
    const uint64_t* state_bitmap,
    uint32_t max_vertex
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < max_vertex) {
        // Check if vertex participates in any edge in this state
        uint32_t start = adj.row_offsets[tid];
        uint32_t end = adj.row_offsets[tid + 1];

        bool in_state = false;
        for (uint32_t i = start; i < end && !in_state; ++i) {
            if (edge_in_state(adj.edge_ids[i], state_bitmap)) {
                in_state = true;
            }
        }

        if (in_state) {
            colors[tid] = compute_initial_color(tid, &adj, state_bitmap);
        } else {
            colors[tid] = 0;  // Not in state
        }
    }
}

__global__ void wl_iteration_kernel(
    const uint64_t* current_colors,
    uint64_t* next_colors,
    DeviceAdjacency adj,            // BY VALUE
    DeviceEdges edges,              // BY VALUE
    const uint64_t* state_bitmap,
    uint32_t max_vertex,
    uint32_t* changed_flag
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < max_vertex) {
        uint64_t old_color = current_colors[tid];

        if (old_color == 0) {
            // Vertex not in state
            next_colors[tid] = 0;
            return;
        }

        uint64_t new_color = compute_next_color(tid, current_colors, &adj, &edges, state_bitmap);
        next_colors[tid] = new_color;

        if (new_color != old_color) {
            atomicExch(changed_flag, 1);
        }
    }
}

__global__ void wl_finalize_hash_kernel(
    const uint64_t* colors,
    const uint64_t* state_bitmap,
    DeviceEdges edges,              // BY VALUE
    uint32_t num_edges,
    uint64_t* output_hash
) {
    // Collect and sort all unique vertex colors
    // Then hash them together

    // This kernel uses a single block for simplicity
    // For large states, could use parallel reduction

    __shared__ uint64_t unique_colors[1024];
    __shared__ uint32_t num_unique;

    if (threadIdx.x == 0) {
        num_unique = 0;
    }
    __syncthreads();

    // Each thread processes some edges
    for (uint32_t eid = threadIdx.x; eid < num_edges; eid += blockDim.x) {
        if (!edge_in_state(eid, state_bitmap)) continue;

        uint8_t arity = edges.arities[eid];
        for (uint8_t p = 0; p < arity; ++p) {
            VertexId v = edges.get_vertex(eid, p);
            uint64_t c = colors[v];

            // Add to unique colors (with dedup)
            // Simple linear scan - OK for small sets
            bool found = false;
            uint32_t n = num_unique;
            for (uint32_t i = 0; i < n && !found; ++i) {
                if (unique_colors[i] == c) found = true;
            }

            if (!found) {
                uint32_t idx = atomicAdd(&num_unique, 1);
                if (idx < 1024) {
                    unique_colors[idx] = c;
                }
            }
        }
    }
    __syncthreads();

    // Sort unique colors (single thread for simplicity)
    if (threadIdx.x == 0) {
        uint32_t n = num_unique;
        if (n > 1024) n = 1024;

        // Insertion sort
        for (uint32_t i = 1; i < n; ++i) {
            uint64_t key = unique_colors[i];
            int32_t j = i - 1;
            while (j >= 0 && unique_colors[j] > key) {
                unique_colors[j + 1] = unique_colors[j];
                --j;
            }
            unique_colors[j + 1] = key;
        }

        // Combine into final hash
        uint64_t hash = FNV_OFFSET_BASIS;
        for (uint32_t i = 0; i < n; ++i) {
            hash = fnv_hash_combine(hash, unique_colors[i]);
        }

        // Also include edge structure in hash
        // Count of vertices per color class
        hash = fnv_hash_combine(hash, n);

        *output_hash = hash;
    }
}

// =============================================================================
// Warp-Level WL Hash (for megakernel)
// =============================================================================

__device__ uint64_t compute_wl_hash_warp(
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    const DeviceAdjacency* adj,
    uint32_t max_vertex,
    uint64_t* scratch_colors
) {
    // Use cooperative groups for warp-level coordination
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;

    uint64_t* colors_a = scratch_colors;
    uint64_t* colors_b = scratch_colors + max_vertex;

    // Initialize colors (warp-parallel)
    for (uint32_t v = lane; v < max_vertex; v += 32) {
        uint32_t start = adj->row_offsets[v];
        uint32_t end = adj->row_offsets[v + 1];

        bool in_state = false;
        for (uint32_t i = start; i < end && !in_state; ++i) {
            if (edge_in_state(adj->edge_ids[i], state_bitmap)) {
                in_state = true;
            }
        }

        if (in_state) {
            colors_a[v] = compute_initial_color(v, adj, state_bitmap);
        } else {
            colors_a[v] = 0;
        }
    }
    __syncwarp();

    // Iterate until stable
    uint64_t* current = colors_a;
    uint64_t* next = colors_b;

    for (uint32_t iter = 0; iter < WL_MAX_ITERATIONS; ++iter) {
        uint32_t any_changed = 0;

        for (uint32_t v = lane; v < max_vertex; v += 32) {
            uint64_t old_color = current[v];
            if (old_color == 0) {
                next[v] = 0;
                continue;
            }

            uint64_t new_color = compute_next_color(v, current, adj, edges, state_bitmap);
            next[v] = new_color;

            if (new_color != old_color) {
                any_changed = 1;
            }
        }

        // Warp reduce to check if any changed
        any_changed = __reduce_or_sync(0xFFFFFFFF, any_changed);

        if (!any_changed) break;

        // Swap buffers
        uint64_t* tmp = current;
        current = next;
        next = tmp;

        __syncwarp();
    }

    // Finalize hash (lane 0 computes)
    uint64_t final_hash = 0;

    if (lane == 0) {
        // Collect unique colors
        uint64_t unique_colors[256];
        uint32_t num_unique = 0;

        for (uint32_t v = 0; v < max_vertex && num_unique < 256; ++v) {
            uint64_t c = current[v];
            if (c == 0) continue;

            bool found = false;
            for (uint32_t i = 0; i < num_unique && !found; ++i) {
                if (unique_colors[i] == c) found = true;
            }
            if (!found) {
                unique_colors[num_unique++] = c;
            }
        }

        // Sort
        for (uint32_t i = 1; i < num_unique; ++i) {
            uint64_t key = unique_colors[i];
            int32_t j = i - 1;
            while (j >= 0 && unique_colors[j] > key) {
                unique_colors[j + 1] = unique_colors[j];
                --j;
            }
            unique_colors[j + 1] = key;
        }

        // Hash
        final_hash = FNV_OFFSET_BASIS;
        for (uint32_t i = 0; i < num_unique; ++i) {
            final_hash = fnv_hash_combine(final_hash, unique_colors[i]);
        }
        final_hash = fnv_hash_combine(final_hash, num_unique);
    }

    // Broadcast from lane 0
    final_hash = __shfl_sync(0xFFFFFFFF, final_hash, 0);

    return final_hash;
}

// =============================================================================
// Adjacency-Free WL Hash Implementation
// =============================================================================
// This version builds vertex neighborhoods on-the-fly by scanning the bitmap
// and reading edge data directly. Slower than adjacency-based but works for
// dynamically created edges.

// Helper: compute initial color for a vertex by scanning all edges in state
// Matches CPU implementation: degree, then (arity, pos) pairs sorted
__device__ uint64_t compute_initial_color_no_adj(
    VertexId vertex,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    uint32_t num_total_edges
) {
    // Collect all (arity, pos) occurrences for this vertex
    uint8_t occ_arity[128];  // Max 128 occurrences
    uint8_t occ_pos[128];
    uint32_t num_occ = 0;

    // Scan all edges to find those containing this vertex
    for (uint32_t eid = 0; eid < num_total_edges && num_occ < 128; ++eid) {
        // Check if edge is in state
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        if ((state_bitmap[word] & (1ULL << bit)) == 0) continue;

        uint8_t arity = edges->arities[eid];

        // Check if vertex is in this edge (may appear multiple times)
        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = edges->get_vertex(eid, i);
            if (v == vertex && num_occ < 128) {
                occ_arity[num_occ] = arity;
                occ_pos[num_occ] = i;
                ++num_occ;
            }
        }
    }

    // Sort occurrences by (arity, pos) for canonical ordering
    for (uint32_t i = 1; i < num_occ; ++i) {
        uint8_t key_arity = occ_arity[i];
        uint8_t key_pos = occ_pos[i];
        int32_t j = i - 1;
        while (j >= 0 && (occ_arity[j] > key_arity ||
                        (occ_arity[j] == key_arity && occ_pos[j] > key_pos))) {
            occ_arity[j + 1] = occ_arity[j];
            occ_pos[j + 1] = occ_pos[j];
            --j;
        }
        occ_arity[j + 1] = key_arity;
        occ_pos[j + 1] = key_pos;
    }

    // Build hash: degree, then sorted (arity, pos) pairs
    uint64_t color = FNV_OFFSET_BASIS;
    color = fnv_hash_combine(color, num_occ);  // degree
    for (uint32_t i = 0; i < num_occ; ++i) {
        color = fnv_hash_combine(color, occ_arity[i]);  // arity
        color = fnv_hash_combine(color, occ_pos[i]);    // pos
    }

    return color;
}

// Helper: compute next color by aggregating neighbor colors (no adjacency)
// Matches CPU: neighbor color combined with neighbor's position k
__device__ uint64_t compute_next_color_no_adj(
    VertexId vertex,
    const uint64_t* current_colors,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    uint32_t num_total_edges
) {
    // Start with current color
    uint64_t color = current_colors[vertex];

    // Local buffer for sorting neighbor hashes
    uint64_t neighbor_hashes[128];
    uint32_t num_neighbors = 0;

    // Scan all edges to find neighbors
    for (uint32_t eid = 0; eid < num_total_edges && num_neighbors < 128; ++eid) {
        // Check if edge is in state
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        if ((state_bitmap[word] & (1ULL << bit)) == 0) continue;

        uint8_t arity = edges->arities[eid];

        // Find our position(s) in this edge
        for (uint8_t my_pos = 0; my_pos < arity; ++my_pos) {
            if (edges->get_vertex(eid, my_pos) != vertex) continue;

            // For each position where we appear, get neighbor colors
            for (uint8_t k = 0; k < arity && num_neighbors < 128; ++k) {
                if (k == my_pos) continue;
                VertexId neighbor = edges->get_vertex(eid, k);
                // CPU: fnv_combine(current[neighbor_idx], k)
                neighbor_hashes[num_neighbors++] = fnv_hash_combine(current_colors[neighbor], k);
            }
        }
    }

    // Sort neighbor hashes for canonical ordering (insertion sort)
    for (uint32_t i = 1; i < num_neighbors; ++i) {
        uint64_t key = neighbor_hashes[i];
        int32_t j = i - 1;
        while (j >= 0 && neighbor_hashes[j] > key) {
            neighbor_hashes[j + 1] = neighbor_hashes[j];
            --j;
        }
        neighbor_hashes[j + 1] = key;
    }

    // Combine sorted neighbor hashes into new color
    for (uint32_t i = 0; i < num_neighbors; ++i) {
        color = fnv_hash_combine(color, neighbor_hashes[i]);
    }

    return color;
}

// Main function: compute WL hash without pre-built adjacency
// Single-threaded version for simplicity (called by lane 0)
__device__ uint64_t compute_wl_hash_no_adj(
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    uint32_t num_total_edges,
    uint32_t max_vertex,
    uint64_t* scratch
) {
    const uint32_t lane = threadIdx.x % 32;

    __syncwarp();

    uint64_t* colors_a = scratch;
    uint64_t* colors_b = scratch + max_vertex;

    // Phase 1: Find all vertices in the state and initialize colors
    // Warp-parallel vertex processing
    for (uint32_t v = lane; v < max_vertex; v += 32) {
        // Check if vertex is in any edge of this state
        bool in_state = false;
        for (uint32_t eid = 0; eid < num_total_edges && !in_state; ++eid) {
            uint32_t word = eid / 64;
            uint32_t bit = eid % 64;
            if ((state_bitmap[word] & (1ULL << bit)) == 0) continue;

            uint8_t arity = edges->arities[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                if (edges->get_vertex(eid, i) == v) {
                    in_state = true;
                    break;
                }
            }
        }

        if (in_state) {
            colors_a[v] = compute_initial_color_no_adj(v, state_bitmap, edges, num_total_edges);
        } else {
            colors_a[v] = 0;
        }
    }
    __syncwarp();

    // Phase 2: WL refinement iterations
    uint64_t* current = colors_a;
    uint64_t* next = colors_b;

    for (uint32_t iter = 0; iter < WL_MAX_ITERATIONS; ++iter) {
        uint32_t any_changed = 0;

        for (uint32_t v = lane; v < max_vertex; v += 32) {
            uint64_t old_color = current[v];
            if (old_color == 0) {
                next[v] = 0;
                continue;
            }

            uint64_t new_color = compute_next_color_no_adj(v, current, state_bitmap, edges, num_total_edges);
            next[v] = new_color;

            if (new_color != old_color) {
                any_changed = 1;
            }
        }

        // Warp reduce to check if any changed
        any_changed = __reduce_or_sync(0xFFFFFFFF, any_changed);

        if (!any_changed) break;

        // Swap buffers
        uint64_t* tmp = current;
        current = next;
        next = tmp;

        __syncwarp();
    }

    // Phase 3: Build canonical hash from edge structure (lane 0 only)
    // Must match CPU's build_canonical_hash_dense exactly
    uint64_t final_hash = 0;

    if (lane == 0) {
        // Collect all vertex colors (including duplicates for class counting)
        // Store as (color, vertex_index) pairs for sorting
        uint64_t vertex_colors[256];
        uint32_t vertex_indices[256];
        uint32_t num_vertices_in_state = 0;

        for (uint32_t v = 0; v < max_vertex && num_vertices_in_state < 256; ++v) {
            uint64_t c = current[v];
            if (c == 0) continue;  // vertex not in state

            vertex_colors[num_vertices_in_state] = c;
            vertex_indices[num_vertices_in_state] = v;
            ++num_vertices_in_state;
        }

        // Sort by color for canonical ordering (matches CPU's sorted_hashes)
        for (uint32_t i = 1; i < num_vertices_in_state; ++i) {
            uint64_t key_color = vertex_colors[i];
            uint32_t key_idx = vertex_indices[i];
            int32_t j = i - 1;
            while (j >= 0 && vertex_colors[j] > key_color) {
                vertex_colors[j + 1] = vertex_colors[j];
                vertex_indices[j + 1] = vertex_indices[j];
                --j;
            }
            vertex_colors[j + 1] = key_color;
            vertex_indices[j + 1] = key_idx;
        }

        // Start building final hash
        final_hash = FNV_OFFSET_BASIS;

        // Hash vertex equivalence class structure (matches CPU exactly)
        // CPU: for each group of vertices with same hash, emit (hash, class_count)
        uint64_t prev_hash = 0;
        uint32_t class_count = 0;
        for (uint32_t i = 0; i < num_vertices_in_state; ++i) {
            uint64_t vh = vertex_colors[i];
            if (vh != prev_hash && class_count > 0) {
                final_hash = fnv_hash_combine(final_hash, prev_hash);
                final_hash = fnv_hash_combine(final_hash, class_count);
                class_count = 0;
            }
            prev_hash = vh;
            ++class_count;
        }
        if (class_count > 0) {
            final_hash = fnv_hash_combine(final_hash, prev_hash);
            final_hash = fnv_hash_combine(final_hash, class_count);
        }

        // Build edge hashes using vertex colors
        // CPU: edge_hash = FNV_OFFSET → arity → vertex_hashes[0] → vertex_hashes[1]...
        uint64_t edge_hashes[256];
        uint32_t num_edge_hashes = 0;

        for (uint32_t eid = 0; eid < num_total_edges && num_edge_hashes < 256; ++eid) {
            uint32_t word = eid / 64;
            uint32_t bit = eid % 64;
            if ((state_bitmap[word] & (1ULL << bit)) == 0) continue;

            uint8_t arity = edges->arities[eid];

            // Build edge hash from vertex colors (preserving order for directed edges)
            uint64_t edge_hash = FNV_OFFSET_BASIS;
            edge_hash = fnv_hash_combine(edge_hash, arity);
            for (uint8_t i = 0; i < arity; ++i) {
                VertexId v = edges->get_vertex(eid, i);
                edge_hash = fnv_hash_combine(edge_hash, current[v]);
            }
            edge_hashes[num_edge_hashes++] = edge_hash;
        }

        // Sort edge hashes for canonical ordering
        for (uint32_t i = 1; i < num_edge_hashes; ++i) {
            uint64_t key = edge_hashes[i];
            int32_t j = i - 1;
            while (j >= 0 && edge_hashes[j] > key) {
                edge_hashes[j + 1] = edge_hashes[j];
                --j;
            }
            edge_hashes[j + 1] = key;
        }

        // Hash edge structure (matches CPU: num_edges then sorted edge hashes)
        final_hash = fnv_hash_combine(final_hash, num_edge_hashes);
        for (uint32_t i = 0; i < num_edge_hashes; ++i) {
            final_hash = fnv_hash_combine(final_hash, edge_hashes[i]);
        }
    }

    // Broadcast from lane 0
    final_hash = __shfl_sync(0xFFFFFFFF, final_hash, 0);

    return final_hash;
}

// =============================================================================
// Host Interface Implementation
// =============================================================================

void WLHasher::init(uint32_t max_vertices) {
    max_vertices_ = max_vertices;
    CUDA_CHECK(cudaMalloc(&d_colors_a_, max_vertices * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_colors_b_, max_vertices * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_hash_output_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_changed_flag_, sizeof(uint32_t)));
}

void WLHasher::destroy() {
    if (d_colors_a_) cudaFree(d_colors_a_);
    if (d_colors_b_) cudaFree(d_colors_b_);
    if (d_hash_output_) cudaFree(d_hash_output_);
    if (d_changed_flag_) cudaFree(d_changed_flag_);
    d_colors_a_ = d_colors_b_ = nullptr;
    d_hash_output_ = nullptr;
    d_changed_flag_ = nullptr;
}

uint64_t WLHasher::compute_hash(
    const uint64_t* d_state_bitmap,
    DeviceEdges edges,            // BY VALUE
    DeviceAdjacency adj,          // BY VALUE
    uint32_t max_vertex,
    uint32_t num_edges,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (max_vertex + block_size - 1) / block_size;

    // Initialize colors
    wl_init_colors_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_colors_a_, adj, d_state_bitmap, max_vertex
    );

    // Iterate until stable
    uint64_t* current = d_colors_a_;
    uint64_t* next = d_colors_b_;

    for (uint32_t iter = 0; iter < WL_MAX_ITERATIONS; ++iter) {
        CUDA_CHECK(cudaMemsetAsync(d_changed_flag_, 0, sizeof(uint32_t), stream));

        wl_iteration_kernel<<<num_blocks, block_size, 0, stream>>>(
            current, next, adj, edges, d_state_bitmap, max_vertex, d_changed_flag_
        );

        // Check if converged
        uint32_t changed;
        CUDA_CHECK(cudaMemcpyAsync(&changed, d_changed_flag_, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!changed) break;

        // Swap buffers
        uint64_t* tmp = current;
        current = next;
        next = tmp;
    }

    // Finalize hash
    wl_finalize_hash_kernel<<<1, 256, 0, stream>>>(
        current, d_state_bitmap, edges, num_edges, d_hash_output_
    );

    // Copy result
    uint64_t hash;
    CUDA_CHECK(cudaMemcpyAsync(&hash, d_hash_output_, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return hash;
}

// =============================================================================
// Kernel: Build Vertex Hash Cache from WL Colors
// =============================================================================

__global__ void wl_build_cache_kernel(
    const uint64_t* colors,
    const uint64_t* state_bitmap,
    DeviceAdjacency adj,
    uint32_t max_vertex,
    VertexId* cache_vertices,
    uint64_t* cache_hashes,
    uint32_t* cache_count
) {
    // Single-block kernel to collect vertices in state and their hashes
    __shared__ uint32_t count;

    if (threadIdx.x == 0) {
        count = 0;
    }
    __syncthreads();

    // Each thread checks a range of vertices
    for (uint32_t v = threadIdx.x; v < max_vertex; v += blockDim.x) {
        // Check if vertex is in state (has at least one edge in state)
        uint32_t start = adj.row_offsets[v];
        uint32_t end = adj.row_offsets[v + 1];

        bool in_state = false;
        for (uint32_t i = start; i < end && !in_state; ++i) {
            EdgeId eid = adj.edge_ids[i];
            uint32_t word = eid / 64;
            uint32_t bit = eid % 64;
            if (state_bitmap[word] & (1ULL << bit)) {
                in_state = true;
            }
        }

        if (in_state && colors[v] != 0) {
            uint32_t idx = atomicAdd(&count, 1);
            cache_vertices[idx] = v;
            cache_hashes[idx] = colors[v];
        }
    }
    __syncthreads();

    // Write final count
    if (threadIdx.x == 0) {
        *cache_count = count;
    }
}

// Simple bubble sort for small arrays on GPU (cache is typically small)
__global__ void wl_sort_cache_kernel(
    VertexId* vertices,
    uint64_t* hashes,
    uint32_t count
) {
    if (threadIdx.x != 0) return;

    // Bubble sort - fine for small arrays (<1000 vertices typically)
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = i + 1; j < count; ++j) {
            if (vertices[j] < vertices[i]) {
                // Swap
                VertexId tv = vertices[i];
                vertices[i] = vertices[j];
                vertices[j] = tv;

                uint64_t th = hashes[i];
                hashes[i] = hashes[j];
                hashes[j] = th;
            }
        }
    }
}

uint64_t WLHasher::compute_hash_with_cache(
    const uint64_t* d_state_bitmap,
    DeviceEdges edges,
    DeviceAdjacency adj,
    uint32_t max_vertex,
    uint32_t num_edges,
    DeviceVertexHashCache* d_cache,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (max_vertex + block_size - 1) / block_size;

    // Initialize colors
    wl_init_colors_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_colors_a_, adj, d_state_bitmap, max_vertex
    );

    // Iterate until stable
    uint64_t* current = d_colors_a_;
    uint64_t* next = d_colors_b_;

    for (uint32_t iter = 0; iter < WL_MAX_ITERATIONS; ++iter) {
        CUDA_CHECK(cudaMemsetAsync(d_changed_flag_, 0, sizeof(uint32_t), stream));

        wl_iteration_kernel<<<num_blocks, block_size, 0, stream>>>(
            current, next, adj, edges, d_state_bitmap, max_vertex, d_changed_flag_
        );

        // Check if converged
        uint32_t changed;
        CUDA_CHECK(cudaMemcpyAsync(&changed, d_changed_flag_, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!changed) break;

        // Swap buffers
        uint64_t* tmp = current;
        current = next;
        next = tmp;
    }

    // Allocate cache arrays on device (will be filled by kernel)
    VertexId* cache_vertices;
    uint64_t* cache_hashes;
    uint32_t* cache_count;

    CUDA_CHECK(cudaMalloc(&cache_vertices, max_vertex * sizeof(VertexId)));
    CUDA_CHECK(cudaMalloc(&cache_hashes, max_vertex * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&cache_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(cache_count, 0, sizeof(uint32_t), stream));

    // Build cache from final colors
    wl_build_cache_kernel<<<1, 256, 0, stream>>>(
        current, d_state_bitmap, adj, max_vertex,
        cache_vertices, cache_hashes, cache_count
    );

    // Get count back
    uint32_t h_count;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, cache_count, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Sort cache by vertex ID for binary search
    if (h_count > 1) {
        wl_sort_cache_kernel<<<1, 1, 0, stream>>>(cache_vertices, cache_hashes, h_count);
    }

    // Set up output cache struct
    d_cache->vertices = cache_vertices;
    d_cache->hashes = cache_hashes;
    d_cache->count = h_count;

    // Free temp count
    cudaFree(cache_count);

    // Finalize hash
    wl_finalize_hash_kernel<<<1, 256, 0, stream>>>(
        current, d_state_bitmap, edges, num_edges, d_hash_output_
    );

    // Copy result
    uint64_t hash;
    CUDA_CHECK(cudaMemcpyAsync(&hash, d_hash_output_, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return hash;
}

// =============================================================================
// Kernel: Find Edge Correspondence
// =============================================================================

__global__ void wl_edge_correspondence_kernel(
    const uint64_t* state1_bitmap,
    const uint64_t* state2_bitmap,
    DeviceVertexHashCache cache1,
    DeviceVertexHashCache cache2,
    DeviceEdges edges,
    uint32_t num_edges,
    EdgeId* state1_edges,
    EdgeId* state2_edges,
    uint32_t* num_pairs,
    uint32_t* valid_flag
) {
    // First, collect edges from state1 and compute signatures
    // Then find matching edges in state2

    __shared__ uint32_t edge1_count;
    __shared__ uint32_t match_count;

    if (threadIdx.x == 0) {
        edge1_count = 0;
        match_count = 0;
        *valid_flag = 1;  // Assume valid until proven otherwise
    }
    __syncthreads();

    // Build signature -> edge map for state2 (using shared memory for small states)
    // For larger states, this would need a more sophisticated approach
    __shared__ uint64_t state2_sigs[512];
    __shared__ EdgeId state2_eids[512];
    __shared__ uint32_t state2_used[512];  // Use uint32_t for atomicCAS compatibility
    __shared__ uint32_t state2_count;

    if (threadIdx.x == 0) {
        state2_count = 0;
    }
    __syncthreads();

    // Collect state2 edges and their signatures
    for (uint32_t eid = threadIdx.x; eid < num_edges; eid += blockDim.x) {
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        if (state2_bitmap[word] & (1ULL << bit)) {
            uint32_t idx = atomicAdd(&state2_count, 1);
            if (idx < 512) {
                state2_eids[idx] = eid;
                state2_sigs[idx] = compute_edge_signature_from_cache(&edges, eid, &cache2);
                state2_used[idx] = 0;
            }
        }
    }
    __syncthreads();

    // For each edge in state1, find matching edge in state2
    for (uint32_t eid = threadIdx.x; eid < num_edges; eid += blockDim.x) {
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        if (state1_bitmap[word] & (1ULL << bit)) {
            uint64_t sig1 = compute_edge_signature_from_cache(&edges, eid, &cache1);

            // Find matching edge in state2 (linear search, could optimize with hash table)
            bool found = false;
            for (uint32_t i = 0; i < state2_count && i < 512 && !found; ++i) {
                if (state2_sigs[i] == sig1 && atomicCAS(&state2_used[i], 0u, 1u) == 0u) {
                    // Found matching edge
                    uint32_t pair_idx = atomicAdd(&match_count, 1);
                    state1_edges[pair_idx] = eid;
                    state2_edges[pair_idx] = state2_eids[i];
                    found = true;
                }
            }

            if (!found) {
                // No matching edge found - states not isomorphic
                atomicExch(valid_flag, 0);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *num_pairs = match_count;
    }
}

bool WLHasher::find_edge_correspondence(
    const uint64_t* d_state1_bitmap,
    const uint64_t* d_state2_bitmap,
    const DeviceVertexHashCache* d_cache1,
    const DeviceVertexHashCache* d_cache2,
    DeviceEdges edges,
    uint32_t num_edges,
    EdgeId* d_state1_edges,
    EdgeId* d_state2_edges,
    uint32_t* num_edge_pairs,
    cudaStream_t stream
) {
    uint32_t* d_valid_flag;
    uint32_t* d_num_pairs;

    CUDA_CHECK(cudaMalloc(&d_valid_flag, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_num_pairs, sizeof(uint32_t)));

    // Copy caches to local for kernel (they contain device pointers)
    DeviceVertexHashCache h_cache1, h_cache2;
    CUDA_CHECK(cudaMemcpy(&h_cache1, d_cache1, sizeof(DeviceVertexHashCache), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_cache2, d_cache2, sizeof(DeviceVertexHashCache), cudaMemcpyDeviceToHost));

    wl_edge_correspondence_kernel<<<1, 256, 0, stream>>>(
        d_state1_bitmap, d_state2_bitmap,
        h_cache1, h_cache2,
        edges, num_edges,
        d_state1_edges, d_state2_edges,
        d_num_pairs, d_valid_flag
    );

    uint32_t h_valid, h_num_pairs;
    CUDA_CHECK(cudaMemcpyAsync(&h_valid, d_valid_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_num_pairs, d_num_pairs, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    *num_edge_pairs = h_num_pairs;

    cudaFree(d_valid_flag);
    cudaFree(d_num_pairs);

    return h_valid != 0;
}

// =============================================================================
// Kernel: WL Hash Without Pre-built Adjacency
// =============================================================================
// Single warp computes the hash by building neighborhoods on-the-fly

__global__ void wl_hash_no_adj_kernel(
    const uint64_t* state_bitmap,
    DeviceEdges edges,              // BY VALUE
    uint32_t num_edges,
    uint32_t max_vertex,
    uint64_t* scratch,              // [max_vertex * 2]
    uint64_t* output_hash
) {
    // Only one warp does the work
    if (threadIdx.x >= 32) return;

    uint64_t hash = compute_wl_hash_no_adj(
        state_bitmap,
        &edges,
        num_edges,
        max_vertex,
        scratch
    );

    if (threadIdx.x == 0) {
        *output_hash = hash;
    }
}

uint64_t WLHasher::compute_hash_no_adj(
    const uint64_t* d_state_bitmap,
    DeviceEdges edges,
    uint32_t max_vertex,
    uint32_t num_edges,
    cudaStream_t stream
) {
    // Launch single warp kernel
    wl_hash_no_adj_kernel<<<1, 32, 0, stream>>>(
        d_state_bitmap,
        edges,          // BY VALUE
        num_edges,
        max_vertex,
        d_colors_a_,    // Use existing buffers as scratch
        d_hash_output_
    );

    uint64_t hash;
    CUDA_CHECK(cudaMemcpyAsync(&hash, d_hash_output_, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return hash;
}

}  // namespace hypergraph::gpu
