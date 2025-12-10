#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "types.cuh"

namespace hypergraph::gpu {

// =============================================================================
// WL Hash Constants
// =============================================================================

static constexpr uint32_t WL_MAX_ITERATIONS = 100;
static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
static constexpr uint64_t FNV_PRIME = 1099511628211ULL;

// =============================================================================
// WL Hash Device Functions
// =============================================================================

// FNV-1a hash combine
__device__ __forceinline__ uint64_t fnv_hash_combine(uint64_t h, uint64_t value) {
    h ^= value;
    h *= FNV_PRIME;
    return h;
}

// Compute initial vertex color from structural signature
// Based on: (degree, positions in edges, edge arities)
__device__ uint64_t compute_initial_color(
    VertexId vertex,
    const DeviceAdjacency* adj,
    const uint64_t* state_bitmap  // Filter to edges in this state
);

// Single WL iteration: update vertex colors based on neighbors
__device__ uint64_t compute_next_color(
    VertexId vertex,
    const uint64_t* current_colors,
    const DeviceAdjacency* adj,
    const DeviceEdges* edges,
    const uint64_t* state_bitmap
);

// =============================================================================
// WL Hash Kernels
// =============================================================================

// Initialize vertex colors from structural signatures
__global__ void wl_init_colors_kernel(
    uint64_t* colors,               // [max_vertex] output
    DeviceAdjacency adj,            // BY VALUE
    const uint64_t* state_bitmap,   // Edges in this state
    uint32_t max_vertex
);

// Single WL iteration
__global__ void wl_iteration_kernel(
    const uint64_t* current_colors, // [max_vertex] input
    uint64_t* next_colors,          // [max_vertex] output
    DeviceAdjacency adj,            // BY VALUE
    DeviceEdges edges,              // BY VALUE
    const uint64_t* state_bitmap,
    uint32_t max_vertex,
    uint32_t* changed_flag          // Set to 1 if any color changed
);

// Compute final canonical hash from stable colors
__global__ void wl_finalize_hash_kernel(
    const uint64_t* colors,         // [max_vertex] vertex colors
    const uint64_t* state_bitmap,   // Edges in this state
    DeviceEdges edges,              // BY VALUE
    uint32_t num_edges,
    uint64_t* output_hash           // Single output hash
);

// =============================================================================
// Warp-Level WL Hash (for megakernel integration)
// =============================================================================

// Compute WL hash for a single state using one warp
// This version is used inside the megakernel for inline hashing
__device__ uint64_t compute_wl_hash_warp(
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    const DeviceAdjacency* adj,
    uint32_t max_vertex,
    uint64_t* scratch_colors        // [max_vertex * 2] for ping-pong
);

// =============================================================================
// Adjacency-Free WL Hash (builds adjacency on-the-fly from bitmap)
// =============================================================================
// This version doesn't require a pre-built adjacency index.
// It iterates over the state bitmap to find edges and builds vertex
// neighborhoods dynamically. This is slower but works for states with
// edges not in the original adjacency index (i.e., edges created during rewriting).

// Compute WL canonical hash from bitmap without pre-built adjacency
// Uses shared memory for vertex colors and neighborhood building
// max_edges_in_state: hint for max edges expected (for scratch sizing)
__device__ uint64_t compute_wl_hash_no_adj(
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    uint32_t num_total_edges,       // Total edges in system (for bitmap iteration)
    uint32_t max_vertex,            // Max vertex ID
    uint64_t* scratch               // Scratch space: [max_vertex * 2] for colors
);

// =============================================================================
// Vertex Hash Cache (for edge correspondence)
// =============================================================================
// Stores vertex hashes from WL refinement for use in edge correspondence

struct DeviceVertexHashCache {
    VertexId* vertices;     // Sorted vertex IDs
    uint64_t* hashes;       // Corresponding WL hashes
    uint32_t count;         // Number of vertices

    // Device-side lookup (binary search)
    __device__ uint64_t lookup(VertexId v) const {
        // Binary search for vertex
        int32_t lo = 0, hi = static_cast<int32_t>(count) - 1;
        while (lo <= hi) {
            int32_t mid = (lo + hi) / 2;
            if (vertices[mid] == v) {
                return hashes[mid];
            } else if (vertices[mid] < v) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return 0;  // Not found
    }
};

// =============================================================================
// Edge Correspondence (for canonical edge ordering)
// =============================================================================

struct DeviceEdgeCorrespondence {
    EdgeId* state1_edges;   // Edges in state 1
    EdgeId* state2_edges;   // Corresponding edges in state 2
    uint32_t count;         // Number of edge pairs
    bool valid;             // True if correspondence found
};

// =============================================================================
// Event Signature (for Full deduplication mode)
// =============================================================================

struct DeviceEventSignature {
    uint64_t input_state_hash;
    uint64_t output_state_hash;
    uint64_t consumed_edges_sig;
    uint64_t produced_edges_sig;

    __device__ uint64_t hash() const {
        uint64_t h = FNV_OFFSET_BASIS;
        h ^= input_state_hash;
        h *= FNV_PRIME;
        h ^= output_state_hash;
        h *= FNV_PRIME;
        h ^= consumed_edges_sig;
        h *= FNV_PRIME;
        h ^= produced_edges_sig;
        h *= FNV_PRIME;
        return h;
    }

    __device__ bool operator==(const DeviceEventSignature& other) const {
        return input_state_hash == other.input_state_hash &&
               output_state_hash == other.output_state_hash &&
               consumed_edges_sig == other.consumed_edges_sig &&
               produced_edges_sig == other.produced_edges_sig;
    }
};

// =============================================================================
// Edge Signature Computation (device-side)
// =============================================================================

// Compute edge signature from vertex hashes
__device__ inline uint64_t compute_edge_signature_from_cache(
    const DeviceEdges* edges,
    EdgeId eid,
    const DeviceVertexHashCache* cache
) {
    uint64_t sig = FNV_OFFSET_BASIS;
    uint8_t arity = edges->arities[eid];
    sig ^= arity;
    sig *= FNV_PRIME;

    for (uint8_t i = 0; i < arity; ++i) {
        VertexId v = edges->get_vertex(eid, i);
        uint64_t vh = cache->lookup(v);
        sig ^= vh;
        sig *= FNV_PRIME;
    }

    return sig;
}

// =============================================================================
// Host Interface
// =============================================================================

class WLHasher {
public:
    void init(uint32_t max_vertices);
    void destroy();

    // Compute canonical hash for a state
    // Takes structs BY VALUE (they contain device pointers internally)
    uint64_t compute_hash(
        const uint64_t* d_state_bitmap,
        DeviceEdges edges,        // BY VALUE
        DeviceAdjacency adj,      // BY VALUE
        uint32_t max_vertex,
        uint32_t num_edges,
        cudaStream_t stream = 0
    );

    // Compute hash and return vertex cache for edge correspondence
    // Allocates d_cache->vertices and d_cache->hashes on device
    uint64_t compute_hash_with_cache(
        const uint64_t* d_state_bitmap,
        DeviceEdges edges,
        DeviceAdjacency adj,
        uint32_t max_vertex,
        uint32_t num_edges,
        DeviceVertexHashCache* d_cache,  // Output: device-side cache
        cudaStream_t stream = 0
    );

    // Find edge correspondence between two isomorphic states
    // Uses cached vertex hashes for O(E) lookup instead of O(E^2)
    bool find_edge_correspondence(
        const uint64_t* d_state1_bitmap,
        const uint64_t* d_state2_bitmap,
        const DeviceVertexHashCache* d_cache1,
        const DeviceVertexHashCache* d_cache2,
        DeviceEdges edges,
        uint32_t num_edges,
        EdgeId* d_state1_edges,      // Output: edges in state 1
        EdgeId* d_state2_edges,      // Output: corresponding edges in state 2
        uint32_t* num_edge_pairs,    // Output: number of pairs
        cudaStream_t stream = 0
    );

    // Compute canonical hash WITHOUT pre-built adjacency index
    // This version builds vertex neighborhoods on-the-fly by scanning the bitmap
    // Slower but works for states with dynamically created edges
    uint64_t compute_hash_no_adj(
        const uint64_t* d_state_bitmap,
        DeviceEdges edges,        // BY VALUE
        uint32_t max_vertex,
        uint32_t num_edges,
        cudaStream_t stream = 0
    );

private:
    uint64_t* d_colors_a_;      // Ping buffer
    uint64_t* d_colors_b_;      // Pong buffer
    uint64_t* d_hash_output_;   // Single hash output
    uint32_t* d_changed_flag_;  // Iteration termination check
    uint32_t max_vertices_;
};

}  // namespace hypergraph::gpu
