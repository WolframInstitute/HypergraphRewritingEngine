#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "types.cuh"
#include "signature_index.cuh"

namespace hypergraph::gpu {

// =============================================================================
// Pattern Matching Declarations
// =============================================================================

// Count matches in a state (for allocation sizing)
// NOTE: DeviceEdges, StatePool, DeviceAdjacency passed BY VALUE to copy to device
__global__ void count_matches_kernel(
    const StateId* states,
    uint32_t num_states,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    DeviceEdges edges,              // BY VALUE - contains device pointers
    StatePool state_pool,           // BY VALUE
    DeviceAdjacency adj,            // BY VALUE
    uint32_t* match_counts          // [num_states] output
);

// Collect matches into compacted array
__global__ void collect_matches_kernel(
    const StateId* states,
    uint32_t num_states,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    DeviceEdges edges,
    StatePool state_pool,
    DeviceAdjacency adj,
    const uint32_t* output_offsets,  // From prefix sum of match_counts
    DeviceMatch* matches             // Output array
);

// =============================================================================
// Warp-Level Pattern Matching (for megakernel)
// =============================================================================

// Find all matches in a single state using one warp
// Returns number of matches found, writes to output buffer
__device__ uint32_t find_matches_warp(
    StateId state,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    const DeviceEdges* edges,       // Pointer (already on device in megakernel context)
    const StatePool* state_pool,    // Pointer
    const DeviceAdjacency* adj,     // Pointer
    DeviceMatch* output,            // Buffer for matches
    uint32_t max_matches            // Capacity of output buffer
);

// Delta matching: find matches involving at least one produced edge
// Used for match forwarding - only matches NEW patterns, not inherited ones
__device__ uint32_t find_delta_matches_warp(
    StateId state,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    const DeviceEdges* edges,
    const StatePool* state_pool,
    const DeviceAdjacency* adj,
    const EdgeId* produced_edges,   // Edges created by parent rewrite
    uint8_t num_produced,           // Number of produced edges
    DeviceMatch* output,
    uint32_t max_matches
);

// =============================================================================
// Device Helper Functions
// =============================================================================

// Check if pattern edge matches graph edge with current binding
__device__ bool try_match_edge(
    const DevicePatternEdge* pattern,
    EdgeId graph_edge,
    const DeviceEdges* edges,
    DeviceMatch* match              // In/out: bindings
);

// Find candidate edges for first pattern edge
__device__ uint32_t find_anchor_candidates(
    const DevicePatternEdge* pattern,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    const DeviceAdjacency* adj,
    EdgeId* candidates,             // Output buffer
    uint32_t max_candidates
);

// Find candidate edges using signature index (faster than adjacency for first edge)
__device__ uint32_t find_candidates_by_signature(
    const DevicePatternEdge* pattern,
    const uint64_t* state_bitmap,
    const DeviceSignatureIndex* sig_index,
    EdgeId* candidates,
    uint32_t max_candidates
);

// Find candidate edges using inverted vertex index (when variable is bound)
__device__ uint32_t find_candidates_by_vertex(
    VertexId vertex,
    uint8_t position,              // Expected position in edge
    uint8_t arity,                 // Expected edge arity
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    const DeviceInvertedVertexIndex* inv_index,
    EdgeId* candidates,
    uint32_t max_candidates
);

// Pattern match dispatcher - selects 1/2/3-edge specialized implementation
// Uses nested loops, not backtracking (despite the old name)
__device__ void find_matches_dispatch(
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    const DeviceAdjacency* adj,
    StateId source_state,
    uint64_t source_hash,
    uint16_t rule_index,
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
);

// =============================================================================
// HGMatch Task Processing (Generic Task Queue)
// =============================================================================
// These functions process SCAN/EXPAND tasks and push results to a generic
// Task queue. Used by hgmatch_megakernel for fine-grained parallelism.

// Forward declaration
template<typename T> struct WorkQueueView;

// Process a SCAN task: find candidates for first pattern edge, spawn EXPAND tasks
__device__ void process_scan_task(
    const Task& task,
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    WorkQueueView<Task>* task_queue,  // Generic task queue for spawning EXPAND
    DeviceMatch* output,              // Optional direct output (can be nullptr)
    uint32_t* num_matches,            // Optional match counter (can be nullptr)
    uint32_t max_matches,
    uint32_t num_edges
);

// Process an EXPAND task: extend partial match, spawn EXPAND or SINK tasks
__device__ void process_expand_task(
    const Task& task,
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    WorkQueueView<Task>* task_queue,  // Generic task queue
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
);

// =============================================================================
// Host Interface
// =============================================================================

class PatternMatcher {
public:
    void init();
    void destroy();

    // Find all matches across multiple states
    // Returns number of matches found
    // NOTE: edges, state_pool, adj are HOST structs containing DEVICE pointers
    uint32_t find_matches(
        const StateId* d_states,
        uint32_t num_states,
        const DeviceRewriteRule* d_rules,
        uint32_t num_rules,
        DeviceEdges edges,              // BY VALUE - host struct with device ptrs
        StatePool state_pool,           // BY VALUE
        DeviceAdjacency adj,            // BY VALUE
        DeviceMatch* d_matches,
        uint32_t max_matches,
        cudaStream_t stream = 0
    );

private:
    uint32_t* d_match_counts_;
    uint32_t* d_output_offsets_;
    void* d_temp_storage_;
    size_t temp_storage_bytes_;
};

}  // namespace hypergraph::gpu
