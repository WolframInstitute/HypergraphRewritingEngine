#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "types.cuh"

namespace hypergraph::gpu {

// =============================================================================
// Rewriting Declarations
// =============================================================================

// Result of applying a single rewrite
struct RewriteOutput {
    StateId new_state;          // New state ID (raw, not canonical)
    StateId canonical_state;    // Canonical state ID (after dedup)
    EventId event;              // Event created
    uint8_t num_produced;       // Number of edges produced
    EdgeId produced_edges[MAX_PATTERN_EDGES];
    bool was_new_state;         // True if first time seeing this canonical state
    bool success;               // True if rewrite succeeded
    uint32_t num_edges_snapshot; // Edge count after this rewrite (for WL hash)
    uint32_t max_vertex_snapshot; // Vertex count after this rewrite (for WL hash)
};

// Apply rewrites for a batch of matches
// NOTE: DeviceEdges, StatePool, EventPool passed BY VALUE
__global__ void apply_rewrites_kernel(
    const DeviceMatch* matches,
    uint32_t num_matches,
    const DeviceRewriteRule* rules,
    DeviceEdges edges,              // BY VALUE
    StatePool state_pool,           // BY VALUE
    EventPool event_pool,           // BY VALUE
    uint32_t* vertex_counter,
    RewriteOutput* outputs,
    uint32_t current_step
);

// =============================================================================
// Warp-Level Rewriting (for megakernel)
// =============================================================================

// Apply a single match using one warp
// Returns RewriteOutput with results
__device__ RewriteOutput apply_rewrite_warp(
    const DeviceMatch* match,
    const DeviceRewriteRule* rule,
    DeviceEdges* edges,
    StatePool* state_pool,
    EventPool* event_pool,
    uint32_t* vertex_counter,
    uint32_t current_step
);

// =============================================================================
// Device Helper Functions
// =============================================================================

// Allocate fresh vertices for RHS pattern
__device__ void allocate_fresh_vertices(
    const DeviceRewriteRule* rule,
    DeviceMatch* match,             // Extended with fresh vertex bindings
    uint32_t* vertex_counter
);

// Create an edge from RHS pattern
__device__ EdgeId create_rhs_edge(
    const DevicePatternEdge* pattern,
    const DeviceMatch* match,
    DeviceEdges* edges,
    uint32_t* edge_count_ptr,
    uint32_t vertex_offset,         // Offset into vertex_data
    uint32_t* vertex_data           // Destination for vertices
);

// Build new state bitmap: parent - consumed + produced
__device__ void build_new_state_bitmap(
    uint64_t* new_bitmap,
    const uint64_t* parent_bitmap,
    const EdgeId* consumed,
    uint8_t num_consumed,
    const EdgeId* produced,
    uint8_t num_produced
);

// =============================================================================
// Host Interface
// =============================================================================

class Rewriter {
public:
    void init();
    void destroy();

    // Apply rewrites for a batch of matches
    // Returns number of successful rewrites
    // NOTE: edges, state_pool, event_pool are HOST structs with DEVICE pointers
    uint32_t apply_rewrites(
        const DeviceMatch* d_matches,
        uint32_t num_matches,
        const DeviceRewriteRule* d_rules,
        DeviceEdges edges,              // BY VALUE
        StatePool state_pool,           // BY VALUE
        EventPool event_pool,           // BY VALUE
        uint32_t* d_vertex_counter,
        RewriteOutput* d_outputs,
        uint32_t current_step,
        cudaStream_t stream = 0
    );

private:
    // No state needed - all work is done in kernels
};

}  // namespace hypergraph::gpu
