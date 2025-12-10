#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "types.cuh"
#include "hash_table.cuh"

namespace hypergraph::gpu {

// =============================================================================
// Causal Graph Device Structures
// =============================================================================

// Reachability tracking for online transitive reduction
// Uses a sparse bitset per event for descendant tracking
struct ReachabilityInfo {
    // For small event counts, use dense bitset
    // For large counts, use hash set
    uint64_t* descendant_bitmaps;   // [MAX_EVENTS * words_per_event]
    uint32_t words_per_event;
    uint32_t max_events;

    __device__ bool is_reachable(EventId source, EventId target) const {
        if (source == target) return true;
        uint32_t word = target / 64;
        uint32_t bit = target % 64;
        uint64_t* bitmap = descendant_bitmaps + source * words_per_event;
        return (bitmap[word] & (1ULL << bit)) != 0;
    }

    __device__ void add_reachable(EventId source, EventId target) {
        uint32_t word = target / 64;
        uint32_t bit = target % 64;
        uint64_t* bitmap = descendant_bitmaps + source * words_per_event;
        atomicOr((unsigned long long*)&bitmap[word], 1ULL << bit);
    }

    // Merge target's descendants into source's descendants
    __device__ void merge_descendants(EventId source, EventId target) {
        uint64_t* src_bitmap = descendant_bitmaps + source * words_per_event;
        uint64_t* tgt_bitmap = descendant_bitmaps + target * words_per_event;
        for (uint32_t w = 0; w < words_per_event; ++w) {
            atomicOr((unsigned long long*)&src_bitmap[w], tgt_bitmap[w]);
        }
    }
};

// =============================================================================
// Causal Edge Kernels
// =============================================================================

// Register edge producers (set by rewrite kernel)
// This kernel just initializes edge_producer_map for initial edges
__global__ void init_edge_producers_kernel(
    const EdgeId* initial_edges,
    uint32_t num_initial_edges,
    EdgeProducerMap* edge_producer_map
);

// Compute causal edges from events
// Uses rendezvous pattern: producer writes then reads consumers,
// consumer writes then reads producer
__global__ void compute_causal_edges_kernel(
    EventPool events,                   // BY VALUE - struct copied to device
    uint32_t num_events,
    EdgeProducerMap edge_producer_map,  // BY VALUE
    CausalEdge* causal_output,
    uint32_t* causal_count,
    GPUHashSet<> seen_causal_pairs,     // BY VALUE - Deduplication
    bool transitive_reduction_enabled,
    ReachabilityInfo* reachability,
    uint32_t* redundant_count           // Statistics
);

// =============================================================================
// Branchial Edge Kernels
// =============================================================================

// Per-state event lists for branchial computation
struct StateEventLists {
    uint32_t* event_offsets;        // [MAX_STATES+1] Start offset per state
    EventId* event_ids;             // Flattened event IDs
    uint32_t num_entries;
};

// Build state -> events mapping
__global__ void build_state_event_lists_kernel(
    EventPool events,               // BY VALUE
    uint32_t num_events,
    StateEventLists lists           // BY VALUE
);

// Second pass to populate lists after prefix sum
__global__ void populate_state_event_lists_kernel(
    EventPool events,               // BY VALUE
    uint32_t num_events,
    StateEventLists lists,          // BY VALUE
    uint32_t* counters              // Per-state counters for insertion
);

// Compute branchial edges (events sharing consumed edge from same state)
__global__ void compute_branchial_edges_kernel(
    EventPool events,               // BY VALUE
    StateEventLists state_events,   // BY VALUE
    uint32_t num_states,
    BranchialEdge* branchial_output,
    uint32_t* branchial_count,
    GPUHashSet<> seen_branchial_pairs  // BY VALUE - Deduplication
);

// =============================================================================
// Online Transitive Reduction
// =============================================================================

// Add causal edge with online TR check
// Returns true if edge was added (not redundant)
__device__ bool add_causal_edge_with_tr(
    EventId producer,
    EventId consumer,
    EdgeId edge,
    CausalEdge* causal_output,
    uint32_t* causal_count,
    GPUHashSet<>& seen_pairs,           // BY REFERENCE (passed by value to kernel)
    ReachabilityInfo* reachability,
    uint32_t* redundant_count
);

// Update reachability after adding edge
__device__ void update_reachability_for_edge(
    EventId producer,
    EventId consumer,
    ReachabilityInfo* reachability
);

// =============================================================================
// Host Interface
// =============================================================================

class CausalGraphGPU {
public:
    void init(uint32_t max_events);
    void destroy();

    // Enable/disable online transitive reduction
    void set_transitive_reduction(bool enabled) { tr_enabled_ = enabled; }

    // Compute causal and branchial edges from events
    void compute_causal_edges(
        EventPool events,               // BY VALUE
        uint32_t num_events,
        EdgeProducerMap edge_producers, // BY VALUE
        CausalEdge* d_causal_output,
        uint32_t* d_causal_count,
        cudaStream_t stream = 0
    );

    void compute_branchial_edges(
        EventPool events,               // BY VALUE
        uint32_t num_events,
        uint32_t num_states,
        BranchialEdge* d_branchial_output,
        uint32_t* d_branchial_count,
        cudaStream_t stream = 0
    );

    // Get statistics
    uint32_t get_redundant_count() const;

private:
    bool tr_enabled_;
    GPUHashSet<> seen_causal_pairs_;
    GPUHashSet<> seen_branchial_pairs_;
    ReachabilityInfo reachability_;
    uint32_t* d_redundant_count_;

    // State event lists for branchial
    StateEventLists state_events_;
};

}  // namespace hypergraph::gpu
