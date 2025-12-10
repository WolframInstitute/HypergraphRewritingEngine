#include "causal.cuh"
#include "types.cuh"

namespace hypergraph::gpu {

// =============================================================================
// Online Transitive Reduction Device Functions
// =============================================================================

__device__ bool add_causal_edge_with_tr(
    EventId producer,
    EventId consumer,
    EdgeId edge,
    CausalEdge* causal_output,
    uint32_t* causal_count,
    GPUHashSet<>& seen_pairs,           // BY REFERENCE
    ReachabilityInfo* reachability,
    uint32_t* redundant_count
) {
    // Check if this exact pair already processed
    uint64_t pair_key = (uint64_t(producer) << 32) | consumer;
    if (!seen_pairs.insert(pair_key)) {
        // Already seen this pair
        return false;
    }

    // Online TR check: is consumer already reachable from producer?
    if (reachability != nullptr && reachability->is_reachable(producer, consumer)) {
        // Edge is redundant - consumer already reachable via transitive path
        if (redundant_count != nullptr) {
            atomicAdd(redundant_count, 1);
        }
        return false;
    }

    // Add the causal edge
    uint32_t idx = atomicAdd(causal_count, 1);
    causal_output[idx].producer = producer;
    causal_output[idx].consumer = consumer;
    causal_output[idx].edge = edge;

    // Update reachability for future TR checks
    if (reachability != nullptr) {
        update_reachability_for_edge(producer, consumer, reachability);
    }

    return true;
}

__device__ void update_reachability_for_edge(
    EventId producer,
    EventId consumer,
    ReachabilityInfo* reachability
) {
    // Add consumer to producer's descendants
    reachability->add_reachable(producer, consumer);

    // Merge consumer's descendants into producer's descendants
    // (transitive closure update)
    reachability->merge_descendants(producer, consumer);

    // Note: For full correctness with concurrent updates, we'd need to
    // also update all ancestors of producer. This simplified version
    // assumes mostly sequential event creation (which is true for
    // causal graphs where earlier events have lower IDs).
}

// =============================================================================
// Causal Edge Kernel
// =============================================================================

__global__ void init_edge_producers_kernel(
    const EdgeId* initial_edges,
    uint32_t num_initial_edges,
    EdgeProducerMap* edge_producer_map
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_initial_edges) return;

    // Initial edges have no producer
    // edge_producer_map already initialized to INVALID_ID
}

__global__ void compute_causal_edges_kernel(
    EventPool events,                       // BY VALUE
    uint32_t num_events,
    EdgeProducerMap edge_producer_map,      // BY VALUE
    CausalEdge* causal_output,
    uint32_t* causal_count,
    GPUHashSet<> seen_causal_pairs,         // BY VALUE
    bool transitive_reduction_enabled,
    ReachabilityInfo* reachability,
    uint32_t* redundant_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const DeviceEvent* event = &events.events[tid];
    EventId consumer = tid;

    // For each edge consumed by this event, check if there's a producer
    for (uint8_t i = 0; i < event->num_consumed; ++i) {
        EdgeId edge = events.consumed_edges[event->consumed_offset + i];
        EventId producer = edge_producer_map.get_producer(edge);

        if (producer != INVALID_ID && producer != consumer) {
            // Found causal relationship: producer -> consumer
            if (transitive_reduction_enabled) {
                add_causal_edge_with_tr(
                    producer, consumer, edge,
                    causal_output, causal_count,
                    seen_causal_pairs, reachability, redundant_count
                );
            } else {
                // No TR - just add edge with deduplication
                uint64_t pair_key = (uint64_t(producer) << 32) | consumer;
                if (seen_causal_pairs.insert(pair_key)) {
                    uint32_t idx = atomicAdd(causal_count, 1);
                    causal_output[idx].producer = producer;
                    causal_output[idx].consumer = consumer;
                    causal_output[idx].edge = edge;
                }
            }
        }
    }
}

// =============================================================================
// Branchial Edge Kernels
// =============================================================================

__global__ void build_state_event_lists_kernel(
    EventPool events,               // BY VALUE
    uint32_t num_events,
    StateEventLists lists           // BY VALUE
) {
    // Phase 1: Count events per state (parallel histogram)
    // This is a simplified version - real impl would use CUB for efficiency

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const DeviceEvent* event = &events.events[tid];
    StateId input_state = event->input_state;

    // Atomically increment count for this state
    atomicAdd(&lists.event_offsets[input_state + 1], 1);
}

// Second pass to populate lists after prefix sum
__global__ void populate_state_event_lists_kernel(
    EventPool events,               // BY VALUE
    uint32_t num_events,
    StateEventLists lists,          // BY VALUE
    uint32_t* counters  // Per-state counters for insertion
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_events) return;

    const DeviceEvent* event = &events.events[tid];
    StateId input_state = event->input_state;

    // Get insertion position
    uint32_t offset = lists.event_offsets[input_state];
    uint32_t pos = atomicAdd(&counters[input_state], 1);

    lists.event_ids[offset + pos] = tid;
}

__global__ void compute_branchial_edges_kernel(
    EventPool events,                       // BY VALUE
    StateEventLists state_events,           // BY VALUE
    uint32_t num_states,
    BranchialEdge* branchial_output,
    uint32_t* branchial_count,
    GPUHashSet<> seen_branchial_pairs       // BY VALUE
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    uint32_t start = state_events.event_offsets[tid];
    uint32_t end = state_events.event_offsets[tid + 1];
    uint32_t num_events_in_state = end - start;

    if (num_events_in_state < 2) return;  // Need at least 2 events for branchial

    // Compare all pairs of events from this state
    for (uint32_t i = 0; i < num_events_in_state; ++i) {
        EventId e1 = state_events.event_ids[start + i];
        const DeviceEvent* event1 = &events.events[e1];

        for (uint32_t j = i + 1; j < num_events_in_state; ++j) {
            EventId e2 = state_events.event_ids[start + j];
            const DeviceEvent* event2 = &events.events[e2];

            // Check for shared consumed edge
            EdgeId shared = INVALID_ID;
            for (uint8_t a = 0; a < event1->num_consumed && shared == INVALID_ID; ++a) {
                EdgeId edge1 = events.consumed_edges[event1->consumed_offset + a];
                for (uint8_t b = 0; b < event2->num_consumed; ++b) {
                    EdgeId edge2 = events.consumed_edges[event2->consumed_offset + b];
                    if (edge1 == edge2) {
                        shared = edge1;
                        break;
                    }
                }
            }

            if (shared != INVALID_ID) {
                // Found branchial relationship
                EventId min_e = min(e1, e2);
                EventId max_e = max(e1, e2);
                uint64_t pair_key = (uint64_t(min_e) << 32) | max_e;

                if (seen_branchial_pairs.insert(pair_key)) {
                    uint32_t idx = atomicAdd(branchial_count, 1);
                    branchial_output[idx].event1 = min_e;
                    branchial_output[idx].event2 = max_e;
                    branchial_output[idx].shared_edge = shared;
                }
            }
        }
    }
}

// =============================================================================
// Host Interface Implementation
// =============================================================================

void CausalGraphGPU::init(uint32_t max_events) {
    tr_enabled_ = false;

    seen_causal_pairs_.init(max_events * 4);
    seen_branchial_pairs_.init(max_events * 2);

    CUDA_CHECK(cudaMalloc(&d_redundant_count_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_redundant_count_, 0, sizeof(uint32_t)));

    // Initialize reachability info
    uint32_t words_per_event = (max_events + 63) / 64;
    reachability_.words_per_event = words_per_event;
    reachability_.max_events = max_events;

    size_t bitmap_size = max_events * words_per_event * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&reachability_.descendant_bitmaps, bitmap_size));
    CUDA_CHECK(cudaMemset(reachability_.descendant_bitmaps, 0, bitmap_size));

    // Initialize state event lists
    CUDA_CHECK(cudaMalloc(&state_events_.event_offsets, (MAX_STATES + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&state_events_.event_ids, max_events * sizeof(EventId)));
}

void CausalGraphGPU::destroy() {
    seen_causal_pairs_.destroy();
    seen_branchial_pairs_.destroy();

    if (d_redundant_count_) cudaFree(d_redundant_count_);
    if (reachability_.descendant_bitmaps) cudaFree(reachability_.descendant_bitmaps);
    if (state_events_.event_offsets) cudaFree(state_events_.event_offsets);
    if (state_events_.event_ids) cudaFree(state_events_.event_ids);
}

void CausalGraphGPU::compute_causal_edges(
    EventPool events,                       // BY VALUE
    uint32_t num_events,
    EdgeProducerMap edge_producers,         // BY VALUE
    CausalEdge* d_causal_output,
    uint32_t* d_causal_count,
    cudaStream_t stream
) {
    if (num_events == 0) return;

    const int block_size = 256;
    const int num_blocks = (num_events + block_size - 1) / block_size;

    // Need to copy reachability info to device-accessible pointer
    ReachabilityInfo* d_reachability = nullptr;
    if (tr_enabled_) {
        CUDA_CHECK(cudaMalloc(&d_reachability, sizeof(ReachabilityInfo)));
        CUDA_CHECK(cudaMemcpy(d_reachability, &reachability_,
                              sizeof(ReachabilityInfo), cudaMemcpyHostToDevice));
    }

    // Pass structs BY VALUE - they contain device pointers, CUDA copies the struct
    compute_causal_edges_kernel<<<num_blocks, block_size, 0, stream>>>(
        events, num_events, edge_producers,
        d_causal_output, d_causal_count,
        seen_causal_pairs_, tr_enabled_, d_reachability, d_redundant_count_
    );

    if (d_reachability) {
        cudaFree(d_reachability);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CausalGraphGPU::compute_branchial_edges(
    EventPool events,                       // BY VALUE
    uint32_t num_events,
    uint32_t num_states,
    BranchialEdge* d_branchial_output,
    uint32_t* d_branchial_count,
    cudaStream_t stream
) {
    if (num_events == 0 || num_states == 0) return;

    const int block_size = 256;

    // Phase 1: Count events per state
    CUDA_CHECK(cudaMemset(state_events_.event_offsets, 0,
                          (num_states + 1) * sizeof(uint32_t)));

    int num_blocks = (num_events + block_size - 1) / block_size;
    // Pass structs BY VALUE
    build_state_event_lists_kernel<<<num_blocks, block_size, 0, stream>>>(
        events, num_events, state_events_
    );

    // Phase 2: Prefix sum (simplified - use CUB in real implementation)
    // For now, do on host
    CUDA_CHECK(cudaStreamSynchronize(stream));  // Need to sync before reading
    std::vector<uint32_t> h_offsets(num_states + 1);
    CUDA_CHECK(cudaMemcpy(h_offsets.data(), state_events_.event_offsets,
                          (num_states + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t total = 0;
    for (uint32_t i = 0; i <= num_states; ++i) {
        uint32_t count = h_offsets[i];
        h_offsets[i] = total;
        total += count;
    }
    state_events_.num_entries = total;

    CUDA_CHECK(cudaMemcpy(state_events_.event_offsets, h_offsets.data(),
                          (num_states + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Phase 3: Populate lists
    uint32_t* d_counters;
    CUDA_CHECK(cudaMalloc(&d_counters, num_states * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_counters, 0, num_states * sizeof(uint32_t)));

    // Pass structs BY VALUE
    populate_state_event_lists_kernel<<<num_blocks, block_size, 0, stream>>>(
        events, num_events, state_events_, d_counters
    );

    cudaFree(d_counters);

    // Phase 4: Compute branchial edges
    num_blocks = (num_states + block_size - 1) / block_size;
    // Pass structs BY VALUE
    compute_branchial_edges_kernel<<<num_blocks, block_size, 0, stream>>>(
        events, state_events_, num_states,
        d_branchial_output, d_branchial_count, seen_branchial_pairs_
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

uint32_t CausalGraphGPU::get_redundant_count() const {
    uint32_t count;
    CUDA_CHECK(cudaMemcpy(&count, d_redundant_count_, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    return count;
}

}  // namespace hypergraph::gpu
