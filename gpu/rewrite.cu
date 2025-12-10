#include "rewrite.cuh"
#include "types.cuh"

namespace hypergraph::gpu {

// =============================================================================
// Device Helper Functions
// =============================================================================

// Allocate fresh vertices for RHS pattern
__device__ void allocate_fresh_vertices(
    const DeviceRewriteRule* rule,
    DeviceMatch* match,
    uint32_t* vertex_counter
) {
    // Fresh variables start at first_fresh_var
    uint8_t first_fresh = rule->first_fresh_var;

    // Count how many fresh vertices we need
    uint8_t num_fresh = 0;
    for (uint8_t e = 0; e < rule->num_rhs; ++e) {
        for (uint8_t p = 0; p < rule->rhs[e].arity; ++p) {
            uint8_t var = rule->rhs[e].vars[p];
            if (var >= first_fresh && !match->is_bound(var)) {
                ++num_fresh;
            }
        }
    }

    if (num_fresh == 0) return;

    // Allocate fresh vertices
    uint32_t base = atomicAdd(vertex_counter, num_fresh);

    // Assign to unbound fresh variables
    uint32_t next_vertex = base;
    for (uint8_t e = 0; e < rule->num_rhs; ++e) {
        for (uint8_t p = 0; p < rule->rhs[e].arity; ++p) {
            uint8_t var = rule->rhs[e].vars[p];
            if (var >= first_fresh && !match->is_bound(var)) {
                match->bindings[var] = next_vertex++;
                match->bound_mask |= (1u << var);
            }
        }
    }
}

// Create a single RHS edge
// Note: Edge data is written BEFORE alloc_edge() increments the counter
// so other threads cannot see the edge until data is fully written.
__device__ EdgeId create_rhs_edge(
    const DevicePatternEdge* pattern,
    const DeviceMatch* match,
    DeviceEdges* edges,
    uint32_t vertex_offset,
    uint32_t* vertex_data
) {
    // Write vertex data FIRST (before edge is visible)
    for (uint8_t p = 0; p < pattern->arity; ++p) {
        uint8_t var = pattern->vars[p];
        vertex_data[vertex_offset + p] = match->bindings[var];
    }

    // Allocate edge slot using device counter
    // This makes the edge visible to other threads
    uint32_t eid = edges->alloc_edge();
    if (eid == INVALID_ID) {
        return INVALID_ID;
    }

    // Set arity and offset
    edges->arities[eid] = pattern->arity;
    edges->vertex_offsets[eid] = vertex_offset;

    // Memory fence to ensure all writes are visible before we return
    // (bitmap operations will see this edge's data)
    __threadfence();

    return eid;
}

// Build new state bitmap
__device__ void build_new_state_bitmap(
    uint64_t* new_bitmap,
    const uint64_t* parent_bitmap,
    const EdgeId* consumed,
    uint8_t num_consumed,
    const EdgeId* produced,
    uint8_t num_produced
) {
    // Copy parent bitmap
    for (uint32_t i = 0; i < BITMAP_WORDS; ++i) {
        new_bitmap[i] = parent_bitmap[i];
    }

    // Clear consumed edges
    for (uint8_t i = 0; i < num_consumed; ++i) {
        EdgeId eid = consumed[i];
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        new_bitmap[word] &= ~(1ULL << bit);
    }

    // Set produced edges
    for (uint8_t i = 0; i < num_produced; ++i) {
        EdgeId eid = produced[i];
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        new_bitmap[word] |= (1ULL << bit);
    }
}

// Count edges in bitmap
__device__ uint32_t count_edges_in_bitmap(const uint64_t* bitmap) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < BITMAP_WORDS; ++i) {
        count += __popcll(bitmap[i]);
    }
    return count;
}

// =============================================================================
// Kernel Implementation
// =============================================================================

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
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_matches) return;

    const DeviceMatch* match = &matches[tid];
    const DeviceRewriteRule* rule = &rules[match->rule_index];
    RewriteOutput* output = &outputs[tid];

    // Initialize output
    output->new_state = INVALID_ID;
    output->canonical_state = INVALID_ID;
    output->event = INVALID_ID;
    output->num_produced = 0;
    output->was_new_state = false;
    output->success = false;

    // Copy match for modification (add fresh vertex bindings)
    DeviceMatch extended_match = *match;

    // Allocate fresh vertices
    allocate_fresh_vertices(rule, &extended_match, vertex_counter);

    // Calculate vertex data offset for new edges
    uint32_t total_rhs_arity = 0;
    for (uint8_t e = 0; e < rule->num_rhs; ++e) {
        total_rhs_arity += rule->rhs[e].arity;
    }

    // Allocate space for vertex data using the edges' vertex_data counter
    uint32_t vertex_data_offset = edges.alloc_vertex_data(total_rhs_arity);

    // Create RHS edges
    EdgeId produced[MAX_PATTERN_EDGES];
    uint8_t num_produced_count = 0;
    uint32_t current_offset = vertex_data_offset;

    for (uint8_t e = 0; e < rule->num_rhs; ++e) {
        EdgeId eid = create_rhs_edge(
            &rule->rhs[e], &extended_match, &edges,
            current_offset, edges.vertex_data
        );

        if (eid == INVALID_ID) {
            // Edge allocation failed
            return;
        }

        produced[num_produced_count++] = eid;
        current_offset += rule->rhs[e].arity;
    }

    // Allocate new state using the device counter
    uint32_t state_idx = state_pool.alloc_state();
    if (state_idx == INVALID_ID) {
        return;  // State pool full
    }

    // Get parent bitmap
    const uint64_t* parent_bitmap = state_pool.all_bitmaps +
                                    match->source_state * BITMAP_WORDS;

    // Get new state's bitmap
    uint64_t* new_bitmap = state_pool.all_bitmaps + state_idx * BITMAP_WORDS;

    // Build new state bitmap
    build_new_state_bitmap(
        new_bitmap, parent_bitmap,
        match->matched_edges, match->num_edges,
        produced, num_produced_count
    );

    // Initialize state metadata
    DeviceState* new_state = &state_pool.states[state_idx];
    new_state->bitmap = new_bitmap;
    new_state->canonical_hash = 0;  // Will be computed later
    new_state->parent_state = match->source_state;
    new_state->step = current_step;
    new_state->edge_count = count_edges_in_bitmap(new_bitmap);

    // Allocate event using the device counter
    uint32_t event_idx = event_pool.alloc_event();
    if (event_idx == INVALID_ID) {
        // Event pool full, but state was created
        output->new_state = state_idx;
        output->num_produced = num_produced_count;
        for (uint8_t i = 0; i < num_produced_count; ++i) {
            output->produced_edges[i] = produced[i];
        }
        output->success = true;
        return;
    }

    // Store consumed edges
    uint32_t cons_offset = event_pool.alloc_consumed(match->num_edges);
    for (uint8_t i = 0; i < match->num_edges; ++i) {
        event_pool.consumed_edges[cons_offset + i] = match->matched_edges[i];
    }

    // Store produced edges
    uint32_t prod_offset = event_pool.alloc_produced(num_produced_count);
    for (uint8_t i = 0; i < num_produced_count; ++i) {
        event_pool.produced_edges[prod_offset + i] = produced[i];
    }

    // Initialize event
    DeviceEvent* event = &event_pool.events[event_idx];
    event->input_state = match->source_state;
    event->output_state = state_idx;
    event->consumed_offset = cons_offset;
    event->produced_offset = prod_offset;
    event->num_consumed = match->num_edges;
    event->num_produced = num_produced_count;
    event->rule_index = match->rule_index;
    event->step = current_step;

    // Mark edges as produced by this event
    for (uint8_t i = 0; i < num_produced_count; ++i) {
        edges.creator_events[produced[i]] = event_idx;
    }

    new_state->parent_event = event_idx;
    output->event = event_idx;

    // Fill output
    output->new_state = state_idx;
    output->num_produced = num_produced_count;
    for (uint8_t i = 0; i < num_produced_count; ++i) {
        output->produced_edges[i] = produced[i];
    }
    output->success = true;
}

// =============================================================================
// Warp-Level Rewriting (for megakernel)
// =============================================================================
// NOTE: This function receives pointers to device-resident structs (from EvolutionContext)
// The structs themselves must have been copied to device memory before calling this.

__device__ RewriteOutput apply_rewrite_warp(
    const DeviceMatch* match,
    const DeviceRewriteRule* rule,
    DeviceEdges* edges,
    StatePool* state_pool,
    EventPool* event_pool,
    uint32_t* vertex_counter,
    uint32_t current_step
) {
    const uint32_t lane = threadIdx.x % 32;
    RewriteOutput output;

    // Only lane 0 does the actual rewrite
    if (lane == 0) {
        output.new_state = INVALID_ID;
        output.canonical_state = INVALID_ID;
        output.event = INVALID_ID;
        output.num_produced = 0;
        output.was_new_state = false;
        output.success = false;

        DeviceMatch extended_match = *match;
        allocate_fresh_vertices(rule, &extended_match, vertex_counter);

        // Calculate vertex data offset using edges allocator
        uint32_t total_rhs_arity = 0;
        for (uint8_t e = 0; e < rule->num_rhs; ++e) {
            total_rhs_arity += rule->rhs[e].arity;
        }

        // Use edges' vertex_data counter (not the vertex ID counter!)
        uint32_t vertex_data_offset = edges->alloc_vertex_data(total_rhs_arity);

        // Create RHS edges
        EdgeId produced[MAX_PATTERN_EDGES];
        uint8_t num_produced = 0;
        uint32_t current_offset = vertex_data_offset;

        for (uint8_t e = 0; e < rule->num_rhs; ++e) {
            EdgeId eid = create_rhs_edge(
                &rule->rhs[e], &extended_match, edges,
                current_offset, edges->vertex_data
            );

            if (eid == INVALID_ID) {
                return output;
            }

            produced[num_produced++] = eid;
            current_offset += rule->rhs[e].arity;
        }

        // Allocate state using the alloc method (uses device counter)
        uint32_t state_idx = state_pool->alloc_state();
        if (state_idx == INVALID_ID) {
            return output;
        }

        const uint64_t* parent_bitmap = state_pool->all_bitmaps +
                                        match->source_state * BITMAP_WORDS;
        uint64_t* new_bitmap = state_pool->all_bitmaps + state_idx * BITMAP_WORDS;

        build_new_state_bitmap(
            new_bitmap, parent_bitmap,
            match->matched_edges, match->num_edges,
            produced, num_produced
        );

        DeviceState* new_state = &state_pool->states[state_idx];
        new_state->bitmap = new_bitmap;
        new_state->canonical_hash = 0;
        new_state->parent_state = match->source_state;
        new_state->step = current_step;
        new_state->edge_count = count_edges_in_bitmap(new_bitmap);

        // Allocate event using the alloc method
        uint32_t event_idx = event_pool->alloc_event();
        if (event_idx != INVALID_ID) {
            uint32_t cons_offset = event_pool->alloc_consumed(match->num_edges);
            for (uint8_t i = 0; i < match->num_edges; ++i) {
                event_pool->consumed_edges[cons_offset + i] = match->matched_edges[i];
            }

            uint32_t prod_offset = event_pool->alloc_produced(num_produced);
            for (uint8_t i = 0; i < num_produced; ++i) {
                event_pool->produced_edges[prod_offset + i] = produced[i];
            }

            DeviceEvent* event = &event_pool->events[event_idx];
            event->input_state = match->source_state;
            event->output_state = state_idx;
            event->consumed_offset = cons_offset;
            event->produced_offset = prod_offset;
            event->num_consumed = match->num_edges;
            event->num_produced = num_produced;
            event->rule_index = match->rule_index;
            event->step = current_step;

            for (uint8_t i = 0; i < num_produced; ++i) {
                edges->creator_events[produced[i]] = event_idx;
            }

            new_state->parent_event = event_idx;
            output.event = event_idx;
        }

        output.new_state = state_idx;
        output.num_produced = num_produced;
        for (uint8_t i = 0; i < num_produced; ++i) {
            output.produced_edges[i] = produced[i];
        }
        output.success = true;

        // Snapshot counters AFTER all edge/state creation is complete
        // This ensures any other thread using these values sees fully written data
        __threadfence();  // Ensure all writes are visible
        output.num_edges_snapshot = *edges->num_edges;
        output.max_vertex_snapshot = *vertex_counter;
    }

    // Broadcast from lane 0 to all lanes
    output.new_state = __shfl_sync(0xFFFFFFFF, output.new_state, 0);
    output.event = __shfl_sync(0xFFFFFFFF, output.event, 0);
    output.success = __shfl_sync(0xFFFFFFFF, output.success ? 1 : 0, 0);
    output.num_edges_snapshot = __shfl_sync(0xFFFFFFFF, output.num_edges_snapshot, 0);
    output.max_vertex_snapshot = __shfl_sync(0xFFFFFFFF, output.max_vertex_snapshot, 0);

    return output;
}

// =============================================================================
// Host Interface Implementation
// =============================================================================

void Rewriter::init() {
    // No state needed
}

void Rewriter::destroy() {
    // No state to free
}

uint32_t Rewriter::apply_rewrites(
    const DeviceMatch* d_matches,
    uint32_t num_matches,
    const DeviceRewriteRule* d_rules,
    DeviceEdges edges,              // BY VALUE
    StatePool state_pool,           // BY VALUE
    EventPool event_pool,           // BY VALUE
    uint32_t* d_vertex_counter,
    RewriteOutput* d_outputs,
    uint32_t current_step,
    cudaStream_t stream
) {
    if (num_matches == 0) return 0;

    const int block_size = 256;
    const int num_blocks = (num_matches + block_size - 1) / block_size;

    apply_rewrites_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_matches, num_matches, d_rules, edges, state_pool, event_pool,
        d_vertex_counter, d_outputs, current_step
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Count successful rewrites (would need to check outputs on device)
    // For now, return num_matches as estimate
    return num_matches;
}

}  // namespace hypergraph::gpu
