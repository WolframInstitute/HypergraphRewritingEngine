#include "evolution.cuh"
#include "types.cuh"
#include "wl_hash.cuh"
#include "pattern_match.cuh"
#include "rewrite.cuh"
#include "causal.cuh"
#include "signature_index.cuh"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>

namespace hypergraph::gpu {

// =============================================================================
// Helper: Allocate a MatchNode from the pool
// =============================================================================
__device__ MatchNode* alloc_match_node(MatchNode* pool, uint32_t* next_idx) {
    uint32_t idx = atomicAdd(next_idx, 1);
    if (idx >= MAX_MATCHES_PER_STEP) return nullptr;
    return &pool[idx];
}

// =============================================================================
// Helper: Allocate a ChildNode from the pool
// =============================================================================
__device__ ChildNode* alloc_child_node(ChildNode* pool, uint32_t* next_idx) {
    uint32_t idx = atomicAdd(next_idx, 1);
    if (idx >= MAX_STATES) return nullptr;
    return &pool[idx];
}

// =============================================================================
// Helper: Store a match for a state (for forwarding to children)
// =============================================================================
__device__ uint64_t store_match_for_state(
    StateId state,
    MatchRecord& match,
    StateMatchList* state_matches,
    uint64_t* global_epoch,
    MatchNode* match_node_pool,
    uint32_t* match_node_next
) {
    // Get epoch BEFORE storing (for ordering)
    uint64_t epoch = atomicAdd((unsigned long long*)global_epoch, 1);
    match.storage_epoch = epoch;

    // Allocate node and push to list
    MatchNode* node = alloc_match_node(match_node_pool, match_node_next);
    if (node != nullptr) {
        node->match = match;
        state_matches[state].push(node);
    }

    return epoch;
}

// =============================================================================
// Helper: Register child with parent (for push/pull coordination)
// =============================================================================
__device__ uint64_t register_child_with_parent(
    StateId parent,
    StateId child,
    const EdgeId* consumed_edges,
    uint8_t num_consumed,
    uint32_t creation_step,
    StateChildrenList* state_children,
    ParentInfo* state_parents,
    uint64_t* global_epoch,
    ChildNode* child_node_pool,
    uint32_t* child_node_next
) {
    if (parent == INVALID_ID) return 0;

    // Build ChildInfo
    ChildNode* node = alloc_child_node(child_node_pool, child_node_next);
    if (node == nullptr) return 0;

    node->info.child_state = child;
    node->info.num_consumed = num_consumed;
    node->info.creation_step = creation_step;
    for (uint8_t i = 0; i < num_consumed; ++i) {
        node->info.consumed_edges[i] = consumed_edges[i];
    }

    // Push child to parent's list FIRST (before getting epoch)
    state_children[parent].push(node);

    // Memory fence ensures child is visible before epoch is read
    __threadfence();

    // Get registration epoch
    uint64_t epoch = atomicAdd((unsigned long long*)global_epoch, 1);
    node->info.registration_epoch = epoch;

    // Store parent info for this child (for ancestor chain walking)
    ParentInfo* pi = &state_parents[child];
    pi->parent_state = parent;
    pi->num_consumed = num_consumed;
    for (uint8_t i = 0; i < num_consumed; ++i) {
        pi->consumed_edges[i] = consumed_edges[i];
    }

    return epoch;
}

// =============================================================================
// Helper: Push match to immediate children
// =============================================================================
__device__ void push_match_to_children(
    StateId parent,
    const MatchRecord& match,
    uint32_t step,
    StateChildrenList* state_children,
    StateMatchList* state_matches,
    uint64_t* global_epoch,
    MatchNode* match_node_pool,
    uint32_t* match_node_next,
    GPUHashSetView<>* seen_match_hashes,
    WorkQueueView<RewriteTaskWithMatch>* rewrite_queue,
    TerminationDetectorView* termination
) {
    // Iterate over all children of parent
    state_children[parent].for_each([&](const ChildInfo& child_info) {
        // Skip if match overlaps with consumed edges
        if (child_info.match_overlaps_consumed(match.matched_edges, match.num_edges)) {
            return;
        }

        // Create forwarded match for child state
        MatchRecord forwarded = match;
        forwarded.source_state = child_info.child_state;
        // Note: canonical_source would be looked up from state if needed

        // Deduplicate
        uint64_t h = forwarded.hash();
        if (!seen_match_hashes->insert(h)) {
            return;  // Already seen
        }

        // Store match in child (so grandchildren can find it)
        store_match_for_state(
            child_info.child_state, forwarded,
            state_matches, global_epoch,
            match_node_pool, match_node_next
        );

        // Recursive push to grandchildren
        push_match_to_children(
            child_info.child_state, forwarded,
            child_info.creation_step + 1,
            state_children, state_matches, global_epoch,
            match_node_pool, match_node_next,
            seen_match_hashes, rewrite_queue,
            termination
        );

        // CRITICAL: Increment work count BEFORE pushing
        termination->work_created();
        // Spawn REWRITE task
        RewriteTaskWithMatch rtask;
        rtask.match = forwarded;
        rtask.step = child_info.creation_step + 1;
        if (!rewrite_queue->push_wait(rtask)) {
            termination->work_finished();  // Rollback work count on failure
        }
    });
}

// =============================================================================
// Helper: Forward matches from ancestor chain (pull model)
// =============================================================================
__device__ void forward_from_ancestors(
    StateId child,
    StateId parent,
    const EdgeId* consumed_at_child,
    uint8_t num_consumed_at_child,
    uint32_t step,
    StateMatchList* state_matches,
    ParentInfo* state_parents,
    GPUHashSetView<>* seen_match_hashes,
    WorkQueueView<RewriteTaskWithMatch>* rewrite_queue,
    TerminationDetectorView* termination
) {
    // Accumulate consumed edges along the path
    EdgeId accumulated_consumed[MAX_PATTERN_EDGES * 8];
    uint8_t total_consumed = 0;

    // Add edges consumed to create this child
    for (uint8_t i = 0; i < num_consumed_at_child && total_consumed < MAX_PATTERN_EDGES * 8; ++i) {
        accumulated_consumed[total_consumed++] = consumed_at_child[i];
    }

    // Walk up the ancestor chain
    StateId current_ancestor = parent;
    while (current_ancestor != INVALID_ID) {
        // Forward matches from this ancestor
        state_matches[current_ancestor].for_each([&](const MatchRecord& ancestor_match) {
            // Check if match overlaps with ANY consumed edge along the path
            bool overlaps = ancestor_match.overlaps_edges(accumulated_consumed, total_consumed);
            if (overlaps) return;

            // Create forwarded match for child
            MatchRecord forwarded = ancestor_match;
            forwarded.source_state = child;

            // Deduplicate
            uint64_t h = forwarded.hash();
            if (!seen_match_hashes->insert(h)) {
                return;  // Already seen
            }

            // Spawn REWRITE task
            RewriteTaskWithMatch rtask;
            rtask.match = forwarded;
            rtask.step = step;
            // CRITICAL: Increment work count BEFORE push, retry with backoff
            termination->work_created();
            if (!rewrite_queue->push_wait(rtask)) {
                termination->work_finished();  // Rollback work count on failure
            }
        });

        // Move to next ancestor
        ParentInfo* pi = &state_parents[current_ancestor];
        if (!pi->has_parent()) break;

        // Add this ancestor's consumed edges
        for (uint8_t i = 0; i < pi->num_consumed && total_consumed < MAX_PATTERN_EDGES * 8; ++i) {
            accumulated_consumed[total_consumed++] = pi->consumed_edges[i];
        }
        current_ancestor = pi->parent_state;
    }
}

// =============================================================================
// Megakernel with Match Forwarding (Task-Parallel, Eager Rewriting)
// =============================================================================
__global__ void evolution_megakernel_with_forwarding(
    EvolutionContext* ctx,
    WorkQueueView<MatchTaskWithContext>* match_queue,
    WorkQueueView<RewriteTaskWithMatch>* rewrite_queue,
    StateMatchList* state_matches,
    StateChildrenList* state_children,
    ParentInfo* state_parents,
    uint64_t* global_epoch,
    MatchNode* match_node_pool,
    uint32_t* match_node_next,
    ChildNode* child_node_pool,
    uint32_t* child_node_next,
    TerminationDetectorView* termination,
    uint32_t max_steps,
    bool match_forwarding_enabled,
    bool batched_matching
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warps_per_block = blockDim.x / 32;

    __shared__ MatchTaskWithContext shared_match_tasks[8];
    __shared__ RewriteTaskWithMatch shared_rewrite_tasks[8];
    __shared__ uint32_t shared_has_work[8];

    // Safety counter to prevent infinite loops due to bugs
    uint32_t iterations = 0;
    const uint32_t MAX_ITERATIONS = 1000000;

    while (!termination->is_done() && iterations < MAX_ITERATIONS) {
        iterations++;
        bool did_work = false;

        // =====================================================================
        // Try to get MATCH work
        // =====================================================================
        if (lane == 0) {
            shared_has_work[warp_id] = match_queue->try_pop(&shared_match_tasks[warp_id]) ? 1 : 0;
        }
        __syncwarp();

        if (shared_has_work[warp_id]) {
            did_work = true;
            // Work was already counted when pushed to queue

            MatchTaskWithContext& task = shared_match_tasks[warp_id];

            if (task.step <= max_steps) {
                StateId state = task.state_id;
                bool has_parent = task.context.has_parent();

                // ===========================================================
                // PHASE 1: Forward from ancestors (if match forwarding enabled)
                // ===========================================================
                if (match_forwarding_enabled && has_parent && lane == 0) {
                    forward_from_ancestors(
                        state,
                        task.context.parent_state,
                        task.context.consumed_edges,
                        task.context.num_consumed,
                        task.step,
                        state_matches,
                        state_parents,
                        (GPUHashSetView<>*)ctx->seen_match_hashes,
                        rewrite_queue,
                        termination
                    );
                }
                __syncwarp();

                // ===========================================================
                // PHASE 2: Find new matches (delta or full)
                // ===========================================================
                uint32_t global_warp_idx = blockIdx.x * warps_per_block + warp_id;
                DeviceMatch* match_output = ctx->matches_buffer + global_warp_idx * ctx->matches_per_warp;

                uint32_t num_matches;
                if (match_forwarding_enabled && has_parent) {
                    // Delta matching: only on produced edges
                    num_matches = find_delta_matches_warp(
                        state,
                        ctx->rules,
                        ctx->num_rules,
                        ctx->edges,
                        ctx->states,
                        ctx->adjacency,
                        task.context.produced_edges,
                        task.context.num_produced,
                        match_output,
                        1024
                    );
                } else {
                    // Full matching: initial state or forwarding disabled
                    num_matches = find_matches_warp(
                        state,
                        ctx->rules,
                        ctx->num_rules,
                        ctx->edges,
                        ctx->states,
                        ctx->adjacency,
                        match_output,
                        1024
                    );
                }

                // ===========================================================
                // PHASE 3: Process matches (EAGER - spawn REWRITEs immediately)
                // ===========================================================
                for (uint32_t i = lane; i < num_matches; i += 32) {
                    DeviceMatch& m = match_output[i];

                    // Build MatchRecord from DeviceMatch
                    MatchRecord record;
                    record.rule_index = m.rule_index;
                    record.num_edges = m.num_edges;
                    record.source_state = state;
                    record.bound_mask = m.bound_mask;
                    for (uint8_t j = 0; j < m.num_edges; ++j) {
                        record.matched_edges[j] = m.matched_edges[j];
                    }
                    for (uint8_t j = 0; j < MAX_VARS; ++j) {
                        record.bindings[j] = m.bindings[j];
                    }

                    // Deduplicate
                    uint64_t h = record.hash();
                    GPUHashSetView<>* seen = (GPUHashSetView<>*)ctx->seen_match_hashes;
                    bool inserted = seen->insert(h);

                    if (!inserted) {
                        continue;  // Already seen
                    }

                    // Store match for forwarding to children
                    if (match_forwarding_enabled) {
                        store_match_for_state(
                            state, record,
                            state_matches, global_epoch,
                            match_node_pool, match_node_next
                        );

                        // Push to existing children
                        push_match_to_children(
                            state, record, task.step,
                            state_children, state_matches, global_epoch,
                            match_node_pool, match_node_next,
                            seen, rewrite_queue,
                            termination
                        );
                    }

                    // EAGER: Spawn REWRITE immediately
                    RewriteTaskWithMatch rtask;
                    rtask.match = record;
                    rtask.step = task.step;
                    // CRITICAL: Increment work count BEFORE push, retry with backoff
                    termination->work_created();
                    bool push_ok = rewrite_queue->push_wait(rtask);
                    if (!push_ok) {
                        termination->work_finished();  // Rollback work count on failure
                    }
                }
                __syncwarp();
                __threadfence();  // Ensure pushes are visible
            }

            // CRITICAL: Decrement work count AFTER all child work is created
            if (lane == 0) {
                termination->work_finished();
            }
        }

        // =====================================================================
        // Try to get REWRITE work
        // =====================================================================
        if (lane == 0) {
            shared_has_work[warp_id] = rewrite_queue->try_pop(&shared_rewrite_tasks[warp_id]) ? 1 : 0;
        }
        __syncwarp();

        if (shared_has_work[warp_id]) {
            did_work = true;
            // Work was already counted when pushed to queue

            RewriteTaskWithMatch& task = shared_rewrite_tasks[warp_id];

            if (task.step <= max_steps) {
                // Convert MatchRecord to DeviceMatch for rewrite
                DeviceMatch m;
                m.rule_index = task.match.rule_index;
                m.num_edges = task.match.num_edges;
                m.source_state = task.match.source_state;
                m.bound_mask = task.match.bound_mask;
                for (uint8_t j = 0; j < task.match.num_edges; ++j) {
                    m.matched_edges[j] = task.match.matched_edges[j];
                }
                for (uint8_t j = 0; j < MAX_VARS; ++j) {
                    m.bindings[j] = task.match.bindings[j];
                }

                // Apply rewrite
                RewriteOutput output = apply_rewrite_warp(
                    &m,
                    &ctx->rules[task.match.rule_index],
                    ctx->edges,
                    ctx->states,
                    ctx->events,
                    ctx->vertex_counter,
                    task.step
                );

                if (output.success) {
                    // Ensure all edge writes are globally visible before computing WL hash
                    // This is critical for correctness when multiple warps write edges concurrently
                    __threadfence();

                    // Compute WL canonical hash for new state
                    uint64_t* state_bitmap = ctx->states->all_bitmaps +
                                             output.new_state * BITMAP_WORDS;

                    // Find max edge ID in bitmap using __clzll (O(BITMAP_WORDS) but with fast intrinsic)
                    // Safer than global counters which can include concurrent writes
                    uint32_t max_edge_in_bitmap = 0;
                    if (lane == 0) {
                        for (int32_t word = BITMAP_WORDS - 1; word >= 0; --word) {
                            if (state_bitmap[word] != 0) {
                                // Use __clzll to find highest set bit efficiently
                                int leading_zeros = __clzll(state_bitmap[word]);
                                max_edge_in_bitmap = word * 64 + (63 - leading_zeros) + 1;
                                break;
                            }
                        }
                    }
                    max_edge_in_bitmap = __shfl_sync(0xFFFFFFFF, max_edge_in_bitmap, 0);

                    // Use snapshot for max_vertex but calculated max for edges
                    uint32_t max_vertex = output.max_vertex_snapshot;
                    uint32_t num_edges = max_edge_in_bitmap;

                    // Use per-warp WL scratch space to avoid races
                    // Each warp in each block gets its own portion of the scratch buffer
                    uint32_t global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_id;
                    uint64_t* my_scratch = ctx->wl_scratch + global_warp_id * ctx->wl_scratch_per_block;

                    // Cap max_vertex to avoid out-of-bounds scratch access
                    // Scratch needs 2 * max_vertex entries (colors_a and colors_b)
                    uint32_t max_scratch_vertices = ctx->wl_scratch_per_block / 2;
                    if (max_vertex > max_scratch_vertices) {
                        max_vertex = max_scratch_vertices;
                    }

                    // Use WL hash with on-the-fly adjacency building
                    // Warp-parallel: all lanes participate in hash computation
                    uint64_t hash = compute_wl_hash_no_adj(
                        state_bitmap,
                        ctx->edges,
                        num_edges,
                        max_vertex,
                        my_scratch
                    );

                    // Only lane 0 does the rest (hash already broadcast to all lanes)
                    if (lane == 0) {
                        ctx->states->states[output.new_state].canonical_hash = hash;

                        // Deduplicate by canonical hash
                        GPUHashTableView<>* canonical_map = (GPUHashTableView<>*)ctx->canonical_state_map;
                        auto [existing_state, was_new] = canonical_map->insert(hash, output.new_state);
                        output.canonical_state = was_new ? output.new_state : existing_state;

                        // Register child with parent (for match forwarding)
                        if (match_forwarding_enabled) {
                            register_child_with_parent(
                                task.match.source_state,
                                output.new_state,
                                task.match.matched_edges,
                                task.match.num_edges,
                                task.step,
                                state_children,
                                state_parents,
                                global_epoch,
                                child_node_pool,
                                child_node_next
                            );
                        }

                        // Online causal/branchial edge tracking
                        // Register for ALL events (not just canonical ones)
                        // Event canonicalization is only for counting unique events,
                        // not for filtering causal/branchial registration
                        if (ctx->online_causal != nullptr && output.event != INVALID_ID) {
                            OnlineCausalGraph* cg = ctx->online_causal;

                            // Register produced edges (causal: producer side)
                            for (uint8_t i = 0; i < output.num_produced; ++i) {
                                cg->register_edge_producer(output.produced_edges[i], output.event);
                            }

                            // Register consumed edges (causal: consumer side)
                            for (uint8_t i = 0; i < task.match.num_edges; ++i) {
                                EdgeId edge = task.match.matched_edges[i];
                                uint32_t producer = ctx->edges->creator_events[edge];
                                cg->register_edge_consumer(edge, output.event, producer);
                            }

                            // Register event from state (branchial tracking)
                            // Use the RAW input state for grouping (matching unified behavior)
                            // Branchial edges connect events that share the same input state
                            // and consume overlapping edges
                            cg->register_event_from_state(
                                task.match.source_state,  // Raw input state for branchial grouping
                                output.event,
                                task.match.matched_edges,
                                task.match.num_edges,
                                ctx->events
                            );
                        }

                        // Build context for child's MATCH task
                        MatchTaskWithContext mtask;
                        mtask.state_id = output.new_state;
                        mtask.step = task.step + 1;
                        mtask.context.parent_state = task.match.source_state;
                        mtask.context.num_consumed = task.match.num_edges;
                        for (uint8_t j = 0; j < task.match.num_edges; ++j) {
                            mtask.context.consumed_edges[j] = task.match.matched_edges[j];
                        }
                        mtask.context.num_produced = output.num_produced;
                        for (uint8_t j = 0; j < output.num_produced; ++j) {
                            mtask.context.produced_edges[j] = output.produced_edges[j];
                        }

                        // CRITICAL: Increment work count BEFORE push, retry with backoff
                        termination->work_created();
                        if (!match_queue->push_wait(mtask)) {
                            termination->work_finished();  // Rollback work count on failure
                        }
                        __threadfence();  // Ensure push is visible
                    }
                }
            }

            // CRITICAL: Decrement work count AFTER all child work is created
            if (lane == 0) {
                termination->work_finished();
            }
        }

        // =====================================================================
        // No work available - sleep briefly to reduce contention
        // Termination is detected automatically when work_count reaches 0
        // =====================================================================
        if (!did_work) {
            __nanosleep(100);
        }
    }

    // Debug: report if we hit the iteration limit
    if (iterations >= MAX_ITERATIONS && threadIdx.x == 0 && blockIdx.x == 0) {
        printf("WARNING: Kernel hit iteration limit! is_done=%d\n", termination->is_done());
    }
}

// =============================================================================
// HGMatch Megakernel with Generic Task Queue
// =============================================================================
// Single queue holds SCAN, EXPAND, SINK, REWRITE tasks.
// Workers pop tasks and dispatch based on task type.
// This follows the HGMatch paper's dataflow model exactly.

__global__ void hgmatch_megakernel(
    EvolutionContext* ctx,
    WorkQueueView<Task>* task_queue,
    StateMatchList* state_matches,
    StateChildrenList* state_children,
    ParentInfo* state_parents,
    uint64_t* global_epoch,
    MatchNode* match_node_pool,
    uint32_t* match_node_next,
    ChildNode* child_node_pool,
    uint32_t* child_node_next,
    TerminationDetectorView* termination,
    uint32_t max_steps,
    bool match_forwarding_enabled
) {
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;

    __shared__ Task shared_tasks[8];  // Up to 8 warps per block
    __shared__ uint32_t shared_has_work[8];

    uint32_t iterations = 0;
    const uint32_t MAX_ITERATIONS = 1000000;

    while (!termination->is_done() && iterations < MAX_ITERATIONS) {
        iterations++;
        bool did_work = false;

        // Pop a task from the generic queue
        if (lane == 0) {
            shared_has_work[warp_id] = task_queue->try_pop(&shared_tasks[warp_id]) ? 1 : 0;
        }
        __syncwarp();

        if (shared_has_work[warp_id]) {
            did_work = true;
            Task& task = shared_tasks[warp_id];

            // Dispatch based on task type
            switch (task.type) {
                case TaskType::SCAN: {
                    // SCAN: Find candidates for first pattern edge, spawn EXPAND tasks
                    if (task.step <= max_steps && lane == 0) {
                        const DeviceRewriteRule* rule = &ctx->rules[task.rule_index];
                        const uint64_t* bitmap = ctx->states->all_bitmaps + task.source_state * BITMAP_WORDS;
                        uint32_t num_edges = *ctx->edges->num_edges;

                        // Call process_scan_task which spawns EXPAND tasks to queue
                        process_scan_task(task, rule, bitmap, ctx->edges, task_queue,
                                         nullptr, nullptr, 0, num_edges);
                    }
                    break;
                }

                case TaskType::EXPAND: {
                    // EXPAND: Extend partial match, spawn EXPAND or SINK tasks
                    if (task.step <= max_steps && lane == 0) {
                        const DeviceRewriteRule* rule = &ctx->rules[task.rule_index];
                        const uint64_t* bitmap = ctx->states->all_bitmaps + task.source_state * BITMAP_WORDS;
                        uint32_t num_edges = *ctx->edges->num_edges;

                        // Process expand - may spawn more EXPAND or complete (SINK)
                        process_expand_task(task, rule, bitmap, ctx->edges, task_queue,
                                           nullptr, nullptr, 0, num_edges);
                    }
                    break;
                }

                case TaskType::SINK: {
                    // SINK: Process complete match, spawn REWRITE task
                    if (task.step <= max_steps && lane == 0) {
                        // Deduplicate match
                        MatchRecord record;
                        record.rule_index = task.rule_index;
                        record.num_edges = task.num_matched;
                        record.source_state = task.source_state;
                        record.bound_mask = task.bound_mask;
                        for (uint8_t j = 0; j < task.num_matched; ++j) {
                            record.matched_edges[j] = task.matched_edges[j];
                        }
                        for (uint8_t j = 0; j < MAX_VARS; ++j) {
                            record.bindings[j] = task.bindings[j];
                        }

                        uint64_t h = record.hash();
                        GPUHashSetView<>* seen = (GPUHashSetView<>*)ctx->seen_match_hashes;
                        if (seen->insert(h)) {
                            // Spawn REWRITE task
                            Task rewrite_task = Task::make_rewrite(
                                task.source_state, task.source_hash, task.rule_index, task.step,
                                task.bindings, task.bound_mask,
                                task.matched_edges, task.num_matched
                            );
                            termination->work_created();
                            if (!task_queue->push(rewrite_task)) {
                                termination->work_finished();
                            }
                        }
                    }
                    break;
                }

                case TaskType::REWRITE: {
                    // REWRITE: Apply rule, create new state, spawn SCAN tasks
                    if (task.step <= max_steps) {
                        DeviceMatch m;
                        m.rule_index = task.rule_index;
                        m.num_edges = task.num_matched;
                        m.source_state = task.source_state;
                        m.bound_mask = task.bound_mask;
                        for (uint8_t j = 0; j < task.num_matched; ++j) {
                            m.matched_edges[j] = task.matched_edges[j];
                        }
                        for (uint8_t j = 0; j < MAX_VARS; ++j) {
                            m.bindings[j] = task.bindings[j];
                        }

                        RewriteOutput output = apply_rewrite_warp(
                            &m, &ctx->rules[task.rule_index],
                            ctx->edges, ctx->states, ctx->events,
                            ctx->vertex_counter, task.step
                        );

                        if (output.success && lane == 0) {
                            // Compute WL hash for new state
                            uint64_t* state_bitmap = ctx->states->all_bitmaps + output.new_state * BITMAP_WORDS;

                            // Find max edge ID in bitmap using __clzll intrinsic
                            uint32_t max_edge_in_bitmap = 0;
                            for (int32_t word = BITMAP_WORDS - 1; word >= 0; --word) {
                                if (state_bitmap[word] != 0) {
                                    int leading_zeros = __clzll(state_bitmap[word]);
                                    max_edge_in_bitmap = word * 64 + (63 - leading_zeros) + 1;
                                    break;
                                }
                            }

                            uint32_t max_vertex = output.max_vertex_snapshot;
                            uint32_t num_edges = max_edge_in_bitmap;

                            uint32_t global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_id;
                            uint64_t* my_scratch = ctx->wl_scratch + global_warp_id * ctx->wl_scratch_per_block;
                            uint32_t max_scratch_vertices = ctx->wl_scratch_per_block / 2;
                            if (max_vertex > max_scratch_vertices) max_vertex = max_scratch_vertices;

                            uint64_t hash = compute_wl_hash_no_adj(state_bitmap, ctx->edges, num_edges, max_vertex, my_scratch);
                            ctx->states->states[output.new_state].canonical_hash = hash;

                            // Deduplicate by canonical hash
                            GPUHashTableView<>* canonical_map = (GPUHashTableView<>*)ctx->canonical_state_map;
                            [[maybe_unused]] auto [existing_state, was_new] = canonical_map->insert(hash, output.new_state);

                            // Spawn SCAN tasks for each rule on new state
                            uint32_t next_step = task.step + 1;
                            if (next_step <= max_steps) {
                                for (uint16_t r = 0; r < ctx->num_rules; ++r) {
                                    Task scan_task = Task::make_scan(output.new_state, hash, r, next_step);
                                    termination->work_created();
                                    if (!task_queue->push(scan_task)) {
                                        termination->work_finished();
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }

            // Work finished for this task
            if (lane == 0) {
                termination->work_finished();
            }
        }

        if (!did_work) {
            __nanosleep(100);
        }
    }
}

// =============================================================================
// GPUEvolutionEngine Implementation
// =============================================================================

GPUEvolutionEngine::GPUEvolutionEngine()
    : max_steps_(0)
    , max_states_(MAX_STATES)
    , max_events_(MAX_EVENTS)
    , tr_enabled_(false)
    , match_forwarding_enabled_(false)
    , batched_matching_(false)
    , event_canon_mode_(EventCanonicalizationMode::ByState)
    , task_granularity_(TaskGranularity::Coarse)  // Default to coarse (warp-cooperative)
    , initialized_(false)
    , d_rules_(nullptr)
    , num_rules_(0)
    , d_state_matches_(nullptr)
    , d_state_children_(nullptr)
    , d_state_parents_(nullptr)
    , d_global_epoch_(nullptr)
    , d_match_node_pool_(nullptr)
    , d_match_node_next_(nullptr)
    , d_child_node_pool_(nullptr)
    , d_child_node_next_(nullptr)
    , d_causal_edges_(nullptr)
    , d_num_causal_edges_(nullptr)
    , d_branchial_edges_(nullptr)
    , d_num_branchial_edges_(nullptr)
    , d_online_causal_(nullptr)
    , d_edge_consumer_heads_(nullptr)
    , d_consumer_nodes_(nullptr)
    , d_num_consumer_nodes_(nullptr)
    , d_state_event_heads_(nullptr)
    , d_state_event_nodes_(nullptr)
    , d_num_state_event_nodes_(nullptr)
    , d_seen_causal_triples_(nullptr)
    , d_seen_branchial_pairs_(nullptr)
    , d_seen_causal_event_pairs_(nullptr)
    , d_num_causal_event_pairs_(nullptr)
    , use_online_causal_(true)  // Enable online causal tracking by default
    , d_matches_(nullptr)
    , d_rewrite_outputs_(nullptr)
    , d_wl_scratch_(nullptr)
    , wl_scratch_per_block_(0)
    , matches_per_warp_(0)
    , stream_(0)
    , num_sms_(0)
{
    // Query device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    num_sms_ = prop.multiProcessorCount;
}

GPUEvolutionEngine::~GPUEvolutionEngine() {
    destroy_gpu_resources();
}

void GPUEvolutionEngine::add_rule(
    const std::vector<std::vector<uint8_t>>& lhs,
    const std::vector<std::vector<uint8_t>>& rhs,
    uint8_t first_fresh_var
) {
    DeviceRewriteRule rule;
    rule.num_lhs = static_cast<uint8_t>(lhs.size());
    rule.num_rhs = static_cast<uint8_t>(rhs.size());
    rule.first_fresh_var = first_fresh_var;

    for (size_t i = 0; i < lhs.size() && i < MAX_PATTERN_EDGES; ++i) {
        rule.lhs[i].arity = static_cast<uint8_t>(lhs[i].size());
        for (size_t j = 0; j < lhs[i].size() && j < MAX_EDGE_ARITY; ++j) {
            rule.lhs[i].vars[j] = lhs[i][j];
        }
    }

    for (size_t i = 0; i < rhs.size() && i < MAX_PATTERN_EDGES; ++i) {
        rule.rhs[i].arity = static_cast<uint8_t>(rhs[i].size());
        for (size_t j = 0; j < rhs[i].size() && j < MAX_EDGE_ARITY; ++j) {
            rule.rhs[i].vars[j] = rhs[i][j];
        }
    }

    h_rules_.push_back(rule);
}

void GPUEvolutionEngine::init_gpu_resources() {
    if (initialized_) return;

    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Initialize memory pools
    memory_.init();

    // Initialize adjacency (will be built after uploading initial state)
    d_adjacency_.row_offsets = nullptr;
    d_adjacency_.edge_ids = nullptr;
    d_adjacency_.positions = nullptr;
    d_adjacency_.edge_arities = nullptr;
    d_adjacency_.num_vertices = 0;
    d_adjacency_.num_entries = 0;

    // Initialize hash tables
    canonical_state_map_.init(HASH_TABLE_SIZE);
    seen_match_hashes_.init(MAX_MATCHES_PER_STEP);
    seen_event_hashes_.init(MAX_EVENTS);  // For event deduplication
    edge_producer_map_.init(MAX_EDGES);

    // Initialize work queues
    match_queue_.init(WORK_QUEUE_SIZE);
    rewrite_queue_.init(WORK_QUEUE_SIZE);
    termination_.init();

    // Initialize causal graph
    causal_graph_.init(MAX_EVENTS);
    causal_graph_.set_transitive_reduction(tr_enabled_);

    CUDA_CHECK(cudaMalloc(&d_causal_edges_, MAX_EVENTS * 4 * sizeof(CausalEdge)));
    CUDA_CHECK(cudaMalloc(&d_num_causal_edges_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_branchial_edges_, MAX_EVENTS * sizeof(BranchialEdge)));
    CUDA_CHECK(cudaMalloc(&d_num_branchial_edges_, sizeof(uint32_t)));

    // Allocate online causal/branchial tracking structures
    CUDA_CHECK(cudaMalloc(&d_online_causal_, sizeof(OnlineCausalGraph)));
    CUDA_CHECK(cudaMalloc(&d_edge_consumer_heads_, MAX_EDGES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_consumer_nodes_, MAX_CONSUMER_NODES * sizeof(ConsumerNode)));
    CUDA_CHECK(cudaMalloc(&d_num_consumer_nodes_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_state_event_heads_, MAX_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_state_event_nodes_, MAX_STATE_EVENT_NODES * sizeof(StateEventNode)));
    CUDA_CHECK(cudaMalloc(&d_num_state_event_nodes_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_seen_causal_triples_, CAUSAL_DEDUP_CAPACITY * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_seen_branchial_pairs_, BRANCHIAL_DEDUP_CAPACITY * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_seen_causal_event_pairs_, CAUSAL_PAIRS_CAPACITY * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_num_causal_event_pairs_, sizeof(uint32_t)));

    // Initialize WL hasher
    wl_hasher_.init(MAX_EDGES * MAX_EDGE_ARITY);  // Max vertices

    // Query available GPU memory to size buffers appropriately
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // Calculate total warps for per-warp allocations
    uint32_t warps_per_block = 8;  // 256 threads / 32
    uint32_t num_blocks = num_sms_ * 2;  // 2 blocks per SM
    uint32_t total_warps = num_blocks * warps_per_block;

    // Size match buffer based on available memory
    // Reserve ~50% of free memory for other allocations
    size_t available_for_matches = free_mem / 4;  // Use 25% for match buffer
    size_t match_size = sizeof(DeviceMatch);
    size_t max_matches_by_memory = available_for_matches / match_size;

    // Cap at MAX_MATCHES_PER_STEP but also ensure we don't exceed memory
    size_t actual_max_matches = std::min((size_t)MAX_MATCHES_PER_STEP, max_matches_by_memory);

    // Per-warp allocation: divide total slots among warps
    // Each warp gets equal share, minimum 64 to be useful
    uint32_t matches_per_warp = std::max(64u, (uint32_t)(actual_max_matches / total_warps));
    matches_per_warp = std::min(matches_per_warp, 1024u);  // Cap at 1024 per warp

    // Allocate scratch buffers - PER-WARP match output buffers
    size_t total_match_slots = (size_t)total_warps * matches_per_warp;
    CUDA_CHECK(cudaMalloc(&d_matches_, total_match_slots * sizeof(DeviceMatch)));
    CUDA_CHECK(cudaMalloc(&d_rewrite_outputs_, MAX_MATCHES_PER_STEP * sizeof(RewriteOutput)));

    // Store matches_per_warp for context setup
    matches_per_warp_ = matches_per_warp;

    // WL scratch: allocate per-warp to avoid races
    // Each warp needs max_vertex * 2 uint64_t values for color ping-pong
    // Use smaller scratch if memory constrained
    uint32_t max_vertices_per_warp = 512;  // Reduced from 1024
    wl_scratch_per_block_ = max_vertices_per_warp * 2;
    CUDA_CHECK(cudaMalloc(&d_wl_scratch_, total_warps * wl_scratch_per_block_ * sizeof(uint64_t)));

    // Allocate match forwarding structures
    CUDA_CHECK(cudaMalloc(&d_state_matches_, MAX_STATES * sizeof(StateMatchList)));
    CUDA_CHECK(cudaMemset(d_state_matches_, 0, MAX_STATES * sizeof(StateMatchList)));

    CUDA_CHECK(cudaMalloc(&d_state_children_, MAX_STATES * sizeof(StateChildrenList)));
    CUDA_CHECK(cudaMemset(d_state_children_, 0, MAX_STATES * sizeof(StateChildrenList)));

    CUDA_CHECK(cudaMalloc(&d_state_parents_, MAX_STATES * sizeof(ParentInfo)));
    // Initialize all parents to INVALID_ID
    std::vector<ParentInfo> h_parents(MAX_STATES);
    for (auto& p : h_parents) {
        p.parent_state = INVALID_ID;
        p.num_consumed = 0;
    }
    CUDA_CHECK(cudaMemcpy(d_state_parents_, h_parents.data(),
                          MAX_STATES * sizeof(ParentInfo), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_global_epoch_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_global_epoch_, 0, sizeof(uint64_t)));

    CUDA_CHECK(cudaMalloc(&d_match_node_pool_, MAX_MATCHES_PER_STEP * sizeof(MatchNode)));
    CUDA_CHECK(cudaMalloc(&d_match_node_next_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_match_node_next_, 0, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_child_node_pool_, MAX_STATES * sizeof(ChildNode)));
    CUDA_CHECK(cudaMalloc(&d_child_node_next_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_child_node_next_, 0, sizeof(uint32_t)));

    initialized_ = true;
}

void GPUEvolutionEngine::destroy_gpu_resources() {
    if (!initialized_) return;

    memory_.destroy();
    canonical_state_map_.destroy();
    seen_match_hashes_.destroy();
    seen_event_hashes_.destroy();
    edge_producer_map_.destroy();

    match_queue_.destroy();
    rewrite_queue_.destroy();
    termination_.destroy();

    causal_graph_.destroy();

    if (d_rules_) cudaFree(d_rules_);
    if (d_causal_edges_) cudaFree(d_causal_edges_);
    if (d_num_causal_edges_) cudaFree(d_num_causal_edges_);
    if (d_branchial_edges_) cudaFree(d_branchial_edges_);
    if (d_num_branchial_edges_) cudaFree(d_num_branchial_edges_);

    // Free online causal tracking
    if (d_online_causal_) cudaFree(d_online_causal_);
    if (d_edge_consumer_heads_) cudaFree(d_edge_consumer_heads_);
    if (d_consumer_nodes_) cudaFree(d_consumer_nodes_);
    if (d_num_consumer_nodes_) cudaFree(d_num_consumer_nodes_);
    if (d_state_event_heads_) cudaFree(d_state_event_heads_);
    if (d_state_event_nodes_) cudaFree(d_state_event_nodes_);
    if (d_num_state_event_nodes_) cudaFree(d_num_state_event_nodes_);
    if (d_seen_causal_triples_) cudaFree(d_seen_causal_triples_);
    if (d_seen_branchial_pairs_) cudaFree(d_seen_branchial_pairs_);
    if (d_seen_causal_event_pairs_) cudaFree(d_seen_causal_event_pairs_);
    if (d_num_causal_event_pairs_) cudaFree(d_num_causal_event_pairs_);

    if (d_matches_) cudaFree(d_matches_);
    if (d_rewrite_outputs_) cudaFree(d_rewrite_outputs_);
    if (d_wl_scratch_) cudaFree(d_wl_scratch_);

    // Free match forwarding structures
    if (d_state_matches_) cudaFree(d_state_matches_);
    if (d_state_children_) cudaFree(d_state_children_);
    if (d_state_parents_) cudaFree(d_state_parents_);
    if (d_global_epoch_) cudaFree(d_global_epoch_);
    if (d_match_node_pool_) cudaFree(d_match_node_pool_);
    if (d_match_node_next_) cudaFree(d_match_node_next_);
    if (d_child_node_pool_) cudaFree(d_child_node_pool_);
    if (d_child_node_next_) cudaFree(d_child_node_next_);

    // Free adjacency index
    if (d_adjacency_.row_offsets) cudaFree(d_adjacency_.row_offsets);
    if (d_adjacency_.edge_ids) cudaFree(d_adjacency_.edge_ids);
    if (d_adjacency_.positions) cudaFree(d_adjacency_.positions);
    if (d_adjacency_.edge_arities) cudaFree(d_adjacency_.edge_arities);

    // Free signature index
    if (d_sig_index_.buckets) cudaFree(d_sig_index_.buckets);
    if (d_sig_index_.edge_ids) cudaFree(d_sig_index_.edge_ids);
    if (d_inv_vertex_index_.row_offsets) cudaFree(d_inv_vertex_index_.row_offsets);
    if (d_inv_vertex_index_.edge_ids) cudaFree(d_inv_vertex_index_.edge_ids);
    sig_index_builder_.destroy();

    wl_hasher_.destroy();

    if (stream_) cudaStreamDestroy(stream_);

    initialized_ = false;
}

void GPUEvolutionEngine::upload_rules() {
    if (h_rules_.empty()) return;

    num_rules_ = static_cast<uint32_t>(h_rules_.size());
    CUDA_CHECK(cudaMalloc(&d_rules_, num_rules_ * sizeof(DeviceRewriteRule)));
    CUDA_CHECK(cudaMemcpy(d_rules_, h_rules_.data(),
                          num_rules_ * sizeof(DeviceRewriteRule),
                          cudaMemcpyHostToDevice));
}

StateId GPUEvolutionEngine::upload_initial_state(
    const std::vector<std::vector<uint32_t>>& edges
) {
    // Track max vertex ID
    uint32_t max_vertex = 0;
    for (const auto& edge : edges) {
        for (uint32_t v : edge) {
            max_vertex = std::max(max_vertex, v + 1);
        }
    }

    // Upload vertex counter
    memory_.vertex_allocator.set(max_vertex);

    // Create edges
    uint32_t num_edges = static_cast<uint32_t>(edges.size());
    std::vector<uint32_t> h_offsets(num_edges + 1);
    std::vector<uint32_t> h_vertices;
    std::vector<uint8_t> h_arities(num_edges);

    uint32_t offset = 0;
    for (size_t i = 0; i < edges.size(); ++i) {
        h_offsets[i] = offset;
        h_arities[i] = static_cast<uint8_t>(edges[i].size());
        for (uint32_t v : edges[i]) {
            h_vertices.push_back(v);
        }
        offset += edges[i].size();
    }
    h_offsets[num_edges] = offset;

    // Upload edge data
    CUDA_CHECK(cudaMemcpy(memory_.edge_offsets, h_offsets.data(),
                          (num_edges + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(memory_.vertex_data.device_ptr(), h_vertices.data(),
                          h_vertices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(memory_.edge_arities, h_arities.data(),
                          num_edges * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(memory_.num_edges, &num_edges, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // CRITICAL: Update the vertex_data counter to reflect initial edges
    // This prevents new edges from overwriting initial vertex data
    uint32_t initial_vertex_data_size = static_cast<uint32_t>(h_vertices.size());
    CUDA_CHECK(cudaMemcpy(memory_.vertex_data.counter_ptr(), &initial_vertex_data_size,
                          sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Set up d_edges_ pointers
    d_edges_.vertex_offsets = memory_.edge_offsets;
    d_edges_.vertex_data = memory_.vertex_data.device_ptr();
    d_edges_.arities = memory_.edge_arities;
    d_edges_.creator_events = memory_.edge_creators;
    d_edges_.num_edges = memory_.num_edges;       // Device counter pointer
    d_edges_.num_vertex_data = memory_.vertex_data.counter_ptr();  // Device counter for vertex_data
    d_edges_.cached_num_edges = num_edges;        // Cached value for read-only access

    // Create initial state
    uint32_t state_idx = memory_.state_pool.host_alloc();  // Allocate slot 0

    // Build bitmap
    std::vector<uint64_t> h_bitmap(BITMAP_WORDS, 0);
    for (uint32_t eid = 0; eid < num_edges; ++eid) {
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        h_bitmap[word] |= (1ULL << bit);
    }

    CUDA_CHECK(cudaMemcpy(memory_.bitmap_pool.device_ptr(), h_bitmap.data(),
                          BITMAP_WORDS * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Compute WL canonical hash for initial state
    // This must match the hash computed in the megakernel for child states
    uint64_t canonical_hash = wl_hasher_.compute_hash_no_adj(
        memory_.bitmap_pool.device_ptr(),  // d_state_bitmap
        d_edges_,                           // edges (BY VALUE, contains device pointers)
        max_vertex,                         // max_vertex
        num_edges,                          // num_edges
        stream_                             // stream
    );

    // Set up d_state_pool_ pointers (using device counters)
    d_state_pool_.states = memory_.state_pool.device_ptr();
    d_state_pool_.all_bitmaps = memory_.bitmap_pool.device_ptr();
    d_state_pool_.num_states = memory_.d_num_states;

    // Set up d_event_pool_ pointers
    d_event_pool_.events = memory_.event_pool.device_ptr();
    d_event_pool_.consumed_edges = memory_.consumed_edges.device_ptr();
    d_event_pool_.produced_edges = memory_.produced_edges.device_ptr();
    d_event_pool_.num_events = memory_.d_num_events;
    d_event_pool_.consumed_offset = memory_.d_consumed_offset;
    d_event_pool_.produced_offset = memory_.d_produced_offset;

    // Set initial state count to 1
    memory_.set_num_states(1);

    // Build adjacency index for pattern matching acceleration
    build_adjacency_index(edges, max_vertex);

    // Set up state metadata
    DeviceState h_state;
    h_state.bitmap = memory_.bitmap_pool.device_ptr();
    h_state.canonical_hash = canonical_hash;
    h_state.parent_state = INVALID_ID;
    h_state.parent_event = INVALID_ID;
    h_state.step = 0;
    h_state.edge_count = num_edges;

    CUDA_CHECK(cudaMemcpy(memory_.state_pool.device_ptr(), &h_state,
                          sizeof(DeviceState), cudaMemcpyHostToDevice));

    // Register initial state in canonical_state_map (from host, need to use host_insert)
    canonical_state_map_.host_insert(canonical_hash, state_idx);

    // Build signature index for faster pattern matching
    sig_index_builder_.init(num_edges, max_vertex);
    sig_index_builder_.build(edges, d_sig_index_, d_inv_vertex_index_);

    return state_idx;
}

void GPUEvolutionEngine::build_adjacency_index(
    const std::vector<std::vector<uint32_t>>& edges,
    uint32_t num_vertices
) {
    // Count entries per vertex (sum of arities for edges containing that vertex)
    std::vector<uint32_t> h_counts(num_vertices + 1, 0);
    uint32_t total_entries = 0;

    for (size_t eid = 0; eid < edges.size(); ++eid) {
        const auto& edge = edges[eid];
        for (size_t pos = 0; pos < edge.size(); ++pos) {
            uint32_t v = edge[pos];
            h_counts[v]++;
            total_entries++;
        }
    }

    // Build row offsets (prefix sum)
    std::vector<uint32_t> h_row_offsets(num_vertices + 2, 0);
    h_row_offsets[0] = 0;
    for (uint32_t v = 0; v < num_vertices; ++v) {
        h_row_offsets[v + 1] = h_row_offsets[v] + h_counts[v];
    }
    h_row_offsets[num_vertices + 1] = total_entries;

    // Build edge_ids and positions arrays
    std::vector<uint32_t> h_edge_ids(total_entries);
    std::vector<uint8_t> h_positions(total_entries);
    std::vector<uint8_t> h_edge_arities(total_entries);

    // Reset counts to use as current write positions
    std::fill(h_counts.begin(), h_counts.end(), 0);

    for (size_t eid = 0; eid < edges.size(); ++eid) {
        const auto& edge = edges[eid];
        uint8_t arity = static_cast<uint8_t>(edge.size());
        for (size_t pos = 0; pos < edge.size(); ++pos) {
            uint32_t v = edge[pos];
            uint32_t offset = h_row_offsets[v] + h_counts[v];
            h_edge_ids[offset] = static_cast<uint32_t>(eid);
            h_positions[offset] = static_cast<uint8_t>(pos);
            h_edge_arities[offset] = arity;
            h_counts[v]++;
        }
    }

    // Allocate and upload to device
    CUDA_CHECK(cudaMalloc(&d_adjacency_.row_offsets, (num_vertices + 2) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_adjacency_.row_offsets, h_row_offsets.data(),
                          (num_vertices + 2) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    if (total_entries > 0) {
        CUDA_CHECK(cudaMalloc(&d_adjacency_.edge_ids, total_entries * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_adjacency_.edge_ids, h_edge_ids.data(),
                              total_entries * sizeof(uint32_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_adjacency_.positions, total_entries * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_adjacency_.positions, h_positions.data(),
                              total_entries * sizeof(uint8_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_adjacency_.edge_arities, total_entries * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_adjacency_.edge_arities, h_edge_arities.data(),
                              total_entries * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    d_adjacency_.num_vertices = num_vertices;
    d_adjacency_.num_entries = total_entries;
}

void GPUEvolutionEngine::evolve(
    const std::vector<std::vector<uint32_t>>& initial_edges,
    uint32_t steps
) {
    max_steps_ = steps;

    init_gpu_resources();
    upload_rules();
    StateId initial_state = upload_initial_state(initial_edges);

    // Clear counters
    CUDA_CHECK(cudaMemset(d_num_causal_edges_, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_num_branchial_edges_, 0, sizeof(uint32_t)));

    // Initialize online causal tracking structures
    if (use_online_causal_) {
        CUDA_CHECK(cudaMemset(d_num_consumer_nodes_, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_num_state_event_nodes_, 0, sizeof(uint32_t)));
        // Initialize list heads to INVALID_ID (all 1s = 0xFFFFFFFF)
        CUDA_CHECK(cudaMemset(d_edge_consumer_heads_, 0xFF, MAX_EDGES * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_state_event_heads_, 0xFF, MAX_STATES * sizeof(uint32_t)));
        // Initialize dedup hash sets to empty (all 0xFF)
        CUDA_CHECK(cudaMemset(d_seen_causal_triples_, 0xFF, CAUSAL_DEDUP_CAPACITY * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_seen_branchial_pairs_, 0xFF, BRANCHIAL_DEDUP_CAPACITY * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_seen_causal_event_pairs_, 0xFF, CAUSAL_PAIRS_CAPACITY * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_num_causal_event_pairs_, 0, sizeof(uint32_t)));

        // Copy OnlineCausalGraph structure to device
        OnlineCausalGraph h_online_causal;
        h_online_causal.edge_consumer_heads = d_edge_consumer_heads_;
        h_online_causal.consumer_nodes = d_consumer_nodes_;
        h_online_causal.num_consumer_nodes = d_num_consumer_nodes_;
        h_online_causal.state_event_heads = d_state_event_heads_;
        h_online_causal.state_event_nodes = d_state_event_nodes_;
        h_online_causal.num_state_event_nodes = d_num_state_event_nodes_;
        h_online_causal.causal_edges = d_causal_edges_;
        h_online_causal.num_causal_edges = d_num_causal_edges_;
        h_online_causal.branchial_edges = d_branchial_edges_;
        h_online_causal.num_branchial_edges = d_num_branchial_edges_;
        h_online_causal.seen_causal_triples = d_seen_causal_triples_;
        h_online_causal.seen_causal_capacity = CAUSAL_DEDUP_CAPACITY;
        h_online_causal.seen_causal_mask = CAUSAL_DEDUP_CAPACITY - 1;
        h_online_causal.seen_branchial_pairs = d_seen_branchial_pairs_;
        h_online_causal.seen_branchial_capacity = BRANCHIAL_DEDUP_CAPACITY;
        h_online_causal.seen_branchial_mask = BRANCHIAL_DEDUP_CAPACITY - 1;
        h_online_causal.seen_causal_event_pairs = d_seen_causal_event_pairs_;
        h_online_causal.seen_causal_pairs_capacity = CAUSAL_PAIRS_CAPACITY;
        h_online_causal.seen_causal_pairs_mask = CAUSAL_PAIRS_CAPACITY - 1;
        h_online_causal.num_causal_event_pairs = d_num_causal_event_pairs_;
        CUDA_CHECK(cudaMemcpy(d_online_causal_, &h_online_causal, sizeof(OnlineCausalGraph), cudaMemcpyHostToDevice));
    }

    // Push initial state to match queue with empty context (no parent)
    MatchTaskWithContext initial_task;
    initial_task.state_id = initial_state;
    initial_task.step = 1;
    initial_task.context.parent_state = INVALID_ID;
    initial_task.context.num_consumed = 0;
    initial_task.context.num_produced = 0;

    // CRITICAL: Increment work count BEFORE pushing to queue
    termination_.host_work_created();
    match_queue_.host_push(initial_task);

    // Run evolution using megakernel with match forwarding
    run_megakernel_evolution();

    // Compute causal/branchial edges
    // If online tracking is enabled, edges are computed incrementally during rewriting
    // Otherwise, use phased (post-hoc) computation
    if (!use_online_causal_) {
        uint32_t num_events = memory_.get_num_events();
        uint32_t num_states = memory_.get_num_states();

        // Pass structs BY VALUE - they contain device pointers
        causal_graph_.compute_causal_edges(
            d_event_pool_, num_events, edge_producer_map_,
            d_causal_edges_, d_num_causal_edges_, stream_
        );

        causal_graph_.compute_branchial_edges(
            d_event_pool_, num_events, num_states,
            d_branchial_edges_, d_num_branchial_edges_, stream_
        );
    }
    // Note: If use_online_causal_ is true, causal/branchial edges were computed
    // incrementally during apply_rewrite_warp() calls

    download_results();
}

void GPUEvolutionEngine::run_megakernel_evolution() {
    // Allocate device copies of hash table views
    // These are POD structs with device pointers inside, so can be safely copied
    auto canonical_view = canonical_state_map_.get_view();
    auto match_hash_view = seen_match_hashes_.get_view();
    auto event_hash_view = seen_event_hashes_.get_view();

    GPUHashTableView<>* d_canonical_view;
    GPUHashSetView<>* d_match_hash_view;
    GPUHashSetView<>* d_event_hash_view;

    CUDA_CHECK(cudaMalloc(&d_canonical_view, sizeof(GPUHashTableView<>)));
    CUDA_CHECK(cudaMalloc(&d_match_hash_view, sizeof(GPUHashSetView<>)));
    CUDA_CHECK(cudaMalloc(&d_event_hash_view, sizeof(GPUHashSetView<>)));

    CUDA_CHECK(cudaMemcpy(d_canonical_view, &canonical_view, sizeof(GPUHashTableView<>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_match_hash_view, &match_hash_view, sizeof(GPUHashSetView<>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_event_hash_view, &event_hash_view, sizeof(GPUHashSetView<>), cudaMemcpyHostToDevice));

    // Allocate device copies of data structures (POD structs with device pointers)
    DeviceEdges* d_edges_ptr;
    StatePool* d_state_pool_ptr;
    EventPool* d_event_pool_ptr;
    DeviceAdjacency* d_adjacency_ptr;

    CUDA_CHECK(cudaMalloc(&d_edges_ptr, sizeof(DeviceEdges)));
    CUDA_CHECK(cudaMalloc(&d_state_pool_ptr, sizeof(StatePool)));
    CUDA_CHECK(cudaMalloc(&d_event_pool_ptr, sizeof(EventPool)));
    CUDA_CHECK(cudaMalloc(&d_adjacency_ptr, sizeof(DeviceAdjacency)));

    CUDA_CHECK(cudaMemcpy(d_edges_ptr, &d_edges_, sizeof(DeviceEdges), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state_pool_ptr, &d_state_pool_, sizeof(StatePool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_event_pool_ptr, &d_event_pool_, sizeof(EventPool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacency_ptr, &d_adjacency_, sizeof(DeviceAdjacency), cudaMemcpyHostToDevice));

    // Allocate device copies of work queue and termination detector views
    auto match_queue_view = match_queue_.get_view();
    auto rewrite_queue_view = rewrite_queue_.get_view();
    auto termination_view = termination_.get_view();

    WorkQueueView<MatchTaskWithContext>* d_match_queue_view;
    WorkQueueView<RewriteTaskWithMatch>* d_rewrite_queue_view;
    TerminationDetectorView* d_termination_view;

    CUDA_CHECK(cudaMalloc(&d_match_queue_view, sizeof(WorkQueueView<MatchTaskWithContext>)));
    CUDA_CHECK(cudaMalloc(&d_rewrite_queue_view, sizeof(WorkQueueView<RewriteTaskWithMatch>)));
    CUDA_CHECK(cudaMalloc(&d_termination_view, sizeof(TerminationDetectorView)));

    CUDA_CHECK(cudaMemcpy(d_match_queue_view, &match_queue_view, sizeof(WorkQueueView<MatchTaskWithContext>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rewrite_queue_view, &rewrite_queue_view, sizeof(WorkQueueView<RewriteTaskWithMatch>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_termination_view, &termination_view, sizeof(TerminationDetectorView), cudaMemcpyHostToDevice));

    // Build context with device pointers
    EvolutionContext h_ctx;
    h_ctx.edges = d_edges_ptr;
    h_ctx.states = d_state_pool_ptr;
    h_ctx.events = d_event_pool_ptr;
    h_ctx.adjacency = d_adjacency_ptr;
    h_ctx.rules = d_rules_;
    h_ctx.num_rules = num_rules_;
    h_ctx.wl_scratch = d_wl_scratch_;  // WL hash scratch space
    h_ctx.wl_scratch_per_block = wl_scratch_per_block_;
    h_ctx.matches_buffer = d_matches_;  // Per-warp match output buffer
    h_ctx.matches_per_warp = matches_per_warp_;  // Determined at init based on available memory
    h_ctx.canonical_state_map = d_canonical_view;
    h_ctx.seen_match_hashes = d_match_hash_view;
    h_ctx.seen_event_hashes = d_event_hash_view;
    h_ctx.event_canon_mode = event_canon_mode_;
    h_ctx.online_causal = use_online_causal_ ? d_online_causal_ : nullptr;
    h_ctx.causal_edges = d_causal_edges_;
    h_ctx.num_causal_edges = d_num_causal_edges_;
    h_ctx.branchial_edges = d_branchial_edges_;
    h_ctx.num_branchial_edges = d_num_branchial_edges_;
    h_ctx.edge_producer_map = edge_producer_map_.device_ptr();
    h_ctx.vertex_counter = memory_.vertex_allocator.device_ptr();
    h_ctx.max_steps = max_steps_;
    h_ctx.max_states = max_states_;
    h_ctx.max_events = max_events_;
    h_ctx.transitive_reduction_enabled = tr_enabled_;
    h_ctx.task_granularity = task_granularity_;

    EvolutionContext* d_ctx;
    CUDA_CHECK(cudaMalloc(&d_ctx, sizeof(EvolutionContext)));
    CUDA_CHECK(cudaMemcpy(d_ctx, &h_ctx, sizeof(EvolutionContext), cudaMemcpyHostToDevice));

    // Launch megakernel with enough blocks to saturate GPU
    // DEBUG: 1 block, multiple warps
    int num_blocks = 1;
    int block_size = 256;

    // Set larger stack size for complex megakernel (default is often too small)
    size_t stack_size = 8192;  // 8KB per thread
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));

    evolution_megakernel_with_forwarding<<<num_blocks, block_size, 0, stream_>>>(
        d_ctx,
        d_match_queue_view,
        d_rewrite_queue_view,
        d_state_matches_,
        d_state_children_,
        d_state_parents_,
        d_global_epoch_,
        d_match_node_pool_,
        d_match_node_next_,
        d_child_node_pool_,
        d_child_node_next_,
        d_termination_view,
        max_steps_,
        match_forwarding_enabled_,
        batched_matching_
    );

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Clean up device allocations
    cudaFree(d_ctx);
    cudaFree(d_canonical_view);
    cudaFree(d_match_hash_view);
    cudaFree(d_event_hash_view);
    cudaFree(d_edges_ptr);
    cudaFree(d_state_pool_ptr);
    cudaFree(d_event_pool_ptr);
    cudaFree(d_adjacency_ptr);
    cudaFree(d_match_queue_view);
    cudaFree(d_rewrite_queue_view);
    cudaFree(d_termination_view);
}

void GPUEvolutionEngine::download_results() {
    results_.num_states = memory_.get_num_states();
    results_.num_canonical_states = canonical_state_map_.get_size();

    // Download raw event count
    uint32_t raw_events = memory_.get_num_events();

    // Compute canonical event count based on event canonicalization mode
    if (event_canon_mode_ == EventCanonicalizationMode::None) {
        results_.num_events = raw_events;
    } else {
        // Download events and states to compute canonical event count
        std::vector<DeviceEvent> h_events(raw_events);
        std::vector<DeviceState> h_states(results_.num_states);

        CUDA_CHECK(cudaMemcpy(h_events.data(), d_event_pool_.events,
                              raw_events * sizeof(DeviceEvent), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_states.data(), d_state_pool_.states,
                              results_.num_states * sizeof(DeviceState), cudaMemcpyDeviceToHost));

        // Build hash -> canonical state ID mapping (first state seen with each hash)
        std::unordered_map<uint64_t, uint32_t> hash_to_canonical;
        for (uint32_t sid = 0; sid < results_.num_states; ++sid) {
            uint64_t hash = h_states[sid].canonical_hash;
            if (hash_to_canonical.find(hash) == hash_to_canonical.end()) {
                hash_to_canonical[hash] = sid;
            }
        }

        // Compute canonical event signatures and count unique ones
        // Use canonical STATE IDs (first state seen with each hash), matching unified's approach
        std::unordered_set<uint64_t> unique_event_sigs;

        for (uint32_t eid = 0; eid < raw_events; ++eid) {
            const auto& event = h_events[eid];

            // Get canonical hashes for input and output states
            uint64_t input_hash = h_states[event.input_state].canonical_hash;
            uint64_t output_hash = h_states[event.output_state].canonical_hash;

            // Get canonical state IDs (first state seen with each hash)
            uint32_t canonical_input_id = hash_to_canonical[input_hash];
            uint32_t canonical_output_id = hash_to_canonical[output_hash];

            // Compute event signature based on mode
            // IMPORTANT: Include rule_index - events from different rules are never equivalent
            uint64_t sig;
            if (event_canon_mode_ == EventCanonicalizationMode::ByState) {
                // Hash: rule_index, canonical_input_id, canonical_output_id
                // Using FNV-1a to combine all three components
                uint64_t h = 14695981039346656037ULL;
                h ^= event.rule_index;
                h *= 1099511628211ULL;
                h ^= canonical_input_id;
                h *= 1099511628211ULL;
                h ^= canonical_output_id;
                h *= 1099511628211ULL;
                sig = h;
            } else {
                // ByStateAndEdges - would need edge correspondence computation
                // For now, fall back to ByState with rule_index
                uint64_t h = 14695981039346656037ULL;
                h ^= event.rule_index;
                h *= 1099511628211ULL;
                h ^= canonical_input_id;
                h *= 1099511628211ULL;
                h ^= canonical_output_id;
                h *= 1099511628211ULL;
                sig = h;
            }

            unique_event_sigs.insert(sig);
        }

        results_.num_events = unique_event_sigs.size();
    }

    // Use temp variables to avoid size_t/uint32_t mismatch
    uint32_t causal_pair_count = 0, branchial_count = 0;

    // Use num_causal_event_pairs for v1 compatibility (unique event pairs, not triples)
    if (use_online_causal_ && d_num_causal_event_pairs_) {
        CUDA_CHECK(cudaMemcpy(&causal_pair_count, d_num_causal_event_pairs_,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
    } else {
        // Fallback to num_causal_edges if online tracking not used
        CUDA_CHECK(cudaMemcpy(&causal_pair_count, d_num_causal_edges_,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(&branchial_count, d_num_branchial_edges_,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    results_.num_causal_edges = causal_pair_count;
    results_.num_branchial_edges = branchial_count;

    // Note: Actual edges are downloaded on-demand via get_causal_edges() / get_branchial_edges()
    // to minimize transfer costs when only counts are needed.

    results_.num_redundant_edges_skipped = causal_graph_.get_redundant_count();
}

std::vector<HostCausalEdge> GPUEvolutionEngine::get_causal_edges() const {
    uint32_t count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_num_causal_edges_, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::vector<HostCausalEdge> edges;
    if (count > 0 && d_causal_edges_) {
        edges.resize(count);
        // CausalEdge and HostCausalEdge have identical layout
        CUDA_CHECK(cudaMemcpy(edges.data(), d_causal_edges_,
                              count * sizeof(CausalEdge), cudaMemcpyDeviceToHost));
    }
    return edges;
}

std::vector<HostBranchialEdge> GPUEvolutionEngine::get_branchial_edges() const {
    uint32_t count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_num_branchial_edges_, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::vector<HostBranchialEdge> edges;
    if (count > 0 && d_branchial_edges_) {
        edges.resize(count);
        // BranchialEdge and HostBranchialEdge have identical layout
        CUDA_CHECK(cudaMemcpy(edges.data(), d_branchial_edges_,
                              count * sizeof(BranchialEdge), cudaMemcpyDeviceToHost));
    }
    return edges;
}

EvolutionResults GPUEvolutionEngine::get_results() const {
    return results_;
}

}  // namespace hypergraph::gpu
