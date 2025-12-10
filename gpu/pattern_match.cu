#include "pattern_match.cuh"
#include "types.cuh"
#include "work_queue.cuh"
#include <cub/cub.cuh>

namespace hypergraph::gpu {

// =============================================================================
// HGMatch Dataflow Model - GPU Implementation
// =============================================================================
//
// Following the HGMatch paper exactly:
//
// ONE GENERIC TASK QUEUE holds all task types (SCAN, EXPAND, SINK, REWRITE).
// Workers pop tasks and dispatch based on task type.
//
// TSCAN task: Takes a partition of data hyperedges, spawns TEXPAND tasks
//             for EACH hyperedge that matches the first pattern edge's signature.
//
// TEXPAND task: Takes a partial embedding, generates candidates for the next
//               pattern edge, spawns TEXPAND or TSINK task for EACH expanded
//               partial embedding.
//
// TSINK task: Takes a complete embedding, spawns TREWRITE task.
//
// TREWRITE task: Applies the rewrite rule, creates new state, spawns TSCAN
//                tasks for the new state.
//
// Key insight: All tasks go to ONE queue. Any available thread/warp can pick
// up any task type. This enables GPU-wide parallelism and load balancing.
//
// The Task type (with TaskType enum) is defined in types.cuh

// =============================================================================
// Device Helper Functions
// =============================================================================

// Check if edge is in state
__device__ __forceinline__ bool edge_in_state(EdgeId eid, const uint64_t* bitmap) {
    uint32_t word = eid / 64;
    uint32_t bit = eid % 64;
    return (bitmap[word] & (1ULL << bit)) != 0;
}

// Try to match a pattern edge against a graph edge with current bindings
__device__ bool try_match_edge(
    const DevicePatternEdge* pattern,
    EdgeId graph_edge,
    const DeviceEdges* edges,
    DeviceMatch* match
) {
    // Check arity matches
    uint8_t graph_arity = edges->arities[graph_edge];
    if (graph_arity != pattern->arity) return false;

    // Try to extend bindings
    uint32_t new_bindings[MAX_EDGE_ARITY];
    uint32_t new_bound_mask = match->bound_mask;

    for (uint8_t pos = 0; pos < pattern->arity; ++pos) {
        uint8_t var = pattern->vars[pos];
        VertexId graph_vertex = edges->get_vertex(graph_edge, pos);

        if (match->is_bound(var)) {
            // Variable already bound - must match
            if (match->bindings[var] != graph_vertex) {
                return false;
            }
        } else {
            // New binding
            new_bindings[pos] = graph_vertex;
            new_bound_mask |= (1u << var);
        }
    }

    // All positions match - apply new bindings
    for (uint8_t pos = 0; pos < pattern->arity; ++pos) {
        uint8_t var = pattern->vars[pos];
        if (!match->is_bound(var)) {
            match->bindings[var] = new_bindings[pos];
        }
    }
    match->bound_mask = new_bound_mask;

    return true;
}

// Find candidate edges for a pattern edge based on current bindings
__device__ uint32_t find_edge_candidates(
    const DevicePatternEdge* pattern,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    const DeviceAdjacency* adj,
    const DeviceMatch* match,
    EdgeId* candidates,
    uint32_t max_candidates,
    uint32_t num_edges
) {
    uint32_t count = 0;

    // Check if adjacency index is available (row_offsets != nullptr)
    bool adj_available = (adj != nullptr && adj->row_offsets != nullptr);

    // If we have any bound variables and adjacency is available, use it
    bool has_bound = false;
    uint8_t first_bound_var = 0;
    uint8_t first_bound_pos = 0;

    for (uint8_t pos = 0; pos < pattern->arity; ++pos) {
        uint8_t var = pattern->vars[pos];
        if (match->is_bound(var)) {
            has_bound = true;
            first_bound_var = var;
            first_bound_pos = pos;
            break;
        }
    }

    if (has_bound && adj_available) {
        // Use adjacency index from bound vertex
        VertexId bound_vertex = match->bindings[first_bound_var];

        // Bounds check for adjacency
        if (bound_vertex < adj->num_vertices) {
            uint32_t start = adj->row_offsets[bound_vertex];
            uint32_t end = adj->row_offsets[bound_vertex + 1];

            for (uint32_t i = start; i < end && count < max_candidates; ++i) {
                EdgeId eid = adj->edge_ids[i];
                uint8_t pos = adj->positions[i];

                // Check position matches where we expect bound variable
                if (pos != first_bound_pos) continue;

                // Check edge is in state and has right arity
                if (!edge_in_state(eid, state_bitmap)) continue;
                if (edges->arities[eid] != pattern->arity) continue;

                candidates[count++] = eid;
            }
        }
        // If bound vertex out of range, fall through to full scan
        if (count > 0) return count;
    }

    // Fallback: scan all edges in state with matching arity
    for (uint32_t eid = 0; eid < num_edges && count < max_candidates; ++eid) {
        if (!edge_in_state(eid, state_bitmap)) continue;
        if (edges->arities[eid] != pattern->arity) continue;
        candidates[count++] = eid;
    }

    return count;
}

// Find candidate edges using signature index (faster for first/anchor edge)
__device__ uint32_t find_candidates_by_signature(
    const DevicePatternEdge* pattern,
    const uint64_t* state_bitmap,
    const DeviceSignatureIndex* sig_index,
    EdgeId* candidates,
    uint32_t max_candidates
) {
    if (sig_index == nullptr || sig_index->buckets == nullptr) {
        return 0;  // No signature index available
    }

    // Compute pattern signature
    DeviceEdgeSignature pattern_sig = DeviceEdgeSignature::from_pattern(
        pattern->vars, pattern->arity
    );
    uint64_t sig_hash = pattern_sig.hash();

    // Look up in signature index
    return sig_index->get_edges_for_signature(
        sig_hash, state_bitmap, candidates, max_candidates
    );
}

// Find candidate edges using inverted vertex index (when variable is bound)
__device__ uint32_t find_candidates_by_vertex(
    VertexId vertex,
    uint8_t position,
    uint8_t arity,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    const DeviceInvertedVertexIndex* inv_index,
    EdgeId* candidates,
    uint32_t max_candidates
) {
    if (inv_index == nullptr || inv_index->row_offsets == nullptr) {
        return 0;  // No inverted index available
    }

    if (vertex >= inv_index->num_vertices) {
        return 0;
    }

    uint32_t start = inv_index->row_offsets[vertex];
    uint32_t end = inv_index->row_offsets[vertex + 1];
    uint32_t count = 0;

    for (uint32_t i = start; i < end && count < max_candidates; ++i) {
        EdgeId eid = inv_index->edge_ids[i];

        // Check if edge is in state
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        if (!(state_bitmap[word] & (1ULL << bit))) continue;

        // Check arity matches
        if (edges->arities[eid] != arity) continue;

        // Check vertex is at expected position
        if (edges->get_vertex(eid, position) != vertex) continue;

        candidates[count++] = eid;
    }

    return count;
}

// Helper: Try to match a single pattern edge and update bindings
// Returns true if match succeeds, bindings updated in-place
__device__ bool try_match_single_edge(
    const DevicePatternEdge* pattern,
    EdgeId graph_edge,
    const DeviceEdges* edges,
    uint32_t* bindings,
    uint32_t* bound_mask
) {
    uint8_t graph_arity = edges->arities[graph_edge];
    if (graph_arity != pattern->arity) return false;

    // Check all positions
    for (uint8_t pos = 0; pos < pattern->arity; ++pos) {
        uint8_t var = pattern->vars[pos];
        VertexId graph_vertex = edges->get_vertex(graph_edge, pos);

        if (*bound_mask & (1u << var)) {
            // Variable already bound - must match
            if (bindings[var] != graph_vertex) {
                return false;
            }
        }
    }

    // Success - bind unbound variables
    for (uint8_t pos = 0; pos < pattern->arity; ++pos) {
        uint8_t var = pattern->vars[pos];
        if (!(*bound_mask & (1u << var))) {
            bindings[var] = edges->get_vertex(graph_edge, pos);
            *bound_mask |= (1u << var);
        }
    }

    return true;
}

// =============================================================================
// HGMatch-Style Pattern Matching (following unified/pattern_matcher.hpp)
// =============================================================================
// Key approach from unified/ and HGMatch paper:
// - SCAN: Get initial candidates via signature (arity) partition
// - EXPAND: For each candidate, validate binding, then recurse/iterate forward
// - No backtracking - just forward iteration with set operations
// - Use inverted index intersection when variables are bound
// - validate_candidate checks binding consistency and extends it

// Validate candidate edge against pattern edge, extending binding
// Returns true if validation succeeds, binding is modified in place
// This is the core matching logic from unified/pattern_matcher.hpp
__device__ bool validate_candidate_edge(
    const DevicePatternEdge* pattern,
    EdgeId graph_edge,
    const DeviceEdges* edges,
    uint32_t* bindings,
    uint32_t* bound_mask
) {
    uint8_t edge_arity = edges->arities[graph_edge];
    if (edge_arity != pattern->arity) return false;

    // Check and extend bindings
    for (uint8_t i = 0; i < edge_arity; ++i) {
        VertexId actual = edges->get_vertex(graph_edge, i);
        uint8_t var = pattern->vars[i];

        if (*bound_mask & (1u << var)) {
            // Variable already bound - must match
            if (bindings[var] != actual) {
                return false;
            }
        } else {
            // Bind new variable
            bindings[var] = actual;
            *bound_mask |= (1u << var);
        }
    }
    return true;
}

// Check if edge is already in matched set
__device__ bool edge_already_matched(EdgeId eid, const EdgeId* matched, uint8_t num_matched) {
    for (uint8_t i = 0; i < num_matched; ++i) {
        if (matched[i] == eid) return true;
    }
    return false;
}

// Emit a complete match
// Note: This is called single-threaded (from lane 0 only), so no atomics needed
// for num_matches. The output buffer is per-warp, so no contention.
__device__ void emit_complete_match(
    const EdgeId* matched_edges,
    uint8_t num_edges,
    const uint32_t* bindings,
    uint32_t bound_mask,
    StateId source_state,
    uint64_t source_hash,
    uint16_t rule_index,
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches
) {
    if (*num_matches >= max_matches) return;

    DeviceMatch match;
    match.bound_mask = bound_mask;
    match.num_edges = num_edges;
    match.source_state = source_state;
    match.source_hash = source_hash;
    match.rule_index = rule_index;

    for (uint8_t i = 0; i < num_edges; ++i) {
        match.matched_edges[i] = matched_edges[i];
    }
    for (uint8_t i = 0; i < MAX_VARS; ++i) {
        match.bindings[i] = bindings[i];
    }

    output[*num_matches] = match;
    ++(*num_matches);
}

// =============================================================================
// TEXPAND Task Processing (HGMatch Dataflow Model)
// =============================================================================
// EXPAND is a task that processes partial matches.
// It spawns NEW EXPAND tasks to a GLOBAL work queue.
// Any available warp can pick up EXPAND tasks from the queue.
//
// This function processes ONE ExpandTask:
// - Generate candidates for the next pattern edge
// - For each valid candidate, either:
//   - Output complete match (SINK)
//   - Push new EXPAND task to global queue

__device__ void process_expand_task(
    const Task& task,
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    WorkQueueView<Task>* task_queue,  // GENERIC task queue
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    // Check if complete (shouldn't happen - complete matches go to SINK)
    if (task.num_matched >= rule->num_lhs) {
        emit_complete_match(task.matched_edges, task.num_matched,
                           task.bindings, task.bound_mask,
                           task.source_state, task.source_hash, task.rule_index,
                           output, num_matches, max_matches);
        return;
    }

    if (*num_matches >= max_matches) return;

    // Get next pattern edge to match
    uint8_t pattern_idx = task.num_matched;
    const DevicePatternEdge* pattern = &rule->lhs[pattern_idx];

    // Generate candidates - iterate edges with matching arity (signature)
    for (uint32_t eid = 0; eid < num_edges; ++eid) {
        if (*num_matches >= max_matches) return;

        // Skip if not in state
        if (!edge_in_state(eid, state_bitmap)) continue;

        // Skip if arity doesn't match (signature filtering)
        if (edges->arities[eid] != pattern->arity) continue;

        // Skip if already matched
        if (task.contains_edge(eid)) continue;

        // Try to validate and extend binding
        uint32_t extended_bindings[MAX_VARS];
        uint32_t extended_mask = task.bound_mask;
        for (uint8_t i = 0; i < MAX_VARS; ++i) extended_bindings[i] = task.bindings[i];

        if (!validate_candidate_edge(pattern, eid, edges, extended_bindings, &extended_mask)) {
            continue;
        }

        // Valid candidate - build extended match state
        EdgeId new_edges[MAX_PATTERN_EDGES];
        for (uint8_t i = 0; i < task.num_matched; ++i) {
            new_edges[i] = task.matched_edges[i];
        }
        new_edges[task.num_matched] = eid;
        uint8_t new_num_matched = task.num_matched + 1;

        // If this completes the match, spawn SINK task (or output directly for now)
        if (new_num_matched >= rule->num_lhs) {
            // For now, output directly. Full HGMatch would spawn SINK task.
            emit_complete_match(new_edges, new_num_matched,
                               extended_bindings, extended_mask,
                               task.source_state, task.source_hash,
                               task.rule_index,
                               output, num_matches, max_matches);
        } else {
            // Spawn EXPAND task to generic task queue
            Task new_task = Task::make_expand(
                task.source_state, task.source_hash, task.rule_index, task.step,
                extended_bindings, extended_mask,
                new_edges, new_num_matched
            );
            task_queue->push(new_task);
        }
    }
}

// =============================================================================
// TSCAN Task (HGMatch Dataflow Model)
// =============================================================================
// SCAN iterates over edges with matching signature (arity) for the first
// pattern edge, and spawns EXPAND tasks to the GLOBAL queue.

__device__ void process_scan_task(
    const Task& task,
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    WorkQueueView<Task>* task_queue,  // GENERIC task queue
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    if (rule->num_lhs == 0) return;

    const DevicePatternEdge* first_pattern = &rule->lhs[0];

    // SCAN: find candidates for first pattern edge
    for (uint32_t eid = 0; eid < num_edges; ++eid) {
        if (*num_matches >= max_matches) return;

        // Skip if not in state
        if (!edge_in_state(eid, state_bitmap)) continue;

        // Skip if arity doesn't match (signature filtering)
        if (edges->arities[eid] != first_pattern->arity) continue;

        // Validate candidate and build initial binding
        uint32_t bindings[MAX_VARS] = {0};
        uint32_t bound_mask = 0;

        if (!validate_candidate_edge(first_pattern, eid, edges, bindings, &bound_mask)) {
            continue;
        }

        EdgeId matched_edges[MAX_PATTERN_EDGES];
        matched_edges[0] = eid;

        if (rule->num_lhs == 1) {
            // Single edge pattern - SINK directly (or spawn SINK task)
            emit_complete_match(matched_edges, 1, bindings, bound_mask,
                               task.source_state, task.source_hash, task.rule_index,
                               output, num_matches, max_matches);
        } else {
            // Multi-edge pattern - spawn EXPAND task to generic queue
            Task expand_task = Task::make_expand(
                task.source_state, task.source_hash, task.rule_index, task.step,
                bindings, bound_mask,
                matched_edges, 1
            );
            task_queue->push(expand_task);
        }
    }
}

// =============================================================================
// Legacy interface for standalone matching (without global queue)
// Uses local processing - for kernels that don't use the megakernel model
// =============================================================================

__device__ void find_matches_general(
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    StateId source_state,
    uint64_t source_hash,
    uint16_t rule_index,
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    // For standalone matching without global queue, we need to process
    // EXPAND tasks locally. This is less parallel but works for simple kernels.
    // The megakernel should use scan_pattern + process_expand_task with global queue.

    if (rule->num_lhs == 0) return;

    const DevicePatternEdge* first_pattern = &rule->lhs[0];

    // SCAN + immediate local EXPAND (nested loops, not backtracking)
    for (uint32_t e0 = 0; e0 < num_edges; ++e0) {
        if (*num_matches >= max_matches) return;
        if (!edge_in_state(e0, state_bitmap)) continue;
        if (edges->arities[e0] != first_pattern->arity) continue;

        uint32_t bind0[MAX_VARS] = {0};
        uint32_t mask0 = 0;
        if (!validate_candidate_edge(first_pattern, e0, edges, bind0, &mask0)) continue;

        if (rule->num_lhs == 1) {
            EdgeId matched[1] = {e0};
            emit_complete_match(matched, 1, bind0, mask0, source_state, source_hash, rule_index, output, num_matches, max_matches);
            continue;
        }

        // 2nd pattern edge
        const DevicePatternEdge* pat1 = &rule->lhs[1];
        for (uint32_t e1 = 0; e1 < num_edges; ++e1) {
            if (*num_matches >= max_matches) return;
            if (e1 == e0) continue;
            if (!edge_in_state(e1, state_bitmap)) continue;
            if (edges->arities[e1] != pat1->arity) continue;

            uint32_t bind1[MAX_VARS];
            uint32_t mask1 = mask0;
            for (uint8_t i = 0; i < MAX_VARS; ++i) bind1[i] = bind0[i];
            if (!validate_candidate_edge(pat1, e1, edges, bind1, &mask1)) continue;

            if (rule->num_lhs == 2) {
                EdgeId matched[2] = {e0, e1};
                emit_complete_match(matched, 2, bind1, mask1, source_state, source_hash, rule_index, output, num_matches, max_matches);
                continue;
            }

            // 3rd pattern edge
            const DevicePatternEdge* pat2 = &rule->lhs[2];
            for (uint32_t e2 = 0; e2 < num_edges; ++e2) {
                if (*num_matches >= max_matches) return;
                if (e2 == e0 || e2 == e1) continue;
                if (!edge_in_state(e2, state_bitmap)) continue;
                if (edges->arities[e2] != pat2->arity) continue;

                uint32_t bind2[MAX_VARS];
                uint32_t mask2 = mask1;
                for (uint8_t i = 0; i < MAX_VARS; ++i) bind2[i] = bind1[i];
                if (!validate_candidate_edge(pat2, e2, edges, bind2, &mask2)) continue;

                if (rule->num_lhs == 3) {
                    EdgeId matched[3] = {e0, e1, e2};
                    emit_complete_match(matched, 3, bind2, mask2, source_state, source_hash, rule_index, output, num_matches, max_matches);
                    continue;
                }

                // 4th pattern edge (extend as needed for larger patterns)
                const DevicePatternEdge* pat3 = &rule->lhs[3];
                for (uint32_t e3 = 0; e3 < num_edges; ++e3) {
                    if (*num_matches >= max_matches) return;
                    if (e3 == e0 || e3 == e1 || e3 == e2) continue;
                    if (!edge_in_state(e3, state_bitmap)) continue;
                    if (edges->arities[e3] != pat3->arity) continue;

                    uint32_t bind3[MAX_VARS];
                    uint32_t mask3 = mask2;
                    for (uint8_t i = 0; i < MAX_VARS; ++i) bind3[i] = bind2[i];
                    if (!validate_candidate_edge(pat3, e3, edges, bind3, &mask3)) continue;

                    EdgeId matched[4] = {e0, e1, e2, e3};
                    emit_complete_match(matched, 4, bind3, mask3, source_state, source_hash, rule_index, output, num_matches, max_matches);
                }
            }
        }
    }
}

// Main dispatcher for pattern matching
// Uses general HGMatch-style algorithm for all pattern sizes
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
) {
    // Use the general HGMatch-style algorithm for all cases
    // This handles patterns with any number of LHS edges (1 to MAX_PATTERN_EDGES)
    find_matches_general(rule, state_bitmap, edges, source_state, source_hash,
                        rule_index, output, num_matches, max_matches, num_edges);
}

// =============================================================================
// Kernel Implementations
// =============================================================================

__global__ void count_matches_kernel(
    const StateId* states,
    uint32_t num_states,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    DeviceEdges edges,              // BY VALUE
    StatePool state_pool,           // BY VALUE
    DeviceAdjacency adj,            // BY VALUE
    uint32_t* match_counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    StateId sid = states[tid];
    const DeviceState* state = state_pool.states + sid;
    const uint64_t* bitmap = state_pool.all_bitmaps + sid * BITMAP_WORDS;

    uint32_t count = 0;

    // Count matches for each rule
    // This is a simplified count - actual implementation would need
    // to run full matching but discard results
    // For now, use heuristic based on edge count and pattern size

    for (uint32_t r = 0; r < num_rules; ++r) {
        const DeviceRewriteRule* rule = &rules[r];
        // Rough estimate: edges^pattern_size matches
        uint32_t estimated = state->edge_count;
        for (uint8_t i = 1; i < rule->num_lhs; ++i) {
            estimated = estimated * state->edge_count / (i + 1);
        }
        count += estimated;
    }

    match_counts[tid] = min(count, (uint32_t)1000);  // Cap estimate
}

__global__ void collect_matches_kernel(
    const StateId* states,
    uint32_t num_states,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    DeviceEdges edges,              // BY VALUE
    StatePool state_pool,           // BY VALUE
    DeviceAdjacency adj,            // BY VALUE
    const uint32_t* output_offsets,
    DeviceMatch* matches
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    StateId sid = states[tid];
    const DeviceState* state = state_pool.states + sid;
    const uint64_t* bitmap = state_pool.all_bitmaps + sid * BITMAP_WORDS;

    uint32_t output_start = output_offsets[tid];
    uint32_t max_out = output_offsets[tid + 1] - output_start;

    DeviceMatch* output = matches + output_start;
    uint32_t num_matches = 0;

    // Find matches for each rule
    for (uint32_t r = 0; r < num_rules && num_matches < max_out; ++r) {
        find_matches_dispatch(
            &rules[r], bitmap, &edges, &adj,
            sid, state->canonical_hash, r,
            output + num_matches, &num_matches, max_out - num_matches,
            edges.cached_num_edges  // Use cached value, not device pointer
        );
    }
}

// =============================================================================
// Warp-Level Matching (for megakernel)
// =============================================================================

__device__ uint32_t find_matches_warp(
    StateId state,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    const DeviceEdges* edges,
    const StatePool* state_pool,
    const DeviceAdjacency* adj,
    DeviceMatch* output,
    uint32_t max_matches
) {
    const uint32_t lane = threadIdx.x % 32;

    const DeviceState* s = state_pool->states + state;
    const uint64_t* bitmap = state_pool->all_bitmaps + state * BITMAP_WORDS;

    // Use shared memory for warp-local match collection
    __shared__ uint32_t warp_num_matches[8];  // Up to 8 warps per block
    uint32_t warp_id = threadIdx.x / 32;

    if (lane == 0) {
        warp_num_matches[warp_id] = 0;
    }
    __syncwarp();

    // Process all rules sequentially (lane 0 does the work)
    // Note: Could parallelize across warps for multiple rules, but for simplicity
    // we serialize here - most patterns have few rules anyway
    // IMPORTANT: Use cumulative_matches to track total matches across all rules
    // and offset the output pointer so each rule's matches don't overwrite previous
    uint32_t cumulative_matches = 0;
    for (uint32_t r = 0; r < num_rules; r++) {
        if (lane == 0) {
            // Use live num_edges counter to see dynamically created edges
            // This is safe because:
            // 1. The bitmap determines which edges are in this state
            // 2. Any edge in the bitmap has had its data fully written (threadfenced)
            // 3. Edge IDs only grow, so num_edges is monotonically increasing
            uint32_t current_num_edges = *edges->num_edges;

            // Use local counter starting at 0 for this rule
            uint32_t local_matches = 0;
            // Write to offset position in output buffer
            find_matches_dispatch(
                &rules[r], bitmap, edges, adj,
                state, s->canonical_hash, r,
                output + cumulative_matches, &local_matches,
                max_matches > cumulative_matches ? max_matches - cumulative_matches : 0,
                current_num_edges
            );
            cumulative_matches += local_matches;
        }
        __syncwarp();  // Ensure all lanes see updated match count before next rule
    }

    if (lane == 0) {
        warp_num_matches[warp_id] = cumulative_matches;
    }
    __syncwarp();

    return warp_num_matches[warp_id];
}

// =============================================================================
// Delta Matching: Find matches involving at least one produced edge
// =============================================================================
// Following unified/pattern_matcher.hpp find_delta_matches:
// For each produced edge, try it at each pattern position.
// Deduplication handles overlaps when multiple produced edges are in one match.

// Helper: Check if an edge is in the produced set
__device__ __forceinline__ bool is_produced_edge(
    EdgeId eid,
    const EdgeId* produced_edges,
    uint8_t num_produced
) {
    for (uint8_t i = 0; i < num_produced; ++i) {
        if (produced_edges[i] == eid) return true;
    }
    return false;
}

// =============================================================================
// Delta EXPAND Task Processing (HGMatch Dataflow Model)
// =============================================================================
// Delta matching: EXPAND from a specific starting position (produced edge).
// Uses the ExpandTask.delta_start_position field to track this.

__device__ void process_delta_expand_task(
    const Task& task,
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    WorkQueueView<Task>* task_queue,  // GENERIC task queue
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    // For delta matching, matched_edges may have gaps (positions filled non-sequentially)
    // Count how many positions are actually filled
    uint8_t filled_count = 0;
    bool position_filled[MAX_PATTERN_EDGES] = {false};
    for (uint8_t i = 0; i < rule->num_lhs; ++i) {
        if (task.matched_edges[i] != INVALID_ID) {
            position_filled[i] = true;
            filled_count++;
        }
    }

    // Check if complete
    if (filled_count >= rule->num_lhs) {
        emit_complete_match(task.matched_edges, rule->num_lhs,
                           task.bindings, task.bound_mask,
                           task.source_state, task.source_hash, task.rule_index,
                           output, num_matches, max_matches);
        return;
    }

    if (*num_matches >= max_matches) return;

    // Find next unfilled position
    uint8_t next_pos = 0;
    for (uint8_t i = 0; i < rule->num_lhs; ++i) {
        if (!position_filled[i]) {
            next_pos = i;
            break;
        }
    }

    const DevicePatternEdge* pattern = &rule->lhs[next_pos];

    // Generate candidates for this position
    for (uint32_t eid = 0; eid < num_edges; ++eid) {
        if (*num_matches >= max_matches) return;

        if (!edge_in_state(eid, state_bitmap)) continue;
        if (edges->arities[eid] != pattern->arity) continue;
        if (task.contains_edge(eid)) continue;

        // Validate and extend binding
        uint32_t extended_bindings[MAX_VARS];
        uint32_t extended_mask = task.bound_mask;
        for (uint8_t i = 0; i < MAX_VARS; ++i) extended_bindings[i] = task.bindings[i];

        if (!validate_candidate_edge(pattern, eid, edges, extended_bindings, &extended_mask)) {
            continue;
        }

        // Build new match state with this position filled
        EdgeId new_edges[MAX_PATTERN_EDGES];
        for (uint8_t i = 0; i < rule->num_lhs; ++i) {
            new_edges[i] = task.matched_edges[i];
        }
        new_edges[next_pos] = eid;

        // Check if all positions now filled
        uint8_t new_filled = filled_count + 1;
        if (new_filled >= rule->num_lhs) {
            emit_complete_match(new_edges, rule->num_lhs,
                               extended_bindings, extended_mask,
                               task.source_state, task.source_hash,
                               task.rule_index,
                               output, num_matches, max_matches);
        } else {
            // Spawn EXPAND task to generic queue (with delta info preserved)
            Task new_task = Task::make_expand(
                task.source_state, task.source_hash, task.rule_index, task.step,
                extended_bindings, extended_mask,
                new_edges, rule->num_lhs,  // num_matched tracks total positions
                task.delta_start_position  // Preserve delta info
            );
            task_queue->push(new_task);
        }
    }
}

// =============================================================================
// Delta SCAN: Start pattern matching from a specific edge at a specific position
// =============================================================================
// This is used for delta matching: the produced edge is fixed at pattern_position,
// and we spawn EXPAND tasks to fill the remaining positions.

__device__ void scan_pattern_from_edge(
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    StateId source_state,
    uint64_t source_hash,
    uint16_t rule_index,
    uint32_t step,
    EdgeId starting_edge,
    uint8_t pattern_position,
    WorkQueueView<Task>* task_queue,  // GENERIC task queue
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    if (rule->num_lhs == 0) return;
    if (pattern_position >= rule->num_lhs) return;

    const DevicePatternEdge* pattern = &rule->lhs[pattern_position];

    // Validate the starting edge matches the pattern at this position
    uint32_t bindings[MAX_VARS] = {0};
    uint32_t bound_mask = 0;

    if (!validate_candidate_edge(pattern, starting_edge, edges, bindings, &bound_mask)) {
        return;
    }

    // Check arity
    if (edges->arities[starting_edge] != pattern->arity) {
        return;
    }

    if (rule->num_lhs == 1) {
        // Single edge pattern - SINK directly
        EdgeId matched_edges[MAX_PATTERN_EDGES];
        matched_edges[0] = starting_edge;
        emit_complete_match(matched_edges, 1, bindings, bound_mask,
                           source_state, source_hash, rule_index,
                           output, num_matches, max_matches);
        return;
    }

    // Multi-edge pattern - spawn EXPAND task to generic queue
    EdgeId matched_edges[MAX_PATTERN_EDGES];
    for (uint8_t i = 0; i < MAX_PATTERN_EDGES; ++i) matched_edges[i] = INVALID_ID;
    matched_edges[pattern_position] = starting_edge;

    Task expand_task = Task::make_expand(
        source_state, source_hash, rule_index, step,
        bindings, bound_mask,
        matched_edges, rule->num_lhs,  // num_matched = total positions for delta
        pattern_position  // delta_start_position
    );
    task_queue->push(expand_task);
}

// Delta matching: find matches that include at least one produced edge
// Following unified/pattern_matcher.hpp find_delta_matches
// Spawns EXPAND tasks to the GENERIC task queue
__device__ void find_delta_matches_general(
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    StateId source_state,
    uint64_t source_hash,
    uint16_t rule_index,
    uint32_t step,
    const EdgeId* produced_edges,
    uint8_t num_produced,
    WorkQueueView<Task>* task_queue,  // GENERIC task queue
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    if (num_produced == 0) return;
    if (rule->num_lhs == 0) return;

    // For each produced edge, try it at each pattern position
    // This ensures we find all matches that include at least one produced edge
    for (uint8_t p = 0; p < num_produced && *num_matches < max_matches; ++p) {
        EdgeId produced = produced_edges[p];

        // Skip if edge not in state
        if (!edge_in_state(produced, state_bitmap)) continue;

        for (uint8_t pos = 0; pos < rule->num_lhs && *num_matches < max_matches; ++pos) {
            scan_pattern_from_edge(
                rule, state_bitmap, edges,
                source_state, source_hash, rule_index,
                step, produced, pos,
                task_queue,  // Pass generic queue
                output, num_matches, max_matches, num_edges
            );
        }
    }
}

// =============================================================================
// Legacy Delta Matching (without global queue - processes locally)
// =============================================================================
// For kernels that don't have access to the global expand queue.
// Uses nested loops locally. The megakernel should use the global queue version.

__device__ void find_delta_matches_local(
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    StateId source_state,
    uint64_t source_hash,
    uint16_t rule_index,
    const EdgeId* produced_edges,
    uint8_t num_produced,
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    if (num_produced == 0) return;
    if (rule->num_lhs == 0) return;

    // For single-edge patterns, just check if produced edges match
    if (rule->num_lhs == 1) {
        const DevicePatternEdge* pat = &rule->lhs[0];
        for (uint8_t p = 0; p < num_produced && *num_matches < max_matches; ++p) {
            EdgeId e = produced_edges[p];
            if (!edge_in_state(e, state_bitmap)) continue;
            if (edges->arities[e] != pat->arity) continue;
            uint32_t bindings[MAX_VARS] = {0};
            uint32_t mask = 0;
            if (validate_candidate_edge(pat, e, edges, bindings, &mask)) {
                EdgeId matched[1] = {e};
                emit_complete_match(matched, 1, bindings, mask, source_state, source_hash, rule_index, output, num_matches, max_matches);
            }
        }
        return;
    }

    // For 2-edge patterns
    if (rule->num_lhs == 2) {
        const DevicePatternEdge* pat0 = &rule->lhs[0];
        const DevicePatternEdge* pat1 = &rule->lhs[1];

        for (uint8_t p = 0; p < num_produced && *num_matches < max_matches; ++p) {
            EdgeId produced = produced_edges[p];
            if (!edge_in_state(produced, state_bitmap)) continue;

            // Try produced at position 0
            if (edges->arities[produced] == pat0->arity) {
                uint32_t bind0[MAX_VARS] = {0};
                uint32_t mask0 = 0;
                if (validate_candidate_edge(pat0, produced, edges, bind0, &mask0)) {
                    for (uint32_t e1 = 0; e1 < num_edges && *num_matches < max_matches; ++e1) {
                        if (e1 == produced) continue;
                        if (!edge_in_state(e1, state_bitmap)) continue;
                        if (edges->arities[e1] != pat1->arity) continue;
                        uint32_t bind1[MAX_VARS];
                        uint32_t mask1 = mask0;
                        for (uint8_t i = 0; i < MAX_VARS; ++i) bind1[i] = bind0[i];
                        if (validate_candidate_edge(pat1, e1, edges, bind1, &mask1)) {
                            EdgeId matched[2] = {produced, e1};
                            emit_complete_match(matched, 2, bind1, mask1, source_state, source_hash, rule_index, output, num_matches, max_matches);
                        }
                    }
                }
            }

            // Try produced at position 1
            if (edges->arities[produced] == pat1->arity) {
                uint32_t bind1[MAX_VARS] = {0};
                uint32_t mask1 = 0;
                if (validate_candidate_edge(pat1, produced, edges, bind1, &mask1)) {
                    for (uint32_t e0 = 0; e0 < num_edges && *num_matches < max_matches; ++e0) {
                        if (e0 == produced) continue;
                        if (!edge_in_state(e0, state_bitmap)) continue;
                        if (edges->arities[e0] != pat0->arity) continue;
                        // Skip if this was already a produced edge (avoid duplicate)
                        if (is_produced_edge(e0, produced_edges, num_produced)) continue;
                        uint32_t bind0[MAX_VARS];
                        uint32_t mask0 = mask1;
                        for (uint8_t i = 0; i < MAX_VARS; ++i) bind0[i] = bind1[i];
                        if (validate_candidate_edge(pat0, e0, edges, bind0, &mask0)) {
                            EdgeId matched[2] = {e0, produced};
                            emit_complete_match(matched, 2, bind0, mask0, source_state, source_hash, rule_index, output, num_matches, max_matches);
                        }
                    }
                }
            }
        }
        return;
    }

    // For larger patterns, use nested loops (extend as needed)
    // This is a simplified local version - the global queue version is more complete
}

// Delta matching dispatcher (legacy - uses local processing)
__device__ void find_delta_matches_dispatch(
    const DeviceRewriteRule* rule,
    const uint64_t* state_bitmap,
    const DeviceEdges* edges,
    StateId source_state,
    uint64_t source_hash,
    uint16_t rule_index,
    const EdgeId* produced_edges,
    uint8_t num_produced,
    DeviceMatch* output,
    uint32_t* num_matches,
    uint32_t max_matches,
    uint32_t num_edges
) {
    find_delta_matches_local(rule, state_bitmap, edges, source_state, source_hash,
                             rule_index, produced_edges, num_produced,
                             output, num_matches, max_matches, num_edges);
}

// Warp-level delta matching for megakernel integration
__device__ uint32_t find_delta_matches_warp(
    StateId state,
    const DeviceRewriteRule* rules,
    uint32_t num_rules,
    const DeviceEdges* edges,
    const StatePool* state_pool,
    const DeviceAdjacency* adj,
    const EdgeId* produced_edges,
    uint8_t num_produced,
    DeviceMatch* output,
    uint32_t max_matches
) {
    const uint32_t lane = threadIdx.x % 32;

    const DeviceState* s = state_pool->states + state;
    const uint64_t* bitmap = state_pool->all_bitmaps + state * BITMAP_WORDS;

    // Use shared memory for warp-local match collection
    __shared__ uint32_t warp_num_matches[8];
    uint32_t warp_id = threadIdx.x / 32;

    if (lane == 0) {
        warp_num_matches[warp_id] = 0;
    }
    __syncwarp();

    // If no produced edges, no delta matches possible
    if (num_produced == 0) {
        return 0;
    }

    // Process all rules sequentially (lane 0 does the work)
    // IMPORTANT: Use cumulative_matches to track total matches across all rules
    // and offset the output pointer so each rule's matches don't overwrite previous
    uint32_t cumulative_matches = 0;
    for (uint32_t r = 0; r < num_rules; r++) {
        if (lane == 0) {
            // For delta matching, we need to see newly created edges
            // The rewrite that produced these edges has completed (threadfence'd)
            // so it's safe to read the live counter and edge data
            uint32_t current_num_edges = *edges->num_edges;

            // Use local counter starting at 0 for this rule
            uint32_t local_matches = 0;
            // Write to offset position in output buffer
            find_delta_matches_dispatch(
                &rules[r], bitmap, edges,
                state, s->canonical_hash, r,
                produced_edges, num_produced,
                output + cumulative_matches, &local_matches,
                max_matches > cumulative_matches ? max_matches - cumulative_matches : 0,
                current_num_edges
            );
            cumulative_matches += local_matches;
        }
        __syncwarp();  // Ensure all lanes see updated match count before next rule
    }

    if (lane == 0) {
        warp_num_matches[warp_id] = cumulative_matches;
    }
    __syncwarp();

    return warp_num_matches[warp_id];
}

// =============================================================================
// Host Interface Implementation
// =============================================================================

void PatternMatcher::init() {
    CUDA_CHECK(cudaMalloc(&d_match_counts_, MAX_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output_offsets_, (MAX_STATES + 1) * sizeof(uint32_t)));

    // Allocate temp storage for prefix sum
    temp_storage_bytes_ = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes_,
                                   d_match_counts_, d_output_offsets_, MAX_STATES + 1);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_, temp_storage_bytes_));

    // Increase stack limit for multi-edge pattern matching
    // Default is 1024 bytes, we need more for complex patterns
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 2048));
}

void PatternMatcher::destroy() {
    if (d_match_counts_) cudaFree(d_match_counts_);
    if (d_output_offsets_) cudaFree(d_output_offsets_);
    if (d_temp_storage_) cudaFree(d_temp_storage_);
    d_match_counts_ = d_output_offsets_ = nullptr;
    d_temp_storage_ = nullptr;
}

uint32_t PatternMatcher::find_matches(
    const StateId* d_states,
    uint32_t num_states,
    const DeviceRewriteRule* d_rules,
    uint32_t num_rules,
    DeviceEdges edges,              // BY VALUE
    StatePool state_pool,           // BY VALUE
    DeviceAdjacency adj,            // BY VALUE
    DeviceMatch* d_matches,
    uint32_t max_matches,
    cudaStream_t stream
) {
    if (num_states == 0) return 0;

    const int block_size = 256;
    const int num_blocks = (num_states + block_size - 1) / block_size;

    // Debug: Check for null pointers before kernel launch
    if (state_pool.states == nullptr) {
        fprintf(stderr, "ERROR: state_pool.states is null\n");
        return 0;
    }
    if (state_pool.all_bitmaps == nullptr) {
        fprintf(stderr, "ERROR: state_pool.all_bitmaps is null\n");
        return 0;
    }

    // Phase 1: Count matches
    count_matches_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_states, num_states, d_rules, num_rules, edges, state_pool, adj,
        d_match_counts_
    );

    // Check for errors after count kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "count_matches_kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Prefix sum to get output offsets
    cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
                                   d_match_counts_, d_output_offsets_,
                                   num_states + 1, stream);

    // Get total count
    uint32_t total_matches;
    CUDA_CHECK(cudaMemcpyAsync(&total_matches, d_output_offsets_ + num_states,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (total_matches > max_matches) {
        total_matches = max_matches;
    }

    // Phase 2: Collect matches
    collect_matches_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_states, num_states, d_rules, num_rules, edges, state_pool, adj,
        d_output_offsets_, d_matches
    );

    // Check for errors after collect kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "collect_matches_kernel launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize error after collect_matches: %s\n", cudaGetErrorString(err));
    }

    return total_matches;
}

}  // namespace hypergraph::gpu
