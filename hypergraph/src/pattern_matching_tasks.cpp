#include <hypergraph/pattern_matching_tasks.hpp>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/debug_log.hpp>
#include <algorithm>
#include <numeric>
#include <string>

namespace hypergraph {

void ScanTask::execute() {
    DEBUG_LOG("[SCAN] EXECUTING SCAN task for pattern edge %zu, partition [%zu,%zu)",
              pattern_edge_idx_, partition_start_, partition_end_);

    if (context_->should_terminate.load()) {
        DEBUG_LOG("[SCAN] Terminating due to should_terminate flag");
        return;
    }

    const auto& target_hg = *context_->target_hypergraph;
    const auto& pattern = *context_->pattern;

    if (pattern_edge_idx_ >= pattern.num_edges()) {
        DEBUG_LOG("[SCAN] Invalid pattern edge index: %zu >= %zu", pattern_edge_idx_, pattern.num_edges());
        return;
    }

    const auto& pattern_edge = pattern.edges()[pattern_edge_idx_];

    // Create signature for this pattern edge
    EdgeSignature pattern_sig = EdgeSignature::from_pattern_edge(
        pattern_edge, context_->label_function
    );

    // Use signature index if available, otherwise iterate through all edges
    std::vector<EdgeId> compatible_edges;
    
    if (context_->signature_index) {
        // Get edges with compatible signatures from the index
        auto edges_with_sig = context_->signature_index->get_edges_with_signature(pattern_sig);
        compatible_edges.insert(compatible_edges.end(), edges_with_sig.begin(), edges_with_sig.end());
        
        // Find additional compatible edges by checking all signatures in the index
        compatible_edges = context_->signature_index->find_compatible_edges(pattern_sig);
    } else {
        // Fallback: iterate through all edges and check signatures
        for (EdgeId edge_id = 0; edge_id < target_hg.num_edges(); ++edge_id) {
            const auto& concrete_edge = target_hg.edges()[edge_id];
            EdgeSignature concrete_sig = EdgeSignature::from_concrete_edge(concrete_edge, context_->label_function);
            
            if (concrete_sig.is_compatible_with_pattern(pattern_sig)) {
                compatible_edges.push_back(edge_id);
            }
        }
    }

    DEBUG_LOG("[SCAN] Pattern edge arity=%zu, target has %zu edges",
              pattern_edge.arity(), target_hg.num_edges());
    DEBUG_LOG("[SCAN] Found %zu compatible edges (arity match)", compatible_edges.size());

    if (compatible_edges.empty()) {
        DEBUG_LOG("[SCAN] No compatible edges found, returning");
        return;
    }

    // Process the assigned partition
    std::size_t actual_end = std::min(partition_end_, compatible_edges.size());

    DEBUG_LOG("[SCAN] Processing partition [%zu,%zu)", partition_start_, actual_end);

    for (std::size_t i = partition_start_; i < actual_end; ++i) {
        if (context_->should_terminate.load()) {
            DEBUG_LOG("[SCAN] Terminating early due to should_terminate flag");
            break;
        }

        EdgeId candidate_edge = compatible_edges[i];
        const auto& concrete_edge = target_hg.get_edge(candidate_edge);

        DEBUG_LOG("[SCAN] Processing candidate edge %zu (index %zu)", candidate_edge, i);

        // Generate all possible variable assignments for this edge match
        auto assignments = generate_edge_assignments(pattern_edge, *concrete_edge);

        DEBUG_LOG("[SCAN] Generated %zu assignments for edge %zu", assignments.size(), candidate_edge);

        for (const auto& assignment : assignments) {
            if (context_->should_terminate.load()) break;

            // Create initial partial match
            PartialMatch partial_match;
            partial_match.add_edge_match(pattern_edge_idx_, candidate_edge, assignment);

            // Set available edges (all edges except the one we just used)
            partial_match.available_edges.clear();
            for (EdgeId eid = 0; eid < target_hg.num_edges(); ++eid) {
                if (eid != candidate_edge) {
                    partial_match.available_edges.insert(eid);
                }
            }

            // Determine anchor vertex for locality (pick first vertex of matched edge)
            if (!concrete_edge->vertices().empty()) {
                partial_match.anchor_vertex = concrete_edge->vertices()[0];
            }

            // If pattern has only one edge, this is already a complete match
            if (pattern.num_edges() == 1) {
                DEBUG_LOG("[SCAN] Single-edge pattern: creating complete match");
                PatternMatch complete_match;
                complete_match.matched_edges = partial_match.matched_edges;
                complete_match.edge_map = partial_match.edge_map;
                complete_match.assignment = partial_match.assignment;
                complete_match.anchor_vertex = partial_match.anchor_vertex;

                DEBUG_LOG("[SCAN] Executing SINK task directly");
                // Create and execute SINK task directly (avoid scheduler overhead)
                SinkTask sink_task(context_, std::move(complete_match));
                sink_task.execute();
            } else {
                DEBUG_LOG("[SCAN] Multi-edge pattern: creating partial match, about to create and execute EXPAND task");
                // Create and execute EXPAND task directly
                ExpandTask expand_task(context_, std::move(partial_match));
                DEBUG_LOG("[SCAN] EXPAND task created, about to execute");
                expand_task.execute();
                DEBUG_LOG("[SCAN] EXPAND task execution completed");
            }
        }
    }
}

void ExpandTask::execute() {
    DEBUG_LOG("[EXPAND] Executing EXPAND task");

    if (context_->should_terminate.load()) return;

    const auto& target_hg = *context_->target_hypergraph;
    const auto& pattern = *context_->pattern;

    // Check if this is already a complete match
    bool edges_complete = partial_match_.is_complete(pattern.num_edges());

    // Also check that all pattern variables are bound
    std::unordered_set<VertexId> all_pattern_variables;
    for (const auto& edge : pattern.edges()) {
        for (const auto& vertex : edge.vertices) {
            if (!vertex.is_concrete()) {
                all_pattern_variables.insert(vertex.id);
            }
        }
    }
    bool variables_complete = partial_match_.assignment.is_complete(all_pattern_variables);

    if (edges_complete && variables_complete) {
        PatternMatch complete_match;
        complete_match.matched_edges = partial_match_.matched_edges;
        complete_match.edge_map = partial_match_.edge_map;
        complete_match.assignment = partial_match_.assignment;
        complete_match.anchor_vertex = partial_match_.anchor_vertex;

        // Execute SINK task directly
        SinkTask sink_task(context_, std::move(complete_match));
        sink_task.execute();
        return;
    } else if (edges_complete && !variables_complete) {
        DEBUG_LOG("[EXPAND] WARNING: All edges matched but variables incomplete - rejecting partial match");
        return;  // Reject incomplete matches
    }

    // Find next pattern edge to match
    std::size_t next_edge_idx = partial_match_.next_pattern_edge_idx;
    if (next_edge_idx >= pattern.num_edges()) return;

    const auto& next_pattern_edge = pattern.edges()[next_edge_idx];

    // Create signature for next pattern edge
    EdgeSignature pattern_sig = EdgeSignature::from_pattern_edge(
        next_pattern_edge, context_->label_function
    );

    // Find compatible edges from remaining available edges
    std::vector<EdgeId> candidates;
    candidates.reserve(partial_match_.available_edges.size());

    for (EdgeId eid : partial_match_.available_edges) {
        const auto& candidate_edge = target_hg.get_edge(eid);
        EdgeSignature candidate_sig = EdgeSignature::from_concrete_edge(
            *candidate_edge, context_->label_function
        );

        if (candidate_sig.is_compatible_with_pattern(pattern_sig)) {
            candidates.push_back(eid);
        }
    }

    // Try to extend partial match with each candidate
    for (EdgeId candidate : candidates) {
        if (context_->should_terminate.load()) break;

        const auto& concrete_edge = target_hg.get_edge(candidate);
        auto assignments = generate_edge_assignments(next_pattern_edge, *concrete_edge);

        for (const auto& assignment : assignments) {
            if (context_->should_terminate.load()) break;

            std::string assignment_str = "[EXPAND] Testing assignment for edge " + std::to_string(candidate) + ": ";
            for (const auto& [var, val] : assignment.variable_to_concrete) {
                assignment_str += "var" + std::to_string(var) + "→" + std::to_string(val) + " ";
            }
            DEBUG_LOG("%s", assignment_str.c_str());

            // Check assignment consistency with existing partial match
            if (!is_assignment_consistent(partial_match_.assignment, assignment)) {
                DEBUG_LOG("[EXPAND] Assignment inconsistent with existing partial match");
                continue;
            }

            // Create extended partial match
            PartialMatch extended_match = partial_match_.branch();

            // Merge assignments
            VariableAssignment merged_assignment = partial_match_.assignment;
            bool assignment_valid = true;
            for (const auto& [var, val] : assignment.variable_to_concrete) {
                if (!merged_assignment.assign(var, val)) {
                    assignment_valid = false;
                    break;
                }
            }

            if (!assignment_valid) {
                DEBUG_LOG("[EXPAND] Merged assignment invalid");
                continue;
            }

            std::string extended_str = "[EXPAND] Extended match successful: ";
            for (const auto& [var, val] : merged_assignment.variable_to_concrete) {
                extended_str += "var" + std::to_string(var) + "→" + std::to_string(val) + " ";
            }
            DEBUG_LOG("%s", extended_str.c_str());

            extended_match.add_edge_match(next_edge_idx, candidate, merged_assignment);

            // Execute another EXPAND task for the extended match
            ExpandTask expand_task(context_, std::move(extended_match));
            expand_task.execute();
        }
    }

    context_->expand_tasks_spawned.fetch_add(1);
}

void SinkTask::execute() {
    DEBUG_LOG("[SINK] Executing SINK task for complete match");

    // Increment match count atomically
    std::size_t current_count = context_->total_matches_found.fetch_add(1);

    DEBUG_LOG("[SINK] Match count increased to %zu", current_count + 1);

    // Check if we've exceeded the match limit
    if (current_count >= context_->max_matches_to_find.load()) {
        DEBUG_LOG("[SINK] Match limit reached (%zu), setting terminate flag", context_->max_matches_to_find.load());
        context_->should_terminate.store(true);
        return;
    }

    // Spawn REWRITE task for this match
    DEBUG_LOG("[SINK] Spawning REWRITE task for match");

    // Create and submit REWRITE task
    auto rewrite_job = job_system::make_job(
        [context = context_, match = complete_match_]() mutable {
            RewriteTask task(context, std::move(match));
            task.execute();
        },
        PatternMatchingTaskType::REWRITE
    );

    context_->job_system->submit(std::move(rewrite_job), job_system::ScheduleMode::FIFO);
    context_->rewrite_tasks_spawned.fetch_add(1);
    context_->sink_tasks_spawned.fetch_add(1);

    DEBUG_LOG("[SINK] REWRITE task submitted successfully");
}

void RewriteTask::execute() {
    std::string assignment_info = "[REWRITE] Executing REWRITE task. Variable assignments: ";
    for (const auto& [var, val] : complete_match_.assignment.variable_to_concrete) {
        assignment_info += "var" + std::to_string(var) + "→" + std::to_string(val) + " ";
    }
    DEBUG_LOG("%s", assignment_info.c_str());

    // Apply the first rewriting rule from context
    if (context_->rewrite_rules.empty()) {
        DEBUG_LOG("[REWRITE] No rewriting rules available");
        return;
    }

    std::string consumed_edges = "[REWRITE] Starting rewrite for state " + std::to_string(context_->current_state_id) + ", consumed edges: ";
    for (GlobalEdgeId eid : complete_match_.matched_edges) {
        consumed_edges += std::to_string(eid) + " ";
    }
    DEBUG_LOG("%s", consumed_edges.c_str());

    const auto& rule = context_->rewrite_rules[0];

    // Create new edges by applying rule RHS
    std::vector<std::vector<GlobalVertexId>> new_edges;
    for (const auto& rhs_edge : rule.rhs.edges()) {
        std::vector<GlobalVertexId> edge_vertices;
        bool all_vertices_resolved = true;

        std::string rhs_info = "[REWRITE] Processing RHS edge with " + std::to_string(rhs_edge.vertices.size()) + " vertices: ";
        for (const auto& pv : rhs_edge.vertices) {
            rhs_info += std::to_string(pv.id) + " ";
        }
        DEBUG_LOG("%s", rhs_info.c_str());

        for (const auto& pattern_vertex : rhs_edge.vertices) {
            if (pattern_vertex.is_variable()) {
                // Use assignment to map pattern variables to concrete vertices
                auto resolved = complete_match_.assignment.resolve(pattern_vertex);
                if (resolved) {
                    // Variable was matched in LHS - convert canonical vertex ID back to global vertex ID
                    auto source_state = context_->multiway_graph->get_state(context_->current_state_id);
                    if (source_state) {
                        GlobalVertexId global_vertex = source_state->canonical_to_global_vertex(*resolved);
                        edge_vertices.push_back(global_vertex);
                        DEBUG_LOG("[REWRITE] Resolved LHS variable %zu → canonical %zu → global %zu",
                                  pattern_vertex.id, *resolved, global_vertex);
                    } else {
                        printf("[REWRITE] ERROR: Could not get source state %zu\n", context_->current_state_id);
                        all_vertices_resolved = false;
                        break;
                    }
                } else {
                    // Variable not in LHS assignment - create fresh vertex
                    GlobalVertexId fresh_vertex = context_->multiway_graph->get_fresh_vertex_id();
                    edge_vertices.push_back(fresh_vertex);
                    DEBUG_LOG("[REWRITE] Variable %zu NOT in LHS, creating fresh vertex %zu",
                              pattern_vertex.id, fresh_vertex);
                }
            } else {
                // Concrete vertex - use as is
                edge_vertices.push_back(static_cast<GlobalVertexId>(pattern_vertex.id));
                DEBUG_LOG("[REWRITE] Concrete vertex %zu", pattern_vertex.id);
            }
        }

        // Only add edge if ALL vertices were resolved
        if (all_vertices_resolved && !edge_vertices.empty()) {
            new_edges.push_back(edge_vertices);
            DEBUG_LOG("[REWRITE] Adding edge with %zu vertices", edge_vertices.size());
        } else if (!all_vertices_resolved) {
            DEBUG_LOG("[REWRITE] SKIPPING edge due to unresolved vertices");
        }
    }

    // Remove matched edges and add new edges to create new state
    std::vector<GlobalEdgeId> edges_to_remove;
    auto source_state = context_->multiway_graph->get_state(context_->current_state_id);
    if (!source_state) {
        printf("[REWRITE] ERROR: Could not get source state %zu\n", context_->current_state_id);
        return;
    }

    std::string source_info = "[REWRITE] Source state has " + std::to_string(source_state->num_edges()) + " edges, matched edges: ";
    for (EdgeId edge_id : complete_match_.matched_edges) {
        source_info += std::to_string(edge_id) + " ";
    }
    DEBUG_LOG("%s", source_info.c_str());

    for (EdgeId canonical_edge_id : complete_match_.matched_edges) {
        // Convert canonical edge index to actual global edge ID in the source state
        const auto& source_edges = source_state->edges();
        if (canonical_edge_id < source_edges.size()) {
            GlobalEdgeId global_edge_id = source_edges[canonical_edge_id].global_id;
            edges_to_remove.push_back(global_edge_id);
            DEBUG_LOG("[REWRITE] Removing canonical edge %zu → global edge %zu",
                      canonical_edge_id, global_edge_id);
        }
    }

    std::string rewrite_info = "[REWRITE] Creating new state: removing " + std::to_string(edges_to_remove.size()) +
                               " edges, adding " + std::to_string(new_edges.size()) + " edges. ";
    rewrite_info += "Edges to remove: ";
    for (auto edge_id : edges_to_remove) {
        rewrite_info += std::to_string(edge_id) + " ";
    }
    rewrite_info += "Edges to add: ";
    for (const auto& edge : new_edges) {
        rewrite_info += "{";
        for (std::size_t i = 0; i < edge.size(); ++i) {
            rewrite_info += std::to_string(edge[i]);
            if (i < edge.size() - 1) rewrite_info += ",";
        }
        rewrite_info += "} ";
    }
    DEBUG_LOG("%s", rewrite_info.c_str());

    // Apply rewriting through private method
    EventId event_id = context_->multiway_graph->apply_rewriting(
        context_->current_state_id, edges_to_remove, new_edges, 0, 0);

    if (event_id != INVALID_EVENT) {
        DEBUG_LOG("[REWRITE] Created new event %zu", event_id);

        // Get the new state created by this rewrite
        auto event_opt = context_->multiway_graph->get_event(event_id);
        if (event_opt) {
            StateId new_state_id = event_opt->output_state_id;
            
            // Check step limit BEFORE incrementing to prevent infinite cascade
            std::size_t current_step = 0;
            if (context_->wolfram_evolution) {
                current_step = context_->wolfram_evolution->get_current_step();
            } else {
                current_step = context_->current_step.load();
            }

            DEBUG_LOG("[REWRITE] Step check: current_step=%zu, max_steps=%zu, state=%zu→%zu",
                      current_step, context_->max_steps, context_->current_state_id, new_state_id);

            // PATCH-BASED MATCHING: Search for new matches around newly added edges
            // Check limit BEFORE spawning to prevent cascade
            if (current_step < context_->max_steps) {
                // Now increment the step after checking the limit
                if (context_->wolfram_evolution) {
                    context_->wolfram_evolution->increment_step();
                } else {
                    context_->current_step.fetch_add(1);
                }
                try {
                    spawn_patch_based_matching_around_new_edges(new_state_id, new_edges, current_step);
                } catch (const std::exception& e) {
                    DEBUG_LOG("[REWRITE] Exception in patch-based matching: %s", e.what());
                }
            } else {
                DEBUG_LOG("[REWRITE] NOT SPAWNING patch-based matching: step %zu >= max_steps %zu", current_step, context_->max_steps);
            }

            DEBUG_LOG("[REWRITE] Event %zu completed, state %zu created", event_id, new_state_id);
        }
    }

    context_->rewrite_tasks_spawned.fetch_add(1);

    DEBUG_LOG("[REWRITE] REWRITE task completed");
}

void RewriteTask::spawn_patch_based_matching_around_new_edges(
    StateId new_state_id,
    const std::vector<std::vector<GlobalVertexId>>& new_edges,
    std::size_t current_step) {

    DEBUG_LOG("[REWRITE] Starting patch-based matching for state %zu, current_step=%zu", new_state_id, current_step);
    
    // Check current global step at spawn time to prevent concurrent tasks from all spawning
    std::size_t global_step_now = 0;
    if (context_->wolfram_evolution) {
        global_step_now = context_->wolfram_evolution->get_current_step();
    } else {
        global_step_now = context_->current_step.load();
    }
    
    if (global_step_now >= context_->max_steps) {  
        return;
    }
    

    // Get the new state
    auto new_state_opt = context_->multiway_graph->get_state(new_state_id);
    if (!new_state_opt) {
        DEBUG_LOG("[REWRITE] New state %zu not found for patch matching", new_state_id);
        return;
    }

    // Safety check: ensure we have valid new edges
    if (new_edges.empty()) {
        DEBUG_LOG("[REWRITE] No new edges to create patch around");
        return;
    }

    // Convert to hypergraph for pattern matching (use shared_ptr for safe sharing across rules)
    std::shared_ptr<Hypergraph> target_hg_ptr;
    try {
        auto temp_hg = new_state_opt->to_canonical_hypergraph();
        target_hg_ptr = std::make_shared<Hypergraph>(std::move(temp_hg));
    } catch (const std::exception& e) {
        DEBUG_LOG("[REWRITE] Exception creating canonical hypergraph: %s", e.what());
        return;
    }

    auto& target_hg = *target_hg_ptr;
    DEBUG_LOG("[REWRITE] New state has %zu vertices, %zu edges for patch matching",
              target_hg.num_vertices(), target_hg.num_edges());

    // Safety check: ensure hypergraph is valid
    if (target_hg.num_edges() == 0) {
        DEBUG_LOG("[REWRITE] Empty hypergraph, skipping patch matching");
        return;
    }

    // Use a conservative radius for patch-based matching
    std::size_t search_radius = 1;

    // Find vertices in the patch around newly added edges
    std::unordered_set<VertexId> patch_vertices;
    for (const auto& new_edge_global : new_edges) {
        for (GlobalVertexId global_vertex : new_edge_global) {
            // Map global vertex to canonical ID using efficient lookup
            try {
                auto canonical_id_opt = new_state_opt->global_to_canonical_vertex(global_vertex);
                if (canonical_id_opt) {
                    patch_vertices.insert(static_cast<VertexId>(*canonical_id_opt));
                    DEBUG_LOG("[REWRITE] Added vertex %zu (global %zu → canonical %zu) to patch",
                              *canonical_id_opt, global_vertex, *canonical_id_opt);
                } else {
                    DEBUG_LOG("[REWRITE] Could not find canonical mapping for global vertex %zu", global_vertex);
                }
            } catch (const std::exception& e) {
                DEBUG_LOG("[REWRITE] Error mapping global vertex %zu: %s", global_vertex, e.what());
                continue;
            }
        }
    }

    if (patch_vertices.empty()) {
        DEBUG_LOG("[REWRITE] No valid patch vertices found, skipping patch matching");
        return;
    }

    // Expand patch by search radius
    std::unordered_set<VertexId> expanded_patch = patch_vertices;
    for (std::size_t radius = 0; radius < search_radius; ++radius) {
        std::unordered_set<VertexId> new_vertices;

        // Safety check: limit expansion size to prevent memory issues
        if (expanded_patch.size() > 1000) {
            DEBUG_LOG("[REWRITE] Patch too large (%zu vertices), limiting expansion", expanded_patch.size());
            break;
        }

        for (VertexId v : expanded_patch) {
            if (v >= target_hg.num_vertices()) continue; // Safety check

            // Find all edges containing this vertex (with bounds checking)
            for (EdgeId edge_id = 0; edge_id < target_hg.num_edges(); ++edge_id) {
                try {
                    const auto& edge = target_hg.edges()[edge_id];
                    const auto& edge_vertices = edge.vertices();

                    if (std::find(edge_vertices.begin(), edge_vertices.end(), v) != edge_vertices.end()) {
                        // Add all vertices from this edge to the patch (with bounds checking)
                        for (VertexId neighbor : edge_vertices) {
                            if (neighbor < target_hg.num_vertices() &&
                                expanded_patch.find(neighbor) == expanded_patch.end()) {
                                new_vertices.insert(neighbor);
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    DEBUG_LOG("[REWRITE] Error processing edge %zu: %s", edge_id, e.what());
                    continue;
                }
            }
        }
        expanded_patch.insert(new_vertices.begin(), new_vertices.end());
    }

    DEBUG_LOG("[REWRITE] Patch expanded from %zu to %zu vertices within radius %zu",
              patch_vertices.size(), expanded_patch.size(), search_radius);

    // Find edges that intersect with the patch (with bounds checking)
    std::unordered_set<EdgeId> patch_edges;
    for (EdgeId edge_id = 0; edge_id < target_hg.num_edges(); ++edge_id) {
        try {
            const auto& edge = target_hg.edges()[edge_id];
            const auto& edge_vertices = edge.vertices();

            bool intersects_patch = false;
            for (VertexId v : edge_vertices) {
                if (v < target_hg.num_vertices() && expanded_patch.find(v) != expanded_patch.end()) {
                    intersects_patch = true;
                    break;
                }
            }
            if (intersects_patch) {
                patch_edges.insert(edge_id);
            }
        } catch (const std::exception& e) {
            DEBUG_LOG("[REWRITE] Error checking edge %zu intersection: %s", edge_id, e.what());
            continue;
        }
    }

    DEBUG_LOG("[REWRITE] Found %zu edges in patch for rule matching", patch_edges.size());

    if (patch_edges.empty()) {
        DEBUG_LOG("[REWRITE] No patch edges found, skipping rule application");
        return;
    }

    // Apply each rule to this patch
    for (std::size_t rule_idx = 0; rule_idx < context_->rewrite_rules.size(); ++rule_idx) {
        const auto& rule = context_->rewrite_rules[rule_idx];

        try {
            // Create edge signature index for patch-based matching
            EdgeSignatureIndex signature_index;
            auto label_func = [](VertexId v) -> VertexLabel { return static_cast<VertexLabel>(v); };

            for (EdgeId edge_id : patch_edges) {
                try {
                    const auto& edge = target_hg.edges()[edge_id];
                    EdgeSignature sig = EdgeSignature::from_concrete_edge(edge, label_func);
                    signature_index.add_edge(edge_id, sig);
                } catch (const std::exception& e) {
                    DEBUG_LOG("[REWRITE] Error creating signature for edge %zu: %s", edge_id, e.what());
                    continue;
                }
            }

            // Create new pattern matching context for this patch
            // The original rules remain in scope for the entire evolution, so we can reference them directly
            auto patch_context = std::make_shared<PatternMatchingContext>(
                target_hg_ptr.get(), &rule.lhs, &signature_index, label_func,
                context_->multiway_graph, context_->job_system);
            patch_context->current_state_id = new_state_id;
            // Reference the original rules directly - they remain in scope
            patch_context->rewrite_rules = context_->rewrite_rules;
            patch_context->max_steps = context_->max_steps;
            patch_context->current_step = current_step;

            // Submit SCAN tasks for this patch (limited to patch edges)
            // Pass the hypergraph pointer to keep it alive during job execution
            spawn_patch_scan_tasks(patch_context, patch_edges, rule_idx, target_hg_ptr);

        } catch (const std::exception& e) {
            DEBUG_LOG("[REWRITE] Exception during patch matching setup for rule %zu: %s", rule_idx, e.what());
            continue;
        }
    }
}

void RewriteTask::spawn_patch_scan_tasks(
    std::shared_ptr<PatternMatchingContext> patch_context,
    const std::unordered_set<EdgeId>& patch_edges,
    std::size_t rule_idx,
    std::shared_ptr<Hypergraph> hypergraph_keeper) {

    DEBUG_LOG("[REWRITE] Spawning patch SCAN tasks for rule %zu over %zu edges",
              rule_idx, patch_edges.size());

    // Safety checks
    if (!patch_context || !patch_context->pattern) {
        DEBUG_LOG("[REWRITE] Invalid patch context, skipping SCAN tasks");
        return;
    }

    if (patch_edges.empty()) {
        DEBUG_LOG("[REWRITE] No patch edges, skipping SCAN tasks");
        return;
    }

    const auto& pattern = *patch_context->pattern;
    if (pattern.num_edges() == 0) {
        DEBUG_LOG("[REWRITE] Empty pattern, skipping SCAN tasks");
        return;
    }

    std::vector<EdgeId> patch_edge_list(patch_edges.begin(), patch_edges.end());

    // Use the shared_ptr directly for safe sharing across lambda captures
    auto shared_keeper = hypergraph_keeper;

    try {
        // Create SCAN tasks for each pattern edge, limited to patch edges
        for (std::size_t pattern_idx = 0; pattern_idx < pattern.num_edges(); ++pattern_idx) {
            std::size_t num_workers = context_->job_system ? context_->job_system->get_num_workers() : 1;
            std::size_t edges_per_partition = std::max(std::size_t(1), patch_edge_list.size() / num_workers);

            for (std::size_t start = 0; start < patch_edge_list.size(); start += edges_per_partition) {
                std::size_t end = std::min(start + edges_per_partition, patch_edge_list.size());

                try {
                    auto task = job_system::make_job(
                        [patch_context, pattern_idx, start, end, patch_edge_list, keeper = shared_keeper]() {
                            try {
                                ScanTask task(patch_context, pattern_idx, start, end);
                                task.execute();
                                // keeper is automatically destroyed when this lambda ends, keeping hypergraph alive
                            } catch (const std::exception& e) {
                                DEBUG_LOG("[REWRITE] Exception in patch SCAN task: %s", e.what());
                            }
                        },
                        PatternMatchingTaskType::SCAN
                    );

                    if (context_->job_system) {
                        context_->job_system->submit(std::move(task), job_system::ScheduleMode::LIFO);
                        patch_context->scan_tasks_spawned.fetch_add(1);
                    }
                } catch (const std::exception& e) {
                    DEBUG_LOG("[REWRITE] Exception creating patch SCAN task: %s", e.what());
                    continue;
                }
            }
        }
    } catch (const std::exception& e) {
        DEBUG_LOG("[REWRITE] Exception during patch SCAN task creation: %s", e.what());
    }
}

void CausalTask::execute() {
    // Implementation for computing causal edges between events
    // Causal edges exist when one event's output edges overlap with another's input edges

    // Steps would include:
    // 1. For each pair of events in event_ids_
    // 2. Check if event A's output edges intersect with event B's input edges
    // 3. If so, create a causal edge A → B
    // 4. Store in the multiway graph's causal edge list

    // This uses FIFO scheduling to ensure timely processing of relationships

    // For demonstration, we just count causal computations
    static std::atomic<std::size_t> causal_count{0};
    causal_count.fetch_add(1);
}

void BranchialTask::execute() {
    // Implementation for computing branchial edges between events
    // Branchial edges exist when events share input edges (alternative evolutions)

    // Steps would include:
    // 1. For each pair of events in event_ids_
    // 2. Check if events share any input edges
    // 3. If so, create a branchial edge A ↔ B
    // 4. Store in the multiway graph's branchial edge list

    // This uses FIFO scheduling to ensure timely processing of relationships

    // For demonstration, we just count branchial computations
    static std::atomic<std::size_t> branchial_count{0};
    branchial_count.fetch_add(1);
}

// Helper functions for task execution

std::vector<VariableAssignment> generate_edge_assignments(
    const PatternEdge& pattern_edge,
    const Hyperedge& concrete_edge) {

    std::vector<VariableAssignment> assignments;

    if (pattern_edge.arity() != concrete_edge.arity()) {
        return assignments;  // Incompatible arities
    }

    // Get concrete vertices
    auto concrete_vertices = concrete_edge.vertices();
    std::vector<std::size_t> indices(concrete_vertices.size());
    std::iota(indices.begin(), indices.end(), 0);

    // For directed hypergraphs, vertex order matters - no permutations!
    VariableAssignment assignment;
    bool valid = true;

    for (std::size_t i = 0; i < pattern_edge.vertices.size(); ++i) {
        const auto& pattern_vertex = pattern_edge.vertices[i];
        VertexId concrete_vertex = concrete_vertices[i];  // Direct mapping, no permutation

        if (pattern_vertex.is_concrete()) {
            // Concrete vertex must match exactly
            if (pattern_vertex.id != concrete_vertex) {
                valid = false;
                break;
            }
        } else {
            // Variable vertex - try to assign
            if (!assignment.assign(pattern_vertex.id, concrete_vertex)) {
                valid = false;
                break;
            }
        }
    }

    if (valid) {
        assignments.push_back(assignment);
    }

    return assignments;
}

bool is_assignment_consistent(const VariableAssignment& a1, const VariableAssignment& a2) {
    // Check that variables have consistent bindings
    for (const auto& [var, val1] : a1.variable_to_concrete) {
        auto it = a2.variable_to_concrete.find(var);
        if (it != a2.variable_to_concrete.end() && it->second != val1) {
            return false;  // Conflict: same variable bound to different values
        }
    }

    // Check the reverse direction
    for (const auto& [var, val2] : a2.variable_to_concrete) {
        auto it = a1.variable_to_concrete.find(var);
        if (it != a1.variable_to_concrete.end() && it->second != val2) {
            return false;  // Conflict
        }
    }

    return true;
}

} // namespace hypergraph