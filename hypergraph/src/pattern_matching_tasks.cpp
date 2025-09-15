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
        // Find compatible edges by checking all signatures in the index (includes deduplication)
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

    {
        std::ostringstream debug_stream;
        debug_stream << "[SCAN] Pattern edge arity=" << pattern_edge.arity()
                     << ", target has " << target_hg.num_edges() << " edges. "
                     << "Found " << compatible_edges.size() << " compatible edges (arity match).";

        if (compatible_edges.empty()) {
            debug_stream << " No compatible edges found, returning.";
            DEBUG_LOG("%s", debug_stream.str().c_str());
            return;
        }
        DEBUG_LOG("%s", debug_stream.str().c_str());
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
        DEBUG_LOG("[SCAN] THREAD_%zu processing candidate_edge=%zu from partition [%zu,%zu) at index %zu",
                  std::hash<std::thread::id>{}(std::this_thread::get_id()) % 1000,
                  candidate_edge, partition_start_, actual_end, i);
        const auto& concrete_edge = target_hg.get_edge(candidate_edge);

        // Generate all possible variable assignments for this edge match
        auto assignments = generate_edge_assignments(pattern_edge, *concrete_edge);

        {
            std::ostringstream debug_stream;
            debug_stream << "[SCAN] Processing candidate edge " << candidate_edge << " (index " << i << "). "
                         << "Generated " << assignments.size() << " assignments for edge " << candidate_edge << ".";
            DEBUG_LOG("%s", debug_stream.str().c_str());
        }

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

    context_->scan_tasks_completed.fetch_add(1);
    DEBUG_LOG("[SCAN] Task completed. Spawned: %zu, Completed: %zu",
              context_->scan_tasks_spawned.load(), context_->scan_tasks_completed.load());
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
            extended_str += "(edges: ";
            for (const auto& [pattern_idx, data_edge] : extended_match.edge_map) {
                const auto& concrete_edge = target_hg.get_edge(data_edge);
                extended_str += "{";
                const auto& vertices = concrete_edge->vertices();
                for (std::size_t i = 0; i < vertices.size(); ++i) {
                    extended_str += std::to_string(vertices[i]);
                    if (i < vertices.size() - 1) extended_str += ",";
                }
                extended_str += "} ";
            }
            extended_str += ")";
            DEBUG_LOG("%s", extended_str.c_str());

            extended_match.add_edge_match(next_edge_idx, candidate, merged_assignment);

            // Execute another EXPAND task for the extended match
            ExpandTask expand_task(context_, std::move(extended_match));
            expand_task.execute();
        }
    }

    context_->expand_tasks_completed.fetch_add(1);
    DEBUG_LOG("[EXPAND] Task completed. Spawned: %zu, Completed: %zu",
              context_->expand_tasks_spawned.load(), context_->expand_tasks_completed.load());
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

    DEBUG_LOG("[SINK] REWRITE task submitted successfully");

    context_->sink_tasks_completed.fetch_add(1);
    DEBUG_LOG("[SINK] Task completed. Spawned: %zu, Completed: %zu",
              context_->sink_tasks_spawned.load(), context_->sink_tasks_completed.load());
}

void RewriteTask::execute() {
    // Check if we should even process this match - avoid wasteful work
    if (context_->current_step >= context_->max_steps) {
        DEBUG_LOG("[REWRITE] Skipping rewrite - current_step %zu >= max_steps %zu",
                  context_->current_step, context_->max_steps);
        context_->rewrite_tasks_completed.fetch_add(1);
        return;
    }

    std::string assignment_info = "[REWRITE] Executing REWRITE task. Variable assignments: ";
    for (const auto& [var, val] : complete_match_.assignment.variable_to_concrete) {
        assignment_info += "var" + std::to_string(var) + "→" + std::to_string(val) + " ";
    }
    DEBUG_LOG("%s", assignment_info.c_str());

    std::string consumed_edges = "[REWRITE] Starting rewrite for state with " + std::to_string(context_->current_state.edges().size()) + " edges, consumed edges: ";
    for (GlobalEdgeId eid : complete_match_.matched_edges) {
        consumed_edges += std::to_string(eid) + " ";
    }
    DEBUG_LOG("%s", consumed_edges.c_str());

    const auto& rule = context_->rewrite_rule;

    // Create new edges by applying rule RHS
    std::vector<std::vector<GlobalVertexId>> new_edges;

    // Maintain consistent mapping from RHS variables to fresh vertices within this rewrite
    std::unordered_map<VertexId, GlobalVertexId> rhs_fresh_vertex_map;

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
                    // Variable was matched in LHS - vertex ID is already global (no conversion needed)
                    GlobalVertexId global_vertex = static_cast<GlobalVertexId>(*resolved);
                    edge_vertices.push_back(global_vertex);
                    DEBUG_LOG("[REWRITE] Resolved LHS variable %zu → global %zu",
                              pattern_vertex.id, global_vertex);
                } else {
                    // Variable not in LHS assignment - use consistent fresh vertex for this rewrite
                    auto it = rhs_fresh_vertex_map.find(pattern_vertex.id);
                    GlobalVertexId fresh_vertex;
                    if (it != rhs_fresh_vertex_map.end()) {
                        // Reuse existing fresh vertex for this variable
                        fresh_vertex = it->second;
                        DEBUG_LOG("[REWRITE] Variable %zu NOT in LHS, reusing fresh vertex %zu",
                                  pattern_vertex.id, fresh_vertex);
                    } else {
                        // Create fresh vertex using atomic counter
                        fresh_vertex = context_->multiway_graph->get_fresh_vertex_id();

                        rhs_fresh_vertex_map[pattern_vertex.id] = fresh_vertex;
                        DEBUG_LOG("[REWRITE] Variable %zu NOT in LHS, creating fresh vertex %zu",
                                  pattern_vertex.id, fresh_vertex);
                    }
                    edge_vertices.push_back(fresh_vertex);
                }
            } else {
                // Concrete vertex - use as is
                edge_vertices.push_back(static_cast<GlobalVertexId>(pattern_vertex.id));
                DEBUG_LOG("[REWRITE] Concrete vertex %zu", pattern_vertex.id);
            }
        }

        // Only add edge if ALL vertices were resolved
        if (all_vertices_resolved && !edge_vertices.empty()) {
            std::string edge_str = "[REWRITE] Adding edge: {";
            for (std::size_t i = 0; i < edge_vertices.size(); ++i) {
                edge_str += std::to_string(edge_vertices[i]);
                if (i < edge_vertices.size() - 1) edge_str += ",";
            }
            edge_str += "}";
            DEBUG_LOG("%s", edge_str.c_str());
            new_edges.push_back(edge_vertices);
        } else if (!all_vertices_resolved) {
            DEBUG_LOG("[REWRITE] SKIPPING edge due to unresolved vertices");
        }
    }

    // Remove matched edges and add new edges to create new state
    // Work directly with the non-canonical hypergraph - no state lookups needed
    std::vector<GlobalEdgeId> edges_to_remove;

    std::string source_info = "[REWRITE] Source hypergraph has " + std::to_string(context_->target_hypergraph->num_edges()) + " edges, matched edges: ";
    for (EdgeId edge_id : complete_match_.matched_edges) {
        source_info += std::to_string(edge_id) + " ";
    }
    DEBUG_LOG("%s", source_info.c_str());

    // Convert edge indices to global edge IDs
    // Note: Edge indices now directly correspond to positions in global_edges_
    for (EdgeId edge_index : complete_match_.matched_edges) {
        // Use edge mapping instead of state lookup
        if (edge_index < context_->edge_id_mapping.size()) {
            GlobalEdgeId global_edge_id = context_->edge_id_mapping[edge_index];
            edges_to_remove.push_back(global_edge_id);
            DEBUG_LOG("[REWRITE] Removing edge %zu → global edge %zu",
                      edge_index, global_edge_id);
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

    // Apply rewrite directly to create new state (states flow through tasks, no storage)
    // Copy current state before modification to avoid affecting other tasks
    WolframState new_state = context_->current_state;

    // Remove consumed edges from the new state
    for (GlobalEdgeId edge_id : edges_to_remove) {
        new_state.remove_global_edge(edge_id);
    }

    // Add produced edges to the new state
    std::vector<GlobalEdgeId> produced_edge_ids;
    for (const auto& edge_vertices : new_edges) {
        GlobalEdgeId new_edge_id = context_->multiway_graph->get_fresh_edge_id();
        new_state.add_global_edge(new_edge_id, edge_vertices);
        produced_edge_ids.push_back(new_edge_id);
    }

    // Create event by calling apply_rewriting with input and output states
    EventId event_id = context_->multiway_graph->apply_rewriting(
        context_->current_state,          // input state (before rewrite)
        new_state,                        // output state (after rewrite)
        edges_to_remove,                   // edges to remove
        produced_edge_ids,                // use the already-created edge IDs
        context_->rule_index,             // rule index
        complete_match_.anchor_vertex,    // anchor vertex
        context_->current_step            // evolution step
    );

    if (event_id != INVALID_EVENT) {
        DEBUG_LOG("[REWRITE] Created new event %zu", event_id);

        // Get the new state created by this rewrite
        auto event_opt = context_->multiway_graph->get_event(event_id);
        if (event_opt) {
            RawStateId raw_new_state_id = event_opt->output_state_id;
            CanonicalStateId canonical_new_state_id = event_opt->canonical_output_state_id;

            // Calculate step for newly created state (parent step + 1)
            std::size_t new_step = context_->current_step + 1;

            DEBUG_LOG("[REWRITE] Step check: new_step=%zu, max_steps=%zu, created event=%zu",
                      new_step, context_->max_steps, event_id);

            // MULTI-RULE APPLICATION: Apply all rules to the newly created state
            // Only spawn if we haven't exceeded step limit
            if (new_step < context_->max_steps) {
                DEBUG_LOG("[REWRITE] Thread %zu: About to spawn rules for NEW state with %zu edges at step %zu",
                         std::hash<std::thread::id>{}(std::this_thread::get_id()) % 1000,
                         new_state.edges().size(), new_step);
                if (context_->wolfram_evolution) {
                    try {
                        context_->wolfram_evolution->apply_all_rules_to_state(new_state, new_step);
                        DEBUG_LOG("[REWRITE] Thread %zu: Successfully spawned rules for new state with %zu edges",
                                 std::hash<std::thread::id>{}(std::this_thread::get_id()) % 1000,
                                 new_state.edges().size());
                    } catch (const std::exception& e) {
                        DEBUG_LOG("[REWRITE] Exception in multi-rule application: %s", e.what());
                    }
                } else {
                    DEBUG_LOG("[REWRITE] No WolframEvolution context for multi-rule application");
                }
            } else {
                DEBUG_LOG("[REWRITE] NOT SPAWNING multi-rule application: step %zu >= max_steps %zu", new_step, context_->max_steps);
            }

            DEBUG_LOG("[REWRITE] Event %zu completed, new state with %zu edges created", event_id, new_state.edges().size());
        }
    }

    DEBUG_LOG("[REWRITE] REWRITE task completed");

    context_->rewrite_tasks_completed.fetch_add(1);
    DEBUG_LOG("[REWRITE] Task completed. Spawned: %zu, Completed: %zu",
              context_->rewrite_tasks_spawned.load(), context_->rewrite_tasks_completed.load());
}


void RewriteTask::spawn_patch_based_matching_around_new_edges(RawStateId new_state_id, const std::vector<std::vector<GlobalVertexId>>& new_edges, std::size_t current_step) {
    DEBUG_LOG("[REWRITE] Starting patch-based matching for state %zu, current_step=%zu", new_state_id.value, current_step);

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


    // TODO: Update this method to receive WolframState parameter instead of StateId
    // For now, this method is not functional in state-flow architecture
    DEBUG_LOG("[REWRITE] patch-based matching disabled - needs state-flow architecture update");
    return;
}

void RewriteTask::spawn_patch_scan_tasks(
    std::shared_ptr<PatternMatchingContext> patch_context,
    const std::unordered_set<EdgeId>& patch_edges,
    std::size_t rule_idx,
    std::shared_ptr<Hypergraph> hypergraph_keeper) {

    DEBUG_LOG("[REWRITE] patch-based matching disabled");
    return;
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