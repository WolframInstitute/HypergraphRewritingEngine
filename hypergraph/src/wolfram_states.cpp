#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/pattern_matching_tasks.hpp>
#include <job_system/job_system.hpp>
#include <sstream>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <unordered_set>

#ifdef ENABLE_BRUTE_FORCE_ISOMORPHISM_CHECK
namespace {
// Brute force directed hypergraph isomorphism checker
bool are_truly_isomorphic(const std::vector<std::vector<hypergraph::GlobalVertexId>>& edges1,
                         const std::vector<std::vector<hypergraph::GlobalVertexId>>& edges2) {
    if (edges1.size() != edges2.size()) return false;

    // Get all vertices from both hypergraphs
    std::set<hypergraph::GlobalVertexId> vertex_set1, vertex_set2;
    for (const auto& edge : edges1) {
        for (auto v : edge) vertex_set1.insert(v);
    }
    for (const auto& edge : edges2) {
        for (auto v : edge) vertex_set2.insert(v);
    }

    if (vertex_set1.size() != vertex_set2.size()) return false;

    std::vector<hypergraph::GlobalVertexId> vertices1(vertex_set1.begin(), vertex_set1.end());
    std::vector<hypergraph::GlobalVertexId> vertices2(vertex_set2.begin(), vertex_set2.end());

    // Try all permutations of vertex2 as potential mappings
    std::sort(vertices2.begin(), vertices2.end());
    do {
        // Create mapping from vertices1 to this permutation of vertices2
        std::map<hypergraph::GlobalVertexId, hypergraph::GlobalVertexId> mapping;
        for (size_t i = 0; i < vertices1.size(); ++i) {
            mapping[vertices1[i]] = vertices2[i];
        }

        // Apply mapping to edges1 and see if we get edges2
        std::vector<std::vector<hypergraph::GlobalVertexId>> mapped_edges1;
        for (const auto& edge : edges1) {
            std::vector<hypergraph::GlobalVertexId> mapped_edge;
            for (auto v : edge) {
                mapped_edge.push_back(mapping[v]);
            }
            mapped_edges1.push_back(mapped_edge);
        }

        // Sort both edge sets for comparison
        auto sorted_mapped = mapped_edges1;
        auto sorted_edges2 = edges2;
        std::sort(sorted_mapped.begin(), sorted_mapped.end());
        std::sort(sorted_edges2.begin(), sorted_edges2.end());

        if (sorted_mapped == sorted_edges2) {
            return true;
        }

    } while (std::next_permutation(vertices2.begin(), vertices2.end()));

    return false;
}
} // anonymous namespace
#endif

namespace hypergraph {

/*
 * WolframState
 */

void WolframState::rebuild_signature_index() {
    edge_signature_index_.clear();

    // Create a simple label function (using vertex ID as label for now)
    auto label_func = [](GlobalVertexId v) -> VertexLabel {
        return static_cast<VertexLabel>(v);
    };

    for (std::size_t i = 0; i < global_edges_.size(); ++i) {
        const GlobalHyperedge& edge = global_edges_[i];

        // Create a temporary Hyperedge for signature generation
        std::vector<VertexId> local_vertices;
        for (GlobalVertexId gv : edge.global_vertices) {
            local_vertices.push_back(static_cast<VertexId>(gv));
        }
        Hyperedge temp_edge(static_cast<EdgeId>(i), local_vertices);

        EdgeSignature sig = EdgeSignature::from_concrete_edge(temp_edge, label_func);
        edge_signature_index_.add_edge(static_cast<EdgeId>(i), sig);
    }
}

void WolframState::add_global_edge(GlobalEdgeId edge_id, const std::vector<GlobalVertexId>& vertices) {
    global_edges_.emplace_back(edge_id, vertices);
    for (GlobalVertexId vertex : vertices) {
        global_vertices_.insert(vertex);
    }
    // No cache to invalidate - canonical forms computed locally

    // Update signature index
    std::vector<VertexId> local_vertices;
    for (GlobalVertexId gv : vertices) {
        local_vertices.push_back(static_cast<VertexId>(gv));
    }
    EdgeId edge_idx = static_cast<EdgeId>(global_edges_.size() - 1);
    Hyperedge temp_edge(edge_idx, local_vertices);
    auto label_func = [](GlobalVertexId v) -> VertexLabel {
        return static_cast<VertexLabel>(v);
    };
    EdgeSignature sig = EdgeSignature::from_concrete_edge(temp_edge, label_func);
    edge_signature_index_.add_edge(static_cast<EdgeId>(global_edges_.size() - 1), sig);
}

bool WolframState::remove_global_edge(GlobalEdgeId edge_id) {
    auto it = std::find_if(global_edges_.begin(), global_edges_.end(),
        [edge_id](const GlobalHyperedge& edge) { return edge.global_id == edge_id; });

    if (it != global_edges_.end()) {
        global_edges_.erase(it);

        // Rebuild vertex set (some vertices might no longer be referenced)
        global_vertices_.clear();
        for (const auto& edge : global_edges_) {
            for (GlobalVertexId vertex : edge.global_vertices) {
                global_vertices_.insert(vertex);
            }
        }

        // No cache to invalidate - canonical forms computed locally
        rebuild_signature_index(); // Rebuild index after edge removal
        return true;
    }
    return false;
}

// These functions were removed as they were only used for unused canonical edge signature computation

Hypergraph WolframState::to_hypergraph() const {
    // Build non-canonical hypergraph for pattern matching
    // Edge indices directly correspond to positions in global_edges_
    Hypergraph hg;
    for (const auto& global_edge : global_edges_) {
        std::vector<VertexId> vertices;
        for (GlobalVertexId global_v : global_edge.global_vertices) {
            vertices.push_back(static_cast<VertexId>(global_v));
        }
        hg.add_edge(vertices);
    }
    return hg;
}

Hypergraph WolframState::to_canonical_hypergraph() const {
    // Convert to list of edges with their original global edge IDs
    std::vector<std::vector<GlobalVertexId>> edges;
    std::vector<GlobalEdgeId> original_edge_ids;
    for (const auto& global_edge : global_edges_) {
        edges.push_back(global_edge.global_vertices);
        original_edge_ids.push_back(global_edge.global_id);
    }

    // Apply FindCanonicalHypergraph.wl algorithm via Canonicalizer
    Canonicalizer canonicalizer;
    auto canonical_result = canonicalizer.canonicalize_edges(edges);
    auto canonical_edges = canonical_result.canonical_form.edges;
    // No caching of vertex mapping for thread safety


    // Build canonical hypergraph
    Hypergraph canonical;
    for (const auto& canonical_edge : canonical_edges) {
        std::vector<VertexId> vertices;
        for (std::size_t vertex : canonical_edge) {
            vertices.push_back(static_cast<VertexId>(vertex));
        }
        canonical.add_edge(vertices);
    }

    return canonical;
}

std::size_t WolframState::compute_hash(bool canonicalization_enabled) const {
    // Convert global edges to vector format
    std::vector<std::vector<GlobalVertexId>> edges;
    for (const auto& global_edge : global_edges_) {
        edges.push_back(global_edge.global_vertices);
    }

    {
        std::ostringstream debug_stream;
        debug_stream << "[DEBUG] Original edges before hashing: ";
        for (const auto& edge : edges) {
            debug_stream << "{";
            for (std::size_t i = 0; i < edge.size(); ++i) {
                debug_stream << edge[i];
                if (i < edge.size() - 1) debug_stream << ",";
            }
            debug_stream << "} ";
        }
        DEBUG_LOG("%s", debug_stream.str().c_str());
    }

    std::vector<std::vector<std::size_t>> edges_to_hash;
    if (canonicalization_enabled) {
        // Use Canonicalizer to get canonical form
        Canonicalizer canonicalizer;
        auto result = canonicalizer.canonicalize_edges(edges);
        edges_to_hash = result.canonical_form.edges;
        // No caching - compute locally for thread safety

        {
            std::ostringstream debug_stream;
            debug_stream << "[DEBUG] Canonical edges for hashing: ";
            for (const auto& edge : edges_to_hash) {
                debug_stream << "{";
                for (std::size_t i = 0; i < edge.size(); ++i) {
                    debug_stream << edge[i];
                    if (i < edge.size() - 1) debug_stream << ",";
                }
                debug_stream << "} ";
            }
            DEBUG_LOG("%s", debug_stream.str().c_str());
        }
    } else {
        // Sort edges for consistent hashing without canonicalization
        for (const auto& edge : edges) {
            std::vector<std::size_t> converted_edge;
            for (auto v : edge) {
                converted_edge.push_back(static_cast<std::size_t>(v));
            }
            edges_to_hash.push_back(converted_edge);
        }
        std::sort(edges_to_hash.begin(), edges_to_hash.end());
    }

    // Compute hash on the edge structure
    std::size_t hash_value = 0x9e3779b97f4a7c15ULL; // Start with non-zero seed
    std::hash<std::size_t> hasher;

    // Use FNV-like hash combining with better avalanche properties
    for (std::size_t edge_idx = 0; edge_idx < edges_to_hash.size(); ++edge_idx) {
        const auto& edge = edges_to_hash[edge_idx];

        // Hash arity and edge position together with bit mixing
        std::size_t edge_hash = edge.size();
        edge_hash ^= edge_idx + 0x9e3779b9;
        edge_hash ^= (edge_hash << 6) ^ (edge_hash >> 2);

        for (std::size_t vertex_idx = 0; vertex_idx < edge.size(); ++vertex_idx) {
            std::size_t vertex = edge[vertex_idx];
            std::size_t vertex_hash = hasher(vertex);

            // Mix vertex hash with position using bit operations for avalanche
            vertex_hash ^= vertex_idx + 0x9e3779b9;
            vertex_hash ^= (vertex_hash << 13) ^ (vertex_hash >> 19);
            vertex_hash *= 0xc2b2ae35;
            vertex_hash ^= vertex_hash >> 16;

            // Combine with edge hash using FNV-like mixing
            edge_hash = (edge_hash ^ vertex_hash) * 0x100000001b3ULL;
        }

        // Combine edge hash with running total using strong mixing
        hash_value = (hash_value ^ edge_hash) * 0x100000001b3ULL;
        hash_value ^= hash_value >> 32;
    }

    DEBUG_LOG("[DEBUG] Computed hash value: %zu", hash_value);

    return hash_value;
}

CanonicalForm WolframState::get_canonical_form() const {
    // Compute canonical form locally without caching for thread safety
    std::vector<std::vector<GlobalVertexId>> edges;
    for (const auto& global_edge : global_edges_) {
        edges.push_back(global_edge.global_vertices);
    }

    Canonicalizer canonicalizer;
    auto result = canonicalizer.canonicalize_edges(edges);
    return result.canonical_form;
}

std::vector<GlobalEdgeId> WolframState::find_edges_by_pattern_signature(const EdgeSignature& pattern_sig) const {
    std::vector<GlobalEdgeId> result;

    // Get edge indices from signature index
    std::vector<EdgeId> edge_indices = edge_signature_index_.find_compatible_edges(pattern_sig);

    // Convert indices to global edge IDs
    for (EdgeId idx : edge_indices) {
        if (idx < global_edges_.size()) {
            result.push_back(global_edges_[idx].global_id);
        }
    }

    return result;
}

WolframState WolframState::clone() const {
    WolframState copy; // New state will compute hash when needed
    copy.global_edges_ = global_edges_;
    copy.global_vertices_ = global_vertices_;
    copy.edge_signature_index_ = edge_signature_index_;
    return copy;
}

std::vector<GlobalEdgeId> WolframState::edges_containing(GlobalVertexId vertex_id) const {
    std::vector<GlobalEdgeId> result;
    for (const auto& edge : global_edges_) {
        if (edge.contains(vertex_id)) {
            result.push_back(edge.global_id);
        }
    }
    return result;
}

const GlobalHyperedge* WolframState::get_edge(GlobalEdgeId edge_id) const {
    auto it = std::find_if(global_edges_.begin(), global_edges_.end(),
        [edge_id](const GlobalHyperedge& edge) { return edge.global_id == edge_id; });
    return (it != global_edges_.end()) ? &(*it) : nullptr;
}

// Canonical edge signature functions removed - they were unused and caused thread safety issues

/*
 * MultiwayGraph
 */

WolframState MultiwayGraph::create_initial_state(const std::vector<std::vector<GlobalVertexId>>& edge_structures) {
    WolframState state;

    // Track the maximum vertex ID seen in the initial state
    GlobalVertexId max_vertex_id = 0;

    for (const auto& vertices : edge_structures) {
        GlobalEdgeId edge_id = next_global_edge_id.fetch_add(1);
        state.add_global_edge(edge_id, vertices);

        // Find the maximum vertex ID in this edge
        for (GlobalVertexId vertex_id : vertices) {
            if (vertex_id > max_vertex_id) {
                max_vertex_id = vertex_id;
            }
        }
    }

    // Update next_global_vertex_id to ensure fresh vertices don't overlap
    if (!edge_structures.empty() && max_vertex_id > 0) {
        GlobalVertexId min_required = max_vertex_id + 1;
        GlobalVertexId current = next_global_vertex_id.load();
        if (current < min_required) {
            // Atomically add the difference to bring it up to the required minimum
            next_global_vertex_id.fetch_add(min_required - current);
        }
    }

    // Store the initial state ID for causal computation
    initial_state_id = state.raw_id();

    // Store state if full_capture is enabled
    if (full_capture && states) {
        auto raw_state_id = state.raw_id();
        auto canonical_state_id = state.canonical_id();

        // Store raw state
        states->insert_or_get(raw_state_id, state);

        // Store canonical mapping if canonicalization is enabled
        if (canonicalization_enabled && canonical_to_raw_mapping) {
            canonical_to_raw_mapping->insert_or_get(canonical_state_id, raw_state_id);
        }
    }

    return state;
}

std::unordered_set<GlobalEdgeId> MultiwayGraph::get_edges_within_radius(
    const WolframState& state,
    const std::vector<GlobalEdgeId>& center_edges,
    std::size_t radius) const {

    std::unordered_set<GlobalEdgeId> result;
    std::unordered_set<GlobalVertexId> frontier_vertices;

    // Add center edges and their vertices to result
    for (GlobalEdgeId edge_id : center_edges) {
        result.insert(edge_id);
        const auto* edge = state.get_edge(edge_id);
        if (edge) {
            for (GlobalVertexId v : edge->global_vertices) {
                frontier_vertices.insert(v);
            }
        }
    }

    // Expand radius layers
    for (std::size_t layer = 0; layer < radius; ++layer) {
        std::unordered_set<GlobalVertexId> next_frontier;

        for (GlobalVertexId vertex : frontier_vertices) {
            // Find all edges containing this vertex
            for (const auto& edge : state.edges()) {
                if (edge.contains(vertex) && result.find(edge.global_id) == result.end()) {
                    result.insert(edge.global_id);

                    // Add new vertices to next frontier
                    for (GlobalVertexId v : edge.global_vertices) {
                        if (frontier_vertices.find(v) == frontier_vertices.end()) {
                            next_frontier.insert(v);
                        }
                    }
                }
            }
        }

        frontier_vertices = std::move(next_frontier);
        if (frontier_vertices.empty()) break; // No more expansion possible
    }

    return result;
}

// Removed: get_all_states - states are not stored

std::vector<WolframEvent> MultiwayGraph::get_all_events() const {
    std::vector<WolframEvent> result;
    result.reserve(events->size());

    events->for_each([&result](const EventId& id, const WolframEvent& event) {
        (void)id; // Unused - only need the event
        result.push_back(event);
    });

    return result;
}

std::vector<EventEdge> MultiwayGraph::get_event_edges() const {
    std::vector<EventEdge> edges;
    EventEdgeNode* current = event_edges_head.load();
    while (current != nullptr) {
        edges.push_back(current->edge);
        current = current->next.load();
    }
    return edges;
}

std::size_t MultiwayGraph::get_causal_edge_count() const {
    std::size_t count = 0;
    EventEdgeNode* current = event_edges_head.load();
    while (current != nullptr) {
        if (current->edge.type == EventRelationType::CAUSAL) {
            count++;
        }
        current = current->next.load();
    }
    return count;
}

std::size_t MultiwayGraph::get_branchial_edge_count() const {
    std::size_t count = 0;
    EventEdgeNode* current = event_edges_head.load();
    while (current != nullptr) {
        if (current->edge.type == EventRelationType::BRANCHIAL) {
            count++;
        }
        current = current->next.load();
    }
    return count;
}

void StateMatchCache::invalidate_matches_using_edges(const std::vector<GlobalEdgeId>& deleted_edges) {
    auto it = std::remove_if(cached_matches.begin(), cached_matches.end(),
        [&deleted_edges](const SharedCachedMatch& match) {
            // Check if this match uses any deleted edge
            for (GlobalEdgeId deleted_edge : deleted_edges) {
                if (std::find(match->matched_edges.begin(), match->matched_edges.end(), deleted_edge)
                    != match->matched_edges.end()) {
                    return true; // This match is invalidated
                }
            }
            return false; // This match is still valid
        });
    cached_matches.erase(it, cached_matches.end());
}

std::vector<CachedMatch> StateMatchCache::get_matches_for_rule(std::size_t rule_index) const {
    std::vector<CachedMatch> result;
    for (const auto& shared_match : cached_matches) {
        if (shared_match->rule_index == rule_index) {
            result.push_back(*shared_match);  // Dereference shared_ptr
        }
    }
    return result;
}

MultiwayGraph& MultiwayGraph::operator=(MultiwayGraph&& other) noexcept {
    if (this != &other) {
        // Move unique_ptrs directly - no locking needed
        // No state storage - only events
        events = std::move(other.events);

        // Clean up old event edges list and take ownership of new one
        EventEdgeNode* old_head = event_edges_head.exchange(other.event_edges_head.exchange(nullptr));
        while (old_head) {
            EventEdgeNode* next = old_head->next.load();
            delete old_head;
            old_head = next;
        }
        event_edges_count = other.event_edges_count.load();

        input_edge_to_events = std::move(other.input_edge_to_events);
        output_edge_to_events = std::move(other.output_edge_to_events);
        // No state-related storage
        next_event_id = other.next_event_id.load();
        next_global_vertex_id = other.next_global_vertex_id.load();
        next_global_edge_id = other.next_global_edge_id.load();
    }
    return *this;
}

std::string MultiwayGraph::get_summary() const {
    std::ostringstream ss;

    auto events_list = get_all_events();

    ss << "=== MULTIWAY GRAPH SUMMARY ===\n";
    ss << "States: " << num_states() << ", Events: " << events_list.size() << "\n\n";

    // Count causal vs branchial edges
    std::size_t causal_count = 0, branchial_count = 0;
    EventEdgeNode* current = event_edges_head.load();
    while (current != nullptr) {
        if (current->edge.type == EventRelationType::CAUSAL) causal_count++;
        else branchial_count++;
        current = current->next.load();
    }

    ss << "Events:\n";
    for (const auto& event : events_list) {
        ss << "  Event " << event.event_id << ": State " << event.input_state_id.value
           << " → State " << event.output_state_id.value << " (rule " << event.rule_index << ")\n";

        // Show consumed edges
        ss << "    Consumed edges: ";
        if (event.consumed_edges.empty()) {
            ss << "(none)";
        } else {
            for (size_t i = 0; i < event.consumed_edges.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << event.consumed_edges[i];
            }
        }
        ss << "\n";

        // Show produced edges
        ss << "    Produced edges: ";
        if (event.produced_edges.empty()) {
            ss << "(none)";
        } else {
            for (size_t i = 0; i < event.produced_edges.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << event.produced_edges[i];
            }
        }
        ss << "\n";

        // Show anchor vertex
        ss << "    Anchor vertex: " << event.anchor_vertex << "\n";
    }

    ss << "\nRelationships: " << causal_count << " causal, " << branchial_count << " branchial\n";
    return ss.str();
}

std::string MultiwayGraph::export_multiway_graph_dot() const {
    std::ostringstream dot;
    dot << "digraph MultiwayGraph {\n";
    dot << "  node [shape=circle]; // states\n";
    dot << "  node [shape=box]; // events\n";

    // State and event nodes omitted for concurrent access safety

    // Add event edges (causal and branchial)
    EventEdgeNode* current = event_edges_head.load();
    while (current != nullptr) {
        const auto& edge = current->edge;
        std::string color = (edge.type == EventRelationType::CAUSAL) ? "red" : "blue";
        std::string label = (edge.type == EventRelationType::CAUSAL) ? "causal" : "branchial";
        std::string style = (edge.type == EventRelationType::CAUSAL) ? "solid" : "dotted";

        dot << "  E" << edge.from_event << " -> E" << edge.to_event
            << " [color=" << color << ", label=\"" << label << "\", style=" << style << "];\n";
        current = current->next.load();
    }

    dot << "}\n";
    return dot.str();
}

EventId MultiwayGraph::apply_rewriting(const WolframState& input_state,
                       const WolframState& output_state,
                       const std::vector<GlobalEdgeId>& edges_to_remove,
                       const std::vector<GlobalEdgeId>& produced_edge_ids,
                       std::size_t rule_index,
                       GlobalVertexId anchor_vertex,
                       std::size_t evolution_step) {
    // State has already been built by RewriteTask - no need to rebuild
    // Generate state IDs based on canonicalization setting
    RawStateId raw_output_state_id = output_state.raw_id();  // Always compute raw form for pattern matching
    CanonicalStateId canonical_output_state_id{0};
    if (canonicalization_enabled) {
        canonical_output_state_id = output_state.canonical_id();
    }

    DEBUG_LOG("[DEBUG] Computing state hash for produced edges count=%zu",
             produced_edge_ids.size());
    DEBUG_LOG("[DEBUG] Canonical output state hash: %zu, Raw output state hash: %zu",
             canonical_output_state_id.value, raw_output_state_id.value);

    // States flow through tasks - always treat as new
    bool was_inserted = true;

#ifdef ENABLE_BRUTE_FORCE_ISOMORPHISM_CHECK
    const WolframState& existing_state = output_state;
    if (!was_inserted) {
        // Check if the existing state is truly isomorphic to the new state
        std::vector<std::vector<hypergraph::GlobalVertexId>> existing_edges;
        for (const auto& edge : existing_state.edges()) {
            existing_edges.push_back(edge.global_vertices);
        }

        std::vector<std::vector<hypergraph::GlobalVertexId>> new_edges;
        for (const auto& edge : output_state.edges()) {
            new_edges.push_back(edge.global_vertices);
        }

        if (!are_truly_isomorphic(existing_edges, new_edges)) {
            std::ostringstream error_msg;
            error_msg << "CANONICALIZATION BUG in apply_rewriting: States with same hash are NOT isomorphic!\n";
            error_msg << "Hash: " << canonical_output_state_id.value << "\n";
            error_msg << "Existing state edges: ";
            for (const auto& edge : existing_edges) {
                error_msg << "{";
                for (size_t i = 0; i < edge.size(); ++i) {
                    error_msg << edge[i];
                    if (i < edge.size() - 1) error_msg << ",";
                }
                error_msg << "} ";
            }
            error_msg << "\nNew state edges: ";
            for (const auto& edge : new_edges) {
                error_msg << "{";
                for (size_t i = 0; i < edge.size(); ++i) {
                    error_msg << edge[i];
                    if (i < edge.size() - 1) error_msg << ",";
                }
                error_msg << "} ";
            }
            error_msg << "\n";

            // This is a critical error - canonicalization is broken
            throw std::runtime_error(error_msg.str());
        }
    }
#endif

    if (was_inserted) {
        // New unique state - we successfully inserted it
        {
            std::ostringstream debug_stream;
            debug_stream << "NEW UNIQUE STATE " << canonical_output_state_id.value << " (raw: " << raw_output_state_id.value << ")\nRAW: ";
            for (const auto& edge : output_state.edges()) {
                debug_stream << "{";
                for (std::size_t i = 0; i < edge.global_vertices.size(); ++i) {
                    debug_stream << edge.global_vertices[i];
                    if (i < edge.global_vertices.size() - 1) debug_stream << ",";
                }
                debug_stream << "} ";
            }

            // Show canonicalized form too
            debug_stream << "\nCANON: ";
            if (canonicalization_enabled) {
                try {
                    auto canonical_hg = output_state.to_canonical_hypergraph();
                    for (const auto& edge : canonical_hg.edges()) {
                        debug_stream << "{";
                        const auto& vertices = edge.vertices();
                        for (std::size_t i = 0; i < vertices.size(); ++i) {
                            debug_stream << vertices[i];
                            if (i < vertices.size() - 1) debug_stream << ",";
                        }
                        debug_stream << "} ";
                    }
                } catch (...) {
                    debug_stream << "CANONICALIZATION_ERROR";
                }
            } else {
                debug_stream << "DISABLED";
            }

            DEBUG_LOG("%s", debug_stream.str().c_str());
        }
        // No state counting - states aren't stored
    } else {
        // State already exists - deduplicated
        DEBUG_LOG("DUPLICATE: reusing canonical state %zu (raw would have been %zu)",
                 canonical_output_state_id.value, raw_output_state_id.value);
    }

    // Create event with both canonical and raw StateIds
    EventId event_id = next_event_id.fetch_add(1);

    // Compute input state IDs directly from provided input_state
    RawStateId raw_input_state_id = input_state.raw_id();
    CanonicalStateId canonical_input_state_id{0};
    if (canonicalization_enabled) {
        canonical_input_state_id = input_state.canonical_id();
    }

    // Create event using existing constructor, then set canonical StateIds
    WolframEvent event(event_id, raw_input_state_id, raw_output_state_id,
                      edges_to_remove, produced_edge_ids,
                      rule_index, anchor_vertex, evolution_step);

    // Set canonical StateIds manually
    event.canonical_input_state_id = canonical_input_state_id;
    event.canonical_output_state_id = canonical_output_state_id;

    events->insert(event_id, std::move(event));
    events_count.fetch_add(1);

    // Update edge-to-event mappings and event relationships
    // update_edge_mappings(event_id, edges_to_remove, produced_edge_ids);
    // update_event_relationships(event_id);

    // No match forwarding - states flow through tasks

    return event_id;
}

void MultiwayGraph::update_edge_mappings(EventId event_id,
                         const std::vector<GlobalEdgeId>& input_edges,
                         const std::vector<GlobalEdgeId>& output_edges) {
    // Add input edge mappings - these edges were consumed by this event
    for (GlobalEdgeId edge_id : input_edges) {
        auto result = input_edge_to_events->insert_or_get(edge_id, new LockfreeList<EventId>());
        // Add event to the list (thread-safe)
        result.first->push_front(event_id);
    }

    // Add output edge mappings - these edges were produced by this event
    for (GlobalEdgeId edge_id : output_edges) {
        auto result = output_edge_to_events->insert_or_get(edge_id, new LockfreeList<EventId>());
        // Add event to the list (thread-safe)
        result.first->push_front(event_id);
    }
}

void MultiwayGraph::update_event_relationships(EventId new_event_id) {
    // Relationship computation moved to post-processing phase to avoid race conditions
    // This method is now a placeholder - edge mappings are populated in update_edge_mappings
    // Actual causal and branchial relationships will be computed after evolution completes
    (void)new_event_id; // Suppress unused parameter warning
}

// Hash function for CanonicalEdgeSignature (vector of size_t)
/* struct SignatureHash {
    std::size_t operator()(const CanonicalEdgeSignature& sig) const {
        std::size_t hash = 0;
        for (std::size_t val : sig) {
            hash ^= std::hash<std::size_t>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
}; */

void MultiwayGraph::compute_all_event_relationships(job_system::JobSystem<PatternMatchingTaskType>* job_system) {
    if (!job_system) {
        return;
    }

    size_t branchial_edges_created = 0;

    // Get all events and precompute their canonical edge signature sets
    std::vector<std::pair<EventId, WolframEvent>> all_events_raw;
    events->for_each([&](const EventId& id, const WolframEvent& event) {
        all_events_raw.emplace_back(id, event);
    });

    // IncludeInitialEvent=False filtering: exclude artificial init events
    const EventId INIT_EVENT_THRESHOLD = SIZE_MAX - 100;
    std::vector<std::pair<EventId, WolframEvent>> all_events;

    for (const auto& [event_id, event] : all_events_raw) {
        if (event_id < INIT_EVENT_THRESHOLD) {
            all_events.emplace_back(event_id, event);
        } else {
            DEBUG_LOG("[RELATIONSHIPS] Excluding artificial init event %zu from causal computation", event_id);
        }
    }

    // Sort by event ID for deterministic processing
    std::sort(all_events.begin(), all_events.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<EventId> all_event_ids;
    std::vector<std::unordered_set<GlobalEdgeId>> input_edge_sets;
    std::vector<std::unordered_set<GlobalEdgeId>> output_edge_sets;

    for (const auto& [event_id, event] : all_events) {
        all_event_ids.push_back(event_id);

        // Debug log event details with input/output states
        {
            std::ostringstream ss;
            ss << "[RELATIONSHIPS] Event " << event_id << " (input_state: " << event.input_state_id.value << ", output_state: " << event.output_state_id.value << "):\n";
            ss << "  Consumed edges (global): ";
            for (auto e : event.consumed_edges) ss << e << " ";
            ss << "\n  Produced edges (global): ";
            for (auto e : event.produced_edges) ss << e << " ";
            DEBUG_LOG("%s", ss.str().c_str());
        }

        // Create input edge set (using global edge IDs)
        std::unordered_set<GlobalEdgeId> input_set(
            event.consumed_edges.begin(), event.consumed_edges.end());
        input_edge_sets.push_back(std::move(input_set));

        // Create output edge set (using global edge IDs)
        std::unordered_set<GlobalEdgeId> output_set(
            event.produced_edges.begin(), event.produced_edges.end());
        output_edge_sets.push_back(std::move(output_set));
    }

    if (all_event_ids.size() < 2) {
        return; // Need at least 2 events to have relationships
    }

    // Create event pairs to process - each pair will be processed exactly once
    std::vector<std::pair<size_t, size_t>> event_pair_indices;
    for (size_t i = 0; i < all_event_ids.size(); ++i) {
        for (size_t j = i + 1; j < all_event_ids.size(); ++j) {
            event_pair_indices.emplace_back(i, j);
        }
    }

    if (event_pair_indices.empty()) {
        return;
    }

    // Fast set intersection helper with deterministic ordering for global edge IDs
    auto fast_intersect = [](const std::unordered_set<GlobalEdgeId>& set1,
                            const std::unordered_set<GlobalEdgeId>& set2) -> std::vector<GlobalEdgeId> {
        std::vector<GlobalEdgeId> intersection;
        const auto& smaller = (set1.size() < set2.size()) ? set1 : set2;
        const auto& larger = (set1.size() < set2.size()) ? set2 : set1;

        intersection.reserve(std::min(smaller.size(), larger.size()));
        for (const GlobalEdgeId& edge : smaller) {
            if (larger.find(edge) != larger.end()) {
                intersection.push_back(edge);
            }
        }

        // Sort to ensure deterministic ordering
        //std::sort(intersection.begin(), intersection.end());
        return intersection;
    };

    DEBUG_LOG("[RELATIONSHIPS] Total events: %zu, Processing %zu event pairs single-threaded for determinism", all_event_ids.size(), event_pair_indices.size());

    // Count events by step for debugging
    std::map<std::size_t, int> events_per_step;
    for (const auto& [event_id, event] : all_events) {
        events_per_step[event.step]++;
    }
    for (const auto& [step, count] : events_per_step) {
        DEBUG_LOG("[RELATIONSHIPS] Step %zu has %d events", step, count);
    }

    // TREE-BASED CAUSAL RELATIONSHIPS: Build event tree based on state dependencies
    // Map from output state -> list of events that consume it as input
    std::map<RawStateId, std::vector<EventId>> state_consumers;
    std::map<EventId, RawStateId> event_input_states;
    std::map<EventId, RawStateId> event_output_states;

    // Build state dependency mapping
    for (size_t i = 0; i < all_event_ids.size(); ++i) {
        EventId event_id = all_event_ids[i];
        const auto& event = all_events[i].second;

        event_input_states[event_id] = event.input_state_id;
        event_output_states[event_id] = event.output_state_id;

        // Map: which events consume each state as input
        state_consumers[event.input_state_id].push_back(event_id);

        DEBUG_LOG("[RELATIONSHIPS] TREE: Event %zu: input_state=%zu -> output_state=%zu",
                  event_id, event.input_state_id.value, event.output_state_id.value);
    }


    // Process all pairs single-threaded for determinism debugging - TREE-BASED CAUSALITY
    int causal_created = 0;
    int causal_skipped_no_overlap = 0;

    for (size_t pair_idx = 0; pair_idx < event_pair_indices.size(); ++pair_idx) {
        size_t i = event_pair_indices[pair_idx].first;
        size_t j = event_pair_indices[pair_idx].second;

        EventId event_i = all_event_ids[i];
        EventId event_j = all_event_ids[j];

        // Check for causal relationship: event_i outputs ∩ event_j inputs + TREE ANCESTRY
        std::vector<GlobalEdgeId> causal_forward = fast_intersect(output_edge_sets[i], input_edge_sets[j]);
        if (!causal_forward.empty() /* && is_ancestor(event_i, event_j) */) {
            {
                std::ostringstream ss;
                ss << "[RELATIONSHIPS] CAUSAL TREE: Event " << event_i << " -> Event " << event_j;
                ss << " (overlap edges: ";
                for (const auto& edge : causal_forward) {
                    ss << edge << " ";
                }
                ss << ") (ancestor->descendant)";
                DEBUG_LOG("%s", ss.str().c_str());
            }
            EventEdge causal_edge(event_i, event_j, EventRelationType::CAUSAL, causal_forward);

            // Add to event edges list atomically
            EventEdgeNode* new_node = new EventEdgeNode(causal_edge);
            EventEdgeNode* current_head = event_edges_head.load();
            do {
                new_node->next.store(current_head);
            } while (!event_edges_head.compare_exchange_weak(current_head, new_node));

            event_edges_count.fetch_add(1);
            causal_created++;
        } else {
            causal_skipped_no_overlap++;
        }

        // Check for reverse causal relationship: event_j outputs ∩ event_i inputs + TREE ANCESTRY
        std::vector<GlobalEdgeId> causal_reverse = fast_intersect(output_edge_sets[j], input_edge_sets[i]);
        if (!causal_reverse.empty()/*  && is_ancestor(event_j, event_i) */) {
            {
                std::ostringstream ss;
                ss << "[RELATIONSHIPS] CAUSAL TREE: Event " << event_j << " -> Event " << event_i;
                ss << " (overlap edges: ";
                for (const auto& edge : causal_reverse) {
                    ss << edge << " ";
                }
                ss << ") (reverse ancestor->descendant)";
                DEBUG_LOG("%s", ss.str().c_str());
            }
            EventEdge causal_edge(event_j, event_i, EventRelationType::CAUSAL, causal_reverse);

            // Add to event edges list atomically
            EventEdgeNode* new_node = new EventEdgeNode(causal_edge);
            EventEdgeNode* current_head = event_edges_head.load();
            do {
                new_node->next.store(current_head);
            } while (!event_edges_head.compare_exchange_weak(current_head, new_node));

            event_edges_count.fetch_add(1);
            causal_created++;
        }
        // Check for branchial relationship: shared input edges
        std::vector<GlobalEdgeId> branchial_overlap = fast_intersect(input_edge_sets[i], input_edge_sets[j]);

        // Get input states for analysis
        RawStateId input_state_i = all_events[i].second.input_state_id;
        RawStateId input_state_j = all_events[j].second.input_state_id;
        bool same_input_state = (input_state_i == input_state_j);

        if (!branchial_overlap.empty() && same_input_state) {
            DEBUG_LOG("[RELATIONSHIPS] Event %zu (input_state: %zu) <-> Event %zu (input_state: %zu): intersection size = %zu, same_input_state = %s",
                    event_i, input_state_i.value, event_j, input_state_j.value, branchial_overlap.size(), same_input_state ? "true" : "false");

            EventEdge branchial_edge(event_i, event_j, EventRelationType::BRANCHIAL, branchial_overlap);

            // Add to event edges list atomically
            EventEdgeNode* new_node = new EventEdgeNode(branchial_edge);
            EventEdgeNode* current_head = event_edges_head.load();
            do {
                new_node->next.store(current_head);
            } while (!event_edges_head.compare_exchange_weak(current_head, new_node));

            event_edges_count.fetch_add(1);
            branchial_edges_created++;
        }
    }

    // Apply transitive reduction to causal graph
    DEBUG_LOG("[RELATIONSHIPS] Applying transitive reduction to %d causal edges...", causal_created);

    // Separate causal and branchial edges for proper handling
    std::vector<std::pair<EventId, EventId>> all_causal_edges;
    std::vector<EventEdge> all_branchial_edges;

    // Collect edges from the linked list, separating by type
    EventEdgeNode* current = event_edges_head.load();
    while (current != nullptr) {
        if (current->edge.type == EventRelationType::CAUSAL) {
            all_causal_edges.push_back({current->edge.from_event, current->edge.to_event});
        } else if (current->edge.type == EventRelationType::BRANCHIAL) {
            all_branchial_edges.push_back(current->edge);
        }
        current = current->next.load();
    }

    // Build adjacency list for transitive reduction (causal only)
    std::map<EventId, std::set<EventId>> causal_graph;
    for (const auto& [from_event, to_event] : all_causal_edges) {
        causal_graph[from_event].insert(to_event);
    }

    // Apply proper transitive reduction: remove edge (A,C) if ANY path A->...->C exists with length >= 2
    std::set<std::pair<EventId, EventId>> edges_to_remove;

    // Build reachability matrix using DFS to find all paths of length >= 2
    auto has_path_excluding_direct = [&](EventId start, EventId end) -> bool {
        if (start == end) return false;

        // DFS to find if there's a path from start to end that doesn't use the direct edge
        std::set<EventId> visited;
        std::function<bool(EventId, int)> dfs = [&](EventId current, int depth) -> bool {
            if (depth > 0 && current == end) return true;  // Found path with length >= 2
            if (visited.count(current)) return false;      // Avoid cycles

            visited.insert(current);
            if (causal_graph.count(current)) {
                for (EventId next : causal_graph[current]) {
                    // Skip direct edge on first hop to ensure path length >= 2
                    if (depth == 0 && current == start && next == end) continue;
                    if (dfs(next, depth + 1)) {
                        visited.erase(current);  // Backtrack
                        return true;
                    }
                }
            }
            visited.erase(current);  // Backtrack
            return false;
        };

        return dfs(start, 0);
    };

    // Check each direct edge to see if it can be removed
    for (const auto& [from_event, to_event] : all_causal_edges) {
        if (has_path_excluding_direct(from_event, to_event)) {
            edges_to_remove.insert({from_event, to_event});
            DEBUG_LOG("[RELATIONSHIPS] Transitive reduction: removing edge %zu -> %zu (alternate path exists)",
                     from_event, to_event);
        }
    }

    // Rebuild the linked list with reduced causal edges + all branchial edges
    int edges_removed = edges_to_remove.size();
    event_edges_head.store(nullptr);
    event_edges_count.store(0);

    // Add reduced causal edges
    for (const auto& [event1, event2] : all_causal_edges) {
        if (edges_to_remove.count({event1, event2}) == 0) {
            EventEdge causal_edge(event1, event2, EventRelationType::CAUSAL, {});
            EventEdgeNode* new_node = new EventEdgeNode(causal_edge);
            EventEdgeNode* current_head = event_edges_head.load();
            do {
                new_node->next.store(current_head);
            } while (!event_edges_head.compare_exchange_weak(current_head, new_node));
            event_edges_count.fetch_add(1);
        }
    }

    // Add all branchial edges back
    for (const auto& branchial_edge : all_branchial_edges) {
        EventEdgeNode* new_node = new EventEdgeNode(branchial_edge);
        EventEdgeNode* current_head = event_edges_head.load();
        do {
            new_node->next.store(current_head);
        } while (!event_edges_head.compare_exchange_weak(current_head, new_node));
        event_edges_count.fetch_add(1);
    }

    DEBUG_LOG("[RELATIONSHIPS] Transitive reduction removed %d causal edges (%d -> %d), kept %zu branchial edges",
              edges_removed, causal_created, causal_created - edges_removed, all_branchial_edges.size());
    causal_created -= edges_removed;

    DEBUG_LOG("[RELATIONSHIPS] SUMMARY: Created %d causal edges, skipped %d (no overlap)",
              causal_created, causal_skipped_no_overlap);

    DEBUG_LOG("[RELATIONSHIPS] Completed: created %zu branchial edges, count function reports %zu branchial edges, %zu causal edges",
        branchial_edges_created, get_branchial_edge_count(), get_causal_edge_count());
}

std::vector<EventId> MultiwayGraph::get_concurrent_events(const std::vector<EventId>& candidate_events) const {
    std::vector<EventId> concurrent;

    for (EventId event_id : candidate_events) {
        bool has_causal_dependency = false;

        // Check if this event has causal dependencies on other candidate events
        EventEdgeNode* current = event_edges_head.load();
        while (current != nullptr) {
            if (current->edge.type == EventRelationType::CAUSAL &&
                current->edge.to_event == event_id &&
                std::find(candidate_events.begin(), candidate_events.end(), current->edge.from_event)
                != candidate_events.end()) {
                has_causal_dependency = true;
                break;
            }
            current = current->next.load();
        }

        if (!has_causal_dependency) {
            concurrent.push_back(event_id);
        }
    }

    return concurrent;
}

void MultiwayGraph::clear() {
    // Clear events
    events.reset(new ConcurrentHashMap<EventId, WolframEvent>());
    event_edges_head.store(nullptr);
    events_count.store(0);

    // Clear optional state storage if full_capture is enabled
    if (full_capture) {
        states.reset(new ConcurrentHashMap<RawStateId, WolframState>());
        canonical_to_raw_mapping.reset(new ConcurrentHashMap<CanonicalStateId, RawStateId>());
        match_caches.reset(new ConcurrentHashMap<RawStateId, StateMatchCache>());
        exhausted_states.clear();
    }

    input_edge_to_events->clear();
    output_edge_to_events->clear();
    next_event_id.store(1);
    next_global_vertex_id = 0;
    next_global_edge_id = 0;
}

// === OPTIONAL STATE STORAGE METHODS (only work when full_capture is enabled) ===

std::vector<WolframState> MultiwayGraph::get_all_states() const {
    if (!full_capture || !states) {
        return {}; // Return empty vector if full_capture is disabled
    }

    std::vector<WolframState> result;
    states->for_each([&](const RawStateId&, const WolframState& state) {
        result.push_back(state);
    });
    return result;
}

void MultiwayGraph::mark_state_exhausted(RawStateId state_id) {
    if (!full_capture) {
        return; // No-op if full_capture is disabled
    }

    exhausted_states.insert(state_id);
    // Optionally remove from states storage to free memory
    if (states) {
        states->erase(state_id);
    }
    if (match_caches) {
        match_caches->erase(state_id);
    }
}

bool MultiwayGraph::is_state_exhausted(RawStateId state_id) const {
    if (!full_capture) {
        return false; // Return false if full_capture is disabled
    }

    return exhausted_states.find(state_id) != exhausted_states.end();
}

std::size_t MultiwayGraph::num_live_states() const {
    if (!full_capture) {
        return 0; // Return 0 if full_capture is disabled
    }

    // Return size of states hash map minus exhausted states
    if (states) {
        return states->size() - exhausted_states.size();
    }
    return 0;
}

void MultiwayGraph::cache_pattern_match(RawStateId state_id, std::size_t rule_index,
                       const std::vector<GlobalEdgeId>& matched_edges,
                       const VariableAssignment& assignment,
                       GlobalVertexId anchor_vertex) {
    if (!full_capture || !match_caches) {
        return; // No-op if full_capture is disabled
    }

    auto cache_result = match_caches->insert_or_get(state_id, StateMatchCache(state_id));
    CachedMatch match(rule_index, matched_edges, assignment, anchor_vertex);
    cache_result.first.add_match(match);
}

std::vector<CachedMatch> MultiwayGraph::get_cached_matches(RawStateId state_id, std::size_t rule_index) const {
    if (!full_capture || !match_caches) {
        return {}; // Return empty vector if full_capture is disabled
    }

    auto cache_opt = match_caches->find(state_id);
    if (!cache_opt) {
        return {};
    }

    return cache_opt.value().get_matches_for_rule(rule_index);
}

void MultiwayGraph::forward_matches(RawStateId input_state_id, RawStateId output_state_id,
                    const std::vector<GlobalEdgeId>& deleted_edges) {
    if (!full_capture || !match_caches) {
        return; // No-op if full_capture is disabled
    }

    auto input_cache_opt = match_caches->find(input_state_id);
    if (!input_cache_opt) {
        return; // No matches to forward
    }

    auto output_cache_result = match_caches->insert_or_get(output_state_id, StateMatchCache(output_state_id));

    // Forward matches that don't use deleted edges
    for (const auto& shared_match : input_cache_opt.value().cached_matches) {
        bool uses_deleted_edge = false;
        for (GlobalEdgeId deleted_edge : deleted_edges) {
            if (std::find(shared_match->matched_edges.begin(), shared_match->matched_edges.end(), deleted_edge)
                != shared_match->matched_edges.end()) {
                uses_deleted_edge = true;
                break;
            }
        }

        if (!uses_deleted_edge) {
            output_cache_result.first.add_shared_match(shared_match);
        }
    }
}

bool MultiwayGraph::has_cached_matches(RawStateId state_id) const {
    if (!full_capture || !match_caches) {
        return false; // Return false if full_capture is disabled
    }

    auto cache_opt = match_caches->find(state_id);
    return cache_opt && cache_opt.value().has_matches();
}

void MultiwayGraph::clear_match_cache(RawStateId state_id) {
    if (!full_capture || !match_caches) {
        return; // No-op if full_capture is disabled
    }

    match_caches->erase(state_id);
}

std::optional<WolframState> MultiwayGraph::get_state_efficient(RawStateId state_id) const {
    if (full_capture && states) {
        // Try to get from stored states first
        auto state_opt = states->find(state_id);
        if (state_opt) {
            return state_opt.value();
        }
    }

    // Fall back to reconstruction
    return reconstruct_state(state_id);
}

std::optional<std::size_t> MultiwayGraph::get_state_hash(RawStateId state_id) const {
    auto state_opt = get_state_efficient(state_id);
    if (!state_opt) {
        return std::nullopt;
    }

    return state_opt->compute_hash(canonicalization_enabled);
}

std::optional<WolframState> MultiwayGraph::reconstruct_state(RawStateId target_state_id) const {
    // For now, just return nullopt since we don't have event replay implemented yet
    // This would need to be implemented to replay events from initial state
    (void)target_state_id; // Suppress unused parameter warning
    return std::nullopt;
}

std::vector<EventId> MultiwayGraph::find_event_path_to_state(RawStateId target_state_id) const {
    // For now, just return empty vector since we don't have path finding implemented yet
    // This would need to traverse the event graph to find a path
    (void)target_state_id; // Suppress unused parameter warning
    return {};
}

} // namespace hypergraph