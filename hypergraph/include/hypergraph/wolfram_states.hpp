#ifndef HYPERGRAPH_WOLFRAM_STATES_HPP
#define HYPERGRAPH_WOLFRAM_STATES_HPP

#include "hypergraph/debug_log.hpp"
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/types.hpp>
#include <hypergraph/canonicalization.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <hypergraph/concurrent_hash_map.hpp>
#include <hypergraph/lockfree_list.hpp>
#include <map>
#include <set>
#include <atomic>
#include <chrono>
#include <sstream>
#include <optional>
#include <iostream>
#include <numeric>

namespace hypergraph {

// Forward declarations
class WolframState;
class WolframEvent;
enum class PatternMatchingTaskType;

} // namespace hypergraph

namespace job_system {
    template<typename TaskType> class JobSystem;
}

namespace hypergraph {


/**
 * Hyperedge with global ID that persists across canonicalization.
 */
struct GlobalHyperedge {
    GlobalEdgeId global_id;
    std::vector<GlobalVertexId> global_vertices;

    GlobalHyperedge(GlobalEdgeId id, const std::vector<GlobalVertexId>& vertices)
        : global_id(id), global_vertices(vertices) {}

    std::size_t arity() const { return global_vertices.size(); }

    bool contains(GlobalVertexId vertex_id) const {
        return std::find(global_vertices.begin(), global_vertices.end(), vertex_id)
               != global_vertices.end();
    }

    bool operator==(const GlobalHyperedge& other) const {
        return global_vertices == other.global_vertices; // Compare structure, not ID
    }
};

/**
 * Canonical edge signature - represents the structural identity of an edge
 * in its canonical form, independent of global IDs.
 */
using CanonicalEdgeSignature = std::vector<std::size_t>;  // Sorted canonical vertex IDs

/**
 * Wolfram Physics state containing a hypergraph with global IDs.
 */
class WolframState {
private:
    std::vector<GlobalHyperedge> global_edges_;
    std::unordered_set<GlobalVertexId> global_vertices_;
    EdgeSignatureIndex edge_signature_index_;  // Fast edge lookup by signature

    /**
     * Rebuild the edge signature index.
     */
    void rebuild_signature_index();

public:
    explicit WolframState() {}

    CanonicalStateId canonical_id() const {
        return CanonicalStateId(compute_hash(true));
    }

    RawStateId raw_id() const {
        return RawStateId(compute_hash(false));
    }

    // Legacy method for backward compatibility - will be removed gradually
    std::size_t id(bool canonicalization_enabled = true) const {
        return compute_hash(canonicalization_enabled);
    }

    /**
     * Add edge with global IDs.
     */
    void add_global_edge(GlobalEdgeId edge_id, const std::vector<GlobalVertexId>& vertices);

    /**
     * Remove edge by global ID.
     */
    bool remove_global_edge(GlobalEdgeId edge_id);

    const std::vector<GlobalHyperedge>& edges() const { return global_edges_; }
    const std::unordered_set<GlobalVertexId>& vertices() const { return global_vertices_; }

    std::size_t num_edges() const { return global_edges_.size(); }
    std::size_t num_vertices() const { return global_vertices_.size(); }

    // Canonical vertex mapping functions removed for thread safety


private:

public:
    /**
     * Convert to hypergraph for pattern matching (non-canonical, preserves edge order).
     * Edge indices directly correspond to positions in global_edges_.
     */
    Hypergraph to_hypergraph() const;

    /**
     * Convert to canonical hypergraph and build edge mapping.
     */
    Hypergraph to_canonical_hypergraph() const;

    /**
     * Compute hash for state identification
     * Uses canonical form if canonicalization_enabled, otherwise raw hypergraph
     */
    std::size_t compute_hash(bool canonicalization_enabled = true) const;

    /**
     * Legacy method for compatibility - uses canonical hash
     */
    std::size_t compute_canonical_hash() const {
        return compute_hash(true);
    }

    /**
     * Get canonical form (cached for performance).
     */
    CanonicalForm get_canonical_form() const;

    /**
     * Check structural equality (same canonical form).
     */
    bool structurally_equal(const WolframState& other) const {
        return get_canonical_form() == other.get_canonical_form();
    }

    /**
     * Get edges containing a specific global vertex.
     */
    std::vector<GlobalEdgeId> edges_containing(GlobalVertexId vertex_id) const;

    /**
     * Get edge by global ID.
     */
    const GlobalHyperedge* get_edge(GlobalEdgeId edge_id) const;

    /**
     * Find edges compatible with a pattern signature.
     * This is key for fast pattern matching using the signature index.
     */
    std::vector<GlobalEdgeId> find_edges_by_pattern_signature(const EdgeSignature& pattern_sig) const;

    // Canonical edge signature functions removed for thread safety

    WolframState clone() const;
};

/**
 * Wolfram Physics event representing a rewriting operation.
 */
struct WolframEvent {
    EventId event_id;
    RawStateId input_state_id;            // Raw/non-canonical state before rewriting (for pattern matching)
    RawStateId output_state_id;           // Raw/non-canonical state after rewriting (for pattern matching)

    CanonicalStateId canonical_input_state_id;   // Canonical state before rewriting (for equivalencing)
    CanonicalStateId canonical_output_state_id;  // Canonical state after rewriting (for equivalencing)

    std::vector<GlobalEdgeId> consumed_edges;  // Global edges removed
    std::vector<GlobalEdgeId> produced_edges;  // Global edges added

    // Canonical signatures for structural equivalence comparison
    std::vector<CanonicalEdgeSignature> consumed_edge_signatures;
    std::vector<CanonicalEdgeSignature> produced_edge_signatures;

    std::size_t rule_index;   // Index into rule set that was applied
    GlobalVertexId anchor_vertex; // Vertex around which rewriting occurred
    std::size_t step;         // Evolution step when this event occurred

    std::chrono::steady_clock::time_point timestamp;

    WolframEvent() = default;

    WolframEvent(EventId id, RawStateId input_state, RawStateId output_state,
                const std::vector<GlobalEdgeId>& consumed,
                const std::vector<GlobalEdgeId>& produced,
                std::size_t rule_idx, GlobalVertexId anchor, std::size_t evolution_step)
        : event_id(id), input_state_id(input_state), output_state_id(output_state)
        , consumed_edges(consumed), produced_edges(produced)
        , rule_index(rule_idx), anchor_vertex(anchor), step(evolution_step)
        , timestamp(std::chrono::steady_clock::now()) {}

    WolframEvent(EventId id, RawStateId input_state, RawStateId output_state,
                const std::vector<GlobalEdgeId>& consumed,
                const std::vector<GlobalEdgeId>& produced,
                const std::vector<CanonicalEdgeSignature>& consumed_sigs,
                const std::vector<CanonicalEdgeSignature>& produced_sigs,
                std::size_t rule_idx, GlobalVertexId anchor, std::size_t evolution_step)
        : event_id(id), input_state_id(input_state), output_state_id(output_state)
        , consumed_edges(consumed), produced_edges(produced)
        , consumed_edge_signatures(consumed_sigs), produced_edge_signatures(produced_sigs)
        , rule_index(rule_idx), anchor_vertex(anchor), step(evolution_step)
        , timestamp(std::chrono::steady_clock::now()) {}
};

/**
 * Causal and branchial edge types between events in the multiway graph.
 */
enum class EventRelationType {
    CAUSAL,    // Directed: event A's output edges overlap with event B's input edges
    BRANCHIAL  // Undirected: events share overlapping input edges
};

/**
 * Edge connecting two events in the multiway graph.
 */
struct EventEdge {
    EventId from_event;
    EventId to_event;
    EventRelationType type;
    std::vector<GlobalEdgeId> overlapping_edges;  // The specific edges that overlap

    EventEdge(EventId from, EventId to, EventRelationType relation_type,
              const std::vector<GlobalEdgeId>& edges = {})
        : from_event(from), to_event(to), type(relation_type), overlapping_edges(edges) {}
};

/**
 * Cached pattern match for a specific rule on a specific state.
 * Uses shared_ptr for memory efficiency when forwarding matches between states.
 */
struct CachedMatch {
    std::size_t rule_index;                    // Which rule this match is for
    std::vector<GlobalEdgeId> matched_edges;   // Global edges involved in the match
    VariableAssignment assignment;             // Variable assignments for the match
    GlobalVertexId anchor_vertex;              // Anchor vertex for the match

    CachedMatch(std::size_t rule_idx, const std::vector<GlobalEdgeId>& edges,
                const VariableAssignment& var_assignment, GlobalVertexId anchor)
        : rule_index(rule_idx), matched_edges(edges), assignment(var_assignment), anchor_vertex(anchor) {}
};

using SharedCachedMatch = std::shared_ptr<const CachedMatch>;

/**
 * Match cache for a specific state, storing all valid pattern matches.
 * Uses shared_ptr to avoid copying matches when forwarding between states.
 */
struct StateMatchCache {
    RawStateId state_id;  // Cache is indexed by raw StateId (for pattern matching)
    std::vector<SharedCachedMatch> cached_matches;   // All valid matches for this state

    explicit StateMatchCache(RawStateId id) : state_id(id) {}

    /**
     * Add a new match to the cache (creates shared_ptr).
     */
    void add_match(const CachedMatch& match) {
        cached_matches.push_back(std::make_shared<const CachedMatch>(match));
    }

    /**
     * Add a shared match to the cache (for forwarding).
     */
    void add_shared_match(SharedCachedMatch shared_match) {
        cached_matches.push_back(std::move(shared_match));
    }

    /**
     * Remove matches that use any of the specified edges (for invalidation).
     */
    void invalidate_matches_using_edges(const std::vector<GlobalEdgeId>& deleted_edges);

    /**
     * Get all valid matches for a specific rule.
     */
    std::vector<CachedMatch> get_matches_for_rule(std::size_t rule_index) const;

    /**
     * Check if cache has any matches for any rule.
     */
    bool has_matches() const {
        return !cached_matches.empty();
    }

    /**
     * Clear all cached matches.
     */
    void clear() {
        cached_matches.clear();
    }
};

/**
 * Multiway graph containing states, events, and their relationships.
 * This is the main structure being populated in parallel by the Wolfram Physics engine.
 */
class MultiwayGraph {
    friend class WolframEvolution;  // Allow evolution to access internal methods
    friend class RewriteTask;       // Allow REWRITE tasks to create new states

    // Lock-free event edge node for linked list
    struct EventEdgeNode {
        EventEdge edge;
        std::atomic<EventEdgeNode*> next{nullptr};
        EventEdgeNode(const EventEdge& e) : edge(e) {}
    };

private:
    // Event storage - using lock-free concurrent map
    std::unique_ptr<ConcurrentHashMap<EventId, WolframEvent>> events;

    // Optional state storage - only used when full_capture is enabled
    std::unique_ptr<ConcurrentHashMap<RawStateId, WolframState>> states;  // Raw states for pattern matching
    std::unique_ptr<ConcurrentHashMap<CanonicalStateId, RawStateId>> canonical_to_raw_mapping;  // For deduplication
    std::unique_ptr<ConcurrentHashMap<RawStateId, StateMatchCache>> match_caches;  // Pattern match caches
    std::unordered_set<RawStateId> exhausted_states;  // States with no more applicable rules

    // Event edge tracking
    std::atomic<EventEdgeNode*> event_edges_head{nullptr};  // Lock-free linked list for event edges
    std::atomic<size_t> event_edges_count{0};
    std::atomic<size_t> events_count{0};
    std::atomic<size_t> states_count{0};

    // Edge-to-event mapping indices for efficient overlap detection
    std::unique_ptr<ConcurrentHashMap<GlobalEdgeId, LockfreeList<EventId>*>> input_edge_to_events;
    std::unique_ptr<ConcurrentHashMap<GlobalEdgeId, LockfreeList<EventId>*>> output_edge_to_events;

    // Global ID generators
    std::atomic<EventId> next_event_id{0};
    std::atomic<GlobalVertexId> next_global_vertex_id{0};
    std::atomic<GlobalEdgeId> next_global_edge_id{0};

    // Configuration flags
    bool canonicalization_enabled = true;
    bool full_capture = false;  // Controls optional state storage

    // Track initial state for causal computation
    std::optional<RawStateId> initial_state_id;


public:
    // Disable copy operations (due to atomic members)
    MultiwayGraph(const MultiwayGraph&) = delete;
    MultiwayGraph& operator=(const MultiwayGraph&) = delete;

    // Enable move operations (no mutex locking needed during construction)
    MultiwayGraph(MultiwayGraph&& other) noexcept
        : events(std::move(other.events))
        , input_edge_to_events(std::move(other.input_edge_to_events))
        , output_edge_to_events(std::move(other.output_edge_to_events))
        , next_event_id(other.next_event_id.load())
        , next_global_vertex_id(other.next_global_vertex_id.load())
        , next_global_edge_id(other.next_global_edge_id.load()) {}

    MultiwayGraph& operator=(MultiwayGraph&& other) noexcept;

    explicit MultiwayGraph(bool enable_full_capture = false)
        : events(std::make_unique<ConcurrentHashMap<EventId, WolframEvent>>())
        , input_edge_to_events(std::make_unique<ConcurrentHashMap<GlobalEdgeId, LockfreeList<EventId>*>>())
        , output_edge_to_events(std::make_unique<ConcurrentHashMap<GlobalEdgeId, LockfreeList<EventId>*>>())
        , full_capture(enable_full_capture)
    {
        if (full_capture) {
            states = std::make_unique<ConcurrentHashMap<RawStateId, WolframState>>();
            canonical_to_raw_mapping = std::make_unique<ConcurrentHashMap<CanonicalStateId, RawStateId>>();
            match_caches = std::make_unique<ConcurrentHashMap<RawStateId, StateMatchCache>>();
        }
    }

    /**
     * Create initial state with given hypergraph structure.
     * Returns the state directly - no storage, states flow through tasks.
     */
    WolframState create_initial_state(const std::vector<std::vector<GlobalVertexId>>& edge_structures);

    /**
     * Get fresh global vertex ID.
     */
    GlobalVertexId get_fresh_vertex_id() {
        return next_global_vertex_id.fetch_add(1);
    }

    /**
     * Get fresh global edge ID.
     */
    GlobalEdgeId get_fresh_edge_id() {
        return next_global_edge_id.fetch_add(1);
    }

    /**
     * Enable or disable canonicalization-based state deduplication.
     */
    void set_canonicalization_enabled(bool enabled) {
        canonicalization_enabled = enabled;
    }

    bool is_canonicalization_enabled() const {
        return canonicalization_enabled;
    }

    /**
     * Get final atomic counter values for determinism testing.
     */
    GlobalVertexId get_final_vertex_id() const {
        return next_global_vertex_id.load();
    }

    GlobalEdgeId get_final_edge_id() const {
        return next_global_edge_id.load();
    }

    EventId get_final_event_id() const {
        return next_event_id.load();
    }

private:
    /**
     * Apply rewriting rule and create new state and event.
     * ONLY called from job system tasks - not public API!
     */
    EventId apply_rewriting(const WolframState& input_state,
                           const WolframState& output_state,
                           const std::vector<GlobalEdgeId>& edges_to_remove,
                           const std::vector<GlobalEdgeId>& produced_edge_ids,
                           std::size_t rule_index,
                           GlobalVertexId anchor_vertex,
                           std::size_t evolution_step);

    /**
     * Update edge-to-event mappings for a new event.
     */
    void update_edge_mappings(EventId event_id,
                             const std::vector<GlobalEdgeId>& input_edges,
                             const std::vector<GlobalEdgeId>& output_edges);

    /**
     * Update causal and branchial relationships between events based on edge overlap.
     * Note: This is now done in post-processing phase to avoid concurrent map updates
     */
    void update_event_relationships(EventId new_event_id);

    /**
     * Compute all event relationships (causal and branchial edges) in parallel post-processing phase.
     * Creates one task per event to check relationships against all other events.
     * This avoids race conditions by running after evolution completes.
     */
    void compute_all_event_relationships(job_system::JobSystem<PatternMatchingTaskType>* job_system);

    // Removed: get_state - states are not stored, they flow through tasks

    /**
     * Get event by ID.
     */
    std::optional<WolframEvent> get_event(EventId event_id) const {
        return events->find(event_id);
    }


    /**
     * Get events that can be processed concurrently (based on causal dependencies).
     */
    std::vector<EventId> get_concurrent_events(const std::vector<EventId>& candidate_events) const;

public:
    std::size_t num_states() const {
        if (full_capture && states) {
            // When full_capture is enabled, use the size of stored states hash map
            if (canonicalization_enabled && canonical_to_raw_mapping) {
                // If canonicalization is enabled, count unique canonical states
                return canonical_to_raw_mapping->size();
            } else {
                // If canonicalization is disabled, count raw states
                return states->size();
            }
        } else {
            // When full_capture is disabled, count unique states from events based on canonicalization setting
            auto all_events = get_all_events();
            if (canonicalization_enabled) {
                // Count unique canonical states from events
                std::unordered_set<CanonicalStateId> unique_canonical_states;
                for (const auto& event : all_events) {
                    unique_canonical_states.insert(event.canonical_input_state_id);
                    unique_canonical_states.insert(event.canonical_output_state_id);
                }
                return unique_canonical_states.size();
            } else {
                // Count unique raw states from events
                std::unordered_set<RawStateId> unique_raw_states;
                for (const auto& event : all_events) {
                    unique_raw_states.insert(event.input_state_id);
                    unique_raw_states.insert(event.output_state_id);
                }
                return unique_raw_states.size();
            }
        }
    }

    std::size_t num_events() const {
        return events_count.load();
    }

    /**
     * Get count of causal event edges.
     */
    std::size_t get_causal_edge_count() const;

    /**
     * Get count of branchial event edges.
     */
    std::size_t get_branchial_edge_count() const;

    /**
     * Get all states for external access.
     * Only works when full_capture is enabled.
     */
    std::vector<WolframState> get_all_states() const;

    /**
     * Get all events for external access.
     */
    std::vector<WolframEvent> get_all_events() const;

    /**
     * Get event edges (causal and branchial) for external access.
     */
    std::vector<EventEdge> get_event_edges() const;

    /**
     * Mark a state as exhausted (no more applicable rules) and clean up memory.
     * Only works when full_capture is enabled.
     */
    void mark_state_exhausted(RawStateId state_id);

    /**
     * Check if a state is exhausted.
     * Only works when full_capture is enabled.
     */
    bool is_state_exhausted(RawStateId state_id) const;

    /**
     * Get count of live (non-exhausted) states.
     * Only works when full_capture is enabled.
     */
    std::size_t num_live_states() const;

    /**
     * Get edges within radius of specified center edges (for patch-based matching).
     */
    std::unordered_set<GlobalEdgeId> get_edges_within_radius(
        const WolframState& state,
        const std::vector<GlobalEdgeId>& center_edges,
        std::size_t radius) const;

    /**
     * Add pattern match to state cache.
     * Only works when full_capture is enabled.
     */
    void cache_pattern_match(RawStateId state_id, std::size_t rule_index,
                           const std::vector<GlobalEdgeId>& matched_edges,
                           const VariableAssignment& assignment,
                           GlobalVertexId anchor_vertex);

    /**
     * Get cached matches for a state and rule.
     * Only works when full_capture is enabled.
     */
    std::vector<CachedMatch> get_cached_matches(RawStateId state_id, std::size_t rule_index) const;

    /**
     * Forward valid matches from input state to output state, invalidating those using deleted edges.
     * Uses shared_ptr to avoid copying match data - much more memory efficient.
     * Only works when full_capture is enabled.
     */
    void forward_matches(RawStateId input_state_id, RawStateId output_state_id,
                        const std::vector<GlobalEdgeId>& deleted_edges);

    /**
     * Check if a state has any cached matches.
     * Only works when full_capture is enabled.
     */
    bool has_cached_matches(RawStateId state_id) const;

    /**
     * Clear match cache for a state (when state is exhausted).
     * Only works when full_capture is enabled.
     */
    void clear_match_cache(RawStateId state_id);

    /**
     * Reconstruct a state by replaying events from the initial state.
     * This is memory-efficient as we don't store full state copies.
     */
    std::optional<WolframState> reconstruct_state(RawStateId target_state_id) const;

    /**
     * Find the sequence of events needed to reach a target state from the initial state.
     */
    std::vector<EventId> find_event_path_to_state(RawStateId target_state_id) const;

    /**
     * Get state by ID, reconstructing it if necessary.
     * This replaces direct state storage with on-demand reconstruction.
     */
    std::optional<WolframState> get_state_efficient(RawStateId state_id) const;

    /**
     * Get canonical hash for a state without storing the full state.
     */
    std::optional<std::size_t> get_state_hash(RawStateId state_id) const;

    void clear();

    /**
     * Export multiway graph in DOT format showing states and events.
     */
    std::string export_multiway_graph_dot() const;

    /**
     * Print concise summary of multiway graph state.
     */
    std::string get_summary() const;

    void print_summary() const {
        std::cout << get_summary();
    }
};

} // namespace hypergraph

// Hash functions for Wolfram Physics types
namespace std {
/* template<>
struct hash<hypergraph::WolframEvent> {
    std::size_t operator()(const hypergraph::WolframEvent& event) const {
        std::size_t hash_value = 0;
        std::hash<std::size_t> size_hasher;

        // Hash basic event properties
        hash_value ^= size_hasher(event.event_id) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        hash_value ^= size_hasher(event.input_state_id.value) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        hash_value ^= size_hasher(event.output_state_id.value) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        hash_value ^= size_hasher(event.rule_index) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        hash_value ^= size_hasher(event.anchor_vertex) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);

        // Hash consumed edges
        for (hypergraph::GlobalEdgeId edge_id : event.consumed_edges) {
            hash_value ^= size_hasher(edge_id) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }

        // Hash produced edges
        for (hypergraph::GlobalEdgeId edge_id : event.produced_edges) {
            hash_value ^= size_hasher(edge_id) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }

        return hash_value;
    }
}; */

/* template<>
struct hash<hypergraph::WolframState> {
    std::size_t operator()(const hypergraph::WolframState& state) const {
        // Hash the canonical form of the state
        return std::hash<hypergraph::CanonicalForm>{}(state.get_canonical_form());
    }
}; */
}

#endif // HYPERGRAPH_WOLFRAM_STATES_HPP