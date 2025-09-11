#ifndef HYPERGRAPH_WOLFRAM_STATES_HPP
#define HYPERGRAPH_WOLFRAM_STATES_HPP

#include "hypergraph/debug_log.hpp"
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/types.hpp>
#include <hypergraph/canonicalization.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <hypergraph/concurrent_hash_map.hpp>
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
 * Wolfram Physics state containing a hypergraph with global IDs.
 */
class WolframState {
private:
    std::vector<GlobalHyperedge> global_edges_;
    std::unordered_set<GlobalVertexId> global_vertices_;
    mutable std::optional<CanonicalForm> cached_canonical_form_;
    EdgeSignatureIndex edge_signature_index_;  // Fast edge lookup by signature
    mutable std::optional<VertexMapping> cached_vertex_mapping_;
    mutable std::optional<std::vector<GlobalEdgeId>> canonical_edge_to_global_mapping_;

    /**
     * Rebuild the edge signature index.
     */
    void rebuild_signature_index();

public:
    explicit WolframState() {}

    StateId id(bool canonicalization_enabled = true) const { 
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

    /**
     * Get mapping from canonical vertex IDs to original global vertex IDs.
     * This is needed during rewrite operations to map back from canonical space.
     */
    GlobalVertexId canonical_to_global_vertex(std::size_t canonical_id) const;

    /**
     * Get mapping from global vertex ID to canonical vertex ID.
     */
    std::optional<std::size_t> global_to_canonical_vertex(GlobalVertexId global_id) const;

    /**
     * Get mapping from canonical edge IDs to original global edge IDs.
     */
    GlobalEdgeId canonical_to_global_edge(std::size_t canonical_edge_id) const;

private:

public:
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
    const CanonicalForm& get_canonical_form() const;

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

    WolframState clone() const;
};

/**
 * Wolfram Physics event representing a rewriting operation.
 */
struct WolframEvent {
    EventId event_id;
    StateId input_state_id;   // State before rewriting
    StateId output_state_id;  // State after rewriting

    std::vector<GlobalEdgeId> consumed_edges;  // Global edges removed
    std::vector<GlobalEdgeId> produced_edges;  // Global edges added

    std::size_t rule_index;   // Index into rule set that was applied
    GlobalVertexId anchor_vertex; // Vertex around which rewriting occurred

    std::chrono::steady_clock::time_point timestamp;

    WolframEvent() = default;

    WolframEvent(EventId id, StateId input_state, StateId output_state,
                const std::vector<GlobalEdgeId>& consumed,
                const std::vector<GlobalEdgeId>& produced,
                std::size_t rule_idx, GlobalVertexId anchor)
        : event_id(id), input_state_id(input_state), output_state_id(output_state)
        , consumed_edges(consumed), produced_edges(produced)
        , rule_index(rule_idx), anchor_vertex(anchor)
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
    StateId state_id;
    std::vector<SharedCachedMatch> cached_matches;   // All valid matches for this state

    explicit StateMatchCache(StateId id) : state_id(id) {}

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
    // State and event storage - using lock-free concurrent maps
    std::unique_ptr<ConcurrentHashMap<StateId, WolframState>> states_;
    std::unique_ptr<ConcurrentHashMap<EventId, WolframEvent>> events_;

    // No longer needed - states are identified by their hash directly
    std::atomic<EventEdgeNode*> event_edges_head_{nullptr};  // Lock-free linked list for event edges
    std::atomic<size_t> event_edges_count_{0};
    std::atomic<size_t> states_count_{0};
    std::atomic<size_t> events_count_{0};

    // Edge-to-event mapping indices for efficient overlap detection
    std::unique_ptr<ConcurrentHashMap<GlobalEdgeId, std::vector<EventId>>> input_edge_to_events_;
    std::unique_ptr<ConcurrentHashMap<GlobalEdgeId, std::vector<EventId>>> output_edge_to_events_;

    // State lifecycle management for memory optimization
    std::unique_ptr<ConcurrentHashSet<StateId>> exhausted_states_;

    // Match forwarding and invalidation tracking
    std::unique_ptr<ConcurrentHashMap<StateId, StateMatchCache>> match_caches_;

    // Global ID generators  
    std::atomic<EventId> next_event_id_{1};
    std::atomic<GlobalVertexId> next_global_vertex_id_{0};
    std::atomic<GlobalEdgeId> next_global_edge_id_{0};

    // Configuration flags
    bool canonicalization_enabled_ = true;


public:
    // Disable copy operations (due to atomic members)
    MultiwayGraph(const MultiwayGraph&) = delete;
    MultiwayGraph& operator=(const MultiwayGraph&) = delete;

    // Enable move operations (no mutex locking needed during construction)
    MultiwayGraph(MultiwayGraph&& other) noexcept
        : states_(std::move(other.states_))
        , events_(std::move(other.events_))
        , input_edge_to_events_(std::move(other.input_edge_to_events_))
        , output_edge_to_events_(std::move(other.output_edge_to_events_))
        , exhausted_states_(std::move(other.exhausted_states_))
        , match_caches_(std::move(other.match_caches_))
        , next_event_id_(other.next_event_id_.load())
        , next_global_vertex_id_(other.next_global_vertex_id_.load())
        , next_global_edge_id_(other.next_global_edge_id_.load()) {}

    MultiwayGraph& operator=(MultiwayGraph&& other) noexcept;

    MultiwayGraph()
        : states_(std::make_unique<ConcurrentHashMap<StateId, WolframState>>())
        , events_(std::make_unique<ConcurrentHashMap<EventId, WolframEvent>>())
        , input_edge_to_events_(std::make_unique<ConcurrentHashMap<GlobalEdgeId, std::vector<EventId>>>())
        , output_edge_to_events_(std::make_unique<ConcurrentHashMap<GlobalEdgeId, std::vector<EventId>>>())
        , exhausted_states_(std::make_unique<ConcurrentHashSet<StateId>>())
        , match_caches_(std::make_unique<ConcurrentHashMap<StateId, StateMatchCache>>())
    {}

    /**
     * Create initial state with given hypergraph structure.
     */
    StateId create_initial_state(const std::vector<std::vector<GlobalVertexId>>& edge_structures);

    /**
     * Get fresh global vertex ID.
     */
    GlobalVertexId get_fresh_vertex_id() {
        return next_global_vertex_id_.fetch_add(1);
    }

    /**
     * Get fresh global edge ID.
     */
    GlobalEdgeId get_fresh_edge_id() {
        return next_global_edge_id_.fetch_add(1);
    }

    /**
     * Enable or disable canonicalization-based state deduplication.
     */
    void set_canonicalization_enabled(bool enabled) {
        canonicalization_enabled_ = enabled;
    }

private:
    /**
     * Apply rewriting rule and create new state and event.
     * ONLY called from job system tasks - not public API!
     */
    EventId apply_rewriting(StateId input_state_id,
                           const std::vector<GlobalEdgeId>& edges_to_remove,
                           const std::vector<std::vector<GlobalVertexId>>& edges_to_add,
                           std::size_t rule_index,
                           GlobalVertexId anchor_vertex);

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
     * Get state by ID.
     */
    std::optional<WolframState> get_state(StateId state_id) const {
        return states_->find(state_id);
    }

    /**
     * Get event by ID.
     */
    std::optional<WolframEvent> get_event(EventId event_id) const {
        return events_->find(event_id);
    }


    /**
     * Get events that can be processed concurrently (based on causal dependencies).
     */
    std::vector<EventId> get_concurrent_events(const std::vector<EventId>& candidate_events) const;

public:
    std::size_t num_states() const {
        // Use hash map's internal atomic size to avoid race condition
        return states_->size();
    }

    std::size_t num_events() const {
        return events_count_.load();
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
     * Simple and effective - let shared_ptr handle the rest.
     */
    void mark_state_exhausted(StateId state_id);

    /**
     * Check if a state is exhausted.
     */
    bool is_state_exhausted(StateId state_id) const {
        return exhausted_states_->contains(state_id);
    }

    /**
     * Get count of live (non-exhausted) states.
     */
    std::size_t num_live_states() const {
        return states_count_.load(); // Only live states remain in states_ map
    }

    /**
     * Get edges within radius of specified center edges (for patch-based matching).
     */
    std::unordered_set<GlobalEdgeId> get_edges_within_radius(
        const WolframState& state,
        const std::vector<GlobalEdgeId>& center_edges,
        std::size_t radius) const;

    /**
     * Add pattern match to state cache.
     */
    void cache_pattern_match(StateId state_id, std::size_t rule_index,
                           const std::vector<GlobalEdgeId>& matched_edges,
                           const VariableAssignment& assignment,
                           GlobalVertexId anchor_vertex);

    /**
     * Get cached matches for a state and rule.
     */
    std::vector<CachedMatch> get_cached_matches(StateId state_id, std::size_t rule_index) const;

    /**
     * Forward valid matches from input state to output state, invalidating those using deleted edges.
     * Uses shared_ptr to avoid copying match data - much more memory efficient.
     */
    void forward_matches(StateId input_state_id, StateId output_state_id,
                        const std::vector<GlobalEdgeId>& deleted_edges);

    /**
     * Check if a state has any cached matches.
     */
    bool has_cached_matches(StateId state_id) const {

        auto cache_opt = match_caches_->find(state_id);
        return cache_opt && cache_opt.value().has_matches();
    }

    /**
     * Clear match cache for a state (when state is exhausted).
     */
    void clear_match_cache(StateId state_id) {
        match_caches_->erase(state_id);
    }

    /**
     * Reconstruct a state by replaying events from the initial state.
     * This is memory-efficient as we don't store full state copies.
     */
    std::optional<WolframState> reconstruct_state(StateId target_state_id) const;

    /**
     * Find the sequence of events needed to reach a target state from the initial state.
     */
    std::vector<EventId> find_event_path_to_state(StateId target_state_id) const;

    /**
     * Get state by ID, reconstructing it if necessary.
     * This replaces direct state storage with on-demand reconstruction.
     */
    std::optional<WolframState> get_state_efficient(StateId state_id) const;

    /**
     * Get canonical hash for a state without storing the full state.
     */
    std::optional<std::size_t> get_state_hash(StateId state_id) const;

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
template<>
struct hash<hypergraph::WolframEvent> {
    std::size_t operator()(const hypergraph::WolframEvent& event) const {
        std::size_t hash_value = 0;
        std::hash<std::size_t> size_hasher;

        // Hash basic event properties
        hash_value ^= size_hasher(event.event_id) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        hash_value ^= size_hasher(event.input_state_id) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        hash_value ^= size_hasher(event.output_state_id) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
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
};

template<>
struct hash<hypergraph::WolframState> {
    std::size_t operator()(const hypergraph::WolframState& state) const {
        // Hash the canonical form of the state
        return std::hash<hypergraph::CanonicalForm>{}(state.get_canonical_form());
    }
};
}

#endif // HYPERGRAPH_WOLFRAM_STATES_HPP