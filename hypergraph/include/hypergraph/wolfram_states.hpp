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
class MultiwayGraph;
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
    friend class MultiwayGraph;  // For accessing next_state_id
private:
    StateID state_id;  // Unique ID assigned on construction
    std::vector<GlobalHyperedge> global_edges_;
    std::unordered_set<GlobalVertexId> global_vertices_;

    // Adjacency/incidence index: vertex -> list of (edge_id, position_in_edge) pairs
    // Enables O(degree(v)) lookups instead of O(E) scans for finding edges containing a vertex
    std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>> vertex_to_edge_positions_;

    // Edge ID to index map: enables O(1) edge lookup by global ID (maintained alongside global_edges_)
    std::unordered_map<GlobalEdgeId, std::size_t> edge_id_to_index_;

    mutable std::optional<std::size_t> cached_hash_;  // Lazy-computed hash
    mutable std::optional<StateID> canonical_state_id;  // Canonical ID when known
    mutable VertexMapping vertex_mapping_;  // Isomorphism to canonical form (for event canonicalization)

public:
    // Delete default constructor
    WolframState() = delete;
    
    // Constructor gets fresh ID from MultiwayGraph
    explicit WolframState(MultiwayGraph& graph);
    
    // Constructor for creating state with specific edges - gets fresh ID
    explicit WolframState(MultiwayGraph& graph, const std::vector<std::vector<GlobalVertexId>>& edge_structures);

    /**
     * Batch constructor: Create state from edge list with pre-allocated edge IDs.
     * Much more efficient than calling add_global_edge() N times.
     * @param graph MultiwayGraph to get fresh state ID from
     * @param edges List of (GlobalEdgeId, vertices) pairs
     * @param parent_state Optional parent state to copy indices from (for incremental updates)
     */
    explicit WolframState(MultiwayGraph& graph,
                         const std::vector<std::pair<GlobalEdgeId, std::vector<GlobalVertexId>>>& edges,
                         const WolframState* parent_state = nullptr);
    
    // Delete copy operations to prevent ID duplication
    WolframState(const WolframState&) = delete;
    WolframState& operator=(const WolframState&) = delete;

    // Allow move operations
    WolframState(WolframState&&) = default;
    WolframState& operator=(WolframState&&) = default;
    
    /**
     * Get the unique state ID.
     */
    StateID id() const { return state_id; }
    
    /**
     * Get hash of the state structure with lazy computation and caching.
     * @param strategy_type Hash strategy to use (UNIQUENESS_TREE or CANONICALIZATION)
     */
    std::size_t get_hash(HashStrategyType strategy_type) const {
        if (!cached_hash_.has_value()) {
            cached_hash_ = compute_hash(strategy_type);
        }
        return cached_hash_.value();
    }

    /**
     * Get canonical state ID if known, otherwise return raw ID.
     */
    StateID get_canonical_id() const {
        return canonical_state_id.value_or(state_id);
    }

    /**
     * Set canonical state ID (called during canonicalization).
     */
    void set_canonical_id(StateID canonical_id) const {
        canonical_state_id = canonical_id;
    }

    /**
     * Check if canonical ID is known.
     */
    bool has_canonical_id() const {
        return canonical_state_id.has_value();
    }

    /**
     * Set vertex mapping to canonical form (only when event canonicalization enabled).
     */
    void set_vertex_mapping(const VertexMapping& mapping) const {
        vertex_mapping_ = mapping;
    }

    /**
     * Get vertex mapping to canonical form.
     */
    const VertexMapping& get_vertex_mapping() const {
        return vertex_mapping_;
    }

    /**
     * Add edge with global IDs (incrementally updates adjacency index).
     */
    void add_global_edge(GlobalEdgeId edge_id, const std::vector<GlobalVertexId>& vertices);

    /**
     * Batch add edges (incrementally updates adjacency index).
     */
    void add_global_edges(const std::vector<std::pair<GlobalEdgeId, std::vector<GlobalVertexId>>>& edges);

    /**
     * Remove edge by global ID (incrementally updates adjacency index).
     */
    bool remove_global_edge(GlobalEdgeId edge_id);

    /**
     * Batch remove edges (incrementally updates adjacency index).
     */
    void remove_global_edges(const std::vector<GlobalEdgeId>& edge_ids);

    const std::vector<GlobalHyperedge>& edges() const { return global_edges_; }
    const std::unordered_set<GlobalVertexId>& vertices() const { return global_vertices_; }

    std::size_t num_edges() const { return global_edges_.size(); }
    std::size_t num_vertices() const { return global_vertices_.size(); }

    /**
     * Get edge by global ID using O(1) index lookup.
     */
    const GlobalHyperedge& get_edge_by_id(GlobalEdgeId edge_id) const {
        return global_edges_[edge_id_to_index_.at(edge_id)];
    }

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
     * Compute isomorphism-invariant hash for state identification.
     * Only called when canonicalization is enabled.
     */
    std::size_t compute_hash(HashStrategyType strategy_type) const;

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
     * Get adjacency index: vertex -> list of (edge_id, position) pairs.
     * Enables O(1) lookup of all edges containing a vertex.
     */
    const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>&
    get_vertex_to_edge_positions() const {
        return vertex_to_edge_positions_;
    }

    /**
     * Get edge by global ID.
     */
    const GlobalHyperedge* get_edge(GlobalEdgeId edge_id) const;

    // Canonical edge signature functions removed for thread safety

    WolframState clone(MultiwayGraph& graph) const;
};

/**
 * Wolfram Physics event representing a rewriting operation.
 */
struct WolframEvent {
    EventId event_id;
    StateID input_state_id;            // Raw state before rewriting (for relationships)
    StateID output_state_id;           // Raw state after rewriting (for relationships)

    StateID canonical_input_state_id;  // Canonical state before rewriting (for storage)
    StateID canonical_output_state_id; // Canonical state after rewriting (for storage)

    std::optional<EventId> canonical_event_id;  // Points to first event with same signature (when event canonicalization enabled)

    std::vector<GlobalEdgeId> consumed_edges;  // Global edge IDs removed
    std::vector<GlobalEdgeId> produced_edges;  // Global edge IDs added

    std::size_t rule_index;   // Index into rule set that was applied
    std::size_t step;         // Evolution step when this event occurred

    std::chrono::steady_clock::time_point timestamp;

    WolframEvent() = default;

    /**
     * Check if canonical input state ID is valid.
     */
    bool has_canonical_input_state_id() const {
        return canonical_input_state_id != INVALID_STATE;
    }

    /**
     * Check if canonical output state ID is valid.
     */
    bool has_canonical_output_state_id() const {
        return canonical_output_state_id != INVALID_STATE;
    }

    WolframEvent(EventId id, StateID input_state, StateID output_state,
                const std::vector<GlobalEdgeId>& consumed,
                const std::vector<GlobalEdgeId>& produced,
                std::size_t rule_idx, std::size_t evolution_step)
        : event_id(id), input_state_id(input_state), output_state_id(output_state)
        , canonical_input_state_id(INVALID_STATE), canonical_output_state_id(INVALID_STATE)
        , consumed_edges(consumed), produced_edges(produced)
        , rule_index(rule_idx), step(evolution_step)
        , timestamp(std::chrono::steady_clock::now()) {}

};

/**
 * Event signature for canonicalization - tuple of (canonical_input_state, canonical_output_state, step)
 */
struct EventSignature {
    // TODO: Automatic mode event canonicalization has issues - disabled pending investigation
    // For Automatic mode: canonical edge positions (indices)
    std::vector<std::size_t> canonical_consumed_edges;
    std::vector<std::size_t> canonical_produced_edges;

    // For Full mode: canonical state IDs only (edges empty)
    StateID canonical_input_state_id{INVALID_STATE};
    StateID canonical_output_state_id{INVALID_STATE};
    std::size_t evolution_step{0};
    bool include_state_ids{false};  // True for Full mode, False for Automatic mode (currently disabled)

    bool operator==(const EventSignature& other) const {
        if (include_state_ids != other.include_state_ids) {
            DEBUG_LOG("[EventSig==] include_state_ids mismatch: this=%d other=%d", include_state_ids, other.include_state_ids);
            return false;
        }

        if (include_state_ids) {
            // Full mode: compare state IDs only (edges are empty)
            return canonical_input_state_id == other.canonical_input_state_id &&
                   canonical_output_state_id == other.canonical_output_state_id;
        } else {
#if 1  // RE-ENABLED: Testing Automatic mode event canonicalization
            // Automatic mode: compare edge positions, canonical state IDs, and step
            // Events are identified by which canonical edges they transform in which canonical states
            bool consumed_match = canonical_consumed_edges == other.canonical_consumed_edges;
            bool produced_match = canonical_produced_edges == other.canonical_produced_edges;
            bool input_state_match = canonical_input_state_id == other.canonical_input_state_id;
            bool output_state_match = canonical_output_state_id == other.canonical_output_state_id;
            bool step_match = evolution_step == other.evolution_step;

            bool result = consumed_match && produced_match && input_state_match && output_state_match && step_match;

            DEBUG_LOG("[EventSig==] this(in=%zu,out=%zu,step=%zu,c=%zu,p=%zu) vs other(in=%zu,out=%zu,step=%zu,c=%zu,p=%zu) => consumed=%d produced=%d in_state=%d out_state=%d step=%d RESULT=%d",
                     canonical_input_state_id.value, canonical_output_state_id.value, evolution_step, canonical_consumed_edges.size(), canonical_produced_edges.size(),
                     other.canonical_input_state_id.value, other.canonical_output_state_id.value, other.evolution_step, other.canonical_consumed_edges.size(), other.canonical_produced_edges.size(),
                     consumed_match, produced_match, input_state_match, output_state_match, step_match, result);

            return result;
#else
            throw std::runtime_error("Automatic mode event canonicalization comparison is disabled");
#endif
        }
    }
};

} // namespace hypergraph

// Hash specialization for EventSignature
namespace std {
    template <>
    struct hash<hypergraph::EventSignature> {
        std::size_t operator()(const hypergraph::EventSignature& sig) const {
            std::size_t h = 0;

            if (sig.include_state_ids) {
                // Full mode: hash state IDs only
                h ^= std::hash<std::size_t>{}(sig.canonical_input_state_id.value) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<std::size_t>{}(sig.canonical_output_state_id.value) + 0x9e3779b9 + (h << 6) + (h >> 2);
                DEBUG_LOG("[std::hash<EventSig>] FULL mode (in=%zu,out=%zu) => hash=%zu",
                         sig.canonical_input_state_id.value, sig.canonical_output_state_id.value, h);
            } else {
#if 1  // RE-ENABLED: Testing Automatic mode event canonicalization
                // Automatic mode: hash edge positions, canonical state IDs, and step
                // Events are identified by which canonical edges they transform in which canonical states

                // Hash canonical input state ID
                h ^= std::hash<std::size_t>{}(sig.canonical_input_state_id.value) + 0x9e3779b9 + (h << 6) + (h >> 2);

                // Hash canonical output state ID
                h ^= std::hash<std::size_t>{}(sig.canonical_output_state_id.value) + 0x9e3779b9 + (h << 6) + (h >> 2);

                // Hash consumed edge positions
                for (const auto& edge_idx : sig.canonical_consumed_edges) {
                    h ^= std::hash<std::size_t>{}(edge_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }

                // Hash produced edge positions
                for (const auto& edge_idx : sig.canonical_produced_edges) {
                    h ^= std::hash<std::size_t>{}(edge_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }

                // Hash step
                h ^= std::hash<std::size_t>{}(sig.evolution_step) + 0x9e3779b9 + (h << 6) + (h >> 2);

                DEBUG_LOG("[std::hash<EventSig>] AUTO mode (in=%zu,out=%zu,step=%zu,consumed=%zu,produced=%zu) => hash=%zu",
                         sig.canonical_input_state_id.value, sig.canonical_output_state_id.value, sig.evolution_step,
                         sig.canonical_consumed_edges.size(), sig.canonical_produced_edges.size(), h);
#else
                throw std::runtime_error("Automatic mode event canonicalization hashing is disabled");
#endif
            }

            return h;
        }
    };
}

namespace hypergraph {

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

    CachedMatch(std::size_t rule_idx, const std::vector<GlobalEdgeId>& edges,
                const VariableAssignment& var_assignment)
        : rule_index(rule_idx), matched_edges(edges), assignment(var_assignment) {}
};

using SharedCachedMatch = std::shared_ptr<const CachedMatch>;

/**
 * Match cache for a specific state, storing all valid pattern matches.
 * Uses shared_ptr to avoid copying matches when forwarding between states.
 */
struct StateMatchCache {
    StateID state_id;  // Cache is indexed by raw StateId (for pattern matching)
    std::vector<SharedCachedMatch> cached_matches;   // All valid matches for this state

    explicit StateMatchCache(StateID id) : state_id(id) {}

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


private:
    // Event storage - using lock-free concurrent map
    std::unique_ptr<ConcurrentHashMap<EventId, WolframEvent>> events;

    // Optional state storage - only used when full_capture is enabled
    std::unique_ptr<ConcurrentHashMap<StateID, std::shared_ptr<WolframState>>> states;  // States by raw ID for direct access
    std::unique_ptr<ConcurrentHashMap<StateID, StateMatchCache>> match_caches;  // Pattern match caches
    std::unique_ptr<ConcurrentHashMap<std::size_t, StateID>> seen_hashes;  // Maps canonical hash to canonical StateID

    // Event canonicalization - maps (input_state, output_state, step) hash to first event ID
    std::unique_ptr<ConcurrentHashMap<EventSignature, EventId>> seen_event_signatures;

    std::unordered_set<StateID> exhausted_states;  // States with no more applicable rules

    // Event edge tracking
    LockfreeList<EventEdge> event_edges;  // Lock-free list for event edges
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
    std::atomic<std::size_t> next_state_id{0};

    // Configuration flags
    bool canonicalize_states = true;
    HashStrategyType hash_strategy_type;  // Runtime-selectable when canonicalization enabled (initialized in constructor)
    bool canonicalize_events = false;  // Track canonical event IDs for events with same (input, output, step)
    bool deduplicate_events = false;  // When true, return existing event ID for duplicates (simple graph mode)
    bool full_event_canonicalization = true;  // Full mode (true) uses state IDs only; Automatic mode (false) includes edges & step
    bool full_capture = false;  // Controls optional state storage
    bool full_capture_non_canonicalised = false;  // Store all states including duplicates
    bool transitive_reduction_enabled = true;  // Controls transitive reduction of causal graph
    bool early_termination_enabled = false;  // Stop processing seen states

    // Track initial states for causal computation and state reconstruction
    std::vector<StateID> initial_state_ids;

    // Pruning control parameters (0 = unlimited)
    std::size_t max_successor_states_per_parent_{0};
    std::size_t max_states_per_step_{0};
    double exploration_probability_{1.0};  // 1.0 = always explore

    // Tracking structures for hard limits (thread-safe concurrent maps with shared_ptr to atomics)
    std::unique_ptr<ConcurrentHashMap<StateID, std::shared_ptr<std::atomic<std::size_t>>>> successor_counts_;
    std::unique_ptr<ConcurrentHashMap<std::size_t, std::shared_ptr<std::atomic<std::size_t>>>> states_per_step_;

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

    explicit MultiwayGraph(bool enable_full_capture = false, bool enable_transitive_reduction = true, bool enable_early_termination = false, bool enable_full_capture_non_canonicalised = false,
                          std::size_t max_successor_states_per_parent = 0, std::size_t max_states_per_step = 0, double exploration_probability = 1.0, bool enable_canonicalize_events = false, bool enable_deduplicate_events = false, bool enable_full_event_canonicalization = true)
        : events(std::make_unique<ConcurrentHashMap<EventId, WolframEvent>>())
        , input_edge_to_events(std::make_unique<ConcurrentHashMap<GlobalEdgeId, LockfreeList<EventId>*>>())
        , output_edge_to_events(std::make_unique<ConcurrentHashMap<GlobalEdgeId, LockfreeList<EventId>*>>())
        , hash_strategy_type(HashStrategyType::UNIQUENESS_TREE)
        , canonicalize_events(enable_canonicalize_events)
        , deduplicate_events(enable_deduplicate_events)
        , full_event_canonicalization(enable_full_event_canonicalization)
        , full_capture(enable_full_capture)
        , full_capture_non_canonicalised(enable_full_capture_non_canonicalised)
        , transitive_reduction_enabled(enable_transitive_reduction)
        , early_termination_enabled(enable_early_termination)
        , max_successor_states_per_parent_(max_successor_states_per_parent)
        , max_states_per_step_(max_states_per_step)
        , exploration_probability_(exploration_probability)
    {
        if (full_capture || full_capture_non_canonicalised) {
            states = std::make_unique<ConcurrentHashMap<StateID, std::shared_ptr<WolframState>>>();
            match_caches = std::make_unique<ConcurrentHashMap<StateID, StateMatchCache>>();
        }
        if (canonicalize_states) {
            seen_hashes = std::make_unique<ConcurrentHashMap<std::size_t, StateID>>();
        }
        // Event canonicalization only makes sense when state canonicalization is enabled
        if (canonicalize_states && canonicalize_events) {
            seen_event_signatures = std::make_unique<ConcurrentHashMap<EventSignature, EventId>>();
        }
        // Initialize pruning tracking structures if limits are enabled
        if (max_successor_states_per_parent_ > 0) {
            successor_counts_ = std::make_unique<ConcurrentHashMap<StateID, std::shared_ptr<std::atomic<std::size_t>>>>();
        }
        if (max_states_per_step_ > 0) {
            states_per_step_ = std::make_unique<ConcurrentHashMap<std::size_t, std::shared_ptr<std::atomic<std::size_t>>>>();
        }
    }

    /**
     * Create initial state with given hypergraph structure.
     * Returns shared_ptr to avoid copies.
     */
    std::shared_ptr<WolframState> create_initial_state(const std::vector<std::vector<GlobalVertexId>>& edge_structures);

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
     * Get fresh state ID.
     */
    StateID get_fresh_state_id() {
        return StateID{next_state_id.fetch_add(1)};
    }

    /**
     * Get all initial state IDs.
     * Returns a vector of state IDs for all states created as initial states.
     */
    const std::vector<StateID>& get_initial_state_ids() const {
        return initial_state_ids;
    }

    /**
     * Enable or disable canonicalization-based state deduplication.
     */
    void set_canonicalize_states(bool enabled) {
        canonicalize_states = enabled;
    }

    bool is_canonicalize_states() const {
        return canonicalize_states;
    }

    /**
     * Set hash strategy type (CANONICALIZATION or UNIQUENESS_TREE).
     * Only used when canonicalization is enabled.
     */
    void set_hash_strategy_type(HashStrategyType type) {
        hash_strategy_type = type;
    }

    HashStrategyType get_hash_strategy_type() const {
        return hash_strategy_type;
    }

    /**
     * Enable or disable early termination on duplicate states.
     */
    void set_early_termination_enabled(bool enabled) {
        early_termination_enabled = enabled;
    }

    bool is_early_termination_enabled() const {
        return early_termination_enabled;
    }

    /**
     * Pruning control accessors.
     */
    std::size_t get_max_successor_states_per_parent() const {
        return max_successor_states_per_parent_;
    }

    std::size_t get_max_states_per_step() const {
        return max_states_per_step_;
    }

    double get_exploration_probability() const {
        return exploration_probability_;
    }

    /**
     * Try to reserve a successor slot for a parent state (hard limit enforcement).
     * Returns true if successfully reserved, false if limit reached.
     */
    bool try_reserve_successor_slot(StateID parent_state_id);

    /**
     * Release a reserved successor slot (for rollback in multi-limit scenarios).
     */
    void release_successor_slot(StateID parent_state_id);

    /**
     * Try to reserve a state slot for a given step (hard limit enforcement).
     * Returns true if successfully reserved, false if limit reached.
     */
    bool try_reserve_step_slot(std::size_t step);

    /**
     * Release a reserved step slot (for rollback in multi-limit scenarios).
     */
    void release_step_slot(std::size_t step);

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
     * Record a state transition as an event in the multiway graph.
     * ONLY called from job system tasks - not public API!
     */
    EventId record_state_transition(const std::shared_ptr<WolframState>& input_state,
                                   const std::shared_ptr<WolframState>& output_state,
                                   const std::vector<GlobalEdgeId>& consumed_edges,
                                   const std::vector<GlobalEdgeId>& produced_edges,
                                   const std::size_t rule_index,
                                   const std::size_t evolution_step);

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
        if (canonicalize_states && seen_hashes) {
            return seen_hashes->size();  // Unique canonical states when canonicalization enabled
        } else if (full_capture && states) {
            return states->size();  // Raw state count when canonicalization disabled
        } else {
            return next_state_id.load();  // All states unique when no canonicalization and no storage
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
    std::vector<std::shared_ptr<WolframState>> get_all_states() const;

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
    void mark_state_exhausted(StateID state_id);

    /**
     * Check if a state is exhausted.
     * Only works when full_capture is enabled.
     */
    bool is_state_exhausted(StateID state_id) const;

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
    void cache_pattern_match(StateID state_id, std::size_t rule_index,
                           const std::vector<GlobalEdgeId>& matched_edges,
                           const VariableAssignment& assignment);

    /**
     * Get cached matches for a state and rule.
     * Only works when full_capture is enabled.
     */
    std::vector<CachedMatch> get_cached_matches(StateID state_id, std::size_t rule_index) const;

    /**
     * Forward valid matches from input state to output state, invalidating those using deleted edges.
     * Uses shared_ptr to avoid copying match data - much more memory efficient.
     * Only works when full_capture is enabled.
     */
    void forward_matches(StateID input_state_id, StateID output_state_id,
                        const std::vector<GlobalEdgeId>& deleted_edges);

    /**
     * Check if a state has any cached matches.
     * Only works when full_capture is enabled.
     */
    bool has_cached_matches(StateID state_id) const;

    /**
     * Clear match cache for a state (when state is exhausted).
     * Only works when full_capture is enabled.
     */
    void clear_match_cache(StateID state_id);

    /**
     * Reconstruct a state by replaying events from the initial state.
     * This is memory-efficient as we don't store full state copies.
     */
    std::optional<WolframState> reconstruct_state(StateID target_state_id) const;

    /**
     * Find the sequence of events needed to reach a target state from the initial state.
     */
    std::vector<EventId> find_event_path_to_state(StateID target_state_id) const;

    /**
     * Get state by ID, reconstructing it if necessary.
     * This replaces direct state storage with on-demand reconstruction.
     */
    std::shared_ptr<WolframState> get_state_efficient(StateID state_id) const;

    /**
     * Get canonical hash for a state without storing the full state.
     */
    std::optional<std::size_t> get_state_hash(StateID state_id) const;

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

#endif // HYPERGRAPH_WOLFRAM_STATES_HPP