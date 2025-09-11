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
    void rebuild_signature_index() {
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

public:
    explicit WolframState() {}

    StateId id(bool canonicalization_enabled = true) const { 
        return compute_hash(canonicalization_enabled);
    }

    /**
     * Add edge with global IDs.
     */
    void add_global_edge(GlobalEdgeId edge_id, const std::vector<GlobalVertexId>& vertices) {
        global_edges_.emplace_back(edge_id, vertices);
        for (GlobalVertexId vertex : vertices) {
            global_vertices_.insert(vertex);
        }
        cached_canonical_form_.reset(); // Invalidate cache

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

    /**
     * Remove edge by global ID.
     */
    bool remove_global_edge(GlobalEdgeId edge_id) {
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

            cached_canonical_form_.reset(); // Invalidate cache
            rebuild_signature_index(); // Rebuild index after edge removal
            return true;
        }
        return false;
    }

    const std::vector<GlobalHyperedge>& edges() const { return global_edges_; }
    const std::unordered_set<GlobalVertexId>& vertices() const { return global_vertices_; }

    std::size_t num_edges() const { return global_edges_.size(); }
    std::size_t num_vertices() const { return global_vertices_.size(); }

    /**
     * Get mapping from canonical vertex IDs to original global vertex IDs.
     * This is needed during rewrite operations to map back from canonical space.
     */
    GlobalVertexId canonical_to_global_vertex(std::size_t canonical_id) const {
        if (!cached_vertex_mapping_.has_value()) {
            to_canonical_hypergraph(); // This computes the mapping
        }
        if (canonical_id < cached_vertex_mapping_->canonical_to_original.size()) {
            return cached_vertex_mapping_->canonical_to_original[canonical_id];
        }
        return canonical_id; // Fallback - shouldn't happen
    }

    /**
     * Get mapping from global vertex ID to canonical vertex ID.
     */
    std::optional<std::size_t> global_to_canonical_vertex(GlobalVertexId global_id) const {
        if (!cached_vertex_mapping_.has_value()) {
            to_canonical_hypergraph(); // This computes the mapping
        }

        auto it = cached_vertex_mapping_->original_to_canonical.find(global_id);
        if (it != cached_vertex_mapping_->original_to_canonical.end()) {
            return it->second;
        }

        return std::nullopt;
    }

    /**
     * Get mapping from canonical edge IDs to original global edge IDs.
     */
    GlobalEdgeId canonical_to_global_edge(std::size_t canonical_edge_id) const {
        if (!canonical_edge_to_global_mapping_.has_value()) {
            to_canonical_hypergraph(); // This computes the mapping
        }
        if (canonical_edge_id < canonical_edge_to_global_mapping_->size()) {
            return (*canonical_edge_to_global_mapping_)[canonical_edge_id];
        }
        return canonical_edge_id; // Fallback - shouldn't happen
    }

private:

public:
    /**
     * Convert to canonical hypergraph and build edge mapping.
     */
    Hypergraph to_canonical_hypergraph() const {
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
        cached_vertex_mapping_ = canonical_result.vertex_mapping;

        // Build mapping from canonical edge index to original global edge ID
        // The canonicalization sorts the edges, so we need to track the mapping
        std::vector<GlobalEdgeId> edge_mapping;
        for (std::size_t canon_idx = 0; canon_idx < canonical_edges.size(); ++canon_idx) {
            // Find which original edge this canonical edge came from
            for (std::size_t orig_idx = 0; orig_idx < edges.size(); ++orig_idx) {
                // Apply the same vertex mapping to the original edge
                std::vector<std::size_t> mapped_orig_edge;
                for (auto v : edges[orig_idx]) {
                    // Find vertex in canonical mapping
                    auto it = cached_vertex_mapping_->original_to_canonical.find(v);
                    if (it != cached_vertex_mapping_->original_to_canonical.end()) {
                        mapped_orig_edge.push_back(it->second);
                    }
                }

                if (mapped_orig_edge == canonical_edges[canon_idx]) {
                    edge_mapping.push_back(original_edge_ids[orig_idx]);
                    break;
                }
            }
        }
        canonical_edge_to_global_mapping_ = edge_mapping;

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

    /**
     * Compute hash for state identification
     * Uses canonical form if canonicalization_enabled, otherwise raw hypergraph
     */
    std::size_t compute_hash(bool canonicalization_enabled = true) const {
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
            DEBUG_LOG("%s\n", debug_stream.str().c_str());
        }

        std::vector<std::vector<std::size_t>> edges_to_hash;
        if (canonicalization_enabled) {
            // Use Canonicalizer to get canonical form
            Canonicalizer canonicalizer;
            auto result = canonicalizer.canonicalize_edges(edges);
            edges_to_hash = result.canonical_form.edges;
            // Cache the result for later use
            cached_canonical_form_ = result.canonical_form;
            cached_vertex_mapping_ = result.vertex_mapping;
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
                DEBUG_LOG("%s\n", debug_stream.str().c_str());
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
        std::size_t hash_value = 0;
        std::hash<std::size_t> hasher;

        for (const auto& edge : edges_to_hash) {
            std::size_t edge_hash = edge.size(); // Start with arity
            for (std::size_t vertex : edge) {
                edge_hash ^= hasher(vertex) + 0x9e3779b9 + (edge_hash << 6) + (edge_hash >> 2);
            }
            hash_value ^= edge_hash + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }

        return hash_value;
    }

    /**
     * Legacy method for compatibility - uses canonical hash
     */
    std::size_t compute_canonical_hash() const {
        return compute_hash(true);
    }

    /**
     * Get canonical form (cached for performance).
     */
    const CanonicalForm& get_canonical_form() const {
        if (!cached_canonical_form_) {
            // Use Canonicalizer to get canonical form
            std::vector<std::vector<GlobalVertexId>> edges;
            for (const auto& global_edge : global_edges_) {
                edges.push_back(global_edge.global_vertices);
            }
            
            Canonicalizer canonicalizer;
            auto result = canonicalizer.canonicalize_edges(edges);
            cached_canonical_form_ = result.canonical_form;
            cached_vertex_mapping_ = result.vertex_mapping;
        }
        return *cached_canonical_form_;
    }

    /**
     * Check structural equality (same canonical form).
     */
    bool structurally_equal(const WolframState& other) const {
        return get_canonical_form() == other.get_canonical_form();
    }

    /**
     * Get edges containing a specific global vertex.
     */
    std::vector<GlobalEdgeId> edges_containing(GlobalVertexId vertex_id) const {
        std::vector<GlobalEdgeId> result;
        for (const auto& edge : global_edges_) {
            if (edge.contains(vertex_id)) {
                result.push_back(edge.global_id);
            }
        }
        return result;
    }

    /**
     * Get edge by global ID.
     */
    const GlobalHyperedge* get_edge(GlobalEdgeId edge_id) const {
        auto it = std::find_if(global_edges_.begin(), global_edges_.end(),
            [edge_id](const GlobalHyperedge& edge) { return edge.global_id == edge_id; });
        return (it != global_edges_.end()) ? &(*it) : nullptr;
    }

    /**
     * Find edges compatible with a pattern signature.
     * This is key for fast pattern matching using the signature index.
     */
    std::vector<GlobalEdgeId> find_edges_by_pattern_signature(const EdgeSignature& pattern_sig) const {
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

    WolframState clone() const {
        WolframState copy; // New state will compute hash when needed
        copy.global_edges_ = global_edges_;
        copy.global_vertices_ = global_vertices_;
        copy.edge_signature_index_ = edge_signature_index_;
        return copy;
    }
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
    void invalidate_matches_using_edges(const std::vector<GlobalEdgeId>& deleted_edges) {
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

    /**
     * Get all valid matches for a specific rule.
     */
    std::vector<CachedMatch> get_matches_for_rule(std::size_t rule_index) const {
        std::vector<CachedMatch> result;
        for (const auto& shared_match : cached_matches) {
            if (shared_match->rule_index == rule_index) {
                result.push_back(*shared_match);  // Dereference shared_ptr
            }
        }
        return result;
    }

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

    MultiwayGraph& operator=(MultiwayGraph&& other) noexcept {
        if (this != &other) {
            // Move unique_ptrs directly - no locking needed
            states_ = std::move(other.states_);
            events_ = std::move(other.events_);

            // Clean up old event edges list and take ownership of new one
            EventEdgeNode* old_head = event_edges_head_.exchange(other.event_edges_head_.exchange(nullptr));
            while (old_head) {
                EventEdgeNode* next = old_head->next.load();
                delete old_head;
                old_head = next;
            }
            event_edges_count_ = other.event_edges_count_.load();

            input_edge_to_events_ = std::move(other.input_edge_to_events_);
            output_edge_to_events_ = std::move(other.output_edge_to_events_);
            exhausted_states_ = std::move(other.exhausted_states_);
            match_caches_ = std::move(other.match_caches_);
            next_event_id_ = other.next_event_id_.load();
            next_global_vertex_id_ = other.next_global_vertex_id_.load();
            next_global_edge_id_ = other.next_global_edge_id_.load();
        }
        return *this;
    }

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
    StateId create_initial_state(const std::vector<std::vector<GlobalVertexId>>& edge_structures) {
        WolframState state;

        // Track the maximum vertex ID seen in the initial state
        GlobalVertexId max_vertex_id = 0;

        for (const auto& vertices : edge_structures) {
            GlobalEdgeId edge_id = next_global_edge_id_.fetch_add(1);
            state.add_global_edge(edge_id, vertices);

            // Find the maximum vertex ID in this edge
            for (GlobalVertexId vertex_id : vertices) {
                if (vertex_id > max_vertex_id) {
                    max_vertex_id = vertex_id;
                }
            }
        }

        // Update next_global_vertex_id_ to ensure fresh vertices don't overlap
        if (!edge_structures.empty() && max_vertex_id > 0) {
            GlobalVertexId min_required = max_vertex_id + 1;
            GlobalVertexId current = next_global_vertex_id_.load();
            if (current < min_required) {
                // Atomically add the difference to bring it up to the required minimum
                next_global_vertex_id_.fetch_add(min_required - current);
            }
        }

        // Get state hash (this will use canonicalization if enabled)
        StateId state_hash = state.id(canonicalization_enabled_);
        
        // Store state using its hash as the key
        bool was_inserted = states_->insert(state_hash, std::move(state));
        if (was_inserted) {
            states_count_.fetch_add(1);
        }
        return state_hash;
    }

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
                           GlobalVertexId anchor_vertex) {
        // No locking needed with concurrent hash map
        auto input_state_opt = states_->find(input_state_id);
        if (!input_state_opt.has_value()) {
            return INVALID_EVENT;
        }

        // Create new state by cloning input state
        WolframState output_state = input_state_opt->clone();
        
        // Remove consumed edges
        for (GlobalEdgeId edge_id : edges_to_remove) {
            output_state.remove_global_edge(edge_id);
        }

        // Add produced edges
        std::vector<GlobalEdgeId> produced_edge_ids;
        for (const auto& vertices : edges_to_add) {
            GlobalEdgeId edge_id = next_global_edge_id_.fetch_add(1);
            output_state.add_global_edge(edge_id, vertices);
            produced_edge_ids.push_back(edge_id);
        }

        // Get state hash (this will use canonicalization if enabled)
        StateId output_state_id = output_state.id(canonicalization_enabled_);
        
        DEBUG_LOG("[DEBUG] Computing state hash for edges count=%zu", 
                 edges_to_add.size());
        DEBUG_LOG("[DEBUG] Output state hash: %zu", 
                 output_state_id);
        
        // Use insert_or_get to atomically handle deduplication
        auto [existing_state, was_inserted] = states_->insert_or_get(output_state_id, output_state);
        if (was_inserted) {
            // New unique state - we successfully inserted it
            {
                std::ostringstream debug_stream;
                debug_stream << "NEW UNIQUE STATE " << output_state_id << "\nRAW: ";
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
                if (canonicalization_enabled_) {
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
            states_count_.fetch_add(1);
        } else {
            // State already exists - deduplicated
            DEBUG_LOG("DUPLICATE: reusing state %zu", 
                     output_state_id);
        }

        // Create event
        EventId event_id = next_event_id_.fetch_add(1);
        WolframEvent event(event_id, input_state_id, output_state_id,
                          edges_to_remove, produced_edge_ids,
                          rule_index, anchor_vertex);

        events_->insert(event_id, std::move(event));
        events_count_.fetch_add(1);

        // Update edge-to-event mappings and event relationships
        update_edge_mappings(event_id, edges_to_remove, produced_edge_ids);
        update_event_relationships(event_id);

        // Forward valid matches from input state to output state
        forward_matches(input_state_id, output_state_id, edges_to_remove);

        return event_id;
    }

    /**
     * Update edge-to-event mappings for a new event.
     */
    void update_edge_mappings(EventId event_id,
                             const std::vector<GlobalEdgeId>& input_edges,
                             const std::vector<GlobalEdgeId>& output_edges) {
        // Intentionally unused - method is a no-op placeholder
        (void)event_id;
        
        // Add input edge mappings
        for (GlobalEdgeId edge_id : input_edges) {
            (void)edge_id; // Intentionally unused
            // Note: This should be removed since we build mappings on-demand\n            // input_edge_to_events_[edge_id].push_back(event_id);
        }

        // Add output edge mappings
        for (GlobalEdgeId edge_id : output_edges) {
            (void)edge_id; // Intentionally unused
            // Note: This should be removed since we build mappings on-demand\n            // output_edge_to_events_[edge_id].push_back(event_id);
        }
    }

    /**
     * Update causal and branchial relationships between events based on edge overlap.
     * Note: This is now done in post-processing phase to avoid concurrent map updates
     */
    void update_event_relationships(EventId new_event_id) {
        // Event relationships are now computed in the post-processing phase
        // This eliminates the need for concurrent edge-to-event mappings during evolution
        // See compute_event_relationships() method in the lock-free implementation
        (void)new_event_id; // Suppress unused parameter warning
    }

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
    std::vector<EventId> get_concurrent_events(const std::vector<EventId>& candidate_events) const {

        std::vector<EventId> concurrent;

        for (EventId event_id : candidate_events) {
            bool has_causal_dependency = false;

            // Check if this event has causal dependencies on other candidate events
            EventEdgeNode* current = event_edges_head_.load();
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
    std::size_t get_causal_edge_count() const {
        std::size_t count = 0;
        EventEdgeNode* current = event_edges_head_.load();
        while (current != nullptr) {
            if (current->edge.type == EventRelationType::CAUSAL) {
                count++;
            }
            current = current->next.load();
        }
        return count;
    }

    /**
     * Get count of branchial event edges.
     */
    std::size_t get_branchial_edge_count() const {
        std::size_t count = 0;
        EventEdgeNode* current = event_edges_head_.load();
        while (current != nullptr) {
            if (current->edge.type == EventRelationType::BRANCHIAL) {
                count++;
            }
            current = current->next.load();
        }
        return count;
    }

    /**
     * Get all states for external access.
     */
    std::vector<WolframState> get_all_states() const {
        std::vector<WolframState> result;
        result.reserve(states_->size());

        states_->for_each([&result](const StateId& id, const WolframState& state) {
            (void)id; // Unused - only need the state
            result.push_back(state);
        });

        return result;
    }

    /**
     * Get all events for external access.
     */
    std::vector<WolframEvent> get_all_events() const {
        std::vector<WolframEvent> result;
        result.reserve(events_->size());

        events_->for_each([&result](const EventId& id, const WolframEvent& event) {
            (void)id; // Unused - only need the event
            result.push_back(event);
        });

        return result;
    }

    /**
     * Get event edges (causal and branchial) for external access.
     */
    std::vector<EventEdge> get_event_edges() const {
        std::vector<EventEdge> edges;
        EventEdgeNode* current = event_edges_head_.load();
        while (current != nullptr) {
            edges.push_back(current->edge);
            current = current->next.load();
        }
        return edges;
    }

    /**
     * Mark a state as exhausted (no more applicable rules) and clean up memory.
     * Simple and effective - let shared_ptr handle the rest.
     */
    void mark_state_exhausted(StateId state_id) {
        exhausted_states_->insert(state_id);

        // Free the state from memory to reduce memory usage
        if (states_->contains(state_id)) {
            DEBUG_LOG("  Freeing exhausted state %zu from memory", state_id);
            states_->erase(state_id);
            states_count_.fetch_sub(1);
        }

        // Clear match cache - shared_ptr will handle cleanup automatically
        match_caches_->erase(state_id);
    }

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

    /**
     * Add pattern match to state cache.
     */
    void cache_pattern_match(StateId state_id, std::size_t rule_index,
                           const std::vector<GlobalEdgeId>& matched_edges,
                           const VariableAssignment& assignment,
                           GlobalVertexId anchor_vertex) {

        if (!match_caches_->contains(state_id)) {
            match_caches_->insert(state_id, StateMatchCache(state_id));
        }

        auto cache_opt = match_caches_->find(state_id);
        if (cache_opt) {
            // Since find() returns a copy, we need to modify and re-insert
            auto cache = cache_opt.value();
            cache.add_match(CachedMatch(rule_index, matched_edges, assignment, anchor_vertex));
            match_caches_->insert(state_id, cache);
        }
    }

    /**
     * Get cached matches for a state and rule.
     */
    std::vector<CachedMatch> get_cached_matches(StateId state_id, std::size_t rule_index) const {

        auto cache_opt = match_caches_->find(state_id);
        if (cache_opt) {
            return cache_opt.value().get_matches_for_rule(rule_index);
        }
        return {};
    }

    /**
     * Forward valid matches from input state to output state, invalidating those using deleted edges.
     * Uses shared_ptr to avoid copying match data - much more memory efficient.
     */
    void forward_matches(StateId input_state_id, StateId output_state_id,
                        const std::vector<GlobalEdgeId>& deleted_edges) {
        auto input_cache_opt = match_caches_->find(input_state_id);
        if (!input_cache_opt) {
            return; // No matches to forward
        }

        // Create output cache if it doesn't exist
        if (!match_caches_->contains(output_state_id)) {
            match_caches_->insert(output_state_id, StateMatchCache(output_state_id));
        }

        auto output_cache_opt = match_caches_->find(output_state_id);
        if (!output_cache_opt) {
            return; // Failed to create output cache
        }

        auto input_cache = input_cache_opt.value();
        auto output_cache = output_cache_opt.value();

        // Forward matches that don't use deleted edges (using shared_ptr - no copying!)
        for (const auto& shared_match : input_cache.cached_matches) {
            bool uses_deleted_edge = false;

            // Check if this match uses any deleted edge
            for (GlobalEdgeId deleted_edge : deleted_edges) {
                if (std::find(shared_match->matched_edges.begin(), shared_match->matched_edges.end(), deleted_edge)
                    != shared_match->matched_edges.end()) {
                    uses_deleted_edge = true;
                    break;
                }
            }

            // Forward the shared_ptr if the match is still valid (no copying!)
            if (!uses_deleted_edge) {
                output_cache.add_shared_match(shared_match);  // Just copies shared_ptr, not data
            }
        }

        // Re-insert the updated output cache
        match_caches_->insert(output_state_id, output_cache);
    }

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
    std::optional<WolframState> reconstruct_state(StateId target_state_id) const {

        // Find the path of events from initial state to target state
        std::vector<EventId> event_path = find_event_path_to_state(target_state_id);

        // If path is empty, this might be the initial state itself
        if (event_path.empty()) {
            auto state_opt = states_->find(target_state_id);
            if (state_opt) {
                return *state_opt; // Return the state directly if it exists
            }
            return std::nullopt; // State not found
        }

        // Start with the initial state (assume state ID 1 is always initial)
        StateId initial_state_id = 1;
        auto initial_state_opt = states_->find(initial_state_id);
        if (!initial_state_opt) {
            return std::nullopt; // Initial state not found
        }

        // Clone initial state and replay events
        WolframState reconstructed = initial_state_opt.value().clone();
        // Hash will be computed automatically when needed

        for (EventId event_id : event_path) {
            auto event_opt = events_->find(event_id);
            if (!event_opt) {
                return std::nullopt; // Event not found
            }

            const auto& event = event_opt.value();

            // Apply the event: remove consumed edges, add produced edges
            for (GlobalEdgeId edge_id : event.consumed_edges) {
                reconstructed.remove_global_edge(edge_id);
            }

            // For produced edges, we need to reconstruct their vertices from the rule application
            // This would require storing more information in events or having access to rules
            // For now, this is a placeholder - we'd need the actual edge structures
        }

        return reconstructed;
    }

    /**
     * Find the sequence of events needed to reach a target state from the initial state.
     */
    std::vector<EventId> find_event_path_to_state(StateId target_state_id) const {
        // This should be called while holding the lock

        std::vector<EventId> path;
        StateId current_state = target_state_id;

        // Trace backwards through events to find path to initial state
        while (current_state != 1) { // Assume state 1 is initial
            bool found_parent = false;

            events_->for_each([&](EventId event_id, const WolframEvent& event) {
                if (!found_parent && event.output_state_id == current_state) {
                    path.insert(path.begin(), event_id); // Insert at beginning
                    current_state = event.input_state_id;
                    found_parent = true;
                }
            });

            if (!found_parent) {
                return {}; // Cannot trace back to initial state
            }
        }

        return path;
    }

    /**
     * Get state by ID, reconstructing it if necessary.
     * This replaces direct state storage with on-demand reconstruction.
     */
    std::optional<WolframState> get_state_efficient(StateId state_id) const {

        // Check if we have the state in memory
        auto state_opt = states_->find(state_id);
        if (state_opt) {
            return state_opt.value();
        }

        // If not in memory, reconstruct it by replaying events
        return reconstruct_state(state_id);
    }

    /**
     * Get canonical hash for a state without storing the full state.
     */
    std::optional<std::size_t> get_state_hash(StateId state_id) const {
        auto state_opt = get_state_efficient(state_id);
        if (state_opt) {
            // Use canonical form hash instead of WolframState hash
            return std::hash<CanonicalForm>{}(state_opt->get_canonical_form());
        }
        return std::nullopt;
    }

    void clear() {
        // Clear lock-free structures by resetting atomic pointers
        // States and events are cleared by creating new empty structures
        states_.reset(new ConcurrentHashMap<StateId, WolframState>());
        events_.reset(new ConcurrentHashMap<EventId, WolframEvent>());

        // Reset event edges linked list
        event_edges_head_.store(nullptr);
        event_edges_count_.store(0);
        states_count_.store(0);
        events_count_.store(0);

        input_edge_to_events_->clear();
        output_edge_to_events_->clear();
        exhausted_states_->clear();
        match_caches_->clear();
        // No need to reset state counter - states are identified by hash
        next_event_id_.store(1);
        next_global_vertex_id_ = 0;
        next_global_edge_id_ = 0;
    }

    /**
     * Export multiway graph in DOT format showing states and events.
     */
    std::string export_multiway_graph_dot() const {

        std::ostringstream dot;
        dot << "digraph MultiwayGraph {\n";
        dot << "  node [shape=circle]; // states\n";
        dot << "  node [shape=box]; // events\n";

        // State and event nodes omitted for concurrent access safety

        // Add event edges (causal and branchial)
        EventEdgeNode* current = event_edges_head_.load();
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

    /**
     * Print concise summary of multiway graph state.
     */
    std::string get_summary() const {
        std::ostringstream ss;
        
        auto states = get_all_states();
        auto events = get_all_events();

        ss << "=== MULTIWAY GRAPH SUMMARY ===\n";
        ss << "States: " << states.size() << ", Events: " << events.size() << "\n\n";

        // Count causal vs branchial edges
        std::size_t causal_count = 0, branchial_count = 0;
        EventEdgeNode* current = event_edges_head_.load();
        while (current != nullptr) {
            if (current->edge.type == EventRelationType::CAUSAL) causal_count++;
            else branchial_count++;
            current = current->next.load();
        }

        ss << "States:\n";
        for (const auto& state : states) {
            ss << "  State " << state.id() << ": " << state.num_edges() << " edges\n";
            for (const auto& edge : state.edges()) {
                ss << "    Edge " << edge.global_id << ": {";
                for (size_t i = 0; i < edge.global_vertices.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << edge.global_vertices[i];
                }
                ss << "}\n";
            }
        }

        ss << "\nEvents:\n";
        for (const auto& event : events) {
            ss << "  Event " << event.event_id << ": State " << event.input_state_id
               << " → State " << event.output_state_id << " (rule " << event.rule_index << ")\n";

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