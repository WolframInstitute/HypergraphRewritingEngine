#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <functional>
#include <thread>
#include <cstring>
#include <random>
#include <algorithm>

#include "types.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "unified_hypergraph.hpp"
#include "pattern.hpp"
#include "pattern_matcher.hpp"
#include "rewriter.hpp"
#include "index.hpp"
#include "causal_graph.hpp"
#include "concurrent_map.hpp"
#include "lock_free_list.hpp"
#include "segmented_array.hpp"
#include "hypergraph/debug_log.hpp"

#include <job_system/job_system.hpp>

// Visualization event emission (compiles to no-op when disabled)
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
#include <events/viz_event_sink.hpp>
#endif

namespace hypergraph::unified {

// =============================================================================
// Match Record
// =============================================================================
// Represents a complete match found during pattern matching.

struct MatchRecord {
    uint16_t rule_index;
    EdgeId matched_edges[MAX_PATTERN_EDGES];
    uint8_t num_edges;
    VariableBinding binding;
    StateId source_state;
    StateId canonical_source;  // Canonical state for deterministic deduplication
    uint64_t source_canonical_hash{0};  // Canonical hash of source state (deterministic)
    uint64_t storage_epoch{0};  // Epoch when this match was stored (for ordering)

    // Hash for deduplication - uses source_state + matched edges + binding
    // MUST use source_state (raw state ID), NOT source_canonical_hash!
    //
    // Why: Multiple raw states can share the same canonical hash (isomorphic states).
    // If two raw states S1 and S2 both contain edge E (inherited from common ancestor),
    // matches on E in both states would have same (canonical_hash, edge, binding) and
    // incorrectly deduplicate. Using source_state ensures matches in different raw
    // states always have different hashes.
    //
    // The raw state IDs are non-deterministic across runs, but that's OK - deduplication
    // only needs to work WITHIN a single run to avoid processing the same match twice.
    uint64_t hash() const {
        // FNV-1a style hash for better distribution and collision resistance
        uint64_t h = 14695981039346656037ULL;  // FNV offset basis
        constexpr uint64_t FNV_PRIME = 1099511628211ULL;

        // Mix in rule_index
        h ^= rule_index;
        h *= FNV_PRIME;

        // Mix in source_state (raw state ID for collision-free deduplication)
        h ^= source_state;
        h *= FNV_PRIME;
        h ^= (source_state >> 16);  // Extra mixing for better distribution
        h *= FNV_PRIME;

        // Mix in matched edges
        for (uint8_t i = 0; i < num_edges; ++i) {
            h ^= matched_edges[i];
            h *= FNV_PRIME;
        }

        // Mix in bound_mask
        h ^= binding.bound_mask;
        h *= FNV_PRIME;

        // Mix in bound variables
        for (uint8_t i = 0; i < MAX_VARS; ++i) {
            if (binding.is_bound(i)) {
                h ^= (static_cast<uint64_t>(i) << 32) | binding.get(i);
                h *= FNV_PRIME;
            }
        }

        return h;
    }

    bool operator==(const MatchRecord& other) const {
        if (rule_index != other.rule_index || num_edges != other.num_edges ||
            source_state != other.source_state) return false;
        // Compare matched edges
        for (uint8_t i = 0; i < num_edges; ++i) {
            if (matched_edges[i] != other.matched_edges[i]) return false;
        }
        // Compare bindings
        for (uint8_t i = 0; i < MAX_VARS; ++i) {
            if (binding.is_bound(i) != other.binding.is_bound(i)) return false;
            if (binding.is_bound(i) && binding.get(i) != other.binding.get(i)) return false;
        }
        return true;
    }
};

// =============================================================================
// Evolution Statistics
// =============================================================================

struct EvolutionStats {
    std::atomic<size_t> states_created{0};
    std::atomic<size_t> events_created{0};
    std::atomic<size_t> matches_found{0};
    std::atomic<size_t> matches_forwarded{0};
    std::atomic<size_t> matches_invalidated{0};
    std::atomic<size_t> new_matches_discovered{0};
    std::atomic<size_t> full_pattern_matches{0};
    std::atomic<size_t> delta_pattern_matches{0};
};

// =============================================================================
// Job Types for Parallel Evolution
// =============================================================================

enum class EvolutionJobType {
    SCAN,       // Find initial candidates for first pattern edge
    EXPAND,     // Extend partial match by one edge
    SINK,       // Process complete match, spawn REWRITE
    MATCH,      // Orchestrate matching for a state (spawn SCAN tasks or fallback to sync)
    REWRITE,    // Apply a match to produce new state
};

// =============================================================================
// RewriteTask
// =============================================================================
// Captures all data needed to apply a single match in parallel.

struct RewriteTask {
    MatchRecord match;
    uint32_t step;

    RewriteTask() : step(0) {}
    RewriteTask(const MatchRecord& m, uint32_t s) : match(m), step(s) {}
};

// =============================================================================
// RewriteResult for parallel collection
// =============================================================================

struct ParallelRewriteResult {
    StateId new_state{INVALID_ID};   // Canonical state ID (for deduplication)
    StateId raw_state{INVALID_ID};   // Raw state ID (with actual produced edges, for evolution)
    EventId event{INVALID_ID};
    bool was_new_state{false};
    bool success{false};
    MatchRecord match;  // The match that produced this result

    // For match forwarding
    SparseBitset deleted_edges;
    SparseBitset created_edges;
    uint8_t num_produced{0};
    EdgeId produced_edges[MAX_PATTERN_EDGES];
};

// =============================================================================
// MatchContext for Match Forwarding
// =============================================================================
// Carries information needed for incremental match discovery.
// When a REWRITE creates a new state, it passes this context to enable
// forwarding valid parent matches and finding only NEW matches.

struct MatchContext {
    StateId parent_state{INVALID_ID};
    EdgeId consumed_edges[MAX_PATTERN_EDGES];
    uint8_t num_consumed{0};
    EdgeId produced_edges[MAX_PATTERN_EDGES];
    uint8_t num_produced{0};

    bool has_parent() const { return parent_state != INVALID_ID; }

    bool edge_was_consumed(EdgeId eid) const {
        for (uint8_t i = 0; i < num_consumed; ++i) {
            if (consumed_edges[i] == eid) return true;
        }
        return false;
    }

    bool edge_was_produced(EdgeId eid) const {
        for (uint8_t i = 0; i < num_produced; ++i) {
            if (produced_edges[i] == eid) return true;
        }
        return false;
    }
};

// =============================================================================
// SCAN/EXPAND/SINK Task Data Structures (HGMatch Dataflow Model)
// =============================================================================
// These structures capture all data needed to execute matching tasks in parallel.
// Following HGMatch paper: SCAN→EXPAND*→SINK pipeline.

// SCAN task: Find initial candidates for first pattern edge
struct ScanTaskData {
    StateId state;                          // State to match in
    uint16_t rule_index;                    // Which rule to match
    uint32_t step;                          // Evolution step
    StateId canonical_state;                // For deterministic deduplication
    uint64_t source_canonical_hash;         // Canonical hash of source state
    // For delta matching (only find NEW matches involving produced edges)
    bool is_delta;                          // If true, only match involving produced_edges
    EdgeId produced_edges[MAX_PATTERN_EDGES];
    uint8_t num_produced;
};

// EXPAND task: Extend partial match by one edge
// Also used for SINK (when match is complete)
struct ExpandTaskData {
    StateId state;                          // State being matched
    uint16_t rule_index;                    // Rule being matched
    uint8_t num_pattern_edges;              // Total edges in pattern
    uint8_t next_pattern_idx;               // Which pattern edge to match next (0-based)
    EdgeId matched_edges[MAX_PATTERN_EDGES];// Data edges matched so far
    uint8_t match_order[MAX_PATTERN_EDGES]; // Pattern indices in match order
    uint8_t num_matched;                    // Number of edges matched
    VariableBinding binding;                // Current variable bindings
    uint32_t step;                          // Evolution step
    StateId canonical_state;                // For deterministic deduplication
    uint64_t source_canonical_hash;         // Canonical hash of source state

    bool is_complete() const {
        return num_matched >= num_pattern_edges;
    }

    bool contains_edge(EdgeId eid) const {
        for (uint8_t i = 0; i < num_matched; ++i) {
            if (matched_edges[i] == eid) return true;
        }
        return false;
    }

    // Get next pattern index to match (first unmatched)
    uint8_t get_next_pattern_idx() const {
        uint32_t matched_mask = 0;
        for (uint8_t i = 0; i < num_matched; ++i) {
            matched_mask |= (1u << match_order[i]);
        }
        for (uint8_t idx = 0; idx < num_pattern_edges; ++idx) {
            if (!(matched_mask & (1u << idx))) return idx;
        }
        return num_pattern_edges;  // All matched
    }

    // Convert matched edges to pattern order
    void to_pattern_order(EdgeId* out) const {
        for (uint8_t i = 0; i < num_matched; ++i) {
            out[match_order[i]] = matched_edges[i];
        }
    }
};

// =============================================================================
// ChildInfo for Match Forwarding (Push Model)
// =============================================================================
// Tracks child states and their consumed edges so parent can push matches.

struct ChildInfo {
    StateId child_state;
    EdgeId consumed_edges[MAX_PATTERN_EDGES];
    uint8_t num_consumed;
    uint32_t creation_step{0};  // Step at which child was created
    uint64_t registration_epoch{0};  // Epoch when child was registered

    bool match_overlaps_consumed(const EdgeId* matched_edges, uint8_t num_edges) const {
        for (uint8_t i = 0; i < num_edges; ++i) {
            for (uint8_t j = 0; j < num_consumed; ++j) {
                if (matched_edges[i] == consumed_edges[j]) return true;
            }
        }
        return false;
    }
};

// =============================================================================
// ParentInfo for Match Forwarding (Pull Model from Ancestors)
// =============================================================================
// Tracks each state's parent and consumed edges so we can forward from ancestors.

struct ParentInfo {
    StateId parent_state;
    EdgeId consumed_edges[MAX_PATTERN_EDGES];
    uint8_t num_consumed;

    ParentInfo() : parent_state(INVALID_ID), num_consumed(0) {}

    bool has_parent() const { return parent_state != INVALID_ID; }

    bool match_overlaps_consumed(const EdgeId* matched_edges, uint8_t num_edges) const {
        for (uint8_t i = 0; i < num_edges; ++i) {
            for (uint8_t j = 0; j < num_consumed; ++j) {
                if (matched_edges[i] == consumed_edges[j]) return true;
            }
        }
        return false;
    }
};

// =============================================================================
// ParallelEvolutionEngine
// =============================================================================
// Dataflow-driven evolution engine with maximal parallelism.
// Uses job system for concurrent match finding and rewrite application.
//
// DATAFLOW MODEL (No synchronization barriers):
// - MATCH tasks find matches in a state, spawn REWRITE tasks
// - REWRITE tasks apply matches, spawn MATCH tasks for new states
// - Single wait_for_completion() at the end
// - Work proceeds continuously as dependencies are satisfied
//
// MATCH FORWARDING (Incremental Pattern Matching):
// - Full pattern matching only on initial states
// - For child states: forward valid parent matches, find only NEW matches
// - NEW matches must involve at least one newly produced edge
//
// Thread safety model (LOCK-FREE):
// - UnifiedHypergraph uses lock-free data structures
// - ConcurrentHeterogeneousArena for thread-safe allocation
// - Match deduplication uses ConcurrentMap (lock-free)
// - State tracking uses ConcurrentMap (lock-free)

class ParallelEvolutionEngine {
    UnifiedHypergraph* hg_;
    Rewriter rewriter_;

    // Rules
    std::vector<RewriteRule> rules_;

    // Global match deduplication (lock-free)
    // Use non-zero EMPTY/LOCKED keys to avoid conflicts with valid hash values
    static constexpr uint64_t MATCH_MAP_EMPTY = 1ULL << 63;
    static constexpr uint64_t MATCH_MAP_LOCKED = (1ULL << 63) | 1;
    ConcurrentMap<uint64_t, bool, MATCH_MAP_EMPTY, MATCH_MAP_LOCKED> seen_match_hashes_;

    // Track which raw states have been matched (lock-free)
    // Prevents duplicate MATCH tasks for the same raw state
    // Use uint64_t as key to avoid template issues with 32-bit StateId
    // StateId is 32-bit, so we use keys outside that range for EMPTY/LOCKED
    static constexpr uint64_t STATE_MAP_EMPTY = 1ULL << 62;
    static constexpr uint64_t STATE_MAP_LOCKED = (1ULL << 62) | 1;
    ConcurrentMap<uint64_t, bool, STATE_MAP_EMPTY, STATE_MAP_LOCKED> matched_raw_states_;

    // Per-state match storage for match forwarding
    // Maps state -> list of matches found in that state
    // Used to forward matches to child states
    // Uses ConcurrentMap for thread-safe "get or create" semantics
    static constexpr uint64_t MATCH_STATE_MAP_EMPTY = (1ULL << 62) + 100;
    static constexpr uint64_t MATCH_STATE_MAP_LOCKED = (1ULL << 62) + 101;
    ConcurrentMap<uint64_t, LockFreeList<MatchRecord>*, MATCH_STATE_MAP_EMPTY, MATCH_STATE_MAP_LOCKED> state_matches_;

    // Per-state children tracking for push-based match forwarding
    // Maps parent state -> list of children (with their consumed edges)
    // When parent finds a match, it pushes to all children where match is valid
    static constexpr uint64_t CHILDREN_MAP_EMPTY = (1ULL << 62) + 200;
    static constexpr uint64_t CHILDREN_MAP_LOCKED = (1ULL << 62) + 201;
    ConcurrentMap<uint64_t, LockFreeList<ChildInfo>*, CHILDREN_MAP_EMPTY, CHILDREN_MAP_LOCKED> state_children_;

    // Per-state parent tracking for pull-based match forwarding from ancestors
    // Maps child state -> parent info pointer (with consumed edges for validation)
    static constexpr uint64_t PARENT_MAP_EMPTY = (1ULL << 62) + 300;
    static constexpr uint64_t PARENT_MAP_LOCKED = (1ULL << 62) + 301;
    ConcurrentMap<uint64_t, ParentInfo*, PARENT_MAP_EMPTY, PARENT_MAP_LOCKED> state_parent_;

    // Per-state registration epoch for epoch-based match forwarding
    // Tracks when each state was registered as a child (for epoch comparison)
    static constexpr uint64_t EPOCH_MAP_EMPTY = (1ULL << 62) + 400;
    static constexpr uint64_t EPOCH_MAP_LOCKED = (1ULL << 62) + 401;
    ConcurrentMap<uint64_t, uint64_t, EPOCH_MAP_EMPTY, EPOCH_MAP_LOCKED> state_registration_epoch_;

    // Global epoch counter for ordering matches and registrations
    // Used to determine if push or pull should handle each (match, child) pair:
    // - If match.epoch < child.epoch: child pulls (match was stored before child registered)
    // - If match.epoch >= child.epoch: parent pushes (match stored after child registered)
    std::atomic<uint64_t> global_epoch_{1};  // Start at 1 to avoid 0 confusion

    // Match forwarding enabled flag
    bool enable_match_forwarding_{true};

    // Batched matching: collect all matches then spawn REWRITEs (vs eager spawning)
    // Batching eliminates race conditions in match forwarding, but eager may have
    // better cache locality for some workloads. When disabled with match forwarding,
    // requires push-based forwarding to cover race windows.
    bool batched_matching_{false};  // Disabled to test eager single-threaded

    // Validation mode: compare forwarded+delta vs full matching
    bool validate_match_forwarding_{false};  // Enabled for debugging
    std::atomic<size_t> validation_mismatches_{0};

    // Genesis events: create synthetic events for initial states that produce
    // all initial edges. This enables causal edges from initial state to gen 1.
    // Disabled by default to maintain backwards compatibility with tests.
    bool enable_genesis_events_{false};

    // Task-based matching: use SCAN→EXPAND→SINK task decomposition (HGMatch model)
    // When enabled, pattern matching spawns fine-grained tasks for better parallelism.
    // When disabled (default), uses synchronous find_matches() within MATCH task.
    bool task_based_matching_{true};

    // Track missing hashes to verify they arrive later via push
    // Value is (state_id << 16) | rule_index for debugging
    ConcurrentMap<uint64_t, uint64_t> missing_match_hashes_{4096};
    std::atomic<size_t> late_arrivals_{0};  // Matches that arrived after validation

    // Job system
    std::unique_ptr<job_system::JobSystem<EvolutionJobType>> job_system_;
    size_t num_threads_{0};

    // Evolution control
    std::atomic<bool> should_stop_{false};
    size_t max_steps_{0};
    size_t max_states_{0};
    size_t max_events_{0};

    // Pruning and random termination (v1 compatibility)
    double exploration_probability_{1.0};          // Probability of exploring each new state (1.0 = always)
    size_t max_successor_states_per_parent_{0};    // Max children per parent state (0 = unlimited)
    size_t max_states_per_step_{0};                // Max new states per generation/step (0 = unlimited)

    // Per-parent successor count tracking (for max_successor_states_per_parent)
    static constexpr uint64_t SUCCESSOR_MAP_EMPTY = (1ULL << 62) + 500;
    static constexpr uint64_t SUCCESSOR_MAP_LOCKED = (1ULL << 62) + 501;
    ConcurrentMap<uint64_t, std::atomic<size_t>*, SUCCESSOR_MAP_EMPTY, SUCCESSOR_MAP_LOCKED> parent_successor_count_;

    // Per-step state count tracking (for max_states_per_step)
    static constexpr uint64_t STEP_MAP_EMPTY = (1ULL << 62) + 600;
    static constexpr uint64_t STEP_MAP_LOCKED = (1ULL << 62) + 601;
    ConcurrentMap<uint64_t, std::atomic<size_t>*, STEP_MAP_EMPTY, STEP_MAP_LOCKED> states_per_step_;

    // Statistics (atomics for thread-safety)
    std::atomic<size_t> total_matches_found_{0};
    std::atomic<size_t> total_rewrites_{0};
    std::atomic<size_t> total_new_states_{0};
    std::atomic<size_t> total_events_{0};
    std::atomic<size_t> forwarded_matches_{0};
    std::atomic<size_t> rejected_duplicates_{0};

    // Evolution statistics
    EvolutionStats stats_;

public:
    ParallelEvolutionEngine()
        : hg_(nullptr)
        , rewriter_(nullptr)
    {}

    explicit ParallelEvolutionEngine(UnifiedHypergraph* hg, size_t num_threads = 0)
        : hg_(hg)
        , rewriter_(hg)
        , num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency())
    {
        job_system_ = std::make_unique<job_system::JobSystem<EvolutionJobType>>(num_threads_);
        job_system_->start();
    }

    ~ParallelEvolutionEngine() {
        if (job_system_) {
            job_system_->shutdown();
        }
    }

    // Non-copyable
    ParallelEvolutionEngine(const ParallelEvolutionEngine&) = delete;
    ParallelEvolutionEngine& operator=(const ParallelEvolutionEngine&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    void add_rule(const RewriteRule& rule) {
        rules_.push_back(rule);
    }

    void set_max_steps(size_t max) { max_steps_ = max; }
    void set_max_states(size_t max) { max_states_ = max; }
    void set_max_events(size_t max) { max_events_ = max; }
    void set_match_forwarding(bool enable) { enable_match_forwarding_ = enable; }
    void set_batched_matching(bool enable) { batched_matching_ = enable; }
    void set_validate_match_forwarding(bool enable) { validate_match_forwarding_ = enable; }

    // Enable online transitive reduction for causal edges (Goranci algorithm)
    // When enabled, redundant causal edges are filtered out at insertion time.
    // Disabled by default for v1 compatibility.
    void set_transitive_reduction(bool enable) {
        if (hg_) hg_->causal_graph().set_transitive_reduction(enable);
    }

    // Enable genesis events for initial states.
    // When enabled, a synthetic event is created for each initial state that
    // "produces" all edges in that state. This allows causal edges to be
    // tracked from the initial state's edges to events that consume them.
    void set_genesis_events(bool enable) { enable_genesis_events_ = enable; }

    // Enable task-based matching (HGMatch SCAN→EXPAND→SINK model)
    // When enabled, pattern matching spawns fine-grained tasks for better parallelism.
    // When disabled (default), uses synchronous find_matches() within MATCH task.
    void set_task_based_matching(bool enable) { task_based_matching_ = enable; }
    bool task_based_matching() const { return task_based_matching_; }

    // Pruning options (v1 compatibility)
    void set_exploration_probability(double p) {
        exploration_probability_ = std::clamp(p, 0.0, 1.0);
    }
    void set_max_successor_states_per_parent(size_t max) {
        max_successor_states_per_parent_ = max;
    }
    void set_max_states_per_step(size_t max) {
        max_states_per_step_ = max;
    }

    double exploration_probability() const { return exploration_probability_; }
    size_t max_successor_states_per_parent() const { return max_successor_states_per_parent_; }
    size_t max_states_per_step() const { return max_states_per_step_; }

    size_t validation_mismatches() const { return validation_mismatches_.load(); }
    size_t late_arrivals() const { return late_arrivals_.load(); }
    size_t still_missing() const {
        // Count how many "missing" matches never arrived
        size_t count = 0;
        missing_match_hashes_.for_each([&](uint64_t h, bool) {
            if (!seen_match_hashes_.contains(h)) {
                ++count;
            }
        });
        return count;
    }

    void dump_still_missing() const {
        DEBUG_LOG("STILL MISSING HASHES:");
        missing_match_hashes_.for_each([&](uint64_t h, uint64_t debug_info) {
            if (!seen_match_hashes_.contains(h)) {
                uint32_t state_id = debug_info >> 16;
                uint16_t rule_index = debug_info & 0xFFFF;
                DEBUG_LOG("  hash=%lu state=%u rule=%u", h, state_id, rule_index);
            }
        });
    }

    size_t num_threads() const { return num_threads_; }
    size_t num_states() const { return hg_ ? hg_->num_states() : 0; }
    size_t num_canonical_states() const { return hg_ ? hg_->num_canonical_states() : 0; }
    size_t num_events() const { return hg_ ? hg_->num_events() : 0; }
    size_t num_causal_edges() const { return hg_ ? hg_->causal_graph().num_causal_event_pairs() : 0; }
    size_t num_branchial_edges() const { return hg_ ? hg_->causal_graph().num_branchial_edges() : 0; }
    size_t num_redundant_edges_skipped() const {
        return hg_ ? hg_->causal_graph().num_redundant_edges_skipped() : 0;
    }

    const EvolutionStats& stats() const { return stats_; }

    // Request early termination of evolution
    // This is non-blocking; evolution will stop as soon as currently queued jobs check the flag.
    // Call wait_for_idle() after request_stop() to ensure all jobs have completed.
    void request_stop() {
        should_stop_.store(true, std::memory_order_release);
    }

    // Check if stop has been requested
    bool stop_requested() const {
        return should_stop_.load(std::memory_order_acquire);
    }

    // =========================================================================
    // Main Evolution Loop - Dataflow Driven
    // =========================================================================
    //
    // MAXIMAL PARALLELISM: No intermediate synchronization barriers.
    //
    // Flow:
    // 1. Submit MATCH task for initial state
    // 2. MATCH tasks find matches → spawn REWRITE tasks
    // 3. REWRITE tasks apply matches → spawn MATCH tasks for new states
    // 4. Single wait_for_completion() at the end
    //
    // Work proceeds continuously as dependencies are satisfied.
    // The job system work-steals to keep all CPUs busy.

    void evolve(const std::vector<std::vector<VertexId>>& initial_edges, size_t steps) {
        if (!hg_ || rules_.empty()) return;

        max_steps_ = steps;
        should_stop_.store(false, std::memory_order_relaxed);

        // Create initial state
        std::vector<EdgeId> edge_ids;
        for (const auto& edge : initial_edges) {
            EdgeId eid = hg_->create_edge(edge.data(), static_cast<uint8_t>(edge.size()));
            edge_ids.push_back(eid);

            // Track max vertex ID to ensure fresh vertices don't collide
            for (VertexId v : edge) {
                hg_->reserve_vertices(v);
            }
        }

        SparseBitset initial_edge_set;
        for (EdgeId eid : edge_ids) {
            initial_edge_set.set(eid, hg_->arena());
        }

        // For initial state, use incremental path with no parent (will compute from scratch)
        // This also returns the cache to store for future incremental computation
        auto [canonical_hash, vertex_cache] = hg_->compute_canonical_hash_incremental(
            initial_edge_set,
            INVALID_ID,  // No parent state
            nullptr, 0,  // No consumed edges
            edge_ids.data(), static_cast<uint8_t>(edge_ids.size())  // All edges are "produced"
        );
        auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
            std::move(initial_edge_set), canonical_hash, 0, INVALID_ID);

        // Store cache for the initial state
        hg_->store_state_cache(raw_state, vertex_cache);

        // Emit visualization event for initial state
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        {
            const auto& state_data = hg_->get_state(raw_state);
            VIZ_EMIT_STATE_CREATED(
                raw_state,                // state id
                0,                        // parent state id (0 = none)
                0,                        // generation (initial state is gen 0)
                state_data.edges.count(), // edge count
                0                         // vertex count (not tracked per-state)
            );
            // Emit hyperedge data for each edge in the initial state
            uint32_t edge_idx = 0;
            state_data.edges.for_each([&](EdgeId eid) {
                const Edge& edge = hg_->get_edge(eid);
                VIZ_EMIT_HYPEREDGE(raw_state, edge_idx++, edge.vertices, edge.arity);
            });
        }
#endif

        // Create genesis event if enabled
        // This allows causal edges from initial state edges to be tracked
        if (enable_genesis_events_) {
            EventId genesis_event = hg_->create_genesis_event(
                raw_state,
                edge_ids.data(),
                static_cast<uint8_t>(edge_ids.size())
            );

            // Emit visualization event for the genesis event
            // Genesis events are always canonical (unique by definition)
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
            VIZ_EMIT_REWRITE_APPLIED(
                viz::VIZ_NO_SOURCE_STATE,  // source_state (none - genesis)
                raw_state,      // target_state (initial state)
                static_cast<RuleIndex>(-1),  // rule_index (none)
                genesis_event,  // event_id (raw)
                genesis_event,  // canonical_event_id (same as raw for genesis)
                0,              // destroyed edges (none)
                static_cast<uint8_t>(edge_ids.size())  // created edges
            );
#endif
        }

        // Mark initial state as matched (waiting version for correctness)
        matched_raw_states_.insert_if_absent_waiting(raw_state, true);

        // Submit MATCH task for initial state - this kicks off the dataflow
        submit_match_task(raw_state, 1);

        // Single synchronization point at the end
        job_system_->wait_for_completion();

        // CRITICAL: Acquire fence to ensure all writes from worker threads are visible
        // This pairs with release semantics of atomic operations in worker threads
        std::atomic_thread_fence(std::memory_order_acquire);

        // Emit visualization event for evolution completion
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        VIZ_EMIT_EVOLUTION_COMPLETE(
            hg_->num_states(),      // total states
            hg_->num_events(),      // total events
            max_steps_,             // max generation
            hg_->num_states()       // final state count (approximation)
        );
#endif
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    size_t total_matches() const { return total_matches_found_.load(std::memory_order_relaxed); }
    size_t total_rewrites() const { return total_rewrites_.load(std::memory_order_relaxed); }

private:
    // =========================================================================
    // Helper: Get or create the match list for a state (thread-safe)
    // =========================================================================
    LockFreeList<MatchRecord>* get_or_create_state_matches(StateId state) {
        uint64_t key = state;

        // First, try to look up existing list
        auto result = state_matches_.lookup(key);
        if (result.has_value()) {
            return *result;
        }

        // Need to create - allocate new list from arena
        auto* new_list = hg_->arena().template create<LockFreeList<MatchRecord>>();

        // Try to insert - if another thread beat us, use theirs
        auto [existing, inserted] = state_matches_.insert_if_absent(key, new_list);

        // Return whichever list is now in the map
        return inserted ? new_list : existing;
    }

    // =========================================================================
    // Helper: Store a match for a state (for later forwarding)
    // Sets the storage_epoch on the match and returns it.
    // =========================================================================
    uint64_t store_match_for_state(StateId state, MatchRecord& match, bool with_fence = false) {
        // Get epoch BEFORE storing (for ordering)
        uint64_t epoch = global_epoch_.fetch_add(1, std::memory_order_acq_rel);
        match.storage_epoch = epoch;

        LockFreeList<MatchRecord>* list = get_or_create_state_matches(state);
        list->push(match, hg_->arena());

        // In non-batched (eager) mode, we need a fence after each store to ensure
        // visibility before push_match_to_children runs. In batched mode, we use
        // a single fence after all stores.
        if (with_fence) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        return epoch;
    }

    // =========================================================================
    // Helper: Get or create the children list for a state (thread-safe)
    // =========================================================================
    LockFreeList<ChildInfo>* get_or_create_state_children(StateId state) {
        uint64_t key = state;

        // First, try to look up existing list
        auto result = state_children_.lookup(key);
        if (result.has_value()) {
            return *result;
        }

        // Need to create - allocate new list from arena
        auto* new_list = hg_->arena().template create<LockFreeList<ChildInfo>>();

        // Try to insert - if another thread beat us, use theirs
        auto [existing, inserted] = state_children_.insert_if_absent(key, new_list);

        // Return whichever list is now in the map
        return inserted ? new_list : existing;
    }

    // =========================================================================
    // Helper: Register a child with its parent (for push-based forwarding)
    // Returns the epoch at which the child was registered.
    // =========================================================================
    //
    // CRITICAL: The order of operations is carefully designed for correctness:
    // 1. First, push child to parent's children list (make it visible)
    // 2. Then, get epoch (determines push vs pull responsibility)
    //
    // This ordering ensures:
    // - If a match is stored BEFORE our epoch (match.epoch < child.epoch):
    //   Pull is responsible, and the match is already visible in the list
    // - If a match is stored AFTER our epoch (match.epoch >= child.epoch):
    //   Push is responsible, and the child is already visible in the list
    //
    // With the opposite order (epoch first, then push), there's a race where
    // a match could be stored with epoch >= child.epoch (so push is responsible)
    // but the child isn't visible yet (so push misses it).
    uint64_t register_child_with_parent(StateId parent, StateId child,
                                     const EdgeId* consumed_edges, uint8_t num_consumed,
                                     uint32_t child_step = 0) {
        if (parent == INVALID_ID) return 0;

        // Build ChildInfo (epoch will be set after push)
        ChildInfo info;
        info.child_state = child;
        info.num_consumed = num_consumed;
        info.creation_step = child_step;  // Step at which child was created
        info.registration_epoch = 0;  // Temporary, will update after push
        for (uint8_t i = 0; i < num_consumed; ++i) {
            info.consumed_edges[i] = consumed_edges[i];
        }

        // Push child to list (for push_match_to_children if called recursively)
        LockFreeList<ChildInfo>* children = get_or_create_state_children(parent);
        children->push(info, hg_->arena());

        // Note: Fences removed - with batched matching:
        // - Parent stores ALL matches before spawning REWRITEs
        // - So no push_match_to_children race to worry about
        // - Epoch-based ordering was for push vs pull; now we use CRDT-style (both try)
        // Uncomment if epoch ordering is reintroduced:
        // std::atomic_thread_fence(std::memory_order_seq_cst);

        // Get epoch (for tracking, though less critical now with batching)
        uint64_t epoch = global_epoch_.fetch_add(1, std::memory_order_acq_rel);

        // Store registration epoch for this child
        state_registration_epoch_.insert_if_absent(child, epoch);

        // Track child's parent (for walking ancestor chain during forward)
        ParentInfo pi_init;
        pi_init.parent_state = parent;
        pi_init.num_consumed = num_consumed;
        for (uint8_t i = 0; i < num_consumed; ++i) {
            pi_init.consumed_edges[i] = consumed_edges[i];
        }
        ParentInfo* parent_info = hg_->arena().template create<ParentInfo>(pi_init);
        state_parent_.insert_if_absent(child, parent_info);

        return epoch;
    }

    // =========================================================================
    // Helper: Push a match to immediate children (single-level push)
    // =========================================================================
    // PUSH mechanism: when a match is discovered, push to immediate children.
    //
    // EPOCH-BASED ORDERING:
    // - Push handles: children with registration_epoch <= match.storage_epoch
    // - Pull handles: children with registration_epoch > match.storage_epoch
    // This ensures exactly one of push/pull handles each (match, child) pair.
    //
    // VISIBILITY HANDLING:
    // A child might have registration_epoch <= match.storage_epoch (so push is
    // responsible) but not yet be visible in the children list. We retry if
    // the global epoch changes during push, indicating potential new children.
    //
    // Note: We also push to grandchildren recursively. Each store gets a new
    // epoch, so the epoch comparison is correct at each level.
    void push_match_to_children(StateId parent, const MatchRecord& match, uint32_t step) {
        if (batched_matching_) {
            // With batched matching, no retry loop needed:
            // - Parent's MATCH task stores ALL matches before spawning any REWRITEs
            // - No children can be registered until REWRITEs run
            // - So children list is stable during this call
            push_match_to_children_impl(parent, match, step);
        } else {
            // EAGER MODE: Retry loop to close race window.
            // Race: child registers AFTER we read children list but BEFORE epoch increments.
            // Solution: capture epoch before, push, check if epoch changed, retry if so.
            //
            // The retry ensures we see any children that registered during our push.
            // At most a few retries since children only register once.
            uint64_t epoch_before = global_epoch_.load(std::memory_order_acquire);
            push_match_to_children_impl(parent, match, step);

            // Retry if new children may have registered during push
            uint64_t epoch_after = global_epoch_.load(std::memory_order_acquire);
            while (epoch_after != epoch_before) {
                epoch_before = epoch_after;
                push_match_to_children_impl(parent, match, step);
                epoch_after = global_epoch_.load(std::memory_order_acquire);
            }
        }
    }

    void push_match_to_children_impl(StateId parent, const MatchRecord& match, uint32_t step) {
        // Use lookup_waiting to handle concurrent inserts (LOCKED slots).
        // Even with batching, grandchildren may race with sibling match stores.
        auto result = state_children_.lookup_waiting(parent);
        if (!result.has_value()) return;  // No children registered

        LockFreeList<ChildInfo>* children = *result;
        children->for_each([&](const ChildInfo& child_info) {
            // CRDT-STYLE: No epoch filtering - push to ALL children.
            // Both push and pull may try to deliver the same match, but
            // seen_match_hashes_.insert_if_absent provides deduplication.
            // This avoids the race where both push and pull miss due to
            // visibility windows in epoch checks.

            // Skip if match overlaps with consumed edges (match was used to create this child)
            if (child_info.match_overlaps_consumed(match.matched_edges, match.num_edges)) {
                stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            // Create forwarded match with child as source
            MatchRecord forwarded = match;
            forwarded.source_state = child_info.child_state;
            forwarded.canonical_source = hg_->get_canonical_state(child_info.child_state);
            forwarded.source_canonical_hash = hg_->get_state(child_info.child_state).canonical_hash;

            // Deduplicate - check if this exact (rule, state, binding) was already processed
            // Use waiting version to avoid race during resize
            uint64_t h = forwarded.hash();
            auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
            if (!inserted) {
                return;  // Already processed
            }

            // Check if this was a "missing" match that arrived late via push
            if (validate_match_forwarding_) {
                auto missing = missing_match_hashes_.lookup(h);
                if (missing.has_value()) {
                    late_arrivals_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            total_matches_found_.fetch_add(1, std::memory_order_relaxed);
            stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

            DEBUG_LOG("PUSH parent=%u -> child=%u rule=%u hash=%lu step=%u epoch=%lu",
                      parent, child_info.child_state, match.rule_index, h, step, match.storage_epoch);

            // Store match in child (so grandchildren can find it via forward_existing)
            store_match_for_state(child_info.child_state, forwarded);

            // Note: With batched matching, this push mechanism is not used in the main flow.
            // Pull from ancestors handles everything. Push remains for non-batched paths.
            // Fences removed - if push is needed, the caller should provide fencing.
            // Uncomment if non-batched paths are reintroduced:
            // std::atomic_thread_fence(std::memory_order_seq_cst);

            // CRITICAL FIX: Use child's MATCH step, not parent's step!
            // The child was created at child_info.creation_step, so its MATCH
            // runs at creation_step + 1. Matches forwarded to child should spawn
            // REWRITEs at the same step as the child's MATCH (creation_step + 1),
            // which create grandchildren whose MATCH runs at creation_step + 2.
            uint32_t child_step = child_info.creation_step + 1;

            // RECURSIVE: Push to child's existing children (grandchildren)
            push_match_to_children(child_info.child_state, forwarded, child_step);

            // Spawn REWRITE task for this forwarded match
            submit_rewrite_task(forwarded, child_step);
        });
    }

    // =========================================================================
    // Helper: Forward existing parent matches to a newly created child
    // =========================================================================
    // Called when a child is created, to catch up on matches found in the entire
    // ancestor chain (parent, grandparent, etc.). This ensures that matches found
    // in ancestors AFTER intermediate states were created still reach descendants.
    // BATCHED VERSION: Forward existing parent matches, adding to batch
    void forward_existing_parent_matches(
        StateId parent, StateId child,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        uint32_t step,
        std::vector<MatchRecord>& batch
    ) {
        // Accumulate all consumed edges along the path from child to ancestors
        // Start with the edges consumed to create this child
        EdgeId accumulated_consumed[MAX_PATTERN_EDGES * 8];  // Allow for deep trees
        uint8_t total_consumed = 0;
        for (uint8_t i = 0; i < num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
            accumulated_consumed[total_consumed++] = consumed_edges[i];
        }

        // Walk up the ancestor chain, forwarding matches from each ancestor
        StateId current_ancestor = parent;
        while (current_ancestor != INVALID_ID) {
            forward_matches_from_single_ancestor(current_ancestor, child,
                                                  accumulated_consumed, total_consumed, step, batch);

            // Move to the next ancestor and accumulate its consumed edges
            // Use waiting lookup to handle concurrent registration
            auto parent_result = state_parent_.lookup_waiting(current_ancestor);
            if (!parent_result.has_value()) break;

            ParentInfo* pi = *parent_result;
            if (!pi || !pi->has_parent()) break;

            // Add this ancestor's consumed edges to the accumulated set
            for (uint8_t i = 0; i < pi->num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
                accumulated_consumed[total_consumed++] = pi->consumed_edges[i];
            }
            current_ancestor = pi->parent_state;
        }
    }

    // BATCHED VERSION: Forward matches from single ancestor, adding to batch
    // With batching, parent's matches are guaranteed visible before child is created,
    // so no retry loop is needed.
    void forward_matches_from_single_ancestor(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t step,
        std::vector<MatchRecord>& batch
    ) {
        forward_matches_from_single_ancestor_impl(
            ancestor, child, accumulated_consumed, total_consumed, step,
            0,  // child_registration_epoch unused
            batch);
    }

    // BATCHED VERSION: Adds forwarded matches to batch instead of spawning immediately
    void forward_matches_from_single_ancestor_impl(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t /* step */, uint64_t /* child_registration_epoch - unused */,
        std::vector<MatchRecord>& batch  // Output: matches to add
    ) {
        // Use lookup_waiting to handle concurrent inserts (LOCKED slots).
        // Even with batching, grandchildren may race with sibling match stores.
        auto result = state_matches_.lookup_waiting(ancestor);
        if (!result.has_value()) return;  // Ancestor has no matches yet

        LockFreeList<MatchRecord>* ancestor_matches = *result;
        ancestor_matches->for_each([&](const MatchRecord& ancestor_match) {
            // Skip if match overlaps with ANY consumed edge along the path
            bool overlaps = false;
            for (uint8_t i = 0; i < ancestor_match.num_edges && !overlaps; ++i) {
                for (uint8_t j = 0; j < total_consumed; ++j) {
                    if (ancestor_match.matched_edges[i] == accumulated_consumed[j]) {
                        overlaps = true;
                        break;
                    }
                }
            }

            if (overlaps) {
                stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            // Create forwarded match for child state
            MatchRecord forwarded = ancestor_match;
            forwarded.source_state = child;
            forwarded.canonical_source = hg_->get_canonical_state(child);
            forwarded.source_canonical_hash = hg_->get_state(child).canonical_hash;

            // Deduplicate (use waiting version to avoid race during resize)
            uint64_t h = forwarded.hash();
            auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
            if (!inserted) {
                DEBUG_LOG("FWD_DUP ancestor=%u -> child=%u rule=%u hash=%lu",
                          ancestor, child, ancestor_match.rule_index, h);
                return;  // Already seen
            }

            // Check if this was a "missing" match that arrived late via forward_existing
            if (validate_match_forwarding_) {
                auto missing = missing_match_hashes_.lookup(h);
                if (missing.has_value()) {
                    late_arrivals_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            total_matches_found_.fetch_add(1, std::memory_order_relaxed);
            stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

            DEBUG_LOG("FWD ancestor=%u -> child=%u rule=%u hash=%lu epoch=%lu",
                      ancestor, child, ancestor_match.rule_index, h, ancestor_match.storage_epoch);

            // Add to batch - will be stored and REWRITE spawned in Phase 2
            batch.push_back(forwarded);
        });
    }

    // =========================================================================
    // EAGER MODE: Forward existing parent matches with retry loop
    // =========================================================================
    // Called when a child is created in non-batched mode. Pulls from ancestors
    // with retry to close the race window where a parent stores a match after
    // we read the ancestor's match list.

    void forward_existing_parent_matches_eager(
        StateId parent, StateId child,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        uint32_t step
    ) {
        // Accumulate all consumed edges along the path from child to ancestors
        EdgeId accumulated_consumed[MAX_PATTERN_EDGES * 8];
        uint8_t total_consumed = 0;
        for (uint8_t i = 0; i < num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
            accumulated_consumed[total_consumed++] = consumed_edges[i];
        }

        // RETRY LOOP: Pull with epoch tracking to close race window.
        // We retry the entire ancestor chain walk if global_epoch changes,
        // indicating new matches may have been stored.
        uint64_t epoch_before = global_epoch_.load(std::memory_order_acquire);

        // Walk up the ancestor chain, forwarding matches from each ancestor
        StateId current_ancestor = parent;
        while (current_ancestor != INVALID_ID) {
            forward_matches_from_single_ancestor_eager(
                current_ancestor, child,
                accumulated_consumed, total_consumed, step);

            // Move to next ancestor and accumulate consumed edges
            auto parent_result = state_parent_.lookup_waiting(current_ancestor);
            if (!parent_result.has_value()) break;

            ParentInfo* pi = *parent_result;
            if (!pi || !pi->has_parent()) break;

            for (uint8_t i = 0; i < pi->num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
                accumulated_consumed[total_consumed++] = pi->consumed_edges[i];
            }
            current_ancestor = pi->parent_state;
        }

        // Check if epoch changed - if so, retry to catch any new matches
        uint64_t epoch_after = global_epoch_.load(std::memory_order_acquire);
        while (epoch_after != epoch_before) {
            epoch_before = epoch_after;

            // Reset consumed edges and re-walk
            total_consumed = 0;
            for (uint8_t i = 0; i < num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
                accumulated_consumed[total_consumed++] = consumed_edges[i];
            }

            current_ancestor = parent;
            while (current_ancestor != INVALID_ID) {
                forward_matches_from_single_ancestor_eager(
                    current_ancestor, child,
                    accumulated_consumed, total_consumed, step);

                auto parent_result = state_parent_.lookup_waiting(current_ancestor);
                if (!parent_result.has_value()) break;

                ParentInfo* pi = *parent_result;
                if (!pi || !pi->has_parent()) break;

                for (uint8_t i = 0; i < pi->num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
                    accumulated_consumed[total_consumed++] = pi->consumed_edges[i];
                }
                current_ancestor = pi->parent_state;
            }

            epoch_after = global_epoch_.load(std::memory_order_acquire);
        }
    }

    // EAGER VERSION: Forward matches from single ancestor, spawning immediately
    void forward_matches_from_single_ancestor_eager(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t step
    ) {
        auto result = state_matches_.lookup_waiting(ancestor);
        if (!result.has_value()) return;

        // Build consumed set once for O(1) amortized overlap checks
        std::unordered_set<EdgeId> consumed_set(accumulated_consumed, accumulated_consumed + total_consumed);

        LockFreeList<MatchRecord>* ancestor_matches = *result;
        ancestor_matches->for_each([&](const MatchRecord& ancestor_match) {
            // Skip if match overlaps with ANY consumed edge - O(n) vs O(n*m)
            bool overlaps = false;
            for (uint8_t i = 0; i < ancestor_match.num_edges && !overlaps; ++i) {
                if (consumed_set.count(ancestor_match.matched_edges[i])) {
                    overlaps = true;
                }
            }

            if (overlaps) {
                stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            // Create forwarded match for child state
            MatchRecord forwarded = ancestor_match;
            forwarded.source_state = child;
            forwarded.canonical_source = hg_->get_canonical_state(child);
            forwarded.source_canonical_hash = hg_->get_state(child).canonical_hash;

            // Deduplicate - seen_match_hashes_ protects against both push and pull duplicates
            uint64_t h = forwarded.hash();
            auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
            if (!inserted) {
                DEBUG_LOG("FWD_EAGER_DUP ancestor=%u -> child=%u rule=%u hash=%lu",
                          ancestor, child, ancestor_match.rule_index, h);
                return;  // Already seen (possibly via push)
            }

            // Check if this was a "missing" match that arrived late
            if (validate_match_forwarding_) {
                auto missing = missing_match_hashes_.lookup(h);
                if (missing.has_value()) {
                    late_arrivals_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            total_matches_found_.fetch_add(1, std::memory_order_relaxed);
            stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

            DEBUG_LOG("FWD_EAGER ancestor=%u -> child=%u rule=%u hash=%lu step=%u",
                      ancestor, child, ancestor_match.rule_index, h, step);

            // EAGER: Immediately spawn REWRITE task
            // No need to store - match is already stored in ancestor, descendants can pull from there
            submit_rewrite_task(forwarded, step);
        });
    }

    // =========================================================================
    // Task Submission (Dataflow)
    // =========================================================================

    // Submit a MATCH task for a state (full matching for initial state)
    void submit_match_task(StateId state, uint32_t step) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("SUBMIT_MATCH state=%u step=%u (full)", state, step);

        auto job = job_system::make_job<EvolutionJobType>(
            [this, state, step]() {
                execute_match_task(state, step, MatchContext{});
            },
            EvolutionJobType::MATCH
        );
        job_system_->submit(std::move(job));
    }

    // Submit a MATCH task with context (for match forwarding)
    void submit_match_task_with_context(StateId state, uint32_t step, const MatchContext& ctx) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("SUBMIT_MATCH state=%u step=%u parent=%u produced=%u consumed=%u (delta)",
                  state, step, ctx.parent_state, ctx.num_produced, ctx.num_consumed);

        auto job = job_system::make_job<EvolutionJobType>(
            [this, state, step, ctx]() {
                execute_match_task(state, step, ctx);
            },
            EvolutionJobType::MATCH
        );
        job_system_->submit(std::move(job));
    }

    // Submit a REWRITE task for a match
    void submit_rewrite_task(const MatchRecord& match, uint32_t step) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("SUBMIT_REWRITE state=%u rule=%u step=%u", match.source_state, match.rule_index, step);

        auto job = job_system::make_job<EvolutionJobType>(
            [this, match, step]() {
                execute_rewrite_task(match, step);
            },
            EvolutionJobType::REWRITE
        );
        job_system_->submit(std::move(job));
    }

    // Submit a SCAN task for initial candidate generation
    void submit_scan_task(const ScanTaskData& data) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("SUBMIT_SCAN state=%u rule=%u step=%u delta=%d",
                  data.state, data.rule_index, data.step, data.is_delta);

        auto job = job_system::make_job<EvolutionJobType>(
            [this, data]() {
                execute_scan_task(data);
            },
            EvolutionJobType::SCAN
        );
        // SCAN tasks use FIFO - start broadly, then depth-first via EXPAND
        job_system_->submit(std::move(job));
    }

    // Submit an EXPAND task for partial match extension
    // Uses LIFO scheduling for depth-first search (bounded memory, cache-friendly)
    void submit_expand_task(const ExpandTaskData& data) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("SUBMIT_EXPAND state=%u rule=%u matched=%u/%u step=%u",
                  data.state, data.rule_index, data.num_matched, data.num_pattern_edges, data.step);

        auto job = job_system::make_job<EvolutionJobType>(
            [this, data]() {
                execute_expand_task(data);
            },
            EvolutionJobType::EXPAND
        );
        // LIFO scheduling: depth-first traversal, bounded memory O(|E(q)|² × |E(H)|)
        job_system_->submit(std::move(job), job_system::ScheduleMode::LIFO);
    }

    // Submit a SINK task for complete match processing
    void submit_sink_task(const ExpandTaskData& data) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("SUBMIT_SINK state=%u rule=%u matched=%u step=%u",
                  data.state, data.rule_index, data.num_matched, data.step);

        auto job = job_system::make_job<EvolutionJobType>(
            [this, data]() {
                execute_sink_task(data);
            },
            EvolutionJobType::SINK
        );
        job_system_->submit(std::move(job));
    }

    // =========================================================================
    // MATCH Task Execution
    // =========================================================================
    // Finds matches in a state and spawns REWRITE tasks for each.
    //
    // With match forwarding:
    // - If no parent (initial state): full pattern matching
    // - If has parent: forward valid parent matches + find NEW matches

    void execute_match_task(StateId state, uint32_t step, const MatchContext& ctx) {
        if (should_stop_.load(std::memory_order_relaxed)) return;
        if (max_steps_ > 0 && step > max_steps_) return;

        const State& s = hg_->get_state(state);

        // Get canonical state for deterministic deduplication
        StateId canonical_state = hg_->get_canonical_state(state);

        // Edge accessor
        auto get_edge = [this](EdgeId eid) -> const Edge& {
            return hg_->get_edge(eid);
        };

        // Signature accessor (cached signatures for O(1) lookup)
        auto get_signature = [this](EdgeId eid) -> const EdgeSignature& {
            return hg_->edge_signature(eid);
        };

        // =======================================================================
        // BATCHED MATCHING: Collect all matches first, then spawn REWRITEs
        // =======================================================================
        // This eliminates the race condition where children are created before
        // all parent matches are stored. With batching:
        // 1. Find ALL matches for this state
        // 2. Store ALL matches
        // 3. THEN spawn REWRITEs (which create children)
        // Children can then reliably pull ALL parent matches.

        std::vector<MatchRecord> batch;
        batch.reserve(32);  // Pre-allocate for typical cases
        size_t delta_start = 0;  // Index where delta (discovered) matches start

        // Collector callback - in batched mode, adds to batch for later processing.
        // In eager mode, stores immediately and spawns REWRITE, with push to children.
        auto collect_match = [&, state, canonical_state](
            uint16_t rule_index,
            const EdgeId* edges,
            uint8_t num_edges,
            const VariableBinding& binding
        ) {
            if (should_stop_.load(std::memory_order_relaxed)) return;

            MatchRecord match;
            match.rule_index = rule_index;
            match.num_edges = num_edges;
            match.binding = binding;
            match.source_state = state;
            match.canonical_source = canonical_state;
            match.source_canonical_hash = s.canonical_hash;
            for (uint8_t i = 0; i < num_edges; ++i) {
                match.matched_edges[i] = edges[i];
            }

            // Deduplicate using lock-free ConcurrentMap (waiting version to avoid race during resize)
            uint64_t h = match.hash();
            auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
            if (!inserted) {
                rejected_duplicates_.fetch_add(1, std::memory_order_relaxed);
                return;  // Already seen
            }

            total_matches_found_.fetch_add(1, std::memory_order_relaxed);
            stats_.new_matches_discovered.fetch_add(1, std::memory_order_relaxed);

            DEBUG_LOG("NEW state=%u rule=%u hash=%lu step=%u", state, rule_index, h, step);

            if (batched_matching_) {
                // BATCHED MODE: Collect for later processing
                batch.push_back(match);
            } else {
                // EAGER MODE: Store immediately with fence, push to children, spawn REWRITE
                if (enable_match_forwarding_) {
                    store_match_for_state(state, match, true);  // with fence
                    push_match_to_children(state, match, step);
                }
                submit_rewrite_task(match, step);
            }
        };

        // Match callback for pattern matching
        auto on_match = [&](
            uint16_t rule_index,
            const EdgeId* edges,
            uint8_t num_edges,
            const VariableBinding& binding,
            StateId /*source_state*/
        ) {
            collect_match(rule_index, edges, num_edges, binding);
        };

        if (enable_match_forwarding_ && ctx.has_parent()) {
            // === DELTA MATCHING MODE (child state) ===
            // With batching, parent's MATCH task completed before this child was created.
            // So all parent matches are already stored. We just pull them.
            //
            // Invariant: Forwarded (pull) + Delta = Full matches
            stats_.delta_pattern_matches.fetch_add(1, std::memory_order_relaxed);

            if (batched_matching_) {
                // BATCHED MODE: Pull all ancestor matches into batch
                // They're guaranteed visible because parent stored all before spawning REWRITE
                forward_existing_parent_matches(
                    ctx.parent_state, state,
                    ctx.consumed_edges, ctx.num_consumed, step, batch);
            } else {
                // EAGER MODE: Pull from ancestors with retry loop.
                // Race: parent may store a match AFTER we read ancestor's match list.
                // Solution: capture epoch before pull, pull, check if epoch changed, retry if so.
                //
                // We directly spawn REWRITEs for each pulled match (no batching).
                forward_existing_parent_matches_eager(
                    ctx.parent_state, state,
                    ctx.consumed_edges, ctx.num_consumed, step);
            }

            // Track where forwarded matches end - we only need to STORE delta matches
            // (forwarded matches are already stored in ancestors and can be pulled by descendants)
            delta_start = batch.size();

            if (task_based_matching_) {
                // Task-based delta matching: spawn SCAN tasks for each rule with is_delta=true
                // SCAN→EXPAND→SINK handles match discovery and REWRITEs
                for (uint16_t r = 0; r < rules_.size(); ++r) {
                    ScanTaskData scan_data;
                    scan_data.state = state;
                    scan_data.rule_index = r;
                    scan_data.step = step;
                    scan_data.canonical_state = canonical_state;
                    scan_data.source_canonical_hash = s.canonical_hash;
                    scan_data.is_delta = true;
                    scan_data.num_produced = ctx.num_produced;
                    for (uint8_t i = 0; i < ctx.num_produced; ++i) {
                        scan_data.produced_edges[i] = ctx.produced_edges[i];
                    }
                    submit_scan_task(scan_data);
                }
                // Spawn REWRITEs for forwarded matches before returning
                // (SCAN tasks only handle delta matches, not the already-forwarded ones)
                for (size_t i = 0; i < delta_start; ++i) {
                    submit_rewrite_task(batch[i], step);
                }
                return;  // SCAN tasks handle delta matches asynchronously
            }

            // Synchronous delta matching: find patterns involving the newly produced edges (SCAN→EXPAND→SINK fused)
            DEBUG_LOG("SYNC_DELTA_MATCH state=%u step=%u rules=%zu produced=%u (SCAN->EXPAND->SINK fused)",
                      state, step, rules_.size(), ctx.num_produced);
            for (uint16_t r = 0; r < rules_.size(); ++r) {
                find_delta_matches(
                    rules_[r], r, state, s.edges,
                    hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature, on_match,
                    ctx.produced_edges, ctx.num_produced
                );
            }

            // VALIDATION: Compare forwarded+delta vs full matching
            if (validate_match_forwarding_) {
                size_t missing = 0;
                auto count_missing = [&, state, canonical_state](
                    uint16_t rule_index,
                    const EdgeId* edges,
                    uint8_t num_edges,
                    const VariableBinding& binding,
                    StateId /*source_state*/
                ) {
                    MatchRecord match;
                    match.rule_index = rule_index;
                    match.num_edges = num_edges;
                    match.binding = binding;
                    match.source_state = state;
                    match.canonical_source = canonical_state;
                    match.source_canonical_hash = s.canonical_hash;
                    for (uint8_t i = 0; i < num_edges; ++i) {
                        match.matched_edges[i] = edges[i];
                    }
                    uint64_t h = match.hash();
                    // Check if this match was found via forwarding+delta
                    if (!seen_match_hashes_.contains(h)) {
                        ++missing;
                        // Store in missing_match_hashes_ with state/rule info for debugging
                        uint64_t debug_info = (static_cast<uint64_t>(state) << 16) | rule_index;
                        missing_match_hashes_.insert_if_absent(h, debug_info);

                        // Debug: check if this match involves produced edges
                        uint8_t produced_count = 0;
                        for (uint8_t i = 0; i < num_edges; ++i) {
                            for (uint8_t j = 0; j < ctx.num_produced; ++j) {
                                if (edges[i] == ctx.produced_edges[j]) {
                                    ++produced_count;
                                    break;
                                }
                            }
                        }
                        (void)produced_count;  // Suppress unused variable warning
                    }
                };
                for (uint16_t r = 0; r < rules_.size(); ++r) {
                    find_matches(
                        rules_[r], r, state, s.edges,
                        hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature, count_missing
                    );
                }
                if (missing > 0) {
                    validation_mismatches_.fetch_add(missing, std::memory_order_relaxed);
                }
            }
        } else {
            // === FULL MATCHING MODE (initial state or forwarding disabled) ===
            stats_.full_pattern_matches.fetch_add(1, std::memory_order_relaxed);

            if (task_based_matching_) {
                // Task-based matching: spawn SCAN tasks for each rule
                // SCAN→EXPAND→SINK handles match discovery and REWRITEs
                for (uint16_t r = 0; r < rules_.size(); ++r) {
                    ScanTaskData scan_data;
                    scan_data.state = state;
                    scan_data.rule_index = r;
                    scan_data.step = step;
                    scan_data.canonical_state = canonical_state;
                    scan_data.source_canonical_hash = s.canonical_hash;
                    scan_data.is_delta = false;
                    scan_data.num_produced = 0;
                    submit_scan_task(scan_data);
                }
                return;  // SCAN tasks handle everything asynchronously
            }

            // Synchronous matching: find all matches directly (SCAN→EXPAND→SINK fused)
            DEBUG_LOG("SYNC_MATCH state=%u step=%u rules=%zu (SCAN->EXPAND->SINK fused)",
                      state, step, rules_.size());
            for (uint16_t r = 0; r < rules_.size(); ++r) {
                find_matches(
                    rules_[r], r, state, s.edges,
                    hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature, on_match
                );
            }
        }

        // =======================================================================
        // PHASE 2: Store all matches, then spawn all REWRITEs (BATCHED MODE ONLY)
        // =======================================================================
        // In batched mode, this ordering guarantees that when a child is created
        // (by REWRITE), all parent matches are already stored and visible for pulling.
        // In eager mode, matches were already stored and REWRITEs spawned above.

        if (batched_matching_) {
            if (enable_match_forwarding_) {
                // Only store DELTA matches (discovered in this state).
                // Forwarded matches are already stored in ancestors - descendants can pull from there.
                // This dramatically reduces memory usage.
                for (size_t i = delta_start; i < batch.size(); ++i) {
                    store_match_for_state(state, batch[i]);
                }
                // Memory fence ensures all stores are visible before REWRITEs run
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }

            // Now spawn REWRITEs - children will see all stored matches
            for (const auto& match : batch) {
                submit_rewrite_task(match, step);
            }
        }
    }

    // =========================================================================
    // SCAN Task Execution (HGMatch Dataflow Model)
    // =========================================================================
    // Finds initial candidates for first pattern edge, spawns EXPAND tasks.
    // For delta matching, starts from produced edges instead.

    void execute_scan_task(const ScanTaskData& data) {
        if (should_stop_.load(std::memory_order_relaxed)) return;
        if (max_steps_ > 0 && data.step > max_steps_) return;

        DEBUG_LOG("EXEC_SCAN state=%u rule=%u step=%u delta=%d",
                  data.state, data.rule_index, data.step, data.is_delta);

        const State& s = hg_->get_state(data.state);
        const RewriteRule& rule = rules_[data.rule_index];

        if (rule.num_lhs_edges == 0) return;

        // Edge accessor
        auto get_edge = [this](EdgeId eid) -> const Edge& {
            return hg_->get_edge(eid);
        };

        // Signature accessor
        auto get_signature = [this](EdgeId eid) -> const EdgeSignature& {
            return hg_->edge_signature(eid);
        };

        // Pre-compute pattern signatures
        EdgeSignature pattern_sigs[MAX_PATTERN_EDGES];
        CompatibleSignatureCache sig_caches[MAX_PATTERN_EDGES];
        for (uint8_t i = 0; i < rule.num_lhs_edges; ++i) {
            pattern_sigs[i] = rule.lhs[i].signature();
            sig_caches[i] = CompatibleSignatureCache::from_pattern(pattern_sigs[i]);
        }

        if (data.is_delta) {
            // Delta matching: start from produced edges
            for (uint8_t p = 0; p < data.num_produced; ++p) {
                EdgeId produced = data.produced_edges[p];
                if (!s.edges.contains(produced)) continue;

                // Try this produced edge at each pattern position
                for (uint8_t pos = 0; pos < rule.num_lhs_edges; ++pos) {
                    if (should_stop_.load(std::memory_order_relaxed)) return;

                    const PatternEdge& pattern_edge = rule.lhs[pos];
                    const auto& edge = get_edge(produced);

                    // Check signature compatibility
                    const EdgeSignature& data_sig = get_signature(produced);
                    if (!signature_compatible(data_sig, pattern_sigs[pos])) continue;

                    // Validate candidate
                    VariableBinding binding;
                    if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, binding)) continue;

                    // Create EXPAND task data
                    ExpandTaskData expand_data;
                    expand_data.state = data.state;
                    expand_data.rule_index = data.rule_index;
                    expand_data.num_pattern_edges = rule.num_lhs_edges;
                    expand_data.next_pattern_idx = 0;  // Will be computed
                    expand_data.matched_edges[0] = produced;
                    expand_data.match_order[0] = pos;
                    expand_data.num_matched = 1;
                    expand_data.binding = binding;
                    expand_data.step = data.step;
                    expand_data.canonical_state = data.canonical_state;
                    expand_data.source_canonical_hash = data.source_canonical_hash;

                    if (rule.num_lhs_edges == 1) {
                        // Single-edge pattern: complete match
                        submit_sink_task(expand_data);
                    } else {
                        // Multi-edge: spawn EXPAND
                        submit_expand_task(expand_data);
                    }
                }
            }
        } else {
            // Full matching: start from first pattern edge
            const PatternEdge& first_edge = rule.lhs[0];
            const EdgeSignature& first_sig = pattern_sigs[0];
            const CompatibleSignatureCache& first_cache = sig_caches[0];

            // Generate candidates for first edge
            generate_candidates(
                first_edge, first_sig, first_cache,
                VariableBinding{}, s.edges,
                hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature,
                [&](EdgeId candidate) {
                    if (should_stop_.load(std::memory_order_relaxed)) return;

                    // Validate candidate
                    const auto& edge = get_edge(candidate);
                    VariableBinding binding;
                    if (!validate_candidate(edge.vertices, edge.arity, first_edge, binding)) return;

                    // Create EXPAND task data
                    ExpandTaskData expand_data;
                    expand_data.state = data.state;
                    expand_data.rule_index = data.rule_index;
                    expand_data.num_pattern_edges = rule.num_lhs_edges;
                    expand_data.next_pattern_idx = 1;
                    expand_data.matched_edges[0] = candidate;
                    expand_data.match_order[0] = 0;
                    expand_data.num_matched = 1;
                    expand_data.binding = binding;
                    expand_data.step = data.step;
                    expand_data.canonical_state = data.canonical_state;
                    expand_data.source_canonical_hash = data.source_canonical_hash;

                    if (rule.num_lhs_edges == 1) {
                        // Single-edge pattern: complete match
                        submit_sink_task(expand_data);
                    } else {
                        // Multi-edge: spawn EXPAND
                        submit_expand_task(expand_data);
                    }
                }
            );
        }
    }

    // =========================================================================
    // EXPAND Task Execution (HGMatch Dataflow Model)
    // =========================================================================
    // Extends partial match by one edge, spawns more EXPAND or SINK tasks.

    void execute_expand_task(const ExpandTaskData& data) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("EXEC_EXPAND state=%u rule=%u matched=%u/%u step=%u",
                  data.state, data.rule_index, data.num_matched, data.num_pattern_edges, data.step);

        // Check if complete (shouldn't happen, but safety check)
        if (data.is_complete()) {
            submit_sink_task(data);
            return;
        }

        const State& s = hg_->get_state(data.state);
        const RewriteRule& rule = rules_[data.rule_index];

        // Edge accessor
        auto get_edge = [this](EdgeId eid) -> const Edge& {
            return hg_->get_edge(eid);
        };

        // Signature accessor
        auto get_signature = [this](EdgeId eid) -> const EdgeSignature& {
            return hg_->edge_signature(eid);
        };

        // Get next pattern edge to match
        uint8_t pattern_idx = data.get_next_pattern_idx();
        if (pattern_idx >= rule.num_lhs_edges) return;

        const PatternEdge& pattern_edge = rule.lhs[pattern_idx];
        EdgeSignature pattern_sig = pattern_edge.signature();
        CompatibleSignatureCache sig_cache = CompatibleSignatureCache::from_pattern(pattern_sig);

        // Generate candidates
        generate_candidates(
            pattern_edge, pattern_sig, sig_cache,
            data.binding, s.edges,
            hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature,
            [&](EdgeId candidate) {
                if (should_stop_.load(std::memory_order_relaxed)) return;

                // Skip if already matched
                if (data.contains_edge(candidate)) return;

                // Validate candidate
                const auto& edge = get_edge(candidate);
                VariableBinding extended = data.binding;
                if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, extended)) return;

                // Create new EXPAND task with extended match
                ExpandTaskData new_data = data;
                new_data.matched_edges[new_data.num_matched] = candidate;
                new_data.match_order[new_data.num_matched] = pattern_idx;
                new_data.num_matched++;
                new_data.binding = extended;

                if (new_data.is_complete()) {
                    // Complete match: spawn SINK
                    submit_sink_task(new_data);
                } else {
                    // Not complete: spawn another EXPAND
                    submit_expand_task(new_data);
                }
            }
        );
    }

    // =========================================================================
    // SINK Task Execution (HGMatch Dataflow Model)
    // =========================================================================
    // Processes complete match: deduplicate, store for forwarding, spawn REWRITE.

    void execute_sink_task(const ExpandTaskData& data) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        DEBUG_LOG("EXEC_SINK state=%u rule=%u matched=%u step=%u",
                  data.state, data.rule_index, data.num_matched, data.step);

        // Convert matched edges to pattern order
        EdgeId edges_in_order[MAX_PATTERN_EDGES];
        data.to_pattern_order(edges_in_order);

        // Build MatchRecord
        MatchRecord match;
        match.rule_index = data.rule_index;
        match.num_edges = data.num_pattern_edges;
        match.binding = data.binding;
        match.source_state = data.state;
        match.canonical_source = data.canonical_state;
        match.source_canonical_hash = data.source_canonical_hash;
        for (uint8_t i = 0; i < data.num_pattern_edges; ++i) {
            match.matched_edges[i] = edges_in_order[i];
        }

        // Deduplicate using lock-free ConcurrentMap
        uint64_t h = match.hash();
        auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
        if (!inserted) {
            rejected_duplicates_.fetch_add(1, std::memory_order_relaxed);
            return;  // Already seen
        }

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.new_matches_discovered.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("SINK state=%u rule=%u hash=%lu step=%u", data.state, data.rule_index, h, data.step);

        // Store and push match for forwarding (mirroring synchronous eager path)
        // Push closes the race window: children created before store still get notified
        if (enable_match_forwarding_) {
            store_match_for_state(data.state, match, true);  // with fence
            push_match_to_children(data.state, match, data.step);
        }

        // Spawn REWRITE task
        submit_rewrite_task(match, data.step);
    }

    // =========================================================================
    // Pruning Helpers (v1 compatibility)
    // =========================================================================

    // Try to reserve a successor slot for the parent state.
    // Returns true if allowed to create another child, false if limit reached.
    bool try_reserve_successor_slot(StateId parent_state) {
        if (max_successor_states_per_parent_ == 0) return true;  // Unlimited

        // Get or create atomic counter for this parent
        uint64_t key = parent_state;
        auto result = parent_successor_count_.lookup(key);
        std::atomic<size_t>* counter = nullptr;

        if (result.has_value()) {
            counter = *result;
        } else {
            // Allocate new counter from arena
            counter = hg_->arena().template create<std::atomic<size_t>>(0);
            auto [existing, inserted] = parent_successor_count_.insert_if_absent(key, counter);
            if (!inserted) {
                counter = existing;  // Another thread beat us
            }
        }

        // Try to increment, fail if at limit
        size_t old_val = counter->fetch_add(1, std::memory_order_relaxed);
        if (old_val >= max_successor_states_per_parent_) {
            counter->fetch_sub(1, std::memory_order_relaxed);  // Rollback
            return false;
        }
        return true;
    }

    // Try to reserve a state slot for the given step/generation.
    // Returns true if allowed to create another state at this step, false if limit reached.
    bool try_reserve_step_slot(uint32_t step) {
        if (max_states_per_step_ == 0) return true;  // Unlimited

        // Get or create atomic counter for this step
        uint64_t key = step;
        auto result = states_per_step_.lookup(key);
        std::atomic<size_t>* counter = nullptr;

        if (result.has_value()) {
            counter = *result;
        } else {
            // Allocate new counter from arena
            counter = hg_->arena().template create<std::atomic<size_t>>(0);
            auto [existing, inserted] = states_per_step_.insert_if_absent(key, counter);
            if (!inserted) {
                counter = existing;  // Another thread beat us
            }
        }

        // Try to increment, fail if at limit
        size_t old_val = counter->fetch_add(1, std::memory_order_relaxed);
        if (old_val >= max_states_per_step_) {
            counter->fetch_sub(1, std::memory_order_relaxed);  // Rollback
            return false;
        }
        return true;
    }

    // Check if we should explore a new state based on exploration_probability.
    // Uses thread-local random state to avoid contention.
    bool should_explore() {
        if (exploration_probability_ >= 1.0) return true;
        if (exploration_probability_ <= 0.0) return false;

        // Thread-local random number generation
        thread_local std::mt19937 rng(std::random_device{}());
        thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

        return dist(rng) < exploration_probability_;
    }

    // =========================================================================
    // REWRITE Task Execution
    // =========================================================================
    // Applies a match and spawns MATCH tasks for new states.
    // Passes MatchContext to enable match forwarding.

    void execute_rewrite_task(const MatchRecord& match, uint32_t step) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        // Check step limit - don't spawn REWRITEs past max_steps
        // Note: step for REWRITE is the step at which it was created, child MATCH uses step+1
        if (max_steps_ > 0 && step > max_steps_) return;

        // Check limits before applying
        if (max_states_ > 0 && hg_->num_states() >= max_states_) {
            should_stop_.store(true, std::memory_order_relaxed);
            return;
        }
        if (max_events_ > 0 && hg_->num_events() >= max_events_) {
            should_stop_.store(true, std::memory_order_relaxed);
            return;
        }

        // Pruning: check max_successor_states_per_parent
        if (!try_reserve_successor_slot(match.source_state)) {
            return;  // Parent has too many children already
        }

        // Pruning: check max_states_per_step (child will be at step+1)
        if (!try_reserve_step_slot(step + 1)) {
            return;  // Too many states at this generation
        }

        const RewriteRule& rule = rules_[match.rule_index];

        // Apply the rewrite
        RewriteResult rr = rewriter_.apply(
            rule,
            match.source_state,
            match.matched_edges,
            match.num_edges,
            match.binding,
            step
        );

        if (rr.new_state != INVALID_ID) {
            total_rewrites_.fetch_add(1, std::memory_order_relaxed);
            total_events_.fetch_add(1, std::memory_order_relaxed);

            if (rr.was_new_state) {
                total_new_states_.fetch_add(1, std::memory_order_relaxed);
            }

            // Emit visualization events for canonical states only
            // For events: emit ALL events but include canonical_event_id for deduplication
            // The visualization can then deduplicate edges using canonical event IDs
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
            if (rr.was_new_state) {
                // Emit StateCreated only for new canonical states
                const auto& state_data = hg_->get_state(rr.new_state);
                VIZ_EMIT_STATE_CREATED(
                    rr.new_state,             // state id (canonical)
                    match.source_state,       // parent state id
                    step + 1,                 // generation
                    state_data.edges.count(), // edge count
                    0                         // vertex count (not tracked)
                );
                // Emit hyperedge data for each edge in the new state
                uint32_t edge_idx = 0;
                state_data.edges.for_each([&](EdgeId eid) {
                    const Edge& edge = hg_->get_edge(eid);
                    VIZ_EMIT_HYPEREDGE(rr.new_state, edge_idx++, edge.vertices, edge.arity);
                });
            }
            // Emit RewriteApplied for ALL events
            // Use canonical state as target (for consistent edge endpoints)
            // The event_id allows tracking raw→canonical mapping for causal/branchial
            // canonical_event_id allows deduplication: only create viz edges for canonical events
            VIZ_EMIT_REWRITE_APPLIED(
                match.source_state,       // source state
                rr.new_state,             // target state (canonical)
                match.rule_index,         // rule index
                rr.event,                 // raw event id (for tracking)
                rr.canonical_event,       // canonical event id (for deduplication)
                match.num_edges,          // destroyed edges count
                rr.num_produced           // created edges count
            );
#endif

            // Spawn MATCH task for the new raw state if it hasn't been matched yet
            // In multiway rewriting, each raw state (even if isomorphic) needs to be
            // matched to find all possible transitions
            // Use waiting version to avoid race during resize
            auto [existing, inserted] = matched_raw_states_.insert_if_absent_waiting(
                rr.raw_state, true);

            if (inserted) {
                DEBUG_LOG("STATE parent=%u -> child=%u (canonical=%u) rule=%u step=%u new=%d",
                          match.source_state, rr.raw_state, rr.new_state, match.rule_index, step, rr.was_new_state);

                // Build MatchContext for match forwarding
                MatchContext ctx;
                ctx.parent_state = match.source_state;
                ctx.num_consumed = match.num_edges;
                for (uint8_t i = 0; i < match.num_edges; ++i) {
                    ctx.consumed_edges[i] = match.matched_edges[i];
                }
                ctx.num_produced = rr.num_produced;
                for (uint8_t i = 0; i < rr.num_produced; ++i) {
                    ctx.produced_edges[i] = rr.produced_edges[i];
                }

                // Pruning: check exploration_probability (v1 style)
                // State is created and event recorded, but we may skip further exploration.
                // This is checked AFTER state creation to match v1 behavior.
                if (!should_explore()) {
                    // State exists but won't be explored further
                    return;
                }

                // Register child's parent pointer for ancestor chain walking.
                // With batched matching, we don't need push-based forwarding:
                // - Parent's MATCH task stores ALL matches before spawning REWRITEs
                // - Child's MATCH task pulls from ancestors (all matches visible)
                // - No race condition!
                if (enable_match_forwarding_) {
                    // Child is created at current step, its MATCH runs at step+1
                    register_child_with_parent(
                        match.source_state, rr.raw_state,
                        match.matched_edges, match.num_edges,
                        step);  // Pass step so pushed matches use correct child step
                    // Note: forward_existing is done in child's MATCH task, not here.
                    // This avoids redundant work and keeps the logic centralized.
                }

                // Submit MATCH task with context for match forwarding
                submit_match_task_with_context(rr.raw_state, step + 1, ctx);
            }
        }
    }
};

}  // namespace hypergraph::unified
