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
#include "hypergraph.hpp"
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

namespace hypergraph {

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
    StateId state{INVALID_ID};              // State to match in
    uint16_t rule_index{0};                 // Which rule to match
    uint32_t step{0};                       // Evolution step
    StateId canonical_state{INVALID_ID};    // For deterministic deduplication
    uint64_t source_canonical_hash{0};      // Canonical hash of source state
    // For delta matching (only find NEW matches involving produced edges)
    bool is_delta{false};                   // If true, only match involving produced_edges
    EdgeId produced_edges[MAX_PATTERN_EDGES]{};  // Zero-initialized
    uint8_t num_produced{0};
};

// EXPAND task: Extend partial match by one edge
// Also used for SINK (when match is complete)
struct ExpandTaskData {
    StateId state{INVALID_ID};              // State being matched
    uint16_t rule_index{0};                 // Rule being matched
    uint8_t num_pattern_edges{0};           // Total edges in pattern
    uint8_t next_pattern_idx{0};            // Which pattern edge to match next (0-based)
    EdgeId matched_edges[MAX_PATTERN_EDGES]{};  // Data edges matched so far
    uint8_t match_order[MAX_PATTERN_EDGES]{};   // Pattern indices in match order
    uint8_t num_matched{0};                 // Number of edges matched
    VariableBinding binding{};              // Current variable bindings
    uint32_t step{0};                       // Evolution step
    StateId canonical_state{INVALID_ID};    // For deterministic deduplication
    uint64_t source_canonical_hash{0};      // Canonical hash of source state

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
// - Hypergraph uses lock-free data structures
// - ConcurrentHeterogeneousArena for thread-safe allocation
// - Match deduplication uses ConcurrentMap (lock-free)
// - State tracking uses ConcurrentMap (lock-free)

class ParallelEvolutionEngine {
    Hypergraph* hg_;
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

    // Uniform random mode: control flags for step-synchronized evolution
    // When true, MATCH tasks don't spawn REWRITEs (main loop does selection)
    // When true, REWRITE tasks don't spawn MATCH tasks (main loop collects new states)
    bool uniform_random_mode_{false};
    LockFreeList<StateId>* pending_new_states_simple_{nullptr};  // Simple state collection (no forwarding)
    LockFreeList<MatchRecord>* pending_matches_{nullptr};  // Temporary match collection (cleared each step)

    // Early termination: stop pattern matching for a job when its reservoir is full
    // Trades strict uniform sampling (over ALL matches) for speed
    // When true: uniform over matches found before termination (faster, less uniform)
    // When false: uniform over ALL possible matches (slower, strictly uniform)
    bool early_terminate_on_reservoir_full_{false};

    // Genesis events: create synthetic events for initial states that produce
    // all initial edges. This enables causal edges from initial state to gen 1.
    // Disabled by default to match v1 behavior.
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

    // Exploration deduplication: only explore from canonical state representatives.
    // When enabled, states equivalent to already-seen states are created (with events)
    // but MATCH tasks are not spawned - we don't explore further from them.
    // This focuses compute on discovering new states rather than all transition paths.
    bool explore_from_canonical_states_only_{false};

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

    explicit ParallelEvolutionEngine(Hypergraph* hg, size_t num_threads = 0);

    ~ParallelEvolutionEngine();

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
    void set_early_terminate_on_reservoir_full(bool enable) {
        early_terminate_on_reservoir_full_ = enable;
    }

    void set_explore_from_canonical_states_only(bool enable) {
        explore_from_canonical_states_only_ = enable;
    }
    bool explore_from_canonical_states_only() const {
        return explore_from_canonical_states_only_;
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
                [[maybe_unused]] uint32_t state_id = debug_info >> 16;
                [[maybe_unused]] uint16_t rule_index = debug_info & 0xFFFF;
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

    void evolve(const std::vector<std::vector<VertexId>>& initial_edges, size_t steps);

    // Overload for multiple initial states (without abort callback)
    // Each initial state is evolved from independently, exploring the full multiway system
    void evolve(const std::vector<std::vector<std::vector<VertexId>>>& initial_states, size_t steps);

    // Evolve with abort callback - allows external abort control (e.g., from Mathematica)
    // The abort_check callback is called from the main thread periodically (~50ms)
    // If it returns true, evolution stops early and this function returns true (aborted)
    // Returns false if evolution completed normally
    template<typename AbortCheck>
    bool evolve_with_abort(const std::vector<std::vector<VertexId>>& initial_edges,
                           size_t steps,
                           AbortCheck&& abort_check) {
        if (!hg_ || rules_.empty()) return false;

        max_steps_ = steps;
        should_stop_.store(false, std::memory_order_relaxed);

        // Propagate abort flag to hypergraph for long-running hash computations
        hg_->set_abort_flag(&should_stop_);

        // Create initial state (same setup as evolve())
        std::vector<EdgeId> edge_ids;
        for (const auto& edge : initial_edges) {
            EdgeId eid = hg_->create_edge(edge.data(), static_cast<uint8_t>(edge.size()));
            edge_ids.push_back(eid);
            for (VertexId v : edge) {
                hg_->reserve_vertices(v);
            }
        }

        SparseBitset initial_edge_set;
        for (EdgeId eid : edge_ids) {
            initial_edge_set.set(eid, hg_->arena());
        }

        auto [canonical_hash, vertex_cache] = hg_->compute_canonical_hash_incremental(
            initial_edge_set,
            INVALID_ID,
            nullptr, 0,
            edge_ids.data(), static_cast<uint8_t>(edge_ids.size())
        );
        auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
            std::move(initial_edge_set), canonical_hash, 0, INVALID_ID);

        hg_->store_state_cache(raw_state, vertex_cache);

        if (enable_genesis_events_) {
            hg_->create_genesis_event(raw_state, edge_ids.data(), static_cast<uint8_t>(edge_ids.size()));
        }

        matched_raw_states_.insert_if_absent_waiting(raw_state, true);
        submit_match_task(raw_state, 1);

        // Wait with abort checking - abort_check called from main thread
        bool aborted = job_system_->wait_for_completion_with_abort([&]() {
            if (abort_check()) {
                request_stop();
                return true;
            }
            return false;
        });

        // CRITICAL: If aborted, we must still wait for all executing jobs to finish
        // before returning. Jobs that passed the should_stop_ check are still running
        // and accessing hg_. Returning early would cause use-after-free.
        if (aborted) {
            job_system_->wait_for_completion();
        }

        finalize_evolution();
        return aborted;
    }

    // Overload for multiple initial states
    // Each initial state is evolved from independently, exploring the full multiway system
    template<typename AbortCheck>
    bool evolve_with_abort(const std::vector<std::vector<std::vector<VertexId>>>& initial_states,
                           size_t steps,
                           AbortCheck&& abort_check) {
        if (!hg_ || rules_.empty() || initial_states.empty()) return false;

        max_steps_ = steps;
        should_stop_.store(false, std::memory_order_relaxed);

        // Propagate abort flag to hypergraph for long-running hash computations
        hg_->set_abort_flag(&should_stop_);

        // Create all initial states - they will all be explored
        for (const auto& state_edges : initial_states) {
            create_and_register_initial_state(state_edges);
        }

        // Wait with abort checking - abort_check called from main thread
        bool aborted = job_system_->wait_for_completion_with_abort([&]() {
            if (abort_check()) {
                request_stop();
                return true;
            }
            return false;
        });

        // CRITICAL: If aborted, we must still wait for all executing jobs to finish
        // before returning. Jobs that passed the should_stop_ check are still running
        // and accessing hg_. Returning early would cause use-after-free.
        if (aborted) {
            job_system_->wait_for_completion();
        }

        finalize_evolution();
        return aborted;
    }

    // =========================================================================
    // Uniform Random Evolution - Step-Synchronized
    // =========================================================================
    // Evolves by completing all MATCH tasks at each step, collecting all matches,
    // randomly selecting which to apply, then completing all REWRITEs before
    // moving to the next step. This enables uniform random sampling across the
    // entire multiway system at each generation.
    //
    // Parameters:
    //   initial_edges: Initial hypergraph edges
    //   steps: Maximum number of evolution steps
    //   matches_per_step: How many matches to randomly select per step (0 = all)

    void evolve_uniform_random(
        const std::vector<std::vector<VertexId>>& initial_edges,
        size_t steps,
        size_t matches_per_step = 1
    );

private:
    void finalize_evolution();

    // Helper: Create an initial state from a set of edges WITHOUT submitting for matching
    // Used by uniform random mode which does synchronous matching
    StateId create_initial_state_only(const std::vector<std::vector<VertexId>>& edges);

    // Helper: Create an initial state from a set of edges and register it for matching
    // Returns the raw state ID, or INVALID_ID if creation failed
    StateId create_and_register_initial_state(const std::vector<std::vector<VertexId>>& edges);

public:
    // =========================================================================
    // Statistics
    // =========================================================================

    size_t total_matches() const { return total_matches_found_.load(std::memory_order_relaxed); }
    size_t total_rewrites() const { return total_rewrites_.load(std::memory_order_relaxed); }

    // Job system diagnostics
    size_t pending_jobs() const { return job_system_ ? job_system_->get_pending_count() : 0; }
    size_t executing_jobs() const { return job_system_ ? job_system_->get_executing_count() : 0; }

    // Error state - check after evolution completes
    bool has_error() const { return job_system_ && job_system_->has_error(); }
    job_system::ErrorType get_error_type() const {
        return job_system_ ? job_system_->get_error_type() : job_system::ErrorType::None;
    }
    const char* get_error_description() const {
        return job_system_ ? job_system_->get_error_description() : "No job system";
    }

private:
    // Helper: Get or create the match list for a state (thread-safe)
    LockFreeList<MatchRecord>* get_or_create_state_matches(StateId state);

    // Helper: Store a match for a state (for later forwarding)
    uint64_t store_match_for_state(StateId state, MatchRecord& match, bool with_fence = false);

    // Helper: Get or create the children list for a state (thread-safe)
    LockFreeList<ChildInfo>* get_or_create_state_children(StateId state);

    // Helper: Register a child with its parent (for push-based forwarding)
    uint64_t register_child_with_parent(StateId parent, StateId child,
                                     const EdgeId* consumed_edges, uint8_t num_consumed,
                                     uint32_t child_step = 0);

    // Helper: Push a match to immediate children (single-level push)
    void push_match_to_children(StateId parent, const MatchRecord& match, uint32_t step);

    void push_match_to_children_impl(StateId parent, const MatchRecord& match, uint32_t step);

    // Helper: Forward existing parent matches to a newly created child
    void forward_existing_parent_matches(
        StateId parent, StateId child,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        uint32_t step,
        std::vector<MatchRecord>& batch
    );

    void forward_matches_from_single_ancestor(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t step,
        std::vector<MatchRecord>& batch
    );

    void forward_matches_from_single_ancestor_impl(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t step, uint64_t child_registration_epoch,
        std::vector<MatchRecord>& batch
    );

    // EAGER MODE: Forward existing parent matches with retry loop
    void forward_existing_parent_matches_eager(
        StateId parent, StateId child,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        uint32_t step
    );

    void forward_matches_from_single_ancestor_eager(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t step
    );

    // Task Submission
    void submit_match_task(StateId state, uint32_t step);
    void submit_match_task_with_context(StateId state, uint32_t step, const MatchContext& ctx);
    void submit_rewrite_task(const MatchRecord& match, uint32_t step);
    void submit_scan_task(const ScanTaskData& data);

    void submit_expand_task(const ExpandTaskData& data);
    void submit_sink_task(const ExpandTaskData& data);

    // Task Execution
    void execute_match_task(StateId state, uint32_t step, const MatchContext& ctx);
    void execute_scan_task(const ScanTaskData& data);
    void execute_expand_task(const ExpandTaskData& data);
    void execute_sink_task(const ExpandTaskData& data);
    void execute_rewrite_task(const MatchRecord& match, uint32_t step);

    // Pruning helpers
    bool can_create_states_at_step(uint32_t step) const;
    bool can_have_more_children(StateId parent) const;
    bool try_reserve_successor_slot(StateId parent);
    bool try_reserve_step_slot(uint32_t step);
    void release_step_slot(uint32_t step);
    bool should_explore();

    // Bias mitigation: returns rule indices in shuffled order
    std::vector<uint16_t> get_shuffled_rule_indices() const;
};

}  // namespace hypergraph
