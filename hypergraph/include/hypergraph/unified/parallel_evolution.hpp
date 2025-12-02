#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <set>
#include <map>
#include <functional>
#include <thread>
#include <cstring>

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

#include <job_system/job_system.hpp>

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

    // Hash for deduplication - uses canonical state for determinism
    uint64_t hash() const {
        uint64_t h = rule_index;
        h = h * 31 + canonical_source;
        // Hash the binding (vertex assignments) instead of edge IDs
        // This makes deduplication isomorphism-aware
        for (uint8_t i = 0; i < MAX_VARS; ++i) {
            if (binding.is_bound(i)) {
                h = h * 31 + binding.get(i);
            }
        }
        return h;
    }

    bool operator==(const MatchRecord& other) const {
        if (rule_index != other.rule_index || num_edges != other.num_edges ||
            canonical_source != other.canonical_source) return false;
        // Compare bindings instead of edge IDs
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
};

// =============================================================================
// Job Types for Parallel Evolution
// =============================================================================

enum class EvolutionJobType {
    REWRITE,    // Apply a match to produce new state
    MATCH       // Find matches in a state (future use)
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
// ParallelEvolutionEngine
// =============================================================================
// Evolution engine with parallel REWRITE task execution.
// Uses job system for concurrent match application.
//
// Thread safety model (LOCK-FREE):
// - UnifiedHypergraph uses lock-free data structures
// - ConcurrentHeterogeneousArena for thread-safe allocation
// - Match deduplication uses ConcurrentMap (lock-free)
// - Results collected via lock-free list

class ParallelEvolutionEngine {
    UnifiedHypergraph* hg_;
    Rewriter rewriter_;

    // Rules
    std::vector<RewriteRule> rules_;

    // Global match deduplication (lock-free)
    ConcurrentMap<uint64_t, bool> seen_match_hashes_;

    // Job system
    job_system::JobSystem<EvolutionJobType>* job_system_;
    bool owns_job_system_{false};
    size_t num_threads_{0};

    // Evolution control
    std::atomic<bool> should_stop_{false};
    size_t max_steps_{0};
    size_t max_states_{0};
    size_t max_events_{0};

    // Statistics (atomics for thread-safety)
    std::atomic<size_t> total_matches_found_{0};
    std::atomic<size_t> total_rewrites_{0};
    std::atomic<size_t> total_new_states_{0};
    std::atomic<size_t> total_events_{0};
    std::atomic<size_t> forwarded_matches_{0};
    std::atomic<size_t> rejected_duplicates_{0};

    // Per-step result collection (lock-free)
    LockFreeList<ParallelRewriteResult> step_results_;

    // Evolution statistics
    EvolutionStats stats_;

public:
    ParallelEvolutionEngine()
        : hg_(nullptr)
        , rewriter_(nullptr)
        , job_system_(nullptr)
    {}

    explicit ParallelEvolutionEngine(UnifiedHypergraph* hg, size_t num_threads = 0)
        : hg_(hg)
        , rewriter_(hg)
        , num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency())
    {
        // Create job system
        job_system_ = new job_system::JobSystem<EvolutionJobType>(num_threads_);
        owns_job_system_ = true;
        job_system_->start();
    }

    ~ParallelEvolutionEngine() {
        if (owns_job_system_ && job_system_) {
            job_system_->shutdown();
            delete job_system_;
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

    size_t num_threads() const { return num_threads_; }
    size_t num_states() const { return hg_ ? hg_->num_states() : 0; }
    size_t num_canonical_states() const { return hg_ ? hg_->num_canonical_states() : 0; }
    size_t num_events() const { return hg_ ? hg_->num_events() : 0; }
    size_t num_causal_edges() const { return hg_ ? hg_->causal_graph().num_causal_edges() : 0; }
    size_t num_branchial_edges() const { return hg_ ? hg_->causal_graph().num_branchial_edges() : 0; }

    const EvolutionStats& stats() const { return stats_; }

    // =========================================================================
    // Main Evolution Loop
    // =========================================================================

    void evolve(const std::vector<std::vector<VertexId>>& initial_edges, size_t steps) {
        if (!hg_ || rules_.empty()) return;

        max_steps_ = steps;

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

        uint64_t canonical_hash = hg_->compute_canonical_hash(initial_edge_set);
        // Use create_or_get_canonical_state to register in canonical map
        // For initial state, raw_state == canonical state since it's the first
        auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
            std::move(initial_edge_set), canonical_hash, 0, INVALID_ID);

        // Evolution loop - use raw_state for evolution (has actual edges)
        std::vector<StateId> current_states = {raw_state};

        for (size_t step = 1; step <= steps && !should_stop_.load(std::memory_order_relaxed); ++step) {
            std::vector<RewriteTask> tasks;

            // Find all matches in current states
            for (StateId sid : current_states) {
                find_matches_in_state(sid, static_cast<uint32_t>(step), tasks);
            }

            if (tasks.empty()) break;

            // Submit all rewrite tasks to job system
            for (const auto& task : tasks) {
                auto job = job_system::make_job<EvolutionJobType>(
                    [this, task]() {
                        apply_match_parallel(task);
                    },
                    EvolutionJobType::REWRITE
                );
                job_system_->submit(std::move(job));
            }

            // Wait for all tasks to complete
            job_system_->wait_for_completion();

            // Collect results (lock-free iteration)
            // Use raw_state (not canonical state) for evolution, as it contains the
            // actual produced edges. The canonical state may have different edge IDs.
            // But only add ONE raw state per canonical state to avoid duplicate work.
            //
            // IMPORTANT: Pick the raw_state with smallest StateId for each canonical
            // state to ensure deterministic behavior across runs. The lock-free list
            // has non-deterministic iteration order, so we can't just take the first.
            std::map<StateId, StateId> canonical_to_raw;  // canonical -> smallest raw
            step_results_.for_each([&](const ParallelRewriteResult& result) {
                if (result.success) {
                    auto it = canonical_to_raw.find(result.new_state);
                    if (it == canonical_to_raw.end()) {
                        canonical_to_raw[result.new_state] = result.raw_state;
                    } else {
                        // Keep the smaller raw_state ID for determinism
                        if (result.raw_state < it->second) {
                            it->second = result.raw_state;
                        }
                    }
                }
            });

            // Extract raw states in canonical order (std::map is sorted)
            std::vector<StateId> next_states;
            for (const auto& [canonical, raw] : canonical_to_raw) {
                next_states.push_back(raw);
            }

            // Clear results for next step by creating new empty list
            step_results_ = LockFreeList<ParallelRewriteResult>();

            if (next_states.empty()) break;

            current_states = std::move(next_states);

            // Check limits
            if (max_states_ > 0 && hg_->num_states() >= max_states_) break;
            if (max_events_ > 0 && hg_->num_events() >= max_events_) break;
        }
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    size_t total_matches() const { return total_matches_found_.load(std::memory_order_relaxed); }
    size_t total_rewrites() const { return total_rewrites_.load(std::memory_order_relaxed); }

private:
    // =========================================================================
    // Match Finding (single-threaded per state, but states can be processed in parallel)
    // =========================================================================

    void find_matches_in_state(StateId state, uint32_t step, std::vector<RewriteTask>& out_tasks) {
        const State& s = hg_->get_state(state);

        // Get canonical state for deterministic deduplication
        StateId canonical_state = hg_->get_canonical_state(state);

        // Edge accessor
        auto get_edge = [this](EdgeId eid) -> const Edge& {
            return hg_->get_edge(eid);
        };

        // Match callback
        auto on_match = [&, state, canonical_state, step](
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
            for (uint8_t i = 0; i < num_edges; ++i) {
                match.matched_edges[i] = edges[i];
            }

            // Deduplicate using lock-free ConcurrentMap
            uint64_t h = match.hash();
            auto [existing, inserted] = seen_match_hashes_.insert_if_absent(h, true);
            if (!inserted) {
                rejected_duplicates_.fetch_add(1, std::memory_order_relaxed);
                return;  // Already seen
            }

            total_matches_found_.fetch_add(1, std::memory_order_relaxed);
            out_tasks.emplace_back(match, step);
        };

        // Find all matches for each rule
        for (uint16_t r = 0; r < rules_.size(); ++r) {
            find_matches(
                rules_[r], r, state, s.edges,
                hg_->signature_index(), hg_->inverted_index(), get_edge, on_match
            );
        }
    }

    // =========================================================================
    // Parallel Rewrite Application
    // =========================================================================

    void apply_match_parallel(const RewriteTask& task) {
        const RewriteRule& rule = rules_[task.match.rule_index];

        // Apply the rewrite
        RewriteResult rr = rewriter_.apply(
            rule,
            task.match.source_state,
            task.match.matched_edges,
            task.match.num_edges,
            task.match.binding,
            task.step
        );

        ParallelRewriteResult result;
        result.match = task.match;

        if (rr.new_state != INVALID_ID) {
            result.new_state = rr.new_state;
            result.raw_state = rr.raw_state;
            result.event = rr.event;
            result.was_new_state = rr.was_new_state;
            result.success = true;

            // Copy produced edges for later match forwarding
            result.num_produced = rr.num_produced;
            std::memcpy(result.produced_edges, rr.produced_edges, rr.num_produced * sizeof(EdgeId));

            // Track edges for forwarding
            const State& parent_state = hg_->get_state(task.match.source_state);
            for (uint8_t i = 0; i < task.match.num_edges; ++i) {
                result.deleted_edges.set(task.match.matched_edges[i], hg_->arena());
            }
            for (uint8_t i = 0; i < rr.num_produced; ++i) {
                result.created_edges.set(rr.produced_edges[i], hg_->arena());
            }

            total_rewrites_.fetch_add(1, std::memory_order_relaxed);
            if (rr.was_new_state) {
                total_new_states_.fetch_add(1, std::memory_order_relaxed);
            }
            total_events_.fetch_add(1, std::memory_order_relaxed);
        }

        // Add to lock-free results list
        step_results_.push(result, hg_->arena());
    }
};

}  // namespace hypergraph::unified
