#pragma once
#include "hgcommon/namespace.hpp"

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
#include <algorithm>

#include "types.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "hypergraph.hpp"
#include "hgcommon/portable_intrinsics.hpp"
#include "hypergraph/scratch_alloc.hpp"
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

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// Match Core (immutable, shared)
// =============================================================================
// The heavy payload of a match: the rule, the matched data edges, and the
// variable binding. Allocated once in the arena when a match is first
// discovered and never mutated afterwards, so a match forwarded to descendant
// states shares one MatchCore by pointer instead of deep-copying the payload
// into every descendant. Immutability makes that sharing race-free.
struct MatchCore {
    uint16_t rule_index{0};
    uint8_t num_edges{0};
    EdgeId matched_edges[MAX_PATTERN_EDGES]{};
    VariableBinding binding{};
};

// =============================================================================
// Match Record
// =============================================================================
// A match as forwarded to and stored in a descendant state: a pointer to the
// shared immutable core plus this descendant's own source_state. Forwarding a
// match to a descendant copies only these two words; the edge list and binding
// stay behind the core pointer, shared across all descendants of a discovery.
//
// source_state is per-descendant because the dedup hash keys on it (see hash())
// so each descendant deduplicates independently. All heavy-field reads go
// through core.
struct MatchRecord {
    const MatchCore* core{nullptr};
    StateId source_state{INVALID_ID};

    uint16_t rule_index() const { return core->rule_index; }
    uint8_t num_edges() const { return core->num_edges; }
    const EdgeId* matched_edges() const { return core->matched_edges; }
    const VariableBinding& binding() const { return core->binding; }

    // Hash for deduplication - uses source_state + matched edges + binding.
    // MUST use source_state (raw state ID), NOT any canonical identifier!
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
        const MatchCore& c = *core;
        // FNV-1a style hash for better distribution and collision resistance
        uint64_t h = 14695981039346656037ULL;  // FNV offset basis
        constexpr uint64_t FNV_PRIME = 1099511628211ULL;

        // Mix in rule_index
        h ^= c.rule_index;
        h *= FNV_PRIME;

        // Mix in source_state (raw state ID for collision-free deduplication)
        h ^= source_state;
        h *= FNV_PRIME;
        h ^= (source_state >> 16);  // Extra mixing for better distribution
        h *= FNV_PRIME;

        // Mix in matched edges
        for (uint8_t i = 0; i < c.num_edges; ++i) {
            h ^= c.matched_edges[i];
            h *= FNV_PRIME;
        }

        // Mix in bound_mask
        h ^= c.binding.bound_mask;
        h *= FNV_PRIME;

        // Mix in bound variables, walking the SET BITS of the mask rather than all MAX_VARS
        // positions. Identical result -- the loop body was already guarded by is_bound, which is
        // that same mask -- for as many iterations as there are bound variables instead of 32.
        // This is the engine's most-probed hash: once per discovered match, and again per
        // (match, descendant) pair that forwarding carries.
        uint32_t mask = c.binding.bound_mask;
        while (mask) {
            const uint32_t i = hgcommon::ctz(mask);
            mask &= mask - 1;
            h ^= (static_cast<uint64_t>(i) << 32) | c.binding.get(static_cast<uint8_t>(i));
            h *= FNV_PRIME;
        }

        return h;
    }

    bool operator==(const MatchRecord& other) const {
        const MatchCore& a = *core;
        const MatchCore& b = *other.core;
        if (a.rule_index != b.rule_index || a.num_edges != b.num_edges ||
            source_state != other.source_state) return false;
        // Compare matched edges
        for (uint8_t i = 0; i < a.num_edges; ++i) {
            if (a.matched_edges[i] != b.matched_edges[i]) return false;
        }
        // Compare bindings
        for (uint8_t i = 0; i < MAX_VARS; ++i) {
            if (a.binding.is_bound(i) != b.binding.is_bound(i)) return false;
            if (a.binding.is_bound(i) && a.binding.get(i) != b.binding.get(i)) return false;
        }
        return true;
    }
};

// =============================================================================
// Evolution Statistics
// =============================================================================

// Every worker bumps several of these per match and per state. Packed, nine of them fit
// in barely more than one cache line, so counters that record unrelated events would
// contend for the same line -- pure interference, since nothing ever reads them during a
// run. One line each; the struct is a singleton, so the padding is free.
struct EvolutionStats {
    alignas(64) std::atomic<size_t> states_created{0};
    alignas(64) std::atomic<size_t> events_created{0};
    alignas(64) std::atomic<size_t> matches_found{0};
    alignas(64) std::atomic<size_t> matches_forwarded{0};
    alignas(64) std::atomic<size_t> matches_invalidated{0};
    alignas(64) std::atomic<size_t> new_matches_discovered{0};
    alignas(64) std::atomic<size_t> full_pattern_matches{0};
    alignas(64) std::atomic<size_t> delta_pattern_matches{0};
    // Extra ancestor re-walks / child re-scans the forwarding rendezvous performs
    // when its epoch changes during a push or pull (a measure of cross-worker churn).
    alignas(64) std::atomic<size_t> forwarding_rewalks{0};

    // Whether push_match_to_children finds anything to push to, split by WHEN it is called.
    // The two sites answer different questions and the split is the measurement:
    //
    //   DISCOVERY -- from SINK, as a match is first completed. Under batched submission a
    //   state's matching finishes before any of its rewrites are submitted, so no child of
    //   that state should exist yet and every one of these calls should find an empty
    //   registry. If that holds, the call is a no-op under batched and the work the child
    //   would have received arrives instead through its pull at creation, when the parent's
    //   match set is already complete.
    //
    //   FORWARDING -- a match arriving from an ancestor, propagated onward. These CAN find
    //   children, because the state was matched earlier and its children already exist.
    //
    // An empty-registry fraction below 1.0 at the discovery site falsifies the reasoning
    // above about when children become visible, and the cause is then elsewhere.
    alignas(64) std::atomic<size_t> push_discovery_calls{0};
    alignas(64) std::atomic<size_t> push_discovery_empty{0};
    alignas(64) std::atomic<size_t> push_forwarding_calls{0};
    alignas(64) std::atomic<size_t> push_forwarding_empty{0};
    // Transitions kept by the spine rather than by a passing draw (drain minimum-key spawns plus
    // late spines). draws_survived_ does not include these, so kept = survived + spine_forced.
    alignas(64) std::atomic<size_t> spine_forced{0};
};

// =============================================================================
// Job Types for Parallel Evolution
// =============================================================================

enum class EvolutionJobType {
    SCAN,       // Find initial candidates for first pattern edge
    EXPAND,     // Extend partial match by one edge
    MATCH,      // Orchestrate matching for a state (spawn SCAN tasks or fallback to sync)
    REWRITE,    // Apply a contiguous RANGE of one state's matches
};

// A contiguous range of one state's matches, applied by a single task.
//
// The unit of work is a state expansion, not an individual match. The locality in this
// computation lives at the STATE: every child of one parent is built from the same edge set,
// canonicalized against the same starting adjacency, and every branchial pair among them is
// internal to that parent. Tasking per match discards all of it -- the parent's adjacency and
// initial partition are rebuilt for each child, which is why incremental canonicalization had
// no amortization hook and stayed refuted, and canonicalization is about half of all
// instructions.
//
// Ranges rather than whole states so a parent with thousands of matches still balances. The
// match array is arena-resident and immutable, so chunks of one parent share it read-only --
// shared clean lines, which cost nothing to replicate across cores.
struct ExpandChunk {
    const MatchRecord* matches = nullptr;   // arena-resident, shared by all chunks of a parent
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t step = 0;
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
// SCAN/EXPAND Task Data Structures (HGMatch Dataflow Model)
// =============================================================================
// These structures capture all data needed to execute matching tasks in parallel.
// Following HGMatch paper: a SCAN seeds the join and EXPAND* extends it one pattern
// edge at a time. A completed match is finished in place by complete_match, and the
// matches one task completes are expanded together.

// SCAN task: Find initial candidates for first pattern edge
struct ScanTaskData {
    StateId state{INVALID_ID};              // State to match in
    uint16_t rule_index{0};                 // Which rule to match
    uint32_t step{0};                       // Evolution step
    // For delta matching (only find NEW matches involving produced edges)
    bool is_delta{false};                   // If true, only match involving produced_edges
    EdgeId produced_edges[MAX_PATTERN_EDGES]{};  // Zero-initialized
    uint8_t num_produced{0};
};

// EXPAND task: Extend partial match by one edge
// Also carries a completed match into complete_match
struct ExpandTaskData {
    StateId state{INVALID_ID};              // State being matched
    uint16_t rule_index{0};                 // Rule being matched
    uint8_t num_pattern_edges{0};           // Total edges in pattern
    EdgeId matched_edges[MAX_PATTERN_EDGES]{};  // Data edges matched so far
    uint8_t match_order[MAX_PATTERN_EDGES]{};   // Pattern indices in match order
    uint8_t num_matched{0};                 // Number of edges matched
    VariableBinding binding{};              // Current variable bindings
    uint32_t step{0};                       // Evolution step

    bool is_complete() const {
        return num_matched >= num_pattern_edges;
    }

    bool contains_edge(EdgeId eid) const {
        for (uint8_t i = 0; i < num_matched; ++i) {
            if (matched_edges[i] == eid) return true;
        }
        return false;
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
public:
    // How the engine runs its work.
    //
    // Parallel: worker threads with work-stealing deques, sized by num_threads.
    //   Serial: no thread is spawned at all. Every job runs inline on the thread that called
    //           evolve(), in submission order, so the run is deterministic by construction --
    //           and it needs neither threads nor the atomics the deques spin on, which is what
    //           a WebAssembly target requires. Passing num_threads = 1 does NOT give this: it
    //           still spawns a worker and still goes through the deques.
    enum class ExecutionMode { Parallel, Serial };

    Hypergraph* hg_;
    Rewriter rewriter_;

    // Rules
    std::vector<RewriteRule> rules_;

    // Global match deduplication (lock-free)
    // Use non-zero EMPTY/LOCKED keys to avoid conflicts with valid hash values
    static constexpr uint64_t MATCH_MAP_EMPTY = 1ULL << 63;
    static constexpr uint64_t MATCH_MAP_LOCKED = (1ULL << 63) | 1;
    // Match dedup, keyed by hash but DECIDED by content.
    //
    // The value is the match itself, not a bool, because the hash alone cannot answer the
    // question being asked. Two DISTINCT matches whose 64-bit hashes collide are
    // indistinguishable to a key-only set, so the second is discarded as a duplicate and never
    // rewritten -- and every downstream artifact (states, events, causal edges, branchial edges,
    // transitive reduction) is computed from the match set, so the loss is silent and
    // self-consistent: the run simply produces less and looks internally fine.
    //
    // The probability is n^2 / 2^65, which is unreachable on the test corpus (~5e-8 at a million
    // matches) and reachable at the scale this engine claims (~3e-2 at a billion). It is the same
    // defect class already proved fatal for STATES, where a hash was likewise trusted to stand in
    // for the object.
    //
    // So on equal hash the stored record is COMPARED against the candidate. Equal means a true
    // duplicate. Unequal means a collision, and the candidate probes the next key rather than
    // being dropped. See claim_match().
    // The value is a POINTER, not the record. A MatchRecord is 16 bytes, and a 16-byte atomic
    // is not lock-free on this target -- it links against __atomic_load_16 and takes a lock.
    // The engine is lock-free throughout, so the map stores an 8-byte pointer to an
    // arena-resident record instead.
    ConcurrentMap<uint64_t, const MatchRecord*, MATCH_MAP_EMPTY, MATCH_MAP_LOCKED>
        seen_match_hashes_;

    // Probes attempted before a collision is treated as unresolvable. A single collision is
    // already astronomically unlikely; needing this many consecutive ones is not a scenario that
    // occurs, and the counter below exists so the assumption is measured rather than believed.
    //
    // claim_match's loop, this constant, and dedup_probe_key are TRANSCRIBED in
    // verification/genmc/claim_match_rendezvous.cpp, which model-checks exactly-once claiming and
    // no-drop-on-collision over the real ConcurrentMap. A change here must be mirrored there or
    // the harness verifies a rendezvous the engine no longer runs.
    static constexpr uint32_t kMaxDedupProbes = 8;
    std::atomic<size_t> hash_collisions_{0};
    std::atomic<size_t> dedup_probe_exhaustions_{0};
    // Stable MatchRecord copies allocated inside claim_match, and how many of those lost the
    // claim. A loser is permanent arena: the allocator is a bump pointer with no per-object
    // free, so the copy is never reclaimed. This is the number the batched default's extra
    // arena is made of.
    std::atomic<size_t> dedup_allocs_{0};
    std::atomic<size_t> dedup_allocs_wasted_{0};

    // True when the two records denote the same match: same source state, rule, edge tuple and
    // binding. This is the predicate the dedup is defined by; the hash only selects where to
    // look.
    static bool match_records_equal(const MatchRecord& a, const MatchRecord& b) {
        if (a.core == b.core) return a.source_state == b.source_state;  // forwarded copies share
        if (!a.core || !b.core) return false;
        return a == b;   // MatchRecord::operator== is the definition; do not restate it here
    }

    // Derive the key for probe attempt `n`, skipping the map's reserved sentinels.
    static uint64_t dedup_probe_key(uint64_t h, uint32_t n) {
        uint64_t k = h + static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ull;
        if (k == MATCH_MAP_EMPTY || k == MATCH_MAP_LOCKED) k += 0x9E3779B97F4A7C15ull;
        return k;
    }

public:
    // Claim `rec` for processing. Returns true when it is NEW and the caller must process it,
    // false when an equal match was already claimed.
    //
    // Public because the hash is a PARAMETER, which is what makes the collision path reachable
    // from a test: passing one hash with two distinct records is exactly the case a key-only set
    // answers wrongly, and it cannot be provoked through the evolution API at any workload size.
    //
    // `make_stable` returns a pointer that outlives this map, and is invoked AT MOST ONCE and
    // only when the match may actually be new. True duplicates are answered from the lookup and
    // never reach it, which matters because they are routine: delta matching finds a match on k
    // produced edges k times, once anchored on each.
    // True when an equal match has already been claimed.
    //
    // The read-only twin of claim_match: it walks the SAME probe chain and decides by the SAME
    // content comparison. A presence check written as "is this hash in the set" would answer yes
    // when a DIFFERENT match occupies the slot, which is the very error the completeness
    // validator exists to detect -- so the check cannot itself be written that way.
    bool contains_match(uint64_t h, const MatchRecord& rec) const {
        for (uint32_t n = 0; n < kMaxDedupProbes; ++n) {
            auto seen = seen_match_hashes_.lookup(dedup_probe_key(h, n));
            if (!seen) return false;      // an empty slot terminates the chain
            if (*seen && match_records_equal(**seen, rec)) return true;
        }
        return false;
    }

    template <typename MakeStable>
    bool claim_match(uint64_t h, const MatchRecord& rec, MakeStable&& make_stable) {
        const MatchRecord* stable = nullptr;
        for (uint32_t n = 0; n < kMaxDedupProbes; ++n) {
            const uint64_t key = dedup_probe_key(h, n);
            if (auto seen = seen_match_hashes_.lookup(key)) {
                if (*seen && match_records_equal(**seen, rec)) return false;   // true duplicate
                // Same key, a different match: a hash collision. Probing rather than discarding
                // is the whole point -- discarding here is what silently loses work.
                hash_collisions_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            // The stable copy has to exist before the exchange that publishes it, so it is
            // allocated on the strength of a lookup that just missed. Another thread can claim
            // the key in that window, and the arena is bump-allocated with no per-object free,
            // so a copy that loses is a permanent cost. Counted rather than assumed: this is the
            // arena the batched default pays for over eager, and a fix has to move THIS number.
            if (!stable) {
                stable = make_stable();
                dedup_allocs_.fetch_add(1, std::memory_order_relaxed);
            }
            auto [existing, inserted] = seen_match_hashes_.insert_if_absent(key, stable);
            if (inserted) return true;
            if (existing && match_records_equal(*existing, rec)) {
                dedup_allocs_wasted_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            hash_collisions_.fetch_add(1, std::memory_order_relaxed);
        }
        dedup_probe_exhaustions_.fetch_add(1, std::memory_order_relaxed);
        return true;   // process it: a redundant rewrite is recoverable, a lost one is not
    }

    // Distinct matches that landed on an equal key, and claims that ran out of probes. Both are
    // expected to be 0; they are counted so "collisions do not happen here" is a measurement.
    size_t hash_collisions() const { return hash_collisions_.load(std::memory_order_relaxed); }
    size_t dedup_probe_exhaustions() const {
        return dedup_probe_exhaustions_.load(std::memory_order_relaxed);
    }
    size_t dedup_allocs() const { return dedup_allocs_.load(std::memory_order_relaxed); }
    size_t dedup_allocs_wasted() const {
        return dedup_allocs_wasted_.load(std::memory_order_relaxed);
    }
private:

    // Track which raw states have been matched (lock-free)
    // Prevents duplicate MATCH tasks for the same raw state
    // Use uint64_t as key to avoid template issues with 32-bit StateId
    // StateId is 32-bit, so we use keys outside that range for EMPTY/LOCKED
    static constexpr uint64_t STATE_MAP_EMPTY = 1ULL << 62;
    static constexpr uint64_t STATE_MAP_LOCKED = (1ULL << 62) | 1;
    ConcurrentMap<uint64_t, uint8_t, STATE_MAP_EMPTY, STATE_MAP_LOCKED> matched_raw_states_;

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


    // Match forwarding enabled flag
    bool enable_match_forwarding_{true};

    // Batched matching: the parent finishes matching, THEN its children are created. Eager
    // creates each child as its match is found, so the parent is still matching when the child
    // exists and a match found afterwards has to reach it by push.
    //
    // Default TRUE, and the reason is the SHAPE of the two, not a defect rate. Both are measured
    // complete: MatchCompleteness.ForwardedPlusDeltaFindsEveryMatch runs the oracle corpus x
    // workers {1,4,8} x reps under EAGER with validate_match_forwarding on and reports 0 LOST
    // (matches counted absent by the validator and still absent when the run ended, tested with
    // contains_match), against a positive control -- disabling push_match_to_children makes the
    // same gate report 10 lost in 7 runs. Batched reports 0 of 51 with no residual at all.
    //
    // Batched CLOSES the window; eager COVERS it with the push rendezvous. Forwarding is
    // INDUCTIVE, so a match lost at depth d removes the whole subtree below it while the run
    // stays self-consistent and simply produces less -- nothing downstream can notice. A window
    // that cannot open is worth more than a window a rendezvous is measured to cover, because the
    // measurement is over the interleavings that happened to run.
    //
    // It costs: 13.52% more arena (cost_matrix, 17 cases; worst case star4-automorphic at
    // 20.88%), because push_match_to_children walks a populated child registry here where under
    // eager it finds an empty one. That overlap is skippable in principle and is tracked as #77.
    bool batched_matching_{true};  // false: submit each match eagerly; true: batch per step

    // Validation mode: cross-check forwarded+delta matches against a full scan
    bool validate_match_forwarding_{false};
    std::atomic<size_t> validation_mismatches_{0};
    // How many times the validator actually EXECUTED. A mismatch count of zero means nothing
    // unless this is nonzero, and the task-based path returns before reaching the check.
    std::atomic<size_t> validations_performed_{0};
    // Attribution of a missed match to the obligation that should have supplied it.
    // A match using ONLY edges that survived from the parent was already a match in the
    // parent, so FORWARDING owed it. A match touching a produced edge could not have existed
    // in the parent, so DELTA owed it. The two have different causes and different fixes.
    std::atomic<size_t> missing_owed_by_forwarding_{0};
    std::atomic<size_t> missing_owed_by_delta_{0};
    // How many thinning draws were TAKEN and how many survived. A draw is deterministic in its
    // key, so two modes that disagree on the kept fraction must disagree on the SET of keys they
    // draw on -- and the count is what shows that without dumping every key.
    mutable std::atomic<size_t> draws_taken_{0};
    mutable std::atomic<size_t> draws_survived_{0};
    // Draws attributed to each call site, so a mode difference names its site instead of
    // being inferred from call order. 0 push, 1 batched pull, 2 eager pull, 3 collect, 4 sink.
    mutable std::atomic<size_t> draws_by_site_[5]{};


    // Transition-level thinning: keep each transition with this probability. 1.0 = keep all.
    //
    // This is the sampler for "a sparse subgraph whose observables match the full graph's".
    // Thinning transitions INDEPENDENTLY yields a sub-branching-process whose offspring
    // distribution is the thinned original, so the branching factor's shape and variance
    // survive. A fixed count per state does not: it collapses the offspring distribution to a
    // point mass, destroying the feature the sample exists to preserve.
    double transition_rate_{1.0};
    // PER-RULE MULTIPLIERS on transition_rate_. Empty means every rule samples at the same
    // rate, which is what a caller that sets nothing gets. A weight scales its rule's rate, so
    // {1.0, 0.0} explores rule 0 fully and drops rule 1 entirely, and the two knobs compose
    // rather than one overriding the other.
    //
    // Indexed by RuleIndex. A rule past the end takes weight 1.0 rather than being an error:
    // the vector is a partial override, and a caller weighting the first of five rules should
    // not have to spell out four ones.
    std::vector<double> rule_weights_;



    // Genesis events: create synthetic events for initial states that produce
    // all initial edges. This enables causal edges from initial state to gen 1.
    // Disabled by default.
    bool enable_genesis_events_{false};

    // Task-based matching: use the SCAN→EXPAND join decomposition (HGMatch model)
    // When enabled, pattern matching spawns fine-grained tasks for better parallelism.
    // When disabled (default), uses synchronous find_matches() within MATCH task.
    bool task_based_matching_{true};

    // Track missing hashes to verify they arrive later via push
    // Value is (state_id << 16) | rule_index for debugging
    // Hash -> a STABLE copy of the match that was missing, not a debug word.
    //
    // Deciding at the end of the run whether that match ever arrived needs the same test the
    // validator used to decide it was absent: contains_match, which probes the whole dedup chain
    // AND compares the record. A hash alone cannot be re-tested that way -- a colliding different
    // match occupying the probe slot reads as "arrived" -- and the state/rule pair a debug word
    // carries is recoverable from the record anyway (source_state, core->rule_index).
    ConcurrentMap<uint64_t, const MatchRecord*> missing_match_hashes_{4096};
    std::atomic<size_t> late_arrivals_{0};  // Matches that arrived after validation

    // Job system
    std::unique_ptr<job_system::JobSystem<EvolutionJobType>> job_system_;
    size_t num_threads_{0};
    ExecutionMode mode_{ExecutionMode::Parallel};

    // A match task the step budget refused. Its state exists and its matches were never
    // computed, so this is precisely the work a continuation resumes -- and precisely the
    // frontier, since a state is refused a match task only for being one step past the budget.
    //
    // The state and its step, and nothing else. Carrying the delta context as well cost 30.1 MB
    // across the oracle corpus -- +9.3% of the arena -- on every run, continued or not. It also
    // buys nothing: a delta context says which edges are new relative to the parent so the scan
    // can skip what forwarding already offered, and a frontier state had no scan at all, so the
    // full scan is the correct resume rather than a fallback.
    struct DeferredMatch {
        StateId state;
        uint32_t step;
    };
    bool continuable_{false};
    LockFreeList<DeferredMatch> deferred_frontier_;

    // A rewrite the step budget refused. Match forwarding STORES a forwarded match on the child
    // and then asks for its rewrite; when the child sits one step past the budget that ask is
    // refused, and the stored match is not something the child's own match task will re-offer --
    // it deduplicates against what is already stored. So the deferred rewrites are their own
    // frontier, and without them a continuation reaches every state and misses transitions.
    struct DeferredRewrite {
        MatchRecord match;
        uint32_t step;
    };
    LockFreeList<DeferredRewrite> deferred_rewrites_;
    std::atomic<size_t> deferred_count_{0};

    void defer_match_task(StateId state, uint32_t step);
    void defer_rewrite_task(const MatchRecord& match, uint32_t step);

    // Evolution control
    std::atomic<bool> should_stop_{false};
    size_t max_steps_{0};
    size_t max_states_{0};
    size_t max_events_{0};

    // Pruning and random termination
    double exploration_probability_{1.0};          // Probability of exploring each new state (1.0 = always)
    size_t max_successor_states_per_parent_{0};    // Max children per parent state (0 = unlimited)
    size_t max_states_per_step_{0};                // Max new states per generation/step (0 = unlimited)
    uint64_t random_seed_{0};                      // Sampling RNG seed (0 = random_device each run)
    // Bumped at the start of every dataflow evolve() so each run re-seeds its
    // per-thread sampling RNGs from random_seed_, making repeated same-seed runs
    // draw identical exploration/shuffle streams.
    mutable std::atomic<uint64_t> sampling_generation_{0};

    // Exploration deduplication: only explore from canonical state representatives.
    // When enabled, states equivalent to already-seen states are created (with events)
    // but MATCH tasks are not spawned - we don't explore further from them.
    // This focuses compute on discovering new states rather than all transition paths.
    bool explore_from_canonical_states_only_{false};
    // Run-level notices (an option combination adjusted, an optimisation disabled). Never a
    // substitute for a result: the run still produces exactly what was requested.
    // Mutable because raise_worker_error() is const and RECORDS rather than throws for a
    // capacity limit: the run is over, nothing else observes this, and the alternative is making
    // the whole error path non-const to append one string.
    mutable std::vector<std::string> warnings_;
    bool quotient_initial_states_{false};

    // Per-parent successor count tracking (for max_successor_states_per_parent)
    static constexpr uint64_t SUCCESSOR_MAP_EMPTY = (1ULL << 62) + 500;
    static constexpr uint64_t SUCCESSOR_MAP_LOCKED = (1ULL << 62) + 501;
    ConcurrentMap<uint64_t, std::atomic<size_t>*, SUCCESSOR_MAP_EMPTY, SUCCESSOR_MAP_LOCKED> parent_successor_count_;

    // Per-step state count tracking (for max_states_per_step)
    static constexpr uint64_t STEP_MAP_EMPTY = (1ULL << 62) + 600;
    static constexpr uint64_t STEP_MAP_LOCKED = (1ULL << 62) + 601;
    ConcurrentMap<uint64_t, std::atomic<size_t>*, STEP_MAP_EMPTY, STEP_MAP_LOCKED> states_per_step_;

    // Per-state match-task join. See docs/ASYNC_SAMPLING_DESIGN.md §5.
    //
    // Matching one state is a tree of MATCH/SCAN/EXPAND tasks, so no single task sees all of
    // that state's matches. Anything that has to act on the state's matches AS A SET needs to
    // know when that tree has drained. Today that is the sampling spine: a state whose every
    // draw failed keeps its lowest-keyed own-found transition, and "every draw failed" is only
    // decidable once no more draws can arrive. A size-k cap per (state, rule) would be the
    // other such consumer and is #58's remaining half -- it must choose AT the drain, because
    // exactly-k needs the population, which is what makes it different from a cap by arrival
    // order.
    //
    // Two monotone counters and the task that equalises them is the drainer. This is a JOIN
    // over one state's own tasks, not a barrier: every other state runs through untouched, and
    // nothing global is consulted.
    static constexpr uint64_t MATCH_JOIN_EMPTY  = (1ULL << 62) + 700;
    static constexpr uint64_t MATCH_JOIN_LOCKED = (1ULL << 62) + 701;
    struct MatchJoin {
        std::atomic<size_t> pushed{0};
        std::atomic<size_t> completed{0};
        // Matches this state has accepted, post-dedup. The drain gate needs it to show the
        // drain fired after the last one rather than merely once.
        std::atomic<size_t> matches{0};
        // Sampling spine bookkeeping (transition_rate_ < 1 only). A fixed rate is a knife-edge:
        // below 1/branching the sampled evolution goes extinct before reaching depth. The spine
        // keeps the minimum-canonical-key OWN-FOUND transition alive when none of the state's
        // own draws passed.
        //
        // OWN-FOUND ONLY, deliberately: a state's own matching completes exactly at its drain,
        // so the minimum over own keys is a pure function of the state -- while the stored list
        // also holds forwarded arrivals, which race the drain, and a spine over the snapshot
        // made WHICH transition survived depend on the schedule (caught at 8 workers by
        // SamplingReproducibility). Forwarded draws neither mark nor force: the surviving set is
        // own-passers, plus forwarded-passers, plus the own-minimum when no own draw passed --
        // every term key-deterministic. A state with NO own-found matches has no spine and
        // relies on its forwarded draws; that hole is documented, and measured not to bite on
        // the corpus.
        std::atomic<uint32_t> own_spawned{0};
        std::atomic<uint64_t> own_min_key{~0ULL};
    };
    ConcurrentMap<uint64_t, MatchJoin*, MATCH_JOIN_EMPTY, MATCH_JOIN_LOCKED> match_join_;

    // Fires once per state, after that state's last match task. Set by tests today; nothing in
    // the shipping path installs one, and the spine reaches the drain through spine_at_drain
    // rather than through this hook.
    std::function<void(StateId, uint32_t)> on_state_matches_complete_;
    std::atomic<size_t> states_drained_{0};

    // Per-depth join, derived from the per-state one above. Flat and sized once per run: depth
    // is bounded by the step budget, so a map would buy nothing and cost a lookup on a path
    // every match task walks.
    //
    // `live` counts the tasks submitted to run at this depth that have not finished. A task at
    // depth d only ever submits at depths ABOVE d, so once d-1 has settled nothing can put work
    // at d -- which is what lets a depth be declared done without a barrier.
    //
    // Counting STATES that arrived would not do: match forwarding submits a rewrite task that
    // is booked against no state, and that rewrite creates a state and submits its match task,
    // so an arrival can appear at a depth with no live state accounting for it.
    // ONE CACHE LINE PER DEPTH. `live` is the engine's hottest counter -- every task increments
    // it when submitted and decrements it when done -- and FOUR depths fit in a 64-byte line at
    // its natural 16-byte size. Depths run CONCURRENTLY by construction (a task at depth d only
    // submits above d, which is what lets a depth settle without a barrier), so those four
    // counters are written by different threads at the same time and the line ping-pongs between
    // cores for no reason: nothing reads a neighbour's field.
    //
    // The whole array is sized once per run at max_steps + 2, so the padding costs 768 bytes on
    // a ten-step run against 192. There is no size at which this trade reverses.
    struct alignas(64) DepthJoin {
        std::atomic<size_t>  live{0};       // tasks submitted at this depth, minus those done
        std::atomic<uint8_t> complete{0};
    };
    std::vector<DepthJoin> depth_join_;
    // Depth 0 cannot settle before the roots are in: until then arrived is still moving and an
    // early match would fire the signal on an empty depth.
    std::atomic<bool> roots_seeded_{false};
    // Whether the arrival invariant this signal needs holds for this run. See the note on
    // set_on_depth_complete.
    bool depth_signal_available_{false};
    std::function<void(uint32_t)> on_depth_complete_;
    // Arrivals at a depth that had already settled. The signal's whole claim is that this
    // cannot happen, so it is counted rather than assumed: a non-zero value means a depth was
    // reported complete while work could still land in it.
    std::atomic<size_t> depth_late_arrivals_{0};

    void reset_depth_join();
    // Every task is booked at the depth it RUNS at: pushed before it can be seen, done after
    // every effect of it is visible -- the same discipline as the per-state join.
    void note_depth_task_pushed(uint32_t depth);
    void note_depth_task_done(uint32_t depth);
    void try_complete_depth(uint32_t depth);

    // Books one task against its depth however its function exits.
    class DepthTaskGuard {
    public:
        DepthTaskGuard(ParallelEvolutionEngine& engine, uint32_t depth)
            : engine_(engine), depth_(depth) {}
        ~DepthTaskGuard() { engine_.note_depth_task_done(depth_); }
        DepthTaskGuard(const DepthTaskGuard&) = delete;
        DepthTaskGuard& operator=(const DepthTaskGuard&) = delete;
    private:
        ParallelEvolutionEngine& engine_;
        uint32_t depth_;
    };

    // Statistics (atomics for thread-safety)
    // Four hot counters in 32 bytes would share a line; one each (see EvolutionStats).
    alignas(64) std::atomic<size_t> total_matches_found_{0};
    alignas(64) std::atomic<size_t> total_rewrites_{0};
    alignas(64) std::atomic<size_t> total_events_{0};
    alignas(64) std::atomic<size_t> rejected_duplicates_{0};

    // Evolution statistics
    EvolutionStats stats_;

public:
    ParallelEvolutionEngine()
        : hg_(nullptr)
        , rewriter_(nullptr)
    {}

    explicit ParallelEvolutionEngine(Hypergraph* hg, size_t num_threads = 0,
                                     ExecutionMode mode = ExecutionMode::Parallel);

    ~ParallelEvolutionEngine();

    // Non-copyable
    ParallelEvolutionEngine(const ParallelEvolutionEngine&) = delete;
    ParallelEvolutionEngine& operator=(const ParallelEvolutionEngine&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    void add_rule(const RewriteRule& rule) {
        rules_.push_back(rule);
        // Finalize the rule's matching data at the single point of registration:
        // join order plus the per-edge signature / compatible-signature caches the
        // matcher reads (lhs_sig / lhs_cache). A hand-built RewriteRule need only
        // populate lhs/rhs and the edge counts; compute_var_counts derives the rest
        // from them. Idempotent for callers that already computed it.
        rules_.back().compute_var_counts();
    }

    void set_max_steps(size_t max) { max_steps_ = max; }
    // The depth this run is budgeted to, which evolve_more RAISES rather than replaces. A caller
    // that continues an exploration and then reports on it needs the accumulated depth, not the
    // increment it just asked for: a step index counted from the end ("the final step") is
    // defined against this total.
    size_t max_steps() const { return max_steps_; }
    void set_max_states(size_t max) { max_states_ = max; }
    void set_max_events(size_t max) { max_events_ = max; }
    void set_match_forwarding(bool enable) { enable_match_forwarding_ = enable; }
    void set_batched_matching(bool enable) { batched_matching_ = enable; }
    void set_validate_match_forwarding(bool enable) { validate_match_forwarding_ = enable; }

    // Enable online transitive reduction for causal edges (Goranci algorithm)
    // When enabled, redundant causal edges are filtered out at insertion time.
    void set_transitive_reduction(bool enable) {
        if (hg_) hg_->causal_graph().set_transitive_reduction(enable);
    }

    // Enable genesis events for initial states.
    // When enabled, a synthetic event is created for each initial state that
    // "produces" all edges in that state. This allows causal edges to be
    // tracked from the initial state's edges to events that consume them.
    void set_genesis_events(bool enable) { enable_genesis_events_ = enable; }
    bool genesis_events() const { return enable_genesis_events_; }

    // Enable task-based matching (HGMatch SCAN→EXPAND join model)
    // When enabled, pattern matching spawns fine-grained tasks for better parallelism.
    // When disabled (default), uses synchronous find_matches() within MATCH task.
    void set_task_based_matching(bool enable) { task_based_matching_ = enable; }
    bool task_based_matching() const { return task_based_matching_; }

    // Pruning options.
    //
    // Sampling / determinism contract:
    //  - Hard caps (max_successor_states_per_parent, max_states_per_step) bound the
    //    number of states kept DETERMINISTICALLY (atomic counters). WHICH states are
    //    kept when a cap binds is scheduling-order dependent, so the retained subset
    //    can differ run-to-run even though the count is capped.
    //  - exploration_probability and the uniform-random evolve path are Monte-Carlo
    //    sampling of the multiway system. Both draw from RNGs seeded from
    //    random_seed_: 0 (default) draws a fresh random_device seed each run (every
    //    run differs); set_random_seed(nonzero) fixes the seed so BOTH the
    //    ExplorationProbability draw and the uniform-random path are reproducible
    //    run-to-run with a single thread. With multiple threads each thread's stream
    //    is still deterministic, but job scheduling perturbs which successor gets
    //    which draw, so multi-thread runs are not bit-reproducible.
    //  - Rule application order is shuffled per task for fairness, but that is
    //    order-only and does NOT change the canonical result.
    void set_exploration_probability(double p) {
        exploration_probability_ = std::clamp(p, 0.0, 1.0);
    }
    void set_max_successor_states_per_parent(size_t max) {
        max_successor_states_per_parent_ = max;
    }
    void set_max_states_per_step(size_t max) {
        max_states_per_step_ = max;
    }
    // Seed for the sampling RNGs (both the ExplorationProbability draw and the
    // uniform-random evolve path). 0 (default) draws a fresh random_device seed each
    // run; nonzero makes the sample reproducible run-to-run on a single thread.
    void set_random_seed(uint64_t seed) {
        random_seed_ = seed;
    }
    // Keep k of each state's matches, chosen uniformly at random from ALL of that state's
    // matches, and rewrite only those. 0 disables sampling entirely.
    //
    // The population is one state's matches, and saying so is the point: it is a population
    // that completes locally, so the sample can be finalised the moment that state's match tree
    // drains, with no step barrier and no other state waiting. Selection is Algorithm R keyed
    // on the stream position, so the retained set is the same whatever the schedule and
    // whichever worker sees which match.
    //
    // Keep each transition with probability q, drawn independently per transition. 1.0 keeps
    // everything. This is the general sampler.
    //
    // Independent thinning is what makes the sample representative: the result is a
    // sub-branching-process whose offspring distribution is the thinned original, so branching
    // shape and variance survive. It also needs no population and no completion, which is why
    // it is eager, needs no join, and applies identically to a match this state discovered and
    // one forwarded to it from an ancestor -- the two failures that a per-state count could not
    // survive (docs/ASYNC_SAMPLING_DESIGN.md §3a).
    //
    // The draw is keyed on the transition's identity rather than on a worker's RNG, so it does
    // not depend on WHICH thread drew or on the order matches arrived. It does still depend on
    // the raw source-state id, which work-stealing assigns nondeterministically, so the sample
    // is reproducible run to run at one worker and not yet across thread counts. Keying on the
    // canonical transition identity is what closes that.
    //
    // Each path of length L survives with probability q^L, so deep structure thins faster than
    // shallow. That is inherent to online thinning -- a deep path cannot be kept without its
    // prefix -- and the answer is a depth-dependent q, not a different mechanism.
    void set_transition_rate(double q) { transition_rate_ = q; }
    double transition_rate() const { return transition_rate_; }
    void set_rule_weights(std::vector<double> w) { rule_weights_ = std::move(w); }
    const std::vector<double>& rule_weights() const { return rule_weights_; }

    // The rate THIS rule's transitions are drawn at. Clamped, because a weight is a caller's
    // number and a rate outside [0,1] is not a probability.
    double rate_for_rule(uint16_t rule) const {
        double w = 1.0;
        if (rule < rule_weights_.size()) w = rule_weights_[rule];
        const double r = transition_rate_ * w;
        return r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
    }

    // Whether ANY draw can fail. The draw sites used to test transition_rate_ < 1.0 directly,
    // which would skip sampling entirely for a caller who left the rate at 1 and weighted a
    // single rule to zero.
    bool sampling_active() const {
        if (transition_rate_ < 1.0) return true;
        for (double w : rule_weights_) if (w < 1.0) return true;
        return false;
    }
    // The spine's per-seed ordering of a state's own transitions: splitmix of (key, seed).
    // Measured dead ends recorded in the probe: extra coins per arrival depth and per arriving
    // ancestor class both left the union-recovery curve unchanged, because recovery is limited
    // by REACHABILITY (a coin cannot fire on a transition whose source was never created), not
    // by per-transition survival. Seeding the spine attacks reachability directly: each seed
    // keeps a different one-per-state skeleton.
    uint64_t spine_rank(uint64_t canonical_key) const {
        uint64_t x = canonical_key ^ (random_seed_ * 0x9E3779B97F4A7C15ULL) ^ 0xA5A5A5A5A5A5A5A5ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    // Called once per state, after that state's last match task and before any of its matches
    // could be superseded. Anything keyed on "the matches of one state" as a set belongs here.
    // Set it before evolve(); it runs on a worker thread.
    void set_on_state_matches_complete(std::function<void(StateId, uint32_t)> cb) {
        on_state_matches_complete_ = std::move(cb);
    }

    // Called once per DEPTH, after every state that entered that depth has drained and every
    // shallower depth has done the same -- so nothing can still arrive there. Fires on a worker
    // thread, in depth order, without any barrier: it is derived from the per-state drain
    // above, not from a wait.
    //
    // AVAILABLE UNDER FULL CAPTURE ONLY, and depth_signal_available() says so before a run
    // commits to it. Under quotient exploration a child is submitted at its parent's LIVE
    // MINIMUM depth plus one, and a shorter path found later lowers that minimum -- so a task
    // running at a deep step can put a new state at a shallow depth, after that depth's
    // predecessor has already settled. The invariant that makes this signal sound is exactly
    // the one relaxation breaks, and repairing it would need "no live task can still submit
    // here", which is the global barrier this exists to avoid. Quotient callers have the
    // per-state drain and the run's own completion.
    void set_on_depth_complete(std::function<void(uint32_t)> cb) {
        on_depth_complete_ = std::move(cb);
    }
    // Whether the depth signal will fire for the configuration this engine is set to. False
    // under quotient exploration; see set_on_depth_complete.
    bool depth_signal_available() const { return !explore_from_canonical_states_only_; }
    // States that arrived at a depth already reported complete. Must be zero.
    size_t depth_late_arrivals() const {
        return depth_late_arrivals_.load(std::memory_order_relaxed);
    }
    size_t states_drained() const { return states_drained_.load(std::memory_order_relaxed); }

    // Matches this state has accepted so far. Read inside the drain callback it is that state's
    // final count, which is what makes "the drain fired after the last match" checkable.
    size_t matches_found_for_state(StateId state) const;

    // Quotient exploration: expand each canonical state exactly once, at the
    // shortest depth that reaches it (maintained by lock-free depth relaxation
    // over the canonical transitions), so a run costs the canonical closure
    // rather than the provenance count. Deterministic: the expanded set and the
    // (input, output, rule) transition multiset depend only on the graph, not on
    // scheduling or rule order. Causal/branchial edges are recorded only for the
    // expanded representatives; the full expansion's exact multisets are
    // reconstructed offline from this skeleton together with per-state
    // multiplicities (tools/quotient_reconstruction_probe.cpp). Requires
    // StateCanonicalizationMode::Full. Default false: expand every provenance,
    // the reference/MultiwayReference.wl semantics with exact online causal and
    // branchial tracking.
    // When true, isomorphic initial states collapse to one canonical root under
    // explore_from_canonical_states_only. Default false: each provided initial
    // state is a distinct entry point (reference MultiwaySystem semantics).
    void set_quotient_initial_states(bool enable) { quotient_initial_states_ = enable; }
    bool quotient_initial_states() const { return quotient_initial_states_; }

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
    size_t validations_performed() const { return validations_performed_.load(); }
    size_t missing_owed_by_forwarding() const { return missing_owed_by_forwarding_.load(); }
    size_t missing_owed_by_delta() const { return missing_owed_by_delta_.load(); }
    size_t draws_taken() const { return draws_taken_.load(); }
    size_t draws_survived() const { return draws_survived_.load(); }
    size_t draws_at_site(int i) const { return draws_by_site_[i].load(); }
    size_t late_arrivals() const { return late_arrivals_.load(); }
    // Matches recorded absent by the validator and STILL absent when the run ended.
    //
    // Tested with contains_match, the validator's own membership test: it probes the whole dedup
    // chain and compares the RECORD. Testing probe slot 0 for the key alone -- which this did --
    // reports a colliding different match as an arrival and silently under-counts.
    size_t still_missing() const {
        size_t count = 0;
        missing_match_hashes_.for_each([&](uint64_t h, const MatchRecord* rec) {
            if (rec && !contains_match(h, *rec)) ++count;
        });
        return count;
    }

    // Every still-absent match, with the state and rule read off the record itself.
    template <typename F>
    void for_each_still_missing(F&& f) const {
        missing_match_hashes_.for_each([&](uint64_t h, const MatchRecord* rec) {
            if (rec && !contains_match(h, *rec))
                f(h, rec->source_state, rec->core ? rec->core->rule_index : uint16_t{0xFFFF},
                  rec->num_edges());
        });
    }

    size_t num_threads() const { return num_threads_; }
    // Serial runs execute every job on the calling thread and spawn nothing.
    bool is_serial() const { return mode_ == ExecutionMode::Serial; }
    size_t num_states() const { return hg_ ? hg_->num_states() : 0; }
    size_t num_canonical_states() const { return hg_ ? hg_->num_canonical_states() : 0; }
    size_t num_events() const { return hg_ ? hg_->num_events() : 0; }
    size_t num_causal_edges() const { return hg_ ? hg_->causal_graph().num_causal_event_pairs() : 0; }
    size_t num_branchial_edges() const { return hg_ ? hg_->causal_graph().num_branchial_edges() : 0; }
    size_t num_redundant_edges_skipped() const {
        return hg_ ? hg_->causal_graph().num_redundant_edges_skipped() : 0;
    }

    const EvolutionStats& stats() const { return stats_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

    // Request early termination of evolution
    // This is non-blocking; evolution will stop as soon as currently queued jobs check the flag.
    // Call wait_for_idle() after request_stop() to ensure all jobs have completed.
    // Relaxed, deliberately, and matching the ~22 in-loop checks on the task paths.
    // Nothing is PUBLISHED through this flag -- it carries no data, it only asks the
    // workers to stop starting work -- so there is no writer-side state for an acquire
    // to pick up, and an acquire/release pair here would order nothing while adding a
    // fence to every one of those checks. Coherence alone guarantees the store becomes
    // visible; observing it late costs at most one more task.
    void request_stop() {
        should_stop_.store(true, std::memory_order_relaxed);
    }

    // Check if stop has been requested
    bool stop_requested() const {
        return should_stop_.load(std::memory_order_relaxed);
    }

    // Error latched by a worker during the last evolve(), or None.
    job_system::ErrorType last_error() const {
        return job_system_ ? job_system_->get_error_type() : job_system::ErrorType::None;
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

    // `steps` is the exact generation budget: events fire at generations 1..steps, and
    // steps == 0 yields the initial state alone. A run to closure (terminating rule
    // systems only) passes SIZE_MAX.
    void evolve(const std::vector<std::vector<VertexId>>& initial_edges, size_t steps);

    // Record what a continuation would resume from. Off by default: the frontier costs 12.5 MB
    // across the oracle corpus, +3.9% of the arena, and a run that is never continued pays all
    // of it for nothing. Set before evolve().
    void set_continuable(bool on) { continuable_ = on; }
    bool continuable() const { return continuable_; }

    // Carry the SAME run `additional_steps` further, from the frontier where the budget stopped
    // it. Equivalent to having asked for the total in the first place; the states, events and
    // relations already built are kept rather than recomputed.
    //
    // Throws unless the run was made continuable before evolve(): without the frontier there is
    // nothing to resume from, and returning the unchanged graph would be a wrong answer that
    // looks like a converged one.
    // `only_from` STEERS the continuation: when non-null, only those frontier states are
    // expanded and every other frontier entry is PUT BACK, so a later call can still resume it.
    // Retention is the point -- dropping the unselected entries would make "explore this branch"
    // mean "abandon the others", and a caller comparing a steered exploration against an
    // exhaustive one would find states missing with nothing to say why. A deferred rewrite is
    // selected by the state its match sits on, since submitting it while retaining that state
    // would half-expand a branch the caller asked to leave alone.
    void evolve_more(size_t additional_steps,
                     const std::unordered_set<StateId>* only_from = nullptr);

    // The states a continuation would resume from, with the step each is waiting at. Read
    // between runs (no worker is running), which is also the only time it is meaningful: during
    // a run the frontier is the set of refusals so far, not a boundary.
    std::vector<std::pair<StateId, uint32_t>> frontier() const;

    // Overload for multiple initial states (without abort callback)
    // Each initial state is evolved from independently, exploring the full multiway system
    void evolve(const std::vector<std::vector<std::vector<VertexId>>>& initial_states, size_t steps);

private:
    // Raise whatever a worker latched during the run. wait_for_completion() returns the
    // moment a worker latches an error, so tasks are still outstanding and the graph is
    // truncated; without this the run returns looking complete. Aborted is the caller's
    // own request, so it returns quietly with whatever was built.
    void raise_worker_error() const;

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
    size_t job_system_park_waits() const { return job_system_ ? job_system_->park_waits() : 0; }
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
    void store_match_for_state(StateId state, MatchRecord& match, bool with_fence = false);

    // Helper: Get or create the children list for a state (thread-safe)
    LockFreeList<ChildInfo>* get_or_create_state_children(StateId state);

    // Helper: Register a child with its parent (for push-based forwarding)
    void register_child_with_parent(StateId parent, StateId child,
                                    const EdgeId* consumed_edges, uint8_t num_consumed,
                                    uint32_t child_step = 0);

    // Which moment a push is issued from. The two are counted separately because they answer
    // different questions about whether the push has anything to do -- see EvolutionStats.
    enum class PushSite { Discovery, Forwarding };

    // Helper: Push a match to immediate children (single-level push)
    void push_match_to_children(StateId parent, const MatchRecord& match, uint32_t step,
                                PushSite site = PushSite::Forwarding);

    void push_match_to_children_impl(StateId parent, const MatchRecord& match, uint32_t step,
                                     PushSite site);

    // Diagnostic: record when a forwarded match had been flagged as missing during
    // validation (validate_match_forwarding_), meaning it arrived after the check.
    void note_late_arrival(uint64_t match_hash);

    // Helper: Forward existing parent matches to a newly created child
    void forward_existing_parent_matches(
        StateId parent, StateId child,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        uint32_t step,
        SVec<MatchRecord>& batch
    );

    void forward_matches_from_single_ancestor(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t step,
        SVec<MatchRecord>& batch
    );

    void forward_matches_from_single_ancestor_impl(
        StateId ancestor, StateId child,
        const EdgeId* accumulated_consumed, uint8_t total_consumed,
        uint32_t step,
        SVec<MatchRecord>& batch
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

    // Apply one chunk's matches. Runs them back to back on this thread so the parent's data
    // stays hot for the whole range.
    void execute_expand_chunk(const ExpandChunk& chunk);
    void submit_expand_chunk(const ExpandChunk& chunk);

    // Hand a state's collected matches to chunk tasks. The first chunk runs INLINE on the
    // calling thread: most states have few matches, and for those this keeps the whole
    // expansion on the core that just built the match list, with no scheduling at all.
    void dispatch_expansion(StateId state, uint32_t step, const MatchRecord* matches,
                            size_t count);

    // Matches per chunk. Small enough that a wide state still spreads across workers, large
    // enough that the per-task cost is amortised over a run of children that share a parent.
    static constexpr size_t kExpandChunkSize = 16;
    void submit_scan_task(const ScanTaskData& data);

    void submit_expand_task(const ExpandTaskData& data);

    // Task Execution
    void execute_match_task(StateId state, uint32_t step, const MatchContext& ctx);
    void execute_scan_task(const ScanTaskData& data);
    void execute_expand_task(const ExpandTaskData& data);
    // Finish a complete match: dedup it, promote its core to the arena, and register it for
    // forwarding. Returns false when the match is a duplicate or a limit forbids it; on true,
    // `out` is ready to rewrite. The caller batches these rather than tasking each one -- the
    // work here is a hash insert and two list pushes, less than the cost of scheduling it.
    bool complete_match(const ExpandTaskData& data, MatchRecord& out);
    void execute_rewrite_task(const MatchRecord& match, uint32_t step);

    // Pruning helpers
    bool can_create_states_at_step(uint32_t step) const;
    bool can_have_more_children(StateId parent) const;
    // Ancestor-chain-scoped epoch for the pull-side retry; see the definition.
    static constexpr uint32_t kMaxAncestorHops = 1u << 20;   // guards a malformed parent cycle
    uintptr_t ancestor_match_epoch(StateId parent) const;

    // Per-state match join. The two ordering rules below are what make the drain exact, and
    // both are invariants rather than observations:
    //
    //   note_match_task_pushed  runs BEFORE the task it counts can be seen. Pushing after would
    //                           let the drain fire on a tree that is still growing, so the
    //                           decision would be taken over part of the population.
    //   note_match_task_done    runs AFTER every effect of its task is visible -- which is what
    //                           a scope guard buys, since the match tasks have many exits.
    //
    // Together they give exactly-one drain per state: when completed equals pushed, every
    // counted task has finished, and only a running task could push another.
    MatchJoin* match_join_for(StateId state);
    void note_match_task_pushed(StateId state);
    void note_match_task_done(StateId state, uint32_t step);

    // Isomorphism-invariant identity of the transition (state, rule, consumed edges), from the
    // shared EVENT_SIG_TRANSITION lattice point. Builds the state's canonical rank table on
    // first use, which is why it is not const.
    //
    // This is the key the sampler draws on, and every component of it has to be invariant: a
    // raw state id or an edge id is assigned by whichever worker got there first, so a draw
    // keyed on one selects a different subgraph every run and there is nothing to compare
    // against the unpruned evolution.
    uint64_t canonical_transition_key(StateId state, const MatchRecord& match);

    // The sampling draw with the spine guarantee: a passing draw records that its source state
    // has a survivor; a failing draw on a state that has already drained with NO survivor is
    // forced through instead (the late spine). See MatchJoin.
    bool transition_survives_spined(StateId source, uint64_t canonical_key, int site,
                                   uint16_t rule);

    // At a state's drain with sampling active and no survivor spawned: submit the stored match
    // with the minimum canonical transition key, so every reachable state keeps at least one
    // outgoing transition at any rate.
    void spine_at_drain(StateId state, uint32_t step, MatchJoin* join);

    // The transition-level draw, on the key above, so the same transition gets the same verdict
    // however the run is scheduled. Every acceptance point -- both discovery paths and both
    // forwarding paths -- consults exactly this.
    bool transition_survives(uint64_t transition_key, int site, uint16_t rule) const;


    // Books one match task's completion however its function exits.
    class MatchTaskGuard {
    public:
        MatchTaskGuard(ParallelEvolutionEngine& engine, StateId state, uint32_t step)
            : engine_(engine), state_(state), step_(step) {}
        ~MatchTaskGuard() { engine_.note_match_task_done(state_, step_); }
        MatchTaskGuard(const MatchTaskGuard&) = delete;
        MatchTaskGuard& operator=(const MatchTaskGuard&) = delete;
    private:
        ParallelEvolutionEngine& engine_;
        StateId state_;
        uint32_t step_;
    };

    static bool try_claim_budget(std::atomic<size_t>* counter, size_t limit);
    bool try_reserve_successor_slot(StateId parent);
    void release_successor_slot(StateId parent);
    bool try_reserve_step_slot(uint32_t step);
    void release_step_slot(uint32_t step);
    // Keep this state, with probability exploration_probability_, drawn on an
    // isomorphism-invariant key so the surviving set is the same at any worker count.
    //
    // WHICH key differs by exploration mode, and the difference is the semantics:
    //   quotient      the canonical state hash -- one draw per CLASS, so a class reached by N
    //                 transitions is kept with probability p rather than 1-(1-p)^N.
    //   full capture  the canonical key of the transition that CREATED the state. Raw states
    //                 stand in bijection with the events that make them, so a per-raw-state
    //                 coin and a per-transition coin are the same decision there -- which is
    //                 why ExplorationProbability is only a distinct knob under quotient.
    bool should_explore(uint64_t invariant_key) const;

    // Worker-RNG draw. Reachable only where no invariant key exists; the surviving set it
    // produces depends on which worker drew, so it cannot be reproduced across worker counts.
    bool should_explore();

    // Disables causal transitive reduction under quotient exploration; see the definition.
    void guard_quotient_transitive_reduction();

    // Applies the identity-mode rules at the top of every evolve():
    //   Positional + quotient -> quotient exploration disabled (the identity needs raw
    //   presentations) and a warning recorded; Automatic -> the reconstruction runs under BOTH
    //   exploration strategies, so their observables agree by construction.
    void configure_identity_and_quotient();


    // The per-thread sampling RNG lives in parallel_evolution.cpp, as a file-local function
    // taking the two values it reads -- sampling_generation_ and random_seed_. Declaring it
    // here instead would spell std::mt19937 in the header, and <random> is the second most
    // expensive standard header this engine's headers reach: dropping it and <sstream> from
    // the closure together is 196 ms off a 1198 ms translation unit.

    // Quotient exploration: canonical transitions discovered so far, parent canonical
    // state to child canonical state. Relaxing a state's depth walks these to push the
    // improvement to its descendants, which needs no re-matching because matches do not
    // depend on depth.
    SegmentedArray<LockFreeList<StateId>> canon_children_;

    // Push a depth improvement to the descendants of a canonical state. The state itself
    // has already been relaxed by the caller, which owns the match context needed to
    // expand it with forwarding; descendants are reached without one, and reaching them
    // at all is rare.
    void propagate_explore_depth(StateId canonical_state, uint32_t depth);

    // May this canonical state be expanded? The claim holds matching to once per canonical
    // state, and the exploration-probability draw is taken once, AT the claim, so a class
    // reached by N transitions is kept with probability p rather than 1-(1-p)^N -- matching the
    // GPU, which flips once per deduped state. A state whose draw fails stays claimed, so no
    // later transition re-flips for it.
    //
    // The DEPTH BUDGET is deliberately not tested here. The three callers stand in different
    // relations to it: the rewrite path and the relaxation walk each test it to choose between
    // expanding and deferring, and the continuation resubmits entries that failed it under a
    // bound that has since risen. What they must NOT differ on is the claim and the draw, which
    // decide which classes exist at all.
    bool claim_canonical_for_expansion(StateId canonical_state);

    // Bias mitigation: returns rule indices in shuffled order
    SVec<uint16_t> get_shuffled_rule_indices() const;
};

}  // namespace engine
}  // namespace HG_NAMESPACE