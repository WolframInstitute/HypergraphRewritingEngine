#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <set>
#include <utility>
#include <cstdint>
#include <cstring>
#include <vector>

#include "types.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"

// Visualization event emission (compiles to no-op when disabled)
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
#include <events/viz_event_sink.hpp>
#endif

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// CausalGraph
// =============================================================================
// Manages online computation of causal and branchial relationships between events.
//
// Key design principles:
// - Causal edges computed incrementally as events are created (not batch)
// - Uses rendezvous pattern for thread-safe producer/consumer discovery
// - Branchial edges computed by tracking events per canonical input state
// - All storage is lock-free append-only (no deletions during evolution)
//
// Rendezvous Pattern (for causal edges), symmetric two-set handshake:
// - Producer: PUSH itself into the edge's producer set, then SCAN the consumer set
//   → create an edge to every consumer.
// - Consumer: PUSH itself into the edge's consumer set, then SCAN the producer set
//   → create an edge from every producer.
// - A seq_cst fence sits between each side's push and scan, so the two form an SC
//   store-load pair: for any (producer, consumer) at least one side observes the
//   other (no store-buffer miss), hence no causal edge is dropped at any timing.
// - Both sides are append-only sets, so the emitted edge set is exactly
//   producers × consumers for the edge -- independent of thread schedule and of the
//   order producers/consumers arrive. Under quotient recurrence a canonical edge can
//   have several producers; keeping ALL of them (not an order-dependent first writer)
//   is what makes the causal attribution deterministic.
//
// Thread safety: Fully lock-free. Multiple events can be created concurrently.
//
// Memory: O(E) for edge causal info, O(causal_edges) for causal graph.
//         Arena-allocated, freed in bulk at end of evolution.

class CausalGraph {
    // The rendezvous is keyed by a CANONICAL EDGE KEY, not a raw edge id. Under quotient
    // the key is fnv(canonical_hash(state), edge_orbit_in_state) so every parent that
    // produces, and every event that consumes, the same canonical edge orbit meets at one
    // key -- the orbit is the only edge identity invariant across the different labelings
    // by which distinct parents reach one canonical state (a raw edge id or position is
    // not, once the state has automorphisms). Without quotient the key is just the raw
    // edge id, so isomorphic-but-distinct raw states keep disjoint causal edges. Keys are
    // arbitrary 64-bit values, so a hash map (not a dense array) backs each set.

    // Per-key producer set: the events that produced this canonical edge. A set because
    // under quotient recurrence one canonical edge is produced by several events; keeping
    // ALL of them (not an order-dependent first writer) is what makes attribution
    // deterministic. Symmetric with the consumer set below.
    // Storage keys on the raw 64-bit value (ConcurrentMap CASes an atomic integer key);
    // the strongly-typed CanonicalEdgeKey is unwrapped to its .value only at this boundary.
    // Sentinels sit in a reserved high band so that a non-quotient key -- a raw EdgeId,
    // which is 32-bit and includes 0 -- is always a valid key; causal_edge_keys masks its
    // orbit-hash keys into [0, 2^63) so they never collide with the band either.
    static constexpr uint64_t CE_MAP_EMPTY = 1ULL << 63;
    static constexpr uint64_t CE_MAP_LOCKED = (1ULL << 63) + 1;
    ConcurrentMap<uint64_t, LockFreeList<EventId>*, CE_MAP_EMPTY, CE_MAP_LOCKED> edge_producers_;

    // Per-key consumer set (appended when the canonical edge is consumed).
    ConcurrentMap<uint64_t, LockFreeList<EventId>*, CE_MAP_EMPTY, CE_MAP_LOCKED> edge_consumers_;

    // Per-state event lists for branchial tracking
    // Maps StateId -> list of events that have this state as input
    // Uses ConcurrentMap for thread-safe "get or create" semantics
    // Key: StateId (as uint64_t), Value: pointer to LockFreeList
    // Use special EMPTY_KEY and LOCKED_KEY outside valid StateId range (0 to 2^32-1)
    static constexpr uint64_t STATE_MAP_EMPTY = 1ULL << 62;
    static constexpr uint64_t STATE_MAP_LOCKED = (1ULL << 62) + 1;
    ConcurrentMap<uint64_t, LockFreeList<EventId>*, STATE_MAP_EMPTY, STATE_MAP_LOCKED> state_events_;

    // Per-(input state, consumed edge) inverted index: the events that consumed a
    // given edge at a given input state. Two events at the same input state are
    // branchially related iff they consumed a common edge, so scanning this bucket
    // finds all co-consumers in O(bucket size) instead of an O(events^2) pairwise
    // scan of the whole state's event list. Key = (state << 32) | edge.
    static constexpr uint64_t STATE_EDGE_MAP_EMPTY = (1ULL << 62) + 8;
    static constexpr uint64_t STATE_EDGE_MAP_LOCKED = (1ULL << 62) + 9;
    ConcurrentMap<uint64_t, LockFreeList<EventId>*, STATE_EDGE_MAP_EMPTY, STATE_EDGE_MAP_LOCKED> state_edge_events_;

    // Causal edges (producer -> consumer)
    LockFreeList<CausalEdge> causal_edges_;

    // Branchial edges (event <-> event with shared input)
    LockFreeList<BranchialEdge> branchial_edges_;

    // Deduplication map for causal edges: hash(producer, consumer, edge) -> true
    // The rendezvous pattern can cause both producer and consumer to add the same edge
    ConcurrentMap<uint64_t, uint8_t> seen_causal_triples_;

    // Deduplication map for causal event pairs: (producer << 32 | consumer) -> true
    // Counts unique event pairs that have a causal relationship.
    ConcurrentMap<uint64_t, uint8_t> seen_causal_event_pairs_;

    // Pack an ordered event pair into a map key. Both ids are offset by one so that the
    // self-loop (0,0) -- a real canonical self-loop under quotient, where distinct raw events
    // of one canonical type collapse to a single representative -- cannot land on the map's
    // EMPTY sentinel and be silently dropped. Packing stays injective: event ids are far below
    // 2^32-1, so the +1 cannot carry into the neighbouring field.
    static uint64_t causal_pair_key(EventId producer, EventId consumer) {
        return id_key(producer, consumer);
    }

    std::atomic<bool> ids_are_topological_{true};

    // Deduplication map for branchial edges: (e1 << 32 | e2) -> true
    ConcurrentMap<uint64_t, uint8_t> seen_branchial_pairs_;

    // =========================================================================
    // Online Transitive Reduction (backward-reachability oracle)
    // =========================================================================
    // An edge (p,c) is redundant iff c is already reachable from p via other kept
    // edges. Reachability is answered directly from the reduced predecessor adjacency
    // preds_ by an on-demand backward search from c toward p, so no per-event
    // descendant closure is materialized: memory is O(causal pairs), not the
    // O(events * depth) of a stored transitive closure, and no closure-propagation
    // union runs when an edge is kept.
    //
    // Query is_reachable(p, c): p reaches c iff some kept in-edge q->c has q == p or p
    // reaches q. A backward BFS from c over preds_ searches for p, visiting only nodes
    // with id >= id(p): event ids are monotonic along every causal edge (a producer's
    // event is created before its consumer's), so an ancestor has a strictly smaller
    // id than its descendant and any node with id < id(p) can neither be p nor have p
    // as an ancestor. This makes the search exact and self-pruning.
    //
    // Thread-safety: fully lock-free; the reduction is exact at any thread count. Two
    // rewriter invariants guarantee it:
    //   1. An edge's producer is set while the edge is still private to the rewrite
    //      that created it, before the state holding it is enqueued for rewriting. So
    //      no consumer can observe an edge whose producer is unset, and every causal
    //      edge is created by add_edge_consumer -- hence preds_[c] is written only by
    //      c's own thread (single writer per list).
    //   2. All in-edges of an event are added by that event's own thread, in
    //      descending producer-event-id order. Since a producer's event precedes its
    //      consumer's, descending id is reverse topological order: when p reaches x
    //      and both produce edges consumed by c, x->c is processed before p->c, so x
    //      is already in preds_[c] when the p->c redundancy search runs and the path
    //      p->..->x->c is found.
    // Ancestors of an event have completed their own causal registration before it
    // runs, so the backward search reads a settled sub-DAG above c.

    // Direct causal predecessors per consumer event (the producer events of the kept
    // causal edges it consumes), indexed by event id. This reduced adjacency preserves
    // reachability, so a backward search over it answers "does p reach c" exactly.
    // O(causal pairs) storage.
    SegmentedArray<LockFreeList<EventId>> preds_;

    // Whether online transitive reduction is enabled (the FFI turns this on by default).
    std::atomic<bool> transitive_reduction_enabled_{false};

    // Statistics for TR
    std::atomic<size_t> num_redundant_edges_skipped_{0};

    // Arena for allocations (supports concurrent access)
    ConcurrentHeterogeneousArena* arena_;

    // Statistics
    std::atomic<size_t> num_causal_edges_{0};        // Per-edge causal relationships
    std::atomic<size_t> num_causal_event_pairs_{0};  // Unique event pairs with a causal relationship
    std::atomic<size_t> num_branchial_edges_{0};

public:
    CausalGraph() : arena_(nullptr) {}

    explicit CausalGraph(ConcurrentHeterogeneousArena* arena) : arena_(arena) {}

    // Enable/disable online transitive reduction
    void set_transitive_reduction(bool enabled) {
        transitive_reduction_enabled_.store(enabled, std::memory_order_relaxed);
    }

    bool transitive_reduction_enabled() const {
        return transitive_reduction_enabled_.load(std::memory_order_relaxed);
    }

    // Set arena (for deferred initialization). Also re-homes every member map's table
    // storage onto the arena (no malloc). Single-threaded setup only, before any
    // event is registered.
    void set_arena(ConcurrentHeterogeneousArena* arena) {
        arena_ = arena;
        seen_causal_triples_.set_arena(arena);
        seen_causal_event_pairs_.set_arena(arena);
        seen_branchial_pairs_.set_arena(arena);
        state_events_.set_arena(arena);
        state_edge_events_.set_arena(arena);
        edge_producers_.set_arena(arena);
        edge_consumers_.set_arena(arena);
    }

    // =========================================================================
    // Edge Causal Tracking
    // =========================================================================

    // Get or create the producer set for a canonical edge key (thread-safe)
    LockFreeList<EventId>* get_or_create_edge_producers(CanonicalEdgeKey edge_key);

    // Get or create the consumer set for a canonical edge key (thread-safe)
    LockFreeList<EventId>* get_or_create_edge_consumers(CanonicalEdgeKey edge_key);

    // Redundancy check: is consumer already reachable from producer via kept edges?
    // Backward search over preds_ (the reduced adjacency); exact and lock-free.
    bool is_reachable(EventId producer, EventId consumer) const;

    // Is the reduction computed on read rather than maintained incrementally? True exactly
    // when the reduction is on and the id assignment does not support the incremental rule.
    bool reduces_on_read() const {
        return transitive_reduction_enabled_.load(std::memory_order_relaxed) &&
               !ids_are_topological_.load(std::memory_order_relaxed);
    }

    // The reduced pair set, computed from the stored relation. Unique for a given relation, so
    // the answer does not depend on the schedule that produced it.
    std::set<std::pair<EventId, EventId>> reduced_pairs() const;

    // WHETHER EVENT IDS INCREASE ALONG EVERY CAUSAL EDGE. True for full capture, which mints
    // an event only after the events that produced its inputs; is_reachable uses it to skip
    // the walk for producer >= consumer and to prune it to ids >= producer.
    //
    // FALSE for the quotient reconstruction: it emits between CANONICAL event ids, assigned
    // first-writer-wins, which are not monotonic under recurrence -- measured on chain6,
    // producer 9 -> consumer 8 among others. With the assumption false the pruned walk misses
    // paths that exist and the reduction over-keeps, so it runs unpruned instead.
    void set_ids_are_topological(bool on) {
        ids_are_topological_.store(on, std::memory_order_relaxed);
    }
    bool ids_are_topological() const {
        return ids_are_topological_.load(std::memory_order_relaxed);
    }

    // Get or create the event list for a state (thread-safe)
    LockFreeList<EventId>* get_or_create_state_events(StateId state);

    // Get or create the co-consumer list for a (state, consumed edge) bucket
    LockFreeList<EventId>* get_or_create_state_edge_events(StateId state, EdgeId edge);

    // Called when a canonical edge (edge_key) is produced by an event: adds the event to
    // the edge's producer set and rendezvous-emits causal edges to all consumers. raw_edge
    // is the concrete edge id recorded on the CausalEdge for viz/debug. Returns true if
    // this producer was newly added (not already in the set).
    bool set_edge_producer(CanonicalEdgeKey edge_key, EventId producer, EdgeId raw_edge);

    // Called when a canonical edge (edge_key) is consumed by an event.
    void add_edge_consumer(CanonicalEdgeKey edge_key, EventId consumer, EdgeId raw_edge);

    // Carry an edge's producer set across a rewrite it survives: register every producer
    // of from_key as a producer of to_key (rendezvous-emitting to to_key's consumers).
    void propagate_producers(CanonicalEdgeKey from_key, CanonicalEdgeKey to_key, EdgeId raw_edge);

    // A representative producer of a canonical edge -- the largest producer event id in
    // the set (the closest producer, for the reverse-topological TR insertion heuristic),
    // or INVALID_ID if the edge has no producer. Deterministic: a function of the
    // order-independent producer set, not of arrival order.
    EventId get_edge_producer(CanonicalEdgeKey edge_key) const;

    // =========================================================================
    // Branchial Tracking
    // =========================================================================

    // More detailed branchial check with access to event data
    //
    // Thread-safety: Uses "add first, check all, deduplicate on insert" pattern.
    // The per-state event list, read back through for_each_state_events. It is what an
    // all-siblings view of the branchial state graph is built from -- every pair of output
    // states of one input state, with no overlap test -- and that view is the only reader.
    // Recorded separately from the pair relation below because the two answer different
    // questions and a caller asking for one should not pay for the other.
    void record_state_event(EventId event, StateId input_state) {
        get_or_create_state_events(input_state)->push(event, *arena_);
    }

    // The branchial PAIR relation: two events branch iff they consumed a common edge at a
    // shared input state.
    //
    // Both events in a pair may detect the overlap and try to add the edge, but only one
    // succeeds due to ConcurrentMap deduplication.
    void record_branchial_overlaps(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed
    ) {
        // Inverted index: for each consumed edge, publish this event into that edge's
        // co-consumer bucket, then scan the same bucket. Per bucket this is
        // "add first, then check", so both events of a pair see each other (whichever
        // scans the shared bucket second finds the first); seen_branchial_pairs_
        // dedups the double add. Work is proportional to the actual number of
        // co-consumers, replacing the O(events^2) pairwise scan of the whole state's
        // event list (one bucket lookup per consumed edge, not two).
        for (uint8_t i = 0; i < num_consumed; ++i) {
            EdgeId shared = consumed_edges[i];
            LockFreeList<EventId>* bucket = get_or_create_state_edge_events(input_state, shared);
            bucket->push(event, *arena_);
            bucket->for_each([&](EventId other_event) {
                if (other_event == event) return;  // Skip self
                EventId e1 = std::min(event, other_event);
                EventId e2 = std::max(event, other_event);
                auto [_, inserted] = seen_branchial_pairs_.insert_if_absent(id_key(e1, e2), true);
                if (inserted) {
                    add_branchial_edge(e1, e2, shared);
                }
            });
        }
    }

    // =========================================================================
    // Graph Access
    // =========================================================================

    // Add a causal edge (producer -> consumer) with deduplication and optional TR
    void add_causal_edge(EventId producer, EventId consumer, EdgeId edge);

    // Record a kept edge's producer in the reduced predecessor adjacency preds_.
    void record_reduced_edge(EventId producer, EventId consumer);

    // Add a branchial edge (event <-> event)
    void add_branchial_edge(EventId e1, EventId e2, EdgeId shared);

    // Iterate over causal edges
    template<typename Visitor>
    void for_each_causal_edge(Visitor&& visit) const {
        if (reduces_on_read()) {
            const auto keep = reduced_pairs();
            causal_edges_.for_each([&](const CausalEdge& edge) {
                if (keep.count({edge.producer, edge.consumer})) visit(edge);
            });
            return;
        }
        causal_edges_.for_each([&](const CausalEdge& edge) { visit(edge); });
    }

    // Iterate over branchial edges
    template<typename Visitor>
    void for_each_branchial_edge(Visitor&& visit) const {
        branchial_edges_.for_each([&](const BranchialEdge& edge) {
            visit(edge);
        });
    }

    // Statistics
    size_t num_causal_edges() const {
        // Reducing on read means the stored relation is the FULL one, so the live edge count is
        // what the filtered iteration yields, not the admitted-triple counter.
        if (reduces_on_read()) {
            size_t n = 0;
            for_each_causal_edge([&](const CausalEdge&) { ++n; });
            return n;
        }
        return num_causal_edges_.load(std::memory_order_relaxed);
    }

    // Number of unique event pairs with a causal relationship.
    size_t num_causal_event_pairs() const {
        if (reduces_on_read()) return reduced_pairs().size();
        return num_causal_event_pairs_.load(std::memory_order_relaxed);
    }

    // Pairs the branchial dedup actually claimed. add_branchial_edge is reached ONLY on a
    // winning claim, so this must equal num_branchial_edges() at every point in a run.
    //
    // The two counts are maintained by different mechanisms -- one is a map's occupancy, the
    // other a counter incremented by the winner -- so equality is a real check on the map's
    // exactly-once contract rather than a restatement of it. A duplicate admitted under
    // contention shows up here on the run that produced it, with that run's parameters, instead
    // of only as two runs disagreeing afterwards: the branchial graph went non-deterministic at
    // 8 threads once in 24 runs (30064 edges against 30063, states/events/causal identical),
    // and a spread across runs cannot say WHICH run was wrong or why.
    size_t num_branchial_pairs_claimed() const {
        return seen_branchial_pairs_.count_unique();
    }

    size_t num_branchial_edges() const {
        return num_branchial_edges_.load(std::memory_order_relaxed);
    }

    // Number of redundant causal edges skipped by online TR
    size_t num_redundant_edges_skipped() const {
        return num_redundant_edges_skipped_.load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Utility
    // =========================================================================

    // Collect causal edges into vector (for export/testing)
    std::vector<CausalEdge> get_causal_edges() const;

    // Collect branchial edges into vector (for export/testing)
    std::vector<BranchialEdge> get_branchial_edges() const;

    // Iterate over all (input_state -> events) mappings
    // Visitor signature: void(StateId input_state, LockFreeList<EventId>* event_list)
    // Caller can use event_list->for_each() to iterate events
    template<typename Visitor>
    void for_each_state_events(Visitor&& visit) const {
        state_events_.for_each([&](uint64_t state_key, LockFreeList<EventId>* event_list) {
            visit(static_cast<StateId>(id_from_key(state_key)), event_list);
        });
    }

};

}  // namespace engine
}  // namespace HG_NAMESPACE