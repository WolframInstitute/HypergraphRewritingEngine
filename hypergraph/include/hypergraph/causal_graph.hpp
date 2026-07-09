#pragma once

#include <atomic>
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

namespace hypergraph {

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
// Rendezvous Pattern (for causal edges):
// - Producer: WRITE producer, then READ consumers → create edges to consumers
// - Consumer: WRITE to consumers, then READ producer → create edge from producer
// - Guarantees: at least one side sees the other, no edges missed
//
// Thread safety: Fully lock-free. Multiple events can be created concurrently.
//
// Memory: O(E) for edge causal info, O(causal_edges) for causal graph.
//         Arena-allocated, freed in bulk at end of evolution.

class CausalGraph {
    // Per-edge producer tracking (set once when edge created). Edge ids are dense and
    // sequential, so a direct-indexed array beats a hash map: no hashing, no per-edge
    // heap node, contiguous storage. get_or_default(edge) lazily materializes the slot
    // (default producer == INVALID_ID); the reader-visible size grows with it.
    SegmentedArray<EdgeCausalInfo> edge_producers_;

    // Per-edge consumer lists (appended when edge consumed), likewise indexed by edge id.
    SegmentedArray<LockFreeList<EventId>> edge_consumers_;

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
    ConcurrentMap<uint64_t, bool> seen_causal_triples_;

    // Deduplication map for causal event pairs: (producer << 32 | consumer) -> true
    // Counts unique event pairs that have a causal relationship.
    ConcurrentMap<uint64_t, bool> seen_causal_event_pairs_;

    // Deduplication map for branchial edges: (e1 << 32 | e2) -> true
    ConcurrentMap<uint64_t, bool> seen_branchial_pairs_;

    // =========================================================================
    // Online Transitive Reduction (Goranci Algorithm)
    // =========================================================================
    // Maintains Desc[u] (events reachable from u) and Anc[u] (events reaching u)
    // for O(1) redundancy checking. Edge (p,c) is redundant iff c ∈ Desc[p].
    //
    // When edge (p,c) is added (if not redundant):
    //   - For all a ∈ {p} ∪ Anc[p]: Desc[a] ∪= {c} ∪ Desc[c]
    //   - For all d ∈ {c} ∪ Desc[c]: Anc[d] ∪= {p} ∪ Anc[p]
    //
    // Thread-safety: concurrent sets with atomic operations. The reduction is
    // exact -- a redundant edge is never emitted, at any thread count. Two
    // invariants established by the rewriter guarantee it:
    //   1. An edge's producer is set while the edge is still private to the
    //      rewrite that created it, before the state holding it is enqueued for
    //      rewriting. So no consumer can observe an edge whose producer is unset,
    //      and every causal edge is created by add_edge_consumer.
    //   2. All in-edges of an event are added by that event's own thread, in
    //      descending producer-event-id order. Event ids are monotonic and a
    //      producer's event is created before its consumer's, so descending id is
    //      reverse topological order: when p reaches x and both produce edges
    //      consumed by c, x->c is added before p->c, placing c in Desc[p] so the
    //      p->c redundancy check finds it.
    // Ancestors of an event have completed their own causal registration before it
    // runs, so the Desc lookup backing the check reads a settled closure.

    // Desc[u] = all events reachable from u (transitive closure)
    // Anc[u] = all events that can reach u (transitive closure)
    // Uses ConcurrentMap<EventId, bool> as lock-free set (key present = in set)
    static constexpr uint64_t DESC_ANC_SET_EMPTY = (1ULL << 62) + 4;
    static constexpr uint64_t DESC_ANC_SET_LOCKED = (1ULL << 62) + 5;
    using DescAncSet = ConcurrentMap<uint64_t, bool, DESC_ANC_SET_EMPTY, DESC_ANC_SET_LOCKED>;

    static constexpr uint64_t DESC_ANC_EMPTY = (1ULL << 62) + 6;
    static constexpr uint64_t DESC_ANC_LOCKED = (1ULL << 62) + 7;
    ConcurrentMap<uint64_t, DescAncSet*, DESC_ANC_EMPTY, DESC_ANC_LOCKED> desc_;
    ConcurrentMap<uint64_t, DescAncSet*, DESC_ANC_EMPTY, DESC_ANC_LOCKED> anc_;

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

    // Set arena (for deferred initialization)
    void set_arena(ConcurrentHeterogeneousArena* arena) {
        arena_ = arena;
    }

    // =========================================================================
    // Edge Causal Tracking
    // =========================================================================

    // Get or create producer info for an edge (thread-safe)
    EdgeCausalInfo* get_or_create_edge_producer(EdgeId edge);

    // Get or create consumer list for an edge (thread-safe)
    LockFreeList<EventId>* get_or_create_edge_consumers(EdgeId edge);

    // Get or create Desc set for an event (Goranci algorithm)
    DescAncSet* get_or_create_desc(EventId event);

    // Get or create Anc set for an event (Goranci algorithm)
    DescAncSet* get_or_create_anc(EventId event);

    // O(1) redundancy check: is consumer reachable from producer?
    bool is_reachable_via_desc(EventId producer, EventId consumer) const;

    // Get or create the event list for a state (thread-safe)
    LockFreeList<EventId>* get_or_create_state_events(StateId state);

    // Get or create the co-consumer list for a (state, consumed edge) bucket
    LockFreeList<EventId>* get_or_create_state_edge_events(StateId state, EdgeId edge);

    // Called when an edge is produced by an event
    bool set_edge_producer(EdgeId edge, EventId producer);

    // Called when an edge is consumed by an event
    void add_edge_consumer(EdgeId edge, EventId consumer);

    // Get producer of an edge (may be INVALID_ID for initial edges)
    EventId get_edge_producer(EdgeId edge) const;

    // =========================================================================
    // Branchial Tracking
    // =========================================================================

    // More detailed branchial check with access to event data
    //
    // Thread-safety: Uses "add first, check all, deduplicate on insert" pattern.
    // Both events in a pair may detect the overlap and try to add the edge,
    // but only one succeeds due to ConcurrentMap deduplication.
    template<typename GetEventConsumedEdges>
    void register_event_from_state_with_overlap_check(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed,
        GetEventConsumedEdges&& /* get_consumed - unused: the index identifies
                                   co-consumers directly, no per-event edge lookup */
    ) {
        // The full per-state event list is exposed to the FFI via for_each_state_events;
        // branchial detection uses the per-consumed-edge inverted index below.
        get_or_create_state_events(input_state)->push(event, *arena_);

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
                uint64_t key = (uint64_t(e1) << 32) | e2;
                auto [_, inserted] = seen_branchial_pairs_.insert_if_absent(key, true);
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

    // Update Desc and Anc sets after adding edge (producer -> consumer)
    void update_transitive_closure(EventId producer, EventId consumer);

    // Add a branchial edge (event <-> event)
    void add_branchial_edge(EventId e1, EventId e2, EdgeId shared);

    // Iterate over causal edges
    template<typename Visitor>
    void for_each_causal_edge(Visitor&& visit) const {
        causal_edges_.for_each([&](const CausalEdge& edge) {
            visit(edge);
        });
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
        return num_causal_edges_.load(std::memory_order_relaxed);
    }

    // Number of unique event pairs with a causal relationship.
    size_t num_causal_event_pairs() const {
        return num_causal_event_pairs_.load(std::memory_order_relaxed);
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
            visit(static_cast<StateId>(state_key), event_list);
        });
    }

};

}  // namespace hypergraph
