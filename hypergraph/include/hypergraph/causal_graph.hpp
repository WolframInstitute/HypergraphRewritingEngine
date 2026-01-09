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
    // Per-edge producer tracking (set once when edge created)
    // Use ConcurrentMap for thread-safe "get or create" semantics
    // Key: EdgeId (as uint64_t), Value: pointer to EdgeCausalInfo
    static constexpr uint64_t EDGE_PROD_MAP_EMPTY = 1ULL << 62;
    static constexpr uint64_t EDGE_PROD_MAP_LOCKED = (1ULL << 62) + 1;
    ConcurrentMap<uint64_t, EdgeCausalInfo*, EDGE_PROD_MAP_EMPTY, EDGE_PROD_MAP_LOCKED> edge_producers_;

    // Per-edge consumer lists (appended when edge consumed)
    // Use ConcurrentMap for thread-safe "get or create" semantics
    static constexpr uint64_t EDGE_CONS_MAP_EMPTY = (1ULL << 62) + 2;
    static constexpr uint64_t EDGE_CONS_MAP_LOCKED = (1ULL << 62) + 3;
    ConcurrentMap<uint64_t, LockFreeList<EventId>*, EDGE_CONS_MAP_EMPTY, EDGE_CONS_MAP_LOCKED> edge_consumers_;

    // Per-state event lists for branchial tracking
    // Maps StateId -> list of events that have this state as input
    // Uses ConcurrentMap for thread-safe "get or create" semantics
    // Key: StateId (as uint64_t), Value: pointer to LockFreeList
    // Use special EMPTY_KEY and LOCKED_KEY outside valid StateId range (0 to 2^32-1)
    static constexpr uint64_t STATE_MAP_EMPTY = 1ULL << 62;
    static constexpr uint64_t STATE_MAP_LOCKED = (1ULL << 62) + 1;
    ConcurrentMap<uint64_t, LockFreeList<EventId>*, STATE_MAP_EMPTY, STATE_MAP_LOCKED> state_events_;

    // Causal edges (producer -> consumer)
    LockFreeList<CausalEdge> causal_edges_;

    // Branchial edges (event <-> event with shared input)
    LockFreeList<BranchialEdge> branchial_edges_;

    // Deduplication map for causal edges: hash(producer, consumer, edge) -> true
    // The rendezvous pattern can cause both producer and consumer to add the same edge
    ConcurrentMap<uint64_t, bool> seen_causal_triples_;

    // Deduplication map for causal event pairs: (producer << 32 | consumer) -> true
    // Used for counting unique event pairs with causal relationships (v1 semantics)
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
    // Thread-safety: Uses concurrent sets with atomic operations.
    // Some redundant edges may slip through in races, but no duplicates.

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

    // Whether online TR is enabled (default: disabled for v1 compatibility)
    std::atomic<bool> transitive_reduction_enabled_{false};

    // Statistics for TR
    std::atomic<size_t> num_redundant_edges_skipped_{0};

    // Arena for allocations (supports concurrent access)
    ConcurrentHeterogeneousArena* arena_;

    // Statistics
    std::atomic<size_t> num_causal_edges_{0};        // Per-edge causal relationships
    std::atomic<size_t> num_causal_event_pairs_{0};  // Unique event pairs (v1 semantics)
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
    EdgeCausalInfo* get_or_create_edge_producer(EdgeId edge) {
        uint64_t key = edge;

        auto result = edge_producers_.lookup(key);
        if (result.has_value()) {
            return *result;
        }

        auto* new_info = arena_->template create<EdgeCausalInfo>();
        auto [existing, inserted] = edge_producers_.insert_if_absent(key, new_info);
        return inserted ? new_info : existing;
    }

    // Get or create consumer list for an edge (thread-safe)
    LockFreeList<EventId>* get_or_create_edge_consumers(EdgeId edge) {
        uint64_t key = edge;

        auto result = edge_consumers_.lookup(key);
        if (result.has_value()) {
            return *result;
        }

        auto* new_list = arena_->template create<LockFreeList<EventId>>();
        auto [existing, inserted] = edge_consumers_.insert_if_absent(key, new_list);
        return inserted ? new_list : existing;
    }

    // Get or create Desc set for an event (Goranci algorithm)
    // Returns lock-free ConcurrentMap used as a set (key present = in set)
    DescAncSet* get_or_create_desc(EventId event) {
        uint64_t key = event;

        auto result = desc_.lookup(key);
        if (result.has_value()) {
            return *result;
        }

        auto* new_set = arena_->template create<DescAncSet>();
        auto [existing, inserted] = desc_.insert_if_absent(key, new_set);
        return inserted ? new_set : existing;
    }

    // Get or create Anc set for an event (Goranci algorithm)
    // Returns lock-free ConcurrentMap used as a set (key present = in set)
    DescAncSet* get_or_create_anc(EventId event) {
        uint64_t key = event;

        auto result = anc_.lookup(key);
        if (result.has_value()) {
            return *result;
        }

        auto* new_set = arena_->template create<DescAncSet>();
        auto [existing, inserted] = anc_.insert_if_absent(key, new_set);
        return inserted ? new_set : existing;
    }

    // O(1) redundancy check: is consumer reachable from producer?
    // Uses Desc[producer] set from Goranci algorithm (lock-free)
    bool is_reachable_via_desc(EventId producer, EventId consumer) const {
        if (producer == consumer) return true;

        auto desc_result = desc_.lookup(producer);
        if (!desc_result.has_value()) return false;

        return (*desc_result)->contains(consumer);  // O(1) lock-free lookup
    }

    // Get or create the event list for a state (thread-safe)
    // Uses ConcurrentMap for proper "create once" semantics
    LockFreeList<EventId>* get_or_create_state_events(StateId state) {
        uint64_t key = state;

        // First, try to look up existing list
        auto result = state_events_.lookup(key);
        if (result.has_value()) {
            return *result;
        }

        // Need to create - allocate new list from arena
        auto* new_list = arena_->template create<LockFreeList<EventId>>();

        // Try to insert - if another thread beat us, use theirs
        auto [existing, inserted] = state_events_.insert_if_absent(key, new_list);

        // Return whichever list is now in the map
        return inserted ? new_list : existing;
    }

    // Called when an edge is produced by an event
    // Returns true if producer was set (first time), false if already set
    bool set_edge_producer(EdgeId edge, EventId producer) {
        EdgeCausalInfo* info = get_or_create_edge_producer(edge);
        LockFreeList<EventId>* consumers = get_or_create_edge_consumers(edge);

        // Try to set producer atomically (only succeeds if INVALID_ID)
        EventId expected = INVALID_ID;
        bool was_set = info->producer.compare_exchange_strong(
            expected, producer,
            std::memory_order_release,
            std::memory_order_acquire
        );

        if (was_set) {
            // We are the producer. Check for any consumers that arrived first.
            // (Rendezvous pattern: write then read)
            consumers->for_each([&](EventId consumer) {
                add_causal_edge(producer, consumer, edge);
            });
        }

        return was_set;
    }

    // Called when an edge is consumed by an event
    void add_edge_consumer(EdgeId edge, EventId consumer) {
        EdgeCausalInfo* info = get_or_create_edge_producer(edge);
        LockFreeList<EventId>* consumers = get_or_create_edge_consumers(edge);

        // Add self to consumers list (write)
        consumers->push(consumer, *arena_);

        // Check for producer (read)
        EventId producer = info->producer.load(std::memory_order_acquire);
        if (producer != INVALID_ID) {
            add_causal_edge(producer, consumer, edge);
        }
    }

    // Get producer of an edge (may be INVALID_ID for initial edges)
    EventId get_edge_producer(EdgeId edge) const {
        auto result = edge_producers_.lookup(edge);
        if (!result.has_value()) return INVALID_ID;
        return (*result)->producer.load(std::memory_order_acquire);
    }

    // =========================================================================
    // Branchial Tracking
    // =========================================================================

    // Called when an event is created from a state
    // Checks for branchial relationships with other events from the same state
    void register_event_from_state(
        EventId event,
        StateId input_state,
        [[maybe_unused]] const EdgeId* consumed_edges,
        [[maybe_unused]] uint8_t num_consumed
    ) {
        // Get or create the event list for this state (thread-safe)
        LockFreeList<EventId>* list = get_or_create_state_events(input_state);

        // Check for branchial relationships with existing events from this state
        list->for_each([]([[maybe_unused]] EventId other_event) {
            // Check for edge overlap
            // Note: we need access to other event's consumed edges
            // This is deferred - the caller must provide the check function
        });

        // Add self to state's event list
        list->push(event, *arena_);
    }

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
        GetEventConsumedEdges&& get_consumed
    ) {
        // Get or create the event list for this state (thread-safe)
        LockFreeList<EventId>* list = get_or_create_state_events(input_state);

        // Add self to state's event list FIRST (before checking)
        list->push(event, *arena_);

        // Check for branchial relationships with ALL events in the list
        // Both sides of a pair may try to add the edge - deduplication handles it
        list->for_each([&](EventId other_event) {
            if (other_event == event) return;  // Skip self

            // Get other event's consumed edges
            const EdgeId* other_consumed;
            uint8_t other_num_consumed;
            get_consumed(other_event, other_consumed, other_num_consumed);

            // Check for overlap
            EdgeId shared = find_shared_edge(
                consumed_edges, num_consumed,
                other_consumed, other_num_consumed
            );

            if (shared != INVALID_ID) {
                // Use ordered pair as key for deduplication
                EventId e1 = std::min(event, other_event);
                EventId e2 = std::max(event, other_event);
                uint64_t key = (uint64_t(e1) << 32) | e2;

                auto [_, inserted] = seen_branchial_pairs_.insert_if_absent(key, true);
                if (inserted) {
                    add_branchial_edge(e1, e2, shared);
                }
            }
        });
    }

    // Version with canonical event checking and edge equivalence
    // - skip_if_same_canonical: callback(e1, e2) returns true if events are canonically equivalent
    // - edges_equivalent: callback(e1, e2) returns true if edges are equivalent (in same class)
    template<typename GetEventConsumedEdges, typename SameCanonicalEvent, typename EdgesEquivalent>
    void register_event_from_state_with_canonicalization(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed,
        GetEventConsumedEdges&& get_consumed,
        SameCanonicalEvent&& same_canonical,
        EdgesEquivalent&& edges_equivalent
    ) {
        // Get or create the event list for this state (thread-safe)
        LockFreeList<EventId>* list = get_or_create_state_events(input_state);

        // Add self to state's event list FIRST (before checking)
        list->push(event, *arena_);

        // Check for branchial relationships with ALL events in the list
        list->for_each([&](EventId other_event) {
            if (other_event == event) return;  // Skip self

            // Skip if events are canonically equivalent (same canonical event)
            if (same_canonical(event, other_event)) return;

            // Get other event's consumed edges
            const EdgeId* other_consumed;
            uint8_t other_num_consumed;
            get_consumed(other_event, other_consumed, other_num_consumed);

            // Check for overlap using edge equivalence
            EdgeId shared = find_shared_edge_with_equivalence(
                consumed_edges, num_consumed,
                other_consumed, other_num_consumed,
                edges_equivalent
            );

            if (shared != INVALID_ID) {
                // Use ordered pair as key for deduplication
                EventId e1 = std::min(event, other_event);
                EventId e2 = std::max(event, other_event);
                uint64_t key = (uint64_t(e1) << 32) | e2;

                auto [_, inserted] = seen_branchial_pairs_.insert_if_absent(key, true);
                if (inserted) {
                    add_branchial_edge(e1, e2, shared);
                }
            }
        });
    }

    // =========================================================================
    // Graph Access
    // =========================================================================

    // Add a causal edge (producer -> consumer) with deduplication and optional TR
    // The rendezvous pattern can cause both set_edge_producer and add_edge_consumer
    // to detect the same (producer, consumer, edge) triple, so we deduplicate here.
    // Each edge creates its own causal relationship.
    //
    // Online Transitive Reduction (Goranci Algorithm):
    // Before storing, check if consumer ∈ Desc[producer]. If so, edge is redundant.
    // After storing, update Desc/Anc sets for transitive closure maintenance.
    void add_causal_edge(EventId producer, EventId consumer, EdgeId edge) {
        // Online TR check using Goranci algorithm: O(1) lookup
        if (transitive_reduction_enabled_.load(std::memory_order_relaxed)) {
            // First check if we already have a DIRECT edge (producer -> consumer)
            // If so, this is a duplicate, not a TR skip - let deduplication handle it
            uint64_t pair_key = (static_cast<uint64_t>(producer) << 32) | consumer;
            auto existing_pair = seen_causal_event_pairs_.lookup(pair_key);

            if (!existing_pair.has_value()) {
                // No direct edge yet - check if consumer already reachable via Desc set
                if (is_reachable_via_desc(producer, consumer)) {
                    // Consumer already reachable - edge is redundant
                    num_redundant_edges_skipped_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
        }

        // Hash (producer, consumer, edge) into 64 bits for per-edge deduplication
        // FNV-1a style hash for good distribution
        uint64_t triple_key = 14695981039346656037ULL;
        triple_key ^= producer;
        triple_key *= 1099511628211ULL;
        triple_key ^= consumer;
        triple_key *= 1099511628211ULL;
        triple_key ^= edge;
        triple_key *= 1099511628211ULL;

        auto [_, inserted] = seen_causal_triples_.insert_if_absent(triple_key, true);
        if (inserted) {
            causal_edges_.push(CausalEdge(producer, consumer, edge), *arena_);
            num_causal_edges_.fetch_add(1, std::memory_order_relaxed);

            // Emit visualization event for causal edge
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
            VIZ_EMIT_CAUSAL_EDGE(producer, consumer, edge);
#endif

            // Update Desc/Anc sets for Goranci algorithm (maintains transitive closure)
            if (transitive_reduction_enabled_.load(std::memory_order_relaxed)) {
                update_transitive_closure(producer, consumer);
            }

            // Also track unique event pairs (for v1 compatibility)
            uint64_t pair_key = (static_cast<uint64_t>(producer) << 32) | consumer;
            auto [_2, pair_inserted] = seen_causal_event_pairs_.insert_if_absent(pair_key, true);
            if (pair_inserted) {
                num_causal_event_pairs_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Update Desc and Anc sets after adding edge (producer -> consumer)
    // Goranci algorithm: propagate reachability through transitive closure
    // Thread-safe: uses lock-free ConcurrentMap operations throughout
    void update_transitive_closure(EventId producer, EventId consumer) {
        // Get consumer's descendants (events reachable from consumer)
        DescAncSet* desc_consumer = get_or_create_desc(consumer);

        // Get producer's ancestors (events that can reach producer)
        DescAncSet* anc_producer = get_or_create_anc(producer);

        // Collect events to update: {producer} ∪ Anc[producer]
        // Reserve based on current size (approximate, but avoids most reallocs)
        std::vector<EventId> ancestors_to_update;
        ancestors_to_update.reserve(1 + anc_producer->size());
        ancestors_to_update.push_back(producer);
        anc_producer->for_each([&](uint64_t id, bool) {
            ancestors_to_update.push_back(static_cast<EventId>(id));
        });

        // Collect new reachable events: {consumer} ∪ Desc[consumer]
        std::vector<EventId> descendants_to_add;
        descendants_to_add.reserve(1 + desc_consumer->size());
        descendants_to_add.push_back(consumer);
        desc_consumer->for_each([&](uint64_t id, bool) {
            descendants_to_add.push_back(static_cast<EventId>(id));
        });

        // Update Desc sets: for all a ∈ {producer} ∪ Anc[producer], add {consumer} ∪ Desc[consumer]
        for (EventId a : ancestors_to_update) {
            DescAncSet* desc_a = get_or_create_desc(a);
            for (EventId d : descendants_to_add) {
                desc_a->insert_if_absent(d, true);  // Lock-free, idempotent
            }
        }

        // Update Anc sets: for all d ∈ {consumer} ∪ Desc[consumer], add {producer} ∪ Anc[producer]
        // Reuse ancestors_to_update - no need to rebuild
        for (EventId d : descendants_to_add) {
            DescAncSet* anc_d = get_or_create_anc(d);
            for (EventId a : ancestors_to_update) {
                anc_d->insert_if_absent(a, true);  // Lock-free, idempotent
            }
        }
    }

    // Add a branchial edge (event <-> event)
    void add_branchial_edge(EventId e1, EventId e2, EdgeId shared) {
        branchial_edges_.push(BranchialEdge(e1, e2, shared), *arena_);
        num_branchial_edges_.fetch_add(1, std::memory_order_relaxed);

        // Emit visualization event for branchial edge
        // Note: We use generation=0 here as we don't have generation info in this context
        // The actual generation can be determined by the visualization from event data
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        VIZ_EMIT_BRANCHIAL_EDGE(e1, e2, 0);
#endif
    }

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

    // Number of unique event pairs with causal relationships (v1 semantics)
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
    std::vector<CausalEdge> get_causal_edges() const {
        std::vector<CausalEdge> result;
        for_each_causal_edge([&](const CausalEdge& e) {
            result.push_back(e);
        });
        return result;
    }

    // Collect branchial edges into vector (for export/testing)
    std::vector<BranchialEdge> get_branchial_edges() const {
        std::vector<BranchialEdge> result;
        for_each_branchial_edge([&](const BranchialEdge& e) {
            result.push_back(e);
        });
        return result;
    }

    // Iterate over all (input_state -> events) mappings
    // Visitor signature: void(StateId input_state, LockFreeList<EventId>* event_list)
    // Caller can use event_list->for_each() to iterate events
    template<typename Visitor>
    void for_each_state_events(Visitor&& visit) const {
        state_events_.for_each([&](uint64_t state_key, LockFreeList<EventId>* event_list) {
            visit(static_cast<StateId>(state_key), event_list);
        });
    }

private:
    // Find first shared edge between two edge sets (O(n*m) but sets are small)
    static EdgeId find_shared_edge(
        const EdgeId* edges1, uint8_t n1,
        const EdgeId* edges2, uint8_t n2
    ) {
        for (uint8_t i = 0; i < n1; ++i) {
            for (uint8_t j = 0; j < n2; ++j) {
                if (edges1[i] == edges2[j]) {
                    return edges1[i];
                }
            }
        }
        return INVALID_ID;
    }

    // Find first shared edge using edge equivalence (O(n*m) but sets are small)
    // Two edges are considered "shared" if they are equivalent (same canonical class)
    template<typename EdgesEquivalent>
    static EdgeId find_shared_edge_with_equivalence(
        const EdgeId* edges1, uint8_t n1,
        const EdgeId* edges2, uint8_t n2,
        EdgesEquivalent&& edges_equivalent
    ) {
        for (uint8_t i = 0; i < n1; ++i) {
            for (uint8_t j = 0; j < n2; ++j) {
                if (edges_equivalent(edges1[i], edges2[j])) {
                    return edges1[i];
                }
            }
        }
        return INVALID_ID;
    }
};

}  // namespace hypergraph
