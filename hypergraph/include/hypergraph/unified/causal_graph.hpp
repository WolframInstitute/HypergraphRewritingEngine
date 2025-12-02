#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#include "types.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"

namespace hypergraph::unified {

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
    SegmentedArray<EdgeCausalInfo> edge_producers_;

    // Per-edge consumer lists (appended when edge consumed)
    SegmentedArray<LockFreeList<EventId>> edge_consumers_;

    // Per-state event lists for branchial tracking
    // Maps StateId -> list of events that have this state as input
    SegmentedArray<LockFreeList<EventId>> state_events_;

    // Causal edges (producer -> consumer)
    LockFreeList<CausalEdge> causal_edges_;

    // Branchial edges (event <-> event with shared input)
    LockFreeList<BranchialEdge> branchial_edges_;

    // Deduplication map for branchial edges: (e1 << 32 | e2) -> true
    ConcurrentMap<uint64_t, bool> seen_branchial_pairs_;

    // Arena for allocations (supports concurrent access)
    ConcurrentHeterogeneousArena* arena_;

    // Statistics
    std::atomic<size_t> num_causal_edges_{0};
    std::atomic<size_t> num_branchial_edges_{0};

public:
    CausalGraph() : arena_(nullptr) {}

    explicit CausalGraph(ConcurrentHeterogeneousArena* arena) : arena_(arena) {}

    // Set arena (for deferred initialization)
    void set_arena(ConcurrentHeterogeneousArena* arena) {
        arena_ = arena;
    }

    // =========================================================================
    // Edge Causal Tracking
    // =========================================================================

    // Ensure storage for edge ID exists (thread-safe)
    void ensure_edge_capacity(EdgeId max_edge) {
        edge_producers_.ensure_size(max_edge + 1, *arena_);
        edge_consumers_.ensure_size(max_edge + 1, *arena_);
    }

    // Ensure storage for state ID exists (thread-safe)
    void ensure_state_capacity(StateId max_state) {
        state_events_.ensure_size(max_state + 1, *arena_);
    }

    // Called when an edge is produced by an event
    // Returns true if producer was set (first time), false if already set
    bool set_edge_producer(EdgeId edge, EventId producer) {
        ensure_edge_capacity(edge);

        // Try to set producer atomically (only succeeds if INVALID_ID)
        EventId expected = INVALID_ID;
        bool was_set = edge_producers_[edge].producer.compare_exchange_strong(
            expected, producer,
            std::memory_order_release,
            std::memory_order_acquire
        );

        if (was_set) {
            // We are the producer. Check for any consumers that arrived first.
            // (Rendezvous pattern: write then read)
            edge_consumers_[edge].for_each([&](EventId consumer) {
                add_causal_edge(producer, consumer, edge);
            });
        }

        return was_set;
    }

    // Called when an edge is consumed by an event
    void add_edge_consumer(EdgeId edge, EventId consumer) {
        ensure_edge_capacity(edge);

        // Add self to consumers list (write)
        edge_consumers_[edge].push(consumer, *arena_);

        // Check for producer (read)
        EventId producer = edge_producers_[edge].producer.load(std::memory_order_acquire);
        if (producer != INVALID_ID) {
            add_causal_edge(producer, consumer, edge);
        }
    }

    // Get producer of an edge (may be INVALID_ID for initial edges)
    EventId get_edge_producer(EdgeId edge) const {
        if (edge >= edge_producers_.size()) return INVALID_ID;
        return edge_producers_[edge].producer.load(std::memory_order_acquire);
    }

    // =========================================================================
    // Branchial Tracking
    // =========================================================================

    // Called when an event is created from a state
    // Checks for branchial relationships with other events from the same state
    void register_event_from_state(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed
    ) {
        ensure_state_capacity(input_state);

        // Check for branchial relationships with existing events from this state
        state_events_[input_state].for_each([&](EventId other_event) {
            // Check for edge overlap
            // Note: we need access to other event's consumed edges
            // This is deferred - the caller must provide the check function
        });

        // Add self to state's event list
        state_events_[input_state].push(event, *arena_);
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
        ensure_state_capacity(input_state);

        // Add self to state's event list FIRST (before checking)
        state_events_[input_state].push(event, *arena_);

        // Check for branchial relationships with ALL events in the list
        // Both sides of a pair may try to add the edge - deduplication handles it
        state_events_[input_state].for_each([&](EventId other_event) {
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

    // =========================================================================
    // Graph Access
    // =========================================================================

    // Add a causal edge (producer -> consumer)
    void add_causal_edge(EventId producer, EventId consumer, EdgeId edge) {
        causal_edges_.push(CausalEdge(producer, consumer, edge), *arena_);
        num_causal_edges_.fetch_add(1, std::memory_order_relaxed);
    }

    // Add a branchial edge (event <-> event)
    void add_branchial_edge(EventId e1, EventId e2, EdgeId shared) {
        branchial_edges_.push(BranchialEdge(e1, e2, shared), *arena_);
        num_branchial_edges_.fetch_add(1, std::memory_order_relaxed);
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

    size_t num_branchial_edges() const {
        return num_branchial_edges_.load(std::memory_order_relaxed);
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
};

}  // namespace hypergraph::unified
