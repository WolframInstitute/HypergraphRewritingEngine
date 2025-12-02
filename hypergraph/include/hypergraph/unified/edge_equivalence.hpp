#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "types.hpp"
#include "lock_free_list.hpp"
#include "segmented_array.hpp"
#include "arena.hpp"

namespace hypergraph::unified {

// =============================================================================
// Lock-Free Union-Find for Edge Equivalence Classes
// =============================================================================
// Thread-safe union-find with path compression.
// Used to group edges that correspond across isomorphic states.
//
// Key operations:
// - find(edge_id) -> canonical representative of equivalence class
// - unite(edge_a, edge_b) -> merge equivalence classes, return new representative
//
// Thread safety:
// - find() uses atomic loads with path compression via CAS
// - unite() uses CAS to atomically update parent pointers
// - Linearizable operations

// Wrapper for atomic values that are default-constructible and movable
// (moves value, not atomically - only for use in containers during resize)
struct AtomicEdgeId {
    std::atomic<EdgeId> value{INVALID_ID};

    AtomicEdgeId() = default;
    AtomicEdgeId(EdgeId v) { value.store(v, std::memory_order_relaxed); }

    // Move constructor copies value (not atomic, for container resize only)
    AtomicEdgeId(AtomicEdgeId&& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    AtomicEdgeId& operator=(AtomicEdgeId&& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    // Copy constructor (same behavior)
    AtomicEdgeId(const AtomicEdgeId& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    AtomicEdgeId& operator=(const AtomicEdgeId& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
};

struct AtomicUint32 {
    std::atomic<uint32_t> value{0};

    AtomicUint32() = default;
    AtomicUint32(uint32_t v) { value.store(v, std::memory_order_relaxed); }

    AtomicUint32(AtomicUint32&& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    AtomicUint32& operator=(AtomicUint32&& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
    AtomicUint32(const AtomicUint32& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    AtomicUint32& operator=(const AtomicUint32& other) noexcept {
        value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
};

class LockFreeUnionFind {
    // Parent array: parent_[i] points to parent of edge i
    // If parent_[i] == i, then i is a root (representative)
    std::vector<AtomicEdgeId> parent_;

    // Rank array for union by rank (approximation)
    std::vector<AtomicUint32> rank_;

public:
    LockFreeUnionFind() = default;

    // Initialize with capacity for edge_count edges
    void init(uint32_t edge_count) {
        parent_.resize(edge_count);
        rank_.resize(edge_count);
        for (uint32_t i = 0; i < edge_count; ++i) {
            parent_[i].value.store(i, std::memory_order_relaxed);
            rank_[i].value.store(0, std::memory_order_relaxed);
        }
    }

    // Ensure capacity for at least edge_id + 1 edges
    void ensure_capacity(EdgeId edge_id) {
        uint32_t needed = edge_id + 1;
        if (needed > parent_.size()) {
            // Grow arrays - this is not fully thread-safe but acceptable
            // since edges are created sequentially in practice
            uint32_t old_size = parent_.size();
            parent_.resize(needed);
            rank_.resize(needed);
            for (uint32_t i = old_size; i < needed; ++i) {
                parent_[i].value.store(i, std::memory_order_relaxed);
                rank_[i].value.store(0, std::memory_order_relaxed);
            }
        }
    }

    // Find representative with path compression
    EdgeId find(EdgeId edge_id) {
        if (edge_id >= parent_.size()) {
            return edge_id;  // Not yet in union-find
        }

        EdgeId root = edge_id;

        // Find root
        while (true) {
            EdgeId parent = parent_[root].value.load(std::memory_order_acquire);
            if (parent == root) break;
            root = parent;
        }

        // Path compression: point all nodes on path directly to root
        EdgeId current = edge_id;
        while (current != root) {
            EdgeId parent = parent_[current].value.load(std::memory_order_acquire);
            parent_[current].value.compare_exchange_weak(
                parent, root, std::memory_order_release, std::memory_order_relaxed);
            current = parent;
        }

        return root;
    }

    // Unite two edges into the same equivalence class
    // Returns the representative of the merged class
    EdgeId unite(EdgeId a, EdgeId b) {
        while (true) {
            EdgeId root_a = find(a);
            EdgeId root_b = find(b);

            if (root_a == root_b) {
                return root_a;  // Already in same class
            }

            // Union by rank: attach smaller tree under larger
            uint32_t rank_a = rank_[root_a].value.load(std::memory_order_relaxed);
            uint32_t rank_b = rank_[root_b].value.load(std::memory_order_relaxed);

            EdgeId new_root, old_root;
            if (rank_a < rank_b) {
                new_root = root_b;
                old_root = root_a;
            } else {
                new_root = root_a;
                old_root = root_b;
            }

            // Try to make old_root point to new_root
            EdgeId expected = old_root;
            if (parent_[old_root].value.compare_exchange_strong(
                    expected, new_root,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                // Success! Update rank if needed
                if (rank_a == rank_b) {
                    rank_[new_root].value.fetch_add(1, std::memory_order_relaxed);
                }
                return new_root;
            }
            // CAS failed, retry
        }
    }

    // Check if two edges are in the same equivalence class
    bool same_class(EdgeId a, EdgeId b) {
        return find(a) == find(b);
    }

    // Get current size (number of tracked elements)
    size_t size() const {
        return parent_.size();
    }
};

// =============================================================================
// EdgeEquivalenceClass
// =============================================================================
// Tracks causal information at the equivalence class level.
// When edges are merged via correspondence, their causal relationships
// are combined to enable cross-branch causal edges.

struct EdgeEquivalenceClass {
    EquivClassId id;
    EdgeId representative;  // Canonical representative edge

    // Note: producers and consumers are stored in parallel arrays
    // in the containing class to avoid allocation issues

    EdgeEquivalenceClass()
        : id(INVALID_ID)
        , representative(INVALID_ID)
    {}

    EdgeEquivalenceClass(EquivClassId id_, EdgeId rep)
        : id(id_)
        , representative(rep)
    {}
};

// =============================================================================
// EdgeEquivalenceManager
// =============================================================================
// Manages edge equivalence classes with union-find and causal tracking.
// Integrates with CausalGraph for cross-branch causal edge creation.

class EdgeEquivalenceManager {
    // Union-find for edge equivalence
    LockFreeUnionFind union_find_;

    // Arena for allocations (supports concurrent access)
    ConcurrentHeterogeneousArena* arena_;

    // Per-equivalence-class producer/consumer lists
    // Indexed by representative edge ID (after find())
    SegmentedArray<LockFreeList<EventId>> class_producers_;
    SegmentedArray<LockFreeList<EventId>> class_consumers_;

    // Callback for creating causal edges when classes are merged
    std::function<void(EventId, EventId)> create_causal_edge_callback_;

public:
    EdgeEquivalenceManager() : arena_(nullptr) {}

    void set_arena(ConcurrentHeterogeneousArena* arena) {
        arena_ = arena;
    }

    void set_causal_edge_callback(std::function<void(EventId, EventId)> callback) {
        create_causal_edge_callback_ = std::move(callback);
    }

    // Initialize for expected edge count
    void init(uint32_t expected_edges) {
        union_find_.init(expected_edges);
    }

    // Ensure capacity for edge
    void ensure_capacity(EdgeId edge_id) {
        union_find_.ensure_capacity(edge_id);
        class_producers_.ensure_size(edge_id + 1, *arena_);
        class_consumers_.ensure_size(edge_id + 1, *arena_);
    }

    // Get representative of edge's equivalence class
    EdgeId find(EdgeId edge_id) {
        return union_find_.find(edge_id);
    }

    // Register an event as producer of an edge
    void add_producer(EdgeId edge_id, EventId event_id) {
        ensure_capacity(edge_id);
        EdgeId rep = find(edge_id);
        class_producers_[rep].push(event_id, *arena_);
    }

    // Register an event as consumer of an edge
    void add_consumer(EdgeId edge_id, EventId event_id) {
        ensure_capacity(edge_id);
        EdgeId rep = find(edge_id);
        class_consumers_[rep].push(event_id, *arena_);
    }

    // Merge two edges into the same equivalence class
    // Creates cross-branch causal edges between their producers/consumers
    void merge(EdgeId edge_a, EdgeId edge_b) {
        // Ensure both edges are in the system
        ensure_capacity(edge_a);
        ensure_capacity(edge_b);

        EdgeId rep_a = find(edge_a);
        EdgeId rep_b = find(edge_b);

        if (rep_a == rep_b) {
            return;  // Already in same class
        }

        // Ensure representatives are also in bounds (should be, but be safe)
        ensure_capacity(rep_a);
        ensure_capacity(rep_b);

        // Create cross-branch causal edges BEFORE merging
        // producers(a) × consumers(b) and producers(b) × consumers(a)
        if (create_causal_edge_callback_) {
            // Collect producers and consumers to avoid iterator invalidation
            std::vector<EventId> producers_a, producers_b;
            std::vector<EventId> consumers_a, consumers_b;

            if (rep_a < class_producers_.size()) {
                class_producers_[rep_a].for_each([&](EventId e) {
                    producers_a.push_back(e);
                });
            }
            if (rep_b < class_producers_.size()) {
                class_producers_[rep_b].for_each([&](EventId e) {
                    producers_b.push_back(e);
                });
            }
            if (rep_a < class_consumers_.size()) {
                class_consumers_[rep_a].for_each([&](EventId e) {
                    consumers_a.push_back(e);
                });
            }
            if (rep_b < class_consumers_.size()) {
                class_consumers_[rep_b].for_each([&](EventId e) {
                    consumers_b.push_back(e);
                });
            }

            // Create edges: producers(a) → consumers(b)
            for (EventId prod : producers_a) {
                for (EventId cons : consumers_b) {
                    create_causal_edge_callback_(prod, cons);
                }
            }

            // Create edges: producers(b) → consumers(a)
            for (EventId prod : producers_b) {
                for (EventId cons : consumers_a) {
                    create_causal_edge_callback_(prod, cons);
                }
            }
        }

        // Now merge the classes
        EdgeId new_rep = union_find_.unite(edge_a, edge_b);

        // Ensure new_rep is in bounds
        ensure_capacity(new_rep);

        // Merge producer/consumer lists into new representative
        // The old representative's list stays (as it might still be accessed)
        // but new registrations will go to new_rep
        EdgeId old_rep = (new_rep == rep_a) ? rep_b : rep_a;

        // Copy old rep's producers/consumers to new rep (with bounds checks)
        if (old_rep < class_producers_.size() && new_rep < class_producers_.size()) {
            class_producers_[old_rep].for_each([&](EventId e) {
                class_producers_[new_rep].push(e, *arena_);
            });
        }
        if (old_rep < class_consumers_.size() && new_rep < class_consumers_.size()) {
            class_consumers_[old_rep].for_each([&](EventId e) {
                class_consumers_[new_rep].push(e, *arena_);
            });
        }
    }

    // Iterate over producers of an edge's equivalence class
    template<typename F>
    void for_each_producer(EdgeId edge_id, F&& f) {
        EdgeId rep = find(edge_id);
        if (rep < class_producers_.size()) {
            class_producers_[rep].for_each(std::forward<F>(f));
        }
    }

    // Iterate over consumers of an edge's equivalence class
    template<typename F>
    void for_each_consumer(EdgeId edge_id, F&& f) {
        EdgeId rep = find(edge_id);
        if (rep < class_consumers_.size()) {
            class_consumers_[rep].for_each(std::forward<F>(f));
        }
    }

    // Check if two edges are equivalent
    bool are_equivalent(EdgeId a, EdgeId b) {
        return union_find_.same_class(a, b);
    }

    // Get current size (number of edges tracked)
    size_t size() const {
        return union_find_.size();
    }
};

// =============================================================================
// EventCanonicalizer
// =============================================================================
// Computes canonical hashes for events based on their semantic content.
// Two events are equivalent if:
// - Same rule
// - Input states have same canonical hash
// - Consumed edges are pairwise equivalent (via edge equivalence)
// - Output states have same canonical hash
//
// The canonical hash for an event uses:
// - Rule index
// - Input state canonical hash
// - Sorted list of consumed edge representatives (from union-find)
// - Output state canonical hash

class EventCanonicalizer {
    static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    static constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

    static uint64_t fnv_hash(uint64_t h, uint64_t value) {
        h ^= value;
        h *= FNV_PRIME;
        return h;
    }

public:
    // Compute canonical hash for an event
    // Uses edge equivalence to normalize consumed edge references
    static uint64_t compute_canonical_hash(
        RuleIndex rule_index,
        uint64_t input_state_hash,
        const EdgeId* consumed_edges,
        uint8_t num_consumed,
        uint64_t output_state_hash,
        EdgeEquivalenceManager& edge_equiv
    ) {
        uint64_t h = FNV_OFFSET;

        // Mix in rule
        h = fnv_hash(h, rule_index);

        // Mix in input state hash
        h = fnv_hash(h, input_state_hash);

        // Get canonical representatives for consumed edges and sort them
        std::vector<EdgeId> consumed_reps;
        consumed_reps.reserve(num_consumed);
        for (uint8_t i = 0; i < num_consumed; ++i) {
            consumed_reps.push_back(edge_equiv.find(consumed_edges[i]));
        }
        std::sort(consumed_reps.begin(), consumed_reps.end());

        // Mix in sorted edge representatives
        for (EdgeId rep : consumed_reps) {
            h = fnv_hash(h, rep);
        }

        // Mix in output state hash
        h = fnv_hash(h, output_state_hash);

        return h;
    }

    // Check if two events are equivalent
    static bool are_equivalent(
        const Event& e1,
        const Event& e2,
        uint64_t input_hash_1,
        uint64_t input_hash_2,
        uint64_t output_hash_1,
        uint64_t output_hash_2,
        EdgeEquivalenceManager& edge_equiv
    ) {
        // Must have same rule
        if (e1.rule_index != e2.rule_index) return false;

        // Must have same input state structure
        if (input_hash_1 != input_hash_2) return false;

        // Must have same output state structure
        if (output_hash_1 != output_hash_2) return false;

        // Must consume same number of edges
        if (e1.num_consumed != e2.num_consumed) return false;

        // Consumed edges must be pairwise equivalent
        // First, get representatives for both
        std::vector<EdgeId> reps1, reps2;
        for (uint8_t i = 0; i < e1.num_consumed; ++i) {
            reps1.push_back(edge_equiv.find(e1.consumed_edges[i]));
        }
        for (uint8_t i = 0; i < e2.num_consumed; ++i) {
            reps2.push_back(edge_equiv.find(e2.consumed_edges[i]));
        }

        // Sort and compare
        std::sort(reps1.begin(), reps1.end());
        std::sort(reps2.begin(), reps2.end());

        return reps1 == reps2;
    }
};

}  // namespace hypergraph::unified
