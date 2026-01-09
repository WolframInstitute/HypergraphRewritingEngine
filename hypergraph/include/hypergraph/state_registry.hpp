#pragma once

#include <cstdint>
#include <atomic>

#include "types.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"

namespace hypergraph::unified {

// =============================================================================
// State Child Tracking
// =============================================================================
// Tracks parent-child relationships between states for match cascading.

struct StateChildren {
    LockFreeList<StateId> children;  // Direct children of this state
    std::atomic<uint32_t> num_children{0};
};

// =============================================================================
// State Registry
// =============================================================================
// Manages states with:
// - Parent-child relationships for match cascading
// - Canonical state deduplication (Level 1+)
// - Per-state match lists

class StateRegistry {
    // State storage
    SegmentedArray<State> states_;

    // Per-state child tracking
    SegmentedArray<StateChildren> state_children_;

    // Per-state match lists
    SegmentedArray<LockFreeList<MatchId>> state_matches_;

    // Canonical state deduplication (Level 1+)
    ConcurrentMap<uint64_t, StateId> canonical_states_;

    // Arena for allocations (supports concurrent access)
    ConcurrentHeterogeneousArena* arena_;

    // ID counter
    GlobalCounters* counters_;

    // Configuration
    uint8_t canonicalization_level_;

public:
    StateRegistry(ConcurrentHeterogeneousArena* arena, GlobalCounters* counters, uint8_t canon_level = 0)
        : arena_(arena)
        , counters_(counters)
        , canonicalization_level_(canon_level)
    {}

    // Non-copyable
    StateRegistry(const StateRegistry&) = delete;
    StateRegistry& operator=(const StateRegistry&) = delete;

    // Create a new state
    // Returns (state_id, is_new_state)
    // If canonicalization is enabled and equivalent state exists, returns existing
    std::pair<StateId, bool> create_state(
        SparseBitset&& edge_set,
        uint32_t step,
        uint64_t canonical_hash,
        EventId parent_event,
        StateId parent_state = INVALID_ID
    ) {
        // Check for equivalent state (Level 1+)
        if (canonicalization_level_ >= 1) {
            auto [existing, inserted] = canonical_states_.insert_if_absent(
                canonical_hash, counters_->next_state.load());

            if (!inserted) {
                // Equivalent state exists - return it
                // Note: Still need to record parent-child relationship
                if (parent_state != INVALID_ID) {
                    add_child(parent_state, existing);
                }
                return {existing, false};
            }
        }

        // Create new state
        StateId sid = counters_->alloc_state();

        // Ensure storage is large enough
        while (states_.size() <= sid) {
            states_.emplace(*arena_);
            state_children_.emplace(*arena_);
            state_matches_.emplace(*arena_);
        }

        // Initialize state
        State& state = states_[sid];
        state.id = sid;
        state.edges = std::move(edge_set);
        state.step = step;
        state.canonical_hash = canonical_hash;
        state.parent_event = parent_event;

        // Add parent-child relationship
        if (parent_state != INVALID_ID) {
            add_child(parent_state, sid);
        }

        return {sid, true};
    }

    // Get state by ID
    const State& get_state(StateId sid) const {
        return states_[sid];
    }

    State& get_state(StateId sid) {
        return states_[sid];
    }

    // Get state's edge bitset
    const SparseBitset& get_state_edges(StateId sid) const {
        return states_[sid].edges;
    }

    // Add child to parent
    void add_child(StateId parent, StateId child) {
        if (parent >= state_children_.size()) return;

        state_children_[parent].children.push(child, *arena_);
        state_children_[parent].num_children.fetch_add(1);
    }

    // Iterate over children of a state
    template<typename Visitor>
    void for_each_child(StateId parent, Visitor&& visit) const {
        if (parent >= state_children_.size()) return;

        state_children_[parent].children.for_each([&](StateId child) {
            visit(child);
        });
    }

    // Get number of children
    uint32_t num_children(StateId parent) const {
        if (parent >= state_children_.size()) return 0;
        return state_children_[parent].num_children.load();
    }

    // Add match to state's match list
    void add_match(StateId sid, MatchId mid) {
        if (sid >= state_matches_.size()) return;
        state_matches_[sid].push(mid, *arena_);
    }

    // Iterate over matches in a state
    template<typename Visitor>
    void for_each_match(StateId sid, Visitor&& visit) const {
        if (sid >= state_matches_.size()) return;

        state_matches_[sid].for_each([&](MatchId mid) {
            visit(mid);
        });
    }

    // Inherit valid matches from parent state
    // A match is valid in child if all its edges are still present
    template<typename MatchAccessor, typename EdgeAccessor>
    void inherit_matches(
        StateId parent,
        StateId child,
        const MatchAccessor& get_match,
        const EdgeAccessor& /* get_edge */
    ) {
        const SparseBitset& child_edges = get_state_edges(child);

        for_each_match(parent, [&](MatchId mid) {
            const auto& match = get_match(mid);

            // Check if all matched edges are present in child
            bool valid = true;
            for (uint8_t i = 0; i < match.num_edges && valid; ++i) {
                if (!child_edges.contains(match.matched_edges[i])) {
                    valid = false;
                }
            }

            if (valid) {
                add_match(child, mid);
            }
        });
    }

    // Cascade a new match to all descendants
    // Called when a match is discovered in a state
    template<typename MatchAccessor>
    void cascade_match_to_descendants(
        MatchId mid,
        StateId origin,
        const MatchAccessor& get_match
    ) {
        const auto& match = get_match(mid);

        // BFS through descendants
        LockFreeList<StateId> to_visit;
        SparseBitset visited;

        // Start with children of origin
        for_each_child(origin, [&](StateId child) {
            to_visit.push(child, *arena_);
        });

        while (true) {
            // Pop next state to visit (simple iteration since we're single-threaded here)
            StateId current = INVALID_ID;
            to_visit.for_each([&](StateId sid) {
                if (current == INVALID_ID && !visited.contains(sid)) {
                    current = sid;
                }
            });

            if (current == INVALID_ID) break;

            visited.set(current, *arena_);

            // Check if match is valid in this state
            const SparseBitset& state_edges = get_state_edges(current);
            bool valid = true;
            for (uint8_t i = 0; i < match.num_edges && valid; ++i) {
                if (!state_edges.contains(match.matched_edges[i])) {
                    valid = false;
                }
            }

            if (valid) {
                // Add match to this state
                add_match(current, mid);

                // Continue to children
                for_each_child(current, [&](StateId child) {
                    if (!visited.contains(child)) {
                        to_visit.push(child, *arena_);
                    }
                });
            }
            // If not valid, don't visit children (they can't have consumed edges back)
        }
    }

    // Number of states
    size_t size() const {
        return states_.size();
    }

    // Check if state exists
    bool contains(StateId sid) const {
        return sid < states_.size() && states_[sid].id != INVALID_ID;
    }

    // Access underlying storage (for iteration)
    const SegmentedArray<State>& states() const { return states_; }
};

// =============================================================================
// Match Inheritance Helper
// =============================================================================
// Handles match inheritance when creating new states from rewrites.

template<typename MatchAccessor, typename EdgeAccessor>
void setup_state_matches(
    StateRegistry& registry,
    StateId new_state,
    StateId parent_state,
    const EdgeId* consumed_edges,
    uint8_t num_consumed,
    const MatchAccessor& get_match,
    const EdgeAccessor& get_edge
) {
    if (parent_state == INVALID_ID) return;

    const SparseBitset& new_edges = registry.get_state_edges(new_state);

    // Inherit matches from parent that are still valid
    registry.for_each_match(parent_state, [&](MatchId mid) {
        const auto& match = get_match(mid);

        // Check if any matched edge was consumed
        bool consumed = false;
        for (uint8_t i = 0; i < match.num_edges && !consumed; ++i) {
            for (uint8_t j = 0; j < num_consumed && !consumed; ++j) {
                if (match.matched_edges[i] == consumed_edges[j]) {
                    consumed = true;
                }
            }
        }

        // If no edges consumed, match is still valid
        if (!consumed) {
            // Double-check all edges present (handles edge case)
            bool valid = true;
            for (uint8_t i = 0; i < match.num_edges && valid; ++i) {
                if (!new_edges.contains(match.matched_edges[i])) {
                    valid = false;
                }
            }

            if (valid) {
                registry.add_match(new_state, mid);
            }
        }
    });
}

}  // namespace hypergraph::unified
