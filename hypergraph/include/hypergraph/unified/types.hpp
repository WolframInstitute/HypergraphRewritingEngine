#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#include "bitset.hpp"

namespace hypergraph::unified {

// =============================================================================
// Identifiers
// =============================================================================
// All IDs are uint32_t: 4 billion sufficient, less cache pressure than uint64.
// Allocated via atomic fetch_add from global counters.

using VertexId = uint32_t;
using EdgeId = uint32_t;
using StateId = uint32_t;
using EventId = uint32_t;
using MatchId = uint32_t;
using EquivClassId = uint32_t;
using RuleIndex = uint16_t;

constexpr uint32_t INVALID_ID = UINT32_MAX;

// =============================================================================
// VariableBinding
// =============================================================================
// Fixed-size inline array for pattern matching bindings.
// No heap allocation.

constexpr uint8_t MAX_VARS = 32;

struct VariableBinding {
    VertexId bindings[MAX_VARS];
    uint32_t bound_mask;  // Bitmask of which vars are bound

    VariableBinding() : bound_mask(0) {
        std::memset(bindings, 0xFF, sizeof(bindings));  // Initialize to INVALID_ID
    }

    bool is_bound(uint8_t var_index) const {
        return (bound_mask & (1u << var_index)) != 0;
    }

    VertexId get(uint8_t var_index) const {
        return bindings[var_index];
    }

    void bind(uint8_t var_index, VertexId vertex) {
        bindings[var_index] = vertex;
        bound_mask |= (1u << var_index);
    }

    void unbind(uint8_t var_index) {
        bindings[var_index] = INVALID_ID;
        bound_mask &= ~(1u << var_index);
    }

    bool empty() const {
        return bound_mask == 0;
    }

    uint8_t count() const {
        return static_cast<uint8_t>(__builtin_popcount(bound_mask));
    }
};

// =============================================================================
// Edge
// =============================================================================
// Represents a hyperedge in the unified hypergraph.
// Immutable after creation except for equiv_class (atomic update for Level 2).
// Allocated from arena.

struct Edge {
    EdgeId id;
    VertexId* vertices;      // Arena-allocated array
    uint8_t arity;
    EventId creator_event;   // INVALID_ID for initial edges
    uint32_t step;
    std::atomic<EquivClassId> equiv_class;  // Updated when correspondence found

    Edge(EdgeId id_, VertexId* verts, uint8_t arity_, EventId creator, uint32_t step_)
        : id(id_)
        , vertices(verts)
        , arity(arity_)
        , creator_event(creator)
        , step(step_)
        , equiv_class(id_)  // Initially each edge is its own equivalence class
    {}

    // Default constructor for array allocation
    Edge()
        : id(INVALID_ID)
        , vertices(nullptr)
        , arity(0)
        , creator_event(INVALID_ID)
        , step(0)
        , equiv_class(INVALID_ID)
    {}
};

// =============================================================================
// Match
// =============================================================================
// Represents a complete pattern match. Immutable after creation.
// Allocated from arena.

struct Match {
    MatchId id;
    RuleIndex rule_index;
    EdgeId* matched_edges;   // Arena-allocated array
    uint8_t num_edges;
    VariableBinding binding;
    StateId origin_state;    // Where first discovered

    Match(MatchId id_, RuleIndex rule, EdgeId* edges, uint8_t n_edges,
          const VariableBinding& bind, StateId origin)
        : id(id_)
        , rule_index(rule)
        , matched_edges(edges)
        , num_edges(n_edges)
        , binding(bind)
        , origin_state(origin)
    {}

    // Default constructor for array allocation
    Match()
        : id(INVALID_ID)
        , rule_index(0)
        , matched_edges(nullptr)
        , num_edges(0)
        , binding()
        , origin_state(INVALID_ID)
    {}
};

// =============================================================================
// Event
// =============================================================================
// Represents a rewrite event. Immutable after creation.
// Allocated from arena.

struct Event {
    EventId id;
    StateId input_state;
    StateId output_state;
    RuleIndex rule_index;
    EdgeId* consumed_edges;  // Arena-allocated array
    EdgeId* produced_edges;  // Arena-allocated array
    uint8_t num_consumed;
    uint8_t num_produced;
    VariableBinding binding;

    Event(EventId id_, StateId input, StateId output, RuleIndex rule,
          EdgeId* consumed, uint8_t n_consumed,
          EdgeId* produced, uint8_t n_produced,
          const VariableBinding& bind)
        : id(id_)
        , input_state(input)
        , output_state(output)
        , rule_index(rule)
        , consumed_edges(consumed)
        , produced_edges(produced)
        , num_consumed(n_consumed)
        , num_produced(n_produced)
        , binding(bind)
    {}

    // Default constructor for array allocation
    Event()
        : id(INVALID_ID)
        , input_state(INVALID_ID)
        , output_state(INVALID_ID)
        , rule_index(0)
        , consumed_edges(nullptr)
        , produced_edges(nullptr)
        , num_consumed(0)
        , num_produced(0)
        , binding()
    {}
};

// =============================================================================
// State
// =============================================================================
// Represents a state in the multiway system - a view into the unified hypergraph.
// The SparseBitset tracks which edges are present in this state.
// Immutable after creation.
// Allocated from arena.

struct State {
    StateId id;
    SparseBitset edges;       // Which edges are present in this state
    uint32_t step;
    uint64_t canonical_hash;  // Computed via uniqueness tree
    EventId parent_event;     // Event that created this, INVALID_ID for initial

    State(StateId id_, SparseBitset&& edge_set, uint32_t step_,
          uint64_t hash, EventId parent)
        : id(id_)
        , edges(std::move(edge_set))
        , step(step_)
        , canonical_hash(hash)
        , parent_event(parent)
    {}

    // Default constructor
    State()
        : id(INVALID_ID)
        , edges()
        , step(0)
        , canonical_hash(0)
        , parent_event(INVALID_ID)
    {}

    // Move constructor
    State(State&& other) noexcept
        : id(other.id)
        , edges(std::move(other.edges))
        , step(other.step)
        , canonical_hash(other.canonical_hash)
        , parent_event(other.parent_event)
    {
        other.id = INVALID_ID;
    }

    // Move assignment
    State& operator=(State&& other) noexcept {
        if (this != &other) {
            id = other.id;
            edges = std::move(other.edges);
            step = other.step;
            canonical_hash = other.canonical_hash;
            parent_event = other.parent_event;
            other.id = INVALID_ID;
        }
        return *this;
    }

    // Delete copy to prevent accidental aliasing
    State(const State&) = delete;
    State& operator=(const State&) = delete;
};

// =============================================================================
// Global ID Counters
// =============================================================================
// Thread-safe ID allocation via atomic fetch_add.

struct GlobalCounters {
    std::atomic<VertexId> next_vertex{0};
    std::atomic<EdgeId> next_edge{0};
    std::atomic<StateId> next_state{0};
    std::atomic<EventId> next_event{0};
    std::atomic<MatchId> next_match{0};
    std::atomic<EquivClassId> next_equiv_class{0};

    VertexId alloc_vertex() {
        return next_vertex.fetch_add(1, std::memory_order_relaxed);
    }

    EdgeId alloc_edge() {
        return next_edge.fetch_add(1, std::memory_order_relaxed);
    }

    StateId alloc_state() {
        return next_state.fetch_add(1, std::memory_order_relaxed);
    }

    EventId alloc_event() {
        return next_event.fetch_add(1, std::memory_order_relaxed);
    }

    MatchId alloc_match() {
        return next_match.fetch_add(1, std::memory_order_relaxed);
    }

    EquivClassId alloc_equiv_class() {
        return next_equiv_class.fetch_add(1, std::memory_order_relaxed);
    }

    void reset() {
        next_vertex.store(0, std::memory_order_relaxed);
        next_edge.store(0, std::memory_order_relaxed);
        next_state.store(0, std::memory_order_relaxed);
        next_event.store(0, std::memory_order_relaxed);
        next_match.store(0, std::memory_order_relaxed);
        next_equiv_class.store(0, std::memory_order_relaxed);
    }
};

// =============================================================================
// EdgeCausalInfo
// =============================================================================
// Per-edge causal tracking for online causal edge computation.
// Uses rendezvous pattern: both producer and consumers write-then-read.
//
// Thread safety: Lock-free via atomic producer and LockFreeList consumers.

struct EdgeCausalInfo {
    std::atomic<EventId> producer{INVALID_ID};  // Set once when edge created
    // Note: consumers stored separately in SegmentedArray<LockFreeList<EventId>>
    // to avoid including lock_free_list.hpp here (circular dependency)
};

// =============================================================================
// CausalEdge / BranchialEdge
// =============================================================================
// Represent relationships between events.

struct CausalEdge {
    EventId producer;   // Source event (produces the edge)
    EventId consumer;   // Target event (consumes the edge)
    EdgeId edge;        // The edge that connects them (for debugging/viz)

    CausalEdge(EventId p, EventId c, EdgeId e)
        : producer(p), consumer(c), edge(e) {}

    CausalEdge() : producer(INVALID_ID), consumer(INVALID_ID), edge(INVALID_ID) {}

    bool operator==(const CausalEdge& other) const {
        return producer == other.producer && consumer == other.consumer;
    }
};

struct BranchialEdge {
    EventId event1;     // First event
    EventId event2;     // Second event (event1 < event2 by convention)
    EdgeId shared_edge; // One of the shared input edges (for debugging/viz)

    BranchialEdge(EventId e1, EventId e2, EdgeId se)
        : event1(e1 < e2 ? e1 : e2)
        , event2(e1 < e2 ? e2 : e1)
        , shared_edge(se) {}

    BranchialEdge() : event1(INVALID_ID), event2(INVALID_ID), shared_edge(INVALID_ID) {}

    bool operator==(const BranchialEdge& other) const {
        return event1 == other.event1 && event2 == other.event2;
    }
};

// =============================================================================
// StateBranchialInfo
// =============================================================================
// Per-state tracking for branchial edge computation.
// Events that originate from this state are tracked here.
// When a new event is created, we check for overlap with existing events.
//
// Note: events_from_here stored separately in SegmentedArray<LockFreeList<EventId>>

struct StateBranchialInfo {
    // Placeholder - actual event list stored externally due to dependency ordering
};

}  // namespace hypergraph::unified
