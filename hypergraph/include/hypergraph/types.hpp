#pragma once

#include <atomic>
#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include <cstring>

#include "bitset.hpp"
#include "hgcommon/core.hpp"
#include "hgcommon/event_core.hpp"

// The quotient/canonical identity cluster: a separate concern from the raw graph below,
// and the only part of this file that needs hgcommon/quotient_replay_core.hpp.
#include "quotient_types.hpp"

namespace hypergraph {

// =============================================================================
// Identifiers
// =============================================================================
// All IDs are uint32_t: 4 billion sufficient, less cache pressure than uint64.
// Allocated via atomic fetch_add from global counters.

using hgcommon::VertexId;
using hgcommon::EdgeId;
using hgcommon::StateId;
using hgcommon::EventId;
using hgcommon::MatchId;
using hgcommon::INVALID_ID;
using RuleIndex = uint16_t;  // host-only width (the GPU port uses a 32-bit RuleId)

// Both engines reserve one value for the empty state's canonical hash; it is defined once, in
// hgcommon/core.hpp, and named here so host call sites reach it unqualified.
using hgcommon::EMPTY_STATE_CANONICAL_HASH;

// Both engines answer the same question -- which artifacts must this run build -- so the record
// set is defined once, in hgcommon/core.hpp, and named here so host call sites reach it
// unqualified.
using hgcommon::RecordSet;

// =============================================================================
// AbortedException
// =============================================================================
// Thrown when a long-running operation detects abort request.
// Caught by job system's exception handler, which sets ErrorType::Exception.

struct AbortedException : std::exception {
    const char* what() const noexcept override { return "Operation aborted"; }
};

// =============================================================================
// VariableBinding
// =============================================================================
// Fixed-size inline array for pattern matching bindings.
// No heap allocation.

using hgcommon::MAX_VARS;

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
        return static_cast<uint8_t>(hgcommon::popcount(bound_mask));
    }
};

// =============================================================================
// Edge
// =============================================================================
// Represents a hyperedge in the hypergraph.
// Immutable after creation. Allocated from arena.

struct Edge {
    // Arity up to INLINE_ARITY is stored inline in the edge itself, so a binary edge
    // (the common case) needs no separate vertex allocation and no pointer-chase to
    // reach its vertices. `vertices` always points at the live storage — the inline
    // buffer for small edges, arena memory for larger ones — so every reader
    // dereferences it identically regardless of where the vertices live. The field
    // order packs the inline buffer into padding the pointer would otherwise waste,
    // keeping sizeof(Edge) at 32 bytes.
    static constexpr uint8_t INLINE_ARITY = 2;

    EdgeId id;
    uint8_t arity;
    EventId creator_event;   // INVALID_ID for initial edges
    uint32_t step;
    VertexId* vertices;      // -> inline_vertices (arity<=INLINE_ARITY) or arena array
    VertexId inline_vertices[INLINE_ARITY];

    // src supplies the arity vertex ids. spill is arena storage used only when arity
    // exceeds INLINE_ARITY; for inline edges it is ignored and may be nullptr.
    Edge(EdgeId id_, const VertexId* src, uint8_t arity_, VertexId* spill,
         EventId creator, uint32_t step_)
        : id(id_)
        , arity(arity_)
        , creator_event(creator)
        , step(step_)
    {
        VertexId* dst = (arity_ <= INLINE_ARITY) ? inline_vertices : spill;
        for (uint8_t i = 0; i < arity_; ++i) dst[i] = src[i];
        vertices = dst;
    }

    // Default constructor for array allocation
    Edge()
        : id(INVALID_ID)
        , arity(0)
        , creator_event(INVALID_ID)
        , step(0)
        , vertices(nullptr)
    {}

    // Copy/move re-home an inline `vertices` pointer to this object's own buffer, so a
    // copied edge never aliases the source's inline storage. Out-of-line vertices are
    // arena-owned and immutable, so the pointer is shared as-is.
    Edge(const Edge& o)
        : id(o.id)
        , arity(o.arity)
        , creator_event(o.creator_event)
        , step(o.step)
    {
        copy_vertices_from(o);
    }

    Edge& operator=(const Edge& o) {
        id = o.id;
        arity = o.arity;
        creator_event = o.creator_event;
        step = o.step;
        copy_vertices_from(o);
        return *this;
    }

    Edge(Edge&& o) noexcept : Edge(static_cast<const Edge&>(o)) {}
    Edge& operator=(Edge&& o) noexcept { return *this = static_cast<const Edge&>(o); }

private:
    void copy_vertices_from(const Edge& o) {
        if (o.vertices == o.inline_vertices) {
            for (uint8_t i = 0; i < arity; ++i) inline_vertices[i] = o.inline_vertices[i];
            vertices = inline_vertices;
        } else {
            vertices = o.vertices;  // arena-owned, shared
        }
    }
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
    EventId canonical_event_id;  // Points to canonical event if this is a duplicate, INVALID_ID if this is canonical

    // The identity this run computed for the event, from hgcommon::event_signature. 0 under
    // EventSignatureKeys None, where events are kept distinct by computing no signature at all.
    //
    // Retained rather than discarded after it has served as the canonical_event_map_ key,
    // because whether two runs agree on event identity is a question about the VALUES and not
    // about how many distinct ones there were: a permutation of signatures across events leaves
    // every count intact. DeviceEvent carries the same field for the same reason.
    uint64_t signature;

    // The match's VariableBinding is NOT stored on the event: it is consumed during
    // RHS instantiation and never read from a persistent event afterwards (the event
    // records consumed/produced edges explicitly). Keeping it cost 132 B per event,
    // retained for the whole run.
    Event(EventId id_, StateId input, StateId output, RuleIndex rule,
          EdgeId* consumed, uint8_t n_consumed,
          EdgeId* produced, uint8_t n_produced,
          EventId canonical_id = INVALID_ID)
        : id(id_)
        , input_state(input)
        , output_state(output)
        , rule_index(rule)
        , consumed_edges(consumed)
        , produced_edges(produced)
        , num_consumed(n_consumed)
        , num_produced(n_produced)
        , canonical_event_id(canonical_id)
        , signature(0)
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
        , canonical_event_id(INVALID_ID)
        , signature(0)
    {}

    // Check if this event is canonical (not a duplicate)
    bool is_canonical() const { return canonical_event_id == INVALID_ID; }
};

// =============================================================================
// State
// =============================================================================
// Represents a state in the multiway system - a view into the hypergraph.
// The SparseBitset tracks which edges are present in this state.
// Immutable after creation.
// Allocated from arena.

struct State {
    StateId id;
    SparseBitset edges;       // Which edges are present in this state
    uint32_t step;
    uint64_t canonical_hash;  // Isomorphism-invariant canonical hash
    EventId parent_event;     // Event that created this, INVALID_ID for initial
    StateId canonical_id;     // Canonical representative (cached, set on creation)
    // Quotient exploration (explore_from_canonical_states_only). explore_depth is the
    // shortest path length known to reach this canonical state, INVALID_ID until first
    // relaxed; expanded records that its matches have been computed, which happens once.
    // Both are reached through std::atomic_ref, never assigned directly after creation.
    uint32_t explore_depth;
    uint32_t expanded;

    State(StateId id_, SparseBitset&& edge_set, uint32_t step_,
          uint64_t hash, EventId parent, StateId canonical = INVALID_ID)
        : id(id_)
        , edges(std::move(edge_set))
        , step(step_)
        , canonical_hash(hash)
        , parent_event(parent)
        , canonical_id(canonical == INVALID_ID ? id_ : canonical)
        , explore_depth(INVALID_ID)
        , expanded(0)
    {}

    // Default constructor
    State()
        : id(INVALID_ID)
        , edges()
        , step(0)
        , canonical_hash(0)
        , parent_event(INVALID_ID)
        , canonical_id(INVALID_ID)
        , explore_depth(INVALID_ID)
        , expanded(0)
    {}

    // Move constructor
    State(State&& other) noexcept
        : id(other.id)
        , edges(std::move(other.edges))
        , step(other.step)
        , canonical_hash(other.canonical_hash)
        , parent_event(other.parent_event)
        , canonical_id(other.canonical_id)
        , explore_depth(other.explore_depth)
        , expanded(other.expanded)
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
            canonical_id = other.canonical_id;
            explore_depth = other.explore_depth;
            expanded = other.expanded;
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

// Packed together these four are 16 bytes -- one cache line -- and every worker
// fetch_adds them on every edge, state and event it creates. The line would then
// ping-pong between cores on allocations that have nothing to do with each other, so
// each counter gets a line of its own. The struct is a singleton, so the padding costs
// nothing that matters.
struct GlobalCounters {
    alignas(64) std::atomic<VertexId> next_vertex{0};
    alignas(64) std::atomic<EdgeId> next_edge{0};
    alignas(64) std::atomic<StateId> next_state{0};
    alignas(64) std::atomic<EventId> next_event{0};

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
        // Release pairs with the acquire load in num_events(), so a reader that sees the
        // count also sees the event this id was allocated for. The counter's own value
        // needs no ordering to be fresh -- coherence gives that.
        return next_event.fetch_add(1, std::memory_order_release);
    }

    void reset() {
        next_vertex.store(0, std::memory_order_relaxed);
        next_edge.store(0, std::memory_order_relaxed);
        next_state.store(0, std::memory_order_relaxed);
        next_event.store(0, std::memory_order_relaxed);
    }
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
// Canonicalization vs Exploration Deduplication
// =============================================================================
// There are THREE orthogonal modes that control multiway evolution behavior:
//
// 1. StateCanonicalizationMode (Hypergraph):
//    - Controls state BOOKKEEPING - which states are considered equivalent
//    - None: Pure tree mode, each state is unique (no equivalence checking)
//    - Automatic: Content hash (fast, not isomorphism-invariant)
//    - Full: Isomorphism-invariant hash (WL approximate / IR exact)
//    - Affects: num_canonical_states(), get_canonical_state(), was_new_state
//
// 2. EventSignatureKeys (Hypergraph):
//    - Controls event BOOKKEEPING - which events are considered equivalent
//    - Affects: canonical_event_id, event multiplicity counting
//    - Independent of state mode (always uses isomorphism hashes internally)
//
// 3. explore_from_canonical_states_only (ParallelEvolutionEngine):
//    - Controls EXPLORATION - which states to explore from
//    - false: Expand every provenance (reference semantics, exact online
//      causal/branchial)
//    - true: Quotient exploration - expand each canonical state once, at its
//      shortest depth (lock-free depth relaxation); deterministic; causal and
//      branchial multisets of the full expansion are reconstructed offline
//      from the skeleton (tools/quotient_reconstruction_probe.cpp)
//    - Requires StateCanonicalizationMode::Full to have any effect
//
// Common configurations:
// - Pure tree: State=None, Event=None, Explore=false
// - Full bookkeeping: State=Full, Event=Full, Explore=false
// - Exploration dedup: State=Full, Event=Full, Explore=true
// =============================================================================

// =============================================================================
// EventSignatureKeys: Bitflags controlling event equivalence
// =============================================================================
// Events with identical signatures are considered equivalent and deduplicated.
// Corresponds to Multicomputation's CanonicalEventFunction key selection.
// When 0 (None), no event canonicalization occurs.

// The key bits, the presets and the signature rule live in hgcommon so the device computes
// event identity the same way; these names keep working unqualified.
using hgcommon::EventSignatureKey;
using hgcommon::EventSignatureKeys;
using hgcommon::EventKey_InputState;
using hgcommon::EventKey_OutputState;
using hgcommon::EventKey_Step;
using hgcommon::EventKey_Rule;
using hgcommon::EventKey_ConsumedEdges;
using hgcommon::EventKey_ProducedEdges;
using hgcommon::EVENT_SIG_NONE;
using hgcommon::EVENT_SIG_FULL;
using hgcommon::EVENT_SIG_AUTOMATIC;

// =============================================================================
// StateCanonicalizationMode: Controls state canonicalization/deduplication
// =============================================================================
// Corresponds to Multicomputation's CanonicalStateFunction modes.

enum class StateCanonicalizationMode : uint8_t {
    None,       // Tree mode: no deduplication, each state is unique
    Automatic,  // Content-ordered hash: hash(edge_contents) - fast but not isomorphism-invariant
    Full        // Isomorphism-invariant hash: WL approximate / IR exact - detects isomorphic states
};

// =============================================================================
// EdgeCorrespondence: Mapping between edges in two isomorphic states
// =============================================================================

struct EdgeCorrespondence {
    EdgeId* state1_edges;
    EdgeId* state2_edges;
    uint32_t count;
    bool valid;

    EdgeCorrespondence() : state1_edges(nullptr), state2_edges(nullptr), count(0), valid(false) {}
};

// =============================================================================
// EventSignature: Signature for event deduplication
// =============================================================================

using hgcommon::FNV_OFFSET;
using hgcommon::FNV_PRIME;
using hgcommon::mix64;
using hgcommon::fnv_hash;

struct EventSignature {
    uint64_t input_state_hash;
    uint64_t output_state_hash;
    uint64_t consumed_edges_sig;
    uint64_t produced_edges_sig;

    uint64_t hash() const {
        uint64_t h = FNV_OFFSET;
        h = fnv_hash(h, input_state_hash);
        h = fnv_hash(h, output_state_hash);
        h = fnv_hash(h, consumed_edges_sig);
        h = fnv_hash(h, produced_edges_sig);
        return h;
    }
};

}  // namespace hypergraph
