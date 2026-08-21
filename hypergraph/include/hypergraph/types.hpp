#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include <cstring>
// AbortedException derives from std::exception, and three constructors below move their
// members. Neither is optional and neither was declared: on libstdc++ both arrive through
// some other header first and the omission is invisible; on libc++ they do not, so any
// translation unit that includes this header BEFORE the one that happened to supply them
// fails to compile. A new .cpp is exactly such a unit.
#include <exception>
#include <utility>

#include "bitset.hpp"
#include "hgcommon/core.hpp"
#include "hgcommon/event_core.hpp"

// The quotient/canonical identity cluster: a separate concern from the raw graph below,
// and the only part of this file that needs hgcommon/quotient_replay_core.hpp.
#include "quotient_types.hpp"

namespace HG_NAMESPACE {
namespace engine {

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
    const char* what() const noexcept override;
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

    VariableBinding();

    bool is_bound(uint8_t var_index) const;
    VertexId get(uint8_t var_index) const;
    void bind(uint8_t var_index, VertexId vertex);
    void unbind(uint8_t var_index);
    bool empty() const;
    uint8_t count() const;
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
         EventId creator, uint32_t step_);

    // Default constructor for array allocation
    Edge();

    // Copy/move re-home an inline `vertices` pointer to this object's own buffer, so a
    // copied edge never aliases the source's inline storage. Out-of-line vertices are
    // arena-owned and immutable, so the pointer is shared as-is.
    Edge(const Edge& o);
    Edge& operator=(const Edge& o);
    Edge(Edge&& o) noexcept;
    Edge& operator=(Edge&& o) noexcept;

private:
    void copy_vertices_from(const Edge& o);
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
          EventId canonical_id = INVALID_ID);

    // Default constructor for array allocation
    Event();

    // Check if this event is canonical (not a duplicate)
    bool is_canonical() const;
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
          uint64_t hash, EventId parent, StateId canonical = INVALID_ID);

    // Default constructor
    State();

    // Move constructor
    State(State&& other) noexcept;

    // Move assignment
    State& operator=(State&& other) noexcept;

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

    VertexId alloc_vertex();
    EdgeId alloc_edge();
    StateId alloc_state();

    // Release pairs with the acquire load in num_events(), so a reader that sees the
    // count also sees the event this id was allocated for. The counter's own value
    // needs no ordering to be fresh -- coherence gives that.
    EventId alloc_event();

    void reset();
};

// =============================================================================
// CausalEdge / BranchialEdge
// =============================================================================
// Represent relationships between events.

struct CausalEdge {
    EventId producer;   // Source event (produces the edge)
    EventId consumer;   // Target event (consumes the edge)
    EdgeId edge;        // The edge that connects them (for debugging/viz)

    CausalEdge(EventId p, EventId c, EdgeId e);
    CausalEdge();

    bool operator==(const CausalEdge& other) const;
};

struct BranchialEdge {
    EventId event1;     // First event
    EventId event2;     // Second event (event1 < event2 by convention)
    EdgeId shared_edge; // One of the shared input edges (for debugging/viz)

    // event1 < event2 by convention, imposed here so every construction site gets it.
    BranchialEdge(EventId e1, EventId e2, EdgeId se);
    BranchialEdge();

    bool operator==(const BranchialEdge& other) const;
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

    EdgeCorrespondence();
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

    uint64_t hash() const;
};

}  // namespace engine
}  // namespace HG_NAMESPACE