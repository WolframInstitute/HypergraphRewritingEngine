#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include <cstring>
#include <stdexcept>

#include "bitset.hpp"
#include "hgcommon/core.hpp"

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

// Canonical hash of the state holding no edges. Any rule whose RHS is empty reaches it,
// so it is an ordinary canonical form and needs a hash of its own. It cannot be 0, which
// carries two other meanings a state hash must stay clear of: State::canonical_hash uses
// 0 for "not computed yet" (get_canonical_state_for_event falls back to the raw state on
// it), and 0 is ConcurrentMap's EMPTY_KEY, so a 0-hashed state is unstorable in every map
// this hash keys. The value is arbitrary -- the fractional part of the golden ratio, a
// mixing constant already used in this file -- and only has to be non-zero and fixed.
inline constexpr uint64_t EMPTY_STATE_CANONICAL_HASH = 0x9E3779B97F4A7C15ULL;

using RuleIndex = uint16_t;  // host-only width (the GPU port uses a 32-bit RuleId)
// edge's producer events with its consumer events. Strongly typed and distinct from:
//   * EdgeId     -- one concrete hyperedge instance in one state (a dense counter), and
//   * CausalEdge -- a producer->consumer relation at the multiway-graph level.
// Minted ONLY by Hypergraph::causal_edge_keys: under quotient it is
// fnv(canonical_hash(state), edge_orbit_in_state) -- iso-invariant, so every raw edge
// instance that distinct parents produce/consume for one canonical edge orbit collapses to
// a single key (the orbit is the only edge identity invariant across the labelings by
// which parents reach a canonical state). Without quotient it degrades to the raw EdgeId
// so isomorphic-but-distinct raw states keep disjoint causal edges. A hash, hence 64-bit,
// not a dense counter -- never construct one from an EdgeId except through causal_edge_keys.
struct CanonicalEdgeKey {
    uint64_t value{0};
    CanonicalEdgeKey() = default;
    explicit constexpr CanonicalEdgeKey(uint64_t v) : value(v) {}
    constexpr bool operator==(CanonicalEdgeKey o) const { return value == o.value; }
    constexpr bool operator!=(CanonicalEdgeKey o) const { return value != o.value; }
};

// A state's canonical edge-orbit table, computed once when the state is canonicalized
// (Full mode) and cached. `orbit[i]` is the automorphism-orbit id of the edge `edges[i]`;
// `edges` is sorted ascending for binary-search lookup. Orbit ids are numbered canonically
// (iso-invariant), so the same orbit id denotes corresponding edges across every raw state
// isomorphic to this one -- the identity the quotient causal reconstruction indexes on.
// Arena-allocated; `num_orbits` is the count of distinct orbits.
struct EdgeOrbitTable {
    uint32_t n = 0;
    uint32_t num_orbits = 0;
    const EdgeId* edges = nullptr;      // sorted ascending, length n
    const uint32_t* orbit = nullptr;    // length n, parallel to edges
    const uint32_t* orbit_size = nullptr;  // length num_orbits
    // Per-edge SLOT: the edge's rank when the state's edges are ordered by (canonical
    // content class, EdgeId). Slots are a permutation of [0,n). The class prefix makes the
    // ordering canonical, so slot i names structurally corresponding edges across every raw
    // instance of one canonical state -- which is what lets a transition recorded on the
    // expanded representative be replayed against an arbitrary instance. Ties within a class
    // break on EdgeId, which is arbitrary but harmless: edges sharing a content class are
    // interchangeable, and every match over them fires, so the emitted set is unchanged.
    const uint32_t* slot = nullptr;     // length n, parallel to edges
    const uint32_t* klass = nullptr;    // length n, parallel to edges (canonical content class)

    uint32_t index_of(EdgeId e) const {
        // Binary search the sorted edge array; returns n if absent (edge not in state).
        uint32_t lo = 0, hi = n;
        while (lo < hi) {
            uint32_t mid = lo + ((hi - lo) >> 1);
            if (edges[mid] < e) lo = mid + 1; else hi = mid;
        }
        return (lo < n && edges[lo] == e) ? lo : n;
    }
    uint32_t orbit_of(EdgeId e) const {
        const uint32_t i = index_of(e);
        return i < n ? orbit[i] : 0;
    }
    uint32_t slot_of(EdgeId e) const {
        const uint32_t i = index_of(e);
        return (i < n && slot) ? slot[i] : 0;
    }
};

// One distinct canonical transition out of a canonical state, in edge-orbit terms -- the
// unit the quotient causal reconstruction propagates over. All raw events sharing the same
// (from canonical state, to canonical state, rule, consumed orbits, surviving orbit map)
// collapse to one of these; `canon_event` is a representative canonical event id used as
// the producer/consumer identity when emitting causal edges. Orbit arrays are arena-
// allocated and sorted. `to_hash` is the child canonical state. See the validated
// reconstruction in tools/quotient_causal_support_probe.cpp.
struct CanonicalTransition {
    uint64_t to_hash = 0;
    uint64_t sig = 0;              // dedup signature over the fields below
    EventId canon_event = INVALID_ID;
    RuleIndex rule = 0;
    uint32_t num_consumed = 0, num_produced = 0, num_survivors = 0;
    const uint32_t* consumed_orbits = nullptr;   // length num_consumed, sorted
    const uint32_t* produced_orbits = nullptr;   // length num_produced, sorted
    const uint32_t* surv_from = nullptr;         // length num_survivors (orbit in `from`)
    const uint32_t* surv_to = nullptr;           // length num_survivors (orbit in `to`)

};

// One match of the expanded representative of a canonical state, named in SLOTS -- the unit
// the per-instance raw reconstruction replays.
//
// Distinct from CanonicalTransition: that record is DEDUPLICATED by an orbit signature, which
// is right for the aggregate producer-set propagation (it needs only the support) but wrong
// here twice over. Slots are finer than orbits, so two matches consuming different edges of
// one orbit collapse into a single transition; and the record carries no multiplicity, so the
// collapse is not recoverable. Full-capture fires both matches, so the reconstruction must see
// both. Hence the representative's matches are kept in full, undeduplicated.
//
// Consumed/produced stay in MATCH order (not sorted): a replay reads
// producer[consumed_slots[i]] and writes the new raw event into producer[produced_slots[i]],
// so the i-th entry must remain the i-th matched / i-th RHS edge. `from_slots`/`to_slots` are
// the slot counts of the source and child states -- the sizes of the producer vectors this
// match reads and writes.
struct SlotMatch {
    uint64_t to_hash = 0;
    uint32_t id = 0;               // dense id, unique per captured match (claims key on it)
    RuleIndex rule = 0;
    uint32_t from_slots = 0, to_slots = 0;
    uint32_t num_consumed = 0, num_produced = 0, num_survivors = 0;
    const uint32_t* consumed_slots = nullptr;    // length num_consumed (slot in `from`)
    const uint32_t* produced_slots = nullptr;    // length num_produced (slot in `to`)
    const uint32_t* surv_from_slot = nullptr;    // length num_survivors (slot in `from`)
    const uint32_t* surv_to_slot = nullptr;      // length num_survivors (slot in `to`)
};


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

struct GlobalCounters {
    std::atomic<VertexId> next_vertex{0};
    std::atomic<EdgeId> next_edge{0};
    std::atomic<StateId> next_state{0};
    std::atomic<EventId> next_event{0};

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
        // Use release ordering so counter increment synchronizes with acquire fences
        // in wait_for_completion. This ensures num_events() returns accurate count.
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

enum EventSignatureKey : uint8_t {
    EventKey_InputState     = 1 << 0,  // Canonical input state ID
    EventKey_OutputState    = 1 << 1,  // Canonical output state ID
    EventKey_Step           = 1 << 2,  // Evolution step number
    EventKey_Rule           = 1 << 3,  // Rule index
    EventKey_ConsumedEdges  = 1 << 4,  // Canonical positions of consumed edges
    EventKey_ProducedEdges  = 1 << 5,  // Canonical positions of produced edges
};

using EventSignatureKeys = uint8_t;

// Presets matching Multicomputation's CanonicalEventFunction modes
constexpr EventSignatureKeys EVENT_SIG_NONE = 0;
constexpr EventSignatureKeys EVENT_SIG_FULL =
    EventKey_InputState | EventKey_OutputState;
constexpr EventSignatureKeys EVENT_SIG_AUTOMATIC =
    EventKey_InputState | EventKey_OutputState | EventKey_Step |
    EventKey_ConsumedEdges | EventKey_ProducedEdges;

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
// SubtreeBloomFilter: Compact representation of vertices in a subtree
// =============================================================================
// Uses bloom filter to track subtree membership with O(1) membership test.
// False positives possible (may say vertex is in subtree when it isn't),
// but no false negatives (never says vertex is not in subtree when it is).
// This is safe: false positives just cause unnecessary recomputation.

struct SubtreeBloomFilter {
    static constexpr size_t NUM_BITS = 256;  // 32 bytes per filter
    static constexpr size_t NUM_WORDS = NUM_BITS / 64;
    static constexpr size_t NUM_HASHES = 3;  // Number of hash functions

    uint64_t bits[NUM_WORDS] = {0};

    void clear() {
        for (size_t i = 0; i < NUM_WORDS; ++i) bits[i] = 0;
    }

    void add(VertexId v) {
        // Use different hash functions (simple mixing)
        uint64_t h1 = v * 0x9e3779b97f4a7c15ULL;
        uint64_t h2 = v * 0xc6a4a7935bd1e995ULL;
        uint64_t h3 = v * 0x85ebca6b;

        bits[(h1 >> 6) % NUM_WORDS] |= (1ULL << (h1 & 63));
        bits[(h2 >> 6) % NUM_WORDS] |= (1ULL << (h2 & 63));
        bits[(h3 >> 6) % NUM_WORDS] |= (1ULL << (h3 & 63));
    }

    bool might_contain(VertexId v) const {
        uint64_t h1 = v * 0x9e3779b97f4a7c15ULL;
        uint64_t h2 = v * 0xc6a4a7935bd1e995ULL;
        uint64_t h3 = v * 0x85ebca6b;

        return (bits[(h1 >> 6) % NUM_WORDS] & (1ULL << (h1 & 63))) &&
               (bits[(h2 >> 6) % NUM_WORDS] & (1ULL << (h2 & 63))) &&
               (bits[(h3 >> 6) % NUM_WORDS] & (1ULL << (h3 & 63)));
    }

    // Check if any vertex in the given set might be in this subtree
    template<typename Container>
    bool might_contain_any(const Container& vertices) const {
        for (VertexId v : vertices) {
            if (might_contain(v)) return true;
        }
        return false;
    }
};

// =============================================================================
// VertexHashCache: Cached vertex subtree hashes for a state
// =============================================================================
// Used by the WL hash implementation
// Includes subtree bloom filters for O(1) dirty detection

struct VertexHashCache {
    // The hash for each vertex in the state
    // Using simple arrays + count for arena-friendly storage
    VertexId* vertices;
    uint64_t* hashes;
    SubtreeBloomFilter* subtree_filters;  // Bloom filter for each vertex's subtree
    void* adjacency_ptr;  // Type-erased pointer to adjacency map for this state
    uint32_t count;
    uint32_t capacity;

    VertexHashCache() : vertices(nullptr), hashes(nullptr), subtree_filters(nullptr), adjacency_ptr(nullptr), count(0), capacity(0) {}

    uint64_t lookup(VertexId v) const {
        // vertices[] is built sorted ascending (WL hash builds it from a sorted,
        // deduplicated vertex list), so binary-search for v and read the parallel
        // hashes[] slot at the same index. Returns 0 when v is absent.
        const VertexId* pos = std::lower_bound(vertices, vertices + count, v);
        if (pos != vertices + count && *pos == v) {
            return hashes[pos - vertices];
        }
        return 0;
    }

    // Lookup hash and return subtree filter if found
    std::pair<uint64_t, const SubtreeBloomFilter*> lookup_with_subtree(VertexId v) const {
        for (uint32_t i = 0; i < count; ++i) {
            if (vertices[i] == v) {
                return {hashes[i], subtree_filters ? &subtree_filters[i] : nullptr};
            }
        }
        return {0, nullptr};
    }

    void insert(VertexId v, uint64_t hash) {
        // Note: caller must ensure capacity
        vertices[count] = v;
        hashes[count] = hash;
        ++count;
    }

    void insert_with_subtree(VertexId v, uint64_t hash, const SubtreeBloomFilter& filter) {
        vertices[count] = v;
        hashes[count] = hash;
        if (subtree_filters) {
            subtree_filters[count] = filter;
        }
        ++count;
    }
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
