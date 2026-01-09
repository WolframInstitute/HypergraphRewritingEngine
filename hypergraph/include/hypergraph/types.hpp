#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "bitset.hpp"

namespace hypergraph {

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
    EventId canonical_event_id;  // Points to canonical event if this is a duplicate, INVALID_ID if this is canonical

    Event(EventId id_, StateId input, StateId output, RuleIndex rule,
          EdgeId* consumed, uint8_t n_consumed,
          EdgeId* produced, uint8_t n_produced,
          const VariableBinding& bind,
          EventId canonical_id = INVALID_ID)
        : id(id_)
        , input_state(input)
        , output_state(output)
        , rule_index(rule)
        , consumed_edges(consumed)
        , produced_edges(produced)
        , num_consumed(n_consumed)
        , num_produced(n_produced)
        , binding(bind)
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
        , binding()
        , canonical_event_id(INVALID_ID)
    {}

    // Check if this event is canonical (not a duplicate)
    bool is_canonical() const { return canonical_event_id == INVALID_ID; }
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
    StateId canonical_id;     // Canonical representative (cached, set on creation)

    State(StateId id_, SparseBitset&& edge_set, uint32_t step_,
          uint64_t hash, EventId parent, StateId canonical = INVALID_ID)
        : id(id_)
        , edges(std::move(edge_set))
        , step(step_)
        , canonical_hash(hash)
        , parent_event(parent)
        , canonical_id(canonical == INVALID_ID ? id_ : canonical)
    {}

    // Default constructor
    State()
        : id(INVALID_ID)
        , edges()
        , step(0)
        , canonical_hash(0)
        , parent_event(INVALID_ID)
        , canonical_id(INVALID_ID)
    {}

    // Move constructor
    State(State&& other) noexcept
        : id(other.id)
        , edges(std::move(other.edges))
        , step(other.step)
        , canonical_hash(other.canonical_hash)
        , parent_event(other.parent_event)
        , canonical_id(other.canonical_id)
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
        // Use release ordering so counter increment synchronizes with acquire fences
        // in wait_for_completion. This ensures num_events() returns accurate count.
        return next_event.fetch_add(1, std::memory_order_release);
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

// =============================================================================
// Canonicalization vs Exploration Deduplication
// =============================================================================
// There are THREE orthogonal modes that control multiway evolution behavior:
//
// 1. StateCanonicalizationMode (UnifiedHypergraph):
//    - Controls state BOOKKEEPING - which states are considered equivalent
//    - None: Pure tree mode, each state is unique (no equivalence checking)
//    - Automatic: Content hash (fast, not isomorphism-invariant)
//    - Full: Isomorphism-invariant hash (WL/UT)
//    - Affects: num_canonical_states(), get_canonical_state(), was_new_state
//
// 2. EventSignatureKeys (UnifiedHypergraph):
//    - Controls event BOOKKEEPING - which events are considered equivalent
//    - Affects: canonical_event_id, event multiplicity counting
//    - Independent of state mode (always uses isomorphism hashes internally)
//
// 3. explore_from_canonical_states_only (ParallelEvolutionEngine):
//    - Controls EXPLORATION - which states to explore from
//    - false: Explore all states (default multiway behavior)
//    - true: Only explore from first canonical representative (deduplication)
//    - Requires StateCanonicalizationMode::Full to have any effect
//    - States/events are still created, just MATCH tasks are not spawned
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
    Full        // Isomorphism-invariant hash: WL/UT - detects isomorphic states
};

// =============================================================================
// EdgeOccurrence: Position of a vertex within an edge
// =============================================================================

struct EdgeOccurrence {
    EdgeId edge_id;
    uint8_t position;
    uint8_t arity;

    EdgeOccurrence() : edge_id(INVALID_ID), position(0), arity(0) {}
    EdgeOccurrence(EdgeId eid, uint8_t pos, uint8_t ar)
        : edge_id(eid), position(pos), arity(ar) {}
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
// Used by both uniqueness tree and WL implementations
// Now includes subtree bloom filters for O(1) dirty detection

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
        for (uint32_t i = 0; i < count; ++i) {
            if (vertices[i] == v) return hashes[i];
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

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

// Mix a raw integer value for better avalanche (MurmurHash3 finalizer)
// Use this when hashing small raw integers like vertex IDs
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Combine a pre-hashed value into an accumulator (FNV-1a style)
// Use this when the value is already well-distributed (e.g., another hash)
inline uint64_t fnv_hash(uint64_t h, uint64_t value) {
    h ^= value;
    h *= FNV_PRIME;
    return h;
}

// Combine a raw integer value with mixing for better avalanche
// Use this when the value is a small raw integer (e.g., vertex ID)
inline uint64_t fnv_hash_raw(uint64_t h, uint64_t raw_value) {
    return fnv_hash(h, mix64(raw_value));
}

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

// =============================================================================
// StateIncrementalCache: Cached data for incremental hash computation
// =============================================================================
// Stores per-state vertex hash cache to enable incremental computation.
// When computing a child state's hash, we can reuse unchanged vertex hashes
// from the parent state.
//
// Also stores a pointer to cached adjacency (arena-allocated) to avoid
// rebuilding adjacency for each child state.
//
// Uses atomic pointer for thread-safe initialization without torn writes.

struct StateIncrementalCacheData {
    VertexHashCache vertex_cache;
    void* adjacency_ptr;  // Pointer to arena-allocated adjacency map (type-erased)

    StateIncrementalCacheData() : vertex_cache(), adjacency_ptr(nullptr) {}
};

struct StateIncrementalCache {
    std::atomic<StateIncrementalCacheData*> data_ptr{nullptr};

    StateIncrementalCache() = default;
};

}  // namespace hypergraph
