#pragma once

#include <cstdint>
#include <cstring>
#include <atomic>
#include <vector>
#include <memory>
#include <unordered_map>

#include "types.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "index.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "causal_graph.hpp"
#include "wl_hash.hpp"
#include "concurrent_map.hpp"

// Shared types: CanonicalizationResult, CanonicalForm, VertexMapping
#include "canonical_types.hpp"

namespace hypergraph {

// =============================================================================
// Hypergraph
// =============================================================================
// Central storage for all hypergraph data in the multiway system.
//
// Key design principles:
// - All edges are stored once (shared storage)
// - States are SparseBitset views over the edge pool
// - Thread-safe allocation via atomic counters
// - Arena allocation for cache-friendly memory layout
// - Lock-free indices for concurrent pattern matching
//
// Thread safety:
// - Edge/state/event/match creation: Lock-free via atomic counters
// - Index updates: Lock-free via ConcurrentMap and LockFreeList
// - Reading: Always safe (immutable after creation)

class Hypergraph {
    // Global ID counters (thread-safe)
    GlobalCounters counters_;

    // Arena for all allocations (thread-safe for parallel evolution)
    ConcurrentHeterogeneousArena arena_;

    // Edge storage
    SegmentedArray<Edge> edges_;

    // Cached edge signatures (computed once at edge creation, immutable)
    SegmentedArray<EdgeSignature> edge_signatures_;

    // State storage
    SegmentedArray<State> states_;

    // Event storage
    SegmentedArray<Event> events_;

    // Pattern matching indices
    PatternMatchingIndex match_index_;

    // Causal and branchial graph
    CausalGraph causal_graph_;

    // Canonical state deduplication map: canonical_hash -> StateId
    // Used to find existing equivalent states before creating new ones
    ConcurrentMap<uint64_t, StateId> canonical_state_map_;

    // Event canonicalization state map: always keyed by isomorphism-invariant hash
    // Unlike canonical_state_map_ (keyed differently based on state_canonicalization_mode_),
    // this map is ALWAYS keyed by canonical_hash (WL/IR) regardless of state mode.
    // Used by event signature computation to find canonical representatives for
    // edge correspondence when state_canonicalization_mode_ is None or Automatic.
    ConcurrentMap<uint64_t, StateId> event_canonical_state_map_;

    // State canonicalization mode: controls how states are deduplicated
    // None: tree mode - no deduplication, each state is unique
    // Automatic: content-ordered hash (not yet implemented, behaves like Full)
    // Full: isomorphism-invariant hash via WL approximate or IR exact
    // NOTE: Must be atomic for ARM64 memory ordering - ensures visibility to worker threads
    std::atomic<StateCanonicalizationMode> state_canonicalization_mode_{StateCanonicalizationMode::None};

    // Whether the evolution quotients isomorphic states (explore-from-canonical-only). When
    // set, causal edges are keyed by canonical edge orbit so attribution is schedule-
    // independent across the labelings by which parents reach one canonical state.
    std::atomic<bool> quotient_causal_{false};

    // Per-state canonical edge-orbit tables, computed once at state canonicalization in
    // quotient mode (piggybacked on the dedup IR canonicalization, so no extra canon pass)
    // and cached by state id. The quotient causal reconstruction reads edge orbits from
    // here rather than recomputing per event (which would re-run IR canonicalization on
    // every event -- catastrophic on high-automorphism states). Key: StateId as uint64_t.
    ConcurrentMap<uint64_t, EdgeOrbitTable*> state_orbit_tables_;

    // The captured quotient causal skeleton: the distinct canonical transitions out of each
    // canonical state (keyed by the source state's canonical hash), plus a dedup set over
    // transition signatures. Built online as events fire in quotient mode; the depth-indexed
    // producer-set reconstruction propagates over it.
    ConcurrentMap<uint64_t, LockFreeList<CanonicalTransition>*> transitions_from_;
    ConcurrentMap<uint64_t, bool> seen_transitions_;

    // Depth-indexed producer-set reconstruction (the online form of the validated DP).
    // qc_dsup_ maps key(state_hash, depth, orbit) -> set of producer canonical-event ids
    // (append-only); qc_dsup_seen_ dedups (key, producer); qc_reached_ marks (state_hash,
    // depth). Producers cascade forward monotonically as transitions and reachability are
    // discovered, emitting causal edges into causal_graph_. Bounded by qc_max_steps_.
    ConcurrentMap<uint64_t, LockFreeList<EventId>*> qc_dsup_;
    ConcurrentMap<uint64_t, bool> qc_dsup_seen_;
    ConcurrentMap<uint64_t, bool> qc_reached_;
    std::atomic<int> qc_max_steps_{0};

    // The expanded representative's FULL match list per canonical state, in slots -- the
    // input to the per-instance raw reconstruction. Deliberately not the deduplicated
    // transitions_from_: slots are finer than orbits and SlotMatch carries no multiplicity,
    // so two matches over one orbit must both survive (full-capture fires both).
    // qc_expansion_rep_ pins the one raw state whose events define the expansion, so a second
    // raw state of the same class (a dedup race) cannot append a duplicate expansion.
    ConcurrentMap<uint64_t, LockFreeList<SlotMatch>*> qc_expansion_;
    ConcurrentMap<uint64_t, uint64_t> qc_expansion_rep_;   // canonical hash -> StateId + 1
    std::atomic<uint32_t> qc_next_match_id_{0};

    // The slot FRAME of a canonical class: the first raw state seen for the class, whose slot
    // numbering every other instance of that class is aligned into.
    //
    // Slots are read off a state's canonical labeling, and two raw states of one class have
    // labelings differing by an automorphism -- different reference frames. Without a frame the
    // reconstruction mixes them: a child's producer vector would be written in the producing
    // event's own output-state numbering but read against a different state's, and which state
    // that is depends on thread scheduling. Pinning one frame per class removes the choice.
    ConcurrentMap<uint64_t, uint64_t> qc_frame_;           // canonical hash -> StateId + 1

    // Fills out[i] with the frame slot of orb->edges[i]. Identity when `s` IS the frame (the
    // common case -- the expanded representative usually claims it), otherwise one edge
    // correspondence against the frame state. Runs only while capturing the expansion, i.e.
    // once per canonical match, never on the per-instance path.
    bool qc_frame_slots(uint64_t state_hash, StateId s, const EdgeOrbitTable* orb, uint32_t* out);

    // Diagnostic: a state's frame slots must be a function of the state, so two calls for one
    // state must agree. qc_frame_sig_ records the first result; disagreements are counted.
    ConcurrentMap<uint64_t, uint64_t> qc_frame_sig_;       // StateId + 1 -> slot-vector hash
    std::atomic<size_t> qc_frame_disagree_{0};
    std::atomic<size_t> qc_align_fail_{0};      // captures dropped because alignment failed
    std::atomic<size_t> qc_align_badcorr_{0};   // of those, an invalid/short edge correspondence
    void qc_check_frame_stable(StateId s, const uint32_t* slots, uint32_t n);

    // Per-instance raw reconstruction. One QcInstance is one raw state of the full expansion,
    // carrying the producing reconstructed-event id per slot; replaying every expansion match
    // against every instance regenerates the raw event set the quotient never explores.
    // Reconstructed event ids come from a counter -- counts and causal edges need only ids,
    // not Event records, so this does not undo the quotient's state/edge compression.
    struct QcInstance {
        uint32_t id = 0;
        uint32_t nslots = 0;
        const uint32_t* prod = nullptr;   // length nslots; QC_NO_PRODUCER for initial edges
    };
    static constexpr uint32_t QC_NO_PRODUCER = 0xFFFFFFFFu;
    ConcurrentMap<uint64_t, LockFreeList<QcInstance>*> qc_instances_;   // key(hash,depth,0)
    // Claims a (instance, match) application. Both the instance side and the match side drive
    // the rendezvous, and unlike the producer-set DP an application is NOT idempotent -- each
    // one emits a raw event -- so the pair must be claimed exactly once. O(raw) entries.
    ConcurrentMap<uint64_t, bool> qc_applied_;
    std::atomic<uint32_t> qc_next_instance_{0};
    std::atomic<uint32_t> qc_next_raw_event_{0};
    std::atomic<bool> quotient_reconstruction_{false};

    // Reconstructed causal relation over raw event ids. ONE base with TWO views: every pair is
    // recorded, and each carries whether it survives transitive reduction, so TR-on is a filter
    // rather than a mode and either view is available in any order without recomputation.
    // Reconstructed ids are topological by construction -- qc_apply mints a producer's id
    // before creating the child instance whose later application mints the consumer's -- and
    // when a consumer is applied its whole ancestor sub-DAG is already emitted, so the
    // reduction decision is exact at insertion.
    ConcurrentMap<uint64_t, bool> qc_causal_pairs_;              // distinct (producer, consumer)
    ConcurrentMap<uint64_t, LockFreeList<uint32_t>*> qc_preds_;  // kept (reduced) predecessors
    // Isomorphism-invariant signature per reconstructed event: fnv(from hash, to hash, rule).
    // Reconstructed events carry no Event record, so this is the only identity they have -- it
    // is what schedule-independence is fingerprinted on, and what a later materialisation of
    // the raw event list would key off.
    SegmentedArray<uint64_t> qc_event_sig_;
    std::atomic<size_t> qc_num_causal_edges_{0};   // per consumed edge (the T1 multiset)
    std::atomic<size_t> qc_num_causal_pairs_{0};   // distinct pairs, un-reduced view
    std::atomic<size_t> qc_num_tr_pairs_{0};       // distinct pairs surviving reduction
    std::atomic<size_t> qc_num_branchial_{0};      // sibling matches of one instance, overlapping
    bool qc_reachable(uint32_t producer, uint32_t consumer) const;
    void qc_record_causal(uint32_t producer, uint32_t consumer);

    static uint64_t qc_key(uint64_t state_hash, uint32_t depth, uint32_t orbit) {
        uint64_t h = 1469598103934665603ULL;
        h ^= state_hash; h *= 1099511628211ULL;
        h ^= (static_cast<uint64_t>(depth) << 32) | orbit; h *= 1099511628211ULL;
        return h;
    }
    static uint64_t qc_rkey(uint64_t state_hash, uint32_t depth) {
        uint64_t h = 1469598103934665603ULL;
        h ^= state_hash; h *= 1099511628211ULL;
        h ^= depth; h *= 1099511628211ULL;
        return h ? h : 1;
    }
    LockFreeList<EventId>* qc_dsup_list(uint64_t key);
    void qc_capture_expansion(EventId e);
    void qc_add_instance(uint64_t state_hash, uint32_t depth, const uint32_t* prod, uint32_t nslots);
    void qc_apply(const QcInstance& inst, const SlotMatch& m, uint64_t state_hash, uint32_t depth);
    void qc_add_producer(uint64_t state_hash, uint32_t depth, uint32_t orbit, EventId producer);
    void qc_process_transition(const CanonicalTransition& t, uint64_t from_hash, uint32_t depth);
    void qc_reach(uint64_t state_hash, uint32_t depth);
    void qc_emit(EventId producer, EventId consumer);

    // Weisfeiler-Leman hash implementation (fast approximate state hash)
    std::unique_ptr<WLHash> wl_hash_;

    // Selects the algorithm for compute_canonical_hash:
    //   true  -> WL approximate hash (fast hot path)
    //   false -> IR exact canonicalization (isomorphism-invariant)
    bool use_wl_hash_{true};


    // Event canonicalization: maps event signature to first EventId
    // Signature computed from keys specified by event_signature_keys_ bitflag
    ConcurrentMap<uint64_t, EventId> canonical_event_map_;
    std::atomic<uint32_t> canonical_event_count_{0};
    EventSignatureKeys event_signature_keys_{EVENT_SIG_NONE};

    // Genesis state: the empty state (no edges) from which all initial states originate
    // Created lazily on first call to get_or_create_genesis_state()
    // Uses lock-free initialization: 0=uninit, 1=in_progress, 2=done
    StateId genesis_state_{INVALID_ID};
    std::atomic<int> genesis_state_init_{0};

public:
    Hypergraph()
        // Route every map's table storage through the arena (no malloc, no per-map
        // heap contention). Ordered by member-declaration order. arena_ is declared
        // before these maps, so it is fully constructed here.
        : canonical_state_map_(
              decltype(canonical_state_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
        , event_canonical_state_map_(
              decltype(event_canonical_state_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
        , wl_hash_(std::make_unique<WLHash>(&arena_))
        , canonical_event_map_(
              decltype(canonical_event_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
    {
        causal_graph_.set_arena(&arena_);
    }

    // Non-copyable
    Hypergraph(const Hypergraph&) = delete;
    Hypergraph& operator=(const Hypergraph&) = delete;

    // =========================================================================
    // Vertex Management
    // =========================================================================

    // Allocate a new vertex ID
    VertexId alloc_vertex() {
        return counters_.alloc_vertex();
    }

    // Allocate N consecutive vertex IDs
    VertexId alloc_vertices(uint32_t count) {
        VertexId first = counters_.next_vertex.fetch_add(count, std::memory_order_relaxed);
        return first;
    }

    // Get current vertex count (upper bound)
    uint32_t num_vertices() const {
        return counters_.next_vertex.load(std::memory_order_relaxed);
    }

    // Ensure vertex ID space is at least `max_id + 1`
    void reserve_vertices(VertexId max_id) {
        VertexId current = counters_.next_vertex.load(std::memory_order_relaxed);
        while (current <= max_id) {
            if (counters_.next_vertex.compare_exchange_weak(
                    current, max_id + 1, std::memory_order_relaxed)) {
                break;
            }
        }
    }

    // =========================================================================
    // Edge Management
    // =========================================================================

    // Create a new edge
    EdgeId create_edge(
        const VertexId* vertices,
        uint8_t arity,
        EventId creator_event = INVALID_ID,
        uint32_t step = 0
    );

    // Create edge from initializer list (convenience)
    EdgeId create_edge(std::initializer_list<VertexId> vertices,
                       EventId creator_event = INVALID_ID,
                       uint32_t step = 0);

    // Get edge by ID
    const Edge& get_edge(EdgeId eid) const {
        return edges_[eid];
    }

    Edge& get_edge(EdgeId eid) {
        return edges_[eid];
    }

    // Edge accessor (for pattern matching)
    auto edge_accessor() const {
        return [this](EdgeId eid) -> const Edge& {
            return edges_[eid];
        };
    }

    // Number of edges
    uint32_t num_edges() const {
        return counters_.next_edge.load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Edge Accessors for the WL hash
    // =========================================================================
    // These provide the interface needed by WLHash::compute_state_hash*()

    // Get vertex array for an edge (returns pointer to vertices)
    const VertexId* edge_vertices(EdgeId eid) const {
        return edges_[eid].vertices;
    }

    // Get arity of an edge
    uint8_t edge_arity(EdgeId eid) const {
        return edges_[eid].arity;
    }

    // Get cached signature for an edge (computed once at creation)
    const EdgeSignature& edge_signature(EdgeId eid) const {
        return edge_signatures_[eid];
    }

    // Lightweight accessor for the WL hash that provides pointer indexing
    // Returns pointer to edge's inline vertex array - no copying or allocation
    class EdgeVertexAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeVertexAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

        const VertexId* operator[](EdgeId eid) const {
            return hg_->edges_[eid].vertices;
        }
    };

    // Direct arity accessor - reads from struct field, O(1)
    class EdgeArityAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeArityAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

        uint8_t operator[](EdgeId eid) const {
            return hg_->edges_[eid].arity;
        }
    };

    EdgeVertexAccessorRaw edge_vertex_accessor_raw() const {
        return EdgeVertexAccessorRaw(this);
    }

    EdgeArityAccessorRaw edge_arity_accessor_raw() const {
        return EdgeArityAccessorRaw(this);
    }

    // =========================================================================
    // State Management
    // =========================================================================

    // Create a new state from edge set
    StateId create_state(
        SparseBitset&& edge_set,
        uint32_t step = 0,
        uint64_t canonical_hash = 0,
        EventId parent_event = INVALID_ID
    );

    // Create state from edge IDs (convenience)
    StateId create_state(
        const EdgeId* edge_ids,
        uint32_t num_edges,
        uint32_t step = 0,
        uint64_t canonical_hash = 0,
        EventId parent_event = INVALID_ID
    );

    // Create state from initializer list (convenience)
    StateId create_state(std::initializer_list<EdgeId> edge_ids,
                         uint32_t step = 0,
                         uint64_t canonical_hash = 0,
                         EventId parent_event = INVALID_ID);

    // Get state by ID
    const State& get_state(StateId sid) const {
        // CRITICAL: Acquire fence to ensure we see all state data written by
        // the thread that created this state. Pairs with release fence in create_state.
        std::atomic_thread_fence(std::memory_order_acquire);
        return states_[sid];
    }

    State& get_state(StateId sid) {
        std::atomic_thread_fence(std::memory_order_acquire);
        return states_[sid];
    }

    // Get state's edge set
    const SparseBitset& get_state_edges(StateId sid) const {
        // CRITICAL: Acquire fence to ensure we see all state data written by
        // the thread that created this state. Pairs with release fence in create_state.
        std::atomic_thread_fence(std::memory_order_acquire);
        return states_[sid].edges;
    }

    // Get content-ordered hash for a state (for Automatic state canonicalization)
    // This is the same hash function used during evolution for state deduplication
    // in Automatic mode, ensuring consistency between evolution and display.
    uint64_t get_state_content_hash(StateId sid) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return compute_content_ordered_hash(states_[sid].edges);
    }

    // Number of states
    uint32_t num_states() const {
        return counters_.next_state.load(std::memory_order_relaxed);
    }

    // Get the genesis state ID (creates it lazily if needed)
    // The genesis state is an empty state (no edges) that serves as the origin
    // for all initial states via genesis events.
    StateId get_or_create_genesis_state();

    // Check if a state is the genesis state
    bool is_genesis_state(StateId sid) const {
        return genesis_state_init_.load(std::memory_order_acquire) == 2 && sid == genesis_state_;
    }

    // Check if an event is a genesis event (connects from genesis state to initial state)
    bool is_genesis_event(EventId eid) const {
        if (genesis_state_init_.load(std::memory_order_acquire) != 2) {
            return false;
        }
        if (eid >= events_.size()) {
            return false;
        }
        const Event& event = events_[eid];
        return event.input_state == genesis_state_;
    }

    // Get genesis state ID (returns INVALID_ID if not created)
    StateId genesis_state() const {
        if (genesis_state_init_.load(std::memory_order_acquire) == 2) {
            return genesis_state_;
        }
        return INVALID_ID;
    }

    // =========================================================================
    // Canonical State Deduplication
    // =========================================================================

    // Result of trying to create a canonical state
    struct CanonicalStateResult {
        StateId canonical_state_id;  // The canonical state ID (existing or new)
        StateId created_state_id;    // The state ID we created (always new, with actual edges)
        bool was_new;                // true if new canonical state, false if existing found
    };

    // Create state if no equivalent exists, otherwise return existing
    // This is the main API for state creation with canonicalization.
    // If Level 2 is enabled and a duplicate is found, edge correspondence is computed.
    //
    // Thread safety: Fully linearizable. We create the state first, then try to
    // insert into the canonical map. If another thread wins, we return their state
    // (the created state becomes "wasted" but this is correct).
    // canonical_hash is computed internally (mode-aware): the exact IR hash in Full
    // mode (reused as both identity and dedup key), the fast WL hash otherwise.
    // The optional incr_* delta (parent state + consumed/produced edges) lets the WL
    // hash be computed incrementally from the parent's cached history when
    // incremental WL is enabled; it is bit-identical, so dedup is unaffected.
    CanonicalStateResult create_or_get_canonical_state(
        SparseBitset&& edge_set,
        uint32_t step = 0,
        EventId parent_event = INVALID_ID,
        StateId incr_parent = INVALID_ID,
        const EdgeId* incr_consumed = nullptr, uint8_t incr_num_consumed = 0,
        const EdgeId* incr_produced = nullptr, uint8_t incr_num_produced = 0
    );


    // Lookup existing canonical state by hash (waits for concurrent inserts)
    std::optional<StateId> find_canonical_state(uint64_t canonical_hash) const {
        return canonical_state_map_.lookup_waiting(canonical_hash);
    }

    // Get the canonical representative for a given state
    // Behavior depends on state_canonicalization_mode_:
    // - None: returns raw_state (no canonicalization)
    // - Automatic/Full: returns cached canonical_id (may differ from raw_state)
    // NOTE: Uses acquire fence to ensure visibility of canonical_id on ARM64
    StateId get_canonical_state(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;
        if (state_canonicalization_mode_.load(std::memory_order_acquire) == StateCanonicalizationMode::None) {
            return raw_state;
        }
        // Acquire fence ensures we see the canonical_id write from create_or_get_canonical_state
        // This is critical for ARM64's weak memory model
        std::atomic_thread_fence(std::memory_order_acquire);
        const State& state = get_state(raw_state);
        return state.canonical_id;
    }

    // Get the canonical state for event canonicalization purposes.
    // Always uses the isomorphism-invariant hash (WL/IR) to find the canonical
    // representative, regardless of state_canonicalization_mode_.
    // This is needed for computing edge correspondence when state mode is None.
    StateId get_canonical_state_for_event(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;

        // Get the isomorphism-invariant hash for this state
        const State& state = get_state(raw_state);
        uint64_t hash = state.canonical_hash;

        // If hash is 0, the state's hash wasn't computed - fall back to raw state
        if (hash == 0) return raw_state;

        // Lookup in event_canonical_state_map_ which is always keyed by canonical_hash
        auto result = event_canonical_state_map_.lookup_waiting(hash);
        return result.value_or(raw_state);
    }

    // Get the canonical hash for a state (compute on-demand if not available)
    // This is used for event canonicalization, which needs isomorphism-invariant
    // state hashes regardless of whether state_canonicalization_mode_ is None.
    uint64_t get_or_compute_canonical_hash(StateId state_id);

    // Quotient exploration support. try_lower_explore_depth records a shorter path to a
    // canonical state, returning true only when it improved on what was known. Depth is a
    // shortest-path label, a property of the graph, so the set of states reachable within
    // the step budget does not depend on the order paths are found. try_claim_expanded
    // succeeds exactly once per canonical state, so its matches are computed once and the
    // matches-per-instance it records are well defined.
    bool try_lower_explore_depth(StateId canonical_id, uint32_t depth);
    bool try_claim_expanded(StateId canonical_id);

    // Current shortest known depth of a canonical state (INVALID_ID until first relaxed).
    // A child's arrival depth is derived from its parent's live minimum here, so that a
    // later shorter path to the parent pulls the child's subtree into budget even after the
    // parent was first expanded at a deeper claim depth.
    uint32_t explore_depth_of(StateId canonical_id) const;

    // Number of unique canonical states
    // Uses count_unique() for accurate counting after evolution completes,
    // handling the case where ConcurrentMap may have duplicate keys due to
    // concurrent insertions of the same canonical hash.
    size_t num_canonical_states() const {
        return canonical_state_map_.count_unique();
    }

    // =========================================================================
    // State Canonicalization Configuration
    // =========================================================================

    // State canonicalization mode: controls state deduplication strategy
    // Uses release semantics to ensure visibility to worker threads on ARM64
    void set_state_canonicalization_mode(StateCanonicalizationMode mode) {
        state_canonicalization_mode_.store(mode, std::memory_order_release);
    }

    // Uses acquire semantics to see updates from main thread on ARM64
    StateCanonicalizationMode state_canonicalization_mode() const {
        return state_canonicalization_mode_.load(std::memory_order_acquire);
    }

    // Select the WL approximate hash for compute_canonical_hash (fast hot path)
    void enable_wl_hash() {
        use_wl_hash_ = true;
    }

    // Select IR exact canonicalization for compute_canonical_hash
    void disable_wl_hash() {
        use_wl_hash_ = false;
    }

    // Whether compute_canonical_hash uses the WL approximate hash
    bool wl_hash_enabled() const {
        return use_wl_hash_;
    }

    // Full canonicalization mode: IR-based exact dedup, edge correspondence, and canonical output
    bool is_full_canonicalization() const {
        return state_canonicalization_mode_.load(std::memory_order_acquire) == StateCanonicalizationMode::Full;
    }

    // =========================================================================
    // Event Management
    // =========================================================================

    // Create a new event with optional canonicalization
    // Returns: (event_id, canonical_event_id, is_canonical)
    // - event_id: the ID of the created event
    // - canonical_event_id: for duplicate events, points to the first event with same signature
    // - is_canonical: true if this is a new canonical event, false if duplicate
    struct CreateEventResult {
        EventId event_id;
        EventId canonical_event_id;  // Same as event_id if is_canonical, otherwise first event
        bool is_canonical;
    };

    CreateEventResult create_event(
        StateId input_state,
        StateId output_state,
        RuleIndex rule_index,
        const EdgeId* consumed,
        uint8_t num_consumed,
        const EdgeId* produced,
        uint8_t num_produced
    );

    // Get event by ID
    const Event& get_event(EventId eid) const {
        return events_[eid];
    }

    Event& get_event(EventId eid) {
        return events_[eid];
    }

    // Number of events (returns canonical count when canonicalization enabled)
    uint32_t num_events() const {
        if (event_signature_keys_ != EVENT_SIG_NONE) {
            return canonical_event_count_.load(std::memory_order_acquire);
        }
        // Use acquire to synchronize with release stores in alloc_event
        return counters_.next_event.load(std::memory_order_acquire);
    }

    // Number of raw events (always returns total count)
    uint32_t num_raw_events() const {
        return counters_.next_event.load(std::memory_order_acquire);
    }

    // Iterate over canonical events only (skips duplicates)
    // Callback signature: void(EventId eid, const Event& event)
    template<typename Callback>
    void for_each_canonical_event(Callback&& callback) const {
        uint32_t count = num_raw_events();
        for (uint32_t eid = 0; eid < count; ++eid) {
            const Event& event = events_[eid];
            if (event.id == INVALID_ID) continue;
            if (!event.is_canonical()) continue;
            callback(eid, event);
        }
    }

    // Check if an event is canonical (not a duplicate)
    bool is_event_canonical(EventId eid) const {
        if (eid >= num_raw_events()) return false;
        return events_[eid].is_canonical();
    }

    // Get the canonical event ID for a raw event ID
    EventId get_canonical_event(EventId eid) const {
        if (eid >= num_raw_events()) return INVALID_ID;
        const Event& event = events_[eid];
        return event.is_canonical() ? eid : event.canonical_event_id;
    }

    // Event signature keys (bitflag controlling event equivalence)
    void set_event_signature_keys(EventSignatureKeys keys) {
        event_signature_keys_ = keys;
    }

    EventSignatureKeys event_signature_keys() const {
        return event_signature_keys_;
    }


    // =========================================================================
    // Index Access
    // =========================================================================

    const SignatureIndex& signature_index() const {
        return match_index_.signature_index();
    }

    const InvertedVertexIndex& inverted_index() const {
        return match_index_.inverted_index();
    }

    const PatternMatchingIndex& match_index() const {
        return match_index_;
    }

    // =========================================================================
    // Causal Graph Access
    // =========================================================================

    CausalGraph& causal_graph() { return causal_graph_; }
    const CausalGraph& causal_graph() const { return causal_graph_; }

    // Set edge producer: register `producer` as a producer of the canonical edge `key`
    // (mint keys with causal_edge_keys). raw_edge is the concrete edge id kept on the
    // CausalEdge record for viz.
    void set_edge_producer(CanonicalEdgeKey key, EventId producer, EdgeId raw_edge) {
        causal_graph_.set_edge_producer(key, producer, raw_edge);
    }

    // Mint the canonical edge key for each of the n `edges` belonging to `state`, writing
    // results into out. Under quotient (and Full canonicalization) the key is
    // fnv(canonical_hash(state), edge_orbit_in_state) -- iso-invariant, so every raw edge
    // instance of one canonical edge orbit maps to the same key regardless of which parent
    // produced it or which labeling a consumer matched. Otherwise (full multiway, or WL
    // mode) the key is the raw EdgeId, keeping isomorphic-but-distinct raw states' causal
    // edges disjoint. This is the ONLY place a CanonicalEdgeKey is minted from (state, edge).
    void causal_edge_keys(StateId state, const EdgeId* edges, uint32_t n,
                          CanonicalEdgeKey* out) const;

    // Compute the canonical edge-orbit table for `edges` and cache it under state id `s`,
    // returning the state's canonical hash (the same IR canonicalization serves both, so
    // this replaces the plain dedup hash in quotient mode at no extra canon cost).
    uint64_t compute_and_cache_state_orbits(StateId s, const SparseBitset& edges);

    // The cached edge-orbit table for a state (null if not computed -- e.g. full-capture
    // mode, or before canonicalization).
    const EdgeOrbitTable* state_orbits(StateId s) const {
        auto r = state_orbit_tables_.lookup(static_cast<uint64_t>(s) + 1);  // +1: key 0 is the map's EMPTY sentinel
        return r.has_value() ? *r : nullptr;
    }

    // Capture the canonical transition an event realizes into the quotient causal skeleton
    // (idempotent per distinct canonical transition). No-op if either endpoint's orbit
    // table is missing. Quotient mode only.
    void register_quotient_transition(EventId e);

    // Seed the quotient causal reconstruction at an initial state (depth 0): mark it
    // reachable and give each of its edge orbits the sentinel INIT producer (INVALID_ID,
    // skipped at emission -- initial edges have no producer). max_steps bounds the depth.
    void quotient_causal_seed(StateId initial_state, int max_steps);

    // Visit the distinct canonical transitions out of the canonical state `from_hash`.
    template <typename F>
    void for_each_transition_from(uint64_t from_hash, F&& f) const {
        auto r = transitions_from_.lookup(from_hash);
        if (r.has_value()) (*r)->for_each([&](const CanonicalTransition& t) { f(t); });
    }

    // Visit every match of the expanded representative of the canonical state `from_hash`,
    // in slots and undeduplicated -- the input to the per-instance raw reconstruction.
    template <typename F>
    void for_each_expansion_match(uint64_t from_hash, F&& f) const {
        auto r = qc_expansion_.lookup(from_hash);
        if (r.has_value()) (*r)->for_each([&](const SlotMatch& m) { f(m); });
    }

    // Per-instance raw reconstruction: replays the captured expansion against every raw
    // instance so quotient mode can report the raw observables it never explores. Off by
    // default while it is proven out against full-capture.
    void set_quotient_reconstruction(bool on) {
        quotient_reconstruction_.store(on, std::memory_order_relaxed);
    }
    bool quotient_reconstruction() const {
        return quotient_reconstruction_.load(std::memory_order_relaxed);
    }
    // Raw observables recovered by the reconstruction (the full-capture counts).
    size_t num_reconstructed_events() const {
        return qc_next_raw_event_.load(std::memory_order_relaxed);
    }
    size_t num_reconstructed_causal_edges() const {
        return qc_num_causal_edges_.load(std::memory_order_relaxed);
    }
    // TR-off view: every distinct (producer, consumer). TR-on view: those tagged in-reduction.
    size_t num_reconstructed_causal_pairs(bool transitively_reduced = false) const {
        return (transitively_reduced ? qc_num_tr_pairs_ : qc_num_causal_pairs_)
                   .load(std::memory_order_relaxed);
    }
    size_t num_reconstructed_branchial() const {
        return qc_num_branchial_.load(std::memory_order_relaxed);
    }
    size_t num_frame_alignment_disagreements() const {
        return qc_frame_disagree_.load(std::memory_order_relaxed);
    }
    size_t num_alignment_failures() const { return qc_align_fail_.load(std::memory_order_relaxed); }
    size_t num_bad_correspondences() const { return qc_align_badcorr_.load(std::memory_order_relaxed); }

    // Visit the reconstructed causal relation as pairs of isomorphism-invariant event
    // signatures. `reduced` selects the view: false walks every recorded pair (TR off), true
    // walks only those tagged in-reduction (TR on). Both come from the same online base, so
    // either view is available in any order at no extra cost.
    template <typename F>
    void for_each_reconstructed_causal(bool reduced, F&& f) const {
        auto sig = [&](uint32_t e) -> uint64_t {
            const uint64_t* s = qc_event_sig_.get(e);
            return s ? *s : 0;
        };
        if (reduced) {
            qc_preds_.for_each([&](uint64_t k, LockFreeList<uint32_t>* lst) {
                const uint32_t c = static_cast<uint32_t>(k - 1);
                lst->for_each([&](uint32_t p) { f(sig(p), sig(c)); });
            });
        } else {
            qc_causal_pairs_.for_each([&](uint64_t k, bool) {
                f(sig(static_cast<uint32_t>(k >> 32)), sig(static_cast<uint32_t>(k & 0xFFFFFFFFu)));
            });
        }
    }

    // ==========================================================================
    // Observables (SPEC section 5)
    // ==========================================================================
    // The engine reaches the same observable two ways: full-capture explores every raw state,
    // quotient explores one per isomorphism class and reconstructs the rest. These accessors
    // hide that choice. They are deliberately NOT the num_events()/causal_graph() accessors,
    // which report what is MATERIALISED -- internal code iterates records by id against those,
    // and would break if they started reporting counts with no records behind them.

    size_t observable_num_events() const {
        return quotient_reconstruction() ? num_reconstructed_events() : num_events();
    }
    size_t observable_num_causal_edges() const {
        return quotient_reconstruction() ? num_reconstructed_causal_edges()
                                         : causal_graph_.num_causal_edges();
    }
    size_t observable_num_causal_pairs(bool transitively_reduced) const {
        return quotient_reconstruction() ? num_reconstructed_causal_pairs(transitively_reduced)
                                         : causal_graph_.num_causal_event_pairs();
    }
    size_t observable_num_branchial() const {
        return quotient_reconstruction() ? num_reconstructed_branchial()
                                         : causal_graph_.num_branchial_edges();
    }

    // Get a representative edge producer for a canonical edge key (INVALID_ID if none).
    EventId get_edge_producer(CanonicalEdgeKey key) const {
        return causal_graph_.get_edge_producer(key);
    }

    // Add edge consumer: register `consumer` as a consumer of the canonical edge `key`.
    void add_edge_consumer(CanonicalEdgeKey key, EventId consumer, EdgeId raw_edge) {
        causal_graph_.add_edge_consumer(key, consumer, raw_edge);
    }

    // Carry a surviving edge's producers from its parent-state orbit key to its
    // child-state orbit key (see CausalGraph::propagate_producers).
    void propagate_producers(CanonicalEdgeKey from, CanonicalEdgeKey to, EdgeId raw_edge) {
        causal_graph_.propagate_producers(from, to, raw_edge);
    }

    // Whether causal edges are keyed by canonical edge orbit (quotient exploration). Set
    // by the evolution engine before evolving; read when minting causal edge keys.
    void set_quotient_causal(bool q) { quotient_causal_.store(q, std::memory_order_relaxed); }
    bool quotient_causal() const { return quotient_causal_.load(std::memory_order_relaxed); }

    // Create a genesis event for an initial state.
    // This synthetic event connects the empty genesis state to the initial state.
    // It "produces" all edges in the initial state, enabling causal tracking from gen 0.
    // Returns the genesis event ID.
    EventId create_genesis_event(StateId initial_state, const EdgeId* edges, uint8_t num_edges);

    // Register event for branchial tracking
    // When event canonicalization is enabled, uses edge equivalence for overlap detection
    // and skips branchial edges between canonically equivalent events
    void register_event_for_branchial(
        EventId event,
        StateId input_state,
        const EdgeId* consumed_edges,
        uint8_t num_consumed,
        EventId canonical_event = INVALID_ID  // Pass canonical_event_id for deduplication
    );

    // Get causal/branchial statistics
    size_t num_causal_edges() const { return causal_graph_.num_causal_edges(); }
    size_t num_causal_event_pairs() const { return causal_graph_.num_causal_event_pairs(); }
    size_t num_branchial_edges() const { return causal_graph_.num_branchial_edges(); }

    // =========================================================================
    // Arena Access
    // =========================================================================

    ConcurrentHeterogeneousArena& arena() { return arena_; }
    const ConcurrentHeterogeneousArena& arena() const { return arena_; }

    // =========================================================================
    // Counter Access
    // =========================================================================

    GlobalCounters& counters() { return counters_; }
    const GlobalCounters& counters() const { return counters_; }

    // =========================================================================
    // Utility
    // =========================================================================

    // Compute simple hash for a state's edge set (fast but not isomorphism-invariant)
    static uint64_t compute_state_hash(const SparseBitset& edges) {
        uint64_t h = 14695981039346656037ULL;
        edges.for_each([&](EdgeId eid) {
            h ^= eid;
            h *= 1099511628211ULL;
        });
        return h;
    }

    // Compute content-ordered hash for Automatic state canonicalization mode
    // Hashes edge contents in order by edge ID: (arity, v1, v2, ...) for each edge
    // Fast but not isomorphism-invariant.
    uint64_t compute_content_ordered_hash(const SparseBitset& edges) const;

    // Compute canonical hash (isomorphism-invariant).
    // With the WL hash enabled (use_wl_hash_), uses the fast approximate hash;
    // otherwise falls back to IR exact canonicalization.
    uint64_t compute_canonical_hash(const SparseBitset& edges) const;

    // Compute the Weisfeiler-Leman approximate canonical hash for a set of
    // edges. This is the fast hot-path hash backing compute_canonical_hash (in
    // WL mode), the per-state canonical_hash recorded during evolution, and the
    // isomorphism-invariant key for event canonicalization.
    uint64_t compute_wl_hash(const SparseBitset& edges) const;


    // Find edge correspondence between two isomorphic states. Uses IR in Full
    // canonicalization mode, WL subtree hashes otherwise.
    // Returns mapping from state1 edges to state2 edges.
    EdgeCorrespondence find_edge_correspondence_dispatch(
        const SparseBitset& state1_edges,
        const SparseBitset& state2_edges
    ) const;

    // Count edges in a state
    uint32_t count_state_edges(StateId sid) const {
        uint32_t count = 0;
        states_[sid].edges.for_each([&](EdgeId) {
            count++;
        });
        return count;
    }
};

}  // namespace hypergraph
