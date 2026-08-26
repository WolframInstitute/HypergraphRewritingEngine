#pragma once
#include "hgcommon/transitive_reduction.hpp"
#include "hgcommon/namespace.hpp"

#include <cstdint>
#include <cstring>
#include <atomic>
#include <vector>
#include <memory>
#include <unordered_map>

#include "types.hpp"
#include "atomic_compat.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "index.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "hgcommon/quotient_causal_core.hpp"
#include "hgcommon/quotient_replay_core.hpp"
#include "lock_free_list.hpp"
#include "causal_graph.hpp"
#include "wl_hash.hpp"
#include "concurrent_map.hpp"
#include "concurrent_key_set.hpp"

// Shared types: CanonicalizationResult, CanonicalForm, VertexMapping
#include "canonical_types.hpp"

namespace HG_NAMESPACE {
namespace engine {

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
    ConcurrentMap<uint64_t, StateId, uint64_t{0}, ~uint64_t{0}, INVALID_ID> canonical_state_map_;

    // Event canonicalization state map, keyed by canonical_hash rather than by the state
    // mode's dedup key, so event identity does not follow the state-merging choice.
    // Used by event signature computation to find canonical representatives for edge
    // correspondence when state_canonicalization_mode_ is None or Automatic.
    //
    // The key is the EXACT IR invariant in every state mode whenever event canonicalization is
    // on (compute_reported_canonical_hash), so the event identity resolved through this map is
    // the same identity whatever the state mode -- mode_matrix_probe measures identical event
    // counts down every state-mode column. SPEC.md sec 4 states the axes independent, and for
    // EVENTS that holds. The CAUSAL relation under Automatic event identity is the
    // canonical-class relation only when the orbit tables exist (the Full state mode computes
    // them); in the other state modes the engine warns and serves the raw-edge rendezvous.
    ConcurrentMap<uint64_t, StateId, uint64_t{0}, ~uint64_t{0}, INVALID_ID> event_canonical_state_map_;

    // State canonicalization mode: controls how states are deduplicated, via the map_key
    // create_or_get_canonical_state builds -- which is a DIFFERENT quantity from the
    // canonical_hash it reports.
    //   None:      dedup key is the raw state id, so nothing merges
    //   Automatic: dedup key is compute_content_ordered_hash -- merges states with identical
    //              edge content, which is NOT isomorphism-invariant
    //   Full:      dedup key is the exact IR hash -- merges isomorphic states
    // NOTE: Must be atomic for ARM64 memory ordering - ensures visibility to worker threads
    std::atomic<StateCanonicalizationMode> state_canonicalization_mode_{StateCanonicalizationMode::None};

    // Whether the evolution quotients isomorphic states (explore-from-canonical-only). When
    // set, causal edges are keyed by canonical edge orbit so attribution is schedule-
    // independent across the labelings by which parents reach one canonical state.
    std::atomic<bool> quotient_causal_{false};
    std::atomic<bool> record_causal_{true};
    std::atomic<bool> record_branchial_{true};
    std::atomic<bool> record_state_events_{true};
    std::atomic<bool> record_raw_events_{true};

    // Per-state canonical edge-orbit tables, computed once at state canonicalization in
    // quotient mode (piggybacked on the dedup IR canonicalization, so no extra canon pass)
    // and cached by state id. The quotient causal reconstruction reads edge orbits from
    // here rather than recomputing per event (which would re-run IR canonicalization on
    // every event -- catastrophic on high-automorphism states). Key: StateId as uint64_t.
    ConcurrentMap<uint64_t, EdgeOrbitTable*> state_orbit_tables_;

    // Canonical RANK of each edge of a state, built once per state when event
    // canonicalization is on and read by every event that consumes or produces one of those
    // edges. Edges ascend (SparseBitset iterates in id order), so a lookup binary-searches.
    ConcurrentMap<uint64_t, EdgeRankTable*> state_edge_rank_tables_;

    // The captured quotient causal skeleton: the distinct canonical transitions out of each
    // canonical state (keyed by the source state's canonical hash), plus a dedup set over
    // transition signatures. Built online as events fire in quotient mode; the depth-indexed
    // producer-set reconstruction propagates over it.
    ConcurrentMap<uint64_t, LockFreeList<CanonicalTransition>*> transitions_from_;
    ConcurrentKeySet<uint64_t> seen_transitions_;

    // Depth-indexed producer-set reconstruction (the online form of the validated DP).
    // qc_dsup_ maps key(state_hash, depth, orbit) -> set of producer canonical-event ids
    // (append-only); qc_dsup_seen_ dedups (key, producer); qc_reached_ marks (state_hash,
    // depth). Producers cascade forward monotonically as transitions and reachability are
    // discovered, emitting causal edges into causal_graph_. Bounded by qc_max_steps_.
    ConcurrentMap<uint64_t, LockFreeList<EventId>*> qc_dsup_;
    ConcurrentKeySet<uint64_t> qc_dsup_seen_;
    ConcurrentKeySet<uint64_t> qc_reached_;
    // The same points qc_reached_ marks, enumerable. The map's key mixes the hash and the
    // depth irreversibly, and raising the depth budget has to revisit the points that stood
    // at the old terminal depth: each was marked reached, but every transition out of it was
    // declined by the bound, so its producers and instances are recorded and unexpanded.
    struct QcReachPoint { uint64_t state_hash; uint32_t depth; };
    LockFreeList<QcReachPoint> qc_reached_list_;
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
    ConcurrentMap<uint64_t, uint64_t, uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}> qc_frame_sig_;       // StateId + 1 -> slot-vector hash
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
    // The slot-has-no-producer sentinel, from hgcommon: the replay core writes it into a
    // child's producer vector and this class reads it back, so one value or neither works.
    static constexpr uint32_t QC_NO_PRODUCER = hgcommon::QR_NO_PRODUCER;
    ConcurrentMap<uint64_t, LockFreeList<QcInstance>*> qc_instances_;   // key(hash,depth,0)
    // Claims a (instance, match) application. Both the instance side and the match side drive
    // the rendezvous, and unlike the producer-set DP an application is NOT idempotent -- each
    // one emits a raw event -- so the pair must be claimed exactly once. O(raw) entries.
    ConcurrentKeySet<uint64_t> qc_applied_;
    // Claims an unordered branchial pair {instance, match a, match b}. Both members of a pair
    // can see each other, so the pair is claimed directly rather than a reporter being elected.

    // The matches already applied to one instance, indexed by dense instance id. Branchial
    // pairing scans THIS, not the expansion list, and that is what makes the pairing provable
    // rather than merely observed to work.
    //
    // Each application pushes here before it scans. A push is a release CAS on the list head;
    // the scan is an acquire load of the same head. A thread's load after its own successful
    // CAS cannot return a value earlier in that head's modification order than its own node,
    // and the stack's prev chain holds every node pushed before it. So of two applications,
    // whichever pushed LATER in the head's modification order necessarily sees the earlier
    // one -- no appeal to timeliness, only to modification-order coherence on one atomic.
    // Scanning the expansion list instead gave no such guarantee: the two sides read a
    // structure neither had written, so nothing ordered their reads against each other.
    //
    // A SegmentedArray, not a ConcurrentMap: instance ids are dense, so a direct-indexed slot
    // has no resize chain and no sentinel-key domain, and get_or_default hands both threads
    // the same list object. A map could hand them two lists during a resize window, which
    // would put the two sides on different heads and void the argument above.
    struct QcAppliedMatch {
        uint32_t id;
        uint32_t event;                   // the raw event this application minted
        uint32_t num_consumed;
        const uint32_t* consumed_slots;   // arena-backed, stable

        uint32_t consumed(uint32_t j) const;
    };
    SegmentedArray<LockFreeList<QcAppliedMatch>> qc_inst_applied_;
    std::atomic<uint32_t> qc_next_instance_{0};
    std::atomic<uint32_t> qc_next_raw_event_{0};
    // Reconstructed events under the RUN'S event identity, as opposed to the raw count above.
    // qc_event_sig_ carries a fixed (input, output, rule) triple, which is its own identity and
    // not the one the caller selected -- EVENT_SIG_FULL keys on the endpoint states alone,
    // EVENT_SIG_AUTOMATIC adds the step and the canonical ranks. Under an identity mode the
    // observable is the count of DISTINCT identities, so the mode's signature is computed here
    // and the distinct ones counted.
    ConcurrentKeySet<uint64_t> qc_canon_event_seen_{4096};
    std::atomic<size_t> qc_num_canon_events_{0};
    std::atomic<bool> quotient_reconstruction_{false};

    // Reconstructed causal relation over raw event ids: THE base, and the only thing stored.
    // Both views come from it -- TR-off enumerates it, TR-on reduces it under hgcommon::tr_reduce
    // -- so neither view depends on the order the pairs arrived in.
    //
    // The set is what makes that true. A pair (p,c) is in the reduction exactly when no longer
    // path bypasses it, which is a property of the finished relation; deciding it as each pair
    // lands answers against the pairs seen so far and gets a different answer depending on
    // whether the bypassing path arrived first.
    ConcurrentKeySet<uint64_t> qc_causal_pairs_;                    // distinct (producer, consumer)
    // Isomorphism-invariant signature per reconstructed event: fnv(from hash, to hash, rule).
    // Reconstructed events carry no Event record, so this is the only description they have --
    // it is what schedule-independence is fingerprinted on, and what a graph over reconstructed
    // events is built from. Held as the three COMPONENTS rather than their hash: the hash
    // identifies an event and cannot describe one, and a vertex needs its endpoints.
    SegmentedArray<QcEventContent> qc_event_sig_;
    // The same events under the RUN'S event identity, indexed the same way. The pair accessors
    // need this and not qc_event_sig_: a caller comparing the reconstructed causal or branchial
    // relation against full capture is comparing against Event::signature, which is the run's
    // identity, so emitting the internal triple instead compares two different functions and
    // every pair looks like a disagreement. Left at 0 when no identity mode is selected, which
    // is what full capture leaves Event::signature at in that case.
    SegmentedArray<uint64_t> qc_event_runsig_;
    std::atomic<size_t> qc_num_causal_edges_{0};   // per consumed edge (the T1 multiset)
    std::atomic<size_t> qc_num_causal_pairs_{0};   // distinct pairs, un-reduced view
    // Scans of an instance's applied list, and elements visited across them. visits/scans is
    // the mean fan-out m; pairs are bounded by sum m(m-1)/2 while the SCAN costs sum m^2.
    // CAPTURES DROPPED BECAUSE AN ENDPOINT'S ORBITS WERE NOT THERE YET.
    //
    // qc_capture_expansion needs both endpoints' EdgeOrbitTable and its slot array to record a
    // match in frame slots. Both are filled when the state is canonicalized, and this runs on
    // the event, so whether this thread OBSERVES them is a question about publication order --
    // schedule-dependent, and the match is dropped when the answer is no.
    //
    // It was not counted. A drop here removes a match from the capture, so the replay never
    // applies it to any instance: fewer applications, hence fewer causal and branchial pairs,
    // while the canonical state and event counts are untouched because the state was still
    // explored. That is the exact shape of the intermittent quotient determinism failures, and
    // every instrument reported zero because nothing was looking here.
    mutable std::atomic<size_t> qc_capture_no_orbits_{0};
    // The class already has a representative raw state and it is not this one. BY DESIGN -- one
    // raw state per class captures -- but counted, because "by design" and "measured" are
    // different claims and only one of them was on record.
    mutable std::atomic<size_t> qc_capture_not_rep_{0};
    mutable std::atomic<size_t> qc_applied_scans_{0};
    mutable std::atomic<size_t> qc_applied_visits_{0};
    std::atomic<size_t> qc_num_branchial_{0};      // sibling matches of one instance, overlapping
    void qc_record_causal(uint32_t producer, uint32_t consumer);

    // An ordered pair of event ids as one map key, for the causal and branchial pair sets.
    //
    // Both ids are offset by one before packing, which makes the key INJECTIVE and never zero:
    // the high word is at least 1, and ConcurrentMap reserves 0 as EMPTY. Packing raw and
    // nudging a zero result to 1 instead -- which is what the causal site did -- collides pair
    // (0,0) with pair (0,1), and insert_if_absent then drops the second as already present.
    //
    // Ids are engine-minted and bounded well below INVALID_ID, so neither offset can wrap and
    // the key cannot reach the LOCKED sentinel either.
    static uint64_t qc_pair_key(uint32_t a, uint32_t b);

    // The DP's key spaces come from hgcommon so the device indexes the same ones.
    static uint64_t qc_key(uint64_t state_hash, uint32_t depth, uint32_t orbit);
    static uint64_t qc_rkey(uint64_t state_hash, uint32_t depth);

    LockFreeList<EventId>* qc_dsup_list(uint64_t key);

    // The storage face hgcommon/quotient_causal_core.hpp drives. It supplies WHERE things are
    // held and nothing else -- when a point is entered, what a producer landing does, and which
    // rendezvous scan follows which publish are in the core, which is the same body the device
    // runs. Nested so it reaches this class's private state without a friend declaration.
    struct QcCtx {
        using Transition = CanonicalTransition;
        Hypergraph& hg;
        uint32_t steps;

        uint32_t max_steps() const;
        // The host recurses on the ordinary stack, which is heap-sized here, so no depth is out
        // of reach and the cascade is bounded by max_steps alone.
        bool enter(uint32_t) const;
        // The DP's depth-advancing edges. The host descends by CALLING, on a thread stack with
        // about a thousand levels of room at the tightest platform it ships on against a bound
        // of `steps` -- so it has nothing to gain from a worklist and nothing to prove by one.
        void defer_reach(uint64_t state_hash, uint32_t depth);
        void defer_producer(uint64_t state_hash, uint32_t depth, uint32_t orbit,
                            uint32_t producer);
        bool mark_reached(uint64_t rkey, uint64_t state_hash, uint32_t depth);
        bool mark_producer_seen(uint64_t seen_key);
        void push_producer(uint64_t key, uint32_t producer);
        template <class F>
        void for_each_producer(uint64_t key, F&& f) {
            auto r = hg.qc_dsup_.lookup(key);
            if (r.has_value()) (*r)->for_each([&](EventId p) { f(p); });
        }
        template <class F>
        void for_each_transition_from(uint64_t hash, F&& f) {
            hg.for_each_transition_from(hash, [&](const CanonicalTransition& t) { f(t); });
        }
        void emit(uint32_t producer, uint32_t consumer);
        void fence();
    };
    QcCtx qc_ctx();

    // The storage face hgcommon/quotient_replay_core.hpp drives. Same division as QcCtx above:
    // WHERE a producer vector, an applied list or a claim set lives is here; what an
    // application DOES -- what it claims, what it identifies the event by, which causal and
    // branchial relations follow -- is in the core, which is the body the device runs too.
    struct QrCtx {
        using Instance = QcInstance;
        using Match    = SlotMatch;
        using Applied  = QcAppliedMatch;
        Hypergraph& hg;
        // Branchial pairs counted into a LOCAL and published once, by the destructor, at the
        // end of the apply this context was made for. Incremented in place it is one atomic
        // read-modify-write per pair on a counter every worker shares -- 133,218,996 of them
        // on disc-l3a2g2r2 depth 3, against 970,584 applies -- so the line carrying it moves
        // between cores once per pair. The published total is identical.
        size_t branchial_seen = 0;
        ~QrCtx();

        bool claim(uint64_t apply_key);
        uint32_t mint_event();
        void record_content(uint32_t ev, uint64_t from_class, uint64_t to_class, uint32_t rule);
        hgcommon::EventSignatureKeys keys() const;
        uint32_t frame_step(uint64_t class_hash, uint32_t fallback) const;
        // Kept per event as well as counted, so the causal and branchial accessors report the
        // relation under the identity the CALLER selected -- reporting the internal triple
        // instead makes every pair look like a disagreement with full capture.
        void record_runsig(uint32_t ev, uint64_t csig);
        bool want_causal() const;
        bool want_branchial() const;
        uint32_t producer_at(const QcInstance& inst, uint32_t slot) const;
        void record_causal(uint32_t producer, uint32_t consumer);
        using AppliedRef = const LockFreeList<QcAppliedMatch>::Node*;
        static bool applied_ref_valid(AppliedRef r);
        AppliedRef publish_applied(const QcInstance& inst, const SlotMatch& m, uint32_t ev);
        template <class F>
        void for_each_applied_before(const QcInstance& inst, AppliedRef mine, F&& f) {
            // THE FAN-OUT, counted. The scan visits the applications published before this one
            // and tests slot overlap inside, so its cost is m per application and m^2/2 per
            // instance. Measured on disc-l3a2g2r2 depth 3: 970,584 scans, 163,228,620 visits,
            // 133,218,996 pairs -- 81.6% of visits emit, so the visits are not the waste and an
            // inverted index by (instance, slot) would save 18% rather than a quadratic.
            //
            // The visit count is summed in a LOCAL and published once per scan. Incremented in
            // the loop it is one atomic read-modify-write per visit on a counter every worker
            // shares -- 163,228,620 of them on the workload above against 970,584 scans -- so
            // the line carrying it moves between cores once per visit and the instrument costs
            // more than the scan it measures. Per scan the published total is identical.
            hg.qc_applied_scans_.fetch_add(1, std::memory_order_relaxed);
            size_t visits = 0;
            hg.qc_inst_applied_.get_or_default(inst.id, hg.arena_).for_each_before(
                mine, [&](const QcAppliedMatch& a) {
                    ++visits;
                    f(a);
                });
            hg.qc_applied_visits_.fetch_add(visits, std::memory_order_relaxed);
        }
        void record_branchial_pair(uint32_t lo, uint32_t hi);
        void descend(const SlotMatch& m, uint32_t depth, uint32_t ev, const QcInstance& parent);
    };
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
    ConcurrentMap<uint64_t, EventId, uint64_t{0}, ~uint64_t{0}, INVALID_ID> canonical_event_map_;
    std::atomic<uint32_t> canonical_event_count_{0};

    // Times an event signature used a RAW edge id because no edge correspondence was found.
    // Such a signature is not an isomorphism invariant, so a non-zero count means the event
    // set is approximate; see the fallback in create_event.
    std::atomic<uint64_t> event_sig_raw_fallbacks_{0};

    // Times a canonical hash was actually COMPUTED, against the number of states that hold
    // one. Both are needed: the ratio is the question, and a raw call count says nothing
    // without the denominator.
    //
    // Incremented at the LEAVES ONLY -- compute_exact_canonical_hash, compute_wl_hash,
    // compute_and_cache_state_orbits. NOT at compute_canonical_hash, which is a dispatcher
    // that tail-calls one of them: counting it too counted one canonicalization twice and
    // reported a uniform 2.00 per state on all 17 cost_matrix cases. The uniformity was the
    // tell -- real duplication varies with the workload, an off-by-a-factor does not.
    //
    // Why it exists: a 2026-07-25 profile recorded "IR canonicalization up to 3x per state"
    // and named it the biggest measurable win. get_or_compute_canonical_hash
    // has memoized into State::canonical_hash since, so the steady state is one computation per
    // state plus whatever racing writers duplicate -- and NOTHING MEASURED THAT. A number that
    // cannot be re-derived is not a number to plan against.
    mutable std::atomic<uint64_t> canonical_hash_computations_{0};
    EventSignatureKeys event_signature_keys_{EVENT_SIG_NONE};
    std::atomic<bool> positional_event_identity_{false};

    // Genesis state: the empty state (no edges) from which all initial states originate
    // Created lazily on first call to get_or_create_genesis_state()
    // Uses lock-free initialization: 0=uninit, 1=in_progress, 2=done
    std::atomic<StateId> genesis_state_{INVALID_ID};

public:
    Hypergraph();

    // Non-copyable
    Hypergraph(const Hypergraph&) = delete;
    Hypergraph& operator=(const Hypergraph&) = delete;

    // =========================================================================
    // Vertex Management
    // =========================================================================

    // Allocate a new vertex ID
    VertexId alloc_vertex();

    // Allocate N consecutive vertex IDs
    VertexId alloc_vertices(uint32_t count);

    // Get current vertex count (upper bound)
    uint32_t num_vertices() const;

    // Ensure vertex ID space is at least `max_id + 1`
    void reserve_vertices(VertexId max_id);

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
    const Edge& get_edge(EdgeId eid) const;
    Edge& get_edge(EdgeId eid);

    // Edge accessor (for pattern matching)
    // STAYS IN THE HEADER, and it is the only body in this class that does: the return type is
    // a lambda's closure type, which is deduced from the body, so a caller cannot name it and
    // the definition has to be visible.
    auto edge_accessor() const {
        return [this](EdgeId eid) -> const Edge& { return edges_[eid]; };
    }

    // Number of edges
    uint32_t num_edges() const;

    // PUBLISHED edges, the bound for enumeration. See num_published_states for why the claim
    // counter above is not that bound.
    uint32_t num_published_edges() const;

    // =========================================================================
    // Edge Accessors for the WL hash
    // =========================================================================
    // These provide the interface needed by WLHash::compute_state_hash*()

    // Get vertex array for an edge (returns pointer to vertices)
    const VertexId* edge_vertices(EdgeId eid) const;

    // Get arity of an edge
    uint8_t edge_arity(EdgeId eid) const;

    // Get cached signature for an edge (computed once at creation)
    const EdgeSignature& edge_signature(EdgeId eid) const;

    // Lightweight accessor for the WL hash that provides pointer indexing
    // Returns pointer to edge's inline vertex array - no copying or allocation
    class EdgeVertexAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeVertexAccessorRaw(const Hypergraph* hg);
        const VertexId* operator[](EdgeId eid) const;
    };

    // Direct arity accessor - reads from struct field, O(1)
    class EdgeArityAccessorRaw {
        const Hypergraph* hg_;
    public:
        explicit EdgeArityAccessorRaw(const Hypergraph* hg);
        uint8_t operator[](EdgeId eid) const;
    };

    EdgeVertexAccessorRaw edge_vertex_accessor_raw() const;
    EdgeArityAccessorRaw edge_arity_accessor_raw() const;

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
    const State& get_state(StateId sid) const;
    State& get_state(StateId sid);

    // Get state's edge set
    const SparseBitset& get_state_edges(StateId sid) const;

    // Get content-ordered hash for a state (for Automatic state canonicalization)
    // This is the same hash function used during evolution for state deduplication
    // in Automatic mode, ensuring consistency between evolution and display.
    uint64_t get_state_content_hash(StateId sid) const;

    // Number of states
    uint32_t num_states() const;

    // How many states are PUBLISHED, as against how many ids have been CLAIMED.
    //
    // num_states() above is the claim counter, and it runs ahead: an id is taken by an atomic
    // increment before the state is constructed, and an id whose state is never emplaced --
    // claimed and then abandoned -- leaves the counter permanently above what exists. A reader
    // that loops to num_states() and indexes therefore reaches slots that hold no element, and
    // SegmentedArray's guard throws rather than hand back arena default bytes.
    //
    // This is the bound for enumerating states. It is exact once the workers are quiescent,
    // which is the only time enumeration is meaningful anyway: mid-run the array is being
    // written and no bound is stable.
    uint32_t num_published_states() const;

    // Get the genesis state ID (creates it lazily if needed)
    // The genesis state is an empty state (no edges) that serves as the origin
    // for all initial states via genesis events.
    StateId get_or_create_genesis_state();

    // Check if a state is the genesis state. INVALID_ID until one is published, and no
    // state id equals INVALID_ID, so the comparison alone answers both questions.
    bool is_genesis_state(StateId sid) const;

    // Check if an event is a genesis event (connects from genesis state to initial state)
    bool is_genesis_event(EventId eid) const;

    // Get genesis state ID (returns INVALID_ID if not created)
    StateId genesis_state() const;

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
    std::optional<StateId> find_canonical_state(uint64_t canonical_hash) const;

    // Get the canonical representative for a given state
    // Behavior depends on state_canonicalization_mode_:
    // - None: returns raw_state (no canonicalization)
    // - Automatic/Full: returns cached canonical_id (may differ from raw_state)
    // NOTE: Uses acquire fence to ensure visibility of canonical_id on ARM64
    StateId get_canonical_state(StateId raw_state) const;

    // Get the canonical state for event canonicalization purposes.
    // Always uses the isomorphism-invariant hash (WL/IR) to find the canonical
    // representative, regardless of state_canonicalization_mode_.
    // This is needed for computing edge correspondence when state mode is None.
    StateId get_canonical_state_for_event(StateId raw_state) const;

    // Get the canonical hash for a state (compute on-demand if not available)
    // This is used for event canonicalization, which needs isomorphism-invariant
    // state hashes regardless of whether state_canonicalization_mode_ is None.
    uint64_t get_or_compute_canonical_hash(StateId state_id);

    // Build the state's canonical rank table and return the exact canonical hash from the
    // SAME individualization-refinement pass -- the event path needs both, and running IR
    // twice for them is the difference between one pass per state and two per event.
    uint64_t cache_state_edge_ranks(StateId state_id, const SparseBitset& edges);

    // cache_state_edge_ranks, skipped when the table is already there. cache_ runs a full IR
    // pass every call and only then discards the result on a losing insert, so a caller that
    // may ask repeatedly for the same state -- a sampler keyed on canonical ranks does, once
    // per match -- must ask through this instead.
    void ensure_state_edge_ranks(StateId state_id, const SparseBitset& edges);

    // Canonical rank of `edge` within `state`, or UINT32_MAX when the state has no table.
    uint32_t edge_rank_in_state(StateId state_id, EdgeId edge) const;

    // Event signatures that fell back to a raw edge id. Non-zero means the event identity is
    // approximate rather than canonical.
    uint64_t event_signature_raw_fallbacks() const;

    // How many times a reported canonical hash was computed. Divide by the state count for the
    // per-state figure; anything above 1.0 is duplication, and under contention a small excess
    // is expected rather than a defect (racing writers compute the same value and the last
    // store wins).
    uint64_t canonical_hash_computations() const;

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
    size_t num_canonical_states() const;

    // =========================================================================
    // State Canonicalization Configuration
    // =========================================================================

    // State canonicalization mode: controls state deduplication strategy
    // Uses release semantics to ensure visibility to worker threads on ARM64
    void set_state_canonicalization_mode(StateCanonicalizationMode mode);

    // Uses acquire semantics to see updates from main thread on ARM64
    StateCanonicalizationMode state_canonicalization_mode() const;

    // Select the WL approximate hash for compute_canonical_hash (fast hot path)
    void enable_wl_hash();

    // Select IR exact canonicalization for compute_canonical_hash
    void disable_wl_hash();

    // Whether compute_canonical_hash uses the WL approximate hash
    bool wl_hash_enabled() const;

    // Full canonicalization mode: IR-based exact dedup, edge correspondence, and canonical output
    bool is_full_canonicalization() const;

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
    const Event& get_event(EventId eid) const;
    Event& get_event(EventId eid);

    // Number of events (returns canonical count when canonicalization enabled)
    uint32_t num_events() const;

    // Number of raw events (always returns total count)
    uint32_t num_raw_events() const;

    // PUBLISHED events, the bound for enumeration. See num_published_states for why the claim
    // counter above is not that bound.
    uint32_t num_published_events() const;

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
    bool is_event_canonical(EventId eid) const;

    // Get the canonical event ID for a raw event ID
    EventId get_canonical_event(EventId eid) const;

    // Event signature keys (bitflag controlling event equivalence)
    void set_event_signature_keys(EventSignatureKeys keys);
    EventSignatureKeys event_signature_keys() const;

    // WHERE the consumed/produced ranks in an Automatic-keyed signature are read from.
    //
    // false (Automatic): the class's pinned frame, via the reconstruction's signing -- the
    // linked-hypergraph convention of Wolfram/Multicomputation, adjudicated step-exact against
    // it (reference/adjudicate_gap1_authority.wls: 1,5,12,86 / 52 / 10 / 1,5,21). Runs under
    // BOTH exploration strategies, so quotient and full capture agree by construction.
    //
    // true ("Positional"): each raw state's own canonical labelling, per raw event. Distinguishes
    // events that differ only by which member of the labelling coset the canonicalizer's
    // tie-break selected, so it is THIS ENGINE'S positional identity: deterministic across
    // schedules and devices, but not a function of the abstract multiway system -- measured, it
    // differs from the reference oracle's like-named column where tie-breaks differ (23 vs 25 on
    // two-rules-overlap step 3) and from the authority (21). It requires raw presentations, so
    // requesting it disables quotient exploration (the engine reports that in warnings()).
    void set_positional_event_identity(bool on);
    bool positional_event_identity() const;


    // =========================================================================
    // Index Access
    // =========================================================================

    const SignatureIndex& signature_index() const;
    const InvertedVertexIndex& inverted_index() const;
    const PatternMatchingIndex& match_index() const;

    // =========================================================================
    // Causal Graph Access
    // =========================================================================

    CausalGraph& causal_graph();
    const CausalGraph& causal_graph() const;

    // Set edge producer: register `producer` as a producer of the canonical edge `key`
    // (mint keys with causal_edge_keys). raw_edge is the concrete edge id kept on the
    // CausalEdge record for viz.
    void set_edge_producer(CanonicalEdgeKey key, EventId producer, EdgeId raw_edge);

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
    const EdgeOrbitTable* state_orbits(StateId s) const;

    // Capture the canonical transition an event realizes into the quotient causal skeleton
    // (idempotent per distinct canonical transition). No-op if either endpoint's orbit
    // table is missing. Quotient mode only.
    void register_quotient_transition(EventId e);

    // Seed the quotient causal reconstruction at an initial state (depth 0): mark it
    // reachable and give each of its edge orbits the sentinel INIT producer (INVALID_ID,
    // skipped at emission -- initial edges have no producer). max_steps bounds the depth.
    void quotient_causal_seed(StateId initial_state, int max_steps);

    // Extend the reconstruction's depth budget for a continued run. The replay refuses to
    // expand an instance past it, so a continuation that raised the engine's budget and not
    // this one resumes the exploration and leaves the reconstruction where it stopped.
    void raise_quotient_max_steps(int max_steps);

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
    void set_quotient_reconstruction(bool on);
    bool quotient_reconstruction() const;
    // Raw observables recovered by the reconstruction (the full-capture counts).
    size_t num_reconstructed_events() const;
    size_t num_reconstructed_raw_events() const;
    // Instances the replay recorded: one per raw occurrence of a class at a depth. The
    // population every captured match is replayed against, so the relations it produces are a
    // function of it -- which makes it the first thing to compare when two runs disagree.
    size_t num_reconstructed_instances() const;
    size_t num_reconstructed_causal_edges() const;
    // TR-off view: every distinct (producer, consumer). TR-on view: those surviving reduction.
    // BOTH are DERIVED from the stored set, for the reason given on num_reconstructed_branchial
    // below -- a count and an enumeration that are maintained separately are the same number
    // only until one of them is wrong, and here the reduced count is what a caller compares
    // against what for_each_reconstructed_causal_as emits.
    size_t num_reconstructed_causal_pairs(bool transitively_reduced = false) const;
    size_t applied_scans() const;

    // HOW MANY (instance, match) PAIRS THE REPLAY CLAIMED, distinct.
    //
    // The claim key is a 64-bit FNV mix of two dense counters (hgcommon::qr_apply_key), and the
    // match side offers every captured match to every instance standing at its class, so the
    // number of DISTINCT keys is the cross product rather than the number of applications that
    // survive it -- the width check that rejects most of them runs AFTER the claim. Two distinct
    // pairs that mix to one key make the second look already-claimed, and its application is
    // dropped silently. That probability is n^2/2^65, so the count is the whole question and
    // nothing was reporting it.
    size_t applied_claims() const;

    // THE SHAPE OF THE APPLIED LISTS: the sorted multiset of per-instance application counts,
    // hashed. Two runs with the same TOTAL number of applications can still distribute them
    // differently across instances, and the branchial relation is built per instance -- so the
    // pair count would vary while the event count did not. That is exactly the shape the suite
    // reports, and nothing measured it: the totals agreed on every run and the distribution was
    // never looked at.
    // The multiset itself, sorted. A hash says two runs differ and not HOW: an instance
    // appearing and an existing instance going from nine applications to ten are the same
    // one-event difference in every scalar the suite reports, and different defects.
    std::vector<uint32_t> applied_shape() const;
    uint64_t applied_shape_fingerprint() const;

    // Matches the capture never recorded. The first is a race and a defect; the second is the
    // one-representative-per-class rule doing its job.
    size_t capture_dropped_no_orbits() const;
    size_t capture_skipped_not_representative() const;
    size_t applied_visits() const;

    // THE TWO POPULATIONS THE APPLICATIONS ARE DRAWN FROM. An application is one (instance,
    // match) pair, so a run with one application too many either replayed a pair it should not
    // have, or was handed an extra member of one of these two sets. Reporting only the
    // applications cannot tell those apart: captured_matches() is every SlotMatch pushed onto a
    // class's expansion list, reconstruction_instances() is every instance record minted.
    size_t captured_matches() const;
    size_t reconstruction_instances() const;

    // THE CLAIM TALLY AGAINST WHAT THE SET CAN HAND BACK. applied_claims() counts inserts that
    // reported a win; this walks the keys those wins are supposed to correspond to. They are the
    // same number only while every (instance, match) pair wins its claim exactly once, so a
    // shortfall here IS a pair replayed twice under one key -- which is the only thing the claim
    // stands between the two paths into qc_apply, and cannot be seen in any other count.
    size_t applied_unique() const;

    // DERIVED FROM THE SET, not from the counter that fed it. qc_num_branchial_ is incremented
    // when insert() reports a win; this returns what for_each will actually emit. The two are
    // the same number only while every winning claim stays reachable, and that invariant broke
    // once -- migrate_into dropped keys the caller had already been told it won (f694c062),
    // giving 20,558 pairs enumerated against 30,063 claimed. Reporting the enumeration removes
    // the class of divergence rather than re-synchronising one more site.
    size_t num_reconstructed_branchial() const;
    size_t num_frame_alignment_disagreements() const;
    size_t num_alignment_failures() const;
    size_t num_bad_correspondences() const;

    // Visit the DISTINCT event identities the reconstruction produced, under the run's
    // EventCanonicalizationMode. The counterpart of for_each_reconstructed_causal for events:
    // comparing these against full capture's Event::signature values says WHICH identities the
    // two paths disagree about, where comparing counts only says that they do.
    template <typename F>
    void for_each_reconstructed_event_signature(F&& f) const {
        qc_canon_event_seen_.for_each([&](uint64_t sig) { f(sig); });
    }

    // The state whose labelling defines a canonical class -- the class FRAME. The reconstruction
    // pins one to align slots, so a class hash resolves to a state a caller can point at without
    // anything being materialised for it. INVALID_ID when the class has no frame, which happens
    // for a class no captured transition touched.
    StateId class_frame_state(uint64_t class_hash) const;

    // Visit each DISTINCT reconstructed event once, as (dense id, content).
    //
    // The dense id names a vertex: the identity signatures are 64-bit hashes, which cannot be a
    // vertex label a user reads, and the reconstruction's raw event ids are per-application, so
    // there are more of them than there are events to show. Ids are assigned in ascending raw
    // event order, which is the order the replay minted them, so the numbering is a function of
    // the run rather than of the map's layout.
    //
    // Under EVENT_SIG_NONE every application is its own event and each raw event is visited;
    // under an identity mode the FIRST raw event carrying each identity stands for it, and its
    // content describes the class transition they all share.
    template <typename F>
    void for_each_reconstructed_event(F&& f) const {
        const uint32_t n = qc_next_raw_event_.load(std::memory_order_relaxed);
        const bool by_identity = event_signature_keys() != hgcommon::EVENT_SIG_NONE;
        std::set<uint64_t> seen;
        uint32_t dense = 0;
        for (uint32_t e = 0; e < n; ++e) {
            const QcEventContent* c = qc_event_sig_.get(e);
            if (!c) continue;
            if (by_identity) {
                const uint64_t* sig = qc_event_runsig_.get(e);
                if (!sig || !seen.insert(*sig).second) continue;
            }
            f(dense++, e, *c);
        }
    }

    // Visit each reconstructed RAW event's content triple hash(input class, output class, rule).
    // Schedule-stable and mode-stable -- a function of the multiway structure alone -- unlike
    // the run-identity signatures, whose slot components are labels relative to the class frame
    // a given run pinned and legitimately vary across schedules on symmetric classes. Use THIS
    // for cross-run and cross-thread fingerprints; use the identity signatures for identity
    // counts and identity-keyed relations.
    template <typename F>
    void for_each_reconstructed_raw_triple(F&& f) const {
        const uint32_t n = qc_next_raw_event_.load(std::memory_order_relaxed);
        for (uint32_t i = 0; i < n; ++i) {
            const QcEventContent* c = qc_event_sig_.get(i);
            if (c) f(c->triple_hash());
        }
    }

    // The identity a reconstructed PAIR endpoint is reported under: the run's, when one was
    // selected, so the relation can be set-compared against full capture's, which keys its own
    // pairs on Event::signature. Falls back to the internal (input, output, rule) triple when no
    // identity mode is selected -- full capture leaves Event::signature at 0 in that case, so
    // neither value is comparable then and the internal one at least distinguishes events.
    uint64_t event_pair_signature(uint32_t e) const;

    // Visit the reconstructed causal relation as pairs of isomorphism-invariant event
    // signatures. `reduced` selects the view: false walks every recorded pair (TR off), true
    // walks only those tagged in-reduction (TR on). Both come from the same online base, so
    // either view is available in any order at no extra cost.
    template <typename F>
    void for_each_reconstructed_causal(bool reduced, F&& f) const {
        for_each_reconstructed_causal_as(
            reduced, [&](uint32_t e) { return event_pair_signature(e); }, f);
    }

    // The same walk under a CALLER-CHOSEN endpoint identity.
    //
    // Which identity an endpoint is reported under is a real choice, not a detail. The run
    // identity (event_pair_signature, the default above) is what full capture keys its pairs on,
    // so it is what a set-comparison against full capture needs -- but its slot components are
    // labels relative to the class frame THIS run pinned, and on a symmetric class two runs
    // legitimately pin different members of the labelling coset. Fingerprinting the relation
    // under it therefore compares labels, not the relation, and reports a difference where the
    // structure is identical. For a cross-run or cross-thread comparison the endpoint identity
    // must be the schedule-stable content triple (for_each_reconstructed_raw_triple's value,
    // reachable per event through qc_event_sig_).
    //
    // One walk, two identities: a second copy of the traversal is how the two would drift.
    template <typename Id, typename F>
    void for_each_reconstructed_causal_as(bool reduced, Id&& id, F&& f) const {
        if (reduced) {
            // Reduced FROM THE STORED SET, under the one shared rule. The reduction of a DAG is
            // unique, so this is a function of the relation and not of the schedule that built
            // it -- which a tag decided as each pair arrived is not, because the pairs that
            // bypass (p,c) may land after it and nothing retracts the tag.
            hgcommon::tr_reduce(
                [&](auto&& add) {
                    qc_causal_pairs_.for_each([&](uint64_t k) {
                        const IdPair q = id_pair_from_key(k);
                        add(q.a, q.b);
                    });
                },
                [&](uint32_t p, uint32_t c) { f(id(p), id(c)); },
                // qc_apply mints a producer's id before creating the child instance whose later
                // application mints the consumer's, so ids increase along every edge here.
                /*ids_topological=*/true);
        } else {
            qc_causal_pairs_.for_each([&](uint64_t k) {
                const IdPair p = id_pair_from_key(k);
                f(id(p.a), id(p.b));
            });
        }
    }

    // The schedule-stable content triple of ONE reconstructed event: hash(input class, output
    // class, rule). 0 when the event has no recorded triple.
    uint64_t reconstructed_raw_triple(uint32_t e) const;

    // The event's content itself, for a caller that must DESCRIBE the event rather than
    // identify it. Null when no such reconstructed event exists.
    const QcEventContent* reconstructed_event_content(uint32_t e) const;

    // Visit the reconstructed branchial relation as pairs of isomorphism-invariant event
    // signatures, so it can be set-compared against full capture's branchial edges rather than
    // only count-compared. Full capture keys its pairs on (e1,e2); these are packed the same way,
    // so a diff of the two sets names WHICH pair is missing -- which a count cannot.
    //
    // The pair key cannot collide with a ConcurrentMap sentinel: the two events of a pair are
    // distinct, so lo < hi strictly, and neither an all-zero nor an all-ones key is reachable.
    template <typename F>
    void for_each_reconstructed_branchial(F&& f) const {
        for_each_reconstructed_branchial_as(
            [&](uint32_t e) { return event_pair_signature(e); }, f);
    }

    // The same walk under a CALLER-CHOSEN endpoint identity, for the same reason
    // for_each_reconstructed_causal_as exists: the run identity's slot components are labels
    // relative to the class frame THIS run pinned, so a cross-run or cross-thread comparison
    // must use the schedule-stable content triple instead. One walk, two identities.
    template <typename Id, typename F>
    void for_each_reconstructed_branchial_as(Id&& id, F&& f) const {
        // DERIVED FROM THE APPLICATIONS, not from a stored list of pairs. The pairs are the
        // per-instance applications that share a consumed slot, so the applications ARE the
        // relation in the form it is generated -- 970,584 of them against the 133,218,996 pairs
        // they imply on disc-l3a2g2r2 depth 3. Storing the expansion cost ~1.07 GB and is what
        // the device's 2^22 branchial ceiling was up against.
        //
        // The SAME rule the replay applies: a pair belongs to the LATER of its two
        // applications, so walking each node against the ones published before it yields every
        // pair exactly once, with nothing to dedup against.
        // THE WALK IS QUIESCENT -- this runs at serialization time, after the evolution -- so
        // each instance's list is FLATTENED once into contiguous scratch and the quadratic
        // pair loop runs over the flat array. The list costs one dependent load per node; the
        // pair loop visits each element k times, so walking the LIST there is k dependent
        // loads per element where the flat array costs none: 162,258,036 node visits against
        // 970,584 flatten loads on disc-l3a2g2r2 depth 3. Order is preserved exactly --
        // for_each_node is newest-first and for_each_before(mine) is everything older, which
        // flat[i] against flat[j], j > i, reproduces pair for pair.
        const uint32_t n = qc_inst_applied_.size();
        std::vector<QcAppliedMatch> flat;
        for (uint32_t i = 0; i < n; ++i) {
            const LockFreeList<QcAppliedMatch>* lst = qc_inst_applied_.get(i);
            if (!lst) continue;
            flat.clear();
            lst->for_each([&](const QcAppliedMatch& m) { flat.push_back(m); });
            for (uint32_t ai = 0; ai < flat.size(); ++ai) {
                const QcAppliedMatch& a = flat[ai];
                for (uint32_t bi = ai + 1; bi < flat.size(); ++bi) {
                    const QcAppliedMatch& b = flat[bi];
                    if (a.event == b.event) continue;
                    bool overlaps = false;
                    for (uint32_t x = 0; x < a.num_consumed && !overlaps; ++x)
                        for (uint32_t y = 0; y < b.num_consumed; ++y)
                            if (a.consumed(x) == b.consumed(y)) { overlaps = true; break; }
                    if (!overlaps) continue;
                    const uint32_t lo = a.event < b.event ? a.event : b.event;
                    const uint32_t hi = a.event < b.event ? b.event : a.event;
                    f(id(lo), id(hi));
                }
            }
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

    size_t observable_num_events() const;
    size_t observable_num_causal_edges() const;
    size_t observable_num_causal_pairs(bool transitively_reduced) const;
    size_t observable_num_branchial() const;

    // Get a representative edge producer for a canonical edge key (INVALID_ID if none).
    EventId get_edge_producer(CanonicalEdgeKey key) const;

    // Add edge consumer: register `consumer` as a consumer of the canonical edge `key`.
    void add_edge_consumer(CanonicalEdgeKey key, EventId consumer, EdgeId raw_edge);

    // Carry a surviving edge's producers from its parent-state orbit key to its
    // child-state orbit key (see CausalGraph::propagate_producers).
    void propagate_producers(CanonicalEdgeKey from, CanonicalEdgeKey to, EdgeId raw_edge);

    // Whether causal edges are keyed by canonical edge orbit (quotient exploration). Set
    // by the evolution engine before evolving; read when minting causal edge keys.
    void set_quotient_causal(bool q);
    bool quotient_causal() const;

    // Which artifacts this run builds. Set before evolving and read by the workers, so the
    // two components are stored as atomics like every other pre-evolution switch here.
    void set_record_set(RecordSet r);
    RecordSet record_set() const;

    // Create a genesis event for an initial state.
    // This synthetic event connects the empty genesis state to the initial state.
    // It "produces" all edges in the initial state, enabling causal tracking from gen 0.
    // Returns the genesis event ID.
    EventId create_genesis_event(StateId initial_state, const EdgeId* edges, uint8_t num_edges);

    // Register event for branchial tracking
    // When event canonicalization is enabled, uses edge equivalence for overlap detection
    // and skips branchial edges between canonically equivalent events
    // The per-state event list and the branchial pair relation, recorded independently: they
    // feed different outputs, so a run that needs one need not build the other.
    void record_state_event(EventId event, StateId input_state);
    void record_branchial_overlaps(EventId event, StateId input_state,
                                   const EdgeId* consumed_edges, uint8_t num_consumed);

    // Get causal/branchial statistics
    size_t num_causal_edges() const;
    size_t num_causal_event_pairs() const;
    size_t num_branchial_edges() const;

    // =========================================================================
    // Arena Access
    // =========================================================================

    ConcurrentHeterogeneousArena& arena();
    const ConcurrentHeterogeneousArena& arena() const;

    // =========================================================================
    // Counter Access
    // =========================================================================

    GlobalCounters& counters();
    const GlobalCounters& counters() const;

    // =========================================================================
    // Utility
    // =========================================================================

    // Compute simple hash for a state's edge set (fast but not isomorphism-invariant)
    static uint64_t compute_state_hash(const SparseBitset& edges);

    // Compute content-ordered hash for Automatic state canonicalization mode
    // Hashes edge contents in order by edge ID: (arity, v1, v2, ...) for each edge
    // Fast but not isomorphism-invariant.
    uint64_t compute_content_ordered_hash(const SparseBitset& edges) const;

    // Compute canonical hash (isomorphism-invariant).
    // With the WL hash enabled (use_wl_hash_), uses the fast approximate hash;
    // otherwise falls back to IR exact canonicalization.
    uint64_t compute_canonical_hash(const SparseBitset& edges) const;

    // Isomorphism-invariant and EXACT, whatever the state mode selects. The event path uses
    // this, because event identity is defined over isomorphism classes.
    uint64_t compute_exact_canonical_hash(const SparseBitset& edges) const;

    // What a state reports and what the event path resolves representatives through: exact
    // when event canonicalization is on, mode-aware otherwise.
    uint64_t compute_reported_canonical_hash(const SparseBitset& edges) const;

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
    uint32_t count_state_edges(StateId sid) const;
};

}  // namespace engine
}  // namespace HG_NAMESPACE