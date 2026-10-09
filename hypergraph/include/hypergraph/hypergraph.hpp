#pragma once
#include "hgcommon/core.hpp"
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
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "hgcommon/quotient_causal_core.hpp"
#include "hgcommon/quotient_replay_core.hpp"
#include "hgcommon/ir_core.hpp"
#include "hgcommon/canonical_form_core.hpp"
#include "hgcommon/token_core.hpp"
#include "hgcommon/state_invariants_core.hpp"
#include "lock_free_list.hpp"
#include "causal_graph.hpp"
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

    // One past the largest state and event id each worker has published, one cache line per
    // worker. num_published_states and num_published_events take the maximum when read; a shared
    // high-water mark on the arrays was one compare-and-swap per state and per event on one line.
    // The owning worker writes its mark with a release store and readers load it with acquire,
    // so a read while workers run is not a data race. A thread that is not a worker publishes
    // through the atomics.
    struct alignas(64) PublishedMark {
        uint32_t states = 0;
        uint32_t events = 0;
    };
    // By pointer: 16 KB, and a Hypergraph is constructed on the stack by tests and probes.
    std::unique_ptr<PublishedMark[]> published_ = std::make_unique<PublishedMark[]>(MAX_ARENA_WORKERS);
    std::atomic<uint32_t> published_states_outside_{0};
    std::atomic<uint32_t> published_events_outside_{0};
    static uint32_t published_mark(const uint32_t& mark);
    void note_published_state(StateId sid);
    void note_published_event(EventId eid);

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
    // Each event's marked forms, indexed by event id, written before the event's identity is
    // claimed; filled only when hgcommon::event_keys_mark_edges(event_signature_keys_).
    SegmentedArray<hgcommon::EventMarkedForms> event_forms_;
    // Each class's invariants, filled only when record_state_invariants_: canonical hash -> a
    // cell the worker that claims the hash fills with the record (record_state_invariants).
    ConcurrentMap<uint64_t, const hgcommon::StateInvariantRecord**> state_invariants_;

    // Pattern matching indices

    // Causal and branchial graph
    CausalGraph causal_graph_;

    // Canonical state deduplication map for the None and Automatic modes: dedup key ->
    // representative StateId.
    ConcurrentMap<uint64_t, StateId, uint64_t{0}, ~uint64_t{0}, INVALID_ID> canonical_state_map_;

    // Full mode's deduplication map: probe key of the IR canonical hash -> the class's record,
    // which holds the representative StateId and the IR canonical form. A state joins the class
    // under a key only when its form equals the record's (claim_canonical_state).
    ConcurrentMap<uint64_t, const hgcommon::CanonicalFormRecord*> canonical_form_map_;

    // Event canonicalization state map, keyed by canonical_hash rather than by the state
    // mode's dedup key, so event identity does not follow the state-merging choice.
    // Used by event signature computation to find canonical representatives for edge
    // correspondence when state_canonicalization_mode_ is None or Automatic.
    //
    // The key is the EXACT IR invariant in every state mode whenever event canonicalization is
    // on (the create path stores it), so the event identity resolved through this map is
    // the same identity whatever the state mode -- mode_matrix_probe measures identical event
    // counts down every state-mode column. SPEC.md sec 4 states the axes independent, and for
    // EVENTS that holds. The CAUSAL relation under Automatic event identity is the
    // canonical-class relation only when the orbit tables exist (the Full state mode computes
    // them); in the other state modes the engine warns and serves the raw-edge rendezvous.
    //
    // Under None and Automatic the key is claimed on the state's IR canonical form (claim_identity),
    // so two non-isomorphic states whose IR hashes collide get two keys; under Full the event
    // path resolves through canonical_form_map_ and this map is empty.
    ConcurrentMap<uint64_t, const hgcommon::CanonicalFormRecord*> event_canonical_state_map_;

    // State canonicalization mode: controls how states are deduplicated, via the map_key
    // create_or_get_canonical_state builds -- which is a DIFFERENT quantity from the
    // canonical_hash it reports.
    //   None:      dedup key is the raw state id, so nothing merges
    //   Automatic: dedup key is compute_content_ordered_hash -- merges states with identical
    //              edge content, which is NOT isomorphism-invariant
    //   Full:      dedup key is the IR canonical hash (canonical_form_map_), and a state merges
    //              with the key's class only when their IR canonical forms are equal -- merges
    //              isomorphic states
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
    std::atomic<bool> record_raw_counts_only_{false};
    std::atomic<bool> record_multiplicities_{false};
    std::atomic<bool> record_state_invariants_{false};

    // A state's edge-orbit table and canonical rank table are State::edge_orbits and
    // State::edge_ranks. The orbits are computed once at state canonicalization in quotient mode
    // (from the dedup IR canonicalization, so no extra pass) and read by the quotient causal
    // reconstruction for every event. The ranks are built once per state when event
    // canonicalization is on and read by every event that consumes or produces one of its
    // edges; edges ascend (SparseBitset iterates in id order), so a lookup binary-searches.
    // Published by compare-and-swap from null (publish_table): the first table built stays.
    template <class T>
    static void publish_table(T*& field, T* tbl) {
        T* expected = nullptr;
        hgcommon::atomic_ref<T*>(field).compare_exchange_strong(
            expected, tbl, std::memory_order_release, std::memory_order_relaxed);
    }
    template <class T>
    static T* read_table(T*& field) {
        return hgcommon::atomic_ref<T*>(field).load(std::memory_order_acquire);
    }

    // The (class, depth) points at the depth bound that hold replay instances or multiplicity
    // mass. Those are recorded and never expanded, so raising the bound has to revisit them.
    // Pushed once per point per structure, by the thread that created the point's entry.
    struct QcPoint { uint64_t state_hash; uint32_t depth; };
    LockFreeList<QcPoint> qc_blocked_;
    std::atomic<int> qc_max_steps_{0};
    // One past the deepest depth that holds a replay instance or a multiplicity point. Raised
    // before the instance or point is published, and read by a captured match after its
    // rendezvous fence: the match visits depths below it only, so its work follows the depth
    // reached and not the step budget.
    std::atomic<uint32_t> qc_depth_hi_{0};
    void qc_note_depth(uint32_t depth) {
        const uint32_t want = depth + 1;
        uint32_t cur = qc_depth_hi_.load(std::memory_order_relaxed);
        while (cur < want && !qc_depth_hi_.compare_exchange_weak(cur, want, std::memory_order_relaxed)) {
        }
    }
    uint32_t qc_depth_bound(uint32_t max_steps) const {
        const uint32_t hi = qc_depth_hi_.load(std::memory_order_relaxed);
        return hi < max_steps ? hi : max_steps;
    }

    // The expanded representative's FULL match list per canonical state, in slots -- the
    // input to the per-instance raw reconstruction. Two matches over one orbit both survive
    // (full capture fires both).
    // qc_expansion_rep_ pins the one raw state whose events define the expansion, so a second
    // raw state of the same class (a dedup race) cannot append a duplicate expansion.
    struct QcExpansion {
        LockFreeList<SlotMatch> list;
        std::atomic<uint32_t> n{0};   // matches captured; SlotMatch::local is taken from it
        // B(c): pairs of the class's matches whose consumed slots overlap, the sum of their b_j
        // (hgcommon/quotient_multiplicity_core.hpp). Accumulated at capture when branchial pairs
        // are counted.
        std::atomic<uint64_t> pairs{0};
    };
    ConcurrentMap<uint64_t, QcExpansion*> qc_expansion_;
    ConcurrentMap<uint64_t, uint64_t> qc_expansion_rep_;   // canonical hash -> StateId + 1
    std::atomic<uint32_t> qc_next_match_id_{0};

    // The slot FRAME of a canonical class: the first raw state seen for the class, whose slot
    // numbering every other instance of that class is aligned into.
    //
    // Slots are read off a state's canonical labeling, and two raw states of one class have
    // labelings differing by an automorphism -- different reference frames. Without a frame the
    // reconstruction mixes them: a child's slots would be written in the producing event's own
    // output-state numbering but read against a different state's, and which state that is
    // depends on thread scheduling. Pinning one frame per class removes the choice.
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
    // with its lineage (hgcommon::qr_producer_of gives a slot's producing event from it);
    // replaying every expansion match against every instance regenerates the raw event set the
    // quotient never explores.
    // Reconstructed event ids come from a counter -- counts and causal edges need only ids,
    // not Event records, so this does not undo the quotient's state/edge compression.
    // An instance's parent, the match that made it and that match's event; a root has none.
    struct QcLineage {
        const QcLineage* parent = nullptr;
        const SlotMatch* via = nullptr;
        uint32_t event = 0;
    };
    // Each initial state's root lineage and genesis event, written by quotient_causal_seed before
    // any worker runs.
    struct QcGenesisRoot { const QcLineage* root; EventId genesis; };
    std::vector<QcGenesisRoot> qc_genesis_roots_;
    // One block of an instance's claim chain (hgcommon::qr_claim_chain): `words` 64-bit claim
    // words follow the header in the same allocation.
    struct QcClaimBlock {
        std::atomic<QcClaimBlock*> next{nullptr};
        uint32_t words = 0;
        std::atomic<uint64_t>* bits() { return reinterpret_cast<std::atomic<uint64_t>*>(this + 1); }
    };
    QcClaimBlock* qc_new_claim_block(uint32_t words);
    struct QcInstance {
        uint32_t id = 0;
        uint32_t nslots = 0;
        const QcLineage* lineage = nullptr;
        // The first block of the instance's claim chain; null for an instance made at the depth
        // bound, whose pairs (applied only when a continuation raises the bound) claim in
        // qc_applied_.
        QcClaimBlock* claims = nullptr;
    };
    // The instances of one (class, depth). Every instance is pushed to `first` until a push
    // loses its compare-and-swap there; that push installs `more`, kInstShards further lists,
    // and from then on a worker pushes to more[worker index % kInstShards]. A reader walks
    // `first` and, when installed, every list of `more` (kInstLists in all, `first` at index
    // 0). On a rule with few classes most of the replay pushes to a few entries, and the shards,
    // each on its own cache line, split those pushes (unpadded heads, measured: multirule
    // depth 7 quotient 22.7 -> 72.8 ms at 16 threads). Most entries hold one or two instances
    // and never install `more`: 128 bytes each where eight padded heads took 528 (wpp depth 8
    // quotient, 348,615 classes: arena 2,658 -> 2,509 MB).
    static constexpr uint32_t kInstShards = 8;
    static constexpr uint32_t kInstLists = kInstShards + 1;
    struct alignas(64) QcInstanceShard {
        LockFreeList<QcInstance> list;
    };
    struct alignas(64) QcInstanceShards {
        QcInstanceShard first;
        std::atomic<QcInstanceShard*> more{nullptr};
        uint64_t class_hash = 0;
        uint32_t depth = 0;
        // List l of kInstLists, or nullptr when `more` is not installed.
        const LockFreeList<QcInstance>* list(uint32_t l) const {
            if (l == 0) return &first.list;
            const QcInstanceShard* m = more.load(std::memory_order_acquire);
            return m ? &m[l - 1].list : nullptr;
        }
        template <typename F>
        void for_each(F&& f) const {
            for (uint32_t l = 0; l < kInstLists; ++l)
                if (const LockFreeList<QcInstance>* x = list(l)) x->for_each(f);
        }
    };
    void qc_push_instance(QcInstanceShards* sh, const QcInstance& inst);
    // Keyed by qc_key(hash, depth, 0) & qc_key_mask_ through the keyed-claim walk
    // (qc_point_claim / qc_point_find): the key selects where to look and the entry's
    // (class_hash, depth) decides, so two points whose keys collide each get a key.
    ConcurrentMap<uint64_t, QcInstanceShards*> qc_instances_;
    QcInstanceShards* qc_instances_at(uint64_t class_hash, uint32_t depth) const;
    // set_qc_spawn's function and context.
    void (*qc_spawn_)(void*, Hypergraph*, const SlotMatch*, uint64_t, uint32_t, uint32_t) = nullptr;
    void* qc_spawn_ctx_ = nullptr;
    // Claims a (instance, match) application. Both the instance side and the match side drive
    // the rendezvous, and unlike the producer-set DP an application is NOT idempotent -- each
    // one emits a raw event -- so the pair must be claimed exactly once. O(raw) entries.
    ShardedKeySet<uint64_t> qc_applied_;
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
    // has no resize chain and no sentinel-key domain, and slot() hands both threads
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
    // Instance ids: taken by a worker in blocks of kIdBlock from qc_next_instance_ (one
    // shared increment per block), by any other thread one at a time. An instance id only has to
    // be unique -- it keys the application claim and the applied list -- so gaps are harmless;
    // the instance count is the per-worker counts summed. 64-bit so that it cannot wrap; ids at or
    // past qc_id_limit_ are refused (alloc_instance_id).
    // Alone on its line: every worker's block refill writes it, and the members after it
    // (qc_inst_blocks_ is read on every instance) would otherwise be reloaded after each refill
    // by another worker (multirule depth 7 quotient, 16 threads: 31% of qc_add_instance's
    // samples on the load of qc_inst_blocks_).
    alignas(64) std::atomic<uint64_t> qc_next_instance_{0};
    alignas(64) std::atomic<uint64_t> qc_instances_made_outside_{0};
    static constexpr uint32_t kIdBlock = 64;
    struct alignas(64) IdBlock {
        uint32_t next = 0;
        uint32_t end = 0;
        uint64_t made = 0;
    };
    std::unique_ptr<IdBlock[]> qc_inst_blocks_ = std::make_unique<IdBlock[]>(MAX_ARENA_WORKERS);
    uint32_t alloc_instance_id();
    template <typename F>
    void for_each_instance_at(uint64_t state_hash, uint32_t depth, F&& f) {
        const QcInstanceShards* sh = qc_instances_at(state_hash, depth);
        if (!sh) return;
        sh->for_each(f);
    }
    // Raw event ids: a worker takes them in blocks of kEventIdBlock from qc_next_raw_event_,
    // and uses its block only for an event whose producers are all below the block's next id.
    // Otherwise it takes a new block, which lies above every id taken so far. Ids therefore
    // increase along every causal edge. Ids left in an abandoned block are never written;
    // QcEventContent::written tells a reader so, and the event count is the per-worker counts
    // summed. 64-bit so that it cannot wrap; ids at or past qc_id_limit_ are refused
    // (alloc_event_id).
    // Alone on its line, for the reason qc_next_instance_ is.
    alignas(64) std::atomic<uint64_t> qc_next_raw_event_{0};
    // hgcommon::QR_ID_LIMIT except in tests (set_replay_id_limit); ids refused at it.
    alignas(64) uint32_t qc_id_limit_ = hgcommon::QR_ID_LIMIT;
    std::atomic<uint64_t> qc_ids_refused_{0};
    std::atomic<uint64_t> qc_captures_dropped_{0};
    // An id counter read as a bound on the ids it issued.
    uint32_t qc_id_bound(const std::atomic<uint64_t>& counter) const {
        const uint64_t v = counter.load(std::memory_order_relaxed);
        return v < qc_id_limit_ ? static_cast<uint32_t>(v) : qc_id_limit_;
    }
    std::atomic<uint64_t> qc_events_made_outside_{0};
    static constexpr uint32_t kEventIdBlock = 64;
    std::unique_ptr<IdBlock[]> qc_event_blocks_ = std::make_unique<IdBlock[]>(MAX_ARENA_WORKERS);
    uint32_t alloc_event_id(uint32_t above);

    // Reconstructed events under the RUN'S event identity, as opposed to the raw count above.
    // qc_event_sig_ carries a fixed (input, output, rule) triple, which is its own identity and
    // not the one the caller selected -- EVENT_SIG_FULL keys on the endpoint states alone,
    // EVENT_SIG_AUTOMATIC adds the step and the canonical ranks. Under an identity mode the
    // observable is the count of DISTINCT identities, so the mode's signature is computed here
    // and the distinct ones counted.
    // Probe key of an event signature -> the first replay application of the event class, from
    // which hgcommon::qr_signature_values recomputes the class's values on a key hit.
    struct QrEventRef { const SlotMatch* m; uint64_t from_hash; uint32_t out_step; };
    ConcurrentMap<uint64_t, const QrEventRef*> qc_canon_events_;
    std::atomic<size_t> qc_num_canon_events_{0};
    std::atomic<bool> quotient_reconstruction_{false};

    // RAW COUNTS FROM CLASS MULTIPLICITIES (hgcommon/quotient_multiplicity_core.hpp), run in
    // place of the replay when the raw events and branchial pairs are read only as counts. The
    // work is per (class, depth, match); the replay's is per raw state.
    std::atomic<bool> quotient_multiplicity_{false};
    // Whether the reconstruction materialises instances. Off when the raw events and branchial
    // pairs are read only as counts, which the multiplicities then answer.
    std::atomic<bool> quotient_replay_{true};
    // A (class, depth) point: m(class, depth), the raw states the class stands for at that
    // depth, and whether a run of the point is queued.
    struct QmPoint {
        std::atomic<uint64_t> mass{0};
        std::atomic<uint32_t> queued{0};
        uint32_t depth = 0;
        uint64_t class_hash = 0;
    };
    ConcurrentMap<uint64_t, QmPoint*> qm_points_;   // keyed as qc_instances_
    QmPoint* qm_point_at(uint64_t class_hash, uint32_t depth) const;
    // ANDed into qc_key for the first probe key of qc_instances_ and qm_points_. All ones except
    // in tests, which narrow it so that different (class, depth) points share keys.
    uint64_t qc_key_mask_{~uint64_t{0}};
    // consumed_j(depth): the mass match j has passed on from its class at that depth.
    ConcurrentMap<uint64_t, std::atomic<uint64_t>*> qm_consumed_;
    // b_j + 1, keyed by match id + 1. Present once the match is ready.
    ConcurrentMap<uint64_t, uint64_t> qm_overlaps_;
    std::atomic<uint64_t> qm_events_{0};

    // The reconstructed causal relation over raw event ids, every pair: the TR-off view. The
    // reduction is kept separately, per consumer, in qc_kept_.
    //
    // (producer, consumer) PAIRS, ONE APPEND-ONLY LIST PER WORKER, and no dedup structure.
    //
    // This was a shared set because the pairs looked like they needed deduplicating. They do
    // not: record_causal is reached only from qr_apply's producer loop, whose consumer is the
    // event THIS application just minted, so (producer, ev) cannot repeat across applications.
    // The one way it repeats is a producer appearing twice in a single application's own
    // producer list, and that list is local, bounded by MAX_PATTERN_EDGES and already sorted --
    // qr_collect_producers drops the adjacent repeats before they are recorded.
    //
    // So the set's only remaining job was STORAGE for the enumeration below, and storage needs
    // no shared line: each worker appends to its own list and a reader walks all of them.
    // MEASURED as the reason: this set took 95,600 inserts on cycle4 against qc_applied_'s
    // 68,184, and ConcurrentKeySet::insert went from 12.9% of the run at one thread to 41.2% at
    // four on a part whose cores do not share a last-level cache.
    // A worker's list is written by that worker alone: chunks of kCap pairs, the newest at
    // `head`, each with a count a reader loads with acquire after the writer's release, so an
    // append is a store and a count increment, with no compare-and-swap and no node per pair.
    // One head per 64-byte line (eight 8-byte heads on one line measured 44% of a 16-thread
    // multirule depth 7 quotient run in the push). A thread that is not a worker appends to
    // qc_causal_pairs_outside_, which several such threads may share.
    struct QcPairChunk {
        static constexpr uint32_t kCap = 252;
        std::atomic<uint32_t> n{0};
        QcPairChunk* prev = nullptr;
        uint64_t v[kCap];
    };
    struct alignas(64) QcPairList {
        std::atomic<QcPairChunk*> head{nullptr};
        template <typename Arena>
        void push(uint64_t key, Arena& arena) {
            QcPairChunk* c = head.load(std::memory_order_relaxed);
            uint32_t n = c ? c->n.load(std::memory_order_relaxed) : QcPairChunk::kCap;
            if (n == QcPairChunk::kCap) {
                QcPairChunk* made = arena.template create<QcPairChunk>();
                made->prev = c;
                head.store(made, std::memory_order_release);
                c = made;
                n = 0;
            }
            c->v[n] = key;
            c->n.store(n + 1, std::memory_order_release);
        }
        template <typename F>
        void for_each(F&& f) const {
            for (const QcPairChunk* c = head.load(std::memory_order_acquire); c; c = c->prev) {
                const uint32_t n = c->n.load(std::memory_order_acquire);
                for (uint32_t i = 0; i < n; ++i) f(c->v[i]);
            }
        }
        size_t size() const {
            size_t n = 0;
            for (const QcPairChunk* c = head.load(std::memory_order_acquire); c; c = c->prev)
                n += c->n.load(std::memory_order_acquire);
            return n;
        }
    };
    QcPairList qc_causal_pairs_[MAX_ARENA_WORKERS];
    LockFreeList<uint64_t> qc_causal_pairs_outside_;

    template <typename F>
    void qc_causal_pairs_for_each(F&& f) const {
        for (const QcPairList& l : qc_causal_pairs_) l.for_each(f);
        qc_causal_pairs_outside_.for_each(f);
    }
    size_t qc_causal_pairs_count() const {
        size_t n = qc_causal_pairs_outside_.size();
        for (const QcPairList& l : qc_causal_pairs_) n += l.size();
        return n;
    }
    // Isomorphism-invariant signature per reconstructed event: fnv(from hash, to hash, rule).
    // Reconstructed events carry no Event record, so this is the only description they have --
    // it is what schedule-independence is fingerprinted on, and what a graph over reconstructed
    // events is built from. Held as the three COMPONENTS rather than their hash: the hash
    // identifies an event and cannot describe one, and a vertex needs its endpoints.
    SegmentedArray<QcEventContent> qc_event_sig_;

    // Per reconstructed event, the producers its transitive reduction kept: THE reduced relation,
    // enumerated as it stands, and the predecessor adjacency the online search walks. Written
    // once, by the event's own application before its descent; indexed through qc_ev_slot. A
    // slot never written reads as zero producers.
    // Up to three producers inline, which is every left-hand side of three edges or fewer.
    struct QcKept {
        uint32_t n;
        uint32_t inl[3];
        const uint32_t* more;   // producers 3.. when n > 3
        uint32_t at(uint32_t i) const { return i < 3 ? inl[i] : more[i - 3]; }
    };
    std::unique_ptr<SegmentedArray<QcKept>> qc_kept_;   // by pointer: a SegmentedArray is 32 KB

    // WHERE EVENT `e`'S CONTENT LIVES: slot e. Event and instance ids are taken in per-worker
    // blocks of 64 (kEventIdBlock, kIdBlock), so a worker writes a run of consecutive slots and
    // two workers share a cache line only at a block's edge. Any permutation of slots within a
    // block of ids larger than 64 puts slots of different workers on one line.
    static uint32_t qc_ev_slot(uint32_t e) { return e; }
    // These four arrays are written through emplace_at on an uncounted array or slot(), and read
    // through find(): no shared high-water counter is written per event or read per lookup. A
    // reader bounds its walk by the id counter. Ids below it that were never written (instance
    // ids skipped at the end of a worker's block) read as empty lists.
    uint32_t qc_applied_slot_bound() const {
        return qc_id_bound(qc_next_instance_);
    }
    // The same events under the RUN'S event identity, indexed the same way. The pair accessors
    // need this and not qc_event_sig_: a caller comparing the reconstructed causal or branchial
    // relation against full capture is comparing against Event::signature, which is the run's
    // identity, so emitting the internal triple instead compares two different functions and
    // every pair looks like a disagreement. Left at 0 when no identity mode is selected, which
    // is what full capture leaves Event::signature at in that case.
    SegmentedArray<uint64_t> qc_event_runsig_;

    // PER-WORKER SLOTS FOR THE COUNTERS BUMPED ONCE PER ITEM. A single atomic incremented per
    // causal edge, per pair or per branchial pair is one cache line taken exclusive by every
    // worker in turn, and the count IS the workload: 129,384 causal-edge bumps on cycle4 at
    // depth 6, and 133,218,996 branchial bumps on disc-l3a2g2r2 at depth 3. EvolutionStats was
    // given this treatment already; the audit that did it named these four and did not move
    // them.
    //
    // Safe here in a way the event-id allocator is NOT: these are REPORTING counters. Nothing
    // reads them to decide anything, so a per-worker sum changes no output -- where per-worker
    // event id BLOCKS changed which raw event represents an identity and broke three gates.
    // ONE LINE PER WORKER, HOLDING EVERY COUNTER THAT WORKER BUMPS -- not one padded array per
    // counter. Five separate alignas(64) arrays of MAX_ARENA_WORKERS entries is 80 KB of member
    // data for 40 bytes of counters per worker, and Hypergraph is stack-allocated by several
    // probes: at 399 KB it overflowed the 1 MB default stack on Windows and
    // ctest's quotient_reconstruction died with SEGFAULT. Packed this way the same separation
    // costs 16 KB, because the point was never a line per COUNTER -- it was a line per WORKER,
    // so that two workers never share one.
    struct alignas(64) QcCounterSlot {
        size_t causal_edges = 0;
        size_t causal_pairs = 0;
        size_t applications = 0;   // reconstruction applications this worker performed
        size_t reduced_pairs = 0;  // pairs the online reduction kept (qr_apply)
        size_t bit_claims = 0;     // (instance, match) claims won in an instance's claim bits
    };
    mutable QcCounterSlot qc_ctr_[kCounterSlots];

    template <typename M>
    static void qc_count(QcCounterSlot* slots, M member, size_t n = 1) {
        const int slot = counter_slot();
        counter_add(slots[slot].*member, slot, n);
    }
    template <typename M>
    size_t qc_ctr_total(M member) const {
        size_t n = 0;
        for (const QcCounterSlot& s : qc_ctr_) n += counter_read(s.*member);
        return n;
    }

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
    // Times an endpoint's table was not cached and qc_orbits_or_build rebuilt it. The
    // reconstruction is unaffected -- the rebuilt table is the same function of the same edge
    // set -- so this is a cache-miss rate and not a correctness signal, and it is deliberately
    // absent from the compared fingerprint for that reason.
    mutable std::atomic<size_t> qc_capture_orbit_rebuilds_{0};
    // THE FIRST DROP KEEPS ITS EVIDENCE, because a count alone cannot say which of two defects
    // it is. Packed as (endpoint << 32) | state id, endpoint 0 for the event's input and 1 for
    // its output, claimed once by compare-exchange from the all-ones initial value. Which
    // endpoint separates a state another thread is still creating from one this thread made
    // itself; the state id lets a reader ask, after the run, whether that state EVER got a
    // table -- never computed and not yet visible are different bugs with different fixes.
    mutable std::atomic<uint64_t> qc_no_orbits_witness_{~uint64_t{0}};
    // The class already has a representative raw state and it is not this one. BY DESIGN -- one
    // raw state per class captures -- but counted, because "by design" and "measured" are
    // different claims and only one of them was on record.
    mutable std::atomic<size_t> qc_capture_not_rep_{0};
    void qc_record_causal(uint32_t producer, uint32_t consumer, bool distinct_pair);

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

    // The (class, depth) key space comes from hgcommon so the device indexes the same one.
    static uint64_t qc_key(uint64_t state_hash, uint32_t depth, uint32_t orbit);

    // The storage face hgcommon/quotient_replay_core.hpp drives. Same division as QcCtx above:
    // WHERE an instance's lineage, an applied list or a claim set lives is here; what an
    // application DOES -- what it claims, what it identifies the event by, which causal and
    // branchial relations follow -- is in the core, which is the body the device runs too.
    struct QrCtx {
        using Instance = QcInstance;
        using Match    = SlotMatch;
        Hypergraph& hg;

        bool claim(const QcInstance& inst, const SlotMatch& m);
        uint32_t mint_event(uint32_t above);
        void record_content(uint32_t ev, uint64_t from_class, uint64_t to_class, uint32_t rule);
        hgcommon::EventSignatureKeys keys() const;
        // Kept per event as well as counted, so the causal and branchial accessors report the
        // relation under the identity the CALLER selected -- reporting the internal triple
        // instead makes every pair look like a disagreement with full capture.
        void record_runsig(uint32_t ev, const SlotMatch& m, uint64_t from_class, uint32_t out_step);
        bool want_causal() const;
        bool want_branchial() const;
        uint32_t producer_at(const QcInstance& inst, uint32_t slot) const;
        // hgcommon::qr_producer_of's face.
        static bool lineage_root(const QcLineage* n) { return n->via == nullptr; }
        static uint32_t lineage_source(const QcLineage* n, uint32_t slot) {
            return slot < n->via->to_slots ? n->via->child_source[slot]
                                           : hgcommon::QR_SOURCE_NONE;
        }
        static uint32_t lineage_event(const QcLineage* n) { return n->event; }
        static const QcLineage* lineage_parent(const QcLineage* n) { return n->parent; }
        void record_causal(uint32_t producer, uint32_t consumer, bool distinct_pair);
        uint32_t redundant(const uint32_t* producers, uint32_t n) const;
        void record_kept(uint32_t ev, const uint32_t* kept, uint32_t nkept);
        void publish_applied(const QcInstance& inst, const SlotMatch& m, uint32_t ev);
        void descend(const SlotMatch& m, uint32_t depth, uint32_t ev, const QcInstance& parent);
    };

    // The storage face hgcommon/quotient_multiplicity_core.hpp drives. The queue is the
    // cascade's own, in this worker's scratch arena (QmQueue, defined in hypergraph.cpp).
    struct QmQueue;
    struct QmCtx {
        using Match = SlotMatch;
        Hypergraph& hg;
        QmQueue& queue;

        uint32_t max_steps() const;
        bool ready(const SlotMatch& m, uint64_t& b) const;
        uint64_t mass(uint64_t class_hash, uint32_t depth) const;
        void add_mass(uint64_t class_hash, uint32_t depth, uint64_t delta);
        uint64_t consumed(const SlotMatch& m, uint32_t depth);
        bool advance(const SlotMatch& m, uint32_t depth, uint64_t& expected, uint64_t desired);
        void count(uint64_t events);
        hgcommon::EventSignatureKeys keys() const;
        void note_signature(const SlotMatch& m, uint64_t from_class, uint32_t out_step);
        bool claim_queued(uint64_t class_hash, uint32_t depth);
        void push(uint64_t class_hash, uint32_t depth);
        bool pop(uint64_t& class_hash, uint32_t& depth);
        template <class F>
        void for_each_match(uint64_t class_hash, F&& f) {
            hg.for_each_expansion_match(class_hash, f);
        }
        void fence();
    };
    static uint64_t qm_consumed_key(uint32_t match_id, uint32_t depth);
    QmPoint* qm_point(uint64_t class_hash, uint32_t depth);
    std::atomic<uint64_t>* qm_consumed_cell(uint32_t match_id, uint32_t depth);
    // Add `delta` to a counter, stopping at QM_SATURATED.
    void qm_add(std::atomic<uint64_t>& counter, uint64_t delta);
    // One cascade: `start` passes or queues the first mass, then the queue drains.
    template <class F>
    void qm_cascade(F&& start);
    void qc_capture_expansion(EventId e);
    const EdgeOrbitTable* qc_orbits_or_build(StateId s);
    void qc_add_instance(uint64_t state_hash, uint32_t depth, const QcLineage* lineage,
                         uint32_t nslots, const SlotMatch* via = nullptr);
    void qc_apply(const QcInstance& inst, const SlotMatch& m, uint64_t state_hash, uint32_t depth);

    // Event canonicalization: probe key of the event signature -> the class's first event, whose
    // values event_values_of recomputes on a key hit (claim_event). The signature is computed
    // from the keys event_signature_keys_ selects.
    ConcurrentMap<uint64_t, EventId, uint64_t{0}, ~uint64_t{0}, INVALID_ID> canonical_event_map_;

    // Keyed rewrites (hgcommon/token_core.hpp). An edge's token is computed on demand from its
    // creator event's rewrite id (edge_token, event_rewrite_id) and cached in *edge_tokens_ (0: not
    // cached), where a rewrite with a known id also writes its produced tokens; the cache is built
    // at the switch to KEYED_INTERNING (note_inherited_rewrite). Each state's token
    // sum is State::token_sum (0: not computed). rewrite_map_ interns a rewrite (rule, consumed tokens)
    // as a record whose id is the rewrite id (intern_rewrite). twin_map_ maps a token sum to the
    // first raw state with that token set (claim_twin).
    std::atomic<SegmentedArray<uint64_t>*> edge_tokens_{nullptr};
    uint32_t edge_token_seg_shift_ = 0;
    ConcurrentMap<uint64_t, const hgcommon::CanonicalFormRecord*> rewrite_map_;
    std::atomic<uint32_t> next_rewrite_id_{1};
    ConcurrentMap<uint64_t, StateId, uint64_t{0}, ~uint64_t{0}, INVALID_ID> twin_map_;
    bool keyed_rewrites_{true};
    // Set at the start of each run (set_reads_rank_tuples): hgcommon::run_reads_rank_tuples.
    bool reads_rank_tuples_{false};
    // The run's keyed-rewrite state. OFF: no tokens (configuration, or stopped). ARMED: nothing is
    // interned yet. INTERNING: every rewrite is interned and every new state gets a token sum,
    // from the run's first application of an inherited match (one whose edges all predate its
    // state, match_predates_state). That is the run's first repeated rewrite: a rewrite applied
    // once has its produced tokens in one state and that state's descendants, so a repeat before it
    // would need an inherited match.
    std::atomic<uint8_t> keyed_state_{0};
    // Twin claims made; after keyed_claim_limit_ claims without a twin the run goes OFF.
    uint32_t keyed_claim_limit_ = 1024;
    // Test lever: the twin claim key is the token sum ANDed with this (set_twin_key_mask).
    uint64_t twin_key_mask_ = ~uint64_t{0};
    std::atomic<uint32_t> keyed_claims_{0};
    std::atomic<bool> twin_seen_{false};
#if HG_ENGINE_STATS
    std::atomic<uint64_t> twin_reuses_{0};
#endif
    std::atomic<uint32_t> canonical_event_count_{0};

    // Times an event signature used a RAW edge id because no edge correspondence was found.
    // Such a signature is not an isomorphism invariant, so a non-zero count means the event
    // set is approximate; see the fallback in create_event.
    std::atomic<uint64_t> event_sig_raw_fallbacks_{0};
    // Matches that named an edge their input state does not hold, and were therefore dropped
    // without being applied. See Rewriter::apply and note_invalid_match().
    std::atomic<uint64_t> invalid_matches_{0};

    // Times a canonical hash was actually COMPUTED, against the number of states that hold
    // one. Both are needed: the ratio is the question, and a raw call count says nothing
    // without the denominator.
    //
    // Incremented at the LEAVES ONLY -- compute_canonical_hash and
    // compute_and_cache_state_orbits.
    //
    // Why it exists: a 2026-07-25 profile recorded "IR canonicalization up to 3x per state"
    // and named it the biggest measurable win. get_or_compute_canonical_hash
    // has memoized into State::canonical_hash since, so the steady state is one computation per
    // state plus whatever racing writers duplicate -- and NOTHING MEASURED THAT. A number that
    // cannot be re-derived is not a number to plan against.
    mutable std::atomic<uint64_t> canonical_hash_computations_{0};
    // The individualisation-refinement search's work, summed over every host canonicalisation
    // (stats builds): calls, calls that went past the discrete fast path, leaves (discrete
    // partitions reached), nodes (individualisations), the sum of the deepest level reached,
    // calls retried at a larger depth or generator budget, and calls that fell back to the
    // unbounded implementation. Leaves per searched call is the search's size on a workload.
    mutable std::atomic<uint64_t> ir_calls_{0}, ir_searched_{0}, ir_leaves_{0}, ir_nodes_{0},
        ir_depth_sum_{0}, ir_retries_{0}, ir_fallbacks_{0};
    // Full mode: keys of canonical_form_map_ found holding a state with a different canonical
    // form (stats builds). Each one moved a state to its next probe key.
    std::atomic<uint64_t> canonical_key_collisions_{0};
    // ANDed into the canonical hash before it becomes the first probe key. All ones except in
    // tests, which narrow it so that non-isomorphic states share keys.
    uint64_t canonical_key_mask_{~uint64_t{0}};
    // The same for event signatures and for the IR key claimed under None and Automatic.
    uint64_t event_key_mask_{~uint64_t{0}};
    EventSignatureKeys event_signature_keys_{EVENT_SIG_NONE};
    std::atomic<bool> positional_event_identity_{false};

    // Genesis state: the empty state (no edges) from which all initial states originate
    // Created lazily on first call to get_or_create_genesis_state()
    // Uses lock-free initialization: 0=uninit, 1=in_progress, 2=done
    std::atomic<StateId> genesis_state_{INVALID_ID};

public:
    // capacity_scale multiplies the segment size of every append-only array, and it is the ONLY
    // way past the container ceiling. The arrays hold MAX_SEGMENTS segments of segment_size
    // elements; past that the engine raises CapacityExhausted, serves the states, events and
    // relations it has, and warns that the evolution is truncated. A caller that hits it and
    // wants the whole evolution passes a larger scale.
    //
    // IT IS A SEGMENT SIZE AND NOT A SEGMENT COUNT because the segment table is an inline array:
    // raising MAX_SEGMENTS grows every Hypergraph object by eight bytes per segment per array,
    // and this type is constructed on the stack in places with a one-megabyte limit. Segments
    // are allocated on demand, so a larger scale costs nothing until the elements exist -- only
    // the first segment of each array is bigger.
    //
    // Rounded up to a power of two, because the index decomposition is a shift and a mask.
    explicit Hypergraph(uint32_t capacity_scale = 1);

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
        size_t requested_arity,
        EventId creator_event = INVALID_ID,
        uint32_t step = 0
    );

    // A rewrite's edge ids and fresh vertex ids, consecutive, in one increment of the shared
    // counter, and create_edge at an id so taken. Edge ids increase along every state's ancestry
    // either way, which Automatic dedup and the canonical ranks' tie-break read.
    void alloc_edges_and_vertices(uint32_t num_edges, uint32_t num_vertices, EdgeId& first_edge,
                                  VertexId& first_vertex);
    EdgeId create_edge_at(EdgeId eid, const VertexId* vertices, size_t requested_arity,
                          EventId creator_event, uint32_t step);

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


    // =========================================================================
    // Edge Accessors
    // =========================================================================

    // Get vertex array for an edge (returns pointer to vertices)
    const VertexId* edge_vertices(EdgeId eid) const;

    // Get arity of an edge
    uint8_t edge_arity(EdgeId eid) const;

    // Get cached signature for an edge (computed once at creation)
    const EdgeSignature& edge_signature(EdgeId eid) const;

    // =========================================================================
    // State Management
    // =========================================================================

    // Create a new state from edge set
    // A state with no parent state is a root and indexes every edge it has; a derived state
    // indexes the edges it produced and continues its chain at `parent_state` (ancestry.hpp).
    StateId create_state(
        SparseBitset&& edge_set,
        uint32_t step = 0,
        uint64_t canonical_hash = 0,
        EventId parent_event = INVALID_ID,
        StateId parent_state = INVALID_ID,
        const EdgeId* produced = nullptr,
        uint8_t num_produced = 0
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
        const EdgeId* incr_produced = nullptr, uint8_t incr_num_produced = 0,
        uint32_t keyed_rewrite = 0
    );


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
    // `out_form`, when given, receives the state's IR canonical form from the same pass.
    uint64_t cache_state_edge_ranks(StateId state_id, const SparseBitset& edges,
                                    std::vector<uint32_t>* out_form = nullptr);

    // cache_state_edge_ranks, skipped when the table is already there. cache_ runs a full IR
    // pass every call and only then discards the result on a losing insert, so a caller that
    // may ask repeatedly for the same state -- a sampler keyed on canonical ranks does, once
    // per match -- must ask through this instead.
    void ensure_state_edge_ranks(StateId state_id, const SparseBitset& edges);

    // The canonical rank table of `state_id`, or nullptr when the state has none. One map
    // lookup; look the table up once per state and each edge's rank in it.
    const EdgeRankTable* edge_rank_table(StateId state_id) const;
    // Canonical rank of `edge` in `t`, or UINT32_MAX when `t` is null or does not hold it.
    static uint32_t edge_rank_in(const EdgeRankTable* t, EdgeId edge);

    // Event signatures that fell back to a raw edge id. Non-zero means the event identity is
    // approximate rather than canonical.
    uint64_t event_signature_raw_fallbacks() const;

    // A MATCH THAT WAS DROPPED RATHER THAN APPLIED. Rewriter::apply refuses a match naming an
    // edge the input state does not hold, and returns an empty result; the caller reads that as
    // "this rewrite produced nothing", releases its budget slots and moves on. Nothing else
    // records it, so the run simply comes back one event short with no error and no warning --
    // indistinguishable from non-determinism when compared against another thread count.
    //
    // It should be zero. A match is either produced by matching the state it is applied to, or
    // FORWARDED to a child from its parent, and the forwarding is supposed to carry only matches
    // that survive the parent's rewrite.
#if HG_ENGINE_STATS
    uint64_t invalid_matches() const;
#endif
    void note_invalid_match();

#if HG_ENGINE_STATS
    // How many times a reported canonical hash was computed. Divide by the state count for the
    // per-state figure; anything above 1.0 is duplication, and under contention a small excess
    // is expected rather than a defect (racing writers compute the same value and the last
    // store wins).
    uint64_t canonical_hash_computations() const;
    uint64_t canonical_key_collisions() const;
#endif
    // Test hook, set before evolution: the first probe key of a Full-mode state is its
    // canonical hash ANDed with `mask`. A narrow mask makes non-isomorphic states share keys.
    void set_canonical_key_mask(uint64_t mask) { canonical_key_mask_ = mask; }
    // Test hook, set before evolution: the first probe key of an event signature (and of the IR
    // key claimed under None and Automatic) is ANDed with `mask`.
    void set_event_key_mask(uint64_t mask) { event_key_mask_ = mask; }
    // Test hook, set before evolution: the first probe key of a quotient reconstruction point
    // (qc_instances_, qm_points_) is ANDed with `mask`.
    void set_quotient_key_mask(uint64_t mask) { qc_key_mask_ = mask; }
    // Set before evolution: tokens and twin reuse (hgcommon/token_core.hpp). On by default.
    void set_keyed_rewrites(bool on) {
        keyed_rewrites_ = on;
        update_keyed_state();
    }
    // Test hook, set before evolution: twin claims made without a twin before the run stops.
    void set_keyed_claim_limit(uint32_t limit) { keyed_claim_limit_ = limit; }
    // Test hook, set before evolution: the twin claim key is the token sum ANDed with `mask`. A
    // narrow mask makes states with different token sets share keys.
    void set_twin_key_mask(uint64_t mask) { twin_key_mask_ = mask; }
    // Values of keyed_state_.
    enum : uint8_t { KEYED_OFF = 0, KEYED_ARMED = 1, KEYED_INTERNING = 2 };
    uint8_t keyed_state() const { return keyed_state_.load(std::memory_order_acquire); }
    bool keyed_active() const { return keyed_state() != KEYED_OFF; }
    // keyed_state_ from the configuration: ARMED when hgcommon::keyed_rewrites_apply admits the
    // run, OFF otherwise.
    void update_keyed_state() {
        const bool on = hgcommon::keyed_rewrites_apply(
            keyed_rewrites_,
            state_canonicalization_mode_.load(std::memory_order_relaxed) ==
                StateCanonicalizationMode::Full,
            positional_event_identity_.load(std::memory_order_relaxed), reads_rank_tuples_);
        keyed_state_.store(on ? KEYED_ARMED : KEYED_OFF, std::memory_order_relaxed);
    }
    // Called by the engine before a run; re-derives keyed_state_ only when the value changes, so a
    // session's later calls keep the state the earlier ones left.
    void set_reads_rank_tuples(bool reads) {
        if (reads == reads_rank_tuples_) return;
        reads_rank_tuples_ = reads;
        update_keyed_state();
    }
    // ARMED to INTERNING, at the run's first application of an inherited match; builds the token
    // cache.
    void note_inherited_rewrite();
    // Whether every edge of a match found in state `s` is older than the edges `s` was made with.
    // Then `s`'s parent holds the same match and applies it, so applying it in `s` repeats a
    // rewrite. A state's produced edges have larger ids than every edge it holds from before.
    bool match_predates_state(StateId s, const EdgeId* edges, uint8_t n) const {
        const State& st = states_[s];
        if (st.parent_state == INVALID_ID) return false;
        if (st.num_delta_edges == 0) return true;
        const EdgeId first = st.delta_edges[0];
        for (uint8_t i = 0; i < n; ++i)
            if (edges[i] >= first) return false;
        return true;
    }
    // Whether any state took its canonical results from a twin.
    bool twin_seen() const { return twin_seen_.load(std::memory_order_relaxed); }
#if HG_ENGINE_STATS
    // States that took their canonical results from a twin with the same token set.
    uint64_t twin_reuses() const { return twin_reuses_.load(std::memory_order_relaxed); }
#endif
    // The rewrite id of (rule, consumed edges in match order), exact over their tokens:
    // REWRITE_ID_NONE when the id space is exhausted. `repeated` is set when the key was interned
    // before this call.
    uint32_t intern_rewrite(uint16_t rule, const EdgeId* consumed, uint8_t num_consumed,
                            bool& repeated);
    // The token of edge `e`, and the rewrite id of event `ev`, computed on first use: 0 and
    // REWRITE_ID_NONE when the id space is exhausted. An edge is read once its rewrite has
    // returned, or through the cache, which a rewrite with a known id fills before its state
    // exists; either way its creator Event is stored when the cache misses.
    uint64_t edge_token(EdgeId e);
    // Caches the token of edge `e` (edge_tokens_; nothing before the cache exists).
    void cache_edge_token(EdgeId e, uint64_t token) {
        if (SegmentedArray<uint64_t>* c = edge_tokens_.load(std::memory_order_acquire))
            hgcommon::atomic_ref<uint64_t>(c->slot(e, arena_)).store(token, std::memory_order_relaxed);
    }
    uint32_t event_rewrite_id(EventId ev);
    // The token sum of state `s`, computed over its edges on first use; 0 when a token is 0.
    uint64_t state_token_sum(StateId s);
    // The token sum of a new state from its parent's by the consumed and produced tokens (the
    // produced tokens from rewrite id `rid`); 0 when a token is 0.
    uint64_t child_token_sum(StateId parent, const EdgeId* consumed, uint8_t num_consumed,
                             uint32_t rid, uint8_t num_produced);
    // Whether two states hold the same token set. Fills ids[i], the i-th edge of `a` in id order,
    // and at[i], the position in `b`'s id order of the edge with the same token; both hold
    // `a`'s edge count.
    bool same_tokens(StateId a, StateId b, EdgeId* ids, uint32_t* at);
    // The first raw state with the token set of `s` (sum `sum`): `s` itself when it is the
    // first, INVALID_ID when the claim could not decide. ids and at are same_tokens' for the
    // state returned.
    StateId claim_twin(StateId s, uint64_t sum, EdgeId* ids, uint32_t* at);
    // `s` takes its twin `t`'s canonical results: the class key and, when the run keeps them,
    // the rank and orbit tables carried across by token (ids and at from claim_twin). False when
    // `t` has not published them.
    bool take_twin(StateId s, StateId t, bool ranks, bool orbits, uint64_t& key, StateId& rep,
                   const EdgeId* ids, const uint32_t* at);
    // Test hook, set before evolution: the replay refuses raw event and instance ids at or past
    // `limit` (hgcommon::QR_ID_LIMIT otherwise).
    void set_replay_id_limit(uint32_t limit) { qc_id_limit_ = limit; }
    // Applications and instances the replay dropped at the id limit; non-zero means the
    // reconstructed raw events and relations are truncated ("ReplayIdsExhausted").
    uint64_t replay_ids_refused() const { return qc_ids_refused_.load(std::memory_order_relaxed); }
    // Captures dropped because their frames could not be aligned; non-zero means the reconstructed
    // raw events and relations are truncated ("CapturesDropped").
    uint64_t captures_dropped() const { return qc_captures_dropped_.load(std::memory_order_relaxed); }
    struct IrWorkTotals {
        uint64_t calls, searched, leaves, nodes, depth_sum, retries, fallbacks;
    };
#if HG_ENGINE_STATS
    IrWorkTotals ir_work() const;
#endif
    // One canonicalisation's search work as its calls accumulate it, booked into the totals
    // by book_ir once the escalation over depths and generator budgets has settled.
    struct IrBooking {
        uint64_t calls = 0, searched = 0, leaves = 0, nodes = 0, depth_sum = 0, retries = 0;
        bool fallback = false;
    };
    void book_ir(const IrBooking& b) const;
    void book_ir_call(const hgcommon::IrWork& work, bool retried) const;

    // Quotient exploration support. try_lower_explore_depth records a shorter path to a
    // canonical state, returning true only when it improved on what was known. Depth is a
    // shortest-path label, a property of the graph, so the set of states reachable within
    // the step budget does not depend on the order paths are found. try_claim_expanded
    // succeeds exactly once per canonical state, so its matches are computed once and the
    // matches-per-instance it records are well defined.
    bool try_lower_explore_depth(StateId canonical_id, uint32_t depth);
    // One compare-exchange on the depth, for hgcommon/explore_depth_core.hpp. On failure
    // `expected` holds the current depth.
    bool explore_depth_cas(StateId canonical_id, uint32_t& expected, uint32_t desired);
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
    // create_event under `eid`, an id from reserve_event_id, with the event's rewrite id when
    // known (hgcommon/token_core.hpp; REWRITE_ID_UNSET otherwise).
    CreateEventResult create_event_at(
        EventId eid,
        StateId input_state,
        StateId output_state,
        RuleIndex rule_index,
        const EdgeId* consumed,
        uint8_t num_consumed,
        const EdgeId* produced,
        uint8_t num_produced,
        uint32_t rewrite_id
    );
    // An event id for a rewrite whose produced edges are created before its event, so that each
    // edge is created with its creator. The Event is stored by create_event_at.
    EventId reserve_event_id() { return counters_.alloc_event(); }

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
    // `out_form`, when given, receives the state's IR canonical form from the same pass.
    // The state's edge-orbit table, built if absent (qc_orbits_or_build); null for a state id
    // that names no state.
    const EdgeOrbitTable* edge_orbits(StateId s) { return qc_orbits_or_build(s); }
    uint64_t compute_and_cache_state_orbits(StateId s, const SparseBitset& edges,
                                            bool cache = true,
                                            std::vector<uint32_t>* out_form = nullptr);

    // Full mode: claims the canonical class of `sid`, whose canonical hash is `hash` and whose
    // IR canonical form is `form`, in canonical_form_map_. The probe keys are
    // hgcommon::dedup_probe_key(hash & canonical_key_mask_, n, 0, ~0); a key whose record has a
    // different form is a collision and the claim moves to the next key. Returns the class's
    // representative and its key, which is the class's identity from then on.
    struct CanonicalClaim { StateId rep; uint64_t key; bool won; };
    // A claim over a map of records: hgcommon::dedup_claim from the first probe key
    // `first_key`, with a record holding `id` and `words`. A key whose record holds other words
    // is a collision and the claim moves to the next key. `rep` is the id of the class's record,
    // `key` its key, `won` whether this call's record was stored.
    using IdentityMap = ConcurrentMap<uint64_t, const hgcommon::CanonicalFormRecord*>;
    CanonicalClaim claim_identity(IdentityMap& map, uint64_t first_key, const uint32_t* words,
                                  uint32_t n, uint32_t id);
    // The signature values of event `e` (hgcommon::event_signature_values under the run's keys),
    // read from the stored Event; returns their count. `count_fallbacks` counts each raw edge id
    // that stands in for a missing rank.
    uint32_t event_values_of(EventId e, uint64_t* out, bool count_fallbacks);
    // The canonical hash of state `s` with the `n` edges in `marked` marked
    // (hgcommon::ir_colour_pad).
    uint64_t marked_form_hash(StateId s, const EdgeId* marked, uint8_t n);
    // Event `e`'s marked forms into event_forms_, when the run's keys read them.
    void record_event_forms(EventId e);
    // The invariants of class `hash` into state_invariants_, from state `sid` and its IR canonical
    // form `form` (computed here when null), on the calling worker when this call claims the hash.
    void record_state_invariants(StateId sid, uint64_t hash, const std::vector<uint32_t>* form);
    // Event `e`'s identity, claimed in canonical_event_map_ on its signature values. The Event is
    // stored before the call; a key hit compares against the class's first event. The claim's
    // key is the event's reported signature.
    CanonicalClaim claim_event(EventId e, const uint64_t* values, uint32_t n, uint64_t sig);
    // The identity of stored event `eid` under the run's keys: claimed, with its signature and
    // canonical id written to the Event. `canonical` is the class's first event.
    struct EventIdentity { EventId canonical; bool is_canonical; };
    EventIdentity assign_event_identity(EventId eid);
    // The event class of match `m` applied from class `from_class` with output step `out_step`,
    // claimed in qc_canon_events_ on its run signature's values. `won` is set for the claim that
    // stored the class.
    CanonicalClaim claim_replay_event(const SlotMatch& m, uint64_t from_class, uint32_t out_step);
    // Automatic identity: the claim of `sid` in canonical_state_map_ from the content hash
    // `hash`; a key hit compares the content (hgcommon::content_equal) with the class's first
    // state.
    CanonicalClaim claim_content_state(StateId sid, uint64_t hash, const SparseBitset& edges);
    CanonicalClaim claim_canonical_state(StateId sid, uint64_t hash,
                                         const std::vector<uint32_t>& form);

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
    // `genesis`, when not INVALID_ID, is the initial state's genesis event (ShowGenesisEvents),
    // recorded against the root instance for reconstructed_genesis_pairs.
    void quotient_causal_seed(StateId initial_state, int max_steps, EventId genesis = INVALID_ID);

    // Under ShowGenesisEvents, the causal pairs (genesis event, application) of the
    // reconstruction: an application that consumed an edge of its initial state is caused by
    // that state's genesis event. With `reduced` (CausalTransitiveReduction) only the
    // applications that consumed no produced edge: every produced edge's producer is reached
    // from the same genesis event, so the pair is the end of a longer path. Read after the run.
    std::vector<std::pair<EventId, uint32_t>> reconstructed_genesis_pairs(bool reduced) const;
    // The genesis events recorded by quotient_causal_seed.
    size_t num_reconstructed_genesis_events() const { return qc_genesis_roots_.size(); }

    // Extend the reconstruction's depth budget for a continued run. The replay refuses to
    // expand an instance past it, so a continuation that raised the engine's budget and not
    // this one resumes the exploration and leaves the reconstruction where it stopped.
    //
    // Returns the budget this call REPLACED, or -1 when it raised nothing. Raising the bound
    // creates work -- the points the old bound left standing -- and that work is the caller's
    // to place, because the caller is what owns a thread pool. Enumerate it with
    // for_each_quotient_blocked_point and drive each point with quotient_redrive_point.
    int raise_quotient_max_steps(int max_steps);

    // The points that the OLD bound made terminal: reached, their producers and instances
    // recorded, and every transition out of them declined by the bound. Under the raised bound
    // they are ordinary interior points and each is one independent unit of work.
    //
    // Enumerate BEFORE driving any of them. The drive cascades -- a point's expansion reaches
    // deeper points and drives them inline on the same thread -- and those land in this list
    // too, so enumerating while drives are running walks a list that is growing underneath it.
    // Everything the cascade creates is already driven by its creator, so the snapshot is
    // complete.
    template <typename F>
    void for_each_quotient_blocked_point(int old_bound, int new_bound, F&& f) const {
        qc_blocked_.for_each([&](const QcPoint& p) {
            // Below the old bound the point was already driven, and AT the new bound it must
            // not be: the final depth is produced into and never read, so expanding it would
            // replay a step the run was not asked for.
            const int d = static_cast<int>(p.depth);
            if (d < old_bound || d >= new_bound) return;
            f(p.state_hash, p.depth);
        });
    }

    // Drive one blocked point: its declined transitions, then its instances against the
    // expansion's matches. Independent of every other point -- each step is claimed
    // (qm's queued flag, qc_applied_), so two threads driving the same point, or one
    // driving a point the cascade already reached, is a no-op rather than a race.
    void quotient_redrive_point(uint64_t state_hash, uint32_t depth);

    // Where a match capture's scan units run. A capture applies its match to every instance of
    // its class at every depth; it runs the first non-empty (depth, list) unit itself and hands
    // each other one to this function, which the engine sets to submit a job calling
    // qc_apply_list. Unset, the capture runs every unit itself.
    using QcSpawn = void (*)(void* ctx, Hypergraph* hg, const SlotMatch* m, uint64_t from,
                             uint32_t depth, uint32_t list);
    void set_qc_spawn(QcSpawn fn, void* ctx) { qc_spawn_ = fn; qc_spawn_ctx_ = ctx; }
    // Apply the stored match `m` to every instance in list `list` of (from, depth).
    void qc_apply_list(const SlotMatch* m, uint64_t from, uint32_t depth, uint32_t list);

    // Visit every match of the expanded representative of the canonical state `from_hash`,
    // in slots and undeduplicated -- the input to the per-instance raw reconstruction.
    template <typename F>
    void for_each_expansion_match(uint64_t from_hash, F&& f) const {
        auto r = qc_expansion_.lookup(from_hash);
        if (r.has_value()) (*r)->list.for_each([&](const SlotMatch& m) { f(m); });
    }

    // Per-instance raw reconstruction: replays the captured expansion against every raw
    // instance so quotient mode can report the raw observables it never explores. Off by
    // default while it is proven out against full-capture.
    void set_quotient_reconstruction(bool on);
    bool quotient_reconstruction() const;
    // Under the reconstruction, count the raw states each class stands for at each depth
    // (hgcommon/quotient_multiplicity_core.hpp).
    void set_quotient_multiplicity(bool on);
    bool quotient_multiplicity() const;
    // Under the reconstruction, replay the captured matches against one instance per raw state.
    // With it off the raw event and branchial counts come from the multiplicities, and no raw
    // event, causal pair or branchial pair is enumerable.
    void set_quotient_replay(bool on);
    // The number of threads that will write the replay's per-event arrays. Above eight, the
    // arrays filled through qc_ev_slot create each next segment from the first element placed in
    // a segment (SegmentedArray::set_scattered): with the second-half trigger, up to 26 of 32
    // threads each allocated and zero-filled the same 1M-entry segment. At eight threads or fewer
    // the second-half trigger is faster (bigpath n128 depth 3 quotient, 8 threads: 14.7 against
    // 15.8 ms), since the early segment is zero-filled before it is needed.
    void set_replay_writers(unsigned n) {
        qc_event_sig_.set_scattered(n > 8);
        qc_kept_->set_scattered(n > 8);
    }
    bool quotient_replay() const;
    // Every (class, depth) point the multiplicity count reached, after the run: f(class_hash,
    // depth, m), with m the raw states the class stands for at that depth (saturating at
    // QM_SATURATED).
    template <class F>
    void for_each_class_multiplicity(F&& f) const {
        qm_points_.for_each([&](uint64_t, QmPoint* p) {
            const uint64_t m = p->mass.load(std::memory_order_acquire);
            if (m) f(p->class_hash, p->depth, m);
        });
    }
    // Raw observables recovered by the reconstruction (the full-capture counts).
    uint64_t num_reconstructed_events() const;
    uint64_t num_reconstructed_raw_events() const;
    // Every reconstructed event id is below this. Ids a worker's block left unused are gaps:
    // reconstructed_event_content() is null for them.
    uint32_t reconstructed_event_id_bound() const {
        return qc_id_bound(qc_next_raw_event_);
    }
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

    // Matches the capture never recorded. The first cannot happen for a state with an edge set
    // -- the table is rebuilt on a miss -- and the second is the one-representative-per-class
    // rule doing its job.
#if HG_ENGINE_STATS
    size_t capture_dropped_no_orbits() const;
    // Reconstruction applications performed by each arena worker index, MAX_ARENA_WORKERS entries.
    // The total is a function of the run; the split shows how the replay was divided.
    std::vector<size_t> reconstruction_applications_by_worker() const;
#endif
    // Endpoint tables that were not cached and were rebuilt. A cache-miss rate: the rebuilt
    // table is the same function of the same immutable edge set, so nothing downstream moves.
#if HG_ENGINE_STATS
    size_t capture_orbit_rebuilds() const;
#endif
#if HG_ENGINE_STATS
    size_t capture_skipped_not_representative() const;
#endif
    // The first cache miss's evidence: the state it was on, INVALID_ID when there was none, and
    // why -- 1 the map held no entry, 2 it held one with no slot array.
    StateId capture_no_orbits_state() const;
    uint32_t capture_no_orbits_reason() const;

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

    // The branchial pairs of the raw unfolding: the sum over the (class, depth) points below the
    // step bound of W(c, d) * B(c), W the class multiplicity when the multiplicities ran and the
    // replay's instance count otherwise. They equal the pairs the readback enumerates
    // (OracleCorpus.MultiplicityCountsMatchTheReplay).
    uint64_t num_reconstructed_branchial() const;
#if HG_ENGINE_STATS
    size_t num_frame_alignment_disagreements() const;
    size_t num_alignment_failures() const;
    size_t num_bad_correspondences() const;
#endif

    // Visit the DISTINCT event identities the reconstruction produced, under the run's
    // EventCanonicalizationMode. The counterpart of for_each_reconstructed_causal for events:
    // comparing these against full capture's Event::signature values says WHICH identities the
    // two paths disagree about, where comparing counts only says that they do.
    template <typename F>
    void for_each_reconstructed_event_signature(F&& f) const {
        qc_canon_events_.for_each([&](uint64_t sig, const QrEventRef*) { f(sig); });
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
        const uint32_t n = qc_id_bound(qc_next_raw_event_);
        const bool by_identity = event_signature_keys() != hgcommon::EVENT_SIG_NONE;
        std::set<uint64_t> seen;
        uint32_t dense = 0;
        for (uint32_t e = 0; e < n; ++e) {
            const QcEventContent* c = qc_event_sig_.find(qc_ev_slot(e));
            if (!c || !c->written) continue;
            if (by_identity) {
                const uint64_t* sig = qc_event_runsig_.find(qc_ev_slot(e));
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
        const uint32_t n = qc_id_bound(qc_next_raw_event_);
        for (uint32_t i = 0; i < n; ++i) {
            const QcEventContent* c = qc_event_sig_.find(qc_ev_slot(i));
            if (c && c->written) f(c->triple_hash());
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
    // walks the pairs the online reduction kept (TR on). Both are maintained as the replay runs,
    // so either is a walk over stored pairs.
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
            // The kept sets ARE the reduction: qr_apply decides it online, exactly (see
            // quotient_replay_core.hpp), so reading it is a walk over what was kept.
            const uint32_t n = qc_id_bound(qc_next_raw_event_);
            for (uint32_t c = 0; c < n; ++c) {
                const QcKept* k = qc_kept_->find(qc_ev_slot(c));
                if (!k) continue;
                for (uint32_t i = 0; i < k->n; ++i) f(id(k->at(i)), id(c));
            }
        } else {
            qc_causal_pairs_for_each([&](uint64_t k) {
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
        // they imply on disc-l3a2g2r2 depth 3. Each instance's list is flattened and paired by
        // hgcommon::qr_instance_branchial_pairs, which the device readback
        // (QeState::reconstructed_pairs_host) calls too.
        const uint32_t n = qc_applied_slot_bound();
        std::vector<QcAppliedMatch> flat;
        std::vector<std::pair<uint32_t, uint32_t>> entries;
        for (uint32_t i = 0; i < n; ++i) {
            const LockFreeList<QcAppliedMatch>* lst = qc_inst_applied_.find(i);
            if (!lst) continue;
            flat.clear();
            lst->for_each([&](const QcAppliedMatch& m) { flat.push_back(m); });
            hgcommon::qr_instance_branchial_pairs(
                flat.data(), static_cast<uint32_t>(flat.size()), entries,
                [&](uint32_t lo, uint32_t hi) { f(id(lo), id(hi)); });
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

    uint64_t observable_num_events() const;
    size_t observable_num_causal_edges() const;
    size_t observable_num_causal_pairs(bool transitively_reduced) const;
    uint64_t observable_num_branchial() const;




    // Whether causal edges are keyed by canonical edge orbit (quotient exploration). Set
    // by the evolution engine before evolving; read when minting causal edge keys.
    void set_quotient_causal(bool q);
    bool quotient_causal() const;

    // Which artifacts this run builds. Set before evolving and read by the workers, so the
    // two components are stored as atomics like every other pre-evolution switch here.
    void set_record_set(RecordSet r);
    RecordSet record_set() const;

    // The invariants of the class with canonical hash `hash` (RecordSet::state_invariants), or
    // nullptr when the run did not record them. Read after the run.
    const hgcommon::StateInvariantRecord* state_invariants(uint64_t hash) const {
        const auto cell = state_invariants_.lookup(hgcommon::avoid_reserved_keys(hash));
        return cell ? hgcommon::atomic_ref<const hgcommon::StateInvariantRecord*>(**cell)
                          .load(std::memory_order_acquire)
                    : nullptr;
    }

    // Create a genesis event for an initial state.
    // This synthetic event connects the empty genesis state to the initial state.
    // It "produces" all edges in the initial state, enabling causal tracking from gen 0.
    // Returns the genesis event ID.
    // The largest initial state a genesis event can describe: it produces every initial edge
    // and Event::num_produced is one byte wide.
    static constexpr size_t MAX_GENESIS_EDGES = 255;

    EventId create_genesis_event(StateId initial_state, const EdgeId* edges,
                                 size_t requested_num_edges);

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

    // Compute content-ordered hash for Automatic state canonicalization mode
    // Hashes edge contents in order by edge ID: (arity, v1, v2, ...) for each edge
    // Fast but not isomorphism-invariant.
    uint64_t compute_content_ordered_hash(const SparseBitset& edges) const;

    // The canonical hash (isomorphism-invariant). The event path resolves representatives
    // through it, and it is the hash a state reports.
    // `out_form`, when given, receives the state's IR canonical form from the same pass.
    uint64_t compute_canonical_hash(const SparseBitset& edges,
                                    std::vector<uint32_t>* out_form = nullptr) const;


    // Count edges in a state
    uint32_t count_state_edges(StateId sid) const;
};

}  // namespace engine
}  // namespace HG_NAMESPACE