#pragma once
//
// Expansion capture, device side: the per-class list of matches in FRAME SLOTS -- the device
// twin of Hypergraph::qc_capture_expansion and for_each_expansion_match
// (hypergraph/src/hypergraph.cpp).
//
// WHY THIS EXISTS. Under quotient exploration only one raw state per isomorphism class is
// expanded, so the raw events the other instances would have produced are never created. The
// host recovers them by REPLAY: it records each class's matches once, expressed in the class's
// own frame rather than in any raw state's edge ids, then replays that record against every
// instance of the class. This file is the record; the replay is the next step of the port.
//
// The device currently has the causal half of that machinery (quotient_causal.hpp: the
// depth-indexed producer-set DP, keyed on orbits) but not this half, so under quotient it
// cannot produce the counts the CPU serves through observable_num_events(). The measured gap is
// pinned by gpu/tests CanonicalEventCount.ReconstructionGapIsStillOpen (CPU 21 / GPU 23 on the
// rank frame, CPU 144 / GPU 15 under quotient + mode None).
//
// WHAT A SLOT IS. Defined once, in hgcommon/slot_core.hpp, and read from there by both engines
// -- the host fills a whole state at once (slots_from_orbits), this file reads one edge at a
// time (slot_rank), and the two forms are asserted equal. Nothing about the rule is restated
// here, because a second statement of it is exactly how the two would drift.
//
// ONE CLAIM, NOT TWO. The host keeps qc_expansion_rep_ (which raw state's events define the
// class's expansion) and qc_frame_ (which raw state's labelling defines the class's slots) as
// separate claims, and aligns a non-frame state's edges onto the frame when they differ. Here
// they are ONE claim, so the state that defines the expansion is by construction the state
// whose labelling the slots are in, and the capture path needs no alignment. Alignment is still
// required to replay a record against an arbitrary INSTANCE; it belongs with the replay.

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/cuda_check.hpp"
#include "hg_gpu/exploration.hpp"   // DedupMap
#include "hgcommon/ir_core.hpp"     // ir_isort_u64
#include "hgcommon/slot_core.hpp"  // slot_rank -- the frame-slot rule, shared with the host

#include <cuda/atomic>

namespace hg_gpu {

// One captured match of a canonical class, in that class's frame slots. The slot arrays live in
// the expansion word arena at arr_offset: consumed | produced | surv_from | surv_to,
// contiguously -- the same layout DeviceCanonicalTransition uses for its orbit arrays.
struct DeviceSlotMatch {
    uint64_t to_hash = 0;
    uint32_t id = 0;              // dense; the replay's (instance, match) claim keys on it
    uint32_t rule = 0;
    uint32_t from_slots = 0, to_slots = 0;
    uint32_t num_consumed = 0, num_produced = 0, num_survivors = 0;
    uint32_t arr_offset = 0;
};

// A captured match reference, bucketed by from_hash; the node carries its exact hash so the
// walkers can filter a shared bucket.
struct QeMatchRef {
    uint64_t from_hash;
    uint32_t record;
};

// (class, depth) as one key. Same mixing as the DP's qc_key with orbit 0, because the two
// index the same (class, depth) space and a reader comparing them must not have to check that
// two spellings agree.
__device__ __forceinline__ uint64_t qe_inst_key(uint64_t state_hash, uint32_t depth) {
    uint64_t h = 1469598103934665603ULL;
    h ^= state_hash; h *= 1099511628211ULL;
    h ^= (static_cast<uint64_t>(depth) << 32); h *= 1099511628211ULL;
    return h;
}

// One raw occurrence of a canonical class, at one depth. `prod_offset` addresses `nslots` words
// in the expansion arena: per FRAME SLOT, the event that produced the edge now in that slot, or
// kQeNoProducer for an edge the initial state came with.
//
// Slots rather than edge ids is the whole point: the class's captured matches are in frame
// slots, so an instance built from any raw state of the class replays them without knowing
// which raw edges the frame state happened to have.
struct DeviceQcInstance {
    uint32_t id = 0;           // dense; the replay's (instance, match) claim keys on it
    uint32_t nslots = 0;
    uint32_t prod_offset = 0;
};

// An instance reference bucketed by key(hash, depth); the node carries the exact key so a
// shared bucket can be filtered, exactly as QeMatchRef does for the matches.
struct QeInstRef {
    uint64_t key;
    uint32_t record;
};

// One application recorded against the instance it expanded. The consumed slots are carried by
// OFFSET into the expansion arena rather than by pointer: the arena is device memory reached
// through the view, and an offset stays valid however the view is passed.
struct QeAppliedMatch {
    uint32_t instance;          // the bucket is shared, so the record carries its own instance
    uint32_t match_id;
    uint32_t event;
    uint32_t num_consumed;
    uint32_t consumed_offset;   // into arr_words
};

// One KEPT causal pair, bucketed by its consumer; the node carries the consumer so a shared
// bucket can be filtered, as every other bucketed record here does.
struct QePredRef {
    uint32_t consumer;
    uint32_t producer;
};

// No event produced this slot's edge: it was in the initial state. Matches the host's
// Hypergraph::QC_NO_PRODUCER.
inline constexpr uint32_t kQeNoProducer = 0xFFFFFFFFu;

struct QeView {
    typename Pool<DeviceSlotMatch>::DeviceView          matches;
    typename LockFreeList<QeMatchRef>::DeviceView       by_from;   // bucket(from_hash)

    typename Pool<DeviceQcInstance>::DeviceView         instances;
    typename LockFreeList<QeInstRef>::DeviceView        by_key;    // bucket(key(hash, depth))
    uint32_t* inst_next_id;    // device atomic; dense instance ids

    // Claims an (instance, match) application. An application mints a raw event, so unlike the
    // producer-set DP it is not idempotent and the pair must be claimed exactly once.
    DedupMap::DeviceView applied;
    uint32_t* next_raw_event;  // device atomic; dense raw-event ids

    // Slots the frame MOVED -- resolved through a state that did not hold the frame, and landing
    // somewhere other than the state's own slot. Counting corrections rather than lookups is
    // what makes it evidence: a lookup that returns the state's own slot changes nothing.
    // align_fail is the host's qc_align_fail_ / qc_align_badcorr_.
    uint32_t* align_moved;
    uint32_t* align_fail;

    // Distinct run identities and their count. Empty under EVENT_SIG_NONE, where every
    // application is its own event and the raw count is already the answer.
    DedupMap::DeviceView canon_seen;
    uint32_t* num_canon;
    EventSignatureKeys keys;

    // The reconstructed causal relation. `pairs` claims each (producer, consumer) exactly once;
    // `num_causal_edges` counts every consumed-edge occurrence, so a pair joined by several
    // edges is one pair and several edges -- the two the host reports separately.
    DedupMap::DeviceView causal_pairs;
    uint32_t* num_causal_pairs;
    uint32_t* num_causal_edges;

    // The KEPT predecessor adjacency: the reduction's only stored structure. A pair enters it
    // exactly when it survives, so the walk testing the next pair reads the reduction rather
    // than the full relation -- keeping an edge costs one list push instead of an
    // ancestors x descendants cross-product of map inserts.
    typename LockFreeList<QePredRef>::DeviceView preds;
    // The same decision, keyed so the relation can be handed back as a SET: preds is a chained
    // list the host cannot walk, and a count says only THAT two engines disagree. Written at
    // the point the counter increments, so the two cannot drift.
    DedupMap::DeviceView reduced_pairs;
    uint32_t* num_reduced_pairs;

    // The reconstructed branchial relation. `inst_applied` is bucketed by instance id: an
    // application publishes itself there and then scans the bucket, so the later of any two
    // sees the earlier. `branchial_pairs` claims the unordered pair, since both sides can see
    // each other when their pushes and scans interleave.
    // Per raw event, the schedule-stable content triple hash(input class, output class, rule).
    // Indexed by raw event id, which is what the pair keys hold; the triple is what a
    // cross-engine comparison can be made on. The host's qc_event_sig_.
    uint64_t* event_sig;
    uint32_t  event_sig_capacity;

    typename LockFreeList<QeAppliedMatch>::DeviceView inst_applied;
    DedupMap::DeviceView branchial_pairs;
    uint32_t* num_branchial;

    // canonical hash -> (StateId + 1) of the state whose matches define this class's expansion.
    // +1 because the map reserves 0 as its EMPTY sentinel, so a raw key of StateId 0 could never
    // be stored -- the same offset, for the same reason, as the None-mode dedup key.
    DedupMap::DeviceView rep;

    // canonical hash -> (StateId + 1) of the state whose labelling is this class's FRAME, and
    // that state's step. Separate from `rep`: a class is given a frame by both endpoints of
    // every captured transition, so a class first seen as an output owns its frame from a state
    // that need never expand. The step is what the Automatic signature keys on.
    DedupMap::DeviceView frame;
    DedupMap::DeviceView frame_step;   // canonical hash -> step + 1

    // Bump arena for the matches' slot arrays.
    uint32_t* arr_words;
    uint32_t* arr_cursor;      // device atomic
    uint32_t  arr_capacity;

    uint32_t* next_id;         // device atomic; dense match ids
    uint32_t  max_steps = 0;
    // How deep the replay may recurse before the per-thread stack runs out. The cycle
    // qe_apply -> qe_add_instance -> qe_drive_instance descends once per depth, so this is a
    // property of the stack the engine was given (EngineState::qe_max_recursion_depth) and not
    // of the workload. Past it the replay stops and records, which loses deep events and says
    // so; without it the next frame faults and the whole run returns nothing.
    uint32_t  max_recursion_depth = 0;
    uint32_t  enabled   = 0;
};

// The rendezvous is mutually recursive: publishing an instance drives the matches, publishing a
// match drives the instances, and an application publishes a child instance. Declared here so
// each publisher can drive without the definitions having to be ordered around each other.
struct DeviceQcInstance;
__device__ inline void qe_drive_instance(DeviceState ds, QeView qe,
                                         const DeviceQcInstance& inst,
                                         uint64_t state_hash, uint32_t depth);
__device__ inline void qe_drive_match(DeviceState ds, QeView qe, const DeviceSlotMatch& m,
                                      uint64_t from_hash);

// Bucket a hash into a list's key space. Same mixing as the DP's qc_bucket, so a shared bucket
// count distributes the two the same way.
__device__ __forceinline__ uint32_t qe_bucket(uint64_t h, uint32_t num_keys) {
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return static_cast<uint32_t>(h % (num_keys ? num_keys : 1u));
}

// The frame slot of `edge` in `sid`: its rank under (orbit, EdgeId).
//
// Computed by counting rather than by materialising the order, because the count is what the
// host's stable_sort produces and a device sort per lookup would not be. O(n) in the state's
// edge count, with n bounded by the state rather than by the run.
//
// UINT32_MAX when the edge is not in the state or the state has no orbits -- the caller drops
// the capture rather than recording a slot that means nothing, because a record built from a
// wrong slot replays as a wrong event and would be invisible.
__device__ __forceinline__ uint32_t qe_slot_of(DeviceState ds, StateId sid, EdgeId edge) {
    if (!ds.state_edge_orbit || sid >= ds.max_states) return UINT32_MAX;
    const StateEdgeSlice sl = ds.state_edge_slices[sid];

    // Locate the edge; the slice is sorted ascending (the DP binary-searches it the same way).
    uint32_t lo = 0, hi = sl.count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (ds.state_edge_ids[sl.offset + mid] < edge) lo = mid + 1; else hi = mid;
    }
    if (lo >= sl.count || ds.state_edge_ids[sl.offset + lo] != edge) return UINT32_MAX;
    if (ds.state_edge_orbit[sl.offset + lo] == UINT32_MAX) return UINT32_MAX;

    // The rule itself is hgcommon's, not this file's: the host records the same coordinates
    // (hypergraph.cpp, via slots_from_orbits) and two readings that drift by one tie-break
    // would replay wrong events invisibly.
    return hgcommon::slot_rank(ds.state_edge_orbit + sl.offset, sl.count, lo);
}

// The canonical rank of `edge` within `sid` -- its position in the state's canonical order,
// from the same individualization-refinement pass that produced the state's exact hash.
// UINT32_MAX when the edge is absent or no rank was computed.
__device__ __forceinline__ uint32_t qe_rank_of(DeviceState ds, StateId sid, EdgeId edge) {
    if (!ds.state_edge_rank || sid >= ds.max_states) return UINT32_MAX;
    const StateEdgeSlice sl = ds.state_edge_slices[sid];
    uint32_t lo = 0, hi = sl.count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (ds.state_edge_ids[sl.offset + mid] < edge) lo = mid + 1; else hi = mid;
    }
    if (lo >= sl.count || ds.state_edge_ids[sl.offset + lo] != edge) return UINT32_MAX;
    return ds.state_edge_rank[sl.offset + lo];
}

// Register `sid` as the frame of its class if no state holds it yet, recording the step the
// signature reads. Idempotent, and the winner is whichever state gets there first -- which is
// all the frame has to be, since every state of the class is isomorphic to it.
__device__ __forceinline__ void qe_register_frame(QeView qe, uint64_t class_hash, StateId sid,
                                                  uint32_t step) {
    if (qe.frame.insert_if_absent(class_hash, static_cast<uint32_t>(sid) + 1u).inserted)
        qe.frame_step.insert_if_absent(class_hash, step + 1u);
}

// The slot `edge` of `sid` occupies IN ITS CLASS'S FRAME.
//
// When `sid` holds the frame this is its own slot. Otherwise the two states are isomorphic and
// the correspondence is by canonical position: the frame's edge of equal rank is this edge's
// image, and its slot is the answer. The correspondence is defined only up to an automorphism,
// which is the harmless freedom -- an automorphism permutes the frame coherently and carries
// matches to matches. Each state using its OWN labelling is what is not harmless, and is what
// this removes.
//
// UINT32_MAX when no image exists, which every caller turns into dropping the capture rather
// than recording a slot that means nothing.
__device__ inline uint32_t qe_frame_slot_of(DeviceState ds, QeView qe, uint64_t class_hash,
                                            StateId sid, EdgeId edge) {
    const auto held = qe.frame.lookup_waiting(class_hash);
    if (!held.found || held.value == 0) { atomicAdd(qe.align_fail, 1u); return UINT32_MAX; }
    const StateId frame = static_cast<StateId>(held.value - 1u);
    if (frame == sid) return qe_slot_of(ds, sid, edge);

    if (!ds.state_edge_rank || !ds.state_edge_orbit || frame >= ds.max_states) {
        atomicAdd(qe.align_fail, 1u);
        return UINT32_MAX;
    }
    const uint32_t r = qe_rank_of(ds, sid, edge);
    if (r != UINT32_MAX) {
        const StateEdgeSlice fsl = ds.state_edge_slices[frame];
        for (uint32_t k = 0; k < fsl.count; ++k) {
            if (ds.state_edge_rank[fsl.offset + k] != r) continue;
            const uint32_t fs = hgcommon::slot_rank(ds.state_edge_orbit + fsl.offset, fsl.count, k);
            if (fs != qe_slot_of(ds, sid, edge)) atomicAdd(qe.align_moved, 1u);
            return fs;
        }
    }
    atomicAdd(qe.align_fail, 1u);
    return UINT32_MAX;
}

// Survivor pairs one capture can hold in local scratch. A class with more surviving edges than
// this records kScratchOverflow and drops the capture: the events reachable only through it are
// then missing, which the warning reports rather than silently mis-attributing. Matches the
// DP's kQcMaxSurvivors so the two halves fail at the same size.
constexpr uint32_t kQeMaxSurvivors = 256;

// Capture one raw event as its class's expansion match, in frame slots.
//
// Only the class's claimed state contributes: the first parent to claim the class defines both
// the expansion and the frame, and every later parent of the same class returns immediately.
// That is what makes the record a property of the CLASS rather than of whichever raw state
// happened to be expanded first by this schedule.
//
// `depth` is the parent's depth (the event's step - 1).
__device__ inline void qe_capture_expansion(DeviceState ds, QeView qe,
                                            StateId parent, StateId child, EventId event,
                                            uint32_t rule, uint32_t depth) {
    if (!qe.enabled || depth > qe.max_steps) return;

    const uint64_t from = ds.state_canonical_hash[parent];
    const uint64_t to   = ds.state_canonical_hash[child];

    // One raw state's matches define the class's expansion; every later parent of the same class
    // drops out here, so the record is a property of the CLASS and not of the schedule.
    const uint32_t claim = static_cast<uint32_t>(parent) + 1u;
    if (qe.rep.insert_if_absent(from, claim).value != claim) return;

    // Both endpoints are given a frame before any slot is taken, so every slot below resolves.
    qe_register_frame(qe, from, parent, depth);
    qe_register_frame(qe, to, child, depth + 1u);

    const DeviceEvent& ev = ds.event_pool.at(event);
    const uint32_t nc = ev.num_consumed, np = ev.num_produced;

    uint32_t consumed[kMaxPatternEdges];
    uint32_t produced[kMaxPatternEdges];
    for (uint32_t i = 0; i < nc; ++i) {
        consumed[i] = qe_frame_slot_of(ds, qe, from, parent, ev.consumed_edges[i]);
        if (consumed[i] == UINT32_MAX) return;   // no frame slot: drop rather than corrupt
    }
    for (uint32_t i = 0; i < np; ++i) {
        produced[i] = qe_frame_slot_of(ds, qe, to, child, ev.produced_edges[i]);
        if (produced[i] == UINT32_MAX) return;
    }

    // Survivors: child edges that were not freshly produced passed through from the parent (the
    // child's slice is parent-minus-consumed plus produced by construction). Recorded as
    // (slot in parent << 32 | slot in child) so one sort orders the pairs.
    uint64_t surv[kQeMaxSurvivors];
    uint32_t ns = 0;
    {
        const StateEdgeSlice csl = ds.state_edge_slices[child];
        for (uint32_t k = 0; k < csl.count; ++k) {
            const EdgeId oe = ds.state_edge_ids[csl.offset + k];
            bool produced_here = false;
            for (uint32_t j = 0; j < np; ++j)
                if (ev.produced_edges[j] == oe) { produced_here = true; break; }
            if (produced_here) continue;
            const uint32_t ps = qe_frame_slot_of(ds, qe, from, parent, oe);
            const uint32_t cs = qe_frame_slot_of(ds, qe, to, child, oe);
            if (ps == UINT32_MAX || cs == UINT32_MAX) continue;
            if (ns >= kQeMaxSurvivors) { ds.errors.record(ErrorKind::kScratchOverflow); return; }
            surv[ns++] = (static_cast<uint64_t>(ps) << 32) | cs;
        }
        hgcommon::ir_isort_u64(surv, ns);
    }

    // Copy the slot arrays into the expansion arena, then publish the record.
    const uint32_t need = nc + np + 2u * ns;
    uint32_t off = 0;
    if (need) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> cur(*qe.arr_cursor);
        off = cur.fetch_add(need, cuda::memory_order_relaxed);
        if (off + need > qe.arr_capacity) { ds.errors.record(ErrorKind::kQcNodes); return; }
        uint32_t* w = qe.arr_words + off;
        for (uint32_t i = 0; i < nc; ++i) *w++ = consumed[i];
        for (uint32_t i = 0; i < np; ++i) *w++ = produced[i];
        for (uint32_t i = 0; i < ns; ++i) *w++ = static_cast<uint32_t>(surv[i] >> 32);
        for (uint32_t i = 0; i < ns; ++i) *w++ = static_cast<uint32_t>(surv[i]);
    }

    const uint32_t rec = qe.matches.claim();
    if (rec == Pool<DeviceSlotMatch>::kInvalid) { ds.errors.record(ErrorKind::kQcNodes); return; }
    DeviceSlotMatch& m = qe.matches.at(rec);
    m.to_hash = to;
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> nid(*qe.next_id);
        m.id = nid.fetch_add(1u, cuda::memory_order_relaxed);
    }
    m.rule = rule;
    m.from_slots = ds.state_edge_slices[parent].count;
    m.to_slots   = ds.state_edge_slices[child].count;
    m.num_consumed = nc; m.num_produced = np; m.num_survivors = ns;
    m.arr_offset = off;

    if (qe.by_from.push(qe_bucket(from, qe.by_from.num_keys), QeMatchRef{from, rec}) == INVALID_ID)
        ds.errors.record(ErrorKind::kQcNodes);

    qe_drive_match(ds, qe, m, from);
}

// Walk the captured matches of one class. The bucket is shared, so the exact hash on each node
// is what selects this class's records out of it.
template <typename F>
__device__ inline void qe_for_each_match_from(QeView qe, uint64_t from_hash, F&& f) {
    qe.by_from.for_each(qe_bucket(from_hash, qe.by_from.num_keys), [&](const QeMatchRef& r) {
        if (r.from_hash != from_hash) return;
        f(qe.matches.at(r.record));
    });
}

// Reserve `n` words of the expansion arena. Returns UINT32_MAX when the arena is exhausted,
// which the caller reports as a capacity overflow rather than writing past the end.
__device__ __forceinline__ uint32_t qe_alloc_words(DeviceState ds, QeView qe, uint32_t n) {
    if (n == 0) return 0;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> cur(*qe.arr_cursor);
    const uint32_t off = cur.fetch_add(n, cuda::memory_order_relaxed);
    if (off + n > qe.arr_capacity) { ds.errors.record(ErrorKind::kQcNodes); return UINT32_MAX; }
    return off;
}

// Record one instance of `state_hash` at `depth`, whose per-slot producers are already written
// at `prod_offset`. The device twin of Hypergraph::qc_add_instance.
__device__ inline uint32_t qe_add_instance(DeviceState ds, QeView qe, uint64_t state_hash,
                                           uint32_t depth, uint32_t prod_offset,
                                           uint32_t nslots) {
    if (!qe.enabled || depth > qe.max_steps) return UINT32_MAX;

    const uint32_t rec = qe.instances.claim();
    if (rec == Pool<DeviceQcInstance>::kInvalid) {
        ds.errors.record(ErrorKind::kQcNodes);
        return UINT32_MAX;
    }
    DeviceQcInstance& inst = qe.instances.at(rec);
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> nid(*qe.inst_next_id);
        inst.id = nid.fetch_add(1u, cuda::memory_order_relaxed);
    }
    inst.nslots      = nslots;
    inst.prod_offset = prod_offset;

    // Published only after the record is complete: a walker that reaches the reference must not
    // find a half-written instance.
    __threadfence();
    const uint64_t key = qe_inst_key(state_hash, depth);
    if (qe.by_key.push(qe_bucket(key, qe.by_key.num_keys), QeInstRef{key, rec}) == INVALID_ID)
        ds.errors.record(ErrorKind::kQcNodes);
    return rec;
}

// The root instance of a class: every slot's edge came with the initial state, so no event
// produced any of them. Claims the class frame first, so the root's producer vector and the
// expansion captured from it are in the SAME labelling by construction -- the host does the
// same, and for the same reason.
__device__ inline void qe_seed_root_instance(DeviceState ds, QeView qe, StateId root) {
    if (!qe.enabled) return;
    const uint64_t h = ds.state_canonical_hash[root];
    const uint32_t nslots = ds.state_edge_slices[root].count;

    qe_register_frame(qe, h, root, 0u);

    const uint32_t off = qe_alloc_words(ds, qe, nslots);
    if (off == UINT32_MAX) return;
    for (uint32_t i = 0; i < nslots; ++i) qe.arr_words[off + i] = kQeNoProducer;
    const uint32_t rec = qe_add_instance(ds, qe, h, 0u, off, nslots);
    if (rec != UINT32_MAX) qe_drive_instance(ds, qe, qe.instances.at(rec), h, 0u);
}

// Visit every instance recorded for `state_hash` at `depth`.
template <typename F>
__device__ inline void qe_for_each_instance(QeView qe, uint64_t state_hash, uint32_t depth,
                                            F&& f) {
    const uint64_t key = qe_inst_key(state_hash, depth);
    qe.by_key.for_each(qe_bucket(key, qe.by_key.num_keys), [&](const QeInstRef& r) {
        if (r.key == key) f(qe.instances.at(r.record));
    });
}

// The (instance, match) claim key. Same mixing as the host's apply_key, and nudged off both
// map sentinels for the same reason.
__device__ __forceinline__ uint64_t qe_apply_key(uint32_t instance, uint32_t match) {
    uint64_t k = 1469598103934665603ULL;
    k ^= instance; k *= 1099511628211ULL;
    k ^= match;    k *= 1099511628211ULL;
    return (k == 0 || k == ~0ULL) ? 1 : k;
}

__device__ inline void qe_apply(DeviceState ds, QeView qe, const DeviceQcInstance& inst,
                                const DeviceSlotMatch& m, uint64_t state_hash, uint32_t depth);

// Instance side of the rendezvous: replay every match already captured for this class.
__device__ inline void qe_drive_instance(DeviceState ds, QeView qe,
                                         const DeviceQcInstance& inst,
                                         uint64_t state_hash, uint32_t depth) {
    if (depth >= qe.max_steps) return;   // final-depth instances are recorded, never expanded
    if (depth >= qe.max_recursion_depth) {
        // Out of stack, not out of work. Recorded and returned rather than descended: one more
        // frame faults, and a fault takes the entire run's result with it.
        ds.errors.record(ErrorKind::kScratchOverflow);
        return;
    }
    // Published before scanning; pairs with the fence on the match side so a concurrent
    // instance and match cannot both miss each other.
    __threadfence();
    qe_for_each_match_from(qe, state_hash, [&](const DeviceSlotMatch& m) {
        qe_apply(ds, qe, inst, m, state_hash, depth);
    });
}

// Match side of the rendezvous: replay this match against every instance already standing at
// this class, at every depth it could stand at.
__device__ inline void qe_drive_match(DeviceState ds, QeView qe, const DeviceSlotMatch& m,
                                      uint64_t from_hash) {
    __threadfence();
    for (uint32_t d = 0; d < qe.max_steps; ++d) {
        qe_for_each_instance(qe, from_hash, d, [&](const DeviceQcInstance& inst) {
            qe_apply(ds, qe, inst, m, from_hash, d);
        });
    }
}

// Walk scratch. A cone wider than this records kScratchOverflow and the pair under test is
// KEPT: over-keeping is reported and recoverable, where dropping silently removes a causal edge.
constexpr uint32_t kQeReachStack = 192;

// Is `consumer` already reachable from `producer` over the KEPT predecessors?
//
// Backward walk from the consumer, pruned to ids above the producer. Raw reconstructed event
// ids increase along every causal edge -- a producer wrote the slot its consumer reads, so its
// application minted the lower id -- which is what makes the prune sound. That is a property of
// THIS id assignment and not of the engine, which is why the host had to parameterise the same
// prune for canonical ids.
__device__ inline bool qe_reachable(DeviceState ds, QeView qe, uint32_t producer,
                                    uint32_t consumer) {
    if (producer >= consumer) return false;

    // One array, holding every node the walk has REACHED, with a cursor separating the ones
    // already expanded from the ones still to expand. That makes it the visited set and the
    // worklist at once, which is what keeps the walk linear in the cone's NODES: a node reached
    // by k paths is expanded once, not k times. A plain stack without the membership test costs
    // one expansion per PATH, which is exponential in the worst case -- though not usually: this
    // walks the KEPT predecessors, and reduction leaves most events with one, so the paths and
    // the nodes are close on ordinary workloads and the bound is what is being fixed rather than
    // a measured cost. Same shape as the host's walk (Hypergraph::qc_reachable), which carries an explicit
    // visited set beside its stack; here the two are one array because the cursor never goes
    // backwards.
    uint32_t seen[kQeReachStack];
    uint32_t n = 0, cursor = 0;
    seen[n++] = consumer;
    while (cursor < n) {
        const uint32_t x = seen[cursor++];
        bool found = false, full = false;
        qe.preds.for_each(qe_bucket(hgcommon::id_key(x), qe.preds.num_keys),
                          [&](const QePredRef& r) {
            if (found || r.consumer != x) return;
            if (r.producer == producer) { found = true; return; }
            if (r.producer <= producer) return;            // outside the cone
            for (uint32_t i = 0; i < n; ++i) if (seen[i] == r.producer) return;   // already reached
            if (n < kQeReachStack) seen[n++] = r.producer;
            else                   full = true;
        });
        if (found) return true;
        if (full) {
            // The answer would be a guess, and a wrong "reachable" DROPS a causal edge.
            ds.errors.record(ErrorKind::kScratchOverflow);
            return false;
        }
    }
    return false;
}

// One application of `m` to `inst`: mint the raw event, then mint the child instance whose
// producers this application determines.
__device__ inline void qe_apply(DeviceState ds, QeView qe, const DeviceQcInstance& inst,
                                const DeviceSlotMatch& m, uint64_t state_hash, uint32_t depth) {
    if (!qe.enabled || depth >= qe.max_steps) return;

    // Exactly once, however many times the two sides reach this pair.
    const uint64_t ck = qe_apply_key(inst.id, m.id);
    if (!qe.applied.insert_if_absent(ck, 1u).inserted) return;
    // The capture and the instance disagree on the class's width: drop rather than corrupt.
    if (m.from_slots != inst.nslots) return;

    // The raw event this instance's copy of the match stands for. An id suffices -- counts and
    // causal edges are expressed over ids, so no Event record has to be materialised.
    uint32_t ev;
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> nre(*qe.next_raw_event);
        ev = nre.fetch_add(1u, cuda::memory_order_relaxed);
    }

    // The event's content triple: isomorphism-invariant and independent of the schedule, so it
    // is the identity a cross-run or cross-engine comparison of the relations is made on.
    if (ev < qe.event_sig_capacity) {
        uint64_t s = 1469598103934665603ULL;
        s ^= state_hash; s *= 1099511628211ULL;
        s ^= m.to_hash;  s *= 1099511628211ULL;
        s ^= m.rule;     s *= 1099511628211ULL;
        qe.event_sig[ev] = s;
    }

    // The run's event identity. Under EVENT_SIG_NONE there is none: every application is its
    // own event, and the raw count above is what a caller is told.
    if (qe.keys != hgcommon::EVENT_SIG_NONE) {
        // The canonical OUTPUT state's step, which is one value per class, not the depth this
        // instance sits at. Falls back to the depth when the output class holds no frame.
        uint32_t out_step = depth;
        const auto fs = qe.frame_step.lookup_waiting(m.to_hash);
        if (fs.found && fs.value != 0) out_step = fs.value - 1u;

        const uint32_t* a = qe.arr_words + m.arr_offset;
        uint64_t csig = hgcommon::event_signature(
            qe.keys, state_hash, m.to_hash, out_step, static_cast<uint16_t>(m.rule),
            a, static_cast<uint8_t>(m.num_consumed),
            a + m.num_consumed, static_cast<uint8_t>(m.num_produced));
        if (csig == 0 || csig == ~0ULL) csig = 1;
        if (qe.canon_seen.insert_if_absent(csig, 1u).inserted) atomicAdd(qe.num_canon, 1u);
    }

    // Causal: one relationship per consumed edge that has a producer. Fed in DESCENDING
    // producer order, so nearer producers enter the relation before farther ones -- the same
    // discipline the full-capture rendezvous keeps.
    if (ds.record_causal) {
        uint32_t producers[kMaxPatternEdges];
        uint32_t np = 0;
        const uint32_t* cs = qe.arr_words + m.arr_offset;
        for (uint32_t i = 0; i < m.num_consumed && np < kMaxPatternEdges; ++i) {
            const uint32_t s = cs[i];
            if (s >= inst.nslots) continue;
            const uint32_t p = qe.arr_words[inst.prod_offset + s];
            if (p != kQeNoProducer) producers[np++] = p;
        }
        // Descending, by insertion sort: np is at most kMaxPatternEdges.
        for (uint32_t i = 1; i < np; ++i) {
            const uint32_t v = producers[i];
            uint32_t j = i;
            while (j > 0 && producers[j - 1] < v) { producers[j] = producers[j - 1]; --j; }
            producers[j] = v;
        }
        for (uint32_t i = 0; i < np; ++i) {
            atomicAdd(qe.num_causal_edges, 1u);
            const uint64_t pk = hgcommon::id_key(producers[i], ev);
            if (!qe.causal_pairs.insert_if_absent(pk, 1u).inserted) continue;
            atomicAdd(qe.num_causal_pairs, 1u);

            // One base, two views: tag whether this pair survives the reduction. A pair
            // bypassed by a longer path is not in it; otherwise it is kept and becomes part of
            // the predecessor adjacency later decisions walk.
            if (qe_reachable(ds, qe, producers[i], ev)) continue;
            atomicAdd(qe.num_reduced_pairs, 1u);
            qe.reduced_pairs.insert_if_absent(pk, 1u);
            if (qe.preds.push(qe_bucket(hgcommon::id_key(ev), qe.preds.num_keys),
                              QePredRef{ev, producers[i]}) == INVALID_ID)
                ds.errors.record(ErrorKind::kQcNodes);
        }
    }

    // Branchial: siblings expanding the SAME instance whose consumed edges overlap. Publish
    // before scanning -- membership of the list is the proof the other application happened,
    // and an application that never claims never publishes.
    if (m.num_consumed && ds.record_branchial) {
        const uint32_t bucket = qe_bucket(hgcommon::id_key(inst.id), qe.inst_applied.num_keys);
        if (qe.inst_applied.push(bucket,
                QeAppliedMatch{inst.id, m.id, ev, m.num_consumed, m.arr_offset}) == INVALID_ID) {
            ds.errors.record(ErrorKind::kQcNodes);
        } else {
            __threadfence();
            const uint32_t* mine = qe.arr_words + m.arr_offset;
            qe.inst_applied.for_each(bucket, [&](const QeAppliedMatch& other) {
                // The bucket is shared, so the record's own instance is what selects this
                // instance's applications out of it. Slots are positions in the class frame,
                // so comparing them across two different instances would compare coordinates
                // in the same frame that belong to different occurrences of it.
                if (other.instance != inst.id) return;
                if (other.event == ev) return;   // self
                const uint32_t* theirs = qe.arr_words + other.consumed_offset;
                bool overlaps = false;
                for (uint32_t i = 0; i < m.num_consumed && !overlaps; ++i)
                    for (uint32_t j = 0; j < other.num_consumed; ++j)
                        if (mine[i] == theirs[j]) { overlaps = true; break; }
                if (!overlaps) return;
                const uint32_t lo = ev < other.event ? ev : other.event;
                const uint32_t hi = ev < other.event ? other.event : ev;
                if (qe.branchial_pairs.insert_if_absent(hgcommon::id_key(lo, hi), 1u).inserted)
                    atomicAdd(qe.num_branchial, 1u);
            });
        }
    }

    // The child instance: survivors carry their producer across, produced slots take THIS event.
    const uint32_t off = qe_alloc_words(ds, qe, m.to_slots);
    if (off == UINT32_MAX) return;
    for (uint32_t i = 0; i < m.to_slots; ++i) qe.arr_words[off + i] = kQeNoProducer;

    const uint32_t* aw = qe.arr_words + m.arr_offset;
    const uint32_t* surv_from = aw + m.num_consumed + m.num_produced;
    const uint32_t* surv_to   = surv_from + m.num_survivors;
    for (uint32_t i = 0; i < m.num_survivors; ++i) {
        const uint32_t f = surv_from[i], t = surv_to[i];
        if (f < inst.nslots && t < m.to_slots)
            qe.arr_words[off + t] = qe.arr_words[inst.prod_offset + f];
    }
    for (uint32_t i = 0; i < m.num_produced; ++i) {
        const uint32_t s = aw[m.num_consumed + i];
        if (s < m.to_slots) qe.arr_words[off + s] = ev;
    }

    const uint32_t rec = qe_add_instance(ds, qe, m.to_hash, depth + 1u, off, m.to_slots);
    if (rec == UINT32_MAX) return;
    qe_drive_instance(ds, qe, qe.instances.at(rec), m.to_hash, depth + 1u);
}

// Host-side owner of the capture's device structures, so a run's records are one body of
// state whether the host seeding or the device loop wrote them. Token-sized when the route is
// off, and cleared between runs rather than rebuilt, for the same reason QcState is: the pools
// total tens of MB of cudaMalloc that an interactive caller would otherwise pay every evolve.
class QeState {
public:
    QeState(bool on, uint32_t max_events)
        : matches_(on ? max_events : 1u),
          by_from_(on ? (1u << 16) : 1u, on ? max_events : 1u),
          instances_(on ? max_events : 1u),
          by_key_(on ? (1u << 16) : 1u, on ? max_events : 1u),
          rep_(on ? max_events : 8u),
          frame_step_(on ? max_events : 8u),
          applied_(on ? max_events * 4u : 8u),
          canon_seen_(on ? max_events * 2u : 8u),
          causal_pairs_(on ? max_events * 4u : 8u),
          branchial_pairs_(on ? max_events * 4u : 8u),
          reduced_pairs_(on ? max_events * 4u : 8u),
          inst_applied_(on ? (1u << 16) : 1u, on ? max_events * 2u : 1u),
          preds_(on ? (1u << 16) : 1u, on ? max_events * 4u : 1u),
          frame_(on ? max_events * 2u : 8u),
          arr_cap_(on ? max_events * 16u : 1u),
          on_(on) {
        HG_CUDA_CHECK(cudaMalloc(&arr_, sizeof(uint32_t) * arr_cap_), "QeState arr alloc");
        HG_CUDA_CHECK(cudaMalloc(&cursor_, sizeof(uint32_t)), "QeState cursor alloc");
        HG_CUDA_CHECK(cudaMalloc(&next_id_, sizeof(uint32_t)), "QeState next_id alloc");
        HG_CUDA_CHECK(cudaMalloc(&inst_next_id_, sizeof(uint32_t)), "QeState inst id alloc");
        HG_CUDA_CHECK(cudaMalloc(&next_raw_event_, sizeof(uint32_t)), "QeState raw ev alloc");
        HG_CUDA_CHECK(cudaMalloc(&align_moved_, sizeof(uint32_t)), "QeState align moved alloc");
        HG_CUDA_CHECK(cudaMalloc(&align_fail_, sizeof(uint32_t)), "QeState align fail alloc");
        HG_CUDA_CHECK(cudaMalloc(&num_canon_, sizeof(uint32_t)), "QeState canon alloc");
        HG_CUDA_CHECK(cudaMalloc(&num_causal_pairs_, sizeof(uint32_t)), "QeState c-pairs alloc");
        HG_CUDA_CHECK(cudaMalloc(&num_causal_edges_, sizeof(uint32_t)), "QeState c-edges alloc");
        HG_CUDA_CHECK(cudaMalloc(&num_branchial_, sizeof(uint32_t)), "QeState branchial alloc");
        HG_CUDA_CHECK(cudaMalloc(&num_reduced_pairs_, sizeof(uint32_t)), "QeState reduced alloc");
        event_sig_capacity_ = on ? max_events : 1u;
        HG_CUDA_CHECK(cudaMalloc(&event_sig_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event sig alloc");
        clear();
    }
    ~QeState() {
        if (arr_)     cudaFree(arr_);
        if (cursor_)  cudaFree(cursor_);
        if (next_id_) cudaFree(next_id_);
        if (inst_next_id_) cudaFree(inst_next_id_);
        if (next_raw_event_) cudaFree(next_raw_event_);
        if (align_moved_) cudaFree(align_moved_);
        if (align_fail_) cudaFree(align_fail_);
        if (num_canon_) cudaFree(num_canon_);
        if (num_causal_pairs_) cudaFree(num_causal_pairs_);
        if (num_causal_edges_) cudaFree(num_causal_edges_);
        if (num_branchial_) cudaFree(num_branchial_);
        if (num_reduced_pairs_) cudaFree(num_reduced_pairs_);
        if (event_sig_) cudaFree(event_sig_);
    }
    QeState(const QeState&)            = delete;
    QeState& operator=(const QeState&) = delete;

    bool enabled() const { return on_; }

    // Between runs: every map, list and record pool starts empty. The slot-array words need no
    // wipe -- records reference them by offset and the cursor restarts at zero.
    void clear() {
        frame_.clear();
        by_from_.clear();
        matches_.reset();
        by_key_.clear();
        instances_.reset();
        rep_.clear();
        frame_step_.clear();
        applied_.clear();
        canon_seen_.clear();
        causal_pairs_.clear();
        branchial_pairs_.clear();
        reduced_pairs_.clear();
        inst_applied_.clear();
        preds_.clear();
        HG_CUDA_CHECK(cudaMemset(inst_next_id_, 0, sizeof(uint32_t)), "QeState inst id clear");
        HG_CUDA_CHECK(cudaMemset(next_raw_event_, 0, sizeof(uint32_t)), "QeState raw ev clear");
        HG_CUDA_CHECK(cudaMemset(align_moved_, 0, sizeof(uint32_t)), "QeState align moved clear");
        HG_CUDA_CHECK(cudaMemset(align_fail_, 0, sizeof(uint32_t)), "QeState align fail clear");
        HG_CUDA_CHECK(cudaMemset(num_canon_, 0, sizeof(uint32_t)), "QeState canon clear");
        HG_CUDA_CHECK(cudaMemset(num_causal_pairs_, 0, sizeof(uint32_t)), "QeState c-pairs clear");
        HG_CUDA_CHECK(cudaMemset(num_causal_edges_, 0, sizeof(uint32_t)), "QeState c-edges clear");
        HG_CUDA_CHECK(cudaMemset(num_branchial_, 0, sizeof(uint32_t)), "QeState branchial clear");
        HG_CUDA_CHECK(cudaMemset(num_reduced_pairs_, 0, sizeof(uint32_t)), "QeState reduced clear");
        HG_CUDA_CHECK(cudaMemset(event_sig_, 0, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event sig clear");
        HG_CUDA_CHECK(cudaMemset(cursor_, 0, sizeof(uint32_t)), "QeState cursor clear");
        HG_CUDA_CHECK(cudaMemset(next_id_, 0, sizeof(uint32_t)), "QeState next_id clear");
    }

    // Records captured this run: one per match of each class's frame state. The number the
    // host's for_each_expansion_match yields when summed over classes, and the gate for the
    // capture being wired correctly.
    uint32_t num_matches_host() { return matches_.size_host(); }

    // Raw events the replay minted: one per (instance, match) application. The host's
    // qc_next_raw_event_, and the number a quotient run reports as its raw event count.
    uint32_t num_raw_events_host() { return read_counter(next_raw_event_, "QeState raw event read"); }

    // The reconstructed causal relation: distinct (producer, consumer) pairs, and the
    // consumed-edge occurrences behind them. The host's num_reconstructed_causal_pairs(false)
    // and num_reconstructed_causal_edges.
    uint32_t num_causal_pairs_host() { return read_counter(num_causal_pairs_, "QeState c-pairs read"); }
    uint32_t num_causal_edges_host() { return read_counter(num_causal_edges_, "QeState c-edges read"); }

    // Pairs tagged in-reduction: the TR view of the same relation. The host's
    // num_reconstructed_causal_pairs(true).
    uint32_t num_reduced_pairs_host() { return read_counter(num_reduced_pairs_, "QeState reduced read"); }

    // Distinct branchial pairs: sibling applications of one instance whose consumed edges
    // overlap. The host's num_reconstructed_branchial.
    uint32_t num_branchial_host() { return read_counter(num_branchial_, "QeState branchial read"); }

    // The reconstructed relations as pairs of CONTENT TRIPLES. A count says two engines
    // disagree; a pair set says which pair is missing, which a count cannot.
    void reconstructed_pairs_host(std::vector<std::pair<uint64_t, uint64_t>>& causal,
                                  std::vector<std::pair<uint64_t, uint64_t>>& causal_reduced,
                                  std::vector<std::pair<uint64_t, uint64_t>>& branchial) {
        causal.clear();
        causal_reduced.clear();
        branchial.clear();
        const uint32_t n = num_raw_events_host();
        if (n == 0) return;
        std::vector<uint64_t> sigs(event_sig_capacity_);
        HG_CUDA_CHECK(cudaMemcpy(sigs.data(), event_sig_,
                                 sizeof(uint64_t) * event_sig_capacity_, cudaMemcpyDeviceToHost),
                      "QeState event sig read");
        auto sig_of = [&](uint32_t e) -> uint64_t {
            return e < sigs.size() ? sigs[e] : 0ull;
        };
        auto drain = [&](DedupMap& m, std::vector<std::pair<uint64_t, uint64_t>>& out) {
            std::vector<uint64_t> keys;
            m.copy_keys_to_host(keys);
            out.reserve(keys.size());
            for (uint64_t k : keys) {
                const hgcommon::IdPair p = hgcommon::id_pair_from_key(k);
                out.emplace_back(sig_of(p.a), sig_of(p.b));
            }
        };
        drain(causal_pairs_, causal);
        drain(reduced_pairs_, causal_reduced);
        drain(branchial_pairs_, branchial);
    }

    // Distinct event identities the replay produced under the run's mode. The host's
    // qc_num_canon_events_, and what a caller is told the event count is when a mode is selected.
    uint32_t num_canon_events_host() { return read_counter(num_canon_, "QeState canon read"); }

    // Slots the frame moved off the state's own labelling, and slots no frame image existed for.
    uint32_t num_aligned_host() { return read_counter(align_moved_, "QeState align moved read"); }
    uint32_t num_align_failures_host() { return read_counter(align_fail_, "QeState align fail read"); }

    // Instances recorded this run. One per raw occurrence of a class at a depth: one per root
    // before any replay, and one more per application once the replay lands.
    uint32_t num_instances_host() { return instances_.size_host(); }

    QeView view(uint32_t max_steps, EventSignatureKeys keys, uint32_t max_recursion_depth) {
        QeView q{};
        q.matches      = matches_.view();
        q.by_from      = by_from_.view();
        q.instances      = instances_.view();
        q.by_key         = by_key_.view();
        q.inst_next_id   = inst_next_id_;
        q.rep            = rep_.view();
        q.frame_step     = frame_step_.view();
        q.applied        = applied_.view();
        q.align_moved    = align_moved_;
        q.canon_seen     = canon_seen_.view();
        q.num_canon      = num_canon_;
        q.event_sig        = event_sig_;
        q.event_sig_capacity = event_sig_capacity_;
        q.inst_applied     = inst_applied_.view();
        q.branchial_pairs  = branchial_pairs_.view();
        q.num_branchial    = num_branchial_;
        q.preds            = preds_.view();
        q.reduced_pairs    = reduced_pairs_.view();
        q.num_reduced_pairs = num_reduced_pairs_;
        q.causal_pairs   = causal_pairs_.view();
        q.num_causal_pairs = num_causal_pairs_;
        q.num_causal_edges = num_causal_edges_;
        q.keys           = keys;
        q.align_fail     = align_fail_;
        q.next_raw_event = next_raw_event_;
        q.frame        = frame_.view();
        q.arr_words    = arr_;
        q.arr_cursor   = cursor_;
        q.arr_capacity = arr_cap_;
        q.next_id      = next_id_;
        q.max_steps    = max_steps;
        q.max_recursion_depth = max_recursion_depth;
        q.enabled      = on_ ? 1u : 0u;
        return q;
    }

private:

    static uint32_t read_counter(const uint32_t* p, const char* what) {
        uint32_t v = 0;
        HG_CUDA_CHECK(cudaMemcpy(&v, p, sizeof(uint32_t), cudaMemcpyDeviceToHost), what);
        return v;
    }

    Pool<DeviceSlotMatch>     matches_;
    LockFreeList<QeMatchRef>  by_from_;
    Pool<DeviceQcInstance>    instances_;
    LockFreeList<QeInstRef>   by_key_;
    DedupMap                  rep_;
    DedupMap                  frame_step_;
    DedupMap                  applied_;
    DedupMap                  canon_seen_;
    DedupMap                  causal_pairs_;
    DedupMap                  branchial_pairs_;
    DedupMap                  reduced_pairs_;
    LockFreeList<QeAppliedMatch> inst_applied_;
    LockFreeList<QePredRef>   preds_;
    uint32_t*                 inst_next_id_ = nullptr;
    uint32_t*                 next_raw_event_ = nullptr;
    uint32_t*                 align_moved_    = nullptr;
    uint32_t*                 align_fail_     = nullptr;
    uint32_t*                 num_canon_        = nullptr;
    uint32_t*                 num_causal_pairs_ = nullptr;
    uint32_t*                 num_causal_edges_ = nullptr;
    uint32_t*                 num_branchial_    = nullptr;
    uint32_t*                 num_reduced_pairs_ = nullptr;
    uint64_t*                 event_sig_        = nullptr;
    uint32_t                  event_sig_capacity_ = 0;
    DedupMap                  frame_;
    uint32_t*                 arr_ = nullptr;
    uint32_t*                 cursor_ = nullptr;
    uint32_t*                 next_id_ = nullptr;
    uint32_t                  arr_cap_ = 0;
    bool                      on_ = false;
};

}  // namespace hg_gpu
