#pragma once
#include <unordered_map>
#include "hgcommon/transitive_reduction.hpp"
#include "hgcommon/namespace.hpp"
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
#include "hgcommon/core.hpp"        // isort_u64
#include "hgcommon/slot_core.hpp"  // slot_rank -- the frame-slot rule, shared with the host
#include "hgcommon/quotient_replay_core.hpp"  // qr_apply -- the replay, and the identity it mints

#include <cuda/atomic>

namespace HG_NAMESPACE {
namespace gpu {

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

    // The four slot arrays live contiguously in the expansion word arena at arr_offset:
    // consumed | produced | surv_from | surv_to. `words` is that arena's base, which the
    // record cannot hold because it is a device pointer the host rebuilds per run -- so the
    // view below binds the two together for hgcommon/quotient_replay_core.hpp, which reads
    // both engines' layouts through one set of calls.
    __device__ const uint32_t* at(const uint32_t* words) const { return words + arr_offset; }
};

// A DeviceSlotMatch bound to the arena its slots live in. What the shared replay sees.
struct QeMatchView {
    const uint32_t* w;            // consumed | produced | surv_from | surv_to
    uint64_t to_hash;
    uint32_t id, rule, from_slots, to_slots;
    uint32_t num_consumed, num_produced, num_survivors;

    __device__ QeMatchView(const DeviceSlotMatch& m, const uint32_t* words)
        : w(m.at(words)), to_hash(m.to_hash), id(m.id), rule(m.rule),
          from_slots(m.from_slots), to_slots(m.to_slots), num_consumed(m.num_consumed),
          num_produced(m.num_produced), num_survivors(m.num_survivors) {}

    __device__ uint32_t consumed(uint32_t i)  const { return w[i]; }
    __device__ uint32_t produced(uint32_t i)  const { return w[num_consumed + i]; }
    __device__ uint32_t surv_from(uint32_t i) const {
        return w[num_consumed + num_produced + i];
    }
    __device__ uint32_t surv_to(uint32_t i) const {
        return w[num_consumed + num_produced + num_survivors + i];
    }
    __device__ const uint32_t* consumed_ptr() const { return w; }
    __device__ const uint32_t* produced_ptr() const { return w + num_consumed; }
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
    uint64_t h = hgcommon::FNV_OFFSET;
    h ^= state_hash; h *= hgcommon::FNV_PRIME;
    h ^= (static_cast<uint64_t>(depth) << 32); h *= hgcommon::FNV_PRIME;
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

// A QeAppliedMatch bound to the arena its consumed slots live in.
struct QeAppliedView {
    const uint32_t* w;
    uint32_t event, num_consumed;
    __device__ QeAppliedView(const QeAppliedMatch& a, const uint32_t* words)
        : w(words + a.consumed_offset), event(a.event), num_consumed(a.num_consumed) {}
    __device__ uint32_t consumed(uint32_t j) const { return w[j]; }
};

// The slot-has-no-producer sentinel, from hgcommon: the replay core writes it into a
// child's producer vector and this file reads it back, so one value or neither works.
inline constexpr uint32_t kQeNoProducer = hgcommon::QR_NO_PRODUCER;

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


    // The reconstructed branchial relation lives in `inst_applied` and nowhere else. It is
    // bucketed by instance id: an application publishes itself there and then scans the nodes
    // linked BEFORE its own, so of any two exactly one sees the other and the pair is emitted
    // once. The pairs themselves are never stored -- reconstructed_pairs_host regroups the
    // applications when a caller asks for the relation.
    //
    // Per raw event, the schedule-stable content triple hash(input class, output class, rule).
    // Indexed by raw event id, which is what the pair keys hold; the triple is what a
    // cross-engine comparison can be made on. The host's qc_event_sig_.
    uint64_t* event_sig;
    // Per raw event, the identity under the RUN'S MODE -- what observable_num_events counts
    // distinct values of. Kept BESIDE the content triple, not instead of it: the triple is the
    // schedule-stable key the relations compare on, and this is what a caller must group events
    // by to build a graph whose vertex set is the set the count describes. Recording only the
    // COUNT of distinct values, which is all this did, cannot say which event carries which, so
    // a graph could not be built over them at all.
    uint64_t* event_runsig;
    uint32_t  event_sig_capacity;

    typename LockFreeList<QeAppliedMatch>::DeviceView inst_applied;
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
    // Whether the captured expansion is REPLAYED against instances, as against merely captured.
    //
    // The two halves of this subsystem have different costs and different consumers. Capture --
    // the per-class frame and its matches in frame slots -- is what Automatic event identity is
    // signed from, and costs what the canonical answer costs. Replay materialises one instance
    // per raw state of the full unfolding to recover the raw event set, and costs what the RAW
    // answer costs, which is exponential in depth while the canonical answer is not.
    //
    // So a run that does not record raw events, causal or branchial captures but does not
    // replay: identity is unchanged and the exponential is not paid. This mirrors the host,
    // where qc_capture_expansion runs unconditionally and only the instance seeding and the
    // match-side scan are gated (hypergraph.cpp:987, :1206).
    uint32_t  replay    = 0;
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
    if (!ds.state_edge_orbit) return UINT32_MAX;
    const uint32_t i = state_edge_index(ds, sid, edge);
    if (i == UINT32_MAX || ds.state_edge_orbit[i] == UINT32_MAX) return UINT32_MAX;
    const StateEdgeSlice sl = ds.state_edge_slices[sid];
    // The rule itself is hgcommon's, not this file's: the host records the same coordinates
    // (hypergraph.cpp, via slots_from_orbits) and two readings that drift by one tie-break
    // would replay wrong events invisibly.
    return hgcommon::slot_rank(ds.state_edge_orbit + sl.offset, sl.count, i - sl.offset);
}

// The canonical rank of `edge` within `sid` -- its position in the state's canonical order,
// from the same individualization-refinement pass that produced the state's exact hash.
// UINT32_MAX when the edge is absent or no rank was computed.
__device__ __forceinline__ uint32_t qe_rank_of(DeviceState ds, StateId sid, EdgeId edge) {
    if (!ds.state_edge_rank) return UINT32_MAX;
    const uint32_t i = state_edge_index(ds, sid, edge);
    return i == UINT32_MAX ? UINT32_MAX : ds.state_edge_rank[i];
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
        hgcommon::isort_u64(surv, ns);
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

    // The frame above is registered whatever the caller records: event identity reads it. The
    // instance below is the root of the replay, and without it no descendant instance exists,
    // so this one guard removes the whole cascade.
    if (!qe.replay) return;

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
    uint64_t k = hgcommon::FNV_OFFSET;
    k ^= instance; k *= hgcommon::FNV_PRIME;
    k ^= match;    k *= hgcommon::FNV_PRIME;
    return (k == 0 || k == ~0ULL) ? 1 : k;
}

// The storage face hgcommon/quotient_replay_core.hpp drives. WHERE a producer vector, an
// applied list or a claim set lives is here; what an application DOES -- what it claims, what
// it identifies the event by, which causal and branchial relations follow -- is in the core,
// which is the body the host runs too.
// Defined below, over the shared replay core; the two drivers here are its callers, so the
// mutual recursion needs the declaration first.
//
// __forceinline__ IS LOAD-BEARING, and the declaration has to carry it too so the two agree.
// This sits inside the recursion cycle whose per-level cost EngineState::kDeviceStackBytesPerDepth
// records, and the body is only "build the Ctx and forward", so a separate frame for it buys a
// call's worth of ABI save area per level of reconstruction depth and nothing else. Measured with
// tools/dev/ptx_frame_sizes.py: as its own frame it holds a 64-byte depot.
__device__ __forceinline__ void qe_apply(DeviceState ds, QeView qe, const DeviceQcInstance& inst,
                                         const DeviceSlotMatch& m, uint64_t state_hash,
                                         uint32_t depth);

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

struct DeviceQrCtx {
    using Instance = DeviceQcInstance;
    using Match    = QeMatchView;
    using Applied  = QeAppliedView;
    // REFERENCES, not copies. This Ctx is constructed inside qe_apply, which is in the replay's
    // recursion cycle (qe_apply -> descend -> qe_add_instance -> qe_drive_instance -> qe_apply),
    // so anything it holds by value is paid once PER LEVEL. DeviceState and QeView are large
    // aggregates, and EngineState::qe_max_recursion_depth is calibrated against a measured
    // 5461 bytes per level -- inflating the frame makes the guard fire after the frame that
    // faults instead of before it, which is an illegal memory access rather than a bounded
    // partial result. The caller's copies outlive this object.
    DeviceState& ds;
    QeView& qe;

    __device__ bool claim(uint64_t apply_key) {
        return qe.applied.insert_if_absent(apply_key, 1u).inserted;
    }
    __device__ uint32_t mint_event() {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> nre(*qe.next_raw_event);
        return nre.fetch_add(1u, cuda::memory_order_relaxed);
    }
    // The event's content triple, from hgcommon rather than open-coded here. The open-coding
    // this replaces seeded FNV with the 64-bit basis missing its last digit, so every
    // reconstructed identity the device reported was a relabelling of the host's; routing the
    // call is what makes that unrepeatable rather than merely fixed.
    __device__ void record_content(uint32_t ev, uint64_t from_class, uint64_t to_class,
                                   uint32_t rule) {
        if (ev < qe.event_sig_capacity)
            qe.event_sig[ev] = hgcommon::qr_content_hash(from_class, to_class, rule);
    }
    __device__ hgcommon::EventSignatureKeys keys() const { return qe.keys; }
    // The canonical OUTPUT class's step, which is one value per class rather than the depth this
    // instance happens to sit at; the caller's depth stands in when the class holds no frame.
    __device__ uint32_t frame_step(uint64_t class_hash, uint32_t fallback) const {
        const auto fs = qe.frame_step.lookup_waiting(class_hash);
        return (fs.found && fs.value != 0) ? fs.value - 1u : fallback;
    }
    __device__ void record_runsig(uint32_t ev, uint64_t csig) {
        if (ev < qe.event_sig_capacity) qe.event_runsig[ev] = csig;
        if (qe.canon_seen.insert_if_absent(csig, 1u).inserted) atomicAdd(qe.num_canon, 1u);
    }
    __device__ bool want_causal() const    { return ds.record_causal != 0; }
    __device__ bool want_branchial() const { return ds.record_branchial != 0; }
    __device__ uint32_t producer_at(const DeviceQcInstance& inst, uint32_t slot) const {
        return qe.arr_words[inst.prod_offset + slot];
    }
    __device__ void record_causal(uint32_t producer, uint32_t consumer) {
        atomicAdd(qe.num_causal_edges, 1u);
        const uint64_t pk = hgcommon::id_key(producer, consumer);
        if (!qe.causal_pairs.insert_if_absent(pk, 1u).inserted) return;
        atomicAdd(qe.num_causal_pairs, 1u);
        // THE BASE SET, and nothing else. Which pairs survive transitive reduction is a
        // property of the finished relation, so it is decided by hgcommon::tr_reduce when the
        // relation is handed back -- see reconstructed_pairs_host. Deciding it here would ask
        // whether a bypassing path exists using only the pairs recorded so far, and on a device
        // that order is whatever the warps produced.
    }
    using AppliedRef = uint32_t;
    __device__ static bool applied_ref_valid(AppliedRef r) { return r != INVALID_ID; }
    __device__ AppliedRef publish_applied(const DeviceQcInstance& inst, const QeMatchView& m,
                                          uint32_t ev) {
        const uint32_t bucket = qe_bucket(hgcommon::id_key(inst.id), qe.inst_applied.num_keys);
        const uint32_t at = qe.inst_applied.push(bucket,
                QeAppliedMatch{inst.id, m.id, ev, m.num_consumed,
                               static_cast<uint32_t>(m.w - qe.arr_words)});
        if (at == INVALID_ID) {
            ds.errors.record(ErrorKind::kQcNodes);
            return INVALID_ID;
        }
        __threadfence();
        return at;
    }
    template <class F>
    __device__ void for_each_applied_before(const DeviceQcInstance& inst, AppliedRef mine, F&& f) {
        // The bucket is shared between instances, so walking below `mine` gives every
        // application published earlier and the instance filter selects this one's.
        qe.inst_applied.for_each_before(mine, [&](const QeAppliedMatch& other) {
            // The bucket is shared, so the record's own instance is what selects this
            // instance's applications out of it. Slots are positions in the class frame, so
            // comparing them across two instances would compare coordinates in the same frame
            // belonging to different occurrences of it.
            if (other.instance != inst.id) return;
            f(QeAppliedView(other, qe.arr_words));
        });
    }
    __device__ void record_branchial_pair(uint32_t lo, uint32_t hi) {
        // Counted on every emission, because the replay emits each pair exactly once: the pair
        // belongs to the later of its two applications and only that one scans the other. The
        // map is storage for the readback, not a dedup the count depends on.
        (void)lo; (void)hi;
        atomicAdd(qe.num_branchial, 1u);
    }
    // __forceinline__ IS LOAD-BEARING. qr_apply calls this exactly once, at its tail, and the
    // call closes the recursion cycle whose per-level cost EngineState::kDeviceStackBytesPerDepth
    // records. Left as its own frame it holds a 1104-byte depot plus a call's ABI save area, on
    // every level of reconstruction depth; folded into qr_apply the depot merges and the frame
    // is not paid. Measured with tools/dev/ptx_frame_sizes.py.
    __device__ __forceinline__ void descend(const QeMatchView& m, uint32_t depth, uint32_t ev,
                                            const DeviceQcInstance& parent) {
        const uint32_t off = qe_alloc_words(ds, qe, m.to_slots);
        if (off == UINT32_MAX) return;
        for (uint32_t i = 0; i < m.to_slots; ++i)
            qe.arr_words[off + i] = hgcommon::QR_NO_PRODUCER;
        for (uint32_t i = 0; i < m.num_survivors; ++i) {
            const uint32_t f = m.surv_from(i), t = m.surv_to(i);
            if (f < parent.nslots && t < m.to_slots)
                qe.arr_words[off + t] = qe.arr_words[parent.prod_offset + f];
        }
        for (uint32_t i = 0; i < m.num_produced; ++i) {
            const uint32_t s = m.produced(i);
            if (s < m.to_slots) qe.arr_words[off + s] = ev;
        }
        const uint32_t rec = qe_add_instance(ds, qe, m.to_hash, depth + 1u, off, m.to_slots);
        if (rec == UINT32_MAX) return;
        qe_drive_instance(ds, qe, qe.instances.at(rec), m.to_hash, depth + 1u);
    }
};

__device__ __forceinline__ void qe_apply(DeviceState ds, QeView qe, const DeviceQcInstance& inst,
                                         const DeviceSlotMatch& m, uint64_t state_hash,
                                         uint32_t depth) {
    if (!qe.enabled || depth >= qe.max_steps) return;
    DeviceQrCtx c{ds, qe};
    hgcommon::qr_apply(c, inst, QeMatchView(m, qe.arr_words), state_hash, depth);
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
          inst_applied_(on ? (1u << 16) : 1u, on ? max_events * 2u : 1u),
          frame_(on ? max_events * 2u : 8u),
          arr_cap_(on ? max_events * 16u : 1u),
          on_(on) {
        HG_CUDA_CHECK(cudaMalloc(&arr_, sizeof(uint32_t) * arr_cap_), "QeState arr alloc");
        // ELEVEN SCALARS IN ONE BLOCK, so the host reads them in ONE transfer.
        //
        // Each of these was its own cudaMalloc and each accessor its own synchronous cudaMemcpy
        // of four bytes. A synchronous copy of a scalar costs about 24 microseconds on this host
        // whatever its size, and the result path reads ten of them per evolve call. Laid out
        // contiguously, counters_host() fetches the lot in one copy; the individual accessors
        // remain for ad-hoc use and now index the block.
        HG_CUDA_CHECK(cudaMalloc(&counters_, sizeof(uint32_t) * kNumCounters),
              "QeState counters alloc");
        cursor_            = counters_ + 0;
        next_id_           = counters_ + 1;
        inst_next_id_      = counters_ + 2;
        next_raw_event_    = counters_ + 3;
        align_moved_       = counters_ + 4;
        align_fail_        = counters_ + 5;
        num_canon_         = counters_ + 6;
        num_causal_pairs_  = counters_ + 7;
        num_causal_edges_  = counters_ + 8;
        num_branchial_     = counters_ + 9;
        event_sig_capacity_ = on ? max_events : 1u;
        HG_CUDA_CHECK(cudaMalloc(&event_sig_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event sig alloc");
        HG_CUDA_CHECK(cudaMalloc(&event_runsig_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event runsig alloc");
        clear();
    }
    ~QeState() {
        if (arr_)     cudaFree(arr_);
        if (counters_) cudaFree(counters_);
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
        inst_applied_.clear();
        HG_CUDA_CHECK(cudaMemset(inst_next_id_, 0, sizeof(uint32_t)), "QeState inst id clear");
        HG_CUDA_CHECK(cudaMemset(next_raw_event_, 0, sizeof(uint32_t)), "QeState raw ev clear");
        HG_CUDA_CHECK(cudaMemset(align_moved_, 0, sizeof(uint32_t)), "QeState align moved clear");
        HG_CUDA_CHECK(cudaMemset(align_fail_, 0, sizeof(uint32_t)), "QeState align fail clear");
        HG_CUDA_CHECK(cudaMemset(num_canon_, 0, sizeof(uint32_t)), "QeState canon clear");
        HG_CUDA_CHECK(cudaMemset(num_causal_pairs_, 0, sizeof(uint32_t)), "QeState c-pairs clear");
        HG_CUDA_CHECK(cudaMemset(num_causal_edges_, 0, sizeof(uint32_t)), "QeState c-edges clear");
        HG_CUDA_CHECK(cudaMemset(num_branchial_, 0, sizeof(uint32_t)), "QeState branchial clear");
        HG_CUDA_CHECK(cudaMemset(event_sig_, 0, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event sig clear");
        HG_CUDA_CHECK(cudaMemset(event_runsig_, 0, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event runsig clear");
        HG_CUDA_CHECK(cudaMemset(cursor_, 0, sizeof(uint32_t)), "QeState cursor clear");
        HG_CUDA_CHECK(cudaMemset(next_id_, 0, sizeof(uint32_t)), "QeState next_id clear");
    }

    // Records captured this run: one per match of each class's frame state. The number the
    // host's for_each_expansion_match yields when summed over classes, and the gate for the
    // capture being wired correctly.
    // Every scalar the result path needs, in ONE transfer.
    //
    // The individual accessors below each cost a synchronous four-byte copy, about 24 us on this
    // host regardless of size, and the result path calls ten of them per evolve call. Since the
    // counters share one allocation they can be fetched together; the fields are named so a
    // caller reads them the same way it read the accessors.
    struct Counters {
        uint32_t cursor, next_id, instances, raw_events, aligned, align_failures,
                 canon_events, causal_pairs, causal_edges, branchial;
    };
    Counters counters_host() const {
        uint32_t v[kNumCounters] = {};
        HG_CUDA_CHECK(cudaMemcpy(v, counters_, sizeof(v), cudaMemcpyDeviceToHost),
              "QeState counters read");
        return Counters{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]};

    }

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

    // Distinct branchial pairs: sibling applications of one instance whose consumed edges
    // overlap. The host's num_reconstructed_branchial.
    uint32_t num_branchial_host() { return read_counter(num_branchial_, "QeState branchial read"); }

    // The reconstructed relations as pairs of CONTENT TRIPLES. A count says two engines
    // disagree; a pair set says which pair is missing, which a count cannot.
    void reconstructed_pairs_host(std::vector<std::pair<uint64_t, uint64_t>>& causal,
                                  std::vector<std::pair<uint64_t, uint64_t>>& causal_reduced,
                                  std::vector<std::pair<uint64_t, uint64_t>>& branchial,
                                  bool want_branchial,
                                  std::vector<uint64_t>* event_signature = nullptr) {
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
        // Handed back whole, so a caller can identify an EVENT the same way the relations
        // identify their endpoints rather than by a second convention.
        if (event_signature) {
            // The RUN identity, not the content triple: observable_num_events counts distinct
            // values of THIS, so a graph grouped by it has the vertex set the count describes.
            std::vector<uint64_t> rsigs(event_sig_capacity_);
            if (event_sig_capacity_)
                HG_CUDA_CHECK(cudaMemcpy(rsigs.data(), event_runsig_,
                                         sizeof(uint64_t) * event_sig_capacity_,
                                         cudaMemcpyDeviceToHost), "QeState event runsig read");
            event_signature->assign(rsigs.begin(),
                                    rsigs.begin() + std::min<size_t>(rsigs.size(), n));
        }
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

        // THE REDUCED VIEW, from the same stored relation and the same rule the host engine
        // uses. It is computed here rather than on the device for two reasons: which pairs
        // survive is a property of the FINISHED relation, and a device that tagged each pair as
        // it landed would answer against whatever the warps had produced so far; and the
        // reduction runs over event IDS, whose order carries the reachability prune, while
        // these vectors carry signatures and two events may share one.
        std::vector<uint64_t> ckeys;
        causal_pairs_.copy_keys_to_host(ckeys);
        hgcommon::tr_reduce(
            [&](auto&& add) {
                for (uint64_t k : ckeys) {
                    const hgcommon::IdPair p = hgcommon::id_pair_from_key(k);
                    add(static_cast<uint32_t>(p.a), static_cast<uint32_t>(p.b));
                }
            },
            [&](uint32_t a, uint32_t b) { causal_reduced.emplace_back(sig_of(a), sig_of(b)); },
            // A producer wrote the slot its consumer reads, so its application minted the
            // lower id: ids increase along every causal edge of this relation.
            /*ids_topological=*/true);

        if (!want_branchial) return;


        // BRANCHIAL, DERIVED FROM THE APPLICATIONS rather than stored as pairs. A branchial pair
        // is two applications of ONE instance sharing a consumed slot, so the applications are
        // the relation in the form the replay generates it, and the pair list is an expansion of
        // them -- 970,584 against 133,218,996 on the host's disc-l3a2g2r2 depth 3. Storing that
        // expansion on the device is what its 2^22 map ceiling was, and truncating it returned a
        // partial relation with a warning rather than an answer.
        //
        // Order does not matter here, only the SET, so the host groups by instance and takes
        // each unordered pair once. The device's own counter is incremented per emission under
        // the strictly-earlier scan rule, so it and this enumeration are two routes to one
        // number and disagreeing is a defect either can catch.
        std::vector<LockFreeList<QeAppliedMatch>::Node> nodes;
        inst_applied_.copy_nodes_to_host(nodes);
        std::vector<uint32_t> slots(arr_cap_);
        if (arr_cap_)
            HG_CUDA_CHECK(cudaMemcpy(slots.data(), arr_, sizeof(uint32_t) * arr_cap_,
                                     cudaMemcpyDeviceToHost), "QeState arr read");

        std::unordered_map<uint32_t, std::vector<const QeAppliedMatch*>> by_instance;
        for (const auto& nd : nodes) by_instance[nd.value.instance].push_back(&nd.value);
        for (const auto& kv : by_instance) {
            const auto& v = kv.second;
            for (size_t i = 0; i < v.size(); ++i) {
                for (size_t j = i + 1; j < v.size(); ++j) {
                    const QeAppliedMatch& a = *v[i];
                    const QeAppliedMatch& b = *v[j];
                    if (a.event == b.event) continue;
                    bool overlaps = false;
                    for (uint32_t x = 0; x < a.num_consumed && !overlaps; ++x) {
                        const uint32_t ax = a.consumed_offset + x;
                        if (ax >= slots.size()) break;
                        for (uint32_t y = 0; y < b.num_consumed; ++y) {
                            const uint32_t by = b.consumed_offset + y;
                            if (by >= slots.size()) break;
                            if (slots[ax] == slots[by]) { overlaps = true; break; }
                        }
                    }
                    if (!overlaps) continue;
                    const uint32_t lo = a.event < b.event ? a.event : b.event;
                    const uint32_t hi = a.event < b.event ? b.event : a.event;
                    branchial.emplace_back(sig_of(lo), sig_of(hi));
                }
            }
        }

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

    QeView view(uint32_t max_steps, EventSignatureKeys keys, uint32_t max_recursion_depth,
                bool replay) {
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
        q.event_runsig     = event_runsig_;
        q.event_sig_capacity = event_sig_capacity_;
        q.inst_applied     = inst_applied_.view();
        q.num_branchial    = num_branchial_;
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
        q.replay       = (on_ && replay) ? 1u : 0u;
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
    LockFreeList<QeAppliedMatch> inst_applied_;
    uint32_t*                 inst_next_id_ = nullptr;
    uint32_t*                 next_raw_event_ = nullptr;
    uint32_t*                 align_moved_    = nullptr;
    uint32_t*                 align_fail_     = nullptr;
    uint32_t*                 num_canon_        = nullptr;
    uint32_t*                 num_causal_pairs_ = nullptr;
    uint32_t*                 num_causal_edges_ = nullptr;
    uint32_t*                 num_branchial_    = nullptr;
    uint64_t*                 event_sig_        = nullptr;
    uint64_t*                 event_runsig_     = nullptr;
    uint32_t                  event_sig_capacity_ = 0;
    DedupMap                  frame_;
    uint32_t*                 arr_ = nullptr;
    // The eleven scalars above and below live in ONE allocation; these pointers index into it,
    // so counters_host() reads them all in a single transfer.
    static constexpr uint32_t kNumCounters = 10;
    uint32_t*                 counters_ = nullptr;
    uint32_t*                 cursor_ = nullptr;
    uint32_t*                 next_id_ = nullptr;
    uint32_t                  arr_cap_ = 0;
    bool                      on_ = false;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE