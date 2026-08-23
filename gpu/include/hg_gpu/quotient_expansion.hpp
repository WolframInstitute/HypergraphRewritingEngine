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
#include "hgcommon/quotient_causal_core.hpp"  // qc_key -- the (class, depth, orbit) key rule

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
    // THE DP'S KEY RULE, not a second one that happens to agree. This open-coded FNV over
    // (state_hash, depth << 32) computed exactly hgcommon::qc_key(state_hash, depth, 0) -- the
    // orbit term is zero here because an instance is keyed by its class and depth alone -- and
    // two spellings of one rule agree until one of them is edited.
    return hgcommon::qc_key(state_hash, depth, 0u);
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

// ONE PENDING DESCENT. The replay used to descend by CALLING itself, and the cycle
// qe_apply -> qr_apply -> descend -> qe_add_instance -> qe_drive_instance cost 8,704 bytes of
// per-thread stack per level of reconstruction depth. What it actually carried across a level is
// these three scalars. Sixteen bytes against 8,704 is why the depth a run can reconstruct was a
// property of the launch rather than of the workload.
struct QeWorkItem {
    uint64_t hash;    // the class the instance stands at
    uint32_t rec;     // its record in the instance pool
    uint32_t depth;
};

// A driver's private descent stack. LIFO, so the order instances are driven in is the order the
// recursion drove them -- which is what lets the existing corpus gate this change directly.
//
// PRIVATE TO ONE DRIVER, and that is a property of where the replay runs rather than an
// assumption: in the persistent kernel the whole rewrite path is inside `threadIdx.x == 0`, so
// there is one driver per BLOCK, and in the root seeder there is one per root.
struct QeWork {
    QeWorkItem* items = nullptr;
    uint32_t    cap   = 0;
    uint32_t    n     = 0;

    __device__ bool push(uint64_t hash, uint32_t rec, uint32_t depth) {
        if (n >= cap) return false;
        items[n].hash = hash; items[n].rec = rec; items[n].depth = depth;
        ++n;
        return true;
    }
    __device__ bool pop(QeWorkItem& out) {
        if (n == 0) return false;
        out = items[--n];
        return true;
    }
};

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

    // Backing store for the descent stacks, one contiguous slice of `work_cap` items per driver.
    // Sized from the run's step budget the way the IR arena is sized from its state budget, and
    // exhausted the same way: a capacity overflow that reports and returns partial work.
    QeWorkItem* work_items  = nullptr;
    uint32_t    work_cap    = 0;   // items per driver
    uint32_t    work_slices = 0;   // drivers this run can serve

    uint32_t  max_steps = 0;
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
                                         uint64_t state_hash, uint32_t depth, QeWork& work);
__device__ inline void qe_run(DeviceState ds, QeView qe, QeWork& work);
__device__ inline QeWork qe_work_for(DeviceState ds, QeView qe, uint32_t slice);
__device__ inline void qe_drive_match(DeviceState ds, QeView qe, const DeviceSlotMatch& m,
                                      uint64_t from_hash, QeWork& work);

// Bucket a hash into a list's key space.
//
// SAME MIXING AS qc_bucket, DIFFERENT REDUCTION, and the two are not interchangeable: this takes
// the full 64-bit value modulo the key count, while the DP's masks the LOW 32 bits with
// `num_keys - 1` and so requires a power-of-two count. Each list is read with the function it was
// written with, which is what makes both correct; a claim that they distribute alike is not.
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
                                            uint32_t rule, uint32_t depth, uint32_t work_slice) {
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

    QeWork work = qe_work_for(ds, qe, work_slice);
    qe_drive_match(ds, qe, m, from, work);
    qe_run(ds, qe, work);
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
__device__ inline void qe_seed_root_instance(DeviceState ds, QeView qe, StateId root,
                                             uint32_t work_slice) {
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
    if (rec == UINT32_MAX) return;
    QeWork work = qe_work_for(ds, qe, work_slice);
    if (!work.push(h, rec, 0u)) { ds.errors.record(ErrorKind::kScratchOverflow); return; }
    qe_run(ds, qe, work);
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
                                         uint32_t depth, QeWork& work);

// Instance side of the rendezvous: replay every match already captured for this class.
__device__ inline void qe_drive_instance(DeviceState ds, QeView qe,
                                         const DeviceQcInstance& inst,
                                         uint64_t state_hash, uint32_t depth, QeWork& work) {
    if (depth >= qe.max_steps) return;   // final-depth instances are recorded, never expanded
    // NO DEPTH GUARD. There was one, because this function used to re-enter itself and the
    // per-thread stack it re-entered on is a fixed reservation the driver takes across every
    // resident thread. The descent is a worklist now, so what bounds it is that list's capacity
    // -- checked where the push happens, and reported as the capacity overflow it is.
    //
    // Published before scanning; pairs with the fence on the match side so a concurrent
    // instance and match cannot both miss each other.
    __threadfence();
    qe_for_each_match_from(qe, state_hash, [&](const DeviceSlotMatch& m) {
        qe_apply(ds, qe, inst, m, state_hash, depth, work);
    });
}

// Match side of the rendezvous: replay this match against every instance already standing at
// this class, at every depth it could stand at.
__device__ inline void qe_drive_match(DeviceState ds, QeView qe, const DeviceSlotMatch& m,
                                      uint64_t from_hash, QeWork& work) {
    __threadfence();
    for (uint32_t d = 0; d < qe.max_steps; ++d) {
        qe_for_each_instance(qe, from_hash, d, [&](const DeviceQcInstance& inst) {
            qe_apply(ds, qe, inst, m, from_hash, d, work);
        });
    }
}

// Drain the descent stack. Every instance an application publishes lands here rather than on the
// call stack, so this loop is the whole of the replay's depth.
__device__ inline void qe_run(DeviceState ds, QeView qe, QeWork& work) {
    QeWorkItem it;
    while (work.pop(it))
        qe_drive_instance(ds, qe, qe.instances.at(it.rec), it.hash, it.depth, work);
}

// The slice of the descent arena belonging to one driver. Out of range yields an empty stack,
// which pushes nothing and reports -- the same partial-work contract as any other capacity here.
__device__ inline QeWork qe_work_for(DeviceState ds, QeView qe, uint32_t slice) {
    QeWork w;
    if (qe.work_items == nullptr || slice >= qe.work_slices) {
        if (qe.replay) ds.errors.record(ErrorKind::kScratchOverflow);
        return w;
    }
    w.items = qe.work_items + static_cast<size_t>(slice) * qe.work_cap;
    w.cap   = qe.work_cap;
    return w;
}

struct DeviceQrCtx {
    using Instance = DeviceQcInstance;
    using Match    = QeMatchView;
    using Applied  = QeAppliedView;
    // REFERENCES, not copies. DeviceState and QeView are large aggregates and this Ctx is
    // constructed once per application, so holding either by value would copy it that often.
    // The caller's copies outlive this object.
    DeviceState& ds;
    QeView& qe;
    QeWork& work;

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
        (void)lo;
        (void)hi;
        atomicAdd(qe.num_branchial, 1u);
    }
    // __forceinline__ so its 1104-byte depot merges into qr_apply's frame rather than taking
    // one of its own. That frame is now paid ONCE per drive rather than once per level of
    // reconstruction depth -- this function ends the descent by pushing instead of calling --
    // but a depot that need not exist still should not. Measured with
    // tools/dev/ptx_frame_sizes.py.
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
        // PUSHED, NOT CALLED. This was the recursive edge; the driver loop takes it from here.
        if (!work.push(m.to_hash, rec, depth + 1u))
            ds.errors.record(ErrorKind::kScratchOverflow);
    }
};

__device__ __forceinline__ void qe_apply(DeviceState ds, QeView qe, const DeviceQcInstance& inst,
                                         const DeviceSlotMatch& m, uint64_t state_hash,
                                         uint32_t depth, QeWork& work) {
    if (!qe.enabled || depth >= qe.max_steps) return;
    DeviceQrCtx c{ds, qe, work};
    hgcommon::qr_apply(c, inst, QeMatchView(m, qe.arr_words), state_hash, depth);
}

// Host-side owner of the capture's device structures, so a run's records are one body of
// state whether the host seeding or the device loop wrote them. Token-sized when the route is
// off, and cleared between runs rather than rebuilt, for the same reason QcState is: the pools
// total tens of MB of cudaMalloc that an interactive caller would otherwise pay every evolve.
class QeState {
public:
    QeState(bool on, uint32_t max_events);
    ~QeState();
    QeState(const QeState&)            = delete;
    QeState& operator=(const QeState&) = delete;

    bool enabled() const;

    // Between runs: every map, list and record pool starts empty. The slot-array words need no
    // wipe -- records reference them by offset and the cursor restarts at zero.
    void clear();

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
    Counters counters_host() const;

    uint32_t num_matches_host();

    // Raw events the replay minted: one per (instance, match) application. The host's
    // qc_next_raw_event_, and the number a quotient run reports as its raw event count.
    uint32_t num_raw_events_host();

    // The reconstructed causal relation: distinct (producer, consumer) pairs, and the
    // consumed-edge occurrences behind them. The host's num_reconstructed_causal_pairs(false)
    // and num_reconstructed_causal_edges.
    uint32_t num_causal_pairs_host();
    uint32_t num_causal_edges_host();

    // Pairs tagged in-reduction: the TR view of the same relation. The host's
    // num_reconstructed_causal_pairs(true).

    // Distinct branchial pairs: sibling applications of one instance whose consumed edges
    // overlap. The host's num_reconstructed_branchial.
    uint32_t num_branchial_host();

    // The reconstructed relations as pairs of CONTENT TRIPLES. A count says two engines
    // disagree; a pair set says which pair is missing, which a count cannot.
    void reconstructed_pairs_host(std::vector<std::pair<uint64_t, uint64_t>>& causal,
                                  std::vector<std::pair<uint64_t, uint64_t>>& causal_reduced,
                                  std::vector<std::pair<uint64_t, uint64_t>>& branchial,
                                  bool want_branchial,
                                  std::vector<uint64_t>* event_signature,
                                  std::vector<std::pair<uint32_t, uint32_t>>* causal_raw = nullptr,
                                  std::vector<std::pair<uint32_t, uint32_t>>* causal_raw_reduced = nullptr,
                                  std::vector<std::pair<uint32_t, uint32_t>>* branchial_raw = nullptr);

    // Distinct event identities the replay produced under the run's mode. The host's
    // qc_num_canon_events_, and what a caller is told the event count is when a mode is selected.
    uint32_t num_canon_events_host();

    // Slots the frame moved off the state's own labelling, and slots no frame image existed for.
    uint32_t num_aligned_host();
    uint32_t num_align_failures_host();

    // Instances recorded this run. One per raw occurrence of a class at a depth: one per root
    // before any replay, and one more per application once the replay lands.
    uint32_t num_instances_host();

    // Size the descent arena for this run. Called before the launch, so the caller supplies the
    // number of drivers it will start (one per block in the persistent kernel, one per root in
    // the seeder) and the depth budget the stacks must hold.
    //
    // GROWS, NEVER SHRINKS, for the reason the IR arena does: an interactive caller reuses one
    // engine across many runs and a buffer whose contents never outlive a run should not be
    // reallocated on each of them.
    void ensure_work(uint32_t slices, uint32_t max_steps);

    QeView view(uint32_t max_steps, EventSignatureKeys keys,
                bool replay);

private:

    static uint32_t read_counter(const uint32_t* p, const char* what);

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
    // The replay's descent stacks: work_slices_ contiguous slices of work_cap_ items.
    QeWorkItem*               work_items_  = nullptr;
    uint32_t                  work_cap_    = 0;
    uint32_t                  work_slices_ = 0;
    bool                      on_ = false;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE