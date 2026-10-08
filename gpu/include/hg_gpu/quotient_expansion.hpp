#pragma once
#include <unordered_map>
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
#include "hg_gpu/work_log.hpp"      // the replay's task log
#include "hgcommon/core.hpp"        // sort_u64
#include "hgcommon/slot_core.hpp"  // slot_rank -- the frame-slot rule, shared with the host
#include "hgcommon/quotient_replay_core.hpp"  // qr_apply -- the replay, and the identity it mints
#include "hgcommon/quotient_multiplicity_core.hpp"  // qm_pass -- raw counts from class multiplicities
#include "hgcommon/quotient_causal_core.hpp"  // qc_key -- the (class, depth, orbit) key rule
#include "hgcommon/reach_core.hpp"  // redundant_producers -- the online reduction

#include <cuda/atomic>

namespace HG_NAMESPACE {
namespace gpu {

// One captured match of a canonical class, in that class's frame slots. The slot arrays live in
// the expansion word arena at arr_offset: consumed | produced | surv_from | surv_to |
// child_source (to_slots words, hgcommon::qr_fill_child_sources), contiguously.
struct DeviceSlotMatch {
    uint64_t to_hash = 0;
    uint32_t id = 0;              // dense; the replay's (instance, match) claim keys on it
    uint32_t local = 0;           // dense within its class, in capture order; UINT32_MAX if none
    uint32_t rule = 0;
    uint32_t from_slots = 0, to_slots = 0;
    uint32_t num_consumed = 0, num_produced = 0, num_survivors = 0;
    uint32_t arr_offset = 0;
    uint64_t from_hash = 0;
    // The claimed run-signature key and the output step it was claimed for
    // (hgcommon::qr_cached_key); accessed through atomic_ref.
    mutable uint64_t runsig_key = 0;
    mutable uint32_t runsig_step = hgcommon::QR_NO_STEP;

    // The slot arrays live contiguously in the expansion word arena at arr_offset:
    // consumed | produced | surv_from | surv_to | child_source. `words` is that arena's base, which the
    // record cannot hold because it is a device pointer the host rebuilds per run -- so the
    // view below binds the two together for hgcommon/quotient_replay_core.hpp, which reads
    // both engines' layouts through one set of calls.
    __host__ __device__ const uint32_t* at(const uint32_t* words) const { return words + arr_offset; }
};

// A DeviceSlotMatch bound to the arena its slots live in. What the shared replay sees.
struct QeMatchView {
    const DeviceSlotMatch* src;   // the record in the match pool
    const uint32_t* w;            // consumed | produced | surv_from | surv_to | child_source
    uint64_t to_hash;
    uint32_t id, local, rule, from_slots, to_slots;
    uint32_t num_consumed, num_produced, num_survivors;

    __host__ __device__ QeMatchView(const DeviceSlotMatch& m, const uint32_t* words)
        : src(&m), w(m.at(words)), to_hash(m.to_hash), id(m.id), local(m.local), rule(m.rule),
          from_slots(m.from_slots), to_slots(m.to_slots), num_consumed(m.num_consumed),
          num_produced(m.num_produced), num_survivors(m.num_survivors) {}

    __host__ __device__ uint32_t consumed(uint32_t i)  const { return w[i]; }
    __device__ uint32_t produced(uint32_t i)  const { return w[num_consumed + i]; }
    __device__ uint32_t surv_from(uint32_t i) const {
        return w[num_consumed + num_produced + i];
    }
    __device__ uint32_t surv_to(uint32_t i) const {
        return w[num_consumed + num_produced + num_survivors + i];
    }
    __host__ __device__ uint32_t child_source(uint32_t i) const {
        return w[num_consumed + num_produced + 2u * num_survivors + i];
    }
    __device__ const uint32_t* consumed_ptr() const { return w; }
    __device__ const uint32_t* produced_ptr() const { return w + num_consumed; }
    // The device serves the key sets None, Full and Automatic, none of which reads marked forms
    // (hgcommon::event_keys_mark_edges); the paclet runs every other key set on the CPU engine.
    __device__ const hgcommon::EventMarkedForms* marked_forms() const { return nullptr; }
};

// A captured match reference, bucketed by from_hash; the node carries its exact hash so the
// walkers can filter a shared bucket.
struct QeMatchRef {
    uint64_t from_hash;
    uint32_t record;
};

// (class, depth) as one key: hgcommon::qc_key with orbit 0, the key the host's instances and
// multiplicity points use. An instance is keyed by its class and depth alone.
__host__ __device__ __forceinline__ uint64_t qe_inst_key(uint64_t state_hash, uint32_t depth) {
    return hgcommon::qc_key(state_hash, depth, 0u);
}
// One raw occurrence of a canonical class, at one depth, recorded by its lineage: the parent
// instance's record, the record of the match that made it and that match's event, or
// kQeNoParent for a root. A slot's producing event is hgcommon::qr_producer_of over it.
//
// Slots rather than edge ids is the whole point: the class's captured matches are in frame
// slots, so an instance built from any raw state of the class replays them without knowing
// which raw edges the frame state happened to have.
inline constexpr uint32_t kQeNoParent = 0xFFFFFFFFu;
inline constexpr uint32_t kQeNoClaims = 0xFFFFFFFFu;
struct DeviceQcInstance {
    uint32_t id = 0;           // dense; the replay's (instance, match) claim keys on it
    uint32_t nslots = 0;
    uint32_t parent = kQeNoParent;
    uint32_t via = 0;
    uint32_t event = 0;        // a root instance: its initial state's StateId (genesis pairs)
    // The first block of the instance's claim chain (hgcommon::qr_claim_chain) in the expansion
    // arena: the next block's offset (kQeNoClaims for none), the block's 64-bit claim words, then
    // two 32-bit words per claim word. kQeNoClaims for an instance at the bound, and when a block
    // cannot be allocated; those pairs claim in `applied`.
    uint32_t claims = kQeNoClaims;
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


// A (class, instance record, depth) point: an instance the depth bound left standing
// (QeView::blocked), and an entry of the multiplicity cascade's queue.
struct QeWorkItem {
    uint64_t hash;    // the class the instance stands at
    uint32_t rec;     // its record in the instance pool
    uint32_t depth;
};

// ONE REPLAY APPLICATION: apply match record `match` to instance record `rec` of class `hash` at
// `depth`. The replay's unit of work on the device. Producers append tasks to QeView::tasks (a
// capture, one per standing instance of its class; a new instance, one per captured match of its
// class) and whole warps consume them, up to 32 at a time, one per lane. `published` is written
// last, with release order, and a consumer that claimed the index waits for it.
struct QeTask {
    uint64_t hash;
    uint32_t rec;
    uint32_t depth;
    uint32_t match;
    uint32_t published;
};

// The multiplicity cascade's queue: one slice of QeView::work_items per driver.
struct QeWork {
    QeWorkItem* items = nullptr;
    uint32_t    cap   = 0;
    uint32_t    n     = 0;
};

// Words per raw event in QeView::event_kept, and the local arrays the replay's redundancy search
// runs in before its block-scratch fallback (the sizes the full capture's search uses).
constexpr uint32_t kQeKeptStride = 5;
constexpr uint32_t kQeReachStack = 64;
constexpr uint32_t kQeReachTable = 128;   // power of two

struct QeView {
    typename Pool<DeviceSlotMatch>::DeviceView          matches;
    typename LockFreeList<QeMatchRef>::DeviceView       by_from;   // bucket(from_hash)

    typename Pool<DeviceQcInstance>::DeviceView         instances;
    // The instances recorded at the depth bound, which a raised bound has to drive: the device
    // twin of the replay half of Hypergraph::qc_blocked_. QeWorkItem is (class, record, depth).
    typename Pool<QeWorkItem>::DeviceView               blocked;
    typename LockFreeList<QeInstRef>::DeviceView        by_key;    // bucket(key(hash, depth))
    uint32_t* inst_next_id;    // device atomic; dense instance ids

    // Claims an (instance, match) application. An application mints a raw event, so unlike the
    // producer-set DP it is not idempotent and the pair must be claimed exactly once.
    DedupMap::DeviceView applied;
    // Matches captured per class, indexed by the class's representative raw state (rep);
    // class_nmatch_cap is max_states.
    uint32_t* class_nmatch;
    uint32_t  class_nmatch_cap;
    // B(c) per class, indexed as class_nmatch: pairs of the class's matches whose consumed slots
    // overlap, the sum of their b_j. Accumulated at capture when branchial pairs are counted.
    unsigned long long* class_pairs;
    uint32_t* next_raw_event;  // device atomic; dense raw-event ids

    // Slots the frame MOVED -- resolved through a state that did not hold the frame, and landing
    // somewhere other than the state's own slot. Counting corrections rather than lookups is
    // what makes it evidence: a lookup that returns the state's own slot changes nothing.
    // align_fail is the host's qc_align_fail_ / qc_align_badcorr_.
    uint32_t* align_moved;
    uint32_t* align_fail;

    // Distinct run identities and their count. Empty under EVENT_SIG_NONE, where every
    // application is its own event and the raw count is already the answer.
    // Probe key of a run signature -> (match record << 32 | output step) of the class's first
    // application (qe_claim_runsig).
    ConcurrentMap<uint64_t, uint64_t>::DeviceView canon_seen;
    uint32_t* num_canon;
    EventSignatureKeys keys;

    // The reconstructed causal relation. `pairs` claims each (producer, consumer) exactly once;
    // `num_causal_edges` counts every consumed-edge occurrence, so a pair joined by several
    // edges is one pair and several edges -- the two the host reports separately.
    DedupMap::DeviceView causal_pairs;
    uint32_t* num_causal_pairs;
    uint32_t* num_causal_edges;
    // Per raw event, the producers the online transitive reduction kept (qr_apply, through
    // hgcommon::redundant_producers), kQeKeptStride words each: the count, three producers inline,
    // and the arr_words offset of the rest when the count exceeds three. Written by the event's
    // own application before its descent and read by later events' searches through L2. The
    // host's qc_kept_. Null when causal is not recorded.
    uint32_t* event_kept;
    uint32_t* num_reduced_pairs;


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
    // Per raw event, its input class, output class and rule, when a caller reads the
    // applications as events or graphs; null otherwise, and then nothing is written.
    uint64_t* event_from_class;
    uint64_t* event_to_class;
    uint32_t* event_rule;

    typename LockFreeList<QeAppliedMatch>::DeviceView inst_applied;

    // canonical hash -> (StateId + 1) of the state whose matches define this class's expansion.
    // +1 because the map reserves 0 as its EMPTY sentinel, so a raw key of StateId 0 could never
    // be stored -- the same offset, for the same reason, as the None-mode dedup key.
    DedupMap::DeviceView rep;

    // canonical hash -> (StateId + 1) of the state whose labelling is this class's FRAME, and
    // that state's step. Separate from `rep`: a class is given a frame by both endpoints of
    // every captured transition, so a class first seen as an output owns its frame from a state
    // that need never expand. The step is what the Automatic signature keys on.
    FrameMap::DeviceView frame;        // canonical hash -> (sid+1) | (step+1)<<32

    // Bump arena for the matches' slot arrays.
    uint32_t* arr_words;
    uint32_t* arr_cursor;      // device atomic
    uint32_t  arr_capacity;

    uint32_t* next_id;         // device atomic; dense match ids

    // Backing store for the multiplicity queues, one contiguous slice of `work_cap` items per
    // driver.
    // Sized from the run's step budget the way the IR arena is sized from its state budget, and
    // exhausted the same way: a capacity overflow that reports and returns partial work.
    QeWorkItem* work_items  = nullptr;
    uint32_t    work_cap    = 0;   // items per driver
    uint32_t    work_slices = 0;   // drivers this run can serve

    // THE REPLAY'S TASK LOG (hg_gpu/work_log.hpp). A warp of the persistent kernel claims up to
    // 32 consecutive tasks and runs one per lane. Every producer is inside a counted unit (the
    // record whose rewrite captured a match, or the task whose application made an instance).
    //
    // The host runs a descent on the thread that produced it. On the device one lane runs the
    // same code 62.9x slower than a host core (device IR on one state, persistent.cu), so the
    // replay's parallelism has to come from running many applications at once, one per lane.
    WorkLogView<QeTask> tasks{};
    // Per lane of the grid, the arena offset + 1 of its reachability-search slice
    // (DeviceQrCtx::redundant), claimed on its first overflow; 0 until then.
    uint32_t* lane_reach = nullptr;
    uint32_t  lane_reach_slots = 0;

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

    // RAW COUNTS FROM CLASS MULTIPLICITIES (hgcommon/quotient_multiplicity_core.hpp), run in
    // place of the replay when the raw events and branchial pairs are read only as counts. The
    // host's quotient_multiplicity_. Set only together with `replay`.
    uint32_t  multiplicity = 0;
    // (class, depth) point key -> index + 1 into qm_mass / qm_queued.
    DedupMap::DeviceView qm_points;
    // (match id, depth) key -> index + 1 into qm_consumed_cells.
    DedupMap::DeviceView qm_consumed;
    // match id + 1 -> b_j + 1. Present once the match is ready.
    DedupMap::DeviceView qm_overlaps;
    unsigned long long* qm_mass           = nullptr;
    unsigned long long* qm_point_class    = nullptr;   // per point: its class hash
    uint32_t*           qm_point_depth    = nullptr;   // per point: its depth
    uint32_t*           qm_queued         = nullptr;
    unsigned long long* qm_consumed_cells = nullptr;
    uint32_t*           qm_cursor         = nullptr;   // [0] next point, [1] next consumed cell
    uint32_t            qm_capacity       = 0;         // entries in each array above
    unsigned long long* qm_counts         = nullptr;   // [0] raw events, [1] branchial
};

// The rendezvous: publishing an instance appends a task per captured match of its class,
// publishing a match appends a task per standing instance of its class, and an application
// publishes a child instance. Declared here so each publisher can append without the
// definitions having to be ordered around each other.
__device__ inline void qe_task_append(const DeviceState& ds, QeView qe, uint64_t hash, uint32_t rec,
                                      uint32_t depth, uint32_t match);
__device__ inline void qe_drive_instance(const DeviceState& ds, QeView qe, uint32_t rec,
                                         uint64_t state_hash, uint32_t depth);
__device__ inline void qe_drive_match(const DeviceState& ds, QeView qe, uint32_t match_rec,
                                      uint64_t from_hash);
__device__ inline QeWork qe_work_for(const DeviceState& ds, QeView qe, uint32_t slice);
__device__ __forceinline__ uint32_t qe_alloc_words(const DeviceState& ds, QeView qe, uint32_t n);

// Bucket a hash into a list's key space.
//
// The full 64-bit value modulo the key count, so `num_keys` need not be a power of two.
__host__ __device__ __forceinline__ uint32_t qe_bucket(uint64_t h, uint32_t num_keys) {
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return static_cast<uint32_t>(h % (num_keys ? num_keys : 1u));
}

// A (class, depth) key's instance list is split over kQeInstShards buckets; a pusher takes the
// shard of its lane across the grid and a scanner walks all of them. Every lane of a warp
// creates instances, so a shard per block would put a warp's lanes on one list head.
constexpr uint32_t kQeInstShards = 16;
__host__ __device__ __forceinline__ uint32_t qe_inst_bucket_of(uint64_t key, uint32_t shard,
                                                              uint32_t num_keys) {
    return qe_bucket(key + 0x9E3779B97F4A7C15ull * shard, num_keys);
}
__device__ __forceinline__ uint32_t qe_inst_bucket(const QeView& qe, uint64_t key, uint32_t shard) {
    return qe_inst_bucket_of(key, shard, qe.by_key.num_keys);
}
// Two shards of one key can hash to the same bucket. A walk over a key's shards takes a bucket
// from the first shard that lands in it only, or it visits that bucket's instances twice.
__host__ __device__ __forceinline__ bool qe_inst_shard_first(uint64_t key, uint32_t shard,
                                                             uint32_t num_keys) {
    const uint32_t b = qe_inst_bucket_of(key, shard, num_keys);
    for (uint32_t t = 0; t < shard; ++t)
        if (qe_inst_bucket_of(key, t, num_keys) == b) return false;
    return true;
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
__device__ __forceinline__ uint32_t qe_slot_of(const DeviceState& ds, StateId sid, EdgeId edge) {
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
__device__ __forceinline__ uint32_t qe_rank_of(const DeviceState& ds, StateId sid, EdgeId edge) {
    if (!ds.state_edge_rank) return UINT32_MAX;
    const uint32_t i = state_edge_index(ds, sid, edge);
    return i == UINT32_MAX ? UINT32_MAX : ds.state_edge_rank[i];
}

// Register `sid` as the frame of its class if no state holds it yet. Idempotent, and the winner
// is whichever state gets there first -- which is all the frame has to be, since every state of
// the class is isomorphic to it. TRUE when the map is full and the frame could not be recorded:
// a full map refuses every later insert, so the caller reports kCanonicalMapFull rather than
// replaying against classes with no frame.
__device__ __forceinline__ bool qe_register_frame(QeView qe, uint64_t class_hash, StateId sid) {
    return qe.frame.insert_if_absent(
        class_hash, hgcommon::id_key(0u, static_cast<uint32_t>(sid))).overflowed;
}

// Alignment outcomes counted into a caller's local and published when it returns.
//
// qe_capture_expansion calls qe_frame_slot_of up to 2n times for an n-edge state, and each call
// scans the frame's whole slice, so an atomic per call sits inside an O(n^2) nest -- on one
// address, from every block. These are statistics with no in-run reader: only
// num_aligned_host/num_align_failures_host and the differential suite read them, after the run.
//
// The destructor publishes, because the capture returns from inside its loops on exactly the
// paths a failure is counted on -- a publish written at the end would drop the counts it exists
// to record.
struct QeAlignTally {
    uint32_t* moved_out;
    uint32_t* failed_out;
    uint32_t  moved  = 0;
    uint32_t  failed = 0;
    __device__ ~QeAlignTally() {
        if (moved)  atomicAdd(moved_out, moved);
        if (failed) atomicAdd(failed_out, failed);
    }
};

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
__device__ inline uint32_t qe_frame_slot_of(const DeviceState& ds, QeView qe, uint64_t class_hash,
                                            StateId sid, EdgeId edge, QeAlignTally& tally) {
    const auto held = qe.frame.lookup(class_hash);
    if (!held.found || held.value == 0) { ++tally.failed; return UINT32_MAX; }
    const StateId frame = static_cast<StateId>(hgcommon::id_pair_from_key(held.value).b);
    if (frame == sid) return qe_slot_of(ds, sid, edge);

    if (!ds.state_edge_rank || !ds.state_edge_orbit || frame >= ds.max_states) {
        ++tally.failed;
        return UINT32_MAX;
    }
    const uint32_t r = qe_rank_of(ds, sid, edge);
    if (r != UINT32_MAX) {
        const StateEdgeSlice fsl = ds.state_edge_slices[frame];
        for (uint32_t k = 0; k < fsl.count; ++k) {
            if (ds.state_edge_rank[fsl.offset + k] != r) continue;
            const uint32_t fs = hgcommon::slot_rank(ds.state_edge_orbit + fsl.offset, fsl.count, k);
            if (fs != qe_slot_of(ds, sid, edge)) ++tally.moved;
            return fs;
        }
    }
    ++tally.failed;
    return UINT32_MAX;
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

// A replay application's event class, the host's Hypergraph::claim_replay_event: the run
// signature of match `m` from class `from` with output step `out_step`, claimed in canon_seen on
// its values. A key hit recomputes the class's values from its first application's (match,
// output step). The match keeps its claimed key for one output step (hgcommon::qr_cached_key),
// so later applications of the match skip the signature.
struct QeRunsigClaim {
    uint64_t key;
    bool     won;
};

__device__ inline QeRunsigClaim qe_claim_runsig(const DeviceState& ds, const QeView& qe,
                                                const QeMatchView& m, uint64_t from,
                                                uint32_t out_step) {
    struct Cells {
        const DeviceSlotMatch& m;
        __device__ uint32_t step_load() const {
            return cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(m.runsig_step)
                .load(cuda::memory_order_relaxed);
        }
        __device__ bool step_cas(uint32_t expected, uint32_t desired) {
            return cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(m.runsig_step)
                .compare_exchange_strong(expected, desired, cuda::memory_order_relaxed);
        }
        __device__ uint64_t key_load() const {
            return cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(m.runsig_key)
                .load(cuda::memory_order_acquire);
        }
        __device__ void key_store(uint64_t key) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(m.runsig_key)
                .store(key, cuda::memory_order_release);
        }
    } cells{*m.src};
    uint64_t cached = 0;
    if (hgcommon::qr_cached_key(cells, out_step, cached)) return {cached, false};

    hgcommon::QrRunSignature sig;
    hgcommon::qr_signature_values(qe.keys, m, from, out_step, sig);
    const uint32_t record = static_cast<uint32_t>(m.src - qe.matches.data);
    struct P {
        const QeView& qe;
        const hgcommon::QrRunSignature& sig;
        uint32_t record;
        uint32_t out_step;
        __device__ bool same(uint64_t v) const {
            const uint32_t r = static_cast<uint32_t>(v >> 32);
            const uint32_t s = static_cast<uint32_t>(v);
            if (r == record && s == out_step) return true;
            const DeviceSlotMatch& first = qe.matches.at(r);
            hgcommon::QrRunSignature theirs;
            hgcommon::qr_signature_values(qe.keys, QeMatchView(first, qe.arr_words),
                                          first.from_hash, s, theirs);
            return hgcommon::qr_same_values(theirs, sig);
        }
        __device__ bool make(uint64_t& v) const {
            v = (static_cast<uint64_t>(record) << 32) | out_step;
            return true;
        }
        __device__ uint32_t rep_of(uint64_t) const { return 0; }
    } p{qe, sig, record, out_step};
    const StateClaim c = keyed_claim_device(ds, 0u, sig.sig & ds.event_key_mask, qe.canon_seen, p);
    hgcommon::qr_cache_key(cells, out_step, c.key);
    return {c.key, c.fresh};
}

// The saturating add of hgcommon::qm_sat_add on a device counter.
__device__ inline void qe_qm_add(unsigned long long* counter, uint64_t delta) {
    unsigned long long old = *reinterpret_cast<volatile unsigned long long*>(counter);
    for (;;) {
        const unsigned long long next = hgcommon::qm_sat_add(old, delta);
        const unsigned long long seen = atomicCAS(counter, old, next);
        if (seen == old) return;
        old = seen;
    }
}

// The storage face hgcommon/quotient_multiplicity_core.hpp drives; the host's
// Hypergraph::QmCtx. The cascade queue is this driver's descent slice, kept as a heap on depth.
struct DeviceQmCtx {
    using Match = QeMatchView;
    const DeviceState& ds;
    QeView& qe;
    QeWork& work;

    // The index behind `key` in `map`, claiming and zeroing a fresh one when absent.
    // UINT32_MAX when the array or the map is full, which is reported as kQcNodes.
    __device__ uint32_t index(DedupMap::DeviceView& map, uint64_t key, uint32_t cursor,
                              bool is_point, uint64_t class_hash = 0, uint32_t depth = 0) {
        const auto r = map.lookup(key);
        if (r.found && r.value) return r.value - 1u;
        const uint32_t idx = atomicAdd(&qe.qm_cursor[cursor], 1u);
        if (idx >= qe.qm_capacity) { ds.errors.record(ErrorKind::kQcNodes); return UINT32_MAX; }
        if (is_point) {
            qe.qm_mass[idx] = 0ull;
            qe.qm_queued[idx] = 0u;
            qe.qm_point_class[idx] = class_hash;
            qe.qm_point_depth[idx] = depth;
        }
        else          { qe.qm_consumed_cells[idx] = 0ull; }
        __threadfence();
        const auto ins = map.insert_if_absent(key, idx + 1u);
        if (ins.overflowed) { ds.errors.record(ErrorKind::kQcNodes); return UINT32_MAX; }
        return ins.value - 1u;
    }
    __device__ uint32_t point(uint64_t class_hash, uint32_t depth) {
        return index(qe.qm_points, hgcommon::avoid_reserved_keys(hgcommon::qc_key(class_hash, depth, 0u)),
                     0u, true, class_hash, depth);
    }
    __device__ uint32_t cell(uint32_t match_id, uint32_t depth) {
        return index(qe.qm_consumed, hgcommon::qr_apply_key(match_id, depth), 1u, false);
    }

    __device__ uint32_t max_steps() const { return qe.max_steps; }
    __device__ bool ready(const QeMatchView& m, uint64_t& b) const {
        const auto r = qe.qm_overlaps.lookup(static_cast<uint64_t>(m.id) + 1u);
        if (!r.found || r.value == 0) return false;
        b = r.value - 1u;
        return true;
    }
    __device__ uint64_t mass(uint64_t class_hash, uint32_t depth) const {
        const auto r = qe.qm_points.lookup(
            hgcommon::avoid_reserved_keys(hgcommon::qc_key(class_hash, depth, 0u)));
        if (!r.found || r.value == 0) return 0;
        return *reinterpret_cast<volatile unsigned long long*>(&qe.qm_mass[r.value - 1u]);
    }
    __device__ void add_mass(uint64_t class_hash, uint32_t depth, uint64_t delta) {
        const uint32_t p = point(class_hash, depth);
        if (p == UINT32_MAX) return;
        qe_qm_add(&qe.qm_mass[p], delta);
    }
    __device__ uint64_t consumed(const QeMatchView& m, uint32_t depth) {
        const auto r = qe.qm_consumed.lookup(hgcommon::qr_apply_key(m.id, depth));
        if (!r.found || r.value == 0) return 0;
        return *reinterpret_cast<volatile unsigned long long*>(&qe.qm_consumed_cells[r.value - 1u]);
    }
    __device__ bool advance(const QeMatchView& m, uint32_t depth, uint64_t& expected,
                            uint64_t desired) {
        const uint32_t c = cell(m.id, depth);
        if (c == UINT32_MAX) return true;   // reported; the pass is dropped
        const unsigned long long seen = atomicCAS(&qe.qm_consumed_cells[c], expected, desired);
        if (seen == expected) return true;
        expected = seen;
        return false;
    }
    __device__ void count(uint64_t events) {
        qe_qm_add(&qe.qm_counts[0], events);
    }
    __device__ hgcommon::EventSignatureKeys keys() const { return qe.keys; }
    __device__ void note_signature(const QeMatchView& m, uint64_t from_class, uint32_t out_step) {
        if (qe_claim_runsig(ds, qe, m, from_class, out_step).won) atomicAdd(qe.num_canon, 1u);
    }
    __device__ bool claim_queued(uint64_t class_hash, uint32_t depth) {
        const uint32_t p = point(class_hash, depth);
        return p != UINT32_MAX && atomicExch(&qe.qm_queued[p], 1u) == 0u;
    }
    __device__ void push(uint64_t class_hash, uint32_t depth) {
        if (work.n >= work.cap) { ds.errors.record(ErrorKind::kQeWorkOverflow); return; }
        work.items[work.n] = QeWorkItem{class_hash, 0u, depth};
        ++work.n;
        hgcommon::qm_heap_push(work.items, work.n);
    }
    __device__ bool pop(uint64_t& class_hash, uint32_t& depth) {
        if (work.n == 0) return false;
        const QeWorkItem t = hgcommon::qm_heap_pop(work.items, work.n);
        --work.n;
        class_hash = t.hash;
        depth = t.depth;
        const uint32_t p = point(class_hash, depth);
        if (p != UINT32_MAX) atomicExch(&qe.qm_queued[p], 0u);
        return true;
    }
    template <class F>
    __device__ void for_each_match(uint64_t class_hash, F&& f) {
        qe_for_each_match_from(qe, class_hash, [&](const DeviceSlotMatch& m) {
            f(QeMatchView(m, qe.arr_words));
        });
    }
    __device__ void fence() { __threadfence(); }
};

// The root's unit of mass; the multiplicity twin of the root instance. __noinline__, with
// qe_capture_multiplicity: inlined into the persistent kernel, ptxas exceeded 4 GB on
// persistent.cu.
__device__ inline __noinline__ void qe_seed_multiplicity(const DeviceState& ds, QeView qe, uint64_t root_hash,
                                            QeWork& work) {
    DeviceQmCtx c{ds, qe, work};
    hgcommon::qm_credit(c, root_hash, 0u, 1ull);
    hgcommon::qm_drain(c);
}

// The capture side of the multiplicity count: b_j over the matches linked into the class's
// bucket before this one, then ready, then the mass already standing at the class at every
// depth. The host's branch in Hypergraph::qc_capture_expansion.
// b_j: the matches of class `from` linked into the bucket before `at` whose consumed slots
// overlap `consumed`.
__device__ inline uint32_t qe_overlaps_before(QeView qe, uint64_t from, uint32_t at,
                                              const uint32_t* consumed, uint32_t nc) {
    uint32_t b = 0;
    qe.by_from.for_each_before(at, [&](const QeMatchRef& r) {
        if (r.from_hash != from) return;
        if (hgcommon::qr_consumed_overlap(consumed, nc, QeMatchView(qe.matches.at(r.record), qe.arr_words)))
            ++b;
    });
    return b;
}

__device__ inline __noinline__ void qe_capture_multiplicity(const DeviceState& ds, QeView qe, const DeviceSlotMatch& m,
                                               uint64_t from, uint32_t b, QeWork& work) {
    if (qe.qm_overlaps.insert_if_absent(static_cast<uint64_t>(m.id) + 1u, b + 1u).overflowed) {
        ds.errors.record(ErrorKind::kQcNodes);
        return;
    }
    DeviceQmCtx c{ds, qe, work};
    c.fence();
    const QeMatchView v(m, qe.arr_words);
    for (uint32_t d = 0; d < qe.max_steps; ++d) hgcommon::qm_pass(c, v, from, d);
    hgcommon::qm_drain(c);
}

// Drive the points a previous run's depth bound left standing, for one driver (`slice`) of
// `stride`: the multiplicity points and the recorded instances whose depth is in
// [old_bound, qe.max_steps). The device twin of Hypergraph::quotient_redrive_point, for a session
// continued past the depth it last stopped at.
__device__ inline __noinline__ void qe_redrive(const DeviceState& ds, QeView qe, uint32_t old_bound,
                                               uint32_t slice, uint32_t stride) {
    if (qe.multiplicity) {
        QeWork work = qe_work_for(ds, qe, slice);
        const uint32_t n = qe.qm_cursor[0] < qe.qm_capacity ? qe.qm_cursor[0] : qe.qm_capacity;
        DeviceQmCtx c{ds, qe, work};
        for (uint32_t p = slice; p < n; p += stride) {
            const uint32_t d = qe.qm_point_depth[p];
            if (d < old_bound || d >= qe.max_steps) continue;
            if (c.claim_queued(qe.qm_point_class[p], d)) c.push(qe.qm_point_class[p], d);
            hgcommon::qm_drain(c);
        }
    }
    if (qe.replay) {
        const uint32_t n = qe.blocked.size();
        for (uint32_t i = slice; i < n; i += stride) {
            const QeWorkItem it = qe.blocked.at(i);
            if (it.depth < old_bound || it.depth >= qe.max_steps) continue;
            qe_drive_instance(ds, qe, it.rec, it.hash, it.depth);
        }
    }
}

// Capture one raw event as its class's expansion match, in frame slots.
//
// Only the class's claimed state contributes: the first parent to claim the class defines both
// the expansion and the frame, and every later parent of the same class returns immediately.
// That is what makes the record a property of the CLASS rather than of whichever raw state
// happened to be expanded first by this schedule.
//
// `depth` is the parent's depth (the event's step - 1).
//
// ALL 32 LANES OF THE BLOCK'S WARP CALL IT. A frame slot costs a scan of the frame state's
// slice, and a capture takes one per consumed and produced edge and two per surviving edge, so
// the slots are computed one per lane: lane i takes consumed edge i or produced edge i - nc (at
// most 2 * kMaxPatternEdges = 32), and the child's edges are taken 32 at a time with a ballot
// compacting the survivors into `surv_shared` (kLocalSurvivors entries, block-shared) or the
// block's survivor scratch. Lane 0 takes the claim, registers the frames, sorts the survivors
// and publishes the record.
__device__ inline void qe_capture_expansion(const DeviceState& ds, QeView qe,
                                            StateId parent, StateId child, EventId event,
                                            uint32_t rule, uint32_t depth, uint32_t work_slice,
                                            uint64_t* surv_shared) {
    if (!qe.enabled || depth > qe.max_steps) return;
    const uint32_t lane = threadIdx.x & 31u;

    const uint64_t from = ds.state_canonical_hash[parent];
    const uint64_t to   = ds.state_canonical_hash[child];

    // One raw state's matches define the class's expansion; every later parent of the same class
    // drops out here, so the record is a property of the CLASS and not of the schedule.
    uint32_t go = 0;
    if (lane == 0) {
        const uint32_t claim = static_cast<uint32_t>(parent) + 1u;
        const auto won = qe.rep.insert_if_absent(from, claim);
        if (won.overflowed) ds.errors.record(ErrorKind::kQcNodes);
        if (won.value == claim) {
            go = 1;
            // Both endpoints are given a frame before any slot is taken, so every slot below
            // resolves. A short-circuiting || would skip the second whenever the first
            // overflowed, and the second is what the child side's slots resolve against.
            const bool frame_from = qe_register_frame(qe, from, parent);
            const bool frame_to   = qe_register_frame(qe, to, child);
            if (frame_from || frame_to) ds.errors.record(ErrorKind::kCanonicalMapFull);
        }
    }
    if (!__shfl_sync(0xffffffffu, go, 0)) return;

    const DeviceEvent& ev = ds.event_pool.at(event);
    const uint32_t nc = ev.num_consumed, np = ev.num_produced;

    // Publishes on every path out of this function, including the drop-the-capture returns
    // below, which are the ones a failure is counted on.
    QeAlignTally align{qe.align_moved, qe.align_fail};

    uint32_t my_slot = 0;
    bool bad = false;
    if (lane < nc) {
        my_slot = qe_frame_slot_of(ds, qe, from, parent, event_consumed_edge(ds, ev, lane), align);
        bad = my_slot == UINT32_MAX;
    } else if (lane < nc + np) {
        my_slot = qe_frame_slot_of(ds, qe, to, child, ev.first_produced + (lane - nc), align);
        bad = my_slot == UINT32_MAX;
    }
    // No frame slot: drop rather than corrupt.
    if (__any_sync(0xffffffffu, bad)) {
        if (lane == 0) ds.errors.record(ErrorKind::kCapturesDropped);
        return;
    }
    uint32_t consumed[kMaxPatternEdges];
    uint32_t produced[kMaxPatternEdges];
    for (uint32_t i = 0; i < nc; ++i) consumed[i] = __shfl_sync(0xffffffffu, my_slot, i);
    for (uint32_t i = 0; i < np; ++i) produced[i] = __shfl_sync(0xffffffffu, my_slot, nc + i);

    // Survivors: child edges that were not freshly produced passed through from the parent (the
    // child's slice is parent-minus-consumed plus produced by construction). Recorded as one
    // packed (parent slot, child slot) pair so a single sort orders them, through
    // hgcommon::id_key like every other pair in this engine -- its +1 offset is applied to both
    // halves, so it preserves the ordering the sort relies on.
    // At most one survivor per child edge, so the child's size bounds the list (survivor_buffer).
    const StateEdgeSlice csl = ds.state_edge_slices[child];
    uint64_t* surv = survivor_buffer(ds, surv_shared, csl.count, work_slice);
    if (surv == nullptr) {
        if (lane == 0) ds.errors.record(ErrorKind::kQeSurvivorsOverflow);
        return;
    }
    uint32_t ns = 0;
    bool lost = false;   // a survivor with no frame image
    for (uint32_t base = 0; base < csl.count; base += 32u) {
        const uint32_t k = base + lane;
        uint64_t key = 0;
        bool keep = false;
        if (k < csl.count) {
            const EdgeId oe = ds.state_edge_ids[csl.offset + k];
            const bool produced_here = oe - ev.first_produced < np;
            if (!produced_here) {
                const uint32_t ps = qe_frame_slot_of(ds, qe, from, parent, oe, align);
                const uint32_t cs = qe_frame_slot_of(ds, qe, to, child, oe, align);
                if (ps != UINT32_MAX && cs != UINT32_MAX) {
                    key = hgcommon::id_key(ps, cs);
                    keep = true;
                } else {
                    lost = true;
                }
            }
        }
        const uint32_t mask = __ballot_sync(0xffffffffu, keep);
        if (keep) surv[ns + __popc(mask & ((1u << lane) - 1u))] = key;
        ns += __popc(mask);
    }
    // A survivor with no frame image drops the whole capture, as an unaligned consumed or
    // produced edge does above and as the host drops a capture whose frame slots it cannot
    // resolve (Hypergraph's qc_frame_slots).
    if (__any_sync(0xffffffffu, lost)) {
        if (lane == 0) ds.errors.record(ErrorKind::kCapturesDropped);
        return;
    }
    __syncwarp();
    // Lane 0 sorts the survivors and publishes the record; the index is broadcast, and the
    // match side of the rendezvous then scans on every lane (qe_drive_match).
    uint32_t published = UINT32_MAX;
    if (lane == 0) published = [&]() -> uint32_t {
        hgcommon::sort_u64(surv, ns);

        // Copy the slot arrays into the expansion arena, then publish the record.
        const uint32_t to_slots = ds.state_edge_slices[child].count;
        const uint32_t need = nc + np + 2u * ns + to_slots;
        uint32_t off = 0;
        if (need) {
            off = qe_alloc_words(ds, qe, need);
            if (off == UINT32_MAX) return UINT32_MAX;
            uint32_t* w = qe.arr_words + off;
            for (uint32_t i = 0; i < nc; ++i) *w++ = consumed[i];
            uint32_t* pw = w;
            for (uint32_t i = 0; i < np; ++i) *w++ = produced[i];
            uint32_t* fw = w;
            for (uint32_t i = 0; i < ns; ++i) *w++ = hgcommon::id_pair_from_key(surv[i]).a;
            uint32_t* tw = w;
            for (uint32_t i = 0; i < ns; ++i) *w++ = hgcommon::id_pair_from_key(surv[i]).b;
            hgcommon::qr_fill_child_sources(pw, np, fw, tw, ns, to_slots, w);
        }

        const uint32_t rec = qe.matches.claim();
        if (rec == Pool<DeviceSlotMatch>::kInvalid) {
            ds.errors.record(ErrorKind::kQcNodes);
            return UINT32_MAX;
        }
        DeviceSlotMatch& m = qe.matches.at(rec);
        m.to_hash = to;
        {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> nid(*qe.next_id);
            m.id = nid.fetch_add(1u, cuda::memory_order_relaxed);
        }
        // `parent` holds the class's rep claim (above), so it indexes the class.
        m.local = UINT32_MAX;
        if (static_cast<uint32_t>(parent) < qe.class_nmatch_cap) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> n(qe.class_nmatch[parent]);
            m.local = n.fetch_add(1u, cuda::memory_order_acq_rel);
        }
        m.rule = rule;
        m.from_slots = ds.state_edge_slices[parent].count;
        m.to_slots   = to_slots;
        m.num_consumed = nc; m.num_produced = np; m.num_survivors = ns;
        m.arr_offset = off;
        m.from_hash = from;
        m.runsig_key = 0;
        m.runsig_step = hgcommon::QR_NO_STEP;

        const uint32_t at =
            qe.by_from.push(qe_bucket(from, qe.by_from.num_keys), QeMatchRef{from, rec});
        if (at == INVALID_ID) {
            ds.errors.record(ErrorKind::kQcNodes);
            return UINT32_MAX;
        }

        // b_j into the class's B(c) whenever branchial pairs are counted (QeState::count_branchial).
        if (qe.multiplicity || (qe.replay && ds.record_branchial)) {
            const uint32_t b = qe_overlaps_before(qe, from, at, consumed, nc);
            if (b && static_cast<uint32_t>(parent) < qe.class_nmatch_cap)
                atomicAdd(&qe.class_pairs[parent], static_cast<unsigned long long>(b));
            if (qe.multiplicity) {
                QeWork work = qe_work_for(ds, qe, work_slice);
                qe_capture_multiplicity(ds, qe, m, from, b, work);
            }
        }
        return rec;
    }();
    published = __shfl_sync(0xffffffffu, published, 0);
    if (published != UINT32_MAX && qe.replay) qe_drive_match(ds, qe, published, from);
}

// Reserve `n` words of the expansion arena. Returns UINT32_MAX when the arena is exhausted,
// which the caller reports as a capacity overflow rather than writing past the end.
//
// The test is in 64 bits and a refused reservation pulls the cursor back to the capacity, as
// Pool::claim_n does. Refused reservations still advance the cursor first, and on a run that
// overflows repeatedly (bigpath at 3 steps) the cursor passed 2^32: a 32-bit `off + n` then
// wrapped below the capacity and the caller wrote 16 GB past the arena.
__device__ __forceinline__ uint32_t qe_alloc_words(const DeviceState& ds, QeView qe, uint32_t n) {
    if (n == 0) return 0;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> cur(*qe.arr_cursor);
    const uint32_t off = cur.fetch_add(n, cuda::memory_order_relaxed);
    if (static_cast<uint64_t>(off) + n > qe.arr_capacity) {
        cur.fetch_min(qe.arr_capacity, cuda::memory_order_relaxed);
        ds.errors.record(ErrorKind::kQeWordsFull);
        return UINT32_MAX;
    }
    return off;
}

// A zeroed claim block of `words` 64-bit claim words (DeviceQcInstance::claims); kQeNoClaims when
// the arena is full.
__device__ __forceinline__ uint32_t qe_new_claim_block(const DeviceState& ds, QeView qe,
                                                       uint32_t words) {
    const uint32_t off = qe_alloc_words(ds, qe, 2u + 2u * words);
    if (off == UINT32_MAX) return kQeNoClaims;
    qe.arr_words[off] = kQeNoClaims;
    qe.arr_words[off + 1u] = words;
    for (uint32_t i = 0; i < 2u * words; ++i) qe.arr_words[off + 2u + i] = 0u;
    return off;
}

// Record one instance of `state_hash` at `depth`, made from instance record `parent` by match
// record `via`, whose event is `event`; a root has parent kQeNoParent and `event` its initial
// state's StateId. The device twin of Hypergraph::qc_add_instance.
__device__ inline uint32_t qe_add_instance(const DeviceState& ds, QeView qe, uint64_t state_hash,
                                           uint32_t depth, uint32_t parent, uint32_t via,
                                           uint32_t event, uint32_t nslots) {
    if (!qe.enabled || depth > qe.max_steps) return UINT32_MAX;

    const uint32_t rec = qe.instances.claim();
    if (rec == Pool<DeviceQcInstance>::kInvalid) {
        ds.errors.record(ErrorKind::kQeInstancesFull);
        return UINT32_MAX;
    }
    DeviceQcInstance& inst = qe.instances.at(rec);
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> nid(*qe.inst_next_id);
        inst.id = nid.fetch_add(1u, cuda::memory_order_relaxed);
    }
    inst.nslots      = nslots;
    inst.parent      = parent;
    inst.via         = via;
    inst.event       = event;
    inst.claims      = kQeNoClaims;
    // Claim words only for an instance that will be expanded; one at the bound claims nothing.
    if (depth < qe.max_steps) {
        uint32_t class_matches = 0;
        const auto rep = qe.rep.lookup(state_hash);
        if (rep.found && rep.value - 1u < qe.class_nmatch_cap) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> n(qe.class_nmatch[rep.value - 1u]);
            class_matches = n.load(cuda::memory_order_acquire);
        }
        inst.claims = qe_new_claim_block(ds, qe, hgcommon::qr_claim_words(class_matches));
    }

    // At the bound the instance is recorded and not expanded; a continuation drives it.
    if (depth >= qe.max_steps) {
        const uint32_t b = qe.blocked.claim();
        if (b == Pool<QeWorkItem>::kInvalid) ds.errors.record(ErrorKind::kQeInstancesFull);
        else qe.blocked.at(b) = QeWorkItem{state_hash, rec, depth};
    }

    // Published only after the record is complete: a walker that reaches the reference must not
    // find a half-written instance.
    __threadfence();
    const uint64_t key = qe_inst_key(state_hash, depth);
    if (qe.by_key.push(qe_inst_bucket(qe, key, (blockIdx.x * blockDim.x + threadIdx.x) &
                                                   (kQeInstShards - 1u)),
                       QeInstRef{key, rec}) == INVALID_ID)
        ds.errors.record(ErrorKind::kQeInstancesFull);
    return rec;
}

// The root instance of a class: every slot's edge came with the initial state, so no event
// produced any of them. Claims the class frame first, so the root's slots and the expansion
// captured from it are in the SAME labelling by construction -- the host does the same, and for
// the same reason.
__device__ inline void qe_seed_root_instance(const DeviceState& ds, QeView qe, StateId root,
                                             uint32_t work_slice) {
    if (!qe.enabled) return;
    const uint64_t h = ds.state_canonical_hash[root];
    const uint32_t nslots = ds.state_edge_slices[root].count;

    if (qe_register_frame(qe, h, root)) ds.errors.record(ErrorKind::kCanonicalMapFull);

    // The frame above is registered whatever the caller records: event identity reads it. The
    // instance below is the root of the replay, and without it no descendant instance exists,
    // so this one guard removes the whole cascade.
    if (qe.multiplicity) {
        QeWork work = qe_work_for(ds, qe, work_slice);
        qe_seed_multiplicity(ds, qe, h, work);
    }
    if (!qe.replay) return;

    const uint32_t rec = qe_add_instance(ds, qe, h, 0u, kQeNoParent, 0u, root, nslots);
    if (rec == UINT32_MAX) return;
    qe_drive_instance(ds, qe, rec, h, 0u);
}

// Visit every instance recorded for `state_hash` at `depth`.
template <typename F>
__device__ inline void qe_for_each_instance(QeView qe, uint64_t state_hash, uint32_t depth,
                                            F&& f) {
    const uint64_t key = qe_inst_key(state_hash, depth);
    for (uint32_t s = 0; s < kQeInstShards; ++s) {
        if (!qe_inst_shard_first(key, s, qe.by_key.num_keys)) continue;
        qe.by_key.for_each(qe_inst_bucket(qe, key, s), [&](const QeInstRef& r) {
            if (r.key == key) f(qe.instances.at(r.record));
        });
    }
}

// The storage face hgcommon/quotient_replay_core.hpp drives. WHERE an instance's lineage, an
// applied list or a claim set lives is here; what an application DOES -- what it claims, what
// it identifies the event by, which causal and branchial relations follow -- is in the core,
// which is the body the host runs too.
__device__ __forceinline__ void qe_apply(const DeviceState& ds, QeView qe, const DeviceQcInstance& inst,
                                         const DeviceSlotMatch& m, uint64_t state_hash,
                                         uint32_t depth);

// Append one application to the task log and publish it. A full log is a capacity overflow,
// reported and grown by the retry ladder, as every other replay pool's.
__device__ inline void qe_task_append(const DeviceState& ds, QeView qe, uint64_t hash, uint32_t rec,
                                      uint32_t depth, uint32_t match) {
    if (!qe.tasks.append(QeTask{hash, rec, depth, match, 0u}))
        ds.errors.record(ErrorKind::kQeEventsFull);
}

// Instance side of the rendezvous: a task per match already captured for this class.
// Final-depth instances are recorded and never expanded.
__device__ inline void qe_drive_instance(const DeviceState& ds, QeView qe, uint32_t rec,
                                         uint64_t state_hash, uint32_t depth) {
    if (depth >= qe.max_steps) return;
    // Published before scanning; pairs with the fence on the match side so a concurrent
    // instance and match cannot both miss each other.
    __threadfence();
    qe.by_from.for_each(qe_bucket(state_hash, qe.by_from.num_keys), [&](const QeMatchRef& r) {
        if (r.from_hash == state_hash) qe_task_append(ds, qe, state_hash, rec, depth, r.record);
    });
}

// Match side of the rendezvous: a task per instance already standing at this class, at every
// depth it could stand at.
// Called by every lane of the capturing warp after lane 0 pushed the match to by_from. Lane 0
// fences after that push and the warp synchronizes before any lane scans, so each scan follows
// the publish and the fence, as the instance side's does. Each lane walks every 32nd
// (depth, shard) list.
__device__ inline void qe_drive_match(const DeviceState& ds, QeView qe, uint32_t match_rec,
                                      uint64_t from_hash) {
    if ((threadIdx.x & 31u) == 0) __threadfence();
    __syncwarp();
    const uint32_t units = qe.max_steps * kQeInstShards;
    for (uint32_t u = threadIdx.x & 31u; u < units; u += 32u) {
        const uint32_t d = u / kQeInstShards;
        const uint64_t key = qe_inst_key(from_hash, d);
        if (!qe_inst_shard_first(key, u % kQeInstShards, qe.by_key.num_keys)) continue;
        qe.by_key.for_each(qe_inst_bucket(qe, key, u % kQeInstShards), [&](const QeInstRef& r) {
            if (r.key == key) qe_task_append(ds, qe, from_hash, r.record, d, match_rec);
        });
    }
}

// The slice of the descent arena belonging to one driver. Out of range yields an empty stack,
// which pushes nothing and reports -- the same partial-work contract as any other capacity here.
__device__ inline QeWork qe_work_for(const DeviceState& ds, QeView qe, uint32_t slice) {
    QeWork w;
    if (qe.work_items == nullptr || slice >= qe.work_slices) {
        if (qe.multiplicity) ds.errors.record(ErrorKind::kScratchOverflow);
        return w;
    }
    w.items = qe.work_items + static_cast<size_t>(slice) * qe.work_cap;
    w.cap   = qe.work_cap;
    return w;
}

struct DeviceQrCtx {
    using Instance = DeviceQcInstance;
    using Match    = QeMatchView;
    // REFERENCES, not copies. DeviceState and QeView are large aggregates and this Ctx is
    // constructed once per application, so holding either by value would copy it that often.
    // The caller's copies outlive this object.
    const DeviceState& ds;
    QeView& qe;

    // COUNTED INTO LOCALS AND PUBLISHED ONCE, when this context is destroyed at the end of the
    // application it was made for. Incremented in place each is one global atomicAdd per
    // emission, from every resident thread onto a single address, and the L2 serialises those
    // one at a time -- the branchial count alone is 133,218,996 emissions against 970,584
    // applications on disc-l3a2g2r2 depth 3. An application emits far fewer than 2^32, so the
    // published totals are identical to counting in place.
    //
    // Its host twin is Hypergraph::QrCtx, which batches the same counts for the same reason and
    // publishes them from its own destructor.
    uint32_t causal_edges_seen = 0;
    uint32_t causal_pairs_seen = 0;
    uint32_t reduced_pairs_seen = 0;

    __device__ ~DeviceQrCtx() {
        if (causal_edges_seen) atomicAdd(qe.num_causal_edges, causal_edges_seen);
        if (causal_pairs_seen) atomicAdd(qe.num_causal_pairs, causal_pairs_seen);
        if (reduced_pairs_seen) atomicAdd(qe.num_reduced_pairs, reduced_pairs_seen);
    }

    struct ClaimChain {
        const DeviceState& ds;
        QeView qe;
        using Block = uint32_t;
        __device__ bool is_null(Block b) const { return b == kQeNoClaims; }
        __device__ uint32_t words(Block b) const { return qe.arr_words[b + 1u]; }
        __device__ Block next(Block b) const {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> n(qe.arr_words[b]);
            return n.load(cuda::memory_order_acquire);
        }
        __device__ Block install_next(Block b, uint32_t w) {
            const Block made = qe_new_claim_block(ds, qe, w);
            if (made == kQeNoClaims) return kQeNoClaims;
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> n(qe.arr_words[b]);
            uint32_t expected = kQeNoClaims;
            if (n.compare_exchange_strong(expected, made, cuda::memory_order_acq_rel,
                                          cuda::memory_order_acquire))
                return made;
            return expected;
        }
        __device__ bool set_bit(Block b, uint32_t bit) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> w(qe.arr_words[b + 2u + (bit >> 5)]);
            const uint32_t mask = 1u << (bit & 31u);
            return (w.fetch_or(mask, cuda::memory_order_acq_rel) & mask) == 0u;
        }
    };

    __device__ bool claim(const Instance& inst, const Match& m) {
        if (inst.claims != kQeNoClaims) {
            ClaimChain chain{ds, qe};
            const hgcommon::QrClaim c = hgcommon::qr_claim_chain(chain, inst.claims, m.local);
            if (c != hgcommon::QR_CLAIM_NO_ROOM) return c == hgcommon::QR_CLAIM_WON;
        }
        const auto r = qe.applied.insert_if_absent(hgcommon::qr_apply_key(inst.id, m.id), 1u);
        if (r.overflowed) ds.errors.record(ErrorKind::kQePairsFull);
        return r.inserted;
    }
    // One shared counter: every producer's id was taken before this one, so the id is above
    // them all.
    // Refused at ds.replay_id_limit (kReplayIdsExhausted). The pre-check keeps the counter from
    // passing the limit by more than the lanes that pass it together, which the limit's distance
    // below 2^32 covers (hgcommon::QR_ID_LIMIT).
    __device__ uint32_t mint_event(uint32_t /*above*/) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> nre(*qe.next_raw_event);
        if (nre.load(cuda::memory_order_relaxed) < ds.replay_id_limit) {
            const uint32_t id = nre.fetch_add(1u, cuda::memory_order_relaxed);
            if (id < ds.replay_id_limit) return id;
        }
        ds.errors.record(ErrorKind::kReplayIdsExhausted);
        return INVALID_ID;
    }
    // The event's content triple, from hgcommon rather than open-coded here. The open-coding
    // this replaces seeded FNV with the 64-bit basis missing its last digit, so every
    // reconstructed identity the device reported was a relabelling of the host's; routing the
    // call is what makes that unrepeatable rather than merely fixed.
    // An event past event_sig_capacity reports kQeEventsFull here, once; record_runsig and
    // record_kept skip it.
    __device__ void record_content(uint32_t ev, uint64_t from_class, uint64_t to_class,
                                   uint32_t rule) {
        if (ev >= qe.event_sig_capacity) {
            ds.errors.record(ErrorKind::kQeEventsFull);
        } else {
            qe.event_sig[ev] = hgcommon::qr_content_hash(from_class, to_class, rule);
            if (qe.event_from_class) {
                qe.event_from_class[ev] = from_class;
                qe.event_to_class[ev] = to_class;
                qe.event_rule[ev] = rule;
            }
        }
    }
    __device__ hgcommon::EventSignatureKeys keys() const { return qe.keys; }
    __device__ void record_runsig(uint32_t ev, const QeMatchView& m, uint64_t from_class,
                                  uint32_t out_step) {
        const QeRunsigClaim c = qe_claim_runsig(ds, qe, m, from_class, out_step);
        if (ev < qe.event_sig_capacity) qe.event_runsig[ev] = c.key;
        if (c.won) atomicAdd(qe.num_canon, 1u);
    }
    __device__ bool want_causal() const    { return ds.record_causal != 0; }
    __device__ bool want_branchial() const { return ds.record_branchial != 0; }
    __device__ uint32_t producer_at(const DeviceQcInstance& inst, uint32_t slot) const {
        return hgcommon::qr_producer_of(*this, &inst, slot);
    }
    // hgcommon::qr_producer_of's face, over instance records.
    __device__ bool lineage_root(const DeviceQcInstance* n) const { return n->parent == kQeNoParent; }
    __device__ uint32_t lineage_source(const DeviceQcInstance* n, uint32_t slot) const {
        const QeMatchView m(qe.matches.at(n->via), qe.arr_words);
        return slot < m.to_slots ? m.child_source(slot) : hgcommon::QR_SOURCE_NONE;
    }
    __device__ uint32_t lineage_event(const DeviceQcInstance* n) const { return n->event; }
    __device__ const DeviceQcInstance* lineage_parent(const DeviceQcInstance* n) const {
        return &qe.instances.at(n->parent);
    }
    __device__ void record_causal(uint32_t producer, uint32_t consumer, bool distinct_pair) {
        ++causal_edges_seen;
        // A repeat of the previous producer in this application's own list is not a new pair;
        // the caller has already decided that, and the map is left to answer the question it is
        // actually here for -- whether some OTHER application recorded this pair.
        if (!distinct_pair) return;
        const uint64_t pk = hgcommon::id_key(producer, consumer);
        const auto r = qe.causal_pairs.insert_if_absent(pk, 1u);
        if (r.overflowed) ds.errors.record(ErrorKind::kQePairsFull);
        if (!r.inserted) return;
        ++causal_pairs_seen;
    }
    // hgcommon::redundant_producers over the kept sets of earlier events. The search runs in local
    // arrays; when they fill, it runs again in this lane's slice of the expansion arena
    // (QeView::lane_reach, claimed on the lane's first overflow, the size of one block's
    // ds.tr_scratch slice), and past that the overflow is recorded, which keeps the pairs and
    // lets grow-and-retry run again with a larger arena.
    __device__ uint32_t redundant(const uint32_t* producers, uint32_t n) {
        if (n < 2 || qe.event_kept == nullptr) return 0;
        auto preds = [&](uint32_t x, auto&& f) {
            if (x >= qe.event_sig_capacity) return;
            const uint32_t* k = qe.event_kept + static_cast<size_t>(kQeKeptStride) * x;
            const uint32_t cnt = __ldcg(k);
            for (uint32_t i = 0; i < cnt && i < 3u; ++i) f(__ldcg(k + 1 + i));
            if (cnt > 3u) {
                const uint32_t off = __ldcg(k + 4);
                for (uint32_t i = 3; i < cnt; ++i) f(__ldcg(qe.arr_words + off + (i - 3u)));
            }
        };
        uint32_t stack[kQeReachStack];
        uint32_t table[kQeReachTable];
        hgcommon::BoundedReachCtx<decltype(preds)> local(preds, stack, kQeReachStack, table,
                                                         kQeReachTable);
        const uint32_t mask = hgcommon::redundant_producers(local, producers, n, true);
        if (!local.overflow) return mask;
        const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
        if (qe.lane_reach != nullptr && lane < qe.lane_reach_slots) {
            const uint32_t words = ds.tr_scratch_stack + ds.tr_scratch_visited;
            uint32_t off = qe.lane_reach[lane];
            if (off == 0) {
                const uint32_t got = qe_alloc_words(ds, qe, words);
                if (got != UINT32_MAX) { off = got + 1u; qe.lane_reach[lane] = off; }
            }
            if (off != 0) {
                uint32_t* slice = qe.arr_words + (off - 1u);
                hgcommon::BoundedReachCtx<decltype(preds)> wide(preds, slice, ds.tr_scratch_stack,
                                                                slice + ds.tr_scratch_stack,
                                                                ds.tr_scratch_visited);
                const uint32_t wmask = hgcommon::redundant_producers(wide, producers, n, true);
                if (!wide.overflow) return wmask;
            }
        }
        ds.errors.record(ErrorKind::kTrScratchOverflow);
        return mask;
    }
    // The count is written last and the fence publishes the record before the descent, whose
    // instance a later event's search reaches this one through.
    __device__ void record_kept(uint32_t ev, const uint32_t* kept, uint32_t nkept) {
        if (qe.event_kept == nullptr || ev >= qe.event_sig_capacity || nkept == 0) return;
        uint32_t* k = qe.event_kept + static_cast<size_t>(kQeKeptStride) * ev;
        for (uint32_t i = 0; i < nkept && i < 3u; ++i) k[1 + i] = kept[i];
        uint32_t stored = nkept < 3u ? nkept : 3u;
        if (nkept > 3u) {
            const uint32_t off = qe_alloc_words(ds, qe, nkept - 3u);
            if (off != UINT32_MAX) {
                for (uint32_t i = 3; i < nkept; ++i) qe.arr_words[off + (i - 3u)] = kept[i];
                k[4] = off;
                stored = nkept;
            }
        }
        k[0] = stored;
        __threadfence();
        reduced_pairs_seen += stored;
    }
    __device__ void publish_applied(const DeviceQcInstance& inst, const QeMatchView& m,
                                    uint32_t ev) {
        const uint32_t bucket = qe_bucket(hgcommon::id_key(inst.id), qe.inst_applied.num_keys);
        if (qe.inst_applied.push(bucket, QeAppliedMatch{inst.id, m.id, ev, m.num_consumed,
                                                        static_cast<uint32_t>(m.w - qe.arr_words)})
            == INVALID_ID)
            ds.errors.record(ErrorKind::kQeEventsFull);
    }
    // __forceinline__ so its depot merges into qr_apply's frame rather than taking one of its
    // own (tools/dev/ptx_frame_sizes.py measured 1104 bytes as its own frame).
    __device__ __forceinline__ void descend(const QeMatchView& m, uint32_t depth, uint32_t ev,
                                            const DeviceQcInstance& parent) {
        const uint32_t prec = static_cast<uint32_t>(&parent - qe.instances.data);
        const uint32_t mrec = static_cast<uint32_t>(m.src - qe.matches.data);
        const uint32_t rec =
            qe_add_instance(ds, qe, m.to_hash, depth + 1u, prec, mrec, ev, m.to_slots);
        if (rec == UINT32_MAX) return;
        // The child's applications are tasks, which any warp's lanes run.
        qe_drive_instance(ds, qe, rec, m.to_hash, depth + 1u);
    }
};

__device__ __forceinline__ void qe_apply(const DeviceState& ds, QeView qe, const DeviceQcInstance& inst,
                                         const DeviceSlotMatch& m, uint64_t state_hash,
                                         uint32_t depth) {
    if (!qe.enabled || depth >= qe.max_steps) return;
    DeviceQrCtx c{ds, qe};
    hgcommon::qr_apply(c, inst, QeMatchView(m, qe.arr_words), state_hash, depth);
}

// The branchial count of a run, after it: sum over the points below qe.max_steps of
// W(c, d) * B(c) (hgcommon::qm_branchial_add) into qe.qm_counts[1], with W the multiplicity when
// `multiplicity`, else the replay's instance count of (c, d). The host twin is
// Hypergraph::num_reconstructed_branchial. One kernel on the default stream.
void qe_count_branchial(const DeviceState& ds, const QeView& qe, bool multiplicity);

// Host-side owner of the capture's device structures, so a run's records are one body of
// state whether the host seeding or the device loop wrote them. Token-sized when the route is
// off, and cleared between runs rather than rebuilt: the pools total tens of MB of cudaMalloc
// that an interactive caller would otherwise pay every evolve.
class QeState {
public:
    QeState(bool on, const QeEntries& entries);
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
                 canon_events, causal_pairs, causal_edges;
        // The kept producers the replay stored: the size of the reduced causal relation.
        uint32_t reduced_pairs;
        // The multiplicity counts (QeView::qm_counts), read in a second transfer and only for a
        // run that counted multiplicities or branchial pairs; zero otherwise. qm_branchial is
        // count_branchial's sum.
        uint64_t qm_raw_events, qm_branchial;
    };
    Counters counters_host(bool multiplicity) const;
    // For a caller that reads these in a batch with others: the counter block (counter_words()
    // words), the two multiplicity counts, the capture pool's counter, and the parse of host
    // copies of the first two. counters_host is the reads followed by counters_from.
    const uint32_t* counters_device() const { return counters_; }
    static constexpr uint32_t counter_words() { return kNumCounters; }
    const unsigned long long* qm_counts_device() const { return qm_words_ + 2ull * qm_capacity_; }
    const uint32_t* num_matches_device() const { return matches_.view().counter; }
    static Counters counters_from(const uint32_t* v, const unsigned long long* q);

    uint32_t num_matches_host();

    // The multiplicity count's (class, depth, m) points with m > 0, and each class's captured
    // matches per rule as (class, rule, count). Read after the run.
    void class_multiplicities_host(std::vector<ClassMultiplicity>& points,
                                   std::vector<ClassRuleMatches>& matches);

    // Raw events the replay minted: one per (instance, match) application. The host's
    // qc_next_raw_event_, and the number a quotient run reports as its raw event count.
    uint32_t num_raw_events_host();
    // The run's raw event id limit (EngineConfig::replay_id_limit): the counter passes it by the
    // refused attempts, and num_raw_events_host reports the ids issued.
    void set_id_limit(uint32_t limit) { id_limit_ = limit; }

    // The reconstructed causal relation: distinct (producer, consumer) pairs, and the
    // consumed-edge occurrences behind them. The host's num_reconstructed_causal_pairs(false)
    // and num_reconstructed_causal_edges.
    uint32_t num_causal_pairs_host();
    uint32_t num_causal_edges_host();

    // Pairs the online reduction kept: the TR view of the same relation. The host's
    // num_reconstructed_causal_pairs(true).
    uint32_t num_reduced_pairs_host();

    // The reconstructed relations as pairs of CONTENT TRIPLES. A count says two engines
    // disagree; a pair set says which pair is missing, which a count cannot. `raw_events` is
    // counters_host().raw_events, which the caller has already read.
    void reconstructed_pairs_host(std::vector<std::pair<uint64_t, uint64_t>>& causal,
                                  std::vector<std::pair<uint64_t, uint64_t>>& causal_reduced,
                                  std::vector<std::pair<uint64_t, uint64_t>>& branchial,
                                  bool want_branchial,
                                  uint32_t raw_events,
                                  std::vector<uint64_t>* event_signature,
                                  std::vector<std::pair<uint32_t, uint32_t>>* causal_raw = nullptr,
                                  std::vector<std::pair<uint32_t, uint32_t>>* causal_raw_reduced = nullptr,
                                  std::vector<std::pair<uint32_t, uint32_t>>* branchial_raw = nullptr);
    // The run identity of each minted application, the `event_signature` of
    // reconstructed_pairs_host, for a run that reads no relation.
    void event_signature_host(std::vector<uint64_t>& event_signature, uint32_t raw_events);

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
    void ensure_work(uint32_t slices, uint32_t max_steps, uint32_t scale);
    // The per-lane reachability-slice table (QeView::lane_reach), one entry per lane of the grid.
    void ensure_lanes(uint32_t lanes);

    // The per-event input class, output class and rule arrays, allocated on first use and kept
    // across runs like the multiplicity queues.
    void ensure_event_content();
    // The first raw-event-count entries of those arrays.
    void reconstructed_event_content_host(std::vector<uint64_t>& from_class,
                                          std::vector<uint64_t>& to_class,
                                          std::vector<uint32_t>& rule);

    // The reconstruction's genesis pairs (docs/SPEC.md §5.2) as (initial state, raw event): each
    // application is the event of the child instance it created, and is paired with its root
    // instance's initial state when hgcommon::qr_genesis_paired holds, under the transitive
    // reduction when `reduced`. The host's Hypergraph::reconstructed_genesis_pairs.
    void reconstructed_genesis_pairs_host(bool reduced,
                                          std::vector<std::pair<uint32_t, uint32_t>>& out);

    QeView view(uint32_t max_steps, EventSignatureKeys keys,
                bool replay, bool multiplicity, bool event_content);

private:

    static uint32_t read_counter(const uint32_t* p, const char* what);

    Pool<DeviceSlotMatch>     matches_;
    LockFreeList<QeMatchRef>  by_from_;
    Pool<DeviceQcInstance>    instances_;
    Pool<QeWorkItem>          blocked_;
    Pool<QeTask>              tasks_;     // the replay's task log (QeView::tasks)
    LockFreeList<QeInstRef>   by_key_;
    DedupMap                  rep_;
    DedupMap                  applied_;
    ConcurrentMap<uint64_t, uint64_t> canon_seen_;
    DedupMap                  causal_pairs_;
    DedupMap                  qm_points_;
    DedupMap                  qm_consumed_;
    DedupMap                  qm_overlaps_;
    // qm_capacity_ masses, then qm_capacity_ consumed cells, then the two qm_counts.
    unsigned long long*       qm_words_  = nullptr;
    uint32_t*                 qm_queued_ = nullptr;
    unsigned long long*       qm_point_class_ = nullptr;
    uint32_t*                 qm_point_depth_ = nullptr;
    uint32_t                  qm_capacity_ = 0;
    // False until the first clear(), which covers the per-event arrays in full; later clears
    // cover the prefix below the previous run's raw event count.
    bool                      cleared_once_ = false;
    LockFreeList<QeAppliedMatch> inst_applied_;
    uint32_t*                 inst_next_id_ = nullptr;
    uint32_t*                 next_raw_event_ = nullptr;
    uint32_t                  id_limit_       = hgcommon::QR_ID_LIMIT;
    uint32_t*                 align_moved_    = nullptr;
    uint32_t*                 align_fail_     = nullptr;
    uint32_t*                 num_canon_        = nullptr;
    uint32_t*                 num_causal_pairs_ = nullptr;
    uint32_t*                 num_causal_edges_ = nullptr;
    unsigned long long*       class_pairs_      = nullptr;   // class_nmatch_cap_ entries
    uint64_t*                 event_sig_        = nullptr;
    uint64_t*                 event_runsig_     = nullptr;
    uint32_t*                 event_kept_       = nullptr;   // kQeKeptStride words per raw event
    uint32_t*                 num_reduced_pairs_ = nullptr;
    uint32_t*                 class_nmatch_ = nullptr;   // QeView::class_nmatch
    uint32_t                  class_nmatch_cap_ = 0;
    uint32_t                  event_sig_capacity_ = 0;
    uint64_t*                 event_from_class_ = nullptr;
    uint64_t*                 event_to_class_   = nullptr;
    uint32_t*                 event_rule_       = nullptr;
    FrameMap                  frame_;
    uint32_t*                 arr_ = nullptr;
    // The scalars above and below live in ONE allocation; these pointers index into it,
    // so counters_host() reads them all in a single transfer.
    // Slot 13 is the task log's consume cursor, slot 14 the tasks run.
    static constexpr uint32_t kNumCounters = 15;
    uint32_t*                 counters_ = nullptr;
    uint32_t*                 cursor_ = nullptr;
    uint32_t*                 next_id_ = nullptr;
    uint32_t                  arr_cap_ = 0;
    // The multiplicity queues: work_slices_ contiguous slices of work_cap_ items.
    QeWorkItem*               work_items_  = nullptr;
    uint32_t                  work_cap_    = 0;
    uint32_t                  work_slices_ = 0;
    uint32_t*                 lane_reach_       = nullptr;
    uint32_t                  lane_reach_slots_ = 0;
    bool                      on_ = false;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE