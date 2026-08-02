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

struct QeView {
    typename Pool<DeviceSlotMatch>::DeviceView          matches;
    typename LockFreeList<QeMatchRef>::DeviceView       by_from;   // bucket(from_hash)

    // canonical hash -> (StateId + 1) of the state that defines this class's expansion AND its
    // frame. +1 because the map reserves 0 as its EMPTY sentinel, so a raw key of StateId 0
    // could never be stored -- the same offset, for the same reason, as the None-mode dedup key.
    DedupMap::DeviceView frame;

    // Bump arena for the matches' slot arrays.
    uint32_t* arr_words;
    uint32_t* arr_cursor;      // device atomic
    uint32_t  arr_capacity;

    uint32_t* next_id;         // device atomic; dense match ids
    uint32_t  max_steps = 0;
    uint32_t  enabled   = 0;
};

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

    // Claim the class. The winner's labelling is the frame; everyone else drops out here, so a
    // record's slots are always in one labelling and never mix two.
    const uint32_t claim = static_cast<uint32_t>(parent) + 1u;
    if (qe.frame.insert_if_absent(from, claim).value != claim) return;

    const DeviceEvent& ev = ds.event_pool.at(event);
    const uint32_t nc = ev.num_consumed, np = ev.num_produced;

    uint32_t consumed[kMaxPatternEdges];
    uint32_t produced[kMaxPatternEdges];
    for (uint32_t i = 0; i < nc; ++i) {
        consumed[i] = qe_slot_of(ds, parent, ev.consumed_edges[i]);
        if (consumed[i] == UINT32_MAX) return;   // no frame slot: drop rather than corrupt
    }
    for (uint32_t i = 0; i < np; ++i) {
        produced[i] = qe_slot_of(ds, child, ev.produced_edges[i]);
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
            const uint32_t ps = qe_slot_of(ds, parent, oe);
            const uint32_t cs = qe_slot_of(ds, child, oe);
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

// Host-side owner of the capture's device structures, so a run's records are one body of
// state whether the host seeding or the device loop wrote them. Token-sized when the route is
// off, and cleared between runs rather than rebuilt, for the same reason QcState is: the pools
// total tens of MB of cudaMalloc that an interactive caller would otherwise pay every evolve.
class QeState {
public:
    QeState(bool on, uint32_t max_events)
        : matches_(on ? max_events : 1u),
          by_from_(on ? (1u << 16) : 1u, on ? max_events : 1u),
          frame_(on ? max_events * 2u : 8u),
          arr_cap_(on ? max_events * 16u : 1u),
          on_(on) {
        check(cudaMalloc(&arr_, sizeof(uint32_t) * arr_cap_), "QeState arr alloc");
        check(cudaMalloc(&cursor_, sizeof(uint32_t)), "QeState cursor alloc");
        check(cudaMalloc(&next_id_, sizeof(uint32_t)), "QeState next_id alloc");
        clear();
    }
    ~QeState() {
        if (arr_)     cudaFree(arr_);
        if (cursor_)  cudaFree(cursor_);
        if (next_id_) cudaFree(next_id_);
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
        check(cudaMemset(cursor_, 0, sizeof(uint32_t)), "QeState cursor clear");
        check(cudaMemset(next_id_, 0, sizeof(uint32_t)), "QeState next_id clear");
    }

    QeView view(uint32_t max_steps) {
        QeView q{};
        q.matches      = matches_.view();
        q.by_from      = by_from_.view();
        q.frame        = frame_.view();
        q.arr_words    = arr_;
        q.arr_cursor   = cursor_;
        q.arr_capacity = arr_cap_;
        q.next_id      = next_id_;
        q.max_steps    = max_steps;
        q.enabled      = on_ ? 1u : 0u;
        return q;
    }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::QeState ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    Pool<DeviceSlotMatch>     matches_;
    LockFreeList<QeMatchRef>  by_from_;
    DedupMap                  frame_;
    uint32_t*                 arr_ = nullptr;
    uint32_t*                 cursor_ = nullptr;
    uint32_t*                 next_id_ = nullptr;
    uint32_t                  arr_cap_ = 0;
    bool                      on_ = false;
};

}  // namespace hg_gpu
