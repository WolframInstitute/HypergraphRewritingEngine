#pragma once
//
// Quotient causal reconstruction, device side: the depth-indexed producer-set DP over
// canonical transitions -- the device twin of Hypergraph::register_quotient_transition and
// the qc_* DP (hypergraph/src/hypergraph.cpp).
//
// Under quotient exploration the raw-edge causal rendezvous is schedule-dependent: isomorphic
// children race for the canonical slot, only the winner's raw instance expands, and causal
// edges attached to the winner's raw edges follow the race (tools/quotient_causal_probe_gpu
// pins it). The DP's keys are (state canonical hash, depth, edge ORBIT) -- no raw state or
// edge id appears in any of them -- so its emitted causal set is a function of the canonical
// evolution alone, identical for every race winner and every schedule.
//
// The propagation, exactly as on the host:
//
//   qc_add_producer(h, d, orbit, p)   record p as a producer of (h, d, orbit); rendezvous
//                                     with transitions already registered from h -- emit for
//                                     consumers of the orbit, carry p across survivor pairs.
//   qc_process_transition(t, h, d)    the transition side: produced orbits gain t's canonical
//                                     event as producer at (to, d+1); consumed orbits emit
//                                     from every producer already at (h, d, orbit); survivor
//                                     pairs carry every such producer to (to, d+1, orbit').
//   qc_reach(h, d)                    first time (h, d) is reachable: process every
//                                     transition registered from h at this depth.
//   qc_register_transition(...)       orbit-map one raw event into a deduplicated canonical
//                                     transition, publish it, and drive it at every depth its
//                                     source is already reached at.
//
// Each register/reach and producer/transition pair is a publish-then-scan rendezvous fenced
// seq_cst on both sides, mirroring the host's fences: the two sides write different locations
// then read the other's, and without the fences a concurrent pair can miss each other and the
// (transition, depth) work is processed by neither.
//
// Emissions go through the raw causal machinery (try_add_causal_edge) between CANONICAL event
// ids with shared-edge id 0; the (producer, consumer) pair dedup keeps one edge per pair,
// which is the DP's support-level contract. TR stays disabled under quotient (the host's
// guard_quotient_transitive_reduction, mirrored by the caller).

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/cuda_check.hpp"
#include "hg_gpu/rewrite.hpp"       // try_add_causal_edge
#include "hgcommon/core.hpp"        // isort_u64

#include <cuda/atomic>

namespace hg_gpu {

// One deduplicated canonical transition. The orbit arrays live in the qc word arena at
// arr_offset: consumed | produced | surv_from | surv_to, contiguously.
struct DeviceCanonicalTransition {
    uint64_t from_hash = 0;
    uint64_t to_hash   = 0;
    EventId  canon_event = INVALID_ID;
    uint32_t num_consumed = 0, num_produced = 0, num_survivors = 0;
    uint32_t arr_offset = 0;
};

// A producer registered under a qc_key. The dsup list is bucketed, so the node carries its
// exact key for the walkers to filter on.
struct QcProducerNode {
    uint64_t key;
    EventId  producer;
};

// A registered transition reference, bucketed by from_hash; the record carries the exact hash.
struct QcTransitionRef {
    uint64_t from_hash;
    uint32_t record;
};

struct QcView {
    typename Pool<DeviceCanonicalTransition>::DeviceView transitions;
    typename LockFreeList<QcTransitionRef>::DeviceView   trans_from;   // bucket(from_hash)
    DedupMap::DeviceView seen_transitions;

    // Bump arena for the transitions' orbit arrays.
    uint32_t* arr_words;
    uint32_t* arr_cursor;      // device atomic
    uint32_t  arr_capacity;

    typename LockFreeList<QcProducerNode>::DeviceView dsup;            // bucket(qc_key)
    DedupMap::DeviceView dsup_seen;
    DedupMap::DeviceView reached;

    uint32_t max_steps = 0;
    // How deep this DP may recurse before the per-thread stack runs out. qc_reach ->
    // qc_process_transition -> qc_add_producer -> qc_reach descends once per depth, exactly as
    // the replay's cycle does, so the two share one bound (EngineState::qe_max_recursion_depth).
    // Past it the cascade stops and records; without it the next frame faults and the fault
    // takes the whole run's result, not just the part past the bound.
    uint32_t max_recursion_depth = 0;
    uint32_t enabled   = 0;
};

// Host-side owner of the DP's device structures, so the host seeding and the device loop share one run's
// registrations and producer sets are a single body of state whichever loop drives them.
// Token-sized when the route is off. Owned by the ENGINE and cleared between runs rather than
// rebuilt: the maps and pools total tens of MB of cudaMalloc, which an interactive caller
// would otherwise pay on every evolve.
class QcState {
public:
    QcState(bool on, uint32_t max_events)
        : transitions_(on ? max_events : 1u),
          trans_from_(on ? (1u << 16) : 1u, on ? max_events : 1u),
          seen_(on ? max_events * 2u : 8u),
          dsup_(on ? (1u << 18) : 1u, on ? max_events * 8u : 8u),
          dsup_seen_(on ? max_events * 16u : 8u),
          reached_(on ? (1u << 20) : 8u),
          arr_cap_(on ? max_events * 16u : 1u),
          on_(on) {
        HG_CUDA_CHECK(cudaMalloc(&arr_, sizeof(uint32_t) * arr_cap_), "QcState arr alloc");
        HG_CUDA_CHECK(cudaMalloc(&cursor_, sizeof(uint32_t)), "QcState cursor alloc");
        clear();
    }
    ~QcState() {
        if (arr_)    cudaFree(arr_);
        if (cursor_) cudaFree(cursor_);
    }
    QcState(const QcState&)            = delete;
    QcState& operator=(const QcState&) = delete;

    bool enabled() const { return on_; }

    // Between runs: every map, list and record pool starts empty. The orbit-array words need
    // no wipe -- records reference them by offset and the cursor restarts at zero.
    void clear() {
        seen_.clear();
        dsup_seen_.clear();
        reached_.clear();
        trans_from_.clear();
        dsup_.clear();
        transitions_.reset();
        HG_CUDA_CHECK(cudaMemset(cursor_, 0, sizeof(uint32_t)), "QcState cursor clear");
    }

    QcView view(uint32_t max_steps, uint32_t max_recursion_depth) {
        QcView q{};
        q.transitions      = transitions_.view();
        q.trans_from       = trans_from_.view();
        q.seen_transitions = seen_.view();
        q.arr_words        = arr_;
        q.arr_cursor       = cursor_;
        q.arr_capacity     = arr_cap_;
        q.dsup             = dsup_.view();
        q.dsup_seen        = dsup_seen_.view();
        q.reached          = reached_.view();
        q.max_steps        = max_steps;
        q.max_recursion_depth = max_recursion_depth;
        q.enabled          = on_ ? 1u : 0u;
        return q;
    }

private:

    Pool<DeviceCanonicalTransition> transitions_;
    LockFreeList<QcTransitionRef>   trans_from_;
    DedupMap                        seen_;
    LockFreeList<QcProducerNode>    dsup_;
    DedupMap                        dsup_seen_;
    DedupMap                        reached_;
    uint32_t*                       arr_ = nullptr;
    uint32_t*                       cursor_ = nullptr;
    uint32_t                        arr_cap_ = 0;
    bool                            on_ = false;
};

__device__ __forceinline__ uint64_t qc_key(uint64_t state_hash, uint32_t depth,
                                           uint32_t orbit) {
    uint64_t h = 1469598103934665603ULL;
    h ^= state_hash; h *= 1099511628211ULL;
    h ^= (static_cast<uint64_t>(depth) << 32) | orbit; h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ uint64_t qc_rkey(uint64_t state_hash, uint32_t depth) {
    uint64_t h = 1469598103934665603ULL;
    h ^= state_hash; h *= 1099511628211ULL;
    h ^= depth; h *= 1099511628211ULL;
    return h ? h : 1;
}

__device__ __forceinline__ uint32_t qc_bucket(uint64_t key, uint32_t num_keys) {
    uint64_t h = key;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return static_cast<uint32_t>(h) & (num_keys - 1u);
}

// Orbit of `edge` within state `sid`: binary search the sorted CSR slice, read the parallel
// orbit array. UINT32_MAX when the edge is not in the state or no orbits were scattered.
__device__ __forceinline__ uint32_t qc_orbit_of(DeviceState ds, QcView qc,
                                                StateId sid, EdgeId edge) {
    const StateEdgeSlice sl = ds.state_edge_slices[sid];
    uint32_t lo = 0, hi = sl.count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (ds.state_edge_ids[sl.offset + mid] < edge) lo = mid + 1; else hi = mid;
    }
    if (lo >= sl.count || ds.state_edge_ids[sl.offset + lo] != edge) return UINT32_MAX;
    return ds.state_edge_orbit[sl.offset + lo];
}

__device__ __forceinline__ EventId qc_canonical_event(DeviceState ds, EventId e) {
    if (e == INVALID_ID) return INVALID_ID;
    const EventId c = ds.event_pool.at(e).canonical_id;
    return c == INVALID_ID ? e : c;
}

// The INIT sentinel (INVALID_ID producer) marks initial edges; a producer == consumer pair is
// a canonical self-loop and is kept, as in full capture.
__device__ __forceinline__ void qc_emit(DeviceState ds, EventId producer, EventId consumer) {
    if (producer == INVALID_ID || consumer == INVALID_ID) return;
    try_add_causal_edge(ds, producer, consumer, 0);
}

__device__ void qc_add_producer(DeviceState ds, QcView qc, uint64_t state_hash,
                                uint32_t depth, uint32_t orbit, EventId producer);
__device__ void qc_reach(DeviceState ds, QcView qc, uint64_t state_hash,
                         uint32_t depth);

__device__ inline void qc_process_transition(DeviceState ds, QcView qc,
                                             const DeviceCanonicalTransition& t,
                                             uint64_t from_hash, uint32_t depth) {
    if (depth + 1 > qc.max_steps) return;
    qc_reach(ds, qc, t.to_hash, depth + 1);
    const uint32_t* consumed  = qc.arr_words + t.arr_offset;
    const uint32_t* produced  = consumed + t.num_consumed;
    const uint32_t* surv_from = produced + t.num_produced;
    const uint32_t* surv_to   = surv_from + t.num_survivors;
    for (uint32_t i = 0; i < t.num_produced; ++i)
        qc_add_producer(ds, qc, t.to_hash, depth + 1, produced[i], t.canon_event);
    // Rendezvous with producers already present at (from, depth): publish (reach/produce
    // above) before this scan.
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    for (uint32_t i = 0; i < t.num_consumed; ++i) {
        const uint64_t k = qc_key(from_hash, depth, consumed[i]);
        qc.dsup.for_each(qc_bucket(k, qc.dsup.num_keys), [&](const QcProducerNode& nd) {
            if (nd.key == k) qc_emit(ds, nd.producer, t.canon_event);
        });
    }
    for (uint32_t i = 0; i < t.num_survivors; ++i) {
        const uint64_t k = qc_key(from_hash, depth, surv_from[i]);
        const uint32_t to_orbit = surv_to[i];
        qc.dsup.for_each(qc_bucket(k, qc.dsup.num_keys), [&](const QcProducerNode& nd) {
            if (nd.key == k)
                qc_add_producer(ds, qc, t.to_hash, depth + 1, to_orbit, nd.producer);
        });
    }
}

__device__ inline void qc_for_each_transition_from(DeviceState ds, QcView qc,
                                                   uint64_t from_hash, uint32_t depth) {
    qc.trans_from.for_each(qc_bucket(from_hash, qc.trans_from.num_keys),
                           [&](const QcTransitionRef& ref) {
        if (ref.from_hash != from_hash) return;
        qc_process_transition(ds, qc, qc.transitions.at(ref.record), from_hash, depth);
    });
}

__device__ inline void qc_reach(DeviceState ds, QcView qc, uint64_t state_hash,
                                uint32_t depth) {
    if (depth > qc.max_steps) return;
    if (depth >= qc.max_recursion_depth) { ds.errors.record(ErrorKind::kScratchOverflow); return; }
    if (!qc.reached.insert_if_absent(qc_rkey(state_hash, depth), 1u).inserted) return;
    // Publish (the insert above) before scanning; pairs with the fence in
    // qc_register_transition. Without seq_cst on BOTH sides a thread reaching (state, depth)
    // and a thread registering a transition out of that state can each read the other as
    // absent, and the (transition, depth) pair is processed by neither.
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    qc_for_each_transition_from(ds, qc, state_hash, depth);
}

__device__ inline void qc_add_producer(DeviceState ds, QcView qc, uint64_t state_hash,
                                       uint32_t depth, uint32_t orbit, EventId producer) {
    if (depth > qc.max_steps) return;
    if (depth >= qc.max_recursion_depth) { ds.errors.record(ErrorKind::kScratchOverflow); return; }
    const uint64_t key = qc_key(state_hash, depth, orbit);
    uint64_t seenk = key ^ (static_cast<uint64_t>(producer) + 0x9e3779b97f4a7c15ULL);
    seenk *= 1099511628211ULL; if (seenk == 0 || seenk == ~0ULL) seenk = 1;
    if (!qc.dsup_seen.insert_if_absent(seenk, 1u).inserted) return;
    if (qc.dsup.push(qc_bucket(key, qc.dsup.num_keys),
                     QcProducerNode{key, producer}) == INVALID_ID) {
        ds.errors.record(ErrorKind::kQcNodes);
    }

    // A producer landing at (state, depth) witnesses reachability, so mark it and process its
    // transitions once; a producer arriving via the survivor cascade would otherwise leave
    // (state, depth) unreached and a later consuming transition would be skipped.
    qc_reach(ds, qc, state_hash, depth);

    // Producers landing at the final depth are stored and dead: the DP processes depths
    // 0..steps-1, producing into depth steps but never reading it.
    if (depth >= qc.max_steps) return;

    // Rendezvous with transitions already known from this state: publish before scan.
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    qc.trans_from.for_each(qc_bucket(state_hash, qc.trans_from.num_keys),
                           [&](const QcTransitionRef& ref) {
        if (ref.from_hash != state_hash) return;
        const DeviceCanonicalTransition& t = qc.transitions.at(ref.record);
        const uint32_t* consumed  = qc.arr_words + t.arr_offset;
        const uint32_t* surv_from = consumed + t.num_consumed + t.num_produced;
        const uint32_t* surv_to   = surv_from + t.num_survivors;
        for (uint32_t i = 0; i < t.num_consumed; ++i)
            if (consumed[i] == orbit) { qc_emit(ds, producer, t.canon_event); break; }
        for (uint32_t i = 0; i < t.num_survivors; ++i)
            if (surv_from[i] == orbit)
                qc_add_producer(ds, qc, t.to_hash, depth + 1, surv_to[i], producer);
    });
}

// Survivor pairs a registration can hold in local scratch. A state with more surviving edges
// than this records kScratchOverflow and skips the transition: causal edges reachable only
// through it are then missing, which the warning reports rather than silently mis-attributes.
constexpr uint32_t kQcMaxSurvivors = 256;

// Orbit-map one raw event into a canonical transition, publish it once (deduplicated by the
// orbit signature), and drive it at every depth its source state is already reached at.
// `depth` is the PARENT state's depth (the event's step - 1).
__device__ inline void qc_register_transition(DeviceState ds, QcView qc,
                                              StateId parent, StateId child, EventId event,
                                              uint32_t rule, uint32_t depth) {
    const uint64_t from = ds.state_canonical_hash[parent];
    const uint64_t to   = ds.state_canonical_hash[child];
    const DeviceEvent& ev = ds.event_pool.at(event);

    uint32_t consumed[kMaxPatternEdges];
    uint32_t produced[kMaxPatternEdges];
    const uint32_t nc = ev.num_consumed, np = ev.num_produced;
    for (uint32_t i = 0; i < nc; ++i)
        consumed[i] = qc_orbit_of(ds, qc, parent, ev.consumed_edges[i]);
    for (uint32_t i = 0; i < np; ++i)
        produced[i] = qc_orbit_of(ds, qc, child, ev.produced_edges[i]);
    // Insertion sort; nc, np <= kMaxPatternEdges.
    for (uint32_t i = 1; i < nc; ++i) {
        const uint32_t v = consumed[i]; uint32_t j = i;
        while (j > 0 && consumed[j - 1] > v) { consumed[j] = consumed[j - 1]; --j; }
        consumed[j] = v;
    }
    for (uint32_t i = 1; i < np; ++i) {
        const uint32_t v = produced[i]; uint32_t j = i;
        while (j > 0 && produced[j - 1] > v) { produced[j] = produced[j - 1]; --j; }
        produced[j] = v;
    }

    // Survivors: child edges that are not freshly produced passed through from the parent
    // (the child's CSR is parent-minus-consumed plus produced by construction). Recorded as
    // (orbit in parent << 32 | orbit in child), sorted as one word.
    uint64_t surv[kQcMaxSurvivors];
    uint32_t ns = 0;
    {
        const StateEdgeSlice csl = ds.state_edge_slices[child];
        for (uint32_t k = 0; k < csl.count; ++k) {
            const EdgeId oe = ds.state_edge_ids[csl.offset + k];
            bool produced_here = false;
            for (uint32_t j = 0; j < np; ++j)
                if (ev.produced_edges[j] == oe) { produced_here = true; break; }
            if (produced_here) continue;
            const uint32_t po = qc_orbit_of(ds, qc, parent, oe);
            if (po == UINT32_MAX) continue;
            if (ns >= kQcMaxSurvivors) { ds.errors.record(ErrorKind::kScratchOverflow); return; }
            surv[ns++] = (static_cast<uint64_t>(po) << 32) | ds.state_edge_orbit[csl.offset + k];
        }
        hgcommon::isort_u64(surv, ns);
    }

    // Dedup signature over (from, to, rule, consumed orbits, survivor orbit pairs) -- the
    // host's key exactly.
    uint64_t sig = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) { sig ^= v; sig *= 1099511628211ULL; };
    mix(from); mix(to); mix(rule);
    for (uint32_t i = 0; i < nc; ++i) { mix(0x1111); mix(consumed[i]); }
    for (uint32_t i = 0; i < ns; ++i) {
        mix(0x2222); mix(surv[i] >> 32); mix(surv[i] & 0xFFFFFFFFu);
    }
    if (sig == 0 || sig == ~0ULL) sig = 1;
    if (!qc.seen_transitions.insert_if_absent(sig, 1u).inserted) return;

    // Copy the orbit arrays into the qc arena, then publish the record.
    const uint32_t need = nc + np + 2u * ns;
    uint32_t off = 0;
    if (need) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> cur(*qc.arr_cursor);
        off = cur.fetch_add(need, cuda::memory_order_relaxed);
        if (off + need > qc.arr_capacity) {
            ds.errors.record(ErrorKind::kQcNodes);
            return;
        }
        uint32_t* w = qc.arr_words + off;
        for (uint32_t i = 0; i < nc; ++i) *w++ = consumed[i];
        for (uint32_t i = 0; i < np; ++i) *w++ = produced[i];
        for (uint32_t i = 0; i < ns; ++i) *w++ = static_cast<uint32_t>(surv[i] >> 32);
        for (uint32_t i = 0; i < ns; ++i) *w++ = static_cast<uint32_t>(surv[i]);
    }

    const uint32_t rec = qc.transitions.claim();
    if (rec == Pool<DeviceCanonicalTransition>::kInvalid) {
        ds.errors.record(ErrorKind::kQcNodes);
        return;
    }
    DeviceCanonicalTransition& t = qc.transitions.at(rec);
    t.from_hash = from; t.to_hash = to;
    t.canon_event = qc_canonical_event(ds, event);
    t.num_consumed = nc; t.num_produced = np; t.num_survivors = ns;
    t.arr_offset = off;
    if (qc.trans_from.push(qc_bucket(from, qc.trans_from.num_keys),
                           QcTransitionRef{from, rec}) == INVALID_ID) {
        ds.errors.record(ErrorKind::kQcNodes);
        return;
    }

    // Drive the transition at every depth its source is already reached at; pairs with the
    // fence in qc_reach.
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    for (uint32_t d = 0; d <= qc.max_steps; ++d)
        if (qc.reached.lookup(qc_rkey(from, d)).found)
            qc_process_transition(ds, qc, qc.transitions.at(rec), from, d);
}

}  // namespace hg_gpu
