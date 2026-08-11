#pragma once
#include "hgcommon/namespace.hpp"
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
#include "hgcommon/quotient_causal_core.hpp"   // the DP itself

#include <cuda/atomic>

namespace HG_NAMESPACE {
namespace gpu {

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

// The DP's key spaces come from hgcommon, so the device indexes the ones the host does.
using hgcommon::qc_key;
using hgcommon::qc_rkey;

__device__ __forceinline__ uint32_t qc_bucket(uint64_t key, uint32_t num_keys) {
    uint64_t h = key;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return static_cast<uint32_t>(h) & (num_keys - 1u);
}

// Orbit of `edge` within `sid`. UINT32_MAX when the edge is absent or no orbits were scattered.
__device__ __forceinline__ uint32_t qc_orbit_of(DeviceState ds, QcView,
                                                StateId sid, EdgeId edge) {
    const uint32_t i = state_edge_index(ds, sid, edge);
    return i == UINT32_MAX ? UINT32_MAX : ds.state_edge_orbit[i];
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

// The storage face hgcommon/quotient_causal_core.hpp drives. It supplies WHERE things are
// held and nothing else: the decisions -- when a point is entered, what a producer landing
// does, which rendezvous scan follows which publish -- are in the core, which is the same body
// the host runs.
//
// The transition record is read through accessors because the device packs its four orbit
// arrays into one contiguous word arena while the host holds four pointers. The DP walks both
// through the same four calls and knows neither layout.
struct DeviceQcTransition {
    const uint32_t* words;                  // consumed | produced | surv_from | surv_to
    uint64_t to_hash;
    EventId  canon_event;
    uint32_t num_consumed, num_produced, num_survivors;

    __device__ DeviceQcTransition(const DeviceCanonicalTransition& t, const uint32_t* arr)
        : words(arr + t.arr_offset), to_hash(t.to_hash), canon_event(t.canon_event),
          num_consumed(t.num_consumed), num_produced(t.num_produced),
          num_survivors(t.num_survivors) {}

    __device__ uint32_t consumed(uint32_t i)  const { return words[i]; }
    __device__ uint32_t produced(uint32_t i)  const { return words[num_consumed + i]; }
    __device__ uint32_t surv_from(uint32_t i) const {
        return words[num_consumed + num_produced + i];
    }
    __device__ uint32_t surv_to(uint32_t i) const {
        return words[num_consumed + num_produced + num_survivors + i];
    }
};

struct DeviceQcCtx {
    using Transition = DeviceQcTransition;
    DeviceState ds;
    QcView qc;

    __device__ uint32_t max_steps() const { return qc.max_steps; }
    // The device recurses on a per-thread stack the launch reserved, so past the bound the
    // cascade stops and records rather than faulting -- a fault takes the whole run's result,
    // not just the part past the bound.
    __device__ bool enter(uint32_t depth) {
        if (depth < qc.max_recursion_depth) return true;
        ds.errors.record(ErrorKind::kScratchOverflow);
        return false;
    }
    __device__ bool mark_reached(uint64_t rkey, uint64_t, uint32_t) {
        return qc.reached.insert_if_absent(rkey, 1u).inserted;
    }
    __device__ bool mark_producer_seen(uint64_t seen_key) {
        return qc.dsup_seen.insert_if_absent(seen_key, 1u).inserted;
    }
    __device__ void push_producer(uint64_t key, uint32_t producer) {
        if (qc.dsup.push(qc_bucket(key, qc.dsup.num_keys),
                         QcProducerNode{key, producer}) == INVALID_ID) {
            ds.errors.record(ErrorKind::kQcNodes);
        }
    }
    template <class F>
    __device__ void for_each_producer(uint64_t key, F&& f) {
        // The dsup list is bucketed, so a walker filters on the node's exact key.
        qc.dsup.for_each(qc_bucket(key, qc.dsup.num_keys), [&](const QcProducerNode& nd) {
            if (nd.key == key) f(nd.producer);
        });
    }
    template <class F>
    __device__ void for_each_transition_from(uint64_t hash, F&& f) {
        qc.trans_from.for_each(qc_bucket(hash, qc.trans_from.num_keys),
                               [&](const QcTransitionRef& ref) {
            if (ref.from_hash != hash) return;
            f(DeviceQcTransition(qc.transitions.at(ref.record), qc.arr_words));
        });
    }
    __device__ void emit(uint32_t producer, uint32_t consumer) {
        qc_emit(ds, producer, consumer);
    }
    __device__ void fence() {
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    }
};

__device__ inline void qc_add_producer(DeviceState ds, QcView qc, uint64_t state_hash,
                                       uint32_t depth, uint32_t orbit, EventId producer) {
    DeviceQcCtx c{ds, qc};
    hgcommon::qc_add_producer(c, state_hash, depth, orbit, producer);
}

__device__ inline void qc_reach(DeviceState ds, QcView qc, uint64_t state_hash,
                                uint32_t depth) {
    DeviceQcCtx c{ds, qc};
    hgcommon::qc_reach(c, state_hash, depth);
}

__device__ inline void qc_process_transition(DeviceState ds, QcView qc,
                                             const DeviceCanonicalTransition& t,
                                             uint64_t from_hash, uint32_t depth) {
    DeviceQcCtx c{ds, qc};
    hgcommon::qc_process_transition(c, DeviceQcTransition(t, qc.arr_words), from_hash, depth);
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

    // Dedup signature over (from, to, rule, consumed orbits, survivor orbit pairs). One body,
    // shared with the host, because it decides which raw events ARE the same transition.
    const uint64_t sig = hgcommon::qc_transition_sig(from, to, rule, consumed, nc, surv, ns);
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

}  // namespace gpu
}  // namespace HG_NAMESPACE