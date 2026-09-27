#pragma once
#include "hgcommon/namespace.hpp"
// THE QUOTIENT REACH, one body for host and device.
//
// The quotient route explores CANONICAL states. The raw relations are reconstructed by the
// replay (quotient_replay_core.hpp); this marks which (class, depth) points the canonical
// transitions reach, which a continuation reads to find the points its old depth bound left
// unexpanded. Two mutually recursive steps:
//
//   reach(state, depth)               mark a (state, depth) point live once, then drive every
//                                     transition out of that state at that depth
//   process(transition, depth)        reach the transition's target at depth + 1
//
// Reach and transition registration are a RENDEZVOUS: each side publishes its own write and
// then scans for the other's, with a sequentially consistent `fence()` between. Without the
// fence on both sides a thread reaching (state, depth) and a thread registering a transition
// out of that state can each read the other as absent, and the pair is processed by neither.
//
// A Ctx must supply:
//
//   using Transition = ...;                     the engine's canonical-transition record
//   uint32_t max_steps() const;                 points are reached at depths 0..max_steps
//   bool enter(uint32_t depth);                 false to stop the cascade at this depth
//   bool mark_reached(uint64_t rkey, uint64_t state_hash, uint32_t depth);
//                                               insert-if-absent on the reached set; true when
//                                               THIS call was the one that inserted
//   void defer_reach(uint64_t state_hash, uint32_t depth);   reach, now or from a worklist
//   template <class F> void for_each_transition_from(uint64_t hash, F&& f);  f(const Transition&)
//   void fence();                               sequentially consistent, engine-scoped
//
// A Transition must supply to_hash.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// The (class, depth) keys. Shared because host and device index one set each: the replay's
// instances and multiplicity points by qc_key(state, depth, 0), the reached set by qc_rkey.
// Two spellings of a key are two key spaces.
HG_HD inline uint64_t qc_key(uint64_t state_hash, uint32_t depth, uint32_t orbit) {
    uint64_t h = FNV_OFFSET;
    h ^= state_hash; h *= FNV_PRIME;
    h ^= (static_cast<uint64_t>(depth) << 32) | orbit; h *= FNV_PRIME;
    return h;
}

// Nonzero, because the reached set's map reserves 0 as its EMPTY sentinel: a key of 0 is
// silently never stored, and a (state, depth) point that cannot be marked is re-expanded
// forever.
HG_HD inline uint64_t qc_rkey(uint64_t state_hash, uint32_t depth) {
    uint64_t h = FNV_OFFSET;
    h ^= state_hash; h *= FNV_PRIME;
    h ^= depth; h *= FNV_PRIME;
    return h ? h : 1;
}

// A canonical transition's dedup key: the (source class, target class) pair, which is all the
// reach marks read. One body, because host and device index one set of transitions. Never 0 or
// all-ones, because the seen set's map reserves both as sentinels.
HG_HD inline uint64_t qc_transition_key(uint64_t from, uint64_t to) {
    return avoid_reserved_keys(fnv_hash(fnv_hash(FNV_OFFSET, from), to));
}

// The edges an event carries across unchanged -- every edge of its output state it did not
// produce -- as index pairs into the input and output states' edge arrays, which both list
// edges in ascending id, so one merge walk pairs them. An output edge the event did not produce
// is an input edge by construction; the walk skips one that is not. ONE BODY on both engines:
// the host's transition record and capture read orbit and slot through the indices, the device
// reads its per-state orbit array through them, and the transition signature both compute
// over the survivors is one function of the same pairs in the same order.
template <class F>
HG_HD inline void qc_for_each_survivor(const EdgeId* in_edges, uint32_t in_n,
                                       const EdgeId* out_edges, uint32_t out_n,
                                       const EdgeId* produced, uint32_t num_produced, F&& f) {
    for (uint32_t i = 0, j = 0; i < out_n; ++i) {
        const EdgeId oe = out_edges[i];
        bool produced_here = false;
        for (uint32_t k = 0; k < num_produced; ++k)
            if (produced[k] == oe) { produced_here = true; break; }
        if (produced_here) continue;
        while (j < in_n && in_edges[j] < oe) ++j;
        if (j < in_n && in_edges[j] == oe) f(j, i);
    }
}

template <class Ctx>
HG_HD void qc_reach(Ctx& c, uint64_t state_hash, uint32_t depth);

template <class Ctx>
HG_HD void qc_process_transition(Ctx& c, const typename Ctx::Transition& t, uint32_t depth) {
    if (depth + 1 > c.max_steps()) return;
    // DEFERRED, NOT CALLED: the edge that advances DEPTH goes through the Ctx, which lets a
    // device carry the depth in a worklist instead of on a per-thread stack. A host Ctx calls
    // straight through.
    c.defer_reach(t.to_hash, depth + 1);
}

template <class Ctx>
HG_HD void qc_reach(Ctx& c, uint64_t state_hash, uint32_t depth) {
    if (depth > c.max_steps()) return;
    if (!c.enter(depth)) return;
    if (!c.mark_reached(qc_rkey(state_hash, depth), state_hash, depth)) return;
    // Publish the mark before scanning; pairs with the fence on the registration side.
    c.fence();
    c.for_each_transition_from(state_hash, [&](const typename Ctx::Transition& t) {
        qc_process_transition(c, t, depth);
    });
}

}  // namespace common
}  // namespace HG_NAMESPACE
