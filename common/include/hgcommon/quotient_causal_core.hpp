#pragma once
// THE QUOTIENT-CAUSAL DP, one body for host and device.
//
// The quotient route explores CANONICAL states and reconstructs the raw causal relation from
// the class skeleton rather than from the expanded multiway graph. The reconstruction is a
// fixpoint over three mutually recursive steps:
//
//   reach(state, depth)               mark a (state, depth) point live once, then drive every
//                                     transition out of that state at that depth
//   process(transition, from, depth)  reach the child, register the transition's produced
//                                     orbits as produced BY it, then rendezvous with the
//                                     producers already standing at the consumed and surviving
//                                     orbits of the parent
//   add_producer(state, depth, orbit, producer)
//                                     record that `producer` produced the edge in `orbit` at
//                                     (state, depth), then rendezvous with the transitions
//                                     already registered out of that state
//
// Every step is a RENDEZVOUS: the DP is driven concurrently by whatever thread happens to
// register a transition or land a producer, in whatever order, so each side publishes its own
// write and then scans for the other's. Both scans must be sequenced, which is what the
// `fence()` calls are -- without one on BOTH sides a thread reaching (state, depth) and a
// thread registering a transition out of that state can each read the other as absent, and the
// (transition, depth) pair is processed by neither.
//
// WHAT THE CTX SUPPLIES is storage and nothing else. The two engines differ in how a producer
// set is held (a lock-free list per key against a bucketed node pool), how transitions are
// enumerated, whether a recursion bound exists, and how a causal edge is recorded -- and in
// none of the decisions above. A second body for the decisions is what this exists to prevent:
// the identification a reconstructed event gets is not a performance property, and two copies
// agreeing today agree until one is edited.
//
// A Ctx must supply:
//
//   using Transition = ...;                     the engine's canonical-transition record
//   uint32_t max_steps() const;                 the DP runs depths 0..max_steps-1, producing
//                                               into max_steps but never reading it
//   bool enter(uint32_t depth);                 false to stop the cascade at this depth; the
//                                               Ctx records why. A host with a heap-sized
//                                               stack always returns true.
//   bool mark_reached(uint64_t rkey, uint64_t state_hash, uint32_t depth);
//                                               insert-if-absent on the reached set; true when
//                                               THIS call was the one that inserted
//   bool mark_producer_seen(uint64_t seen_key); same, on the (key, producer) seen set
//   void push_producer(uint64_t key, uint32_t producer);
//   template <class F> void for_each_producer(uint64_t key, F&& f);   f(producer)
//   template <class F> void for_each_transition_from(uint64_t hash, F&& f);  f(const Transition&)
//   void emit(uint32_t producer, uint32_t consumer);   record a causal edge
//   void fence();                               sequentially consistent, engine-scoped
//
// A Transition must supply: to_hash, canon_event, num_consumed, num_produced, num_survivors,
// and consumed(i) / produced(i) / surv_from(i) / surv_to(i).

#include <cstdint>

#include "hgcommon/core.hpp"

namespace hgcommon {

// The DP's three key spaces. Shared because host and device index ONE conceptual set each:
// a producer set keyed by (state, depth, orbit), a reached set keyed by (state, depth), and a
// seen set keyed by (producer set key, producer). Two spellings of a key are two key spaces.
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

// The (producer-set key, producer) pair, on the same terms.
HG_HD inline uint64_t qc_seen_key(uint64_t key, uint32_t producer) {
    uint64_t k = key ^ (static_cast<uint64_t>(producer) + 0x9e3779b97f4a7c15ULL);
    k *= FNV_PRIME;
    return (k == 0 || k == ~uint64_t(0)) ? 1 : k;
}

template <class Ctx>
HG_HD void qc_add_producer(Ctx& c, uint64_t state_hash, uint32_t depth, uint32_t orbit,
                           uint32_t producer);
template <class Ctx>
HG_HD void qc_reach(Ctx& c, uint64_t state_hash, uint32_t depth);

template <class Ctx>
HG_HD void qc_process_transition(Ctx& c, const typename Ctx::Transition& t,
                                 uint64_t from_hash, uint32_t depth) {
    if (depth + 1 > c.max_steps()) return;
    qc_reach(c, t.to_hash, depth + 1);
    // The produced edges are produced by THIS canonical event, at the child depth.
    for (uint32_t i = 0; i < t.num_produced; ++i)
        qc_add_producer(c, t.to_hash, depth + 1, t.produced(i), t.canon_event);
    // Rendezvous with producers already standing at (from, depth): publish the reach and the
    // produces above before scanning for them.
    c.fence();
    for (uint32_t i = 0; i < t.num_consumed; ++i) {
        const uint64_t k = qc_key(from_hash, depth, t.consumed(i));
        c.for_each_producer(k, [&](uint32_t p) { c.emit(p, t.canon_event); });
    }
    for (uint32_t i = 0; i < t.num_survivors; ++i) {
        const uint64_t k = qc_key(from_hash, depth, t.surv_from(i));
        const uint32_t to_orbit = t.surv_to(i);
        c.for_each_producer(k, [&](uint32_t p) {
            qc_add_producer(c, t.to_hash, depth + 1, to_orbit, p);
        });
    }
}

template <class Ctx>
HG_HD void qc_reach(Ctx& c, uint64_t state_hash, uint32_t depth) {
    if (depth > c.max_steps()) return;
    if (!c.enter(depth)) return;
    if (!c.mark_reached(qc_rkey(state_hash, depth), state_hash, depth)) return;
    // Publish the mark before scanning; pairs with the fence on the registration side.
    c.fence();
    c.for_each_transition_from(state_hash, [&](const typename Ctx::Transition& t) {
        qc_process_transition(c, t, state_hash, depth);
    });
}

template <class Ctx>
HG_HD void qc_add_producer(Ctx& c, uint64_t state_hash, uint32_t depth, uint32_t orbit,
                           uint32_t producer) {
    if (depth > c.max_steps()) return;
    if (!c.enter(depth)) return;
    const uint64_t key = qc_key(state_hash, depth, orbit);
    if (!c.mark_producer_seen(qc_seen_key(key, producer))) return;
    c.push_producer(key, producer);

    // A producer landing at (state, depth) witnesses that the point is reachable, so mark it
    // and drive its transitions once. Without this a producer arriving via the survivor cascade
    // leaves (state, depth) unreached and a transition registered later out of it is skipped.
    qc_reach(c, state_hash, depth);

    // The DP reads depths 0..max_steps-1. A producer landing at the final depth is stored and
    // dead, so the scan below would find nothing to do.
    if (depth >= c.max_steps()) return;

    // Rendezvous with transitions already registered out of this state: publish before scan.
    c.fence();
    c.for_each_transition_from(state_hash, [&](const typename Ctx::Transition& t) {
        for (uint32_t i = 0; i < t.num_consumed; ++i)
            if (t.consumed(i) == orbit) { c.emit(producer, t.canon_event); break; }
        for (uint32_t i = 0; i < t.num_survivors; ++i)
            if (t.surv_from(i) == orbit)
                qc_add_producer(c, t.to_hash, depth + 1, t.surv_to(i), producer);
    });
}

}  // namespace hgcommon
