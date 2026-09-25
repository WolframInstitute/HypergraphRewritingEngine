#pragma once
#include "hgcommon/namespace.hpp"
// RAW COUNTS FROM CLASS MULTIPLICITIES, one body for host and device.
//
// Under quotient exploration a class c at depth d stands for m(c, d) raw states: the number of
// raw paths from the initial state that reach a member of c in d steps. Every raw state of c has
// the matches of c's expanded representative, in the class's slot frame, so the raw counts
// follow from m alone:
//
//   m(root, 0) = 1,    m(c', d + 1) = sum over matches j of c into c' of m(c, d),
//   raw events       = sum over c and d < steps of m(c, d) * M(c),
//   branchial pairs  = sum over c and d < steps of m(c, d) * B(c),
//
// where M(c) is the number of matches captured on c and B(c) the number of pairs of them whose
// consumed slots overlap. The per-instance replay (quotient_replay_core.hpp) reaches the same
// counts by materialising one instance per raw state; here the work is one pass per (class,
// depth, match) plus one per arrival of mass.
//
// ONLINE. m(c, d) is an accumulator that grows as parents pass mass on, and a match of c may be
// captured after mass has reached c. Each match j keeps, per depth, the mass it has passed on,
// consumed_j(d), and advances it by compare-and-swap to the accumulator's current value, passing
// on the difference. Each unit of mass therefore crosses each match exactly once, whichever side
// arrives first.
//
// ONE RUN PER POINT PER CASCADE. Mass arriving at (c, d) is added, and the point is queued unless
// it is already queued; a queued point's run passes everything that has arrived by then. The
// queue is drained in increasing depth, so within one cascade every arrival at depth d lands
// before any point at depth d runs. Passing on each arrival as it lands does the same arithmetic
// but runs a point once per path into it, which is the raw count again once the child's matches
// are captured: 0.92 s at depth 10 and more than 120 s at depth 12 on
// {{1,1},{1,1}} -> {{1,1},{1,1},{1,1}}. A cascade starts at the seed, at each captured match, and
// at each point a raised depth bound re-drives.
//
// TWO RENDEZVOUS, each publish-then-scan with a fence on both sides:
//   match j of c:        record b_j (which makes j ready), then pass every m(c, d) through j;
//   mass arriving:       add to m(c, d), then claim the point's queue flag;
//   a queued point runs: clear its flag, then read m(c, d) and scan c's ready matches.
// An arrival that finds the flag held leaves its mass to the run that holds it, which clears the
// flag before it reads.
//
// b_j is the number of matches linked into c's capture list before j whose consumed slots
// overlap j's. Each overlapping pair is counted by its later member only, so the b_j of c sum
// to B(c).
//
// SATURATION. Counts stop at QM_SATURATED, the largest value the int64 reply field holds, and
// the Ctx records that it happened. m grows exponentially with depth on rules with many matches
// per state (60x per step on {{1,1},{1,1}} -> {{1,1},{1,1},{1,1}}).
//
// A Ctx must supply:
//
//   using Match = ...;       id, to_hash and the fields qr_run_signature reads
//   uint32_t max_steps() const;
//   bool     ready(const Match&, uint64_t& b) const;      false until b_j is recorded
//   uint64_t mass(uint64_t class_hash, uint32_t depth) const;       0 when none has arrived
//   void     add_mass(uint64_t class_hash, uint32_t depth, uint64_t delta);   saturating
//   uint64_t consumed(const Match&, uint32_t depth);
//   bool     advance(const Match&, uint32_t depth, uint64_t& expected, uint64_t desired);
//                              compare-and-swap on consumed_j(depth); on failure `expected`
//                              holds the current value
//   void     count(uint64_t events, uint64_t branchial);   saturating adds
//   hgcommon::EventSignatureKeys keys() const;  uint32_t frame_step(uint64_t, uint32_t) const;
//   void     note_signature(uint64_t csig);      the run's distinct-event set
//   bool     claim_queued(uint64_t class_hash, uint32_t depth);   set the flag; true if it was clear
//   void     push(uint64_t class_hash, uint32_t depth);
//   bool     pop(uint64_t& class_hash, uint32_t& depth);   the shallowest queued point; clears
//                                                           its flag
//   template <class F> void for_each_match(uint64_t class_hash, F&& f);
//   void     fence();

#include <cstdint>

#include "hgcommon/core.hpp"
#include "hgcommon/event_core.hpp"
#include "hgcommon/quotient_replay_core.hpp"

namespace HG_NAMESPACE {
namespace common {

constexpr uint64_t QM_SATURATED = 0x7FFFFFFFFFFFFFFFull;
// The warning both engines attach to a saturated count, under the kind "CountSaturated".
constexpr const char* QM_SATURATED_MESSAGE =
    "a raw event or branchial count exceeds 2^63 - 1; NumEvents and NumBranchialEdges report "
    "2^63 - 1";

HG_HD inline uint64_t qm_sat_add(uint64_t a, uint64_t b) {
    return (a >= QM_SATURATED || b >= QM_SATURATED - a) ? QM_SATURATED : a + b;
}

HG_HD inline uint64_t qm_sat_mul(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    return a > QM_SATURATED / b ? QM_SATURATED : a * b;
}

// The cascade queue: a binary min-heap on `depth` over the caller's array. T has a `depth`
// field; `n` counts the elements including the one being pushed, or before the pop.
template <class T>
HG_HD void qm_heap_push(T* a, uint32_t n) {
    uint32_t i = n - 1;
    const T x = a[i];
    while (i > 0) {
        const uint32_t p = (i - 1) / 2;
        if (a[p].depth <= x.depth) break;
        a[i] = a[p];
        i = p;
    }
    a[i] = x;
}

// Removes and returns the shallowest element; n - 1 remain.
template <class T>
HG_HD T qm_heap_pop(T* a, uint32_t n) {
    const T top = a[0];
    const T x = a[--n];
    uint32_t i = 0;
    for (;;) {
        uint32_t c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && a[c + 1].depth < a[c].depth) ++c;
        if (x.depth <= a[c].depth) break;
        a[i] = a[c];
        i = c;
    }
    if (n) a[i] = x;
    return top;
}

template <class Ctx>
HG_HD void qm_credit(Ctx& c, uint64_t class_hash, uint32_t depth, uint64_t delta);

// Pass the mass of (state_hash, depth) that match `m` has not yet passed on.
template <class Ctx>
HG_HD void qm_pass(Ctx& c, const typename Ctx::Match& m, uint64_t state_hash, uint32_t depth) {
    uint64_t b = 0;
    if (!c.ready(m, b)) return;
    uint64_t done = c.consumed(m, depth);
    uint64_t have = 0;
    for (;;) {
        have = c.mass(state_hash, depth);
        if (have <= done) return;
        if (c.advance(m, depth, done, have)) break;
    }
    const uint64_t delta = have - done;
    c.count(delta, qm_sat_mul(delta, b));
    if (c.keys() != EVENT_SIG_NONE) c.note_signature(qr_run_signature(c, m, state_hash, depth));
    qm_credit(c, m.to_hash, depth + 1, delta);
}

// Mass arriving at (class_hash, depth). Mass at the depth bound is recorded and never passed on,
// as the replay records and never expands an instance there.
template <class Ctx>
HG_HD void qm_credit(Ctx& c, uint64_t class_hash, uint32_t depth, uint64_t delta) {
    if (depth > c.max_steps()) return;
    c.add_mass(class_hash, depth, delta);
    if (depth >= c.max_steps()) return;
    c.fence();
    if (c.claim_queued(class_hash, depth)) c.push(class_hash, depth);
}

// Run every queued point, shallowest first, including the points the runs queue.
template <class Ctx>
HG_HD void qm_drain(Ctx& c) {
    uint64_t h = 0;
    uint32_t d = 0;
    while (c.pop(h, d)) {
        c.fence();
        c.for_each_match(h, [&](const typename Ctx::Match& m) { qm_pass(c, m, h, d); });
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
