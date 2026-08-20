#pragma once
#include "hgcommon/namespace.hpp"
// THE PER-INSTANCE REPLAY, one body for host and device.
//
// The quotient route explores CANONICAL states. A canonical class is expanded ONCE, and its
// matches are recorded in SLOTS -- positions in the class's frame. Every raw state of that
// class is isomorphic to the frame, so one recorded match can be replayed against any
// INSTANCE of the class, and replaying it is what reconstructs the raw events the full
// expansion would have fired.
//
// This is that replay: one (instance, match) pair, applied once, producing one raw event and
// everything that follows from it.
//
//   claim         the pair, exactly once. Unlike the producer-set DP this is NOT idempotent:
//                 every application mints an event, so both sides of the rendezvous -- an
//                 instance arriving and replaying known matches, a match arriving and
//                 replaying known instances -- must not both fire it.
//   identify      the event twice over: the CONTENT triple (from class, to class, rule), which
//                 is isomorphism-invariant and is what a cross-run or cross-engine comparison
//                 is made on; and the RUN's signature under the caller's event-identity mode.
//   causal        one relation per consumed slot that carries a producer, fed in DESCENDING
//                 producer order so nearer producers enter the kept adjacency before farther
//                 ones are tested -- what makes the reduction tag exact rather than
//                 insertion-order dependent.
//   branchial     siblings expanding the SAME instance whose consumed slots overlap. Publish
//                 into the instance's applied list, THEN scan it: membership of that list is
//                 the proof the other application happened, and an application that never
//                 claims never publishes. Both sides can see each other, so the unordered pair
//                 is claimed and the winner counts it.
//   descend       the child instance -- survivors carry their producer across, produced slots
//                 take THIS event -- and drive it.
//
// Every one of those is a decision about the reconstructed relation, and none of them is a
// storage question. What differs between the engines is only where things are held: a
// scratch vector against a packed word arena, a ConcurrentMap against a DedupMap, an arena
// allocation against a bump offset.
//
// A Ctx must supply:
//
//   using Instance = ...;  using Match = ...;
//   bool     claim(uint64_t apply_key);        exactly-once on the (instance, match) pair
//   uint32_t mint_event();
//   void     record_content(uint32_t ev, uint64_t from_class, uint64_t to_class, uint32_t rule);
//   EventSignatureKeys keys() const;           EVENT_SIG_NONE to skip the run signature
//   uint32_t frame_step(uint64_t class_hash, uint32_t fallback) const;
//   void     record_runsig(uint32_t ev, uint64_t csig);
//   bool     want_causal() const;  bool want_branchial() const;
//   uint32_t producer_at(const Instance&, uint32_t slot) const;   NO_PRODUCER when none
//   void     record_causal(uint32_t producer, uint32_t consumer);
//   AppliedRef publish_applied(const Instance&, const Match&, uint32_t ev);   a position in the
//                              instance's applied list, or a value the Ctx reports as not
//                              published through applied_ref_valid
//   bool     applied_ref_valid(AppliedRef) const;
//   template <class F> void for_each_applied_before(const Instance&, AppliedRef, F&& f);
//                              f(const Applied&) over the applications published STRICTLY
//                              EARLIER than the given position
//   void     record_branchial_pair(uint32_t lo, uint32_t hi);
//   void     descend(const Match&, uint32_t depth, uint32_t ev, const Instance& parent);
//
// A Match supplies: id, to_hash, rule, from_slots, to_slots, num_consumed/produced/survivors,
// and consumed(i)/produced(i)/surv_from(i)/surv_to(i). An Instance supplies id and nslots. An
// Applied supplies event, num_consumed and consumed(j).

#include <cstdint>

#include "hgcommon/core.hpp"
#include "hgcommon/event_core.hpp"

namespace HG_NAMESPACE {
namespace common {

// A slot with no producer: the edge came with the initial state, so no event made it.
constexpr uint32_t QR_NO_PRODUCER = 0xFFFFFFFFu;

// The (instance, match) pair, mixed the same way on both engines because it is one claim set.
HG_HD inline uint64_t qr_apply_key(uint32_t instance, uint32_t match) {
    uint64_t k = FNV_OFFSET;
    k ^= instance; k *= FNV_PRIME;
    k ^= match;    k *= FNV_PRIME;
    return (k == 0 || k == ~uint64_t(0)) ? 1 : k;
}

// The event's CONTENT triple. Isomorphism-invariant and schedule-independent, so it is the
// identity a cross-run or cross-engine comparison of the relations is made on -- which is
// exactly why it cannot be spelled twice.
HG_HD inline uint64_t qr_content_hash(uint64_t from_class, uint64_t to_class, uint32_t rule) {
    uint64_t s = FNV_OFFSET;
    s ^= from_class; s *= FNV_PRIME;
    s ^= to_class;   s *= FNV_PRIME;
    s ^= rule;       s *= FNV_PRIME;
    return s;
}

// Producers of the slots this match consumes, DESCENDING, written into `out` (capacity
// MAX_PATTERN_EDGES). Returns how many. Descending because the causal recorder tests each pair
// against the adjacency built from the ones before it, so nearer producers must be in it first.
template <class Ctx>
HG_HD uint32_t qr_collect_producers(const Ctx& c, const typename Ctx::Instance& inst,
                                    const typename Ctx::Match& m, uint32_t* out) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < m.num_consumed && n < MAX_PATTERN_EDGES; ++i) {
        const uint32_t s = m.consumed(i);
        if (s >= inst.nslots) continue;
        const uint32_t p = c.producer_at(inst, s);
        if (p != QR_NO_PRODUCER) out[n++] = p;
    }
    // Insertion sort, descending; n is at most MAX_PATTERN_EDGES.
    for (uint32_t i = 1; i < n; ++i) {
        const uint32_t v = out[i];
        uint32_t j = i;
        while (j > 0 && out[j - 1] < v) { out[j] = out[j - 1]; --j; }
        out[j] = v;
    }
    return n;
}

// Apply one match to one instance. Returns the minted event id, or INVALID_ID when the pair
// was already claimed or the capture and the instance disagree on the class's width.
template <class Ctx>
HG_HD uint32_t qr_apply(Ctx& c, const typename Ctx::Instance& inst,
                        const typename Ctx::Match& m, uint64_t state_hash, uint32_t depth) {
    if (!c.claim(qr_apply_key(inst.id, m.id))) return INVALID_ID;
    // The capture and the instance disagree on how wide the class is: drop rather than
    // corrupt. A record built from a slot that means nothing replays as a wrong event, and a
    // wrong event is invisible.
    if (m.from_slots != inst.nslots) return INVALID_ID;

    // The raw event this instance's copy of the match stands for. An id suffices: counts and
    // causal edges are expressed over ids, so no Event record -- and hence no raw state and no
    // raw edge -- has to be materialised here.
    const uint32_t ev = c.mint_event();
    c.record_content(ev, state_hash, m.to_hash, m.rule);

    // The RUN's event identity, which is a different question from the invariant above.
    // Slots ARE the canonical ranks the signature wants, and consumed/produced stay in
    // match/RHS order as it requires.
    if (c.keys() != EVENT_SIG_NONE) {
        // The canonical OUTPUT state's step, not this replay's depth. Full capture signs with
        // one value per class; the depth is where this instance happens to sit, so signing
        // with it makes the two signature sets disjoint for every event.
        const uint32_t out_step = c.frame_step(m.to_hash, depth);
        uint64_t csig = event_signature(c.keys(), state_hash, m.to_hash, out_step, m.rule,
                                        m.consumed_ptr(), static_cast<uint8_t>(m.num_consumed),
                                        m.produced_ptr(), static_cast<uint8_t>(m.num_produced));
        if (csig == 0 || csig == ~uint64_t(0)) csig = 1;
        c.record_runsig(ev, csig);
    }

    if (c.want_causal()) {
        uint32_t producers[MAX_PATTERN_EDGES];
        const uint32_t np = qr_collect_producers(c, inst, m, producers);
        for (uint32_t i = 0; i < np; ++i) c.record_causal(producers[i], ev);
    }

    // Branchial: siblings expanding the SAME instance whose consumed slots overlap. The order
    // that matters is APPLICATION order, not match-id order -- ids come from a global counter
    // while the list is appended concurrently, so a lower id can arrive after a higher one has
    // scanned. Publication order is the order the two applications agree on.
    //
    // Each pair is reported by the LATER of the two applications, and only by it: the scan
    // visits the applications published strictly before this one, so of any two exactly one
    // sees the other. That is what makes the relation exact WITHOUT a set of pairs to dedup
    // against -- and a set of pairs is the thing that does not fit, at 133,218,996 entries on
    // disc-l3a2g2r2 depth 3 against the 970,584 applications they are derived from.
    typename Ctx::AppliedRef mine{};
    if (m.num_consumed && c.want_branchial() &&
        c.applied_ref_valid(mine = c.publish_applied(inst, m, ev))) {
        // This application's consumed slots are read once, not once per sibling. The scan below
        // visits every application already published against the instance -- 167 of them on
        // average on disc-l3a2g2r2 depth 3 -- and reading them through the view inside the
        // comparison made the accessor alone a measurable share of the run.
        //
        // A 64-bit fold of the slots, tested against each sibling's fold to reject the disjoint
        // ones in one AND, was measured and REJECTED: 81.6% of these comparisons DO share a
        // slot on this workload, so the old loop usually decided on its first comparison, and
        // computing a fold per sibling added a pass rather than replacing one --
        // 14,509,903,885 instructions to 16,752,095,733, +15.5%.
        uint32_t mine_slots[MAX_PATTERN_EDGES];
        uint32_t mine_n = 0;
        for (uint32_t i = 0; i < m.num_consumed && mine_n < MAX_PATTERN_EDGES; ++i)
            mine_slots[mine_n++] = m.consumed(i);
        c.for_each_applied_before(inst, mine, [&](const typename Ctx::Applied& other) {
            if (other.event == ev) return;                 // this application
            const uint32_t on = other.num_consumed;
            // Swapping these loops so the sibling's slot is read in the OUTER one -- fewer
            // accessor calls, one per slot instead of one per pair of slots -- was measured and
            // REJECTED: 14,187,138,966 instructions to 15,749,617,571, +11.0%. The comparison
            // almost always decides on the first pair, so the accessor count is not what this
            // loop costs, and the original nesting is what the compiler schedules better.
            bool overlaps = false;
            if (on <= 3) {
                // MAX_PATTERN_EDGES is 16 and a left-hand side of one to three edges is what
                // every rule in the corpus has, so the general loop below spends most of this
                // test on bookkeeping for a trip count of three. Reading the sibling's slots
                // into registers and comparing without an inner loop is the same comparison in
                // the same order, with the loop gone.
                const uint32_t o0 = on > 0 ? other.consumed(0) : ~0u;
                const uint32_t o1 = on > 1 ? other.consumed(1) : ~0u;
                const uint32_t o2 = on > 2 ? other.consumed(2) : ~0u;
                for (uint32_t i = 0; i < mine_n; ++i) {
                    const uint32_t s = mine_slots[i];
                    if (s == o0 || s == o1 || s == o2) { overlaps = true; break; }
                }
            } else {
                for (uint32_t i = 0; i < mine_n && !overlaps; ++i)
                    for (uint32_t j = 0; j < on; ++j)
                        if (mine_slots[i] == other.consumed(j)) { overlaps = true; break; }
            }
            if (!overlaps) return;
            // Keyed on the two EVENTS, so the pair reads back as a pair of event signatures and
            // set-compares against full capture, which keys its own branchial edges the same
            // way. A key over match ids has the same scope and nothing can be recovered from it.
            const uint32_t lo = ev < other.event ? ev : other.event;
            const uint32_t hi = ev < other.event ? other.event : ev;
            c.record_branchial_pair(lo, hi);
        });
    }

    // The child instance: survivors carry their producer across, produced slots take THIS
    // event. Building and driving it is the Ctx's, because where a producer vector lives is
    // the one thing about it that differs.
    c.descend(m, depth, ev, inst);
    return ev;
}

}  // namespace common
}  // namespace HG_NAMESPACE
