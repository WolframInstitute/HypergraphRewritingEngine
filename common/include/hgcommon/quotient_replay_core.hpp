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
//   causal        one relation per consumed slot that carries a producer, and its transitive
//                 reduction decided here, online: a producer is dropped when it is a proper
//                 ancestor of another producer of this event (hgcommon::redundant_producers).
//                 Every producer's own kept set was recorded in its application, before the
//                 descent that made the instance this application runs on, so every path into
//                 the event exists when it is judged and no later event adds one: the kept set
//                 is the unique reduction of the relation, whatever the schedule.
//   branchial     siblings expanding the SAME instance whose consumed slots overlap. The
//                 application is published into the instance's applied list, from which the
//                 readback enumerates the pairs; the pair COUNT comes from class
//                 multiplicities (quotient_multiplicity_core.hpp), which count the same pairs
//                 without enumerating them.
//   descend       the child instance, recorded by its lineage (parent instance, the match that
//                 made it, this event), and drive it.
//
// Every one of those is a decision about the reconstructed relation, and none of them is a
// storage question. What differs between the engines is only where things are held: a
// scratch vector against a packed word arena, a ConcurrentMap against a DedupMap, an arena
// allocation against a bump offset.
//
// A Ctx must supply:
//
//   using Instance = ...;  using Match = ...;
//   bool     claim(const Instance&, const Match&);   exactly-once on the (instance, match) pair
//   uint32_t mint_event(uint32_t above);   a fresh id, greater than `above` when above is not
//                                          QR_NO_PRODUCER; INVALID_ID when the id space is
//                                          exhausted (QR_ID_LIMIT), which the Ctx reports
//   void     record_content(uint32_t ev, uint64_t from_class, uint64_t to_class, uint32_t rule);
//   EventSignatureKeys keys() const;           EVENT_SIG_NONE to skip the run signature
//   void     record_runsig(uint32_t ev, const Match& m, uint64_t from_class, uint32_t out_step);
//                                          the run signature of `m` applied: a function of the
//                                          three (qr_signature_values)
//   bool     want_causal() const;  bool want_branchial() const;
//   uint32_t producer_at(const Instance&, uint32_t slot) const;   NO_PRODUCER when none; the
//                                          Ctx answers it with qr_producer_of below
//   void     record_causal(uint32_t producer, uint32_t consumer, bool distinct_pair);
//   uint32_t redundant(const uint32_t* producers, uint32_t n);
//                              hgcommon::redundant_producers over the recorded kept sets
//   void     record_kept(uint32_t ev, const uint32_t* kept, uint32_t nkept);
//                              the event's kept producers, before descend
//                              `distinct_pair` is false when this producer repeats the previous
//                              one in the same application's list, which is the ONLY way a
//                              (producer, consumer) pair can repeat -- see below. The edge
//                              multiset counts every call; the PAIR is recorded only when set.
//   void     publish_applied(const Instance&, const Match&, uint32_t ev);   into the
//                              instance's applied list, for the readback's enumeration
//   void     descend(const Match&, uint32_t depth, uint32_t ev, const Instance& parent);
//
// A Match supplies: id, to_hash, rule, from_slots, to_slots, num_consumed/produced/survivors,
// consumed(i)/produced(i)/surv_from(i)/surv_to(i), and marked_forms(): the marked forms of the
// raw event it was captured from (EventMarkedForms). An Instance supplies id and nslots.

#include <algorithm>
#include <cstdint>
#include <utility>

#include "hgcommon/core.hpp"
#include "hgcommon/event_core.hpp"

namespace HG_NAMESPACE {
namespace common {

// A slot with no producer: the edge came with the initial state, so no event made it.
constexpr uint32_t QR_NO_PRODUCER = 0xFFFFFFFFu;

// Raw event and instance ids are 32-bit and index per-event arrays. Both engines mint them below
// this limit and refuse past it: the application that asked is dropped and the run reports
// "ReplayIdsExhausted". The 2^20 below 2^32 bounds the device counter's overshoot by the lanes
// that pass the limit check together, so the counter cannot wrap.
constexpr uint32_t QR_ID_LIMIT = 0xFFF00000u;
// A captured transition whose edges have no image in its class's frame cannot be replayed, so it
// is dropped; both engines count such captures and report "CapturesDropped".
constexpr const char* QR_CAPTURES_DROPPED_MESSAGE =
    "a captured transition could not be aligned to its class's frame and was dropped; the "
    "reconstructed raw events and relations are TRUNCATED at that point";
constexpr const char* QR_IDS_EXHAUSTED_MESSAGE =
    "the replay minted its limit of raw event or instance ids; the reconstructed raw events and "
    "relations are TRUNCATED at that point";

// WHERE A SLOT'S EDGE CAME FROM, per captured match: for each slot of the child frame, the
// parent slot it survived from, QR_SOURCE_PRODUCED when the match produced it, or
// QR_SOURCE_NONE. `out` holds to_slots entries.
constexpr uint32_t QR_SOURCE_PRODUCED = 0xFFFFFFFEu;
constexpr uint32_t QR_SOURCE_NONE     = 0xFFFFFFFFu;
HG_HD inline void qr_fill_child_sources(const uint32_t* produced, uint32_t np,
                                        const uint32_t* surv_from, const uint32_t* surv_to,
                                        uint32_t ns, uint32_t to_slots, uint32_t* out) {
    for (uint32_t i = 0; i < to_slots; ++i) out[i] = QR_SOURCE_NONE;
    for (uint32_t i = 0; i < ns; ++i)
        if (surv_to[i] < to_slots) out[surv_to[i]] = surv_from[i];
    for (uint32_t i = 0; i < np; ++i)
        if (produced[i] < to_slots) out[produced[i]] = QR_SOURCE_PRODUCED;
}

// THE PRODUCER OF A SLOT, BY LINEAGE. An instance records its parent instance, the match that
// made it and that match's event, not a producer per slot: a slot the match produced was
// produced by that event, and any other slot survived from a parent slot, whose producer is the
// parent's answer for it. The walk is at most the instance's depth, which is the step count.
// A per-slot vector per instance is the replay's largest store on large states (17 GB on a
// 256-edge path at three steps); a lineage record is three words.
//
// The Ctx supplies, over its own lineage handle L:
//   bool     lineage_root(L) const;              the instance has no parent (a root)
//   uint32_t lineage_source(L, uint32_t slot) const;   the making match's child source table
//   uint32_t lineage_event(L) const;             the making match's event
//   L        lineage_parent(L) const;
template <class Ctx, class L>
HG_HD uint32_t qr_producer_of(const Ctx& c, L node, uint32_t slot) {
    for (;;) {
        if (c.lineage_root(node)) return QR_NO_PRODUCER;
        const uint32_t src = c.lineage_source(node, slot);
        if (src == QR_SOURCE_PRODUCED) return c.lineage_event(node);
        if (src == QR_SOURCE_NONE) return QR_NO_PRODUCER;
        slot = src;
        node = c.lineage_parent(node);
    }
}

// The root instance of a lineage, the initial state's: the parents walked to the end.
template <class Ctx, class L>
HG_HD L qr_lineage_root(const Ctx& c, L node) {
    while (!c.lineage_root(node)) node = c.lineage_parent(node);
    return node;
}

// THE GENESIS PAIR RULE (docs/SPEC.md §5.2). An application is paired with its initial state's
// genesis event when it consumed an edge of that state, and under the transitive reduction only
// when it consumed no produced edge: each producer is reached from the same genesis event. Full
// capture reads the two facts from the consumed edges' producers, the reconstruction from
// qr_genesis_paired below.
HG_HD inline bool qr_genesis_pair_kept(bool consumed_initial, bool consumed_produced,
                                       bool reduced) {
    return consumed_initial && !(reduced && consumed_produced);
}

// The rule over one reconstructed application: `m` applied to the instance whose lineage is
// `parent`. A consumed slot with no producer (qr_producer_of) is an edge of the initial state.
template <class Ctx, class L, class M>
HG_HD bool qr_genesis_paired(const Ctx& c, L parent, const M& m, bool reduced) {
    bool initial = false, produced = false;
    for (uint32_t j = 0; j < m.num_consumed; ++j) {
        if (qr_producer_of(c, parent, m.consumed(j)) == QR_NO_PRODUCER) initial = true;
        else produced = true;
    }
    return qr_genesis_pair_kept(initial, produced, reduced);
}

// The (instance, match) pair, mixed the same way on both engines because it is one claim set.
HG_HD inline uint64_t qr_apply_key(uint32_t instance, uint32_t match) {
    uint64_t k = FNV_OFFSET;
    k ^= instance; k *= FNV_PRIME;
    k ^= match;    k *= FNV_PRIME;
    return avoid_reserved_keys(k);
}

// WHERE A PAIR IS CLAIMED. An instance carries a chain of claim blocks with one bit per class
// match, in per-class match order. The first block is made with the instance: one 64-bit word
// for every 64 matches its class held then, and at least one. A match past the chain's end
// installs the next block, which starts where the chain ends and holds the larger of the words
// that reach the match and the words of the block before it. A block is installed by one
// compare-and-swap on its predecessor's next link and never changes after, so both sides of the
// rendezvous walk to the same bit.
HG_HD inline uint32_t qr_claim_words(uint32_t class_matches) {
    return class_matches ? (class_matches + 63u) / 64u : 1u;
}
HG_HD inline uint32_t qr_claim_bits(uint32_t words) { return words * 64u; }

enum QrClaim : int { QR_CLAIM_LOST = 0, QR_CLAIM_WON = 1, QR_CLAIM_NO_ROOM = 2 };

// Claim bit `local` in the chain starting at `first`. QR_CLAIM_NO_ROOM when a block the claim
// needs could not be allocated; the caller then claims the pair elsewhere. `B` supplies:
//   using Block = ...;
//   bool     is_null(Block) const;
//   uint32_t words(Block) const;
//   Block    next(Block) const;                       acquire
//   Block    install_next(Block b, uint32_t words);   a zeroed block of `words` words linked
//                                                     after b by compare-and-swap; the block
//                                                     another claimer linked when that one won;
//                                                     null when allocation failed
//   bool     set_bit(Block, uint32_t bit);            fetch_or; true when this call set it
template <class B>
HG_HD QrClaim qr_claim_chain(B& b, typename B::Block blk, uint32_t local) {
    uint32_t base = 0;
    for (;;) {
        const uint32_t words = b.words(blk);
        const uint32_t end = base + qr_claim_bits(words);
        if (local < end) return b.set_bit(blk, local - base) ? QR_CLAIM_WON : QR_CLAIM_LOST;
        typename B::Block n = b.next(blk);
        if (b.is_null(n)) {
            const uint32_t need = (local - end) / 64u + 1u;
            n = b.install_next(blk, need > words ? need : words);
            if (b.is_null(n)) return QR_CLAIM_NO_ROOM;
        }
        blk = n;
        base = end;
    }
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

// The event's signature under the RUN's identity mode, the values it digests, and the two
// arguments besides the match that the values are a function of.
struct QrRunSignature {
    uint64_t sig;
    uint64_t from_hash;
    uint32_t out_step;
    uint32_t n;
    uint64_t values[EVENT_SIG_MAX_VALUES];
};

// The signature of match `m` applied from class `from_hash`, with `out_step` as the output step.
// Under event_keys_mark_edges the edges enter as the captured event's marked forms: every
// instance of a class is the frame under an isomorphism that carries the captured event's edges
// to the match's slots, so the forms are the same for every application of the match.
template <class Match>
HG_HD void qr_signature_values(EventSignatureKeys keys, const Match& m, uint64_t from_hash,
                               uint32_t out_step, QrRunSignature& out) {
    out.from_hash = from_hash;
    out.out_step = out_step;
    out.n = event_signature_values(keys, from_hash, m.to_hash, out_step, m.rule,
                                   m.consumed_ptr(), static_cast<uint8_t>(m.num_consumed),
                                   m.produced_ptr(), static_cast<uint8_t>(m.num_produced),
                                   out.values,
                                   event_keys_mark_edges(keys) ? m.marked_forms() : nullptr);
    out.sig = avoid_reserved_keys(event_signature_of_values(out.values, out.n));
}

// A match's cached run-signature key. The run signature of a match is a function of the match,
// its from class and the output step, and a match has one from class, so its claimed key is kept
// on the match for one output step. `A` supplies the match's two cells:
//   uint32_t step_load() const;                      relaxed
//   bool     step_cas(uint32_t expected, uint32_t desired);
//   uint64_t key_load() const;                       acquire
//   void     key_store(uint64_t key);                release
// The step is set once, from QR_NO_STEP, by the thread that then stores the key, so a reader
// that sees a key sees the step it was claimed for. A key claimed for another step is not cached.
constexpr uint32_t QR_NO_STEP = ~0u;

template <class A>
HG_HD bool qr_cached_key(const A& a, uint32_t out_step, uint64_t& key) {
    const uint64_t k = a.key_load();
    if (k == 0 || a.step_load() != out_step) return false;
    key = k;
    return true;
}

template <class A>
HG_HD void qr_cache_key(A& a, uint32_t out_step, uint64_t key) {
    if (a.step_cas(QR_NO_STEP, out_step)) a.key_store(key);
}

// True when two signatures have the same values.
HG_HD inline bool qr_same_values(const QrRunSignature& a, const QrRunSignature& b) {
    if (a.n != b.n) return false;
    for (uint32_t i = 0; i < a.n; ++i)
        if (a.values[i] != b.values[i]) return false;
    return true;
}

// The step a run signature records: the event's own step, the step of the raw state it
// produces. An application to an instance at `depth` produces a state at depth + 1, as a raw
// event from a state at step s produces one at s + 1 under full capture (the host's output
// state's step, the device's DeviceEvent::step). Known when the event is made, so it does not
// depend on which state of a class was explored first.
HG_HD inline uint32_t qr_out_step(uint32_t depth) { return depth + 1u; }

// Whether two matches of one state consume a common slot: the branchial test. `mine` holds the
// first match's consumed slots; `other` supplies num_consumed and consumed(j).
template <class Other>
HG_HD inline bool qr_consumed_overlap(const uint32_t* mine, uint32_t mine_n, const Other& other) {
    const uint32_t on = other.num_consumed;
    // Swapping these loops so the sibling's slot is read in the OUTER one -- fewer accessor
    // calls, one per slot instead of one per pair of slots -- was measured and REJECTED:
    // 14,187,138,966 instructions to 15,749,617,571, +11.0%. The comparison almost always
    // decides on the first pair, so the accessor count is not what this loop costs, and the
    // original nesting is what the compiler schedules better.
    if (on <= 3) {
        // MAX_PATTERN_EDGES is 16 and a left-hand side of one to three edges is what every rule
        // in the corpus has, so the general loop below spends most of this test on bookkeeping
        // for a trip count of three. Reading the sibling's slots into registers and comparing
        // without an inner loop is the same comparison in the same order, with the loop gone.
        const uint32_t o0 = on > 0 ? other.consumed(0) : ~0u;
        const uint32_t o1 = on > 1 ? other.consumed(1) : ~0u;
        const uint32_t o2 = on > 2 ? other.consumed(2) : ~0u;
        bool overlaps = false;
        for (uint32_t i = 0; i < mine_n; ++i) {
            const uint32_t s = mine[i];
            if (s == o0 || s == o1 || s == o2) { overlaps = true; break; }
        }
        return overlaps;
    }
    bool overlaps = false;
    for (uint32_t i = 0; i < mine_n && !overlaps; ++i)
        for (uint32_t j = 0; j < on; ++j)
            if (mine[i] == other.consumed(j)) { overlaps = true; break; }
    return overlaps;
}

// THE BRANCHIAL PAIRS OF ONE INSTANCE, for a readback of the relation: every unordered pair of
// the instance's applications whose consumed slots overlap, once each, as emit(lo event,
// hi event). Up to QR_PAIR_TEST_MAX applications every pair is tested. Above it the
// applications' (slot, index) entries are sorted by slot and a pair is emitted from the group of
// its LOWEST common slot only, so the work is the sort plus the pairs that share a slot, where
// testing every pair is m(m-1)/2. `entries` is the caller's buffer of
// std::pair<uint32_t, uint32_t>. App supplies event, num_consumed and consumed(j). Host only.
constexpr uint32_t QR_PAIR_TEST_MAX = 32;
template <class App, class Buf, class Emit>
HG_INLINE void qr_instance_branchial_pairs(const App* apps, uint32_t n, Buf& entries, Emit&& emit) {
    if (n <= QR_PAIR_TEST_MAX) {
        for (uint32_t i = 0; i < n; ++i) {
            const App& a = apps[i];
            for (uint32_t j = i + 1; j < n; ++j) {
                const App& b = apps[j];
                if (a.event == b.event) continue;
                bool overlaps = false;
                for (uint32_t x = 0; x < a.num_consumed && !overlaps; ++x)
                    for (uint32_t y = 0; y < b.num_consumed; ++y)
                        if (a.consumed(x) == b.consumed(y)) { overlaps = true; break; }
                if (!overlaps) continue;
                const uint32_t lo = a.event < b.event ? a.event : b.event;
                const uint32_t hi = a.event < b.event ? b.event : a.event;
                emit(lo, hi);
            }
        }
        return;
    }
    entries.clear();
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < apps[i].num_consumed; ++j)
            entries.push_back({apps[i].consumed(j), i});
    // An application that names a slot twice contributes it once.
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    auto lowest_common = [](const App& a, const App& b) {
        uint32_t lo = ~0u;
        for (uint32_t x = 0; x < a.num_consumed; ++x)
            for (uint32_t y = 0; y < b.num_consumed; ++y)
                if (a.consumed(x) == b.consumed(y) && a.consumed(x) < lo) lo = a.consumed(x);
        return lo;
    };
    for (size_t g = 0; g < entries.size();) {
        size_t e = g;
        while (e < entries.size() && entries[e].first == entries[g].first) ++e;
        const uint32_t slot = entries[g].first;
        for (size_t x = g; x < e; ++x)
            for (size_t y = x + 1; y < e; ++y) {
                const App& a = apps[entries[x].second];
                const App& b = apps[entries[y].second];
                if (a.event == b.event || lowest_common(a, b) != slot) continue;
                emit(a.event < b.event ? a.event : b.event, a.event < b.event ? b.event : a.event);
            }
        g = e;
    }
}

// Apply one match to one instance. Returns the minted event id, or INVALID_ID when the pair
// was already claimed or the capture and the instance disagree on the class's width.
template <class Ctx>
HG_HD uint32_t qr_apply(Ctx& c, const typename Ctx::Instance& inst,
                        const typename Ctx::Match& m, uint64_t state_hash, uint32_t depth) {
    if (!c.claim(inst, m)) return INVALID_ID;
    // The capture and the instance disagree on how wide the class is: drop rather than
    // corrupt. A record built from a slot that means nothing replays as a wrong event, and a
    // wrong event is invisible.
    if (m.from_slots != inst.nslots) return INVALID_ID;

    // The raw event this instance's copy of the match stands for. An id suffices: counts and
    // causal edges are expressed over ids, so no Event record -- and hence no raw state and no
    // raw edge -- has to be materialised here.
    // Collected before the mint: the event's id is minted above its largest producer, so ids
    // increase along every causal edge, which the reduction's search requires.
    const bool causal = c.want_causal();
    // producers[0] is written before the collect: the optimiser may load it for the mint's
    // argument below even when np is 0, and a model checker reads that load as a read of
    // uninitialised memory (GenMC, quotient_capture_composition).
    uint32_t producers[MAX_PATTERN_EDGES];
    producers[0] = QR_NO_PRODUCER;
    const uint32_t np = causal ? qr_collect_producers(c, inst, m, producers) : 0;
    const uint32_t ev = c.mint_event(np ? producers[0] : QR_NO_PRODUCER);
    if (ev == INVALID_ID) return INVALID_ID;
    c.record_content(ev, state_hash, m.to_hash, m.rule);

    // The RUN's event identity, which is a different question from the invariant above.
    // Under the Automatic preset the signature reads the match's slots in match/RHS order where
    // full capture reads canonical ranks; the endpoint classes are in the same signature, and
    // both are injective on a state's edges, so the count of distinct signatures is the same.
    if (c.keys() != EVENT_SIG_NONE) {
        c.record_runsig(ev, m, state_hash, qr_out_step(depth));
    }

    if (causal) {
        // THE PAIR CANNOT REPEAT ACROSS APPLICATIONS, because `ev` was minted for this one.
        // It can repeat WITHIN this one, when two consumed slots carry the same producer, and
        // that is the entire duplicate population: measured on cycle4, recording every call as
        // a pair gives 129,384 against the 95,600 distinct ones. The list is sorted, so the
        // repeats are adjacent and the test is a comparison with the previous element.
        //
        // Telling the Ctx which calls are distinct is what lets it store the pairs without a
        // shared dedup structure. The edge multiset still counts every call, so the two
        // observables keep their separate meanings.
        uint32_t distinct[MAX_PATTERN_EDGES];
        uint32_t nd = 0;
        for (uint32_t i = 0; i < np; ++i) {
            const bool first = i == 0 || producers[i] != producers[i - 1];
            c.record_causal(producers[i], ev, first);
            if (first) distinct[nd++] = producers[i];
        }
        const uint32_t dropped = c.redundant(distinct, nd);
        uint32_t kept[MAX_PATTERN_EDGES];
        uint32_t nkept = 0;
        for (uint32_t i = 0; i < nd; ++i)
            if (!(dropped & (1u << i))) kept[nkept++] = distinct[i];
        c.record_kept(ev, kept, nkept);
    }

    // Branchial: published for the readback, which pairs the instance's applications whose
    // consumed slots overlap. The run counts the pairs from class multiplicities, so nothing
    // here scans the siblings.
    if (m.num_consumed && c.want_branchial()) c.publish_applied(inst, m, ev);

    // The child instance, recorded by its lineage (qr_producer_of). Building and driving it is
    // the Ctx's, because where an instance record lives is the one thing about it that differs.
    c.descend(m, depth, ev, inst);
    return ev;
}

}  // namespace common
}  // namespace HG_NAMESPACE
