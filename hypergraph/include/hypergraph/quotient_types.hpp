#pragma once
#include "hgcommon/namespace.hpp"
//
// What a state, an edge and an event ARE once isomorphic states are identified.
//
// These six are one concern and they are separated from the raw-graph types for that reason:
// they describe identity under quotient, where types.hpp's Edge/Event/State describe the
// concrete graph. They are also the only types that need hgcommon/quotient_replay_core.hpp,
// which types.hpp would otherwise hand to every file that includes it -- and types.hpp is
// included by every engine header.
//
// hgcommon supplies the mixing (qr_content_hash) rather than this file open-coding it, because
// the DEVICE stores the hash where the host stores the triple, and two open-codings of one
// arithmetic agree only until one is edited.

#include <cstdint>

#include "hgcommon/core.hpp"                   // EMPTY_STATE_CANONICAL_HASH, the id widths
#include "hgcommon/quotient_replay_core.hpp"   // qr_content_hash -- the event content identity
#include "hgcommon/event_core.hpp"

namespace HG_NAMESPACE {
namespace engine {

using hgcommon::VertexId;
using hgcommon::EdgeId;
using hgcommon::StateId;
using hgcommon::EventId;
using hgcommon::INVALID_ID;
using RuleIndex = uint16_t;  // host-only width (the GPU port uses a 32-bit RuleId)

// One reconstructed event's content: the class it went from, the class it went to, and the rule
// that took it. A reconstructed event has no Event record -- the replay mints an id and
// materialises nothing -- so this is what describes it.
//
// The triple HASH is its isomorphism-invariant identity, derived here rather than stored beside
// the components so the mixing has one definition and the two cannot drift.
struct QcEventContent {
    uint64_t from_class = 0;
    uint64_t to_class = 0;
    uint32_t rule = 0;

    // From hgcommon, because the DEVICE stores this hash where the host stores the triple and
    // hashes on demand. Two engines writing the same identity is exactly one rule, and two
    // open-codings of the same arithmetic agree only until one is edited.
    uint64_t triple_hash() const;
};

// The quotient-aware identity of a HYPEREDGE, used as the rendezvous key that meets an
// edge's producer events with its consumer events. Strongly typed and distinct from:
//   * EdgeId     -- one concrete hyperedge instance in one state (a dense counter), and
//   * CausalEdge -- a producer->consumer relation at the multiway-graph level.
// Minted ONLY by Hypergraph::causal_edge_keys: under quotient it is
// fnv(canonical_hash(state), edge_orbit_in_state) -- iso-invariant, so every raw edge
// instance that distinct parents produce/consume for one canonical edge orbit collapses to
// a single key (the orbit is the only edge identity invariant across the labelings by
// which parents reach a canonical state). Without quotient it degrades to the raw EdgeId
// so isomorphic-but-distinct raw states keep disjoint causal edges. A hash, hence 64-bit,
// not a dense counter -- never construct one from an EdgeId except through causal_edge_keys.
struct CanonicalEdgeKey {
    uint64_t value{0};
    CanonicalEdgeKey() = default;
    explicit constexpr CanonicalEdgeKey(uint64_t v) : value(v) {}
    constexpr bool operator==(CanonicalEdgeKey o) const { return value == o.value; }
    constexpr bool operator!=(CanonicalEdgeKey o) const { return value != o.value; }
};

// A state's canonical edge-orbit table, computed once when the state is canonicalized
// (Full mode) and cached. `orbit[i]` is the automorphism-orbit id of the edge `edges[i]`;
// `edges` is sorted ascending for binary-search lookup. Orbit ids are numbered canonically
// (iso-invariant), so the same orbit id denotes corresponding edges across every raw state
// isomorphic to this one -- the identity the quotient causal reconstruction indexes on.
// Arena-allocated; `num_orbits` is the count of distinct orbits.
// Per-edge canonical RANK for one state: the edge's position when the state's edges are
// ordered by (canonical content, original index). Distinct for every edge -- the index
// tie-break separates duplicate-content edges, which Positional event identity requires
// since it must not quotient state automorphisms. Arena-allocated; edges ascend so a lookup
// binary-searches.
struct EdgeRankTable {
    uint32_t n = 0;
    const EdgeId* edges = nullptr;   // sorted ascending, length n
    const uint32_t* rank = nullptr;  // length n, parallel to edges
};

struct EdgeOrbitTable {
    uint32_t n = 0;
    uint32_t num_orbits = 0;
    const EdgeId* edges = nullptr;      // sorted ascending, length n
    const uint32_t* orbit = nullptr;    // length n, parallel to edges
    const uint32_t* orbit_size = nullptr;  // length num_orbits
    // Per-edge SLOT: the edge's rank when the state's edges are ordered by (Aut ORBIT,
    // EdgeId). Slots are a permutation of [0,n). Orbit, not content class: which content
    // class an edge lands in depends on which canonical labeling the IR run picked (two
    // labelings differ by an automorphism, which can permute distinct contents), so a
    // per-edge class is defined only up to the Aut action. The orbit is the Aut-closure
    // itself, so the orbit-block structure of the slots is identical in every raw instance
    // of one canonical state -- which is what lets a transition recorded on the expanded
    // representative be replayed against an arbitrary instance. Ties within an orbit break
    // on EdgeId, which is arbitrary but harmless: the match set is closed under Aut, so a
    // within-orbit permutation maps matches to matches and the emitted set is unchanged.
    const uint32_t* slot = nullptr;     // length n, parallel to edges
    const uint32_t* klass = nullptr;    // length n, parallel to edges (canonical content class)
    // The edge's rank in the canonical form, from the same search as its orbit: a permutation
    // of [0,n). Two raw states of one canonical class have the same form, so equal rank is
    // the isomorphism between them, and a state is aligned onto its class frame by rank.
    const uint32_t* rank = nullptr;     // length n, parallel to edges

    // Binary search the sorted edge array; returns n if absent (edge not in state).
    uint32_t index_of(EdgeId e) const;
    uint32_t orbit_of(EdgeId e) const;
    uint32_t slot_of(EdgeId e) const;
};

// One distinct canonical transition out of a canonical state, in edge-orbit terms -- the
// unit the quotient causal reconstruction propagates over. All raw events sharing the same
// (from canonical state, to canonical state, rule, consumed orbits, surviving orbit map)
// collapse to one of these; `canon_event` is a representative canonical event id used as
// the producer/consumer identity when emitting causal edges. Orbit arrays are arena-
// allocated and sorted. `to_hash` is the child canonical state. See the validated
// reconstruction in tools/quotient_causal_support_probe.cpp.
struct CanonicalTransition {
    uint64_t to_hash = 0;
    uint64_t sig = 0;              // dedup signature over the fields below
    EventId canon_event = INVALID_ID;
    RuleIndex rule = 0;
    uint32_t num_consumed = 0, num_produced = 0, num_survivors = 0;
    const uint32_t* consumed_orbits = nullptr;   // length num_consumed, sorted
    const uint32_t* produced_orbits = nullptr;   // length num_produced, sorted
    const uint32_t* surv_from_orbits = nullptr;  // length num_survivors (orbit in `from`)
    const uint32_t* surv_to_orbits = nullptr;    // length num_survivors (orbit in `to`)

    // Accessors, because hgcommon/quotient_causal_core.hpp reads the orbit arrays through them
    // and the device packs its four into one contiguous word arena. The DP does not know or
    // care which layout it is walking.
    uint32_t consumed(uint32_t i) const;
    uint32_t produced(uint32_t i) const;
    uint32_t surv_from(uint32_t i) const;
    uint32_t surv_to(uint32_t i) const;
};

// One match of the expanded representative of a canonical state, named in SLOTS -- the unit
// the per-instance raw reconstruction replays.
//
// Distinct from CanonicalTransition: that record is DEDUPLICATED by an orbit signature, which
// is right for the aggregate producer-set propagation (it needs only the support) but wrong
// here twice over. Slots are finer than orbits, so two matches consuming different edges of
// one orbit collapse into a single transition; and the record carries no multiplicity, so the
// collapse is not recoverable. Full-capture fires both matches, so the reconstruction must see
// both. Hence the representative's matches are kept in full, undeduplicated.
//
// Consumed/produced stay in MATCH order (not sorted): a replay reads
// producer[consumed_slots[i]] and writes the new raw event into producer[produced_slots[i]],
// so the i-th entry must remain the i-th matched / i-th RHS edge. `from_slots`/`to_slots` are
// the slot counts of the source and child states -- the sizes of the producer vectors this
// match reads and writes.
struct SlotMatch {
    uint64_t to_hash = 0;
    uint32_t id = 0;               // dense id, unique per captured match (claims key on it)
    RuleIndex rule = 0;
    uint32_t from_slots = 0, to_slots = 0;
    uint32_t num_consumed = 0, num_produced = 0, num_survivors = 0;
    const uint32_t* consumed_slots = nullptr;    // length num_consumed (slot in `from`)
    const uint32_t* produced_slots = nullptr;    // length num_produced (slot in `to`)
    const uint32_t* surv_from_slot = nullptr;    // length num_survivors (slot in `from`)
    const uint32_t* surv_to_slot = nullptr;      // length num_survivors (slot in `to`)

    // Accessors, because hgcommon/quotient_replay_core.hpp reads the slot arrays through them
    // and the device packs its four into one contiguous word arena. The replay walks both
    // through the same calls and knows neither layout.
    uint32_t consumed(uint32_t i) const;
    uint32_t produced(uint32_t i) const;
    uint32_t surv_from(uint32_t i) const;
    uint32_t surv_to(uint32_t i) const;
    // The signature reads consumed/produced as CONTIGUOUS runs in match/RHS order.
    const uint32_t* consumed_ptr() const;
    const uint32_t* produced_ptr() const;
};

}  // namespace engine
}  // namespace HG_NAMESPACE