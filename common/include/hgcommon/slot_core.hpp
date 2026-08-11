#pragma once
#include "hgcommon/namespace.hpp"
//
// FRAME SLOTS: one definition, host and device.
//
// A slot is an edge's rank when a state's edges are ordered by (Aut ORBIT, EdgeId). It is the
// coordinate system a canonical class's matches are recorded in, so that a match found on one
// raw instance of the class can be replayed against any other instance. Two implementations of
// this rule that drift by one tie-break produce replayed events that are wrong and invisible,
// which is why the rule lives here and not in either engine.
//
// ORBIT, NOT CONTENT CLASS. An automorphism can permute edges between content classes, so a
// per-edge class is defined only up to the Aut action and two raw states of one canonical class
// can disagree about which class an edge belongs to. Orbits are Aut-invariant, so the orbit
// block structure is identical in every instance. Order WITHIN an orbit is arbitrary and
// harmless: the match set is closed under Aut, so permuting inside an orbit maps matches to
// matches and leaves the replayed set unchanged. That is exactly the property a class rank
// lacks.
//
// TWO ACCESS PATTERNS, ONE RULE.
//   slot_rank(orbit, n, i)                  one edge, no scratch  -- THE DEFINITION
//   slots_from_orbits(orbit, n, out, ...)   all edges, counting   -- the bulk form
// The bulk form must equal the definition for every i. That is not a comment: it is asserted
// over random orbit vectors by SlotCore.BulkFormEqualsDefinition in the host suite, and the
// device's own reading of the rule is checked against it in gpu/tests/test_quotient_expansion.
//
// The caller supplies edges in ASCENDING id order, which both engines already do (the host's
// SparseBitset iterates in id order; the device's CSR slice is sorted and binary-searched), so
// index order IS EdgeId order and the tie-break needs no ids.

#include "hgcommon/core.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace common {

// Rank of edge `i` under (orbit, index). O(n), no scratch. This is the definition of a slot;
// every other form is an optimisation of it and must agree with it.
HG_HD inline uint32_t slot_rank(const uint32_t* orbit, uint32_t n, uint32_t i) {
    const uint32_t mine = orbit[i];
    uint32_t rank = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t o = orbit[k];
        if (o < mine) { ++rank; continue; }
        if (o == mine && k < i) ++rank;   // ties break on index == ascending EdgeId
    }
    return rank;
}

// Every slot at once by counting sort: O(n + num_orbits) rather than n calls to slot_rank.
// `counts` is caller-provided scratch of length num_orbits and is fully overwritten.
//
// Equivalent to a stable sort of the indices by orbit followed by numbering them in order:
// counting ascending indices into per-orbit cursors visits each orbit's members in ascending
// index order, which is what stability means here.
HG_HD inline void slots_from_orbits(const uint32_t* orbit, uint32_t n,
                                    uint32_t* out_slot,
                                    uint32_t* counts, uint32_t num_orbits) {
    for (uint32_t j = 0; j < num_orbits; ++j) counts[j] = 0;
    for (uint32_t i = 0; i < n; ++i) counts[orbit[i]]++;
    uint32_t running = 0;
    for (uint32_t j = 0; j < num_orbits; ++j) {
        const uint32_t c = counts[j];
        counts[j] = running;          // counts becomes the per-orbit cursor
        running += c;
    }
    for (uint32_t i = 0; i < n; ++i) out_slot[i] = counts[orbit[i]]++;
}

}  // namespace common
}  // namespace HG_NAMESPACE
