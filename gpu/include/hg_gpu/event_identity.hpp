#pragma once
// Event identity on the device.
//
// One statement of the signature rule, read by the persistent kernel and by the host-driven
// identity phase alike. A second copy would not crash; it would silently identify a different
// set of events, which is the defect class the shared-core work exists to close.
//
// The identity is defined over ISOMORPHISM CLASSES independently of how states are being
// identified (SPEC.md sec 4), so every component here reads DeviceState::state_exact_hash and
// DeviceState::state_edge_rank rather than the state mode's dedup key. Both are filled by
// fill_event_identity_inputs below, which runs the individualization pass once per state and
// takes the ranks off the same pass.

#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/exploration.hpp"   // DedupMap
#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/persistent.hpp"    // default_persistent_grid — the arena's slot-holder bound
#include "hg_gpu/types.hpp"

#include "hgcommon/event_core.hpp"

#include <cstdint>

namespace hg_gpu {

// True when the run's key set reads per-edge canonical ranks, so the rank array is worth its
// four bytes per edge slot. Derived from the keys rather than passed alongside them, so the
// flag cannot disagree with the key set it describes.
inline HG_HD bool event_keys_need_ranks(EventSignatureKeys keys) {
    return (keys & (hgcommon::EventKey_ConsumedEdges |
                    hgcommon::EventKey_ProducedEdges)) != 0;
}

// Rank of `edge` inside `sid`, from the array the canonicalization pass filled. A linear scan
// over the state's own slice: slices are the size of a state's edge set and a rule consumes at
// most kMaxPatternEdges of them, so this is bounded by the rule rather than by the run.
//
// UINT32_MAX when the state has no ranks or the edge is not in it. The caller substitutes the
// raw edge id and counts it, because a signature built from an id is not an isomorphism
// invariant and a silent substitution would make that invisible.
__device__ __forceinline__ uint32_t edge_rank_in_state_device(DeviceState ds, StateId sid,
                                                              EdgeId edge) {
    if (!ds.state_edge_rank || sid >= ds.max_states) return UINT32_MAX;
    StateEdgeSlice sl = ds.state_edge_slices[sid];
    for (uint32_t k = 0; k < sl.count; ++k)
        if (ds.state_edge_ids[sl.offset + k] == edge) return ds.state_edge_rank[sl.offset + k];
    return UINT32_MAX;
}

// Stamp one event with the identity the run's key set asks for, and APPLY that identity: two
// applications whose signatures agree are the same event, so the second to arrive records the
// first as its canonical id. Without the insert the signature would be computed and dropped,
// and the mode would be accepted while changing nothing about the result.
//
// Ranks are resolved in the states they belong to -- consumed in the input, produced in the
// output -- because a rank is a position in THAT state's canonical labeling and means nothing
// in any other.
__device__ inline void stamp_event_signature(DeviceState ds, EventId eid,
                                             EventSignatureKeys keys,
                                             uint64_t in_hash, uint64_t out_hash,
                                             StateId in_state, StateId out_state,
                                             uint32_t step, RuleId rule,
                                             DedupMap::DeviceView event_map) {
    DeviceEvent& ev = ds.event_pool.at(eid);
    uint32_t consumed_ranks[kMaxPatternEdges];
    uint32_t produced_ranks[kMaxPatternEdges];
    uint32_t fallbacks = 0;

    if (keys & hgcommon::EventKey_ConsumedEdges) {
        for (uint8_t i = 0; i < ev.num_consumed && i < kMaxPatternEdges; ++i) {
            uint32_t r = edge_rank_in_state_device(ds, in_state, ev.consumed_edges[i]);
            if (r == UINT32_MAX) { ++fallbacks; r = ev.consumed_edges[i]; }
            consumed_ranks[i] = r;
        }
    }
    if (keys & hgcommon::EventKey_ProducedEdges) {
        for (uint8_t i = 0; i < ev.num_produced && i < kMaxPatternEdges; ++i) {
            uint32_t r = edge_rank_in_state_device(ds, out_state, ev.produced_edges[i]);
            if (r == UINT32_MAX) { ++fallbacks; r = ev.produced_edges[i]; }
            produced_ranks[i] = r;
        }
    }
    if (fallbacks && ds.event_sig_raw_fallbacks)
        atomicAdd(ds.event_sig_raw_fallbacks, fallbacks);

    const uint64_t sig = hgcommon::event_signature(
        keys, in_hash, out_hash, step, rule,
        consumed_ranks, ev.num_consumed, produced_ranks, ev.num_produced);
    ev.signature = sig;

    // event_signature never returns 0 or the bare FNV offset, which is what keeps a signature
    // from colliding with the map's EMPTY and LOCKED sentinels -- a key equal to either is
    // silently never stored.
    auto r = event_map.insert_if_absent(sig, eid);
    if (r.inserted) {
        ev.canonical_id = INVALID_ID;
        if (ds.canonical_event_count) atomicAdd(ds.canonical_event_count, 1u);
    } else {
        ev.canonical_id = r.value;
    }
}

// Fill state_exact_hash (and state_edge_rank when the keys read it) for states [lo, hi).
//
// A phase of its own, because the rewrite kernel writes an event before the output state has
// been canonicalized and the signature cannot be filled inline there. Running it between
// hashing and dedup gives the stamping kernel both endpoint hashes.
//
// In Full state mode the exact hash is the mode's key and is already in state_canonical_hash;
// `key_is_exact` says so, and the pass then only has to produce ranks. `want_orbits`
// additionally scatters per-edge automorphism orbits (the quotient-causal DP's keys), and
// makes the pass run even under EventSignatureKeys None.
void fill_event_identity_inputs(EngineState& engine, uint32_t lo, uint32_t hi,
                                EventSignatureKeys keys, bool key_is_exact,
                                DeviceArena& arena, bool want_orbits = false);

// Stamp and deduplicate events [lo, hi). Reads the exact hashes and ranks the call above filled.
void stamp_event_identity_range(EngineState& engine, uint32_t lo, uint32_t hi,
                                EventSignatureKeys keys, DedupMap& event_map);

}  // namespace hg_gpu
