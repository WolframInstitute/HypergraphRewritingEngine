#pragma once
#include "hgcommon/namespace.hpp"
// Event identity on the device.
//
// The signature rule the persistent kernel stamps events with.
//
// The identity is defined over ISOMORPHISM CLASSES independently of how states are being
// identified (SPEC.md sec 4), so every component here reads DeviceState::state_exact_hash and
// DeviceState::state_edge_rank rather than the state mode's dedup key. The persistent kernel
// fills both when run_needs_exact_hash and run_needs_edge_ranks say the run reads them.

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/exploration.hpp"   // DedupMap
#include "hg_gpu/types.hpp"

#include "hgcommon/event_core.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

using hgcommon::event_keys_need_ranks;

// WHETHER THIS RUN MUST COMPUTE CANONICAL EDGE RANKS. Three things read them -- the event
// identity, the class-frame expansion, and the transition draw -- and the question is asked in
// TWO places: once on the host, to seed the root states, and once in the kernel, for every child
// it creates. Spelled twice they drifted the moment a third reader appeared: the roots were
// seeded without ranks under sampling, so the draw keyed the FIRST transition on absent ranks
// and the two engines disagreed about that one transition while agreeing everywhere else. The
// run is bimodal on it, so the symptom was CPU and GPU each keeping one of two subgraphs, at
// swapped parameters -- which reads like a seeding bug and is not one.
HG_HD inline bool run_needs_edge_ranks(EventSignatureKeys event_keys, bool expansion_enabled,
                                       double transition_rate, uint32_t num_rule_weights,
                                       uint32_t matches_per_state_rule) {
    return event_keys_need_ranks(event_keys) || expansion_enabled ||
           transition_rate < 1.0 || num_rule_weights != 0u || matches_per_state_rule != 0u;
}

// Whether this run reads each state's exact isomorphism hash (state_exact_hash): the event
// identity does, and so does the transition key of the draw, the spine and the per-state cap
// (transition_key_device). In Full mode it is the state's key; in None and Automatic it is a
// separate individualization-refinement pass.
HG_HD inline bool run_needs_exact_hash(EventSignatureKeys event_keys, double transition_rate,
                                       uint32_t num_rule_weights, uint32_t matches_per_state_rule) {
    return event_keys != hgcommon::EVENT_SIG_NONE || transition_rate < 1.0 ||
           num_rule_weights != 0u || matches_per_state_rule != 0u;
}

// Rank of `edge` inside `sid`, from the array the canonicalization pass filled. A linear scan
// over the state's own slice: slices are the size of a state's edge set and a rule consumes at
// most kMaxPatternEdges of them, so this is bounded by the rule rather than by the run.
//
// UINT32_MAX when the state has no ranks or the edge is not in it. The caller substitutes the
// raw edge id and counts it, because a signature built from an id is not an isomorphism
// invariant and a silent substitution would make that invisible.
__device__ __forceinline__ uint32_t edge_rank_in_state_device(const DeviceState& ds, StateId sid,
                                                              EdgeId edge) {
    if (!ds.state_edge_rank || sid >= ds.max_states) return UINT32_MAX;
    StateEdgeSlice sl = ds.state_edge_slices[sid];
    for (uint32_t k = 0; k < sl.count; ++k)
        if (ds.state_edge_ids[sl.offset + k] == edge) return ds.state_edge_rank[sl.offset + k];
    return UINT32_MAX;
}

// The signature values of stored event `eid` (hgcommon::event_signature_values under `keys`):
// its states' exact hashes, its step, its rule, and the ranks of its consumed and produced edges,
// each resolved in the state it belongs to -- consumed in the input, produced in the output --
// because a rank is a position in THAT state's canonical labeling and means nothing in any
// other. A missing rank stands in the raw edge id; `fallbacks` counts them.
__device__ inline uint32_t event_values_device(const DeviceState& ds, EventId eid,
                                               EventSignatureKeys keys, uint64_t* out,
                                               uint32_t& fallbacks) {
    const DeviceEvent& ev = ds.event_pool.at(eid);
    uint32_t consumed_ranks[kMaxPatternEdges];
    uint32_t produced_ranks[kMaxPatternEdges];
    const uint8_t nc = ev.num_consumed < kMaxPatternEdges ? ev.num_consumed
                                                          : static_cast<uint8_t>(kMaxPatternEdges);
    const uint8_t np = ev.num_produced < kMaxPatternEdges ? ev.num_produced
                                                          : static_cast<uint8_t>(kMaxPatternEdges);
    if (keys & hgcommon::EventKey_ConsumedEdges) {
        for (uint8_t i = 0; i < nc; ++i) {
            const EdgeId e = event_consumed_edge(ds, ev, i);
            uint32_t r = edge_rank_in_state_device(ds, ev.input_state, e);
            if (r == UINT32_MAX) { ++fallbacks; r = e; }
            consumed_ranks[i] = r;
        }
    }
    if (keys & hgcommon::EventKey_ProducedEdges) {
        for (uint8_t i = 0; i < np; ++i) {
            const EdgeId e = ev.first_produced + i;
            uint32_t r = edge_rank_in_state_device(ds, ev.output_state, e);
            if (r == UINT32_MAX) { ++fallbacks; r = e; }
            produced_ranks[i] = r;
        }
    }
    return hgcommon::event_signature_values(
        keys, ds.state_exact_hash[ev.input_state], ds.state_exact_hash[ev.output_state],
        ev.step, ev.rule, consumed_ranks, nc, produced_ranks, np, out);
}

// Stamp one event with the identity the run's key set asks for, and APPLY that identity: two
// applications whose signature values agree are the same event, so the second to arrive records
// the first as its canonical id. The signature selects the key and a key hit compares the values
// recomputed from the class's first event (the host's Hypergraph::claim_event); the class's key
// is the signature stamped. Both endpoint states' exact hashes are published before this runs.
__device__ inline void stamp_event_signature(const DeviceState& ds, EventId eid,
                                             EventSignatureKeys keys,
                                             DedupMap::DeviceView event_map) {
    uint64_t values[hgcommon::EVENT_SIG_MAX_VALUES];
    uint32_t fallbacks = 0;
    const uint32_t n = event_values_device(ds, eid, keys, values, fallbacks);
    if (fallbacks && ds.event_sig_raw_fallbacks)
        atomicAdd(ds.event_sig_raw_fallbacks, fallbacks);
    const uint64_t sig = hgcommon::event_signature_of_values(values, n);

    struct P {
        const DeviceState& ds;
        EventSignatureKeys keys;
        EventId eid;
        const uint64_t* values;
        uint32_t n;
        __device__ bool same(uint32_t rep) const {
            uint64_t theirs[hgcommon::EVENT_SIG_MAX_VALUES];
            uint32_t unused = 0;
            if (event_values_device(ds, rep, keys, theirs, unused) != n) return false;
            for (uint32_t i = 0; i < n; ++i)
                if (theirs[i] != values[i]) return false;
            return true;
        }
        __device__ bool make(uint32_t& v) const { v = eid; __threadfence(); return true; }
        __device__ uint32_t rep_of(uint32_t v) const { return v; }
    } p{ds, keys, eid, values, n};
    const StateClaim c = keyed_claim_device(ds, eid, sig & ds.event_key_mask, event_map, p);

    DeviceEvent& ev = ds.event_pool.at(eid);
    ev.signature = c.key;
    if (c.fresh) {
        ev.canonical_id = INVALID_ID;
        if (ds.canonical_event_count) atomicAdd(ds.canonical_event_count, 1u);
    } else {
        ev.canonical_id = c.canonical;
    }
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
