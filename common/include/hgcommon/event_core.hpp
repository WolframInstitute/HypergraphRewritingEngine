#pragma once
// Shared CPU/GPU event identity.
//
// Two applications are the same EVENT when their signatures agree. Which components go into
// that signature is the event-identity axis of SPEC.md sec 4.2, selected by the key bits
// below, and it is a refinement lattice: ByEndpointStates keys on the two canonical states
// alone, ByConsumedProducedEdges also on which edges moved, DistinctApplications keeps every
// application apart by computing no signature at all.
//
// Every component is a quantity BOTH devices can produce: canonical state hashes, the step,
// the rule, and the canonical RANKS of the consumed and produced edges. Ranks rather than edge
// ids because an id is run-local and carries no isomorphism meaning, while a rank is the
// position an edge takes in its own state's canonical labeling -- so a signature built from
// them is a property of the event, not of the schedule that produced it, and does not move
// when the state-identity mode does.
//
// The caller resolves the ranks. That keeps this a pure function and leaves the host free to
// count the cases where a rank was unavailable rather than substitute one silently.

#include <cstdint>
#include "hgcommon/core.hpp"

namespace hgcommon {

// Components of an event signature. The presets below name the points of the lattice.
enum EventSignatureKey : uint8_t {
    EventKey_InputState     = 1 << 0,  // canonical input state
    EventKey_OutputState    = 1 << 1,  // canonical output state
    EventKey_Step           = 1 << 2,
    EventKey_Rule           = 1 << 3,
    EventKey_ConsumedEdges  = 1 << 4,  // canonical ranks of the consumed edges, in match order
    EventKey_ProducedEdges  = 1 << 5,  // canonical ranks of the produced edges, in RHS order
};

using EventSignatureKeys = uint8_t;

constexpr EventSignatureKeys EVENT_SIG_NONE = 0;
constexpr EventSignatureKeys EVENT_SIG_FULL =
    EventKey_InputState | EventKey_OutputState;
constexpr EventSignatureKeys EVENT_SIG_AUTOMATIC =
    EventKey_InputState | EventKey_OutputState | EventKey_Step |
    EventKey_ConsumedEdges | EventKey_ProducedEdges;

// Signature of one application. Ranks are consumed IN ORDER -- match order for the consumed
// edges, RHS order for the produced -- because Positional identity distinguishes which role an
// edge played, not merely which edges took part.
//
// The result is never 0 and never the bare FNV offset: both are reserved by the maps that key
// on it, and a signature colliding with a sentinel would be dropped rather than stored.
HG_HD inline uint64_t event_signature(
    EventSignatureKeys keys,
    uint64_t input_state_hash, uint64_t output_state_hash,
    uint32_t step, uint16_t rule_index,
    const uint32_t* consumed_ranks, uint8_t num_consumed,
    const uint32_t* produced_ranks, uint8_t num_produced)
{
    uint64_t sig = FNV_OFFSET;
    if (keys & EventKey_InputState)  sig = fnv_hash(sig, input_state_hash);
    if (keys & EventKey_OutputState) sig = fnv_hash(sig, output_state_hash);
    if (keys & EventKey_Step)        sig = fnv_hash(sig, static_cast<uint64_t>(step));
    if (keys & EventKey_Rule)        sig = fnv_hash(sig, static_cast<uint64_t>(rule_index));
    if (keys & EventKey_ConsumedEdges)
        for (uint8_t i = 0; i < num_consumed; ++i)
            sig = fnv_hash(sig, static_cast<uint64_t>(consumed_ranks[i]));
    if (keys & EventKey_ProducedEdges)
        for (uint8_t i = 0; i < num_produced; ++i)
            sig = fnv_hash(sig, static_cast<uint64_t>(produced_ranks[i]));
    if (sig == 0 || sig == FNV_OFFSET) sig = 1;
    return sig;
}

}  // namespace hgcommon
