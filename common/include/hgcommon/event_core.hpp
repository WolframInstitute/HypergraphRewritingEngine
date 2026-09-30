#pragma once
#include "hgcommon/namespace.hpp"
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

namespace HG_NAMESPACE {
namespace common {

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

// The identity of a transition BEFORE it is applied: which state, which rule, which edges.
//
// It is the only point of this lattice available at MATCH time, since the output state does
// not exist yet -- so it is the key a sampler must use. Every component is
// isomorphism-invariant, which is the property that matters: a sampler keyed on it selects the
// same subgraph however the run was scheduled, on however many threads, on either device.
// Keyed on anything run-local (a raw state id, a worker's RNG) it would select a different
// subgraph each run, and a sample that cannot be reproduced cannot be compared against the
// evolution it claims to represent.
constexpr EventSignatureKeys EVENT_SIG_TRANSITION =
    EventKey_InputState | EventKey_Rule | EventKey_ConsumedEdges;

// The values an event's signature is taken over, in order: the selected keys' fields, one value
// per rank. Ranks are consumed IN ORDER -- match order for the consumed edges, RHS order for the
// produced -- because Positional identity distinguishes which role an edge played, not merely
// which edges took part. Two events are the same under `keys` exactly when these values are
// equal; the signature is a 64-bit digest of them, and a table keyed by it compares the values
// on a hit (hgcommon/canonical_form_core.hpp, two words per value).
constexpr uint32_t EVENT_SIG_MAX_VALUES = 4u + 2u * MAX_PATTERN_EDGES;

HG_HD inline uint32_t event_signature_values(
    EventSignatureKeys keys,
    uint64_t input_state_hash, uint64_t output_state_hash,
    uint32_t step, uint16_t rule_index,
    const uint32_t* consumed_ranks, uint8_t num_consumed,
    const uint32_t* produced_ranks, uint8_t num_produced,
    uint64_t* out)
{
    uint32_t n = 0;
    if (keys & EventKey_InputState)  out[n++] = input_state_hash;
    if (keys & EventKey_OutputState) out[n++] = output_state_hash;
    if (keys & EventKey_Step)        out[n++] = static_cast<uint64_t>(step);
    if (keys & EventKey_Rule)        out[n++] = static_cast<uint64_t>(rule_index);
    if (keys & EventKey_ConsumedEdges)
        for (uint8_t i = 0; i < num_consumed; ++i) out[n++] = consumed_ranks[i];
    if (keys & EventKey_ProducedEdges)
        for (uint8_t i = 0; i < num_produced; ++i) out[n++] = produced_ranks[i];
    return n;
}

// The signature of those values. Never 0 and never the bare FNV offset: both are reserved by the
// maps that key on it, and a signature equal to a sentinel is never stored.
HG_HD inline uint64_t event_signature_of_values(const uint64_t* values, uint32_t n) {
    uint64_t sig = FNV_OFFSET;
    for (uint32_t i = 0; i < n; ++i) sig = fnv_hash(sig, values[i]);
    if (sig == 0 || sig == FNV_OFFSET) sig = 1;
    return sig;
}

// Signature of one application: event_signature_of_values over event_signature_values.
HG_HD inline uint64_t event_signature(
    EventSignatureKeys keys,
    uint64_t input_state_hash, uint64_t output_state_hash,
    uint32_t step, uint16_t rule_index,
    const uint32_t* consumed_ranks, uint8_t num_consumed,
    const uint32_t* produced_ranks, uint8_t num_produced)
{
    uint64_t v[EVENT_SIG_MAX_VALUES];
    const uint32_t n = event_signature_values(keys, input_state_hash, output_state_hash, step,
                                              rule_index, consumed_ranks, num_consumed,
                                              produced_ranks, num_produced, v);
    return event_signature_of_values(v, n);
}

// The identity words of `n` signature values: low word, then high word, per value.
HG_HD inline uint32_t event_identity_words(const uint64_t* values, uint32_t n, uint32_t* out) {
    for (uint32_t i = 0; i < n; ++i) {
        out[2 * i]     = static_cast<uint32_t>(values[i]);
        out[2 * i + 1] = static_cast<uint32_t>(values[i] >> 32);
    }
    return 2u * n;
}

// True when the key set reads per-edge canonical ranks.
HG_HD inline bool event_keys_need_ranks(EventSignatureKeys keys) {
    return (keys & (EventKey_ConsumedEdges | EventKey_ProducedEdges)) != 0;
}

}  // namespace common
}  // namespace HG_NAMESPACE
