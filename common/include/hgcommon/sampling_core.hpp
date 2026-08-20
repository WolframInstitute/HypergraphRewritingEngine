#pragma once
#include "hgcommon/namespace.hpp"
// THE SAMPLING DECISIONS, one body for host and device.
//
// Which transitions a thinned run keeps is a DECISION, not a storage question, so it is spelled
// once. It was spelled once and only the host could reach it: `TransitionRate` and `RuleWeights`
// were accepted, applied on the CPU, and reported as unimplemented on the GPU -- not because the
// device lacks anything, but because the rule lived in ParallelEvolutionEngine where no kernel
// can call it. Everything below is a pure function of values both engines already hold.
//
// THE DRAW IS KEYED ON THE TRANSITION, NEVER ON THREAD STATE. Drawing from a worker RNG would
// make the surviving subgraph depend on which thread reached the transition first, so the same
// run would sample differently at a different thread count -- and on a different DEVICE -- and
// "representative sample" would have nothing stable to be representative of. Keyed this way, the
// same seed gives the same subgraph at any worker count and on either engine, which is what
// makes a CPU/GPU differential test of a SAMPLED run meaningful at all.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// The rate a rule's transitions survive at: the run's rate scaled by that rule's weight, clamped.
// `weights` may be null (every rule weighted 1) or shorter than the rule set (rules past its end
// are weighted 1), which is what a partial override means on the WL side.
HG_HD inline double sampling_rate_for_rule(double transition_rate, const double* weights,
                                           uint32_t num_weights, uint32_t rule) {
    double w = 1.0;
    if (weights != nullptr && rule < num_weights) w = weights[rule];
    const double r = transition_rate * w;
    return r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
}

// Does this transition survive the draw?
//
// `transition_key` is the transition's isomorphism-invariant identity -- on both engines
// event_signature(EVENT_SIG_TRANSITION, ...) over the input state's canonical hash, the rule, and
// the consumed edges' canonical ranks WITHIN that state. Two runs that reach the same transition
// compute the same key, which is the whole point.
HG_HD inline bool transition_survives(uint64_t transition_key, uint64_t random_seed, double rate) {
    if (rate >= 1.0) return true;
    if (rate <= 0.0) return false;

    // splitmix64 of (seed, transition). The seed is mixed by multiplication rather than xor so
    // that seed 0 is not the identity -- a caller who leaves the seed alone still gets a draw.
    uint64_t x = transition_key ^ (random_seed * 0x9E3779B97F4A7C15ULL);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= (x >> 31);

    // Compared in [0,1) via the top 53 bits, so the threshold means the same thing at any rate.
    const double u = static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
    return u < rate;
}

// THE ORDER A PER-(state, rule) CAP KEEPS ITS k IN. Seeded, so a different seed keeps a
// different k rather than the same skeleton every run, and derived from the transition's own
// identity, so the kept set does not depend on which worker or which DEVICE reached it first.
//
// A DIFFERENT MIX FROM transition_survives, deliberately: the same key must not produce a rank
// correlated with whether it survived a rate draw, or the two controls would compound instead
// of composing.
HG_HD inline uint64_t transition_rank(uint64_t transition_key, uint64_t random_seed) {
    uint64_t x = transition_key ^ (random_seed * 0x9E3779B97F4A7C15ULL) ^ 0xA5A5A5A5A5A5A5A5ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Whether ANY draw can fail. Testing `transition_rate < 1` alone would skip sampling entirely for
// a caller who left the rate at 1 and weighted a single rule to zero.
HG_HD inline bool sampling_active(double transition_rate, const double* weights,
                                  uint32_t num_weights, uint32_t matches_per_state_rule) {
    if (transition_rate < 1.0) return true;
    if (matches_per_state_rule != 0) return true;
    for (uint32_t i = 0; weights != nullptr && i < num_weights; ++i)
        if (weights[i] < 1.0) return true;
    return false;
}

}  // namespace common
}  // namespace HG_NAMESPACE
