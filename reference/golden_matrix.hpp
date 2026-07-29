#pragma once
// The cached golden matrix: every corpus workload across every identity mode, with what each
// row was checked against recorded next to it.
//
// WHY CACHED. The brute-force oracle is O(V! * E log E), so running it inside the test caps the
// depth at whatever keeps states under 8 vertices, and the WL reference needs wolframscript,
// which is slow and intermittently unlicensed. Neither can run on every build. Caching the
// expected values lets the gate compare in milliseconds and lets the expensive checks run
// deliberately, when the golden is regenerated.
//
// WHY PROVENANCE IS PER ROW, AND NOT A FOOTNOTE. A golden generated from the engine and then
// compared against the engine proves stability, not correctness: it pins whatever the engine
// did, bug included. So each row says which of these it is, and the gate reports the mix rather
// than letting a pin pass for an oracle:
//
//   Oracle     cross-checked against reference/oracle_corpus.hpp's brute-force isomorphism
//              count, which shares no code with the engine's canonicalization. Available only
//              where every reachable state stays inside the brute force's vertex bound.
//   Reference  cross-checked against reference/MultiwaySystem.wl, itself cross-checked against
//              the authoritative Wolfram/Multicomputation paclet.
//   Pin        engine output. A regression tripwire, and nothing more. A cell that can only be
//              pinned is a cell with no independent check, which is a gap to be named.
//
// A row's expectation is a FINGERPRINT, not a count. Two different state sets of the same size
// would satisfy a count, and the modes here differ precisely in which states they identify, so
// counts are the one thing least able to tell the cells apart.

#include "hypergraph/parallel_evolution.hpp"

#include <cstdint>
#include <string>

namespace golden {

using namespace hypergraph;

enum class Provenance : uint8_t { Oracle, Reference, Pin };

inline const char* provenance_name(Provenance p) {
    switch (p) {
        case Provenance::Oracle:    return "oracle";
        case Provenance::Reference: return "reference";
        case Provenance::Pin:       return "pin";
    }
    return "?";
}

// One cell of the matrix: a workload evolved under one identity configuration.
struct Row {
    std::string case_name;
    StateCanonicalizationMode state_mode;
    EventSignatureKeys        event_keys;
    bool                      quotient;
    uint32_t                  steps;

    uint64_t states;
    uint64_t events;
    uint64_t causal_edges;
    uint64_t causal_event_pairs;
    uint64_t branchial_edges;
    // Order-independent digest of the multiset of canonical state hashes. This is what makes
    // the row about WHICH states, not how many.
    uint64_t state_fingerprint;

    Provenance provenance;
};

inline const char* state_mode_name(StateCanonicalizationMode m) {
    switch (m) {
        case StateCanonicalizationMode::None:      return "None";
        case StateCanonicalizationMode::Automatic: return "Automatic";
        case StateCanonicalizationMode::Full:      return "Full";
    }
    return "?";
}

inline bool state_mode_from_name(const std::string& s, StateCanonicalizationMode& out) {
    if (s == "None")      { out = StateCanonicalizationMode::None;      return true; }
    if (s == "Automatic") { out = StateCanonicalizationMode::Automatic; return true; }
    if (s == "Full")      { out = StateCanonicalizationMode::Full;      return true; }
    return false;
}

inline const char* event_keys_name(EventSignatureKeys k) {
    if (k == EVENT_SIG_NONE)      return "None";
    if (k == EVENT_SIG_AUTOMATIC) return "Automatic";
    if (k == EVENT_SIG_FULL)      return "Full";
    return "?";
}

inline bool event_keys_from_name(const std::string& s, EventSignatureKeys& out) {
    if (s == "None")      { out = EVENT_SIG_NONE;      return true; }
    if (s == "Automatic") { out = EVENT_SIG_AUTOMATIC; return true; }
    if (s == "Full")      { out = EVENT_SIG_FULL;      return true; }
    return false;
}

// Order-independent over states, because the order they are created in is the scheduler's
// business and changes with the worker count. Commutative combination means a permutation of
// the same states digests the same, while a different state set does not.
inline uint64_t fold_fingerprint(uint64_t acc, uint64_t state_hash) {
    uint64_t x = state_hash;
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return acc + x;   // addition: commutative, and sensitive to multiplicity
}

}  // namespace golden
