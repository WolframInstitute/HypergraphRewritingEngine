#pragma once
// Shared CPU/GPU core definitions — the single source of truth for identifiers,
// fixed-size limits, and integer hash primitives used by both the host engine
// (namespace hypergraph) and the CUDA port (namespace hg_gpu).
//
// CUDA-safe: depends only on <cstdint>. No <atomic>/<vector>/<algorithm>, so it
// includes cleanly from .cu translation units. Functions are annotated HG_HD
// (__host__ __device__ under nvcc, empty otherwise) so one definition serves both.
#include <cstdint>

#if defined(__CUDACC__)
  #define HG_HD __host__ __device__
#else
  #define HG_HD
#endif

// For a function whose inlining must not be at the mercy of unrelated code size. `inline` is
// only a hint, and GCC's budget is per translation unit: adding template instantiations to a
// widely-included header tightens it everywhere in that unit, and something already inlined
// stops being inlined. See SparseBitset::contains for the case that cost 3.7% of an evolution
// without changing any arithmetic.
#if defined(_MSC_VER) && !defined(__clang__)
  #define HG_INLINE __forceinline
#else
  #define HG_INLINE inline __attribute__((always_inline))
#endif

namespace hgcommon {

// Identifiers — all 32-bit (4 billion is ample and halves cache pressure vs 64-bit).
using VertexId = uint32_t;
using EdgeId   = uint32_t;
using StateId  = uint32_t;
using EventId  = uint32_t;
using MatchId  = uint32_t;

constexpr uint32_t INVALID_ID = 0xFFFFFFFFu;  // == UINT32_MAX

// Fixed-size structural limits (stack/shared-memory buffers rely on these).
constexpr uint8_t MAX_ARITY         = 16;
constexpr uint8_t MAX_PATTERN_EDGES = 16;
constexpr uint8_t MAX_VARS          = 32;

// FNV-1a constants.
constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;  // 14695981039346656037
constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;       // 1099511628211

// MurmurHash3 finalizer — avalanche a small raw integer (e.g. a vertex id).
HG_HD inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// FNV-1a combine of an already well-distributed value into an accumulator.
HG_HD inline uint64_t fnv_hash(uint64_t h, uint64_t value) {
    h ^= value;
    h *= FNV_PRIME;
    return h;
}

// splitmix64 finalizer — strong avalanche, so a commutative SUM of these over a
// multiset is an order-independent, collision-resistant hash (used by WL folds).
HG_HD inline uint64_t splitmix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// A map key built from engine ids.
//
// The map reserves EMPTY(0) and LOCKED and REJECTS a key equal to either, so an id packed raw
// collides exactly when the id is zero -- which is the first state, the first edge and the
// first event. Offsetting each id by one is injective and cannot produce zero: the high word is
// at least one. Ids are engine-minted and bounded well below INVALID_ID, so neither offset can
// wrap into LOCKED.
//
// Every map keyed by ids goes through this, on BOTH engines. Two sites packing ids their own
// way is how one of them ends up without the offset.
HG_HD inline constexpr uint64_t id_key(uint32_t a) {
    return static_cast<uint64_t>(a) + 1;
}
HG_HD inline constexpr uint64_t id_key(uint32_t a, uint32_t b) {
    return ((static_cast<uint64_t>(a) + 1) << 32) | (static_cast<uint64_t>(b) + 1);
}
HG_HD inline constexpr uint32_t id_from_key(uint64_t k) {
    return static_cast<uint32_t>(k - 1);
}

// The inverse of the two-id form, beside it for the same reason the forward map is shared: a
// reader that unpacks a packed pair its own way is a second implementation of the packing, and
// the offset is exactly what such a copy leaves out.
struct IdPair { uint32_t a, b; };
HG_HD inline constexpr IdPair id_pair_from_key(uint64_t k) {
    return IdPair{ static_cast<uint32_t>(k >> 32) - 1u,
                   static_cast<uint32_t>(k & 0xFFFFFFFFu) - 1u };
}

// What a run must RECORD, as against what it will later SERIALIZE.
//
// An artifact turned off here is never built: the rendezvous that would produce it does not
// run, and the structures behind it stay empty. That is a different question from which
// artifacts a caller asks to be written out, which is what the FFI's include_* flags decide --
// a run can be asked for states alone and still have paid for the whole causal graph.
//
// Everything defaults to on, so a caller that states nothing gets exactly what it got before.
// States and events are not listed: the evolution IS the states and the events, so there is no
// run that skips them.
struct RecordSet {
    // The causal relation: which event produced the edge another consumed.
    bool causal = true;
    // The branchial PAIR relation: two events that consumed a common edge of one state.
    bool branchial = true;
    // The per-state event list. Read only by an all-siblings view of the branchial state
    // graph -- every pair of output states of one input state, with no overlap test -- which
    // is a different question from the pair relation above and is answered without it.
    bool state_events = true;
};


// Canonical hash of the state holding no edges. Any rule whose RHS is empty reaches it, so it
// is an ordinary canonical form and needs a hash of its own -- a canonicalizer given no edges
// has nothing to compute one from, so it is reserved rather than derived.
//
// It cannot be 0, which carries two other meanings a state hash must stay clear of: 0 means
// "not computed yet" in both engines' per-state hash array, and 0 is the EMPTY sentinel of
// every map this hash keys, so a 0-hashed state is unstorable in all of them.
//
// The value is arbitrary -- the fractional part of the golden ratio, a mixing constant already
// in this file -- and only has to be non-zero, fixed, and THE SAME ON BOTH DEVICES, since a
// caller compares canonical hashes across runs and across devices.
inline constexpr uint64_t EMPTY_STATE_CANONICAL_HASH = 0x9E3779B97F4A7C15ULL;

}  // namespace hgcommon
