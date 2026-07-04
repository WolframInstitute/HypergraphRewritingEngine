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

}  // namespace hgcommon
