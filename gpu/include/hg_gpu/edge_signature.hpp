#pragma once

#include "hg_gpu/types.hpp"

#include <cstdint>

namespace hg_gpu {

// Vertex-repetition pattern signature, mirroring the CPU EdgeSignature in
// hypergraph/include/hypergraph/signature.hpp. Two edges have the same
// signature iff their vertices repeat in the same positional pattern,
// regardless of the actual vertex IDs:
//
//   {3, 3, 4} → pattern [0, 0, 1]   (positions 0,1 same; 2 different)
//   {7, 7, 8} → pattern [0, 0, 1]   (same signature)
//   {5, 6, 8} → pattern [0, 1, 2]
//   {1, 1, 1} → pattern [0, 0, 0]
//   {a, b, a} → pattern [0, 1, 0]
//
// Used for signature-partitioned candidate generation in the match kernel.
// The hash is FNV-1a over (arity, pattern_bytes), kept bit-identical with
// the CPU implementation so candidate sets agree across engines.

struct EdgeSignature {
    uint8_t arity = 0;
    uint8_t pattern[kMaxArity] = {0};

    __host__ __device__ bool operator==(const EdgeSignature& o) const {
        if (arity != o.arity) return false;
        for (uint8_t i = 0; i < arity; ++i) {
            if (pattern[i] != o.pattern[i]) return false;
        }
        return true;
    }
};

// Compute the signature pattern from a vertex tuple.
__host__ __device__ inline EdgeSignature signature_from_vertices(
    const VertexId* vertices, uint8_t arity) {
    EdgeSignature sig;
    sig.arity = arity;
    if (arity == 0) return sig;

    VertexId seen   [kMaxArity];
    uint8_t  labels [kMaxArity];
    uint8_t  next_label = 0;

    for (uint8_t i = 0; i < arity; ++i) {
        VertexId v = vertices[i];
        uint8_t label = next_label;
        for (uint8_t j = 0; j < next_label; ++j) {
            if (seen[j] == v) { label = labels[j]; break; }
        }
        if (label == next_label) {
            seen[next_label]   = v;
            labels[next_label] = next_label;
            ++next_label;
        }
        sig.pattern[i] = label;
    }
    return sig;
}

__host__ __device__ inline uint64_t signature_hash(const EdgeSignature& sig) {
    constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
    constexpr uint64_t kFnvPrime  = 1099511628211ULL;
    uint64_t h = kFnvOffset;
    h ^= sig.arity;
    h *= kFnvPrime;
    for (uint8_t i = 0; i < sig.arity; ++i) {
        h ^= sig.pattern[i];
        h *= kFnvPrime;
    }
    return h;
}

__host__ __device__ inline uint64_t signature_hash_from_vertices(
    const VertexId* vertices, uint8_t arity) {
    return signature_hash(signature_from_vertices(vertices, arity));
}

// Compatibility predicate: a data edge with sig `data` can match a pattern
// edge with sig `pattern` iff for every two positions i,j, pattern[i]=pattern[j]
// implies data[i]=data[j]. Different pattern vars MAY (but need not) collapse
// to the same data vertex — this is Wolfram non-distinct binding semantics.
__host__ __device__ inline bool signature_compatible(const EdgeSignature& data,
                                                     const EdgeSignature& pattern) {
    if (data.arity != pattern.arity) return false;
    for (uint8_t i = 0; i < pattern.arity; ++i) {
        for (uint8_t j = i + 1; j < pattern.arity; ++j) {
            if (pattern.pattern[i] == pattern.pattern[j] &&
                data.pattern[i]    != data.pattern[j]) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace hg_gpu
