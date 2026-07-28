#pragma once

#include "hg_gpu/types.hpp"

#include "hgcommon/signature_core.hpp"  // the rules themselves, shared with the host

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

// The rules themselves live in hgcommon/signature_core.hpp so the host runs the same ones;
// what stays here is only this port's EdgeSignature layout and the wrappers onto it.
__host__ __device__ inline EdgeSignature signature_from_vertices(
    const VertexId* vertices, uint8_t arity) {
    EdgeSignature sig;
    sig.arity = arity;
    hgcommon::signature_pattern_from_vertices(vertices, arity, sig.pattern);
    return sig;
}

__host__ __device__ inline uint64_t signature_hash(const EdgeSignature& sig) {
    return hgcommon::signature_hash(sig.arity, sig.pattern);
}

__host__ __device__ inline uint64_t signature_hash_from_vertices(
    const VertexId* vertices, uint8_t arity) {
    return signature_hash(signature_from_vertices(vertices, arity));
}

__host__ __device__ inline bool signature_compatible(const EdgeSignature& data,
                                                     const EdgeSignature& pattern) {
    return hgcommon::signature_compatible(data.arity, data.pattern,
                                          pattern.arity, pattern.pattern);
}

}  // namespace hg_gpu
