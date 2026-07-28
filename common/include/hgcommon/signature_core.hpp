#pragma once
// Shared CPU/GPU edge-signature rules.
//
// An edge's signature is its vertex-REPETITION pattern: position i gets the label of the
// first position holding the same vertex. It decides which data edges are even considered as
// candidates for a pattern edge, so if the two ports disagree about it they silently search
// different candidate sets and therefore find different matches -- a divergence that would
// show up as missing events rather than as an error.
//
// The rules operate on (arity, pattern[]) directly rather than on a struct, so each port
// keeps its own EdgeSignature layout and only the LOGIC is shared. Pure functions over
// caller-provided arrays: no allocation, no synchronisation, host and device from one
// definition.

#include <cstdint>
#include "hgcommon/core.hpp"

namespace hgcommon {

// Repetition pattern of a vertex tuple. out_pattern must hold at least `arity` bytes.
// {a,b,a} -> {0,1,0}; {a,a,a} -> {0,0,0}; {a,b,c} -> {0,1,2}.
HG_HD inline void signature_pattern_from_vertices(
    const VertexId* vertices, uint8_t arity, uint8_t* out_pattern)
{
    uint8_t next_label = 0;
    VertexId seen[MAX_ARITY];
    uint8_t labels[MAX_ARITY];
    for (uint8_t i = 0; i < arity; ++i) {
        const VertexId v = vertices[i];
        uint8_t label = next_label;
        for (uint8_t j = 0; j < next_label; ++j) {
            if (seen[j] == v) { label = labels[j]; break; }
        }
        if (label == next_label) {
            seen[next_label] = v;
            labels[next_label] = next_label;
            ++next_label;
        }
        out_pattern[i] = label;
    }
}

// FNV-1a over (arity, pattern bytes). Both ports index candidates by this, so it has to agree
// bit for bit, not merely be "a hash of the same thing".
HG_HD inline uint64_t signature_hash(uint8_t arity, const uint8_t* pattern) {
    uint64_t h = FNV_OFFSET;
    h ^= arity;
    h *= FNV_PRIME;
    for (uint8_t i = 0; i < arity; ++i) {
        h ^= pattern[i];
        h *= FNV_PRIME;
    }
    return h;
}

// A data edge can match a pattern edge exactly when every repetition the PATTERN demands is
// present in the DATA. The converse is not required: two distinct pattern variables may bind
// the same vertex, which is Wolfram's non-distinct binding semantics, so data repetition the
// pattern does not ask for is not a conflict.
HG_HD inline bool signature_compatible(
    uint8_t data_arity, const uint8_t* data_pattern,
    uint8_t pattern_arity, const uint8_t* pattern_pattern)
{
    if (data_arity != pattern_arity) return false;
    for (uint8_t i = 0; i < pattern_arity; ++i) {
        for (uint8_t j = i + 1; j < pattern_arity; ++j) {
            if (pattern_pattern[i] == pattern_pattern[j] &&
                data_pattern[i] != data_pattern[j]) {
                return false;
            }
        }
    }
    return true;
}

// Distinct vertices the signature describes: the largest label plus one.
HG_HD inline uint8_t signature_num_distinct(uint8_t arity, const uint8_t* pattern) {
    if (arity == 0) return 0;
    uint8_t max_label = 0;
    for (uint8_t i = 0; i < arity; ++i) if (pattern[i] > max_label) max_label = pattern[i];
    return static_cast<uint8_t>(max_label + 1);
}

}  // namespace hgcommon
