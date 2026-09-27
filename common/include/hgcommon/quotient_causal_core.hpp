#pragma once
#include "hgcommon/namespace.hpp"
// The quotient replay's (class, depth) key and the survivor walk its match capture uses, one
// body for host and device.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// The (class, depth) key. Shared because host and device index one set each: the replay's
// instances and the multiplicity points by qc_key(state, depth, 0). Two spellings of a key are
// two key spaces.
HG_HD inline uint64_t qc_key(uint64_t state_hash, uint32_t depth, uint32_t orbit) {
    uint64_t h = FNV_OFFSET;
    h ^= state_hash; h *= FNV_PRIME;
    h ^= (static_cast<uint64_t>(depth) << 32) | orbit; h *= FNV_PRIME;
    return h;
}

// The edges an event carries across unchanged -- every edge of its output state it did not
// produce -- as index pairs into the input and output states' edge arrays, which both list
// edges in ascending id, so one merge walk pairs them. An output edge the event did not produce
// is an input edge by construction; the walk skips one that is not. ONE BODY on both engines:
// each engine's match capture reads slots through the indices.
template <class F>
HG_HD inline void qc_for_each_survivor(const EdgeId* in_edges, uint32_t in_n,
                                       const EdgeId* out_edges, uint32_t out_n,
                                       const EdgeId* produced, uint32_t num_produced, F&& f) {
    for (uint32_t i = 0, j = 0; i < out_n; ++i) {
        const EdgeId oe = out_edges[i];
        bool produced_here = false;
        for (uint32_t k = 0; k < num_produced; ++k)
            if (produced[k] == oe) { produced_here = true; break; }
        if (produced_here) continue;
        while (j < in_n && in_edges[j] < oe) ++j;
        if (j < in_n && in_edges[j] == oe) f(j, i);
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
