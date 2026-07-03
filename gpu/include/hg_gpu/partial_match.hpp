#pragma once

#include "hg_gpu/types.hpp"

#include <cstdint>

namespace hg_gpu {

// Per-warp mutable state for an in-progress pattern match. Lives in shared
// memory during MatchKernel execution. One PartialMatch per active warp.
//
// The match kernel DFS extends PartialMatch one pattern-edge at a time:
//   1. Pick an unmatched pattern edge.
//   2. Iterate candidate data edges from CHS.
//   3. For each candidate: try bind_pattern_edge; if successful, recurse.
//   4. On backtrack: unbind.
//
// Wolfram semantics are enforced by check_or_bind_var (variables may share
// vertices) and is_consumed / set_consumed (each pattern edge consumes a
// different data edge).
//
// Memory footprint per PartialMatch:
//   matched_edges:    16 × 4 =   64 B
//   consumed bitmap:  32 × 4 =  128 B
//   var_binding:      32 × 4 =  128 B
//   masks + counts:   ≈    8 B
//   DFS candidate cursors: kept outside PartialMatch so frame replay on
//     backtrack is trivial (pop-and-advance).
//   Total ≈ 328 B per warp.

struct PartialMatch {
    EdgeId   matched_edges[kMaxPatternEdges];  // pattern-edge-idx → data edge; INVALID_ID when unmatched
    uint32_t consumed[kMaxConsumedWords];      // bit j set iff data edge j is consumed by this match
    VertexId var_binding[kMaxVars];            // var-idx → vertex; INVALID_ID when unbound

    uint32_t matched_mask = 0;   // bit j set iff pattern edge j has a candidate bound
    uint32_t bound_mask   = 0;   // bit j set iff var j is bound
    uint8_t  num_pattern_edges = 0;
    uint8_t  num_vars          = 0;

    __device__ void reset(uint8_t n_pattern, uint8_t n_vars) {
        num_pattern_edges = n_pattern;
        num_vars          = n_vars;
        matched_mask = 0;
        bound_mask   = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kMaxPatternEdges; ++i) matched_edges[i] = INVALID_ID;
        #pragma unroll
        for (uint32_t i = 0; i < kMaxConsumedWords; ++i) consumed[i] = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kMaxVars; ++i) var_binding[i] = INVALID_ID;
    }

    // Cooperative reset: distribute element writes across a tile of threads.
    // Each thread handles a strided subset of the arrays.
    template <uint32_t TileSize>
    __device__ void reset_cooperative(uint32_t lane, uint8_t n_pattern, uint8_t n_vars) {
        if (lane == 0) {
            matched_mask = 0;
            bound_mask   = 0;
            num_pattern_edges = n_pattern;
            num_vars          = n_vars;
        }
        for (uint32_t i = lane; i < kMaxPatternEdges;  i += TileSize) matched_edges[i] = INVALID_ID;
        for (uint32_t i = lane; i < kMaxConsumedWords; i += TileSize) consumed[i]      = 0;
        for (uint32_t i = lane; i < kMaxVars;          i += TileSize) var_binding[i]   = INVALID_ID;
    }

    // -------------------------------------------------------------------------
    // Consumed-edges bitmap (one-to-one edge consumption per Wolfram semantics)
    // -------------------------------------------------------------------------

    // Consumed-edge check: scans matched_edges directly. Previously this
    // used a bitmap of kMaxConsumedBits=1024 bits, which overflows for any
    // state where edge IDs exceed 1024. Since a match consumes at most
    // kMaxPatternEdges data edges, a linear scan over matched_edges is
    // both correct for arbitrary edge IDs and cheap (≤16 comparisons).
    // The consumed[] array is retained for ABI stability but unused; will
    // be removed in a follow-up cleanup.
    __device__ bool is_consumed(EdgeId eid) const {
        uint32_t mm = matched_mask;
        while (mm) {
            int p = __ffs(mm) - 1;
            if (matched_edges[p] == eid) return true;
            mm &= mm - 1;
        }
        return false;
    }

    __device__ void set_consumed(EdgeId /*eid*/)   { /* handled by bind_pattern_edge */ }
    __device__ void clear_consumed(EdgeId /*eid*/) { /* handled by unbind_pattern_edge */ }

    // -------------------------------------------------------------------------
    // Variable bindings (Wolfram non-distinct semantics)
    // -------------------------------------------------------------------------

    __device__ bool is_var_bound(uint8_t v) const {
        return (bound_mask >> v) & 1u;
    }

    __device__ VertexId get_var(uint8_t v) const {
        return var_binding[v];
    }

    // Wolfram-style bind-or-check: if var v is already bound, verify the
    // attempted vertex matches the existing binding (vertices CAN repeat
    // across positions, so we don't forbid `vertex == some-other-var`).
    // Returns false only if v is bound to a DIFFERENT vertex.
    __device__ bool check_or_bind_var(uint8_t v, VertexId vertex) {
        if (is_var_bound(v)) return var_binding[v] == vertex;
        var_binding[v] = vertex;
        bound_mask |= (1u << v);
        return true;
    }

    __device__ void unbind_var(uint8_t v) {
        var_binding[v] = INVALID_ID;
        bound_mask &= ~(1u << v);
    }

    // -------------------------------------------------------------------------
    // Pattern-edge slot
    // -------------------------------------------------------------------------

    __device__ bool is_pattern_matched(uint8_t p) const {
        return (matched_mask >> p) & 1u;
    }

    __device__ void bind_pattern_edge(uint8_t p, EdgeId data_edge) {
        matched_edges[p] = data_edge;
        matched_mask |= (1u << p);
    }

    __device__ void unbind_pattern_edge(uint8_t p) {
        matched_edges[p] = INVALID_ID;
        matched_mask &= ~(1u << p);
    }

    __device__ bool is_complete() const {
        uint32_t full = (num_pattern_edges >= 32) ? 0xFFFFFFFFu : ((1u << num_pattern_edges) - 1u);
        return (matched_mask & full) == full;
    }
};

static_assert(sizeof(PartialMatch) <= 352, "PartialMatch grew unexpectedly; re-check shared-mem budget");

}  // namespace hg_gpu
