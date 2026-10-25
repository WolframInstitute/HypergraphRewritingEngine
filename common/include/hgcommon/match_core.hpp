#pragma once
#include "hgcommon/namespace.hpp"
// Shared CPU/GPU matching semantics.
//
// What CONSTITUTES a match is the rewrite system; how candidates are enumerated is the
// hardware. This header holds the first. Both ports had their own copy of it -- the host in
// pattern_matcher.hpp's validate_candidate and the device matcher both reach it through join_core
// plus an arity check at each of its three call sites -- and two copies of a semantic rule is
// how the two devices come to disagree about what a match is.
//
// Everything here is a pure function over caller-provided arrays: no allocation, no
// synchronisation, no container. That is the criterion for living in hgcommon -- shared code is
// allocation-free and synchronisation-free by construction, so anything that allocates or
// synchronises is orchestration and belongs to one device -- and it is what lets one definition
// compile for host and device.

#include <cstdint>
#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// Bind a data edge against a pattern edge under the current variable binding.
//
// Wolfram semantics: distinct pattern variables MAY bind the same vertex, so a position
// constrains only when its variable is already bound; an unbound variable simply takes the
// vertex. Arity must match exactly -- a pattern edge of arity k matches only edges of arity k.
// Under "UndirectedEdges" a two-vertex edge also binds reversed (bind_pattern_edge_oriented).
//
// `binding` is indexed by variable and `bound_mask` carries one bit per variable; both ports
// already store the binding in exactly that shape. On FAILURE the two may be left partially
// updated, because the loop binds as it goes and returns at the first conflict -- a caller
// that backtracks must save and restore them, which both ports do.
HG_HD inline bool bind_pattern_edge(
    const VertexId* edge_verts, uint8_t edge_arity,
    const uint8_t* pattern_vars, uint8_t pattern_arity,
    VertexId* binding, uint32_t& bound_mask)
{
    if (edge_arity != pattern_arity) return false;
    for (uint8_t i = 0; i < edge_arity; ++i) {
        const uint8_t var = pattern_vars[i];
        const uint32_t bit = uint32_t(1) << var;
        if (bound_mask & bit) {
            if (binding[var] != edge_verts[i]) return false;
        } else {
            binding[var] = edge_verts[i];
            bound_mask |= bit;
        }
    }
    return true;
}

// The orientations in which a data edge can bind a pattern edge under the binding so far, as a
// mask: bit 0 the edge as stored, bit 1 its two vertices reversed (bind_pattern_edge_oriented).
// Only an undirected run, a pattern edge of two distinct variables and a data edge of two
// distinct vertices have bit 1, so a self-loop binds once. There a bound variable keeps the
// orientations that put its vertex at its position. Every other case is bit 0, which
// bind_pattern_edge tests.
HG_HD inline uint8_t edge_orientation_mask(bool undirected, const VertexId* edge_verts,
                                           uint8_t edge_arity, const uint8_t* pattern_vars,
                                           uint8_t pattern_arity, const VertexId* binding,
                                           uint32_t bound_mask) {
    if (!(undirected && edge_arity == 2 && pattern_arity == 2 &&
          pattern_vars[0] != pattern_vars[1] && edge_verts[0] != edge_verts[1]))
        return 1u;
    const bool b0 = (bound_mask >> pattern_vars[0]) & 1u;
    const bool b1 = (bound_mask >> pattern_vars[1]) & 1u;
    const bool o0 = (!b0 || binding[pattern_vars[0]] == edge_verts[0]) &&
                    (!b1 || binding[pattern_vars[1]] == edge_verts[1]);
    const bool o1 = (!b0 || binding[pattern_vars[0]] == edge_verts[1]) &&
                    (!b1 || binding[pattern_vars[1]] == edge_verts[0]);
    return static_cast<uint8_t>((o0 ? 1u : 0u) | (o1 ? 2u : 0u));
}

// bind_pattern_edge in orientation `orient` (edge_orientation_mask).
HG_HD inline bool bind_pattern_edge_oriented(
    const VertexId* edge_verts, uint8_t edge_arity, uint8_t orient,
    const uint8_t* pattern_vars, uint8_t pattern_arity,
    VertexId* binding, uint32_t& bound_mask)
{
    if (orient == 0)
        return bind_pattern_edge(edge_verts, edge_arity, pattern_vars, pattern_arity, binding,
                                 bound_mask);
    const VertexId reversed[2] = {edge_verts[1], edge_verts[0]};
    return bind_pattern_edge(reversed, edge_arity, pattern_vars, pattern_arity, binding,
                             bound_mask);
}

}  // namespace common
}  // namespace HG_NAMESPACE
