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

}  // namespace common
}  // namespace HG_NAMESPACE
