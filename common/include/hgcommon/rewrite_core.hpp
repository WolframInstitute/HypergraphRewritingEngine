#pragma once
// Shared CPU/GPU rewrite semantics: which vertices the produced edges carry.
//
// A rewrite consumes the edges the match bound and produces one edge per RHS pattern edge.
// Which edges are consumed is the match, already shared via match_core.hpp. What is left is
// resolving each RHS position to a vertex, and that splits cleanly:
//
//   SEMANTICS  a variable appearing in the LHS is already bound by the match; a variable
//              appearing only in the RHS is NEW and takes a freshly created vertex, and the
//              new variables take their fresh vertices in ASCENDING VARIABLE ORDER.
//   HARDWARE   where the fresh vertex ids come from -- a host counter, a device high-water
//              bump -- and where the produced edges are stored.
//
// The ordering convention is worth sharing even though it looks incidental: it is not
// recoverable from the RHS pattern, and if the ports disagreed they would produce states
// differing by a relabeling -- invisible under Full state identity, which merges isomorphic
// states, and visible under Automatic, which does not.
//
// WHICH variables are new matters more, and the two ports did not agree. The set is
// rhs_var_mask & ~lhs_var_mask, an arbitrary mask. The device instead took the new variables
// to be the index range [num_lhs_vars, num_rhs_vars), and num_lhs_vars is a POPCOUNT -- a
// count, not a highest index. For a rule whose LHS variables are not a dense prefix, say LHS
// {0,2} and RHS {0,1,2}, that range names variable 2, which the match already bound: the
// device would overwrite a matched variable with a fresh vertex and leave variable 1 unbound.
// Taking the mask removes the discrepancy by construction rather than by both sides
// happening to number variables densely.
//
// A binding is read by SENTINEL here: INVALID_ID means unbound. Both ports already satisfy
// that -- the host's VariableBinding fills with INVALID_ID and restores it on unbind, the
// device's array is initialised to it -- so neither representation had to change.

#include <cstdint>
#include "hgcommon/core.hpp"

namespace hgcommon {

// Count trailing zeros of a 32-bit word, host and device. Undefined for x == 0.
HG_HD inline uint8_t rw_ctz32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint8_t>(__ffs(static_cast<int>(x)) - 1);
#elif defined(_MSC_VER)
    unsigned long i; _BitScanForward(&i, x); return static_cast<uint8_t>(i);
#else
    return static_cast<uint8_t>(__builtin_ctz(x));
#endif
}

// Bind the rule's new variables to freshly created vertices, ascending by variable index:
// the n-th new variable takes fresh_ids[n]. Returns how many were bound, which is the number
// of fresh vertices the caller must have supplied.
//
// `binding` is the match's, extended in place -- callers work on a copy, since the match
// itself is shared with the forwarding machinery and must not be mutated.
HG_HD inline uint8_t assign_fresh_variables(
    uint32_t new_var_mask, const VertexId* fresh_ids, VertexId* binding)
{
    uint8_t n = 0;
    while (new_var_mask) {
        const uint8_t var = rw_ctz32(new_var_mask);
        binding[var] = fresh_ids[n++];
        new_var_mask &= new_var_mask - 1;
    }
    return n;
}

// Vertices of one produced edge. Every variable must be bound by now -- by the match if it
// occurs in the LHS, by assign_fresh_variables if it does not. False means the rule names a
// variable that is neither, which is a malformed rule rather than a failed match.
HG_HD inline bool resolve_rhs_vertices(
    const uint8_t* rhs_vars, uint8_t arity, const VertexId* binding, VertexId* out)
{
    for (uint8_t i = 0; i < arity; ++i) {
        const VertexId v = binding[rhs_vars[i]];
        if (v == INVALID_ID) return false;
        out[i] = v;
    }
    return true;
}

}  // namespace hgcommon
