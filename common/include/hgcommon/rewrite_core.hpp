#pragma once
// Shared CPU/GPU rewrite semantics: which vertices the produced edges carry.
//
// A rewrite consumes the edges the match bound and produces one edge per RHS pattern edge.
// Which edges are consumed is the match, already shared via match_core.hpp. What is left is
// resolving each RHS position to a vertex, and that splits cleanly:
//
//   SEMANTICS  a variable appearing in the LHS is already bound by the match; a variable
//              appearing only in the RHS is NEW and takes a freshly created vertex, and the
//              new variables take their fresh vertices in ASCENDING VARIABLE ORDER -- so the
//              n-th new variable takes the n-th fresh vertex, and n is recoverable from the
//              mask without storing anything per rewrite.
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
#include "hgcommon/portable_intrinsics.hpp"

namespace hgcommon {

// How many fresh vertices a rule needs: one per new variable.
HG_HD inline uint8_t num_fresh_variables(uint32_t new_var_mask) {
    return static_cast<uint8_t>(popcount(new_var_mask));
}

// Bind the rule's new variables to a consecutive block of fresh vertices, ascending by
// variable index: the n-th new variable takes first_fresh + n. `fresh` is indexed BY VARIABLE
// and must arrive filled with INVALID_ID; only the new variables' slots are written.
//
// A block rather than a list of ids because both ports reserve one: the device bumps a
// high-water mark once, and the host takes one fetch_add instead of one per variable.
HG_HD inline void assign_fresh_consecutive(
    uint32_t new_var_mask, VertexId first_fresh, VertexId* fresh)
{
    VertexId next = first_fresh;
    while (new_var_mask) {
        const uint8_t var = static_cast<uint8_t>(ctz(new_var_mask));
        fresh[var] = next++;
        new_var_mask &= new_var_mask - 1;
    }
}

// Vertices of one produced edge: from the MATCH if the variable occurs in the LHS, otherwise
// from the fresh assignment. Both arrays are indexed by variable and both use INVALID_ID for
// "not here", so a port that has already merged the fresh vertices into its binding passes
// that same array twice.
//
// Two array lookups rather than deriving the fresh index from the mask by popcount: the
// derived form needs no per-rewrite array, but it cost 72 instructions an event more than the
// lookup on the host, which is the shape this had before it was shared and is the one to keep.
//
// False means the rule names a variable that is neither matched nor new -- a malformed rule,
// not a failed match.
HG_HD inline bool resolve_rhs_vertices(
    const uint8_t* rhs_vars, uint8_t arity,
    const VertexId* binding, const VertexId* fresh, VertexId* out)
{
    for (uint8_t i = 0; i < arity; ++i) {
        const uint8_t var = rhs_vars[i];
        VertexId v = binding[var];
        if (v == INVALID_ID) v = fresh[var];
        if (v == INVALID_ID) return false;
        out[i] = v;
    }
    return true;
}

}  // namespace hgcommon
