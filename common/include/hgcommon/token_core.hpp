#pragma once
#include "hgcommon/namespace.hpp"
// Keyed rewrites: an edge's TOKEN is its identity across states.
//
// An initial edge's token is its edge id plus one. A produced edge's token is (rewrite id, RHS
// index), where the rewrite id is interned exactly from the rule and the consumed edges' tokens
// in match order; the binding, and so every produced edge's vertex tuple, is a function of
// those. A token therefore determines its edge's vertices up to the renaming of fresh vertices,
// and a token occurs at most once in any state: producing it again would need the tokens it
// consumed. Two states with the same token set are isomorphic through the token correspondence,
// so a state can take the canonical results of an earlier state with the same tokens.
//
// Token 0 means "no token identity" (the rewrite id space exhausted); a state holding such an
// edge has no token sum and takes no twin.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

constexpr uint64_t TOKEN_PRODUCED = 1ull << 63;

HG_HD inline uint64_t token_initial(uint32_t edge_id) {
    return static_cast<uint64_t>(edge_id) + 1u;
}

// Event::rewrite_id before it is computed, and when the id space is exhausted.
constexpr uint32_t REWRITE_ID_UNSET = 0;
constexpr uint32_t REWRITE_ID_NONE = 0xFFFFFFFFu;
// Beside a rewrite id (which is below 2^31): the state it makes claims a twin.
constexpr uint32_t REWRITE_TWIN_CANDIDATE = 1u << 31;

// rewrite_id >= 1, index < 256.
HG_HD inline uint64_t token_produced(uint32_t rewrite_id, uint32_t index) {
    return TOKEN_PRODUCED | (static_cast<uint64_t>(rewrite_id) << 8) | (index & 0xFFu);
}

// A state's token sum: the sum over its edges of token_term(token), modulo 2^64. A child's sum
// is its parent's minus the consumed terms plus the produced terms. Equal sums select a
// candidate twin; equal token sets decide it.
HG_HD inline uint64_t token_term(uint64_t token) { return splitmix64(token); }

}  // namespace common
}  // namespace HG_NAMESPACE
