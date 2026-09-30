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
#include "hgcommon/event_core.hpp"

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

// Whether a run reads RANK TUPLES: canonical edge ranks compared as values, by edge-keyed event
// identity (consumed or produced edges; each raw event is signed from its raw states' ranks) and
// by the transition draw (a rate below 1, rule weights, or a per-state match cap). A twin's ranks
// are its earlier state's labelling read through the tokens; on a state with a nontrivial
// automorphism group that is a different labelling from the one the state's own IR gives
// (measured: ranks 1,3,0,2 copied against 0,2,1,3 computed on a 4-edge growshrink3 state), and
// whether the twin is published in time depends on the schedule, so a rank tuple would too.
// Quotient exploration by itself reads ranks only to match edges of equal rank between a state
// and its class frame, and orbits, which any canonical labelling gives alike.
HG_HD inline bool run_reads_rank_tuples(EventSignatureKeys event_keys, double transition_rate,
                                        uint32_t num_rule_weights,
                                        uint32_t matches_per_state_rule) {
    return event_keys_need_ranks(event_keys) || transition_rate < 1.0 ||
           num_rule_weights != 0u || matches_per_state_rule != 0u;
}

// Whether a run keys its rewrites and takes twins: Full mode (the only mode that takes twins),
// event identity not positional (it reads each raw state's own labelling), and no rank tuples.
HG_HD inline bool keyed_rewrites_apply(bool requested, bool full_mode, bool positional_events,
                                       bool reads_rank_tuples) {
    return requested && full_mode && !positional_events && !reads_rank_tuples;
}

// What one twin claim does to a run's keying. A claim that finds another state with the same
// token set marks twins seen, whether that state's results are taken or not yet published; a run
// that has made `claim_limit` claims finding none, with no twin seen, stops keying.
//   Ctx: bool seen() const; void set_seen(); uint32_t add_claim() (the count after this one);
//        void switch_off().
template <class Ctx>
HG_HD inline void keyed_note_claim(Ctx& k, bool found_twin, uint32_t claim_limit) {
    if (found_twin) {
        if (!k.seen()) k.set_seen();
        return;
    }
    if (!k.seen() && k.add_claim() >= claim_limit && !k.seen()) k.switch_off();
}

// rewrite_id >= 1, index < 256.
HG_HD inline uint64_t token_produced(uint32_t rewrite_id, uint32_t index) {
    return TOKEN_PRODUCED | (static_cast<uint64_t>(rewrite_id) << 8) | (index & 0xFFu);
}

// A state's token sum: the sum over its edges of token_term(token), modulo 2^64. A child's sum
// is its parent's minus the consumed terms plus the produced terms. Equal sums select a
// candidate twin; equal token sets decide it.
HG_HD inline uint64_t token_term(uint64_t token) { return splitmix64(token); }

// A rewrite's key: the rule, then each consumed token (match order) as two words, and the hash
// the key is claimed under. Returns the word count, 1 + 2 * n; 0 when a token is 0.
HG_HD inline uint32_t rewrite_key(uint16_t rule, const uint64_t* tokens, uint8_t n,
                                  uint32_t* words, uint64_t& hash) {
    uint32_t w = 0;
    words[w++] = rule;
    hash = fnv_hash(FNV_OFFSET, rule);
    for (uint8_t i = 0; i < n; ++i) {
        if (tokens[i] == 0) return 0;
        words[w++] = static_cast<uint32_t>(tokens[i]);
        words[w++] = static_cast<uint32_t>(tokens[i] >> 32);
        hash = fnv_hash(hash, tokens[i]);
    }
    return w;
}

// A sum is never 0, which marks "no sum".
HG_HD inline uint64_t token_sum_nonzero(uint64_t sum) { return sum == 0 ? 1 : sum; }

// The token sum of a state made by rewrite `rid` from a state with sum `parent_sum`: 0 when the
// parent has no sum or a consumed token is 0.
HG_HD inline uint64_t child_token_sum(uint64_t parent_sum, const uint64_t* consumed,
                                      uint8_t num_consumed, uint32_t rid, uint8_t num_produced) {
    if (parent_sum == 0) return 0;
    uint64_t sum = parent_sum;
    for (uint8_t i = 0; i < num_consumed; ++i) {
        if (consumed[i] == 0) return 0;
        sum -= token_term(consumed[i]);
    }
    for (uint8_t i = 0; i < num_produced; ++i) sum += token_term(token_produced(rid, i));
    return token_sum_nonzero(sum);
}

// A state's edges by token: an open-addressed table from token to the edge's position in the
// state's id order, over storage the caller provides (token_index_capacity slots of each). A
// token occurs at most once in a state; a repeat, or a token 0, clears `valid`.
struct TokenIndex {
    uint64_t* keys;
    uint32_t* pos;
    uint32_t mask;
    uint32_t n;
    bool valid;
};

// The slot count for a state of `count` edges: a power of two, at least 2 * count and 16.
HG_HD inline uint32_t token_index_capacity(uint32_t count) {
    uint32_t cap = 16;
    while (cap < 2 * count) cap <<= 1;
    return cap;
}

HG_HD inline TokenIndex token_index_open(uint64_t* keys, uint32_t* pos, uint32_t cap) {
    for (uint32_t i = 0; i < cap; ++i) keys[i] = 0;
    return TokenIndex{keys, pos, cap - 1, 0, true};
}

// Adds the next edge's token; positions are given in the order tokens are added.
HG_HD inline void token_index_add(TokenIndex& x, uint64_t t) {
    if (t == 0) { x.valid = false; ++x.n; return; }
    for (uint32_t h = static_cast<uint32_t>(splitmix64(t)) & x.mask;; h = (h + 1) & x.mask) {
        if (x.keys[h] == 0) { x.keys[h] = t; x.pos[h] = x.n; break; }
        if (x.keys[h] == t) { x.valid = false; break; }
    }
    ++x.n;
}

// The position of the edge with token `t`, UINT32_MAX when there is none.
HG_HD inline uint32_t token_index_find(const TokenIndex& x, uint64_t t) {
    if (t == 0) return 0xFFFFFFFFu;
    for (uint32_t h = static_cast<uint32_t>(splitmix64(t)) & x.mask;; h = (h + 1) & x.mask) {
        if (x.keys[h] == t) return x.pos[h];
        if (x.keys[h] == 0) return 0xFFFFFFFFu;
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
