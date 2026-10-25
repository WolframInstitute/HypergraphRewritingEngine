#pragma once
#include "hgcommon/namespace.hpp"
// Undirected edges ("UndirectedEdges" -> True), one body for host and device.
//
// Under the option an edge of two vertices is an unordered pair. A pattern edge {a, b} of two
// distinct variables is FLIPPABLE: it binds a data edge (u, v) as a = u, b = v or, when u != v, as
// a = v, b = u (hgcommon::edge_orientation_mask, bind_pattern_edge_oriented in match_core.hpp). Edges
// of three or more vertices are refused by the callers before a run starts.
//
// A match of a rule with k flippable left edges has an ORIENTATION MASK: bit j is set when the
// j-th flippable left edge is bound reversed relative to the stored order of its data edge. The
// engine registers, after the user's rules, the 2^k - 1 ORIENTATION VARIANTS of each rule: the
// rule with the flippable left edges of a mask reversed. A match is emitted with the rule id of
// the variant of its mask, so every component that identifies a match by its rule id and edges,
// and that rebinds a match from them, reads the orientation without a second field. Matching runs
// over the user's rules only.
//
// Two matches are one when they consume the same edges and their bindings differ by an
// AUTOMORPHISM of the rule under the undirected reading: a permutation of the left variables that
// maps the unordered left edges onto themselves and, with a permutation of the fresh variables,
// the unordered right edges onto themselves. Only the least match of each orbit is emitted
// (undirected_orbit_least): least by the edge ids in pattern order, then by the orientation mask.
// The edge ids of one state are in the order of their creation along its derivation, so the
// emitted member is a function of the inputs on both engines and at every thread count.

#include <cstdint>
#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// Flippable left edges a rule may have: 2^8 - 1 variants per rule.
constexpr uint32_t UNDIRECTED_MAX_FLIPPABLE = 8;
// Non-identity automorphisms a rule may have under the undirected reading.
constexpr uint32_t UNDIRECTED_MAX_AUTOMORPHISMS = 15;

// One automorphism: var[v] is the image of left variable v, and pos[i] the left position whose
// unordered edge is the image of left edge i.
struct UndirectedAut {
    uint8_t var[MAX_VARS];
    uint8_t pos[MAX_PATTERN_EDGES];
};

// What a rule carries for undirected matching. `undirected` 0 is a directed run and every other
// field is unread. A variant carries its base rule's info and its own `orientation`.
struct UndirectedRuleInfo {
    uint8_t  undirected;
    uint8_t  num_flippable;
    uint8_t  num_aut;
    uint8_t  pad0;
    uint16_t orientation;      // this entry's mask: 0 for a user rule
    uint16_t variant_first;    // table index of the variant of mask 1
    uint16_t base;             // table index of the user rule this entry orients
    uint16_t pad1;
    uint8_t  flip_pos[MAX_PATTERN_EDGES];    // left positions of the flippable edges, ascending
    uint8_t  flip_var0[MAX_PATTERN_EDGES];   // the base rule's variable at position 0 of each
    UndirectedAut aut[UNDIRECTED_MAX_AUTOMORPHISMS];
};

HG_HD inline bool undirected_flippable(const uint8_t* vars, uint8_t arity) {
    return arity == 2 && vars[0] != vars[1];
}

// The orientation mask of a binding of the base rule: `binding_of(v)` the vertex variable v is
// bound to, `v0_at(p)` the stored first vertex of the data edge at left position p.
template <class BindingOf, class V0At>
HG_HD inline uint32_t undirected_flip_mask(const UndirectedRuleInfo& info, BindingOf&& binding_of,
                                           V0At&& v0_at) {
    uint32_t mask = 0;
    for (uint32_t j = 0; j < info.num_flippable; ++j)
        if (binding_of(info.flip_var0[j]) != v0_at(info.flip_pos[j])) mask |= 1u << j;
    return mask;
}

// The orientation mask of a match from the left positions bound reversed (bit p for position p,
// JoinState::reversed_mask), which is undirected_flip_mask of the match's own binding.
HG_HD inline uint32_t undirected_mask_of_positions(const UndirectedRuleInfo& info,
                                                   uint32_t reversed_positions) {
    uint32_t mask = 0;
    for (uint32_t j = 0; j < info.num_flippable; ++j)
        if ((reversed_positions >> info.flip_pos[j]) & 1u) mask |= 1u << j;
    return mask;
}

// The rule id of the variant of `mask` of user rule `base`.
HG_HD inline uint32_t undirected_variant_id(const UndirectedRuleInfo& info, uint32_t base,
                                            uint32_t mask) {
    return mask == 0 ? base : uint32_t{info.variant_first} + mask - 1u;
}

// The position of a variant in the list of a rule's orientation variants with the first
// flippable left edge most significant, the order of Tuples[{0, 1}, k]: the rule key of the
// event orders after the user rule index.
HG_HD inline uint32_t undirected_orientation_rank(const UndirectedRuleInfo& info, uint32_t mask) {
    uint32_t r = 0;
    for (uint32_t j = 0; j < info.num_flippable; ++j)
        r = (r << 1) | ((mask >> j) & 1u);
    return r;
}

// The user rule of table entry `t`: its base under undirected matching, `t` itself otherwise.
HG_HD inline uint32_t undirected_user_rule(const UndirectedRuleInfo& info, uint32_t t) {
    return info.undirected ? uint32_t{info.base} : t;
}

// The rule key of table entry `t` for "EventOrderingFunction" and "Generational" ("RuleIndex"):
// the user rule, then the orientation rank.
HG_HD inline uint32_t undirected_order_rule_key(const UndirectedRuleInfo& info, uint32_t t) {
    if (!info.undirected) return t;
    return (uint32_t{info.base} << UNDIRECTED_MAX_FLIPPABLE) |
           undirected_orientation_rank(info, info.orientation);
}

// Whether the match (edges[p] at left position p, `binding`, orientation `mask`) is the least of
// its orbit under the rule's automorphisms. `v0_at(e)` reads the stored first vertex of edge e.
template <class EdgeIdT, class V0Of>
HG_HD inline bool undirected_orbit_least(const UndirectedRuleInfo& info, uint32_t num_lhs,
                                         const EdgeIdT* edges, const uint32_t* binding,
                                         uint32_t mask, V0Of&& v0_of) {
    for (uint32_t a = 0; a < info.num_aut; ++a) {
        const UndirectedAut& g = info.aut[a];
        int c = 0;
        for (uint32_t i = 0; i < num_lhs && c == 0; ++i) {
            const EdgeIdT img = edges[g.pos[i]];
            if (img != edges[i]) c = img < edges[i] ? -1 : 1;
        }
        if (c == 0) {
            const uint32_t img_mask = undirected_flip_mask(
                info, [&](uint32_t v) { return binding[g.var[v]]; },
                [&](uint32_t p) { return v0_of(edges[g.pos[p]]); });
            if (img_mask < mask) c = -1;
        }
        if (c < 0) return false;
    }
    return true;
}

// The rule id a match of user rule `base` is emitted under, or false when an automorphic match
// is emitted in its place. `edges[p]` is the data edge at left position p, `mask` the match's
// orientation mask (undirected_mask_of_positions of a join's reversed positions, or
// undirected_flip_mask of its binding) and `v0_of(e)` the stored first vertex of edge e, read only
// for the images under the rule's symmetries.
template <class EdgeIdT, class V0Of>
HG_HD inline bool undirected_resolve(const UndirectedRuleInfo& info, uint32_t base,
                                     uint32_t num_lhs, const EdgeIdT* edges,
                                     const uint32_t* binding, uint32_t mask,
                                     V0Of&& v0_of, uint32_t& rule_out) {
    if (!undirected_orbit_least(info, num_lhs, edges, binding, mask, v0_of)) return false;
    rule_out = undirected_variant_id(info, base, mask);
    return true;
}

// Whether left position `p` is reversed in the variant of orientation `mask`.
HG_HD inline bool undirected_reversed(const UndirectedRuleInfo& info, uint32_t mask, uint32_t p) {
    for (uint32_t j = 0; j < info.num_flippable; ++j)
        if (info.flip_pos[j] == p) return ((mask >> j) & 1u) != 0u;
    return false;
}

// THE RULE TABLE OF AN UNDIRECTED RUN, for both engines: the user rules 0..n-1, then the
// orientation variants of rule 0 (mask 1 upward), then those of rule 1, and so on. `info[r]`
// arrives filled by undirected_rule_info; this sets its `base` and `variant_first`, then calls
// emit(r, mask) for every variant in table order. Returns the table size, or 0 when the table
// would pass 65535 entries.
template <class Emit>
inline uint32_t undirected_table_layout(UndirectedRuleInfo* info, uint32_t n_rules, Emit&& emit) {
    uint32_t next = n_rules;
    for (uint32_t r = 0; r < n_rules; ++r) {
        const uint32_t variants = (1u << info[r].num_flippable) - 1u;
        if (next + variants > 0xFFFFu) return 0;
        info[r].base = static_cast<uint16_t>(r);
        info[r].variant_first = static_cast<uint16_t>(next);
        next += variants;
    }
    for (uint32_t r = 0; r < n_rules; ++r)
        for (uint32_t mask = 1; mask < (1u << info[r].num_flippable); ++mask) emit(r, mask);
    return next;
}

// -----------------------------------------------------------------------------------------
// Registration (host): the flippable edges and the automorphisms of one rule. `Edge` has
// `vars[]` and `arity`; every edge has one or two vertices.
// -----------------------------------------------------------------------------------------

enum UndirectedRuleStatus : uint32_t {
    UNDIRECTED_RULE_OK = 0,
    UNDIRECTED_RULE_TOO_MANY_FLIPPABLE = 1,
    UNDIRECTED_RULE_TOO_SYMMETRIC = 2,
    UNDIRECTED_RULE_TOO_WIDE = 3,           // an edge of more than two vertices
    UNDIRECTED_RULE_TOO_MANY_VARIANTS = 4,  // the table passes 65535 entries
};

// The reason a refusal states: after "rule r", or alone for TOO_MANY_VARIANTS.
inline const char* undirected_refusal_reason(UndirectedRuleStatus s) {
    switch (s) {
        case UNDIRECTED_RULE_TOO_MANY_FLIPPABLE:
            return "has more than 8 left edges of two distinct vertices";
        case UNDIRECTED_RULE_TOO_SYMMETRIC: return "has more than 16 symmetries";
        case UNDIRECTED_RULE_TOO_WIDE: return "has an edge of more than two vertices";
        case UNDIRECTED_RULE_TOO_MANY_VARIANTS: return "the rules have more than 65535 orientations";
        default: return "";
    }
}

namespace undirected_detail {

// Backtracking over a bijection of `n` edges onto themselves under a variable map `img` (0xFF
// unset) that may still be extended at the variables `free_mask` allows, its inverse `inv` kept
// injective. found() is called at each complete bijection and returns false to stop.
template <class Edge, class Found>
inline bool edge_bijections(const Edge* e, uint32_t n, uint32_t i, uint8_t* img, uint8_t* inv,
                            uint32_t free_mask, uint8_t* pos, uint32_t used, Found&& found) {
    if (i == n) return found(pos);
    for (uint32_t t = 0; t < n; ++t) {
        if ((used >> t) & 1u) continue;
        if (e[t].arity != e[i].arity) continue;
        const uint32_t orients = e[i].arity == 2 ? 2u : 1u;
        for (uint32_t o = 0; o < orients; ++o) {
            uint8_t target[2] = {e[t].vars[0], e[t].arity == 2 ? e[t].vars[1] : uint8_t(0)};
            if (o == 1) { const uint8_t s = target[0]; target[0] = target[1]; target[1] = s; }
            if (o == 1 && target[0] == target[1]) continue;
            uint8_t set_vars[2];
            uint32_t n_set = 0;
            bool ok = true;
            for (uint32_t k = 0; k < e[i].arity && ok; ++k) {
                const uint8_t v = e[i].vars[k], w = target[k];
                if (img[v] != 0xFF) { ok = img[v] == w; continue; }
                if (!((free_mask >> v) & 1u) || inv[w] != 0xFF) { ok = false; continue; }
                img[v] = w; inv[w] = v; set_vars[n_set++] = v;
            }
            bool more = true;
            if (ok) {
                pos[i] = static_cast<uint8_t>(t);
                more = edge_bijections(e, n, i + 1, img, inv, free_mask, pos, used | (1u << t), found);
            }
            for (uint32_t k = 0; k < n_set; ++k) { inv[img[set_vars[k]]] = 0xFF; img[set_vars[k]] = 0xFF; }
            if (!more) return false;
        }
    }
    return true;
}

}  // namespace undirected_detail

// Fills `info` for a user rule: undirected 1, the flippable left edges, and the non-identity
// automorphisms. variant_first is the caller's.
template <class Edge>
inline UndirectedRuleStatus undirected_rule_info(const Edge* lhs, uint32_t n_lhs, const Edge* rhs,
                                                 uint32_t n_rhs, UndirectedRuleInfo& info) {
    using namespace undirected_detail;
    info = UndirectedRuleInfo{};
    info.undirected = 1;
    for (uint32_t p = 0; p < n_lhs; ++p)
        if (lhs[p].arity > 2) return UNDIRECTED_RULE_TOO_WIDE;
    for (uint32_t p = 0; p < n_rhs; ++p)
        if (rhs[p].arity > 2) return UNDIRECTED_RULE_TOO_WIDE;
    for (uint32_t p = 0; p < n_lhs; ++p) {
        if (!undirected_flippable(lhs[p].vars, lhs[p].arity)) continue;
        if (info.num_flippable >= UNDIRECTED_MAX_FLIPPABLE) return UNDIRECTED_RULE_TOO_MANY_FLIPPABLE;
        info.flip_pos[info.num_flippable] = static_cast<uint8_t>(p);
        info.flip_var0[info.num_flippable] = lhs[p].vars[0];
        ++info.num_flippable;
    }
    uint32_t lhs_mask = 0, rhs_mask = 0;
    for (uint32_t p = 0; p < n_lhs; ++p)
        for (uint32_t k = 0; k < lhs[p].arity; ++k) lhs_mask |= 1u << lhs[p].vars[k];
    for (uint32_t p = 0; p < n_rhs; ++p)
        for (uint32_t k = 0; k < rhs[p].arity; ++k) rhs_mask |= 1u << rhs[p].vars[k];
    const uint32_t fresh_mask = rhs_mask & ~lhs_mask;

    uint8_t img[MAX_VARS], inv[MAX_VARS], pos[MAX_PATTERN_EDGES], rpos[MAX_PATTERN_EDGES];
    for (uint32_t v = 0; v < MAX_VARS; ++v) { img[v] = 0xFF; inv[v] = 0xFF; }
    bool too_many = false;
    edge_bijections(lhs, n_lhs, 0, img, inv, lhs_mask, pos, 0u, [&](const uint8_t* lp) {
        bool identity = true;
        for (uint32_t i = 0; i < n_lhs && identity; ++i) identity = lp[i] == i;
        for (uint32_t v = 0; v < MAX_VARS && identity; ++v)
            identity = ((lhs_mask >> v) & 1u) == 0 || img[v] == v;
        if (identity) return true;
        // The right side under the left map, the fresh variables free to permute among
        // themselves: the fresh variables' images are taken from the fresh set only.
        uint8_t rimg[MAX_VARS], rinv[MAX_VARS];
        for (uint32_t v = 0; v < MAX_VARS; ++v) { rimg[v] = img[v]; rinv[v] = 0xFF; }
        for (uint32_t v = 0; v < MAX_VARS; ++v)
            if (rimg[v] != 0xFF) rinv[rimg[v]] = static_cast<uint8_t>(v);
        // A left variable's image is a left variable, so a fresh variable is never one's image.
        bool rhs_ok = false;
        edge_bijections(rhs, n_rhs, 0, rimg, rinv, fresh_mask, rpos, 0u, [&](const uint8_t*) {
            rhs_ok = true;
            return false;
        });
        if (!rhs_ok) return true;
        if (info.num_aut >= UNDIRECTED_MAX_AUTOMORPHISMS) { too_many = true; return false; }
        UndirectedAut& g = info.aut[info.num_aut++];
        for (uint32_t v = 0; v < MAX_VARS; ++v) g.var[v] = ((lhs_mask >> v) & 1u) ? img[v] : uint8_t(v);
        for (uint32_t i = 0; i < MAX_PATTERN_EDGES; ++i)
            g.pos[i] = i < n_lhs ? lp[i] : static_cast<uint8_t>(i);
        return true;
    });
    return too_many ? UNDIRECTED_RULE_TOO_SYMMETRIC : UNDIRECTED_RULE_OK;
}

}  // namespace common
}  // namespace HG_NAMESPACE
