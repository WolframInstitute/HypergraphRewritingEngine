#include "hypergraph/pattern.hpp"

// A rule's derived matching data: variable counts and the join order. Both are computed at
// add_rule and read from then on, so they are here rather than in the header that every
// engine translation unit parses.

namespace HG_NAMESPACE {
namespace engine {

void RewriteRule::compute_var_counts() {
    uint32_t lhs_mask = lhs_var_mask();
    uint32_t rhs_mask = rhs_var_mask();
    uint32_t new_mask = rhs_mask & ~lhs_mask;

    num_lhs_vars = static_cast<uint8_t>(hgcommon::popcount(lhs_mask));
    num_rhs_vars = static_cast<uint8_t>(hgcommon::popcount(rhs_mask));
    num_new_vars = static_cast<uint8_t>(hgcommon::popcount(new_mask));

    compute_match_order();

    // Precompute per-edge signature + compatible-signature cache once, so match
    // tasks never repeat the from_pattern Bell enumeration. Indexed by original
    // LHS edge index (see lhs_sig / lhs_cache).
    for (uint8_t i = 0; i < num_lhs_edges; ++i) {
        lhs_sig[i] = lhs[i].signature();
        lhs_cache[i] = CompatibleSignatureCache::from_pattern(lhs_sig[i]);
    }
}

int RewriteRule::edge_constraint_score(uint8_t e) const {
    int score = 0;
    for (uint8_t i = 0; i < lhs[e].arity; ++i)
        for (uint8_t j = static_cast<uint8_t>(i + 1); j < lhs[e].arity; ++j)
            if (lhs[e].var_at(i) == lhs[e].var_at(j)) score += 100;
    for (uint8_t o = 0; o < num_lhs_edges; ++o)
        if (o != e && lhs_edges_connected(e, o)) score += 1;
    return score;
}

void RewriteRule::compute_match_order() {
    for (uint8_t i = 0; i < MAX_PATTERN_EDGES; ++i) match_order[i] = i;
    if (num_lhs_edges <= 1) return;

    bool used[MAX_PATTERN_EDGES] = {};

    // Seed with the most self-constrained edge.
    uint8_t first = 0;
    int best = -1;
    for (uint8_t e = 0; e < num_lhs_edges; ++e) {
        int s = edge_constraint_score(e);
        if (s > best) { best = s; first = e; }
    }
    match_order[0] = first;
    used[first] = true;
    uint32_t bound = lhs[first].var_mask();

    // Greedily append the unmatched edge sharing the most variables with the
    // bound prefix; tie-break by self-constraint, then lower index.
    for (uint8_t pos = 1; pos < num_lhs_edges; ++pos) {
        uint8_t pick = 0;
        int pick_shared = -1, pick_self = -1;
        for (uint8_t e = 0; e < num_lhs_edges; ++e) {
            if (used[e]) continue;
            int shared = hgcommon::popcount(lhs[e].var_mask() & bound);
            int self = edge_constraint_score(e);
            if (shared > pick_shared || (shared == pick_shared && self > pick_self)) {
                pick = e; pick_shared = shared; pick_self = self;
            }
        }
        match_order[pos] = pick;
        used[pick] = true;
        bound |= lhs[pick].var_mask();
    }
}

}  // namespace engine
}  // namespace HG_NAMESPACE
