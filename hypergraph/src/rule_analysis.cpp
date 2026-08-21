#include "hypergraph/rule_analysis.hpp"

// The rule facts, computed ONCE per rule at registration and once per run when the engine
// configures itself. They are here rather than in the header because nothing calls them per
// state, per match or per event: the inlining a header body buys is worth nothing at that
// frequency, and every consumer of the header pays to parse them.

namespace HG_NAMESPACE {
namespace engine {

bool lhs_is_connected(const RewriteRule& r) {
    const uint8_t n = r.num_lhs_edges;
    if (n <= 1) return true;
    bool seen[MAX_PATTERN_EDGES] = {false};
    uint8_t stack[MAX_PATTERN_EDGES];
    uint8_t top = 0, count = 1;
    seen[0] = true;
    stack[top++] = 0;
    while (top) {
        const uint8_t e = stack[--top];
        for (uint8_t o = 0; o < n; ++o) {
            if (seen[o] || !r.lhs_edges_connected(e, o)) continue;
            seen[o] = true;
            ++count;
            stack[top++] = o;
        }
    }
    return count == n;
}

bool lhs_is_acyclic(const RewriteRule& r) {
    uint32_t edge_vars[MAX_PATTERN_EDGES] = {0};   // bitset of variables per edge
    bool alive[MAX_PATTERN_EDGES] = {false};
    const uint8_t n = r.num_lhs_edges;
    for (uint8_t i = 0; i < n; ++i) {
        alive[i] = true;
        for (uint8_t j = 0; j < r.lhs[i].arity; ++j) {
            const uint8_t v = r.lhs[i].vars[j];
            if (v < 32) edge_vars[i] |= (1u << v);
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;

        // (a) Drop a variable that only one live edge mentions: nothing joins on it.
        for (uint8_t v = 0; v < 32; ++v) {
            const uint32_t bit = 1u << v;
            int holder = -1, count = 0;
            for (uint8_t i = 0; i < n; ++i)
                if (alive[i] && (edge_vars[i] & bit)) { holder = i; if (++count > 1) break; }
            if (count == 1) { edge_vars[holder] &= ~bit; changed = true; }
        }

        // (b) Drop an edge with nothing left to constrain. Stripping (a) empties an edge whose
        // variables were all its own, and a single-edge query reaches exactly that state -- it is
        // trivially acyclic, and a rule requiring ANOTHER edge to contain it would never say so.
        for (uint8_t i = 0; i < n; ++i)
            if (alive[i] && edge_vars[i] == 0u) { alive[i] = false; changed = true; }

        // (c) Drop an edge whose variables another live edge already carries. It constrains
        // nothing the other does not.
        for (uint8_t i = 0; i < n && !changed; ++i) {
            if (!alive[i]) continue;
            for (uint8_t k = 0; k < n; ++k) {
                if (k == i || !alive[k]) continue;
                if ((edge_vars[i] & ~edge_vars[k]) == 0u) {   // i subset of k
                    alive[i] = false; changed = true; break;
                }
            }
        }
    }

    for (uint8_t i = 0; i < n; ++i) if (alive[i]) return false;
    return true;
}

uint32_t lhs_edge_cover(const RewriteRule& r) {
    const uint8_t n = r.num_lhs_edges;
    if (n == 0) return 0;
    uint32_t edge_vars[MAX_PATTERN_EDGES] = {0}, all = 0;
    for (uint8_t i = 0; i < n; ++i)
        for (uint8_t j = 0; j < r.lhs[i].arity; ++j) {
            const uint8_t v = r.lhs[i].vars[j];
            if (v < 32) { edge_vars[i] |= (1u << v); all |= (1u << v); }
        }
    uint32_t best = n;
    for (uint32_t mask = 1; mask < (1u << n); ++mask) {
        uint32_t covered = 0, used = 0;
        for (uint8_t i = 0; i < n; ++i)
            if (mask & (1u << i)) { covered |= edge_vars[i]; ++used; }
        if (used < best && covered == all) best = used;
    }
    return best;
}

RuleFacts analyze_rule(const RewriteRule& r) {
    RuleFacts f;
    f.lhs_edges = r.num_lhs_edges;
    f.edge_delta = static_cast<int>(r.num_rhs_edges) - static_cast<int>(r.num_lhs_edges);
    for (uint8_t i = 0; i < r.num_lhs_edges; ++i) f.lhs_arities.push_back(r.lhs[i].arity);
    // The rule already carries this: num_new_vars is variables in the RHS and not the LHS, which
    // is the vertex creation rate. Recomputing it here would be a second implementation of a
    // rule the pattern builder already decides.
    f.new_vertices = r.num_new_vars;
    f.acyclic = lhs_is_acyclic(r);
    f.edge_cover = lhs_edge_cover(r);
    f.edge_cover_tight = f.acyclic;   // acyclic => the LP optimum is integral => cover == rho*
    return f;
}

bool can_branch(const std::vector<RewriteRule>& rules) {
    for (size_t a = 0; a < rules.size(); ++a) {
        if (rules[a].num_lhs_edges >= 2) return true;   // one rule, two matches, shared edge
        for (size_t b = a + 1; b < rules.size(); ++b) {
            for (uint8_t i = 0; i < rules[a].num_lhs_edges; ++i)
                for (uint8_t j = 0; j < rules[b].num_lhs_edges; ++j)
                    if (rules[a].lhs[i].arity == rules[b].lhs[j].arity)
                        return true;                    // two rules, one edge satisfies both
        }
    }
    return false;
}

RuleSetFacts analyze_rules(const std::vector<RewriteRule>& rules) {
    RuleSetFacts s;
    s.may_branch = can_branch(rules);
    s.non_growing = true;
    s.bounded_vertices = true;
    for (const auto& r : rules) {
        if (r.num_rhs_edges > r.num_lhs_edges) s.non_growing = false;
        if (r.num_new_vars > 0) s.bounded_vertices = false;
        if (r.num_lhs_edges >= 2) s.forwarding_pays = true;
        if (r.num_lhs_edges >= 3 && !lhs_is_acyclic(r)) s.has_cyclic_multiedge_lhs = true;
        if (!lhs_is_connected(r)) s.has_disconnected_lhs = true;
    }
    return s;
}


bool edge_cover_is_tight(const RewriteRule& r) { return lhs_is_acyclic(r); }

}  // namespace engine
}  // namespace HG_NAMESPACE
