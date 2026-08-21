#include "hypergraph/pattern.hpp"

// The bodies behind pattern.hpp: PatternEdge, RewriteRule, RuleBuilder, MatchIdentity and
// PartialMatch. A rule's derived matching data -- variable counts and the join order -- is
// computed once at add_rule and read from then on; the rest is small value-type code. None of
// it is a template, so it lives here rather than in the header that every engine translation
// unit parses.

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


// =============================================================================
// PatternEdge
// =============================================================================

PatternEdge::PatternEdge() : arity(0) {
    std::memset(vars, 0, MAX_ARITY);
}

PatternEdge::PatternEdge(std::initializer_list<uint8_t> var_list) : arity(0) {
    if (var_list.size() > MAX_ARITY) {
        throw std::length_error("PatternEdge: arity exceeds MAX_ARITY");
    }
    std::memset(vars, 0, MAX_ARITY);
    for (uint8_t v : var_list) {
        vars[arity++] = v;
    }
}

PatternEdge::PatternEdge(const uint8_t* var_array, uint8_t n) : arity(n) {
    if (n > MAX_ARITY) {
        throw std::length_error("PatternEdge: arity exceeds MAX_ARITY");
    }
    std::memset(vars, 0, MAX_ARITY);
    for (uint8_t i = 0; i < n; ++i) {
        vars[i] = var_array[i];
    }
}

uint8_t PatternEdge::var_at(uint8_t pos) const {
    return vars[pos];
}

EdgeSignature PatternEdge::signature() const {
    return EdgeSignature::from_pattern(vars, arity);
}

uint32_t PatternEdge::var_mask() const {
    uint32_t mask = 0;
    for (uint8_t i = 0; i < arity; ++i) {
        mask |= (1u << vars[i]);
    }
    return mask;
}

bool PatternEdge::operator==(const PatternEdge& other) const {
    if (arity != other.arity) return false;
    for (uint8_t i = 0; i < arity; ++i) {
        if (vars[i] != other.vars[i]) return false;
    }
    return true;
}

bool PatternEdge::operator!=(const PatternEdge& other) const {
    return !(*this == other);
}

// =============================================================================
// RewriteRule: the parts that are not the derived matching data above
// =============================================================================

RewriteRule::RewriteRule()
    : index(0)
    , num_lhs_edges(0)
    , num_rhs_edges(0)
    , num_lhs_vars(0)
    , num_rhs_vars(0)
    , num_new_vars(0)
{
    for (uint8_t i = 0; i < MAX_PATTERN_EDGES; ++i) match_order[i] = i;
}

uint32_t RewriteRule::lhs_var_mask() const {
    uint32_t mask = 0;
    for (uint8_t i = 0; i < num_lhs_edges; ++i) {
        mask |= lhs[i].var_mask();
    }
    return mask;
}

uint32_t RewriteRule::rhs_var_mask() const {
    uint32_t mask = 0;
    for (uint8_t i = 0; i < num_rhs_edges; ++i) {
        mask |= rhs[i].var_mask();
    }
    return mask;
}

uint32_t RewriteRule::new_var_mask() const {
    return rhs_var_mask() & ~lhs_var_mask();
}

bool RewriteRule::lhs_edges_connected(uint8_t edge1, uint8_t edge2) const {
    return (lhs[edge1].var_mask() & lhs[edge2].var_mask()) != 0;
}

// =============================================================================
// RuleBuilder
// =============================================================================

RuleBuilder::RuleBuilder(uint16_t index) {
    rule_.index = index;
}

RuleBuilder& RuleBuilder::lhs(std::initializer_list<uint8_t> vars) {
    if (rule_.num_lhs_edges >= MAX_PATTERN_EDGES) {
        throw std::length_error("RuleBuilder::lhs: exceeds MAX_PATTERN_EDGES");
    }
    rule_.lhs[rule_.num_lhs_edges++] = PatternEdge(vars);
    return *this;
}

RuleBuilder& RuleBuilder::rhs(std::initializer_list<uint8_t> vars) {
    if (rule_.num_rhs_edges >= MAX_PATTERN_EDGES) {
        throw std::length_error("RuleBuilder::rhs: exceeds MAX_PATTERN_EDGES");
    }
    rule_.rhs[rule_.num_rhs_edges++] = PatternEdge(vars);
    return *this;
}

RewriteRule RuleBuilder::build() {
    rule_.compute_var_counts();
    return rule_;
}

RuleBuilder make_rule(uint16_t index) {
    return RuleBuilder(index);
}

// =============================================================================
// MatchIdentity
// =============================================================================

MatchIdentity::MatchIdentity() : rule_index(0), num_edges(0) {
    std::memset(edges, 0xFF, sizeof(edges));
}

MatchIdentity::MatchIdentity(uint16_t rule, const EdgeId* edge_array, uint8_t n)
    : rule_index(rule), num_edges(n) {
    std::memset(edges, 0xFF, sizeof(edges));
    for (uint8_t i = 0; i < n; ++i) {
        edges[i] = edge_array[i];
    }
}

uint64_t MatchIdentity::hash() const {
    uint64_t h = 14695981039346656037ULL;
    h ^= rule_index;
    h *= 1099511628211ULL;
    for (uint8_t i = 0; i < num_edges; ++i) {
        h ^= edges[i];
        h *= 1099511628211ULL;
    }
    return h;
}

bool MatchIdentity::operator==(const MatchIdentity& other) const {
    if (rule_index != other.rule_index) return false;
    if (num_edges != other.num_edges) return false;
    for (uint8_t i = 0; i < num_edges; ++i) {
        if (edges[i] != other.edges[i]) return false;
    }
    return true;
}

bool MatchIdentity::operator!=(const MatchIdentity& other) const {
    return !(*this == other);
}

// =============================================================================
// PartialMatch
// =============================================================================

PartialMatch::PartialMatch()
    : id(INVALID_ID)
    , rule_index(0)
    , num_matched(0)
    , num_pattern_edges(0)
    , binding()
    , origin_state(INVALID_ID)
{
    std::memset(match_order, 0, sizeof(match_order));
    std::memset(matched_edges, 0xFF, sizeof(matched_edges));
}

bool PartialMatch::is_complete() const {
    return num_matched == num_pattern_edges;
}

bool PartialMatch::is_complete(const RewriteRule& rule) const {
    return num_matched == rule.num_lhs_edges;
}

void PartialMatch::add_match(uint8_t pattern_idx, EdgeId data_edge,
                             const VariableBinding& new_binding) {
    match_order[num_matched] = pattern_idx;
    matched_edges[num_matched] = data_edge;
    binding = new_binding;
    num_matched++;
}

PartialMatch PartialMatch::branch() const {
    return *this;
}

bool PartialMatch::contains_edge(EdgeId eid) const {
    for (uint8_t i = 0; i < num_matched; ++i) {
        if (matched_edges[i] == eid) return true;
    }
    return false;
}

void PartialMatch::to_pattern_order(EdgeId* out) const {
    std::memset(out, 0xFF, MAX_PATTERN_EDGES * sizeof(EdgeId));
    for (uint8_t i = 0; i < num_matched; ++i) {
        uint8_t pattern_idx = match_order[i];
        out[pattern_idx] = matched_edges[i];
    }
}

// rule names the pattern this match was built against; the edge permutation it describes is
// already carried in match_order, so the parameter is not read.
MatchIdentity PartialMatch::to_identity([[maybe_unused]] const RewriteRule& rule) const {
    MatchIdentity mid;
    mid.rule_index = rule_index;
    mid.num_edges = num_matched;

    // matched_edges is in match_order, convert to pattern order
    for (uint8_t i = 0; i < num_matched; ++i) {
        uint8_t pattern_idx = match_order[i];
        mid.edges[pattern_idx] = matched_edges[i];
    }
    return mid;
}

}  // namespace engine
}  // namespace HG_NAMESPACE
