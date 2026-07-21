// Quotient exploration (explore_from_canonical_states_only) must discover the same
// canonical closure as the full expansion -- it may only skip re-expanding a state,
// never miss one (TASKS.md item 15). Loop-forming idempotent / mixed rulesets are the
// ones that historically risked leaving a canonical state unexpanded. This gates that
// property, which was previously checked only by an un-wired standalone probe.
#include <gtest/gtest.h>
#include <hypergraph/parallel_evolution.hpp>
#include <vector>

using namespace hypergraph;

namespace {
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;

size_t canonical_count(const Rules& rules, Init init, int steps, bool quotient,
                       unsigned threads) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_explore_from_canonical_states_only(quotient);
    for (const auto& r : rules) e.add_rule(r);
    e.evolve(init, steps);
    return hg.num_canonical_states();
}

Rules mAllThree() {
    return {make_rule(0).lhs({0, 1}).rhs({0, 2}).rhs({2, 1}).build(),
            make_rule(1).lhs({0, 1}).rhs({1, 0}).build(),
            make_rule(2).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).build()};
}
Rules iFlip() { return {make_rule(0).lhs({0, 1}).rhs({1, 0}).build()}; }
Rules mIdemProd() {
    return {make_rule(0).lhs({0, 1}).rhs({1, 0}).build(),
            make_rule(1).lhs({0, 1}).rhs({0, 2}).rhs({2, 1}).build()};
}

struct Case { const char* name; Rules rules; Init init; int steps; };
}  // namespace

TEST(QuotientCompletenessTest, MatchesFullExpansionCanonicalCount) {
    const Init tri = {{0u, 1u}, {1u, 2u}, {2u, 0u}};
    const Init two = {{0u, 1u}, {0u, 2u}};
    const std::vector<Case> cases = {
        {"mAllThree/triangle", mAllThree(), tri, 3},
        {"mAllThree/two-edge", mAllThree(), two, 3},
        {"iFlip/two-edge",     iFlip(),     two, 5},
        {"mIdemProd/two-edge", mIdemProd(), two, 3},
    };
    const unsigned T = std::max(4u, std::thread::hardware_concurrency());
    for (const auto& c : cases) {
        size_t full = canonical_count(c.rules, c.init, c.steps, /*quotient=*/false, T);
        size_t quot = canonical_count(c.rules, c.init, c.steps, /*quotient=*/true, T);
        EXPECT_EQ(full, quot)
            << c.name << ": quotient skeleton discovered " << quot
            << " canonical states, full expansion " << full;
    }
}
