#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"

#include <vector>

// Incremental WL (B) must be bit-identical to full WL: an evolution with incremental
// WL enabled must produce exactly the same canonical states, events, causal and
// branchial edges as with it disabled. (Incremental affects the WL canonical hash
// used in None/Automatic modes + event canonicalization + match forwarding.)

using namespace hypergraph;

namespace {

RewriteRule growth_rule() {
    return make_rule(0).lhs({0, 1}).rhs({0, 2}).rhs({1, 2}).build();
}

RewriteRule swap_rule() {
    return make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).rhs({2, 1}).build();
}

struct Counts {
    size_t states, events, causal, branchial;
    bool operator==(const Counts& o) const {
        return states == o.states && events == o.events && causal == o.causal && branchial == o.branchial;
    }
};

Counts run(const std::vector<RewriteRule>& rules,
           const std::vector<std::vector<VertexId>>& initial,
           size_t steps, bool incremental, size_t threads) {
    Hypergraph hg;
    if (incremental) hg.set_incremental_wl(true);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_match_forwarding(true);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);
    return { engine.num_canonical_states(), engine.num_events(),
             engine.num_causal_edges(), engine.num_branchial_edges() };
}

void expect_match(const std::vector<RewriteRule>& rules,
                  const std::vector<std::vector<VertexId>>& initial,
                  size_t steps, size_t threads) {
    Counts off = run(rules, initial, steps, /*incremental=*/false, threads);
    Counts on  = run(rules, initial, steps, /*incremental=*/true,  threads);
    EXPECT_EQ(off.states, on.states)   << "canonical states differ (steps=" << steps << ", threads=" << threads << ")";
    EXPECT_EQ(off.events, on.events)   << "events differ";
    EXPECT_EQ(off.causal, on.causal)   << "causal edges differ";
    EXPECT_EQ(off.branchial, on.branchial) << "branchial edges differ";
}

}  // namespace

TEST(IncrementalWL, GrowthRuleMatchesFullWL) {
    for (size_t steps = 3; steps <= 6; ++steps)
        expect_match({growth_rule()}, {{0, 1}}, steps, /*threads=*/4);
}

TEST(IncrementalWL, GrowthRuleSingleThread) {
    for (size_t steps = 3; steps <= 6; ++steps)
        expect_match({growth_rule()}, {{0, 1}}, steps, /*threads=*/1);
}

TEST(IncrementalWL, SwapRuleMatchesFullWL) {
    expect_match({swap_rule()}, {{0, 1}, {1, 2}, {2, 3}}, 4, /*threads=*/4);
}

TEST(IncrementalWL, LargerInitialMatchesFullWL) {
    expect_match({growth_rule()}, {{0, 1}, {2, 3}, {4, 5}}, 5, /*threads=*/4);
}
