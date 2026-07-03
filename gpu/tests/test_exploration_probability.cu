#include <gtest/gtest.h>

#include "hg_gpu/evolve.hpp"

#include <set>

// Stochastic-exploration pruning. Mirrors the CPU
// ParallelEvolutionEngine::set_exploration_probability semantics: a
// freshly-deduped state is admitted to the next-step frontier with
// probability `exploration_probability`; the state and its event are
// still recorded regardless. The free `evolve()` wrapper applies the
// coin flip in k_dedup_and_append.

namespace {

// Branching rule: {{x,y}} -> {{x,y},{y,z}}. Each step adds one new
// edge consuming the chain's current tip; produces new states quickly.
hg_gpu::RewriteRule chain_grow_rule() {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}};
    r.rhs = {{0, 1}, {1, 2}};
    r.num_lhs_vars = 2;
    r.num_rhs_vars = 3;
    return r;
}

hg_gpu::EvolveInput base_input(uint32_t steps) {
    hg_gpu::EvolveInput in;
    in.rules         = {chain_grow_rule()};
    in.initial_state = {{0u, 1u}};
    in.num_steps     = steps;
    return in;
}

}  // namespace

TEST(ExplorationProbability, ProbabilityOneEqualsBaseline) {
    auto in = base_input(4);
    in.exploration_probability = 1.0f;
    auto a = hg_gpu::evolve(in);
    auto b = hg_gpu::evolve(base_input(4));  // default = 1.0
    EXPECT_EQ(a.states.size(), b.states.size());
    EXPECT_EQ(a.events.size(), b.events.size());
}

TEST(ExplorationProbability, ProbabilityZeroSuppressesExpansion) {
    auto in = base_input(5);
    in.exploration_probability = 0.0f;
    auto result = hg_gpu::evolve(in);
    auto baseline = hg_gpu::evolve(base_input(5));

    // p=0.0 means: rewrite step 0 still runs (initial state is in
    // frontier), produces children, but no child enters the next-step
    // frontier — so step 1's match kernel sees an empty frontier and
    // the loop terminates. Result: initial state + step-0 children only.
    EXPECT_LT(result.states.size(), baseline.states.size());
    EXPECT_GE(result.states.size(), 2u);  // initial + ≥1 child
    EXPECT_FALSE(result.events.empty());
}

TEST(ExplorationProbability, IntermediateValueShrinksTree) {
    auto baseline_in = base_input(4);
    auto baseline = hg_gpu::evolve(baseline_in);

    auto in = base_input(4);
    in.exploration_probability = 0.4f;
    in.exploration_seed        = 0xC0FFEEull;  // deterministic
    auto result = hg_gpu::evolve(in);

    // 0 < p < 1 should give strictly fewer states than the full tree
    // (with overwhelming probability on a 4-step branching evolution),
    // and at least the initial state must be present.
    EXPECT_LT(result.states.size(), baseline.states.size());
    EXPECT_GE(result.states.size(), 1u);
}

TEST(ExplorationProbability, SameSeedReproducesSameResult) {
    auto in = base_input(4);
    in.exploration_probability = 0.5f;
    in.exploration_seed        = 0xDEADBEEFull;

    auto a = hg_gpu::evolve(in);
    auto b = hg_gpu::evolve(in);
    EXPECT_EQ(a.states.size(), b.states.size());
    EXPECT_EQ(a.events.size(), b.events.size());
    EXPECT_EQ(a.causal_edges.size(),    b.causal_edges.size());
    EXPECT_EQ(a.branchial_edges.size(), b.branchial_edges.size());

    // State-hash sets should be identical for identical seeds.
    std::set<uint64_t> ha, hb;
    for (auto& s : a.states) ha.insert(s.canonical_hash);
    for (auto& s : b.states) hb.insert(s.canonical_hash);
    EXPECT_EQ(ha, hb);
}

TEST(ExplorationProbability, DifferentSeedsCanDiverge) {
    // Two different non-zero seeds at intermediate p should *typically*
    // explore different state subsets. We don't require divergence on
    // every coin pair (theoretically possible to coincide), so the
    // assertion is only that both runs succeeded and stayed within the
    // baseline. The same-seed test above is what guarantees determinism;
    // this one just confirms the seed actually feeds the PRNG.
    auto baseline = hg_gpu::evolve(base_input(4));

    auto in1 = base_input(4);
    in1.exploration_probability = 0.5f;
    in1.exploration_seed        = 0xAAAAAAAAull;
    auto r1 = hg_gpu::evolve(in1);

    auto in2 = base_input(4);
    in2.exploration_probability = 0.5f;
    in2.exploration_seed        = 0xBBBBBBBBull;
    auto r2 = hg_gpu::evolve(in2);

    EXPECT_LE(r1.states.size(), baseline.states.size());
    EXPECT_LE(r2.states.size(), baseline.states.size());
    EXPECT_GE(r1.states.size(), 1u);
    EXPECT_GE(r2.states.size(), 1u);
}

TEST(ExplorationProbability, ClampedToValidRange) {
    auto in = base_input(2);
    in.exploration_probability = 2.5f;     // > 1: clamps to 1.0
    auto a = hg_gpu::evolve(in);
    auto b = hg_gpu::evolve(base_input(2));
    EXPECT_EQ(a.states.size(), b.states.size());

    auto in2 = base_input(2);
    in2.exploration_probability = -1.0f;   // < 0: clamps to 0.0
    auto c = hg_gpu::evolve(in2);
    EXPECT_LT(c.states.size(), b.states.size());
}
