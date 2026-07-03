#include <gtest/gtest.h>

#include "hg_gpu/evolve.hpp"

TEST(HgGpuSmoke, EmptyEvolveYieldsEmptyResult) {
    hg_gpu::EvolveInput in;
    in.num_steps = 0;
    auto result = hg_gpu::evolve(in);
    EXPECT_TRUE(result.states.empty());
    EXPECT_TRUE(result.events.empty());
    EXPECT_TRUE(result.causal_edges.empty());
    EXPECT_TRUE(result.branchial_edges.empty());
}

TEST(HgGpuSmoke, NoRulesYieldsJustInitialState) {
    hg_gpu::EvolveInput in;
    in.initial_state = {{0u, 1u}, {1u, 2u}};
    in.num_steps = 5;
    auto result = hg_gpu::evolve(in);
    // No rules means no matches/events, so evolution can't leave the initial
    // state. The initial state itself is tracked, so exactly one state.
    EXPECT_EQ(result.states.size(), 1u);
    EXPECT_TRUE(result.events.empty());
    if (!result.states.empty()) {
        EXPECT_EQ(result.states[0].edges.size(), 2u);
    }
}
