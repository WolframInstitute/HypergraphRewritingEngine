// Correctness gate across the rule-type space: every workload in the shared corpus
// (single/mixed arity, varied edge counts and connectivity, productive/idempotent/
// reductive, self-loop, disconnected LHS, multi-rule) must match the INDEPENDENT
// brute-force isomorphism oracle exactly, in Full mode, and must do so identically
// at 1 and 4 threads (determinism). This is the "all rule types checked against the
// oracle" guarantee that every optimization must keep passing.
#include <gtest/gtest.h>

#include "reference/oracle_corpus.hpp"

using namespace hypergraph;

TEST(OracleCorpus, EveryRuleTypeMatchesBruteForce) {
    for (const auto& c : oracle::corpus()) {
        bool all_small = true;
        size_t brute = oracle::brute_force_iso_count(c.rules, c.init, c.oracle_steps, &all_small);
        ASSERT_TRUE(all_small)
            << c.name << " (" << c.type << "): state exceeded brute-force size bound at oracle depth";
        size_t full1 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 1);
        EXPECT_EQ(full1, brute)
            << c.name << " (" << c.type << "): Full-mode count != brute-force oracle";
    }
}

TEST(OracleCorpus, DeterministicAcrossThreadCounts) {
    for (const auto& c : oracle::corpus()) {
        size_t t1 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 1);
        size_t t4 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 4);
        size_t t8 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 8);
        EXPECT_EQ(t1, t4) << c.name << ": canonical count differs between 1 and 4 threads";
        EXPECT_EQ(t1, t8) << c.name << ": canonical count differs between 1 and 8 threads";
    }
}
