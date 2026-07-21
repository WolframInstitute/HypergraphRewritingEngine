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

// The causal + branchial graph invariants (edge counts, event-pair count) are
// properties of the multiway system, independent of thread scheduling — so they must
// be identical at every thread count, run after run. This is the gate that guards the
// causal/closure/transitive-reduction redesign: any change to that code must keep these
// counts deterministic across 1/4/8/16 threads. Run deeper (more events => more race
// surface) and repeat to shake out synchronization flakiness.
TEST(OracleCorpus, CausalBranchialCountsDeterministicAcrossThreads) {
    for (const auto& c : oracle::corpus()) {
        oracle::Counts ref = oracle::engine_counts(c.rules, c.init, c.measure_steps, 1);
        for (int rep = 0; rep < 3; ++rep) {
            for (unsigned t : {4u, 8u, 16u}) {
                oracle::Counts got = oracle::engine_counts(c.rules, c.init, c.measure_steps, t);
                EXPECT_EQ(got.canonical_states, ref.canonical_states)
                    << c.name << ": canonical_states differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.events, ref.events)
                    << c.name << ": events differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.causal_edges, ref.causal_edges)
                    << c.name << ": causal_edges differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.causal_event_pairs, ref.causal_event_pairs)
                    << c.name << ": causal_event_pairs differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.branchial_edges, ref.branchial_edges)
                    << c.name << ": branchial_edges differ @" << t << " threads (rep " << rep << ")";
            }
        }
    }
}
