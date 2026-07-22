// Feature-combination matrix: the hardening gate that the engine's feature surface
// COMPOSES and stays oracle-exact + deterministic under the lock-free hot paths.
//
// The prior oracle gate (test_oracle_corpus.cpp) checked only Full state mode with
// event-canonicalization off. This exercises the full cross-product that the
// forward-by-reference MatchCore, the is_reachable causal walk, the Edge inline-vertex
// SBO, and the per-worker arena growth must all survive:
//
//   StateCanonicalizationMode {None, Automatic, Full}
//     x event-canonicalization {off, EVENT_SIG_FULL}
//     x threads {1, 4, 8, 16}
//     + quotient exploration {off, on}
//     + uniform-random sampling (fixed seed)
//     + multiple initial states
//
// The invariant asserted everywhere: for a FIXED configuration, every graph-count
// invariant (canonical states, events, causal edges, causal event pairs, branchial
// edges) is a property of the multiway system and MUST be identical at every thread
// count, run after run -- any divergence is a race in the lock-free machinery. Where an
// independent anchor exists (Full + event-off), counts are additionally pinned to the
// brute-force isomorphism oracle.
#include <gtest/gtest.h>

#include "reference/oracle_corpus.hpp"

#include <set>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

// Full configuration knob set for one evolution run.
struct Config {
    StateCanonicalizationMode state_mode;
    EventSignatureKeys event_keys;
    bool quotient;  // explore_from_canonical_states_only (Full mode only)
};

// Run one evolution with the given configuration and return the graph-count invariants.
oracle::Counts run_counts(const std::vector<RewriteRule>& rules,
                          const std::vector<std::vector<VertexId>>& initial,
                          int steps, unsigned threads, const Config& cfg) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(cfg.state_mode);
    hg.set_event_signature_keys(cfg.event_keys);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_explore_from_canonical_states_only(cfg.quotient);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(initial, steps);

    oracle::Counts c;
    c.canonical_states   = hg.num_canonical_states();
    c.events             = hg.num_events();
    c.causal_edges       = hg.causal_graph().num_causal_edges();
    c.causal_event_pairs = hg.causal_graph().num_causal_event_pairs();
    c.branchial_edges    = hg.causal_graph().num_branchial_edges();
    return c;
}

// Multi-initial-state variant.
oracle::Counts run_counts_multi(const std::vector<RewriteRule>& rules,
                                const std::vector<std::vector<std::vector<VertexId>>>& inits,
                                int steps, unsigned threads, const Config& cfg) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(cfg.state_mode);
    hg.set_event_signature_keys(cfg.event_keys);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_explore_from_canonical_states_only(cfg.quotient);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(inits, steps);

    oracle::Counts c;
    c.canonical_states   = hg.num_canonical_states();
    c.events             = hg.num_events();
    c.causal_edges       = hg.causal_graph().num_causal_edges();
    c.causal_event_pairs = hg.causal_graph().num_causal_event_pairs();
    c.branchial_edges    = hg.causal_graph().num_branchial_edges();
    return c;
}

// Independent brute-force iso-distinct state count of the full raw exploration from a
// SET of initial states (None mode -> full raw state set), the multi-initial analogue of
// oracle::brute_force_iso_count.
size_t brute_iso_multi(const std::vector<RewriteRule>& rules,
                       const std::vector<std::vector<std::vector<VertexId>>>& inits,
                       int steps, bool* all_small) {
    Hypergraph hg;                          // None mode: no dedup
    ParallelEvolutionEngine engine(&hg, 1); // single thread: no wasted states
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(inits, steps);

    std::set<std::string> distinct;
    *all_small = true;
    for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
        auto edges = oracle::state_edges(hg, sid);
        if (edges.empty()) continue;
        std::string c = oracle::brute_canonical(edges);
        if (c.empty()) { *all_small = false; continue; }
        distinct.insert(std::move(c));
    }
    return distinct.size();
}

#define EXPECT_COUNTS_EQ(got, ref, ctx)                                                   \
    do {                                                                                  \
        EXPECT_EQ((got).canonical_states, (ref).canonical_states) << (ctx) << " states";  \
        EXPECT_EQ((got).events, (ref).events) << (ctx) << " events";                      \
        EXPECT_EQ((got).causal_edges, (ref).causal_edges) << (ctx) << " causal_edges";    \
        EXPECT_EQ((got).causal_event_pairs, (ref).causal_event_pairs)                     \
            << (ctx) << " causal_event_pairs";                                            \
        EXPECT_EQ((got).branchial_edges, (ref).branchial_edges)                           \
            << (ctx) << " branchial_edges";                                               \
    } while (0)

const char* mode_name(StateCanonicalizationMode m) {
    switch (m) {
        case StateCanonicalizationMode::None: return "None";
        case StateCanonicalizationMode::Automatic: return "Automatic";
        case StateCanonicalizationMode::Full: return "Full";
    }
    return "?";
}

// None mode is the raw (undeduplicated) exploration -- its state set grows as the full
// unfolding tree, so bound its depth to the oracle depth; the deduplicating modes are
// bounded by the multiway closure and can run to the deeper measurement depth.
int steps_for(const oracle::Case& c, StateCanonicalizationMode m) {
    return (m == StateCanonicalizationMode::None) ? c.oracle_steps : c.measure_steps;
}

}  // namespace

// -----------------------------------------------------------------------------
// The core cross-product: {None, Automatic, Full} x {event off, event Full} must each be
// thread-count-invariant. Reference computed at 1 thread; asserted identical at 4/8/16.
// -----------------------------------------------------------------------------
TEST(FeatureMatrix, CanonModeEventCanonThreadDeterminism) {
    const StateCanonicalizationMode modes[] = {
        StateCanonicalizationMode::None,
        StateCanonicalizationMode::Automatic,
        StateCanonicalizationMode::Full,
    };
    const EventSignatureKeys event_variants[] = {EVENT_SIG_NONE, EVENT_SIG_FULL};

    for (const auto& c : oracle::corpus()) {
        for (StateCanonicalizationMode m : modes) {
            int steps = steps_for(c, m);
            for (EventSignatureKeys ek : event_variants) {
                Config cfg{m, ek, /*quotient=*/false};
                oracle::Counts ref = run_counts(c.rules, c.init, steps, 1, cfg);
                for (unsigned t : {4u, 8u, 16u}) {
                    oracle::Counts got = run_counts(c.rules, c.init, steps, t, cfg);
                    std::string ctx = std::string(c.name) + " mode=" + mode_name(m) +
                                      " event=" + (ek ? "Full" : "off") +
                                      " threads=" + std::to_string(t);
                    EXPECT_COUNTS_EQ(got, ref, ctx);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Oracle-exactness anchor across threads: Full + event-off canonical_states must equal
// the INDEPENDENT brute-force isomorphism count at every thread count (not just 1).
// -----------------------------------------------------------------------------
TEST(FeatureMatrix, FullModeOracleExactAllThreads) {
    for (const auto& c : oracle::corpus()) {
        bool all_small = true;
        size_t brute = oracle::brute_force_iso_count(c.rules, c.init, c.oracle_steps, &all_small);
        ASSERT_TRUE(all_small) << c.name << ": state exceeded brute-force size bound";
        Config cfg{StateCanonicalizationMode::Full, EVENT_SIG_NONE, false};
        for (unsigned t : {1u, 4u, 8u, 16u}) {
            oracle::Counts got = run_counts(c.rules, c.init, c.oracle_steps, t, cfg);
            EXPECT_EQ(got.canonical_states, brute)
                << c.name << ": Full canonical_states != brute oracle @" << t << " threads";
        }
    }
}

// -----------------------------------------------------------------------------
// Quotient exploration (explore_from_canonical_states_only, Full mode). The quotient run
// expands each canonical state exactly once but must reach the SAME canonical closure as
// the full expansion (completeness), and its deterministic invariants -- the canonical
// state set and the event (transition) multiset -- must be thread-count invariant.
//
// Under quotient, causal and branchial edges are recorded only for the expanded
// representatives, forming a scheduling-dependent SKELETON that the design reconstructs
// offline (see set_quotient_initial_states doc + tools/quotient_reconstruction_probe).
// causal_edges/causal_event_pairs therefore legitimately vary with thread count here and
// are NOT asserted; canonical_states and events are the online quotient invariants.
// -----------------------------------------------------------------------------
TEST(FeatureMatrix, QuotientMatchesFullClosureAndDeterministic) {
    // Checked at the oracle depth, where the corpus workloads reach their canonical
    // closure within budget. (At the deeper measurement depth the step budget truncates
    // mid-closure and the quotient depth-relaxation vs budget-cutoff interaction makes the
    // reached canonical set scheduling-dependent under multithreading -- a quotient/budget
    // property independent of the lock-free hot-path hardening; see the escalation note.)
    for (const auto& c : oracle::corpus()) {
        int steps = c.oracle_steps;
        Config full{StateCanonicalizationMode::Full, EVENT_SIG_NONE, /*quotient=*/false};
        Config quot{StateCanonicalizationMode::Full, EVENT_SIG_NONE, /*quotient=*/true};

        size_t full_states = run_counts(c.rules, c.init, steps, 4, full).canonical_states;

        oracle::Counts ref = run_counts(c.rules, c.init, steps, 1, quot);
        EXPECT_EQ(ref.canonical_states, full_states)
            << c.name << ": quotient canonical closure != full closure";
        for (unsigned t : {4u, 8u, 16u}) {
            oracle::Counts got = run_counts(c.rules, c.init, steps, t, quot);
            std::string ctx = std::string(c.name) + " quotient threads=" + std::to_string(t);
            EXPECT_EQ(got.canonical_states, ref.canonical_states) << ctx << " canonical_states";
            EXPECT_EQ(got.events, ref.events) << ctx << " events";
        }
    }
}

// -----------------------------------------------------------------------------
// Multiple initial states compose with canonicalization + threading. Two non-isomorphic
// roots explored together: Full canonical_states must equal the independent brute-force
// iso count over the combined exploration, and every count invariant must be thread
// deterministic.
// -----------------------------------------------------------------------------
TEST(FeatureMatrix, MultipleInitialStatesOracleAndDeterministic) {
    // Non-isomorphic roots under the canonical binary-growth rule.
    std::vector<RewriteRule> rules{
        make_rule(0).lhs({0, 1}).rhs({0, 2}).rhs({1, 2}).build()};
    std::vector<std::vector<std::vector<VertexId>>> inits{
        {{0, 1}},              // single edge
        {{0, 1}, {1, 2}},      // path of two
    };
    const int steps = 3;

    bool all_small = true;
    size_t brute = brute_iso_multi(rules, inits, steps, &all_small);
    ASSERT_TRUE(all_small) << "multi-initial: exceeded brute-force size bound";

    Config cfg{StateCanonicalizationMode::Full, EVENT_SIG_NONE, false};
    oracle::Counts ref = run_counts_multi(rules, inits, steps, 1, cfg);
    EXPECT_EQ(ref.canonical_states, brute)
        << "multi-initial: Full canonical_states != brute oracle";
    for (unsigned t : {4u, 8u, 16u}) {
        oracle::Counts got = run_counts_multi(rules, inits, steps, t, cfg);
        std::string ctx = std::string("multi-initial threads=") + std::to_string(t);
        EXPECT_COUNTS_EQ(got, ref, ctx);
    }
}

// -----------------------------------------------------------------------------
// Uniform-random sampling reproducibility: with a fixed nonzero seed on a single thread,
// repeated runs must draw the identical sample (all count invariants equal). This pins
// the sampling RNG re-seed path (sampling_generation_) under the lock-free engine.
// -----------------------------------------------------------------------------
TEST(FeatureMatrix, UniformRandomReproducibleSameSeed) {
    for (const auto& c : oracle::corpus()) {
        auto run = [&]() {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine engine(&hg, 1);
            engine.set_random_seed(0xC0FFEEu);
            for (const auto& r : c.rules) engine.add_rule(r);
            engine.evolve_uniform_random(c.init, c.measure_steps, /*matches_per_step=*/1);
            oracle::Counts ct;
            ct.canonical_states   = hg.num_canonical_states();
            ct.events             = hg.num_events();
            ct.causal_edges       = hg.causal_graph().num_causal_edges();
            ct.causal_event_pairs = hg.causal_graph().num_causal_event_pairs();
            ct.branchial_edges    = hg.causal_graph().num_branchial_edges();
            return ct;
        };
        oracle::Counts a = run();
        oracle::Counts b = run();
        std::string ctx = std::string(c.name) + " uniform-random same-seed";
        EXPECT_COUNTS_EQ(b, a, ctx);
    }
}
