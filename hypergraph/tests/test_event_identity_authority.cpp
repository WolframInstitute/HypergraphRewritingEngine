#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

// =============================================================================
// Automatic event identity is anchored to Wolfram/Multicomputation, on BOTH
// exploration strategies; Positional is pinned to this engine's own values.
//
// The expected Automatic counts are the AUTHORITY'S numbers, produced by
// reference/adjudicate_gap1_authority.wls against the installed
// Wolfram/Multicomputation 0.1.8 (the latest published version), per step, with
// our brute-force oracle's states and Full-identity columns agreeing on every
// row as the depth anchor. They are external facts, not regression snapshots.
//
// The Positional expectations ARE engine snapshots: positional identity depends
// on the canonicalizer's tie-break, so no external oracle can anchor it
// (measured: our own oracle's like-named column differs, 25 vs 23, where
// tie-breaks differ). Pinning them detects unintended change; it cannot certify
// correctness, and nothing can.
// =============================================================================

namespace {

using namespace hypergraph;

struct Case {
    const char* name;
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> init;
    int steps;
    size_t authority_automatic;   // Wolfram/Multicomputation, CanonicalEventFunction -> Automatic
    size_t engine_positional;     // this engine's Positional identity (snapshot)
};

std::vector<Case> cases() {
    std::vector<Case> c;
    c.push_back({"wolfram-2to4",
        {make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).rhs({2,0}).build()},
        {{0,1},{1,2}}, 4, 86, 87});
    c.push_back({"WPP",
        {make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
        {{0,1},{0,2}}, 4, 52, 54});
    c.push_back({"binary-growth",
        {make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
        {{0,1}}, 4, 10, 10});
    c.push_back({"two-rules-overlap",
        {make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
         make_rule(1).lhs({0,1}).rhs({1,2}).rhs({2,0}).build()},
        {{0,1}}, 3, 21, 23});
    return c;
}

struct Obs {
    size_t events;
    std::set<uint64_t> signatures;
    std::vector<std::string> warnings;
};

Obs run(const Case& c, bool quotient, bool positional, int threads) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    g.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
    g.set_positional_event_identity(positional);
    ParallelEvolutionEngine e(&g, threads);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(quotient);
    for (const auto& r : c.rules) e.add_rule(r);
    e.evolve(c.init, c.steps);

    Obs out;
    out.events = g.observable_num_events();
    if (g.quotient_reconstruction()) {
        g.for_each_reconstructed_raw_triple([&](uint64_t s) { out.signatures.insert(s); });
    } else {
        for (uint32_t i = 0; i < g.num_published_events(); ++i) {
            const Event& ev = g.get_event(i);
            if (ev.id != INVALID_ID && ev.is_canonical()) out.signatures.insert(ev.signature);
        }
    }
    out.warnings = e.warnings();
    return out;
}

}  // namespace

// Automatic == the authority's count, whichever exploration strategy ran; and within each
// strategy the signature VALUES are schedule-stable (equal across thread counts).
//
// Signature values are NOT asserted equal ACROSS strategies: they are labels relative to the
// class frame each run pinned, and the two strategies can legitimately pin different members of
// a symmetric class's labelling coset (measured: 85 of 86 values coincide on wolfram-2to4, the
// one difference in a class with a nontrivial automorphism). The observable contract -- golden
// count columns and the state fingerprint, which the quotient twin check enforces -- is
// partition-level, and the partitions agree.
TEST(EventIdentityAuthority, AutomaticMatchesAuthorityOnBothPaths) {
    for (const auto& c : cases()) {
        Obs full1 = run(c, /*quotient=*/false, /*positional=*/false, 1);
        Obs full4 = run(c, /*quotient=*/false, /*positional=*/false, 4);
        Obs quot1 = run(c, /*quotient=*/true, /*positional=*/false, 1);
        Obs quot4 = run(c, /*quotient=*/true, /*positional=*/false, 4);
        EXPECT_EQ(full1.events, c.authority_automatic) << c.name << " full capture";
        EXPECT_EQ(quot1.events, c.authority_automatic) << c.name << " quotient";
        EXPECT_EQ(full4.events, c.authority_automatic) << c.name << " full capture, 4 threads";
        EXPECT_EQ(quot4.events, c.authority_automatic) << c.name << " quotient, 4 threads";
        // Raw content-triple sets: the reconstructed raw event SET is schedule-stable even
        // where the identity labels are not.
        EXPECT_EQ(full1.signatures, full4.signatures)
            << c.name << ": full capture's raw event set is schedule-dependent";
        EXPECT_EQ(quot1.signatures, quot4.signatures)
            << c.name << ": quotient's raw event set is schedule-dependent";
    }
}

// Positional stays available under its own name, produces this engine's pinned values, and
// forces full capture: requesting it with quotient exploration disables the optimisation and
// says so, rather than silently returning identities the mode cannot define.
TEST(EventIdentityAuthority, PositionalPreservedAndForcesFullCapture) {
    for (const auto& c : cases()) {
        Obs full = run(c, /*quotient=*/false, /*positional=*/true, 1);
        EXPECT_EQ(full.events, c.engine_positional) << c.name << " positional";
        EXPECT_TRUE(full.warnings.empty()) << c.name << ": no warning without quotient";

        Obs forced = run(c, /*quotient=*/true, /*positional=*/true, 1);
        EXPECT_EQ(forced.events, c.engine_positional)
            << c.name << " positional+quotient must produce the SAME identities as positional "
                         "alone (the optimisation is disabled, the semantics are not)";
        ASSERT_EQ(forced.warnings.size(), 1u) << c.name;
        EXPECT_NE(forced.warnings[0].find("Positional"), std::string::npos) << c.name;
    }
}
