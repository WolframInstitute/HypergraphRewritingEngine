// Match deduplication must be decided by CONTENT, never by the hash alone.
//
// Every artifact this engine produces -- states, events, causal edges, branchial edges,
// transitive reduction -- is computed from the match set. A match discarded as a "duplicate"
// when it was actually a distinct match whose hash happened to collide is therefore not a local
// error: the whole subtree below it is never explored, and nothing downstream can detect that,
// because the run stays internally consistent. It simply produces less and looks correct.
//
// The probability is n^2 / 2^65 over a run's matches. That is ~5e-8 at a million matches, which
// no test corpus will ever reach, and ~3e-2 at a billion, which is the scale this engine claims.
// A defect that is unreachable in testing and reachable in production cannot be gated by a
// workload -- it has to be gated by driving the collision path directly.
//
// That is what these tests do: they hand the dedup ONE hash and two DIFFERENT matches. A
// key-only set answers the second one wrongly by construction, no matter how good the hash is.

#include <gtest/gtest.h>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

using namespace hypergraph;

namespace {

MatchCore make_core(uint16_t rule, std::initializer_list<EdgeId> edges) {
    MatchCore c;
    c.rule_index = rule;
    c.num_edges = static_cast<uint8_t>(edges.size());
    uint8_t i = 0;
    for (EdgeId e : edges) c.matched_edges[i++] = e;
    return c;
}

}  // namespace

// The defect, driven directly. Two matches that differ in their matched edges are handed the
// same hash. Both are new work; both must be claimed.
TEST(MatchDedupExactness, DistinctMatchesSharingAHashAreBothClaimed) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 1);

    const MatchCore a = make_core(0, {7});
    const MatchCore b = make_core(0, {9});     // a different match, not a duplicate
    const MatchRecord ra{&a, 3};
    const MatchRecord rb{&b, 3};
    ASSERT_FALSE(ra == rb) << "the fixture must supply genuinely distinct matches";

    const uint64_t collided = 0x0123456789ABCDEFull;

    EXPECT_TRUE(engine.claim_match(collided, ra, [&]{ return &ra; })) << "first match is new";
    EXPECT_TRUE(engine.claim_match(collided, rb, [&]{ return &rb; }))
        << "a DISTINCT match sharing a hash was dropped as a duplicate -- this is the defect";

    EXPECT_GE(engine.hash_collisions(), 1u) << "the collision must be counted, not hidden";
    EXPECT_EQ(engine.dedup_probe_exhaustions(), 0u);
}

// The other half of the contract: real duplicates must still be rejected, or the dedup has
// simply been disabled and every match would be rewritten repeatedly.
TEST(MatchDedupExactness, TrueDuplicatesAreStillRejected) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 1);

    const MatchCore a = make_core(0, {7});
    const MatchRecord ra{&a, 3};
    const uint64_t h = ra.hash();

    EXPECT_TRUE(engine.claim_match(h, ra, [&]{ return &ra; }));
    EXPECT_FALSE(engine.claim_match(h, ra, [&]{ return &ra; })) << "the same match claimed twice is a duplicate";
    EXPECT_FALSE(engine.claim_match(h, ra, [&]{ return &ra; }));
    EXPECT_EQ(engine.hash_collisions(), 0u) << "a repeat of one match is not a collision";
}

// source_state is part of the match's identity and is mixed into the hash, but it is NOT part of
// MatchCore. A comparison that only looked at the core would conflate the same core reached from
// two different raw states -- which is precisely the case the hash comment warns about.
TEST(MatchDedupExactness, SameCoreFromDifferentSourceStatesAreDistinct) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 1);

    const MatchCore a = make_core(0, {7});
    const MatchRecord in_s3{&a, 3};
    const MatchRecord in_s4{&a, 4};      // same core, different raw state: a different match

    const uint64_t collided = 0xFEEDFACECAFEBEEFull;
    EXPECT_TRUE(engine.claim_match(collided, in_s3, [&]{ return &in_s3; }));
    EXPECT_TRUE(engine.claim_match(collided, in_s4, [&]{ return &in_s4; }))
        << "matches differing only in source_state must not be conflated";
}

// Several distinct matches on one key, to confirm the probe walk resolves rather than giving up
// after the first step.
TEST(MatchDedupExactness, ManyDistinctMatchesOnOneKeyAllResolve) {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 1);

    constexpr int kN = 6;
    std::vector<MatchRecord> cores_rec;
    std::vector<MatchCore> cores;
    cores.reserve(kN);
    for (int i = 0; i < kN; ++i) cores.push_back(make_core(0, {static_cast<EdgeId>(100 + i)}));
    for (int i = 0; i < kN; ++i) cores_rec.push_back(MatchRecord{&cores[i], 5});

    const uint64_t collided = 0xDEADBEEF12345678ull;
    for (int i = 0; i < kN; ++i) {
        const MatchRecord r{&cores[i], 5};
        EXPECT_TRUE(engine.claim_match(collided, r, [&]{ return &cores_rec[i]; })) << "distinct match " << i << " was dropped";
    }
    // Every one of them must now read back as a duplicate, which is what proves they were
    // stored rather than merely waved through.
    for (int i = 0; i < kN; ++i) {
        const MatchRecord r{&cores[i], 5};
        EXPECT_FALSE(engine.claim_match(collided, r, [&]{ return &cores_rec[i]; })) << "match " << i << " was not retained";
    }
    EXPECT_EQ(engine.dedup_probe_exhaustions(), 0u);
}
