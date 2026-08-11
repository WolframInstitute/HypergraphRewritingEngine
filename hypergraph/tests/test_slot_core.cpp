// The two access patterns of the frame-slot rule must agree for every edge, on every shape.
// The host fills a state at once; the device reads one edge at a time. Nothing else asserts
// that those are the same function.
#include <gtest/gtest.h>
#include "hgcommon/slot_core.hpp"
#include "hgcommon/quotient_causal_core.hpp"
#include "hgcommon/wl_core.hpp"
#include "hypergraph/atomic_compat.hpp"
#include "hypergraph/types.hpp"
#include <random>
#include <type_traits>
#include <vector>

TEST(SlotCore, BulkFormEqualsDefinition) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 400; ++trial) {
        const uint32_t n = 1 + rng() % 64;
        const uint32_t k = 1 + rng() % n;              // orbit count
        std::vector<uint32_t> orbit(n);
        for (uint32_t i = 0; i < n; ++i) orbit[i] = rng() % k;
        uint32_t num_orbits = 0;
        for (uint32_t o : orbit) num_orbits = std::max(num_orbits, o + 1);

        std::vector<uint32_t> bulk(n), counts(num_orbits);
        hgcommon::slots_from_orbits(orbit.data(), n, bulk.data(), counts.data(), num_orbits);

        std::vector<uint32_t> seen(n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            ASSERT_EQ(bulk[i], hgcommon::slot_rank(orbit.data(), n, i))
                << "trial " << trial << " edge " << i;
            ASSERT_LT(bulk[i], n);
            ++seen[bulk[i]];
        }
        // A frame, not a labelling with holes.
        for (uint32_t c : seen) ASSERT_EQ(c, 1u);
    }
}

// Orbit order dominates; ties inside an orbit follow ascending index (== ascending EdgeId,
// because both engines hand edges over in id order).
TEST(SlotCore, OrbitDominatesAndTiesFollowIndex) {
    const uint32_t orbit[] = {1, 0, 1, 0};
    uint32_t out[4], counts[2];
    hgcommon::slots_from_orbits(orbit, 4, out, counts, 2);
    EXPECT_EQ(out[1], 0u);
    EXPECT_EQ(out[3], 1u);
    EXPECT_EQ(out[0], 2u);
    EXPECT_EQ(out[2], 3u);
}

// ---------------------------------------------------------------------------------------
// The canonical transition's dedup signature. Host and device call this one body to decide
// which raw events ARE the same transition, so the value itself is an interface: a change to
// it changes which transitions dedup, and a disagreement between the two sides means the two
// devices drop different ones.
//
// Pinned rather than merely self-consistent. Two open-coded copies of this existed and agreed
// on a MISTYPED basis (one digit short of the FNV offset), so a test comparing the two would
// have passed while both were wrong. The literal below is the value the shared body produces
// and is what the device's own test must reproduce.

TEST(QcTransitionSig, SeparatorTagsKeepConsumedAndSurvivorOrbitsDistinct) {
    // Same orbit value in the two runs. Without the 0x1111/0x2222 tags these two transitions
    // are the same sequence of mixed inputs and collide, which silently drops one of them.
    const uint32_t consumed[] = {3};
    const uint64_t survivor[] = {(uint64_t{3} << 32) | 0};

    const uint64_t only_consumed =
        hgcommon::qc_transition_sig(11, 22, 1, consumed, 1, nullptr, 0);
    const uint64_t only_survivor =
        hgcommon::qc_transition_sig(11, 22, 1, nullptr, 0, survivor, 1);
    EXPECT_NE(only_consumed, only_survivor);
}

TEST(QcTransitionSig, EveryComponentIsLoadBearing) {
    const uint32_t consumed[] = {2, 5};
    const uint64_t surv[] = {(uint64_t{1} << 32) | 4};
    const uint64_t base = hgcommon::qc_transition_sig(101, 202, 7, consumed, 2, surv, 1);

    // A signature that ignored any of its inputs would dedup two different transitions into one.
    EXPECT_NE(base, hgcommon::qc_transition_sig(999, 202, 7, consumed, 2, surv, 1));
    EXPECT_NE(base, hgcommon::qc_transition_sig(101, 999, 7, consumed, 2, surv, 1));
    EXPECT_NE(base, hgcommon::qc_transition_sig(101, 202, 8, consumed, 2, surv, 1));

    const uint32_t other_consumed[] = {2, 6};
    EXPECT_NE(base, hgcommon::qc_transition_sig(101, 202, 7, other_consumed, 2, surv, 1));
    EXPECT_NE(base, hgcommon::qc_transition_sig(101, 202, 7, consumed, 1, surv, 1));

    const uint64_t other_surv[] = {(uint64_t{1} << 32) | 5};
    EXPECT_NE(base, hgcommon::qc_transition_sig(101, 202, 7, consumed, 2, other_surv, 1));

    // The two halves of a packed survivor are mixed separately, so swapping them must differ.
    const uint64_t swapped[] = {(uint64_t{4} << 32) | 1};
    EXPECT_NE(base, hgcommon::qc_transition_sig(101, 202, 7, consumed, 2, swapped, 1));
}

TEST(QcTransitionSig, NeverReturnsAMapSentinel) {
    // seen_transitions_ is a ConcurrentMap with EMPTY=0 and LOCKED=~0. A key equal to either is
    // REJECTED, not stored, so a signature carrying one would abort the host run and be remapped
    // on the device -- the two sides would then disagree about a transition rather than about a
    // number. The guard belongs in the shared body, which is what this asserts.
    for (uint64_t from = 0; from < 512; ++from) {
        const uint64_t sig = hgcommon::qc_transition_sig(from, from * 7 + 1, 0, nullptr, 0,
                                                         nullptr, 0);
        EXPECT_NE(sig, uint64_t{0});
        EXPECT_NE(sig, ~uint64_t{0});
    }
}

TEST(QcTransitionSig, PinnedValueIsTheDeviceContract) {
    const uint32_t consumed[] = {0, 3};
    const uint64_t surv[] = {(uint64_t{2} << 32) | 5};
    const uint64_t sig = hgcommon::qc_transition_sig(0xABCDEF0123456789ULL,
                                                     0x0123456789ABCDEFULL,
                                                     4, consumed, 2, surv, 1);
    // Recompute independently from the documented definition: FNV-1a over
    // from, to, rule, then (0x1111, orbit) per consumed, then (0x2222, hi, lo) per survivor.
    uint64_t want = hgcommon::FNV_OFFSET;
    want = hgcommon::fnv_hash(want, 0xABCDEF0123456789ULL);
    want = hgcommon::fnv_hash(want, 0x0123456789ABCDEFULL);
    want = hgcommon::fnv_hash(want, 4);
    want = hgcommon::fnv_hash(want, 0x1111); want = hgcommon::fnv_hash(want, 0);
    want = hgcommon::fnv_hash(want, 0x1111); want = hgcommon::fnv_hash(want, 3);
    want = hgcommon::fnv_hash(want, 0x2222); want = hgcommon::fnv_hash(want, 2);
    want = hgcommon::fnv_hash(want, 5);
    EXPECT_EQ(sig, want);

    // And the basis is the real FNV offset, not the digit-dropped literal that shipped in two
    // open-coded copies of this rule.
    EXPECT_EQ(hgcommon::FNV_OFFSET, 14695981039346656037ULL);
}

// ---------------------------------------------------------------------------------------
// Content-ordered identity: what Automatic deduplicates states by, and the LAST rule in this
// codebase that was written twice. The host walks a SparseBitset, the device walks an edge
// slice with a liveness filter, so the iteration cannot be shared -- but every constant and
// every mixing step must be, and hgcommon::ContentHasher is where they now live.
//
// Pinned by VALUE, not by comparing the two callers. Comparing them is what a duplicated rule
// always passes: the marshaller's own third copy of this concept agreed with nothing and no
// test noticed, because no test asked what the answer should BE.

TEST(ContentHasher, EdgeCountIsHashedSoASubStateCannotCollide) {
    // Without the leading count, {{1,2}} and {{1,2},{}} differ only by an edge that contributes
    // nothing, and a state whose edges are a prefix of another's could land on the same key.
    hgcommon::ContentHasher one(1);
    one.edge_begin(2); one.vertex(1); one.vertex(2); one.edge_end();

    hgcommon::ContentHasher two(2);
    two.edge_begin(2); two.vertex(1); two.vertex(2); two.edge_end();

    EXPECT_NE(one.value(), two.value());
}

TEST(ContentHasher, TheSeparatorSeparates) {
    // Same vertex sequence, different edge boundaries. Without edge_end these are one input.
    hgcommon::ContentHasher a(2);
    a.edge_begin(2); a.vertex(1); a.vertex(2); a.edge_end();
    a.edge_begin(1); a.vertex(3); a.edge_end();

    hgcommon::ContentHasher b(2);
    b.edge_begin(1); b.vertex(1); b.edge_end();
    b.edge_begin(2); b.vertex(2); b.vertex(3); b.edge_end();

    EXPECT_NE(a.value(), b.value());
}

TEST(ContentHasher, OrderIsPartOfTheIdentity) {
    // CONTENT-ordered, deliberately not isomorphism-invariant: a relabelling is a different
    // state under Automatic. That is the whole difference from the Full path, so a hasher that
    // ignored order would silently make Automatic mean Full-lite.
    hgcommon::ContentHasher fwd(1);
    fwd.edge_begin(2); fwd.vertex(1); fwd.vertex(2); fwd.edge_end();

    hgcommon::ContentHasher rev(1);
    rev.edge_begin(2); rev.vertex(2); rev.vertex(1); rev.edge_end();

    EXPECT_NE(fwd.value(), rev.value());
}

TEST(ContentHasher, PinnedValueIsTheDeviceContract) {
    hgcommon::ContentHasher ch(2);
    ch.edge_begin(2); ch.vertex(7); ch.vertex(9); ch.edge_end();
    ch.edge_begin(3); ch.vertex(1); ch.vertex(2); ch.vertex(3); ch.edge_end();

    // Recomputed from the documented definition rather than from the object: count, then per
    // edge the arity, its vertices, and the separator -- every step through mix64 and fnv_hash.
    uint64_t want = hgcommon::fnv_hash(hgcommon::FNV_OFFSET, hgcommon::mix64(2));
    want = hgcommon::fnv_hash(want, hgcommon::mix64(2));
    want = hgcommon::fnv_hash(want, hgcommon::mix64(7));
    want = hgcommon::fnv_hash(want, hgcommon::mix64(9));
    want = hgcommon::fnv_hash(want, 0xDEADBEEFCAFEBABEull);
    want = hgcommon::fnv_hash(want, hgcommon::mix64(3));
    want = hgcommon::fnv_hash(want, hgcommon::mix64(1));
    want = hgcommon::fnv_hash(want, hgcommon::mix64(2));
    want = hgcommon::fnv_hash(want, hgcommon::mix64(3));
    want = hgcommon::fnv_hash(want, 0xDEADBEEFCAFEBABEull);

    EXPECT_EQ(ch.value(), want);
}

// ---------------------------------------------------------------------------------------
// THE NAMESPACE ROOT. Linking this engine into another program must add exactly ONE name to
// the global namespace, and the root must be renameable by whoever links it -- a library
// cannot know it has not collided.
//
// Compile-time, because that is what the claim is: these are assertions about NAMES, and a
// runtime check could only observe what the compiler already decided.

TEST(NamespaceRoot, TheRootExistsAndTheShortAliasNamesTheSameEntity) {
    // The symbol genuinely lives under the root...
    static_assert(HG_NAMESPACE::common::FNV_OFFSET == 14695981039346656037ULL,
                  "hgcommon's symbols are not reachable through the namespace root");
    // ...and the short alias is that same entity, not a second declaration of it. If the alias
    // ever bound to a separate namespace, these would be two constants that merely agree.
    static_assert(&HG_NAMESPACE::common::FNV_OFFSET == &hgcommon::FNV_OFFSET,
                  "the short alias does not name the root's namespace");
    SUCCEED();
}

TEST(NamespaceRoot, EverySubsystemIsUnderTheRootAndItsShortNameIsAnAlias) {
    // Eight subsystems moved, and each short name must be an ALIAS of the nested one rather
    // than a namespace that still exists in its own right. A namespace alias and its target are
    // the same scope, so a type named through either is one type -- which is what these assert.
    static_assert(std::is_same_v<hgcommon::ContentHasher, HG_NAMESPACE::common::ContentHasher>);
    static_assert(std::is_same_v<hypergraph::StateCanonicalizationMode,
                                 HG_NAMESPACE::engine::StateCanonicalizationMode>);
    static_assert(std::is_same_v<hgcommon::atomic_ref<uint32_t>,
                                 HG_NAMESPACE::common::atomic_ref<uint32_t>>,
                  "atomic_ref must live in a subsystem, not directly in the root");
    SUCCEED();
}

TEST(NamespaceRoot, TheRootIsAMacroSoALinkerCanRenameIt) {
    // -DHG_NAMESPACE=whatever has to move every symbol without editing the engine, which is
    // only true while the declarations open the root through the macro rather than naming it.
    // A build that hardcoded `namespace hg {` would still compile and would silently ignore the
    // override, so the check is that the macro is what is defined.
#ifndef HG_NAMESPACE
    FAIL() << "HG_NAMESPACE is not defined; the root cannot be overridden at build time";
#endif
    SUCCEED();
}
