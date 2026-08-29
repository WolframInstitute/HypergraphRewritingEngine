#include <gtest/gtest.h>
#include <hypergraph/ir_canonicalization.hpp>
#include <hypergraph/hypergraph.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include "hgcommon/ir_core.hpp"

using namespace hypergraph;

class IRCanonicalizationTest : public ::testing::Test {
protected:
    IRCanonicalizer ir;
};

// =============================================================================
// Basic correctness
// =============================================================================

TEST_F(IRCanonicalizationTest, EmptyHypergraph) {
    std::vector<std::vector<VertexId>> edges;
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 0);
    EXPECT_EQ(result.canonical_form.vertex_count, 0);
}

TEST_F(IRCanonicalizationTest, SingleVertex) {
    std::vector<std::vector<VertexId>> edges = {{5}};
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 1);
    EXPECT_EQ(result.canonical_form.edges[0], std::vector<VertexId>({0}));
    EXPECT_EQ(result.canonical_form.vertex_count, 1);
}

TEST_F(IRCanonicalizationTest, SingleEdge) {
    std::vector<std::vector<VertexId>> edges = {{10, 20}};
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 1);
    EXPECT_EQ(result.canonical_form.vertex_count, 2);
    // Canonical form should have vertices 0 and 1
    ASSERT_EQ(result.canonical_form.edges[0].size(), 2);
    std::set<VertexId> verts(result.canonical_form.edges[0].begin(),
                              result.canonical_form.edges[0].end());
    EXPECT_EQ(verts, (std::set<VertexId>{0, 1}));
}

// =============================================================================
// Isomorphism detection
// =============================================================================

TEST_F(IRCanonicalizationTest, IsomorphicSimple) {
    // {1,2,3} and {4,5,6} are isomorphic (same structure, different labels)
    std::vector<std::vector<VertexId>> edges1 = {{1, 2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{4, 5, 6}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, IsomorphicMultiEdge) {
    // {{1,2},{2,3}} and {{10,20},{20,30}} are isomorphic
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{10, 20}, {20, 30}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, IsomorphicPermuted) {
    // {{1,2},{2,3},{3,1}} and {{5,6},{6,4},{4,5}} (triangle)
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}, {3, 1}};
    std::vector<std::vector<VertexId>> edges2 = {{5, 6}, {6, 4}, {4, 5}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, NonIsomorphicDifferentStructure) {
    // Path vs star
    std::vector<std::vector<VertexId>> path = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> star = {{1, 2}, {1, 3}};
    auto r1 = ir.canonicalize_edges(path);
    auto r2 = ir.canonicalize_edges(star);
    EXPECT_NE(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, DirectedEdgeOrderPreserved) {
    // {{1,2,3}} ≠ {{1,3,2}} since position matters in directed hypergraphs
    std::vector<std::vector<VertexId>> edges1 = {{1, 2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{1, 3, 2}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    // Both have 1 edge with 3 vertices, but internal order differs
    // The position encoding should distinguish them
    // Actually, {1,2,3} and {1,3,2} ARE isomorphic as directed hyperedges
    // because there exists a vertex renaming (swap 2<->3) that maps one to the other.
    // The position encoding ensures position matters within an edge, but
    // swapping vertex labels can swap positions.
    // So {1,2,3} with mapping 1->0,2->1,3->2 = {0,1,2}
    // And {1,3,2} with mapping 1->0,3->1,2->2 = {0,1,2}
    // They ARE isomorphic.
    EXPECT_EQ(r1.canonical_form, r2.canonical_form);
}

TEST_F(IRCanonicalizationTest, DirectedEdgeOrderNonIsomorphic) {
    // {{1,2},{2,1}} vs {{1,2},{1,2}} - first has two edges with opposite directions
    // These should differ
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 1}};
    std::vector<std::vector<VertexId>> edges2 = {{1, 2}, {1, 2}};
    auto r1 = ir.canonicalize_edges(edges1);
    auto r2 = ir.canonicalize_edges(edges2);
    EXPECT_NE(r1.canonical_form, r2.canonical_form);
}

// =============================================================================
// Hash consistency
// =============================================================================

TEST_F(IRCanonicalizationTest, HashConsistency) {
    std::vector<std::vector<VertexId>> edges1 = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> edges2 = {{10, 20}, {20, 30}};
    uint64_t h1 = ir.compute_canonical_hash(edges1);
    uint64_t h2 = ir.compute_canonical_hash(edges2);
    EXPECT_EQ(h1, h2);
    EXPECT_NE(h1, 0u);
}

TEST_F(IRCanonicalizationTest, HashDiffers) {
    std::vector<std::vector<VertexId>> path = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> star = {{1, 2}, {1, 3}};
    uint64_t h1 = ir.compute_canonical_hash(path);
    uint64_t h2 = ir.compute_canonical_hash(star);
    EXPECT_NE(h1, h2);
}

// =============================================================================
// 1-WL-hard pairs: non-isomorphic graphs whose vertices all share one degree, so
// Weisfeiler-Leman colour refinement stabilizes to a uniform colouring and cannot
// separate them. Exact IR must distinguish them -- this is the whole reason IR
// exists over the fast WL hash, and it was asserted nowhere in the suite.
// =============================================================================
namespace {
std::vector<std::vector<VertexId>> makeCycle(int n) {
    std::vector<std::vector<VertexId>> e;
    for (int i = 0; i < n; ++i)
        e.push_back({(VertexId)i, (VertexId)((i + 1) % n)});
    return e;
}
std::vector<std::vector<VertexId>> makeDisjointCycles(int k, int len) {
    std::vector<std::vector<VertexId>> e;
    for (int c = 0; c < k; ++c) {
        int base = c * len;
        for (int i = 0; i < len; ++i)
            e.push_back({(VertexId)(base + i), (VertexId)(base + (i + 1) % len)});
    }
    return e;
}
}  // namespace

TEST_F(IRCanonicalizationTest, DistinguishesOneWLHardPairs) {
    // Each pair is a single n-cycle vs k disjoint (n/k)-cycles: same vertex count,
    // both 2-regular, so 1-WL assigns every vertex the same colour -- yet they are
    // not isomorphic (one component vs several). Exact IR must give them different
    // canonical hashes; a match would be an IR false-merge.
    struct Pair { const char* name; int n; int k; int len; };
    const Pair pairs[] = {
        {"C6 vs 2xC3", 6, 2, 3},  {"C8 vs 2xC4", 8, 2, 4},   {"C9 vs 3xC3", 9, 3, 3},
        {"C10 vs 2xC5", 10, 2, 5}, {"C12 vs 2xC6", 12, 2, 6},
    };
    for (const auto& p : pairs) {
        uint64_t hCycle = ir.compute_canonical_hash(makeCycle(p.n));
        uint64_t hSplit = ir.compute_canonical_hash(makeDisjointCycles(p.k, p.len));
        EXPECT_NE(hCycle, hSplit)
            << p.name << ": IR false-merged a 1-WL-hard non-isomorphic pair";
    }
}

// =============================================================================
// Isomorphism detection on permuted graphs
// =============================================================================

TEST_F(IRCanonicalizationTest, AgreesOnPermutedGraphs) {
    // Each pair should be isomorphic - verify IR detects this
    std::vector<std::pair<std::vector<std::vector<VertexId>>,
                          std::vector<std::vector<VertexId>>>> pairs = {
        {{{1, 2}, {2, 3}, {3, 1}}, {{5, 6}, {6, 4}, {4, 5}}},  // triangle
        {{{1, 2, 3}, {3, 4, 5}}, {{10, 20, 30}, {30, 40, 50}}},  // shared vertex
        {{{1, 2}, {1, 3}, {1, 4}}, {{7, 8}, {7, 9}, {7, 10}}},  // star
    };
    for (size_t i = 0; i < pairs.size(); ++i) {
        auto ir_a = ir.canonicalize_edges(pairs[i].first);
        auto ir_b = ir.canonicalize_edges(pairs[i].second);
        EXPECT_EQ(ir_a.canonical_form, ir_b.canonical_form)
            << "IR failed isomorphism on pair " << i;
    }
}

TEST_F(IRCanonicalizationTest, CanonicalFormConsistency) {
    // Same graph, different vertex labels → same canonical form
    std::vector<std::vector<VertexId>> a = {{3, 7, 1}};
    std::vector<std::vector<VertexId>> b = {{100, 200, 300}};
    auto ra = ir.canonicalize_edges(a);
    auto rb = ir.canonicalize_edges(b);
    EXPECT_EQ(ra.canonical_form, rb.canonical_form);
}

// =============================================================================
// Vertex mapping correctness
// =============================================================================

TEST_F(IRCanonicalizationTest, VertexMappingCorrect) {
    std::vector<std::vector<VertexId>> edges = {{10, 20}, {20, 30}};
    // Request the inverse (original_to_canonical) maps; the shipping path leaves
    // them empty, so this test opts in via want_inverse_maps.
    auto result = ir.canonicalize_edges(edges, /*want_inverse_maps=*/true);

    // Verify mapping is bijective for original vertices
    EXPECT_EQ(result.vertex_mapping.original_to_canonical.size(), 3);
    EXPECT_EQ(result.vertex_mapping.canonical_to_original.size(), 3);

    std::set<VertexId> canonical_ids;
    for (auto& [orig, canon] : result.vertex_mapping.original_to_canonical) {
        canonical_ids.insert(canon);
    }
    EXPECT_EQ(canonical_ids, (std::set<VertexId>{0, 1, 2}));
}

// =============================================================================
// Performance: IR handles larger graphs than brute-force
// =============================================================================

TEST_F(IRCanonicalizationTest, HandlesLargerGraphs) {
    // Create a chain of 20 edges (brute-force can't handle 20 vertices)
    std::vector<std::vector<VertexId>> edges;
    for (VertexId i = 0; i < 20; ++i) {
        edges.push_back({i, i + 1});
    }
    // Should complete without timeout
    auto result = ir.canonicalize_edges(edges);
    EXPECT_EQ(result.canonical_form.edges.size(), 20);
    EXPECT_EQ(result.canonical_form.vertex_count, 21);
}

// =============================================================================
// Integration: IR verification in Hypergraph
// =============================================================================

TEST_F(IRCanonicalizationTest, AreIsomorphicMethod) {
    std::vector<std::vector<VertexId>> path = {{1, 2}, {2, 3}};
    std::vector<std::vector<VertexId>> path2 = {{10, 20}, {20, 30}};
    std::vector<std::vector<VertexId>> star = {{1, 2}, {1, 3}};

    EXPECT_TRUE(ir.are_isomorphic(path, path2));
    EXPECT_FALSE(ir.are_isomorphic(path, star));
    EXPECT_TRUE(ir.are_isomorphic({}, {}));
}

// =========================================================================================
// The shared core's per-edge orbit/class output against the reference implementation
// (IRCanonicalizer::compute_canonical_hash_with_edge_orbits). Same hash, identical class
// ids (canonical content classes are implementation-independent), and the same orbit
// PARTITION -- orbit ids are each implementation's own deterministic numbering over
// union-find roots, so blocks are compared after first-occurrence normalization.
// =========================================================================================

namespace {

// Flatten a heap edge list to the shared core's convention (local vertex ids assigned in
// sorted original-id order, the engines' rule) and run ir_canonical_hash with orbit and
// class outputs, retrying deeper exactly as the engines do.
uint64_t shared_core_orbits(const std::vector<std::vector<hypergraph::VertexId>>& edges,
                            std::vector<uint32_t>& orbit, std::vector<uint32_t>& klass) {
    const uint32_t e = static_cast<uint32_t>(edges.size());
    std::vector<hypergraph::VertexId> verts;
    for (const auto& ed : edges) for (auto v : ed) verts.push_back(v);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    const uint32_t n = static_cast<uint32_t>(verts.size());

    std::vector<uint8_t> ea(e);
    std::vector<uint32_t> eoff(e + 1);
    std::vector<uint32_t> ev;
    for (uint32_t i = 0; i < e; ++i) {
        ea[i] = static_cast<uint8_t>(edges[i].size());
        eoff[i] = static_cast<uint32_t>(ev.size());
        for (auto v : edges[i]) {
            const uint32_t vi = static_cast<uint32_t>(
                std::lower_bound(verts.begin(), verts.end(), v) - verts.begin());
            ev.push_back(vi);
        }
    }
    eoff[e] = static_cast<uint32_t>(ev.size());
    const uint32_t occ = static_cast<uint32_t>(ev.size());

    orbit.assign(e, 0u);
    klass.assign(e, 0u);
    for (uint32_t depth = 1; depth <= 64; depth *= 8) {
        std::vector<uint32_t> scratch(
            hgcommon::ir_scratch_words(n, e, occ, depth, hgcommon::IR_HOST_GENERATORS));
        hgcommon::IrResult r = hgcommon::ir_canonical_hash(
            ea.data(), eoff.data(), ev.data(), e, n, occ, scratch.data(), depth,
            nullptr, hgcommon::IR_HOST_GENERATORS, orbit.data(), klass.data());
        if (r.status != hgcommon::IR_NEED_DEPTH) return r.hash;
    }
    ADD_FAILURE() << "shared core never finished within depth 64";
    return 0;
}

std::vector<uint32_t> first_occurrence_normalized(const std::vector<uint32_t>& a) {
    std::vector<uint32_t> out(a.size());
    std::map<uint32_t, uint32_t> seen;
    for (size_t i = 0; i < a.size(); ++i) {
        auto it = seen.find(a[i]);
        if (it == seen.end())
            it = seen.emplace(a[i], static_cast<uint32_t>(seen.size())).first;
        out[i] = it->second;
    }
    return out;
}

}  // namespace

TEST_F(IRCanonicalizationTest, SharedCoreEdgeOrbitsMatchReference) {
    const std::vector<std::vector<std::vector<VertexId>>> corpus = {
        {{0, 1}, {1, 2}, {2, 3}},                            // path: trivial Aut
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}},    // C6: one edge orbit
        {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}},    // two identical triangles
        {{0, 1}, {0, 1}, {1, 2}},                            // duplicate edge (one class)
        {{0, 1}, {0, 2}, {0, 3}, {0, 4}},                    // star: leaves symmetric
        {{0, 0}},                                            // self-loop
        {{0, 1, 2}, {2, 1, 0}, {0, 1, 2}},                   // arity 3, dup + reversal
        {{0, 1}, {1, 0}},                                    // directed 2-cycle
        {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}},            // diamond with a tail
    };

    for (size_t ci = 0; ci < corpus.size(); ++ci) {
        const auto& edges = corpus[ci];
        std::vector<uint32_t> orbit_ref, klass_ref;
        const uint64_t h_ref =
            ir.compute_canonical_hash_with_edge_orbits(edges, orbit_ref, &klass_ref);

        std::vector<uint32_t> orbit_core, klass_core;
        const uint64_t h_core = shared_core_orbits(edges, orbit_core, klass_core);

        EXPECT_EQ(h_core, h_ref) << "case " << ci << ": hash differs";
        EXPECT_EQ(klass_core, klass_ref) << "case " << ci << ": class ids differ";
        EXPECT_EQ(first_occurrence_normalized(orbit_core),
                  first_occurrence_normalized(orbit_ref))
            << "case " << ci << ": orbit partition differs";
    }
}

// =========================================================================================
// Presentation invariance of the bounded core.
//
// The engine hands the core local vertex ids assigned in ENCOUNTER order over the state's
// edges, and the edges themselves in EdgeId order. Both are properties of the SCHEDULE that
// built the state, not of the state: EdgeIds come from an atomic increment, so two threads
// reaching isomorphic states present them differently. create_or_get_canonical_state uses the
// returned hash as the dedup key, so if the hash moves with the presentation, isomorphic
// states fail to merge and the canonical count inflates.
//
// This is the invariant the dedup key rests on, stated as a test rather than as a comment.
// =========================================================================================
namespace {

// Re-present a state the way a different schedule would have: permute the edge order and
// relabel the vertices, both driven by one counter so the case is indexed rather than random.
std::vector<std::vector<hypergraph::VertexId>> repermute(
        const std::vector<std::vector<hypergraph::VertexId>>& edges, uint64_t seed) {
    uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    auto next = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };

    std::vector<hypergraph::VertexId> verts;
    for (const auto& e : edges) for (auto v : e) verts.push_back(v);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());

    // A relabeling onto a disjoint, shuffled id range: isomorphic by construction.
    std::vector<hypergraph::VertexId> img(verts.size());
    for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<hypergraph::VertexId>(1000 + i);
    for (size_t i = img.size(); i-- > 1;) std::swap(img[i], img[next() % (i + 1)]);

    std::map<hypergraph::VertexId, hypergraph::VertexId> relabel;
    for (size_t i = 0; i < verts.size(); ++i) relabel[verts[i]] = img[i];

    auto out = edges;
    for (auto& e : out) for (auto& v : e) v = relabel[v];
    for (size_t i = out.size(); i-- > 1;) std::swap(out[i], out[next() % (i + 1)]);
    return out;
}

}  // namespace

TEST_F(IRCanonicalizationTest, BoundedCoreHashIsInvariantUnderPresentation) {
    // Shapes chosen for the properties that make the search branch: a star's leaves are
    // interchangeable, a cycle admits a rotation group, and a disconnected state has an
    // automorphism exchanging its components. These are where a presentation dependence
    // would show, and they are the shapes the corpus generator emits.
    std::vector<std::pair<const char*, std::vector<std::vector<VertexId>>>> cases;

    std::vector<std::vector<VertexId>> star;
    for (VertexId i = 1; i <= 12; ++i) star.push_back({0, i});
    cases.emplace_back("star12", star);

    std::vector<std::vector<VertexId>> cycle;
    for (VertexId i = 0; i < 12; ++i) cycle.push_back({i, static_cast<VertexId>((i + 1) % 12)});
    cases.emplace_back("cycle12", cycle);

    // Two isomorphic components: the automorphism group contains the swap between them.
    std::vector<std::vector<VertexId>> disc;
    for (VertexId c = 0; c < 2; ++c)
        for (VertexId i = 0; i < 6; ++i)
            disc.push_back({static_cast<VertexId>(c * 100 + i),
                            static_cast<VertexId>(c * 100 + (i + 1) % 6)});
    cases.emplace_back("disc2x6", disc);

    // A star of triangles: high automorphism AND mixed arity, the combination the corpus's
    // growth rules produce after a few steps.
    std::vector<std::vector<VertexId>> mixed;
    for (VertexId i = 0; i < 5; ++i) {
        mixed.push_back({0, static_cast<VertexId>(1 + i * 2), static_cast<VertexId>(2 + i * 2)});
        mixed.push_back({static_cast<VertexId>(1 + i * 2), static_cast<VertexId>(2 + i * 2)});
    }
    cases.emplace_back("mixedarity", mixed);

    for (const auto& [name, base] : cases) {
        std::vector<uint32_t> orbit, klass;
        const uint64_t h0 = shared_core_orbits(base, orbit, klass);
        for (uint64_t seed = 1; seed <= 64; ++seed) {
            auto perm = repermute(base, seed);
            std::vector<uint32_t> o2, k2;
            const uint64_t h = shared_core_orbits(perm, o2, k2);
            ASSERT_EQ(h, h0) << name << ": bounded core hash moved under presentation "
                             << seed << " -- isomorphic states will not dedup";
        }
    }
}
