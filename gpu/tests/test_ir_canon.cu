// Unit tests for the GPU IR canonicalisation fast path
// (gpu/src/ir_canon.cu, S3.2 / S3.2b).
//
// What we actually test (and what we don't):
//   - The GPU IR hash function is *isomorphism-invariant*: relabel a
//     graph and you get the same hash.
//   - The GPU IR hash function *distinguishes* non-isomorphic graphs that
//     1-WL refinement can separate (which covers everything we test
//     here — the adversarial graphs that need the IR backtrack-tree
//     fallback are deferred to S3.5's tests).
//
// What we do NOT test: byte-for-byte equality between the GPU IR hash
// and the CPU `hypergraph::IRCanonicalizer::compute_canonical_hash`.
// Both produce iso-invariant canonical hashes but use different
// internal mixing / packing, so the numeric values differ. The cross-
// engine equivalence-class agreement is enforced separately by the
// differential test in test_gpu_vs_cpu_differential.cpp, which
// re-canonicalises both engines' state edge lists through the CPU
// IRCanonicalizer post-hoc.

#include <gtest/gtest.h>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/evolve.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace {

using hg_gpu::VertexId;
using EdgeList = std::vector<std::vector<VertexId>>;

hg_gpu::EngineConfig small_cfg() {
    hg_gpu::EngineConfig cfg;
    cfg.max_edges            = 64;
    cfg.max_state_edge_total = 256;
    cfg.max_states           = 8;
    cfg.max_vertex_slots     = 256;
    // Bumped so iso-invariance tests can use vertex IDs in arbitrary
    // ranges (e.g. {7,8,9}, {42,17,99}, {100..103}). The vertex-inverted
    // index is keyed on global VertexId, so vertices must fit in
    // [0, max_vertices) at upload time.
    cfg.max_vertices         = 256;
    cfg.sig_index_buckets    = 16;
    cfg.sig_index_pool       = 64;
    cfg.inverted_pool        = 256;
    return cfg;
}

uint64_t gpu_ir_hash(const EdgeList& edges) {
    hg_gpu::EngineState eng(small_cfg());
    hg_gpu::upload_initial_state(eng, edges);
    return hg_gpu::compute_state_ir_hash_host(eng, /*sid=*/0);
}

}  // namespace

// -------------------------------------------------------------------------
// Empty / trivial cases
// -------------------------------------------------------------------------

TEST(IRCanonGPU, EmptyGraphTakesTheReservedHash) {
    // A canonicalizer given no edges has nothing to compute a hash from, so the empty state's
    // is reserved. The value is shared with the host (hgcommon/core.hpp): it is what each
    // engine deduplicates on and what a caller reads back, so the two must agree on it.
    //
    // Not 0: 0 is what the per-state hash array holds for "not computed yet", and it is the
    // EMPTY sentinel of every map this hash keys.
    EXPECT_EQ(gpu_ir_hash(EdgeList{}), hgcommon::EMPTY_STATE_CANONICAL_HASH);
    EXPECT_NE(gpu_ir_hash(EdgeList{}), 0u);
}

TEST(IRCanonGPU, SingleEdgeIsoInvariant) {
    // {(0,1)} and {(7,9)} are isomorphic — same canonical hash.
    EXPECT_EQ(gpu_ir_hash({{0u, 1u}}), gpu_ir_hash({{7u, 9u}}));
    EXPECT_NE(gpu_ir_hash({{0u, 1u}}), 0u);
}

// -------------------------------------------------------------------------
// Iso-invariance
// -------------------------------------------------------------------------

TEST(IRCanonGPU, TriangleRelabellingsAgree) {
    EdgeList tri_a = {{0u, 1u}, {1u, 2u}, {2u, 0u}};
    EdgeList tri_b = {{7u, 8u}, {8u, 9u}, {9u, 7u}};
    EdgeList tri_c = {{42u, 17u}, {17u, 99u}, {99u, 42u}};
    uint64_t ha = gpu_ir_hash(tri_a);
    uint64_t hb = gpu_ir_hash(tri_b);
    uint64_t hc = gpu_ir_hash(tri_c);
    EXPECT_EQ(ha, hb);
    EXPECT_EQ(ha, hc);
    EXPECT_NE(ha, 0u);
}

TEST(IRCanonGPU, EdgeOrderInInitialStateDoesNotMatter) {
    EdgeList a = {{0u, 1u}, {1u, 2u}, {2u, 0u}};
    EdgeList b = {{2u, 0u}, {0u, 1u}, {1u, 2u}};
    EdgeList c = {{1u, 2u}, {2u, 0u}, {0u, 1u}};
    EXPECT_EQ(gpu_ir_hash(a), gpu_ir_hash(b));
    EXPECT_EQ(gpu_ir_hash(a), gpu_ir_hash(c));
}

TEST(IRCanonGPU, TwoDisjointTrianglesIsoInvariant) {
    // Two disjoint triangles — relabel both components.
    EdgeList g1 = {
        {0u, 1u}, {1u, 2u}, {2u, 0u},
        {3u, 4u}, {4u, 5u}, {5u, 3u},
    };
    EdgeList g2 = {
        {10u, 11u}, {11u, 12u}, {12u, 10u},
        {20u, 21u}, {21u, 22u}, {22u, 20u},
    };
    EXPECT_EQ(gpu_ir_hash(g1), gpu_ir_hash(g2));
}

// -------------------------------------------------------------------------
// Distinguishability — non-isomorphic graphs hash differently
// -------------------------------------------------------------------------

TEST(IRCanonGPU, TriangleVsPathDifferentHash) {
    // Triangle (closed loop) vs path of 3 directed edges (open chain).
    EdgeList tri  = {{0u, 1u}, {1u, 2u}, {2u, 0u}};
    EdgeList path = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
    EXPECT_NE(gpu_ir_hash(tri), gpu_ir_hash(path));
}

TEST(IRCanonGPU, DirectionalityMattersInPathStructure) {
    // {(0,1)} and {(1,0)} alone are isomorphic (φ: 0↔1 maps one to the
    // other) so they have the same canonical hash. Directionality only
    // becomes observable when a vertex's role across multiple edges
    // differs — e.g. vertex 1 in {(0,1),(1,2)} appears once at pos 1
    // (incoming) and once at pos 0 (outgoing): a path with through-flow.
    // Vertex 1 in {(0,1),(2,1)} appears at pos 1 twice (two incoming
    // edges, no outgoing). These graphs have non-isomorphic
    // (in-degree, out-degree) profiles for vertex 1, so they must hash
    // differently.
    EdgeList path        = {{0u, 1u}, {1u, 2u}};   // path: 0→1→2
    EdgeList double_sink = {{0u, 1u}, {2u, 1u}};   // 1 is sink of two
    EXPECT_NE(gpu_ir_hash(path), gpu_ir_hash(double_sink));
}

TEST(IRCanonGPU, SelfLoopDistinguishedFromRegularEdge) {
    EXPECT_NE(gpu_ir_hash({{5u, 5u}}), gpu_ir_hash({{5u, 6u}}));
}

TEST(IRCanonGPU, TwoEdgesSharedVsDisjointVertex) {
    // {(0,1),(1,2)} (path, shared vertex) vs {(0,1),(2,3)} (two disjoint
    // edges) — different connectivity, must differ.
    EdgeList path     = {{0u, 1u}, {1u, 2u}};
    EdgeList disjoint = {{0u, 1u}, {2u, 3u}};
    EXPECT_NE(gpu_ir_hash(path), gpu_ir_hash(disjoint));
}

// -------------------------------------------------------------------------
// Mixed arity — exercises the (arity, position) initial-colour signature
// -------------------------------------------------------------------------

TEST(IRCanonGPU, MixedArityIsoInvariant) {
    EdgeList g1 = {{0u, 1u}, {1u, 2u, 3u}};
    EdgeList g2 = {{10u, 11u}, {11u, 22u, 33u}};
    EXPECT_EQ(gpu_ir_hash(g1), gpu_ir_hash(g2));
}

TEST(IRCanonGPU, BinaryVsTernarySingleEdgeDifferent) {
    EXPECT_NE(gpu_ir_hash({{0u, 1u}}), gpu_ir_hash({{0u, 1u, 2u}}));
}

// -------------------------------------------------------------------------
// Slightly bigger — K4 (complete digraph over 4 vertices: 12 directed edges)
// -------------------------------------------------------------------------

TEST(IRCanonGPU, K4RelabellingsAgree) {
    auto make_k4 = [] (uint32_t base) -> EdgeList {
        EdgeList g;
        for (uint32_t a = 0; a < 4; ++a) {
            for (uint32_t b = 0; b < 4; ++b) {
                if (a != b) g.push_back({base + a, base + b});
            }
        }
        return g;
    };
    EXPECT_EQ(gpu_ir_hash(make_k4(0)), gpu_ir_hash(make_k4(100)));
}

// A LARGE state gets an EXACT, relabelling-invariant hash -- there is no size at which the
// canonicalizer starts keying states by something coarser.
//
// The slot is sized from the state's own edge and occurrence counts and claimed from a device
// arena, so 400 edges is no different in kind from 4. The property matters because a coarser
// key would be a WRONG dedup key rather than a conservative one: 1-WL never separates
// isomorphic states but it does MERGE non-isomorphic ones, which tools/ir_vs_wl demonstrates
// on six-vertex graphs.
TEST(IrCanon, ALargeStateIsStillHashedExactly) {
    const uint32_t kEdges = 400;   // well past the old 128-edge cap

    hg_gpu::EngineConfig cfg;
    cfg.max_edges            = kEdges * 2;
    cfg.max_state_edge_total = kEdges * 4;
    cfg.max_states           = 8;
    cfg.max_vertex_slots     = kEdges * 4;
    cfg.max_vertices         = kEdges * 4;
    cfg.sig_index_buckets    = 64;
    cfg.sig_index_pool       = kEdges * 4;
    cfg.inverted_pool        = kEdges * 4;

    // A long path: enough structure that refinement discretises, and far past the old cap.
    EdgeList edges;
    for (uint32_t i = 0; i < kEdges; ++i)
        edges.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});

    hg_gpu::EngineState eng(cfg);
    hg_gpu::upload_initial_state(eng, edges);
    const uint64_t h = hg_gpu::compute_state_ir_hash_host(eng, /*sid=*/0);
    EXPECT_NE(h, 0u) << "a state of " << kEdges << " edges was not keyed at all";

    // Same graph, vertices relabelled by a shift: an exact canonical hash is invariant.
    EdgeList shifted;
    for (const auto& e : edges)
        shifted.push_back({static_cast<VertexId>(e[0] + 1000), static_cast<VertexId>(e[1] + 1000)});
    hg_gpu::EngineConfig cfg2 = cfg;
    cfg2.max_vertices     = kEdges * 4 + 2000;
    cfg2.max_vertex_slots = kEdges * 4 + 2000;
    hg_gpu::EngineState eng2(cfg2);
    hg_gpu::upload_initial_state(eng2, shifted);
    const uint64_t h2 = hg_gpu::compute_state_ir_hash_host(eng2, /*sid=*/0);
    EXPECT_EQ(h, h2) << "relabelling changed the exact hash";
}

// A state whose individualization search outruns the device depth is RETRIED, not degraded.
//
// Many identical disjoint components give an automorphism group 1-WL cannot discretise, so the
// search individualizes repeatedly and runs past EngineConfig::ir_depth. What must NOT happen
// is a coarser dedup key: 1-WL never separates isomorphic states but it DOES merge
// non-isomorphic ones (tools/ir_vs_wl collides the prism against K3,3 on six vertices), so a
// caller promised exactness under CanonicalizeStates -> Full would silently not get it.
//
// The depth is a config field, so the exhaustion is a capacity overflow the wrapper answers by
// doubling it. Checked through hg_gpu::evolve, which is where the retry lives.
TEST(IrCanon, ADeepIndividualizationSearchIsRetriedRatherThanDegraded) {
    const uint32_t kComponents = 64;    // 64 indistinguishable 2-edge paths

    std::vector<std::vector<hg_gpu::VertexId>> edges;
    for (uint32_t c = 0; c < kComponents; ++c) {
        const hg_gpu::VertexId a = static_cast<hg_gpu::VertexId>(c * 3);
        edges.push_back({a, static_cast<hg_gpu::VertexId>(a + 1)});
        edges.push_back({static_cast<hg_gpu::VertexId>(a + 1),
                         static_cast<hg_gpu::VertexId>(a + 2)});
    }

    hg_gpu::EvolveInput in;
    in.rules = {};                 // no rewriting: the root's canonicalization is the subject
    in.initial_state = edges;
    in.num_steps = 0;
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
    in.explore_from_canonical_states_only = true;

    hg_gpu::EvolveResult r = hg_gpu::evolve(in);

    ASSERT_FALSE(r.states.empty()) << "the state was not returned at all";
    for (const auto& w : r.warnings) {
        EXPECT_NE(w.kind, hg_gpu::ErrorKind::kUncomputedStateHash)
            << "a state reached deduplication with no hash, so the depth retry did not converge";
        EXPECT_NE(w.kind, hg_gpu::ErrorKind::kIRDepthExceeded)
            << "the depth was exhausted and the run finished anyway; grow-and-retry should have "
               "doubled EngineConfig::ir_depth until the search fit";
    }
}
