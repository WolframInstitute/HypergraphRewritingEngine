#include "hgcommon/namespace.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/ir_core.hpp"

#include <algorithm>
#include <vector>

namespace HG_NAMESPACE {
namespace engine {

namespace {

// The canonical form itself, for the caller that wants the relabelled state rather than a
// digest of it. `form` is the core's flat [arity, v0, v1, ...] run per edge in canonical edge
// order; `vertex_label` maps a local vertex index to its canonical label and `orig_vertex` maps
// the same index back to the caller's VertexId, so the two together are the vertex mapping.
struct CoreForm {
    std::vector<uint32_t> form;
    std::vector<uint32_t> vertex_label;
    std::vector<uint32_t> orig_vertex;
    uint32_t n_verts = 0;
    bool ok = false;
};

// The shared core, over one state's edges, escalating the individualization depth.
//
// Local vertex indices are assigned by SORTED-UNIQUE order of the caller's vertex ids; see the
// note at the numbering below for why that choice is load-bearing.
uint64_t ir_core_call(const SVec<SVec<VertexId>>& edges,
                      uint32_t* rank, uint32_t* orbit, uint32_t* klass,
                      CoreForm* form_out = nullptr) {
    const uint32_t n_edges = static_cast<uint32_t>(edges.size());
    // PER-THREAD AND REUSED, for the reason the scratch buffer below already records: as locals
    // these four allocate and free on EVERY call, and this function runs once per raw state.
    // Callgrind on path-l2a2g1r1 put malloc, _int_malloc and _int_free together at 5.32% of all
    // instructions with 2.3 million allocations in a single-worker run. The allocator is also a
    // shared lock, so the cost is not only the instructions -- it is a serialization point that
    // every worker reaches once per state, which is parallel efficiency spent on nothing.
    //
    // clear() keeps the capacity, so a thread pays for its largest state once and reuses it.
    // Not reentrant, and does not need to be: ir_core_call calls the core, never itself.
    static thread_local std::vector<uint8_t> ea;
    static thread_local std::vector<uint32_t> eoff, ev, sorted;
    ea.clear(); eoff.clear(); ev.clear();
    ea.reserve(n_edges); eoff.reserve(n_edges);
    for (const auto& e : edges) {
        eoff.push_back(static_cast<uint32_t>(ev.size()));
        ea.push_back(static_cast<uint8_t>(e.size()));
        for (VertexId v : e) ev.push_back(static_cast<uint32_t>(v));
    }
    // SORTED-UNIQUE vertex order, not encounter order, and the difference is not cosmetic.
    // The canonical labelling of a state with a nontrivial automorphism group is a COSET, and
    // the core's within-cell tie-break is what picks one representative out of it. That
    // tie-break reads the vertex NUMBERING, so two numberings pick two different
    // representatives -- each internally valid, agreeing on the canonical HASH, and
    // disagreeing about which edge holds which RANK. Measured: encounter order differs from
    // this on 103 of the equivalence probe's 4063 states, all of them the symmetric ones.
    // This class's callers expect the sorted convention, which is what build_adjacency used.
    sorted.assign(ev.begin(), ev.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    const uint32_t n_verts = static_cast<uint32_t>(sorted.size());
    for (uint32_t& x : ev)
        x = static_cast<uint32_t>(
            std::lower_bound(sorted.begin(), sorted.end(), x) - sorted.begin());
    const uint32_t total_occ = static_cast<uint32_t>(ev.size());

    uint32_t* out_form = nullptr;
    uint32_t* out_label = nullptr;
    if (form_out) {
        form_out->form.assign(hgcommon::ir_canonical_form_words(n_edges, total_occ), 0u);
        form_out->vertex_label.assign(n_verts, 0u);
        form_out->orig_vertex.assign(sorted.begin(), sorted.end());
        form_out->n_verts = n_verts;
        out_form = form_out->form.data();
        out_label = form_out->vertex_label.data();
    }

    // The generator budget escalates only when ORBITS are requested: for search pruning a
    // short table costs time and cannot change the canonical form, since automorphic branches
    // reach the same form either way. For orbits it changes the answer.
    const bool want_orbits = (orbit != nullptr) || (klass != nullptr);
    const uint32_t gen_hi = want_orbits ? (1u << 16) : hgcommon::IR_HOST_GENERATORS;
    // PER-THREAD AND REUSED, because a local one is re-zeroed on every call. The buffer only
    // grows, and assign() below zeroes only when it does, so as a local this paid a full
    // memset per call: callgrind on ir_vs_wl put __memset_avx2 at 15.03% of all instructions
    // with 98.9% of it reached through this function, over 103,474 calls. Hoisting it changes
    // no semantics -- the buffer is still zeroed whenever it grows -- and removes the repeat.
    //
    // The sibling caller of this same core (Hypergraph::compute_exact_canonical_hash) takes the
    // per-worker arena instead and does not zero at all, on the stated grounds that the core
    // writes every word it later reads. That is the stronger claim; this is the one that needs
    // no claim.
    static thread_local std::vector<uint32_t> scratch;
    for (uint32_t depth : {1u, 8u, hgcommon::IR_MAX_DEPTH_DEFAULT}) {
        for (uint32_t gens = hgcommon::IR_HOST_GENERATORS; gens <= gen_hi; gens *= 4u) {
            const uint64_t words =
                hgcommon::ir_scratch_words(n_verts, n_edges, total_occ, depth, gens);
            if (scratch.size() < words + 2) scratch.assign(words + 2, 0);
            auto r = hgcommon::ir_canonical_hash(ea.data(), eoff.data(), ev.data(),
                                                 n_edges, n_verts, total_occ,
                                                 scratch.data(), depth, rank, gens,
                                                 orbit, klass, out_form, out_label);
            if (r.status == hgcommon::IR_OK) {
                if (form_out) form_out->ok = true;
                return r.hash;
            }
            if (r.status == hgcommon::IR_EMPTY) return 0;
            if (r.status == hgcommon::IR_NEED_DEPTH) break;   // more generators cannot help
        }
    }
    return 0;
}

}  // namespace

CanonicalizationResult IRCanonicalizer::canonicalize_edges(
    const SVec<SVec<VertexId>>& edges, bool want_inverse_maps) const {
    if (edges.empty()) {
        CanonicalizationResult result;
        result.canonical_form.vertex_count = 0;
        return result;
    }

    // Canonicalization scratch draws from the per-worker scratch arena; one
    // mark/release reclaims it in bulk. The returned result is heap (caller owns).
    auto scratch_mark = worker_scratch().mark();
    CoreForm cf;
    std::vector<uint32_t> rank(edges.size(), 0u);
    ir_core_call(edges, rank.data(), nullptr, nullptr, &cf);
    worker_scratch().release(scratch_mark);

    CanonicalizationResult result;
    if (!cf.ok) {
        // No vertices to label: every edge is empty, so the canonical form is that many empty
        // edges over zero vertices, and no mapping is meaningful.
        result.canonical_form.vertex_count = 0;
        result.canonical_form.edges.assign(edges.size(), {});
        result.vertex_mapping.canonical_edge_to_original.resize(edges.size());
        for (size_t i = 0; i < edges.size(); ++i)
            result.vertex_mapping.canonical_edge_to_original[i] = i;
        return result;
    }

    result.canonical_form.vertex_count = static_cast<VertexId>(cf.n_verts);

    // The vertex mapping is the core's winning labelling composed with the local numbering:
    // local index -> canonical label (vertex_label) and local index -> caller's id
    // (orig_vertex).
    result.vertex_mapping.canonical_to_original.resize(cf.n_verts);
    if (want_inverse_maps) result.vertex_mapping.original_to_canonical.reserve(cf.n_verts);
    for (uint32_t vi = 0; vi < cf.n_verts; ++vi) {
        const VertexId orig_v = static_cast<VertexId>(cf.orig_vertex[vi]);
        const VertexId canonical_v = static_cast<VertexId>(cf.vertex_label[vi]);
        if (want_inverse_maps) result.vertex_mapping.original_to_canonical[orig_v] = canonical_v;
        result.vertex_mapping.canonical_to_original[canonical_v] = orig_v;
    }

    // The form is flat [arity, v0, v1, ...] runs already in canonical edge order, and rank[e]
    // is the position edge e takes in that order -- so the two are inverse permutations of each
    // other and the edge mapping reads off both directions without a second sort.
    result.canonical_form.edges.reserve(edges.size());
    for (size_t w = 0; w < cf.form.size();) {
        const uint32_t arity = cf.form[w++];
        result.canonical_form.edges.emplace_back(cf.form.begin() + w,
                                                 cf.form.begin() + w + arity);
        w += arity;
    }
    result.vertex_mapping.canonical_edge_to_original.resize(edges.size());
    if (want_inverse_maps) result.vertex_mapping.original_edge_to_canonical.reserve(edges.size());
    for (size_t ei = 0; ei < edges.size(); ++ei) {
        if (want_inverse_maps) result.vertex_mapping.original_edge_to_canonical[ei] = rank[ei];
        result.vertex_mapping.canonical_edge_to_original[rank[ei]] = ei;
    }
    return result;
}

bool IRCanonicalizer::are_isomorphic(
    const std::vector<std::vector<VertexId>>& edges1,
    const std::vector<std::vector<VertexId>>& edges2) const {
    if (edges1.size() != edges2.size()) return false;
    if (edges1.empty()) return true;

    auto r1 = canonicalize_edges(edges1);
    auto r2 = canonicalize_edges(edges2);
    return r1.canonical_form == r2.canonical_form;
}


uint64_t IRCanonicalizer::compute_canonical_hash(
    const SVec<SVec<VertexId>>& edges) const {
    // ZERO, not EMPTY_STATE_CANONICAL_HASH. This entry point's empty-set convention is 0, and
    // it is load-bearing: the value is a dedup key, so changing it moves which states merge
    // and therefore the event counts. An earlier rewrite of this file returned
    // EMPTY_STATE_CANONICAL_HASH here and GoldenMatrix reported an event count that depended
    // on the worker count.
    if (edges.empty()) return 0;

    // The shared core decides the hash. ir_core_equivalence_probe compares this against it
    // over 4063 corpus states, on the hash, the canonical form, and the per-edge rank, class
    // and orbit arrays.
    auto scratch_mark = worker_scratch().mark();
    const uint64_t hash = ir_core_call(edges, nullptr, nullptr, nullptr);
    worker_scratch().release(scratch_mark);
    return hash;
}

// Convenience overloads (tests/tools): copy a heap edge list into the per-worker
// scratch arena, run the scratch-backed path, then reclaim the copy.
CanonicalizationResult IRCanonicalizer::canonicalize_edges(
    const std::vector<std::vector<VertexId>>& edges, bool want_inverse_maps) const {
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> s; s.reserve(edges.size());
    for (const auto& e : edges) s.emplace_back(e.begin(), e.end());
    auto result = canonicalize_edges(s, want_inverse_maps);
    worker_scratch().release(mk);
    return result;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_map(
    const SVec<SVec<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_class) const {
    out_edge_class.assign(edges.size(), 0u);
    if (edges.empty()) return 0;

    // The core numbers the content classes: edges whose canonicalized tuples are equal share
    // a class, numbered by the canonical order of those tuples. The core computes the class on
    // its way to the orbit, so asking for both costs nothing over asking for either -- but the
    // orbit buffer must be supplied, since the class is emitted from the same pass.
    auto scratch_mark = worker_scratch().mark();
    std::vector<uint32_t> orbit(edges.size(), 0u);
    const uint64_t hash = ir_core_call(edges, nullptr, orbit.data(), out_edge_class.data());
    worker_scratch().release(scratch_mark);
    return hash;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_rank(
    const SVec<SVec<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_rank) const {
    out_edge_rank.assign(edges.size(), 0u);
    if (edges.empty()) return 0;

    // The core assigns the rank: the edge's position when edges are ordered by (canonical
    // content, ORIGINAL INDEX). The index tie-break is what makes this a RANK -- a distinct
    // value per edge -- rather than a content class, and it is what keeps duplicate-content
    // edges apart. Positional event identity is defined this way precisely so it does NOT
    // quotient state automorphisms: two symmetric edge-role assignments keep distinct ranks
    // and stay distinct events.
    auto scratch_mark = worker_scratch().mark();
    const uint64_t hash = ir_core_call(edges, out_edge_rank.data(), nullptr, nullptr);
    worker_scratch().release(scratch_mark);
    return hash;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_orbits(
    const SVec<SVec<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_orbit,
    std::vector<uint32_t>* out_edge_class) const {
    out_edge_orbit.assign(edges.size(), 0u);
    if (out_edge_class) out_edge_class->assign(edges.size(), 0u);
    if (edges.empty()) return 0;

    // Orbits come from the core, fused over the automorphism generators its search finds.
    // The generator budget ESCALATES here (ir_core_call raises it while IR_NEED_GENERATORS is
    // returned): orbits are fused over the generators found, so a short table fuses less and
    // yields orbits that are too FINE -- a wrong identity, not a slow run -- and this is the
    // identification the quotient reconstruction keys instances by.
    auto scratch_mark = worker_scratch().mark();
    std::vector<uint32_t> klass_local;
    uint32_t* klass = out_edge_class ? out_edge_class->data()
                                     : (klass_local.assign(edges.size(), 0u), klass_local.data());
    const uint64_t hash = ir_core_call(edges, nullptr, out_edge_orbit.data(), klass);
    worker_scratch().release(scratch_mark);
    return hash;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_orbits(
    const std::vector<std::vector<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_orbit,
    std::vector<uint32_t>* out_edge_class) const {
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> s; s.reserve(edges.size());
    for (const auto& e : edges) s.emplace_back(e.begin(), e.end());
    auto h = compute_canonical_hash_with_edge_orbits(s, out_edge_orbit, out_edge_class);
    worker_scratch().release(mk);
    return h;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_map(
    const std::vector<std::vector<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_class) const {
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> s; s.reserve(edges.size());
    for (const auto& e : edges) s.emplace_back(e.begin(), e.end());
    auto h = compute_canonical_hash_with_edge_map(s, out_edge_class);
    worker_scratch().release(mk);
    return h;
}

uint64_t IRCanonicalizer::compute_canonical_hash(
    const std::vector<std::vector<VertexId>>& edges) const {
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> s; s.reserve(edges.size());
    for (const auto& e : edges) s.emplace_back(e.begin(), e.end());
    auto h = compute_canonical_hash(s);
    worker_scratch().release(mk);
    return h;
}


// =============================================================================
// canonical_types.hpp
// =============================================================================
//
// The canonical-form value types are the OUTPUT of this file's canonicalizer and are read
// nowhere else but hypergraph.hpp, so their bodies live with the code that produces them.

bool CanonicalForm::operator==(const CanonicalForm& other) const {
    return vertex_count == other.vertex_count && edges == other.edges;
}

bool CanonicalForm::operator!=(const CanonicalForm& other) const {
    return !(*this == other);
}

VertexId VertexMapping::map_vertex(VertexId original) const {
    auto it = original_to_canonical.find(original);
    return (it != original_to_canonical.end()) ? it->second : INVALID_VERTEX;
}

VertexId VertexMapping::get_original(VertexId canonical) const {
    return (canonical < canonical_to_original.size()) ?
           canonical_to_original[canonical] : INVALID_VERTEX;
}

std::size_t VertexMapping::map_edge(std::size_t original_idx) const {
    auto it = original_edge_to_canonical.find(original_idx);
    return (it != original_edge_to_canonical.end()) ? it->second : static_cast<std::size_t>(-1);
}

bool CanonicalizationResult::are_isomorphic(const CanonicalizationResult& a,
                                            const CanonicalizationResult& b) {
    return a.canonical_form == b.canonical_form;
}

}  // namespace engine
}  // namespace HG_NAMESPACE
