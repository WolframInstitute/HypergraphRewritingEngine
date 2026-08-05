#pragma once

#include <vector>
#include <cstdint>

#include "types.hpp"
#include "canonical_types.hpp"
#include "scratch_alloc.hpp"

namespace hypergraph {

// The host face of the McKay-style individualization-refinement canonicalizer for directed
// hypergraphs: it adapts this project's edge lists to hgcommon/ir_core.hpp, which holds the
// algorithm and is the same code the device runs. The canonical form is the lexicographically
// smallest relabelled edge list, so it matches brute force.
class IRCanonicalizer {
public:
    // Engine entry points: edges already materialized in the per-worker scratch
    // arena (no heap copy on the hot path). want_inverse_maps builds the
    // VertexMapping::original_to_canonical / original_edge_to_canonical lookups
    // (the map_vertex/map_edge direction); the shipping serialization and hash
    // paths never read them, so it defaults off and the per-canonicalization
    // hash tables are never allocated. Tools/tests that query those maps pass true.
    CanonicalizationResult canonicalize_edges(const SVec<SVec<VertexId>>& edges,
                                              bool want_inverse_maps = false) const;
    uint64_t compute_canonical_hash(const SVec<SVec<VertexId>>& edges) const;

    // Canonical hash, plus for each input edge the index of its canonical edge
    // CONTENT class: edges that canonicalize to the same vertex tuple share a class,
    // numbered by the canonical (sorted) order of those tuples. Content classes
    // rather than per-edge slots because a state is a multiset: among duplicate
    // edges the slot each one lands in depends on input order, the class does not.
    //
    // The class is invariant under vertex relabeling and edge reordering exactly
    // when the state's automorphism group is trivial. With a nontrivial Aut several
    // labelings reach the same canonical form and differ by an automorphism, which
    // permutes edges between classes, so an individual edge's class is defined only
    // up to that action. Callers that must identify edges across two labelings of
    // the same state (for example accumulating per-edge data over the parents that
    // merge into one canonical state) need edge ORBITS under Aut, not these classes.
    // The hash is fully invariant regardless.
    uint64_t compute_canonical_hash_with_edge_map(
        const SVec<SVec<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_class) const;

    // Canonical hash, plus for each input edge its canonical RANK: the edge's position when
    // the edges are ordered by (canonical content, original index). Unlike the content class
    // above, a rank is distinct for every edge -- the index tie-break separates duplicates --
    // which is what Positional event identity requires, since it must NOT quotient state
    // automorphisms. The rank is a property of the state's isomorphism class plus the input
    // edge order, so an event identified by ranks needs no representative state and does not
    // move when the state-identity mode does.
    uint64_t compute_canonical_hash_with_edge_rank(
        const SVec<SVec<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_rank) const;

    // Canonical hash, plus for each input edge the id of its canonical edge ORBIT
    // under the state's automorphism group. Unlike the content class above, the orbit
    // is invariant under vertex relabeling and edge reordering even when Aut is
    // nontrivial, because it quotients out exactly the automorphism action that
    // permutes edges between content classes. This is the identification to use when
    // accumulating per-edge data across the several labelings by which distinct
    // parents reach one canonical state. Orbits are numbered by the canonical order
    // of their smallest content class, so the numbering is itself invariant.
    // `out_edge_class`, when non-null, additionally receives each edge's canonical content
    // class -- the index of its canonicalized vertex tuple among the state's distinct
    // canonicalized tuples, in canonical order. Orbits are unions of content classes under
    // Aut, so the class is strictly finer; it is computed on the way to the orbit and costs
    // nothing extra. It is the identification the quotient's per-instance reconstruction
    // slots on (see Hypergraph::EdgeOrbitTable::slot).
    uint64_t compute_canonical_hash_with_edge_orbits(
        const SVec<SVec<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_orbit,
        std::vector<uint32_t>* out_edge_class = nullptr) const;
    uint64_t compute_canonical_hash_with_edge_orbits(
        const std::vector<std::vector<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_orbit,
        std::vector<uint32_t>* out_edge_class = nullptr) const;

    // Convenience overloads (tests/tools): adapt a heap edge list into scratch.
    CanonicalizationResult canonicalize_edges(
        const std::vector<std::vector<VertexId>>& edges,
        bool want_inverse_maps = false) const;
    uint64_t compute_canonical_hash(
        const std::vector<std::vector<VertexId>>& edges) const;
    uint64_t compute_canonical_hash_with_edge_map(
        const std::vector<std::vector<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_class) const;

    bool are_isomorphic(
        const std::vector<std::vector<VertexId>>& edges1,
        const std::vector<std::vector<VertexId>>& edges2) const;
};

}  // namespace hypergraph
