#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>

#include "types.hpp"
#include "canonical_types.hpp"
#include "scratch_alloc.hpp"

namespace hypergraph {

// Ordered partition of vertex indices for IR refinement. All internal containers
// draw from the per-worker scratch arena (SVec); the whole canonicalization's
// scratch is reclaimed by a single mark/release in canonicalize_edges, so no
// malloc is hit on the IR hot path.
struct IRPartition {
    SVec<SVec<uint32_t>> cells;
    SVec<uint32_t> vertex_to_cell;  // vertex_index -> cell index

    bool is_discrete() const;
    size_t first_non_singleton() const;
};

// McKay-style Individualization-Refinement canonicalizer for directed hypergraphs.
// Works directly on the original vertex set using hypergraph-aware refinement.
// Produces the lexicographically smallest canonical form (matching brute-force).
class IRCanonicalizer {
public:
    // Engine entry points: edges already materialized in the per-worker scratch
    // arena (no heap copy on the hot path).
    CanonicalizationResult canonicalize_edges(const SVec<SVec<VertexId>>& edges) const;
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

    // Canonical hash, plus for each input edge the id of its canonical edge ORBIT
    // under the state's automorphism group. Unlike the content class above, the orbit
    // is invariant under vertex relabeling and edge reordering even when Aut is
    // nontrivial, because it quotients out exactly the automorphism action that
    // permutes edges between content classes. This is the identification to use when
    // accumulating per-edge data across the several labelings by which distinct
    // parents reach one canonical state. Orbits are numbered by the canonical order
    // of their smallest content class, so the numbering is itself invariant.
    uint64_t compute_canonical_hash_with_edge_orbits(
        const SVec<SVec<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_orbit) const;
    uint64_t compute_canonical_hash_with_edge_orbits(
        const std::vector<std::vector<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_orbit) const;

    // Convenience overloads (tests/tools): adapt a heap edge list into scratch.
    CanonicalizationResult canonicalize_edges(
        const std::vector<std::vector<VertexId>>& edges) const;
    uint64_t compute_canonical_hash(
        const std::vector<std::vector<VertexId>>& edges) const;
    uint64_t compute_canonical_hash_with_edge_map(
        const std::vector<std::vector<VertexId>>& edges,
        std::vector<uint32_t>& out_edge_class) const;

    bool are_isomorphic(
        const std::vector<std::vector<VertexId>>& edges1,
        const std::vector<std::vector<VertexId>>& edges2) const;

#ifdef IR_CANON_PROFILE
    // Measurement hooks (populated by the most recent canonicalize_edges call).
    // Compiled in only under IR_CANON_PROFILE; absent from release builds.
    struct Stats {
        uint32_t num_vertices = 0;
        size_t refine_pops = 0;                  // total splitter pops across all refines
        size_t search_nodes = 0;                 // individualization-refinement tree nodes visited
        size_t individualizations = 0;           // search nodes that individualized (non-leaf)
        bool discrete_after_initial_refine = false;  // true => no individualization was needed
    };
    Stats last_stats() const { return last_stats_; }
#endif

private:
#ifdef IR_CANON_PROFILE
    mutable Stats last_stats_;
#endif

    // Vertex occurrence in an edge: (edge_index, position, arity)
    struct VertexOccurrence {
        uint32_t edge_idx;
        uint8_t position;
        uint8_t arity;
    };

    // Precomputed adjacency for refinement (scratch-arena backed; lives for one
    // canonicalize_edges call). edges points at the caller's heap input.
    struct HypergraphAdj {
        uint32_t num_vertices;
        SVec<SVec<VertexOccurrence>> vertex_edges; // vertex_index -> occurrences
        const SVec<SVec<VertexId>>* edges;
        SVec<SVec<uint32_t>> edges_idx;  // edges in vertex-index form (hot path)
        SUMap<VertexId, uint32_t> orig_to_idx;
        SVec<VertexId> idx_to_orig;
    };

    HypergraphAdj build_adjacency(const SVec<SVec<VertexId>>& edges) const;
    IRPartition initial_partition(const HypergraphAdj& adj) const;
    bool refine(const HypergraphAdj& adj, IRPartition& pi) const;
    IRPartition individualize(const IRPartition& pi, size_t cell_idx, uint32_t v) const;
    SVec<uint32_t> extract_labeling(const IRPartition& pi) const;

    SVec<SVec<VertexId>> apply_labeling(
        const SVec<SVec<VertexId>>& edges,
        const HypergraphAdj& adj,
        const SVec<uint32_t>& labeling) const;

    CanonicalizationResult build_result(
        const SVec<SVec<VertexId>>& edges,
        const HypergraphAdj& adj,
        const SVec<uint32_t>& labeling) const;

    // Runs adjacency + refinement + backtracking IR search on scratch-arena edges;
    // writes the canonical labeling (vertex-index -> canonical label) and adjacency to
    // the out-params (both scratch-arena backed -- the caller holds a worker_scratch
    // mark spanning their use). Returns false only for the degenerate no-labeling case.
    // Shared by canonicalize_edges (which builds the full result) and
    // compute_canonical_hash (which hashes the ordering, skipping result materialization).
    // out_generators, when non-null, receives the automorphism generators the search
    // already discovers for orbit pruning (permutations of vertex indices). Empty when
    // refinement alone is discrete, which means the automorphism group is trivial.
    bool find_canonical_labeling(
        const SVec<SVec<VertexId>>& edges,
        HypergraphAdj& adj,
        SVec<uint32_t>& labeling,
        SVec<SVec<uint32_t>>* out_generators = nullptr) const;
};

}  // namespace hypergraph
