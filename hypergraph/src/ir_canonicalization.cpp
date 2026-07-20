#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/portable_intrinsics.hpp"
#include <functional>

#include <algorithm>
#include <cassert>
#include <map>
#include <set>

// Profiling counters compile in only under IR_CANON_PROFILE (see ir_canonicalization.hpp).
#ifdef IR_CANON_PROFILE
#define IR_PROF(stmt) stmt
#else
#define IR_PROF(stmt)
#endif

namespace hypergraph {

// =============================================================================
// IRPartition
// =============================================================================

bool IRPartition::is_discrete() const {
    for (const auto& cell : cells) {
        if (cell.size() > 1) return false;
    }
    return true;
}

size_t IRPartition::first_non_singleton() const {
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].size() > 1) return i;
    }
    return cells.size();
}

// =============================================================================
// Adjacency Construction
// =============================================================================

IRCanonicalizer::HypergraphAdj IRCanonicalizer::build_adjacency(
    const SVec<SVec<VertexId>>& edges) const {
    HypergraphAdj adj;
    adj.edges = &edges;

    // Sorted-unique vertex set: sort + unique on a vector gives the same ascending
    // order (which fixes the vertex-index assignment below) without a set's tree.
    SVec<VertexId> verts;
    for (const auto& edge : edges) {
        for (VertexId v : edge) verts.push_back(v);
    }
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());

    adj.num_vertices = static_cast<uint32_t>(verts.size());
    adj.idx_to_orig.reserve(verts.size());
    uint32_t idx = 0;
    for (VertexId v : verts) {
        adj.orig_to_idx[v] = idx;
        adj.idx_to_orig.push_back(v);
        ++idx;
    }

    adj.vertex_edges.resize(adj.num_vertices);
    adj.edges_idx.resize(edges.size());
    for (uint32_t ei = 0; ei < edges.size(); ++ei) {
        uint8_t arity = static_cast<uint8_t>(edges[ei].size());
        adj.edges_idx[ei].reserve(arity);
        for (uint8_t pos = 0; pos < arity; ++pos) {
            uint32_t vi = adj.orig_to_idx[edges[ei][pos]];
            adj.vertex_edges[vi].push_back({ei, pos, arity});
            adj.edges_idx[ei].push_back(vi);
        }
    }

    return adj;
}

// =============================================================================
// Initial Partition
// =============================================================================

IRPartition IRCanonicalizer::initial_partition(const HypergraphAdj& adj) const {
    IRPartition pi;
    pi.vertex_to_cell.resize(adj.num_vertices);

    // Initial refinement: group by degree signature
    // (sorted list of (arity, position) pairs for all occurrences).
    using DegreeSig = SVec<std::pair<uint8_t, uint8_t>>;
    const uint32_t n = adj.num_vertices;
    SVec<DegreeSig> sigs;
    sigs.reserve(n);
    SVec<uint32_t> order;
    order.reserve(n);
    for (uint32_t vi = 0; vi < n; ++vi) {
        DegreeSig sig;
        sig.reserve(adj.vertex_edges[vi].size());
        for (const auto& occ : adj.vertex_edges[vi]) {
            sig.push_back({occ.arity, occ.position});
        }
        std::sort(sig.begin(), sig.end());
        sigs.push_back(std::move(sig));
        order.push_back(vi);
    }

    // Order vertices by (signature, vertex): the signature is a structural,
    // label-independent key, and the vertex tie-break reproduces the ascending
    // insertion order an ordered map would have kept within each group -- so the
    // cell partition is bit-identical to the map version, without the per-insert
    // red-black tree.
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (sigs[a] != sigs[b]) return sigs[a] < sigs[b];
        return a < b;
    });
    for (uint32_t i = 0; i < order.size();) {
        uint32_t ci = static_cast<uint32_t>(pi.cells.size());
        SVec<uint32_t> verts;
        uint32_t j = i;
        while (j < order.size() && sigs[order[j]] == sigs[order[i]]) {
            pi.vertex_to_cell[order[j]] = ci;
            verts.push_back(order[j]);
            ++j;
        }
        pi.cells.push_back(std::move(verts));
        i = j;
    }

    return pi;
}

// =============================================================================
// Partition Refinement
// =============================================================================

bool IRCanonicalizer::refine(const HypergraphAdj& adj, IRPartition& pi) const {
    bool changed = false;
    const uint32_t n = adj.num_vertices;
    const size_t num_edges = adj.edges_idx.size();

    // Position of each vertex within its current cell, for O(1) swap-removal.
    SVec<uint32_t> pos_in_cell(n);
    for (uint32_t ci = 0; ci < pi.cells.size(); ++ci)
        for (uint32_t j = 0; j < pi.cells[ci].size(); ++j)
            pos_in_cell[pi.cells[ci][j]] = j;

    // Splitter worklist (cell indices), processed lowest-index first. Index order
    // is structurally determined, so refinement stays equivariant under vertex
    // relabeling. Refining a cell C by a splitter S only inspects vertices that
    // share an edge with S, so a split costs O(boundary), not O(|C|); keeping the
    // largest piece at C's index (below) bounds total work to ~O(E log n).
    // Cell indices number at most n, and the worklist is always popped
    // lowest-index-first, so a bit-set scanned with count-trailing-zeros gives the
    // same structurally-determined order as an ordered set, with O(1) insert and no
    // per-splitter tree-node allocation.
    SVec<uint64_t> worklist((n + 63) / 64 + 1, 0);
    auto wl_insert = [&](uint32_t ci) { worklist[ci >> 6] |= (uint64_t(1) << (ci & 63)); };
    auto wl_pop_min = [&]() -> uint32_t {
        for (uint32_t w = 0; w < worklist.size(); ++w) {
            if (worklist[w]) {
                uint32_t b = static_cast<uint32_t>(hgcommon::ctz64(worklist[w]));
                worklist[w] &= worklist[w] - 1;   // clear lowest set bit
                return w * 64u + b;
            }
        }
        return UINT32_MAX;
    };
    for (uint32_t ci = 0; ci < pi.cells.size(); ++ci) wl_insert(ci);

    // Reusable scratch (no per-splitter allocation).
    SVec<uint32_t> inc_edges;
    SVec<uint32_t> edge_epoch(num_edges, 0);
    uint32_t epoch = 0;
    SVec<uint32_t> touched;               // vertices sharing an S-incident edge
    SVec<uint8_t> on_touched(n, 0);
    SVec<SVec<uint64_t>> vsig(n);   // per-vertex signature w.r.t. S (cleared after use)

    uint32_t S;
    while ((S = wl_pop_min()) != UINT32_MAX) {
        IR_PROF(++last_stats_.refine_pops;)

        // Edges incident to S, deduplicated via an epoch stamp.
        ++epoch;
        inc_edges.clear();
        for (uint32_t s : pi.cells[S])
            for (const auto& occ : adj.vertex_edges[s])
                if (edge_epoch[occ.edge_idx] != epoch) {
                    edge_epoch[occ.edge_idx] = epoch;
                    inc_edges.push_back(occ.edge_idx);
                }
        if (inc_edges.empty()) continue;

        // Signature of each touched vertex w.r.t. S: one uint64 per incident edge,
        // bits 56-63 = arity, 48-55 = the vertex's position, 0-47 = bitmask of the
        // positions occupied by S-vertices. Exact for arity <= 48.
        touched.clear();
        for (uint32_t e_idx : inc_edges) {
            const auto& edge = adj.edges_idx[e_idx];
            uint32_t arity = static_cast<uint32_t>(edge.size());
            uint64_t spos = 0;
            for (uint32_t p = 0; p < arity && p < 48; ++p)
                if (pi.vertex_to_cell[edge[p]] == S) spos |= (uint64_t(1) << p);
            for (uint32_t pu = 0; pu < arity; ++pu) {
                uint32_t u = edge[pu];
                uint64_t key = (uint64_t(arity & 0xFF) << 56) | (uint64_t(pu & 0xFF) << 48) | spos;
                if (!on_touched[u]) { on_touched[u] = 1; touched.push_back(u); }
                vsig[u].push_back(key);
            }
        }
        for (uint32_t u : touched) std::sort(vsig[u].begin(), vsig[u].end());

        // Order touched vertices by (cell, signature); both keys are structural.
        std::sort(touched.begin(), touched.end(), [&](uint32_t a, uint32_t b) {
            uint32_t ca = pi.vertex_to_cell[a], cb = pi.vertex_to_cell[b];
            if (ca != cb) return ca < cb;
            return vsig[a] < vsig[b];
        });

        // Each cell's touched vertices form a contiguous run; split that cell.
        size_t i = 0;
        while (i < touched.size()) {
            uint32_t C = pi.vertex_to_cell[touched[i]];
            size_t j = i;
            while (j < touched.size() && pi.vertex_to_cell[touched[j]] == C) ++j;

            size_t adjacent = j - i;
            size_t leftover = pi.cells[C].size() - adjacent;
            size_t num_groups = 0;
            for (size_t k = i; k < j;) {
                size_t m = k;
                while (m < j && vsig[touched[m]] == vsig[touched[k]]) ++m;
                ++num_groups;
                k = m;
            }

            if (num_groups + (leftover > 0 ? 1 : 0) > 1) {
                changed = true;
                // Remove the adjacent vertices from C (O(1) each); C keeps leftover.
                for (size_t k = i; k < j; ++k) {
                    uint32_t u = touched[k];
                    uint32_t p = pos_in_cell[u];
                    uint32_t last = pi.cells[C].back();
                    pi.cells[C][p] = last;
                    pos_in_cell[last] = p;
                    pi.cells[C].pop_back();
                }
                // Adjacent groups, in signature order.
                SVec<SVec<uint32_t>> grp;
                for (size_t k = i; k < j;) {
                    size_t m = k;
                    SVec<uint32_t> g;
                    while (m < j && vsig[touched[m]] == vsig[touched[k]]) { g.push_back(touched[m]); ++m; }
                    grp.push_back(std::move(g));
                    k = m;
                }
                // Keep the largest piece at index C; the rest get new indices and are
                // queued. The kept piece retains C's id, so vertices referencing it
                // are unaffected and need no re-refinement.
                size_t kept_sz = pi.cells[C].size();  // leftover (0 if none)
                int kept = -1;
                for (size_t g = 0; g < grp.size(); ++g)
                    if (grp[g].size() > kept_sz) { kept_sz = grp[g].size(); kept = static_cast<int>(g); }

                SVec<SVec<uint32_t>> to_queue;
                if (kept >= 0) {
                    if (!pi.cells[C].empty()) to_queue.push_back(std::move(pi.cells[C]));
                    pi.cells[C] = std::move(grp[kept]);
                    for (uint32_t t = 0; t < pi.cells[C].size(); ++t) {
                        pos_in_cell[pi.cells[C][t]] = t;
                        pi.vertex_to_cell[pi.cells[C][t]] = C;
                    }
                    for (size_t g = 0; g < grp.size(); ++g)
                        if (static_cast<int>(g) != kept) to_queue.push_back(std::move(grp[g]));
                } else {
                    for (size_t g = 0; g < grp.size(); ++g)
                        to_queue.push_back(std::move(grp[g]));
                }
                for (auto& verts : to_queue) {
                    uint32_t new_ci = static_cast<uint32_t>(pi.cells.size());
                    for (uint32_t t = 0; t < verts.size(); ++t) {
                        pos_in_cell[verts[t]] = t;
                        pi.vertex_to_cell[verts[t]] = new_ci;
                    }
                    pi.cells.push_back(std::move(verts));
                    wl_insert(new_ci);
                }
            }

            for (size_t k = i; k < j; ++k) {
                vsig[touched[k]].clear();
                on_touched[touched[k]] = 0;
            }
            i = j;
        }
    }

    return changed;
}

// =============================================================================
// Individualize
// =============================================================================

IRPartition IRCanonicalizer::individualize(
    const IRPartition& pi, size_t cell_idx, uint32_t v) const {
    IRPartition result;
    result.vertex_to_cell.resize(pi.vertex_to_cell.size());
    result.cells.reserve(pi.cells.size() + 1);

    for (size_t i = 0; i < pi.cells.size(); ++i) {
        if (i == cell_idx) {
            result.cells.push_back({v});
            result.vertex_to_cell[v] = static_cast<uint32_t>(result.cells.size() - 1);

            SVec<uint32_t> rest;
            for (uint32_t u : pi.cells[i]) {
                if (u != v) rest.push_back(u);
            }
            if (!rest.empty()) {
                uint32_t ci_rest = static_cast<uint32_t>(result.cells.size());
                for (uint32_t u : rest) {
                    result.vertex_to_cell[u] = ci_rest;
                }
                result.cells.push_back(std::move(rest));
            }
        } else {
            uint32_t ci = static_cast<uint32_t>(result.cells.size());
            for (uint32_t u : pi.cells[i]) {
                result.vertex_to_cell[u] = ci;
            }
            result.cells.push_back(pi.cells[i]);
        }
    }

    return result;
}

// =============================================================================
// Extract Labeling
// =============================================================================

SVec<uint32_t> IRCanonicalizer::extract_labeling(const IRPartition& pi) const {
    SVec<uint32_t> labeling(pi.vertex_to_cell.size());
    uint32_t label = 0;
    for (const auto& cell : pi.cells) {
        assert(cell.size() == 1);
        labeling[cell[0]] = label++;
    }
    return labeling;
}

// =============================================================================
// Apply Labeling (produce sorted canonical edge list)
// =============================================================================

SVec<SVec<VertexId>> IRCanonicalizer::apply_labeling(
    const SVec<SVec<VertexId>>& edges,
    const HypergraphAdj& adj,
    const SVec<uint32_t>& labeling) const {
    SVec<SVec<VertexId>> result;
    result.reserve(edges.size());

    // adj.edges_idx[ei] holds the vertex indices of edges[ei] already resolved
    // through orig_to_idx (build_adjacency), so relabeling is a direct index into
    // the labeling with no per-vertex hash lookup. This runs at every discrete leaf
    // of the search, so on symmetric states the saving compounds.
    for (uint32_t ei = 0; ei < edges.size(); ++ei) {
        const auto& vis = adj.edges_idx[ei];
        SVec<VertexId> mapped;
        mapped.reserve(vis.size());
        for (uint32_t vi : vis) {
            mapped.push_back(static_cast<VertexId>(labeling[vi]));
        }
        result.push_back(std::move(mapped));
    }
    std::sort(result.begin(), result.end());
    return result;
}

// =============================================================================
// Build Result
// =============================================================================

CanonicalizationResult IRCanonicalizer::build_result(
    const SVec<SVec<VertexId>>& edges,
    const HypergraphAdj& adj,
    const SVec<uint32_t>& labeling) const {
    CanonicalizationResult result;
    result.canonical_form.vertex_count = static_cast<VertexId>(adj.num_vertices);

    // Build vertex mapping
    result.vertex_mapping.canonical_to_original.resize(adj.num_vertices);
    for (uint32_t vi = 0; vi < adj.num_vertices; ++vi) {
        VertexId orig_v = adj.idx_to_orig[vi];
        VertexId canonical_v = static_cast<VertexId>(labeling[vi]);
        result.vertex_mapping.original_to_canonical[orig_v] = canonical_v;
        result.vertex_mapping.canonical_to_original[canonical_v] = orig_v;
    }

    // Transient sort scratch: the collection of mapped edges draws from the per-worker
    // scratch arena (reclaimed by the caller's mark/release). Each MappedEdge::mapped is
    // heap because it is moved into the returned heap CanonicalForm::edges below and
    // becomes the result's edge storage, so it must outlive the scratch reclaim.
    struct MappedEdge {
        std::vector<VertexId> mapped;
        size_t orig_idx;
    };
    SVec<MappedEdge> mapped;
    mapped.reserve(edges.size());
    for (size_t ei = 0; ei < edges.size(); ++ei) {
        MappedEdge me;
        me.orig_idx = ei;
        me.mapped.reserve(edges[ei].size());
        for (VertexId v : edges[ei]) {
            uint32_t vi = adj.orig_to_idx.at(v);
            me.mapped.push_back(static_cast<VertexId>(labeling[vi]));
        }
        mapped.push_back(std::move(me));
    }
    std::sort(mapped.begin(), mapped.end(),
              [](const MappedEdge& a, const MappedEdge& b) { return a.mapped < b.mapped; });

    result.canonical_form.edges.reserve(edges.size());
    result.vertex_mapping.canonical_edge_to_original.resize(edges.size());
    for (size_t ci = 0; ci < mapped.size(); ++ci) {
        result.vertex_mapping.original_edge_to_canonical[mapped[ci].orig_idx] = ci;
        result.vertex_mapping.canonical_edge_to_original[ci] = mapped[ci].orig_idx;
        result.canonical_form.edges.push_back(std::move(mapped[ci].mapped));
    }

    return result;
}

// =============================================================================
// Main Entry Points
// =============================================================================

bool IRCanonicalizer::find_canonical_labeling(
    const SVec<SVec<VertexId>>& edges,
    HypergraphAdj& adj,
    SVec<uint32_t>& labeling,
    SVec<SVec<uint32_t>>* out_generators) const {
    IR_PROF(last_stats_ = Stats{};)
    // All IR scratch (adjacency, partitions, search state) draws from the per-worker
    // scratch arena; the caller's mark/release reclaims it in bulk, so canonicalization
    // hits no heap allocator on its hot path. adj and labeling live on that scratch, so
    // the caller must hold the mark until it has consumed them.
    adj = build_adjacency(edges);
    IR_PROF(last_stats_.num_vertices = adj.num_vertices;)
    IRPartition pi = initial_partition(adj);
    refine(adj, pi);
    IR_PROF(last_stats_.discrete_after_initial_refine = pi.is_discrete();)

    if (pi.is_discrete()) {
        labeling = extract_labeling(pi);
        return true;
    }

    // Backtracking search for the lexicographically smallest canonical form,
    // with nauty-style automorphism orbit pruning. Leaves whose canonical form
    // equals the first leaf's yield an automorphism (sigma = this^-1 . first);
    // at each node, target-cell vertices lying in an already-explored orbit
    // (under the automorphisms that fix the current individualization path) are
    // skipped. This collapses the search on high-automorphism states (cycles,
    // regular/symmetric graphs) from O(cell) branches to O(orbits) without
    // changing the resulting canonical form.
    struct SearchState {
        const IRCanonicalizer* self;
        const HypergraphAdj* adj;
        const SVec<SVec<VertexId>>* edges;
        uint32_t n = 0;

        SVec<uint32_t> best_labeling;
        SVec<SVec<VertexId>> best_canonical;
        bool has_best = false;

        SVec<uint32_t> first_labeling;
        SVec<SVec<VertexId>> first_canonical;
        bool has_first = false;

        SVec<SVec<uint32_t>> generators;  // automorphisms over vertex indices
        SVec<uint32_t> path;              // individualized vertices on this branch
        SVec<uint32_t> uf;               // union-find scratch for orbits

        uint32_t find(uint32_t x) {
            while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
            return x;
        }

        // Mark every target-cell vertex in v's orbit (under the generators that
        // fix the current path) as covered, so automorphic siblings are skipped.
        void cover_orbit(const SVec<uint32_t>& cell, uint32_t v,
                         SVec<char>& covered) {
            for (uint32_t i = 0; i < n; ++i) uf[i] = i;
            for (const auto& g : generators) {
                bool fixes_path = true;
                for (uint32_t p : path) if (g[p] != p) { fixes_path = false; break; }
                if (!fixes_path) continue;
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t a = find(i), b = find(g[i]);
                    if (a != b) uf[a] = b;
                }
            }
            uint32_t rv = find(v);
            for (uint32_t w : cell) if (find(w) == rv) covered[w] = 1;
        }

        // Derive the automorphism relating this same-canonical leaf to the first.
        void record_automorphism(const SVec<uint32_t>& labeling) {
            SVec<uint32_t> this_inv(n);
            for (uint32_t vi = 0; vi < n; ++vi) this_inv[labeling[vi]] = vi;
            SVec<uint32_t> sigma(n);
            bool identity = true;
            for (uint32_t u = 0; u < n; ++u) {
                sigma[u] = this_inv[first_labeling[u]];
                if (sigma[u] != u) identity = false;
            }
            if (!identity) generators.push_back(std::move(sigma));
        }

        void search(IRPartition pi) {
            IR_PROF(++self->last_stats_.search_nodes;)
            self->refine(*adj, pi);

            if (pi.is_discrete()) {
                auto labeling = self->extract_labeling(pi);
                auto canonical = self->apply_labeling(*edges, *adj, labeling);

                if (!has_best || canonical < best_canonical) {
                    best_labeling = labeling;
                    best_canonical = canonical;
                    has_best = true;
                }
                if (!has_first) {
                    first_labeling = std::move(labeling);
                    first_canonical = std::move(canonical);
                    has_first = true;
                } else if (canonical == first_canonical) {
                    record_automorphism(labeling);
                }
                return;
            }

            size_t target = pi.first_non_singleton();
            if (target >= pi.cells.size()) return;

            SVec<uint32_t> cell = pi.cells[target];
            std::sort(cell.begin(), cell.end());

            IR_PROF(++self->last_stats_.individualizations;)
            SVec<char> covered(n, 0);
            for (uint32_t v : cell) {
                if (covered[v]) continue;
                path.push_back(v);
                search(self->individualize(pi, target, v));
                path.pop_back();
                cover_orbit(cell, v, covered);
            }
        }
    };

    SearchState state;
    state.self = this;
    state.adj = &adj;
    state.edges = &edges;
    state.n = adj.num_vertices;
    state.uf.resize(adj.num_vertices);
    state.search(pi);

    if (!state.has_best) return false;
    labeling = std::move(state.best_labeling);
    if (out_generators) *out_generators = std::move(state.generators);
    return true;
}

CanonicalizationResult IRCanonicalizer::canonicalize_edges(
    const SVec<SVec<VertexId>>& edges) const {
    if (edges.empty()) {
        CanonicalizationResult result;
        result.canonical_form.vertex_count = 0;
        return result;
    }

    // Canonicalization scratch draws from the per-worker scratch arena; one
    // mark/release reclaims it in bulk. The returned result is heap (caller owns).
    auto scratch_mark = worker_scratch().mark();
    HypergraphAdj adj;
    SVec<uint32_t> labeling;
    if (!find_canonical_labeling(edges, adj, labeling)) {
        worker_scratch().release(scratch_mark);
        CanonicalizationResult result;
        result.canonical_form.vertex_count = 0;
        return result;
    }

    auto result = build_result(edges, adj, labeling);
    worker_scratch().release(scratch_mark);
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
    if (edges.empty()) return 0;

    // Hash-only path: the canonical labeling fully determines the hash, so hash the
    // canonical edge ordering directly and skip build_result -- the CanonicalizationResult
    // it would materialize (per-edge maps and vectors) is never read here. This is the
    // per-state hot path; the mark/release reclaims all scratch in bulk.
    auto scratch_mark = worker_scratch().mark();
    HypergraphAdj adj;
    SVec<uint32_t> labeling;
    bool ok = find_canonical_labeling(edges, adj, labeling);

    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;

    hash ^= static_cast<uint64_t>(ok ? adj.num_vertices : 0u);
    hash *= prime;

    if (ok) {
        SVec<SVec<VertexId>> canonical = apply_labeling(edges, adj, labeling);
        for (const auto& edge : canonical) {
            for (auto vertex : edge) {
                hash ^= static_cast<uint64_t>(vertex);
                hash *= prime;
            }
            hash ^= 0xDEADBEEF;
            hash *= prime;
        }
    }

    worker_scratch().release(scratch_mark);
    return hash;
}

// Convenience overloads (tests/tools): copy a heap edge list into the per-worker
// scratch arena, run the scratch-backed path, then reclaim the copy.
CanonicalizationResult IRCanonicalizer::canonicalize_edges(
    const std::vector<std::vector<VertexId>>& edges) const {
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> s; s.reserve(edges.size());
    for (const auto& e : edges) s.emplace_back(e.begin(), e.end());
    auto result = canonicalize_edges(s);
    worker_scratch().release(mk);
    return result;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_map(
    const SVec<SVec<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_class) const {
    out_edge_class.assign(edges.size(), 0u);
    if (edges.empty()) return 0;

    auto scratch_mark = worker_scratch().mark();
    HypergraphAdj adj;
    SVec<uint32_t> labeling;
    bool ok = find_canonical_labeling(edges, adj, labeling);

    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;

    hash ^= static_cast<uint64_t>(ok ? adj.num_vertices : 0u);
    hash *= prime;

    if (ok) {
        // Relabel each edge, then order by canonical content. The hash consumes the
        // same sequence compute_canonical_hash does, so both agree bit for bit.
        struct MappedEdge {
            std::vector<VertexId> mapped;
            size_t orig_idx;
        };
        std::vector<MappedEdge> mapped;
        mapped.reserve(edges.size());
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            MappedEdge me;
            me.orig_idx = ei;
            me.mapped.reserve(edges[ei].size());
            for (VertexId v : edges[ei]) {
                uint32_t vi = adj.orig_to_idx.at(v);
                me.mapped.push_back(static_cast<VertexId>(labeling[vi]));
            }
            mapped.push_back(std::move(me));
        }
        std::sort(mapped.begin(), mapped.end(),
                  [](const MappedEdge& a, const MappedEdge& b) { return a.mapped < b.mapped; });

        for (const auto& me : mapped) {
            for (auto vertex : me.mapped) {
                hash ^= static_cast<uint64_t>(vertex);
                hash *= prime;
            }
            hash ^= 0xDEADBEEF;
            hash *= prime;
        }

        uint32_t cls = 0;
        for (size_t i = 0; i < mapped.size(); ++i) {
            if (i > 0 && mapped[i].mapped != mapped[i - 1].mapped) ++cls;
            out_edge_class[mapped[i].orig_idx] = cls;
        }
    }

    worker_scratch().release(scratch_mark);
    return hash;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_orbits(
    const SVec<SVec<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_orbit) const {
    out_edge_orbit.assign(edges.size(), 0u);
    if (edges.empty()) return 0;

    auto scratch_mark = worker_scratch().mark();
    HypergraphAdj adj;
    SVec<uint32_t> labeling;
    SVec<SVec<uint32_t>> generators;
    bool ok = find_canonical_labeling(edges, adj, labeling, &generators);

    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    hash ^= static_cast<uint64_t>(ok ? adj.num_vertices : 0u);
    hash *= prime;

    if (ok) {
        const uint32_t n = adj.num_vertices;

        // Canonical content of each edge, and the canonically ordered distinct contents.
        std::vector<std::vector<VertexId>> mapped(edges.size());
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            mapped[ei].reserve(edges[ei].size());
            for (VertexId v : edges[ei])
                mapped[ei].push_back(static_cast<VertexId>(labeling[adj.orig_to_idx.at(v)]));
        }
        std::vector<std::vector<VertexId>> sorted_edges = mapped;
        std::sort(sorted_edges.begin(), sorted_edges.end());
        for (const auto& e : sorted_edges) {
            for (auto vertex : e) { hash ^= static_cast<uint64_t>(vertex); hash *= prime; }
            hash ^= 0xDEADBEEF; hash *= prime;
        }

        std::vector<std::vector<VertexId>> uniq = sorted_edges;
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        auto class_of = [&](const std::vector<VertexId>& c) -> uint32_t {
            return static_cast<uint32_t>(
                std::lower_bound(uniq.begin(), uniq.end(), c) - uniq.begin());
        };

        // Union content classes that an automorphism of the canonical form identifies.
        // A generator sigma permutes vertex indices; tau = labeling . sigma . labeling^-1
        // is the corresponding automorphism of the canonical form.
        std::vector<uint32_t> uf(uniq.size());
        for (size_t i = 0; i < uf.size(); ++i) uf[i] = static_cast<uint32_t>(i);
        std::function<uint32_t(uint32_t)> find = [&](uint32_t x) {
            while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
            return x;
        };
        std::vector<uint32_t> tau(n);
        for (const auto& sigma : generators) {
            for (uint32_t u = 0; u < n; ++u) tau[labeling[u]] = labeling[sigma[u]];
            for (uint32_t c = 0; c < uniq.size(); ++c) {
                std::vector<VertexId> img;
                img.reserve(uniq[c].size());
                for (VertexId v : uniq[c]) img.push_back(static_cast<VertexId>(tau[v]));
                uint32_t d = class_of(img);
                uint32_t a = find(c), b = find(d);
                if (a != b) uf[a] = b;
            }
        }

        // Number orbits by the canonical order of their smallest content class.
        std::map<uint32_t, uint32_t> root_to_orbit;
        for (uint32_t c = 0; c < uniq.size(); ++c) root_to_orbit.emplace(find(c), 0u);
        uint32_t next = 0;
        for (auto& kv : root_to_orbit) kv.second = next++;

        for (size_t ei = 0; ei < edges.size(); ++ei)
            out_edge_orbit[ei] = root_to_orbit[find(class_of(mapped[ei]))];
    }

    worker_scratch().release(scratch_mark);
    return hash;
}

uint64_t IRCanonicalizer::compute_canonical_hash_with_edge_orbits(
    const std::vector<std::vector<VertexId>>& edges,
    std::vector<uint32_t>& out_edge_orbit) const {
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> s; s.reserve(edges.size());
    for (const auto& e : edges) s.emplace_back(e.begin(), e.end());
    auto h = compute_canonical_hash_with_edge_orbits(s, out_edge_orbit);
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

}  // namespace hypergraph
