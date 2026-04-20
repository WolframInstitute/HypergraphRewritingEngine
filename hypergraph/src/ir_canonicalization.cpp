#include "hypergraph/ir_canonicalization.hpp"

#include <algorithm>
#include <cassert>
#include <map>
#include <set>

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
    const std::vector<std::vector<VertexId>>& edges) const {
    HypergraphAdj adj;
    adj.edges = &edges;

    std::set<VertexId> verts;
    for (const auto& edge : edges) {
        for (VertexId v : edge) verts.insert(v);
    }

    adj.num_vertices = static_cast<uint32_t>(verts.size());
    adj.idx_to_orig.reserve(verts.size());
    uint32_t idx = 0;
    for (VertexId v : verts) {
        adj.orig_to_idx[v] = idx;
        adj.idx_to_orig.push_back(v);
        ++idx;
    }

    adj.vertex_edges.resize(adj.num_vertices);
    for (uint32_t ei = 0; ei < edges.size(); ++ei) {
        uint8_t arity = static_cast<uint8_t>(edges[ei].size());
        for (uint8_t pos = 0; pos < arity; ++pos) {
            uint32_t vi = adj.orig_to_idx[edges[ei][pos]];
            adj.vertex_edges[vi].push_back({ei, pos, arity});
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

    // Initially all vertices in one cell (no coloring for original vertices)
    // The refinement step will separate them based on structure
    std::vector<uint32_t> all;
    all.reserve(adj.num_vertices);
    for (uint32_t i = 0; i < adj.num_vertices; ++i) {
        all.push_back(i);
    }

    // Initial refinement: group by degree signature
    // (sorted list of (arity, position) pairs for all occurrences)
    using DegreeSig = std::vector<std::pair<uint8_t, uint8_t>>;
    std::map<DegreeSig, std::vector<uint32_t>> sig_groups;

    for (uint32_t vi = 0; vi < adj.num_vertices; ++vi) {
        DegreeSig sig;
        sig.reserve(adj.vertex_edges[vi].size());
        for (const auto& occ : adj.vertex_edges[vi]) {
            sig.push_back({occ.arity, occ.position});
        }
        std::sort(sig.begin(), sig.end());
        sig_groups[sig].push_back(vi);
    }

    for (auto& [sig, verts] : sig_groups) {
        uint32_t ci = static_cast<uint32_t>(pi.cells.size());
        for (uint32_t v : verts) {
            pi.vertex_to_cell[v] = ci;
        }
        pi.cells.push_back(std::move(verts));
    }

    return pi;
}

// =============================================================================
// Partition Refinement
// =============================================================================

bool IRCanonicalizer::refine(const HypergraphAdj& adj, IRPartition& pi) const {
    bool changed = false;
    bool any_split = true;

    while (any_split) {
        any_split = false;

        for (size_t ci = 0; ci < pi.cells.size(); ++ci) {
            if (pi.cells[ci].size() <= 1) continue;

            // For each vertex, compute signature based on co-occurrence with
            // vertices in each cell across all edges.
            // Signature: for each edge containing this vertex, collect
            //   (arity, position, sorted cell indices of other vertices)
            using EdgeSig = std::vector<uint32_t>; // (arity, position, cells of co-vertices...)
            std::map<std::vector<EdgeSig>, std::vector<uint32_t>> sig_to_verts;

            for (uint32_t vi : pi.cells[ci]) {
                std::vector<EdgeSig> vertex_sig;

                for (const auto& occ : adj.vertex_edges[vi]) {
                    EdgeSig esig;
                    esig.push_back(occ.arity);
                    esig.push_back(occ.position);

                    const auto& edge = (*adj.edges)[occ.edge_idx];
                    for (uint8_t p = 0; p < edge.size(); ++p) {
                        if (p == occ.position) continue;
                        uint32_t other_vi = adj.orig_to_idx.at(edge[p]);
                        esig.push_back(pi.vertex_to_cell[other_vi]);
                    }
                    std::sort(esig.begin() + 2, esig.end());
                    vertex_sig.push_back(std::move(esig));
                }
                std::sort(vertex_sig.begin(), vertex_sig.end());
                sig_to_verts[vertex_sig].push_back(vi);
            }

            if (sig_to_verts.size() > 1) {
                any_split = true;
                changed = true;

                auto it = sig_to_verts.begin();
                pi.cells[ci] = it->second;
                for (uint32_t v : pi.cells[ci]) {
                    pi.vertex_to_cell[v] = static_cast<uint32_t>(ci);
                }
                ++it;
                for (; it != sig_to_verts.end(); ++it) {
                    uint32_t new_ci = static_cast<uint32_t>(pi.cells.size());
                    for (uint32_t v : it->second) {
                        pi.vertex_to_cell[v] = new_ci;
                    }
                    pi.cells.push_back(it->second);
                }
            }
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

            std::vector<uint32_t> rest;
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

std::vector<uint32_t> IRCanonicalizer::extract_labeling(const IRPartition& pi) const {
    std::vector<uint32_t> labeling(pi.vertex_to_cell.size());
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

std::vector<std::vector<VertexId>> IRCanonicalizer::apply_labeling(
    const std::vector<std::vector<VertexId>>& edges,
    const HypergraphAdj& adj,
    const std::vector<uint32_t>& labeling) const {
    std::vector<std::vector<VertexId>> result;
    result.reserve(edges.size());

    for (const auto& edge : edges) {
        std::vector<VertexId> mapped;
        mapped.reserve(edge.size());
        for (VertexId v : edge) {
            uint32_t vi = adj.orig_to_idx.at(v);
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
    const std::vector<std::vector<VertexId>>& edges,
    const HypergraphAdj& adj,
    const std::vector<uint32_t>& labeling) const {
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

    // Map and sort edges, tracking original indices
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

CanonicalizationResult IRCanonicalizer::canonicalize_edges(
    const std::vector<std::vector<VertexId>>& edges) const {
    if (edges.empty()) {
        CanonicalizationResult result;
        result.canonical_form.vertex_count = 0;
        return result;
    }

    HypergraphAdj adj = build_adjacency(edges);
    IRPartition pi = initial_partition(adj);
    refine(adj, pi);

    if (pi.is_discrete()) {
        auto labeling = extract_labeling(pi);
        return build_result(edges, adj, labeling);
    }

    // Backtracking search for lexicographically smallest canonical form
    struct SearchState {
        const IRCanonicalizer* self;
        const HypergraphAdj* adj;
        const std::vector<std::vector<VertexId>>* edges;
        std::vector<uint32_t> best_labeling;
        std::vector<std::vector<VertexId>> best_canonical;
        bool has_best = false;

        void search(IRPartition pi) {
            self->refine(*adj, pi);

            if (pi.is_discrete()) {
                auto labeling = self->extract_labeling(pi);
                auto canonical = self->apply_labeling(*edges, *adj, labeling);

                if (!has_best || canonical < best_canonical) {
                    best_labeling = std::move(labeling);
                    best_canonical = std::move(canonical);
                    has_best = true;
                }
                return;
            }

            size_t target = pi.first_non_singleton();
            if (target >= pi.cells.size()) return;

            std::vector<uint32_t> target_verts = pi.cells[target];
            std::sort(target_verts.begin(), target_verts.end());

            for (uint32_t v : target_verts) {
                auto child_pi = self->individualize(pi, target, v);
                search(std::move(child_pi));
            }
        }
    };

    SearchState state;
    state.self = this;
    state.adj = &adj;
    state.edges = &edges;
    state.search(pi);

    if (!state.has_best) {
        CanonicalizationResult result;
        result.canonical_form.vertex_count = 0;
        return result;
    }

    return build_result(edges, adj, state.best_labeling);
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
    const std::vector<std::vector<VertexId>>& edges) const {
    if (edges.empty()) return 0;

    auto result = canonicalize_edges(edges);

    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;

    hash ^= static_cast<uint64_t>(result.canonical_form.vertex_count);
    hash *= prime;

    for (const auto& edge : result.canonical_form.edges) {
        for (auto vertex : edge) {
            hash ^= static_cast<uint64_t>(vertex);
            hash *= prime;
        }
        hash ^= 0xDEADBEEF;
        hash *= prime;
    }

    return hash;
}

EdgeCorrespondence IRCanonicalizer::find_edge_correspondence(
    const std::vector<std::vector<VertexId>>& edges1,
    const std::vector<std::vector<VertexId>>& edges2) const {
    EdgeCorrespondence result;

    if (edges1.size() != edges2.size()) return result;

    auto r1 = canonicalize_edges(edges1);
    auto r2 = canonicalize_edges(edges2);

    if (r1.canonical_form != r2.canonical_form) return result;

    result.count = static_cast<uint32_t>(edges1.size());
    result.state1_edges = new EdgeId[result.count];
    result.state2_edges = new EdgeId[result.count];

    for (uint32_t ci = 0; ci < result.count; ++ci) {
        result.state1_edges[ci] = static_cast<EdgeId>(r1.vertex_mapping.canonical_edge_to_original[ci]);
        result.state2_edges[ci] = static_cast<EdgeId>(r2.vertex_mapping.canonical_edge_to_original[ci]);
    }

    result.valid = true;
    return result;
}

}  // namespace hypergraph
