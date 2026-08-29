#pragma once
// Candidate edges of a state, by ancestry.
//
// A state's edges are the edges its chain of parent states brought -- every edge of the root,
// the produced edges of each derived state -- less what later events consumed. Each state
// indexes its own contribution by vertex once, when it is created (State::vertex_index), so
// the edges of a state that contain a vertex are found by walking the chain and searching each
// contribution, and an edge the walk yields is in the state unless consumed further down the
// chain, which the state's edge set settles. A query costs the chain's length times one search
// of a small sorted array, nearly every edge it tests is in the state, and nothing is
// maintained per edge on the write side.
#include "hgcommon/namespace.hpp"
#include <algorithm>
#include <cstdint>
#include "types.hpp"
#include "signature.hpp"
#include "bitset.hpp"
#include "hypergraph.hpp"

namespace HG_NAMESPACE {
namespace engine {

struct AncestryCandidates {
    const Hypergraph* hg;
    StateId state;
    const SparseBitset* edges;

    // Edges of the state containing `v`, each once, with the edge record.
    template <typename F>
    void for_each_edge_at(VertexId v, F&& visit) const {
        for (StateId s = state; s != INVALID_ID;) {
            const State& st = hg->get_state(s);
            const RootVertexEntry* b = st.vertex_index;
            const RootVertexEntry* e = b + st.vertex_index_size;
            const RootVertexEntry* it = std::lower_bound(
                b, e, v, [](const RootVertexEntry& a, VertexId x) { return a.vertex < x; });
            for (; it != e && it->vertex == v; ++it)
                if (edges->contains(it->edge)) visit(it->edge, hg->get_edge(it->edge));
            s = st.parent_state;
        }
    }

    // Edges of the state containing every vertex in `vertices`, each once, with the edge
    // record. The walk is seeded from the first bound vertex; the others are checked on the
    // edge itself, O(arity) per candidate.
    template <typename F>
    void for_each_edge_containing_all(const VertexId* vertices, uint8_t count, F&& visit) const {
        if (count == 0) return;
        for_each_edge_at(vertices[0], [&](EdgeId eid, const Edge& edge) {
            for (uint8_t i = 1; i < count; ++i) {
                bool found = false;
                for (uint8_t j = 0; j < edge.arity && !found; ++j)
                    if (edge.vertices[j] == vertices[i]) found = true;
                if (!found) return;
            }
            visit(eid, edge);
        });
    }

    // Edges of the state whose signature is compatible with `pattern_sig`, each once: a
    // derived state's produced edges along the chain, and the root's edges from its edge set.
    template <typename SignatureAccessor, typename F>
    void for_each_edge_compatible(const EdgeSignature& pattern_sig,
                                  const SignatureAccessor& get_signature, F&& visit) const {
        auto offer = [&](EdgeId eid) {
            if (edges->contains(eid) && signature_compatible(get_signature(eid), pattern_sig))
                visit(eid);
        };
        for (StateId s = state; s != INVALID_ID;) {
            const State& st = hg->get_state(s);
            if (st.parent_state == INVALID_ID) {
                st.edges.for_each(offer);
                return;
            }
            for (uint32_t i = 0; i < st.num_delta_edges; ++i) offer(st.delta_edges[i]);
            s = st.parent_state;
        }
    }
};

}  // namespace engine
}  // namespace HG_NAMESPACE
