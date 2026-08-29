#pragma once
// Candidate edges of a state, by ancestry.
//
// Every edge of a state was produced by exactly one event on the state's chain of parent
// events, or belongs to the root state at the end of that chain, whose edges no event
// produced. The edges of a state that contain a vertex are therefore found by walking the
// chain: each event's produced edges are its delta, tested for the vertex, and the root's edges
// are indexed by vertex once when the root is created. An edge the walk yields is in the state
// unless an event further down the chain consumed it, which the state's edge set settles.
//
// A walk costs the chain's length times the few edges each event produced, and nearly every
// edge it tests is in the state. Nothing is maintained per edge on the write side.
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
        StateId s = state;
        for (;;) {
            const State& st = hg->get_state(s);
            if (st.parent_event == INVALID_ID) {
                const RootVertexEntry* b = st.root_index;
                const RootVertexEntry* e = b + st.root_index_size;
                const RootVertexEntry* it = std::lower_bound(
                    b, e, v, [](const RootVertexEntry& a, VertexId x) { return a.vertex < x; });
                for (; it != e && it->vertex == v; ++it)
                    if (edges->contains(it->edge)) visit(it->edge, hg->get_edge(it->edge));
                return;
            }
            const Event& ev = hg->get_event(st.parent_event);
            for (uint8_t i = 0; i < ev.num_produced; ++i) {
                const EdgeId eid = ev.produced_edges[i];
                const Edge& edge = hg->get_edge(eid);
                bool has = false;
                for (uint8_t k = 0; k < edge.arity; ++k)
                    if (edge.vertices[k] == v) { has = true; break; }
                if (has && edges->contains(eid)) visit(eid, edge);
            }
            s = ev.input_state;
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

    // Edges of the state whose signature is compatible with `pattern_sig`, each once. The
    // chain's deltas are tested edge by edge; the root's edges are walked from its edge set.
    template <typename SignatureAccessor, typename F>
    void for_each_edge_compatible(const EdgeSignature& pattern_sig,
                                  const SignatureAccessor& get_signature, F&& visit) const {
        StateId s = state;
        for (;;) {
            const State& st = hg->get_state(s);
            if (st.parent_event == INVALID_ID) {
                st.edges.for_each([&](EdgeId eid) {
                    if (edges->contains(eid) && signature_compatible(get_signature(eid), pattern_sig))
                        visit(eid);
                });
                return;
            }
            const Event& ev = hg->get_event(st.parent_event);
            for (uint8_t i = 0; i < ev.num_produced; ++i) {
                const EdgeId eid = ev.produced_edges[i];
                if (edges->contains(eid) && signature_compatible(get_signature(eid), pattern_sig))
                    visit(eid);
            }
            s = ev.input_state;
        }
    }
};

}  // namespace engine
}  // namespace HG_NAMESPACE
