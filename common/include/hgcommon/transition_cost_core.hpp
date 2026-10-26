#pragma once
#include "hgcommon/namespace.hpp"
// "TransitionCost": the cost of a transition from a parent state P to a child state C, read from
// a field on the vertices of both (docs/TRANSITION_COST.md). A rewrite keeps the ids of the
// vertices it does not delete, so a vertex of both states has one id in both.
//
// The cost is the heat residual of the field: for each vertex v of both states,
//
//   r_v = f_C(v) - f_P(v) - dt * sum over the neighbours w of v in G_P of (f_P(w) - f_P(v))
//
// G_P the parent's graph under "StatisticsGraph" (state_geometry_core.hpp, sg_build), and the
// cost the sum of r_v^2 over the vertices with a finite r_v.
//
// ONE PARENT, MANY CHILDREN. A child that leaves v's field unchanged has the residual
// e_v = -dt ((A - D) f_P)(v) at v, so the cost is B - sum over T of e_v^2 + sum over T of r_v^2,
// where B is the sum of every e_v^2 and T holds the vertices whose field the transition can
// change. tc_parent_of computes the field, e_v and B once per parent; tc_child_cost computes a
// child from the hyperedges the transition consumes and produces. The degree in G, the loop
// count and the edge multiplicity of v read only the hyperedges that contain v, so T is the
// parent's vertices of the consumed and produced hyperedges, and the child's field is read from
// the child's hyperedges at T. The curvature reads a wider neighbourhood, so T is every vertex
// and the child's field is the whole child's. The sums are double-double (det_math.hpp), so the
// subtraction loses nothing to B.
//
// tc_child_cost is the one implementation: the CPU engine calls it for each costed transition
// (ParallelEvolutionEngine::transition_cost_of), and tc_transition_cost, which the Python client's
// heat_residual calls with two whole states, calls it with the hyperedges that differ.
#include "hgcommon/core.hpp"
#include "hgcommon/det_math.hpp"
#include "hgcommon/state_geometry_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace HG_NAMESPACE {
namespace common {

// The fields. TC_FIELD_NONE is no transition cost.
enum : uint32_t {
    TC_FIELD_NONE = 0,
    TC_FIELD_DEGREE = 1,             // the degree of v in G
    TC_FIELD_CURVATURE = 2,          // the Ollivier-Ricci curvature k_v; NaN with no neighbour
    TC_FIELD_LOOP_COUNT = 3,         // the hyperedges whose every vertex is v
    TC_FIELD_EDGE_MULTIPLICITY = 4,  // the hyperedges that contain v and are not loops
};

struct TransitionCostSpec {
    uint32_t field = TC_FIELD_NONE;
    uint32_t graph = SG_GRAPH_CONSECUTIVE;   // "StatisticsGraph"
    double dt = 0.0;                         // "TimeStep"
};

HG_HD inline bool tc_active(const TransitionCostSpec& s) { return s.field != TC_FIELD_NONE; }

// Whether v's field reads only the hyperedges that contain v.
HG_HD inline bool tc_field_is_local(uint32_t field) { return field != TC_FIELD_CURVATURE; }

// A state's hyperedges in compressed rows: hyperedge i is v[off[i] .. off[i+1]). An EdgeList of
// sg_build.
struct TcEdges {
    const uint32_t* off = nullptr;
    const uint32_t* v = nullptr;
    uint32_t m = 0;
    HG_HD uint32_t count() const { return m; }
    HG_HD uint32_t arity(uint32_t i) const { return off[i + 1] - off[i]; }
    HG_HD uint32_t at(uint32_t i, uint32_t k) const { return v[off[i] + k]; }
    // Hyperedges first .. m.
    HG_HD TcEdges from(uint32_t first) const { return TcEdges{off + first, v, m - first}; }
};

// The vertices of `el` in ascending id order, which is G's vertex numbering (sg_build), into
// verts[0 .. n). `verts` holds at least the number of incidences. Returns n.
template <class EdgeList>
HG_HD uint32_t tc_vertices(const EdgeList& el, uint32_t* verts) {
    size_t s = 0;
    for (uint32_t i = 0; i < el.count(); ++i)
        for (uint32_t k = 0; k < el.arity(i); ++k) verts[s++] = el.at(i, k);
    sg_sort(verts, s);
    return static_cast<uint32_t>(sg_unique(verts, s));
}

// The field `field` at each vertex of `el` except TC_FIELD_CURVATURE, into f[0 .. n), with `g`
// the state's graph and `verts` its n vertices (tc_vertices).
template <class EdgeList>
HG_HD void tc_local_field(const EdgeList& el, uint32_t field, const SgGraph& g,
                          const uint32_t* verts, uint32_t n, double* f) {
    for (uint32_t i = 0; i < n; ++i)
        f[i] = field == TC_FIELD_DEGREE ? static_cast<double>(g.degree(i)) : 0.0;
    if (field == TC_FIELD_DEGREE) return;
    for (uint32_t i = 0; i < el.count(); ++i) {
        const uint32_t k = el.arity(i);
        if (k == 0) continue;
        bool loop = true;
        for (uint32_t j = 1; j < k; ++j) loop = loop && el.at(i, j) == el.at(i, 0);
        if (field == TC_FIELD_LOOP_COUNT) {
            if (loop) f[sg_index_of(verts, n, el.at(i, 0))] += 1.0;
            continue;
        }
        if (loop) continue;
        // Each distinct vertex of the hyperedge once.
        for (uint32_t j = 0; j < k; ++j) {
            bool first = true;
            for (uint32_t l = 0; l < j; ++l) first = first && el.at(i, l) != el.at(i, j);
            if (first) f[sg_index_of(verts, n, el.at(i, j))] += 1.0;
        }
    }
}

// e_v = -dt ((A - D) f)(v) at vertex index i of G: the residual at v of a child that leaves v's
// field unchanged.
HG_HD inline double tc_unchanged_residual(const SgGraph& g, const double* f, uint32_t i,
                                          double dt) {
    double lap = 0.0;
    for (uint32_t e = g.off[i]; e < g.off[i + 1]; ++e) lap += f[g.nbr[e]] - f[i];
    return -dm_mul(dt, lap);
}

// r_v from the child's field, the parent's field and e_v.
HG_HD inline double tc_residual(double child, double parent, double unchanged) {
    return child - parent + unchanged;
}

// r^2 exactly, and 0 for a residual that is not finite.
HG_HD inline DmDD tc_square(double r) {
    return r - r == 0.0 ? dm_two_prod(r, r) : DmDD{};
}

// ---------------------------------------------------------------------------------------------
// Host

// A state's hyperedges as TcEdges storage.
struct TcEdgeLists {
    std::vector<uint32_t> off{0};
    std::vector<uint32_t> v;
    void clear() { off.assign(1, 0); v.clear(); }
    void add(const uint32_t* verts, uint32_t arity) {
        v.insert(v.end(), verts, verts + arity);
        off.push_back(static_cast<uint32_t>(v.size()));
    }
    uint32_t count() const { return static_cast<uint32_t>(off.size() - 1); }
    TcEdges view() const { return TcEdges{off.data(), v.data(), count()}; }
};

// One state's field with its graph, built in `bytes`.
struct TcFieldOf {
    std::vector<unsigned char> bytes;
    std::vector<uint32_t> verts;
    std::vector<double> f;
    SgGraph g;
};

inline void tc_field_of(const TcEdges& el, const TransitionCostSpec& spec, TcFieldOf& out) {
    size_t slots = 0, pairs = 0;
    sg_sizes(el, slots, pairs, spec.graph);
    out.verts.resize(slots + 1);
    const uint32_t n = tc_vertices(el, out.verts.data());
    out.verts.resize(n);
    out.bytes.resize(sg_build_bytes(slots, pairs));
    SgArena arena;
    arena.base = out.bytes.data();
    arena.capacity = out.bytes.size();
    sg_build(el, arena, out.g, spec.graph);
    out.f.assign(size_t{n} + 1, 0.0);
    if (spec.field != TC_FIELD_CURVATURE) {
        tc_local_field(el, spec.field, out.g, out.verts.data(), n, out.f.data());
        return;
    }
    // k_v, as stats::vertex_curvature computes it for "VertexCurvature".
    std::vector<unsigned char> scratch(size_t{1} << 16);
    std::vector<double> ball(slots + 1);
    SgGeometry geo;
    size_t needed = 0;
    while (!sg_state_geometry(el, scratch.data(), scratch.size(), geo, ball.data(),
                              static_cast<uint32_t>(ball.size()), needed, SG_OLLIVIER, IrSerial{},
                              nullptr, 0, nullptr, out.f.data(), nullptr, nullptr, 0, spec.graph))
        scratch.resize(needed > 2 * scratch.size() ? needed : 2 * scratch.size());
}

// A parent's part of every transition cost out of it: its hyperedges, field and graph, e_v at
// each vertex, B, and each vertex's hyperedges (inc[inc_off[i] .. inc_off[i+1]), each once).
struct TcParent {
    TcEdgeLists edges;
    TcFieldOf field;
    std::vector<double> unchanged;
    DmDD base;
    std::vector<uint32_t> inc_off, inc;
};

inline void tc_parent_of(const TcEdges& el, const TransitionCostSpec& spec, TcParent& p) {
    p.edges.clear();
    for (uint32_t i = 0; i < el.count(); ++i) p.edges.add(el.v + el.off[i], el.arity(i));
    tc_field_of(p.edges.view(), spec, p.field);
    const uint32_t n = static_cast<uint32_t>(p.field.verts.size());
    p.unchanged.resize(n);
    p.base = DmDD{};
    for (uint32_t i = 0; i < n; ++i) {
        p.unchanged[i] = tc_unchanged_residual(p.field.g, p.field.f.data(), i, spec.dt);
        p.base = dm_dd_add(p.base, tc_square(p.unchanged[i]));
    }
    const TcEdges e = p.edges.view();
    p.inc_off.assign(size_t{n} + 1, 0);
    auto each_vertex = [&](auto&& visit) {
        for (uint32_t i = 0; i < e.count(); ++i)
            for (uint32_t j = 0; j < e.arity(i); ++j) {
                bool first = true;
                for (uint32_t l = 0; l < j; ++l) first = first && e.at(i, l) != e.at(i, j);
                if (first) visit(sg_index_of(p.field.verts.data(), n, e.at(i, j)), i);
            }
    };
    each_vertex([&](uint32_t x, uint32_t) { ++p.inc_off[x + 1]; });
    for (uint32_t i = 0; i < n; ++i) p.inc_off[i + 1] += p.inc_off[i];
    p.inc.resize(p.inc_off[n]);
    std::vector<uint32_t> at(p.inc_off.begin(), p.inc_off.end() - 1);
    each_vertex([&](uint32_t x, uint32_t i) { p.inc[at[x]++] = i; });
}

// THE TRANSITION COST of the child that removes the parent's hyperedges `consumed` (nc distinct
// indices into p.edges) and adds `produced`.
inline double tc_child_cost(const TcParent& p, const uint32_t* consumed, uint32_t nc,
                            const TcEdges& produced, const TransitionCostSpec& spec) {
    HG_THREAD_LOCAL(std::vector<uint32_t>, touched);
    HG_THREAD_LOCAL(std::vector<uint32_t>, kept);
    HG_THREAD_LOCAL(TcEdgeLists, child);
    HG_THREAD_LOCAL(TcFieldOf, cf);
    const TcEdges pe = p.edges.view();
    const std::vector<uint32_t>& pv = p.field.verts;
    const uint32_t n = static_cast<uint32_t>(pv.size());
    auto gone = [&](uint32_t i) {
        for (uint32_t c = 0; c < nc; ++c)
            if (consumed[c] == i) return true;
        return false;
    };
    touched.clear();
    kept.clear();
    if (tc_field_is_local(spec.field)) {
        for (uint32_t c = 0; c < nc; ++c)
            for (uint32_t k = 0; k < pe.arity(consumed[c]); ++k)
                touched.push_back(sg_index_of(pv.data(), n, pe.at(consumed[c], k)));
        for (uint32_t i = 0; i < produced.count(); ++i)
            for (uint32_t k = 0; k < produced.arity(i); ++k) {
                const uint32_t x = sg_index_of(pv.data(), n, produced.at(i, k));
                if (x < n && pv[x] == produced.at(i, k)) touched.push_back(x);
            }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (uint32_t x : touched)
            for (uint32_t j = p.inc_off[x]; j < p.inc_off[x + 1]; ++j)
                if (!gone(p.inc[j])) kept.push_back(p.inc[j]);
        std::sort(kept.begin(), kept.end());
        kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
    } else {
        for (uint32_t x = 0; x < n; ++x) touched.push_back(x);
        for (uint32_t i = 0; i < pe.count(); ++i)
            if (!gone(i)) kept.push_back(i);
    }
    // The child's hyperedges at `touched`: every one of them that contains a touched vertex, or
    // the whole child.
    child.clear();
    for (uint32_t i : kept) child.add(pe.v + pe.off[i], pe.arity(i));
    for (uint32_t i = 0; i < produced.count(); ++i)
        child.add(produced.v + produced.off[i], produced.arity(i));
    tc_field_of(child.view(), spec, cf);
    const uint32_t cn = static_cast<uint32_t>(cf.verts.size());
    DmDD sum = p.base;
    for (uint32_t x : touched) {
        sum = dm_dd_sub(sum, tc_square(p.unchanged[x]));
        const uint32_t y = sg_index_of(cf.verts.data(), cn, pv[x]);
        if (y < cn && cf.verts[y] == pv[x])
            sum = dm_dd_add(sum, tc_square(tc_residual(cf.f[y], p.field.f[x], p.unchanged[x])));
    }
    return sum.hi + sum.lo;
}

// The transition cost from `parent` to `child`, two whole states: the parent's hyperedges that
// the child does not have are consumed and the child's that the parent does not have are
// produced, as multisets.
inline double tc_transition_cost(const TcEdges& parent, const TcEdges& child,
                                 const TransitionCostSpec& spec) {
    HG_THREAD_LOCAL(TcParent, p);
    tc_parent_of(parent, spec, p);
    auto content_order = [](const TcEdges& e) {
        std::vector<uint32_t> order(e.count());
        for (uint32_t i = 0; i < e.count(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return std::lexicographical_compare(e.v + e.off[a], e.v + e.off[a + 1],
                                                e.v + e.off[b], e.v + e.off[b + 1]);
        });
        return order;
    };
    const std::vector<uint32_t> po = content_order(parent), co = content_order(child);
    auto less = [](const TcEdges& x, uint32_t a, const TcEdges& y, uint32_t b) {
        return std::lexicographical_compare(x.v + x.off[a], x.v + x.off[a + 1], y.v + y.off[b],
                                            y.v + y.off[b + 1]);
    };
    std::vector<uint32_t> consumed;
    TcEdgeLists produced;
    size_t i = 0, j = 0;
    while (i < po.size() || j < co.size()) {
        if (j == co.size() || (i < po.size() && less(parent, po[i], child, co[j]))) {
            consumed.push_back(po[i++]);
        } else if (i == po.size() || less(child, co[j], parent, po[i])) {
            produced.add(child.v + child.off[co[j]], child.arity(co[j]));
            ++j;
        } else {
            ++i;
            ++j;
        }
    }
    return tc_child_cost(p, consumed.data(), static_cast<uint32_t>(consumed.size()),
                         produced.view(), spec);
}

// A caller's transition cost (the Python client's callback): the parent's and the child's
// hyperedges in, the cost out.
using TransitionCostHook = std::function<double(const TcEdges& parent, const TcEdges& child)>;

}  // namespace common
}  // namespace HG_NAMESPACE
