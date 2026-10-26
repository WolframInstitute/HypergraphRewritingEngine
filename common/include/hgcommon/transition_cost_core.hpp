#pragma once
#include "hgcommon/namespace.hpp"
// "TransitionCost": the cost of a transition from a parent state P to a child state C, read from
// a field on the vertices of both (docs/TRANSITION_COST.md), one body for host and device. A
// rewrite keeps the ids of the vertices it does not delete, so a vertex of both states has one
// id in both.
//
// The cost is the heat residual of the field: for each vertex v of both states,
//
//   r_v = f_C(v) - f_P(v) - dt * sum over the neighbours w of v in G_P of (f_P(w) - f_P(v))
//
// G_P the parent's graph under "StatisticsGraph" (state_geometry_core.hpp, sg_build), and the
// cost the sum of r_v^2 over the vertices with a finite r_v.
//
// ONE PARENT, MANY CHILDREN. A child that leaves v's field unchanged has the residual
// e_v = -dt ((A - D) f_P)(v) at v, so the cost is B - (sum over T of e_v^2) + (sum over T of
// r_v^2), where B is the sum of every e_v^2 and T holds the vertices whose field the transition
// can change. tc_parent_build computes the field, e_v and B once per parent; tc_child_cost
// computes a child from the hyperedges the transition consumes and produces, after dropping
// each consumed hyperedge that a produced one equals. The degree in G, the loop count and the
// edge multiplicity of v read only the hyperedges that contain v, so T is the parent's vertices
// of those hyperedges, and the child's field is read from the child's hyperedges at T. The
// curvature reads a wider neighbourhood, so T is every vertex and the child's field is the
// whole child's.
//
// THE SUMS DO NOT READ THE VERTEX IDS. Each of the three sums adds its exact squares
// (double-double, det_math.hpp) in ascending order of value, so the cost is a function of the
// multisets of per-vertex values, and two engines that number the vertices differently get the
// same double.
//
// tc_child_cost is the one implementation: the CPU engine (ParallelEvolutionEngine::
// transition_cost_of) and the device (transition_cost_device) call it for each costed
// transition, and tc_transition_cost, which the Python client's heat_residual calls with two
// whole states, calls it with the hyperedges that differ.
#include "hgcommon/core.hpp"
#include "hgcommon/det_math.hpp"
#include "hgcommon/state_geometry_core.hpp"

#include <cstdint>

#ifndef __CUDACC__
#include <algorithm>
#include <functional>
#include <vector>
#endif

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

HG_HD inline bool tc_same_edge(const TcEdges& a, uint32_t i, const TcEdges& b, uint32_t j) {
    if (a.arity(i) != b.arity(j)) return false;
    for (uint32_t k = 0; k < a.arity(i); ++k)
        if (a.at(i, k) != b.at(j, k)) return false;
    return true;
}

// A squared residual, ordered by value.
struct TcTerm {
    double hi = 0.0;
    double lo = 0.0;
    HG_HD bool operator<(const TcTerm& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
    HG_HD bool operator==(const TcTerm& o) const { return hi == o.hi && lo == o.lo; }
};

// r^2 exactly, and 0 for a residual that is not finite.
HG_HD inline TcTerm tc_square(double r) {
    if (!(r - r == 0.0)) return TcTerm{};
    const DmDD s = dm_two_prod(r, r);
    return TcTerm{s.hi, s.lo};
}

// The sum of `n` terms in ascending order of value; sorts `t`.
HG_HD inline DmDD tc_sum(TcTerm* t, uint32_t n) {
    sg_sort(t, n);
    DmDD s{};
    for (uint32_t i = 0; i < n; ++i) s = dm_dd_add(s, DmDD{t[i].hi, t[i].lo});
    return s;
}

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

// Whether vertex k of hyperedge i of `el` is the hyperedge's first occurrence of that vertex.
template <class EdgeList>
HG_HD bool tc_first_occurrence(const EdgeList& el, uint32_t i, uint32_t k) {
    for (uint32_t l = 0; l < k; ++l)
        if (el.at(i, l) == el.at(i, k)) return false;
    return true;
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
        for (uint32_t j = 0; j < k; ++j)
            if (tc_first_occurrence(el, i, j)) f[sg_index_of(verts, n, el.at(i, j))] += 1.0;
    }
}

// A state's field in an arena: its vertices, its graph and the value at each vertex.
struct TcField {
    const uint32_t* verts = nullptr;
    uint32_t n = 0;
    const double* f = nullptr;
    SgGraph g;
};

// The field of `el` into `a`. False when `a` is short; for the curvature, a.extra then holds the
// bytes the geometry asked for beyond what `a` had left.
HG_HD inline bool tc_field_build(const TcEdges& el, const TransitionCostSpec& spec, SgArena& a,
                                 TcField& out) {
    size_t slots = 0, pairs = 0;
    sg_sizes(el, slots, pairs, spec.graph);
    uint32_t* verts = a.take<uint32_t>(slots + 1);
    if (a.failed) return false;
    const uint32_t n = tc_vertices(el, verts);
    if (!sg_build(el, a, out.g, spec.graph)) return false;
    double* f = a.take<double>(size_t{n} + 1);
    if (a.failed) return false;
    out.verts = verts;
    out.n = n;
    out.f = f;
    for (uint32_t i = 0; i <= n; ++i) f[i] = 0.0;
    if (spec.field != TC_FIELD_CURVATURE) {
        tc_local_field(el, spec.field, out.g, verts, n, f);
        return true;
    }
    // k_v, as stats::vertex_curvature computes it for "VertexCurvature".
    double* ball = a.take<double>(slots + 1);
    if (a.failed) return false;
    const size_t at = (a.used + 7) & ~size_t{7};
    const size_t left = at < a.capacity ? a.capacity - at : 0;
    SgGeometry geo;
    size_t needed = 0;
    if (!sg_state_geometry(el, a.base + at, left, geo, ball, static_cast<uint32_t>(slots + 1),
                           needed, SG_OLLIVIER, IrSerial{}, nullptr, 0, nullptr, f, nullptr,
                           nullptr, 0, spec.graph)) {
        a.failed = true;
        a.extra = needed;
        return false;
    }
    a.used = at + needed;
    return true;
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

// A parent's part of every transition cost out of it: its hyperedges, field and graph, e_v at
// each vertex index, B, and each vertex's hyperedges (inc[inc_off[i] .. inc_off[i+1]), each
// once).
struct TcParentView {
    TcEdges edges;
    TcField field;
    const double* unchanged = nullptr;
    DmDD base;
    const uint32_t* inc_off = nullptr;
    const uint32_t* inc = nullptr;
};

// The parent's part of `el` (which must outlive `p`) into `a`. False when `a` is short.
HG_HD inline bool tc_parent_build(const TcEdges& el, const TransitionCostSpec& spec, SgArena& a,
                                  TcParentView& p) {
    p.edges = el;
    if (!tc_field_build(el, spec, a, p.field)) return false;
    const uint32_t n = p.field.n;
    double* e = a.take<double>(size_t{n} + 1);
    TcTerm* terms = a.take<TcTerm>(size_t{n} + 1);
    uint32_t* inc_off = a.take<uint32_t>(size_t{n} + 2);
    size_t slots = 0;
    for (uint32_t i = 0; i < el.count(); ++i) slots += el.arity(i);
    uint32_t* inc = a.take<uint32_t>(slots + 1);
    uint32_t* fill = a.take<uint32_t>(size_t{n} + 1);
    if (a.failed) return false;
    for (uint32_t i = 0; i < n; ++i) {
        e[i] = tc_unchanged_residual(p.field.g, p.field.f, i, spec.dt);
        terms[i] = tc_square(e[i]);
    }
    p.unchanged = e;
    p.base = tc_sum(terms, n);
    for (uint32_t i = 0; i <= n; ++i) inc_off[i] = 0;
    for (uint32_t i = 0; i < el.count(); ++i)
        for (uint32_t k = 0; k < el.arity(i); ++k)
            if (tc_first_occurrence(el, i, k))
                ++inc_off[sg_index_of(p.field.verts, n, el.at(i, k)) + 1];
    for (uint32_t i = 0; i < n; ++i) inc_off[i + 1] += inc_off[i];
    for (uint32_t i = 0; i < n; ++i) fill[i] = inc_off[i];
    for (uint32_t i = 0; i < el.count(); ++i)
        for (uint32_t k = 0; k < el.arity(i); ++k)
            if (tc_first_occurrence(el, i, k))
                inc[fill[sg_index_of(p.field.verts, n, el.at(i, k))]++] = i;
    p.inc_off = inc_off;
    p.inc = inc;
    return true;
}

// THE TRANSITION COST of the child that removes the parent's hyperedges `consumed` (nc distinct
// indices into p.edges) and adds `produced`, into `out`. False when `a` is short.
HG_HD inline bool tc_child_cost(const TcParentView& p, const uint32_t* consumed, uint32_t nc,
                                const TcEdges& produced, const TransitionCostSpec& spec,
                                SgArena& a, double& out) {
    const TcEdges& pe = p.edges;
    const uint32_t n = p.field.n;
    const uint32_t* pv = p.field.verts;
    const bool local = tc_field_is_local(spec.field);
    // A consumed hyperedge that a produced one equals changes nothing: both are dropped, each
    // produced hyperedge matching at most one consumed one.
    uint32_t* gone = a.take<uint32_t>(size_t{nc} + 1);
    uint8_t* made = a.take<uint8_t>(size_t{produced.count()} + 1);
    if (a.failed) return false;
    uint32_t ng = 0;
    for (uint32_t j = 0; j < produced.count(); ++j) made[j] = 1u;
    for (uint32_t c = 0; c < nc; ++c) {
        bool same = false;
        for (uint32_t j = 0; j < produced.count() && !same; ++j)
            if (made[j] && tc_same_edge(pe, consumed[c], produced, j)) {
                made[j] = 0u;
                same = true;
            }
        if (!same) gone[ng++] = consumed[c];
    }
    auto is_gone = [&](uint32_t i) {
        for (uint32_t c = 0; c < ng; ++c)
            if (gone[c] == i) return true;
        return false;
    };
    // T: the parent's vertex indices whose field can change.
    size_t cap_t = n;
    if (local) {
        cap_t = 0;
        for (uint32_t c = 0; c < ng; ++c) cap_t += pe.arity(gone[c]);
        for (uint32_t j = 0; j < produced.count(); ++j) cap_t += produced.arity(j);
    }
    uint32_t* touched = a.take<uint32_t>(cap_t + 1);
    if (a.failed) return false;
    uint32_t nt = 0;
    if (local) {
        for (uint32_t c = 0; c < ng; ++c)
            for (uint32_t k = 0; k < pe.arity(gone[c]); ++k)
                touched[nt++] = sg_index_of(pv, n, pe.at(gone[c], k));
        for (uint32_t j = 0; j < produced.count(); ++j) {
            if (!made[j]) continue;
            for (uint32_t k = 0; k < produced.arity(j); ++k) {
                const uint32_t x = sg_index_of(pv, n, produced.at(j, k));
                if (x < n && pv[x] == produced.at(j, k)) touched[nt++] = x;
            }
        }
        sg_sort(touched, nt);
        nt = static_cast<uint32_t>(sg_unique(touched, nt));
    } else {
        for (uint32_t x = 0; x < n; ++x) touched[nt++] = x;
    }
    // The parent's hyperedges the child keeps that contain a vertex of T (all of them for the
    // curvature), then the produced ones: every hyperedge of the child at T.
    size_t cap_k = pe.count();
    if (local) {
        cap_k = 0;
        for (uint32_t t = 0; t < nt; ++t)
            cap_k += p.inc_off[touched[t] + 1] - p.inc_off[touched[t]];
    }
    uint32_t* kept = a.take<uint32_t>(cap_k + 1);
    if (a.failed) return false;
    uint32_t nk = 0;
    if (local) {
        for (uint32_t t = 0; t < nt; ++t)
            for (uint32_t j = p.inc_off[touched[t]]; j < p.inc_off[touched[t] + 1]; ++j)
                if (!is_gone(p.inc[j])) kept[nk++] = p.inc[j];
        sg_sort(kept, nk);
        nk = static_cast<uint32_t>(sg_unique(kept, nk));
    } else {
        for (uint32_t i = 0; i < pe.count(); ++i)
            if (!is_gone(i)) kept[nk++] = i;
    }
    uint32_t np = 0;
    size_t slots = 0;
    for (uint32_t k = 0; k < nk; ++k) slots += pe.arity(kept[k]);
    for (uint32_t j = 0; j < produced.count(); ++j)
        if (made[j]) {
            slots += produced.arity(j);
            ++np;
        }
    uint32_t* coff = a.take<uint32_t>(size_t{nk} + np + 1);
    uint32_t* cv = a.take<uint32_t>(slots + 1);
    if (a.failed) return false;
    uint32_t w = 0, row = 0;
    coff[0] = 0;
    for (uint32_t k = 0; k < nk; ++k) {
        for (uint32_t x = 0; x < pe.arity(kept[k]); ++x) cv[w++] = pe.at(kept[k], x);
        coff[++row] = w;
    }
    for (uint32_t j = 0; j < produced.count(); ++j) {
        if (!made[j]) continue;
        for (uint32_t x = 0; x < produced.arity(j); ++x) cv[w++] = produced.at(j, x);
        coff[++row] = w;
    }
    const TcEdges child{coff, cv, row};
    TcField cf;
    if (!tc_field_build(child, spec, a, cf)) return false;
    TcTerm* sub = a.take<TcTerm>(size_t{nt} + 1);
    TcTerm* add = a.take<TcTerm>(size_t{nt} + 1);
    if (a.failed) return false;
    uint32_t na = 0;
    for (uint32_t t = 0; t < nt; ++t) {
        const uint32_t x = touched[t];
        sub[t] = tc_square(p.unchanged[x]);
        const uint32_t y = sg_index_of(cf.verts, cf.n, pv[x]);
        if (y < cf.n && cf.verts[y] == pv[x])
            add[na++] = tc_square(tc_residual(cf.f[y], p.field.f[x], p.unchanged[x]));
    }
    const DmDD s = dm_dd_add(dm_dd_sub(p.base, tc_sum(sub, nt)), tc_sum(add, na));
    out = s.hi + s.lo;
    return true;
}

#ifndef __CUDACC__
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

// Runs build(arena) in `bytes`, growing it until the build fits.
template <class Build>
inline void tc_with_arena(std::vector<unsigned char>& bytes, Build&& build) {
    if (bytes.size() < 4096) bytes.resize(4096);
    for (;;) {
        SgArena a;
        a.base = bytes.data();
        a.capacity = bytes.size();
        if (build(a)) return;
        const size_t want = a.used + a.extra + 64;
        bytes.resize(want > 2 * bytes.size() ? want : 2 * bytes.size());
    }
}

// One state's field, in its own storage.
struct TcFieldOf {
    std::vector<unsigned char> bytes;
    TcField field;
};

inline void tc_field_of(const TcEdges& el, const TransitionCostSpec& spec, TcFieldOf& out) {
    tc_with_arena(out.bytes, [&](SgArena& a) { return tc_field_build(el, spec, a, out.field); });
}

// A parent's part, with its hyperedges, in its own storage.
struct TcParent {
    TcEdgeLists edges;
    std::vector<unsigned char> bytes;
    TcParentView view;
};

inline void tc_parent_of(const TcEdges& el, const TransitionCostSpec& spec, TcParent& p) {
    p.edges.clear();
    for (uint32_t i = 0; i < el.count(); ++i) p.edges.add(el.v + el.off[i], el.arity(i));
    tc_with_arena(p.bytes,
                  [&](SgArena& a) { return tc_parent_build(p.edges.view(), spec, a, p.view); });
}

inline double tc_child_cost(const TcParent& p, const uint32_t* consumed, uint32_t nc,
                            const TcEdges& produced, const TransitionCostSpec& spec) {
    HG_THREAD_LOCAL(std::vector<unsigned char>, bytes);
    double out = 0.0;
    tc_with_arena(bytes, [&](SgArena& a) {
        return tc_child_cost(p.view, consumed, nc, produced, spec, a, out);
    });
    return out;
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
#endif

}  // namespace common
}  // namespace HG_NAMESPACE
