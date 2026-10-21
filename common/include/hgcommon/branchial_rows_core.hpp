#pragma once
#include "hgcommon/namespace.hpp"
// ONE ROW OF THE "StepStatisticsBranchial" OVERLAP PASS, one body for the host's job-system parts
// (paclet_source/paclet_support.cpp, BranchialStepWork::rows) and the device's warps
// (gpu/src/branchial_rows.cu).
//
// A step's k states are numbered 0..k-1 by effective id. Row a counts every state b > a that
// shares a vertex with a, through the inverted index of holders (each vertex's holders,
// ascending), and with the branchial graph every b > a in a's component, sharing or not, with its
// distance. Each pair is emitted once as (distance, b, |B|, shared): distance 0 for every pair that
// shares a vertex, and its distance d > 0 for every pair of one component. The caller counts the
// emissions per key, or keeps each pair's shared count for the state matrices, which
// bo_matrix_row then turns into the row's matrix cells; |A| is the row's.
// The same rows over the states' edge numbers in place of their vertices give the shared edges.
//
// The lanes of the policy (IrSerial on the host, IrTile on the device) split each holder list and
// each breadth-first level. The search's distances and the shared counts do not depend on which
// lane visits what, so neither do the emissions.
//
// A Ctx supplies the row's scratch, each array reached only through ld and st:
//   uint32_t* count        k entries, all 0 between rows
//   uint32_t* touched      k entries
//   int32_t*  dist         graph nodes entries, all -1 between rows
//   uint32_t* queue        graph nodes entries
//   uint32_t* n_touched, n_queue      counters, 0 between rows
//   ld(p), st(p, v)        a load and a store of the scratch, seen by every lane after par.sync()
//   uint32_t add_count(uint32_t b)     count[b] += 1, returns the old value; atomic on the device
//   bool     claim_dist(uint32_t x, int32_t d)   dist[x] from -1 to d; true for the one claimer
//   uint32_t push(uint32_t* counter)  returns the old value and adds 1; atomic on the device
//   void     emit(uint32_t distance, uint32_t b, uint32_t size_b, uint32_t shared)

#include <stdint.h>

#include "hgcommon/branchial_overlap_core.hpp"
#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

constexpr uint32_t BR_NONE = 0xFFFFFFFFu;

// One step of the overlap pass, in compressed rows.
struct BrStep {
    uint32_t k = 0;                         // states
    const uint32_t* size = nullptr;         // |state s|, k entries
    const uint32_t* row_off = nullptr;      // k + 1: state s's vertices are row_vert[row_off[s]..]
    const uint32_t* row_vert = nullptr;     // dense vertex numbers
    uint32_t vertices = 0;
    const uint32_t* hold_off = nullptr;     // vertices + 1: vertex v's holders, ascending
    const uint32_t* hold = nullptr;
    // The branchial graph, when its distances are asked for (to_graph null otherwise): n nodes.
    uint32_t n = 0;
    const uint32_t* adj_off = nullptr;      // n + 1
    const uint32_t* adj = nullptr;
    const uint32_t* to_graph = nullptr;     // k entries, BR_NONE for a state not in the graph
    const uint32_t* from_graph = nullptr;   // n entries, BR_NONE for a node with no state
};

// The cell of pair a < b in a step's upper triangle of k states, row by row: k (k - 1) / 2 cells.
HG_HD inline uint64_t br_pair_index(uint32_t k, uint32_t a, uint32_t b) {
    return uint64_t{a} * k - uint64_t{a} * (a + 1) / 2 + (b - a - 1);
}

// The state matrices of one level (vertex sets or edge sets) of a step's k states: `shared` holds
// the k (k - 1) / 2 shared counts by br_pair_index, `size` each state's size, `universe` |U|, and
// m[kind] is a k x k row-major array (kind 0 the overlap, 1 the cosine, 2 the mutual information)
// or null when that matrix is not asked for.
struct BoMatrices {
    uint32_t k = 0;
    const uint32_t* size = nullptr;
    const uint32_t* shared = nullptr;
    uint64_t universe = 0;
    double* m[3] = {nullptr, nullptr, nullptr};
};

// One pair's value of `kind` from n11 and the two sizes, taken as (smaller, larger): the key the
// overlap summaries read, so a cell equals the value its summary counts.
HG_HD inline double bo_pair_value(uint32_t kind, uint32_t both, uint32_t a, uint32_t b,
                                  uint64_t universe) {
    const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
    if (kind == 0) return bo_jaccard_of_union(both, uint64_t{lo} + hi - both);
    if (kind == 1) return bo_cosine(both, lo, hi);
    return bo_mutual_information(both, lo, hi, universe);
}

// Row a of the state matrices, once row a's shared counts are in x.shared: the cell (a, a) and,
// for every b > a, the cells (a, b) and (b, a), each pair's value computed once. Lane `lane` of
// `lanes` takes every lanes-th b. No other row writes these cells.
HG_HD inline void bo_matrix_row(const BoMatrices& x, uint32_t a, uint32_t lane, uint32_t lanes) {
    const uint32_t sa = x.size[a];
    for (uint32_t kind = 0; kind < 3; ++kind) {
        double* m = x.m[kind];
        if (!m) continue;
        if (lane == 0) m[uint64_t{a} * x.k + a] = bo_pair_value(kind, sa, sa, sa, x.universe);
        for (uint32_t b = a + 1 + lane; b < x.k; b += lanes) {
            const double v = bo_pair_value(kind, x.shared[br_pair_index(x.k, a, b)], sa,
                                           x.size[b], x.universe);
            m[uint64_t{a} * x.k + b] = v;
            m[uint64_t{b} * x.k + a] = v;
        }
    }
}

// The first index in hs[0..len) whose value is above a; hs ascending.
HG_HD inline uint32_t br_upper_bound(const uint32_t* hs, uint32_t len, uint32_t a) {
    uint32_t lo = 0, hi = len;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (hs[mid] <= a) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

template <class Ctx, class Par>
HG_HD void br_row(Ctx& c, const BrStep& s, uint32_t a, Par par) {
    // Shared vertices: one count per b > a, each b listed once in `touched`.
    for (uint32_t i = s.row_off[a]; i < s.row_off[a + 1]; ++i) {
        const uint32_t v = s.row_vert[i];
        const uint32_t* hs = s.hold + s.hold_off[v];
        const uint32_t len = s.hold_off[v + 1] - s.hold_off[v];
        const uint32_t from = br_upper_bound(hs, len, a);
        par.fan(len - from, [&](uint32_t j) {
            const uint32_t b = hs[from + j];
            if (c.add_count(b) == 0) c.st(&c.touched[c.push(c.n_touched)], b);
        });
    }
    par.sync();
    // Distances: a breadth-first search of a's component, one level at a time.
    uint32_t reached = 0;
    if (s.to_graph && s.to_graph[a] != BR_NONE) {
        if (par.leader()) {
            c.st(&c.dist[s.to_graph[a]], 0);
            c.st(&c.queue[0], s.to_graph[a]);
            c.st(c.n_queue, 1u);
        }
        par.sync();
        uint32_t begin = 0, end = par.bcast(c.ld(c.n_queue));
        for (int32_t d = 1; begin < end; ++d) {
            par.fan(end - begin, [&](uint32_t j) {
                const uint32_t x = c.ld(&c.queue[begin + j]);
                for (uint32_t e = s.adj_off[x]; e < s.adj_off[x + 1]; ++e)
                    if (c.claim_dist(s.adj[e], d)) c.st(&c.queue[c.push(c.n_queue)], s.adj[e]);
            });
            par.sync();
            begin = end;
            end = par.bcast(c.ld(c.n_queue));
        }
        reached = end;
        par.fan(reached, [&](uint32_t j) {
            const uint32_t x = c.ld(&c.queue[j]);
            const uint32_t b = s.from_graph[x];
            if (b != BR_NONE && b > a)
                c.emit(static_cast<uint32_t>(c.ld(&c.dist[x])), b, s.size[b], c.ld(&c.count[b]));
        });
        par.sync();
    }
    const uint32_t n_touched = par.bcast(c.ld(c.n_touched));
    par.fan(n_touched, [&](uint32_t j) {
        const uint32_t b = c.ld(&c.touched[j]);
        c.emit(0, b, s.size[b], c.ld(&c.count[b]));
        c.st(&c.count[b], 0u);
    });
    par.fan(reached, [&](uint32_t j) { c.st(&c.dist[c.ld(&c.queue[j])], -1); });
    if (par.leader()) {
        c.st(c.n_touched, 0u);
        c.st(c.n_queue, 0u);
    }
    par.sync();
}

}  // namespace common
}  // namespace HG_NAMESPACE
