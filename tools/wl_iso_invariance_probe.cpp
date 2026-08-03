// tools/wl_iso_invariance_probe.cpp
//
// Does the engine's WL canonical hash actually separate isomorphism classes?
//
// The WL hash is what `canonical_hash` carries in EVERY mode except Full -- and None is the
// default -- and it is also what resolves the canonical representative for EVENT
// canonicalization in ALL modes. So if it assigns two different values to two presentations
// of ONE graph, states that are the same get treated as different, and event identity and
// edge correspondence split with them.
//
// Method: take a state, enumerate presentations of it (every permutation of the edge order x
// every relabelling of the vertices), and group them by the EXACT IR canonical hash, which is
// the oracle for "same graph". Within one IR class, every WL hash must agree. Report any
// class WL splits.
//

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <functional>

using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

// Engine WL hash of a state built from `edges`.
static uint64_t engine_wl(const Edges& edges) {
    Hypergraph hg;
    SparseBitset set;
    for (const auto& e : edges) set.set(hg.create_edge(e.data(), (uint8_t)e.size()), hg.arena());
    return hg.compute_wl_hash(set);
}

static uint64_t exact_ir(const Edges& edges) {
    IRCanonicalizer ir;
    return ir.compute_canonical_hash(edges);
}

// Every presentation: all vertex relabellings x all edge orders.
static void for_each_presentation(const Edges& base, uint32_t nv,
                                  const std::function<void(const Edges&)>& f) {
    std::vector<VertexId> perm(nv);
    for (uint32_t i = 0; i < nv; ++i) perm[i] = i;
    do {
        Edges relabelled;
        relabelled.reserve(base.size());
        for (const auto& e : base) {
            std::vector<VertexId> ne;
            ne.reserve(e.size());
            for (VertexId v : e) ne.push_back(perm[v]);
            relabelled.push_back(std::move(ne));
        }
        std::vector<size_t> order(relabelled.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        do {
            Edges permuted;
            permuted.reserve(order.size());
            for (size_t i : order) permuted.push_back(relabelled[i]);
            f(permuted);
        } while (std::next_permutation(order.begin(), order.end()));
    } while (std::next_permutation(perm.begin(), perm.end()));
}

static bool check(const char* name, const Edges& base, uint32_t nv) {
    std::map<uint64_t, std::set<uint64_t>> ir_to_wl;   // IR class -> the WL values seen in it
    for_each_presentation(base, nv, [&](const Edges& p) {
        ir_to_wl[exact_ir(p)].insert(engine_wl(p));
    });
    size_t split = 0;
    for (const auto& [ir, wls] : ir_to_wl) if (wls.size() > 1) ++split;
    std::printf("  %-26s IR classes=%-3zu  WL-split classes=%-3zu  %s\n",
                name, ir_to_wl.size(), split,
                split ? "*** WL SPLITS AN ISOMORPHISM CLASS ***" : "ok");
    return split == 0;
}

int main() {
    std::printf("Engine WL hash vs the exact IR oracle, over every presentation:\n");
    bool ok = true;
    // No repeated vertex inside any edge -> nn(v) <= total_occ, so no truncation.
    ok &= check("binary path (no repeats)",  {{0,1},{1,2}}, 3);
    ok &= check("arity-3, no repeats",       {{0,1,2},{2,1,0}}, 3);
    // A repeated vertex inside an edge of arity >= 3 makes nn(v) exceed total_occ.
    ok &= check("self-loop arity 3",         {{0,0,0}}, 1);
    ok &= check("arity-3 with one repeat",   {{0,0,1}}, 2);
    ok &= check("the reported minimal case", {{1,0,1,1},{1,1,0}}, 2);
    ok &= check("arity-4 repeat + partner",  {{0,0,0,1},{1,0}}, 2);
    std::printf("%s\n", ok ? "ALL ISOMORPHISM CLASSES INTACT"
                           : "ISOMORPHISM-INVARIANCE IS BROKEN");
    return ok ? 0 : 1;
}
