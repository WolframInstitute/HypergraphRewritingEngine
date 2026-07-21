// The engine's ONLINE transitive reduction of the causal graph must equal the exact
// OFFLINE transitive reduction of the full (TR-off) causal graph -- on real evolution
// output, not just hand-built DAGs. Previously the causal/branchial suite tested TR
// only on manually-constructed micro-DAGs; this closes that gap (audit R7). The
// offline reduction here is the same reference the standalone probe uses.
#include <gtest/gtest.h>
#include <hypergraph/parallel_evolution.hpp>

#include <functional>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace hypergraph;

namespace {
using PairSet = std::set<std::pair<uint32_t, uint32_t>>;

PairSet pairs_of(const std::vector<CausalEdge>& edges) {
    PairSet p;
    for (const auto& c : edges) p.insert({c.producer, c.consumer});
    return p;
}

// Exact minimal transitive reduction of a pair set (unique for a DAG): drop an edge
// (u,w) iff w is reachable from a different successor of u.
PairSet offline_tr(const PairSet& pairs) {
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> succ;
    for (const auto& pc : pairs) succ[pc.first].insert(pc.second);

    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> reach;
    std::unordered_set<uint32_t> done;
    std::function<std::unordered_set<uint32_t>&(uint32_t)> R =
        [&](uint32_t u) -> std::unordered_set<uint32_t>& {
        auto& s = reach[u];
        if (done.count(u)) return s;
        done.insert(u);
        auto it = succ.find(u);
        if (it != succ.end())
            for (uint32_t w : it->second) {
                s.insert(w);
                auto& rw = R(w);
                s.insert(rw.begin(), rw.end());
            }
        return s;
    };

    PairSet kept;
    for (const auto& pc : pairs) {
        bool redundant = false;
        for (uint32_t w : succ[pc.first]) {
            if (w == pc.second) continue;
            if (R(w).count(pc.second)) { redundant = true; break; }
        }
        if (!redundant) kept.insert(pc);
    }
    return kept;
}

PairSet run(RewriteRule rule, std::vector<std::vector<VertexId>> init, int steps,
            bool tr, unsigned threads) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_transitive_reduction(tr);
    e.set_explore_from_canonical_states_only(false);
    e.add_rule(rule);
    e.evolve(init, steps);
    return pairs_of(hg.causal_graph().get_causal_edges());
}

RewriteRule rWolfram() {
    return make_rule(0).lhs({0, 1}).lhs({0, 2}).rhs({0, 1}).rhs({0, 3}).rhs({1, 3}).rhs({2, 3}).build();
}
RewriteRule rSplit() { return make_rule(0).lhs({0, 1}).rhs({0, 2}).rhs({2, 1}).build(); }
RewriteRule rMixed() {
    return make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).rhs({2, 1}).rhs({1, 0}).build();
}

struct Case { const char* name; RewriteRule (*rule)(); std::vector<std::vector<VertexId>> init; int steps; };
}  // namespace

TEST(CausalTrExactnessTest, OnlineTrMatchesOfflineTr) {
    const std::vector<Case> cases = {
        {"wolfram/s4", rWolfram, {{0u, 1u}, {0u, 2u}}, 4},
        {"split/s5",   rSplit,   {{0u, 1u}}, 5},
        {"mixed/s4",   rMixed,   {{0u, 1u}, {1u, 2u}}, 4},
    };
    // Single-threaded so the two evolutions assign identical event IDs (TR changes
    // which causal edges are recorded, not which events are created), making the
    // producer/consumer pair sets directly comparable. The engine's determinism
    // across thread counts is covered by the determinism-fuzzing suite.
    for (const auto& c : cases) {
        PairSet full    = run(c.rule(), c.init, c.steps, /*tr=*/false, /*threads=*/1);
        PairSet online  = run(c.rule(), c.init, c.steps, /*tr=*/true, /*threads=*/1);
        PairSet offline = offline_tr(full);
        EXPECT_EQ(online, offline)
            << c.name << ": online TR (" << online.size() << " edges) != offline TR ("
            << offline.size() << ") of full (" << full.size() << ")";
    }
}
