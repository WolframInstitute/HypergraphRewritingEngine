// The engine's ONLINE transitive reduction of the causal graph must equal the exact
// OFFLINE transitive reduction of the full (TR-off) causal graph -- on real evolution
// output, not just hand-built DAGs. Previously the causal/branchial suite tested TR
// only on manually-constructed micro-DAGs; this closes that gap (audit R7). The
// offline reduction here is the same reference the standalone probe uses.
#include <gtest/gtest.h>
#include <hypergraph/parallel_evolution.hpp>
#include <hgcommon/transitive_reduction.hpp>

#include <functional>
#include <map>
#include <set>
#include <string>
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

// A SURPLUS EDGE AND THE PATH THAT IMPLIES IT, in the failure message.
//
// The count says the online rule kept something it should not have; it does not say which pair,
// or which path was already there to make it redundant. Those are what identify the moment the
// rule answered, because the answer is a search over exactly that path -- so a firing that names
// them is a diagnosis, and one that reports only a number costs a rare interleaving to learn
// nothing new.
std::string witness(const PairSet& online, const PairSet& self) {
    std::string out;
    int shown = 0;
    for (const auto& pc : online) {
        if (self.count(pc) || shown >= 3) continue;
        ++shown;
        // Shortest path p -> .. -> c through the kept set, avoiding the direct edge. Its
        // interior nodes are the events whose in-edges had to be present for the rule to have
        // answered "redundant", and their ids say where they sit relative to p and c.
        std::map<uint32_t, uint32_t> prev;
        std::vector<uint32_t> frontier{pc.first};
        prev[pc.first] = pc.first;
        while (!frontier.empty() && !prev.count(pc.second)) {
            std::vector<uint32_t> next;
            for (uint32_t u : frontier)
                for (const auto& e : online) {
                    if (e.first != u) continue;
                    if (e.first == pc.first && e.second == pc.second) continue;  // not the edge itself
                    if (prev.count(e.second)) continue;
                    prev[e.second] = u;
                    next.push_back(e.second);
                }
            frontier.swap(next);
        }
        out += "  surplus edge " + std::to_string(pc.first) + " -> " + std::to_string(pc.second);
        if (prev.count(pc.second)) {
            std::vector<uint32_t> path;
            for (uint32_t x = pc.second; x != pc.first; x = prev[x]) path.push_back(x);
            path.push_back(pc.first);
            out += ", implied by ";
            for (size_t i = path.size(); i-- > 0;) {
                out += std::to_string(path[i]);
                if (i) out += "->";
            }
        } else {
            out += ", and NO path through the kept set implies it -- the oracle and the online "
                   "rule disagree about the same set, which is a fault in one of them";
        }
        out += "\n";
    }
    return out;
}
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

// A REDUCTION IS ITS OWN REDUCTION, and that is checkable at any thread count.
//
// The reduction of a DAG preserves its reachability, so no edge of the reduction is implied by a
// path through the others: for a kept set K that is the reduction of the full relation F,
// tr_reduce(K) == K. A kept edge that a path through other kept edges already implies is a
// definite fault in the online rule, whatever F was.
//
// This needs neither a second run nor a mapping between two runs' event ids, which is what
// confines OnlineTrMatchesOfflineTr above to one thread. So it runs where the online rule is
// actually under pressure, and it names the fault on the run that produced it -- the determinism
// gate can only report, later, that two runs disagreed.
TEST(CausalTrExactnessTest, OnlineTrIsItsOwnReductionUnderConcurrency) {
    const std::vector<Case> cases = {
        // The depths the determinism gate runs, so this checks the reduction on the same
        // relation sizes that gate compares -- a reduction that is exact on a hundred edges
        // says nothing about one on twenty-five thousand.
        {"wolfram/s6", rWolfram, {{0u, 1u}, {0u, 2u}}, 6},
        {"mixed/s5",   rMixed,   {{0u, 1u}, {1u, 2u}}, 5},
        {"split/s6",   rSplit,   {{0u, 1u}}, 6},
    };
    const unsigned hw = std::max(4u, std::thread::hardware_concurrency());
    for (const auto& c : cases) {
        for (unsigned th : {4u, hw, hw * 2}) {
            for (int rep = 0; rep < 4; ++rep) {
                const PairSet online = run(c.rule(), c.init, c.steps, /*tr=*/true, th);
                PairSet self;
                // ids_topological is left off: the oracle then assumes nothing about the ids of
                // the set it is handed, so it stays a check on the online rule rather than on
                // the same premise the online rule was written under.
                hgcommon::tr_reduce(
                    [&](auto&& add) { for (const auto& pc : online) add(pc.first, pc.second); },
                    [&](uint32_t p, uint32_t q) { self.insert({p, q}); });
                ASSERT_EQ(online.size(), self.size())
                    << c.name << " at threads=" << th << " rep=" << rep << ": the kept set holds "
                    << (online.size() - self.size()) << " edge(s) that a path through other kept "
                       "edges already implies, so it is not the reduction of anything.\n"
                    << witness(online, self);
            }
        }
    }
}
