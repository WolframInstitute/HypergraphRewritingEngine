// The shared join, checked against brute force.
//
// hgcommon/join_core.hpp is the ONE body both engines will run. Before either is pointed at it,
// it is checked here against an independent enumeration -- all ordered tuples of distinct edges,
// filtered by a consistent binding -- which is the definition the reference implementation uses
// (reference/MultiwayReference.wl:141 findMatches) rather than a restatement of the join.
//
// Two conventions are asserted explicitly because they are the ones that decide what a match IS,
// and getting either wrong changes the event set rather than the speed:
//   - EDGE-INJECTIVE: two pattern edges may not take the same data edge.
//   - VERTEX-NON-INJECTIVE: distinct pattern variables MAY bind the same vertex.

#include <gtest/gtest.h>

#include "hgcommon/join_core.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

namespace {

using EdgeId = uint32_t;
using VertexId = uint32_t;
constexpr uint32_t kMaxEdges = 4;
constexpr uint32_t kMaxVars = 16;

using St = hgcommon::JoinState<kMaxEdges, kMaxVars, EdgeId, VertexId>;

struct Edge {
    std::vector<VertexId> v;
};

// The smallest Ctx that satisfies the join's requirements: candidates are every edge in the
// state. A real adapter narrows this with an index; narrowing may not change WHICH matches
// exist, which is what comparing against brute force checks.
struct ScanCtx {
    const std::vector<Edge>* edges;
    const std::vector<std::vector<uint8_t>>* lhs;
    const std::vector<uint8_t>* order;   // depth -> pattern index

    uint8_t num_lhs_edges() const { return static_cast<uint8_t>(lhs->size()); }
    uint8_t order_at(uint8_t k) const { return (*order)[k]; }
    const uint8_t* pattern_vars(uint8_t p) const { return (*lhs)[p].data(); }
    uint8_t pattern_arity(uint8_t p) const { return static_cast<uint8_t>((*lhs)[p].size()); }

    // This enumerator yields bare ids, so a candidate IS an id and the accessors are identities.
    EdgeId candidate_of(EdgeId e) const { return e; }
    EdgeId candidate_id(EdgeId e) const { return e; }
    const VertexId* edge_vertices(EdgeId e) const { return (*edges)[e].v.data(); }
    uint8_t edge_arity(EdgeId e) const { return static_cast<uint8_t>((*edges)[e].v.size()); }
    bool usable(EdgeId) const { return true; }
    bool aborted() const { return false; }

    template <typename F>
    void for_each_candidate(uint8_t, const St&, F&& f) const {
        for (EdgeId e = 0; e < edges->size(); ++e) f(e);
    }
};

// Independent enumeration: every ordered tuple of DISTINCT edges whose positional binding is
// consistent. Deliberately not written in terms of the join.
std::set<std::vector<EdgeId>> brute_force(const std::vector<Edge>& edges,
                                          const std::vector<std::vector<uint8_t>>& lhs) {
    std::set<std::vector<EdgeId>> out;
    std::vector<EdgeId> pick(lhs.size());
    std::vector<uint8_t> used(edges.size(), 0);

    auto rec = [&](auto& self, size_t pos) -> void {
        if (pos == lhs.size()) { out.insert(pick); return; }
        for (EdgeId e = 0; e < edges.size(); ++e) {
            if (used[e]) continue;                       // edge-injective
            if (edges[e].v.size() != lhs[pos].size()) continue;
            pick[pos] = e; used[e] = 1;
            // Consistency of the whole assignment so far, recomputed from scratch.
            bool ok = true;
            VertexId bind[kMaxVars];
            bool has[kMaxVars] = {};
            for (size_t q = 0; q <= pos && ok; ++q)
                for (size_t i = 0; i < lhs[q].size() && ok; ++i) {
                    const uint8_t var = lhs[q][i];
                    const VertexId val = edges[pick[q]].v[i];
                    if (has[var]) { if (bind[var] != val) ok = false; }
                    else          { bind[var] = val; has[var] = true; }
                }
            if (ok) self(self, pos + 1);
            used[e] = 0;
        }
    };
    rec(rec, 0);
    return out;
}

// Run the shared join and collect matches keyed by PATTERN position, so the result is
// comparable with brute force whatever order the join bound them in.
std::set<std::vector<EdgeId>> via_join(const std::vector<Edge>& edges,
                                       const std::vector<std::vector<uint8_t>>& lhs,
                                       const std::vector<uint8_t>& order) {
    ScanCtx ctx{&edges, &lhs, &order};
    St st; st.reset();
    std::set<std::vector<EdgeId>> out;
    hgcommon::join_dfs(ctx, st, [&](const St& s) {
        std::vector<EdgeId> by_pattern(lhs.size());
        for (uint8_t d = 0; d < s.depth; ++d) by_pattern[s.pattern[d]] = s.matched[d];
        out.insert(by_pattern);
    });
    return out;
}

std::vector<uint8_t> identity_order(size_t n) {
    std::vector<uint8_t> o(n);
    for (size_t i = 0; i < n; ++i) o[i] = static_cast<uint8_t>(i);
    return o;
}

}  // namespace

// The position rule, directly: the recursive join and the task-based scheduler
// (ParallelEvolutionEngine::execute_expand_task) both select with this, so it is pinned on its
// own rather than only through what the join happens to emit.
TEST(JoinCore, NextPositionIsTheFirstUNBOUNDOneInTheSchedule) {
    const std::vector<uint8_t> order{2, 0, 1};
    auto at = [&](uint8_t k) { return order[k]; };

    EXPECT_EQ(hgcommon::join_next_position(at, 3, 0b000), 2) << "nothing bound: schedule head";
    EXPECT_EQ(hgcommon::join_next_position(at, 3, 0b100), 0) << "2 bound: next in schedule";
    EXPECT_EQ(hgcommon::join_next_position(at, 3, 0b101), 1);

    // Seeded away from the schedule head: counting by depth would take order[1]=0 and leave
    // position 2 unbound forever. The rule takes 2, the first unbound one.
    EXPECT_EQ(hgcommon::join_next_position(at, 3, 0b010), 2) << "anchor at position 1";

    EXPECT_EQ(hgcommon::join_next_position(at, 3, 0b111), 0xFFu) << "all bound";
}

TEST(JoinCore, MatchesBruteForceOnAPath) {
    std::vector<Edge> edges{{{0,1}}, {{1,2}}, {{2,3}}, {{3,4}}};
    std::vector<std::vector<uint8_t>> lhs{{0,1},{1,2}};
    EXPECT_EQ(via_join(edges, lhs, identity_order(lhs.size())), brute_force(edges, lhs));
}

TEST(JoinCore, MatchesBruteForceOnACycleWithHighSymmetry) {
    std::vector<Edge> edges{{{0,1}}, {{1,2}}, {{2,3}}, {{3,0}}};
    std::vector<std::vector<uint8_t>> lhs{{0,1},{1,2}};
    EXPECT_EQ(via_join(edges, lhs, identity_order(lhs.size())), brute_force(edges, lhs));
}

TEST(JoinCore, MixedArityAndThreeEdgePattern) {
    std::vector<Edge> edges{{{0,1}}, {{1,2,3}}, {{3,4}}, {{4,5,6}}, {{2,3}}};
    std::vector<std::vector<uint8_t>> lhs{{0,1},{1,2,3},{3,4}};
    EXPECT_EQ(via_join(edges, lhs, identity_order(lhs.size())), brute_force(edges, lhs));
}

// A self-loop is a legal binding target for {x,y}: distinct variables may take one vertex.
TEST(JoinCore, VertexBindingIsNotInjective) {
    std::vector<Edge> edges{{{7,7}}, {{7,8}}};
    std::vector<std::vector<uint8_t>> lhs{{0,1}};
    auto got = via_join(edges, lhs, identity_order(lhs.size()));
    EXPECT_EQ(got, brute_force(edges, lhs));
    EXPECT_TRUE(got.count(std::vector<EdgeId>{0})) << "{{x,y}} must match the self-loop {7,7}";
}

// One data edge may not serve two pattern edges, even when the binding would be consistent.
TEST(JoinCore, EdgeBindingIsInjective) {
    std::vector<Edge> edges{{{1,1}}};
    std::vector<std::vector<uint8_t>> lhs{{0,1},{1,0}};   // both satisfiable by edge 0 alone
    auto got = via_join(edges, lhs, identity_order(lhs.size()));
    EXPECT_TRUE(got.empty()) << "one edge cannot fill two pattern positions";
    EXPECT_EQ(got, brute_force(edges, lhs));
}

// THE ORDER IS A SCHEDULE, NOT A SEMANTIC. The host indirects through match_order at match time
// and the device physically reorders its LHS at build time; both must yield the same match set.
// Every permutation of the binding order is checked to produce exactly the same matches.
TEST(JoinCore, EveryBindingOrderYieldsTheSameMatches) {
    std::vector<Edge> edges{{{0,1}}, {{1,2}}, {{2,3}}, {{3,0}}, {{0,2}}};
    std::vector<std::vector<uint8_t>> lhs{{0,1},{1,2},{2,3}};
    const auto expect = brute_force(edges, lhs);

    std::vector<uint8_t> order = identity_order(lhs.size());
    int perms = 0;
    do {
        EXPECT_EQ(via_join(edges, lhs, order), expect)
            << "binding order {" << int(order[0]) << "," << int(order[1]) << ","
            << int(order[2]) << "} changed the match set";
        ++perms;
    } while (std::next_permutation(order.begin(), order.end()));
    EXPECT_EQ(perms, 6);
}

// Delta matching is the same join with one position pinned: every emitted match must use the
// anchor edge at the anchor position, and together the anchors must cover every match that
// contains the edge at all.
TEST(JoinCore, SeedingAtAPositionIsTheSameJoinAnchored) {
    std::vector<Edge> edges{{{0,1}}, {{1,2}}, {{2,3}}, {{3,4}}};
    std::vector<std::vector<uint8_t>> lhs{{0,1},{1,2}};
    const auto order = identity_order(lhs.size());
    ScanCtx ctx{&edges, &lhs, &order};

    const EdgeId anchor = 1;
    std::set<std::vector<EdgeId>> anchored;
    for (uint8_t p = 0; p < lhs.size(); ++p) {
        St st;
        hgcommon::join_seed(ctx, st, anchor, p, [&](const St& s) {
            std::vector<EdgeId> by_pattern(lhs.size());
            for (uint8_t d = 0; d < s.depth; ++d) by_pattern[s.pattern[d]] = s.matched[d];
            EXPECT_EQ(by_pattern[p], anchor) << "seeded match does not use the anchor at p";
            anchored.insert(by_pattern);
        });
    }

    std::set<std::vector<EdgeId>> expect;
    for (const auto& m : brute_force(edges, lhs))
        if (std::find(m.begin(), m.end(), anchor) != m.end()) expect.insert(m);
    EXPECT_EQ(anchored, expect)
        << "anchoring at every position must find exactly the matches containing the edge";
}
