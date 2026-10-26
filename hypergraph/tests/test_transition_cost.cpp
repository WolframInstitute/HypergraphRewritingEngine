#include <gtest/gtest.h>
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/transition_cost_core.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <cstring>
#include <tuple>
#include <vector>

using namespace hypergraph;

// "TransitionCost" and "Search" -> "BestFirst" (docs/TRANSITION_COST.md).

namespace {

using Edges = std::vector<std::vector<uint32_t>>;

double residual(const Edges& parent, const Edges& child, uint32_t field, double dt) {
    hgcommon::TcEdgeLists p, c;
    for (const auto& e : parent) p.add(e.data(), static_cast<uint32_t>(e.size()));
    for (const auto& e : child) c.add(e.data(), static_cast<uint32_t>(e.size()));
    hgcommon::TransitionCostSpec spec;
    spec.field = field;
    spec.dt = dt;
    return hgcommon::tc_transition_cost(p.view(), c.view(), spec);
}

std::vector<double> field_of(const Edges& edges, uint32_t field) {
    hgcommon::TcEdgeLists l;
    for (const auto& e : edges) l.add(e.data(), static_cast<uint32_t>(e.size()));
    hgcommon::TransitionCostSpec spec;
    spec.field = field;
    hgcommon::TcFieldOf f;
    hgcommon::tc_field_of(l.view(), spec, f);
    return std::vector<double>(f.field.f, f.field.f + f.field.n);
}

const Edges kCycle = {{1, 2}, {2, 3}, {3, 4}, {4, 1}};

Edges with(Edges base, const Edges& more) {
    base.insert(base.end(), more.begin(), more.end());
    return base;
}

}  // namespace

// The residual on a 4-cycle, by hand. Two loops at vertex 1 give f_P = (2, 0, 0, 0) and
// (A - D) f_P = (-4, 2, 0, 2); with dt = 1/4 one Euler step predicts (1, 1/2, 0, 1/2). A child
// with one loop at 1 and one at 2 has f_C = (1, 1, 0, 0): residuals (0, 1/2, 0, -1/2), cost 1/2.
TEST(TransitionCost, HeatResidualOnFourCycleMatchesHandValue) {
    const Edges parent = with(kCycle, {{1, 1}, {1, 1}});
    const Edges child = with(kCycle, {{1, 1}, {2, 2}});
    EXPECT_DOUBLE_EQ(residual(parent, child, hgcommon::TC_FIELD_LOOP_COUNT, 0.25), 0.5);
    // dt = 0 compares the fields: (1, -1, 0, 0), cost 2.
    EXPECT_DOUBLE_EQ(residual(parent, child, hgcommon::TC_FIELD_LOOP_COUNT, 0.0), 2.0);
    // The degree field is constant 2 on the cycle, so (A - D) f_P = 0; a chord 1-3 gives
    // f_C = (3, 2, 3, 2): residuals (1, 0, 1, 0), cost 2 at any dt.
    const Edges chord = with(kCycle, {{1, 3}});
    EXPECT_DOUBLE_EQ(residual(kCycle, chord, hgcommon::TC_FIELD_DEGREE, 0.7), 2.0);
    // A deleted or created vertex has no residual: a pendant vertex 5 adds nothing.
    const Edges pendant = with(kCycle, {{4, 5}});
    EXPECT_DOUBLE_EQ(residual(kCycle, pendant, hgcommon::TC_FIELD_DEGREE, 0.0), 1.0);
}

TEST(TransitionCost, FieldsCountTheirIncidences) {
    // Vertices 1, 2, 3, 4 in id order.
    const Edges e = {{1, 2}, {1, 2}, {2, 3}, {3, 4}, {4, 1}, {3, 3}, {3, 3, 3}};
    EXPECT_EQ(field_of(e, hgcommon::TC_FIELD_DEGREE), (std::vector<double>{2, 2, 2, 2}));
    EXPECT_EQ(field_of(e, hgcommon::TC_FIELD_EDGE_MULTIPLICITY),
              (std::vector<double>{3, 3, 2, 2}));
    EXPECT_EQ(field_of(e, hgcommon::TC_FIELD_LOOP_COUNT), (std::vector<double>{0, 0, 2, 0}));
    // The 4-cycle's Ollivier-Ricci curvature, which "VertexCurvature" reports: the lazy walk puts
    // 1/2 at x and 1/4 at each neighbour, W1 between two neighbours is 1/2, so k_v = 1/2.
    for (double k : field_of(kCycle, hgcommon::TC_FIELD_CURVATURE)) EXPECT_NEAR(k, 0.5, 1e-12);
}

namespace {

// Moves a loop along an edge: {{x,x},{x,y}} -> {{x,y},{y,y}}.
RewriteRule loop_move() {
    return make_rule(0).lhs({0, 0}).lhs({0, 1}).rhs({0, 1}).rhs({1, 1}).build();
}

std::vector<std::vector<VertexId>> edges_of(const Hypergraph& hg, StateId s) {
    std::vector<std::vector<VertexId>> out;
    hg.get_state_edges(s).for_each([&](EdgeId e) {
        const Edge& x = hg.get_edge(e);
        out.emplace_back(x.vertices, x.vertices + x.arity);
    });
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

// The engine's transition cost reads the parent and the child through the rewrite's vertex
// correspondence: the loop move on the 4-cycle costs the hand value 1/2. The two loops at 1 also
// match {x,y} as {1,1}; that child equals the parent, f_C = (2, 0, 0, 0), and costs
// 1 + 1/4 + 1/4. Under None each of the four transitions makes a state; under Full the two
// transitions to one class make one state, and the two back to the initial class make none.
TEST(TransitionCost, EngineCostsTheLoopMoveAtTheHandValue) {
    for (auto mode : {StateCanonicalizationMode::None, StateCanonicalizationMode::Full}) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(mode);
        ParallelEvolutionEngine e(&hg, 2);
        e.add_rule(loop_move());
        hgcommon::TransitionCostSpec spec;
        spec.field = hgcommon::TC_FIELD_LOOP_COUNT;
        spec.dt = 0.25;
        e.set_transition_cost(spec);
        e.set_best_first(true, 10);
        e.evolve(std::vector<std::vector<std::vector<VertexId>>>{
                     {{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 1}, {1, 1}}},
                 1);
        std::map<uint32_t, std::multiset<double>> by_step;
        for (const auto& [sid, cost] : e.path_costs())
            by_step[hg.get_state(sid).step].insert(cost);
        const std::map<uint32_t, std::multiset<double>> want =
            mode == StateCanonicalizationMode::None
                ? std::map<uint32_t, std::multiset<double>>{{0, {0.0}}, {1, {0.5, 0.5, 1.5, 1.5}}}
                : std::map<uint32_t, std::multiset<double>>{{0, {0.0}}, {1, {0.5}}};
        EXPECT_EQ(by_step, want);
    }
}

namespace {

// A constructed trap for the step cap. From {{x}}: {{x}} -> {{x,x}} costs 1 and its only
// continuation {{x,x}} -> {{x,x,x,x}} costs 10; {{x}} -> {{x,x,x}} costs 2 and its continuation
// {{x,x,x}} -> {{x,x,x,x,x}} costs 0. The cost is a hook on the child's largest arity.
std::vector<RewriteRule> trap_rules() {
    return {make_rule(0).lhs({0}).rhs({0, 0}).build(),
            make_rule(1).lhs({0}).rhs({0, 0, 0}).build(),
            make_rule(2).lhs({0, 0}).rhs({0, 0, 0, 0}).build(),
            make_rule(3).lhs({0, 0, 0}).rhs({0, 0, 0, 0, 0}).build()};
}

double trap_cost(const hgcommon::TcEdges&, const hgcommon::TcEdges& child) {
    uint32_t a = 0;
    for (uint32_t i = 0; i < child.count(); ++i) a = std::max(a, child.arity(i));
    switch (a) {
        case 2: return 1.0;
        case 3: return 2.0;
        case 4: return 10.0;
        default: return 0.0;
    }
}

// The largest arity of each state at step 2, with its path cost (best-first) or 0 (the cap).
std::multiset<std::pair<uint32_t, double>> step_two(const Hypergraph& hg,
                                                    const ParallelEvolutionEngine& e) {
    std::multiset<std::pair<uint32_t, double>> out;
    for (StateId s = 0; s < hg.num_states(); ++s) {
        if (hg.get_state(s).step != 2) continue;
        uint32_t a = 0;
        for (const auto& edge : edges_of(hg, s)) a = std::max<uint32_t>(a, edge.size());
        const auto it = e.path_costs().find(s);
        out.insert({a, it == e.path_costs().end() ? 0.0 : it->second});
    }
    return out;
}

}  // namespace

TEST(BestFirst, FindsALowerTotalCostThanTheStepCap) {
    // The step cap keeps the cheaper first step and reaches {{x,x,x,x}} at a total of 1 + 10.
    {
        Hypergraph hg;
        ParallelEvolutionEngine e(&hg, 4);
        for (const auto& r : trap_rules()) e.add_rule(r);
        e.set_max_states_per_step(1);
        e.set_transition_cost(hgcommon::TransitionCostSpec{}, trap_cost);
        e.evolve(std::vector<std::vector<std::vector<VertexId>>>{{{1}}}, 2);
        const auto two = step_two(hg, e);
        ASSERT_EQ(two.size(), 1u);
        EXPECT_EQ(two.begin()->first, 4u);   // path cost 1 + 10 = 11
    }
    // Best-first takes {{x}} -> {{x,x}} (1), {{x}} -> {{x,x,x}} (2), then {{x,x,x,x,x}} at a
    // path cost of 2 before {{x,x,x,x}} at 11. With a budget of 3 it stops there.
    for (size_t threads : {size_t(1), size_t(4)}) {
        Hypergraph hg;
        ParallelEvolutionEngine e(&hg, threads);
        for (const auto& r : trap_rules()) e.add_rule(r);
        e.set_transition_cost(hgcommon::TransitionCostSpec{}, trap_cost);
        e.set_best_first(true, 3);
        e.evolve(std::vector<std::vector<std::vector<VertexId>>>{{{1}}}, 2);
        EXPECT_EQ(step_two(hg, e), (std::multiset<std::pair<uint32_t, double>>{{5u, 2.0}}));
        EXPECT_EQ(e.path_costs().size(), 4u);
    }
    {
        Hypergraph hg;
        ParallelEvolutionEngine e(&hg, 4);
        for (const auto& r : trap_rules()) e.add_rule(r);
        e.set_transition_cost(hgcommon::TransitionCostSpec{}, trap_cost);
        e.set_best_first(true, 100);
        e.evolve(std::vector<std::vector<std::vector<VertexId>>>{{{1}}}, 2);
        EXPECT_EQ(step_two(hg, e),
                  (std::multiset<std::pair<uint32_t, double>>{{4u, 11.0}, {5u, 2.0}}));
    }
}

namespace {

// Each created state's (step, canonical hash, path cost), sorted: the search's result up to
// the run-local ids.
std::vector<std::tuple<uint32_t, uint64_t, double>> search_result(StateCanonicalizationMode mode,
                                                                  size_t threads, uint32_t field,
                                                                  size_t budget) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(mode);
    ParallelEvolutionEngine e(&hg, threads);
    e.add_rule(make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build());
    e.add_rule(make_rule(1).lhs({0, 1}).rhs({0, 1}).rhs({1, 1}).build());
    hgcommon::TransitionCostSpec spec;
    spec.field = field;
    spec.dt = 0.3;
    e.set_transition_cost(spec);
    e.set_best_first(true, budget);
    e.evolve(std::vector<std::vector<std::vector<VertexId>>>{
                 {{1, 2}, {2, 3}, {3, 4}, {4, 1}}},
             6);
    std::vector<std::tuple<uint32_t, uint64_t, double>> out;
    for (const auto& [sid, cost] : e.path_costs())
        out.emplace_back(hg.get_state(sid).step,
                         IRCanonicalizer{}.compute_canonical_hash(edges_of(hg, sid)), cost);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

// The states a search creates and their path costs depend on the inputs, not the thread count.
TEST(BestFirst, ResultDoesNotDependOnThreadCount) {
    for (auto mode : {StateCanonicalizationMode::None, StateCanonicalizationMode::Full}) {
        for (uint32_t field : {hgcommon::TC_FIELD_DEGREE, hgcommon::TC_FIELD_LOOP_COUNT,
                               hgcommon::TC_FIELD_CURVATURE}) {
            const auto one = search_result(mode, 1, field, 60);
            ASSERT_EQ(one.size(), 61u);
            for (size_t threads : {size_t(4), size_t(16)})
                EXPECT_EQ(search_result(mode, threads, field, 60), one)
                    << "field " << field << " at " << threads << " workers";
        }
    }
}

// The search creates states one at a time, so the raw ids are the creation order. Under None
// each state has one history: its path cost is the sum of the residuals along it, recomputed
// here from the stored states, and the path costs never decrease in creation order.
TEST(BestFirst, PathCostIsTheSumAlongTheHistoryInCreationOrder) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::None);
    ParallelEvolutionEngine e(&hg, 4);
    e.add_rule(make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build());
    e.add_rule(make_rule(1).lhs({0, 1}).rhs({0, 1}).rhs({1, 1}).build());
    hgcommon::TransitionCostSpec spec;
    spec.field = hgcommon::TC_FIELD_DEGREE;
    spec.dt = 0.3;
    e.set_transition_cost(spec);
    e.set_best_first(true, 400);
    e.evolve(std::vector<std::vector<std::vector<VertexId>>>{{{1, 2}, {2, 3}, {3, 4}, {4, 1}}},
             5);
    ASSERT_EQ(e.path_costs().size(), 401u);
    double last = 0.0;
    for (StateId s = 0; s < hg.num_states(); ++s) {
        const auto it = e.path_costs().find(s);
        ASSERT_NE(it, e.path_costs().end()) << s;
        EXPECT_GE(it->second, last) << s;
        last = it->second;
        const StateId parent = hg.get_state(s).parent_state;
        if (parent == INVALID_ID) {
            EXPECT_EQ(it->second, 0.0);
            continue;
        }
        const double step = residual(edges_of(hg, parent), edges_of(hg, s),
                                     hgcommon::TC_FIELD_DEGREE, 0.3);
        EXPECT_NEAR(it->second, e.path_costs().at(parent) + step, 1e-9) << s;
    }
}

// Under the step cap a successor's cost is the lowest cost of a transition into it. From the
// path {{1,2},{2,3}}, deleting either edge makes one class under Full, at costs 2 ({{2,3}}) and
// 1 ({{1,2}}): the class costs 1 and ranks before the doubled edge at 1.5, whichever of its two
// transitions has the lower rank under the seed.
TEST(TransitionCost, StepCapCostsASuccessorByItsCheapestTransition) {
    const auto cost = [](const hgcommon::TcEdges&, const hgcommon::TcEdges& child) {
        if (child.count() != 1) return 1.5;
        return static_cast<double>(std::min(child.at(0, 0), child.at(0, 1)));
    };
    for (uint64_t seed = 1; seed <= 16; ++seed) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, 4);
        e.add_rule(make_rule(0).lhs({0, 1}).build());
        e.add_rule(make_rule(1).lhs({0, 1}).rhs({0, 1}).rhs({0, 1}).build());
        e.set_max_states_per_step(1);
        e.set_random_seed(seed);
        e.set_transition_cost(hgcommon::TransitionCostSpec{}, cost);
        e.evolve(std::vector<std::vector<std::vector<VertexId>>>{{{1, 2}, {2, 3}}}, 1);
        size_t one_edge = 0, others = 0;
        for (StateId s = 0; s < hg.num_states(); ++s) {
            if (hg.get_state(s).step != 1) continue;
            (edges_of(hg, s).size() == 1 ? one_edge : others) += 1;
        }
        EXPECT_EQ(one_edge, 2u) << "seed " << seed;   // both transitions into the kept class
        EXPECT_EQ(others, 0u) << "seed " << seed;
    }
}

namespace {

// Each created state's (step, canonical hash, path cost), sorted, for a search with `hook`.
std::vector<std::tuple<uint32_t, uint64_t, double>> hooked(
    const std::vector<RewriteRule>& rules, const std::vector<std::vector<VertexId>>& init,
    size_t steps, size_t budget, size_t threads, hgcommon::TransitionCostHook hook,
    StateCanonicalizationMode mode = StateCanonicalizationMode::None) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(mode);
    ParallelEvolutionEngine e(&hg, threads);
    for (const auto& r : rules) e.add_rule(r);
    e.set_transition_cost(hgcommon::TransitionCostSpec{}, std::move(hook));
    e.set_best_first(true, budget);
    e.evolve(std::vector<std::vector<std::vector<VertexId>>>{init}, steps);
    std::vector<std::tuple<uint32_t, uint64_t, double>> out;
    for (const auto& [sid, cost] : e.path_costs())
        out.emplace_back(hg.get_state(sid).step,
                         IRCanonicalizer{}.compute_canonical_hash(edges_of(hg, sid)), cost);
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        // NaN-free order: step, hash, then the cost's bits.
        uint64_t x = 0, y = 0;
        std::memcpy(&x, &std::get<2>(a), sizeof x);
        std::memcpy(&y, &std::get<2>(b), sizeof y);
        return std::tie(std::get<0>(a), std::get<1>(a), x) <
               std::tie(std::get<0>(b), std::get<1>(b), y);
    });
    return out;
}

const std::vector<RewriteRule> kGrow = {
    make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build(),
    make_rule(1).lhs({0, 1}).rhs({0, 1}).rhs({1, 1}).build()};
const std::vector<std::vector<VertexId>> kSquare = {{1, 2}, {2, 3}, {3, 4}, {4, 1}};

}  // namespace

// A budget of 0 creates no state; a state with no match ends the search; a rewrite to the empty
// state costs the residuals of the deleted vertices.
TEST(BestFirst, BudgetZeroNoMatchAndTheEmptyState) {
    auto zero = [](const hgcommon::TcEdges&, const hgcommon::TcEdges&) { return 0.0; };
    EXPECT_EQ(hooked(kGrow, kSquare, 4, 0, 4, zero).size(), 1u);
    // {{x,y,z}} matches nothing in a graph of pairs.
    EXPECT_EQ(hooked({make_rule(0).lhs({0, 1, 2}).rhs({0, 1}).build()}, kSquare, 3, 10, 4, zero)
                  .size(),
              1u);
    // {{x}} -> {} from {{1}}: vertex 1 has degree 0 and loop count 0, so every residual is 0.
    Hypergraph hg;
    ParallelEvolutionEngine e(&hg, 2);
    e.add_rule(make_rule(0).lhs({0}).build());
    hgcommon::TransitionCostSpec spec;
    spec.field = hgcommon::TC_FIELD_DEGREE;
    spec.dt = 0.5;
    e.set_transition_cost(spec);
    e.set_best_first(true, 5);
    e.evolve(std::vector<std::vector<std::vector<VertexId>>>{{{1}}}, 3);
    ASSERT_EQ(e.path_costs().size(), 2u);
    for (const auto& [sid, cost] : e.path_costs()) EXPECT_EQ(cost, 0.0);
    // The 4-cycle with a chord {1,3} deleted: f_P = (3, 2, 3, 2), (A - D) f_P = (-2, 2, -2, 2);
    // at dt = 1/2 with f_C = (2, 2, 2, 2) the residuals are (0, -1, 0, -1): 2.
    EXPECT_DOUBLE_EQ(residual({{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 3}}, kCycle,
                              hgcommon::TC_FIELD_DEGREE, 0.5),
                     2.0);
}

// Every cost equal: the rank alone orders the frontier, and the search is the same at 1, 4 and
// 16 workers. Negative and non-finite costs order the frontier as numbers (NaN as +Infinity).
TEST(BestFirst, EqualNegativeAndNonFiniteCosts) {
    auto equal = [](const hgcommon::TcEdges&, const hgcommon::TcEdges&) { return 1.0; };
    auto negative = [](const hgcommon::TcEdges&, const hgcommon::TcEdges& c) {
        return -static_cast<double>(c.count());
    };
    auto nonfinite = [](const hgcommon::TcEdges&, const hgcommon::TcEdges& c) {
        return c.count() % 3 == 0 ? std::nan("") : c.count() % 3 == 1 ? HUGE_VAL : 1.0;
    };
    for (auto mode : {StateCanonicalizationMode::None, StateCanonicalizationMode::Full}) {
        for (const hgcommon::TransitionCostHook& h :
             {hgcommon::TransitionCostHook(equal), hgcommon::TransitionCostHook(negative),
              hgcommon::TransitionCostHook(nonfinite)}) {
            const auto one = hooked(kGrow, kSquare, 5, 40, 1, h, mode);
            ASSERT_EQ(one.size(), 41u);
            for (size_t threads : {size_t(4), size_t(16)})
                EXPECT_EQ(hooked(kGrow, kSquare, 5, 40, threads, h, mode), one);
        }
    }
    // With costs -(edges), the search goes deep first: a state at step 5 is created.
    const auto deep = hooked(kGrow, kSquare, 5, 40, 4, negative);
    EXPECT_EQ(std::get<0>(deep.back()), 5u);
    // NaN costs +Infinity: the states created first all have finite path costs.
    for (const auto& [step, hash, cost] : hooked(kGrow, kSquare, 2, 3, 4, nonfinite))
        EXPECT_FALSE(std::isnan(cost));
}

// Self-loops and multi-edges in the fields: {1,1} counts once as a loop and not in the edge
// multiplicity; a doubled {1,2} counts twice in the edge multiplicity and once in the degree.
TEST(TransitionCost, SelfLoopsAndMultiEdgesInEveryField) {
    const Edges single = {{1, 1}};
    EXPECT_EQ(field_of(single, hgcommon::TC_FIELD_LOOP_COUNT), (std::vector<double>{1}));
    EXPECT_EQ(field_of(single, hgcommon::TC_FIELD_DEGREE), (std::vector<double>{0}));
    EXPECT_EQ(field_of(single, hgcommon::TC_FIELD_EDGE_MULTIPLICITY), (std::vector<double>{0}));
    // A lone vertex has no neighbour: k_v is NaN and its residual is left out.
    EXPECT_TRUE(std::isnan(field_of(single, hgcommon::TC_FIELD_CURVATURE)[0]));
    EXPECT_EQ(residual(single, {{1, 1}, {1, 1}}, hgcommon::TC_FIELD_CURVATURE, 0.1), 0.0);
    // Doubling {1,2}: the degree field does not change (cost 0 at dt = 0), the multiplicity
    // rises by 1 at both ends (cost 2).
    EXPECT_EQ(residual({{1, 2}}, {{1, 2}, {1, 2}}, hgcommon::TC_FIELD_DEGREE, 0.0), 0.0);
    EXPECT_EQ(residual({{1, 2}}, {{1, 2}, {1, 2}}, hgcommon::TC_FIELD_EDGE_MULTIPLICITY, 0.0),
              2.0);
}
