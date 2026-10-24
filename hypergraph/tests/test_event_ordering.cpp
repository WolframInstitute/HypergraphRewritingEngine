// "EventOrderingFunction": SetReplace's event order (hgcommon::event_order_compare) and the
// single history that applies, at each step, the state's one match first in that order
// (ParallelEvolutionEngine::set_event_ordering). The engine is checked against SetReplace's
// WolframModel on the recorded corpus reference/setreplace_order_cases.hpp.
#include <gtest/gtest.h>

#include "hgcommon/sampling_core.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "reference/setreplace_order_reader.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace hypergraph;

namespace {

using setreplace_order::Edges;
using SetReplaceCase = setreplace_order::Recorded;
using setreplace_order::read_cases;
using setreplace_order::same_up_to_relabel;

struct EngineRun {
    Edges events;
    Edges expressions;
    std::vector<std::pair<uint32_t, uint32_t>> causal;
};

// The case's rules from its initial state, one history: under "EventOrderingFunction" with its
// order and `steps` steps, or under "Generational".
EngineRun run_engine(const SetReplaceCase& k, size_t threads, bool generational, size_t steps) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::None);
    ParallelEvolutionEngine e(&hg, threads);
    for (size_t r = 0; r < k.rules.size(); ++r) {
        RuleBuilder b(static_cast<uint16_t>(r));
        for (const auto& edge : k.rules[r].first) b.lhs(edge);
        for (const auto& edge : k.rules[r].second) b.rhs(edge);
        e.add_rule(b.build());
    }
    if (generational) {
        // Whole generations up to the case's event limit.
        e.set_generational(true);
        e.set_max_events(k.max_events);
    } else {
        e.set_event_ordering(k.order);
    }
    std::vector<std::vector<VertexId>> init;
    for (const auto& edge : k.init) init.emplace_back(edge.begin(), edge.end());
    e.evolve(init, steps);
    EngineRun out;
    std::map<EventId, uint32_t> position;
    std::vector<EventId> ids;
    for (uint32_t ev = 0; ev < hg.num_published_events(); ++ev) {
        const Event& x = hg.get_event(ev);
        if (x.id == INVALID_ID || hg.is_genesis_event(ev)) continue;
        ids.push_back(ev);
    }
    std::sort(ids.begin(), ids.end());
    for (const EventId ev : ids) {
        const Event& x = hg.get_event(ev);
        std::vector<uint32_t> row{static_cast<uint32_t>(x.rule_index) + 1u};
        for (uint8_t i = 0; i < x.num_consumed; ++i) row.push_back(x.consumed_edges[i] + 1u);
        for (uint8_t i = 0; i < x.num_produced; ++i) row.push_back(x.produced_edges[i] + 1u);
        position[ev] = static_cast<uint32_t>(out.events.size()) + 1u;
        out.events.push_back(row);
    }
    for (EdgeId id = 0; id < hg.num_edges(); ++id) {
        const Edge& edge = hg.get_edge(id);
        out.expressions.emplace_back(edge.vertices, edge.vertices + edge.arity);
    }
    std::set<std::pair<uint32_t, uint32_t>> pairs;
    for (const CausalEdge& c : hg.causal_graph().get_causal_edges())
        pairs.emplace(position.at(c.producer), position.at(c.consumer));
    out.causal.assign(pairs.begin(), pairs.end());
    return out;
}

}  // namespace

// The order of two matches by each key: edge ids sorted ascending, sorted descending, in
// pattern order, and the rule; a prefix comes before a longer list; a reversed key flips the
// comparison.
TEST(EventOrdering, KeysCompareAsSetReplaceDoes) {
    using namespace hgcommon;
    const uint32_t a[] = {3, 1, 7}, b[] = {2, 5, 7}, p[] = {3, 1};
    auto order = [](std::initializer_list<uint32_t> codes) {
        uint32_t o = 0;
        for (uint32_t c : codes) o = event_order_append(o, c);
        return o;
    };
    // Sorted ascending: {1,3,7} < {2,5,7}.
    EXPECT_EQ(event_order_compare(order({kEventOrderOldestEdge}), a, 3, 0, b, 3, 0), -1);
    EXPECT_EQ(event_order_compare(order({kEventOrderLeastOldEdge}), a, 3, 0, b, 3, 0), 1);
    // Sorted descending: {7,3,1} < {7,5,2}.
    EXPECT_EQ(event_order_compare(order({kEventOrderLeastRecentEdge}), a, 3, 0, b, 3, 0), -1);
    EXPECT_EQ(event_order_compare(order({kEventOrderNewestEdge}), a, 3, 0, b, 3, 0), 1);
    // Pattern order: {3,1,7} > {2,5,7}; a prefix comes first.
    EXPECT_EQ(event_order_compare(order({kEventOrderRuleOrdering}), a, 3, 0, b, 3, 0), 1);
    EXPECT_EQ(event_order_compare(order({kEventOrderRuleOrdering}), p, 2, 0, a, 3, 0), -1);
    EXPECT_EQ(event_order_compare(order({kEventOrderReverseRuleOrdering}), p, 2, 0, a, 3, 0), 1);
    // Rule index, after a key that ties.
    EXPECT_EQ(event_order_compare(order({kEventOrderRuleOrdering, kEventOrderRuleIndex}), a, 3, 2,
                                  a, 3, 1),
              1);
    EXPECT_EQ(event_order_compare(order({kEventOrderRuleOrdering, kEventOrderReverseRuleIndex}),
                                  a, 3, 2, a, 3, 1),
              -1);
    EXPECT_EQ(event_order_compare(kStandardEventOrder, a, 3, 1, a, 3, 1), 0);
    EXPECT_EQ(kStandardEventOrder,
              order({kEventOrderLeastRecentEdge, kEventOrderRuleOrdering, kEventOrderRuleIndex}));
}

// Names map to codes; a key whose function is already in the order is left out; an order is
// total when it has a pattern-order key and, with two or more rules, a rule key.
TEST(EventOrdering, NamesRepeatsAndTotality) {
    using namespace hgcommon;
    EXPECT_EQ(event_order_code("OldestEdge"), kEventOrderOldestEdge);
    EXPECT_EQ(event_order_code("ReverseRuleIndex"), kEventOrderReverseRuleIndex);
    EXPECT_EQ(event_order_code("Random"), 0u);
    EXPECT_EQ(event_order_code("Any"), 0u);
    EXPECT_EQ(event_order_code("OldestEdg"), 0u);
    const uint32_t o = event_order_append(kEventOrderOldestEdge, kEventOrderLeastOldEdge);
    EXPECT_EQ(o, kEventOrderOldestEdge);
    EXPECT_FALSE(event_order_total(kEventOrderOldestEdge, 1));
    EXPECT_TRUE(event_order_total(kEventOrderRuleOrdering, 1));
    EXPECT_FALSE(event_order_total(kEventOrderRuleOrdering, 2));
    EXPECT_TRUE(event_order_total(event_order_append(kEventOrderRuleOrdering,
                                                     kEventOrderReverseRuleIndex),
                                  2));
    EXPECT_TRUE(event_order_total(kStandardEventOrder, 5));
}

// Every recorded case: the engine's events (rule, consumed edges, produced edges, numbered from 1
// in creation order), its edges up to a renaming of the vertices it creates, and its causal
// pairs equal SetReplace's, at 1 and 4 workers.
TEST(EventOrdering, MatchesSetReplaceWolframModel) {
    const auto cases = read_cases();
    ASSERT_EQ(cases.size(), setreplace_order::kCases);
    size_t events = 0;
    for (const SetReplaceCase& k : cases) {
        for (size_t threads : {size_t(1), size_t(4)}) {
            const EngineRun r = run_engine(k, threads, false, k.max_events);
            EXPECT_EQ(r.events, k.events) << k.name << " threads " << threads;
            EXPECT_TRUE(same_up_to_relabel(r.expressions, k.expressions))
                << k.name << " threads " << threads;
            EXPECT_EQ(r.causal, k.causal) << k.name << " threads " << threads;
        }
        events += k.events.size();
    }
    EXPECT_GT(events, 15000u);
}

// "Generational" takes each generation's matches in SetReplace's default order, and an edge
// made in generation g is newer than every edge before it, so the events in creation order are
// SetReplace's under its default order: the "Generational" run's events, whole generations up
// to the case's event limit, are a prefix of every recorded default-order case's.
TEST(EventOrdering, GenerationalIsSetReplacesDefaultOrder) {
    size_t checked = 0;
    for (const SetReplaceCase& k : read_cases()) {
        if (k.order != hgcommon::kStandardEventOrder) continue;
        const EngineRun r = run_engine(k, 1, true, k.max_events);
        ASSERT_LE(r.events.size(), k.events.size()) << k.name;
        EXPECT_TRUE(std::equal(r.events.begin(), r.events.end(), k.events.begin())) << k.name;
        if (!k.events.empty()) EXPECT_GT(r.events.size(), 0u) << k.name;
        ++checked;
    }
    EXPECT_GE(checked, 15u);
}
