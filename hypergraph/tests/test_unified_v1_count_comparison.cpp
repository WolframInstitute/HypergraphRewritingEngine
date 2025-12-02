#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <memory>
#include <set>
#include <map>
#include <string>

// v1 includes
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/pattern_matching.hpp"

// v2 (unified) includes
#include "hypergraph/unified/arena.hpp"
#include "hypergraph/unified/types.hpp"
#include "hypergraph/unified/bitset.hpp"
#include "hypergraph/unified/unified_hypergraph.hpp"
#include "hypergraph/unified/pattern.hpp"
#include "hypergraph/unified/rewriter.hpp"
#include "hypergraph/unified/parallel_evolution.hpp"

using namespace hypergraph;
namespace v2 = hypergraph::unified;

// =============================================================================
// v1 vs Unified Count Comparison Tests
// =============================================================================
// These tests verify that the unified architecture produces the same state and event counts as v1.

class V1_Unified_CountComparisonTest : public ::testing::Test {
protected:
    struct EvolutionCounts {
        size_t num_states;
        size_t num_events;
        size_t num_causal;
        size_t num_branchial;
    };

    // Run v1 evolution and get counts
    EvolutionCounts run_v1(
        const std::vector<RewritingRule>& rules,
        const std::vector<std::vector<GlobalVertexId>>& initial,
        size_t steps
    ) {
        WolframEvolution evolution(steps, 1, true, false);  // single thread
        evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::CANONICALIZATION);
        for (const auto& rule : rules) {
            evolution.add_rule(rule);
        }
        evolution.evolve(initial);

        const auto& graph = evolution.get_multiway_graph();
        return {
            graph.num_states(),
            graph.num_events(),
            graph.get_causal_edge_count(),
            graph.get_branchial_edge_count()
        };
    }

    // Run unified evolution and get counts
    EvolutionCounts run_unified(
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), 4);

        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }

        engine.evolve(initial, steps);

        return {
            hg->num_canonical_states(),  // Use canonical count, not raw count
            engine.num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges()
        };
    }

    // Compare counts and print results
    void compare_and_verify(
        const std::string& test_name,
        const EvolutionCounts& v1_counts,
        const EvolutionCounts& unified_counts,
        size_t steps,
        bool strict = true
    ) {
        std::cout << test_name << " (step " << steps << "):\n";
        std::cout << "  v1:      states=" << v1_counts.num_states
                  << ", events=" << v1_counts.num_events
                  << ", causal=" << v1_counts.num_causal
                  << ", branchial=" << v1_counts.num_branchial << "\n";
        std::cout << "  unified: states=" << unified_counts.num_states
                  << ", events=" << unified_counts.num_events
                  << ", causal=" << unified_counts.num_causal
                  << ", branchial=" << unified_counts.num_branchial << "\n";

        if (strict) {
            EXPECT_EQ(v1_counts.num_states, unified_counts.num_states)
                << "State count mismatch at step " << steps;
            EXPECT_EQ(v1_counts.num_events, unified_counts.num_events)
                << "Event count mismatch at step " << steps;
        }
    }
};

TEST_F(V1_Unified_CountComparisonTest, SimpleRule_Step1) {
    // v1 rule: {{1,2}} -> {{1,2},{2,3}}
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    // v2.3 rule
    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}};

    auto v1_counts = run_v1({v1_rule}, v1_initial, 1);
    auto unified_counts = run_unified({v2_rule}, v2_initial, 1);

    compare_and_verify("SimpleRule", v1_counts, unified_counts, 1);
}

TEST_F(V1_Unified_CountComparisonTest, SimpleRule_Step2) {
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}};

    auto v1_counts = run_v1({v1_rule}, v1_initial, 2);
    auto unified_counts = run_unified({v2_rule}, v2_initial, 2);

    compare_and_verify("SimpleRule", v1_counts, unified_counts, 2);
}

TEST_F(V1_Unified_CountComparisonTest, SimpleRule_Step3) {
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}};

    auto v1_counts = run_v1({v1_rule}, v1_initial, 3);
    auto unified_counts = run_unified({v2_rule}, v2_initial, 3);

    compare_and_verify("SimpleRule", v1_counts, unified_counts, 3);
}

TEST_F(V1_Unified_CountComparisonTest, SimpleRule_Step4) {
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}};

    auto v1_counts = run_v1({v1_rule}, v1_initial, 4);
    auto unified_counts = run_unified({v2_rule}, v2_initial, 4);

    compare_and_verify("SimpleRule", v1_counts, unified_counts, 4);
}

TEST_F(V1_Unified_CountComparisonTest, TwoEdgeRule_Triangle_Step1) {
    // v1 rule: {{1,2},{2,3}} -> {{1,2},{2,3},{2,4}}
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(4)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}, {2, 3}, {3, 1}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}, {1, 2}, {2, 0}};

    auto v1_counts = run_v1({v1_rule}, v1_initial, 1);
    auto unified_counts = run_unified({v2_rule}, v2_initial, 1);

    compare_and_verify("TwoEdgeRule_Triangle", v1_counts, unified_counts, 1);
}

TEST_F(V1_Unified_CountComparisonTest, TwoEdgeRule_Triangle_Step2) {
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(4)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}, {2, 3}, {3, 1}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}, {1, 2}, {2, 0}};

    auto v1_counts = run_v1({v1_rule}, v1_initial, 2);
    auto unified_counts = run_unified({v2_rule}, v2_initial, 2);

    compare_and_verify("TwoEdgeRule_Triangle", v1_counts, unified_counts, 2);
}

// Debug test to verify canonical hashing - Step 3
TEST_F(V1_Unified_CountComparisonTest, DebugCanonicalHashes_Step3) {
    // Simple rule: {x,y} -> {x,y}, {y,z}
    // This is where the count mismatch first appears

    // --- v1 version ---
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    WolframEvolution v1_evolution(3, 1, true, false);
    v1_evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::CANONICALIZATION);
    v1_evolution.add_rule(v1_rule);

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}};
    v1_evolution.evolve(v1_initial);

    const auto& v1_graph = v1_evolution.get_multiway_graph();

    // --- unified version ---
    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    auto hg = std::make_unique<v2::UnifiedHypergraph>();
    v2::ParallelEvolutionEngine engine(hg.get(), 4);
    engine.add_rule(v2_rule);

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 3);

    // Print counts
    std::cout << "\n=== Step 3 comparison ===\n";
    std::cout << "v1 states: " << v1_graph.num_states() << ", events: " << v1_graph.num_events() << "\n";
    std::cout << "unified canonical states: " << hg->num_canonical_states()
              << ", raw states: " << hg->num_states()
              << ", events: " << engine.num_events() << "\n";

    // Print unified state details
    std::cout << "\nunified state canonical forms:\n";
    std::map<std::string, std::vector<uint32_t>> canonical_groups;
    for (uint32_t sid = 0; sid < hg->num_states(); ++sid) {
        const v2::SparseBitset& edges = hg->get_state_edges(sid);
        std::string canonical = hg->get_canonical_form_string(edges);
        canonical_groups[canonical].push_back(sid);

        // Also print raw edge content
        std::string raw = hg->get_raw_edges_string(edges);
        std::cout << "  State " << sid << ": raw=" << raw << " canonical=" << canonical << "\n";
    }

    std::cout << "\n  Unique canonical forms: " << canonical_groups.size() << "\n";

    // Show grouped states
    std::cout << "\n  States by canonical form:\n";
    for (const auto& [form, states] : canonical_groups) {
        std::cout << "    " << form << " -> states: ";
        for (auto s : states) std::cout << s << " ";
        if (states.size() > 1) std::cout << " [DUPLICATES!]";
        std::cout << "\n";
    }

    // Check for states with self-loops
    std::cout << "\n  Checking for self-loops...\n";
    bool found_self_loop = false;
    for (uint32_t sid = 0; sid < hg->num_states(); ++sid) {
        const v2::SparseBitset& edges = hg->get_state_edges(sid);
        std::string canonical = hg->get_canonical_form_string(edges);
        if (canonical.find("{0,0}") != std::string::npos) {
            found_self_loop = true;
            std::cout << "    State " << sid << " has self-loop in canonical form: " << canonical << "\n";
        }
    }
    if (!found_self_loop) {
        std::cout << "    No self-loops found\n";
    }

    EXPECT_EQ(v1_graph.num_states(), hg->num_canonical_states());
    EXPECT_EQ(v1_graph.num_events(), engine.num_events());
}

// Debug test for Step 4 event mismatch
TEST_F(V1_Unified_CountComparisonTest, DebugStep4Events) {
    // Simple rule: {x,y} -> {x,y}, {y,z}

    // --- v1 version ---
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    WolframEvolution v1_evolution(4, 1, true, false);
    v1_evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::CANONICALIZATION);
    v1_evolution.add_rule(v1_rule);

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}};
    v1_evolution.evolve(v1_initial);

    const auto& v1_graph = v1_evolution.get_multiway_graph();

    // --- unified version ---
    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    auto hg = std::make_unique<v2::UnifiedHypergraph>();
    v2::ParallelEvolutionEngine engine(hg.get(), 4);
    engine.add_rule(v2_rule);

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}};
    engine.evolve(initial, 4);

    // Print counts
    std::cout << "\n=== Step 4 Debug ===\n";
    std::cout << "v1 states: " << v1_graph.num_states() << ", events: " << v1_graph.num_events() << "\n";
    std::cout << "unified canonical states: " << hg->num_canonical_states()
              << ", raw states: " << hg->num_states()
              << ", events: " << engine.num_events() << "\n";

    // Print unified events
    std::cout << "\nunified events:\n";
    for (uint32_t eid = 0; eid < hg->num_events(); ++eid) {
        const v2::Event& ev = hg->get_event(eid);
        std::cout << "  Event " << eid << ": state " << ev.input_state << " -> " << ev.output_state << "\n";
    }

    // Print evolution stats
    const auto& stats = engine.stats();
    std::cout << "\nunified stats:\n";
    std::cout << "  states_created: " << stats.states_created.load() << "\n";
    std::cout << "  events_created: " << stats.events_created.load() << "\n";
    std::cout << "  matches_found: " << stats.matches_found.load() << "\n";
    std::cout << "  matches_forwarded: " << stats.matches_forwarded.load() << "\n";
    std::cout << "  matches_invalidated: " << stats.matches_invalidated.load() << "\n";
    std::cout << "  new_matches_discovered: " << stats.new_matches_discovered.load() << "\n";
    std::cout << "  full_pattern_matches: " << stats.full_pattern_matches.load() << "\n";

    EXPECT_EQ(v1_graph.num_states(), hg->num_canonical_states());
    EXPECT_EQ(v1_graph.num_events(), engine.num_events());
}

