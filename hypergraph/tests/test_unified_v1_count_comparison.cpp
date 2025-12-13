#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <memory>
#include <set>
#include <map>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <thread>

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
        evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::UNIQUENESS_TREE);
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
        // Disable event canonicalization to match v1's default behavior (canonicalize_events=false)
        hg->set_event_canonicalization_mode(v2::EventCanonicalizationMode::None);
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
    v1_evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::UNIQUENESS_TREE);
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
    hg->set_event_canonicalization_mode(v2::EventCanonicalizationMode::None);
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
    v1_evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::UNIQUENESS_TREE);
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
    hg->set_event_canonicalization_mode(v2::EventCanonicalizationMode::None);
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

// =============================================================================
// Scaling Comparison Tests - Compare v1 vs unified for larger configurations
// =============================================================================

TEST_F(V1_Unified_CountComparisonTest, ScalingComparison_SimpleRule) {
    std::cout << "\n========================================\n";
    std::cout << "SCALING COMPARISON: Simple Rule {x,y} -> {x,y}, {y,z}\n";
    std::cout << "========================================\n\n";

    // Pre-computed v1 results (v1 is too slow for steps >= 7)
    // Format: {steps, states, events, time_ms}
    struct V1Result { int steps; size_t states; size_t events; long long ms; };
    std::vector<V1Result> v1_precomputed = {
        {4,   17,    33,        8},
        {5,   37,   153,      139},
        {6,   85,   873,     4823},
        {7,  200,  5913,   261423},    // ~4.4 minutes
        {8,  492, 46233, 14414000},    // >4 hours (estimated, didn't complete)
    };

    // unified rule
    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    size_t hw_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << hw_threads << " threads\n";
    std::cout << "* = precomputed v1 value (too slow to run live)\n\n";

    std::cout << std::setw(6) << "Steps"
              << std::setw(12) << "v1_states"
              << std::setw(12) << "v2_states"
              << std::setw(12) << "v1_events"
              << std::setw(12) << "v2_events"
              << std::setw(14) << "v1_ms"
              << std::setw(12) << "v2_ms"
              << std::setw(14) << "speedup"
              << "\n";

    for (int steps = 4; steps <= 7; ++steps) {
        // Get precomputed v1 values
        size_t v1_states = 0, v1_events = 0;
        long long v1_ms = 0;
        for (const auto& pre : v1_precomputed) {
            if (pre.steps == steps) {
                v1_states = pre.states;
                v1_events = pre.events;
                v1_ms = pre.ms;
                break;
            }
        }

        // unified - always run
        auto v2_start = std::chrono::high_resolution_clock::now();
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_event_canonicalization_mode(v2::EventCanonicalizationMode::None);
        v2::ParallelEvolutionEngine engine(hg.get(), 0);  // use all threads
        engine.add_rule(v2_rule);
        engine.evolve({{0, 1}}, steps);
        auto v2_end = std::chrono::high_resolution_clock::now();
        auto v2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(v2_end - v2_start).count();

        double speedup = v1_ms > 0 ? static_cast<double>(v1_ms) / static_cast<double>(std::max(1LL, static_cast<long long>(v2_ms))) : 0;

        std::cout << std::setw(6) << steps
                  << std::setw(12) << v1_states
                  << std::setw(12) << hg->num_canonical_states()
                  << std::setw(12) << v1_events
                  << std::setw(12) << engine.num_events()
                  << std::setw(13) << v1_ms << "*"
                  << std::setw(12) << v2_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "\n";

        EXPECT_EQ(v1_states, hg->num_canonical_states());
        EXPECT_EQ(v1_events, engine.num_events());
    }
}

TEST_F(V1_Unified_CountComparisonTest, ScalingComparison_TwoEdgeRule) {
    std::cout << "\n========================================\n";
    std::cout << "SCALING COMPARISON: Two-Edge Rule {x,y},{y,z} -> {x,w},{w,y},{y,z}\n";
    std::cout << "========================================\n\n";

    // v1 rule
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(0), PatternVertex::variable(1)});
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(0), PatternVertex::variable(3)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(3), PatternVertex::variable(1)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    // unified rule
    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 2})
        .build();

    std::cout << std::setw(6) << "Steps"
              << std::setw(12) << "v1_states"
              << std::setw(12) << "v2_states"
              << std::setw(12) << "v1_events"
              << std::setw(12) << "v2_events"
              << std::setw(12) << "v1_ms"
              << std::setw(12) << "v2_ms"
              << std::setw(12) << "speedup"
              << "\n";

    size_t hw_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << hw_threads << " threads for both v1 and unified\n\n";

    for (int steps = 3; steps <= 7; ++steps) {
        // v1 - use hardware_concurrency for fair comparison
        auto v1_start = std::chrono::high_resolution_clock::now();
        WolframEvolution v1_evolution(steps, hw_threads, true, false);
        v1_evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::UNIQUENESS_TREE);
        v1_evolution.add_rule(v1_rule);
        v1_evolution.evolve({{0, 1}, {1, 2}});
        auto v1_end = std::chrono::high_resolution_clock::now();
        auto v1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(v1_end - v1_start).count();
        const auto& v1_graph = v1_evolution.get_multiway_graph();

        // unified
        auto v2_start = std::chrono::high_resolution_clock::now();
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_event_canonicalization_mode(v2::EventCanonicalizationMode::None);
        v2::ParallelEvolutionEngine engine(hg.get(), 0);  // use all threads
        engine.add_rule(v2_rule);
        engine.evolve({{0, 1}, {1, 2}}, steps);
        auto v2_end = std::chrono::high_resolution_clock::now();
        auto v2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(v2_end - v2_start).count();

        double speedup = v1_ms > 0 ? static_cast<double>(v1_ms) / static_cast<double>(std::max(1LL, static_cast<long long>(v2_ms))) : 0;

        std::cout << std::setw(6) << steps
                  << std::setw(12) << v1_graph.num_states()
                  << std::setw(12) << hg->num_canonical_states()
                  << std::setw(12) << v1_graph.num_events()
                  << std::setw(12) << engine.num_events()
                  << std::setw(12) << v1_ms
                  << std::setw(12) << v2_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "\n";

        EXPECT_EQ(v1_graph.num_states(), hg->num_canonical_states());
        EXPECT_EQ(v1_graph.num_events(), engine.num_events());
    }
}

TEST_F(V1_Unified_CountComparisonTest, ScalingComparison_TwoEdgeRule2) {
    std::cout << "\n========================================\n";
    std::cout << "SCALING COMPARISON: Two-Edge Rule {x,y},{y,z} -> {x,w},{w,y},{y,z}\n";
    std::cout << "========================================\n\n";

    // v1 rule
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(0), PatternVertex::variable(1)});
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(0), PatternVertex::variable(3)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(3), PatternVertex::variable(1)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    // unified rule
    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 3})
        .rhs({3, 1})
        .rhs({1, 2})
        .build();

    std::cout << std::setw(6) << "Steps"
              << std::setw(12) << "v1_states"
              << std::setw(12) << "v2_states"
              << std::setw(12) << "v1_events"
              << std::setw(12) << "v2_events"
              << std::setw(12) << "v1_ms"
              << std::setw(12) << "v2_ms"
              << std::setw(12) << "speedup"
              << "\n";

    size_t hw_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << hw_threads << " threads for both v1 and unified\n\n";

    for (int steps = 3; steps <= 6; ++steps) {
        // v1 - use hardware_concurrency for fair comparison
        auto v1_start = std::chrono::high_resolution_clock::now();
        WolframEvolution v1_evolution(steps, hw_threads, true, false);
        v1_evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::UNIQUENESS_TREE);
        v1_evolution.add_rule(v1_rule);
        v1_evolution.evolve({{0, 0}, {0, 0}});
        auto v1_end = std::chrono::high_resolution_clock::now();
        auto v1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(v1_end - v1_start).count();
        const auto& v1_graph = v1_evolution.get_multiway_graph();

        // unified
        auto v2_start = std::chrono::high_resolution_clock::now();
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_event_canonicalization_mode(v2::EventCanonicalizationMode::None);
        v2::ParallelEvolutionEngine engine(hg.get(), 0);  // use all threads
        engine.add_rule(v2_rule);
        engine.evolve({{0, 0}, {0, 0}}, steps);
        auto v2_end = std::chrono::high_resolution_clock::now();
        auto v2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(v2_end - v2_start).count();

        double speedup = v1_ms > 0 ? static_cast<double>(v1_ms) / static_cast<double>(std::max(1LL, static_cast<long long>(v2_ms))) : 0;

        std::cout << std::setw(6) << steps
                  << std::setw(12) << v1_graph.num_states()
                  << std::setw(12) << hg->num_canonical_states()
                  << std::setw(12) << v1_graph.num_events()
                  << std::setw(12) << engine.num_events()
                  << std::setw(12) << v1_ms
                  << std::setw(12) << v2_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "\n";

        EXPECT_EQ(v1_graph.num_states(), hg->num_canonical_states());
        EXPECT_EQ(v1_graph.num_events(), engine.num_events());
    }
}

TEST_F(V1_Unified_CountComparisonTest, ScalingComparison_LargerInitialState) {
    std::cout << "\n========================================\n";
    std::cout << "SCALING COMPARISON: 3-Edge Triangle Initial State\n";
    std::cout << "Rule: {x,y} -> {x,y}, {y,z}\n";
    std::cout << "========================================\n\n";

    // v1 rule
    PatternHypergraph v1_lhs, v1_rhs;
    v1_lhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    RewritingRule v1_rule(v1_lhs, v1_rhs);

    // unified rule
    v2::RewriteRule v2_rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    // 3-edge triangle initial state
    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0, 1}, {1, 2}, {2, 0}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}, {1, 2}, {2, 0}};

    std::cout << std::setw(6) << "Steps"
              << std::setw(12) << "v1_states"
              << std::setw(12) << "v2_states"
              << std::setw(12) << "v1_events"
              << std::setw(12) << "v2_events"
              << std::setw(12) << "v1_ms"
              << std::setw(12) << "v2_ms"
              << std::setw(12) << "speedup"
              << "\n";

    size_t hw_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << hw_threads << " threads for both v1 and unified\n\n";

    for (int steps = 3; steps <= 6; ++steps) {
        // v1 - use hardware_concurrency for fair comparison
        auto v1_start = std::chrono::high_resolution_clock::now();
        WolframEvolution v1_evolution(steps, hw_threads, true, false);
        v1_evolution.get_multiway_graph().set_hash_strategy_type(HashStrategyType::UNIQUENESS_TREE);
        v1_evolution.add_rule(v1_rule);
        v1_evolution.evolve(v1_initial);
        auto v1_end = std::chrono::high_resolution_clock::now();
        auto v1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(v1_end - v1_start).count();
        const auto& v1_graph = v1_evolution.get_multiway_graph();

        // unified
        auto v2_start = std::chrono::high_resolution_clock::now();
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        hg->set_event_canonicalization_mode(v2::EventCanonicalizationMode::None);
        v2::ParallelEvolutionEngine engine(hg.get(), 0);  // use all threads
        engine.add_rule(v2_rule);
        engine.evolve(v2_initial, steps);
        auto v2_end = std::chrono::high_resolution_clock::now();
        auto v2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(v2_end - v2_start).count();

        double speedup = v1_ms > 0 ? static_cast<double>(v1_ms) / static_cast<double>(std::max(1LL, static_cast<long long>(v2_ms))) : 0;

        std::cout << std::setw(6) << steps
                  << std::setw(12) << v1_graph.num_states()
                  << std::setw(12) << hg->num_canonical_states()
                  << std::setw(12) << v1_graph.num_events()
                  << std::setw(12) << engine.num_events()
                  << std::setw(12) << v1_ms
                  << std::setw(12) << v2_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "\n";

        EXPECT_EQ(v1_graph.num_states(), hg->num_canonical_states());
        EXPECT_EQ(v1_graph.num_events(), engine.num_events());
    }
}

