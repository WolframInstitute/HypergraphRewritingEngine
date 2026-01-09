#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <memory>
#include <set>
#include <map>

// v1 includes
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/pattern_matching.hpp"

// v2 (unified) includes
#include "hypergraph/arena.hpp"
#include "hypergraph/types.hpp"
#include "hypergraph/bitset.hpp"
#include "hypergraph/unified_hypergraph.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/parallel_evolution.hpp"

using namespace hypergraph;
namespace v2 = hypergraph::unified;

// =============================================================================
// Match Forwarding Debug Test
// =============================================================================
// Focused test to debug match forwarding with:
// - 2 distinct initial states
// - 2 rules (2 edges LHS -> 3 edges RHS, 2 edges LHS -> 4 edges RHS)
// - Both rules match in both initial states
// - 2 steps of evolution
// - Compares v2 (unified) against v1 results

class MatchForwardingDebugTest : public ::testing::Test {
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
        size_t steps,
        size_t num_threads = 4
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), num_threads);

        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }

        engine.evolve(initial, steps);

        // Print stats
        const auto& stats = engine.stats();
        std::cout << "\n  Unified stats:\n";
        std::cout << "    states_created: " << stats.states_created.load() << "\n";
        std::cout << "    events_created: " << stats.events_created.load() << "\n";
        std::cout << "    matches_found: " << stats.matches_found.load() << "\n";
        std::cout << "    matches_forwarded: " << stats.matches_forwarded.load() << "\n";
        std::cout << "    matches_invalidated: " << stats.matches_invalidated.load() << "\n";
        std::cout << "    new_matches_discovered: " << stats.new_matches_discovered.load() << "\n";
        std::cout << "    full_pattern_matches: " << stats.full_pattern_matches.load() << "\n";
        std::cout << "    delta_pattern_matches: " << stats.delta_pattern_matches.load() << "\n";

        return {
            hg->num_canonical_states(),
            engine.num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges()
        };
    }

    void compare_and_verify(
        const std::string& test_name,
        const EvolutionCounts& v1_counts,
        const EvolutionCounts& unified_counts
    ) {
        std::cout << "\n" << test_name << " comparison:\n";
        std::cout << "  v1:      states=" << v1_counts.num_states
                  << ", events=" << v1_counts.num_events
                  << ", causal=" << v1_counts.num_causal
                  << ", branchial=" << v1_counts.num_branchial << "\n";
        std::cout << "  unified: states=" << unified_counts.num_states
                  << ", events=" << unified_counts.num_events
                  << ", causal=" << unified_counts.num_causal
                  << ", branchial=" << unified_counts.num_branchial << "\n";

        EXPECT_EQ(v1_counts.num_states, unified_counts.num_states)
            << "State count mismatch";
        EXPECT_EQ(v1_counts.num_events, unified_counts.num_events)
            << "Event count mismatch";
        EXPECT_EQ(v1_counts.num_causal, unified_counts.num_causal)
            << "Causal edge count mismatch";
        EXPECT_EQ(v1_counts.num_branchial, unified_counts.num_branchial)
            << "Branchial edge count mismatch";
    }
};

// =============================================================================
// Main Debug Test
// =============================================================================
// 2 initial states, 2 rules, 2 steps
// Rule 1: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {y,w}} (2 edges -> 3 edges, adds branch)
// Rule 2: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {x,z}, {z,w}} (2 edges -> 4 edges, adds triangle + branch)
// Initial 1: {{0,1}, {1,2}} (chain of 2 edges)
// Initial 2: {{0,1}, {1,2}, {2,0}} (triangle)

TEST_F(MatchForwardingDebugTest, TwoRules_TwoInitials_TwoSteps) {
    std::cout << "\n========================================\n";
    std::cout << "Match Forwarding Debug Test\n";
    std::cout << "========================================\n";

    // --- v1 rules ---
    // Rule 1: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {y,w}}
    PatternHypergraph v1_lhs1, v1_rhs1;
    v1_lhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_lhs1.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(4)});
    RewritingRule v1_rule1(v1_lhs1, v1_rhs1);

    // Rule 2: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {x,z}, {z,w}}
    PatternHypergraph v1_lhs2, v1_rhs2;
    v1_lhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_lhs2.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(3)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(3), PatternVertex::variable(4)});
    RewritingRule v1_rule2(v1_lhs2, v1_rhs2);

    // --- v2 rules ---
    // Rule 1: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {y,w}} (2->3)
    v2::RewriteRule v2_rule1 = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    // Rule 2: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {x,z}, {z,w}} (2->4)
    v2::RewriteRule v2_rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({0, 2})
        .rhs({2, 3})
        .build();

    // --- Initial states ---
    // v1: Initial 1 = {{1,2}, {2,3}}, Initial 2 = {{1,2}, {2,3}, {3,1}}
    std::vector<std::vector<GlobalVertexId>> v1_initial1 = {{1, 2}, {2, 3}};
    std::vector<std::vector<GlobalVertexId>> v1_initial2 = {{1, 2}, {2, 3}, {3, 1}};

    // v2: Initial 1 = {{0,1}, {1,2}}, Initial 2 = {{0,1}, {1,2}, {2,0}}
    std::vector<std::vector<v2::VertexId>> v2_initial1 = {{0, 1}, {1, 2}};
    std::vector<std::vector<v2::VertexId>> v2_initial2 = {{0, 1}, {1, 2}, {2, 0}};

    // Combine initial states for multi-initial evolution
    std::vector<std::vector<std::vector<GlobalVertexId>>> v1_combined_initial = {
        v1_initial1,  // First initial state (flattened)
        v1_initial2   // Second initial state
    };
    // For v1, we need to evolve with multiple initial states
    // The WolframEvolution evolve() takes a single flat initial state
    // Let's just test with each initial state separately first

    std::cout << "\n--- Testing Initial State 1 (chain): {{0,1}, {1,2}} ---\n";
    auto v1_counts1 = run_v1({v1_rule1, v1_rule2}, v1_initial1, 2);
    auto v2_counts1 = run_unified({v2_rule1, v2_rule2}, v2_initial1, 2);
    compare_and_verify("Initial1 (chain)", v1_counts1, v2_counts1);

    std::cout << "\n--- Testing Initial State 2 (triangle): {{0,1}, {1,2}, {2,0}} ---\n";
    auto v1_counts2 = run_v1({v1_rule1, v1_rule2}, v1_initial2, 2);
    auto v2_counts2 = run_unified({v2_rule1, v2_rule2}, v2_initial2, 2);
    compare_and_verify("Initial2 (triangle)", v1_counts2, v2_counts2);
}

// =============================================================================
// Large Initial State Test - More likely to trigger race conditions
// =============================================================================

TEST_F(MatchForwardingDebugTest, MultiRule_TriggerRaces) {
    std::cout << "\n========================================\n";
    std::cout << "MultiRule Test (from TestCase4 - reliably fails)\n";
    std::cout << "========================================\n";

    // Rule 1: {{x,y,z}} -> {{x,y},{x,z},{x,w}} (ternary -> 3 binary edges)
    v2::RewriteRule rule1 = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    // Rule 2: {{x,y}} -> {{x,y},{x,z}} (binary -> 2 binary edges)
    v2::RewriteRule rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({0, 2})
        .build();

    // Initial: single ternary edge {{0,1,2}}
    std::vector<std::vector<v2::VertexId>> initial = {{0, 1, 2}};

    const int NUM_RUNS = 50;
    std::set<size_t> unique_states;
    std::set<size_t> unique_events;

    for (int i = 0; i < NUM_RUNS; ++i) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), 0);  // 0 = hardware_concurrency, matches fuzz test
        engine.add_rule(rule1);
        engine.add_rule(rule2);
        engine.evolve(initial, 4);  // 4 steps like TestCase4

        size_t num_canonical = hg->num_canonical_states();
        unique_states.insert(num_canonical);
        unique_events.insert(engine.num_events());

        if (i == 0 || unique_states.size() > 1 || unique_events.size() > 1) {
            std::cout << "Run " << (i+1) << ": states=" << num_canonical
                      << ", events=" << engine.num_events() << "\n";

            // If we got extra states, dump all canonical hashes
            if (num_canonical > 5) {
                std::cout << "  EXTRA STATE DETECTED! Dumping canonical hashes:\n";
                for (uint32_t sid = 0; sid < hg->num_states(); ++sid) {
                    const auto& state = hg->get_state(sid);
                    std::cout << "    State " << sid << " (canonical_hash="
                              << std::hex << state.canonical_hash << std::dec << "): edges={";
                    hg->get_state_edges(sid).for_each([&](uint32_t eid) {
                        std::cout << eid << " ";
                    });
                    std::cout << "}\n";
                }
            }
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Summary after " << NUM_RUNS << " runs:\n";
    std::cout << "  States: " << (unique_states.size() == 1 ? "DETERMINISTIC" : "NON-DETERMINISTIC")
              << " (" << unique_states.size() << " unique values: ";
    for (auto v : unique_states) std::cout << v << " ";
    std::cout << ")\n";
    std::cout << "  Events: " << (unique_events.size() == 1 ? "DETERMINISTIC" : "NON-DETERMINISTIC")
              << " (" << unique_events.size() << " unique values: ";
    for (auto v : unique_events) std::cout << v << " ";
    std::cout << ")\n";

    EXPECT_EQ(unique_states.size(), 1u) << "State count non-deterministic";
    EXPECT_EQ(unique_events.size(), 1u) << "Event count non-deterministic";
}

// =============================================================================
// V1 Comparison Test - verify unified matches v1 for MultiRule case
// Tests with match forwarding ON/OFF and shared tree ON/OFF
// =============================================================================

TEST_F(MatchForwardingDebugTest, MultiRule_V1_Comparison) {
    std::cout << "\n========================================\n";
    std::cout << "MultiRule V1 Comparison Test\n";
    std::cout << "========================================\n";

    // v1 rules
    // Rule 1: {{x,y,z}} -> {{x,y},{x,z},{x,w}}
    PatternHypergraph v1_lhs1, v1_rhs1;
    v1_lhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(3)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(4)});
    RewritingRule v1_rule1(v1_lhs1, v1_rhs1);

    // Rule 2: {{x,y}} -> {{x,y},{x,z}}
    PatternHypergraph v1_lhs2, v1_rhs2;
    v1_lhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(3)});
    RewritingRule v1_rule2(v1_lhs2, v1_rhs2);

    // v2 rules
    v2::RewriteRule v2_rule1 = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1})
        .rhs({0, 2})
        .rhs({0, 3})
        .build();

    v2::RewriteRule v2_rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({0, 2})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0, 1, 2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1, 2}};

    // Run v1 to get ground truth
    auto v1_counts = run_v1({v1_rule1, v1_rule2}, v1_initial, 4);
    std::cout << "V1 ground truth: states=" << v1_counts.num_states
              << ", events=" << v1_counts.num_events << "\n\n";

    const int NUM_RUNS = 20;

    // Helper to run unified with specific settings
    auto run_test = [&](bool match_fwd, bool shared_tree, const char* label) {
        int correct = 0, wrong = 0;
        for (int i = 0; i < NUM_RUNS; ++i) {
            auto hg = std::make_unique<v2::UnifiedHypergraph>();
            if (!shared_tree) {
                hg->disable_shared_tree();
            }
            v2::ParallelEvolutionEngine engine(hg.get(), 8);
            engine.set_match_forwarding(match_fwd);
            engine.add_rule(v2_rule1);
            engine.add_rule(v2_rule2);
            engine.evolve(v2_initial, 4);

            size_t states = hg->num_canonical_states();
            size_t events = engine.num_events();

            if (states == v1_counts.num_states && events == v1_counts.num_events) {
                correct++;
            } else {
                wrong++;
                if (wrong == 1) {  // Only dump first wrong run
                    std::cout << "  " << label << " run " << (i+1)
                              << " WRONG: states=" << states << ", events=" << events << "\n";

                    // Dump all canonical hashes to see what's different
                    std::cout << "  Canonical hashes:\n";
                    std::map<uint64_t, std::vector<v2::StateId>> hash_to_states;
                    for (v2::StateId sid = 0; sid < hg->num_states(); ++sid) {
                        const auto& s = hg->get_state(sid);
                        hash_to_states[s.canonical_hash].push_back(sid);
                    }
                    for (const auto& [hash, sids] : hash_to_states) {
                        std::cout << "    hash=" << hash << " states=[";
                        for (size_t j = 0; j < sids.size(); ++j) {
                            if (j > 0) std::cout << ",";
                            std::cout << sids[j];
                        }
                        std::cout << "] (count=" << sids.size() << ")\n";
                    }
                }
            }
        }
        std::cout << label << ": " << correct << "/" << NUM_RUNS << " correct\n";
        return wrong;
    };

    std::cout << "--- Testing configurations ---\n";

    // Test 1: Full matching (no forwarding), exact canonicalization
    int wrong1 = run_test(false, false, "Full match + exact canon");

    // Test 2: Full matching, shared tree canonicalization
    int wrong2 = run_test(false, true, "Full match + shared tree");

    // Test 3: Match forwarding, exact canonicalization
    int wrong3 = run_test(true, false, "Match fwd + exact canon ");

    // Test 4: Match forwarding, shared tree (current default)
    int wrong4 = run_test(true, true, "Match fwd + shared tree ");

    std::cout << "\n========================================\n";
    std::cout << "Summary:\n";
    std::cout << "  Full matching (no fwd): " << (wrong1 == 0 && wrong2 == 0 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Match forwarding:       " << (wrong3 == 0 && wrong4 == 0 ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n";

    // We expect full matching to always work
    EXPECT_EQ(wrong1, 0) << "Full match + exact canon should work";
    EXPECT_EQ(wrong2, 0) << "Full match + shared tree should work";

    // Match forwarding is what we're debugging
    EXPECT_EQ(wrong3, 0) << "Match fwd + exact canon has bugs";
    EXPECT_EQ(wrong4, 0) << "Match fwd + shared tree has bugs";
}

// =============================================================================
// Repeated Run Test for Non-Determinism Detection
// =============================================================================

TEST_F(MatchForwardingDebugTest, RepeatedRuns_DetectNonDeterminism) {
    std::cout << "\n========================================\n";
    std::cout << "Repeated Runs - Non-Determinism Detection\n";
    std::cout << "========================================\n";

    // Rule 1: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {y,w}} (2->3)
    v2::RewriteRule v2_rule1 = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    // Rule 2: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {x,z}, {z,w}} (2->4)
    v2::RewriteRule v2_rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({0, 2})
        .rhs({2, 3})
        .build();

    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}};

    const int NUM_RUNS = 10;
    std::set<size_t> unique_states;
    std::set<size_t> unique_events;
    std::set<size_t> unique_causal;
    std::set<size_t> unique_branchial;

    for (int i = 0; i < NUM_RUNS; ++i) {
        std::cout << "\n--- Run " << (i + 1) << "/" << NUM_RUNS << " ---\n";
        auto counts = run_unified({v2_rule1, v2_rule2}, initial, 2);
        unique_states.insert(counts.num_states);
        unique_events.insert(counts.num_events);
        unique_causal.insert(counts.num_causal);
        unique_branchial.insert(counts.num_branchial);
    }

    std::cout << "\n========================================\n";
    std::cout << "Summary after " << NUM_RUNS << " runs:\n";
    std::cout << "  States: " << (unique_states.size() == 1 ? "DETERMINISTIC" : "NON-DETERMINISTIC")
              << " (" << unique_states.size() << " unique values: ";
    for (auto v : unique_states) std::cout << v << " ";
    std::cout << ")\n";
    std::cout << "  Events: " << (unique_events.size() == 1 ? "DETERMINISTIC" : "NON-DETERMINISTIC")
              << " (" << unique_events.size() << " unique values: ";
    for (auto v : unique_events) std::cout << v << " ";
    std::cout << ")\n";
    std::cout << "  Causal: " << (unique_causal.size() == 1 ? "DETERMINISTIC" : "NON-DETERMINISTIC")
              << " (" << unique_causal.size() << " unique values: ";
    for (auto v : unique_causal) std::cout << v << " ";
    std::cout << ")\n";
    std::cout << "  Branchial: " << (unique_branchial.size() == 1 ? "DETERMINISTIC" : "NON-DETERMINISTIC")
              << " (" << unique_branchial.size() << " unique values: ";
    for (auto v : unique_branchial) std::cout << v << " ";
    std::cout << ")\n";

    EXPECT_EQ(unique_states.size(), 1u) << "State count non-deterministic";
    EXPECT_EQ(unique_events.size(), 1u) << "Event count non-deterministic";
}

// =============================================================================
// Single-Threaded Baseline Test
// =============================================================================

TEST_F(MatchForwardingDebugTest, SingleThreaded_Baseline) {
    std::cout << "\n========================================\n";
    std::cout << "Single-Threaded Baseline Test\n";
    std::cout << "========================================\n";

    // v1 rules
    PatternHypergraph v1_lhs1, v1_rhs1;
    v1_lhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_lhs1.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs1.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(4)});
    RewritingRule v1_rule1(v1_lhs1, v1_rhs1);

    PatternHypergraph v1_lhs2, v1_rhs2;
    v1_lhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_lhs2.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(2)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(2), PatternVertex::variable(3)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(1), PatternVertex::variable(3)});
    v1_rhs2.add_edge(PatternEdge{PatternVertex::variable(3), PatternVertex::variable(4)});
    RewritingRule v1_rule2(v1_lhs2, v1_rhs2);

    // v2 rules
    v2::RewriteRule v2_rule1 = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({1, 3})
        .build();

    v2::RewriteRule v2_rule2 = v2::make_rule(1)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({0, 2})
        .rhs({2, 3})
        .build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{1, 2}, {2, 3}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0, 1}, {1, 2}};

    std::cout << "\n--- v1 (single-threaded reference) ---\n";
    auto v1_counts = run_v1({v1_rule1, v1_rule2}, v1_initial, 2);

    std::cout << "\n--- v2 with 1 thread ---\n";
    auto v2_counts_1t = run_unified({v2_rule1, v2_rule2}, v2_initial, 2, 1);

    std::cout << "\n--- v2 with 4 threads ---\n";
    auto v2_counts_4t = run_unified({v2_rule1, v2_rule2}, v2_initial, 2, 4);

    compare_and_verify("v1 vs v2(1 thread)", v1_counts, v2_counts_1t);
    compare_and_verify("v1 vs v2(4 threads)", v1_counts, v2_counts_4t);
}
