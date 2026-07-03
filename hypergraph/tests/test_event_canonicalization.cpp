#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "hypergraph/arena.hpp"
#include "hypergraph/bitset.hpp"
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/rewriter.hpp"
#include "hypergraph/types.hpp"

namespace v2 = hypergraph;

// =============================================================================
// Event Canonicalization Counts
// =============================================================================
// Evolves small rules across all event-signature modes and checks that the
// engine is deterministic run-to-run (single-threaded). Counts here are a
// consistency cross-check of the WL hot path — not ground truth. For
// ground-truth counts, use IR canonicalization (Full mode) or wolframscript
// MultiwaySystem output.

class EventCanonicalizationTest : public ::testing::Test {
protected:
    struct EvolutionResult {
        size_t num_states;
        size_t num_events;
        size_t num_causal;
        size_t num_branchial;
        double runtime_ms;

        bool counts_equal(const EvolutionResult& o) const {
            return num_states == o.num_states && num_events == o.num_events &&
                   num_causal == o.num_causal && num_branchial == o.num_branchial;
        }
    };

    EvolutionResult run_once(
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        v2::EventSignatureKeys mode
    ) {
        auto start = std::chrono::high_resolution_clock::now();

        auto hg = std::make_unique<v2::Hypergraph>();
        hg->set_event_signature_keys(mode);
        v2::ParallelEvolutionEngine engine(hg.get(), 1);

        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }
        engine.evolve(initial, steps);

        auto end = std::chrono::high_resolution_clock::now();
        double runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return {
            hg->num_canonical_states(),
            hg->num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges(),
            runtime_ms
        };
    }

    static const char* mode_name(v2::EventSignatureKeys mode) {
        if (mode == v2::EVENT_SIG_NONE) return "None";
        if (mode == v2::EVENT_SIG_FULL) return "ByState";
        if (mode == v2::EVENT_SIG_AUTOMATIC) return "ByStateAndEdges";
        return "Unknown";
    }

    void compare_all_modes(
        const std::string& test_name,
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps
    ) {
        std::cout << "\n" << test_name << " (step " << steps << "):\n";
        std::cout << std::setw(20) << "Mode"
                  << std::setw(8) << "States"
                  << std::setw(8) << "Events"
                  << std::setw(8) << "Causal"
                  << std::setw(10) << "Branchial"
                  << std::setw(12) << "Time (ms)" << "\n";
        std::cout << std::string(66, '-') << "\n";

        for (auto mode : {v2::EVENT_SIG_NONE,
                          v2::EVENT_SIG_FULL,
                          v2::EVENT_SIG_AUTOMATIC}) {

            EvolutionResult first = run_once(rules, initial, steps, mode);
            EvolutionResult second = run_once(rules, initial, steps, mode);

            std::cout << std::setw(20) << mode_name(mode)
                      << std::setw(8) << first.num_states
                      << std::setw(8) << first.num_events
                      << std::setw(8) << first.num_causal
                      << std::setw(10) << first.num_branchial
                      << std::setw(12) << std::fixed << std::setprecision(2) << first.runtime_ms
                      << "\n";

            // Single-threaded evolution must be deterministic run-to-run.
            EXPECT_TRUE(first.counts_equal(second))
                << "Non-deterministic counts in " << mode_name(mode);
        }
    }
};

TEST_F(EventCanonicalizationTest, SimpleRule_Step5) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 1})
        .rhs({1, 2})
        .build();

    compare_all_modes("SimpleRule", {rule}, {{0, 1}}, 5);
}

TEST_F(EventCanonicalizationTest, TwoEdgeRule_Triangle_Step3) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 3})
        .rhs({3, 2})
        .build();

    compare_all_modes("TwoEdgeRule_Triangle", {rule}, {{0, 1}, {1, 2}, {2, 0}}, 3);
}

TEST_F(EventCanonicalizationTest, HyperedgeRule_Step4) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1, 2})
        .rhs({0, 1, 2})
        .rhs({1, 2, 3})
        .build();

    compare_all_modes("HyperedgeRule", {rule}, {{0, 1, 2}}, 4);
}

TEST_F(EventCanonicalizationTest, TwoEdgeRuleWithSelfLoops_Step3) {
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({2, 0})
        .rhs({0, 0})  // self-loop
        .build();

    compare_all_modes("TwoEdgeRuleWithSelfLoops", {rule}, {{0, 1}, {1, 2}, {2, 0}}, 3);
}

TEST_F(EventCanonicalizationTest, LargerGraph_Step4) {
    // Rule that grows the graph more significantly
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .rhs({0, 2})
        .rhs({2, 1})
        .rhs({1, 3})
        .build();

    // Start with a small cycle
    std::vector<std::vector<v2::VertexId>> initial = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

    compare_all_modes("LargerGraph", {rule}, initial, 4);
}

// Large initial state with 2-edge rule (consumes 2, produces 3)
// Single step to measure hashing overhead on large states
TEST_F(EventCanonicalizationTest, LargeInitial_TwoEdgeRule) {
    // Classic Wolfram rule: {x,y},{y,z} -> {x,y},{y,w},{w,z}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 3})
        .rhs({3, 2})
        .build();

    // Large initial state: chain of 20 edges (many matches)
    std::vector<std::vector<v2::VertexId>> initial;
    for (v2::VertexId i = 0; i < 20; ++i) {
        initial.push_back({i, i + 1});
    }

    std::cout << "\nLarge chain: " << initial.size() << " edges\n";
    compare_all_modes("LargeInitial_TwoEdge", {rule}, initial, 1);
}

// Large initial state with 3-edge rule (consumes 3, produces 4)
TEST_F(EventCanonicalizationTest, LargeInitial_ThreeEdgeRule) {
    // 3-edge rule: {x,y},{y,z},{z,w} -> {x,y},{y,z},{z,w},{w,v}
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .lhs({2, 3})
        .rhs({0, 1})
        .rhs({1, 2})
        .rhs({2, 3})
        .rhs({3, 4})
        .build();

    // Large initial state: chain of 25 edges
    std::vector<std::vector<v2::VertexId>> initial;
    for (v2::VertexId i = 0; i < 25; ++i) {
        initial.push_back({i, i + 1});
    }

    std::cout << "\nLarge chain (3-edge rule): " << initial.size() << " edges\n";
    compare_all_modes("LargeInitial_ThreeEdge", {rule}, initial, 1);
}

// Large sparse graph - disconnected components
// This is where incremental SHOULD shine - local rewrites don't affect distant vertices
TEST_F(EventCanonicalizationTest, LargeSparse_TwoEdgeRule) {
    // Simple growth rule
    v2::RewriteRule rule = v2::make_rule(0)
        .lhs({0, 1})
        .lhs({1, 2})
        .rhs({0, 1})
        .rhs({1, 3})
        .rhs({3, 2})
        .build();

    // Large sparse initial state: 30 disconnected edge pairs
    // Each pair is {v, v+1}, {v+1, v+2} - matches the rule
    std::vector<std::vector<v2::VertexId>> initial;
    v2::VertexId v = 0;

    for (int i = 0; i < 30; ++i) {
        initial.push_back({v, v + 1});
        initial.push_back({v + 1, v + 2});
        v += 4;  // Gap between components
    }

    std::cout << "\nLarge sparse state: " << initial.size() << " edges, "
              << v << " vertices (disconnected components)\n";
    compare_all_modes("LargeSparse", {rule}, initial, 1);
}
