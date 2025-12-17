// Comprehensive V1 vs Unified comparison tests
// Tests higher/mixed arity rules, multiple rules, multiple initial states
// All tests should complete quickly (< 1 second each)
// Uses FULL event canonicalization (state IDs only, not edge positions)

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

// V1
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/wolfram_states.hpp"

// Unified (v2)
#include "hypergraph/unified/parallel_evolution.hpp"

namespace v2 = hypergraph::unified;
using hypergraph::PatternHypergraph;
using hypergraph::PatternVertex;
using hypergraph::RewritingRule;
using hypergraph::GlobalVertexId;

// =============================================================================
// Test Fixture
// =============================================================================

class V1UnifiedComprehensiveTest : public ::testing::Test {
protected:
    struct Counts {
        size_t states;
        size_t events;
        size_t causal;
        size_t branchial;
    };

    // Run v1 evolution with full event canonicalization
    Counts run_v1(
        const std::vector<RewritingRule>& rules,
        const std::vector<std::vector<GlobalVertexId>>& initial,
        size_t steps
    ) {
        // Parameters: max_steps, num_threads, canonicalize_states, full_capture,
        //             canonicalize_events, deduplicate_events, transitive_reduction,
        //             early_termination, full_capture_non_canonicalised,
        //             max_successor_states_per_parent, max_states_per_step,
        //             exploration_probability, full_event_canonicalization
        hypergraph::WolframEvolution evolution(
            steps,      // max_steps
            4,          // num_threads
            true,       // canonicalize_states
            true,       // full_capture
            true,       // canonicalize_events - ENABLED for full event canon
            false,      // deduplicate_events
            true,       // transitive_reduction
            false,      // early_termination
            false,      // full_capture_non_canonicalised
            0,          // max_successor_states_per_parent
            0,          // max_states_per_step
            1.0,        // exploration_probability
            true        // full_event_canonicalization - FULL mode (state IDs only)
        );
        for (const auto& rule : rules) {
            evolution.add_rule(rule);
        }
        evolution.initialize_radius_config();
        evolution.evolve(initial);

        const auto& mg = evolution.get_multiway_graph();
        return {
            mg.num_states(),
            mg.num_events(),
            mg.get_causal_edge_count(),
            mg.get_branchial_edge_count()
        };
    }

    // Run unified evolution (uses defaults: match_forwarding=true, batched=false)
    Counts run_unified(
        const std::vector<v2::RewriteRule>& rules,
        const std::vector<std::vector<v2::VertexId>>& initial,
        size_t steps,
        bool debug = false
    ) {
        auto hg = std::make_unique<v2::UnifiedHypergraph>();
        // Match v1's settings: canonicalize_states=true, canonicalize_events=true (full mode)
        hg->set_state_canonicalization_mode(v2::StateCanonicalizationMode::Full);
        hg->set_event_signature_keys(v2::EVENT_SIG_FULL);
        v2::ParallelEvolutionEngine engine(hg.get(), 4);

        for (const auto& rule : rules) {
            engine.add_rule(rule);
        }
        engine.evolve(initial, steps);

        if (debug) {
            std::cout << "  [DEBUG] raw_events=" << hg->num_raw_events()
                      << " canonical_events=" << hg->num_events()
                      << " raw_states=" << hg->num_states()
                      << " canonical_states=" << engine.num_canonical_states()
                      << "\n";
        }

        return {
            engine.num_canonical_states(),
            engine.num_events(),
            engine.num_causal_edges(),
            engine.num_branchial_edges()
        };
    }

    // Compare and print results
    void compare(const std::string& name, const Counts& v1, const Counts& unified) {
        std::cout << name << ": ";
        std::cout << "states=" << v1.states << "/" << unified.states;
        std::cout << " events=" << v1.events << "/" << unified.events;
        std::cout << " causal=" << v1.causal << "/" << unified.causal;
        std::cout << " branchial=" << v1.branchial << "/" << unified.branchial << "\n";

        EXPECT_EQ(v1.states, unified.states) << name << " states mismatch";
        EXPECT_EQ(v1.events, unified.events) << name << " events mismatch";
        EXPECT_EQ(v1.causal, unified.causal) << name << " causal mismatch";
        EXPECT_EQ(v1.branchial, unified.branchial) << name << " branchial mismatch";
    }

    // Helper: make v1 rule from edge lists
    RewritingRule make_v1_rule(
        const std::vector<std::vector<size_t>>& lhs_edges,
        const std::vector<std::vector<size_t>>& rhs_edges
    ) {
        PatternHypergraph lhs, rhs;
        for (const auto& edge : lhs_edges) {
            std::vector<PatternVertex> verts;
            for (size_t v : edge) verts.push_back(PatternVertex::variable(v));
            lhs.add_edge(verts);
        }
        for (const auto& edge : rhs_edges) {
            std::vector<PatternVertex> verts;
            for (size_t v : edge) verts.push_back(PatternVertex::variable(v));
            rhs.add_edge(verts);
        }
        return RewritingRule(lhs, rhs);
    }
};

// =============================================================================
// Binary (Arity-2) Edge Tests - 1->2 and 2->3 rules
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, Binary_1to2_2Steps) {
    // Rule: {x,y} -> {x,y},{y,z}
    auto v1_rule = make_v1_rule({{0,1}}, {{0,1}, {1,2}});
    auto v2_rule = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("Binary_1to2_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, Binary_2to3_2Steps) {
    // Rule: {x,y},{y,z} -> {x,y},{y,z},{z,w}
    auto v1_rule = make_v1_rule({{0,1},{1,2}}, {{0,1}, {1,2}, {2,3}});
    auto v2_rule = v2::make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,2}).rhs({2,3}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {1,2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {1,2}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("Binary_2to3_2Steps", v1, unified);
}

// =============================================================================
// Ternary (Arity-3) Edge Tests
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, Ternary_1to2_2Steps) {
    // Rule: {x,y,z} -> {x,y,z},{x,w}
    auto v1_rule = make_v1_rule({{0,1,2}}, {{0,1,2}, {0,3}});
    auto v2_rule = v2::make_rule(0).lhs({0,1,2}).rhs({0,1,2}).rhs({0,3}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1,2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1,2}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("Ternary_1to2_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, Ternary_ToBinary_2Steps) {
    // Rule: {x,y,z} -> {x,y},{y,z}
    auto v1_rule = make_v1_rule({{0,1,2}}, {{0,1}, {1,2}});
    auto v2_rule = v2::make_rule(0).lhs({0,1,2}).rhs({0,1}).rhs({1,2}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1,2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1,2}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("Ternary_ToBinary_2Steps", v1, unified);
}

// =============================================================================
// Quaternary (Arity-4) Edge Tests
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, Quaternary_1to2_2Steps) {
    // Rule: {x,y,z,w} -> {x,y,z,w},{x,v}
    auto v1_rule = make_v1_rule({{0,1,2,3}}, {{0,1,2,3}, {0,4}});
    auto v2_rule = v2::make_rule(0).lhs({0,1,2,3}).rhs({0,1,2,3}).rhs({0,4}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1,2,3}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1,2,3}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("Quaternary_1to2_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, Quaternary_ToBinary_2Steps) {
    // Rule: {x,y,z,w} -> {x,y},{z,w}
    auto v1_rule = make_v1_rule({{0,1,2,3}}, {{0,1}, {2,3}});
    auto v2_rule = v2::make_rule(0).lhs({0,1,2,3}).rhs({0,1}).rhs({2,3}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1,2,3}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1,2,3}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("Quaternary_ToBinary_2Steps", v1, unified);
}

// =============================================================================
// Mixed Arity Tests
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, MixedArity_BinaryTernaryInitial_2Steps) {
    // Rule: {x,y} -> {x,y},{y,z}
    // Initial: binary + ternary edges (ternary won't match)
    auto v1_rule = make_v1_rule({{0,1}}, {{0,1}, {1,2}});
    auto v2_rule = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {2,3,4}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {2,3,4}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("MixedArity_BinaryTernaryInitial_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, MixedArity_TernaryToTriangle_2Steps) {
    // Rule: {x,y,z} -> {x,y},{y,z},{z,x}
    auto v1_rule = make_v1_rule({{0,1,2}}, {{0,1}, {1,2}, {2,0}});
    auto v2_rule = v2::make_rule(0).lhs({0,1,2}).rhs({0,1}).rhs({1,2}).rhs({2,0}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1,2}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1,2}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("MixedArity_TernaryToTriangle_2Steps", v1, unified);
}

// =============================================================================
// Multiple Rules Tests
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, MultiRule_TwoBinaryRules_2Steps) {
    // Rule 0: {x,y} -> {x,y},{y,z}
    // Rule 1: {x,y} -> {x,z},{z,y}
    auto v1_rule0 = make_v1_rule({{0,1}}, {{0,1}, {1,2}});
    auto v1_rule1 = make_v1_rule({{0,1}}, {{0,2}, {2,1}});

    auto v2_rule0 = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();
    auto v2_rule1 = v2::make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}};

    auto v1 = run_v1({v1_rule0, v1_rule1}, v1_initial, 2);
    auto unified = run_unified({v2_rule0, v2_rule1}, v2_initial, 2);
    compare("MultiRule_TwoBinaryRules_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, MultiRule_BinaryAndTernary_2Steps) {
    // Rule 0: {x,y} -> {x,y},{y,z}
    // Rule 1: {x,y,z} -> {x,y},{y,z}
    auto v1_rule0 = make_v1_rule({{0,1}}, {{0,1}, {1,2}});
    auto v1_rule1 = make_v1_rule({{0,1,2}}, {{0,1}, {1,2}});

    auto v2_rule0 = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();
    auto v2_rule1 = v2::make_rule(1).lhs({0,1,2}).rhs({0,1}).rhs({1,2}).build();

    // Initial with both binary and ternary
    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {2,3,4}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {2,3,4}};

    auto v1 = run_v1({v1_rule0, v1_rule1}, v1_initial, 2);
    auto unified = run_unified({v2_rule0, v2_rule1}, v2_initial, 2);
    compare("MultiRule_BinaryAndTernary_2Steps", v1, unified);
}

// =============================================================================
// Multiple Initial Edges Tests
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, MultiInitial_TwoDisconnectedBinary_2Steps) {
    // Rule: {x,y} -> {y,z}
    auto v1_rule = make_v1_rule({{0,1}}, {{1,2}});
    auto v2_rule = v2::make_rule(0).lhs({0,1}).rhs({1,2}).build();

    // Two disconnected binary edges
    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {10,11}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {2,3}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("MultiInitial_TwoDisconnectedBinary_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, MultiInitial_Triangle_2Steps) {
    // Rule: {x,y} -> {x,y},{y,z}
    auto v1_rule = make_v1_rule({{0,1}}, {{0,1}, {1,2}});
    auto v2_rule = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();

    // Triangle initial state
    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {1,2}, {2,0}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {1,2}, {2,0}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("MultiInitial_Triangle_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, MultiInitial_Chain_2Steps) {
    // Rule: {x,y} -> {x,y},{x,z}
    auto v1_rule = make_v1_rule({{0,1}}, {{0,1}, {0,2}});
    auto v2_rule = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,2}).build();

    // Chain of 3 edges
    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {1,2}, {2,3}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {1,2}, {2,3}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("MultiInitial_Chain_2Steps", v1, unified);
}

// =============================================================================
// Self-Loop Tests
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, SelfLoop_Simple_2Steps) {
    // Rule: {x,y} -> {x,y},{y,y}  (creates self-loop)
    auto v1_rule = make_v1_rule({{0,1}}, {{0,1}, {1,1}});
    auto v2_rule = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,1}).build();

    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}};

    auto v1 = run_v1({v1_rule}, v1_initial, 2);
    auto unified = run_unified({v2_rule}, v2_initial, 2);
    compare("SelfLoop_Simple_2Steps", v1, unified);
}

// =============================================================================
// Complex Combined Tests
// =============================================================================

TEST_F(V1UnifiedComprehensiveTest, Complex_MultiRule_MixedArity_2Steps) {
    // Rule 0: {x,y} -> {x,y},{y,z}
    // Rule 1: {x,y,z} -> {x,y},{y,z},{z,x}
    auto v1_rule0 = make_v1_rule({{0,1}}, {{0,1}, {1,2}});
    auto v1_rule1 = make_v1_rule({{0,1,2}}, {{0,1}, {1,2}, {2,0}});

    auto v2_rule0 = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();
    auto v2_rule1 = v2::make_rule(1).lhs({0,1,2}).rhs({0,1}).rhs({1,2}).rhs({2,0}).build();

    // Mixed initial: binary chain + ternary
    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {1,2}, {10,11,12}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {1,2}, {3,4,5}};

    auto v1 = run_v1({v1_rule0, v1_rule1}, v1_initial, 2);
    auto unified = run_unified({v2_rule0, v2_rule1}, v2_initial, 2);
    compare("Complex_MultiRule_MixedArity_2Steps", v1, unified);
}

TEST_F(V1UnifiedComprehensiveTest, Complex_QuaternaryWithBinary_2Steps) {
    // Rule 0: {x,y} -> {x,y},{y,z}
    // Rule 1: {x,y,z,w} -> {x,y},{z,w},{y,z}
    auto v1_rule0 = make_v1_rule({{0,1}}, {{0,1}, {1,2}});
    auto v1_rule1 = make_v1_rule({{0,1,2,3}}, {{0,1}, {2,3}, {1,2}});

    auto v2_rule0 = v2::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build();
    auto v2_rule1 = v2::make_rule(1).lhs({0,1,2,3}).rhs({0,1}).rhs({2,3}).rhs({1,2}).build();

    // Initial: binary + quaternary
    std::vector<std::vector<GlobalVertexId>> v1_initial = {{0,1}, {10,11,12,13}};
    std::vector<std::vector<v2::VertexId>> v2_initial = {{0,1}, {2,3,4,5}};

    auto v1 = run_v1({v1_rule0, v1_rule1}, v1_initial, 2);
    auto unified = run_unified({v2_rule0, v2_rule1}, v2_initial, 2);
    compare("Complex_QuaternaryWithBinary_2Steps", v1, unified);
}
