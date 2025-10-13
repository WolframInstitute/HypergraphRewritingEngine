#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>
#include "test_helpers.hpp"

class WolframEvolutionTest : public ::testing::Test {
protected:
    // Helper to create a simple rule: {{1, 2}} -> {{1, 2}, {2, 3}}
    hypergraph::RewritingRule create_test_rule() {
        hypergraph::PatternHypergraph lhs, rhs;
        
        lhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(1), 
            hypergraph::PatternVertex::variable(2)
        });
        
        rhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(1), 
            hypergraph::PatternVertex::variable(2)
        });
        rhs.add_edge(hypergraph::PatternEdge{
            hypergraph::PatternVertex::variable(2), 
            hypergraph::PatternVertex::variable(3)
        });
        
        return hypergraph::RewritingRule(lhs, rhs);
    }
};

// === WOLFRAM EVOLUTION TESTS ===

TEST_F(WolframEvolutionTest, EvolutionCreation) {
    hypergraph::WolframEvolution evolution(1, 1, true, false);  // 1 step, 1 thread
    
    // Evolution should be created successfully
    EXPECT_EQ(evolution.get_multiway_graph().num_states(), 0);
    EXPECT_EQ(evolution.get_multiway_graph().num_events(), 0);
}

TEST_F(WolframEvolutionTest, RuleAddition) {
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    
    evolution.add_rule(create_test_rule());
    
    // Rule should be added (we can't directly test this, but evolution should not crash)
    EXPECT_TRUE(true);  // If we get here, rule was added successfully
}

TEST_F(WolframEvolutionTest, BasicEvolution) {
    hypergraph::WolframEvolution evolution(1, 1, true, false);
    evolution.add_rule(create_test_rule());
    
    // Initial state: single edge {{1, 2}}
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial = {{1, 2}};
    
    try {
        evolution.evolve(initial);
        
        const auto& graph = evolution.get_multiway_graph();
        
        // After 1 step with rule {{1,2}} -> {{1,2}, {2,3}}, we should have more states
        EXPECT_GT(graph.num_states(), 0);
        
    } catch (const std::exception& e) {
        FAIL() << "Evolution threw exception: " << e.what();
    }
}

TEST_F(WolframEvolutionTest, MultiStepEvolution) {
    // Test with 3 steps - should now work with step checking fix
    hypergraph::WolframEvolution evolution(3, 1, true, false);
    evolution.add_rule(create_test_rule());
    
    // Initial state: two edges {{1, 2}, {2, 3}}
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial = {{1, 2}, {2, 3}};
    
    try {
        evolution.evolve(initial);
        
        const auto& graph = evolution.get_multiway_graph();
        
        // After 3 steps we should have multiple states and events
        EXPECT_GT(graph.num_states(), 1) << "Should have more than 1 state after 3 steps";
        EXPECT_GT(graph.num_events(), 1) << "Should have more than 1 event after 3 steps";
        
        // Print debug info
        std::cout << "After 3 steps: " << graph.num_states() << " states, " 
                  << graph.num_events() << " events\n";
        
    } catch (const std::exception& e) {
        FAIL() << "Multi-step evolution threw exception: " << e.what();
    }
}

class DebugRelationshipsTest : public ::testing::Test {
protected:
    void print_event_details(const hypergraph::WolframEvent& event) {
        std::cout << "  Event " << event.event_id 
                  << ": State " << event.input_state_id.value << " -> " << event.output_state_id.value
                  << " (rule " << event.rule_index << ")\n";
        
        std::cout << "    Consumed edges (global IDs): ";
        for (auto id : event.consumed_edges) {
            std::cout << id << " ";
        }
        std::cout << "\n";
        
        std::cout << "    Produced edges (global IDs): ";
        for (auto id : event.produced_edges) {
            std::cout << id << " ";
        }
        std::cout << "\n";
    }
    
    void print_state_details(const hypergraph::WolframState& state, hypergraph::StateID id) {
        std::cout << "  State " << id.value << ":\n";
        std::cout << "    Raw edges: ";
        for (const auto& edge : state.edges()) {
            std::cout << edge.global_id << ":{";
            for (size_t i = 0; i < edge.global_vertices.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << edge.global_vertices[i];
            }
            std::cout << "} ";
        }
        std::cout << "\n";
        
        // Show canonical form
        auto canonical = state.to_canonical_hypergraph();
        std::cout << "    Canonical edges: ";
        for (const auto& edge : canonical.edges()) {
            std::cout << "{";
            const auto& vertices = edge.vertices();
            for (size_t i = 0; i < vertices.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << vertices[i];
            }
            std::cout << "} ";
        }
        std::cout << "\n";
    }
};

TEST_F(DebugRelationshipsTest, SimpleRuleDebug) {
    std::cout << "\n=== DEBUG: Simple Rule {{1,2}} -> {{1,2},{2,3}} ===\n";
    
    // Create rule: {{1,2}} -> {{1,2}, {2,3}}
    hypergraph::PatternHypergraph lhs, rhs;
    lhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(1), hypergraph::PatternVertex::variable(2)
    });
    rhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(1), hypergraph::PatternVertex::variable(2)
    });
    rhs.add_edge(hypergraph::PatternEdge{
        hypergraph::PatternVertex::variable(2), hypergraph::PatternVertex::variable(3)
    });
    
    hypergraph::RewritingRule rule(lhs, rhs);
    std::vector<std::vector<hypergraph::GlobalVertexId>> initial = {{1, 2}};
    
    // Run evolution for just 2 steps to keep it simple
    std::cout << "\n--- Running evolution for 2 steps ---\n";
    hypergraph::WolframEvolution evolution(2, 1, true, false); // 2 steps, 1 thread, canonicalization on
    evolution.add_rule(rule);
    evolution.evolve(initial);
    
    const auto& graph = evolution.get_multiway_graph();
    
    // Print all states
    std::cout << "\n--- States ---\n";
    auto states = graph.get_all_states();
    for (const auto& state : states) {
        print_state_details(*state, state->id());
    }
    
    // Print all events with their edge signatures
    std::cout << "\n--- Events ---\n";
    auto events = graph.get_all_events();
    for (const auto& event : events) {
        print_event_details(event);
    }
    
    // Print relationships
    std::cout << "\n--- Event Relationships ---\n";
    auto event_edges = graph.get_event_edges();
    std::cout << "Total causal edges: " << graph.get_causal_edge_count() << "\n";
    std::cout << "Total branchial edges: " << graph.get_branchial_edge_count() << "\n";
    
    for (const auto& edge : event_edges) {
        std::cout << "  " << (edge.type == hypergraph::EventRelationType::CAUSAL ? "CAUSAL" : "BRANCHIAL")
                  << ": Event " << edge.from_event << " -> Event " << edge.to_event << "\n";
    }
    
    // Check determinism
    std::cout << "\n--- Checking Determinism (3 runs) ---\n";
    std::set<size_t> causal_counts, branchial_counts;
    
    for (int run = 0; run < 3; ++run) {
        hypergraph::WolframEvolution evo(2, 1, true, false);
        evo.add_rule(rule);
        evo.evolve(initial);
        const auto& g = evo.get_multiway_graph();
        
        size_t causal = g.get_causal_edge_count();
        size_t branchial = g.get_branchial_edge_count();
        causal_counts.insert(causal);
        branchial_counts.insert(branchial);
        
        std::cout << "  Run " << run << ": " << causal << " causal, " << branchial << " branchial\n";
    }
    
    std::cout << "\nCausal deterministic: " << (causal_counts.size() == 1 ? "YES" : "NO") << "\n";
    std::cout << "Branchial deterministic: " << (branchial_counts.size() == 1 ? "YES" : "NO") << "\n";
    
    EXPECT_EQ(causal_counts.size(), 1) << "Causal edge counts should be deterministic";
    EXPECT_EQ(branchial_counts.size(), 1) << "Branchial edge counts should be deterministic";
}
