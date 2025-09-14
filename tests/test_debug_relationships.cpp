#include <gtest/gtest.h>
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <iostream>
#include <iomanip>

using namespace hypergraph;

class DebugRelationshipsTest : public ::testing::Test {
protected:
    void print_event_details(const WolframEvent& event) {
        std::cout << "  Event " << event.event_id 
                  << ": State " << event.input_state_id.value << " -> " << event.output_state_id.value
                  << " (rule " << event.rule_index << ")\n";
        
        std::cout << "    Consumed edges (global IDs): ";
        for (auto id : event.consumed_edges) {
            std::cout << id << " ";
        }
        std::cout << "\n";
        
        std::cout << "    Consumed signatures: ";
        for (const auto& sig : event.consumed_edge_signatures) {
            std::cout << "{";
            for (size_t i = 0; i < sig.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << sig[i];
            }
            std::cout << "} ";
        }
        std::cout << "\n";
        
        std::cout << "    Produced edges (global IDs): ";
        for (auto id : event.produced_edges) {
            std::cout << id << " ";
        }
        std::cout << "\n";
        
        std::cout << "    Produced signatures: ";
        for (const auto& sig : event.produced_edge_signatures) {
            std::cout << "{";
            for (size_t i = 0; i < sig.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << sig[i];
            }
            std::cout << "} ";
        }
        std::cout << "\n";
    }
    
    void print_state_details(const WolframState& state, RawStateId id) {
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
    PatternHypergraph lhs, rhs;
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });
    
    RewritingRule rule(lhs, rhs);
    std::vector<std::vector<GlobalVertexId>> initial = {{1, 2}};
    
    // Run evolution for just 2 steps to keep it simple
    std::cout << "\n--- Running evolution for 2 steps ---\n";
    WolframEvolution evolution(2, 1, true, false); // 2 steps, 1 thread, canonicalization on
    evolution.add_rule(rule);
    evolution.evolve(initial);
    
    const auto& graph = evolution.get_multiway_graph();
    
    // Print all states
    std::cout << "\n--- States ---\n";
    auto states = graph.get_all_states();
    for (const auto& state : states) {
        print_state_details(state, state.raw_id());
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
        std::cout << "  " << (edge.type == EventRelationType::CAUSAL ? "CAUSAL" : "BRANCHIAL")
                  << ": Event " << edge.from_event << " -> Event " << edge.to_event << "\n";
    }
    
    // Check determinism
    std::cout << "\n--- Checking Determinism (3 runs) ---\n";
    std::set<size_t> causal_counts, branchial_counts;
    
    for (int run = 0; run < 3; ++run) {
        WolframEvolution evo(2, 1, true, false);
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
    
    EXPECT_EQ(causal_counts.size(), 1) << "Causal edges should be deterministic";
    EXPECT_EQ(branchial_counts.size(), 1) << "Branchial edges should be deterministic";
}