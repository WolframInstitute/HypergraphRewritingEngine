// Exact comparison of V1 and Unified parallel evolution
// Goal: Identify EXACTLY where counts diverge

#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <set>
#include <map>

// V1
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/wolfram_states.hpp"

// V2 (unified)
#include "hypergraph/unified/parallel_evolution.hpp"

using namespace std;

namespace v2 = hypergraph::unified;

// =============================================================================
// Simplest possible test: {{0,1}} with rule {{x,y}} -> {{y,z}}, 1 step
// =============================================================================

TEST(V1_Unified_Exact, Step1_SimpleRule) {
    cout << "\n========================================\n";
    cout << "TEST: {{0,1}} with {{x,y}} -> {{y,z}}, 1 step\n";
    cout << "========================================\n\n";

    // -------------------------------------------------------------------------
    // V1 Evolution
    // -------------------------------------------------------------------------
    cout << "=== V1 Evolution ===\n";
    size_t v1_states = 0, v1_canonical = 0, v1_events = 0, v1_causal = 0, v1_branchial = 0;
    {
        hypergraph::WolframEvolution v1_evo(1, 1, true, true);  // 1 step, 1 thread, canonicalize, full_capture

        hypergraph::PatternHypergraph lhs;
        lhs.add_edge({
            hypergraph::PatternVertex::variable(0),
            hypergraph::PatternVertex::variable(1)
        });

        hypergraph::PatternHypergraph rhs;
        rhs.add_edge({
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        });

        hypergraph::RewritingRule rule(lhs, rhs);
        v1_evo.add_rule(rule);
        v1_evo.initialize_radius_config();

        vector<vector<hypergraph::GlobalVertexId>> initial = {{0, 1}};
        v1_evo.evolve(initial);

        const auto& mg = v1_evo.get_multiway_graph();

        v1_states = mg.get_all_states().size();
        v1_canonical = mg.num_states();  // num_states() returns canonical count when canonicalization enabled
        v1_events = mg.num_events();
        v1_causal = mg.get_causal_edge_count();
        v1_branchial = mg.get_branchial_edge_count();

        cout << "States (total stored): " << v1_states << "\n";
        cout << "States (canonical): " << v1_canonical << "\n";
        cout << "Events: " << v1_events << "\n";
        cout << "Causal edges: " << v1_causal << "\n";
        cout << "Branchial edges: " << v1_branchial << "\n";

        cout << "\nStates:\n";
        for (const auto& state : mg.get_all_states()) {
            cout << "  State " << state->id().value << ": {";
            bool first = true;
            for (const auto& edge : state->edges()) {
                if (!first) cout << ", ";
                first = false;
                cout << "{";
                for (size_t i = 0; i < edge.global_vertices.size(); ++i) {
                    if (i > 0) cout << ",";
                    cout << edge.global_vertices[i];
                }
                cout << "}";
            }
            cout << "}\n";
        }

        cout << "\nEvents:\n";
        for (const auto& event : mg.get_all_events()) {
            cout << "  Event " << event.event_id << ": state " << event.input_state_id.value
                 << " -> state " << event.output_state_id.value << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // Unified Parallel Evolution
    // -------------------------------------------------------------------------
    cout << "\n=== Unified Parallel Evolution ===\n";
    size_t v2_states = 0, v2_canonical = 0, v2_events = 0, v2_causal = 0, v2_branchial = 0;
    {
        auto hg = make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), 1);  // 1 thread for determinism

        v2::RewriteRule rule = v2::make_rule(0)
            .lhs({0, 1})    // {{x, y}}
            .rhs({1, 2})    // {{y, z}}
            .build();

        engine.add_rule(rule);

        vector<vector<v2::VertexId>> initial = {{0, 1}};
        engine.evolve(initial, 1);

        v2_states = hg->num_states();
        v2_canonical = hg->num_canonical_states();
        v2_events = hg->num_events();
        v2_causal = hg->causal_graph().num_causal_edges();
        v2_branchial = hg->causal_graph().num_branchial_edges();

        cout << "States (total): " << v2_states << "\n";
        cout << "States (canonical): " << v2_canonical << "\n";
        cout << "Events: " << v2_events << "\n";
        cout << "Causal edges: " << v2_causal << "\n";
        cout << "Branchial edges: " << v2_branchial << "\n";

        cout << "\nStates:\n";
        for (v2::StateId sid = 0; sid < hg->num_states(); ++sid) {
            const auto& state = hg->get_state(sid);
            cout << "  State " << sid << ": {";
            bool first = true;
            state.edges.for_each([&](v2::EdgeId eid) {
                if (!first) cout << ", ";
                first = false;
                const auto& edge = hg->get_edge(eid);
                cout << "{";
                for (uint8_t i = 0; i < edge.arity; ++i) {
                    if (i > 0) cout << ",";
                    cout << edge.vertices[i];
                }
                cout << "}";
            });
            cout << "}\n";
        }

        cout << "\nEvents:\n";
        for (v2::EventId eid = 0; eid < hg->num_events(); ++eid) {
            const auto& event = hg->get_event(eid);
            cout << "  Event " << eid << ": state " << event.input_state
                 << " -> state " << event.output_state << "\n";
        }
    }

    cout << "\n=== COMPARISON ===\n";
    cout << "                    V1      Unified Match?\n";
    cout << "States (total):     " << setw(6) << v1_states << "  " << setw(6) << v2_states << "  " << (v1_states == v2_states ? "YES" : "NO") << "\n";
    cout << "States (canonical): " << setw(6) << v1_canonical << "  " << setw(6) << v2_canonical << "  " << (v1_canonical == v2_canonical ? "YES" : "NO") << "\n";
    cout << "Events:             " << setw(6) << v1_events << "  " << setw(6) << v2_events << "  " << (v1_events == v2_events ? "YES" : "NO") << "\n";
    cout << "Causal edges:       " << setw(6) << v1_causal << "  " << setw(6) << v2_causal << "  " << (v1_causal == v2_causal ? "YES" : "NO") << "\n";
    cout << "Branchial edges:    " << setw(6) << v1_branchial << "  " << setw(6) << v2_branchial << "  " << (v1_branchial == v2_branchial ? "YES" : "NO") << "\n";

    EXPECT_EQ(v1_canonical, v2_canonical) << "Canonical state count mismatch";
    EXPECT_EQ(v1_events, v2_events) << "Event count mismatch";
    EXPECT_EQ(v1_causal, v2_causal) << "Causal edge count mismatch";
    EXPECT_EQ(v1_branchial, v2_branchial) << "Branchial edge count mismatch";
}

// =============================================================================
// Two steps
// =============================================================================

TEST(V1_Unified_Exact, Step2_SimpleRule) {
    cout << "\n========================================\n";
    cout << "TEST: {{0,1}} with {{x,y}} -> {{y,z}}, 2 steps\n";
    cout << "========================================\n\n";

    // -------------------------------------------------------------------------
    // V1 Evolution
    // -------------------------------------------------------------------------
    cout << "=== V1 Evolution ===\n";
    size_t v1_states = 0, v1_canonical = 0, v1_events = 0, v1_causal = 0, v1_branchial = 0;
    {
        hypergraph::WolframEvolution v1_evo(2, 1, true, true);

        hypergraph::PatternHypergraph lhs;
        lhs.add_edge({
            hypergraph::PatternVertex::variable(0),
            hypergraph::PatternVertex::variable(1)
        });

        hypergraph::PatternHypergraph rhs;
        rhs.add_edge({
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        });

        hypergraph::RewritingRule rule(lhs, rhs);
        v1_evo.add_rule(rule);
        v1_evo.initialize_radius_config();

        vector<vector<hypergraph::GlobalVertexId>> initial = {{0, 1}};
        v1_evo.evolve(initial);

        const auto& mg = v1_evo.get_multiway_graph();

        v1_states = mg.get_all_states().size();
        v1_canonical = mg.num_states();
        v1_events = mg.num_events();
        v1_causal = mg.get_causal_edge_count();
        v1_branchial = mg.get_branchial_edge_count();

        cout << "States (total stored): " << v1_states << "\n";
        cout << "States (canonical): " << v1_canonical << "\n";
        cout << "Events: " << v1_events << "\n";
        cout << "Causal edges: " << v1_causal << "\n";
        cout << "Branchial edges: " << v1_branchial << "\n";
    }

    // -------------------------------------------------------------------------
    // Unified Parallel Evolution
    // -------------------------------------------------------------------------
    cout << "\n=== Unified Parallel Evolution ===\n";
    size_t v2_states = 0, v2_canonical = 0, v2_events = 0, v2_causal = 0, v2_branchial = 0;
    {
        auto hg = make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), 1);

        v2::RewriteRule rule = v2::make_rule(0)
            .lhs({0, 1})
            .rhs({1, 2})
            .build();

        engine.add_rule(rule);

        vector<vector<v2::VertexId>> initial = {{0, 1}};
        engine.evolve(initial, 2);

        v2_states = hg->num_states();
        v2_canonical = hg->num_canonical_states();
        v2_events = hg->num_events();
        v2_causal = hg->causal_graph().num_causal_edges();
        v2_branchial = hg->causal_graph().num_branchial_edges();

        cout << "States (total): " << v2_states << "\n";
        cout << "States (canonical): " << v2_canonical << "\n";
        cout << "Events: " << v2_events << "\n";
        cout << "Causal edges: " << v2_causal << "\n";
        cout << "Branchial edges: " << v2_branchial << "\n";

        // Debug: print all events with their consumed/produced edges
        cout << "\n=== Unified Event Details ===\n";
        for (v2::EventId eid = 0; eid < hg->num_events(); ++eid) {
            const auto& event = hg->get_event(eid);
            cout << "Event " << eid << ": input_state=" << event.input_state
                 << " -> output_state=" << event.output_state << "\n";
            cout << "  Consumed edges: ";
            for (uint8_t i = 0; i < event.num_consumed; ++i) {
                cout << event.consumed_edges[i] << " ";
            }
            cout << "\n  Produced edges: ";
            for (uint8_t i = 0; i < event.num_produced; ++i) {
                cout << event.produced_edges[i] << " ";
            }
            cout << "\n";
        }

        // Debug: print causal edges
        cout << "\n=== Unified Causal Edges ===\n";
        hg->causal_graph().for_each_causal_edge([](const v2::CausalEdge& e) {
            cout << "Causal: Event " << e.producer << " -> Event " << e.consumer
                 << " (via edge " << e.edge << ")\n";
        });

        // Debug: print branchial edges
        cout << "\n=== Unified Branchial Edges ===\n";
        hg->causal_graph().for_each_branchial_edge([](const v2::BranchialEdge& e) {
            cout << "Branchial: Event " << e.event1 << " <-> Event " << e.event2
                 << " (shared edge " << e.shared_edge << ")\n";
        });
    }

    cout << "\n=== COMPARISON ===\n";
    cout << "                    V1      Unified Match?\n";
    cout << "States (total):     " << setw(6) << v1_states << "  " << setw(6) << v2_states << "  " << (v1_states == v2_states ? "YES" : "NO") << "\n";
    cout << "States (canonical): " << setw(6) << v1_canonical << "  " << setw(6) << v2_canonical << "  " << (v1_canonical == v2_canonical ? "YES" : "NO") << "\n";
    cout << "Events:             " << setw(6) << v1_events << "  " << setw(6) << v2_events << "  " << (v1_events == v2_events ? "YES" : "NO") << "\n";
    cout << "Causal edges:       " << setw(6) << v1_causal << "  " << setw(6) << v2_causal << "  " << (v1_causal == v2_causal ? "YES" : "NO") << "\n";
    cout << "Branchial edges:    " << setw(6) << v1_branchial << "  " << setw(6) << v2_branchial << "  " << (v1_branchial == v2_branchial ? "YES" : "NO") << "\n";

    EXPECT_EQ(v1_canonical, v2_canonical) << "Canonical state count mismatch";
    EXPECT_EQ(v1_events, v2_events) << "Event count mismatch";
    EXPECT_EQ(v1_causal, v2_causal) << "Causal edge count mismatch";
    EXPECT_EQ(v1_branchial, v2_branchial) << "Branchial edge count mismatch";
}

// =============================================================================
// Two edges initial state, 1 step - tests branchial edges
// =============================================================================

TEST(V1_Unified_Exact, TwoEdges_Step1) {
    cout << "\n========================================\n";
    cout << "TEST: {{0,1}, {1,2}} with {{x,y}} -> {{y,z}}, 1 step\n";
    cout << "========================================\n\n";

    // -------------------------------------------------------------------------
    // V1 Evolution
    // -------------------------------------------------------------------------
    cout << "=== V1 Evolution ===\n";
    size_t v1_states = 0, v1_canonical = 0, v1_events = 0, v1_causal = 0, v1_branchial = 0;
    {
        hypergraph::WolframEvolution v1_evo(1, 1, true, true);

        hypergraph::PatternHypergraph lhs;
        lhs.add_edge({
            hypergraph::PatternVertex::variable(0),
            hypergraph::PatternVertex::variable(1)
        });

        hypergraph::PatternHypergraph rhs;
        rhs.add_edge({
            hypergraph::PatternVertex::variable(1),
            hypergraph::PatternVertex::variable(2)
        });

        hypergraph::RewritingRule rule(lhs, rhs);
        v1_evo.add_rule(rule);
        v1_evo.initialize_radius_config();

        vector<vector<hypergraph::GlobalVertexId>> initial = {{0, 1}, {1, 2}};
        v1_evo.evolve(initial);

        const auto& mg = v1_evo.get_multiway_graph();

        v1_states = mg.get_all_states().size();
        v1_canonical = mg.num_states();
        v1_events = mg.num_events();
        v1_causal = mg.get_causal_edge_count();
        v1_branchial = mg.get_branchial_edge_count();

        cout << "States (total stored): " << v1_states << "\n";
        cout << "States (canonical): " << v1_canonical << "\n";
        cout << "Events: " << v1_events << "\n";
        cout << "Causal edges: " << v1_causal << "\n";
        cout << "Branchial edges: " << v1_branchial << "\n";
    }

    // -------------------------------------------------------------------------
    // Unified Parallel Evolution
    // -------------------------------------------------------------------------
    cout << "\n=== Unified Parallel Evolution ===\n";
    size_t v2_states = 0, v2_canonical = 0, v2_events = 0, v2_causal = 0, v2_branchial = 0;
    {
        auto hg = make_unique<v2::UnifiedHypergraph>();
        v2::ParallelEvolutionEngine engine(hg.get(), 1);

        v2::RewriteRule rule = v2::make_rule(0)
            .lhs({0, 1})
            .rhs({1, 2})
            .build();

        engine.add_rule(rule);

        vector<vector<v2::VertexId>> initial = {{0, 1}, {1, 2}};
        engine.evolve(initial, 1);

        v2_states = hg->num_states();
        v2_canonical = hg->num_canonical_states();
        v2_events = hg->num_events();
        v2_causal = hg->causal_graph().num_causal_edges();
        v2_branchial = hg->causal_graph().num_branchial_edges();

        cout << "States (total): " << v2_states << "\n";
        cout << "States (canonical): " << v2_canonical << "\n";
        cout << "Events: " << v2_events << "\n";
        cout << "Causal edges: " << v2_causal << "\n";
        cout << "Branchial edges: " << v2_branchial << "\n";

        cout << "\nStates:\n";
        for (v2::StateId sid = 0; sid < hg->num_states(); ++sid) {
            const auto& state = hg->get_state(sid);
            cout << "  State " << sid << " (hash=" << hex << state.canonical_hash << dec << "): {";
            bool first = true;
            state.edges.for_each([&](v2::EdgeId eid) {
                if (!first) cout << ", ";
                first = false;
                const auto& edge = hg->get_edge(eid);
                cout << "{";
                for (uint8_t i = 0; i < edge.arity; ++i) {
                    if (i > 0) cout << ",";
                    cout << edge.vertices[i];
                }
                cout << "}";
            });
            cout << "}\n";
        }

        cout << "\nBranchial edge details:\n";
        auto branchial_edges = hg->causal_graph().get_branchial_edges();
        for (const auto& be : branchial_edges) {
            cout << "  Event " << be.event1 << " -- Event " << be.event2 << "\n";
        }
    }

    cout << "\n=== COMPARISON ===\n";
    cout << "                    V1      Unified Match?\n";
    cout << "States (total):     " << setw(6) << v1_states << "  " << setw(6) << v2_states << "  " << (v1_states == v2_states ? "YES" : "NO") << "\n";
    cout << "States (canonical): " << setw(6) << v1_canonical << "  " << setw(6) << v2_canonical << "  " << (v1_canonical == v2_canonical ? "YES" : "NO") << "\n";
    cout << "Events:             " << setw(6) << v1_events << "  " << setw(6) << v2_events << "  " << (v1_events == v2_events ? "YES" : "NO") << "\n";
    cout << "Causal edges:       " << setw(6) << v1_causal << "  " << setw(6) << v2_causal << "  " << (v1_causal == v2_causal ? "YES" : "NO") << "\n";
    cout << "Branchial edges:    " << setw(6) << v1_branchial << "  " << setw(6) << v2_branchial << "  " << (v1_branchial == v2_branchial ? "YES" : "NO") << "\n";

    EXPECT_EQ(v1_canonical, v2_canonical) << "Canonical state count mismatch";
    EXPECT_EQ(v1_events, v2_events) << "Event count mismatch";
    EXPECT_EQ(v1_causal, v2_causal) << "Causal edge count mismatch";
    EXPECT_EQ(v1_branchial, v2_branchial) << "Branchial edge count mismatch";
}
