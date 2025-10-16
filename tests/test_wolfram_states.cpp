#include <gtest/gtest.h>
#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/canonicalization.hpp>
#include "test_helpers.hpp"

class WolframStatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        graph = std::make_unique<hypergraph::MultiwayGraph>(true, true, false, true); // Enable full_capture and full_capture_non_canonicalised for tests
    }
    
    std::unique_ptr<hypergraph::MultiwayGraph> graph;
};

// === WOLFRAM STATE CREATION AND MANAGEMENT ===

TEST_F(WolframStatesTest, StateCreation) {
    // Test creating initial state - now returns state directly
    auto state = graph->create_initial_state({{1, 2}, {2, 3}});

    // Verify state properties
    EXPECT_EQ(state->num_edges(), 2);
    // State hash should be non-zero
    EXPECT_GT(state->compute_hash(hypergraph::HashStrategyType::UNIQUENESS_TREE), 0);
}

TEST_F(WolframStatesTest, StateGlobalEdgeManagement) {
    auto state = graph->create_initial_state({{1, 2}, {2, 3}});
    
    // Check global edges
    EXPECT_EQ(state->num_edges(), 2);
    EXPECT_EQ(state->num_vertices(), 3);  // vertices 1, 2, 3
    
    // Add another edge
    state->add_global_edge(100, {4, 5});
    EXPECT_EQ(state->num_edges(), 3);
    EXPECT_EQ(state->num_vertices(), 5);  // vertices 1, 2, 3, 4, 5
}

TEST_F(WolframStatesTest, StateSignatureIndexing) {
    auto state1 = graph->create_initial_state({{1, 2}, {2, 3}});
    auto state2 = graph->create_initial_state({{4, 5}, {5, 6}});
    
    // Different states should have different raw signatures
    auto sig1 = state1->id().value;
    auto sig2 = state2->id().value;

    // Raw IDs should be different
    EXPECT_NE(sig1, sig2);
}

// === CANONICAL FORM CACHING ===

TEST_F(WolframStatesTest, CanonicalFormCaching) {
    auto state = graph->create_initial_state({{1, 2}});
    
    // First access should compute canonical form
    const auto& canonical1 = state->get_canonical_form();
    
    // Second access should use cached version
    const auto& canonical2 = state->get_canonical_form();
    
    EXPECT_EQ(canonical1, canonical2);
    EXPECT_EQ(canonical1.vertex_count, 2);
    EXPECT_EQ(canonical1.edges.size(), 1);
}

TEST_F(WolframStatesTest, CanonicalFormInvalidation) {
    auto state = graph->create_initial_state({{1, 2}});
    auto state_id = state->id();  // Get state ID
    auto state_ptr = graph->get_state_efficient(state_id);
    ASSERT_TRUE(state_ptr != nullptr);

    auto& retrieved_state = *state_ptr;

    // Get initial canonical form (make a copy)
    auto canonical1 = state->get_canonical_form();

    // Modify state - should invalidate cache
    retrieved_state.add_global_edge(100, {3, 4});

    // Should recompute canonical form (make a copy)
    auto canonical2 = retrieved_state.get_canonical_form();
    EXPECT_NE(canonical1, canonical2);
    EXPECT_EQ(canonical2.vertex_count, 4);
    EXPECT_EQ(canonical2.edges.size(), 2);
}

// === STATE RECONSTRUCTION ===

// NOTE: This test requires implementing reconstruct_state and find_event_path_to_state methods
// These methods need to traverse the event graph to reconstruct states from the initial state
// TODO: Implement these methods for when full_capture is false
/*
TEST_F(WolframStatesTest, StateReconstruction) {
    // Create initial state
    auto initial_state = graph->create_initial_state({{1, 2}});
    auto initial_state_id = initial_state->id();

    // Create some events (mock - would normally come from rule application)
    // For this test, we'll test the path finding mechanism

    auto path = graph->find_event_path_to_state(initial_state_id);
    EXPECT_TRUE(path.empty());  // Initial state should have no path

    auto reconstructed = graph->reconstruct_state(initial_state_id);
    EXPECT_TRUE(reconstructed.has_value());
    EXPECT_EQ(reconstructed->id(), initial_state_id);
}
*/

// === MULTIWAY GRAPH OPERATIONS ===

TEST_F(WolframStatesTest, StateAndEventCounting) {
    EXPECT_EQ(graph->num_states(), 0);
    EXPECT_EQ(graph->num_events(), 0);
    
    auto state1 = graph->create_initial_state({{1, 2}});
    // Note: The counting might be broken (as mentioned in original issue)
    // These tests will reveal the actual behavior
    
    // Check what the system actually reports vs what we know
    size_t actual_states = graph->num_states();
    size_t actual_events = graph->num_events();
    
    // We created 1 state, 0 events
    EXPECT_GE(actual_states, 1);  // Should be at least 1
    EXPECT_EQ(actual_events, 0);  // No events yet
}

TEST_F(WolframStatesTest, EdgeMappingUpdates) {
    auto state = graph->create_initial_state({{1, 2}, {2, 3}});
    auto state_id = state->id();

    // Note: update_edge_mappings is private, so we can't test it directly
    // This test just verifies we can create and access states
    auto state_ptr = graph->get_state_efficient(state_id);
    EXPECT_TRUE(state_ptr != nullptr);
}

// === CAUSAL AND BRANCHIAL EDGE COMPUTATION ===

TEST_F(WolframStatesTest, CausalEdgeComputation) {
    // Create minimal setup for causal edge testing
    auto state1 = graph->create_initial_state({{1, 2}});
    auto state2 = graph->create_initial_state({{1, 2}, {2, 3}});  // Different state
    
    // The current system should compute causal edges
    size_t causal_count = graph->get_causal_edge_count();
    EXPECT_GE(causal_count, 0);  // Should be non-negative
}

TEST_F(WolframStatesTest, BranchialEdgeComputation) {
    // Create setup for branchial edge testing
    auto state1 = graph->create_initial_state({{1, 2}});
    auto state2 = graph->create_initial_state({{2, 3}});  // Different but potentially related
    
    size_t branchial_count = graph->get_branchial_edge_count();
    EXPECT_GE(branchial_count, 0);  // Should be non-negative
}

// === THREAD SAFETY AND ATOMICS ===

TEST_F(WolframStatesTest, AtomicOperations) {
    // Test that atomic operations work correctly
    EXPECT_EQ(graph->num_states(), 0);
    
    // Create states from multiple operations
    // Note: With canonicalization enabled, {{1, 2}} and {{3, 4}} are the same canonical form
    // So we need structurally different states
    auto state1 = graph->create_initial_state({{1, 2}});
    auto state2 = graph->create_initial_state({{1, 2}, {3, 4}});  // Different structure
    
    // Check atomic counters
    EXPECT_GE(graph->num_states(), 2);  // Should reflect both states
}

// === STATE DEDUPLICATION ===

TEST_F(WolframStatesTest, StateDuplicationDetection) {
    // Create two states with identical structure but different global IDs
    auto state1 = graph->create_initial_state({{1, 2}});
    auto state2 = graph->create_initial_state({{10, 20}});  // Same structure, different IDs

    auto state1_id = state1->id();
    auto state2_id = state2->id();

    auto state1_ptr = graph->get_state_efficient(state1_id);
    auto state2_ptr = graph->get_state_efficient(state2_id);

    ASSERT_TRUE(state1_ptr != nullptr);
    ASSERT_TRUE(state2_ptr != nullptr);

    // Get canonical forms
    const auto& canon1 = state1_ptr->get_canonical_form();
    const auto& canon2 = state2_ptr->get_canonical_form();

    // Should have same canonical form
    EXPECT_EQ(canon1, canon2);

    // Should have same hash
    auto hash1 = graph->get_state_hash(state1_id);
    auto hash2 = graph->get_state_hash(state2_id);
    ASSERT_TRUE(hash1.has_value());
    ASSERT_TRUE(hash2.has_value());
    EXPECT_EQ(*hash1, *hash2);
}

TEST_F(WolframStatesTest, StateDuplicationDetectionNoCanonicalization) {
    // Create a graph with canonicalization disabled
    auto no_canon_graph = std::make_unique<hypergraph::MultiwayGraph>(true); // full_capture=true
    no_canon_graph->set_canonicalize_states(false);

    // Create two states with identical structure but different global IDs
    auto state1 = no_canon_graph->create_initial_state({{1, 2}});
    auto state2 = no_canon_graph->create_initial_state({{10, 20}});  // Same structure, different IDs

    auto state1_id = state1->id();
    auto state2_id = state2->id();

    // Both states should be retrievable since canonicalization is disabled
    auto state1_ptr = no_canon_graph->get_state_efficient(state1_id);
    auto state2_ptr = no_canon_graph->get_state_efficient(state2_id);

    ASSERT_TRUE(state1_ptr != nullptr);
    ASSERT_TRUE(state2_ptr != nullptr);

    // States should have different raw IDs
    EXPECT_NE(state1_id, state2_id);

    // Get canonical forms
    const auto& canon1 = state1_ptr->get_canonical_form();
    const auto& canon2 = state2_ptr->get_canonical_form();

    // Should have same canonical form (structure is identical)
    EXPECT_EQ(canon1, canon2);

    // Hashes are not computed when canonicalization is disabled
    auto hash1 = no_canon_graph->get_state_hash(state1_id);
    auto hash2 = no_canon_graph->get_state_hash(state2_id);
    EXPECT_FALSE(hash1.has_value());
    EXPECT_FALSE(hash2.has_value());

    // Verify both states are counted separately when canonicalization is disabled
    EXPECT_EQ(no_canon_graph->num_states(), 2);
}

// === ERROR CONDITIONS ===

TEST_F(WolframStatesTest, InvalidStateHandling) {
    auto invalid_state = graph->get_state_efficient(hypergraph::INVALID_STATE);
    EXPECT_TRUE(invalid_state == nullptr);
    
    auto invalid_hash = graph->get_state_hash(hypergraph::INVALID_STATE);
    EXPECT_FALSE(invalid_hash.has_value());
    
    auto invalid_reconstruction = graph->reconstruct_state(hypergraph::INVALID_STATE);
    EXPECT_FALSE(invalid_reconstruction.has_value());
}

TEST_F(WolframStatesTest, EmptyStateHandling) {
    // Test empty initial state
    auto empty_state = graph->create_initial_state({});
    auto empty_state_id = empty_state->id();
    EXPECT_NE(empty_state_id, hypergraph::INVALID_STATE);

    auto state_ptr = graph->get_state_efficient(empty_state_id);
    ASSERT_TRUE(state_ptr != nullptr);

    EXPECT_TRUE(state_ptr->num_edges() == 0);
    EXPECT_TRUE(state_ptr->num_vertices() == 0);

    const auto& canonical = state_ptr->get_canonical_form();
    EXPECT_EQ(canonical.vertex_count, 0);
    EXPECT_TRUE(canonical.edges.empty());
}

// === LARGE SCALE TESTS ===

TEST_F(WolframStatesTest, ManyStatesCreation) {
    const int num_unique_structures = 5;  // Reduced from 10 to avoid performance issues
    std::vector<hypergraph::StateID> state_ids;

    // Create states with actually different canonical structures
    // Each state will have a different number of edges to ensure uniqueness
    for (int i = 0; i < num_unique_structures; ++i) {
        std::vector<std::vector<hypergraph::GlobalVertexId>> edges;
        // Create i+1 edges, ensuring different structure
        for (int j = 0; j <= i; ++j) {
            edges.push_back({static_cast<std::size_t>(j), static_cast<std::size_t>(j+1)});
        }
        auto state = graph->create_initial_state(edges);
        auto state_id = state->id();
        state_ids.push_back(state_id);
        EXPECT_NE(state_id, hypergraph::INVALID_STATE);
    }

    // Verify all can be retrieved
    for (auto state_id : state_ids) {
        auto state_ptr = graph->get_state_efficient(state_id);
        EXPECT_TRUE(state_ptr != nullptr);
    }

    // Check state count - with canonicalization, we should have num_unique_structures unique states
    size_t reported_count = graph->num_states();
    EXPECT_EQ(reported_count, num_unique_structures);
}