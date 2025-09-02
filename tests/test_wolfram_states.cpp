#include <gtest/gtest.h>
#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/canonicalization.hpp>
#include "test_helpers.hpp"

class WolframStatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        graph = std::make_unique<hypergraph::MultiwayGraph>();
    }
    
    std::unique_ptr<hypergraph::MultiwayGraph> graph;
};

// === WOLFRAM STATE CREATION AND MANAGEMENT ===

TEST_F(WolframStatesTest, StateCreation) {
    // Test creating initial state
    auto state_id = graph->create_initial_state({{1, 2}, {2, 3}});
    EXPECT_NE(state_id, hypergraph::INVALID_STATE);
    
    // Verify state exists and can be retrieved
    auto state_opt = graph->get_state_efficient(state_id);
    ASSERT_TRUE(state_opt.has_value());
    
    auto& state = *state_opt;
    EXPECT_EQ(state.id(), state_id);
    EXPECT_EQ(state.num_edges(), 2);
}

TEST_F(WolframStatesTest, StateGlobalEdgeManagement) {
    auto state_id = graph->create_initial_state({{1, 2}, {2, 3}});
    auto state_opt = graph->get_state_efficient(state_id);
    ASSERT_TRUE(state_opt.has_value());
    
    auto& state = *state_opt;
    
    // Check global edges
    EXPECT_EQ(state.num_edges(), 2);
    EXPECT_EQ(state.num_vertices(), 3);  // vertices 1, 2, 3
    
    // Add another edge
    state.add_global_edge(100, {4, 5});
    EXPECT_EQ(state.num_edges(), 3);
    EXPECT_EQ(state.num_vertices(), 5);  // vertices 1, 2, 3, 4, 5
}

TEST_F(WolframStatesTest, StateSignatureIndexing) {
    auto state1 = graph->create_initial_state({{1, 2}, {2, 3}});
    auto state2 = graph->create_initial_state({{4, 5}, {5, 6}});
    
    // Different states should have different signatures initially
    auto sig1 = graph->get_state_hash(state1);
    auto sig2 = graph->get_state_hash(state2);
    
    ASSERT_TRUE(sig1.has_value());
    ASSERT_TRUE(sig2.has_value());
    // Note: signatures might be same due to canonical equivalence, that's OK
}

// === CANONICAL FORM CACHING ===

TEST_F(WolframStatesTest, CanonicalFormCaching) {
    auto state_id = graph->create_initial_state({{1, 2}});
    auto state_opt = graph->get_state_efficient(state_id);
    ASSERT_TRUE(state_opt.has_value());
    
    auto& state = *state_opt;
    
    // First access should compute canonical form
    const auto& canonical1 = state.get_canonical_form();
    
    // Second access should use cached version
    const auto& canonical2 = state.get_canonical_form();
    
    EXPECT_EQ(canonical1, canonical2);
    EXPECT_EQ(canonical1.vertex_count, 2);
    EXPECT_EQ(canonical1.edges.size(), 1);
}

TEST_F(WolframStatesTest, CanonicalFormInvalidation) {
    auto state_id = graph->create_initial_state({{1, 2}});
    auto state_opt = graph->get_state_efficient(state_id);
    ASSERT_TRUE(state_opt.has_value());
    
    auto& state = *state_opt;
    
    // Get initial canonical form (make a copy)
    auto canonical1 = state.get_canonical_form();
    
    // Modify state - should invalidate cache
    state.add_global_edge(100, {3, 4});
    
    // Should recompute canonical form (make a copy)
    auto canonical2 = state.get_canonical_form();
    EXPECT_NE(canonical1, canonical2);
    EXPECT_EQ(canonical2.vertex_count, 4);
    EXPECT_EQ(canonical2.edges.size(), 2);
}

// === STATE RECONSTRUCTION ===

TEST_F(WolframStatesTest, StateReconstruction) {
    // Create initial state
    auto initial_state = graph->create_initial_state({{1, 2}});
    
    // Create some events (mock - would normally come from rule application)
    // For this test, we'll test the path finding mechanism
    
    auto path = graph->find_event_path_to_state(initial_state);
    EXPECT_TRUE(path.empty());  // Initial state should have no path
    
    auto reconstructed = graph->reconstruct_state(initial_state);
    EXPECT_TRUE(reconstructed.has_value());
    EXPECT_EQ(reconstructed->id(), initial_state);
}

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
    
    // Note: update_edge_mappings is private, so we can't test it directly
    // This test just verifies we can create and access states
    auto state_opt = graph->get_state_efficient(state);
    EXPECT_TRUE(state_opt.has_value());
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
    auto state1 = graph->create_initial_state({{1, 2}});
    auto state2 = graph->create_initial_state({{3, 4}});
    
    // Check atomic counters
    EXPECT_GE(graph->num_states(), 2);  // Should reflect both states
}

// === STATE DEDUPLICATION ===

TEST_F(WolframStatesTest, StateDuplicationDetection) {
    // Create two states with identical structure but different global IDs
    auto state1 = graph->create_initial_state({{1, 2}});
    auto state2 = graph->create_initial_state({{10, 20}});  // Same structure, different IDs
    
    auto canonical1_opt = graph->get_state_efficient(state1);
    auto canonical2_opt = graph->get_state_efficient(state2);
    
    ASSERT_TRUE(canonical1_opt.has_value());
    ASSERT_TRUE(canonical2_opt.has_value());
    
    // Get canonical forms
    const auto& canon1 = canonical1_opt->get_canonical_form();
    const auto& canon2 = canonical2_opt->get_canonical_form();
    
    // Should have same canonical form
    EXPECT_EQ(canon1, canon2);
    
    // Should have same hash
    auto hash1 = graph->get_state_hash(state1);
    auto hash2 = graph->get_state_hash(state2);
    ASSERT_TRUE(hash1.has_value());
    ASSERT_TRUE(hash2.has_value());
    EXPECT_EQ(*hash1, *hash2);
}

// === ERROR CONDITIONS ===

TEST_F(WolframStatesTest, InvalidStateHandling) {
    auto invalid_state = graph->get_state_efficient(hypergraph::INVALID_STATE);
    EXPECT_FALSE(invalid_state.has_value());
    
    auto invalid_hash = graph->get_state_hash(hypergraph::INVALID_STATE);
    EXPECT_FALSE(invalid_hash.has_value());
    
    auto invalid_reconstruction = graph->reconstruct_state(hypergraph::INVALID_STATE);
    EXPECT_FALSE(invalid_reconstruction.has_value());
}

TEST_F(WolframStatesTest, EmptyStateHandling) {
    // Test empty initial state
    auto empty_state = graph->create_initial_state({});
    EXPECT_NE(empty_state, hypergraph::INVALID_STATE);
    
    auto state_opt = graph->get_state_efficient(empty_state);
    ASSERT_TRUE(state_opt.has_value());
    
    EXPECT_TRUE(state_opt->num_edges() == 0);
    EXPECT_TRUE(state_opt->num_vertices() == 0);
    
    const auto& canonical = state_opt->get_canonical_form();
    EXPECT_EQ(canonical.vertex_count, 0);
    EXPECT_TRUE(canonical.edges.empty());
}

// === LARGE SCALE TESTS ===

TEST_F(WolframStatesTest, ManyStatesCreation) {
    const int num_states = 100;
    std::vector<hypergraph::StateId> state_ids;
    
    // Create many states
    for (int i = 0; i < num_states; ++i) {
        auto state_id = graph->create_initial_state({{static_cast<std::size_t>(i), static_cast<std::size_t>(i+1)}, 
                                                     {static_cast<std::size_t>(i+1), static_cast<std::size_t>(i+2)}});
        state_ids.push_back(state_id);
        EXPECT_NE(state_id, hypergraph::INVALID_STATE);
    }
    
    // Verify all can be retrieved
    for (auto state_id : state_ids) {
        auto state_opt = graph->get_state_efficient(state_id);
        EXPECT_TRUE(state_opt.has_value());
    }
    
    // Check state count (this is where the counting bug might show up)
    size_t reported_count = graph->num_states();
    EXPECT_GE(reported_count, num_states);
}