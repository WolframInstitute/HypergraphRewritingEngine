#include <gtest/gtest.h>
#include "../wl_hash.cuh"
#include "../types.cuh"
#include <cstdio>
#include <vector>

using namespace hypergraph::gpu;

// Helper: Set up edges on device for testing
struct TestEdgeSetup {
    DeviceEdges edges;
    uint64_t* bitmap;
    uint32_t num_edges;
    uint32_t max_vertex;
    uint32_t next_vertex_offset;

    void init(uint32_t max_e, uint32_t max_v) {
        num_edges = 0;
        max_vertex = max_v;
        next_vertex_offset = 0;

        CUDA_CHECK(cudaMalloc(&edges.vertex_offsets, (max_e + 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&edges.vertex_data, max_e * MAX_EDGE_ARITY * sizeof(VertexId)));
        CUDA_CHECK(cudaMalloc(&edges.arities, max_e * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&edges.creator_events, max_e * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&edges.num_edges, sizeof(uint32_t)));
        edges.num_vertex_data = nullptr;  // Not used in read-only tests

        CUDA_CHECK(cudaMalloc(&bitmap, BITMAP_WORDS * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(bitmap, 0, BITMAP_WORDS * sizeof(uint64_t)));

        // Initialize vertex_offsets[0] = 0
        uint32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(edges.vertex_offsets, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    void destroy() {
        cudaFree(edges.vertex_offsets);
        cudaFree(edges.vertex_data);
        cudaFree(edges.arities);
        cudaFree(edges.creator_events);
        cudaFree(edges.num_edges);
        cudaFree(bitmap);
    }

    // Add edge and mark it in the bitmap
    void add_edge(const std::vector<VertexId>& verts) {
        uint32_t eid = num_edges++;
        uint8_t arity = verts.size();

        // Set arity
        CUDA_CHECK(cudaMemcpy(edges.arities + eid, &arity, sizeof(uint8_t), cudaMemcpyHostToDevice));

        // Set vertex offset
        CUDA_CHECK(cudaMemcpy(edges.vertex_offsets + eid, &next_vertex_offset, sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Set vertices
        CUDA_CHECK(cudaMemcpy(edges.vertex_data + next_vertex_offset, verts.data(), arity * sizeof(VertexId), cudaMemcpyHostToDevice));
        next_vertex_offset += arity;

        // Set next offset
        CUDA_CHECK(cudaMemcpy(edges.vertex_offsets + eid + 1, &next_vertex_offset, sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Update num_edges
        CUDA_CHECK(cudaMemcpy(edges.num_edges, &num_edges, sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Mark edge in bitmap
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        uint64_t h_word;
        CUDA_CHECK(cudaMemcpy(&h_word, bitmap + word, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        h_word |= (1ULL << bit);
        CUDA_CHECK(cudaMemcpy(bitmap + word, &h_word, sizeof(uint64_t), cudaMemcpyHostToDevice));

        // Update max_vertex
        for (VertexId v : verts) {
            if (v >= max_vertex) max_vertex = v + 1;
        }
    }

    void clear_bitmap() {
        CUDA_CHECK(cudaMemset(bitmap, 0, BITMAP_WORDS * sizeof(uint64_t)));
    }

    void set_bitmap_edge(uint32_t eid, bool present) {
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        uint64_t h_word;
        CUDA_CHECK(cudaMemcpy(&h_word, bitmap + word, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (present) {
            h_word |= (1ULL << bit);
        } else {
            h_word &= ~(1ULL << bit);
        }
        CUDA_CHECK(cudaMemcpy(bitmap + word, &h_word, sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
};

TEST(GPU_WLHash, BasicDistinction) {
    // Test that WL hash distinguishes two non-isomorphic states
    // State 2: {(0,1), (1,2), (1,3)}  - vertex 1 has degree 3
    // State 3: {(0,1), (1,2), (2,3)}  - max degree 2

    TestEdgeSetup setup;
    setup.init(10, 10);

    // Add edges: (0,1), (1,2), (1,3), (2,3)
    setup.add_edge({0, 1});  // edge 0
    setup.add_edge({1, 2});  // edge 1
    setup.add_edge({1, 3});  // edge 2
    setup.add_edge({2, 3});  // edge 3

    WLHasher hasher;
    hasher.init(100);

    // State 2: edges 0, 1, 2 -> {(0,1), (1,2), (1,3)}
    setup.clear_bitmap();
    setup.set_bitmap_edge(0, true);
    setup.set_bitmap_edge(1, true);
    setup.set_bitmap_edge(2, true);

    uint64_t hash_state2 = hasher.compute_hash_no_adj(
        setup.bitmap, setup.edges, setup.max_vertex, setup.num_edges, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("State 2 {(0,1), (1,2), (1,3)} hash: 0x%016lx\n", hash_state2);

    // State 3: edges 0, 1, 3 -> {(0,1), (1,2), (2,3)}
    setup.clear_bitmap();
    setup.set_bitmap_edge(0, true);
    setup.set_bitmap_edge(1, true);
    setup.set_bitmap_edge(3, true);

    uint64_t hash_state3 = hasher.compute_hash_no_adj(
        setup.bitmap, setup.edges, setup.max_vertex, setup.num_edges, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("State 3 {(0,1), (1,2), (2,3)} hash: 0x%016lx\n", hash_state3);

    // They should be different
    EXPECT_NE(hash_state2, hash_state3)
        << "States with different vertex degree distributions should have different hashes";

    hasher.destroy();
    setup.destroy();
}

TEST(GPU_WLHash, IsomorphicStates) {
    // Test that isomorphic states get the same hash
    // State A: {(0,1), (1,2)}  - a chain
    // State B: {(3,4), (4,5)}  - same chain with different vertex IDs

    TestEdgeSetup setup;
    setup.init(10, 10);

    // Add edges for both states
    setup.add_edge({0, 1});  // edge 0
    setup.add_edge({1, 2});  // edge 1
    setup.add_edge({3, 4});  // edge 2
    setup.add_edge({4, 5});  // edge 3

    WLHasher hasher;
    hasher.init(100);

    // State A: edges 0, 1
    setup.clear_bitmap();
    setup.set_bitmap_edge(0, true);
    setup.set_bitmap_edge(1, true);

    uint64_t hash_A = hasher.compute_hash_no_adj(
        setup.bitmap, setup.edges, setup.max_vertex, setup.num_edges, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("State A {(0,1), (1,2)} hash: 0x%016lx\n", hash_A);

    // State B: edges 2, 3
    setup.clear_bitmap();
    setup.set_bitmap_edge(2, true);
    setup.set_bitmap_edge(3, true);

    uint64_t hash_B = hasher.compute_hash_no_adj(
        setup.bitmap, setup.edges, setup.max_vertex, setup.num_edges, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("State B {(3,4), (4,5)} hash: 0x%016lx\n", hash_B);

    // They should be the same (isomorphic)
    EXPECT_EQ(hash_A, hash_B)
        << "Isomorphic states should have the same hash";

    hasher.destroy();
    setup.destroy();
}

TEST(GPU_WLHash, SingleEdge) {
    // Single edge should have a valid non-zero hash
    TestEdgeSetup setup;
    setup.init(10, 10);

    setup.add_edge({0, 1});  // edge 0

    WLHasher hasher;
    hasher.init(100);

    setup.clear_bitmap();
    setup.set_bitmap_edge(0, true);

    uint64_t hash = hasher.compute_hash_no_adj(
        setup.bitmap, setup.edges, setup.max_vertex, setup.num_edges, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Single edge {(0,1)} hash: 0x%016lx\n", hash);

    EXPECT_NE(hash, 0ULL) << "Hash should not be zero";
    EXPECT_NE(hash, FNV_OFFSET_BASIS) << "Hash should not be just FNV offset basis";

    hasher.destroy();
    setup.destroy();
}

// Test case mimicking LargeInitial scenario where different states get same hash
TEST(GPU_WLHash, LargeInitialScenario) {
    // Initial chain: {0,1}, {1,2}, {2,3}, {3,4}, {4,5} (edges 0-4)
    // Rule: {x,y} -> {y,z} replaces matched edge
    //
    // Child states after one step:
    // Match edge 0: {1,6}, {1,2}, {2,3}, {3,4}, {4,5}  - vertex 1 has degree 3
    // Match edge 4: {0,1}, {1,2}, {2,3}, {3,4}, {4,9}  - simple chain
    //
    // These should have DIFFERENT hashes!

    TestEdgeSetup setup;
    setup.init(20, 20);

    // Add all edges that could exist
    // Original chain edges (0-4)
    setup.add_edge({0, 1});  // edge 0
    setup.add_edge({1, 2});  // edge 1
    setup.add_edge({2, 3});  // edge 2
    setup.add_edge({3, 4});  // edge 3
    setup.add_edge({4, 5});  // edge 4
    // New edges from rewrites (5-9)
    setup.add_edge({1, 6});  // edge 5 - from matching edge 0
    setup.add_edge({2, 7});  // edge 6 - from matching edge 1
    setup.add_edge({3, 8});  // edge 7 - from matching edge 2
    setup.add_edge({4, 9});  // edge 8 - from matching edge 3
    setup.add_edge({5, 10}); // edge 9 - from matching edge 4

    WLHasher hasher;
    hasher.init(100);

    // State A: match edge 0 -> edges {5, 1, 2, 3, 4} = {1,6}, {1,2}, {2,3}, {3,4}, {4,5}
    setup.clear_bitmap();
    setup.set_bitmap_edge(5, true);  // {1,6}
    setup.set_bitmap_edge(1, true);  // {1,2}
    setup.set_bitmap_edge(2, true);  // {2,3}
    setup.set_bitmap_edge(3, true);  // {3,4}
    setup.set_bitmap_edge(4, true);  // {4,5}

    uint64_t hash_A = hasher.compute_hash_no_adj(
        setup.bitmap, setup.edges, setup.max_vertex, setup.num_edges, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("State A (match edge 0): edges {1,6},{1,2},{2,3},{3,4},{4,5} hash: 0x%016lx\n", hash_A);

    // State B: match edge 4 -> edges {0, 1, 2, 3, 9} = {0,1}, {1,2}, {2,3}, {3,4}, {5,10}
    setup.clear_bitmap();
    setup.set_bitmap_edge(0, true);  // {0,1}
    setup.set_bitmap_edge(1, true);  // {1,2}
    setup.set_bitmap_edge(2, true);  // {2,3}
    setup.set_bitmap_edge(3, true);  // {3,4}
    setup.set_bitmap_edge(9, true);  // {5,10}

    uint64_t hash_B = hasher.compute_hash_no_adj(
        setup.bitmap, setup.edges, setup.max_vertex, setup.num_edges, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("State B (match edge 4): edges {0,1},{1,2},{2,3},{3,4},{5,10} hash: 0x%016lx\n", hash_B);

    // These states have different structure:
    // State A: vertex 1 has degree 3 (connected to 6, 2, and implicitly via chain)
    // State B: a simple chain with no branching
    EXPECT_NE(hash_A, hash_B) << "Non-isomorphic states should have different hashes!";

    hasher.destroy();
    setup.destroy();
}
