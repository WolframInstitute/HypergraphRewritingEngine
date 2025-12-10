// Granular unit tests for GPU components
// Each test isolates a single component to identify failures precisely

#include <gtest/gtest.h>
#include <cstdio>
#include "../types.cuh"
#include "../work_queue.cuh"
#include "../hash_table.cuh"
#include "../memory_pool.cuh"

using namespace hypergraph::gpu;

// =============================================================================
// Work Queue Tests
// =============================================================================

// Simple kernel to test work queue push/pop
__global__ void test_queue_push_kernel(WorkQueueView<uint32_t>* queue, uint32_t value) {
    if (threadIdx.x == 0) {
        queue->push(value);
    }
}

__global__ void test_queue_pop_kernel(WorkQueueView<uint32_t>* queue, uint32_t* out, bool* success) {
    if (threadIdx.x == 0) {
        *success = queue->try_pop(out);
    }
}

TEST(GPU_WorkQueue, PushPop) {
    WorkQueue<uint32_t> queue;
    queue.init(1024);

    auto view = queue.get_view();
    WorkQueueView<uint32_t>* d_view;
    ASSERT_EQ(cudaMalloc(&d_view, sizeof(WorkQueueView<uint32_t>)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_view, &view, sizeof(WorkQueueView<uint32_t>), cudaMemcpyHostToDevice), cudaSuccess);

    // Push a value
    test_queue_push_kernel<<<1, 32>>>(d_view, 42);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Pop and verify
    uint32_t* d_out;
    bool* d_success;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_success, sizeof(bool)), cudaSuccess);

    test_queue_pop_kernel<<<1, 32>>>(d_view, d_out, d_success);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t h_out;
    bool h_success;
    ASSERT_EQ(cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_TRUE(h_success);
    EXPECT_EQ(h_out, 42u);

    cudaFree(d_view);
    cudaFree(d_out);
    cudaFree(d_success);
    queue.destroy();
}

TEST(GPU_WorkQueue, Empty) {
    WorkQueue<uint32_t> queue;
    queue.init(1024);

    auto view = queue.get_view();
    WorkQueueView<uint32_t>* d_view;
    ASSERT_EQ(cudaMalloc(&d_view, sizeof(WorkQueueView<uint32_t>)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_view, &view, sizeof(WorkQueueView<uint32_t>), cudaMemcpyHostToDevice), cudaSuccess);

    // Try to pop from empty queue
    uint32_t* d_out;
    bool* d_success;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_success, sizeof(bool)), cudaSuccess);

    test_queue_pop_kernel<<<1, 32>>>(d_view, d_out, d_success);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bool h_success;
    ASSERT_EQ(cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_FALSE(h_success);

    cudaFree(d_view);
    cudaFree(d_out);
    cudaFree(d_success);
    queue.destroy();
}

TEST(GPU_WorkQueue, HostPush) {
    WorkQueue<uint32_t> queue;
    queue.init(1024);

    // Push from host
    queue.host_push(99);

    auto view = queue.get_view();
    WorkQueueView<uint32_t>* d_view;
    ASSERT_EQ(cudaMalloc(&d_view, sizeof(WorkQueueView<uint32_t>)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_view, &view, sizeof(WorkQueueView<uint32_t>), cudaMemcpyHostToDevice), cudaSuccess);

    // Pop from device
    uint32_t* d_out;
    bool* d_success;
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_success, sizeof(bool)), cudaSuccess);

    test_queue_pop_kernel<<<1, 32>>>(d_view, d_out, d_success);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t h_out;
    bool h_success;
    ASSERT_EQ(cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_TRUE(h_success);
    EXPECT_EQ(h_out, 99u);

    cudaFree(d_view);
    cudaFree(d_out);
    cudaFree(d_success);
    queue.destroy();
}

// =============================================================================
// Hash Table Tests
// =============================================================================

__global__ void test_hash_insert_kernel(GPUHashTableView<>* table, uint64_t key, uint32_t value,
                                         uint32_t* out_existing, bool* out_was_new) {
    if (threadIdx.x == 0) {
        auto [existing, was_new] = table->insert(key, value);
        *out_existing = existing;
        *out_was_new = was_new;
    }
}

__global__ void test_hash_lookup_kernel(GPUHashTableView<>* table, uint64_t key,
                                        uint32_t* out_value, bool* out_found) {
    if (threadIdx.x == 0) {
        auto result = table->lookup(key);
        *out_value = result.first;
        *out_found = result.second;
    }
}

TEST(GPU_HashTable, InsertLookup) {
    GPUHashTable<> table;
    table.init(1024);

    auto view = table.get_view();
    GPUHashTableView<>* d_view;
    ASSERT_EQ(cudaMalloc(&d_view, sizeof(GPUHashTableView<>)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_view, &view, sizeof(GPUHashTableView<>), cudaMemcpyHostToDevice), cudaSuccess);

    uint32_t* d_existing;
    bool* d_was_new;
    ASSERT_EQ(cudaMalloc(&d_existing, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_was_new, sizeof(bool)), cudaSuccess);

    // Insert key=123, value=456
    test_hash_insert_kernel<<<1, 32>>>(d_view, 123, 456, d_existing, d_was_new);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bool h_was_new;
    ASSERT_EQ(cudaMemcpy(&h_was_new, d_was_new, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(h_was_new);

    // Lookup
    uint32_t* d_value;
    bool* d_found;
    ASSERT_EQ(cudaMalloc(&d_value, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_found, sizeof(bool)), cudaSuccess);

    test_hash_lookup_kernel<<<1, 32>>>(d_view, 123, d_value, d_found);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t h_value;
    bool h_found;
    ASSERT_EQ(cudaMemcpy(&h_value, d_value, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_TRUE(h_found);
    EXPECT_EQ(h_value, 456u);

    cudaFree(d_view);
    cudaFree(d_existing);
    cudaFree(d_was_new);
    cudaFree(d_value);
    cudaFree(d_found);
    table.destroy();
}

TEST(GPU_HashTable, DuplicateInsert) {
    GPUHashTable<> table;
    table.init(1024);

    auto view = table.get_view();
    GPUHashTableView<>* d_view;
    ASSERT_EQ(cudaMalloc(&d_view, sizeof(GPUHashTableView<>)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_view, &view, sizeof(GPUHashTableView<>), cudaMemcpyHostToDevice), cudaSuccess);

    uint32_t* d_existing;
    bool* d_was_new;
    ASSERT_EQ(cudaMalloc(&d_existing, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_was_new, sizeof(bool)), cudaSuccess);

    // First insert
    test_hash_insert_kernel<<<1, 32>>>(d_view, 123, 456, d_existing, d_was_new);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bool h_was_new;
    ASSERT_EQ(cudaMemcpy(&h_was_new, d_was_new, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(h_was_new);

    // Second insert with same key, different value
    test_hash_insert_kernel<<<1, 32>>>(d_view, 123, 789, d_existing, d_was_new);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t h_existing;
    ASSERT_EQ(cudaMemcpy(&h_was_new, d_was_new, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_existing, d_existing, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_FALSE(h_was_new);  // Not new - key already existed
    EXPECT_EQ(h_existing, 456u);  // Returns original value

    cudaFree(d_view);
    cudaFree(d_existing);
    cudaFree(d_was_new);
    table.destroy();
}

// =============================================================================
// Hash Set Tests
// =============================================================================

__global__ void test_hashset_insert_kernel(GPUHashSetView<>* set, uint64_t key, bool* out_was_new) {
    if (threadIdx.x == 0) {
        *out_was_new = set->insert(key);
    }
}

__global__ void test_hashset_contains_kernel(GPUHashSetView<>* set, uint64_t key, bool* out_contains) {
    if (threadIdx.x == 0) {
        *out_contains = set->contains(key);
    }
}

TEST(GPU_HashSet, InsertContains) {
    GPUHashSet<> set;
    set.init(1024);

    auto view = set.get_view();
    GPUHashSetView<>* d_view;
    ASSERT_EQ(cudaMalloc(&d_view, sizeof(GPUHashSetView<>)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_view, &view, sizeof(GPUHashSetView<>), cudaMemcpyHostToDevice), cudaSuccess);

    bool* d_result;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(bool)), cudaSuccess);

    // Insert
    test_hashset_insert_kernel<<<1, 32>>>(d_view, 12345, d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bool h_result;
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(h_result);  // Was new

    // Contains
    test_hashset_contains_kernel<<<1, 32>>>(d_view, 12345, d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(h_result);  // Contains it

    // Doesn't contain other key
    test_hashset_contains_kernel<<<1, 32>>>(d_view, 99999, d_result);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FALSE(h_result);  // Doesn't contain it

    cudaFree(d_view);
    cudaFree(d_result);
    set.destroy();
}

// =============================================================================
// Termination Detector Tests (Work-Counting Pattern)
// =============================================================================

// Test kernel: simulates work creation and completion
__global__ void test_work_counting_kernel(TerminationDetectorView* term, bool* is_done) {
    if (threadIdx.x == 0) {
        // Initially work_count=1 (set by host_work_created before kernel launch)
        // Simulate creating child work
        term->work_created();  // work_count=2

        // Complete initial work
        term->work_finished();  // work_count=1

        // Complete child work
        term->work_finished();  // work_count=0 -> done_flag=1

        *is_done = term->is_done();
    }
}

TEST(GPU_Termination, WorkCounting) {
    TerminationDetector term;
    term.init();

    // Increment work count BEFORE pushing initial work (simulated)
    term.host_work_created();

    auto term_view = term.get_view();
    TerminationDetectorView* d_term;
    bool* d_is_done;

    ASSERT_EQ(cudaMalloc(&d_term, sizeof(TerminationDetectorView)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_is_done, sizeof(bool)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_term, &term_view, sizeof(TerminationDetectorView), cudaMemcpyHostToDevice), cudaSuccess);

    // Run kernel - work counting should trigger termination when count reaches 0
    test_work_counting_kernel<<<1, 32>>>(d_term, d_is_done);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bool h_is_done;
    ASSERT_EQ(cudaMemcpy(&h_is_done, d_is_done, sizeof(bool), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(h_is_done);

    // Also verify host-side query
    EXPECT_TRUE(term.host_is_done());

    cudaFree(d_term);
    cudaFree(d_is_done);
    term.destroy();
}

// =============================================================================
// Memory Pool Tests
// =============================================================================

__global__ void test_state_alloc_kernel(StatePool* pool, uint32_t* out_id) {
    if (threadIdx.x == 0) {
        *out_id = pool->alloc_state();
    }
}

TEST(GPU_MemoryPool, StateAllocation) {
    // Allocate StatePool
    StatePool h_pool;
    ASSERT_EQ(cudaMalloc(&h_pool.states, MAX_STATES * sizeof(DeviceState)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&h_pool.num_states, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(h_pool.num_states, 0, sizeof(uint32_t)), cudaSuccess);
    h_pool.all_bitmaps = nullptr;  // Not needed for this test

    StatePool* d_pool;
    ASSERT_EQ(cudaMalloc(&d_pool, sizeof(StatePool)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_pool, &h_pool, sizeof(StatePool), cudaMemcpyHostToDevice), cudaSuccess);

    uint32_t* d_id;
    ASSERT_EQ(cudaMalloc(&d_id, sizeof(uint32_t)), cudaSuccess);

    // Allocate first state
    test_state_alloc_kernel<<<1, 32>>>(d_pool, d_id);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t h_id;
    ASSERT_EQ(cudaMemcpy(&h_id, d_id, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_id, 0u);  // First allocation gets ID 0

    // Allocate second state
    test_state_alloc_kernel<<<1, 32>>>(d_pool, d_id);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_id, d_id, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_id, 1u);  // Second allocation gets ID 1

    cudaFree(h_pool.states);
    cudaFree(h_pool.num_states);
    cudaFree(d_pool);
    cudaFree(d_id);
}

// =============================================================================
// Large Task Struct Test (Stack Usage)
// =============================================================================

// Test that we can pass large task structs through shared memory without stack overflow
__global__ void test_large_struct_kernel(MatchTaskWithContext* input, MatchTaskWithContext* output) {
    __shared__ MatchTaskWithContext shared_task;

    if (threadIdx.x == 0) {
        shared_task = *input;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *output = shared_task;
    }
}

TEST(GPU_Stack, LargeTaskStruct) {
    MatchTaskWithContext h_input;
    h_input.state_id = 42;
    h_input.step = 5;
    h_input.context.parent_state = 10;
    h_input.context.num_consumed = 2;
    h_input.context.num_produced = 1;

    MatchTaskWithContext* d_input;
    MatchTaskWithContext* d_output;
    ASSERT_EQ(cudaMalloc(&d_input, sizeof(MatchTaskWithContext)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, sizeof(MatchTaskWithContext)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, &h_input, sizeof(MatchTaskWithContext), cudaMemcpyHostToDevice), cudaSuccess);

    test_large_struct_kernel<<<1, 32>>>(d_input, d_output);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    MatchTaskWithContext h_output;
    ASSERT_EQ(cudaMemcpy(&h_output, d_output, sizeof(MatchTaskWithContext), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_output.state_id, 42u);
    EXPECT_EQ(h_output.step, 5u);
    EXPECT_EQ(h_output.context.parent_state, 10u);

    cudaFree(d_input);
    cudaFree(d_output);
}

// Test with multiple warps using shared arrays (like megakernel)
__global__ void test_shared_arrays_kernel(uint32_t* success) {
    __shared__ MatchTaskWithContext shared_match_tasks[8];
    __shared__ RewriteTaskWithMatch shared_rewrite_tasks[8];
    __shared__ uint32_t shared_flags[8];

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;

    if (lane == 0 && warp_id < 8) {
        shared_match_tasks[warp_id].state_id = warp_id;
        shared_rewrite_tasks[warp_id].step = warp_id * 10;
        shared_flags[warp_id] = warp_id + 100;
    }
    __syncthreads();

    if (lane == 0 && warp_id < 8) {
        if (shared_match_tasks[warp_id].state_id != warp_id ||
            shared_rewrite_tasks[warp_id].step != warp_id * 10 ||
            shared_flags[warp_id] != warp_id + 100) {
            atomicAdd(success, 1);  // Error
        }
    }
}

TEST(GPU_Stack, SharedArrays) {
    uint32_t* d_success;
    ASSERT_EQ(cudaMalloc(&d_success, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_success, 0, sizeof(uint32_t)), cudaSuccess);

    // Launch with 256 threads (8 warps)
    test_shared_arrays_kernel<<<1, 256>>>(d_success);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t h_success;
    ASSERT_EQ(cudaMemcpy(&h_success, d_success, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_success, 0u);  // No errors

    cudaFree(d_success);
}
