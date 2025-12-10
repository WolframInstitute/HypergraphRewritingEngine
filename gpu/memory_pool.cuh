#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "types.cuh"

// CUDA error checking macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
} while(0)
#endif

namespace hypergraph::gpu {

// =============================================================================
// Memory Pool Allocator
// =============================================================================
// Pre-allocated pool for fixed-size objects.
// Thread-safe allocation via atomicAdd on free index.
// No deallocation (freed in bulk at evolution end).

template<typename T>
class DevicePool {
private:
    T* data_;
    uint32_t capacity_;
    uint32_t* next_free_;   // Atomic counter for next allocation

public:
    void init(uint32_t capacity) {
        capacity_ = capacity;
        CUDA_CHECK(cudaMalloc(&data_, capacity * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&next_free_, sizeof(uint32_t)));
        clear();
    }

    void destroy() {
        if (data_) cudaFree(data_);
        if (next_free_) cudaFree(next_free_);
        data_ = nullptr;
        next_free_ = nullptr;
    }

    void clear() {
        CUDA_CHECK(cudaMemset(next_free_, 0, sizeof(uint32_t)));
    }

    // Allocate single item, returns index or INVALID_ID if full
    __device__ uint32_t alloc() {
        uint32_t idx = atomicAdd(next_free_, 1);
        if (idx >= capacity_) {
            // Rollback (best effort)
            atomicSub(next_free_, 1);
            return INVALID_ID;
        }
        return idx;
    }

    // Allocate multiple items, returns start index or INVALID_ID
    __device__ uint32_t alloc_n(uint32_t count) {
        uint32_t idx = atomicAdd(next_free_, count);
        if (idx + count > capacity_) {
            atomicSub(next_free_, count);
            return INVALID_ID;
        }
        return idx;
    }

    __device__ T* get(uint32_t idx) {
        return &data_[idx];
    }

    __device__ const T* get(uint32_t idx) const {
        return &data_[idx];
    }

    __device__ T& operator[](uint32_t idx) {
        return data_[idx];
    }

    __device__ const T& operator[](uint32_t idx) const {
        return data_[idx];
    }

    __device__ uint32_t size() const {
        return *next_free_;
    }

    __device__ uint32_t capacity() const {
        return capacity_;
    }

    // Host access
    T* device_ptr() { return data_; }
    const T* device_ptr() const { return data_; }

    uint32_t get_size() const {
        uint32_t h_size;
        CUDA_CHECK(cudaMemcpy(&h_size, next_free_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return h_size;
    }

    // Host-side allocation (for setup code)
    __host__ uint32_t host_alloc() {
        uint32_t current;
        CUDA_CHECK(cudaMemcpy(&current, next_free_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (current >= capacity_) return INVALID_ID;
        uint32_t next = current + 1;
        CUDA_CHECK(cudaMemcpy(next_free_, &next, sizeof(uint32_t), cudaMemcpyHostToDevice));
        return current;
    }

    // Host-side allocate multiple
    __host__ uint32_t host_alloc_n(uint32_t count) {
        uint32_t current;
        CUDA_CHECK(cudaMemcpy(&current, next_free_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (current + count > capacity_) return INVALID_ID;
        uint32_t next = current + count;
        CUDA_CHECK(cudaMemcpy(next_free_, &next, sizeof(uint32_t), cudaMemcpyHostToDevice));
        return current;
    }
};

// =============================================================================
// Vertex Allocator
// =============================================================================
// Simple atomic counter for fresh vertex allocation.

class VertexAllocator {
private:
    uint32_t* counter_;

public:
    void init(uint32_t initial_value = 0) {
        CUDA_CHECK(cudaMalloc(&counter_, sizeof(uint32_t)));
        set(initial_value);
    }

    void destroy() {
        if (counter_) cudaFree(counter_);
        counter_ = nullptr;
    }

    void set(uint32_t value) {
        CUDA_CHECK(cudaMemcpy(counter_, &value, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    __device__ uint32_t alloc() {
        return atomicAdd(counter_, 1);
    }

    __device__ uint32_t alloc_n(uint32_t count) {
        return atomicAdd(counter_, count);
    }

    __device__ uint32_t current() const {
        return *counter_;
    }

    uint32_t get() const {
        uint32_t h_val;
        CUDA_CHECK(cudaMemcpy(&h_val, counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return h_val;
    }

    uint32_t* device_ptr() { return counter_; }
};

// =============================================================================
// Bulk Allocator for Variable-Size Data
// =============================================================================
// Used for vertex data (edges have varying arity) and consumed/produced edges.

class BulkAllocator {
private:
    uint32_t* data_;
    uint32_t capacity_;
    uint32_t* next_free_;

public:
    void init(uint32_t capacity) {
        capacity_ = capacity;
        CUDA_CHECK(cudaMalloc(&data_, capacity * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&next_free_, sizeof(uint32_t)));
        clear();
    }

    void destroy() {
        if (data_) cudaFree(data_);
        if (next_free_) cudaFree(next_free_);
        data_ = nullptr;
        next_free_ = nullptr;
    }

    void clear() {
        CUDA_CHECK(cudaMemset(next_free_, 0, sizeof(uint32_t)));
    }

    // Allocate space for 'count' uint32_t values
    __device__ uint32_t alloc(uint32_t count) {
        uint32_t offset = atomicAdd(next_free_, count);
        if (offset + count > capacity_) {
            atomicSub(next_free_, count);
            return INVALID_ID;
        }
        return offset;
    }

    __device__ uint32_t* get(uint32_t offset) {
        return &data_[offset];
    }

    __device__ void store(uint32_t offset, const uint32_t* values, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            data_[offset + i] = values[i];
        }
    }

    uint32_t* device_ptr() { return data_; }
    uint32_t* counter_ptr() { return next_free_; }
};

// =============================================================================
// Bitmap Pool for States
// =============================================================================
// Each state needs BITMAP_WORDS worth of uint64_t for edge membership.

class BitmapPool {
private:
    uint64_t* data_;
    uint32_t capacity_;         // Number of bitmaps
    uint32_t* next_free_;

public:
    void init(uint32_t num_bitmaps) {
        capacity_ = num_bitmaps;
        CUDA_CHECK(cudaMalloc(&data_, num_bitmaps * BITMAP_WORDS * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&next_free_, sizeof(uint32_t)));
        clear();
    }

    void destroy() {
        if (data_) cudaFree(data_);
        if (next_free_) cudaFree(next_free_);
        data_ = nullptr;
        next_free_ = nullptr;
    }

    void clear() {
        CUDA_CHECK(cudaMemset(next_free_, 0, sizeof(uint32_t)));
        // Note: Individual bitmaps cleared on allocation
    }

    // Allocate a bitmap, returns index
    __device__ uint32_t alloc() {
        uint32_t idx = atomicAdd(next_free_, 1);
        if (idx >= capacity_) {
            atomicSub(next_free_, 1);
            return INVALID_ID;
        }
        // Zero the bitmap
        uint64_t* bm = get(idx);
        for (uint32_t i = 0; i < BITMAP_WORDS; ++i) {
            bm[i] = 0;
        }
        return idx;
    }

    __device__ uint64_t* get(uint32_t idx) {
        return &data_[idx * BITMAP_WORDS];
    }

    // Copy parent bitmap to new slot
    __device__ void copy_from(uint32_t dst_idx, uint32_t src_idx) {
        uint64_t* dst = get(dst_idx);
        uint64_t* src = get(src_idx);
        for (uint32_t i = 0; i < BITMAP_WORDS; ++i) {
            dst[i] = src[i];
        }
    }

    uint64_t* device_ptr() { return data_; }
};

// =============================================================================
// Combined Memory Manager
// =============================================================================
// Manages all GPU memory pools for evolution.

struct GPUMemoryManager {
    // Core pools
    DevicePool<DeviceState> state_pool;
    BitmapPool bitmap_pool;
    DevicePool<DeviceEvent> event_pool;
    BulkAllocator vertex_data;          // For edge vertex storage
    BulkAllocator consumed_edges;       // For event consumed edges
    BulkAllocator produced_edges;       // For event produced edges

    // Edge storage (special handling for offsets)
    uint32_t* edge_offsets;             // [MAX_EDGES+1]
    uint8_t* edge_arities;              // [MAX_EDGES]
    uint32_t* edge_creators;            // [MAX_EDGES]
    uint32_t* num_edges;                // Atomic counter

    // Device counters for StatePool and EventPool
    uint32_t* d_num_states;             // Device counter for states
    uint32_t* d_num_events;             // Device counter for events
    uint32_t* d_consumed_offset;        // Device counter for consumed edges offset
    uint32_t* d_produced_offset;        // Device counter for produced edges offset

    // Allocators
    VertexAllocator vertex_allocator;

    void init() {
        state_pool.init(MAX_STATES);
        bitmap_pool.init(MAX_STATES);
        event_pool.init(MAX_EVENTS);
        vertex_data.init(MAX_EDGES * MAX_EDGE_ARITY);
        consumed_edges.init(MAX_EVENTS * MAX_PATTERN_EDGES);
        produced_edges.init(MAX_EVENTS * MAX_PATTERN_EDGES);

        CUDA_CHECK(cudaMalloc(&edge_offsets, (MAX_EDGES + 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&edge_arities, MAX_EDGES * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&edge_creators, MAX_EDGES * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&num_edges, sizeof(uint32_t)));

        // Allocate device counters for StatePool/EventPool
        CUDA_CHECK(cudaMalloc(&d_num_states, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_num_events, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_consumed_offset, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_produced_offset, sizeof(uint32_t)));

        vertex_allocator.init(0);

        clear();
    }

    void destroy() {
        state_pool.destroy();
        bitmap_pool.destroy();
        event_pool.destroy();
        vertex_data.destroy();
        consumed_edges.destroy();
        produced_edges.destroy();

        if (edge_offsets) cudaFree(edge_offsets);
        if (edge_arities) cudaFree(edge_arities);
        if (edge_creators) cudaFree(edge_creators);
        if (num_edges) cudaFree(num_edges);

        if (d_num_states) cudaFree(d_num_states);
        if (d_num_events) cudaFree(d_num_events);
        if (d_consumed_offset) cudaFree(d_consumed_offset);
        if (d_produced_offset) cudaFree(d_produced_offset);

        vertex_allocator.destroy();
    }

    void clear() {
        state_pool.clear();
        bitmap_pool.clear();
        event_pool.clear();
        vertex_data.clear();
        consumed_edges.clear();
        produced_edges.clear();

        CUDA_CHECK(cudaMemset(num_edges, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(edge_creators, 0xFF, MAX_EDGES * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemset(d_num_states, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_num_events, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_consumed_offset, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_produced_offset, 0, sizeof(uint32_t)));

        vertex_allocator.set(0);
    }

    // Helper to get host-side count from device counter
    uint32_t get_num_states() const {
        uint32_t count;
        CUDA_CHECK(cudaMemcpy(&count, d_num_states, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return count;
    }

    uint32_t get_num_events() const {
        uint32_t count;
        CUDA_CHECK(cudaMemcpy(&count, d_num_events, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return count;
    }

    uint32_t get_num_edges() const {
        uint32_t count;
        CUDA_CHECK(cudaMemcpy(&count, num_edges, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return count;
    }

    // Set initial state count (for after uploading initial state)
    void set_num_states(uint32_t count) {
        CUDA_CHECK(cudaMemcpy(d_num_states, &count, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
};

}  // namespace hypergraph::gpu
