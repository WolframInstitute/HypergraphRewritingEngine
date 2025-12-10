#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

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

// Simple pair struct for device code (avoiding thrust dependency)
template<typename T1, typename T2>
struct DevicePair {
    T1 first;
    T2 second;
    __device__ __host__ DevicePair() : first(), second() {}
    __device__ __host__ DevicePair(T1 f, T2 s) : first(f), second(s) {}
};

// =============================================================================
// GPU Hash Table
// =============================================================================
// Open-addressing hash table with linear probing for CUDA.
// Uses 64-bit keys and 32-bit values.
// Thread-safe via atomicCAS on keys.
//
// Empty slots indicated by EMPTY_KEY.
// Supports concurrent insert and lookup.

// Device-accessible view of a hash table (can be copied to device memory)
template<uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL>
struct GPUHashTableView {
    struct Entry {
        uint64_t key;
        uint32_t value;
    };

    Entry* entries;
    uint32_t capacity;
    uint32_t mask;
    uint32_t* size;

    // Device-side insert
    // Thread-safe: uses atomicExch for value write and atomic read for value
    __device__ DevicePair<uint32_t, bool> insert(uint64_t key, uint32_t value) {
        uint32_t slot = hash(key) & mask;

        for (uint32_t i = 0; i < capacity; ++i) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&entries[slot].key,
                EMPTY_KEY,
                key
            );

            if (prev == EMPTY_KEY) {
                // We won the slot - atomically write value
                atomicExch(&entries[slot].value, value);
                atomicAdd(size, 1);
                return {value, true};
            }

            if (prev == key) {
                // Key already exists - atomically read value
                uint32_t existing = atomicAdd(&entries[slot].value, 0);
                return {existing, false};
            }

            slot = (slot + 1) & mask;
        }

        return {0, false};
    }

    __device__ DevicePair<uint32_t, bool> lookup(uint64_t key) const {
        uint32_t slot = hash(key) & mask;

        for (uint32_t i = 0; i < capacity; ++i) {
            // Atomic read of key for visibility
            uint64_t k = atomicAdd((unsigned long long*)&entries[slot].key, 0);

            if (k == key) {
                // Atomic read of value for visibility
                uint32_t v = atomicAdd((uint32_t*)&entries[slot].value, 0);
                return {v, true};
            }

            if (k == EMPTY_KEY) {
                return {0, false};
            }

            slot = (slot + 1) & mask;
        }

        return {0, false};
    }

    __device__ bool contains(uint64_t key) const {
        return lookup(key).second;
    }

    __device__ bool insert_if_absent(uint64_t key, uint32_t value) {
        return insert(key, value).second;
    }

    __device__ uint32_t get_size() const {
        return *size;
    }

private:
    __device__ __host__ static uint32_t hash(uint64_t key) {
        uint64_t h = 14695981039346656037ULL;
        h ^= key;
        h *= 1099511628211ULL;
        h ^= (key >> 32);
        h *= 1099511628211ULL;
        return static_cast<uint32_t>(h ^ (h >> 32));
    }
};

template<uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL>
class GPUHashTable {
public:
    using Entry = typename GPUHashTableView<EMPTY_KEY>::Entry;

private:
    Entry* entries_;
    uint32_t capacity_;
    uint32_t mask_;         // capacity_ - 1 (for fast modulo)
    uint32_t* size_;        // Atomic counter for number of entries

public:
    // Get a device-accessible view (struct can be copied to device)
    GPUHashTableView<EMPTY_KEY> get_view() const {
        return {entries_, capacity_, mask_, size_};
    }

    // Host-side initialization
    void init(uint32_t capacity) {
        // Round up to power of 2
        capacity_ = 1;
        while (capacity_ < capacity) capacity_ *= 2;
        mask_ = capacity_ - 1;

        CUDA_CHECK(cudaMalloc(&entries_, capacity_ * sizeof(Entry)));
        CUDA_CHECK(cudaMalloc(&size_, sizeof(uint32_t)));

        clear();
    }

    void destroy() {
        if (entries_) cudaFree(entries_);
        if (size_) cudaFree(size_);
        entries_ = nullptr;
        size_ = nullptr;
    }

    void clear() {
        // Fill with empty keys
        Entry empty;
        empty.key = EMPTY_KEY;
        empty.value = 0;

        // Use cudaMemset for bulk initialization
        // Note: This only works because EMPTY_KEY is all 1s (0xFF pattern)
        CUDA_CHECK(cudaMemset(entries_, 0xFF, capacity_ * sizeof(Entry)));
        CUDA_CHECK(cudaMemset(size_, 0, sizeof(uint32_t)));
    }

    // Device-side insert
    // Returns (existing_value, was_inserted)
    // If key already exists, returns existing value and false
    // If inserted new, returns new_value and true
    //
    // Following CPU ConcurrentMap pattern: both key and value use atomics.
    __device__ DevicePair<uint32_t, bool> insert(uint64_t key, uint32_t value) {
        uint32_t slot = hash(key) & mask_;

        for (uint32_t i = 0; i < capacity_; ++i) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&entries_[slot].key,
                EMPTY_KEY,
                key
            );

            if (prev == EMPTY_KEY) {
                // We inserted the key - atomically set value
                atomicExch(&entries_[slot].value, value);
                atomicAdd(size_, 1);
                return {value, true};
            }

            if (prev == key) {
                // Key already exists - atomically read value
                uint32_t existing = atomicAdd(&entries_[slot].value, 0);
                return {existing, false};
            }

            // Collision - linear probe
            slot = (slot + 1) & mask_;
        }

        // Table full (should never happen with proper sizing)
        return {0, false};
    }

    // Device-side lookup
    // Returns (value, found)
    __device__ DevicePair<uint32_t, bool> lookup(uint64_t key) const {
        uint32_t slot = hash(key) & mask_;

        for (uint32_t i = 0; i < capacity_; ++i) {
            // Atomic read of key
            uint64_t k = atomicAdd((unsigned long long*)&entries_[slot].key, 0);

            if (k == key) {
                // Atomic read of value
                uint32_t v = atomicAdd((uint32_t*)&entries_[slot].value, 0);
                return {v, true};
            }

            if (k == EMPTY_KEY) {
                return {0, false};  // Not found
            }

            // Linear probe
            slot = (slot + 1) & mask_;
        }

        return {0, false};  // Not found
    }

    // Device-side contains check
    __device__ bool contains(uint64_t key) const {
        return lookup(key).second;
    }

    // Device-side insert_if_absent (simpler interface)
    // Returns true if newly inserted, false if already existed
    __device__ bool insert_if_absent(uint64_t key, uint32_t value) {
        return insert(key, value).second;
    }

    __device__ uint32_t size() const {
        return *size_;
    }

    __host__ uint32_t get_size() const {
        uint32_t h_size;
        CUDA_CHECK(cudaMemcpy(&h_size, size_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return h_size;
    }

    // Host-side insert (slower, but useful for initialization)
    __host__ bool host_insert(uint64_t key, uint32_t value) {
        uint32_t slot = hash(key) & mask_;

        // Read current entry from device
        Entry entry;
        for (uint32_t i = 0; i < capacity_; ++i) {
            CUDA_CHECK(cudaMemcpy(&entry, &entries_[slot], sizeof(Entry), cudaMemcpyDeviceToHost));

            if (entry.key == EMPTY_KEY) {
                // Empty slot - insert
                entry.key = key;
                entry.value = value;
                CUDA_CHECK(cudaMemcpy(&entries_[slot], &entry, sizeof(Entry), cudaMemcpyHostToDevice));

                // Increment size
                uint32_t h_size;
                CUDA_CHECK(cudaMemcpy(&h_size, size_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                h_size++;
                CUDA_CHECK(cudaMemcpy(size_, &h_size, sizeof(uint32_t), cudaMemcpyHostToDevice));
                return true;
            }

            if (entry.key == key) {
                // Key already exists
                return false;
            }

            // Linear probe
            slot = (slot + 1) & mask_;
        }

        return false;  // Table full
    }

private:
    // FNV-1a hash for 64-bit keys
    __device__ __host__ static uint32_t hash(uint64_t key) {
        uint64_t h = 14695981039346656037ULL;
        h ^= key;
        h *= 1099511628211ULL;
        h ^= (key >> 32);
        h *= 1099511628211ULL;
        return static_cast<uint32_t>(h ^ (h >> 32));
    }
};

// =============================================================================
// GPU Set (hash table with no value, just presence)
// =============================================================================

// Device-accessible view of a hash set
template<uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL>
struct GPUHashSetView {
    uint64_t* keys;
    uint32_t capacity;
    uint32_t mask;
    uint32_t* size;

    __device__ bool insert(uint64_t key) {
        uint32_t slot = hash(key) & mask;

        for (uint32_t i = 0; i < capacity; ++i) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&keys[slot],
                EMPTY_KEY,
                key
            );

            if (prev == EMPTY_KEY) {
                atomicAdd(size, 1);
                return true;
            }

            if (prev == key) {
                return false;
            }

            slot = (slot + 1) & mask;
        }

        return false;
    }

    __device__ bool contains(uint64_t key) const {
        uint32_t slot = hash(key) & mask;

        for (uint32_t i = 0; i < capacity; ++i) {
            // Atomic read for visibility across threads
            uint64_t k = atomicAdd((unsigned long long*)&keys[slot], 0);

            if (k == key) return true;
            if (k == EMPTY_KEY) return false;

            slot = (slot + 1) & mask;
        }

        return false;
    }

    __device__ uint32_t get_size() const {
        return *size;
    }

private:
    __device__ __host__ static uint32_t hash(uint64_t key) {
        uint64_t h = 14695981039346656037ULL;
        h ^= key;
        h *= 1099511628211ULL;
        h ^= (key >> 32);
        h *= 1099511628211ULL;
        return static_cast<uint32_t>(h ^ (h >> 32));
    }
};

template<uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL>
class GPUHashSet {
private:
    uint64_t* keys_;
    uint32_t capacity_;
    uint32_t mask_;
    uint32_t* size_;

public:
    // Get a device-accessible view
    GPUHashSetView<EMPTY_KEY> get_view() const {
        return {keys_, capacity_, mask_, size_};
    }

    void init(uint32_t capacity) {
        capacity_ = 1;
        while (capacity_ < capacity) capacity_ *= 2;
        mask_ = capacity_ - 1;

        CUDA_CHECK(cudaMalloc(&keys_, capacity_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&size_, sizeof(uint32_t)));

        clear();
    }

    void destroy() {
        if (keys_) cudaFree(keys_);
        if (size_) cudaFree(size_);
        keys_ = nullptr;
        size_ = nullptr;
    }

    void clear() {
        CUDA_CHECK(cudaMemset(keys_, 0xFF, capacity_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(size_, 0, sizeof(uint32_t)));
    }

    // Insert key, return true if newly inserted
    __device__ bool insert(uint64_t key) {
        uint32_t slot = hash(key) & mask_;

        for (uint32_t i = 0; i < capacity_; ++i) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&keys_[slot],
                EMPTY_KEY,
                key
            );

            if (prev == EMPTY_KEY) {
                atomicAdd(size_, 1);
                return true;  // Newly inserted
            }

            if (prev == key) {
                return false;  // Already existed
            }

            slot = (slot + 1) & mask_;
        }

        return false;  // Table full
    }

    __device__ bool contains(uint64_t key) const {
        uint32_t slot = hash(key) & mask_;

        for (uint32_t i = 0; i < capacity_; ++i) {
            uint64_t k = keys_[slot];

            if (k == key) return true;
            if (k == EMPTY_KEY) return false;

            slot = (slot + 1) & mask_;
        }

        return false;
    }

    __device__ uint32_t size() const {
        return *size_;
    }

private:
    __device__ __host__ static uint32_t hash(uint64_t key) {
        uint64_t h = 14695981039346656037ULL;
        h ^= key;
        h *= 1099511628211ULL;
        h ^= (key >> 32);
        h *= 1099511628211ULL;
        return static_cast<uint32_t>(h ^ (h >> 32));
    }
};

// =============================================================================
// Atomic Map for Edge -> Event mapping (32-bit keys)
// =============================================================================
// Specialized for edge_producer_map where we map EdgeId -> EventId

class EdgeProducerMap {
    static constexpr uint32_t EMPTY = 0xFFFFFFFF;

private:
    uint32_t* values_;      // Direct indexed: values_[edge_id] = event_id
    uint32_t capacity_;

public:
    void init(uint32_t max_edges) {
        capacity_ = max_edges;
        CUDA_CHECK(cudaMalloc(&values_, capacity_ * sizeof(uint32_t)));
        clear();
    }

    void destroy() {
        if (values_) cudaFree(values_);
        values_ = nullptr;
    }

    void clear() {
        CUDA_CHECK(cudaMemset(values_, 0xFF, capacity_ * sizeof(uint32_t)));
    }

    // Set producer for edge (only first setter wins)
    // Returns true if we set it, false if already set by someone else
    __device__ bool set_producer(uint32_t edge_id, uint32_t event_id) {
        uint32_t prev = atomicCAS(&values_[edge_id], EMPTY, event_id);
        return prev == EMPTY;
    }

    // Get producer for edge (INVALID_ID if none)
    __device__ uint32_t get_producer(uint32_t edge_id) const {
        return values_[edge_id];
    }

    // Check if edge has producer
    __device__ bool has_producer(uint32_t edge_id) const {
        return values_[edge_id] != EMPTY;
    }

    // Host access
    uint32_t* device_ptr() { return values_; }
    const uint32_t* device_ptr() const { return values_; }
};

}  // namespace hypergraph::gpu
