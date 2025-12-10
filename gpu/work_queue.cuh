#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
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
// Device-accessible Work Queue View (POD struct)
// =============================================================================
// MPMC queue using sequence numbers for safe publish/consume.
// Key invariant: sequence[slot] tracks the "turn" for that slot.
// - For push: sequence[slot] should equal the tail value we reserved
// - For pop: sequence[slot] should equal head + 1 (meaning data is ready)
// After pop completes, we update sequence to head + capacity (next turn for producers)
template<typename T>
struct WorkQueueView {
    T* buffer;
    uint32_t capacity;
    uint32_t mask;
    uint32_t* head;
    uint32_t* tail;
    uint32_t* sequence;  // Per-slot sequence numbers for safe publishing

    // Push item to queue
    // Returns true if pushed, false if queue full
    __device__ bool push(const T& item) {
        uint32_t pos;
        uint32_t seq;
        uint32_t slot;

        // Spin to find a valid slot
        for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
            pos = atomicAdd(tail, 0);  // Atomic read of tail
            slot = pos & mask;
            seq = atomicAdd(&sequence[slot], 0);  // Atomic read of sequence

            // For a producer, sequence should equal pos (slot is available)
            if (seq == pos) {
                // Try to claim this position
                if (atomicCAS(tail, pos, pos + 1) == pos) {
                    // We own this slot - write data
                    buffer[slot] = item;
                    __threadfence();  // Ensure data is visible before updating sequence

                    // Signal that slot is ready for consumers
                    // Consumer expects sequence = pos + 1
                    atomicExch(&sequence[slot], pos + 1);
                    return true;
                }
                // CAS failed, another producer won - retry
            } else if (seq < pos) {
                // Queue is full (consumer hasn't freed this slot yet)
                return false;
            }
            // seq > pos means another producer is ahead, retry to get new tail
        }
        return false;  // Too much contention
    }

    // Non-blocking try_pop
    __device__ bool try_pop(T* out) {
        uint32_t pos;
        uint32_t seq;
        uint32_t slot;

        for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
            pos = atomicAdd(head, 0);  // Atomic read of head
            slot = pos & mask;
            seq = atomicAdd(&sequence[slot], 0);  // Atomic read of sequence

            // For a consumer, sequence should equal pos + 1 (data is ready)
            if (seq == pos + 1) {
                // Try to claim this position
                if (atomicCAS(head, pos, pos + 1) == pos) {
                    // We own this slot - read data
                    __threadfence();  // Ensure we see producer's write
                    *out = buffer[slot];
                    __threadfence();  // Ensure read completes before updating sequence

                    // Signal that slot is free for producers
                    // Next producer turn for this slot is pos + capacity
                    atomicExch(&sequence[slot], pos + capacity);
                    return true;
                }
                // CAS failed, another consumer won - retry
            } else if (seq < pos + 1) {
                // Data not ready yet OR queue empty
                // Check if queue is actually empty
                uint32_t t = atomicAdd(tail, 0);
                if (pos >= t) {
                    return false;  // Queue is empty
                }
                // Data not ready yet, spin
            }
            // seq > pos + 1 means slot was already consumed and recycled, retry
        }
        return false;  // Too much contention or empty
    }

    __device__ bool empty() const {
        uint32_t h = atomicAdd((uint32_t*)head, 0);
        uint32_t t = atomicAdd((uint32_t*)tail, 0);
        return h >= t;
    }

    __device__ uint32_t size() const {
        uint32_t h = atomicAdd((uint32_t*)head, 0);
        uint32_t t = atomicAdd((uint32_t*)tail, 0);
        return (t > h) ? (t - h) : 0;
    }

    // Blocking push with bounded retry - spins until push succeeds or limit reached
    // Use this when work MUST NOT be dropped
    // Returns true if pushed successfully, false if gave up after max retries
    __device__ bool push_wait(const T& item, uint32_t max_retries = 10000) {
        for (uint32_t i = 0; i < max_retries; ++i) {
            if (push(item)) {
                return true;
            }
            // Brief pause to reduce contention and allow consumers to drain queue
            __nanosleep(100);
        }
        // After max retries, print warning and return failure
        if (threadIdx.x == 0) {
            printf("WARNING: push_wait failed after %u retries (queue full)\n", max_retries);
        }
        return false;
    }
};

// =============================================================================
// Lock-Free Work Queue
// =============================================================================
// MPMC (multi-producer, multi-consumer) circular buffer.
// Uses atomics for thread-safe concurrent access.
//
// Design:
// - Fixed capacity (power of 2)
// - head: next index to dequeue from
// - tail: next index to enqueue to
// - Full: (tail - head) >= capacity
// - Empty: head == tail

template<typename T>
class WorkQueue {
private:
    T* buffer_;
    uint32_t* sequence_;  // Per-slot sequence numbers
    uint32_t capacity_;
    uint32_t mask_;

    // Separate cache lines to avoid false sharing
    alignas(128) uint32_t* head_;       // Consumer index
    alignas(128) uint32_t* tail_;       // Producer index (reserved slots)

public:
    void init(uint32_t capacity) {
        // Round to power of 2
        capacity_ = 1;
        while (capacity_ < capacity) capacity_ *= 2;
        mask_ = capacity_ - 1;

        CUDA_CHECK(cudaMalloc(&buffer_, capacity_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&sequence_, capacity_ * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&head_, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tail_, sizeof(uint32_t)));

        clear();
    }

    void destroy() {
        if (buffer_) cudaFree(buffer_);
        if (sequence_) cudaFree(sequence_);
        if (head_) cudaFree(head_);
        if (tail_) cudaFree(tail_);
        buffer_ = nullptr;
        sequence_ = nullptr;
        head_ = nullptr;
        tail_ = nullptr;
    }

    void clear() {
        CUDA_CHECK(cudaMemset(head_, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(tail_, 0, sizeof(uint32_t)));
        // Initialize sequence numbers to slot indices (0, 1, 2, ...)
        // This allows producers to start claiming slots from position 0
        std::vector<uint32_t> h_seq(capacity_);
        for (uint32_t i = 0; i < capacity_; ++i) {
            h_seq[i] = i;
        }
        CUDA_CHECK(cudaMemcpy(sequence_, h_seq.data(), capacity_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Push item to queue (device-side, use view in kernel)
    // Returns true if successful, false if full
    __device__ bool push(const T& item) {
        uint32_t tail = atomicAdd(tail_, 1);
        uint32_t head = *head_;  // Non-atomic read OK (tail already reserved)

        // Check if full
        if (tail - head >= capacity_) {
            // Rollback
            atomicSub(tail_, 1);
            return false;
        }

        uint32_t slot = tail & mask_;
        buffer_[slot] = item;
        __threadfence();  // Ensure write visible

        // Mark slot as ready with sequence number
        atomicExch(&sequence_[slot], tail + 1);
        return true;
    }

    // Pop item from queue (device-side, use view in kernel)
    // Returns true if successful (item written to *out), false if empty
    __device__ bool pop(T* out) {
        while (true) {
            uint32_t head = *head_;
            uint32_t tail = *tail_;

            if (head >= tail) {
                return false;  // Empty
            }

            uint32_t slot = head & mask_;
            uint32_t expected_seq = head + 1;
            uint32_t actual_seq = atomicAdd(&sequence_[slot], 0);

            if (actual_seq != expected_seq) {
                continue;  // Data not yet written, retry
            }

            // Try to claim this slot
            if (atomicCAS(head_, head, head + 1) == head) {
                __threadfence();
                *out = buffer_[slot];
                return true;
            }
            // Another thread claimed it, retry
        }
    }

    // Non-blocking try_pop (device-side, use view in kernel)
    __device__ bool try_pop(T* out) {
        uint32_t head = *head_;
        uint32_t tail = *tail_;

        if (head >= tail) {
            return false;
        }

        uint32_t slot = head & mask_;
        uint32_t expected_seq = head + 1;
        uint32_t actual_seq = atomicAdd(&sequence_[slot], 0);

        if (actual_seq != expected_seq) {
            return false;  // Data not yet written
        }

        if (atomicCAS(head_, head, head + 1) == head) {
            __threadfence();
            *out = buffer_[slot];
            return true;
        }

        return false;  // Contention, caller can retry
    }

    __device__ bool empty() const {
        return *head_ >= *tail_;
    }

    __device__ uint32_t size() const {
        uint32_t h = *head_;
        uint32_t t = *tail_;
        return (t > h) ? (t - h) : 0;
    }

    // Host-side accessors
    uint32_t get_size() const {
        uint32_t h, t;
        CUDA_CHECK(cudaMemcpy(&h, head_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&t, tail_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return (t > h) ? (t - h) : 0;
    }

    bool is_empty() const {
        return get_size() == 0;
    }

    // Push from host (for initial state)
    void host_push(const T& item) {
        uint32_t t;
        CUDA_CHECK(cudaMemcpy(&t, tail_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t slot = t & mask_;
        CUDA_CHECK(cudaMemcpy(&buffer_[slot], &item, sizeof(T), cudaMemcpyHostToDevice));
        // Set sequence number for this slot
        uint32_t seq = t + 1;
        CUDA_CHECK(cudaMemcpy(&sequence_[slot], &seq, sizeof(uint32_t), cudaMemcpyHostToDevice));
        t++;
        CUDA_CHECK(cudaMemcpy(tail_, &t, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Get device-accessible view (POD struct that can be copied to device)
    WorkQueueView<T> get_view() const {
        return {buffer_, capacity_, mask_, head_, tail_, sequence_};
    }
};

// =============================================================================
// Work-Stealing Deque (optional, for better load balancing)
// =============================================================================
// Each block has its own deque; can steal from others when empty.
// Uses Chase-Lev algorithm.

template<typename T>
class WorkStealingDeque {
private:
    T* buffer_;
    uint32_t capacity_;
    uint32_t mask_;

    // Owner pushes/pops from bottom, thieves steal from top
    alignas(128) int64_t* top_;      // Thieves steal from here
    alignas(128) int64_t* bottom_;   // Owner pushes/pops here

public:
    void init(uint32_t capacity) {
        capacity_ = 1;
        while (capacity_ < capacity) capacity_ *= 2;
        mask_ = capacity_ - 1;

        CUDA_CHECK(cudaMalloc(&buffer_, capacity_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&top_, sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&bottom_, sizeof(int64_t)));

        clear();
    }

    void destroy() {
        if (buffer_) cudaFree(buffer_);
        if (top_) cudaFree(top_);
        if (bottom_) cudaFree(bottom_);
    }

    void clear() {
        int64_t zero = 0;
        CUDA_CHECK(cudaMemcpy(top_, &zero, sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bottom_, &zero, sizeof(int64_t), cudaMemcpyHostToDevice));
    }

    // Owner: push to bottom
    __device__ void push(const T& item) {
        int64_t b = *bottom_;
        buffer_[b & mask_] = item;
        __threadfence();
        *bottom_ = b + 1;
    }

    // Owner: pop from bottom
    __device__ bool pop(T* out) {
        int64_t b = *bottom_ - 1;
        *bottom_ = b;
        __threadfence();
        int64_t t = *top_;

        if (t <= b) {
            *out = buffer_[b & mask_];
            if (t == b) {
                // Last item - race with thieves
                if (atomicCAS((unsigned long long*)top_, t, t + 1) != (unsigned long long)t) {
                    // Lost race
                    *bottom_ = t + 1;
                    return false;
                }
                *bottom_ = t + 1;
            }
            return true;
        } else {
            // Empty
            *bottom_ = t;
            return false;
        }
    }

    // Thief: steal from top
    __device__ bool steal(T* out) {
        int64_t t = *top_;
        __threadfence();
        int64_t b = *bottom_;

        if (t < b) {
            *out = buffer_[t & mask_];
            if (atomicCAS((unsigned long long*)top_, t, t + 1) != (unsigned long long)t) {
                return false;  // Lost race
            }
            return true;
        }
        return false;  // Empty
    }

    __device__ bool empty() const {
        return *top_ >= *bottom_;
    }

    __device__ int64_t size() const {
        int64_t b = *bottom_;
        int64_t t = *top_;
        return (b > t) ? (b - t) : 0;
    }
};

// =============================================================================
// Device-accessible Termination Detector View (POD struct)
// =============================================================================
// Uses work-counting pattern:
// - Increment work_count BEFORE pushing work to queue
// - Decrement work_count AFTER finishing work (including spawning children)
// - Termination: work_count reaches 0 AND remains 0 (no bouncing)
//
// IMPORTANT: We don't set done_flag on first reaching 0 because work_count
// can bounce (e.g., push_wait retry may cause temporary 0). Instead, is_done()
// checks both done_flag AND work_count == 0.
struct TerminationDetectorView {
    uint32_t* work_count;  // Pending + in-progress work items
    uint32_t* done_flag;

    // Call BEFORE pushing work to any queue
    __device__ void work_created() {
        atomicAdd(work_count, 1);
    }

    // Call AFTER fully processing a work item (after spawning any children)
    __device__ void work_finished() {
        uint32_t old = atomicSub(work_count, 1);
        if (old == 1) {  // Decremented from 1 to 0 - potentially done
            // Set done_flag, but is_done() will verify work_count is still 0
            atomicExch(done_flag, 1);
            __threadfence();  // Ensure flag is visible to all threads
        }
    }

    __device__ bool is_done() const {
        // Check done_flag first (quick check)
        if (atomicAdd(done_flag, 0) == 0) {
            return false;  // Not signaled yet
        }
        // Verify work_count is actually 0 (handles bouncing case)
        uint32_t count = atomicAdd(work_count, 0);
        if (count > 0) {
            // Work was added after done_flag was set - reset done_flag
            atomicExch(done_flag, 0);
            return false;
        }
        return true;  // Both flag set and count is 0
    }

    __device__ void signal_done() {
        atomicExch(done_flag, 1);
    }
};

// =============================================================================
// Termination Detection
// =============================================================================
// Uses work-counting pattern for race-free termination detection.
// - work_count tracks pending + in-progress work items
// - Increment BEFORE pushing work, decrement AFTER finishing
// - Termination: work_count reaches 0

class TerminationDetector {
private:
    uint32_t* work_count_;  // Pending + in-progress work items
    uint32_t* done_flag_;   // Set to 1 when termination detected

public:
    void init() {
        CUDA_CHECK(cudaMalloc(&work_count_, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&done_flag_, sizeof(uint32_t)));
        clear();
    }

    void destroy() {
        if (work_count_) cudaFree(work_count_);
        if (done_flag_) cudaFree(done_flag_);
    }

    void clear() {
        CUDA_CHECK(cudaMemset(work_count_, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(done_flag_, 0, sizeof(uint32_t)));
    }

    // Host-side: call BEFORE pushing initial work
    void host_work_created() {
        uint32_t current;
        CUDA_CHECK(cudaMemcpy(&current, work_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        current += 1;
        CUDA_CHECK(cudaMemcpy(work_count_, &current, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    bool host_is_done() const {
        uint32_t flag;
        CUDA_CHECK(cudaMemcpy(&flag, done_flag_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return flag != 0;
    }

    uint32_t* done_flag_ptr() { return done_flag_; }

    // Get device-accessible view (POD struct that can be copied to device)
    TerminationDetectorView get_view() const {
        return {work_count_, done_flag_};
    }
};

}  // namespace hypergraph::gpu
