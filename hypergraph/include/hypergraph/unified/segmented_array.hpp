#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <thread>

namespace hypergraph::unified {

// =============================================================================
// SegmentedArray<T>: Append-only array with O(1) random access
// =============================================================================
//
// Array of fixed-size segments. Never reallocates existing segments, so
// pointers to elements remain stable. New segments allocated on demand.
//
// Thread safety: Lock-free append via atomic count. Safe concurrent reads.
//
// Usage:
//   SegmentedArray<Edge> edges;
//   EdgeId id = edges.emplace(arena, ...);  // Returns index
//   Edge& e = edges[id];                    // O(1) access
//

template<typename T>
class SegmentedArray {
public:
    static constexpr size_t DEFAULT_SEGMENT_SIZE = 1024;
    static constexpr size_t MAX_SEGMENTS = 4096;  // Supports up to 4M elements with 1K segments

    explicit SegmentedArray(size_t segment_size = DEFAULT_SEGMENT_SIZE)
        : segment_size_(segment_size)
        , count_(0)
        , initialized_count_(0) {
        // Initialize segment pointers to null
        for (size_t i = 0; i < MAX_SEGMENTS; ++i) {
            segments_[i].store(nullptr, std::memory_order_relaxed);
        }
    }

    ~SegmentedArray() {
        // Note: We don't free segments here - they're arena-allocated
        // If using heap allocation, would need to track and free
    }

    // Non-copyable, non-movable
    SegmentedArray(const SegmentedArray&) = delete;
    SegmentedArray& operator=(const SegmentedArray&) = delete;
    SegmentedArray(SegmentedArray&&) = delete;
    SegmentedArray& operator=(SegmentedArray&&) = delete;

    // Append a new element, return its index
    // Thread-safe: ensures segment and element are initialized before count is incremented
    template<typename Arena, typename... Args>
    uint32_t emplace(Arena& arena, Args&&... args) {
        // First, claim an index
        uint32_t idx = count_.fetch_add(1, std::memory_order_acq_rel);

        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;

        // Ensure segment exists - this must complete before we increment initialized_count_
        T* segment = get_or_create_segment(seg_idx, arena);

        // Construct the element
        new (&segment[offset]) T(std::forward<Args>(args)...);

        // Signal that this index is now fully initialized
        // (increment initialized count up to this index)
        uint32_t expected_init = idx;
        while (!initialized_count_.compare_exchange_weak(
                expected_init, idx + 1,
                std::memory_order_release,
                std::memory_order_acquire)) {
            if (expected_init > idx) {
                // Someone else already advanced past us
                break;
            }
            expected_init = idx;
            // Spin - another thread needs to finish first
            std::this_thread::yield();
        }

        return idx;
    }

    // Access element by index - O(1)
    T& operator[](uint32_t idx) {
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        T* segment = segments_[seg_idx].load(std::memory_order_acquire);
        return segment[offset];
    }

    const T& operator[](uint32_t idx) const {
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        T* segment = segments_[seg_idx].load(std::memory_order_acquire);
        return segment[offset];
    }

    // Get pointer to element (may be null if not yet allocated)
    T* get(uint32_t idx) {
        if (idx >= count_.load(std::memory_order_acquire)) {
            return nullptr;
        }
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        T* segment = segments_[seg_idx].load(std::memory_order_acquire);
        return segment ? &segment[offset] : nullptr;
    }

    const T* get(uint32_t idx) const {
        if (idx >= count_.load(std::memory_order_acquire)) {
            return nullptr;
        }
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        T* segment = segments_[seg_idx].load(std::memory_order_acquire);
        return segment ? &segment[offset] : nullptr;
    }

    // Current number of elements (returns only fully initialized elements)
    uint32_t size() const {
        return initialized_count_.load(std::memory_order_acquire);
    }

    // Number of claimed slots (may include uninitialized)
    uint32_t claimed_size() const {
        return count_.load(std::memory_order_acquire);
    }

    bool empty() const {
        return size() == 0;
    }

    // Ensure array has at least 'required' elements, thread-safe
    // Waits for all elements to be fully initialized
    template<typename Arena>
    void ensure_size(uint32_t required, Arena& arena) {
        // First, claim enough slots
        while (claimed_size() < required) {
            emplace(arena);
        }
        // Wait for all required slots to be initialized
        while (size() < required) {
            std::this_thread::yield();
        }
    }

    // Iterate over all elements
    template<typename F>
    void for_each(F&& f) {
        uint32_t n = size();
        for (uint32_t i = 0; i < n; ++i) {
            f((*this)[i]);
        }
    }

    template<typename F>
    void for_each(F&& f) const {
        uint32_t n = size();
        for (uint32_t i = 0; i < n; ++i) {
            f((*this)[i]);
        }
    }

    // Iterate with index
    template<typename F>
    void for_each_indexed(F&& f) {
        uint32_t n = size();
        for (uint32_t i = 0; i < n; ++i) {
            f(i, (*this)[i]);
        }
    }

    template<typename F>
    void for_each_indexed(F&& f) const {
        uint32_t n = size();
        for (uint32_t i = 0; i < n; ++i) {
            f(i, (*this)[i]);
        }
    }

private:
    template<typename Arena>
    T* get_or_create_segment(size_t seg_idx, Arena& arena) {
        T* segment = segments_[seg_idx].load(std::memory_order_acquire);
        if (segment) {
            return segment;
        }

        // Need to allocate new segment
        T* new_segment = arena.template allocate_array<T>(segment_size_);

        // Try to install it
        T* expected = nullptr;
        if (segments_[seg_idx].compare_exchange_strong(
                expected, new_segment,
                std::memory_order_release,
                std::memory_order_acquire)) {
            return new_segment;
        }

        // Another thread beat us - use theirs
        // Our allocation is wasted but will be cleaned up with arena
        return expected;
    }

    size_t segment_size_;
    std::atomic<uint32_t> count_;             // Number of claimed slots
    std::atomic<uint32_t> initialized_count_; // Number of fully initialized slots
    std::atomic<T*> segments_[MAX_SEGMENTS];
};

// =============================================================================
// SingleThreadedSegmentedArray<T>: Non-concurrent version
// =============================================================================
//
// Same as SegmentedArray but without atomics. Use when single-threaded access
// is guaranteed (e.g., during sequential phases).
//

template<typename T>
class SingleThreadedSegmentedArray {
public:
    static constexpr size_t DEFAULT_SEGMENT_SIZE = 1024;
    static constexpr size_t MAX_SEGMENTS = 4096;

    explicit SingleThreadedSegmentedArray(size_t segment_size = DEFAULT_SEGMENT_SIZE)
        : segment_size_(segment_size)
        , count_(0) {
        for (size_t i = 0; i < MAX_SEGMENTS; ++i) {
            segments_[i] = nullptr;
        }
    }

    // Non-copyable, non-movable
    SingleThreadedSegmentedArray(const SingleThreadedSegmentedArray&) = delete;
    SingleThreadedSegmentedArray& operator=(const SingleThreadedSegmentedArray&) = delete;
    SingleThreadedSegmentedArray(SingleThreadedSegmentedArray&&) = delete;
    SingleThreadedSegmentedArray& operator=(SingleThreadedSegmentedArray&&) = delete;

    template<typename Arena, typename... Args>
    uint32_t emplace(Arena& arena, Args&&... args) {
        uint32_t idx = count_++;

        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;

        if (!segments_[seg_idx]) {
            segments_[seg_idx] = arena.template allocate_array<T>(segment_size_);
        }

        new (&segments_[seg_idx][offset]) T(std::forward<Args>(args)...);
        return idx;
    }

    T& operator[](uint32_t idx) {
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        return segments_[seg_idx][offset];
    }

    const T& operator[](uint32_t idx) const {
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        return segments_[seg_idx][offset];
    }

    T* get(uint32_t idx) {
        if (idx >= count_) return nullptr;
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        return segments_[seg_idx] ? &segments_[seg_idx][offset] : nullptr;
    }

    const T* get(uint32_t idx) const {
        if (idx >= count_) return nullptr;
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;
        return segments_[seg_idx] ? &segments_[seg_idx][offset] : nullptr;
    }

    uint32_t size() const { return count_; }
    bool empty() const { return count_ == 0; }

    template<typename F>
    void for_each(F&& f) {
        for (uint32_t i = 0; i < count_; ++i) {
            f((*this)[i]);
        }
    }

    template<typename F>
    void for_each(F&& f) const {
        for (uint32_t i = 0; i < count_; ++i) {
            f((*this)[i]);
        }
    }

    template<typename F>
    void for_each_indexed(F&& f) {
        for (uint32_t i = 0; i < count_; ++i) {
            f(i, (*this)[i]);
        }
    }

private:
    size_t segment_size_;
    uint32_t count_;
    T* segments_[MAX_SEGMENTS];
};

}  // namespace hypergraph::unified