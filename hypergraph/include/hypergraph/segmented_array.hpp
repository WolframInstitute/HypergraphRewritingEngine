#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace hypergraph {

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
        , count_(0) {
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
    // Thread-safe, lock-free
    //
    // MEMORY ORDERING:
    // 1. Element construction happens-before release fence
    // 2. Release fence happens-before segment store (release)
    // 3. Segment store (release) synchronizes-with segment load (acquire) in operator[]
    // 4. Therefore: element data is visible to threads that load the segment
    //
    // IMPORTANT: If iterating by size(), callers must handle the case where
    // the segment might not yet be visible. Use get() which returns nullptr
    // for not-yet-ready slots, or only access indices returned by emplace().
    template<typename Arena, typename... Args>
    uint32_t emplace(Arena& arena, Args&&... args) {
        // Claim an index
        uint32_t idx = count_.fetch_add(1, std::memory_order_relaxed);

        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;

        // Ensure THIS segment exists (lock-free, may race with other threads)
        // get_or_create_segment uses CAS with release semantics
        T* segment = get_or_create_segment(seg_idx, arena);

        // Pre-create next segment when nearing end of current
        // This reduces (but doesn't eliminate) the window for the next segment
        if (offset >= segment_size_ - 1) {
            get_or_create_segment(seg_idx + 1, arena);
        }

        // Construct the element in place
        new (&segment[offset]) T(std::forward<Args>(args)...);

        // CRITICAL: Release fence ensures element construction is complete and visible
        // before the segment pointer store. Without this, readers may see the segment
        // pointer but not the element data due to store reordering.
        std::atomic_thread_fence(std::memory_order_release);

        // Store segment pointer with release (pairs with acquire in operator[])
        segments_[seg_idx].store(segment, std::memory_order_release);

        return idx;
    }

    // Access element by index - O(1)
    //
    // MEMORY ORDERING:
    // We synchronize via count_. The thread that constructs element idx does:
    //   1. construct element
    //   2. release fence
    //   3. CAS count_ to at least idx+1 (release)
    //
    // Readers wait for count_ > idx (acquire), which synchronizes with the
    // constructor's CAS. This ensures the element data is visible.
    //
    // This is necessary because multiple threads share the same segment pointer,
    // so synchronizing on the segment alone doesn't synchronize with a specific
    // element's constructor.
    T& operator[](uint32_t idx) {
        // Wait until element is constructed (count_ > idx)
        // The acquire load syncs with the release CAS in emplace/emplace_at
        while (count_.load(std::memory_order_acquire) <= idx) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
            #endif
        }

        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;

        // CRITICAL: Must use acquire to see segment pointer stored before count_ was updated.
        // The count_ acquire establishes a synchronizes-with relationship with emplace_at's
        // release CAS, but we still need acquire here to see the segment store that
        // happened-before that CAS. Relaxed could return a stale null.
        T* segment = segments_[seg_idx].load(std::memory_order_acquire);

        return segment[offset];
    }

    const T& operator[](uint32_t idx) const {
        while (count_.load(std::memory_order_acquire) <= idx) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
            #endif
        }

        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;

        // CRITICAL: Must use acquire to see segment pointer stored before count_ was updated.
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

    // Current number of elements
    uint32_t size() const {
        return count_.load(std::memory_order_acquire);
    }

    bool empty() const {
        return size() == 0;
    }

    // Ensure array has at least 'required' elements, thread-safe, wait-free
    // Each thread only emplaces what it needs - no waiting for other threads
    template<typename Arena>
    void ensure_size(uint32_t required, Arena& arena) {
        while (size() < required) {
            emplace(arena);
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

    // Ensure element at index is accessible without constructing it
    // Used when elements are externally indexed (e.g., vertex IDs) and default
    // construction is sufficient. The segment allocation already default-constructs
    // all elements, so this just ensures the segment exists.
    //
    // Thread-safe: Multiple threads can call this concurrently for the same or
    // different indices. Only one will create the segment, others will use it.
    template<typename Arena>
    T& get_or_default(uint32_t idx, Arena& arena) {
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;

        // Ensure segment exists (thread-safe via CAS in get_or_create_segment)
        // When segment is created, all elements are default-constructed by arena
        T* segment = get_or_create_segment(seg_idx, arena);

        // Update count_ to at least idx+1 for iteration purposes
        // This is safe because we only increase, never decrease
        uint32_t expected = count_.load(std::memory_order_relaxed);
        while (expected <= idx) {
            if (count_.compare_exchange_weak(expected, idx + 1,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                break;
            }
        }

        return segment[offset];
    }

    // Construct element directly at a specific index
    // Used when the index is managed by an external counter (e.g., edge IDs)
    // This avoids the race condition in ensure_size/emplace where another thread
    // might be constructing the same slot
    //
    // MEMORY ORDERING:
    // 1. Element construction happens-before release fence
    // 2. Release fence happens-before segment store (release)
    // 3. Segment store (release) synchronizes-with segment load (acquire) in operator[]
    // 4. Therefore: element data is visible to threads that load the segment
    template<typename Arena, typename... Args>
    void emplace_at(uint32_t idx, Arena& arena, Args&&... args) {
        size_t seg_idx = idx / segment_size_;
        size_t offset = idx % segment_size_;

        // Ensure segment exists (thread-safe)
        T* segment = get_or_create_segment(seg_idx, arena);

        // Pre-create next segment when nearing end of current
        if (offset >= segment_size_ - 1) {
            get_or_create_segment(seg_idx + 1, arena);
        }

        // Construct the element directly with provided arguments
        new (&segment[offset]) T(std::forward<Args>(args)...);

        // CRITICAL: Release fence ensures element construction is complete and visible
        // before the segment pointer store. Without this, readers may see the segment
        // pointer but not the element data due to store reordering.
        std::atomic_thread_fence(std::memory_order_release);

        // Store segment pointer with release (pairs with acquire in operator[])
        // Note: segments_[seg_idx] was already stored in get_or_create_segment,
        // but that was BEFORE construction. This store creates the synchronization
        // point AFTER construction.
        segments_[seg_idx].store(segment, std::memory_order_release);

        // Update count_ to at least idx+1 for iteration purposes
        // This is safe because we only increase, never decrease
        uint32_t expected = count_.load(std::memory_order_relaxed);
        while (expected <= idx) {
            if (count_.compare_exchange_weak(expected, idx + 1,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                break;
            }
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
    std::atomic<uint32_t> count_;
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

}  // namespace hypergraph