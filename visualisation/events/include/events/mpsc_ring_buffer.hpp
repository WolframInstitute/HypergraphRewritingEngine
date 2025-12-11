#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace viz {

// Lock-free Multi-Producer Single-Consumer ring buffer
// Based on Vyukov's bounded MPSC queue
// - Multiple threads can push concurrently (producers)
// - Single thread drains (consumer)
// - Fixed capacity, bounded memory
// - Returns false on overflow (drops event)
template<typename T, size_t Capacity = 65536>
class MPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    MPSCRingBuffer() : write_index_(0), read_index_(0) {
        // Initialize all slots as empty
        for (size_t i = 0; i < Capacity; ++i) {
            slots_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    // Try to push an item. Returns false if buffer is full.
    // Thread-safe for multiple producers.
    bool try_push(const T& item) {
        Slot* slot;
        size_t pos = write_index_.load(std::memory_order_relaxed);

        for (;;) {
            slot = &slots_[pos & MASK];
            size_t seq = slot->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                // Slot is ready for writing
                if (write_index_.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed)) {
                    break;  // Successfully claimed this slot
                }
                // CAS failed, retry with updated pos
            } else if (diff < 0) {
                // Buffer is full (consumer hasn't caught up)
                return false;
            } else {
                // Another producer took this slot, advance
                pos = write_index_.load(std::memory_order_relaxed);
            }
        }

        // Write the data
        slot->data = item;
        // Publish: mark slot as ready for consumer
        slot->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Try to pop an item. Returns false if buffer is empty.
    // Only safe for single consumer thread.
    bool try_pop(T& item) {
        Slot* slot = &slots_[read_index_ & MASK];
        size_t seq = slot->sequence.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(read_index_ + 1);

        if (diff < 0) {
            // Buffer is empty
            return false;
        }

        // Read the data
        item = slot->data;
        // Mark slot as available for producers
        slot->sequence.store(read_index_ + Capacity, std::memory_order_release);
        ++read_index_;
        return true;
    }

    // Drain all available items, calling callback for each
    // Only safe for single consumer thread.
    template<typename Callback>
    size_t drain(Callback&& callback) {
        size_t count = 0;
        T item;
        while (try_pop(item)) {
            callback(item);
            ++count;
        }
        return count;
    }

    // Drain up to max_count items
    template<typename Callback>
    size_t drain(Callback&& callback, size_t max_count) {
        size_t count = 0;
        T item;
        while (count < max_count && try_pop(item)) {
            callback(item);
            ++count;
        }
        return count;
    }

    // Check if buffer appears empty (approximate, for diagnostics)
    bool empty() const {
        return write_index_.load(std::memory_order_relaxed) == read_index_;
    }

    // Approximate count of items (not exact due to concurrent access)
    size_t size_approx() const {
        size_t w = write_index_.load(std::memory_order_relaxed);
        size_t r = read_index_;
        return w >= r ? w - r : 0;
    }

    static constexpr size_t capacity() { return Capacity; }

    // Reset the buffer (only safe when no producers are active!)
    // WARNING: Only call this when you're certain no threads are writing
    void reset() {
        // Full fence to ensure all prior writes are visible
        std::atomic_thread_fence(std::memory_order_seq_cst);

        read_index_ = 0;
        write_index_.store(0, std::memory_order_release);
        for (size_t i = 0; i < Capacity; ++i) {
            slots_[i].sequence.store(i, std::memory_order_relaxed);
        }

        // Another fence to ensure reset is complete before any subsequent operations
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

private:
    static constexpr size_t MASK = Capacity - 1;

    struct alignas(64) Slot {
        std::atomic<size_t> sequence;
        T data;
    };

    // Pad to avoid false sharing
    alignas(64) std::atomic<size_t> write_index_;
    alignas(64) size_t read_index_;  // Only accessed by consumer, no atomic needed
    alignas(64) Slot slots_[Capacity];
};

// Default event buffer type
struct VizEvent;  // Forward declaration
using VizEventBuffer = MPSCRingBuffer<VizEvent, 65536>;  // 64K slots

} // namespace viz
