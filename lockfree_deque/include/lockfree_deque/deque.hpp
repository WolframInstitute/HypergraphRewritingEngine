#ifndef LOCKFREE_DEQUE_HPP
#define LOCKFREE_DEQUE_HPP

#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>
#include <thread>

namespace lockfree {

// Bounded multi-producer/multi-consumer double-ended queue, lock-free.
//
// State is one 64-bit atomic packed as {tag:32, head:16, tail:16}. A single
// compare-exchange commits both ends at once, so the two ends are safe against each
// other: a pop_front and a pop_back racing for the last element see the same word and
// only one CAS wins. Because both ends move in both directions (push_front decrements
// head, pop_front increments it), the (head,tail) pair recurs, so a monotonic tag is
// incremented on every successful operation: a CAS commits only when nothing changed
// since the load, which defeats ABA (a stale item pointer can never be claimed twice).
// std::atomic<uint64_t> CAS is lock-free on x86-64 and arm64 alike -- no double-width
// CAS and no architecture-specific code. Capacity is limited to 32768 by the 16-bit
// indices, which is ample for a work queue.
//
// Each slot holds one pointer (nullptr = free). An operation checks the slot state
// BEFORE committing (push only into a vacated slot, pop only from a published slot) and
// reports false/nullopt on a transient lag rather than spinning after committing, so
// every operation is non-blocking.
template<typename T>
class Deque {
private:
    static constexpr std::size_t DEFAULT_CAPACITY = 1024;
    static constexpr std::size_t MAX_CAPACITY = 32768;

    static std::size_t round_up_pow2(std::size_t n) {
        std::size_t c = 1;
        while (c < n) c <<= 1;
        return c;
    }

    static constexpr std::uint64_t pack(std::uint32_t tag, std::uint16_t h, std::uint16_t t) {
        return (static_cast<std::uint64_t>(tag) << 32) | (static_cast<std::uint32_t>(h) << 16) | t;
    }
    static constexpr std::uint32_t tag_of(std::uint64_t v)  { return static_cast<std::uint32_t>(v >> 32); }
    static constexpr std::uint16_t head_of(std::uint64_t v) { return static_cast<std::uint16_t>((v >> 16) & 0xffffu); }
    static constexpr std::uint16_t tail_of(std::uint64_t v) { return static_cast<std::uint16_t>(v & 0xffffu); }

    const std::uint32_t capacity_;
    const std::uint32_t mask_;
    alignas(64) std::atomic<std::uint64_t> ht_{0};   // packed {tag, head, tail}
    std::vector<std::atomic<T*>> buffer_;

    // Brief busy-spin then yield, for the blocking variants only. yield() is portable,
    // so there is no architecture-specific pause instruction.
    static void spin_pause(int& n) {
        if (++n < 64) return;
        std::this_thread::yield();
    }

    bool is_full(std::uint64_t v) const {
        return static_cast<std::uint16_t>(tail_of(v) - head_of(v)) >= capacity_;
    }

public:
    explicit Deque(std::size_t cap = DEFAULT_CAPACITY)
        : capacity_(static_cast<std::uint32_t>(round_up_pow2(cap < 4 ? 4 : (cap > MAX_CAPACITY ? MAX_CAPACITY : cap)))),
          mask_(capacity_ - 1),
          buffer_(capacity_) {
        for (auto& slot : buffer_) slot.store(nullptr, std::memory_order_relaxed);
    }

    ~Deque() {
        while (try_pop_front()) {}
    }

    Deque(const Deque&) = delete;
    Deque& operator=(const Deque&) = delete;

    // Non-blocking. Returns false when full, or (transiently) when the target slot is
    // still being drained by a lagging consumer.
    bool try_push_front(T value) {
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        while (true) {
            if (is_full(v)) return false;
            std::uint16_t h = head_of(v), t = tail_of(v);
            std::uint32_t slot = static_cast<std::uint16_t>(h - 1) & mask_;
            if (buffer_[slot].load(std::memory_order_acquire) != nullptr) return false;
            if (ht_.compare_exchange_weak(v, pack(tag_of(v) + 1, h - 1, t),
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
                buffer_[slot].store(new T(std::move(value)), std::memory_order_release);
                return true;
            }
        }
    }

    bool try_push_back(T value) {
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        while (true) {
            if (is_full(v)) return false;
            std::uint16_t h = head_of(v), t = tail_of(v);
            std::uint32_t slot = t & mask_;
            if (buffer_[slot].load(std::memory_order_acquire) != nullptr) return false;
            if (ht_.compare_exchange_weak(v, pack(tag_of(v) + 1, h, t + 1),
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
                buffer_[slot].store(new T(std::move(value)), std::memory_order_release);
                return true;
            }
        }
    }

    // Non-blocking. Returns nullopt when empty, or (transiently) when the front/back
    // slot has not yet been published by a lagging producer.
    std::optional<T> try_pop_front() {
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        while (true) {
            std::uint16_t h = head_of(v), t = tail_of(v);
            if (h == t) return std::nullopt;
            std::uint32_t slot = h & mask_;
            T* item = buffer_[slot].load(std::memory_order_acquire);
            if (item == nullptr) return std::nullopt;
            if (ht_.compare_exchange_weak(v, pack(tag_of(v) + 1, h + 1, t),
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
                buffer_[slot].store(nullptr, std::memory_order_release);
                std::optional<T> result(std::move(*item));
                delete item;
                return result;
            }
        }
    }

    std::optional<T> try_pop_back() {
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        while (true) {
            std::uint16_t h = head_of(v), t = tail_of(v);
            if (h == t) return std::nullopt;
            std::uint32_t slot = static_cast<std::uint16_t>(t - 1) & mask_;
            T* item = buffer_[slot].load(std::memory_order_acquire);
            if (item == nullptr) return std::nullopt;
            if (ht_.compare_exchange_weak(v, pack(tag_of(v) + 1, h, t - 1),
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
                buffer_[slot].store(nullptr, std::memory_order_release);
                std::optional<T> result(std::move(*item));
                delete item;
                return result;
            }
        }
    }

    // Blocking: spin (yielding) until there is room / an item. The item is allocated
    // once and stored only on success, so value is never lost on a failed attempt.
    void push_front(T value) {
        T* item = new T(std::move(value));
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        int n = 0;
        while (true) {
            std::uint16_t h = head_of(v), t = tail_of(v);
            std::uint32_t slot = static_cast<std::uint16_t>(h - 1) & mask_;
            if (is_full(v) || buffer_[slot].load(std::memory_order_acquire) != nullptr) {
                spin_pause(n);
                v = ht_.load(std::memory_order_acquire);
                continue;
            }
            if (ht_.compare_exchange_weak(v, pack(tag_of(v) + 1, h - 1, t),
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
                buffer_[slot].store(item, std::memory_order_release);
                return;
            }
        }
    }

    void push_back(T value) {
        T* item = new T(std::move(value));
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        int n = 0;
        while (true) {
            std::uint16_t h = head_of(v), t = tail_of(v);
            std::uint32_t slot = t & mask_;
            if (is_full(v) || buffer_[slot].load(std::memory_order_acquire) != nullptr) {
                spin_pause(n);
                v = ht_.load(std::memory_order_acquire);
                continue;
            }
            if (ht_.compare_exchange_weak(v, pack(tag_of(v) + 1, h, t + 1),
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
                buffer_[slot].store(item, std::memory_order_release);
                return;
            }
        }
    }

    T pop_front() {
        int n = 0;
        while (true) {
            auto result = try_pop_front();
            if (result.has_value()) return std::move(*result);
            spin_pause(n);
        }
    }

    T pop_back() {
        int n = 0;
        while (true) {
            auto result = try_pop_back();
            if (result.has_value()) return std::move(*result);
            spin_pause(n);
        }
    }

    bool empty() const {
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        return head_of(v) == tail_of(v);
    }

    bool full() const {
        return is_full(ht_.load(std::memory_order_acquire));
    }

    std::size_t size() const {
        std::uint64_t v = ht_.load(std::memory_order_acquire);
        return static_cast<std::uint16_t>(tail_of(v) - head_of(v));
    }

    std::size_t capacity() const { return capacity_; }
};

} // namespace lockfree

#endif // LOCKFREE_DEQUE_HPP
