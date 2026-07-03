#ifndef JOB_SYSTEM_WORK_STEALING_DEQUE_HPP
#define JOB_SYSTEM_WORK_STEALING_DEQUE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace job_system {

// Chase-Lev lock-free work-stealing deque (bounded). T must be a pointer type;
// nullptr is the "no item" sentinel.
//
// A single OWNER thread calls push()/pop() on the bottom; any number of THIEF threads
// call steal() on the top. push/pop are wait-free for the owner (no CAS unless the
// deque is down to its last element and racing a thief); steal is lock-free. Memory
// orderings follow Le, Pop, Cohen, Nardelli, "Correct and Efficient Work-Stealing for
// Weak Memory Models" (2013). 64-bit indices and a single-word top CAS keep it
// lock-free on x86-64 and arm64 with no architecture-specific code.
//
// Bounded: push() returns false when full (capacity items) so the caller can fall
// back (e.g. run the task inline) rather than block.
template<typename T>
class WorkStealingDeque {
    static_assert(std::is_pointer<T>::value, "WorkStealingDeque<T> requires a pointer T");

    static std::size_t round_up_pow2(std::size_t n) {
        std::size_t c = 1;
        while (c < n) c <<= 1;
        return c;
    }

    const std::size_t capacity_;
    const std::int64_t mask_;
    alignas(64) std::atomic<std::int64_t> top_{0};      // stolen end (thieves)
    alignas(64) std::atomic<std::int64_t> bottom_{0};   // owner end
    std::vector<std::atomic<T>> buffer_;

public:
    explicit WorkStealingDeque(std::size_t cap = 1024)
        : capacity_(round_up_pow2(cap < 2 ? 2 : cap)),
          mask_(static_cast<std::int64_t>(capacity_) - 1),
          buffer_(capacity_) {
        for (auto& slot : buffer_) slot.store(nullptr, std::memory_order_relaxed);
    }

    WorkStealingDeque(const WorkStealingDeque&) = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    std::size_t capacity() const { return capacity_; }

    // OWNER only. Returns false if the deque is full. The item is published with a
    // release store on its slot (paired with the acquire load in steal), so the
    // payload happens-before its consumer without relying on a standalone fence.
    bool push(T x) {
        std::int64_t b = bottom_.load(std::memory_order_relaxed);
        std::int64_t t = top_.load(std::memory_order_acquire);
        if (b - t >= static_cast<std::int64_t>(capacity_)) return false;  // full
        buffer_[b & mask_].store(x, std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_release);
        return true;
    }

    // OWNER only. Returns nullptr if empty.
    T pop() {
        std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);  // order bottom store vs top load
        std::int64_t t = top_.load(std::memory_order_relaxed);
        T x = nullptr;
        if (t <= b) {
            x = buffer_[b & mask_].load(std::memory_order_relaxed);
            if (t == b) {
                // Last element: race a concurrent thief for it.
                if (!top_.compare_exchange_strong(t, t + 1,
                        std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    x = nullptr;  // a thief won
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
        } else {
            // Deque was already empty.
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        return x;
    }

    // THIEF (any thread). Returns nullptr if empty or it lost the race.
    T steal() {
        std::int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);  // order top load vs bottom load
        std::int64_t b = bottom_.load(std::memory_order_acquire);
        if (t < b) {
            T x = buffer_[t & mask_].load(std::memory_order_acquire);
            if (top_.compare_exchange_strong(t, t + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return x;
            }
        }
        return nullptr;  // empty or lost the race
    }

    // Observers (approximate under concurrency).
    bool empty() const {
        return bottom_.load(std::memory_order_acquire) <= top_.load(std::memory_order_acquire);
    }
    std::size_t size() const {
        std::int64_t b = bottom_.load(std::memory_order_acquire);
        std::int64_t t = top_.load(std::memory_order_acquire);
        return b > t ? static_cast<std::size_t>(b - t) : 0;
    }
};

} // namespace job_system

#endif // JOB_SYSTEM_WORK_STEALING_DEQUE_HPP
