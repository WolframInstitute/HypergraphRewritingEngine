#ifndef LOCKFREE_DEQUE_HPP
#define LOCKFREE_DEQUE_HPP

#include <atomic>
#include <memory>
#include <optional>
#include <vector>
#include <thread>
#include <stdexcept>
#include <cstdio>

namespace lockfree {

template<typename T>
class Deque {
private:
    static constexpr std::size_t DEFAULT_CAPACITY = 1024;
    
    struct Slot {
        std::atomic<T*> data{nullptr};
        
        Slot() = default;
        
        // Make Slot movable
        Slot(Slot&& other) noexcept {
            data.store(other.data.exchange(nullptr));
        }
        
        Slot& operator=(Slot&& other) noexcept {
            if (this != &other) {
                data.store(other.data.exchange(nullptr));
            }
            return *this;
        }
        
        // Delete copy operations
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
    };
    
    const std::size_t capacity;
    const std::size_t mask;
    alignas(64) std::atomic<std::size_t> head{0};
    alignas(64) std::atomic<std::size_t> tail{0};
    std::vector<Slot> buffer;

    std::size_t next(std::size_t index) const {
        return (index + 1) & mask;
    }
    
    std::size_t prev(std::size_t index) const {
        return (index - 1) & mask;
    }

public:
    explicit Deque(std::size_t cap = DEFAULT_CAPACITY) 
        : capacity(std::max(cap, static_cast<std::size_t>(4))), 
          mask(capacity - 1),
          buffer(capacity) {
        // Ensure capacity is power of 2
        std::size_t actual_cap = 1;
        while (actual_cap < capacity) {
            actual_cap <<= 1;
        }
        const_cast<std::size_t&>(capacity) = actual_cap;
        const_cast<std::size_t&>(mask) = actual_cap - 1;
        buffer.resize(actual_cap);
    }

    ~Deque() {
        while (try_pop_front()) {}
    }

    // Non-blocking operations that return optional
    bool try_push_front(T value) {
#ifdef DEQUE_DEBUG
        static std::atomic<int> push_front_count{0};
        int my_push = push_front_count.fetch_add(1);
        printf("[Deque] try_push_front #%d start\n", my_push);
#endif
        T* item = new T(std::move(value));
        
        while (true) {
            std::size_t h = head.load(std::memory_order_acquire);
            std::size_t new_head = prev(h);
            std::size_t t = tail.load(std::memory_order_acquire);
            
#ifdef DEQUE_DEBUG
            printf("[Deque] try_push_front: head=%zu, new_head=%zu, tail=%zu\n", h, new_head, t);
#endif
            
            // Check if full (leave one slot empty to distinguish full from empty)
            if (new_head == t) {
#ifdef DEQUE_DEBUG
                printf("[Deque] try_push_front: FULL\n");
#endif
                delete item;
                return false; // Full
            }
            
            // Try to atomically reserve the slot by updating head
            if (head.compare_exchange_weak(h, new_head, 
                                          std::memory_order_acq_rel, 
                                          std::memory_order_relaxed)) {
                // We successfully reserved the slot, now place the item
                buffer[new_head].data.store(item, std::memory_order_release);
#ifdef DEQUE_DEBUG
                printf("[Deque] try_push_front #%d SUCCESS at slot %zu\n", my_push, new_head);
#endif
                return true;
            }
            
#ifdef DEQUE_DEBUG
            printf("[Deque] try_push_front CAS failed, retrying\n");
#endif
            // CAS failed, reload values and retry
        }
    }

    bool try_push_back(T value) {
        T* item = new T(std::move(value));
        
        while (true) {
            std::size_t t = tail.load(std::memory_order_acquire);
            std::size_t h = head.load(std::memory_order_acquire);
            
            // Check if full (leave one slot empty to distinguish full from empty)
            if (next(t) == h) {
                delete item;
                return false; // Full
            }
            
            // Try to atomically reserve the slot by updating tail
            if (tail.compare_exchange_weak(t, next(t), 
                                          std::memory_order_acq_rel, 
                                          std::memory_order_relaxed)) {
                // We successfully reserved the slot, now place the item
                buffer[t].data.store(item, std::memory_order_release);
                return true;
            }
            
            // CAS failed, reload values and retry
        }
    }

    std::optional<T> try_pop_front() {
#ifdef DEQUE_DEBUG
        static std::atomic<int> pop_front_count{0};
        int my_pop = pop_front_count.fetch_add(1);
        printf("[Deque] try_pop_front #%d start\n", my_pop);
#endif
        int retries = 0;
        const int max_retries = 1000;
        
        while (retries < max_retries) {
            std::size_t h = head.load(std::memory_order_acquire);
            std::size_t t = tail.load(std::memory_order_acquire);
            
#ifdef DEQUE_DEBUG
            printf("[Deque] try_pop_front: head=%zu, tail=%zu\n", h, t);
#endif
            
            if (h == t) {
#ifdef DEQUE_DEBUG
                printf("[Deque] try_pop_front #%d: EMPTY\n", my_pop);
#endif
                return std::nullopt; // Empty
            }
            
            // Try to atomically extract the item from the slot at head position
            T* expected = nullptr;
            T* item = buffer[h].data.load(std::memory_order_acquire);
            
#ifdef DEQUE_DEBUG
            printf("[Deque] try_pop_front: slot[%zu] has item=%p\n", h, (void*)item);
#endif
            
            // If slot is empty, continue to next iteration
            if (item == nullptr) {
                retries++;
#ifdef DEQUE_DEBUG
                printf("[Deque] try_pop_front: slot empty, retry %d\n", retries);
#endif
                std::this_thread::yield();
                continue;
            }
            
            // Try to atomically take ownership of the item before advancing head
            if (buffer[h].data.compare_exchange_weak(item, nullptr, 
                                                    std::memory_order_acq_rel, 
                                                    std::memory_order_relaxed)) {
#ifdef DEQUE_DEBUG
                printf("[Deque] try_pop_front: got item from slot %zu\n", h);
#endif
                // We got the item, now try to advance head
                std::size_t new_head = next(h);
                if (head.compare_exchange_weak(h, new_head, 
                                              std::memory_order_acq_rel, 
                                              std::memory_order_relaxed)) {
                    // Successfully advanced head and got item
                    std::optional<T> result = std::move(*item);
                    delete item;
#ifdef DEQUE_DEBUG
                    printf("[Deque] try_pop_front #%d SUCCESS\n", my_pop);
#endif
                    return result;
                } else {
                    // Failed to advance head, but we took the item - put it back
#ifdef DEQUE_DEBUG
                    printf("[Deque] try_pop_front: failed to advance head, putting item back\n");
#endif
                    buffer[h].data.store(item, std::memory_order_release);
                    retries++;
                    continue;
                }
            }
            
            // Failed to get the item, retry
#ifdef DEQUE_DEBUG
            printf("[Deque] try_pop_front: CAS failed to get item\n");
#endif
            retries++;
        }
        
#ifdef DEQUE_DEBUG
        printf("[Deque] try_pop_front #%d: MAX RETRIES EXCEEDED\n", my_pop);
#endif
        return std::nullopt; // Exceeded retries
    }

    std::optional<T> try_pop_back() {
        int retries = 0;
        const int max_retries = 1000;
        
        while (retries < max_retries) {
            std::size_t t = tail.load(std::memory_order_acquire);
            std::size_t h = head.load(std::memory_order_acquire);
            
            if (h == t) {
                return std::nullopt; // Empty
            }
            
            std::size_t back_pos = prev(t);
            
            // Try to atomically extract the item from the slot at back position
            T* item = buffer[back_pos].data.load(std::memory_order_acquire);
            
            // If slot is empty, continue to next iteration
            if (item == nullptr) {
                retries++;
                std::this_thread::yield();
                continue;
            }
            
            // Try to atomically take ownership of the item before moving tail
            if (buffer[back_pos].data.compare_exchange_weak(item, nullptr, 
                                                           std::memory_order_acq_rel, 
                                                           std::memory_order_relaxed)) {
                // We got the item, now try to move tail back
                if (tail.compare_exchange_weak(t, back_pos, 
                                              std::memory_order_acq_rel, 
                                              std::memory_order_relaxed)) {
                    // Successfully moved tail back and got item
                    std::optional<T> result = std::move(*item);
                    delete item;
                    return result;
                } else {
                    // Failed to move tail back, but we took the item - put it back
                    buffer[back_pos].data.store(item, std::memory_order_release);
                    retries++;
                    continue;
                }
            }
            
            // Failed to get the item, retry
            retries++;
        }
        
        return std::nullopt; // Exceeded retries
    }

    // Blocking operations that always succeed
    void push_front(T value) {
        while (!try_push_front(std::move(value))) {
            std::this_thread::yield();
        }
    }

    void push_back(T value) {
        while (!try_push_back(std::move(value))) {
            std::this_thread::yield();
        }
    }

    T pop_front() {
        while (true) {
            auto result = try_pop_front();
            if (result.has_value()) {
                return std::move(*result);
            }
            std::this_thread::yield();
        }
    }

    T pop_back() {
        while (true) {
            auto result = try_pop_back();
            if (result.has_value()) {
                return std::move(*result);
            }
            std::this_thread::yield();
        }
    }

    bool empty() const {
        std::size_t h = head.load(std::memory_order_acquire);
        std::size_t t = tail.load(std::memory_order_acquire);
        return h == t;
    }
    
    bool full() const {
        std::size_t h = head.load(std::memory_order_acquire);
        std::size_t t = tail.load(std::memory_order_acquire);
        return next(t) == h;
    }
};

} // namespace lockfree

#endif // LOCKFREE_DEQUE_HPP