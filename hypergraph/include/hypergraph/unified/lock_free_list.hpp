#pragma once

#include <atomic>
#include <cstddef>

namespace hypergraph::unified {

// =============================================================================
// LockFreeList<T>: Append-only lock-free linked list
// =============================================================================
//
// Simple lock-free linked list supporting only prepend (push to head).
// Nodes allocated from arena - never freed individually.
// Iteration is safe concurrent with appends (snapshot semantics on head).
//
// Thread safety: Lock-free push, safe concurrent iteration.
//
// Usage:
//   LockFreeList<EventId> consumers;
//   consumers.push(event_id, arena);
//   consumers.for_each([](EventId e) { ... });
//

template<typename T>
class LockFreeList {
public:
    struct Node {
        T value;
        Node* prev;  // Points to previously inserted node (older)

        Node(const T& v, Node* p) : value(v), prev(p) {}
    };

    LockFreeList() : head_(nullptr) {}

    // Non-copyable, movable
    LockFreeList(const LockFreeList&) = delete;
    LockFreeList& operator=(const LockFreeList&) = delete;

    LockFreeList(LockFreeList&& other) noexcept
        : head_(other.head_.load(std::memory_order_relaxed)) {
        other.head_.store(nullptr, std::memory_order_relaxed);
    }

    LockFreeList& operator=(LockFreeList&& other) noexcept {
        head_.store(other.head_.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        other.head_.store(nullptr, std::memory_order_relaxed);
        return *this;
    }

    // Push value to front of list (lock-free)
    template<typename Arena>
    void push(const T& value, Arena& arena) {
        Node* new_node = arena.template create<Node>(value, nullptr);

        Node* old_head = head_.load(std::memory_order_acquire);
        do {
            new_node->prev = old_head;
        } while (!head_.compare_exchange_weak(
            old_head, new_node,
            std::memory_order_release,
            std::memory_order_acquire));
    }

    // Iterate over all elements (newest to oldest)
    // Safe to call concurrently with push - sees snapshot at call time
    template<typename F>
    void for_each(F&& f) const {
        Node* node = head_.load(std::memory_order_acquire);
        while (node) {
            f(node->value);
            node = node->prev;
        }
    }

    // Iterate with early termination
    // Return false from f to stop iteration
    template<typename F>
    bool for_each_while(F&& f) const {
        Node* node = head_.load(std::memory_order_acquire);
        while (node) {
            if (!f(node->value)) {
                return false;
            }
            node = node->prev;
        }
        return true;
    }

    // Check if list contains value
    bool contains(const T& value) const {
        bool found = false;
        for_each([&](const T& v) {
            if (v == value) {
                found = true;
            }
        });
        return found;
    }

    // Count elements (O(n))
    size_t size() const {
        size_t count = 0;
        for_each([&](const T&) { ++count; });
        return count;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == nullptr;
    }

    // Get head pointer (for advanced usage)
    const Node* head() const {
        return head_.load(std::memory_order_acquire);
    }

private:
    std::atomic<Node*> head_;
};

// =============================================================================
// SingleThreadedList<T>: Non-concurrent version
// =============================================================================
//
// Same interface as LockFreeList but without atomics.
// Use when single-threaded access is guaranteed.
//

template<typename T>
class SingleThreadedList {
public:
    struct Node {
        T value;
        Node* prev;

        Node(const T& v, Node* p) : value(v), prev(p) {}
    };

    SingleThreadedList() : head_(nullptr) {}

    // Non-copyable, movable
    SingleThreadedList(const SingleThreadedList&) = delete;
    SingleThreadedList& operator=(const SingleThreadedList&) = delete;

    SingleThreadedList(SingleThreadedList&& other) noexcept
        : head_(other.head_) {
        other.head_ = nullptr;
    }

    SingleThreadedList& operator=(SingleThreadedList&& other) noexcept {
        head_ = other.head_;
        other.head_ = nullptr;
        return *this;
    }

    template<typename Arena>
    void push(const T& value, Arena& arena) {
        Node* new_node = arena.template create<Node>(value, head_);
        head_ = new_node;
    }

    template<typename F>
    void for_each(F&& f) const {
        Node* node = head_;
        while (node) {
            f(node->value);
            node = node->prev;
        }
    }

    template<typename F>
    bool for_each_while(F&& f) const {
        Node* node = head_;
        while (node) {
            if (!f(node->value)) {
                return false;
            }
            node = node->prev;
        }
        return true;
    }

    bool contains(const T& value) const {
        Node* node = head_;
        while (node) {
            if (node->value == value) return true;
            node = node->prev;
        }
        return false;
    }

    size_t size() const {
        size_t count = 0;
        Node* node = head_;
        while (node) {
            ++count;
            node = node->prev;
        }
        return count;
    }

    bool empty() const {
        return head_ == nullptr;
    }

    const Node* head() const {
        return head_;
    }

private:
    Node* head_;
};

}  // namespace hypergraph::unified
