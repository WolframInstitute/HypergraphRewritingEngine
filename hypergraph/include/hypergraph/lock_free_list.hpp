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

    // Push value to front of list (lock-free, wait-free)
    //
    // MEMORY ORDERING:
    // 1. Node construction happens-before release fence
    // 2. Release fence happens-before release CAS
    // 3. Release CAS synchronizes-with acquire load in for_each()
    // 4. Therefore: node data is visible to threads that see this node
    template<typename Arena>
    void push(const T& value, Arena& arena) {
        Node* new_node = arena.template create<Node>(value, nullptr);

        // CRITICAL: Release fence ensures node construction (including T's copy)
        // is visible before the node pointer becomes visible via the CAS.
        // Without this, readers may see the node pointer but not the node data.
        std::atomic_thread_fence(std::memory_order_release);

        Node* old_head = head_.load(std::memory_order_acquire);
        do {
            new_node->prev = old_head;
        } while (!head_.compare_exchange_weak(
            old_head, new_node,
            std::memory_order_release,
            std::memory_order_acquire));
    }

    // Iterate over all elements (newest to oldest)
    // Keeps checking for new nodes until list is stable - no snapshot batching
    //
    // MEMORY ORDERING:
    // The acquire load on head_ synchronizes-with the release CAS in push().
    // The acquire fence ensures we see ALL writes that happened-before the push,
    // including any data the pushed value references (e.g., if T is an ID, the
    // data at that ID must be visible).
    //
    // Synchronization chain:
    // 1. Writer: constructs data → release fence → push() → CAS(head_, release)
    // 2. Reader: head_.load(acquire) → acquire fence → access data via value
    // 3. Writer's release fence synchronizes-with reader's acquire fence
    template<typename F>
    void for_each(F&& f) const {
        Node* seen_up_to = nullptr;
        while (true) {
            Node* current_head = head_.load(std::memory_order_acquire);
            if (current_head == seen_up_to) break;  // No new nodes since last pass

            // CRITICAL: Acquire fence ensures all writes that happened-before the
            // push (including data referenced by node values) are visible.
            // This pairs with the release fence in push().
            std::atomic_thread_fence(std::memory_order_acquire);

            // Walk from current_head down to seen_up_to (exclusive)
            Node* node = current_head;
            while (node != seen_up_to) {
                f(node->value);
                node = node->prev;
            }
            seen_up_to = current_head;
        }
    }

    // Iterate with early termination
    // Return false from f to stop iteration
    // Keeps checking for new nodes until list is stable or terminated
    //
    // MEMORY ORDERING: Same as for_each() - acquire fence pairs with release fence in push()
    template<typename F>
    bool for_each_while(F&& f) const {
        Node* seen_up_to = nullptr;
        while (true) {
            Node* current_head = head_.load(std::memory_order_acquire);
            if (current_head == seen_up_to) break;

            // CRITICAL: Acquire fence ensures all writes that happened-before the
            // push (including data referenced by node values) are visible.
            std::atomic_thread_fence(std::memory_order_acquire);

            Node* node = current_head;
            while (node != seen_up_to) {
                if (!f(node->value)) {
                    return false;
                }
                node = node->prev;
            }
            seen_up_to = current_head;
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
