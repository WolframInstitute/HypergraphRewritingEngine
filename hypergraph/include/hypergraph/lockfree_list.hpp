#ifndef HYPERGRAPH_LOCKFREE_LIST_HPP
#define HYPERGRAPH_LOCKFREE_LIST_HPP

#include <atomic>
#include <memory>

namespace hypergraph {

/**
 * Lock-free singly-linked list for concurrent append operations
 * Optimized for frequent appends and occasional iteration
 */
template<typename T>
class LockfreeList {
private:
    struct Node {
        T data;
        std::atomic<Node*> next{nullptr};

        explicit Node(const T& value) : data(value) {}
        explicit Node(T&& value) : data(std::move(value)) {}

        template<typename... Args>
        explicit Node(Args&&... args) : data(std::forward<Args>(args)...) {}
    };
    
    std::atomic<Node*> head_{nullptr};
    
public:
    LockfreeList() = default;
    
    ~LockfreeList() {
        clear();
    }
    
    // Non-copyable, non-movable for simplicity
    LockfreeList(const LockfreeList&) = delete;
    LockfreeList& operator=(const LockfreeList&) = delete;
    LockfreeList(LockfreeList&&) = delete;
    LockfreeList& operator=(LockfreeList&&) = delete;
    
    /**
     * Append an element to the front of the list (lock-free)
     */
    void push_front(const T& value) {
        Node* new_node = new Node(value);
        Node* current_head = head_.load(std::memory_order_acquire);
        
        do {
            new_node->next.store(current_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(
            current_head, 
            new_node,
            std::memory_order_release,
            std::memory_order_acquire
        ));
    }
    
    /**
     * Append an element to the front of the list (lock-free, move version)
     */
    void push_front(T&& value) {
        Node* new_node = new Node(std::move(value));
        Node* current_head = head_.load(std::memory_order_acquire);
        
        do {
            new_node->next.store(current_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(
            current_head, 
            new_node,
            std::memory_order_release,
            std::memory_order_acquire
        ));
    }

    /**
     * Append an element to the front of the list (universal reference version)
     */
    template<typename U>
    void push_front(U&& value) {
        Node* new_node = new Node(std::forward<U>(value));
        Node* current_head = head_.load(std::memory_order_acquire);

        do {
            new_node->next.store(current_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(
            current_head,
            new_node,
            std::memory_order_release,
            std::memory_order_acquire
        ));
    }

    /**
     * Construct an element in-place at the front of the list (lock-free)
     */
    template<typename... Args>
    void emplace_front(Args&&... args) {
        Node* new_node = new Node(std::forward<Args>(args)...);
        Node* current_head = head_.load(std::memory_order_acquire);

        do {
            new_node->next.store(current_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(
            current_head,
            new_node,
            std::memory_order_release,
            std::memory_order_acquire
        ));
    }

    /**
     * Check if the list is empty
     */
    bool empty() const {
        return head_.load(std::memory_order_acquire) == nullptr;
    }
    
    /**
     * Iterate over all elements (not thread-safe with concurrent modifications)
     * Use only when you know no other threads are modifying the list
     */
    template<typename Func>
    void for_each(Func&& func) const {
        Node* current = head_.load(std::memory_order_acquire);
        while (current != nullptr) {
            func(current->data);
            current = current->next.load(std::memory_order_acquire);
        }
    }
    
    /**
     * Clear all elements (not thread-safe)
     * Use only when you know no other threads are accessing the list
     */
    void clear() {
        Node* current = head_.load(std::memory_order_acquire);
        while (current != nullptr) {
            Node* next = current->next.load(std::memory_order_acquire);
            delete current;
            current = next;
        }
        head_.store(nullptr, std::memory_order_release);
    }
    
    /**
     * Count elements (not thread-safe with concurrent modifications)
     * Use only when you know no other threads are modifying the list
     */
    std::size_t size() const {
        std::size_t count = 0;
        for_each([&count](const T&) { ++count; });
        return count;
    }
};

} // namespace hypergraph

#endif // HYPERGRAPH_LOCKFREE_LIST_HPP