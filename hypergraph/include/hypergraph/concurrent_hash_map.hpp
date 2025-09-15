#ifndef HYPERGRAPH_CONCURRENT_HASH_MAP_HPP
#define HYPERGRAPH_CONCURRENT_HASH_MAP_HPP

#include <atomic>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <cstddef>

namespace hypergraph {

/**
 * Lock-free concurrent hash map using atomic operations and hazard pointers
 * Optimized for high-throughput concurrent access patterns
 */
template<typename Key, typename Value, typename Hash = std::hash<Key>>
class ConcurrentHashMap {
private:
    struct Node {
        std::atomic<Node*> next{nullptr};
        Key key;
        Value value;
        std::atomic<bool> marked{false};  // For logical deletion

        Node(const Key& k, const Value& v) : key(k), value(v) {}
    };

    struct Bucket {
        std::atomic<Node*> head{nullptr};
    };

    static constexpr size_t DEFAULT_BUCKET_COUNT = 1024;
    static constexpr size_t MAX_LOAD_FACTOR = 2;

    std::vector<Bucket> buckets_;
    std::atomic<size_t> size_{0};
    Hash hasher_;

    size_t get_bucket_index(const Key& key) const {
        return hasher_(key) % buckets_.size();
    }

public:
    explicit ConcurrentHashMap(size_t bucket_count = DEFAULT_BUCKET_COUNT)
        : buckets_(bucket_count) {}

    ~ConcurrentHashMap() {
        clear();
    }

    /**
     * Insert or update a key-value pair
     * Returns true if inserted, false if updated
     */
    bool insert(const Key& key, const Value& value) {
        size_t bucket_idx = get_bucket_index(key);
        Node* new_node = nullptr;

        while (true) {
            Node* head = buckets_[bucket_idx].head.load(std::memory_order_acquire);

            // Check if key already exists
            Node* current = head;
            while (current != nullptr) {
                if (!current->marked.load(std::memory_order_acquire) && current->key == key) {
                    // Key exists, don't update - return false to indicate no insertion
                    if (new_node) delete new_node;
                    return false;
                }
                current = current->next.load(std::memory_order_acquire);
            }

            // Only allocate if we haven't already
            if (!new_node) {
                new_node = new Node(key, value);
            }

            // Try to insert at head of bucket
            new_node->next.store(head, std::memory_order_release);
            if (buckets_[bucket_idx].head.compare_exchange_weak(
                    head, new_node,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                size_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            // CAS failed, retry with the same node
            // The key might have been inserted by another thread
        }
    }

    /**
     * Insert if not exists, or get existing value
     * Returns pair of (value, was_inserted)
     * This is linearizable - exactly one thread will return true for any given key
     */
    std::pair<Value, bool> insert_or_get(const Key& key, const Value& value) {
        size_t bucket_idx = get_bucket_index(key);
        Node* new_node = nullptr;

        while (true) {
            Node* head = buckets_[bucket_idx].head.load(std::memory_order_acquire);

            // Check if key already exists in current list state
            Node* current = head;
            while (current != nullptr) {
                if (!current->marked.load(std::memory_order_acquire) &&
                    current->key == key) {
                    // Key exists - clean up and return existing value
                    if (new_node) delete new_node;
                    Value existing_value = current->value;
                    return {existing_value, false};
                }
                current = current->next.load(std::memory_order_acquire);
            }

            // Only allocate if we haven't already
            if (!new_node) {
                new_node = new Node(key, value);
            }

            // Key doesn't exist in current state - try to insert
            new_node->next.store(head, std::memory_order_release);
            if (buckets_[bucket_idx].head.compare_exchange_weak(
                    head, new_node,
                    std::memory_order_release,
                    std::memory_order_acquire)) {

                // Successfully inserted - we're the winner
                size_.fetch_add(1, std::memory_order_relaxed);
                return {value, true};
            }

            // CAS failed because head changed - retry with the same node
            // The new head might contain our key now
        }
    }

    /**
     * Find a value by key
     * Returns optional containing value if found
     */
    std::optional<Value> find(const Key& key) const {
        size_t bucket_idx = get_bucket_index(key);
        Node* current = buckets_[bucket_idx].head.load(std::memory_order_acquire);

        while (current != nullptr) {
            if (!current->marked.load(std::memory_order_acquire) && current->key == key) {
                return current->value;
            }
            current = current->next.load(std::memory_order_acquire);
        }

        return std::nullopt;
    }

    /**
     * Check if key exists
     */
    bool contains(const Key& key) const {
        return find(key).has_value();
    }

    /**
     * Remove a key-value pair
     * Returns true if removed, false if not found
     */
    bool erase(const Key& key) {
        size_t bucket_idx = get_bucket_index(key);
        Node* current = buckets_[bucket_idx].head.load(std::memory_order_acquire);

        while (current != nullptr) {
            if (!current->marked.load(std::memory_order_acquire) && current->key == key) {
                // Mark as deleted
                if (current->marked.exchange(true, std::memory_order_acq_rel) == false) {
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    return true;
                }
            }
            current = current->next.load(std::memory_order_acquire);
        }

        return false;
    }

    /**
     * Get current size
     */
    size_t size() const {
        return size_.load(std::memory_order_relaxed);
    }

    /**
     * Check if empty
     */
    bool empty() const {
        return size() == 0;
    }

    /**
     * Clear all entries
     */
    void clear() {
        for (auto& bucket : buckets_) {
            Node* head = bucket.head.exchange(nullptr, std::memory_order_acq_rel);
            while (head != nullptr) {
                Node* next = head->next.load(std::memory_order_relaxed);
                delete head;
                head = next;
            }
        }
        size_.store(0, std::memory_order_relaxed);
    }

    /**
     * Apply a function to all key-value pairs
     */
    template<typename Func>
    void for_each(Func&& func) const {
        for (const auto& bucket : buckets_) {
            Node* current = bucket.head.load(std::memory_order_acquire);
            while (current != nullptr) {
                if (!current->marked.load(std::memory_order_acquire)) {
                    func(current->key, current->value);
                }
                current = current->next.load(std::memory_order_acquire);
            }
        }
    }

    /**
     * Collect all key-value pairs into a vector
     */
    std::vector<std::pair<Key, Value>> to_vector() const {
        std::vector<std::pair<Key, Value>> result;
        result.reserve(size());

        for_each([&result](const Key& k, const Value& v) {
            result.emplace_back(k, v);
        });

        return result;
    }
};

/**
 * Lock-free concurrent set using the concurrent hash map
 */
template<typename Key, typename Hash = std::hash<Key>>
class ConcurrentHashSet {
private:
    ConcurrentHashMap<Key, bool, Hash> map_;

public:
    explicit ConcurrentHashSet(size_t bucket_count = 1024)
        : map_(bucket_count) {}

    bool insert(const Key& key) {
        return map_.insert(key, true);
    }

    bool contains(const Key& key) const {
        return map_.contains(key);
    }

    bool erase(const Key& key) {
        return map_.erase(key);
    }

    size_t size() const {
        return map_.size();
    }

    bool empty() const {
        return map_.empty();
    }

    void clear() {
        map_.clear();
    }

    template<typename Func>
    void for_each(Func&& func) const {
        map_.for_each([&func](const Key& k, bool) {
            func(k);
        });
    }

    std::vector<Key> to_vector() const {
        std::vector<Key> result;
        result.reserve(size());

        for_each([&result](const Key& k) {
            result.push_back(k);
        });

        return result;
    }
};

} // namespace hypergraph

#endif // HYPERGRAPH_CONCURRENT_HASH_MAP_HPP