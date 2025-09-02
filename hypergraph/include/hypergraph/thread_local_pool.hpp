#ifndef HYPERGRAPH_THREAD_LOCAL_POOL_HPP
#define HYPERGRAPH_THREAD_LOCAL_POOL_HPP

#include <vector>
#include <memory>

namespace hypergraph {

// Forward declaration - PartialMatch is defined in pattern_matching_tasks.hpp
struct PartialMatch;

/**
 * Thread-local object pool for lock-free allocation
 * Each thread maintains its own pool, eliminating synchronization overhead
 */
template<typename T>
class ThreadLocalPool {
private:
    // Thread-local storage - no synchronization needed
    thread_local static std::vector<std::unique_ptr<T>> available_objects_;
    
public:
    /**
     * Acquire an object from the thread-local pool
     * No locking needed - each thread has its own pool
     */
    static std::unique_ptr<T> acquire() {
        if (!available_objects_.empty()) {
            auto obj = std::move(available_objects_.back());
            available_objects_.pop_back();
            return obj;
        }
        // Create new object if pool is empty
        return std::make_unique<T>();
    }
    
    /**
     * Return an object to the thread-local pool
     * No locking needed - each thread manages its own pool
     */
    static void release(std::unique_ptr<T> obj) {
        if (obj) {
            available_objects_.push_back(std::move(obj));
        }
    }
    
    /**
     * Clear the thread-local pool
     * Called automatically when thread exits
     */
    static void clear() {
        available_objects_.clear();
    }
    
    /**
     * Get current pool size for this thread
     */
    static size_t size() {
        return available_objects_.size();
    }
};

// Thread-local storage definition
template<typename T>
thread_local std::vector<std::unique_ptr<T>> ThreadLocalPool<T>::available_objects_;

/**
 * Specialized thread-local pool for PartialMatch with proper reset logic
 */
class PartialMatchPool {
private:
    thread_local static std::vector<std::unique_ptr<PartialMatch>> available_objects_;
    
public:
    /**
     * Acquire a PartialMatch from thread-local pool
     * No synchronization needed
     */
    static std::unique_ptr<PartialMatch> acquire();
    
    /**
     * Return a PartialMatch to thread-local pool
     * No synchronization needed
     */
    static void release(std::unique_ptr<PartialMatch> obj);
    
    /**
     * Get current pool size for this thread
     */
    static size_t size();
    
    /**
     * Clear thread-local pool
     */
    static void clear();
};

} // namespace hypergraph

#endif // HYPERGRAPH_THREAD_LOCAL_POOL_HPP