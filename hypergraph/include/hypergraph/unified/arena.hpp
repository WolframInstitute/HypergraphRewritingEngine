#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace hypergraph::unified {

// =============================================================================
// Arena<T>: Typed single-threaded arena allocator
// =============================================================================
//
// Allocates objects of type T from chained memory blocks.
// Calls destructors in reverse allocation order when arena is destroyed.
// For trivially destructible types, destructor tracking is skipped entirely.
//
// Thread safety: NONE. Use one per thread (thread_local).
//

template<typename T>
class Arena {
public:
    static constexpr size_t DEFAULT_BLOCK_CAPACITY = 1024;  // Objects per block

    explicit Arena(size_t block_capacity = DEFAULT_BLOCK_CAPACITY)
        : block_capacity_(block_capacity)
        , head_(nullptr)
        , current_block_(nullptr) {
        allocate_new_block();
    }

    ~Arena() {
        // Walk blocks from newest to oldest (head_ is newest)
        Block* block = head_;
        while (block) {
            // Destroy objects in reverse order within block (newest first)
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_t i = block->count; i > 0; --i) {
                    block->objects[i - 1].~T();
                }
            }
            Block* prev = block->prev;
            ::operator delete(block);
            block = prev;
        }
    }

    // Non-copyable, non-movable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) = delete;
    Arena& operator=(Arena&&) = delete;

    // Allocate and construct a new T
    template<typename... Args>
    T* create(Args&&... args) {
        if (current_block_->count >= block_capacity_) {
            allocate_new_block();
        }
        T* obj = new (&current_block_->objects[current_block_->count])
                     T(std::forward<Args>(args)...);
        ++current_block_->count;
        return obj;
    }

    // Allocate array of n objects (default constructed)
    T* allocate_array(size_t n) {
        // For arrays, we need contiguous storage
        if (current_block_->count + n > block_capacity_) {
            // Need new block, possibly oversized
            size_t cap = (n > block_capacity_) ? n : block_capacity_;
            allocate_new_block(cap);
        }
        T* arr = &current_block_->objects[current_block_->count];
        for (size_t i = 0; i < n; ++i) {
            new (&arr[i]) T();
        }
        current_block_->count += n;
        return arr;
    }

    // Statistics
    size_t count() const {
        size_t total = 0;
        Block* block = head_;
        while (block) {
            total += block->count;
            block = block->prev;
        }
        return total;
    }

    size_t capacity() const {
        size_t total = 0;
        Block* block = head_;
        while (block) {
            total += block->capacity;
            block = block->prev;
        }
        return total;
    }

private:
    struct Block {
        Block* prev;      // Previously allocated block (older)
        size_t capacity;
        size_t count;
        T objects[];      // Flexible array member

        static Block* create(size_t cap) {
            void* mem = ::operator new(sizeof(Block) + sizeof(T) * cap);
            Block* block = static_cast<Block*>(mem);
            block->prev = nullptr;
            block->capacity = cap;
            block->count = 0;
            return block;
        }
    };

    void allocate_new_block(size_t cap) {
        Block* new_block = Block::create(cap);
        new_block->prev = head_;
        head_ = new_block;
        current_block_ = new_block;
    }

    void allocate_new_block() {
        allocate_new_block(block_capacity_);
    }

    size_t block_capacity_;
    Block* head_;
    Block* current_block_;
};

// =============================================================================
// ConcurrentArena<T>: Typed thread-safe arena allocator
// =============================================================================
//
// Same as Arena<T> but supports concurrent allocation from multiple threads.
// Uses atomic operations for thread safety.
//

template<typename T>
class ConcurrentArena {
public:
    static constexpr size_t DEFAULT_BLOCK_CAPACITY = 1024;

    explicit ConcurrentArena(size_t block_capacity = DEFAULT_BLOCK_CAPACITY)
        : block_capacity_(block_capacity) {
        allocate_new_block();
    }

    ~ConcurrentArena() {
        Block* block = head_.load(std::memory_order_acquire);
        while (block) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                size_t cnt = block->count.load(std::memory_order_relaxed);
                // Clamp to capacity in case of overflow from failed allocations
                if (cnt > block->capacity) cnt = block->capacity;
                for (size_t i = cnt; i > 0; --i) {
                    block->objects[i - 1].~T();
                }
            }
            Block* prev = block->prev;
            ::operator delete(block);
            block = prev;
        }
    }

    // Non-copyable, non-movable
    ConcurrentArena(const ConcurrentArena&) = delete;
    ConcurrentArena& operator=(const ConcurrentArena&) = delete;
    ConcurrentArena(ConcurrentArena&&) = delete;
    ConcurrentArena& operator=(ConcurrentArena&&) = delete;

    // Allocate and construct a new T
    template<typename... Args>
    T* create(Args&&... args) {
        while (true) {
            Block* block = current_block_.load(std::memory_order_acquire);
            size_t idx = block->count.fetch_add(1, std::memory_order_acq_rel);

            if (idx < block->capacity) {
                T* obj = new (&block->objects[idx]) T(std::forward<Args>(args)...);
                return obj;
            }

            // Block is full, revert and allocate new block
            block->count.fetch_sub(1, std::memory_order_relaxed);
            allocate_new_block();
        }
    }

    // Statistics
    size_t count() const {
        size_t total = 0;
        Block* block = head_.load(std::memory_order_acquire);
        while (block) {
            size_t c = block->count.load(std::memory_order_relaxed);
            total += (c <= block->capacity) ? c : block->capacity;
            block = block->prev;
        }
        return total;
    }

private:
    struct Block {
        Block* prev;
        size_t capacity;
        std::atomic<size_t> count;
        T objects[];

        static Block* create(size_t cap) {
            void* mem = ::operator new(sizeof(Block) + sizeof(T) * cap);
            Block* block = static_cast<Block*>(mem);
            block->prev = nullptr;
            block->capacity = cap;
            block->count.store(0, std::memory_order_relaxed);
            return block;
        }
    };

    void allocate_new_block() {
        Block* new_block = Block::create(block_capacity_);

        Block* old_head = head_.load(std::memory_order_acquire);
        do {
            new_block->prev = old_head;
        } while (!head_.compare_exchange_weak(
            old_head, new_block,
            std::memory_order_release,
            std::memory_order_acquire));

        current_block_.store(new_block, std::memory_order_release);
    }

    size_t block_capacity_;
    std::atomic<Block*> head_{nullptr};
    std::atomic<Block*> current_block_{nullptr};
};

// =============================================================================
// HeterogeneousArena: Untyped single-threaded arena allocator
// =============================================================================
//
// Allocates objects of any type. Tracks destructors for non-trivially-
// destructible types using type-erased destructor nodes.
// Calls destructors in reverse allocation order when arena is destroyed.
//
// Thread safety: NONE. Use one per thread (thread_local).
//

class HeterogeneousArena {
public:
    static constexpr size_t DEFAULT_BLOCK_SIZE = 1024 * 1024;  // 1 MB

    explicit HeterogeneousArena(size_t block_size = DEFAULT_BLOCK_SIZE)
        : block_size_(block_size)
        , head_(nullptr)
        , current_block_(nullptr)
        , destructor_head_(nullptr) {
        allocate_new_block();
    }

    ~HeterogeneousArena() {
        // Call destructors in reverse allocation order (head is newest)
        DestructorNode* node = destructor_head_;
        while (node) {
            node->destroy(node->object);
            node = node->prev;
        }

        // Free all blocks from newest to oldest
        Block* block = head_;
        while (block) {
            Block* prev = block->prev;
            ::operator delete(block);
            block = prev;
        }
    }

    // Non-copyable, non-movable
    HeterogeneousArena(const HeterogeneousArena&) = delete;
    HeterogeneousArena& operator=(const HeterogeneousArena&) = delete;
    HeterogeneousArena(HeterogeneousArena&&) = delete;
    HeterogeneousArena& operator=(HeterogeneousArena&&) = delete;

    // Allocate and construct a new T
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* mem = allocate_raw(sizeof(T), alignof(T));
        T* obj = new (mem) T(std::forward<Args>(args)...);

        if constexpr (!std::is_trivially_destructible_v<T>) {
            register_destructor(obj, [](void* p) {
                static_cast<T*>(p)->~T();
            });
        }

        return obj;
    }

    // Allocate raw memory (no construction, no destructor tracking)
    void* allocate_raw(size_t size, size_t alignment = alignof(std::max_align_t)) {
        size_t aligned_offset = (current_block_->offset + alignment - 1) & ~(alignment - 1);
        size_t new_offset = aligned_offset + size;

        if (new_offset > current_block_->capacity) {
            allocate_new_block();
            aligned_offset = (current_block_->offset + alignment - 1) & ~(alignment - 1);
            new_offset = aligned_offset + size;
        }

        current_block_->offset = new_offset;
        return current_block_->data + aligned_offset;
    }

    // Allocate array of T (default constructed, destructors tracked if needed)
    template<typename T>
    T* allocate_array(size_t n) {
        void* mem = allocate_raw(sizeof(T) * n, alignof(T));
        T* arr = static_cast<T*>(mem);

        for (size_t i = 0; i < n; ++i) {
            new (&arr[i]) T();
        }

        if constexpr (!std::is_trivially_destructible_v<T>) {
            // Register destructors in reverse order so they're called correctly
            for (size_t i = n; i > 0; --i) {
                register_destructor(&arr[i - 1], [](void* p) {
                    static_cast<T*>(p)->~T();
                });
            }
        }

        return arr;
    }

    // Statistics
    size_t bytes_allocated() const {
        size_t total = 0;
        Block* block = head_;
        while (block) {
            total += block->offset;
            block = block->prev;
        }
        return total;
    }

    size_t bytes_capacity() const {
        size_t total = 0;
        Block* block = head_;
        while (block) {
            total += block->capacity;
            block = block->prev;
        }
        return total;
    }

private:
    struct Block {
        Block* prev;      // Previously allocated block (older)
        size_t capacity;
        size_t offset;
        alignas(std::max_align_t) char data[];

        static Block* create(size_t data_capacity) {
            void* mem = ::operator new(sizeof(Block) + data_capacity);
            Block* block = static_cast<Block*>(mem);
            block->prev = nullptr;
            block->capacity = data_capacity;
            block->offset = 0;
            return block;
        }
    };

    struct DestructorNode {
        void* object;
        void (*destroy)(void*);
        DestructorNode* prev;  // Previously registered destructor (older)
    };

    void allocate_new_block() {
        Block* new_block = Block::create(block_size_);
        new_block->prev = head_;
        head_ = new_block;
        current_block_ = new_block;
    }

    void register_destructor(void* obj, void (*destroy)(void*)) {
        // Allocate node from the arena itself
        void* mem = allocate_raw(sizeof(DestructorNode), alignof(DestructorNode));
        DestructorNode* node = static_cast<DestructorNode*>(mem);
        node->object = obj;
        node->destroy = destroy;
        node->prev = destructor_head_;
        destructor_head_ = node;
    }

    size_t block_size_;
    Block* head_;
    Block* current_block_;
    DestructorNode* destructor_head_;
};

// =============================================================================
// ConcurrentHeterogeneousArena: Untyped thread-safe arena allocator
// =============================================================================
//
// Same as HeterogeneousArena but supports concurrent allocation.
//

class ConcurrentHeterogeneousArena {
public:
    static constexpr size_t DEFAULT_BLOCK_SIZE = 1024 * 1024;  // 1 MB

    explicit ConcurrentHeterogeneousArena(size_t block_size = DEFAULT_BLOCK_SIZE)
        : block_size_(block_size)
        , destructor_head_(nullptr) {
        allocate_new_block();
    }

    ~ConcurrentHeterogeneousArena() {
        // Call destructors in reverse allocation order
        DestructorNode* node = destructor_head_.load(std::memory_order_acquire);
        while (node) {
            node->destroy(node->object);
            node = node->prev;
        }

        // Free all blocks
        Block* block = head_.load(std::memory_order_acquire);
        while (block) {
            Block* prev = block->prev;
            ::operator delete(block);
            block = prev;
        }
    }

    // Non-copyable, non-movable
    ConcurrentHeterogeneousArena(const ConcurrentHeterogeneousArena&) = delete;
    ConcurrentHeterogeneousArena& operator=(const ConcurrentHeterogeneousArena&) = delete;
    ConcurrentHeterogeneousArena(ConcurrentHeterogeneousArena&&) = delete;
    ConcurrentHeterogeneousArena& operator=(ConcurrentHeterogeneousArena&&) = delete;

    // Allocate and construct a new T
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* mem = allocate_raw(sizeof(T), alignof(T));
        T* obj = new (mem) T(std::forward<Args>(args)...);

        if constexpr (!std::is_trivially_destructible_v<T>) {
            register_destructor(obj, [](void* p) {
                static_cast<T*>(p)->~T();
            });
        }

        return obj;
    }

    // Allocate raw memory
    void* allocate_raw(size_t size, size_t alignment = alignof(std::max_align_t)) {
        while (true) {
            Block* block = current_block_.load(std::memory_order_acquire);
            size_t offset = block->offset.load(std::memory_order_acquire);

            size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
            size_t new_offset = aligned_offset + size;

            if (new_offset <= block->capacity) {
                if (block->offset.compare_exchange_weak(
                        offset, new_offset,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                    return block->data + aligned_offset;
                }
                continue;
            }

            allocate_new_block();
        }
    }

    // Allocate array of T (default constructed, destructors tracked if needed)
    template<typename T>
    T* allocate_array(size_t n) {
        void* mem = allocate_raw(sizeof(T) * n, alignof(T));
        T* arr = static_cast<T*>(mem);

        for (size_t i = 0; i < n; ++i) {
            new (&arr[i]) T();
        }

        if constexpr (!std::is_trivially_destructible_v<T>) {
            // Register destructors in reverse order so they're called correctly
            for (size_t i = n; i > 0; --i) {
                register_destructor(&arr[i - 1], [](void* p) {
                    static_cast<T*>(p)->~T();
                });
            }
        }

        return arr;
    }

    // Statistics
    size_t bytes_allocated() const {
        size_t total = 0;
        Block* block = head_.load(std::memory_order_acquire);
        while (block) {
            total += block->offset.load(std::memory_order_relaxed);
            block = block->prev;
        }
        return total;
    }

private:
    struct Block {
        Block* prev;
        size_t capacity;
        std::atomic<size_t> offset;
        alignas(std::max_align_t) char data[];

        static Block* create(size_t data_capacity) {
            void* mem = ::operator new(sizeof(Block) + data_capacity);
            Block* block = static_cast<Block*>(mem);
            block->prev = nullptr;
            block->capacity = data_capacity;
            block->offset.store(0, std::memory_order_relaxed);
            return block;
        }
    };

    struct DestructorNode {
        void* object;
        void (*destroy)(void*);
        DestructorNode* prev;
    };

    void allocate_new_block() {
        Block* new_block = Block::create(block_size_);

        Block* old_head = head_.load(std::memory_order_acquire);
        do {
            new_block->prev = old_head;
        } while (!head_.compare_exchange_weak(
            old_head, new_block,
            std::memory_order_release,
            std::memory_order_acquire));

        current_block_.store(new_block, std::memory_order_release);
    }

    void register_destructor(void* obj, void (*destroy)(void*)) {
        void* mem = allocate_raw(sizeof(DestructorNode), alignof(DestructorNode));
        DestructorNode* node = static_cast<DestructorNode*>(mem);
        node->object = obj;
        node->destroy = destroy;

        DestructorNode* old_head = destructor_head_.load(std::memory_order_acquire);
        do {
            node->prev = old_head;
        } while (!destructor_head_.compare_exchange_weak(
            old_head, node,
            std::memory_order_release,
            std::memory_order_acquire));
    }

    size_t block_size_;
    std::atomic<Block*> head_{nullptr};
    std::atomic<Block*> current_block_{nullptr};
    std::atomic<DestructorNode*> destructor_head_;
};

}  // namespace hypergraph::unified
