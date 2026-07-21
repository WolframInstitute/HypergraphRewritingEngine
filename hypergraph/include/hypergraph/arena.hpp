#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

namespace hypergraph {

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

    // Install a fresh block at the head of the chain. current_block_ is then
    // re-synced from head_ so allocators always reach the most-recent block: when
    // two threads race to allocate, the CAS loser must adopt the winner's newer
    // block rather than reinstate its own, or the newer block's capacity is
    // silently wasted.
    void allocate_new_block() {
        Block* new_block = Block::create(block_capacity_);

        Block* old_head = head_.load(std::memory_order_acquire);
        do {
            new_block->prev = old_head;
        } while (!head_.compare_exchange_weak(
            old_head, new_block,
            std::memory_order_release,
            std::memory_order_acquire));

        current_block_.store(head_.load(std::memory_order_acquire),
                             std::memory_order_release);
    }

    size_t block_capacity_;
    std::atomic<Block*> head_{nullptr};
    std::atomic<Block*> current_block_{nullptr};
};

// =============================================================================
// Per-thread arena worker index
// =============================================================================
//
// Each thread that allocates from an arena is assigned a small dense integer in
// [0, MAX_ARENA_WORKERS). That index selects a PRIVATE bump cursor inside every
// arena, so the allocation fast path touches only thread-local state and never a
// shared atomic — this is what lets the shared arena scale to many concurrent
// allocators without CAS contention. Indices are released at thread exit and
// reused, so the ceiling bounds PEAK concurrent threads, not total threads spawned
// over the process lifetime. A thread past the ceiling gets index -1 and falls back
// to the shared bump path (still correct, just contended); the ceiling sits well
// above any realistic worker count.

inline constexpr int MAX_ARENA_WORKERS = 256;

class ArenaWorkerRegistry {
public:
    // Claim the lowest free index, or -1 if all are taken.
    int acquire() {
        for (int i = 0; i < MAX_ARENA_WORKERS; ++i) {
            if (in_use_[i].load(std::memory_order_relaxed)) continue;
            bool expected = false;
            if (in_use_[i].compare_exchange_strong(
                    expected, true,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return i;
            }
        }
        return -1;
    }
    void release(int idx) {
        if (idx >= 0) in_use_[idx].store(false, std::memory_order_release);
    }
private:
    std::atomic<bool> in_use_[MAX_ARENA_WORKERS] = {};
};

inline ArenaWorkerRegistry& arena_worker_registry() {
    static ArenaWorkerRegistry registry;
    return registry;
}

// Acquires an index on first use by a thread, releases it at thread exit.
struct ArenaWorkerIndexHolder {
    int index;
    ArenaWorkerIndexHolder() : index(arena_worker_registry().acquire()) {}
    ~ArenaWorkerIndexHolder() { arena_worker_registry().release(index); }
};

// The calling thread's arena worker index (assigned on first call, stable for the
// thread's lifetime, -1 when the worker ceiling is exceeded).
inline int arena_worker_index() {
    static thread_local ArenaWorkerIndexHolder holder;
    return holder.index;
}

// =============================================================================
// ConcurrentHeterogeneousArena: Untyped thread-safe arena allocator
// =============================================================================
//
// Bump-pointer arena for heterogeneous object types, safe for concurrent
// allocation from multiple threads. The fast path is a PER-WORKER bump cursor:
// each thread carries a private current block and bumps a plain offset with no
// shared atomic, touching shared state only to grab a fresh block (rare, once per
// block_size_ bytes). This keeps concurrent allocation contention-free.
//

class ConcurrentHeterogeneousArena {
public:
    static constexpr size_t DEFAULT_BLOCK_SIZE = 1024 * 1024;  // 1 MB

    // recycle_blocks enables reset()-based block reuse. It is ONLY safe when the
    // arena is used single-threaded (a per-worker scratch arena); leave it false for
    // the shared global arena, where a concurrent reuse would zero a block another
    // thread is allocating from.
    explicit ConcurrentHeterogeneousArena(size_t block_size = DEFAULT_BLOCK_SIZE,
                                          bool recycle_blocks = false)
        : block_size_(block_size)
        , recycle_(recycle_blocks)
        , destructor_head_(nullptr) {
        allocate_new_block();
        // Per-worker bump cursors back the contention-free concurrent fast path. A
        // recycling scratch arena is single-threaded and uses the shared cursor plus
        // mark()/release()/reset() stack discipline instead, so it needs none.
        if (!recycle_) {
            cursors_ = new LocalCursor[MAX_ARENA_WORKERS];
        }
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

        delete[] cursors_;
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

        // Memory barrier: ensure prior reads see prior writes before we construct
        std::atomic_thread_fence(std::memory_order_acquire);

        T* obj = new (mem) T(std::forward<Args>(args)...);

        // Memory barrier: ensure construction is visible before returning
        std::atomic_thread_fence(std::memory_order_release);

        if constexpr (!std::is_trivially_destructible_v<T>) {
            register_destructor(obj, [](void* p) {
                static_cast<T*>(p)->~T();
            });
        }

        return obj;
    }

    // Allocate and construct a new T WITHOUT registering a destructor. Reserved for
    // objects whose cleanup is owned by the arena's bulk reclamation — e.g. an
    // arena-backed ConcurrentMap, whose own destructor is a no-op because its tables
    // live in this arena. Registering such a destructor would add a shared-list CAS
    // per object for no effect, so this path skips it.
    template<typename T, typename... Args>
    T* create_untracked(Args&&... args) {
        void* mem = allocate_raw(sizeof(T), alignof(T));
        std::atomic_thread_fence(std::memory_order_acquire);
        T* obj = new (mem) T(std::forward<Args>(args)...);
        std::atomic_thread_fence(std::memory_order_release);
        return obj;
    }

    // Allocate raw memory. The concurrent fast path bumps this worker's private
    // cursor with no shared atomic; only grabbing a fresh block touches shared state.
    // A recycling scratch arena (cursors_ == nullptr) and any thread past the worker
    // ceiling (index < 0) fall through to the shared bump path.
    void* allocate_raw(size_t size, size_t alignment = alignof(std::max_align_t)) {
        if (cursors_) {
            int wi = arena_worker_index();
            if (wi >= 0) return allocate_local(cursors_[wi], size, alignment);
        }
        return allocate_shared(size, alignment);
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

    // Stack-discipline checkpoint into a recycling scratch arena. mark() captures the
    // current position; release(m) rewinds to it, reclaiming everything allocated
    // since (the blocks stay chained and are recycled on the next advance). LIFO
    // only, single-threaded — for bounding per-call/per-recursion scratch high-water.
    // Does NOT run destructors, so only use for trivially-destructible scratch.
    struct Marker { void* blk; size_t off; };
    Marker mark() {
        Block* b = current_block_.load(std::memory_order_relaxed);
        return { b, b->offset.load(std::memory_order_relaxed) };
    }
    void release(Marker m) {
        Block* b = static_cast<Block*>(m.blk);
        b->offset.store(m.off, std::memory_order_relaxed);
        current_block_.store(b, std::memory_order_relaxed);
    }

    // Reset for reuse WITHOUT freeing blocks — recycles a scratch arena between
    // tasks. Single-threaded: only the owning thread may call this, and only while
    // no other thread allocates from this arena (e.g. a per-worker scratch arena
    // between tasks). Runs+clears registered destructors, zeroes every block, and
    // restarts allocation from the first block.
    void reset() {
        DestructorNode* node = destructor_head_.load(std::memory_order_relaxed);
        while (node) { node->destroy(node->object); node = node->prev; }
        destructor_head_.store(nullptr, std::memory_order_relaxed);
        Block* b = head_.load(std::memory_order_relaxed);
        Block* first = b;
        while (b) { b->offset.store(0, std::memory_order_relaxed); first = b; b = b->prev; }
        current_block_.store(first, std::memory_order_relaxed);
    }

private:
    struct Block {
        Block* prev;   // older block (allocation order, newest->oldest via head_)
        Block* next;   // newer block; lets reset() walk forward to recycle blocks
        size_t capacity;
        std::atomic<size_t> offset;
        alignas(std::max_align_t) char data[];

        static Block* create(size_t data_capacity) {
            void* mem = ::operator new(sizeof(Block) + data_capacity);
            Block* block = static_cast<Block*>(mem);
            block->prev = nullptr;
            block->next = nullptr;
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

    // One private bump cursor per worker index. Only the owning thread touches its
    // cursor, so the offset is a plain integer (no atomic) — the source of the
    // fast path's freedom from contention. Padded to a cache line so cursors of
    // different workers never share one.
    struct alignas(64) LocalCursor {
        Block* block = nullptr;   // this worker's current bump block
        size_t offset = 0;        // bump position within block->data
        size_t capacity = 0;      // cached block->capacity
    };

    // Bump this worker's private cursor. On overflow, grab a fresh block sized for
    // the request and bump from there. block->offset is mirrored (relaxed, to this
    // worker's own block) so bytes_allocated() sees the live high-water mark.
    void* allocate_local(LocalCursor& c, size_t size, size_t alignment) {
        if (c.block) {
            size_t aligned = (c.offset + alignment - 1) & ~(alignment - 1);
            size_t new_offset = aligned + size;
            if (new_offset <= c.capacity) {
                c.offset = new_offset;
                c.block->offset.store(new_offset, std::memory_order_relaxed);
                return c.block->data + aligned;
            }
        }
        // Current block can't fit this request; take a fresh one (shared, but rare).
        size_t cap = block_size_;
        size_t need = size + alignment;  // worst-case alignment slack
        if (need > cap) cap = need;
        Block* nb = grab_block(cap);
        c.block = nb;
        c.capacity = nb->capacity;
        // Fresh block: data is max_align_t-aligned, so an offset-relative alignment
        // (<= max_align_t, as for every arena request) starts at 0.
        size_t aligned = (alignment - 1) & ~(alignment - 1);  // == 0
        size_t new_offset = aligned + size;
        c.offset = new_offset;
        nb->offset.store(new_offset, std::memory_order_relaxed);
        return nb->data + aligned;
    }

    // Shared bump path: an atomic claim on current_block_'s offset. Backs the
    // single-threaded recycling scratch arena (whose mark/release/reset ride
    // current_block_) and the over-ceiling fallback for the concurrent arena.
    void* allocate_shared(size_t size, size_t alignment) {
        while (true) {
            Block* block = current_block_.load(std::memory_order_acquire);
            size_t offset = block->offset.load(std::memory_order_acquire);

            size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
            size_t new_offset = aligned_offset + size;

            if (new_offset <= block->capacity) {
                // Try to claim this region
                if (block->offset.compare_exchange_weak(
                        offset, new_offset,
                        std::memory_order_acq_rel,  // Use acq_rel for stronger ordering
                        std::memory_order_acquire)) {
                    void* result = block->data + aligned_offset;
                    return result;
                }
                continue;
            }

            advance_block();
        }
    }

    // See ConcurrentArena<T>::allocate_new_block for the rationale: sync
    // current_block_ from head_ after installing the new block so the last
    // store always reflects the most-recent head rather than a racing
    // thread's older block.
    // Allocate a block of the given capacity and splice it onto the head of the
    // chain (lock-free). Shared by the per-worker cursor path and allocate_new_block.
    Block* grab_block(size_t cap) {
        Block* new_block = Block::create(cap);

        Block* old_head = head_.load(std::memory_order_acquire);
        do {
            new_block->prev = old_head;
        } while (!head_.compare_exchange_weak(
            old_head, new_block,
            std::memory_order_release,
            std::memory_order_acquire));

        // Forward link (older -> newer). Only ever READ single-threaded after reset()
        // to recycle blocks, so the plain store is fine alongside the lock-free path.
        // Each old_head is superseded by exactly one CAS winner, so this store races
        // with no other write.
        if (old_head) old_head->next = new_block;

        return new_block;
    }

    void allocate_new_block() {
        grab_block(block_size_);

        // Track the most-recent head: a plain store(new_block) lets a racing
        // thread's older block win current_block_ while its newer block sits
        // unreachable mid-chain, stranding that block's capacity.
        current_block_.store(head_.load(std::memory_order_acquire),
                             std::memory_order_release);
    }

    // Advance to the next block when the current one is full: recycle an
    // already-allocated successor (populated after a reset()) if present, else grow.
    // The recycle branch is only reached single-threaded (a per-worker scratch arena
    // refilling after reset); the global concurrent arena's current_block_ is always
    // the head (next == nullptr), so it always grows, preserving prior behaviour.
    void advance_block() {
        if (recycle_) {
            Block* cur = current_block_.load(std::memory_order_relaxed);
            if (Block* nxt = cur->next) {            // single-threaded: safe to reuse
                nxt->offset.store(0, std::memory_order_relaxed);
                current_block_.store(nxt, std::memory_order_relaxed);
                return;
            }
        }
        allocate_new_block();
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
    bool recycle_;
    std::atomic<Block*> head_{nullptr};
    std::atomic<Block*> current_block_{nullptr};
    std::atomic<DestructorNode*> destructor_head_;
    // Per-worker bump cursors (non-null only for a non-recycling concurrent arena).
    LocalCursor* cursors_ = nullptr;
};

// Per-worker scratch arena: thread-local, recycled via reset() between tasks. The
// foundation of the allocation architecture — hot-path temporaries draw from here
// and are reclaimed in bulk by reset() instead of touching the global allocator on
// every call. One instance per thread ⇒ no contention, never freed mid-task.
inline ConcurrentHeterogeneousArena& worker_scratch() {
    static thread_local ConcurrentHeterogeneousArena scratch(
        ConcurrentHeterogeneousArena::DEFAULT_BLOCK_SIZE, /*recycle_blocks=*/true);
    return scratch;
}

// =============================================================================
// ArenaVector<T>: Vector that allocates from ConcurrentHeterogeneousArena
// =============================================================================
//
// Behaves like std::vector but uses arena allocation:
// - No individual free() calls - arena is bulk-freed at end of evolution
// - Growth allocates new array from arena, abandons old (becomes arena garbage)
// - Much faster than heap allocation for temporary vectors
//
// Thread safety: NOT thread-safe. Use one ArenaVector per thread.
// The underlying arena IS thread-safe, but the vector itself is not.
//
// Usage:
//   ArenaVector<int> vec(arena);
//   vec.reserve(100);  // Pre-allocate from arena
//   vec.push_back(42);
//   vec.clear();       // Logical clear, keeps capacity
//

template<typename T>
class ArenaVector {
public:
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    explicit ArenaVector(ConcurrentHeterogeneousArena& arena)
        : arena_(&arena)
        , data_(nullptr)
        , size_(0)
        , capacity_(0)
    {}

    ArenaVector(ConcurrentHeterogeneousArena& arena, size_t initial_capacity)
        : arena_(&arena)
        , data_(nullptr)
        , size_(0)
        , capacity_(0)
    {
        reserve(initial_capacity);
    }

    // No destructor needed - arena handles cleanup

    // Non-copyable (would need arena allocation for copy)
    ArenaVector(const ArenaVector&) = delete;
    ArenaVector& operator=(const ArenaVector&) = delete;

    // Move is okay (just pointer transfer)
    ArenaVector(ArenaVector&& other) noexcept
        : arena_(other.arena_)
        , data_(other.data_)
        , size_(other.size_)
        , capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    ArenaVector& operator=(ArenaVector&& other) noexcept {
        if (this != &other) {
            // Abandon our data (arena garbage)
            arena_ = other.arena_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) return;

        T* new_data = static_cast<T*>(
            arena_->allocate_raw(sizeof(T) * new_capacity, alignof(T))
        );

        // Copy existing elements
        if (size_ > 0) {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(new_data, data_, sizeof(T) * size_);
            } else {
                for (size_t i = 0; i < size_; ++i) {
                    new (&new_data[i]) T(std::move(data_[i]));
                    data_[i].~T();
                }
            }
        }

        // Abandon old data (arena garbage)
        data_ = new_data;
        capacity_ = new_capacity;
    }

    void push_back(const T& value) {
        if (size_ >= capacity_) {
            grow();
        }
        new (&data_[size_]) T(value);
        ++size_;
    }

    void push_back(T&& value) {
        if (size_ >= capacity_) {
            grow();
        }
        new (&data_[size_]) T(std::move(value));
        ++size_;
    }

    template<typename... Args>
    T& emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            grow();
        }
        T* obj = new (&data_[size_]) T(std::forward<Args>(args)...);
        ++size_;
        return *obj;
    }

    void clear() {
        // Call destructors for non-trivial types
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = 0; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = 0;
        // Keep capacity for reuse
    }

    void resize(size_t new_size) {
        if (new_size > capacity_) {
            reserve(new_size);
        }
        // Default construct new elements
        for (size_t i = size_; i < new_size; ++i) {
            new (&data_[i]) T();
        }
        // Destroy excess elements
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = new_size; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = new_size;
    }

    void resize(size_t new_size, const T& value) {
        if (new_size > capacity_) {
            reserve(new_size);
        }
        // Copy construct new elements
        for (size_t i = size_; i < new_size; ++i) {
            new (&data_[i]) T(value);
        }
        // Destroy excess elements
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = new_size; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = new_size;
    }

    // Access
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }

    T* data() { return data_; }
    const T* data() const { return data_; }

    // Iterators
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }

    // Size
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

private:
    void grow() {
        size_t new_capacity = capacity_ == 0 ? 8 : capacity_ * 2;
        reserve(new_capacity);
    }

    ConcurrentHeterogeneousArena* arena_;
    T* data_;
    size_t size_;
    size_t capacity_;
};

}  // namespace hypergraph
