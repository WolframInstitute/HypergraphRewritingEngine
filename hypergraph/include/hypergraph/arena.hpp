#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

// ASan: a block is poisoned as it is grabbed and each allocation unpoisons what it hands out,
// so a read past an allocation's end is reported even inside the arena's own memory.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define HG_ARENA_ASAN 1
#  endif
#elif defined(__SANITIZE_ADDRESS__)
#  define HG_ARENA_ASAN 1
#endif
#ifdef HG_ARENA_ASAN
extern "C" void __asan_poison_memory_region(void const volatile* addr, size_t size);
extern "C" void __asan_unpoison_memory_region(void const volatile* addr, size_t size);
#  define HG_ARENA_POISON(addr, size)   __asan_poison_memory_region((addr), (size))
#  define HG_ARENA_UNPOISON(addr, size) __asan_unpoison_memory_region((addr), (size))
#else
#  define HG_ARENA_POISON(addr, size)   ((void)(addr), (void)(size))
#  define HG_ARENA_UNPOISON(addr, size) ((void)(addr), (void)(size))
#endif

namespace HG_NAMESPACE {
namespace engine {

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

// Overridable so a harness can bound it. The registry's acquire() scans every slot and each is
// an atomic location, so a model checker asked to enumerate 256 of them is enumerating the scan
// rather than the property. A harness defines HG_MAX_ARENA_WORKERS small and checks the SAME
// code; the shipped value is unchanged.
#ifndef HG_MAX_ARENA_WORKERS
#define HG_MAX_ARENA_WORKERS 256
#endif
inline constexpr int MAX_ARENA_WORKERS = HG_MAX_ARENA_WORKERS;

// Both bodies are defined here rather than in arena.cpp, and the reason is a linkage one:
// verification/genmc/arena_worker_index_exclusive.cpp constructs a registry and calls acquire()
// and release() on it, compiling this header on its own without linking the engine library. The
// harness checks THIS code; an out-of-line definition would leave it undefined at link, and
// transcribing the loop into the harness would be a second implementation of the rule.
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

ArenaWorkerRegistry& arena_worker_registry();

#ifdef HG_VERIFICATION

// Under a model checker the index comes from a counter, is claimed on first use, and is never
// released. Two constraints of the checker's interpreter force this shape, and both are about
// how it materialises globals rather than about how the code runs.
//
// A thread_local of CLASS type cannot be interpreted: every global is built through LLVM's
// ExecutionEngine, which has no constant representation for an aggregate and stops before the
// first thread runs. That holds for any class -- an empty one fails the same way -- and a
// function-local thread_local scalar blocks on its initialisation guard, so the one shape that
// survives is a constant-initialised scalar at namespace scope, needing neither a guard nor a
// destructor registration.
//
// The registry itself is an aggregate global of MAX_ARENA_WORKERS atomics, which the same
// machinery faults on rather than diagnoses, so it must not appear in the module at all. What
// keeps it out is that arena_worker_registry() is only declared here -- its local static is
// defined in arena.cpp, which a harness compiles nothing of and links nothing of.
//
// A counter hands out each index once, so it cannot reproduce the acquire/release recycling. That
// costs nothing here: a harness runs a fixed, bounded set of threads that outlive their
// allocations, no index is ever reused, and the free-list behaviour is not among the properties
// any harness states. verification/genmc/README.md lists this substitution with its argument.
inline std::atomic<int> g_arena_worker_next{0};
inline thread_local int t_arena_worker_index = -1;

inline int arena_worker_index() {
    if (t_arena_worker_index < 0) {
        const int next = g_arena_worker_next.fetch_add(1, std::memory_order_relaxed);
        t_arena_worker_index = next < MAX_ARENA_WORKERS ? next : -1;
    }
    return t_arena_worker_index;
}

#else

// Acquires an index on first use by a thread, releases it at thread exit.
struct ArenaWorkerIndexHolder {
    int index;
    ArenaWorkerIndexHolder();
    ~ArenaWorkerIndexHolder();
};

// The calling thread's arena worker index (assigned on first call, stable for the
// thread's lifetime, -1 when the worker ceiling is exceeded).
int arena_worker_index();

#endif  // HG_VERIFICATION

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
    // THE HUGE PAGE IS THE REASON FOR THE SIZE. A block below 2 MB, or not 2 MB aligned, cannot be
    // backed by one however it is advised, so the default block is exactly one huge page. Measured
    // on the EPYC at 32 workers: the 1 MB default took 1,060,520 minor faults on wpp depth 7, with
    // 9.3% of cycles in the kernel's fault and page-zeroing path.
    static constexpr size_t kHugePageBytes = 2u * 1024u * 1024u;
#if defined(HG_VERIFICATION)
    // Under the model checker a block is small and comes from operator new: the huge-page path
    // calls posix_memalign, which the interpreter does not model, and a 2 MB block is a 2 MB
    // memset it has to promote -- one store per byte, each a bounded-loop iteration, so the
    // engine harnesses define the size (verification/genmc/engine_*.cpp). Block size and source
    // are allocation policy on thread-private memory; nothing another thread reads depends on
    // either.
#ifndef HG_ARENA_BLOCK_SIZE
#define HG_ARENA_BLOCK_SIZE 4096
#endif
    static constexpr size_t DEFAULT_BLOCK_SIZE = HG_ARENA_BLOCK_SIZE;
#else
    static constexpr size_t DEFAULT_BLOCK_SIZE = kHugePageBytes;
#endif

    // The first block reserved (per arena, and per worker cursor) is small, and each
    // successive block doubles up to block_size_. A lightly-used arena therefore
    // reserves only a small block instead of a full block_size_, so the heap-BYTES
    // floor tracks actual usage; the geometric ramp keeps the malloc COUNT
    // logarithmic in the bytes served, never per-allocation.
    static constexpr size_t INITIAL_BLOCK_SIZE = 64 * 1024;  // 64 KB

    // recycle_blocks enables reset()-based block reuse. It is ONLY safe when the
    // arena is used single-threaded (a per-worker scratch arena); leave it false for
    // the shared global arena, where a concurrent reuse would zero a block another
    // thread is allocating from.
    explicit ConcurrentHeterogeneousArena(size_t block_size = DEFAULT_BLOCK_SIZE,
                                          bool recycle_blocks = false);

    ~ConcurrentHeterogeneousArena();

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
    // A recycling arena is single-threaded, so it bumps current_block_'s offset with
    // no read-modify-write at all. Any thread past the worker ceiling (index < 0)
    // falls through to the atomic shared bump path.
    // The bump on this worker's own cursor, inline: a fit in the current block is an
    // alignment, a compare and two stores, and it is what nearly every allocation is.
    // Everything else -- a full block, a recycling arena, a thread with no cursor -- is the
    // out-of-line path. Stats builds take the out-of-line path for every allocation, which
    // is where the per-site profile is booked.
    void* allocate_raw(size_t size, size_t alignment = alignof(std::max_align_t)) {
#if !HG_ENGINE_STATS
        if (!recycle_) {
            const int wi = arena_worker_index();
            if (wi >= 0) {
                LocalCursor& c = cursors_[wi];
                if (c.block) {
                    const size_t aligned = (c.offset + alignment - 1) & ~(alignment - 1);
                    const size_t new_offset = aligned + size;
                    if (new_offset <= c.capacity) {
                        c.offset = new_offset;
                        c.block->offset.store(new_offset, std::memory_order_relaxed);
                        void* p = c.block->data + aligned;
                        HG_ARENA_UNPOISON(p, size);
                        return p;
                    }
                }
            }
        }
#endif
        return allocate_raw_slow(size, alignment);
    }
    void* allocate_raw_slow(size_t size, size_t alignment);

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
    size_t bytes_allocated() const;
    // Bytes of blocks this arena holds, and the most it has held: its share of the resident
    // set, against bytes_allocated() which is what the blocks contain.
    size_t block_bytes() const { return block_bytes_.load(std::memory_order_relaxed); }
    size_t block_bytes_high_water() const { return block_bytes_high_water_.load(std::memory_order_relaxed); }

    // Stack-discipline checkpoint into a recycling scratch arena. mark() captures the
    // current position; release(m) rewinds to it, reclaiming everything allocated
    // since (the blocks stay chained and are recycled on the next advance). LIFO
    // only, single-threaded — for bounding per-call/per-recursion scratch high-water.
    // Does NOT run destructors, so only use for trivially-destructible scratch.
    struct Marker { void* blk; size_t off; };
    Marker mark();
    void release(Marker m);

    // Reset for reuse WITHOUT freeing blocks — recycles a scratch arena between
    // tasks. Single-threaded: only the owning thread may call this, and only while
    // no other thread allocates from this arena (e.g. a per-worker scratch arena
    // between tasks). Runs+clears registered destructors, zeroes every block, and
    // restarts allocation from the first block.
    void reset();

private:
    struct Block {
        Block* prev;   // older block (allocation order, newest->oldest via head_)
        Block* next;   // newer block; lets reset() walk forward to recycle blocks
        size_t capacity;
        std::atomic<size_t> offset;
        // WHICH ALLOCATOR PRODUCED THIS BLOCK, recorded rather than inferred. A huge-page block
        // comes from posix_memalign and must go back through free(); an ordinary one comes from
        // operator new. Deducing it from the size would be wrong the moment the threshold moves.
        bool huge;
        alignas(std::max_align_t) char data[];

        static Block* create(size_t data_capacity);
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
        size_t next_size = 0;     // size of this cursor's next block (0 = use initial); doubles up to block_size_
    };

    // Bump this worker's private cursor. On overflow, grab a fresh block sized for
    // the request and bump from there. block->offset is mirrored (relaxed, to this
    // worker's own block) so bytes_allocated() sees the live high-water mark.
    void* allocate_local(LocalCursor& c, size_t size, size_t alignment);

    // Single-threaded bump path for a recycling arena, riding the same
    // current_block_/offset pair that mark()/release()/reset() do.
    void* allocate_single(size_t size, size_t alignment);

    // Shared bump path: an atomic claim on current_block_'s offset. Backs the
    // over-ceiling fallback for the concurrent arena.
    void* allocate_shared(size_t size, size_t alignment);

    // Allocate a block of the given capacity and splice it onto the head of the
    // chain (lock-free). Shared by the per-worker cursor path and allocate_new_block.
    Block* grab_block(size_t cap);

    // Grow the shared chain by one block, sized by the geometric ramp or min_cap,
    // whichever is larger.
    void allocate_new_block(size_t min_cap = 0);

    // Advance to the next block when the current one is full: recycle an
    // already-allocated successor (populated after a reset()) if present, else grow.
    void advance_block(size_t min_cap = 0);

    void register_destructor(void* obj, void (*destroy)(void*));

    size_t block_size_;
    bool recycle_;
    std::atomic<size_t> block_bytes_{0};
    std::atomic<size_t> block_bytes_high_water_{0};
    std::atomic<Block*> head_{nullptr};
    std::atomic<Block*> current_block_{nullptr};
    std::atomic<DestructorNode*> destructor_head_;
    // Geometric growth size for the shared/eager block chain: doubles up to
    // block_size_ each time a shared block is grabbed. Atomic for the rare
    // concurrent over-ceiling fallback path.
    std::atomic<size_t> shared_grow_;
    // Per-worker bump cursors (non-null only for a non-recycling concurrent arena).
    LocalCursor* cursors_ = nullptr;
};

// Per-worker scratch arena: thread-local, recycled via reset() between tasks. The
// foundation of the allocation architecture — hot-path temporaries draw from here
// and are reclaimed in bulk by reset() instead of touching the global allocator on
// every call. One instance per thread ⇒ no contention, never freed mid-task.
ConcurrentHeterogeneousArena& worker_scratch();

// TOTAL BYTES CURRENTLY HELD BY ConcurrentHeterogeneousArena BLOCKS, summed over every arena on
// every thread. bytes_allocated() answers a different question -- the live high-water offset
// inside one arena -- and neither the main arena's figure nor the sum of them says how much the
// process is HOLDING, because a block stays owned once grown: reset() rewinds offsets and keeps
// the chain, which is what makes the next task's allocations free.
//
// That distinction is the whole reason this exists. Resident set grows about 200 MB per worker
// on the shape workload (tools/dev/worker_memory_slope.sh: 624 MB at one thread, 1211 MB at
// four), and "the workers are each holding an arena grown to their own high-water mark" and
// "the workers have more live data" predict the same RSS curve. This number separates them.
size_t arena_block_bytes_live();
// The largest that total has been since the process started: the arenas' share of the
// resident set's high-water mark.
size_t arena_block_bytes_high_water();
// The same high-water for the recycling (per-worker scratch) arenas alone.
size_t arena_scratch_block_bytes_high_water();
#if HG_ENGINE_STATS
// Stats builds: bytes and calls per allocation site (the engine function that asked), the
// largest first. Addresses are symbolised with addr2line against the binary.
void arena_alloc_profile_dump(FILE* out, size_t top);
#endif

// BYTES OF ConcurrentMap TABLES THAT LOST THEIR INSTALL RACE and were abandoned in an arena.
//
// resize() allocates the new table BEFORE the compare-exchange that installs it, so under
// contention every thread but one has allocated a full table it then discards. A heap-backed
// map deletes it; an arena-backed one cannot, and the comment at the discard site calls that
// "rare, small". Whether it is rare and small is a measurement, not a property of the code, and
// this is the number that decides it: it scales with contention where the map's live table
// bytes do not.
//
// EMPTY UNDER HG_VERIFICATION, which is a real change to what the checker sees and is listed in
// verification/genmc/README.md with the others. A GenMC harness compiles concurrent_map.hpp and
// links no library, so these would be unresolved externals -- and defining them inline for its
// benefit would be worse than that: every table create, install and discard would become a
// racing read-modify-write on one shared location, multiplying the execution count of the
// growth harnesses over a variable no property here mentions. Nothing reads a counter to decide
// anything; they are relaxed diagnostics beside the protocol, never in it.
#ifdef HG_VERIFICATION
inline void note_discarded_table_bytes(size_t) {}
inline size_t discarded_table_bytes() { return 0; }
inline void note_installed_table_bytes(size_t) {}
inline size_t installed_table_bytes() { return 0; }
inline size_t installed_table_count() { return 0; }
inline size_t discarded_table_count() { return 0; }
#else
void note_discarded_table_bytes(size_t bytes);
size_t discarded_table_bytes();
void note_installed_table_bytes(size_t bytes);
size_t installed_table_bytes();
size_t installed_table_count();
size_t discarded_table_count();
#endif

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

}  // namespace engine
}  // namespace HG_NAMESPACE