#include "hypergraph/arena.hpp"
#include "hypergraph/scratch_alloc.hpp"

// ASAN CONTAINER ANNOTATIONS.
//
// A bump allocator hands out slices of ONE large allocation, so a write past the end of an
// object lands in the next object's bytes -- inside a region the sanitizer sees as a single
// valid buffer, and therefore invisible to it. That is the shape of a defect this engine
// currently carries: bench_cpu_evolve.exe at one worker exits STATUS_HEAP_CORRUPTION on
// Windows, deterministically, while Linux under valgrind is clean at the same configuration.
//
// Poisoning a block when it is created and unpoisoning exactly the bytes each request returns
// puts the object boundaries back where the sanitizer can see them, so an intra-block overrun
// reports where it happens instead of corrupting whatever is next.
//
// The granularity is eight bytes, so two small allocations sharing a granule cannot be
// separated; an overrun that stays inside one is still invisible. It is a partial instrument,
// and the boundaries it does place are the ones a bump allocator erases.
//
// Present only under a sanitizer build. Both macros are empty otherwise, so nothing here
// reaches the shipping allocator.
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

ArenaWorkerRegistry& arena_worker_registry() {
    static ArenaWorkerRegistry registry;
    return registry;
}

#ifndef HG_VERIFICATION

ArenaWorkerIndexHolder::ArenaWorkerIndexHolder()
    : index(arena_worker_registry().acquire()) {}

ArenaWorkerIndexHolder::~ArenaWorkerIndexHolder() {
    arena_worker_registry().release(index);
}

int arena_worker_index() {
    static thread_local ArenaWorkerIndexHolder holder;
    return holder.index;
}

#endif  // HG_VERIFICATION

// =============================================================================
// ConcurrentHeterogeneousArena
// =============================================================================

ConcurrentHeterogeneousArena::ConcurrentHeterogeneousArena(size_t block_size,
                                                           bool recycle_blocks)
    : block_size_(block_size)
    , recycle_(recycle_blocks)
    , destructor_head_(nullptr)
    , shared_grow_(INITIAL_BLOCK_SIZE < block_size ? INITIAL_BLOCK_SIZE : block_size) {
    allocate_new_block();
    // Per-worker bump cursors back the contention-free concurrent fast path. A
    // recycling scratch arena is single-threaded and bumps current_block_ directly
    // (allocate_single), under mark()/release()/reset() stack discipline, so it
    // needs none. Invariant relied on by allocate_raw: !recycle_ ⇒ cursors_ != null.
    if (!recycle_) {
        cursors_ = new LocalCursor[MAX_ARENA_WORKERS];
    }
}

// Bytes held by every live ConcurrentHeterogeneousArena block. Bumped where blocks are created
// and freed, which is once per 1 MB rather than once per allocation, so nothing on the hot path
// touches it. Relaxed throughout: it is read for reporting, never to decide anything.
static std::atomic<size_t> g_arena_block_bytes{0};

ConcurrentHeterogeneousArena::~ConcurrentHeterogeneousArena() {
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
        // operator delete must not be handed a region this file poisoned.
        HG_ARENA_UNPOISON(block->data, block->capacity);
        g_arena_block_bytes.fetch_sub(sizeof(Block) + block->capacity, std::memory_order_relaxed);
        ::operator delete(block);
        block = prev;
    }

    delete[] cursors_;
}

void* ConcurrentHeterogeneousArena::allocate_raw(size_t size, size_t alignment) {
    // One choke point for every request, which is where the unpoison belongs: the three paths
    // below differ in which block they take from, not in what they hand back.
    void* p;
    if (recycle_) {
        p = allocate_single(size, alignment);
    } else {
        const int wi = arena_worker_index();
        p = (wi >= 0) ? allocate_local(cursors_[wi], size, alignment)
                      : allocate_shared(size, alignment);
    }
    HG_ARENA_UNPOISON(p, size);
    return p;
}

size_t ConcurrentHeterogeneousArena::bytes_allocated() const {
    size_t total = 0;
    Block* block = head_.load(std::memory_order_acquire);
    while (block) {
        total += block->offset.load(std::memory_order_relaxed);
        block = block->prev;
    }
    return total;
}

ConcurrentHeterogeneousArena::Marker ConcurrentHeterogeneousArena::mark() {
    Block* b = current_block_.load(std::memory_order_relaxed);
    return { b, b->offset.load(std::memory_order_relaxed) };
}

void ConcurrentHeterogeneousArena::release(Marker m) {
    Block* b = static_cast<Block*>(m.blk);
    b->offset.store(m.off, std::memory_order_relaxed);
    current_block_.store(b, std::memory_order_relaxed);
}

void ConcurrentHeterogeneousArena::reset() {
    DestructorNode* node = destructor_head_.load(std::memory_order_relaxed);
    while (node) { node->destroy(node->object); node = node->prev; }
    destructor_head_.store(nullptr, std::memory_order_relaxed);
    Block* b = head_.load(std::memory_order_relaxed);
    Block* first = b;
    while (b) {
        b->offset.store(0, std::memory_order_relaxed);
        // Its bytes are unallocated again, and the next request will unpoison what it takes.
        HG_ARENA_POISON(b->data, b->capacity);
        first = b;
        b = b->prev;
    }
    current_block_.store(first, std::memory_order_relaxed);
}

ConcurrentHeterogeneousArena::Block*
ConcurrentHeterogeneousArena::Block::create(size_t data_capacity) {
    void* mem = ::operator new(sizeof(Block) + data_capacity);
    Block* block = static_cast<Block*>(mem);
    block->prev = nullptr;
    block->next = nullptr;
    block->capacity = data_capacity;
    block->offset.store(0, std::memory_order_relaxed);
    HG_ARENA_POISON(block->data, data_capacity);
    g_arena_block_bytes.fetch_add(sizeof(Block) + data_capacity, std::memory_order_relaxed);
    return block;
}

// Bump this worker's private cursor. On overflow, grab a fresh block sized for the request and
// bump from there. block->offset is mirrored (relaxed, to this worker's own block) so
// bytes_allocated() sees the live high-water mark.
void* ConcurrentHeterogeneousArena::allocate_local(LocalCursor& c, size_t size,
                                                   size_t alignment) {
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
    // Ramp this cursor's block size geometrically from INITIAL_BLOCK_SIZE up to
    // block_size_, so a lightly-used worker reserves only a small block.
    size_t grow = c.next_size ? c.next_size
                : (INITIAL_BLOCK_SIZE < block_size_ ? INITIAL_BLOCK_SIZE : block_size_);
    size_t cap = grow;
    size_t need = size + alignment;  // worst-case alignment slack
    if (need > cap) cap = need;      // oversized single request (does not perturb the ramp)
    Block* nb = grab_block(cap);
    c.next_size = grow < block_size_ ? (grow * 2 < block_size_ ? grow * 2 : block_size_)
                                     : block_size_;
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

// Single-threaded bump path for a recycling arena. Bumps the same current_block_/offset pair
// that mark()/release()/reset() ride, so the stack discipline and bytes_allocated() see exactly
// the state they always did -- but with plain loads and stores, since only the owning thread can
// reach this arena. The fields stay atomic for the shared path's sake; relaxed on a single thread
// is a bare mov, where the shared path's compare_exchange is a locked RMW.
void* ConcurrentHeterogeneousArena::allocate_single(size_t size, size_t alignment) {
    while (true) {
        Block* block = current_block_.load(std::memory_order_relaxed);
        size_t offset = block->offset.load(std::memory_order_relaxed);

        size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
        size_t new_offset = aligned_offset + size;

        if (new_offset <= block->capacity) {
            block->offset.store(new_offset, std::memory_order_relaxed);
            return block->data + aligned_offset;
        }
        // Fresh/recycled block starts near offset 0, so the request needs at most
        // size + alignment bytes.
        advance_block(size + alignment);
    }
}

// Shared bump path: an atomic claim on current_block_'s offset. Backs the over-ceiling fallback
// for the concurrent arena.
void* ConcurrentHeterogeneousArena::allocate_shared(size_t size, size_t alignment) {
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

        // Fresh/recycled block starts near offset 0, so the request needs at most
        // size + alignment bytes; pass that so a small ramp block still grows big
        // enough to satisfy an oversized request.
        advance_block(size + alignment);
    }
}

// See ConcurrentArena<T>::allocate_new_block for the rationale: sync current_block_ from head_
// after installing the new block so the last store always reflects the most-recent head rather
// than a racing thread's older block.
// Allocate a block of the given capacity and splice it onto the head of the chain (lock-free).
// Shared by the per-worker cursor path and allocate_new_block.
ConcurrentHeterogeneousArena::Block*
ConcurrentHeterogeneousArena::grab_block(size_t cap) {
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

// Grow the shared chain by one block. Its capacity is the geometric ramp size (shared_grow_,
// doubling up to block_size_) or min_cap, whichever is larger, so a lightly-used arena reserves
// only a small block while an oversized request is still satisfied. shared_grow_ is atomic
// because the concurrent over-ceiling fallback can reach this off the fast path; a race merely
// mis-sizes a block.
void ConcurrentHeterogeneousArena::allocate_new_block(size_t min_cap) {
    size_t grow = shared_grow_.load(std::memory_order_relaxed);
    size_t cap = grow > min_cap ? grow : min_cap;
    grab_block(cap);
    size_t next = grow < block_size_ ? (grow * 2 < block_size_ ? grow * 2 : block_size_)
                                     : block_size_;
    shared_grow_.store(next, std::memory_order_relaxed);

    // Track the most-recent head: a plain store(new_block) lets a racing
    // thread's older block win current_block_ while its newer block sits
    // unreachable mid-chain, stranding that block's capacity.
    current_block_.store(head_.load(std::memory_order_acquire),
                         std::memory_order_release);
}

// Advance to the next block when the current one is full: recycle an already-allocated successor
// (populated after a reset()) if present, else grow. The recycle branch is only reached
// single-threaded (a per-worker scratch arena refilling after reset); the global concurrent
// arena's current_block_ is always the head (next == nullptr), so it always grows. A recycled
// successor too small for the request is handled by allocate_shared's retry loop, which advances
// again until a block fits or the chain ends and grows.
void ConcurrentHeterogeneousArena::advance_block(size_t min_cap) {
    if (recycle_) {
        Block* cur = current_block_.load(std::memory_order_relaxed);
        if (Block* nxt = cur->next) {            // single-threaded: safe to reuse
            nxt->offset.store(0, std::memory_order_relaxed);
            current_block_.store(nxt, std::memory_order_relaxed);
            return;
        }
    }
    allocate_new_block(min_cap);
}

void ConcurrentHeterogeneousArena::register_destructor(void* obj, void (*destroy)(void*)) {
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

ConcurrentHeterogeneousArena& worker_scratch() {
    static thread_local ConcurrentHeterogeneousArena scratch(
        ConcurrentHeterogeneousArena::DEFAULT_BLOCK_SIZE, /*recycle_blocks=*/true);
    return scratch;
}

size_t arena_block_bytes_live() {
    return g_arena_block_bytes.load(std::memory_order_relaxed);
}

// NOT DEFINED UNDER HG_VERIFICATION, because arena.hpp already defines them there as inline
// no-ops and defining them again here is a redefinition the moment this translation unit is
// LINKED INTO a harness rather than left out of one.
//
// The header's reasoning stands and is why the no-ops win rather than these: nothing reads a
// counter to decide anything -- they are relaxed diagnostics beside the protocol, never in it --
// and giving the checker real ones turns every table create, install and discard into a racing
// read-modify-write on one shared location, multiplying executions over a variable no property
// mentions. What has changed is only that a harness can now link the engine, so "the library is
// absent" is no longer what keeps these from colliding; this guard is.
#ifndef HG_VERIFICATION

static std::atomic<size_t> g_discarded_tables{0};
static std::atomic<size_t> g_discarded_table_bytes{0};

void note_discarded_table_bytes(size_t bytes) {
    g_discarded_tables.fetch_add(1, std::memory_order_relaxed);
    g_discarded_table_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

size_t discarded_table_bytes() {
    return g_discarded_table_bytes.load(std::memory_order_relaxed);
}

// Installs, beside the discards, because the ratio is the actionable part: the same waste is a
// tuning problem when it comes from many growth rounds and a protocol problem when it comes from
// a few. Counts as well as bytes, since a table's size varies across the maps.
static std::atomic<size_t> g_installed_table_bytes{0};
static std::atomic<size_t> g_installed_tables{0};

void note_installed_table_bytes(size_t bytes) {
    g_installed_table_bytes.fetch_add(bytes, std::memory_order_relaxed);
    g_installed_tables.fetch_add(1, std::memory_order_relaxed);
}

size_t installed_table_bytes() { return g_installed_table_bytes.load(std::memory_order_relaxed); }
size_t installed_table_count() { return g_installed_tables.load(std::memory_order_relaxed); }
size_t discarded_table_count() { return g_discarded_tables.load(std::memory_order_relaxed); }

#endif  // !HG_VERIFICATION


// =============================================================================
// scratch_alloc.hpp
// =============================================================================
//
// The per-worker PERSISTENT arena and its redirect, alongside worker_scratch() above: the two
// are the same mechanism at two lifetimes, and a reader looking for one wants the other.
// ScratchIdSet comes with them because it allocates from worker_scratch().

ConcurrentHeterogeneousArena*& worker_persistent_target() {
    static thread_local ConcurrentHeterogeneousArena default_arena;
    static thread_local ConcurrentHeterogeneousArena* current = &default_arena;
    return current;
}

ConcurrentHeterogeneousArena& worker_persistent() { return *worker_persistent_target(); }

PersistTarget::PersistTarget(ConcurrentHeterogeneousArena& arena)
    : prev_(worker_persistent_target()) {
    worker_persistent_target() = &arena;
}

PersistTarget::~PersistTarget() { worker_persistent_target() = prev_; }

ScratchIdSet::ScratchIdSet(uint32_t hint) { rehash(round_up_pow2(hint < 8 ? 8 : hint)); }

bool ScratchIdSet::insert(uint32_t key) {
    if (key == kEmpty) { const bool fresh = !has_empty_; has_empty_ = true; return fresh; }
    if (count_ * 4 >= cap_ * 3) rehash(cap_ * 2);   // keep the load factor under 3/4
    return insert_into(slots_, cap_, key);
}

uint32_t ScratchIdSet::round_up_pow2(uint32_t v) {
    uint32_t p = 8; while (p < v) p <<= 1; return p;
}

uint32_t ScratchIdSet::mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

bool ScratchIdSet::insert_into(uint32_t* slots, uint32_t cap, uint32_t key) {
    uint32_t i = mix(key) & (cap - 1);
    for (;;) {
        const uint32_t cur = slots[i];
        if (cur == key) return false;
        if (cur == kEmpty) { slots[i] = key; ++count_; return true; }
        i = (i + 1) & (cap - 1);
    }
}

void ScratchIdSet::rehash(uint32_t cap) {
    uint32_t* fresh = static_cast<uint32_t*>(
        worker_scratch().allocate_raw(sizeof(uint32_t) * cap, alignof(uint32_t)));
    for (uint32_t i = 0; i < cap; ++i) fresh[i] = kEmpty;
    const uint32_t old_cap = cap_;
    uint32_t* old = slots_;
    slots_ = fresh; cap_ = cap; count_ = 0;
    for (uint32_t i = 0; i < old_cap; ++i)
        if (old[i] != kEmpty) insert_into(slots_, cap_, old[i]);
}

}  // namespace engine
}  // namespace HG_NAMESPACE
