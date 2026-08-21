#include "hypergraph/arena.hpp"

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
        ::operator delete(block);
        block = prev;
    }

    delete[] cursors_;
}

void* ConcurrentHeterogeneousArena::allocate_raw(size_t size, size_t alignment) {
    if (recycle_) return allocate_single(size, alignment);
    int wi = arena_worker_index();
    if (wi >= 0) return allocate_local(cursors_[wi], size, alignment);
    return allocate_shared(size, alignment);
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
    while (b) { b->offset.store(0, std::memory_order_relaxed); first = b; b = b->prev; }
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

}  // namespace engine
}  // namespace HG_NAMESPACE
