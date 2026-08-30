#include "hgcommon/core.hpp"
#include "hypergraph/arena.hpp"
#include <hgcommon/portable_intrinsics.hpp>
#include <algorithm>
#include <vector>
#include <cstdio>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif !defined(HG_VERIFICATION)
#include <sys/mman.h>
#endif
#include <cstdlib>
#include "hypergraph/scratch_alloc.hpp"
#include "hgcommon/pool_core.hpp"



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
// The largest the live total has been: what the arenas contributed to the resident set's
// high-water mark, which a count at the end of a run (blocks already freed) cannot say.
static std::atomic<size_t> g_arena_block_bytes_high_water{0};
// The same high-water kept per arena kind: the recycling (per-worker scratch) arenas and
// the rest, so a footprint that grows with the worker count is attributed to one of them.
static std::atomic<size_t> g_scratch_block_bytes{0};
static std::atomic<size_t> g_scratch_block_bytes_high_water{0};
static void note_block_grab(std::atomic<size_t>& live, std::atomic<size_t>& high_water, size_t bytes) {
    const size_t now = live.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    size_t hw = high_water.load(std::memory_order_relaxed);
    while (now > hw && !high_water.compare_exchange_weak(hw, now, std::memory_order_relaxed)) {}
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
        Block* prev = block->prev.load(std::memory_order_relaxed);
        // operator delete must not be handed a region this file poisoned.
        HG_ARENA_UNPOISON(block->data, block->capacity);
        g_arena_block_bytes.fetch_sub(sizeof(Block) + block->capacity, std::memory_order_relaxed);
        if (recycle_) g_scratch_block_bytes.fetch_sub(sizeof(Block) + block->capacity, std::memory_order_relaxed);
        // What the block's next life may not assume zero: everything ever bumped past, or
        // the whole block for a recycling arena, whose reset() rewound the offset over it.
        const size_t used = block->offset.load(std::memory_order_relaxed);
        if (recycle_) block->dirty_end = block->capacity;
        else if (block->dirty_end < used) block->dirty_end = used;
        Block::release(block);
        block = prev;
    }

    delete[] cursors_;
}

#if HG_ENGINE_STATS
// BYTES PER ALLOCATION SITE (stats builds). The site is the return address of allocate_raw,
// which the inline create<T>/allocate_array<T> wrappers make the engine function that asked;
// a run at two worker counts with the same output and the same call counts per site but
// more bytes is then attributed to the sites whose requests grew. Fixed open-addressed table,
// relaxed atomics: an instrument, not a protocol.
namespace {
struct AllocSite { std::atomic<void*> addr{nullptr}; std::atomic<size_t> bytes{0}; std::atomic<size_t> calls{0}; };
constexpr size_t kAllocSites = 4096;
AllocSite g_alloc_sites[kAllocSites];
void note_alloc_site(void* ret, size_t size) {
    size_t h = (reinterpret_cast<size_t>(ret) >> 2) * 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < kAllocSites; ++i) {
        AllocSite& e = g_alloc_sites[(h + i) & (kAllocSites - 1)];
        void* cur = e.addr.load(std::memory_order_relaxed);
        if (cur == ret || (cur == nullptr && (e.addr.compare_exchange_strong(cur, ret, std::memory_order_relaxed) || cur == ret))) {
            e.bytes.fetch_add(size, std::memory_order_relaxed);
            e.calls.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}
}  // namespace

void arena_alloc_profile_dump(FILE* out, size_t top) {
    std::vector<const AllocSite*> v;
    for (const AllocSite& e : g_alloc_sites)
        if (e.addr.load(std::memory_order_relaxed)) v.push_back(&e);
    std::sort(v.begin(), v.end(), [](const AllocSite* a, const AllocSite* b) {
        return a->bytes.load(std::memory_order_relaxed) > b->bytes.load(std::memory_order_relaxed);
    });
    for (size_t i = 0; i < v.size() && i < top; ++i)
        std::fprintf(out, "alloc_site %p bytes=%zu calls=%zu\n", v[i]->addr.load(std::memory_order_relaxed),
                     v[i]->bytes.load(std::memory_order_relaxed), v[i]->calls.load(std::memory_order_relaxed));
}
#endif

void* ConcurrentHeterogeneousArena::allocate_raw_slow(size_t size, size_t alignment, bool* zero) {
    // One choke point for every request, which is where the unpoison belongs: the three paths
    // below differ in which block they take from, not in what they hand back.
    HG_STAT(note_alloc_site(HG_RETURN_ADDRESS(), size));
    void* p;
    if (recycle_) {
        // reset() rewinds offsets over live contents: nothing here is known to be zero.
        if (zero) *zero = false;
        p = allocate_single(size, alignment);
    } else {
        const int wi = arena_worker_index();
        p = (wi >= 0) ? allocate_local(cursors_[wi], size, alignment, zero)
                      : allocate_shared(size, alignment, zero);
    }
    HG_ARENA_UNPOISON(p, size);
    return p;
}

size_t ConcurrentHeterogeneousArena::bytes_allocated() const {
    size_t total = 0;
    Block* block = head_.load(std::memory_order_acquire);
    while (block) {
        total += block->offset.load(std::memory_order_relaxed);
        block = block->prev.load(std::memory_order_relaxed);
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
        b = b->prev.load(std::memory_order_relaxed);
    }
    current_block_.store(first, std::memory_order_relaxed);
}

// A BLOCK IS AN ANONYMOUS MAPPING, for two properties the heap allocator gives neither of.
//
// ZERO BYTES. The kernel zero-fills an anonymous page on first touch, and the arena never
// hands a byte out twice (see Block::dirty_end), so every allocation receives zeros without the arena
// writing them: allocate_array and the hash tables skip their fills on that, which were 3.0% of
// all instructions and 18.9% of all stores on wpp depth 7 (callgrind, one thread).
//
// HUGE PAGES, ASKED FOR EXPLICITLY. Measured on the EPYC at 32 workers, wpp depth 7: 1,060,520
// minor page faults, and 9.3% of all cycles inside the kernel -- clear_page_erms zeroing fresh
// 4 KB pages, down_read_trylock on mmap_sem, and native_queued_spin_lock_slowpath where 32
// threads meet on it. That is the fault path, not the engine. Two things are needed together
// and neither works alone: the block must be 2 MB ALIGNED and at least 2 MB, or there is no
// huge page for the kernel to use, and it must be ASKED FOR, because transparent huge pages
// run in `madvise` mode on this box and on most distributions, where an unrequested mapping
// gets 4 KB pages however large it is. A block of a huge page or more is placed on the first
// huge-page boundary inside a mapping one huge page longer than it needs; the slack at either
// end stays untouched, so it costs address space and no memory, and goes back with the block.
// A smaller block is a plain page-aligned mapping.
//
// Under HG_VERIFICATION a block is operator new: the checker's interpreter models the heap and
// not mmap, and a 2 MB block is a 2 MB memset it has to promote one store at a time (the engine
// harnesses define HG_ARENA_BLOCK_SIZE). Windows has VirtualAlloc, which zero-fills likewise.
//
// THE POOL. A released block goes back to a process-wide pool and the next arena that needs
// its size class takes it from there rather than mapping afresh, which is what the C heap
// did for the arena before and what keeps a run of evolutions from faulting its working set
// in again each time (measured on the EPYC, wpp depth 7, five reps, 16 workers: mapping
// afresh per arena was 118k page faults against 83k and 5% of the wall, all of it the
// sub-huge-page ramp blocks re-faulted under the mm lock). One Treiber stack per power-of-two
// class of mapping length, heads tagged against ABA (16-bit tag above a 48-bit pointer,
// which is every user-space pointer on the platforms built); a pooled block is never
// unmapped, so reading its link after the head is loaded is always a read of mapped memory.
// A block's dirty_end travels with it, so what the pool hands back is zero exactly where
// its previous life never wrote. Absent under HG_VERIFICATION, where a block is operator new.
namespace {
#if !defined(HG_VERIFICATION)
constexpr int kPoolClasses = 48;
std::atomic<uint64_t> g_block_pool[kPoolClasses];
inline int pool_class_of(size_t map_len) {              // ceil(log2(map_len))
    int c = 0;
    while ((size_t(1) << c) < map_len) ++c;
    return c;
}
// The storage half of hgcommon's tagged free-list rule, over the pool heads and Block::prev.
// The link accessors ride the atomic prev field because a rival popper reads it speculatively; the
// decision -- and the reason -- live in pool_core.hpp, which the GenMC harness
// block_pool_exactly_once.cpp checks without this file's mmap machinery.
struct BlockPoolOps {
    std::atomic<uint64_t>& head;
    uint64_t head_load() { return head.load(std::memory_order_acquire); }
    bool head_cas(uint64_t& expected, uint64_t desired) {
        return head.compare_exchange_weak(expected, desired, std::memory_order_acq_rel,
                                          std::memory_order_acquire);
    }
    uint64_t link_load(uint64_t node) {
        return ConcurrentHeterogeneousArena::pool_link_load(node);
    }
    void link_store(uint64_t node, uint64_t v) {
        ConcurrentHeterogeneousArena::pool_link_store(node, v);
    }
};
#endif
}  // namespace

ConcurrentHeterogeneousArena::Block*
ConcurrentHeterogeneousArena::Block::create(size_t data_capacity) {
    size_t total = sizeof(Block) + data_capacity;
    void*  mem      = nullptr;
    void*  map_base = nullptr;
    size_t map_len  = 0;
#if defined(HG_VERIFICATION)
    mem = ::operator new(total);
#else
    const bool huge = total >= kHugePageBytes;
    if (huge) total = (total + kHugePageBytes - 1) & ~(kHugePageBytes - 1);
    map_len = huge ? total + kHugePageBytes : total;
    // A pooled block of the same class (the ramp asks for the same few sizes over and over,
    // so the class is the size). One whose capacity falls short of this request -- the class
    // spans a factor of two -- goes back and the request maps afresh.
    {
        BlockPoolOps ops{g_block_pool[pool_class_of(map_len)]};
        while (uint64_t node = hgcommon::pool_core_pop(ops)) {
            Block* b = reinterpret_cast<Block*>(node);
            if (b->capacity < data_capacity) { hgcommon::pool_core_push(ops, node); break; }
            ops.link_store(node, 0);
            b->next = nullptr;
            b->offset.store(0, std::memory_order_relaxed);
            HG_ARENA_POISON(b->data, b->capacity);
            note_block_grab(g_arena_block_bytes, g_arena_block_bytes_high_water, sizeof(Block) + b->capacity);
            return b;
        }
    }
#if defined(_WIN32)
    map_base = VirtualAlloc(nullptr, map_len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!map_base) throw std::bad_alloc();
#else
    map_base = ::mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map_base == MAP_FAILED) throw std::bad_alloc();
#endif
    mem = map_base;
    if (huge) {
        const uintptr_t base  = reinterpret_cast<uintptr_t>(map_base);
        const uintptr_t start = (base + kHugePageBytes - 1) & ~uintptr_t(kHugePageBytes - 1);
        mem = reinterpret_cast<void*>(start);
#if defined(MADV_HUGEPAGE)
        // Advisory: a kernel that cannot back it says so and the mapping still works. Linux
        // names the advice; macOS has no equivalent and its mapping is what mmap returned.
        ::madvise(mem, total, MADV_HUGEPAGE);
#endif
    }
    data_capacity = total - sizeof(Block);   // the rounding slack is usable, not wasted
#endif
    Block* block = static_cast<Block*>(mem);
    block->prev.store(nullptr, std::memory_order_relaxed);
    block->next = nullptr;
    block->capacity = data_capacity;
    block->map_base = map_base;
    block->map_len  = map_len;
#if defined(HG_VERIFICATION)
    block->dirty_end = data_capacity;
#else
    block->dirty_end = 0;
#endif
    block->offset.store(0, std::memory_order_relaxed);
    HG_ARENA_POISON(block->data, data_capacity);
    note_block_grab(g_arena_block_bytes, g_arena_block_bytes_high_water, sizeof(Block) + data_capacity);
    return block;
}

void ConcurrentHeterogeneousArena::Block::release(Block* block) {
#if defined(HG_VERIFICATION)
    ::operator delete(block);
#else
    pool_push(g_block_pool[pool_class_of(block->map_len)], block);
#endif
}

uint64_t ConcurrentHeterogeneousArena::pool_link_load(uint64_t node) {
    Block* b = reinterpret_cast<Block*>(node);
    return reinterpret_cast<uint64_t>(b->prev.load(std::memory_order_relaxed));
}
void ConcurrentHeterogeneousArena::pool_link_store(uint64_t node, uint64_t v) {
    Block* b = reinterpret_cast<Block*>(node);
    b->prev.store(reinterpret_cast<Block*>(v), std::memory_order_relaxed);
}

void ConcurrentHeterogeneousArena::Block::pool_push(std::atomic<uint64_t>& head, Block* block) {
#if defined(HG_VERIFICATION)
    (void)head; (void)block;
#else
    BlockPoolOps ops{head};
    hgcommon::pool_core_push(ops, reinterpret_cast<uint64_t>(block));
#endif
}

// Bump this worker's private cursor. On overflow, grab a fresh block sized for the request and
// bump from there. block->offset is mirrored (relaxed, to this worker's own block) so
// bytes_allocated() sees the live high-water mark.
void* ConcurrentHeterogeneousArena::allocate_local(LocalCursor& c, size_t size,
                                                   size_t alignment, bool* zero) {
    if (c.block) {
        size_t aligned = (c.offset + alignment - 1) & ~(alignment - 1);
        size_t new_offset = aligned + size;
        if (new_offset <= c.capacity) {
            c.offset = new_offset;
            c.block->offset.store(new_offset, std::memory_order_relaxed);
            if (zero) *zero = aligned >= c.block->dirty_end;
            return c.block->data + aligned;
        }
    }
    // Current block can't fit this request; take a fresh one (shared, but rare).
    // Ramp this cursor's block size geometrically from INITIAL_BLOCK_SIZE up to
    // block_size_, so a lightly-used worker reserves only a small block.
    size_t need = size + alignment;  // worst-case alignment slack
    Block* nb;
    if (c.spare && c.spare->capacity >= need) {
        // The block a give-back emptied earlier; reusing it reserves nothing and leaves the
        // ramp where it is.
        nb = c.spare;
        c.spare = nullptr;
    } else {
        size_t grow = c.next_size ? c.next_size
                    : (INITIAL_BLOCK_SIZE < block_size_ ? INITIAL_BLOCK_SIZE : block_size_);
        size_t cap = grow;
        if (need > cap) cap = need;      // oversized single request (does not perturb the ramp)
        nb = grab_block(cap);
        c.next_size = grow < block_size_ ? (grow * 2 < block_size_ ? grow * 2 : block_size_)
                                         : block_size_;
    }
    c.prev_block    = c.block;
    c.prev_offset   = c.offset;
    c.prev_capacity = c.capacity;
    c.block = nb;
    c.capacity = nb->capacity;
    // Fresh block: data is max_align_t-aligned, so an offset-relative alignment
    // (<= max_align_t, as for every arena request) starts at 0.
    size_t aligned = (alignment - 1) & ~(alignment - 1);  // == 0
    size_t new_offset = aligned + size;
    c.offset = new_offset;
    nb->offset.store(new_offset, std::memory_order_relaxed);
    if (zero) *zero = aligned >= nb->dirty_end;
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
void* ConcurrentHeterogeneousArena::allocate_shared(size_t size, size_t alignment, bool* zero) {
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
                if (zero) *zero = aligned_offset >= block->dirty_end;
                return block->data + aligned_offset;
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
    if (recycle_) note_block_grab(g_scratch_block_bytes, g_scratch_block_bytes_high_water, sizeof(Block) + cap);
    note_block_grab(block_bytes_, block_bytes_high_water_, sizeof(Block) + cap);

    Block* old_head = head_.load(std::memory_order_acquire);
    do {
        // Relaxed: a pool rival that lost this block an instant ago may still be reading
        // the link speculatively; the field is atomic for exactly that overlap, and the
        // publication that matters is the head CAS below.
        new_block->prev.store(old_head, std::memory_order_relaxed);
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
    HG_THREAD_LOCAL(ConcurrentHeterogeneousArena, scratch,
                    ConcurrentHeterogeneousArena::DEFAULT_BLOCK_SIZE, /*recycle_blocks=*/true);
    return scratch;
}

size_t arena_block_bytes_live() {
    return g_arena_block_bytes.load(std::memory_order_relaxed);
}

size_t arena_block_bytes_high_water() {
    return g_arena_block_bytes_high_water.load(std::memory_order_relaxed);
}

size_t arena_scratch_block_bytes_high_water() {
    return g_scratch_block_bytes_high_water.load(std::memory_order_relaxed);
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
    HG_STAT(g_discarded_tables.fetch_add(1, std::memory_order_relaxed));
    HG_STAT(g_discarded_table_bytes.fetch_add(bytes, std::memory_order_relaxed));
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
    HG_STAT(g_installed_table_bytes.fetch_add(bytes, std::memory_order_relaxed));
    HG_STAT(g_installed_tables.fetch_add(1, std::memory_order_relaxed));
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

ConcurrentHeterogeneousArena*& worker_persistent_target() {
    HG_THREAD_LOCAL(ConcurrentHeterogeneousArena, default_arena);
    static thread_local ConcurrentHeterogeneousArena* current = &default_arena;
    return current;
}

ConcurrentHeterogeneousArena& worker_persistent() { return *worker_persistent_target(); }

PersistTarget::PersistTarget(ConcurrentHeterogeneousArena& arena)
    : prev_(worker_persistent_target()) {
    worker_persistent_target() = &arena;
}

PersistTarget::~PersistTarget() { worker_persistent_target() = prev_; }

}  // namespace engine
}  // namespace HG_NAMESPACE
