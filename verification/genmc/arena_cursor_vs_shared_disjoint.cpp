// GenMC harness: the per-worker cursor path and the shared bump path never hand out one address.
//
// WHY THIS PROPERTY. ConcurrentHeterogeneousArena has two allocation disciplines and they are
// safe only while they never bump the same block:
//
//   allocate_local  bumps a PRIVATE cursor and mirrors the result with a plain relaxed store:
//                       c.offset = new_offset; c.block->offset.store(new_offset, relaxed);
//   allocate_shared reserves on current_block_ with a compare-exchange on that SAME field.
//
// A store and a compare-exchange on one offset is not a reservation: the store erases what the
// exchange just claimed, and both callers are handed the same bytes. The two meet because a
// cursor grows through grab_block, which CASes its private block onto head_, so a cursor's live
// block can be the chain head -- and what allocate_new_block publishes into current_block_
// decides whether the shared path then bumps it.
//
// WHY IT NEEDS TWO THREADS AND NOT MORE. A thread past the worker ceiling takes the shared path.
// HG_MAX_ARENA_WORKERS is 1 here, so the first thread to ask for an index takes the cursor and
// the second is turned away to the shared path: the tightest interleaving in which both
// disciplines are live on one arena. HG_ARENA_BLOCK_SIZE is 512.
//
// THE WORKLOAD. Each thread asks for its index first and allocates by role. The cursor thread
// makes two 16-byte allocations: the first grabs its block (grab_block puts it on head_), the
// second bumps it privately. The shared thread makes two 400-byte allocations: the first fits
// the block the constructor made, the second does not and runs allocate_new_block, which
// publishes a block, and the claim that follows lands on whichever block was published. This is
// the smallest workload in which a cursor's block can be on head_ while the shared path
// publishes and both then bump it. The assertion is that the four regions are pairwise
// disjoint. Each thread also WRITES its own byte into what it is given and reads it back, so an
// overlap is additionally a data race on those bytes.
//
// WHAT IS BOUNDED. Every weak compare-exchange on the path (the shared claim, grab_block's head_
// CAS and note_block_grab's two high-water loops per block) has its spurious failures explored,
// at most one in a row per thread and address (GENMC-ARGS). The high-water loop is checked
// alone by arena_high_water_max.
//
// CALIBRATION. Publishing head_ rather than the block the call created --
//     current_block_.store(head_.load(std::memory_order_acquire), std::memory_order_release);
// in allocate_new_block (HG_CALIBRATE_ARENA_PUBLISH_HEAD) -- must make the checker report the
// overlap.
//
// GENMC-ARGS: --disable-estimation --weak-cas-spurious-bound=1
// GENMC-BOUND: weak CAS spurious-failure bound 1: 406,912 complete executions, K = 286,343 dropped (bounded, not exhaustive).
// GENMC-EXPECT: pass
// GENMC-CALIBRATE: -DHG_CALIBRATE_ARENA_PUBLISH_HEAD
//
// Build/run: verification/genmc/run.sh arena_cursor_vs_shared_disjoint

#define HG_MAX_ARENA_WORKERS 1
#define HG_ARENA_BLOCK_SIZE 512

#include <pthread.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>

#include "genmc_support.hpp"

// The allocator's bodies live in arena.cpp, so the harness compiles that translation unit
// rather than linking the whole engine: the property is the allocator's alone, and every
// function it names is defined there.
// The cursor array is over-aligned, so its allocation compiles to
// operator new[](size_t, align_val_t), which GenMC does not implement. The unaligned form
// satisfies it: alignment beyond the natural alignment is not a property this harness states.
//
// genmc_support.cpp carries the same shim for COMPOSED harnesses, and this cannot share that
// definition: a replacement allocation function may not be inline, so it is defined once per
// program, and a single-TU harness and a linked one are two programs. genmc_support.hpp and
// genmc_support.cpp also both define __dso_handle, so a translation unit takes one or the other.
void* operator new(std::size_t n, std::align_val_t) { return ::operator new(n ? n : 1); }
void* operator new[](std::size_t n, std::align_val_t) { return ::operator new(n ? n : 1); }
void operator delete(void* p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::align_val_t) noexcept { ::operator delete(p); }

#include "hypergraph/arena.hpp"
#include "../../hypergraph/src/arena.cpp"


namespace {

constexpr size_t kPerThread = 2;
constexpr size_t kCursorSize = 16;
constexpr size_t kSharedSize = 400;   // two do not fit one 512-byte block

hypergraph::ConcurrentHeterogeneousArena* g_arena;
unsigned char* g_got[2][kPerThread];
size_t g_size[2];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    const unsigned char stamp = static_cast<unsigned char>(id + 1);
    const size_t size = hypergraph::arena_worker_index() >= 0 ? kCursorSize : kSharedSize;
    g_size[id] = size;
    for (size_t i = 0; i < kPerThread; ++i) {
        unsigned char* p = static_cast<unsigned char*>(
            g_arena->allocate_raw(size, alignof(max_align_t), nullptr));
        g_got[id][i] = p;
        // Writing and reading back makes an overlap a data race on these bytes as well as a
        // failed assertion, so the checker reports it even if the assertion is unreachable.
        *p = stamp;
        assert(*p == stamp);
    }
    return nullptr;
}

bool overlaps(const unsigned char* a, size_t na, const unsigned char* b, size_t nb) {
    return a < b + nb && b < a + na;
}

}  // namespace

int main() {
    hypergraph::ConcurrentHeterogeneousArena arena;
    g_arena = &arena;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // Pairwise disjoint, across both threads and within each.
    for (long a = 0; a < 2; ++a)
        for (size_t i = 0; i < kPerThread; ++i)
            for (long b = 0; b < 2; ++b)
                for (size_t j = 0; j < kPerThread; ++j)
                    if (a != b || i != j)
                        assert(!overlaps(g_got[a][i], g_size[a], g_got[b][j], g_size[b]));
    return 0;
}
