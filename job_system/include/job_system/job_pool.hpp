#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>

namespace HG_NAMESPACE {
namespace jobs {

// Per-thread slab pool for job objects, so task submission does no malloc.
//
// Each thread that allocates jobs owns a JobSlotPool. Allocation and same-thread
// deallocation touch only that pool's owner-private free list — no atomics, no
// contention. Because the work-stealing scheduler migrates jobs, a job allocated by
// one thread is often freed by another; a cross-thread free is CAS-pushed onto the
// owning pool's foreign-free list (single-producer-per-push, multi-producer overall)
// and the owner reclaims the whole batch with one atomic exchange when its private
// list runs dry. Push-only-plus-exchange-drain is ABA-free: the only consumer takes
// the entire list atomically, producers only ever prepend.
//
// Slot layout: [SlotHeader (kHeaderSize)][payload (kSlotSize)]. The header records the
// owning pool so a free on any thread can route the slot home. While a slot is free,
// the first pointer of its payload chains the free list. Objects larger than kSlotSize
// fall back to a plain malloc tagged with owner==nullptr; this keeps correctness
// independent of the size estimate (the fixed engine task set fits a slot).
//
// Pools live for the process (a bounded number, one per concurrently-active thread,
// recycled through a free list when a thread exits) so a cross-thread free never
// dereferences a destroyed pool.
class JobSlotPool {
public:
    static constexpr std::size_t kHeaderSize = 16;   // holds owner ptr; keeps payload 16-aligned
    static constexpr std::size_t kSlotSize   = 512;  // payload capacity; covers every engine task
    static constexpr std::size_t kChunkSlots = 256;  // slots per malloc'd chunk

    static void* allocate(std::size_t size);
    static void deallocate(void* payload) noexcept;

private:
    struct SlotHeader { JobSlotPool* owner; };

    static SlotHeader* header(void* payload);
    static void** link(void* payload);

    void* alloc_slot();
    void grow();

    // ---- process-lifetime pool registry ----
    //
    // A retired pool is offered back for reuse by the next thread that needs one. The obvious
    // shape is an intrusive free list, and the obvious lock-free form of that is a Treiber
    // stack -- which here would carry an ABA hazard, because a pool is popped, reused and
    // pushed again, so a CAS on the head can succeed against a value that has been reused
    // underneath it.
    //
    // A SLOT ARRAY HAS NO POINTER TO COMPARE, so there is no ABA to defend against. Pools are
    // never destroyed and their number is bounded by the threads that ever ask for one, so a
    // fixed array of claimable slots holds every pool that will exist. Acquiring is a scan for
    // a slot whose owner can be taken; releasing is a store. Both are lock-free, and neither
    // is on the allocation hot path -- which is a reason the lock was never measured, not a
    // reason to keep it.
    static constexpr std::size_t kMaxPools = 1024;

    struct Registry {
        // occupied[i] true means slots[i] holds a RETIRED pool available for reuse. A slot is
        // claimed by winning the true -> false transition, so exactly one thread can take it.
        std::atomic<JobSlotPool*> slots[kMaxPools];
        std::atomic<bool> occupied[kMaxPools];
        Registry();
    };
    static Registry& registry();

    // A POOL HOMES TO ONE SLOT FOR ITS LIFETIME. Choosing a slot per release would let the same
    // pool sit in two slots at once -- released into a fresh slot while its previous one still
    // held the pointer -- and both could then be claimed, handing one pool to two threads.
    // Recording the index makes release O(1) and the ownership single-valued.
    static JobSlotPool* acquire_pool();
    static void release_pool(JobSlotPool* p);

    static JobSlotPool* tls_pool();

    void* local_free_head_ = nullptr;                                  // owner-thread only
    alignas(64) std::atomic<void*> foreign_free_head_{nullptr};        // cross-thread frees
    char* chunk_list_head_ = nullptr;                                  // for completeness
    int registry_slot_ = -1;                                           // home slot, or -1

    static inline thread_local JobSlotPool* t_pool_ = nullptr;         // trivial; MinGW-safe
};

}  // namespace jobs
}  // namespace HG_NAMESPACE