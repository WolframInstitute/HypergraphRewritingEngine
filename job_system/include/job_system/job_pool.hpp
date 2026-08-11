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

    static void* allocate(std::size_t size) {
        if (size > kSlotSize) {
            char* base = static_cast<char*>(std::malloc(kHeaderSize + size));
            header(base + kHeaderSize)->owner = nullptr;  // heap-fallback marker
            return base + kHeaderSize;
        }
        return tls_pool()->alloc_slot();
    }

    static void deallocate(void* payload) noexcept {
        JobSlotPool* owner = header(payload)->owner;
        if (owner == nullptr) {                          // heap fallback
            std::free(static_cast<char*>(payload) - kHeaderSize);
            return;
        }
        if (owner == t_pool_) {                          // same-thread: owner-private list
            *link(payload) = owner->local_free_head_;
            owner->local_free_head_ = payload;
            return;
        }
        void* old = owner->foreign_free_head_.load(std::memory_order_relaxed);
        do {
            *link(payload) = old;
        } while (!owner->foreign_free_head_.compare_exchange_weak(
            old, payload, std::memory_order_release, std::memory_order_relaxed));
    }

private:
    struct SlotHeader { JobSlotPool* owner; };

    static SlotHeader* header(void* payload) {
        return reinterpret_cast<SlotHeader*>(static_cast<char*>(payload) - kHeaderSize);
    }
    static void** link(void* payload) { return reinterpret_cast<void**>(payload); }

    void* alloc_slot() {
        if (local_free_head_ == nullptr) {
            void* batch = foreign_free_head_.exchange(nullptr, std::memory_order_acquire);
            if (batch) local_free_head_ = batch;
            else grow();
        }
        void* payload = local_free_head_;
        if (payload == nullptr) return nullptr;   // out of memory
        local_free_head_ = *link(payload);
        header(payload)->owner = this;
        return payload;
    }

    void grow() {
        constexpr std::size_t stride = kHeaderSize + kSlotSize;
        char* chunk = static_cast<char*>(std::malloc(kHeaderSize + stride * kChunkSlots));
        if (chunk == nullptr) return;                         // OOM: caller throws bad_alloc
        *reinterpret_cast<char**>(chunk) = chunk_list_head_;  // intrusive chunk chain
        chunk_list_head_ = chunk;
        char* slots = chunk + kHeaderSize;
        void* head = nullptr;
        for (std::size_t i = 0; i < kChunkSlots; ++i) {
            void* payload = slots + i * stride + kHeaderSize;
            *link(payload) = head;
            head = payload;
        }
        local_free_head_ = head;
    }

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
        Registry() {
            for (std::size_t i = 0; i < kMaxPools; ++i) {
                slots[i].store(nullptr, std::memory_order_relaxed);
                occupied[i].store(false, std::memory_order_relaxed);
            }
        }
    };
    static Registry& registry() {
        static Registry r;
        return r;
    }

    // A POOL HOMES TO ONE SLOT FOR ITS LIFETIME. Choosing a slot per release would let the same
    // pool sit in two slots at once -- released into a fresh slot while its previous one still
    // held the pointer -- and both could then be claimed, handing one pool to two threads.
    // Recording the index makes release O(1) and the ownership single-valued.
    static JobSlotPool* acquire_pool() {
        Registry& r = registry();
        for (std::size_t i = 0; i < kMaxPools; ++i) {
            if (!r.occupied[i].load(std::memory_order_acquire)) continue;
            bool expected = true;
            // Acquire on success pairs with the release below, so the retiring thread's writes
            // -- including registry_slot_ -- are visible to whoever claims the pool.
            if (r.occupied[i].compare_exchange_strong(expected, false,
                                                      std::memory_order_acquire,
                                                      std::memory_order_relaxed)) {
                return r.slots[i].load(std::memory_order_relaxed);
            }
        }
        return new JobSlotPool();  // never freed by design (bounded, process-lifetime)
    }

    static void release_pool(JobSlotPool* p) {
        Registry& r = registry();
        if (p->registry_slot_ >= 0) {
            r.occupied[p->registry_slot_].store(true, std::memory_order_release);
            return;
        }
        for (std::size_t i = 0; i < kMaxPools; ++i) {
            JobSlotPool* empty = nullptr;
            if (r.slots[i].compare_exchange_strong(empty, p, std::memory_order_relaxed,
                                                   std::memory_order_relaxed)) {
                p->registry_slot_ = static_cast<int>(i);
                r.occupied[i].store(true, std::memory_order_release);
                return;
            }
        }
        // Registry full: more than kMaxPools threads have ever held a pool. This one is simply
        // not offered for reuse, which costs the next thread one allocation. Never blocks, and
        // never hands out a pool another thread still owns.
    }

    static JobSlotPool* tls_pool() {
        if (t_pool_ == nullptr) {
            // Touch the registry before constructing the guard so it outlives it: the guard
            // runs release_pool() at thread/process exit and must still find a live registry.
            (void)registry();
            // Function-local guard, NOT a class-scope `inline thread_local`: a non-trivial
            // inline thread_local emits a per-TU TLS-init function GCC folds via COMDAT but
            // MinGW's ld does not (the Windows paclet failed to link on t_guard_). A
            // function-local static thread_local carries a single guard tied to this inline
            // function, which folds on every target. t_pool_ itself is a trivial pointer, so
            // its class-scope inline thread_local is MinGW-safe and stays (deallocate reads it).
            struct TlsGuard {
                ~TlsGuard() { if (t_pool_) { release_pool(t_pool_); t_pool_ = nullptr; } }
            };
            static thread_local TlsGuard guard;   // recycles the pool at thread exit
            (void)guard;
            t_pool_ = acquire_pool();
        }
        return t_pool_;
    }

    void* local_free_head_ = nullptr;                                  // owner-thread only
    alignas(64) std::atomic<void*> foreign_free_head_{nullptr};        // cross-thread frees
    char* chunk_list_head_ = nullptr;                                  // for completeness
    int registry_slot_ = -1;                                           // home slot, or -1

    static inline thread_local JobSlotPool* t_pool_ = nullptr;         // trivial; MinGW-safe
};

}  // namespace jobs
}  // namespace HG_NAMESPACE