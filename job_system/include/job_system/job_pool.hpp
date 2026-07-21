#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>

namespace job_system {

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

    // ---- process-lifetime pool registry (off the hot path) ----

    static std::mutex& registry_mutex() {
        static std::mutex m;
        return m;
    }
    static JobSlotPool*& free_list_head() {   // retired pools available for reuse
        static JobSlotPool* head = nullptr;
        return head;
    }

    static JobSlotPool* acquire_pool() {
        std::lock_guard<std::mutex> lock(registry_mutex());
        if (JobSlotPool* p = free_list_head()) {
            free_list_head() = p->next_free_;
            p->next_free_ = nullptr;
            return p;
        }
        return new JobSlotPool();  // never freed by design (bounded, process-lifetime)
    }
    static void release_pool(JobSlotPool* p) {
        std::lock_guard<std::mutex> lock(registry_mutex());
        p->next_free_ = free_list_head();
        free_list_head() = p;
    }

    struct TlsGuard {
        ~TlsGuard() {
            if (t_pool_) { release_pool(t_pool_); t_pool_ = nullptr; }
        }
    };

    static JobSlotPool* tls_pool() {
        if (t_pool_ == nullptr) {
            // Construct the registry statics before the thread-local guard so they are
            // destroyed after it: the main thread's guard runs release_pool() during
            // process exit and must still find a live mutex and free list.
            (void)registry_mutex();
            (void)free_list_head();
            (void)t_guard_;             // ODR-use so the exit hook is registered
            t_pool_ = acquire_pool();
        }
        return t_pool_;
    }

    void* local_free_head_ = nullptr;                                  // owner-thread only
    alignas(64) std::atomic<void*> foreign_free_head_{nullptr};        // cross-thread frees
    char* chunk_list_head_ = nullptr;                                  // for completeness
    JobSlotPool* next_free_ = nullptr;                                 // reuse free list

    static inline thread_local JobSlotPool* t_pool_ = nullptr;
    static inline thread_local TlsGuard t_guard_{};
};

} // namespace job_system
