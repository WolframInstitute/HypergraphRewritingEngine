#include "job_system/job_pool.hpp"

// The bodies behind job_pool.hpp. Everything here is non-template, so a translation unit
// that submits jobs links these rather than recompiling them.

namespace HG_NAMESPACE {
namespace jobs {

void* JobSlotPool::allocate(std::size_t size) {
    if (size > kSlotSize) {
        char* base = static_cast<char*>(std::malloc(kHeaderSize + size));
        header(base + kHeaderSize)->owner = nullptr;  // heap-fallback marker
        return base + kHeaderSize;
    }
    return tls_pool()->alloc_slot();
}

void JobSlotPool::deallocate(void* payload) noexcept {
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

JobSlotPool::SlotHeader* JobSlotPool::header(void* payload) {
    return reinterpret_cast<SlotHeader*>(static_cast<char*>(payload) - kHeaderSize);
}

void** JobSlotPool::link(void* payload) { return reinterpret_cast<void**>(payload); }

void* JobSlotPool::alloc_slot() {
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

void JobSlotPool::grow() {
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

JobSlotPool::Registry::Registry() {
    for (std::size_t i = 0; i < kMaxPools; ++i) {
        slots[i].store(nullptr, std::memory_order_relaxed);
        occupied[i].store(false, std::memory_order_relaxed);
    }
}

JobSlotPool::Registry& JobSlotPool::registry() {
    static Registry r;
    return r;
}

JobSlotPool* JobSlotPool::acquire_pool() {
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

void JobSlotPool::release_pool(JobSlotPool* p) {
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

JobSlotPool* JobSlotPool::tls_pool() {
    if (t_pool_ == nullptr) {
        // Touch the registry before constructing the guard so it outlives it: the guard
        // runs release_pool() at thread/process exit and must still find a live registry.
        (void)registry();
        struct TlsGuard {
            ~TlsGuard() { if (t_pool_) { release_pool(t_pool_); t_pool_ = nullptr; } }
        };
        static thread_local TlsGuard guard;   // recycles the pool at thread exit
        (void)guard;
        t_pool_ = acquire_pool();
    }
    return t_pool_;
}

}  // namespace jobs
}  // namespace HG_NAMESPACE
