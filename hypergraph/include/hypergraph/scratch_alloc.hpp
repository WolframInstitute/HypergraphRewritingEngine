#pragma once

#include "arena.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hypergraph {

// std::allocator that draws from the calling thread's worker_scratch() arena.
// deallocate() is a no-op: memory is reclaimed in bulk by the arena's reset()
// (per task) or release(mark) (per scope). Use ONLY for transient scratch whose
// lifetime is bounded by such a reclaim point — never for data that must outlive
// the current task / mark.
template<class T>
struct ScratchAlloc {
    using value_type = T;
    ScratchAlloc() noexcept = default;
    template<class U> ScratchAlloc(const ScratchAlloc<U>&) noexcept {}
    T* allocate(std::size_t n) {
        return static_cast<T*>(worker_scratch().allocate_raw(n * sizeof(T), alignof(T)));
    }
    void deallocate(T*, std::size_t) noexcept {}
    template<class U> bool operator==(const ScratchAlloc<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const ScratchAlloc<U>&) const noexcept { return false; }
};

// Per-worker PERSISTENT allocation target: thread-local, for data that outlives the
// task that created it (e.g. cached parent WL histories). Per worker => no allocator
// contention; cross-worker READS are safe (arena memory is stable, never moved).
//
// It is a settable POINTER, defaulting to a per-worker arena. A pool of resettable
// arenas can redirect it (via PersistTarget below) so a cached object is built into
// its own arena and reclaimed by resetting that arena on eviction — the A4 reclaim
// mechanism, without stateful/nested-propagating allocators.
inline ConcurrentHeterogeneousArena*& worker_persistent_target() {
    static thread_local ConcurrentHeterogeneousArena default_arena;
    static thread_local ConcurrentHeterogeneousArena* current = &default_arena;
    return current;
}
inline ConcurrentHeterogeneousArena& worker_persistent() { return *worker_persistent_target(); }

// RAII: redirect worker_persistent() to `arena` for this scope (single-threaded per
// worker). Everything allocated via PersistAlloc while in scope lands in `arena`.
struct PersistTarget {
    ConcurrentHeterogeneousArena* prev_;
    explicit PersistTarget(ConcurrentHeterogeneousArena& arena) : prev_(worker_persistent_target()) {
        worker_persistent_target() = &arena;
    }
    ~PersistTarget() { worker_persistent_target() = prev_; }
    PersistTarget(const PersistTarget&) = delete;
    PersistTarget& operator=(const PersistTarget&) = delete;
};

// std::allocator drawing from worker_persistent(). deallocate is a no-op (reclaim
// is bulk, via A4). Use ONLY for data whose lifetime is bounded by that reclaim.
template<class T>
struct PersistAlloc {
    using value_type = T;
    PersistAlloc() noexcept = default;
    template<class U> PersistAlloc(const PersistAlloc<U>&) noexcept {}
    T* allocate(std::size_t n) {
        return static_cast<T*>(worker_persistent().allocate_raw(n * sizeof(T), alignof(T)));
    }
    void deallocate(T*, std::size_t) noexcept {}
    template<class U> bool operator==(const PersistAlloc<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const PersistAlloc<U>&) const noexcept { return false; }
};

template<class T> using SVec = std::vector<T, ScratchAlloc<T>>;
template<class T> using PVec = std::vector<T, PersistAlloc<T>>;
template<class K, class C = std::less<K>> using SSet = std::set<K, C, ScratchAlloc<K>>;
template<class K, class H = std::hash<K>, class E = std::equal_to<K>>
    using SUSet = std::unordered_set<K, H, E, ScratchAlloc<K>>;
template<class K, class V, class C = std::less<K>>
    using SMap = std::map<K, V, C, ScratchAlloc<std::pair<const K, V>>>;
template<class K, class V, class H = std::hash<K>, class E = std::equal_to<K>>
    using SUMap = std::unordered_map<K, V, H, E, ScratchAlloc<std::pair<const K, V>>>;
template<class K, class V, class H = std::hash<K>, class E = std::equal_to<K>>
    using PUMap = std::unordered_map<K, V, H, E, PersistAlloc<std::pair<const K, V>>>;

}  // namespace hypergraph
