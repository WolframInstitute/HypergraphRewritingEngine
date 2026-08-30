#pragma once
#include "hgcommon/namespace.hpp"

#include "arena.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace engine {

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
ConcurrentHeterogeneousArena*& worker_persistent_target();
ConcurrentHeterogeneousArena& worker_persistent();

// RAII: redirect worker_persistent() to `arena` for this scope (single-threaded per
// worker). Everything allocated via PersistAlloc while in scope lands in `arena`.
struct PersistTarget {
    ConcurrentHeterogeneousArena* prev_;
    explicit PersistTarget(ConcurrentHeterogeneousArena& arena);
    ~PersistTarget();
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

// A visited-set for a graph walk, on the scratch arena and without a node per element.
//
// std::unordered_set allocates one node PER INSERT and chains them, so a breadth-first walk pays
// an allocation and a pointer chase for every vertex it touches. Both of this engine's
// reachability walks -- the quotient reduction's qc_reachable and the causal graph's own -- used
// one, and on the reconstruction workload the hash table was 4% of all instructions with the
// allocator it drives accounting for more. Open addressing over a flat span costs one bump for
// the whole walk and one probe per visit.
//
// Keys are dense 32-bit ids and UINT32_MAX marks a free slot -- but that value is ALSO a live key
// here: INVALID_ID is the sentinel producer the reconstruction seeds a root's edges with, and it
// reaches these walks. A set that treated it as an empty slot would report it absent every time,
// so it is tracked in a flag of its own rather than in the table.
//
// With a value type it is the same table carrying one V per key (ScratchIdMap): the vertex
// relabelling in front of the canonical search was a std::unordered_map, one heap node per
// vertex per state -- callgrind on wpp depth 7 put its hashtable at 0.93% of all instructions
// and the malloc/free it drove at 2.7% more. One probing rule for both, so the set and the map
// cannot drift.
template <class V = void>
class ScratchIdTable {
public:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;
    static constexpr bool kHasValue = !std::is_void_v<V>;
    using Value = std::conditional_t<kHasValue, V, uint32_t>;

    explicit ScratchIdTable(uint32_t hint = 64) { rehash(round_up_pow2(hint < 8 ? 8 : hint)); }

    // True iff the key was not already present.
    bool insert(uint32_t key) {
        Value v{};
        return find_or_insert(key, v, v);
    }

    // Map use: `out` receives the value stored for key -- `fresh` when this call added it.
    // True iff the key was not already present.
    bool find_or_insert(uint32_t key, Value fresh, Value& out) {
        if (key == kEmpty) {
            const bool added = !has_empty_;
            if (added) { has_empty_ = true; empty_value_ = fresh; }
            out = empty_value_;
            return added;
        }
        if (count_ * 4 >= cap_ * 3) rehash(cap_ * 2);   // keep the load factor under 3/4
        return insert_into(slots_, vals_, cap_, key, fresh, out);
    }

private:
    static uint32_t round_up_pow2(uint32_t v) {
        uint32_t p = 8; while (p < v) p <<= 1; return p;
    }
    static uint32_t mix(uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return x;
    }
    bool insert_into(uint32_t* slots, Value* vals, uint32_t cap, uint32_t key,
                     Value fresh, Value& out) {
        uint32_t i = mix(key) & (cap - 1);
        for (;;) {
            const uint32_t cur = slots[i];
            if (cur == key) { if constexpr (kHasValue) out = vals[i]; return false; }
            if (cur == kEmpty) {
                slots[i] = key;
                if constexpr (kHasValue) { vals[i] = fresh; out = fresh; }
                ++count_;
                return true;
            }
            i = (i + 1) & (cap - 1);
        }
    }
    void rehash(uint32_t cap) {
        uint32_t* fresh = static_cast<uint32_t*>(
            worker_scratch().allocate_raw(sizeof(uint32_t) * cap, alignof(uint32_t)));
        for (uint32_t i = 0; i < cap; ++i) fresh[i] = kEmpty;
        Value* fresh_vals = nullptr;
        if constexpr (kHasValue)
            fresh_vals = static_cast<Value*>(
                worker_scratch().allocate_raw(sizeof(Value) * cap, alignof(Value)));
        const uint32_t old_cap = cap_;
        uint32_t* old = slots_;
        Value* old_vals = vals_;
        slots_ = fresh; vals_ = fresh_vals; cap_ = cap; count_ = 0;
        for (uint32_t i = 0; i < old_cap; ++i)
            if (old[i] != kEmpty) {
                Value v{};
                insert_into(slots_, vals_, cap_, old[i], kHasValue ? old_vals[i] : v, v);
            }
    }

    uint32_t* slots_ = nullptr;
    Value*    vals_ = nullptr;
    uint32_t  cap_ = 0;
    uint32_t  count_ = 0;
    bool      has_empty_ = false;
    Value     empty_value_{};
};
using ScratchIdSet = ScratchIdTable<void>;
using ScratchIdMap = ScratchIdTable<uint32_t>;

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

}  // namespace engine
}  // namespace HG_NAMESPACE