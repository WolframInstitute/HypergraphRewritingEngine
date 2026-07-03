#pragma once

#include "wl_hash.hpp"
#include "arena.hpp"
#include "scratch_alloc.hpp"
#include "types.hpp"

#include <memory>
#include <vector>

namespace hypergraph {

// Per-worker bounded LRU cache of parent WL histories (the A4 reclaim mechanism for
// the incremental WL's persistent data). Each cached history lives in its OWN
// resettable arena; evicting a history resets its arena, reclaiming all its memory.
// Per worker => thread-local, no allocator contention, no cross-worker arena access.
//
// Usage (B3):
//   auto& hist = cache.get_or_build(wl, parent, parent_edges, ev, ea);
//   PersistTarget redirect(cache.arena_of(parent));   // ext_parent -> the slot arena
//   uint64_t h = wl.compute_state_hash_incremental(child_edges, ev, ea, hist, ...);
class WLHistoryCache {
public:
    explicit WLHistoryCache(size_t capacity = 8) : capacity_(capacity ? capacity : 1) {}

    // The history for parent `state` in evolution `epoch`, building it into a pooled
    // arena if absent (evicting the LRU slot, which resets+reuses its arena). The
    // epoch guards against StateId reuse across evolutions on the same worker thread.
    template<typename VA, typename AA>
    WLHash::WLHistory& get_or_build(uint64_t epoch, const WLHash& wl, StateId state,
                                    const SparseBitset& edges, const VA& ev, const AA& ea) {
        for (auto& s : slots_) if (s.epoch == epoch && s.state == state && s.hist.valid) { s.lru = ++tick_; return s.hist; }
        Slot& slot = pick_slot();
        if (!slot.arena) slot.arena = std::make_unique<ConcurrentHeterogeneousArena>(1u << 20, /*recycle=*/true);
        slot.arena->reset();                 // reclaim the prior occupant's memory
        slot.hist = WLHash::WLHistory{};
        { PersistTarget redirect(*slot.arena);
          slot.hist = wl.compute_state_hash_and_history(edges, ev, ea).second; }
        slot.state = state; slot.epoch = epoch; slot.lru = ++tick_;
        return slot.hist;
    }

    // The arena backing (epoch,state)'s cached history (for redirecting ext_parent),
    // or the default persistent arena if not cached.
    ConcurrentHeterogeneousArena& arena_of(uint64_t epoch, StateId state) {
        for (auto& s : slots_) if (s.epoch == epoch && s.state == state && s.hist.valid) return *s.arena;
        return worker_persistent();
    }

private:
    struct Slot {
        StateId state = INVALID_ID;
        uint64_t epoch = 0;
        std::unique_ptr<ConcurrentHeterogeneousArena> arena;
        WLHash::WLHistory hist;
        uint64_t lru = 0;
    };
    std::vector<Slot> slots_;   // pool metadata (per-worker, bounded to capacity_)
    size_t capacity_;
    uint64_t tick_ = 0;

    Slot& pick_slot() {
        for (auto& s : slots_) if (s.state == INVALID_ID || !s.hist.valid) return s;
        if (slots_.size() < capacity_) { slots_.emplace_back(); return slots_.back(); }
        Slot* lru = &slots_[0];
        for (auto& s : slots_) if (s.lru < lru->lru) lru = &s;
        return *lru;
    }
};

// Per-worker (thread-local) history cache used by the incremental WL path.
inline WLHistoryCache& worker_history_cache() {
    static thread_local WLHistoryCache cache(8);
    return cache;
}

}  // namespace hypergraph
