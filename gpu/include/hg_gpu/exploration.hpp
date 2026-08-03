#pragma once
// Which newly created states get expanded.
//
// Asked in one place. A second copy of this predicate would not crash; it would silently
// explore a different state set, which is the defect class the shared-core work exists to
// close.

#include "hgcommon/core.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/hash_table.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>

namespace hg_gpu {

// Canonical hash -> the first state id seen with that hash. First writer wins, and its id is
// the one that gets expanded.
using DedupMap = ConcurrentMap<uint64_t, uint32_t>;

// True when `sid` should be expanded.
//
// `dedup` selects the exploration semantics. True: only the first state of each canonical hash
// enters the frontier (explore_from_canonical_states_only). False: every new state is explored,
// so `map` is not consulted at all -- deduplicating against it anyway would silently drop
// states whose hashes collided.
//
// `explore_threshold_u32` is the stochastic-pruning coin, encoded as a probability scaled to
// uint32: 0xFFFFFFFF means always explore and skips the draw entirely.
//
// Defined here rather than in a .cu so every translation unit that asks the question links to
// THIS body -- a device function defined in one .cu is not reachable from another target's
// device link, and the answer to "does a second copy appear" must not depend on that.
__device__ inline bool state_survives_dedup(DeviceState ds, StateId sid, uint64_t hash,
                                            DedupMap::DeviceView map, bool dedup,
                                            uint32_t explore_threshold_u32,
                                            uint64_t explore_seed, uint32_t step) {
    if (dedup) {
        if (hash == 0) {
            // Not a hash: keep the state rather than merge every uncomputed one into one slot.
            ds.errors.record(ErrorKind::kUncomputedStateHash);
            return true;
        }
        auto r = map.insert_if_absent(hash, sid);
        if (!r.inserted) return false;
    }

    // Stochastic-exploration coin flip. UINT32_MAX == "always explore"
    // (the threshold encoding for probability 1.0); skip the hash work
    // entirely on that fast path so the existing all-deterministic
    // workloads pay zero overhead.
    if (explore_threshold_u32 != 0xFFFFFFFFu) {
        if (explore_threshold_u32 == 0u) return false;  // probability 0.0
        uint64_t mix = hgcommon::splitmix64(explore_seed
                                  ^ (static_cast<uint64_t>(step) << 32)
                                  ^ static_cast<uint64_t>(sid));
        uint32_t draw = static_cast<uint32_t>(mix);
        if (draw >= explore_threshold_u32) return false;
    }
    return true;
}

}  // namespace hg_gpu
