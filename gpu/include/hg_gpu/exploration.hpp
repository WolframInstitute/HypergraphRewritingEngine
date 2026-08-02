#pragma once
// Which newly created states get expanded.
//
// Asked in one place. A second copy of this predicate would not crash; it would silently
// explore a different state set, which is the defect class the shared-core work exists to
// close.

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
__device__ bool state_survives_dedup(StateId sid, uint64_t hash,
                                     DedupMap::DeviceView map, bool dedup,
                                     uint32_t explore_threshold_u32,
                                     uint64_t explore_seed, uint32_t step);

}  // namespace hg_gpu
