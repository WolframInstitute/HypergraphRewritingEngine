#pragma once

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>
#include <vector>

namespace hg_gpu {

// Device-inline WL hash of a single state. Safe to call from any kernel that
// has the DeviceState. See gpu/src/wl_hash.cu for the algorithm comments.
__device__ uint64_t wl_hash_state_device(DeviceState ds, StateId sid);

// Compute 1-WL canonical hashes for every state in [0, num_states) on the
// device. Writes hashes into `out_hashes` (device pointer, caller-allocated,
// size num_states * sizeof(uint64_t)).
//
// Algorithm (single thread per state for the first cut):
//   1. Extract the state's edges from its bitset; collect unique vertices
//      (linear-probe dedup, bounded by kMaxWlVertices).
//   2. Initial vertex color = FNV of (degree, per-occurrence (arity, position)).
//   3. Refine: new_color = FNV(old_color, sorted multiset {FNV(old[u], pos(u))
//      for each neighbour u across each edge containing v}). Repeat until
//      stable or kMaxWlRefineIters.
//   4. State hash = FNV over sorted multiset of final colors.
//
// 1-WL is isomorphism-invariant and suffices for the common case. Known
// false positives on highly symmetric graphs (Cai-Fürer-Immerman etc.);
// full GPU IR canonicalization (exact, follow-up task) would eliminate
// those, but 1-WL is a correct, well-defined first cut that lets the
// dedup pipeline operate.
void compute_state_wl_hashes(const EngineState& engine,
                             uint32_t  num_states,
                             uint64_t* out_hashes_device);

// Convenience: compute + pull one state's hash back to host. For tests.
uint64_t compute_state_wl_hash_host(const EngineState& engine, StateId sid);

// Hash a range [lo, hi) of state IDs, writing out_hashes[i-lo] = hash of
// state i. Faster than compute_state_wl_hashes for evolving workloads that
// only need new states hashed.
void compute_state_wl_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device);


// Content-ordered (non-isomorphic) hash per state, for
// CanonicalizationMode::Automatic — groups states by literal edge content.
void compute_state_content_hashes_range(const EngineState& engine,
                                        uint32_t lo, uint32_t hi,
                                        uint64_t* out_hashes_device);

}  // namespace hg_gpu
