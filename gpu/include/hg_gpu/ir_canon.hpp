#pragma once
#include <cstdint>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

namespace hg_gpu {

// Exact individualization-refinement canonicalization of a state's edge list.
//
// The algorithm itself lives in hgcommon/ir_core.hpp and is the SAME code the host engine
// runs, so the two devices produce identical canonical hashes by construction rather than by
// two implementations being kept in step. What is device-specific is only the orchestration:
// how a state is flattened out of the CSR, where the core's scratch comes from, and the
// launch shape.
//
// One thread per state, grid-stride: the parallelism in this computation is ACROSS states.
// Each thread owns one slot of a device scratch pool, holding the flattened state and the
// core's working buffers.
//
// A state that exceeds the slot's bounds (kMaxIRVerts / kMaxIREdges / kMaxIROccs in the .cu)
// or whose individualization search wants more depth than the pool is sized for falls back to
// the 1-WL hash, and that hash then serves as the state's DEDUP KEY.
//
// That is a correctness exposure, not a tuning knob. Isomorphism-invariance is one
// directional: WL never separates two isomorphic states, but it does MERGE non-isomorphic
// ones. tools/ir_vs_wl demonstrates it constructively on the prism against K3,3 -- six
// vertices -- and on the rook's 4x4 graph against Shrikhande. Nothing bounds how often an
// evolution reaches such a state, so no measured collision rate over some other corpus
// licenses assuming it is rare.
//
// last_ir_degraded_states() reports how many states took the fallback. A non-zero count means
// the state set may contain wrongly merged states.

uint64_t compute_state_ir_hash_host(const EngineState& engine, StateId sid);

// States in the most recent range that fell back to the coarser 1-WL hash.
uint32_t last_ir_degraded_states();

void compute_state_ir_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device);

}  // namespace hg_gpu
