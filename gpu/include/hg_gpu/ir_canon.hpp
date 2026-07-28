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
// the 1-WL hash. That hash is isomorphism-invariant, so deduplication stays correct; it is
// coarser than the exact hash, so those states are COUNTED -- last_ir_degraded_states()
// reports how many, which is what makes the degradation visible instead of silent.

uint64_t compute_state_ir_hash_host(const EngineState& engine, StateId sid);

// States in the most recent range that fell back to the coarser 1-WL hash.
uint32_t last_ir_degraded_states();

void compute_state_ir_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device);

}  // namespace hg_gpu
