#pragma once
#include <cstdint>

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

namespace hg_gpu {

// McKay-style individualization-refinement (IR) canonicalization of a state's
// edge list, on GPU. Mirrors hypergraph/src/ir_canonicalization.cpp semantics:
//
//   (1) Collect unique vertices from the state's edges. Build per-vertex
//       occurrence list (edge_idx, position, arity).
//   (2) Initial colour = hash over sorted (arity, position) signature across
//       the vertex's occurrences. Vertices with the same occurrence pattern
//       start in the same cell.
//   (3) 1-WL refinement: iterate until fixpoint — each vertex's colour is
//       hashed from its current colour + sorted list of per-occurrence
//       (arity, position, sorted colours of co-occurring vertices).
//   (4) If the resulting partition is discrete (every colour unique),
//       extract a labelling (colour → rank 0..N-1), apply it to the edges,
//       sort edges lexicographically, and emit an FNV-1a hash over the
//       sorted canonical edge list. This is the canonical hash.
//   (5) If non-discrete, fall back to an individualize-refine backtrack
//       search (to be added in a follow-up). For now the kernel emits the
//       sorted-colour-multiset hash and records kScratchOverflow so the
//       host can detect that a workload needed the backtrack path.
//
// One block per state (grid.x = hi - lo). Block size = 32 threads (one warp)
// — threads cooperate on vertex-parallel refinement. Per-block shared-memory
// scratch bounds the state size (see kMaxIRVerts / kMaxIREdges / kMaxIROccs
// in the .cu); the kernel records kScratchOverflow when exceeded.

uint64_t compute_state_ir_hash_host(const EngineState& engine, StateId sid);

void compute_state_ir_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device);

}  // namespace hg_gpu
