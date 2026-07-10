#pragma once

#include "hg_gpu/edge_signature.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>
#include <vector>

namespace hg_gpu {

// Upload an initial-state edge list to the engine and create state 0 from
// those edges. Returns the new StateId.
//
// Host-side: builds vertex/edge buffers in pinned host memory (or just plain
// host memory + cudaMemcpy), uploads in one shot, then launches a kernel
// that populates the signature and vertex-inverted indices for the new
// edges.
//
// All edges land in edge_pool[0..n) and vertex tuples in vertex_pool
// starting at offset 0 (cumulative). Vertex IDs are taken at face value;
// vertex_high_water is set to (max(VertexId in input) + 1).
// Bulk (re)build of the signature and vertex-inverted indices from the edge
// pool; used when lazy index maintenance turns on mid-run.
void rebuild_indices(EngineState& engine, uint32_t num_edges);

StateId upload_initial_state(EngineState&                          engine,
                             const std::vector<std::vector<VertexId>>& initial_edges);

// Upload M initial states in one shot: their edges are concatenated into the
// edge pool (each state's edges are a contiguous ascending ID run, so its CSR
// slice stays sorted), state_count is set to M, and the indices are built over
// all initial edges. Returns M. Isomorphic initial states are separate states
// here; canonical dedup at seed time merges them under explore-from-canonical.
uint32_t upload_initial_states(EngineState& engine,
                               const std::vector<std::vector<std::vector<VertexId>>>& initial_states);

}  // namespace hg_gpu
