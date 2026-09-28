#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/edge_signature.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// Bulk (re)build of the signature and vertex-inverted indices from the edge
// pool; used when lazy index maintenance turns on mid-run.
void rebuild_indices(EngineState& engine, uint32_t num_edges);

// Upload M initial states in one shot: their edges are concatenated into the
// edge pool (each state's edges are a contiguous ascending ID run, so its CSR
// slice stays sorted), state_count is set to M, and the indices are built over
// all initial edges. Returns M. Isomorphic initial states are separate states
// here; canonical dedup at seed time merges them under explore-from-canonical.
// Vertex ids are taken at face value; vertex_high_water is set to the largest
// plus one.
uint32_t upload_initial_states(EngineState& engine,
                               const std::vector<std::vector<std::vector<VertexId>>>& initial_states);

// upload_initial_states with one state, which becomes state 0. Returns 0.
StateId upload_initial_state(EngineState&                          engine,
                             const std::vector<std::vector<VertexId>>& initial_edges);

}  // namespace gpu
}  // namespace HG_NAMESPACE