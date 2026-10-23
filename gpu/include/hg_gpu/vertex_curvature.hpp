#pragma once
#include "hgcommon/namespace.hpp"
// "IncludeVertexCurvature" ON THE DEVICE: each state's per-vertex Ollivier-Ricci curvature from
// hgcommon::sg_state_geometry, the body the host's stats::vertex_curvature runs, one warp per
// state. The states arrive as edge lists in compressed rows; each state's values come back in
// its vertex order (ascending id), NaN for a vertex with no neighbour, as the host computes them.

#include <cstdint>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// The states' edges: state s holds edges [state_edges[s], state_edges[s + 1]), edge e holds
// vertices [edge_verts[e], edge_verts[e + 1]) of `verts`. `vertex_count[s]` is the number of
// distinct vertices of state s; its values go to k[k_off[s] .. k_off[s] + vertex_count[s]),
// with k_off the prefix sum of vertex_count. `done[s]` is set for each state computed; a state
// whose scratch does not fit the device budget is left for the caller. False, with nothing
// computed, when no device is usable. `graph` is the rule that builds each state's graph
// (hgcommon::SG_GRAPH_*).
bool vertex_curvature_device(const std::vector<uint32_t>& state_edges,
                             const std::vector<uint32_t>& edge_verts,
                             const std::vector<uint32_t>& verts,
                             const std::vector<uint32_t>& vertex_count,
                             std::vector<double>& k, std::vector<uint64_t>& k_off,
                             std::vector<uint8_t>& done, uint32_t graph = 0);

}  // namespace gpu
}  // namespace HG_NAMESPACE
