#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

// Content-ordered hash: the edge tuples as they stand, in edge order. This is what
// CanonicalizationMode::Automatic identifies states by -- NOT an isomorphism invariant, which is
// the point of the mode. The exact identity is the shared individualization-refinement body
// (hgcommon/ir_core.hpp), which host and device both run; this is the cheap key the Automatic
// mode asks for, and the rule it applies is hgcommon::ContentHasher, so the two devices agree on
// that identity by construction rather than by comparison.
__device__ uint64_t content_hash_state_device(const DeviceState& ds, StateId sid);

// A state's edges in content order: its slice in order, skipping an id past the edge pool's
// published size (hgcommon::content_equal's cursor). The content hash iterates through it too.
struct DeviceContentCursor {
    const DeviceState& ds;
    StateEdgeSlice sl;
    uint32_t live;
    uint32_t k = 0;

    __device__ DeviceContentCursor(const DeviceState& d, StateId sid)
        : ds(d), sl(sid < d.max_states ? d.state_edge_slices[sid] : StateEdgeSlice{}),
          live(d.edge_pool.size()) {}

    __device__ bool next(uint32_t& arity, const uint32_t*& vertices) {
        while (k < sl.count) {
            const EdgeId eid = ds.state_edge_ids[sl.offset + k++];
            if (eid >= live || eid >= ds.edge_pool.capacity) continue;
            const Edge& e = ds.edge_pool.at(eid);
            arity = e.arity;
            vertices = &ds.vertex_pool.at(e.vertex_offset);
            return true;
        }
        return false;
    }
};

}  // namespace gpu
}  // namespace HG_NAMESPACE
