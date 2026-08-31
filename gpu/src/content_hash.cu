#include "hgcommon/namespace.hpp"
#include "hg_gpu/content_hash.hpp"

#include "hgcommon/content_core.hpp"

#include <cuda_runtime.h>

namespace HG_NAMESPACE {
namespace gpu {

__device__ uint64_t content_hash_state_device(DeviceState ds, StateId sid) {
    if (sid >= ds.max_states) return 0;
    StateEdgeSlice sl = ds.state_edge_slices[sid];
    uint32_t live = ds.edge_pool.size();
    uint32_t ne = 0;
    for (uint32_t k = 0; k < sl.count; ++k) {
        EdgeId eid = ds.state_edge_ids[sl.offset + k];
        if (eid < live && eid < ds.edge_pool.capacity) ++ne;
    }
    // The rule is hgcommon::ContentHasher; only the ITERATION is ours -- the host walks a
    // SparseBitset, this walks the slice and filters dead edges. Sharing the loop is impossible;
    // sharing the constants and the mixing is mandatory.
    hgcommon::ContentHasher ch(ne);
    for (uint32_t k = 0; k < sl.count; ++k) {
        EdgeId eid = ds.state_edge_ids[sl.offset + k];
        if (eid >= live || eid >= ds.edge_pool.capacity) continue;
        const Edge& e = ds.edge_pool.at(eid);
        ch.edge_begin(e.arity);
        for (uint8_t i = 0; i < e.arity; ++i)
            ch.vertex(static_cast<uint64_t>(ds.vertex_pool.at(e.vertex_offset + i)));
        ch.edge_end();
    }
    return ch.value();
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
