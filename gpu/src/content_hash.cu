#include "hgcommon/namespace.hpp"
#include "hg_gpu/content_hash.hpp"

#include "hgcommon/content_core.hpp"

#include <cuda_runtime.h>

namespace HG_NAMESPACE {
namespace gpu {

__device__ uint64_t content_hash_state_device(const DeviceState& ds, StateId sid) {
    if (sid >= ds.max_states) return 0;
    // The rule is hgcommon::ContentHasher; the iteration is DeviceContentCursor, which
    // state_claim_content compares through.
    uint32_t ne = 0, arity = 0;
    const uint32_t* v = nullptr;
    for (DeviceContentCursor c(ds, sid); c.next(arity, v);) ++ne;
    hgcommon::ContentHasher ch(ne);
    for (DeviceContentCursor c(ds, sid); c.next(arity, v);) {
        ch.edge_begin(arity);
        for (uint32_t i = 0; i < arity; ++i) ch.vertex(static_cast<uint64_t>(v[i]));
        ch.edge_end();
    }
    return ch.value();
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
