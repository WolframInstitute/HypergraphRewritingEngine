#include "hgcommon/namespace.hpp"
#include "hg_gpu/content_hash.hpp"

#include "hgcommon/content_core.hpp"

#include <cuda_runtime.h>

namespace HG_NAMESPACE {
namespace gpu {

__device__ uint64_t content_hash_state_device(const DeviceState& ds, StateId sid) {
    if (sid >= ds.max_states) return 0;
    // The rule is hgcommon::content_hash_of; the iteration is DeviceContentCursor, which
    // state_claim_content compares through.
    return hgcommon::content_hash_of(
        hgcommon::undirected_content_cursor(DeviceContentCursor(ds, sid), ds.undirected != 0u));
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
