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
__device__ uint64_t content_hash_state_device(DeviceState ds, StateId sid);

}  // namespace gpu
}  // namespace HG_NAMESPACE
