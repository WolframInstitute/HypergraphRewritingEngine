#pragma once
#include <cstdint>

#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

namespace hg_gpu {

// Exact individualization-refinement canonicalization of a state's edge list.
//
// The algorithm itself lives in hgcommon/ir_core.hpp and is the SAME code the host engine
// runs, so the two devices produce identical canonical hashes by construction rather than by
// two implementations being kept in step. What is device-specific is only the orchestration:
// how a state is flattened out of the CSR and where the core's scratch comes from.
//
// ONE state per call, sized from that state's own counts and taken from a device arena. There
// is no per-state ceiling and so no fallback: a state the exact path cannot key is REPORTED,
// never keyed by something coarser. That is not a tuning stance. Isomorphism-invariance is one
// directional -- 1-WL never separates two isomorphic states, but it does MERGE non-isomorphic
// ones, which tools/ir_vs_wl demonstrates constructively on the prism against K3,3 (six
// vertices) and on the rook's 4x4 graph against Shrikhande. Nothing bounds how often an
// evolution reaches such a state, so no measured collision rate over some other corpus
// licenses assuming it is rare.

// Exact hash of ONE state.
//
// The slot is sized from THIS state's own edge and occurrence counts and claimed from a device
// arena, so the path has no per-state ceiling and states arriving continuously in a
// device-resident loop need no host-side batch measurement.
//
// `slot`/`slot_words` are the caller's scratch, carried across items: a worker reuses its slot
// and claims again only when the next state needs a larger one. Initialise them to
// {nullptr, 0}.
//
// Returns false when the arena is exhausted or the search wants more depth than the device
// attempts. Both are capacity overflows -- record the warning and let the wrapper grow and
// retry, never a coarser hash, and never a host round trip mid-run.
// Why an exact hash could not be produced. Carried rather than collapsed to a bool because the
// three causes call for three different responses, and treating them alike made a recoverable
// capacity failure indistinguishable from a fixed kernel limit:
//
//   kArenaExhausted   the arena had no slot of the size this state needs. The arena is sized
//                     from the config, so growing the config is a real remedy -- the host's
//                     grow-and-retry treats it as retryable.
//   kDepthExceeded    the individualization search wanted to go deeper than the device
//                     attempts. The depth is EngineConfig::ir_depth, so growing the config is a
//                     real remedy -- the host's grow-and-retry doubles it.
//   kMalformedState   the flattening did not fit a shape sized from this state's own counts,
//                     which cannot happen; reported rather than silently hashing something else.
enum class ExactHashStatus : uint8_t {
    kOk = 0,
    kArenaExhausted,
    kDepthExceeded,
    kGeneratorsExceeded,
    kMalformedState,
};

// `want_orbits` additionally scatters each edge's automorphism ORBIT into
// ds.state_edge_orbit (parallel to the CSR slice, UINT32_MAX where the flattening skipped a
// slot) and writes the state's orbit count into ds.state_num_orbits -- the quotient-causal
// DP's keys. Rides the same IR pass as the hash and ranks.
__device__ ExactHashStatus state_exact_hash_device(DeviceState ds, StateId sid,
                                                   DeviceArena::View arena,
                                                   uint32_t*& slot, uint64_t& slot_words,
                                                   uint64_t& out_hash, bool want_ranks = false,
                                                   bool want_orbits = false);

// The ErrorKind a failed exact hash should be recorded as. One place, so a new call site cannot
// pick a different mapping and re-conflate what this separation exists to keep apart.
HG_HD inline ErrorKind error_kind_for(ExactHashStatus s) {
    switch (s) {
        case ExactHashStatus::kArenaExhausted: return ErrorKind::kIRArenaExhausted;
        case ExactHashStatus::kDepthExceeded:  return ErrorKind::kIRDepthExceeded;
        case ExactHashStatus::kGeneratorsExceeded: return ErrorKind::kIRGeneratorsExceeded;
        default:                               return ErrorKind::kScratchOverflow;
    }
}

// Every state in [lo, hi) keyed by state_exact_hash_device, one thread per state, grid-stride.
// A launch shape over the same body the device-resident loop calls, not a second rule: a state
// the exact path cannot key leaves 0 in `out_hashes_device` and records its capacity kind.
void compute_state_ir_hashes_range(EngineState& engine, uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device);

// One state through the same launcher, for callers with a single state to key.
uint64_t compute_state_ir_hash_host(EngineState& engine, StateId sid);

}  // namespace hg_gpu
