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
// how a state is flattened out of the CSR, where the core's scratch comes from, and the
// launch shape.
//
// One thread per state, grid-stride: the parallelism in this computation is ACROSS states.
// Each thread owns one slot of a device scratch pool, holding the flattened state and the
// core's working buffers.
//
// The range entry point sizes its slot to the largest state in the range, so size alone does
// not force a fallback. A state whose individualization search wants more depth than the pool
// is sized for, or a range whose single slot exceeds the whole pool budget, falls back to the
// 1-WL hash, and that hash then serves as the state's DEDUP KEY.
//
// That is a correctness exposure, not a tuning knob. Isomorphism-invariance is one
// directional: WL never separates two isomorphic states, but it does MERGE non-isomorphic
// ones. tools/ir_vs_wl demonstrates it constructively on the prism against K3,3 -- six
// vertices -- and on the rook's 4x4 graph against Shrikhande. Nothing bounds how often an
// evolution reaches such a state, so no measured collision rate over some other corpus
// licenses assuming it is rare.
//
// last_ir_degraded_states() reports how many states took the fallback. A non-zero count means
// the state set may contain wrongly merged states.

uint64_t compute_state_ir_hash_host(const EngineState& engine, StateId sid);

// States in the most recent range that fell back to the coarser 1-WL hash.
uint32_t last_ir_degraded_states();

void compute_state_ir_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device);

// Exact hash of ONE state, for callers with no batch to measure.
//
// The range entry point above sizes its slot on the host from the largest state in the range.
// A device-resident loop has no range: states arrive continuously and the largest is not
// knowable before the launch. This sizes the slot from THIS state's own edge and occurrence
// counts and claims it from a device arena, so the exact path has no per-state ceiling and
// therefore no 1-WL fallback -- the fallback's merge hazard is the reason the ceiling had to
// go.
//
// `slot`/`slot_words` are the caller's scratch, carried across items: a worker reuses its slot
// and claims again only when the next state needs a larger one. Initialise them to
// {nullptr, 0}.
//
// Returns false when the arena is exhausted or the search wants more depth than the device
// attempts. Both are capacity overflows -- record the warning and return partial work, never a
// coarser hash, and never a host round trip to grow.
// Why an exact hash could not be produced. Carried rather than collapsed to a bool because the
// three causes call for three different responses, and treating them alike made a recoverable
// capacity failure indistinguishable from a fixed kernel limit:
//
//   kArenaExhausted   the arena had no slot of the size this state needs. The arena is sized
//                     from the config, so growing the config is a real remedy -- the host's
//                     grow-and-retry treats it as retryable.
//   kDepthExceeded    the individualization search wanted to go deeper than the device
//                     attempts. The depth is a constant the slot is shaped for, so growing
//                     cannot help.
//   kMalformedState   the flattening did not fit a shape sized from this state's own counts,
//                     which cannot happen; reported rather than silently hashing something else.
enum class ExactHashStatus : uint8_t {
    kOk = 0,
    kArenaExhausted,
    kDepthExceeded,
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
        default:                               return ErrorKind::kScratchOverflow;
    }
}

}  // namespace hg_gpu
