#pragma once
#include "hgcommon/namespace.hpp"
#include <cstdint>

#include "hgcommon/ir_core.hpp"
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/types.hpp"

namespace HG_NAMESPACE {
namespace gpu {

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
//
// `out_form`, when non-null, receives a pointer into the slot where the core wrote the state's IR
// canonical form, and `out_form_words` its length (hgcommon::ir_canonical_form_words); both are
// null and 0 for the empty state. The form stays valid until the slot is reused.
template <class Par = hgcommon::IrSerial>
__device__ ExactHashStatus state_exact_hash_device(DeviceState ds, StateId sid,
                                                   DeviceArena::View arena,
                                                   uint32_t*& slot, uint64_t& slot_words,
                                                   uint64_t& out_hash, bool want_ranks = false,
                                                   bool want_orbits = false,
                                                   uint32_t** out_form = nullptr,
                                                   uint32_t* out_form_words = nullptr,
                                                   Par par = Par{});


// Slot geometry for one thread: every field a flattened state needs, then the shared core's
// scratch behind it. Sized from the states that will use it, never from a constant, because a
// state past a fixed bound would have to be keyed by the 1-WL hash and that MERGES
// non-isomorphic states.
//
// The slot lives in global memory: shared memory cannot hold the search's per-level
// partitions, and the core wants one contiguous span.
struct IrSlotShape {
    uint32_t cap_verts = 0;
    uint32_t cap_edges = 0;
    uint32_t cap_occs  = 0;
    // Generator rows the scratch is sized for. Must match the budget handed to the core, or
    // the search would write past what this slot reserved.
    uint32_t generators = hgcommon::IR_DEVICE_GENERATORS;
    uint32_t depth     = 0;

    HG_HD uint32_t ea_words()   const { return (cap_edges + 3) / 4; }
    HG_HD uint32_t eoff_words() const { return cap_edges + 1; }
    // Three cap_edges spans sit between the flattened state and the core's scratch: the ranks
    // the core reports, the CSR slot each flattened edge came from, and the per-edge orbits.
    // The slot map exists because flattening skips edges the slice still holds, so flat index
    // j and slot k diverge and neither ranks nor orbits can be scattered back without it. All
    // three are dwarfed by ir_scratch_words.
    HG_HD uint32_t rank_words() const { return 3u * cap_edges; }
    HG_HD uint64_t scratch_words() const {
        return hgcommon::ir_scratch_words(cap_verts, cap_edges, cap_occs, depth, generators);
    }
    // The canonical form the core writes when asked (one arity word and the labelled vertices
    // per edge), after the scratch: a Full-mode dedup compares it on a key hit.
    HG_HD uint32_t form_words() const { return cap_edges + cap_occs; }
    HG_HD uint64_t words() const {
        return ea_words() + eoff_words() + cap_occs + cap_verts + rank_words()
             + scratch_words() + form_words() + 8;
    }
    // Even, so every slot base keeps the 8-byte alignment the pool starts with.
    HG_HD uint64_t stride() const { return (words() + 1ull) & ~1ull; }
};

// The slot of a state of `edges` edges and `occs` vertex occurrences, searched to `depth` with
// `generators` generator rows.
HG_HD inline IrSlotShape ir_slot_shape(uint32_t edges, uint32_t occs, uint32_t depth,
                                       uint32_t generators) {
    IrSlotShape s;
    s.cap_edges = edges + 1;
    s.cap_occs = occs + 1;
    s.cap_verts = occs + 1;   // every occurrence could be a distinct vertex
    s.depth = depth;
    s.generators = generators;
    return s;
}

// A TILE OF W LANES RUNS THE SEARCH TOGETHER (W a power of two, at most 32). The tile's lanes
// enter the canonicalization with identical arguments and execute identical control flow --
// every branch reads state each lane sees the same, so convergence is by construction. Shared
// writes are the tile leader's (its lowest lane), behind a sync of the tile's lanes; the
// order-safe loops fan lane-strided over the tile; a leader-computed scalar that control
// depends on crosses by shuffle within the tile. Storage the leader writes and the other lanes
// read (the IR scratch slot) is the tile's own. The serial policy in ir_core.hpp compiles the
// same source to the plain single-thread search.
template <uint32_t W>
struct IrTile {
    static_assert(W >= 1u && W <= 32u && (W & (W - 1u)) == 0u, "tile width is a power of two");
    static constexpr bool kFans = true;
    static constexpr uint32_t kWidth = W;
    __device__ static uint32_t rank() { return threadIdx.x & (W - 1u); }
    __device__ static uint32_t mask() {
        return W == 32u ? 0xffffffffu
                        : ((1u << (W & 31u)) - 1u) << ((threadIdx.x & 31u) & ~(W - 1u));
    }
    __device__ bool leader() const { return rank() == 0u; }
    __device__ void sync() const { __syncwarp(mask()); }
    template <class F>
    __device__ void fan(uint32_t n, F&& f) const {
        __syncwarp(mask());
        for (uint32_t i = rank(); i < n; i += W) f(i);
        __syncwarp(mask());
    }
    __device__ uint32_t bcast(uint32_t v) const { return __shfl_sync(mask(), v, 0, W); }
    __device__ uint64_t bcast64(uint64_t v) const {
        return __shfl_sync(mask(), static_cast<unsigned long long>(v), 0, W);
    }
    __device__ uint32_t fetch_add(uint32_t* p, uint32_t v) const { return atomicAdd(p, v); }
};

// The whole warp: the persistent kernel's block is one warp.
using IrWarpAll = IrTile<32>;

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

}  // namespace gpu
}  // namespace HG_NAMESPACE