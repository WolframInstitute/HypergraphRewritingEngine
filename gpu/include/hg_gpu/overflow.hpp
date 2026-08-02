#pragma once

// Capacity-overflow types shared between the device-side error channel
// (errors.hpp, which depends on CUDA) and the public EvolveResult API
// (evolve.hpp, which is host-only). Kept in a CUDA-free header so host
// translation units (bench harness, differential-test driver) can
// include evolve.hpp without dragging cuda_runtime.h in transitively.

#include <cstdint>
#include <string>

namespace hg_gpu {

// A capacity-bounded resource the GPU kernels can hit at runtime. When
// such a resource overflows, the kernel records the kind here (via the
// device-side DeviceErrors channel — see errors.hpp), and the host
// surfaces it back to the caller as an OverflowWarning attached to the
// EvolveResult. Crucially, overflows are warnings, not errors —
// kernels keep running on whatever budget they have left, and the
// caller decides whether the partial result is good enough or to retry
// with bigger pools.
enum class ErrorKind : uint32_t {
    kEdgePoolFull        = 0,
    kVertexPoolFull      = 1,
    kEventPoolFull       = 2,
    kStatePoolFull       = 3,
    kCausalPoolFull      = 4,
    kBranchialPoolFull   = 5,
    kMatchPoolFull       = 6,
    kCausalTripleMapFull = 7,
    kCausalPairMapFull   = 8,
    kBranchialMapFull    = 9,
    kEdgeConsumerNodes   = 12,
    kBranchialIndexNodes = 20,
    // The reduced predecessor adjacency (online TR's only stored structure) ran out of list
    // nodes. Sized from the config (one node per unique kept causal pair), so growing is a
    // real remedy.
    kTrPredsNodes        = 25,
    // A quotient-causal structure (transition records, their orbit-array arena, producer or
    // transition list nodes) ran out of capacity. Config-sized, so growing is a real remedy;
    // the causal edges reachable only through the dropped work are missing from the result.
    kQcNodes             = 26,
    kSigIndexNodes       = 16,
    kInvIndexNodes       = 17,
    kFrontierCapFull     = 18,
    kScratchOverflow     = 19,   // bounded local scratch (TR closure, WL)
    kDeviceOutOfMemory   = 21,   // host-side: an engine of the grown size no longer fits in VRAM
    // A device-resident scheduler ran past its spin budget. It means a defect -- the
    // termination detector should have fired -- and it exists so that defect costs a partial
    // result and a warning rather than a GPU occupied until the machine is rebooted. On a box
    // whose GPU also drives the display that distinction is the difference between a failed
    // run and a lost session.
    kPersistentStall     = 22,
    // The device IR arena could not give a worker a slot for the state it was canonicalizing.
    // Distinct from kScratchOverflow, which is a fixed per-thread bound: this one is sized from
    // the config (the arena scales with max_states), so growing the config is a real remedy and
    // the host's grow-and-retry treats it as retryable. Collapsing the two made a recoverable
    // capacity failure look like an unfixable kernel limit.
    kIRArenaExhausted    = 23,
    // The individualization search needed to go deeper than the device attempts, on the
    // PERSISTENT path. NOT config-controlled -- the depth is a constant the slot is shaped for
    // -- so growing cannot help, and that path takes no 1-WL fallback: the state is left
    // un-canonicalized rather than keyed by a hash that MERGES non-isomorphic states.
    kIRDepthExceeded     = 24,
    // A state was keyed by the 1-WL hash because it did not fit the IR slot, or wanted an
    // individualization depth the slot is not shaped for (k_ir_canon_range). 1-WL is
    // isomorphism-invariant in one direction only: it never separates isomorphic states, but it
    // DOES merge non-isomorphic ones -- tools/ir_vs_wl collides the prism against K3,3 on six
    // vertices, and the rook's graph against Shrikhande. So under CanonicalizeStates -> Full
    // this is a state whose dedup key is not exact, and the caller is promised exactness.
    // Nothing bounds how often an evolution reaches such a state, which is why it is reported
    // rather than tallied privately.
    kIRDegradedToWL      = 27,
    // The counter array is sized kCount and DeviceErrors::record drops any kind whose value is
    // not below it, so kCount must exceed every value above. The values are assigned by hand and
    // are not dense, so an implicit kCount tracks only the LAST entry -- which is how
    // kTrPredsNodes (25) and kQcNodes (26) came to sit above an implicit kCount of 25 and could
    // never be reported at all. Stated explicitly, with the static_assert below as the guard.
    kCount               = 32
};

// Every kind must be recordable. A kind at or above kCount is silently dropped by record(),
// which turns a capacity failure into no signal at all.
static_assert(static_cast<uint32_t>(ErrorKind::kQcNodes) <
              static_cast<uint32_t>(ErrorKind::kCount), "kQcNodes is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kTrPredsNodes) <
              static_cast<uint32_t>(ErrorKind::kCount), "kTrPredsNodes is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kIRDegradedToWL) <
              static_cast<uint32_t>(ErrorKind::kCount), "kIRDegradedToWL is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kPersistentStall) <
              static_cast<uint32_t>(ErrorKind::kCount), "kPersistentStall is unrecordable");

inline const char* error_kind_name(ErrorKind k) {
    switch (k) {
        case ErrorKind::kEdgePoolFull:        return "edge_pool";
        case ErrorKind::kVertexPoolFull:      return "vertex_pool";
        case ErrorKind::kEventPoolFull:       return "event_pool";
        case ErrorKind::kStatePoolFull:       return "state_pool (max_states)";
        case ErrorKind::kCausalPoolFull:      return "causal_edge_pool";
        case ErrorKind::kBranchialPoolFull:   return "branchial_edge_pool";
        case ErrorKind::kMatchPoolFull:       return "match_pool";
        case ErrorKind::kCausalTripleMapFull: return "causal_triple_dedup";
        case ErrorKind::kCausalPairMapFull:   return "causal_pair_dedup";
        case ErrorKind::kBranchialMapFull:    return "branchial_pair_dedup";
        case ErrorKind::kEdgeConsumerNodes:   return "edge_consumers (node pool)";
        case ErrorKind::kBranchialIndexNodes: return "branchial_index (node pool)";
        case ErrorKind::kTrPredsNodes:        return "tr_preds (node pool)";
        case ErrorKind::kQcNodes:             return "quotient-causal records/nodes";
        case ErrorKind::kSigIndexNodes:       return "signature_index (node pool)";
        case ErrorKind::kInvIndexNodes:       return "vertex_inverted_index (node pool)";
        case ErrorKind::kFrontierCapFull:     return "frontier buffer";
        case ErrorKind::kScratchOverflow:     return "per-thread scratch (TR/WL)";
        case ErrorKind::kIRArenaExhausted:    return "device IR arena (retryable: grow config)";
        case ErrorKind::kIRDepthExceeded:     return "IR search depth (not config-controlled)";
        case ErrorKind::kIRDegradedToWL:      return "state keyed by 1-WL, not exact IR";
        case ErrorKind::kDeviceOutOfMemory:   return "device memory (engine allocation)";
        case ErrorKind::kPersistentStall:     return "persistent scheduler spin budget (defect)";
        default:                              return "unknown";
    }
}

// One occurrence of a capacity overflow during evolve(). Counts are the
// per-kernel-launch tally observed on the device, not a cumulative
// total across the whole evolve — that means the same ErrorKind may
// appear multiple times in EvolveResult.warnings if the kernel that
// owns it ran multiple times (per-step kernels typically do).
//
// `context` names the phase ("match kernel step 3", "rewrite kernel
// step 5", "ir hash", etc.) so the operator can locate the bottleneck
// quickly. `count` is a lower bound on how much more capacity was
// needed in that phase — at least N more pool slots, etc.
struct OverflowWarning {
    ErrorKind   kind;
    uint32_t    count;
    std::string context;
};

}  // namespace hg_gpu
