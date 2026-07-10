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
    kDescSetFull         = 10,
    kAncSetFull          = 11,
    kEdgeConsumerNodes   = 12,
    kBranchialIndexNodes = 20,
    kDescListNodes       = 14,
    kAncListNodes        = 15,
    kSigIndexNodes       = 16,
    kInvIndexNodes       = 17,
    kFrontierCapFull     = 18,
    kScratchOverflow     = 19,   // bounded local scratch (TR closure, WL)
    kDeviceOutOfMemory   = 21,   // host-side: an engine of the grown size no longer fits in VRAM
    kCount
};

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
        case ErrorKind::kDescSetFull:         return "desc_set";
        case ErrorKind::kAncSetFull:          return "anc_set";
        case ErrorKind::kEdgeConsumerNodes:   return "edge_consumers (node pool)";
        case ErrorKind::kBranchialIndexNodes: return "branchial_index (node pool)";
        case ErrorKind::kDescListNodes:       return "desc_list (node pool)";
        case ErrorKind::kAncListNodes:        return "anc_list (node pool)";
        case ErrorKind::kSigIndexNodes:       return "signature_index (node pool)";
        case ErrorKind::kInvIndexNodes:       return "vertex_inverted_index (node pool)";
        case ErrorKind::kFrontierCapFull:     return "frontier buffer";
        case ErrorKind::kScratchOverflow:     return "per-thread scratch (TR/WL)";
        case ErrorKind::kDeviceOutOfMemory:   return "device memory (engine allocation)";
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
