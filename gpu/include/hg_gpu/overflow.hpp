#pragma once
#include "hgcommon/namespace.hpp"

// Capacity-overflow types shared between the device-side error channel
// (errors.hpp, which depends on CUDA) and the public EvolveResult API
// (evolve.hpp, which is host-only). Kept in a CUDA-free header so host
// translation units (bench harness, differential-test driver) can
// include evolve.hpp without dragging cuda_runtime.h in transitively.

#include <cstdint>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

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
    // The individualization search found MORE automorphisms than the generator table holds,
    // while ORBITS were being computed. Generators serve two purposes: for search pruning a
    // short table costs time only, because automorphic branches reach the same canonical form
    // either way; for ORBITS it changes the answer, since orbits are fused over the generators
    // found and a short table fuses less, giving orbits that are too FINE. Those orbits are
    // what the quotient reconstruction identifies instances by, so a truncated table is a
    // wrong answer rather than a slow one. The budget is config-controlled, so growing it is a
    // real remedy and the host's grow-and-retry treats this as retryable.
    kIRGeneratorsExceeded = 28,
    // A state reached deduplication with a canonical hash of 0, which is not a hash: 0 is what
    // the per-state hash array holds for "not computed yet", and the empty state -- the one
    // case with nothing to compute a hash from -- has its own reserved value
    // (hgcommon::EMPTY_STATE_CANONICAL_HASH). Such a state is NOT merged, because merging is
    // the destructive choice: a state deduped away is a subtree never explored. It is kept and
    // reported, so the answer is over-complete with a warning rather than short without one.
    kUncomputedStateHash = 27,
    // The canonical dedup map exhausted its probe run. Recorded rather than inferred: exhaustion
    // is indistinguishable from a hit at the map's interface, so before this existed an overfull
    // map silently reported new states as already-seen and dropped them from the answer. Same
    // stance as kUncomputedStateHash above -- keep the state and report, so the answer is
    // over-complete with a warning rather than short without one.
    kCanonicalMapFull    = 29,
    // An event signature stood a RAW edge identifier in place of a canonical edge rank, because
    // the rank was unavailable when the event was stamped. Not a capacity failure and not
    // retryable: the run completes and the answer is well defined, but a signature built from a
    // raw id is not an isomorphism invariant, so two runs that agree on everything else can
    // report different event counts. Reported for the same reason kIRGeneratorsExceeded is --
    // the caller cannot otherwise distinguish it from a disagreement in the evolution.
    kEventSigRawFallback = 30,
    // One (state, rule) pair produced more matches than the drain cap can rank at once, so
    // "MatchesPerStateRule" was not applied to it. Not a wrong answer: an UNCAPPED one, said out
    // loud, which is the engine's partial-result contract rather than a silent substitution.
    kDrainCapBufferFull = 31,
    // The counter array is sized kCount and DeviceErrors::record drops any kind whose value is
    // not below it, so kCount must exceed every value above. The values are assigned by hand and
    // are not dense, so an implicit kCount tracks only the LAST entry -- which is how
    // kTrPredsNodes (25) and kQcNodes (26) came to sit above an implicit kCount of 25 and could
    // never be reported at all. Stated explicitly, with the static_assert below as the guard.
    kCount               = 32
};

// DISTINCT VALUES, NOT MERELY IN-RANGE ONES. record() indexes the counter array by the enum
// value, so two kinds sharing a value share a counter: they become one condition that reports
// under whichever name a switch happens to list first. kIRGeneratorsExceeded and
// kUncomputedStateHash were both 28, which made evolve.cu's retry answer an uncomputed state
// hash by doubling the IR generator budget -- the remedy for the OTHER condition -- up to
// eight times. The in-range assertions below cannot see that; this can.
static_assert(static_cast<uint32_t>(ErrorKind::kIRGeneratorsExceeded) !=
              static_cast<uint32_t>(ErrorKind::kUncomputedStateHash),
              "two ErrorKinds share a counter, so one condition reports as the other");

// Every kind must be recordable. A kind at or above kCount is silently dropped by record(),
// which turns a capacity failure into no signal at all.
static_assert(static_cast<uint32_t>(ErrorKind::kQcNodes) <
              static_cast<uint32_t>(ErrorKind::kCount), "kQcNodes is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kUncomputedStateHash) <
              static_cast<uint32_t>(ErrorKind::kCount), "kUncomputedStateHash is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kCanonicalMapFull) <
              static_cast<uint32_t>(ErrorKind::kCount), "kCanonicalMapFull is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kEventSigRawFallback) <
              static_cast<uint32_t>(ErrorKind::kCount), "kEventSigRawFallback is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kTrPredsNodes) <
              static_cast<uint32_t>(ErrorKind::kCount), "kTrPredsNodes is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kPersistentStall) <
              static_cast<uint32_t>(ErrorKind::kCount), "kPersistentStall is unrecordable");
static_assert(static_cast<uint32_t>(ErrorKind::kDrainCapBufferFull) <
              static_cast<uint32_t>(ErrorKind::kCount), "kDrainCapBufferFull is unrecordable");

const char* error_kind_name(ErrorKind k);

// One occurrence of a capacity overflow during evolve(). A count is the
// per-kernel-launch tally observed on the device, not a cumulative
// total across the whole evolve — that means the same ErrorKind may
// appear multiple times in EvolveResult.warnings if the kernel that
// owns it ran multiple times (per-step kernels typically do).
//
// `context` names the phase ("match kernel step 3", "rewrite kernel
// step 5", "ir hash", etc.) so the operator can locate the bottleneck
// quickly.
//
// `count` SAYS THAT A KIND FIRED, NOT HOW MUCH CAPACITY WAS MISSING.
// DeviceErrors::DeviceView::record latches: the first observers of a
// kind pay the atomic and every later one reads the counter and leaves,
// because under saturation the unlatched tally was one atomicAdd per
// thread per inner-loop iteration onto a single 4-byte address. So the
// number is a small count of racing observers. Nothing decides on its
// magnitude — the retry loop doubles the field the kind names whatever
// the number is (evolve.cu, grow_config_for).
struct OverflowWarning {
    ErrorKind   kind;
    uint32_t    count;
    std::string context;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE