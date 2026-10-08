#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/types.hpp"

#include <cuda/atomic>

#include <cstdint>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// Sentinel for DevicePatternEdge::pivot_var meaning "no bound var to pivot
// from" — only valid on pattern edge 0 (connectivity-scheduling ensures every
// subsequent pattern edge shares at least one var with a prior edge).
constexpr uint8_t kNoPivotVar = 0xFF;

// Device-side pattern edge: per-position variable indices and the pivot.
//
// `pivot_var` is the connectivity-schedule's contribution: for every pattern edge at depth ≥ 1,
// pivot_var is the LHS variable index (guaranteed bound at the point this edge runs) that ties
// this edge to the subgraph matched so far. The match kernel looks up
// `vertex_inverted_index[binding[pivot_var]]` to get a degree-bounded candidate list (typically
// 2–10 entries). Pattern edge 0, and an edge without a pivot, take their candidates from the
// state's own edge slice; hgcommon::bind_pattern_edge, which the host runs too, rejects an edge
// of another arity or with vertex repetitions the pattern does not allow.
struct DevicePatternEdge {
    uint8_t  arity = 0;
    uint8_t  vars[kMaxArity] = {0};
    uint8_t  pivot_var = kNoPivotVar;
};

// RHS edges reference LHS variable indices [0, num_lhs_vars) for re-used vars
// and fresh-var indices [num_lhs_vars, num_rhs_vars) for newly introduced
// variables. The rewrite kernel atomically allocates a fresh VertexId per
// fresh-var per match.
struct DeviceRhsEdge {
    uint8_t arity = 0;
    uint8_t vars[kMaxArity] = {0};
};

struct DeviceRule {
    DevicePatternEdge lhs[kMaxPatternEdges];
    DeviceRhsEdge     rhs[kMaxPatternEdges];
    uint8_t           num_lhs_edges = 0;
    uint8_t           num_lhs_vars  = 0;
    uint8_t           num_rhs_edges = 0;
    uint8_t           num_rhs_vars  = 0;  // total (includes new vars in RHS)

    // Variables occurring in the RHS but not the LHS, as a mask rather than as a count.
    // The set is not an index RANGE: a rule may number its LHS variables sparsely, and
    // num_lhs_vars is a count on the host and a max-index-plus-one here, so neither reading
    // of [num_lhs_vars, num_rhs_vars) names the right variables. hgcommon takes the mask.
    uint32_t          new_var_mask  = 0;
};

// One match found during pattern matching.
//
// `step` is the depth of the state this match was found in, carried on the RECORD so the
// rewrite that consumes it needs nothing from its scheduler: there is no host loop variable to
// read a depth from, and records from several depths are live in the pool at once.
//
// `published` is the record's own publication flag, stored LAST with release ordering.
// Claiming a pool index bumps the pool counter before the record is filled, so a consumer
// running concurrently with the producer -- which is what a device-resident scheduler does --
// can see the index and read an unwritten record. A kernel boundary would hide that; there is
// none inside the evolution. Consumers wait for this flag;
// producers set it once the rest of the record is written.
struct MatchRecord {
    RuleId   rule_id   = 0;
    StateId  state_id  = INVALID_ID;
    uint32_t step      = 0;
    uint32_t published = 0;
    uint8_t  num_edges = 0;
    EdgeId   matched_edges[kMaxPatternEdges] = {INVALID_ID};
};

// Build DeviceRule from the host EvolveInput rule. Pads arrays to kMax*.
DeviceRule make_device_rule(const RewriteRule& rule);

// Threads per block for match_state_rule. Its body stripes the depth-0 candidates across
// exactly these threads, so every scheduler that calls it must launch with this shape.
constexpr uint32_t kMatchBlockThreads = 32;

// How many of ONE (state, rule) pair's matches the per-(state, rule) cap can rank at once.
// 512 x 8 bytes = 4 KB of shared memory, taken only by a block that is capping. Beyond it the
// k-th smallest rank cannot be identified, so the cap is skipped and the run records
// kDrainCapBufferFull rather than applying the cap to the wrong k.
constexpr uint32_t kDrainCapBuffer = 512;

// Publish a filled record: the release store that makes everything written before it visible
// to a consumer that observes the flag.
__device__ __forceinline__ void publish_match(MatchRecord& m) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ref(m.published);
    ref.store(1u, cuda::memory_order_release);
}

// Wait until record `m` is fully written, then read it. Returns immediately for a scheduler
// that separates matching from rewriting with a kernel boundary, because the flag is already
// set by then. The wait always ends: a producer that claimed an index below the pool capacity
// always finishes writing it.
__device__ __forceinline__ void await_match(const MatchRecord& m) {
    cuda::atomic_ref<const uint32_t, cuda::thread_scope_device> ref(m.published);
    while (ref.load(cuda::memory_order_acquire) == 0u) __nanosleep(64);
}

// One (state, rule) pair, matched by ONE BLOCK of kMatchBlockThreads threads. Exposed so a
// scheduler in another translation unit drives this implementation rather than growing a
// second copy of it.
__device__ void match_state_rule(const DeviceState& ds, const DeviceRule* rules,
                                 StateId state_id, uint32_t rid, uint32_t step,
                                 typename Pool<MatchRecord>::DeviceView out);

// Run the match kernel for (state_id, all rules), populating out_matches.
// Returns the number of matches written. `step` is stamped on every record.
// The largest per-thread stack (cudaFuncAttributes::localSizeBytes) among the matching kernels.
size_t match_kernels_stack_bytes();

uint32_t run_match_kernel(const EngineState&            engine,
                          const std::vector<DeviceRule>& rules,
                          StateId                        state_id,
                          Pool<MatchRecord>&             out_matches,
                          uint32_t                       step = 0);

// Batched: every (state_id, rule) pair in one launch. Reads the count through
// Pool::counter rather than a D2H per call.
void run_match_kernel_batch_nosync(const EngineState& engine,
                                   const DeviceRule*  d_rules,
                                   uint32_t           num_rules,
                                   const StateId*     d_state_ids,
                                   uint32_t           num_state_ids,
                                   Pool<MatchRecord>& out_matches,
                                   uint32_t           step = 0);

}  // namespace gpu
}  // namespace HG_NAMESPACE