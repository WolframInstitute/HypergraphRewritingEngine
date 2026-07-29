#pragma once

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/types.hpp"

#include <cuda/atomic>

#include <cstdint>
#include <vector>

namespace hg_gpu {

// Maximum number of compatible data signatures per pattern edge. Equal to the
// largest Bell number we admit: Bell(5)=52, Bell(6)=203 — pick 64 to cover up
// to arity 5 (typical Wolfram rules use arity 2–3).
constexpr uint32_t kMaxCompatibleSigs = 64;

// Sentinel for DevicePatternEdge::pivot_var meaning "no bound var to pivot
// from" — only valid on pattern edge 0 (connectivity-scheduling ensures every
// subsequent pattern edge shares at least one var with a prior edge).
constexpr uint8_t kNoPivotVar = 0xFF;

// Device-side pattern edge: per-position variable indices and the precomputed
// list of compatible data-edge signature hashes. Wolfram non-distinct binding
// allows distinct pattern vars to bind to the same data vertex, so a single
// pattern signature is compatible with multiple data signatures (every
// coarsening of the pattern partition). Precomputing the compat list on the
// host means the match kernel can seed candidates via the signature index for
// pattern edge 0 without per-match enumeration.
//
// `pivot_var` is the connectivity-schedule's contribution: for every pattern
// edge at depth ≥ 1, pivot_var is the LHS variable index (guaranteed bound
// at the point this edge runs) that ties this edge to the subgraph matched
// so far. The match kernel looks up `vertex_inverted_index[binding[pivot_var]]`
// to get a degree-bounded candidate list (typically 2–10 entries) instead of
// walking the global signature_index bucket (1000s of entries on dense
// graphs). This is the adapted-HGMatch pattern: signature_index seeds edge
// 0; inverted_index drives edges 1..R-1.
struct DevicePatternEdge {
    uint8_t  arity = 0;
    uint8_t  vars[kMaxArity] = {0};
    uint8_t  num_compat_sigs = 0;
    uint8_t  pivot_var = kNoPivotVar;
    uint64_t compat_sig_hashes[kMaxCompatibleSigs] = {0};
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
// rewrite that consumes it needs nothing from its scheduler. The level-synchronous loop could
// take it from the loop variable; a device-resident one has no loop, and its records from
// several depths are live in the pool at once.
//
// `published` is the record's own publication flag, stored LAST with release ordering.
// Claiming a pool index bumps the pool counter before the record is filled, so a consumer
// running concurrently with the producer -- which is what a device-resident scheduler does --
// can see the index and read an unwritten record. A kernel boundary hides that for the
// level-synchronous scheduler; nothing hides it without one. Consumers wait for this flag;
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
__device__ void match_state_rule(DeviceState ds, const DeviceRule* rules,
                                 StateId state_id, uint32_t rid, uint32_t step,
                                 typename Pool<MatchRecord>::DeviceView out);

// Run the match kernel for (state_id, all rules), populating out_matches.
// Returns the number of matches written. `step` is stamped on every record.
uint32_t run_match_kernel(const EngineState&            engine,
                          const std::vector<DeviceRule>& rules,
                          StateId                        state_id,
                          Pool<MatchRecord>&             out_matches,
                          uint32_t                       step = 0);

// Batched variant: process all (state_id, rule) pairs across `state_ids` in
// a single kernel launch. Much faster than calling run_match_kernel per
// state because we avoid per-state kernel launch overhead.
//
// `d_rules` must already contain `rules` uploaded to device (caller reuses
// across steps). Returns total number of matches written to out_matches.
uint32_t run_match_kernel_batch(const EngineState& engine,
                                const DeviceRule*  d_rules,
                                uint32_t           num_rules,
                                const StateId*     d_state_ids,
                                uint32_t           num_state_ids,
                                Pool<MatchRecord>& out_matches,
                                uint32_t           step = 0);

// Variant that skips the final size_host D2H — caller reads the count
// separately (e.g. via Pool::counter pointer) to avoid per-step D2H.
void run_match_kernel_batch_nosync(const EngineState& engine,
                                   const DeviceRule*  d_rules,
                                   uint32_t           num_rules,
                                   const StateId*     d_state_ids,
                                   uint32_t           num_state_ids,
                                   Pool<MatchRecord>& out_matches,
                                   uint32_t           step = 0);

}  // namespace hg_gpu
