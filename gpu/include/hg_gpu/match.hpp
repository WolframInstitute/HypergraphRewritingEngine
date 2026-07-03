#pragma once

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/types.hpp"

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
};

// One match found during pattern matching.
struct MatchRecord {
    RuleId   rule_id  = 0;
    StateId  state_id = INVALID_ID;
    uint8_t  num_edges = 0;
    EdgeId   matched_edges[kMaxPatternEdges] = {INVALID_ID};
};

// Build DeviceRule from the host EvolveInput rule. Pads arrays to kMax*.
DeviceRule make_device_rule(const RewriteRule& rule);

// Run the match kernel for (state_id, all rules), populating out_matches.
// Returns the number of matches written.
uint32_t run_match_kernel(const EngineState&            engine,
                          const std::vector<DeviceRule>& rules,
                          StateId                        state_id,
                          Pool<MatchRecord>&             out_matches);

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
                                Pool<MatchRecord>& out_matches);

// Variant that skips the final size_host D2H — caller reads the count
// separately (e.g. via Pool::counter pointer) to avoid per-step D2H.
void run_match_kernel_batch_nosync(const EngineState& engine,
                                   const DeviceRule*  d_rules,
                                   uint32_t           num_rules,
                                   const StateId*     d_state_ids,
                                   uint32_t           num_state_ids,
                                   Pool<MatchRecord>& out_matches);

}  // namespace hg_gpu
