#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/types.hpp"
#include "hg_gpu/work_log.hpp"
#include "hgcommon/sampling_core.hpp"  // SuccessorIdentity

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
    // lhs[] is in the rule's authored order, as the host's RewriteRule::lhs, so a match's
    // matched_edges[p] is the edge bound to authored pattern edge p on both devices; the
    // transition key reads the matched edges' ranks in that order. order[k] is the authored
    // index of the pattern edge the join binds at depth k (make_device_rule).
    DevicePatternEdge lhs[kMaxPatternEdges];
    uint8_t           order[kMaxPatternEdges] = {0};
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

// How many candidates of one (state, rule) pair, or of one state under
// MaxSuccessorStatesPerParent, a capping block ranks in shared memory and its scratch. 512 x 8
// bytes = 4 KB of shared memory, taken only by a block that is capping. A pair or state with more
// is counted, given a spill buffer of exactly its count from the block's arena
// (SuccessorCtx::spill), and ranked again there; an arena too small records kIRArenaExhausted,
// which grow-and-retry doubles. A scheduler without a SuccessorCtx records kDrainCapBufferFull
// and leaves the pair uncapped.
constexpr uint32_t kDrainCapBuffer = 512;

// The 64-bit words of a match block's shared buffer: under a cap, kDrainCapBuffer ranks, rules,
// keep bytes and quota bits; without one, the state's staged edges (5,696 bytes, 438 edges of
// arity 2).
constexpr uint32_t kMatchSharedWords =
    kDrainCapBuffer + kDrainCapBuffer / 4u + kDrainCapBuffer / 8u + kDrainCapBuffer / 64u;

// True when match_state_rule emits every match of a (state, rule) pair as found: no
// per-(state, rule) cap, no per-parent cap and no draw. Only then can a pair's matching be split
// into parts over ranges of its anchor candidates, since no pass needs the whole pair.
__device__ __forceinline__ bool match_state_rule_splits(const DeviceState& ds) {
    // No draw applies where a cap chooses: under MaxStatesPerStep every transition is a candidate
    // of its step, as on the host.
    const bool sampling = (ds.transition_rate < 1.0 || ds.num_rule_weights != 0u) &&
                          ds.max_states_per_step == 0u;
    return ds.matches_per_state_rule == 0u && ds.max_successor_states_per_parent == 0u &&
           !sampling;
}

// The first anchor (an index into the state's slice of `count` edges) of part `part` of
// `parts`; part `parts` gives `count`. The parts cover the slice once, in order.
__host__ __device__ __forceinline__ uint32_t match_split_root_begin(uint32_t count, uint32_t part,
                                                                    uint32_t parts) {
    return static_cast<uint32_t>(uint64_t{count} * part / parts);
}

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

// True when record `m` is fully written; its fields may then be read.
__device__ __forceinline__ bool match_published(const MatchRecord& m) {
    cuda::atomic_ref<const uint32_t, cuda::thread_scope_device> ref(m.published);
    return ref.load(cuda::memory_order_acquire) != 0u;
}

// What the successor caps (MaxSuccessorStatesPerParent, MaxStatesPerStep) need to tell a
// transition's successor state apart (hgcommon::SuccessorIdentity): the run's state mode, and
// for a Full successor the block's IR arena and slot (successor_exact_hash_device).
// "StateCost": candidate `cand` of step `step` needs its successor's cost (step_cost_one).
struct CostEntry {
    uint32_t cand;
    uint32_t step;
    uint32_t published;
};

struct SuccessorCtx {
    CanonicalizationMode mode;
    DeviceArena::View    arena;
    uint32_t**           slot;
    uint64_t*            slot_words;
    // The block's spill slot, claimed from `arena` and reused while large enough: the
    // candidates of a cap whose pair or state has more than kDrainCapBuffer of them.
    uint32_t**           spill = nullptr;
    uint64_t*            spill_words = nullptr;
    // Under a step cap that counts successor states or reads a cost, indexed by candidate-pool
    // index: each candidate's successor identity and cost key, and 1 once both are written
    // (step_cost_one). Null otherwise.
    uint64_t*                    cand_cost = nullptr;
    hgcommon::SuccessorIdentity* cand_id = nullptr;
    uint32_t*                    cand_ready = nullptr;
    // The candidates whose cost an idle block computes (CostEntry), each booked on its step.
    WorkLogView<CostEntry>       cost_log{};
};

// Step d gains n units of work: its selection runs when the count returns to zero
// (step_release, persistent.cu).
__device__ inline void step_book(const DeviceState& ds, uint32_t d, uint32_t n) {
    if (ds.max_states_per_step != 0u && d < ds.step_slots) atomicAdd(&ds.step_pending[d], n);
}

// Per-block scratch of MaxSuccessorStatesPerParent: kDrainCapBuffer candidates of one state, each
// with its matched edges (kMaxPatternEdges, by pattern position), its successor identity, its
// successor rank and the sort scratch of hgcommon::successor_ranks. Indexed by blockIdx.x below
// `blocks`.
struct SuccessorScratch {
    EdgeId*                      edges = nullptr;
    hgcommon::SuccessorIdentity* id = nullptr;
    uint64_t*                    rank = nullptr;
    uint32_t*                    order = nullptr;
    uint8_t*                     lead = nullptr;
    uint32_t                     blocks = 0;
};

// The identity of the state rule `rule` makes from `parent` with the match `matched` (edges by
// pattern position), under ctx.mode: hgcommon::SuccessorIdentity's rule. Every lane of the warp
// calls it with the same arguments (the Full hash runs on the warp). False when the hash could
// not be computed; the error is recorded.
__device__ bool successor_identity_device(const DeviceState& ds, const DeviceRule& rule,
                                          StateId parent, const EdgeId* matched,
                                          const SuccessorCtx& ctx,
                                          hgcommon::SuccessorIdentity& out);

// "StateCost": the cost key (hgcommon::state_cost_key) of the same successor, from its IR
// canonical form (state_cost_of_form_device), on the warp as successor_identity_device runs.
// `full_id`, when non-null, receives the successor's identity under Full states (its canonical
// hash), which the form's pass also gives. False when the form or the record could not be
// computed; the error is recorded and `out` is the key of +Infinity.
__device__ bool successor_cost_device(const DeviceState& ds, const DeviceRule& rule,
                                      StateId parent, const EdgeId* matched,
                                      const SuccessorCtx& ctx, uint64_t& out,
                                      hgcommon::SuccessorIdentity* full_id = nullptr);

// The successor identity of the candidate out[i] (when the step cap counts successor states)
// and, under "StateCost", its cost, into ctx.cand_id, ctx.cand_cost and ctx.cand_ready, on the
// calling warp. A candidate whose form or record could
// not be computed is left unready, and the step's selection computes it.
__device__ void step_cost_one(const DeviceState& ds, const DeviceRule* rules,
                              typename Pool<MatchRecord>::DeviceView out, const SuccessorCtx& ctx,
                              uint32_t i);

// One (state, rule) pair, matched by ONE BLOCK of kMatchBlockThreads threads. Exposed so a
// scheduler in another translation unit drives this implementation rather than growing a
// second copy of it. `succ` and `scratch` serve MaxSuccessorStatesPerParent under Automatic and
// Full states; a scheduler that passes null for them records kScratchOverflow when the cap
// needs them. When match_state_rule_splits(ds) holds, the call matches part `part` of `parts`
// (match_split_root_begin), and the parts of a pair together emit each of its matches once,
// each with its transition_rank_device at its index in `out` when `out_rank` is non-null;
// otherwise `parts` must be 1 and `out_rank` is not written.
__device__ void match_state_rule(const DeviceState& ds, const DeviceRule* rules,
                                 StateId state_id, uint32_t rid, uint32_t step,
                                 typename Pool<MatchRecord>::DeviceView out,
                                 const SuccessorCtx* succ = nullptr,
                                 const SuccessorScratch* scratch = nullptr,
                                 uint32_t part = 0, uint32_t parts = 1,
                                 uint64_t* out_rank = nullptr);

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