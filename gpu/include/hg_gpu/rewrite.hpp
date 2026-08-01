#pragma once

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/match.hpp"

#include <cstdint>
#include <vector>

namespace hg_gpu {

// Run the rewrite kernel for a batch of matches. For each match:
//   1. Re-derive variable bindings from (lhs, matched_edges).
//   2. Atomic-allocate fresh VertexIds for new RHS vars.
//   3. Atomic-allocate a new StateId; its bitset is (parent_bitset
//      minus consumed edges) plus the newly created RHS edges.
//   4. Atomic-allocate new Edge slots for each RHS edge, write vertex
//      tuples into vertex_pool, compute signatures, insert into the
//      signature + vertex inverted indices.
//
// After the kernel, the engine's state_count, edge_pool, vertex_pool, and
// vertex_high_water have advanced and the indices are populated with the
// new edges. No events / causal / branchial structures yet (M6).
//
// Returns the number of new states produced (== num_matches, one per match).
// One match, applied by ONE THREAD: consumes the matched edges, produces the RHS edges, and
// emits the event. Exposed so a scheduler in another translation unit drives this
// implementation rather than growing a second copy of it.
//
// What one application produced. Both halves are needed by a scheduler that finishes the work
// itself: the STATE to hash and re-enqueue, and the EVENT to stamp an identity onto once that
// hash exists. The level-synchronous scheduler discards both and takes the step's whole
// [before, after) range instead.
struct AppliedMatch {
    StateId state = INVALID_ID;
    EventId event = INVALID_ID;   // both are INVALID_ID when a capacity claim failed
};

// `sub`, when non-null, receives clock64()-cycle attribution for the six stretches of one
// application, atomicAdd-ed per call: [0] bind+preflight reservations, [1] RHS edge emission
// (+ index inserts), [2] CSR merge-copy of the child's edge list, [3] event record write,
// [4] causal rendezvous (producer + consumer sides), [5] branchial scan.
__device__ AppliedMatch apply_one_match(DeviceState ds, const DeviceRule* rules,
                                        const MatchRecord& m, uint32_t step,
                                        unsigned long long* sub = nullptr);

// Insert a causal edge (producer -> consumer via shared edge e), first-writer-wins on the
// (p, c, e) triple, with online TR when enabled. EXTERNAL because the quotient-causal DP
// emits its canonical-event pairs through this same machinery (shared edge 0).
__device__ void try_add_causal_edge(DeviceState ds, EventId p, EventId c, EdgeId e);

uint32_t run_rewrite_kernel(EngineState&                   engine,
                            const std::vector<DeviceRule>& rules,
                            const Pool<MatchRecord>&       matches,
                            uint32_t                       num_matches,
                            uint32_t                       step);

// Pre-allocated-rules variant: d_rules is a device pointer the caller has
// already populated. Avoids cudaMalloc/cudaFree on the hot path.
uint32_t run_rewrite_kernel_with(EngineState&             engine,
                                 const DeviceRule*        d_rules,
                                 const Pool<MatchRecord>& matches,
                                 uint32_t                 num_matches,
                                 uint32_t                 step);

// Same as run_rewrite_kernel_with but caller passes the known state count
// before the call and receives it after — avoids the internal D2H round-
// trip that num_states_host() does.
void run_rewrite_kernel_with_nosync(EngineState&             engine,
                                    const DeviceRule*        d_rules,
                                    const Pool<MatchRecord>& matches,
                                    uint32_t                 num_matches,
                                    uint32_t                 step);

}  // namespace hg_gpu
