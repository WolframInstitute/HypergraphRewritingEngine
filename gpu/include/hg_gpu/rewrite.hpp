#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/match.hpp"

#include <cstdint>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

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
// emits the event. The child's kept edges are the caller's to copy (copy_kept_edges) before
// anything reads the child's slice. Exposed so a scheduler in another translation unit drives this
// implementation rather than growing a second copy of it.
//
// What one application produced. A scheduler that finishes the work itself needs both halves:
// the STATE to hash and re-enqueue, and the EVENT to stamp an identity onto once that hash
// exists.
// The kept part of a child's edge list: the parent's edges other than the consumed ones, in
// parent order, written at dst_offset. apply_one_match reserves the child's slice and writes
// its produced edges after the kept ones; copy_kept_edges writes the kept ones.
struct KeptCopy {
    uint32_t src_offset = 0;
    uint32_t src_count  = 0;
    uint32_t dst_offset = 0;
    uint32_t n_consumed = 0;
    EdgeId   consumed[kMaxPatternEdges];
};

struct AppliedMatch {
    StateId  state = INVALID_ID;
    EventId  event = INVALID_ID;   // both are INVALID_ID when a capacity claim failed
    KeptCopy kept{};
    // Keyed rewrites (keyed.hpp): the rewrite id with REWRITE_TWIN_CANDIDATE, or 0.
    uint32_t keyed = 0;
};

// Copy a child's kept edges. Par is a lane policy (hgcommon::IrSerial, or IrTile<W> for W
// lanes together): with kFans each lane of the tile takes every W-th parent edge and a ballot
// over the tile places the kept ones in parent order. Every lane of the tile must call it
// together.
template <class Par>
__device__ inline void copy_kept_edges(const DeviceState& ds, const KeptCopy& k, Par) {
    EdgeId* dst = ds.state_edge_ids + k.dst_offset;
    const EdgeId* src = ds.state_edge_ids + k.src_offset;
    auto kept = [&](EdgeId e) {
        for (uint32_t i = 0; i < k.n_consumed; ++i)
            if (k.consumed[i] == e) return false;
        return true;
    };
    if constexpr (!Par::kFans) {
        uint32_t cursor = 0;
        for (uint32_t i = 0; i < k.src_count; ++i)
            if (kept(src[i])) dst[cursor++] = src[i];
    } else {
        constexpr uint32_t W = Par::kWidth;
        const uint32_t rank = Par::rank();
        const uint32_t tile_mask = Par::mask();
        const uint32_t shift = (threadIdx.x & 31u) & ~(W - 1u);
        uint32_t cursor = 0;
        for (uint32_t base = 0; base < k.src_count; base += W) {
            const uint32_t i = base + rank;
            const EdgeId e = i < k.src_count ? src[i] : INVALID_ID;
            const bool keep = i < k.src_count && kept(e);
            const uint32_t mask = __ballot_sync(tile_mask, keep) >> shift;
            if (keep) dst[cursor + __popc(mask & ((1u << rank) - 1u))] = e;
            cursor += __popc(mask);
        }
        // Each lane publishes its own writes; the leader's later release covers only its own.
        __threadfence();
    }
}

// `sub`, when non-null, receives clock64()-cycle attribution for the six stretches of one
// application, atomicAdd-ed per call: [0] bind+preflight reservations, [1] RHS edge emission
// (+ index inserts), [2] the child's slice header and produced ids, [3] event record write,
// [4] causal rendezvous (producer + consumer sides), [5] branchial scan.
__device__ AppliedMatch apply_one_match(const DeviceState& ds, const DeviceRule* rules,
                                        const MatchRecord& m, uint32_t step,
                                        unsigned long long* sub = nullptr);

// Insert a causal edge (producer -> consumer via shared edge e), first-writer-wins on the
// (p, c, e) triple, with online TR when enabled. EXTERNAL because the quotient-causal DP
// emits its canonical-event pairs through this same machinery (shared edge 0).
__device__ void try_add_causal_edge(const DeviceState& ds, EventId p, EventId c, EdgeId e);

// The transitive-reduction gate's setup: builds the causal chain 1 <- 2 <- ... <- n + 1 <- n + 2
// in the reduced predecessor lists and offers the edge 1 -> n + 2, which the chain makes
// redundant. Needs tr enabled and max_events, tr_preds_nodes above n + 2. The test reads the
// causal edge count and the warnings.
void add_redundant_edge_over_chain(EngineState& engine, uint32_t n);

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

}  // namespace gpu
}  // namespace HG_NAMESPACE