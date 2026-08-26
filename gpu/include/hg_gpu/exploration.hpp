#pragma once
#include "hgcommon/namespace.hpp"
// Which newly created states get expanded.
//
// Asked in one place. A second copy of this predicate would not crash; it would silently
// explore a different state set, which is the defect class the shared-core work exists to
// close.

#include "hgcommon/core.hpp"
#include "hgcommon/event_core.hpp"
#include "hgcommon/sampling_core.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/hash_table.hpp"
#include "hg_gpu/types.hpp"

#include <cuda/atomic>

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

// Canonical hash -> the first state id seen with that hash. First writer wins, and its id is
// the one that gets expanded.
using DedupMap = ConcurrentMap<uint64_t, uint32_t>;

// ONE PUBLICATION PER CLASS: the frame owner AND the step it reads, in a single value.
//
// These were two maps under one flag, and a thread that lost the `frame` insert went on to read
// `frame_step` before the winner had written it. It found the slot EMPTY -- not locked, so there
// was nothing to wait on -- took the caller's own depth as the fallback, and every event it
// replayed signed with its instance depth instead of the class's, which makes the two signature
// sets disjoint (quotient_replay_core.hpp:141).
//
// The old spin in `lookup_waiting` never closed this. It waited on LOCKED, and the losing thread
// arrives at a slot the winner has not claimed AT ALL, which is EMPTY. Removing the lock
// narrowed the window; it did not open it.
//
// A pair in one value removes it by construction: there is one insert, so there is no second
// publication to race. It also deletes the map `quotient.cu` sized at max_events while its twin
// got max_events*2 -- the one that saturates first -- so the device holds less, not more.
using FrameMap = ConcurrentMap<uint64_t, uint64_t>;

// The value is a packed (step, sid) pair through hgcommon::id_key, NOT a hand-rolled shift.
// core.hpp says why in the one place the rule lives: "Every map keyed by ids goes through this,
// on BOTH engines. Two sites packing ids their own way is how one of them ends up without the
// offset." It also supplies exactly the property this map needs -- the pair is injective and
// cannot produce zero, because the high word is at least one -- so a published frame is never
// mistaken for an absent one.

// True when `sid` should be expanded.
//
// `dedup` selects the exploration semantics. True: only the first state of each canonical hash
// enters the frontier (explore_from_canonical_states_only). False: every new state is explored,
// so `map` is not consulted at all -- deduplicating against it anyway would silently drop
// states whose hashes collided.
//
// `explore_threshold_u32` is the stochastic-pruning coin, encoded as a probability scaled to
// uint32: 0xFFFFFFFF means always explore and skips the draw entirely.
//
// Defined here rather than in a .cu so every translation unit that asks the question links to
// THIS body -- a device function defined in one .cu is not reachable from another target's
// device link, and the answer to "does a second copy appear" must not depend on that.
// THE TRANSITION'S IDENTITY, as the host computes it: the input state's canonical hash, the
// rule, and the consumed edges' canonical ranks WITHIN that state, in pattern order. Two engines
// reaching the same transition must produce the same value or nothing keyed on it agrees.
//
// ONE body, because two device call sites need it -- the rate draw in apply_one_match and the
// per-(state, rule) cap in match_state_rule -- and a key spelled twice is a key that will drift.
// __noinline__ ON PURPOSE. This is called from inside the join's innermost completion
// callback, which is instantiated through a template per rule shape; inlined there it
// carried event_signature's whole body into the DFS and ptxas ran out of memory
// assembling match.cu. One call per completed match is not the cost that matters here.
__device__ inline __noinline__ uint64_t transition_key_device(const DeviceState& ds, StateId state_id,
                                                 RuleId rule_id, const EdgeId* matched_edges,
                                                 uint8_t num_edges) {
    uint32_t ranks[kMaxPatternEdges];
    uint8_t n = 0;
    for (uint8_t i = 0; i < num_edges && n < kMaxPatternEdges; ++i) {
        if (matched_edges[i] == INVALID_ID) continue;
        const uint32_t pos = state_edge_index(ds, state_id, matched_edges[i]);
        ranks[n++] = (pos == UINT32_MAX) ? UINT32_MAX : ds.state_edge_rank[pos];
    }
    return hgcommon::event_signature(hgcommon::EVENT_SIG_TRANSITION,
                                     ds.state_canonical_hash[state_id],
                                     /*output_state_hash=*/0, /*step=*/0, rule_id,
                                     ranks, n, /*produced_ranks=*/nullptr, 0);
}

__device__ inline bool state_survives_dedup(DeviceState ds, StateId sid, uint64_t hash,
                                            DedupMap::DeviceView map, bool dedup,
                                            uint32_t explore_threshold_u32,
                                            uint64_t explore_seed, uint32_t step,
                                            StateId parent_sid = INVALID_ID) {
    if (dedup) {
        if (hash == 0) {
            // Not a hash: keep the state rather than merge every uncomputed one into one slot.
            ds.errors.record(ErrorKind::kUncomputedStateHash);
            return true;
        }
        auto r = map.insert_if_absent(hash, sid);
        // A FULL MAP IS NOT A DUPLICATE. Exhaustion returns inserted=false like a genuine hit, so
        // without this test an overfull dedup map drops every new state silently and the run
        // reports a smaller state set as if it were the answer. Keeping the state instead errs
        // toward a duplicate, which is visible and correctable, over a loss, which is neither --
        // and the recorded error puts the run under the engine's partial-result contract.
        if (r.overflowed) {
            ds.errors.record(ErrorKind::kCanonicalMapFull);
            return true;
        }
        if (!r.inserted) return false;
    }

    // Stochastic-exploration coin flip. UINT32_MAX == "always explore"
    // (the threshold encoding for probability 1.0); skip the hash work
    // entirely on that fast path so the existing all-deterministic
    // workloads pay zero overhead.
    if (explore_threshold_u32 != 0xFFFFFFFFu) {
        if (explore_threshold_u32 == 0u) return false;  // probability 0.0
        uint64_t mix = hgcommon::splitmix64(explore_seed
                                  ^ (static_cast<uint64_t>(step) << 32)
                                  ^ static_cast<uint64_t>(sid));
        uint32_t draw = static_cast<uint32_t>(mix);
        if (draw >= explore_threshold_u32) return false;
    }

    // THE TWO HARD BOUNDS, applied AFTER dedup for the reason the host applies them there: the
    // cap counts states RETAINED, and a state that merged into an existing one was never
    // retained, so charging it would make the bound depend on how many duplicates arrived.
    //
    // Both admit the first k to arrive. Which k those are is not reproducible -- the same
    // statement the host's documentation makes about these two options, and the reason
    // "MatchesPerStateRule" exists for a caller who needs the kept set to be stable.
    if (ds.max_states_per_step != 0u && step < ds.max_states_per_step_slots) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> c(ds.states_per_step[step]);
        if (c.fetch_add(1u, cuda::memory_order_relaxed) >= ds.max_states_per_step) return false;
    }
    if (ds.max_successor_states_per_parent != 0u && parent_sid < ds.max_states) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> c(
            ds.successors_per_parent[parent_sid]);
        if (c.fetch_add(1u, cuda::memory_order_relaxed) >= ds.max_successor_states_per_parent)
            return false;
    }
    return true;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE