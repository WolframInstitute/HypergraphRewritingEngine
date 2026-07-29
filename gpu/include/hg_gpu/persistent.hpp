#pragma once
// Device-resident scheduling: workers that PULL work from a queue, instead of being launched
// once per phase per step. See docs/GPU_PERSISTENT_DESIGN.md.
//
// Stage 1, which is what exists here: the MATCH role only, over a queue seeded once. It buys
// no speed and is not on the shipping path -- Engine::Impl still runs the level-synchronous
// loop. What it buys is that the risky parts (queue mechanics, one block claiming one item,
// workers that decide for themselves when to stop) are exercised against a kernel whose
// output is already trusted, before any of it carries semantics.
//
// TerminationDetector is deliberately NOT used yet. Its stable-observation window exists to
// stop a worker exiting during a lull, when in-flight items have finished but not yet emitted
// their follow-ups. A queue seeded once and never grown has no lull: empty means finished, so
// wiring the detector here would exercise none of what it is for. It wires in at stage 2, when
// the roles start feeding each other.

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/rewrite.hpp"
#include "hg_gpu/ring_buffer.hpp"
#include "hg_gpu/termination.hpp"
#include "hg_gpu/types.hpp"

#include <vector>

namespace hg_gpu {

// One unit of matching work: a (state, rule) pair, which is the granularity
// match_state_rule already wants -- one block, its threads striping the depth-0 candidates.
//
// `step` rides on the ITEM. That is what makes a step budget a predicate rather than a loop
// bound, and it is already how the host carries depth (ExpandChunk::step): with no phases there
// is nothing else that could hold it.
struct MatchWorkItem {
    StateId  state_id = INVALID_ID;
    uint32_t rule_id  = 0;
    uint32_t step     = 0;
};

// Match every (state, rule) pair through persistent workers rather than one block per pair.
// Returns the number of matches in `out`, which must equal what run_match_kernel_batch
// produces for the same inputs -- that equality is the point of the stage.
uint32_t run_persistent_match(const EngineState& engine,
                              const std::vector<DeviceRule>& rules,
                              const std::vector<StateId>& states,
                              Pool<MatchRecord>& out,
                              uint32_t blocks = 0);

// Stage 2: MATCH and REWRITE as two roles feeding each other, with no barrier between them.
// A match found by one worker is rewritten by another as soon as it is picked up, rather than
// after every match in the step has been found.
//
// This is where the two hazards the design names actually bite, and both are handled here
// rather than left to be discovered:
//
//   PREMATURE EXIT. A rewrite worker finding its queue empty cannot conclude it is finished --
//   match workers may still be producing. That is exactly what TerminationDetector's stable
//   observation window is for, and stage 2 is the first point at which it earns its keep.
//
// The queue between the roles is the match POOL plus a consume cursor, not a second ring.
// A match's slot is assigned by match_state_rule, whose contract is shared with the
// level-synchronous scheduler and must not change; and since blocks match concurrently, no
// block can say which pool slots are its own -- a before/after counter delta is not
// attributable to one block. Consumers claim indices instead, which sidesteps that entirely.
//
// The remaining hazard, a FULL pool, is the pre-existing capacity-overflow path
// (ErrorKind::kMatchPoolFull) and keeps its existing behaviour: record and return partial
// work, never throw. The "must not block on full" rule still binds and is satisfied trivially
// here, because nothing blocks -- a worker that finds nothing to claim loops and re-checks.
struct PersistentRunStats {
    uint32_t matches_found = 0;
};

PersistentRunStats run_persistent_match_rewrite(EngineState& engine,
                                                const std::vector<DeviceRule>& rules,
                                                const std::vector<StateId>& states,
                                                uint32_t step,
                                                Pool<MatchRecord>& scratch_matches,
                                                uint32_t blocks = 0);

}  // namespace hg_gpu
