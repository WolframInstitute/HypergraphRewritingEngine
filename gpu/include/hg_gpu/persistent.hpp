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
#include "hg_gpu/ring_buffer.hpp"
#include "hg_gpu/types.hpp"

#include <vector>

namespace hg_gpu {

// One unit of matching work: a (state, rule) pair, which is the granularity
// match_state_rule already wants -- one block, its threads striping the depth-0 candidates.
struct MatchWorkItem {
    StateId  state_id = INVALID_ID;
    uint32_t rule_id  = 0;
};

// Match every (state, rule) pair through persistent workers rather than one block per pair.
// Returns the number of matches in `out`, which must equal what run_match_kernel_batch
// produces for the same inputs -- that equality is the point of the stage.
uint32_t run_persistent_match(const EngineState& engine,
                              const std::vector<DeviceRule>& rules,
                              const std::vector<StateId>& states,
                              Pool<MatchRecord>& out,
                              uint32_t blocks = 0);

}  // namespace hg_gpu
