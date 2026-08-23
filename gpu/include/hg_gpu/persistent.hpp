#pragma once
#include "hgcommon/namespace.hpp"
// Device-resident scheduling: workers that PULL work from a queue, instead of being launched
// once per phase per step. See gpu/ARCHITECTURE.md sec 2.
//
// Three entry points, each adding one thing to the one before, so a failure lands in the stage
// that introduced it rather than in the whole model at once. run_persistent_evolve is the
// shipping one; the two below it are the stages it is built from, kept as gates.
//
//   run_persistent_match          the MATCH role alone, over a queue seeded once. Empty means
//                                 finished, because nothing can push after the seed.
//   run_persistent_match_rewrite  match and rewrite as two roles feeding each other, so empty
//                                 no longer means finished and TerminationDetector's stable
//                                 observation window starts earning its keep.
//   run_persistent_evolve         the loop closes: output states are hashed, deduplicated and
//                                 re-enqueued on device, so a whole evolution is one launch.

#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/exploration.hpp"
#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/quotient_causal.hpp"
#include "hg_gpu/quotient_expansion.hpp"
#include "hg_gpu/rewrite.hpp"
#include "hg_gpu/ring_buffer.hpp"
#include "hg_gpu/termination.hpp"
#include "hg_gpu/types.hpp"

#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

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
// Returns the number of matches in `out`, which must equal what the batch driver
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
// A match's slot is assigned by match_state_rule, whose contract must not change; and
// since blocks match concurrently, no
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

// Stage 3: the loop closes. A rewrite's output state is hashed, tested against the exploration
// rule, and its (state, rule) items pushed back into the same match queue -- so a whole
// evolution runs inside ONE launch, with no host in the loop and no per-step barrier.
//
// Three things only bind once the loop is closed:
//
//   THE STEP BUDGET IS A PREDICATE. `max_steps` bounds the depth carried on the item, not a
//   number of kernel launches. An item at the budget is rewritten and its children are not
//   re-enqueued.
//
//   A FULL QUEUE MUST NOT BLOCK. The producers here are the same workers that consume, so a
//   worker waiting for room would wait for itself. On a failed push the pusher runs the match
//   INLINE -- the host's rule in job_system.hpp, and terminating for the same reason: matching
//   writes to the match pool, never back into this ring.
//
//   HASHING HAS NO BATCH TO MEASURE. States arrive continuously, so the IR scratch is claimed
//   per state from a device arena sized from that state's own counts
//   (state_exact_hash_device). Arena exhaustion is a capacity overflow: recorded, partial work
//   returned, never a coarser hash that would merge non-isomorphic states.
// Blocks a persistent kernel launches when the caller does not choose: eight per SM.
//
// A persistent kernel's blocks do not retire and get replaced -- they live for the whole
// evolution -- so the grid IS the worker count. Exposed because the IR arena has to be sized
// from the same number: each resident worker holds its own slot, so arena demand scales with the
// grid, and a caller sizing the arena off the state budget alone starves it as soon as the grid
// grows. See gpu/src/persistent.cu for the sweep behind eight-per-SM.
uint32_t default_persistent_grid();

// Words of IR arena to provide for `holders` concurrent slot holders at `share_words` average
// words each (EngineConfig::ir_arena_share_words). The arena is one shared bump pool; a holder
// keeps one slot at a time and grows it to the largest state it personally canonicalizes, so
// the AVERAGE share is what matters, not a per-holder partition. A workload that outgrows the
// pool records kIRArenaExhausted, which grow-and-retry answers by doubling the share.
uint64_t persistent_arena_words(uint32_t share_words, uint32_t holders);

// A CONTINUABLE evolution's state, owned by the caller across calls.
//
// WHY THIS EXISTS RATHER THAN "just call it again". The state and event POOLS already live in
// EngineState and survive a call, so a second call looks like it should extend. Two designs that
// relied on that were measured and both failed, in opposite directions (rule 2-edge -> 3-edge,
// depth 2 then 3, against 7 states / 6 events for one run at depth 3):
//
//   maps rebuilt per call     10 states, 9 events -- nothing was recognised, so the second call
//                             re-derived every state it already had as new.
//   maps owned by the caller   5 states, 4 events -- everything was recognised, so nothing was
//                             re-expanded and the new depth was never reached.
//
// The cause is that ONE bit answers two questions: the dedup map says "seen", and the worker
// reads that same answer as "already expanded". Sharing it across calls loses recognition or
// loses expansion. So a session carries both the maps AND a FRONTIER: the states whose expansion
// was refused by the STEP BUDGET rather than by dedup. That is the same separation the host makes
// with defer_match_task/evolve_more, and the device twin of it.
struct SessionView {
    DedupMap::DeviceView states;
    DedupMap::DeviceView events;
    // Boundary states, appended when the budget refuses expansion. `count` may exceed `cap`
    // under a race; readers take min(count, cap), which is the same discipline the counted root
    // seeder uses.
    StateId*  frontier      = nullptr;
    uint32_t* frontier_count = nullptr;
    uint32_t  frontier_cap  = 0;
    uint32_t  enabled       = 0;
};

// Host owner of the above. Allocated once for the session, never shrunk, so a Step costs no
// device allocation.
class SessionState {
public:
    SessionState(uint32_t max_states, uint32_t max_events);
    ~SessionState();
    SessionState(const SessionState&)            = delete;
    SessionState& operator=(const SessionState&) = delete;

    uint32_t frontier_size() const;

    SessionView view();

private:
    DedupMap  states_;
    DedupMap  events_;
    StateId*  frontier_ = nullptr;
    uint32_t* count_    = nullptr;
    uint32_t  cap_      = 0;
};

struct PersistentEvolveStats {
    uint32_t matches_found = 0;
    uint32_t states_after  = 0;
    uint64_t arena_words_used = 0;
    // Events that won their signature slot. 0 under EventSignatureKeys None, where no signature
    // is computed and the raw application count is the answer.
    uint32_t canonical_events = 0;
    // Where the worker blocks' time went, as clock64() deltas measured by each block's thread 0
    // and summed across blocks at exit. SM clocks tick independently, so these attribute as
    // FRACTIONS of their sum, not as wall time. canon covers the exact-hash/dedup/stamp stretch
    // of the rewrite branch; wait is the await_match spin on a claimed-but-unpublished record;
    // idle covers failed-pop iterations including their backoff sleep.
    uint64_t cycles_match   = 0;
    uint64_t cycles_rewrite = 0;
    uint64_t cycles_canon   = 0;
    uint64_t cycles_idle    = 0;
    uint64_t cycles_wait    = 0;
    // apply_one_match's six sub-stretches of cycles_rewrite, in rewrite.hpp's order:
    // reserve, emit, csr, event, causal, branchial.
    uint64_t cycles_rw_sub[6] = {0, 0, 0, 0, 0, 0};
};

PersistentEvolveStats run_persistent_evolve(EngineState& engine,
                                            const std::vector<DeviceRule>& rules,
                                            const std::vector<StateId>& roots,
                                            uint32_t max_steps,
                                            Pool<MatchRecord>& scratch_matches,
                                            DeviceArena& arena,
                                            bool dedup,
                                            uint32_t explore_threshold_u32 = 0xFFFFFFFFu,
                                            uint64_t explore_seed = 0,
                                            // How states are identified. The device twin of
                                            // compute_state_dedup_keys: the host seeding and
                                            // the device loop deduplicating different
                                            // equivalences is not a performance difference, it
                                            // is a different evolution.
                                            CanonicalizationMode state_mode =
                                                CanonicalizationMode::Full,
                                            // Which components the event identity is built
                                            // from, and it is built from the EXACT hashes
                                            // whatever state_mode is. EVENT_SIG_AUTOMATIC also
                                            // keys on canonical edge ranks, which ride the same
                                            // individualization pass as the exact hash.
                                            EventSignatureKeys event_keys = EVENT_SIG_NONE,
                                            uint32_t blocks = 0,
                                            // Collapse isomorphic ROOTS to one entry point.
                                            // Default false is the reference semantics: provided
                                            // roots are distinct entry points even when
                                            // isomorphic. This must agree with what
                                            // k_seed_roots does for the host-seeded roots, or
                                            // the option changes the state set depending on
                                            // which path seeded them.
                                            bool quotient_roots = false,
                                            // The quotient-causal DP's device structures
                                            // (quotient_causal.hpp), constructed by the
                                            // caller, so the host seeding and the device loop
                                            // drive one body of state. Disabled (enabled == 0)
                                            // when null.
                                            const QcView* qc = nullptr,
                                            const QeView* qe = nullptr,
                                            // Null for a one-shot run, which is unchanged.
                                            // Non-null makes the run CONTINUABLE: identity is
                                            // remembered across calls and the budget's boundary
                                            // states are recorded so the next call can expand
                                            // them.
                                            SessionView* session = nullptr,
                                            // 0 seeds from `roots` at depth 0. Non-zero seeds
                                            // from the session's FRONTIER at that depth, which
                                            // is what continues an evolution rather than
                                            // restarting it.
                                            uint32_t start_step = 0);

}  // namespace gpu
}  // namespace HG_NAMESPACE