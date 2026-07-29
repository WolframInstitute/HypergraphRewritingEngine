// Device-resident scheduling: workers that pull work from a queue rather than being launched
// once per phase per step. See gpu/include/hg_gpu/persistent.hpp and
// docs/GPU_PERSISTENT_DESIGN.md.
//
// Its own translation unit, not appended to match.cu, and that is a memory decision rather
// than a stylistic one: match.cu already costs several GB to compile, and adding one more
// kernel to it took a single nvcc to 8 GB. This machine is shared, so a translation unit that
// cannot be compiled within a safe ceiling is a defect whether or not it links.

#include "hg_gpu/event_identity.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/wl_hash.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hg_gpu {
namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::persistent ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

// Seed the queue on the device, so the ring's cursors and slot states are only ever touched
// through its own device API rather than by a host write assuming its layout.
__global__ void k_seed_match_queue(typename RingBuffer<MatchWorkItem>::DeviceView queue,
                                   const StateId* states, uint32_t num_states,
                                   uint32_t num_rules, uint32_t step) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states * num_rules) return;
    MatchWorkItem item;
    item.state_id = states[tid / num_rules];
    item.rule_id  = tid - (tid / num_rules) * num_rules;
    item.step     = step;
    queue.try_push(item);   // capacity >= item count, so this cannot fail here
}

// The key this run identifies states BY -- the device twin of compute_state_dedup_keys, and it
// must stay the twin: the two schedulers deduplicating different equivalences is not a
// performance difference, it is a different evolution.
//
//   None       a per-state unique value, so nothing ever deduplicates. Costs no hashing at all.
//   Automatic  the content-ordered hash. Cheap, and deliberately NOT isomorphism-invariant.
//   Full       the exact isomorphism hash, which is the expensive one.
//
// Only the Full arm touches the arena, so a run in the other two modes never claims IR scratch.
// `want_ranks` is passed through to the Full arm so that when the run also needs per-edge ranks
// the single pass produces both, rather than the key here and the ranks in a repeat pass.
__device__ ExactHashStatus state_key_device(DeviceState ds, StateId sid,
                                            CanonicalizationMode mode,
                                            DeviceArena::View arena,
                                            uint32_t*& slot, uint64_t& slot_words,
                                            uint64_t& out_key, bool want_ranks) {
    switch (mode) {
        case CanonicalizationMode::None:
            // Mirrors k_fill_unique_keys: distinct per state, and offset so it can never be the
            // dedup map's EMPTY sentinel.
            out_key = static_cast<uint64_t>(sid) + 1ull;
            return ExactHashStatus::kOk;
        case CanonicalizationMode::Automatic:
            out_key = content_hash_state_device(ds, sid);
            return ExactHashStatus::kOk;
        case CanonicalizationMode::Full:
        default:
            return state_exact_hash_device(ds, sid, arena, slot, slot_words, out_key,
                                           want_ranks);
    }
}

// Insert every root's canonical hash into the map before the loop starts, so a child isomorphic
// to a root deduplicates against it rather than being explored a second time. Runs pre-launch,
// which the no-host-in-the-loop constraint permits: the constraint is on evolution, not on
// seeding. Mirrors what k_seed_roots does for the level-synchronous scheduler.
//
// Every root is enqueued regardless of the result, which is the reference semantics: provided
// roots are distinct entry points even when isomorphic.
__global__ void k_seed_root_hashes(DeviceState ds, const StateId* roots, uint32_t num_roots,
                                   DedupMap::DeviceView map, CanonicalizationMode state_mode,
                                   bool need_exact, bool need_ranks, DeviceArena::View arena) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_roots) return;
    const StateId sid = roots[tid];
    uint32_t* slot = nullptr;
    uint64_t  slot_words = 0;

    uint64_t key = 0;
    {
        const ExactHashStatus st =
            state_key_device(ds, sid, state_mode, arena, slot, slot_words, key, need_ranks);
        if (st != ExactHashStatus::kOk) {
            ds.errors.record(error_kind_for(st));
            return;
        }
    }
    ds.state_canonical_hash[sid] = key;

    // The exact hash is a SECOND quantity, and only in Full mode is it the same one. Computed
    // here only if an event identity will read it; under event mode None nobody does.
    if (need_exact) {
        uint64_t exact = key;
        if (state_mode != CanonicalizationMode::Full) {
            const ExactHashStatus st =
                state_exact_hash_device(ds, sid, arena, slot, slot_words, exact, need_ranks);
            if (st != ExactHashStatus::kOk) {
                ds.errors.record(error_kind_for(st));
                return;
            }
        }
        ds.state_exact_hash[sid] = exact;
    }

    map.insert_if_absent(key == 0 ? 1 : key, sid);
}

// Records a claiming consumer may safely read. The pool's counter counts CLAIMS, and a claim
// past the end returns kInvalid without writing, so the counter can exceed the capacity while
// only the first `capacity` slots hold anything. Reading up to the raw counter would read past
// the allocation.
__device__ __forceinline__ uint32_t readable_records(
        const typename Pool<MatchRecord>::DeviceView& found) {
    const uint32_t claimed = *found.counter;
    return claimed < found.capacity ? claimed : found.capacity;
}

// Spin budgets. Nothing in a correct run reaches them -- the detector fires and the workers
// leave -- so hitting one means a DEFECT, and that is exactly why they exist.
//
// A device-resident scheduler has no host in the loop to notice it is stuck, and a kernel that
// never returns holds the device until the context is destroyed. On a machine whose GPU also
// drives the display, that is not a slow run, it is a lost session. So a stall costs a warning
// and a partial result, which is already this project's contract for anything it cannot
// complete, rather than the machine.
//
// Sized far past any real run: the detector's rounds are ~2 us apart, so ~20 s of quiescence
// checking, and a worker's idle spins are cheaper still. Both are ceilings on PATHOLOGY, not
// tuning parameters, and neither should ever be reached often enough to be worth tuning.
constexpr uint32_t kMaxDetectorRounds = 10u * 1000u * 1000u;
constexpr uint32_t kMaxWorkerIdleSpins = 20u * 1000u * 1000u;

// Reserve the next unconsumed record index, or INVALID_ID when there is none yet.
//
// The reservation is a CAS rather than an unconditional bump, because the cursor is shared and
// a bump has nothing to undo with: a block that bumped past the end and then subtracted can
// have its subtraction cancel a DIFFERENT block's successful claim, which both hands the same
// record to two blocks and strands the one in between. A stranded record is never rewritten,
// so `rewrites_done` never reaches the record count and the run does not terminate.
__device__ __forceinline__ uint32_t claim_next_record(
        uint32_t* cursor, const typename Pool<MatchRecord>::DeviceView& found) {
    uint32_t cur = *cursor;
    for (;;) {
        if (cur >= readable_records(found)) return INVALID_ID;
        const uint32_t prev = atomicCAS(cursor, cur, cur + 1u);
        if (prev == cur) return cur;
        cur = prev;
    }
}

// ---- stage 1: the match role alone ------------------------------------------------------
//
// One block per popped item -- the shape match_state_rule already wants. Only thread 0 touches
// the queue, so a pop is one claim per block rather than a race between its threads.
//
// Exit when the queue is empty. That is exact for a queue seeded once and never grown: no work
// can appear after a failed pop. It is NOT the rule stage 2 uses.
__global__ void k_persistent_match(DeviceState ds,
                                   const DeviceRule* rules,
                                   typename RingBuffer<MatchWorkItem>::DeviceView queue,
                                   typename Pool<MatchRecord>::DeviceView out) {
    __shared__ MatchWorkItem item;
    __shared__ bool have;

    for (;;) {
        if (threadIdx.x == 0) have = queue.try_pop(item);
        __syncthreads();
        if (!have) return;

        match_state_rule(ds, rules, item.state_id, item.rule_id, item.step, out);
        __syncthreads();
    }
}

// ---- stage 2: match and rewrite as two roles --------------------------------------------
//
// The match POOL is the queue between them. Matches appear in it as they are found, and a
// rewrite worker claims the next unconsumed index the moment it exists -- there is no barrier
// between finding a match and applying it, which is the whole point.
//
// A cursor rather than a second RingBuffer because a match's slot in the pool is assigned by
// match_state_rule, whose contract is shared with the level-synchronous scheduler and must not
// change. Blocks match concurrently, so no block can say which pool slots are its own: a
// before/after counter delta is not attributable to one block. The cursor sidesteps that
// entirely -- consumers claim indices, not ranges.
constexpr uint32_t kRoleMatch = 0;

__global__ void k_persistent_match_rewrite(
        DeviceState ds,
        const DeviceRule* rules,
        typename RingBuffer<MatchWorkItem>::DeviceView match_q,
        typename Pool<MatchRecord>::DeviceView found,
        uint32_t* consume_cursor,
        typename TerminationDetector::DeviceView term,
        uint32_t step) {

    if (blockIdx.x == 0) {
        // Detector. Only thread 0 observes; the rest of the block idles, which costs one block
        // of occupancy and buys a termination test that cannot race with its own workers.
        if (threadIdx.x != 0) return;
        uint64_t pushed[TerminationDetector::kMaxRoles];
        uint64_t completed[TerminationDetector::kMaxRoles];
        for (uint32_t round = 0; ; ++round) {
            if (round >= kMaxDetectorRounds) {
                ds.errors.record(ErrorKind::kPersistentStall);
                term.signal_exit();
                return;
            }
            // Finished means BOTH: every seeded match item accounted for, and every match that
            // matching produced already consumed. Checking only the match role would exit with
            // rewrites outstanding; checking only the cursor would exit before matching had
            // produced anything at all.
            const bool matches_done = term.snapshot_quiescent(pushed, completed);
            const uint32_t produced = readable_records(found);
            const uint32_t consumed = *consume_cursor;
            if (matches_done && consumed >= produced) {
                // Quiescent once is not enough: an in-flight match may have just completed
                // without its matches yet being visible. Look again after a backoff, and only
                // signal when it held across both observations.
                __nanosleep(4000);
                if (term.snapshot_quiescent(pushed, completed) &&
                    *consume_cursor >= readable_records(found)) {
                    term.signal_exit();
                    return;
                }
            }
            __nanosleep(2000);
        }
    }

    __shared__ MatchWorkItem mitem;
    __shared__ bool have;
    __shared__ uint32_t claimed;

    for (;;) {
        // Rewrite first: it drains what matching produced, and letting the pool run ahead
        // unboundedly is what makes it overflow.
        if (threadIdx.x == 0) claimed = claim_next_record(consume_cursor, found);
        __syncthreads();
        if (claimed != INVALID_ID) {
            if (threadIdx.x == 0) {
                const MatchRecord& rec = found.at(claimed);
                await_match(rec);
                // rec.step + 1, not rec.step: an event is stamped with the depth of the state
                // it PRODUCES, which is what the level-synchronous loop writes
                // (run_rewrite_kernel_with_nosync is called with step + 1) and what the CPU
                // uses (the canonical OUTPUT state's step). Writing the parent's depth here
                // made every event's reported step differ between the two schedulers.
                (void)apply_one_match(ds, rules, rec, rec.step + 1u);
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) have = match_q.try_pop(mitem);
        __syncthreads();
        if (have) {
            match_state_rule(ds, rules, mitem.state_id, mitem.rule_id, mitem.step, found);
            __syncthreads();
            if (threadIdx.x == 0) term.mark_completed(kRoleMatch);
            __syncthreads();
            continue;
        }

        // Nothing available in either role. Empty does NOT mean finished here -- the other
        // role may still be producing -- so only the detector decides.
        if (term.exit_requested()) return;
        __syncthreads();
    }
}

// ---- stage 3: the loop closes ------------------------------------------------------------
//
// A rewrite's output state is hashed, tested against the exploration rule, and its (state,
// rule) items pushed back into the same match queue. A whole evolution then runs inside one
// launch: no step loop, no host in the middle, no barrier between depths.
//
// Termination cannot be "queue empty" any more, and cannot be a single quiescent snapshot
// either. The exact condition is that NOTHING MADE PROGRESS across an observation window while
// both roles read as drained, so the detector compares a four-counter snapshot against itself:
//
//   pushed[match] / completed[match]   a match item exists but has not finished
//   readable records / rewrites_done   a match record exists but has not been rewritten
//
// Each counter is monotone, and a worker cannot start and finish inside the window without
// moving one of them. So equality of all four across the window, plus both drained conditions,
// means quiescent -- where either condition alone, or either snapshot alone, does not.
//
// Ordering the workers must keep, and the reason:
//   mark_pushed BEFORE try_push        an item is never visible while uncounted
//   rewrites_done LAST                 a rewrite that will still push, or is still running an
//                                      item inline, reads as unfinished
__global__ void k_persistent_evolve(
        DeviceState ds,
        const DeviceRule* rules,
        uint32_t num_rules,
        typename RingBuffer<MatchWorkItem>::DeviceView match_q,
        typename Pool<MatchRecord>::DeviceView found,
        uint32_t* consume_cursor,
        uint32_t* rewrites_done,
        DedupMap::DeviceView dedup_map,
        bool dedup,
        uint32_t explore_threshold_u32,
        uint64_t explore_seed,
        uint32_t max_steps,
        CanonicalizationMode state_mode,
        EventSignatureKeys event_keys,
        DedupMap::DeviceView event_map,
        DeviceArena::View arena,
        typename TerminationDetector::DeviceView term) {

    const bool need_ranks = event_keys_need_ranks(event_keys);

    if (blockIdx.x == 0) {
        if (threadIdx.x != 0) return;
        uint64_t p1[TerminationDetector::kMaxRoles], c1[TerminationDetector::kMaxRoles];
        uint64_t p2[TerminationDetector::kMaxRoles], c2[TerminationDetector::kMaxRoles];
        for (uint32_t round = 0; ; ++round) {
            if (round >= kMaxDetectorRounds) {
                // Quiescence never held. Signal exit anyway so the workers leave and the
                // launch returns: a recorded defect with partial work beats holding the device.
                ds.errors.record(ErrorKind::kPersistentStall);
                term.signal_exit();
                return;
            }
            const bool q1  = term.snapshot_quiescent(p1, c1);
            const uint32_t prod1 = readable_records(found);
            const uint32_t done1 = *rewrites_done;
            if (q1 && done1 >= prod1) {
                __nanosleep(4000);
                const bool q2 = term.snapshot_quiescent(p2, c2);
                const uint32_t prod2 = readable_records(found);
                const uint32_t done2 = *rewrites_done;
                bool unchanged = (prod1 == prod2) && (done1 == done2);
                for (uint32_t r = 0; r < term.num_roles && unchanged; ++r)
                    unchanged = (p1[r] == p2[r]) && (c1[r] == c2[r]);
                if (q2 && done2 >= prod2 && unchanged) {
                    term.signal_exit();
                    return;
                }
            }
            __nanosleep(2000);
        }
    }

    // Per-block IR scratch, carried across items: claimed on first use and re-claimed only
    // when a larger state arrives. Thread 0 does the hashing, so the slot is its own.
    __shared__ uint32_t* ir_slot;
    __shared__ uint64_t  ir_slot_words;
    __shared__ MatchWorkItem mitem;
    __shared__ bool     have;
    __shared__ uint32_t claimed;
    __shared__ uint32_t child_sid;
    __shared__ uint32_t child_event;
    __shared__ uint32_t child_step;
    __shared__ bool     expand_child;
    __shared__ bool     run_rule_inline;
    __shared__ bool     stalled;
    uint32_t idle_spins = 0;

    if (threadIdx.x == 0) { ir_slot = nullptr; ir_slot_words = 0; stalled = false; }
    __syncthreads();

    for (;;) {
        // Rewrite first: it drains what matching produced, and letting the pool run ahead
        // unboundedly is what makes it overflow.
        if (threadIdx.x == 0) claimed = claim_next_record(consume_cursor, found);
        __syncthreads();

        if (claimed != INVALID_ID) {
            if (threadIdx.x == 0) {
                const MatchRecord& rec = found.at(claimed);
                await_match(rec);
                const uint32_t step = rec.step;
                // The event carries the depth of the state it PRODUCES -- see the note in
                // k_persistent_match_rewrite. The exploration depth below is the same value.
                const AppliedMatch applied = apply_one_match(ds, rules, rec, step + 1u);
                child_sid    = applied.state;
                child_event  = applied.event;
                child_step   = step + 1u;
                expand_child = false;

                // Expand the child only if it exists, the step budget allows it, its exact
                // hash is computable, and the exploration rule keeps it. The hash is the
                // dedup KEY, so a state whose hash could not be computed is not enqueued
                // under a coarser one -- 1-WL merges non-isomorphic states.
                if (child_sid != INVALID_ID) {
                    uint64_t h = 0;
                    const ExactHashStatus key_st =
                        state_key_device(ds, child_sid, state_mode, arena, ir_slot,
                                         ir_slot_words, h, need_ranks);
                    if (key_st != ExactHashStatus::kOk) {
                        ds.errors.record(error_kind_for(key_st));
                    } else {
                        // Publish before anything reads it: a transition OUT of this state
                        // needs it as an input hash, and that read happens on another block.
                        ds.state_canonical_hash[child_sid] = h;

                        // The exact isomorphism hash is a different question from the mode's
                        // key and coincides with it only in Full. Computed only when an event
                        // identity will read it -- otherwise this is an
                        // individualization-refinement pass per state bought for nobody.
                        uint64_t exact = h;
                        if (event_keys != EVENT_SIG_NONE) {
                            if (state_mode != CanonicalizationMode::Full) {
                                const ExactHashStatus ex_st =
                                    state_exact_hash_device(ds, child_sid, arena, ir_slot,
                                                            ir_slot_words, exact, need_ranks);
                                if (ex_st != ExactHashStatus::kOk) {
                                    ds.errors.record(error_kind_for(ex_st));
                                    exact = 0;
                                }
                            }
                            ds.state_exact_hash[child_sid] = exact;
                        }

                        // The event identity, at the only point where both halves exist: the
                        // input hash, published when the parent was created, and the output
                        // hash just computed. The rewrite wrote this event BEFORE its output
                        // state was canonicalized, which is precisely why a scheduler with a
                        // phase boundary between rewriting and hashing cannot fill it in --
                        // and why the persistent one can.
                        // Built from the EXACT hashes, never the mode's key: event identity is
                        // defined over isomorphism classes independently of how states are
                        // being identified (SPEC.md sec 4). Keying it off the mode's hash is
                        // the defect b82049f fixed on the host.
                        if (event_keys != EVENT_SIG_NONE && child_event != INVALID_ID) {
                            stamp_event_signature(ds, child_event, event_keys,
                                                  ds.state_exact_hash[rec.state_id], exact,
                                                  rec.state_id, child_sid, step + 1u,
                                                  rec.rule_id, event_map);
                        }

                        expand_child = child_step < max_steps &&
                                       state_survives_dedup(child_sid, h, dedup_map, dedup,
                                                            explore_threshold_u32,
                                                            explore_seed, child_step);
                    }
                }
            }
            __syncthreads();

            for (uint32_t r = 0; expand_child && r < num_rules; ++r) {
                if (threadIdx.x == 0) {
                    MatchWorkItem it;
                    it.state_id = child_sid;
                    it.rule_id  = r;
                    it.step     = child_step;
                    term.mark_pushed(kRoleMatch);
                    run_rule_inline = !match_q.try_push(it);
                    if (run_rule_inline) {
                        // Full queue. The producers here are the same workers that consume,
                        // so waiting for room would be waiting on ourselves -- job_system.hpp
                        // solves it the same way, by running the item on the pusher. It
                        // terminates because matching only writes to the match pool, never
                        // back into this ring.
                        //
                        // The item never entered the queue, so its completion is booked here
                        // and the block runs it below; leaving it counted as outstanding
                        // would stall termination forever.
                        term.mark_completed(kRoleMatch);
                    }
                }
                __syncthreads();
                if (run_rule_inline)
                    match_state_rule(ds, rules, child_sid, r, child_step, found);
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(rewrites_done, 1u);
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) have = match_q.try_pop(mitem);
        __syncthreads();
        if (have) {
            match_state_rule(ds, rules, mitem.state_id, mitem.rule_id, mitem.step, found);
            __syncthreads();
            if (threadIdx.x == 0) term.mark_completed(kRoleMatch);
            __syncthreads();
            continue;
        }

        if (term.exit_requested()) return;
        // Idle, and the detector has not released us. Counted, because a worker that can
        // neither find work nor be told to stop is the same defect from the other side.
        if (threadIdx.x == 0 && ++idle_spins >= kMaxWorkerIdleSpins) {
            ds.errors.record(ErrorKind::kPersistentStall);
            stalled = true;
        }
        __syncthreads();
        if (stalled) return;
    }
}

}  // namespace

uint32_t run_persistent_match(const EngineState& engine,
                              const std::vector<DeviceRule>& rules,
                              const std::vector<StateId>& states,
                              Pool<MatchRecord>& out,
                              uint32_t blocks) {
    if (rules.empty() || states.empty()) return out.size_host();

    const uint32_t num_rules = static_cast<uint32_t>(rules.size());
    const uint32_t num_items = static_cast<uint32_t>(num_rules * states.size());

    DeviceRule* d_rules = nullptr;
    check(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    check(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    StateId* d_states = nullptr;
    check(cudaMalloc(&d_states, sizeof(StateId) * states.size()), "states alloc");
    check(cudaMemcpy(d_states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice), "states copy");

    uint32_t cap = 1;
    while (cap < num_items) cap <<= 1;
    RingBuffer<MatchWorkItem> queue(cap);
    queue.clear();

    {
        const uint32_t block = 128;
        const uint32_t grid = (num_items + block - 1) / block;
        k_seed_match_queue<<<grid, block>>>(queue.view(), d_states,
                                            static_cast<uint32_t>(states.size()), num_rules,
                                            /*step=*/0u);
        check(cudaDeviceSynchronize(), "seed sync");
    }

    // Deliberately FEWER blocks than items: each one loops, which is the whole difference from
    // launching one block per item.
    const uint32_t grid = blocks ? blocks : 64;
    k_persistent_match<<<grid, kMatchBlockThreads>>>(engine.device(), d_rules,
                                                     queue.view(), out.view());
    check(cudaDeviceSynchronize(), "persistent match sync");

    cudaFree(d_states);
    cudaFree(d_rules);
    return out.size_host();
}

PersistentRunStats run_persistent_match_rewrite(EngineState& engine,
                                                const std::vector<DeviceRule>& rules,
                                                const std::vector<StateId>& states,
                                                uint32_t step,
                                                Pool<MatchRecord>& scratch_matches,
                                                uint32_t blocks) {
    PersistentRunStats stats;
    if (rules.empty() || states.empty()) return stats;

    // Records are consumed while they are still being produced, so their publication flags
    // must start clear. The scheduler that relies on the flag is the one that clears it.
    scratch_matches.reset_and_clear();

    const uint32_t num_rules = static_cast<uint32_t>(rules.size());
    const uint32_t num_items = static_cast<uint32_t>(num_rules * states.size());

    DeviceRule* d_rules = nullptr;
    check(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    check(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    StateId* d_states = nullptr;
    check(cudaMalloc(&d_states, sizeof(StateId) * states.size()), "states alloc");
    check(cudaMemcpy(d_states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice), "states copy");

    uint32_t cap = 1;
    while (cap < num_items) cap <<= 1;
    RingBuffer<MatchWorkItem> match_q(cap);
    match_q.clear();
    {
        const uint32_t block = 128;
        const uint32_t grid = (num_items + block - 1) / block;
        k_seed_match_queue<<<grid, block>>>(match_q.view(), d_states,
                                            static_cast<uint32_t>(states.size()), num_rules,
                                            step);
        check(cudaDeviceSynchronize(), "seed sync");
    }

    uint32_t* d_cursor = nullptr;
    check(cudaMalloc(&d_cursor, sizeof(uint32_t)), "cursor alloc");
    check(cudaMemset(d_cursor, 0, sizeof(uint32_t)), "cursor clear");

    TerminationDetector term(/*num_roles=*/1);
    term.clear();
    term.mark_pushed_host(kRoleMatch, num_items);

    // Block 0 is the detector, so at least two blocks are needed for any work to happen.
    const uint32_t grid = blocks ? blocks : 33;
    k_persistent_match_rewrite<<<grid < 2 ? 2 : grid, kMatchBlockThreads>>>(
        engine.device(), d_rules, match_q.view(), scratch_matches.view(),
        d_cursor, term.view(), step);
    check(cudaDeviceSynchronize(), "persistent match+rewrite sync");

    stats.matches_found = scratch_matches.size_host();

    cudaFree(d_cursor);
    cudaFree(d_states);
    cudaFree(d_rules);
    return stats;
}

PersistentEvolveStats run_persistent_evolve(EngineState& engine,
                                            const std::vector<DeviceRule>& rules,
                                            const std::vector<StateId>& roots,
                                            uint32_t max_steps,
                                            Pool<MatchRecord>& scratch_matches,
                                            DeviceArena& arena,
                                            bool dedup,
                                            uint32_t explore_threshold_u32,
                                            uint64_t explore_seed,
                                            CanonicalizationMode state_mode,
                                            EventSignatureKeys event_keys,
                                            uint32_t blocks) {
    PersistentEvolveStats stats;
    if (rules.empty() || roots.empty() || max_steps == 0) return stats;

    // Records are consumed while they are still being produced, so their publication flags
    // must start clear. The scheduler that relies on the flag is the one that clears it.
    scratch_matches.reset_and_clear();

    const uint32_t num_rules = static_cast<uint32_t>(rules.size());
    const uint32_t num_seed  = static_cast<uint32_t>(num_rules * roots.size());

    DeviceRule* d_rules = nullptr;
    check(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    check(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    StateId* d_states = nullptr;
    check(cudaMalloc(&d_states, sizeof(StateId) * roots.size()), "states alloc");
    check(cudaMemcpy(d_states, roots.data(), sizeof(StateId) * roots.size(),
                     cudaMemcpyHostToDevice), "states copy");

    // The ring holds work in flight, not the whole evolution: a run that outgrows it does not
    // fail, it runs the excess inline on the pushing block. Sized to the match pool so the
    // inline path is an escape valve rather than the normal case.
    uint32_t cap = 1;
    while (cap < num_seed) cap <<= 1;
    while (cap < scratch_matches.capacity() && cap < (1u << 20)) cap <<= 1;
    RingBuffer<MatchWorkItem> match_q(cap);
    match_q.clear();
    {
        const uint32_t block = 128;
        const uint32_t grid = (num_seed + block - 1) / block;
        k_seed_match_queue<<<grid, block>>>(match_q.view(), d_states,
                                            static_cast<uint32_t>(roots.size()), num_rules,
                                            /*step=*/0u);
        check(cudaDeviceSynchronize(), "seed sync");
    }

    // The canonical map is the dedup key store for the whole run. Sized to the state pool: one
    // entry per state is the worst case, and the map must not fill, because a full map would
    // silently start admitting duplicates.
    DedupMap canonical(engine.config().max_states * 2u);
    canonical.clear();

    // Signature -> first event with it. Sized off the event budget rather than the state one:
    // an evolution has as many applications as it has matches, which is not bounded by its
    // state count. Allocated whatever the mode, since an unused map costs one clear and the
    // alternative is a null view every stamp site has to test.
    DedupMap event_ids(engine.config().max_events * 2u);
    event_ids.clear();
    if (event_keys != EVENT_SIG_NONE) engine.ensure_event_identity();

    arena.reset();
    {
        const uint32_t block = 64;
        const uint32_t n = static_cast<uint32_t>(roots.size());
        k_seed_root_hashes<<<(n + block - 1) / block, block>>>(
            engine.device(), d_states, n, canonical.view(), state_mode,
            event_keys != EVENT_SIG_NONE,
            event_keys_need_ranks(event_keys),
            arena.view());
        check(cudaDeviceSynchronize(), "root hash seed sync");
    }

    uint32_t* d_cursor = nullptr;
    check(cudaMalloc(&d_cursor, sizeof(uint32_t) * 2), "cursor alloc");
    check(cudaMemset(d_cursor, 0, sizeof(uint32_t) * 2), "cursor clear");
    uint32_t* d_rewrites_done = d_cursor + 1;

    TerminationDetector term(/*num_roles=*/1);
    term.clear();
    term.mark_pushed_host(kRoleMatch, num_seed);

    // Block 0 is the detector, so at least two blocks are needed for any work to happen.
    const uint32_t grid = blocks ? blocks : 33;
    k_persistent_evolve<<<grid < 2 ? 2 : grid, kMatchBlockThreads>>>(
        engine.device(), d_rules, num_rules, match_q.view(), scratch_matches.view(),
        d_cursor, d_rewrites_done, canonical.view(), dedup,
        explore_threshold_u32, explore_seed, max_steps, state_mode, event_keys,
        event_ids.view(), arena.view(), term.view());
    check(cudaDeviceSynchronize(), "persistent evolve sync");

    stats.matches_found    = scratch_matches.size_host();
    stats.states_after     = engine.num_states_host();
    stats.arena_words_used = arena.used_words_host();
    stats.canonical_events = engine.canonical_event_count();

    cudaFree(d_cursor);
    cudaFree(d_states);
    cudaFree(d_rules);
    return stats;
}

}  // namespace hg_gpu
