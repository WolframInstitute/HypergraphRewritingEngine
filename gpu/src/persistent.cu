#include "hgcommon/namespace.hpp"
// Device-resident scheduling: workers that pull work from a queue rather than being launched
// once per phase per step. See gpu/include/hg_gpu/persistent.hpp and
// gpu/ARCHITECTURE.md sec 2.
//
// Its own translation unit, not appended to match.cu, and that is a memory decision rather
// than a stylistic one: match.cu already costs several GB to compile, and adding one more
// kernel to it took a single nvcc to 8 GB. This machine is shared, so a translation unit that
// cannot be compiled within a safe ceiling is a defect whether or not it links.

#include "hg_gpu/event_identity.hpp"
#include "hg_gpu/persistent.hpp"
#include <cstdio>
#include "hg_gpu/quotient_causal.hpp"
#include "hg_gpu/quotient_expansion.hpp"
#include "hg_gpu/content_hash.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {
namespace {

// The single work role the persistent schedulers count. Shared by the seed kernel below and
// the stage-2/stage-3 worker kernels.
constexpr uint32_t kRoleMatch = 0;

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

// Seed variant for a launch chain with no host round trip: the kept-root count lives in device
// memory (written by k_seed_root_hashes earlier in the same stream), so each thread reads it
// and self-selects, and the detector's pushed counter is marked here, item by item, under the
// same discipline the workers keep (mark_pushed BEFORE try_push). The grid covers the CAP --
// every root kept -- and threads past the live count exit, which is what lets the host launch
// this without ever reading the count back.
__global__ void k_seed_match_queue_counted(
        typename RingBuffer<MatchWorkItem>::DeviceView queue,
        const StateId* kept_ids, const uint32_t* kept_count, uint32_t kept_cap,
        uint32_t num_rules, uint32_t step,
        typename TerminationDetector::DeviceView term) {
    const uint32_t kept = min(*kept_count, kept_cap);
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kept * num_rules) return;
    MatchWorkItem item;
    item.state_id = kept_ids[tid / num_rules];
    item.rule_id  = tid - (tid / num_rules) * num_rules;
    item.step     = step;
    term.mark_pushed(kRoleMatch);
    queue.try_push(item);   // capacity >= kept_cap * num_rules, so this cannot fail here
}

// Seed from a SESSION FRONTIER: state ids recorded when the previous call's budget refused to
// expand them. Unlike the root seeder there is no hashing and no dedup consultation -- these
// states are already known and already deduplicated, and consulting dedup here is exactly what
// makes an extend reach nothing (measured: 5 states where one run gives 7).
__global__ void k_seed_frontier(typename RingBuffer<MatchWorkItem>::DeviceView queue,
                                const StateId* ids, const uint32_t* steps,
                                const uint32_t* count, uint32_t cap,
                                uint32_t num_rules,
                                typename TerminationDetector::DeviceView term) {
    const uint32_t live = min(*count, cap);
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= live * num_rules) return;
    MatchWorkItem item;
    // Depth is PER ENTRY: after a steered Step the frontier mixes entries stranded by
    // different budgets, and each resumes at its own recorded depth.
    item.state_id = ids[tid / num_rules];
    item.rule_id  = tid - (tid / num_rules) * num_rules;
    item.step     = steps[tid / num_rules];
    term.mark_pushed(kRoleMatch);
    queue.try_push(item);
}

// The key this run identifies states BY -- the device twin of compute_state_dedup_keys, and it
// must stay the twin: the seeding and the loop deduplicating different equivalences is not a
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
                                            uint64_t& out_key, bool want_ranks,
                                            bool want_orbits = false) {
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
                                           want_ranks, want_orbits);
    }
}

// Insert every root's canonical hash into the map before the loop starts, so a child isomorphic
// to a root deduplicates against it rather than being explored a second time. Runs pre-launch,
// which the no-host-in-the-loop constraint permits: the constraint is on evolution, not on
// seeding, alongside k_seed_roots.
//
// Surviving roots are compacted into out_ids/out_count, and the queue is seeded from those rather
// than from the caller's list, because `quotient_roots` is decided here:
//
//   false  every root is kept whether or not it won its map slot. That is the reference
//          semantics -- provided roots are distinct entry points even when isomorphic.
//   true   a root whose key another root already claimed is still hashed and mapped, but is not
//          appended, so it never enters the queue.
//
// k_seed_roots decides the same thing for the host-seeded roots. Deciding it in only one of
// them made the option change the state set on one scheduler and not the other.
__global__ void k_seed_root_hashes(DeviceState ds, const StateId* roots, uint32_t num_roots,
                                   DedupMap::DeviceView map, CanonicalizationMode state_mode,
                                   bool need_exact, bool need_ranks, DeviceArena::View arena,
                                   bool quotient_roots, QcView qc, QeView qe,
                                   StateId* out_ids, uint32_t* out_count, uint32_t out_cap) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_roots) return;
    const StateId sid = roots[tid];
    uint32_t* slot = nullptr;
    uint64_t  slot_words = 0;

    uint64_t key = 0;
    {
        const ExactHashStatus st =
            state_key_device(ds, sid, state_mode, arena, slot, slot_words, key, need_ranks,
                             qc.enabled != 0);
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

    // Quotient-causal seed, the device twin of Hypergraph::quotient_causal_seed: every orbit
    // of a root gains the INIT sentinel producer (initial edges have no producing event), and
    // the root is marked reached at depth 0. Duplicate roots re-seed the same keys; the DP's
    // per-(key, producer) dedup makes that idempotent.
    if (qc.enabled) {
        const uint32_t norb = ds.state_num_orbits[sid];
        for (uint32_t j = 0; j < norb; ++j)
            qc_add_producer(ds, qc, key, 0, j, INVALID_ID, tid);
        qc_reach(ds, qc, key, 0, tid);
    }
    // The class's root instance: every slot's edge came with the initial state, so no event
    // produced any of them. Idempotent across duplicate roots -- only the state that wins the
    // class frame records one.
    // One driver per ROOT here: this kernel runs one thread per root, so the thread's own
    // index is the slice.
    qe_seed_root_instance(ds, qe, sid, tid);

    bool merged = false;
    if (key == 0) ds.errors.record(ErrorKind::kUncomputedStateHash);   // keep it; see the kind
    else          merged = !map.insert_if_absent(key, sid).inserted;
    // Reference semantics without the option: provided roots are distinct entry points even when
    // isomorphic, so every root is kept regardless of whether it won the map slot.
    if (quotient_roots && merged) return;
    const uint32_t pos = atomicAdd(out_count, 1u);
    // Past capacity the state is not written, and a state missing from the frontier is a
    // subtree that never gets explored -- silently a smaller answer, not a slower one. Recorded
    // so the run reports partial work rather than looking complete; the host's grow-and-retry
    // reads the same kind and doubles max_states.
    if (pos < out_cap) out_ids[pos] = sid;
    else               ds.errors.record(ErrorKind::kFrontierCapFull);
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
// match_state_rule, whose contract the batch driver shares and which must not
// change. Blocks match concurrently, so no block can say which pool slots are its own: a
// before/after counter delta is not attributable to one block. The cursor sidesteps that
// entirely -- consumers claim indices, not ranges.
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
        uint64_t p1[TerminationDetector::kMaxRoles], c1[TerminationDetector::kMaxRoles];
        uint64_t p2[TerminationDetector::kMaxRoles], c2[TerminationDetector::kMaxRoles];

        // THE BUDGET COUNTS LACK OF PROGRESS, NOT ELAPSED ROUNDS.
        //
        // A fixed round ceiling cannot tell a deadlock from a workload that simply takes longer
        // than the ceiling, and it fired on the second. A rule with a disconnected left side
        // produces a cartesian product of matches, every resident block ends up inside a long
        // match, and the queue drains slowly. Measured on disc-l3a2g2r2 at depth 5: the device sat
        // at 97% utilisation -- WORKING, not stuck -- and the detector gave up anyway after ten
        // million rounds, signalled exit and returned a partial result, after which the wrapper
        // grew the pools and re-ran. A workload the CPU finishes in 25 s did not finish in 200.
        //
        // The signature also differs from a real stall. Here role0 read pushed=2972 completed=295:
        // a queue holding thousands of items nobody is popping because every consumer is busy. A
        // genuine stall has pushed == completed, because nobody is working at all.
        //
        // The counters ARE the progress signal and were already read every round. A round in which
        // any of them moves resets the budget; only rounds where nothing changes count against it.
        // A deadlock still trips it -- nothing moves, by definition -- while arbitrarily slow
        // forward progress never does.
        uint32_t stagnant = 0;
        uint32_t last_prod = 0xFFFFFFFFu, last_done = 0xFFFFFFFFu;
        uint64_t last_pc = 0xFFFFFFFFFFFFFFFFull;
        for (uint32_t round = 0; ; ++round) {
            if (stagnant >= kMaxDetectorRounds) {
                ds.errors.record(ErrorKind::kPersistentStall);
                term.signal_exit();
                return;
            }
            // Finished means BOTH: every seeded match item accounted for, and every match that
            // matching produced already consumed. Checking only the match role would exit with
            // rewrites outstanding; checking only the cursor would exit before matching had
            // produced anything at all.
            const bool q1 = term.snapshot_quiescent(p1, c1);
            const uint32_t prod1 = readable_records(found);
            const uint32_t done1 = *consume_cursor;
            {   // Progress check: any movement resets the stagnation budget. Role counters are
                // summed rather than compared elementwise -- any move changes the sum.
                uint64_t pc = 0;
                for (uint32_t r = 0; r < term.num_roles; ++r) pc += p1[r] + c1[r];
                if (prod1 != last_prod || done1 != last_done || pc != last_pc) {
                    last_prod = prod1; last_done = done1; last_pc = pc;
                    stagnant = 0;
                } else {
                    ++stagnant;
                }
            }
            if (q1 && done1 >= prod1) {
                // Quiescent once is not enough: an in-flight match may have just completed
                // without its matches yet being visible. Look again after a backoff, and
                // require every observed quantity to be UNCHANGED across the window -- each
                // is monotone, so a worker that started and finished inside it must have
                // moved one. Re-testing the conditions alone would accept two distinct
                // quiescent moments with activity between them.
                __nanosleep(4000);
                const bool q2 = term.snapshot_quiescent(p2, c2);
                const uint32_t prod2 = readable_records(found);
                const uint32_t done2 = *consume_cursor;
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

    __shared__ MatchWorkItem mitem;
    __shared__ bool have;
    __shared__ uint32_t claimed;
    uint32_t idle_ns = 64;   // thread 0's backoff state; reset whenever work is found

    for (;;) {
        // Rewrite first: it drains what matching produced, and letting the pool run ahead
        // unboundedly is what makes it overflow.
        if (threadIdx.x == 0) claimed = claim_next_record(consume_cursor, found);
        __syncthreads();
        if (claimed != INVALID_ID) {
            if (threadIdx.x == 0) {
                idle_ns = 64;
                const MatchRecord& rec = found.at(claimed);
                await_match(rec);
                // rec.step + 1, not rec.step: an event is stamped with the depth of the state
                // it PRODUCES, which is what the rewrite kernel writes
                // (run_rewrite_kernel_with_nosync is called with step + 1) and what the CPU
                // uses (the canonical OUTPUT state's step). Writing the parent's depth here
                // made every event's reported step differ from the depth it was claimed at.
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
            if (threadIdx.x == 0) { term.mark_completed(kRoleMatch); idle_ns = 64; }
            __syncthreads();
            continue;
        }

        // Nothing available in either role. Empty does NOT mean finished here -- the other
        // role may still be producing -- so only the detector decides. Backed off, because a
        // grid of idle blocks re-polling the cursor words in a tight loop starves the blocks
        // holding work of memory bandwidth.
        if (term.exit_requested()) return;
        if (threadIdx.x == 0) {
            __nanosleep(idle_ns);
            if (idle_ns < 4096u) idle_ns <<= 1;
        }
        __syncthreads();
    }
}

// ---- stage 3: the loop closes ------------------------------------------------------------
//
// A rewrite's output state is hashed, tested against the exploration rule, and its (state,
// rule) items pushed back into the same match queue. A whole evolution then runs inside one
// launch: the device decides what work exists, who takes it, and when it is finished.
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
        typename TerminationDetector::DeviceView term,
        QcView qc,
        QeView qe,
        unsigned long long* phase_cycles,
        SessionView sess) {

    // Ranks are the reconstruction's frame alignment, Automatic's signature, AND the transition
    // draw's key. One predicate answers it for the roots and for every child; see its note.
    const bool need_ranks = run_needs_edge_ranks(event_keys, qe.enabled != 0,
                                                 ds.transition_rate, ds.num_rule_weights,
                                                 ds.matches_per_state_rule);

    if (blockIdx.x == 0) {
        if (threadIdx.x != 0) return;
        uint64_t p1[TerminationDetector::kMaxRoles], c1[TerminationDetector::kMaxRoles];
        uint64_t p2[TerminationDetector::kMaxRoles], c2[TerminationDetector::kMaxRoles];

        // THE BUDGET COUNTS LACK OF PROGRESS, NOT ELAPSED ROUNDS.
        //
        // A fixed round ceiling cannot tell a deadlock from a workload that simply takes longer
        // than the ceiling, and it fired on the second. A rule with a disconnected left side
        // produces a cartesian product of matches, every resident block ends up inside a long
        // match, and the queue drains slowly. Measured on disc-l3a2g2r2 at depth 5: the device sat
        // at 97% utilisation -- WORKING, not stuck -- and the detector gave up anyway after ten
        // million rounds, signalled exit and returned a partial result, after which the wrapper
        // grew the pools and re-ran. A workload the CPU finishes in 25 s did not finish in 200.
        //
        // The signature also differs from a real stall. Here role0 read pushed=2972 completed=295:
        // a queue holding thousands of items nobody is popping because every consumer is busy. A
        // genuine stall has pushed == completed, because nobody is working at all.
        //
        // The counters ARE the progress signal and were already read every round. A round in which
        // any of them moves resets the budget; only rounds where nothing changes count against it.
        // A deadlock still trips it -- nothing moves, by definition -- while arbitrarily slow
        // forward progress never does.
        uint32_t stagnant = 0;
        uint32_t last_prod = 0xFFFFFFFFu, last_done = 0xFFFFFFFFu;
        uint64_t last_pc = 0xFFFFFFFFFFFFFFFFull;
        for (uint32_t round = 0; ; ++round) {
            if (stagnant >= kMaxDetectorRounds) {
                // Quiescence never held. Signal exit anyway so the workers leave and the
                // launch returns: a recorded defect with partial work beats holding the device.
                //
                // NAME THE COUNTER PAIR THAT FAILED TO CONVERGE. A stall is a defect, and the
                // one question worth asking of it is which side is stuck: a role whose pushed
                // exceeds its completed, or the match pool's readable count running ahead of the
                // rewrites that drain it. Without this the only evidence is a wall-clock outlier,
                // which is what made this bug survive several rounds of investigation.
                printf("[hg_gpu STALL] rounds=%u prod=%u done=%u", round,
                       readable_records(found), *rewrites_done);
                for (uint32_t r = 0; r < term.num_roles; ++r)
                    printf(" role%u(pushed=%llu completed=%llu)", r,
                           (unsigned long long)p1[r], (unsigned long long)c1[r]);
                printf("\n");
                // Name the record itself. The pool index below prod whose published flag is
                // clear IS the record a consumer is parked on, and its contents say which
                // producer abandoned it.
                {
                    const uint32_t n = readable_records(found);
                    uint32_t shown = 0;
                    for (uint32_t i = 0; i < n && shown < 4; ++i) {
                        const MatchRecord& r = found.at(i);
                        cuda::atomic_ref<const uint32_t, cuda::thread_scope_device> pf(r.published);
                        if (pf.load(cuda::memory_order_acquire) == 0u) {
                            printf("[hg_gpu STALL] unpublished idx=%u state=%u rule=%u step=%u "
                                   "num_edges=%u\n", i, r.state_id, (uint32_t)r.rule_id,
                                   r.step, (uint32_t)r.num_edges);
                            ++shown;
                        }
                    }
                    if (shown == 0) printf("[hg_gpu STALL] every record below prod IS published\n");
                }
                ds.errors.record(ErrorKind::kPersistentStall);
                term.signal_exit();
                return;
            }
            const bool q1  = term.snapshot_quiescent(p1, c1);
            const uint32_t prod1 = readable_records(found);
            const uint32_t done1 = *rewrites_done;

            // PERIODIC PROGRESS, so a long run is observable rather than opaque.
            //
            // A run that does not finish tells you nothing about WHY unless you can see whether it
            // is grinding steadily or decaying. This prints the counters roughly every two million
            // detector rounds -- about four seconds at the 2 us backoff -- and only when the run
            // has already lasted that long, so an ordinary evolution prints nothing at all. It
            // exists because ten hypotheses about a non-finishing workload were each eliminated by
            // a separate measurement, and none of them could be tested against the run itself.
            if (round > 0 && (round % 2000000u) == 0u) {
                // The phase counters too, read from device memory by the detector block -- the
                // host is blocked in its sync and cannot see them, and the workers now flush
                // every 1024 records precisely so a run that never finishes is still
                // attributable. Fractions of their sum, the same reading as PersistentEvolveStats.
                unsigned long long m = 0, rw = 0, cn = 0, id = 0, wt = 0;
                if (phase_cycles) {
                    m = phase_cycles[0]; rw = phase_cycles[1]; cn = phase_cycles[2];
                    id = phase_cycles[3]; wt = phase_cycles[4];
                }
                const unsigned long long tot = m + rw + cn + id + wt;
                // The canon bucket's five parts, as fractions of the bucket. Without this the
                // bucket reads as "canonicalization" while containing four other calls.
                unsigned long long ir = 0, sg = 0, qc_ = 0, qe_ = 0, dd = 0;
                if (phase_cycles) {
                    ir = phase_cycles[11]; sg = phase_cycles[12]; qc_ = phase_cycles[13];
                    qe_ = phase_cycles[14]; dd = phase_cycles[15];
                }
                const unsigned long long cb = ir + sg + qc_ + qe_ + dd;
                printf("[hg_gpu PROGRESS] round=%u prod=%u done=%u | "
                       "match=%llu%% rewrite=%llu%% canonblk=%llu%% idle=%llu%% || "
                       "ir=%llu%% sig=%llu%% qc=%llu%% qe=%llu%% dedup=%llu%%\n",
                       round, prod1, done1,
                       tot ? 100ull * m  / tot : 0ull, tot ? 100ull * rw / tot : 0ull,
                       tot ? 100ull * cn / tot : 0ull, tot ? 100ull * id / tot : 0ull,
                       cb ? 100ull * ir  / cb : 0ull, cb ? 100ull * sg  / cb : 0ull,
                       cb ? 100ull * qc_ / cb : 0ull, cb ? 100ull * qe_ / cb : 0ull,
                       cb ? 100ull * dd  / cb : 0ull);
            }

            {   // Progress check: any movement resets the stagnation budget. Role counters are
                // summed rather than compared elementwise -- any move changes the sum.
                uint64_t pc = 0;
                for (uint32_t r = 0; r < term.num_roles; ++r) pc += p1[r] + c1[r];
                if (prod1 != last_prod || done1 != last_done || pc != last_pc) {
                    last_prod = prod1; last_done = done1; last_pc = pc;
                    stagnant = 0;
                } else {
                    ++stagnant;
                }
            }
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
    uint32_t idle_ns    = 64;   // thread 0's backoff state; reset whenever work is found

    // Phase attribution, accumulated in thread 0's registers and flushed once at exit so the
    // hot loop carries no extra atomics. See PersistentEvolveStats for what the four mean.
    unsigned long long acc_match = 0, acc_rewrite = 0, acc_canon = 0, acc_idle = 0,
                       acc_wait = 0;
    auto flush_cycles = [&] {
        if (threadIdx.x == 0 && phase_cycles) {
            atomicAdd(&phase_cycles[0], acc_match);
            atomicAdd(&phase_cycles[1], acc_rewrite);
            atomicAdd(&phase_cycles[2], acc_canon);
            atomicAdd(&phase_cycles[3], acc_idle);
            atomicAdd(&phase_cycles[4], acc_wait);
            acc_match = acc_rewrite = acc_canon = acc_idle = acc_wait = 0;
        }
    };

    // FLUSH PERIODICALLY, NOT ONLY AT EXIT.
    //
    // These counters were published once, when a block left the loop, so a run that does not
    // finish attributed nothing at all -- which is exactly the run whose attribution is wanted.
    // A block that has consumed a few thousand records has already said what it needed to say;
    // publishing then costs five atomics against a handful of records of work, and the
    // accumulators reset so nothing is double counted.
    uint32_t records_since_flush = 0;

    if (threadIdx.x == 0) { ir_slot = nullptr; ir_slot_words = 0; stalled = false; }
    __syncthreads();

    for (;;) {
        // Rewrite first: it drains what matching produced, and letting the pool run ahead
        // unboundedly is what makes it overflow.
        if (threadIdx.x == 0) claimed = claim_next_record(consume_cursor, found);
        __syncthreads();

        if (claimed != INVALID_ID) {
            if (threadIdx.x == 0) {
                idle_ns = 64;
                idle_spins = 0;            // consecutive, not cumulative -- see the guard below
                const unsigned long long t0 = clock64();
                const MatchRecord& rec = found.at(claimed);
                await_match(rec);
                const unsigned long long t0b = clock64();
                acc_wait += t0b - t0;
                const uint32_t step = rec.step;
                // The event carries the depth of the state it PRODUCES -- see the note in
                // k_persistent_match_rewrite. The exploration depth below is the same value.
                const AppliedMatch applied = apply_one_match(
                    ds, rules, rec, step + 1u,
                    phase_cycles ? phase_cycles + 5 : nullptr);
                child_sid    = applied.state;
                child_event  = applied.event;
                child_step   = step + 1u;
                expand_child = false;
                acc_rewrite += clock64() - t0b;
                const unsigned long long t1 = clock64();

                // Expand the child only if it exists, the step budget allows it, its exact
                // hash is computable, and the exploration rule keeps it. The hash is the
                // dedup KEY, so a state whose hash could not be computed is not enqueued
                // under a coarser one -- 1-WL merges non-isomorphic states.
                if (child_sid != INVALID_ID) {
                    // SPLIT THE canon BUCKET INTO ITS PARTS.
                    //
                    // acc_canon spans this whole block, so it has been reporting
                    // "canonicalization" for a region that also stamps event signatures, drives
                    // the quotient causal DP, captures the class-frame expansion and consults
                    // dedup. A 99% reading was taken to mean individualization-refinement and does
                    // not: an isolated measurement puts device IR at 62.9x the host on one state,
                    // not the thousands the whole-block figure implied. Slots 11-15 name the
                    // parts so the next question is asked of the right one.
                    uint64_t sub0 = clock64();
                    uint64_t h = 0;
                    const ExactHashStatus key_st =
                        state_key_device(ds, child_sid, state_mode, arena, ir_slot,
                                         ir_slot_words, h, need_ranks, qc.enabled != 0);
                    if (phase_cycles) atomicAdd(&phase_cycles[11], clock64() - sub0);
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
                            const uint64_t s1 = clock64();
                            stamp_event_signature(ds, child_event, event_keys,
                                                  ds.state_exact_hash[rec.state_id], exact,
                                                  rec.state_id, child_sid, step + 1u,
                                                  rec.rule_id, event_map);
                            if (phase_cycles) atomicAdd(&phase_cycles[12], clock64() - s1);
                        }

                        // Quotient causal: register this raw event's canonical transition and
                        // drive the DP. EVERY raw event registers, whether or not the child
                        // survives dedup below -- the host registers per raw event too. Both
                        // endpoint hashes and orbit tables exist at this point (the parent's
                        // from its own canon, the child's from the pass just above).
                        if (child_event != INVALID_ID) {
                            // The DP runs only when its output is recorded. `enabled` still
                            // follows the quotient route, so edge orbits are computed exactly as
                            // before and the answer does not move; only the causal relation's own
                            // work is skipped. Measured as the growing term of this block on
                            // disc-l3a2g2r2: 43% then 66% then 81% while IR fell 53% to 17%.
                            if (qc.enabled && qc.record_causal) {
                                const uint64_t s2 = clock64();
                                qc_register_transition(ds, qc, rec.state_id, child_sid,
                                                       child_event, rec.rule_id, step,
                                                       blockIdx.x);
                                if (phase_cycles) atomicAdd(&phase_cycles[13], clock64() - s2);
                            }
                            // Same event, same endpoints: the class frame's match record.
                            const uint64_t s3 = clock64();
                            // One driver per BLOCK: this whole path is inside
                            // `threadIdx.x == 0`, so blockIdx is the slice.
                            qe_capture_expansion(ds, qe, rec.state_id, child_sid,
                                                 child_event, rec.rule_id, step, blockIdx.x);
                            if (phase_cycles) atomicAdd(&phase_cycles[14], clock64() - s3);
                        }

                        if (child_step < max_steps) {
                            const uint64_t s4 = clock64();
                            expand_child = state_survives_dedup(ds, child_sid, h, dedup_map,
                                                                dedup, explore_threshold_u32,
                                                                explore_seed, child_step,
                                                                rec.state_id);
                            if (phase_cycles) atomicAdd(&phase_cycles[15], clock64() - s4);
                        } else if (sess.enabled) {
                            // AT THE BUDGET, AND THE RUN IS CONTINUABLE. Consult dedup anyway --
                            // a duplicate needs no frontier entry, someone else's copy carries
                            // the expansion -- and record the survivor so the next call can
                            // expand it. Without this the boundary is not merely unexpanded, it
                            // is unrecoverable: nothing else records which states the budget
                            // stopped at.
                            if (state_survives_dedup(ds, child_sid, h, dedup_map, dedup,
                                                     explore_threshold_u32, explore_seed,
                                                     child_step, rec.state_id)) {
                                cuda::atomic_ref<uint32_t, cuda::thread_scope_device>
                                    fc(*sess.frontier_count);
                                const uint32_t at = fc.fetch_add(1u, cuda::memory_order_relaxed);
                                if (at < sess.frontier_cap) {
                                    sess.frontier[at]      = child_sid;
                                    sess.frontier_step[at] = child_step;
                                } else ds.errors.record(ErrorKind::kFrontierCapFull);
                            }
                            expand_child = false;
                        } else {
                            expand_child = false;
                        }
                    }
                }
                acc_canon += clock64() - t1;
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
                const unsigned long long tA =
                    (threadIdx.x == 0 && run_rule_inline) ? clock64() : 0;
                if (run_rule_inline)
                    match_state_rule(ds, rules, child_sid, r, child_step, found);
                __syncthreads();
                if (threadIdx.x == 0 && run_rule_inline) acc_match += clock64() - tA;
            }

            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(rewrites_done, 1u);
                if (++records_since_flush >= 8u) {
                    flush_cycles();
                    records_since_flush = 0;
                }
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) have = match_q.try_pop(mitem);
        __syncthreads();
        if (have) {
            const unsigned long long tA = (threadIdx.x == 0) ? clock64() : 0;
            match_state_rule(ds, rules, mitem.state_id, mitem.rule_id, mitem.step, found);
            __syncthreads();
            if (threadIdx.x == 0) {
                term.mark_completed(kRoleMatch);
                idle_ns = 64;
                idle_spins = 0;            // consecutive, not cumulative -- see the guard below
                acc_match += clock64() - tA;
            }
            __syncthreads();
            continue;
        }

        if (term.exit_requested()) { flush_cycles(); return; }
        // Idle, and the detector has not released us. Counted, because a worker that can
        // neither find work nor be told to stop is the same defect from the other side.
        //
        // Backed off, because a grid of idle blocks re-polling the ring's cursor words in a
        // tight loop contends with the blocks HOLDING work for the very lines their pushes and
        // pops need -- the seed is often a single item, so the ramp is a window where most
        // blocks are idle and the few working ones set the pace, and the drain tail is the
        // same shape. Exponential to a 4 us ceiling: at most one ceiling's latency added to
        // waking up with work, against orders of magnitude less idle traffic. Idle polling is
        // the only queue traffic that scales with the grid: every productive op carries a
        // whole subgraph match or rewrite, so push/pop rates sit orders of magnitude below
        // what an MPMC ring saturates at. Measured on
        // bench_gpu_evolve (WPP, quotient, Full, 6 steps, RTX 4090): medians hold ~10 ms from
        // the SM-count grid through 8x oversubscription (128 blocks 10.2, 256 9.9, 512 10.2,
        // 1024 11.3).
        if (threadIdx.x == 0) {
            const unsigned long long tA = clock64();
            // CONSECUTIVE IDLE ITERATIONS, NOT LIFETIME ONES.
            //
            // This counter exists to catch a worker that can neither find work nor be told to
            // stop, which is a condition about an UNBROKEN run of idling. It was never reset, so
            // it accumulated over the whole kernel: a block that idled briefly between items --
            // the normal thing to do whenever the queue is momentarily empty -- added to it every
            // time, and after twenty million such moments declared a stall and RETIRED, however
            // productive it had been in between.
            //
            // On a short run nothing reaches the cap. On a long one the workers die off one at a
            // time and throughput decays with them, which is why disc-l3a2g2r2 finishes at depth 4
            // in 244 ms and had not finished at depth 5 after 540 s. The backoff sleeps up to
            // 4 us, so twenty million idle moments is tens of seconds of cumulative idling --
            // easily reached by a run lasting minutes, and unreachable by one lasting a quarter of
            // a second.
            //
            // Reset wherever work is found, beside the backoff reset that was already there.
            if (++idle_spins >= kMaxWorkerIdleSpins) {
                ds.errors.record(ErrorKind::kPersistentStall);
                stalled = true;
            } else {
                __nanosleep(idle_ns);
                if (idle_ns < 4096u) idle_ns <<= 1;
            }
            acc_idle += clock64() - tA;
        }
        __syncthreads();
        if (stalled) { flush_cycles(); return; }
    }
}

}  // namespace

// Declared in hg_gpu/persistent.hpp, which carries the contract. One resident block per SM:
// a persistent kernel's blocks do not retire and get replaced, so the grid IS the worker count.
// Each block works one item at a time on thread 0 (the shared match/rewrite/canon routines are
// single-threaded per item, and warp-bursting them was measured SLOWER -- irregular tasks in
// one warp serialize on divergence and burst their atomics into the same lines), so the
// scaling axis is MORE BLOCKS, each an independent serial worker the SM scheduler interleaves.
//
// THE BOUND IS OCCUPANCY, NOT QUEUE CONTENTION. Those predict opposite curves -- contention
// would flatten early or climb as workers pile onto the same cursors -- and the measured curve
// falls monotonically and then plateaus:
//
//   for b in 32 64 128 256 512 1024 2048 3072; do
//     HG_GPU_PERSISTENT_BLOCKS=$b build_gpu/bench_gpu_evolve 7 5 2; done
//
//   grid    32     64    128    256    512   1024   2048   3072
//   ms     338    189    118     86     72     69     62     61
//
// (RTX 4090, 128 SMs, 45317 states / 45316 events, medians of 5.) The plateau begins near 8x the
// SM count, which is where the default sits; the idle-path backoff in the kernels above is what
// keeps oversubscription free when work runs short. Run-to-run spread on this host is ~10%, and
// an explicit 1024 measures the same as the 8/SM default (58.6 vs 58.7 over 9 iterations), which
// is the check that the override and the derived default are the same grid.
//
// Quotient causal is orbit-keyed (quotient_causal.hpp), so the causal set is the same at every
// grid (tools/quotient_causal_probe_gpu holds it constant, and equal to the CPU's).
uint32_t default_persistent_grid() {
    static uint32_t cached = 0;
    if (cached) return cached;
    // Measurement override, read once. Everything grid-derived (worker count, IR arena slots)
    // funnels through this function, so an override scales all of it consistently.
    if (const char* env = std::getenv("HG_GPU_PERSISTENT_BLOCKS")) {
        const long v = std::atol(env);
        if (v > 0) { cached = static_cast<uint32_t>(v); return cached; }
    }
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
        sms <= 0) {
        cudaGetLastError();   // do not leave a sticky status behind for the next launch
        sms = 32;             // a plausible small device; the caller's floor still applies
    }
    cached = static_cast<uint32_t>(sms) * 8u;
    return cached;
}

uint32_t run_persistent_match(const EngineState& engine,
                              const std::vector<DeviceRule>& rules,
                              const std::vector<StateId>& states,
                              Pool<MatchRecord>& out,
                              uint32_t blocks) {
    if (rules.empty() || states.empty()) return out.size_host();

    const uint32_t num_rules = static_cast<uint32_t>(rules.size());
    const uint32_t num_items = static_cast<uint32_t>(num_rules * states.size());

    DeviceRule* d_rules = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    StateId* d_states = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_states, sizeof(StateId) * states.size()), "states alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice), "states copy");

    uint32_t cap = 1;
    while (cap < num_items) cap <<= 1;
    RingBuffer<MatchWorkItem> queue(cap);

    {
        const uint32_t block = 128;
        const uint32_t seed_grid = (num_items + block - 1) / block;
        k_seed_match_queue<<<seed_grid, block>>>(queue.view(), d_states,
                                                 static_cast<uint32_t>(states.size()), num_rules,
                                                 /*step=*/0u);
        HG_CUDA_CHECK(cudaDeviceSynchronize(), "seed sync");
    }

    // Deliberately FEWER blocks than items: each one loops, which is the whole difference from
    // launching one block per item.
    const uint32_t grid = blocks ? blocks : 64;

    k_persistent_match<<<grid, kMatchBlockThreads>>>(engine.device(), d_rules,
                                                     queue.view(), out.view());
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "persistent match sync");

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
    HG_CUDA_CHECK(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    StateId* d_states = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_states, sizeof(StateId) * states.size()), "states alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice), "states copy");

    uint32_t cap = 1;
    while (cap < num_items) cap <<= 1;
    RingBuffer<MatchWorkItem> match_q(cap);
    {
        const uint32_t block = 128;
        const uint32_t seed_grid = (num_items + block - 1) / block;
        k_seed_match_queue<<<seed_grid, block>>>(match_q.view(), d_states,
                                                 static_cast<uint32_t>(states.size()), num_rules,
                                                 step);
        HG_CUDA_CHECK(cudaDeviceSynchronize(), "seed sync");
    }

    uint32_t* d_cursor = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(uint32_t)), "cursor alloc");
    HG_CUDA_CHECK(cudaMemset(d_cursor, 0, sizeof(uint32_t)), "cursor clear");

    TerminationDetector term(/*num_roles=*/1);
    term.clear();
    term.mark_pushed_host(kRoleMatch, num_items);

    // Block 0 is the detector, so at least two blocks are needed for any work to happen.
    const uint32_t grid_req = blocks ? blocks : default_persistent_grid();
    const uint32_t grid = grid_req < 2 ? 2 : grid_req;
    k_persistent_match_rewrite<<<grid, kMatchBlockThreads>>>(
        engine.device(), d_rules, match_q.view(), scratch_matches.view(),
        d_cursor, term.view(), step);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "persistent match+rewrite sync");

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
                                            uint32_t blocks,
                                            bool quotient_roots,
                                            const QcView* qc_in,
                                            const QeView* qe_in,
                                            SessionView* session,
                                            uint32_t start_step) {
    PersistentEvolveStats stats;
    if (rules.empty() || roots.empty() || max_steps == 0) return stats;

    QcView qc{};
    if (qc_in) qc = *qc_in;
    QeView qe{};
    if (qe_in) qe = *qe_in;

    // Records are consumed while they are still being produced, so their publication flags
    // must start clear. The scheduler that relies on the flag is the one that clears it.
    scratch_matches.reset_and_clear();

    const uint32_t num_rules = static_cast<uint32_t>(rules.size());
    const uint32_t num_seed  = static_cast<uint32_t>(num_rules * roots.size());

    DeviceRule* d_rules = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    StateId* d_states = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_states, sizeof(StateId) * roots.size()), "states alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_states, roots.data(), sizeof(StateId) * roots.size(),
                     cudaMemcpyHostToDevice), "states copy");

    // The ring holds work in flight, not the whole evolution: a run that outgrows it does not
    // fail, it runs the excess inline on the pushing block. Sized to the match pool so the
    // inline path is an escape valve rather than the normal case.
    uint32_t cap = 1;
    while (cap < num_seed) cap <<= 1;
    while (cap < scratch_matches.capacity() && cap < (1u << 20)) cap <<= 1;
    RingBuffer<MatchWorkItem> match_q(cap);

    // The canonical map is the dedup key store for the whole run. Sized to the state pool: one
    // entry per state is the worst case, and the map must not fill, because a full map would
    // silently start admitting duplicates.
    // A SESSION OWNS ITS IDENTITY. Rebuilt per call otherwise, which is what a one-shot run
    // wants and what makes a second call re-derive everything as new.
    SessionView sess_v{};
    if (session) sess_v = *session;
    const bool dbgt = std::getenv("HG_GPU_DBG_TIME") != nullptr;
    auto t_maps0 = std::chrono::steady_clock::now();
    DedupMap owned_canonical(session ? 1u : engine.config().max_states * 2u);
    DedupMap& canonical_owner = owned_canonical;
    if (!session) canonical_owner.clear();

    // Signature -> first event with it. Sized off the event budget rather than the state one:
    // an evolution has as many applications as it has matches, which is not bounded by its
    // state count.
    //
    // Sized to nothing when no event identity is being computed. At the default config this map
    // is 2^18 slots, and allocating plus clearing it is milliseconds on runs that take tens --
    // a cost charged to every run for a mode most do not select. The stamp sites are all behind
    // `event_keys != EVENT_SIG_NONE`, so the small map is never touched.
    const bool want_event_ids = (event_keys != EVENT_SIG_NONE);
    DedupMap owned_event_ids(session ? 1u : (want_event_ids ? engine.config().max_events * 2u : 8u));
    if (!session) owned_event_ids.clear();
    if (want_event_ids) engine.ensure_event_identity();

    const double t_maps = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_maps0).count();
    auto t_alloc0 = std::chrono::steady_clock::now();

    // Every allocation happens HERE, before the first kernel goes out: cudaMalloc may
    // synchronize the device, and the evolution's contract is memory traffic at the start and
    // end only, with ONE synchronization -- after the last kernel.
    StateId* d_kept = nullptr;
    uint32_t* d_kept_count = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_kept, sizeof(StateId) * roots.size()), "kept roots alloc");
    HG_CUDA_CHECK(cudaMalloc(&d_kept_count, sizeof(uint32_t)), "kept count alloc");
    HG_CUDA_CHECK(cudaMemset(d_kept_count, 0, sizeof(uint32_t)), "kept count clear");

    uint32_t* d_cursor = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(uint32_t) * 2), "cursor alloc");
    HG_CUDA_CHECK(cudaMemset(d_cursor, 0, sizeof(uint32_t) * 2), "cursor clear");
    uint32_t* d_rewrites_done = d_cursor + 1;

    // 5 top-level phases + apply_one_match's 6 sub-stretches (see rewrite.hpp).
    unsigned long long* d_phase_cycles = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_phase_cycles, sizeof(unsigned long long) * 16), "phase cycles alloc");
    HG_CUDA_CHECK(cudaMemset(d_phase_cycles, 0, sizeof(unsigned long long) * 16), "phase cycles clear");

    TerminationDetector term(/*num_roles=*/1);
    term.clear();

    // The whole evolution is a launch CHAIN on one stream: root hashing decides which roots
    // survive and compacts them into d_kept/d_kept_count; the counted seeder reads that count
    // on the device, enqueues (root, rule) items and books them with the detector; the evolve
    // kernel consumes them. Stream order carries every dependency, so the host synchronizes
    // exactly once, after the last kernel, and reads nothing back before that.
    const double t_alloc = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_alloc0).count();
    auto t_seed0 = std::chrono::steady_clock::now();

    arena.reset();
    // CONTINUING rather than starting: the frontier already holds hashed, deduplicated states,
    // so it is seeded straight into the queue at each entry's own recorded depth. The
    // root path would re-hash them and, worse, consult dedup -- which they already satisfy, so
    // nothing would expand.
    if (start_step > 0 && session) {
        const uint32_t block = 128;
        const uint32_t items = sess_v.frontier_cap * num_rules;
        const uint32_t seed_grid = (items + block - 1) / block;
        if (seed_grid) {
            k_seed_frontier<<<seed_grid, block>>>(
                match_q.view(), sess_v.frontier, sess_v.frontier_step, sess_v.frontier_count,
                sess_v.frontier_cap, num_rules, term.view());
        }
        // THE FRONTIER IS CONSUMED, NOT ACCUMULATED. The states it held are being expanded now,
        // and this run's own boundary takes their place -- so the counter is reset between the
        // seed reading it and the workers appending to it. Stream order is what makes that safe:
        // both are on the default stream, so the seed sees the old count and the workers start
        // from zero. Without this a SECOND extend re-seeds the first extend's boundary, at a
        // depth those states have already passed.
        HG_CUDA_CHECK(cudaMemsetAsync(sess_v.frontier_count, 0, sizeof(uint32_t)),
                      "session frontier consume");
    } else
    {
        const uint32_t block = 64;
        const uint32_t n = static_cast<uint32_t>(roots.size());
        // The device view is taken once: the rank predicate reads the run's sampling parameters
        // out of it, and it must be the SAME view the kernel is handed.
        const DeviceState dsv = engine.device();
        k_seed_root_hashes<<<(n + block - 1) / block, block>>>(
            dsv, d_states, n,
            session ? sess_v.states : canonical_owner.view(), state_mode,
            event_keys != EVENT_SIG_NONE,
            run_needs_edge_ranks(event_keys, qe.enabled != 0, dsv.transition_rate,
                                 dsv.num_rule_weights, dsv.matches_per_state_rule),
            arena.view(), quotient_roots, qc, qe, d_kept, d_kept_count, n);
    }
    if (!(start_step > 0 && session)) {
        const uint32_t block = 128;
        const uint32_t items = static_cast<uint32_t>(roots.size()) * num_rules;
        const uint32_t seed_grid = (items + block - 1) / block;
        k_seed_match_queue_counted<<<seed_grid, block>>>(
            match_q.view(), d_kept, d_kept_count, static_cast<uint32_t>(roots.size()),
            num_rules, /*step=*/0u, term.view());
    }

    // Block 0 is the detector, so at least two blocks are needed for any work to happen.
    const uint32_t grid_req = blocks ? blocks : default_persistent_grid();
    const uint32_t grid = grid_req < 2 ? 2 : grid_req;
    const double t_seed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_seed0).count();
    if (dbgt)
        std::fprintf(stderr, "[persistent setup] dedup_maps=%.2f allocs=%.2f seed=%.2f (ms)\n",
                     t_maps, t_alloc, t_seed);

    k_persistent_evolve<<<grid, kMatchBlockThreads>>>(
        engine.device(), d_rules, num_rules, match_q.view(), scratch_matches.view(),
        d_cursor, d_rewrites_done,
        session ? sess_v.states : canonical_owner.view(), dedup,
        explore_threshold_u32, explore_seed, max_steps, state_mode, event_keys,
        session ? sess_v.events : owned_event_ids.view(),
        arena.view(), term.view(), qc, qe, d_phase_cycles, sess_v);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "persistent evolve sync");

    // states_after and canonical_events are both slots of the engine's counter block, so one
    // transfer fetches them instead of two. The pool and arena counters belong to other objects
    // and still cost a call each.
    const auto ctr = engine.counters_snapshot_host();
    stats.matches_found    = scratch_matches.size_host();
    stats.states_after     = ctr.states;
    stats.arena_words_used = arena.used_words_host();
    stats.canonical_events = ctr.canonical_ev;

    unsigned long long phase[16] = {};
    HG_CUDA_CHECK(cudaMemcpy(phase, d_phase_cycles, sizeof(phase), cudaMemcpyDeviceToHost),
          "phase cycles read");
    stats.cycles_match   = phase[0];
    stats.cycles_rewrite = phase[1];
    stats.cycles_canon   = phase[2];
    stats.cycles_idle    = phase[3];
    stats.cycles_wait    = phase[4];
    for (int i = 0; i < 6; ++i) stats.cycles_rw_sub[i] = phase[5 + i];

    cudaFree(d_phase_cycles);
    cudaFree(d_cursor);
    cudaFree(d_states);
    cudaFree(d_kept);
    cudaFree(d_kept_count);
    cudaFree(d_rules);
    return stats;
}


// =============================================================================
// persistent.hpp host bodies
// =============================================================================
//
// SessionState owns device allocations and reads a counter back across the boundary; none of
// it is device code and none runs per item. persistent_arena_words is arithmetic a launch does
// once. The kernels and the SessionView the device sees stay in the header.

uint64_t persistent_arena_words(uint32_t share_words, uint32_t holders) {
    return static_cast<uint64_t>(holders) * static_cast<uint64_t>(share_words);
}

SessionState::SessionState(uint32_t max_states, uint32_t max_events): states_(max_states * 2u), events_(max_events * 2u), cap_(max_states) {
        states_.clear();
        events_.clear();
        HG_CUDA_CHECK(cudaMalloc(&frontier_, sizeof(StateId) * cap_), "session frontier alloc");
        HG_CUDA_CHECK(cudaMalloc(&step_, sizeof(uint32_t) * cap_), "session frontier step alloc");
        HG_CUDA_CHECK(cudaMalloc(&count_, sizeof(uint32_t)), "session frontier count alloc");
        HG_CUDA_CHECK(cudaMemset(count_, 0, sizeof(uint32_t)),
                      "session frontier count clear");
    }

SessionState::~SessionState() {
        if (frontier_) cudaFree(frontier_);
        if (step_)     cudaFree(step_);
        if (count_)    cudaFree(count_);
    }

uint32_t SessionState::frontier_size() const {
        uint32_t n = 0;
        HG_CUDA_CHECK(cudaMemcpy(&n, count_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                      "session frontier count read");
        return n < cap_ ? n : cap_;
    }

void SessionState::frontier_host(std::vector<StateId>& ids,
                                 std::vector<uint32_t>& steps) const {
        const uint32_t n = frontier_size();
        ids.resize(n);
        steps.resize(n);
        if (n == 0) return;
        HG_CUDA_CHECK(cudaMemcpy(ids.data(), frontier_, sizeof(StateId) * n,
                                 cudaMemcpyDeviceToHost),
                      "session frontier read");
        HG_CUDA_CHECK(cudaMemcpy(steps.data(), step_, sizeof(uint32_t) * n,
                                 cudaMemcpyDeviceToHost),
                      "session frontier step read");
    }

void SessionState::set_frontier_host(const StateId* ids, const uint32_t* steps, uint32_t n) {
        if (n > cap_) n = cap_;
        if (n) {
            HG_CUDA_CHECK(cudaMemcpy(frontier_, ids, sizeof(StateId) * n,
                                     cudaMemcpyHostToDevice),
                          "session frontier write");
            HG_CUDA_CHECK(cudaMemcpy(step_, steps, sizeof(uint32_t) * n,
                                     cudaMemcpyHostToDevice),
                          "session frontier step write");
        }
        HG_CUDA_CHECK(cudaMemcpy(count_, &n, sizeof(uint32_t), cudaMemcpyHostToDevice),
                      "session frontier count write");
    }

SessionView SessionState::view() {
        SessionView v;
        v.states         = states_.view();
        v.events         = events_.view();
        v.frontier       = frontier_;
        v.frontier_step  = step_;
        v.frontier_count = count_;
        v.frontier_cap   = cap_;
        v.enabled        = 1;
        return v;
    }

}  // namespace gpu
}  // namespace HG_NAMESPACE
