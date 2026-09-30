#include "hgcommon/namespace.hpp"
#include "hgcommon/termination_core.hpp"
// Device-resident scheduling: workers that pull work from a queue rather than being launched
// once per phase per step. See gpu/include/hg_gpu/persistent.hpp and
// gpu/ARCHITECTURE.md sec 2.
//
// Its own translation unit, not appended to match.cu, and that is a memory decision rather
// than a stylistic one: match.cu already costs several GB to compile, and adding one more
// kernel to it took a single nvcc to 8 GB. This machine is shared, so a translation unit that
// cannot be compiled within a safe ceiling is a defect whether or not it links.

#include "hg_gpu/event_identity.hpp"
#include "hg_gpu/keyed.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/explore_depth.hpp"
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


// Seed from a SESSION FRONTIER: state ids recorded when the previous call's budget refused to
// expand them. Unlike the root seeder there is no hashing and no dedup consultation -- these
// states are already known and already deduplicated, and consulting dedup here is exactly what
// makes an extend reach nothing (measured: 5 states where one run gives 7).
__global__ void k_seq_ramp(uint64_t* seq, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) seq[i] = i;
}

__global__ void k_seed_frontier(DeviceState ds, ExploreView ev, const StateId* ids,
                                const uint32_t* steps, const uint32_t* count, uint32_t cap) {
    const uint32_t live = min(*count, cap);
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= live) return;
    const StateId s = ids[tid];
    // A state lowered under the old budget later in the run that recorded it was expanded
    // then, and holds the claim.
    if (!ev.claim(s)) return;
    // Depth is PER ENTRY: after a steered Step the frontier mixes entries stranded by
    // different budgets. The state's own depth is the smallest any path reached it by.
    uint32_t d = steps[tid];
    const uint32_t known = ev.depth[s];
    if (known < d) d = known;
    if (!ev.expand.append(ExpandEntry{s, d, 0u})) ds.errors.record(ErrorKind::kStatePoolFull);
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
template <class Par = hgcommon::IrSerial>
__device__ ExactHashStatus state_key_device(DeviceState ds, StateId sid,
                                            CanonicalizationMode mode,
                                            DeviceArena::View arena,
                                            uint32_t*& slot, uint64_t& slot_words,
                                            uint64_t& out_key, bool want_ranks,
                                            bool want_orbits = false,
                                            uint32_t** out_form = nullptr,
                                            uint32_t* out_form_words = nullptr,
                                            Par par = Par{}) {
    if (out_form) { *out_form = nullptr; *out_form_words = 0; }
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
                                           want_ranks, want_orbits, out_form, out_form_words,
                                           par);
    }
}

// Insert every root's canonical hash into the map before the loop starts, so a child isomorphic
// to a root deduplicates against it rather than being explored a second time. Runs pre-launch,
// which the no-host-in-the-loop constraint permits: the constraint is on evolution, not on
// seeding, alongside k_seed_roots.
//
// Every root is compacted into out_ids/out_count, isomorphic ones included, and the queue is
// seeded from those.
// One thread per replay driver: the points the previous run's bound left standing.
__global__ void k_qe_redrive(DeviceState ds, QeView qe, uint32_t old_bound) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= qe.work_slices) return;
    qe_redrive(ds, qe, old_bound, tid, qe.work_slices);
}

__global__ void k_seed_root_hashes(DeviceState ds, const StateId* roots, uint32_t num_roots,
                                   DedupMap::DeviceView map, CanonicalizationMode state_mode,
                                   bool need_exact, bool need_ranks, DeviceArena::View arena,
                                   QcView qc, QeView qe, ExploreView ev,
                                   typename Pool<uint32_t>::DeviceView forms,
                                   DedupMap::DeviceView exact_map) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_roots) return;
    const StateId sid = roots[tid];
    if (ds.keyed.enabled) {
        ds.keyed.state_first_new_edge[sid] = INVALID_ID;
        ds.keyed.state_token_sum[sid] = 0;
    }
    uint32_t* slot = nullptr;
    uint64_t  slot_words = 0;

    uint64_t key = 0;
    uint32_t* form = nullptr;
    uint32_t form_words = 0;
    {
        const ExactHashStatus st =
            state_key_device(ds, sid, state_mode, arena, slot, slot_words, key, need_ranks,
                             qc.enabled != 0, &form, &form_words);
        if (st != ExactHashStatus::kOk) {
            ds.errors.record(error_kind_for(st));
            return;
        }
    }
    // Full and Automatic: the class's claimed key is the state's identity -- claimed on the
    // canonical form in Full, on the content in Automatic.
    const bool full = state_mode == CanonicalizationMode::Full;
    const bool automatic = state_mode == CanonicalizationMode::Automatic;
    StateClaim claim{sid, key, true};
    if (full) {
        claim = state_claim_form(ds, sid, key & ds.canonical_key_mask, form, form_words, map,
                                 forms);
        key = claim.key;
    } else if (automatic) {
        claim = state_claim_content(ds, sid, key, map);
        key = claim.key;
    }
    ds.state_canonical_hash[sid] = key;

    // The exact hash is a SECOND quantity, and only in Full mode is it the same one. Computed
    // here only if an event identity or a transition key will read it (run_needs_exact_hash).
    if (need_exact) {
        uint64_t exact = key;
        if (state_mode != CanonicalizationMode::Full) {
            uint32_t* eform = nullptr;
            uint32_t eform_words = 0;
            const ExactHashStatus st =
                state_exact_hash_device(ds, sid, arena, slot, slot_words, exact, need_ranks,
                                        false, &eform, &eform_words);
            if (st != ExactHashStatus::kOk) {
                ds.errors.record(error_kind_for(st));
                return;
            }
            exact = state_claim_form(ds, sid, exact & ds.event_key_mask, eform, eform_words,
                                     exact_map, forms).key;
        }
        ds.state_exact_hash[sid] = exact;
    }

    // The class's root instance: every slot's edge came with the initial state, so no event
    // produced any of them. Idempotent across duplicate roots -- only the state that wins the
    // class frame records one.
    // One driver per ROOT here: this kernel runs one thread per root, so the thread's own
    // index is the slice.
    qe_seed_root_instance(ds, qe, sid, tid);

    // Every root is expanded at depth 0, isomorphic ones included: each is its own initial
    // state. Its class's canonical state is claimed too, so no later arrival expands the class
    // again (the host's try_claim_expanded on the canonical root).
    if (full || automatic) {
        if (!claim.fresh) {
            atomicMin(&ev.depth[claim.canonical], 0u);
            ev.claim(claim.canonical);
        }
    } else if (key == 0) {
        ds.errors.record(ErrorKind::kUncomputedStateHash);   // keep it; see the kind
    } else {
        const auto r = map.insert_if_absent(key, sid);
        if (!r.inserted && !r.overflowed) {
            atomicMin(&ev.depth[r.value], 0u);
            ev.claim(r.value);
        }
    }
    atomicMin(&ev.depth[sid], 0u);
    ev.claim(sid);
    if (!ev.expand.append(ExpandEntry{sid, 0u, 0u})) ds.errors.record(ErrorKind::kStatePoolFull);
}

// Record `s` on a session's frontier at `step`: the budget refused it and a continuation resumes
// from it. Past the capacity the entry is dropped and reported.
__device__ inline void session_frontier_append(DeviceState& ds, const SessionView& sess,
                                               StateId s, uint32_t step) {
    if (!sess.enabled) return;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> fc(*sess.frontier_count);
    const uint32_t at = fc.fetch_add(1u, cuda::memory_order_relaxed);
    if (at < sess.frontier_cap) {
        sess.frontier[at]      = s;
        sess.frontier_step[at] = step;
    } else {
        ds.errors.record(ErrorKind::kFrontierCapFull);
    }
}

// This engine's face for hgcommon/explore_depth_core.hpp, run by a block's thread 0 over the
// block's frame slice. A state admitted under the budget is claimed and appended to the expand
// log; at or past it, recorded on a session's frontier unclaimed.
struct DeviceExploreCtx {
    using Node = uint32_t;
    DeviceState&       ds;
    ExploreView&       ev;
    const SessionView& sess;
    uint32_t           max_steps;
    uint32_t*          frame_node;
    uint32_t*          frame_depth;
    uint32_t           levels;
    uint32_t           frames = 0;

    __device__ uint32_t depth_load(uint32_t s) const {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> d(ev.depth[s]);
        return d.load(cuda::memory_order_acquire);
    }
    __device__ bool depth_cas(uint32_t s, uint32_t& expected, uint32_t desired) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> d(ev.depth[s]);
        return d.compare_exchange_strong(expected, desired, cuda::memory_order_acq_rel,
                                         cuda::memory_order_acquire);
    }
    __device__ void children_push(uint32_t parent, uint32_t child) {
        if (ev.children.push(parent, child) == Pool<LockFreeList<StateId>::Node>::kInvalid)
            ds.errors.record(ErrorKind::kEventPoolFull);
    }
    __device__ Node children_head(uint32_t s) const { return ev.children.head_index(s); }
    __device__ static bool children_end(Node n) {
        return n == Pool<LockFreeList<StateId>::Node>::kInvalid;
    }
    __device__ uint32_t children_value(Node n) const { return ev.children.node(n)->value; }
    __device__ Node children_next(Node n) const { return ev.children.node(n)->next; }
    __device__ void fence() const { __threadfence(); }
    __device__ void admit(uint32_t s, uint32_t d) {
        if (d >= max_steps) { session_frontier_append(ds, sess, s, d); return; }
        if (!ev.claim(s)) return;
        if (!ev.expand.append(ExpandEntry{s, d, 0u})) ds.errors.record(ErrorKind::kStatePoolFull);
    }
    __device__ bool frame_push(Node at, uint32_t d) {
        if (frames == levels) return false;
        frame_node[frames] = at;
        frame_depth[frames] = d;
        ++frames;
        return true;
    }
    __device__ bool frame_top(Node*& at, uint32_t& d) {
        if (frames == 0) return false;
        at = &frame_node[frames - 1];
        d = frame_depth[frames - 1];
        return true;
    }
    __device__ void frame_pop() { --frames; }
};

// Records a claiming consumer may safely read. The pool's counter counts CLAIMS, and a claim
// past the end returns kInvalid without writing, so the counter can exceed the capacity while
// only the first `capacity` slots hold anything. Reading up to the raw counter would read past
// the allocation.
//
// Acquire for the detector: it pairs this with its acquire snapshot of pushed/completed, and a
// producer publishes its claim after the item it took was booked. A plain load there can pair a
// fresh value with a stale snapshot and pass both quiescence checks with work outstanding
// (verification/gpumc/evolve_ring_termination.cpp reports that execution). Relaxed for a
// claimer, which reads a record only after await_match's acquire on its published flag; an
// acquire load here invalidates the SM's L1 (CCTL.IVALL) on every poll of an idle block.
__device__ __forceinline__ uint32_t readable_records(
        const typename Pool<MatchRecord>::DeviceView& found,
        cuda::memory_order order = cuda::memory_order_acquire) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> cref(*found.counter);
    const uint32_t claimed = cref.load(order);
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
    uint32_t cur = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*cursor)
                       .load(cuda::memory_order_relaxed);
    for (;;) {
        if (cur >= readable_records(found, cuda::memory_order_relaxed)) return INVALID_ID;
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
        // The decision is hgcommon::term_detect_loop; this supplies only where the counters live
        // and what to do at the edges. The rewriting detector below drives the same body with a
        // different consumed-cursor and its own diagnostics.
        struct MatchDetectorCtx {
            typename TerminationDetector::DeviceView& term;
            typename Pool<MatchRecord>::DeviceView&   found;
            uint32_t*                                 consume_cursor;
            DeviceState&                              ds;

            HG_DEV uint32_t num_roles() const { return term.num_roles; }
            HG_DEV uint32_t max_stagnant_rounds() const { return kMaxDetectorRounds; }
            HG_DEV bool snapshot(uint64_t* p, uint64_t* c) const {
                return term.snapshot_quiescent(p, c);
            }
            HG_DEV uint32_t produced() const { return readable_records(found); }
            HG_DEV uint32_t consumed() const {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> r(*consume_cursor);
                return r.load(cuda::memory_order_acquire);
            }
            HG_DEV uint64_t work_progress() const { return 0; }
            HG_DEV void on_round(uint32_t, uint32_t, uint32_t) const {}
            HG_DEV void on_stall(uint32_t, const uint64_t*, const uint64_t*) const {
                ds.errors.record(ErrorKind::kPersistentStall);
            }
            HG_DEV void backoff_long() const { __nanosleep(4000); }
            HG_DEV void backoff_short() const { __nanosleep(2000); }
            HG_DEV void signal_exit() const { term.signal_exit(); }
        } dctx{term, found, consume_cursor, ds};

        hgcommon::term_detect_loop(dctx, p1, c1, p2, c2);
        return;
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
                const AppliedMatch a = apply_one_match(ds, rules, rec, rec.step + 1u);
                if (a.state != INVALID_ID) copy_kept_edges(ds, a.kept, hgcommon::IrSerial{});
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
        DedupMap::DeviceView exact_map,
        DeviceArena::View arena,
        typename TerminationDetector::DeviceView term,
        QcView qc,
        QeView qe,
        unsigned long long* phase_cycles,
        SessionView sess,
        ExploreView ev,
        typename Pool<uint32_t>::DeviceView forms) {

    // Ranks are the reconstruction's frame alignment, Automatic's signature, AND the transition
    // draw's key. One predicate answers it for the roots and for every child; see its note.
    const bool need_ranks = run_needs_edge_ranks(event_keys, qe.enabled != 0,
                                                 ds.transition_rate, ds.num_rule_weights,
                                                 ds.matches_per_state_rule);
    const bool need_exact = run_needs_exact_hash(event_keys, ds.transition_rate,
                                                 ds.num_rule_weights, ds.matches_per_state_rule);

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
        // The decision is hgcommon::term_detect_loop -- the same body the matching detector above
        // drives. This one counts consumed work with rewrites_done and carries the stall dump and
        // the periodic progress report, which are diagnostics rather than part of the decision.
        struct RewriteDetectorCtx {
            typename TerminationDetector::DeviceView& term;
            typename Pool<MatchRecord>::DeviceView&   found;
            uint32_t*                                 rewrites_done;
            DeviceState&                              ds;
            unsigned long long*                       phase_cycles;
            const uint32_t*                           replay_events;   // null without a replay
            // The replay's task log (QeView::tasks); null without a replay.
            const WorkLogView<QeTask>*                tasks;
            const WorkLogView<ExpandEntry>*           expands;

            HG_DEV uint32_t num_roles() const { return term.num_roles; }
            HG_DEV uint32_t max_stagnant_rounds() const { return kMaxDetectorRounds; }
            HG_DEV bool snapshot(uint64_t* p, uint64_t* c) const {
                return term.snapshot_quiescent(p, c);
            }
            // Records, expand entries and replay tasks together. No consumed count can pass its
            // produced count, so the sums are equal exactly when every pair is.
            HG_DEV uint32_t produced() const {
                uint32_t n = readable_records(found) + expands->readable();
                if (tasks) n += tasks->readable();
                return n;
            }
            HG_DEV uint32_t consumed() const {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> r(*rewrites_done);
                uint32_t n = r.load(cuda::memory_order_acquire) + expands->done_count();
                if (tasks) n += tasks->done_count();
                return n;
            }
            // The replay's raw events: the work a block does inline with no role booking it.
            HG_DEV uint64_t work_progress() const {
                if (!replay_events) return 0;
                cuda::atomic_ref<const uint32_t, cuda::thread_scope_device> r(*replay_events);
                return r.load(cuda::memory_order_relaxed);
            }
            HG_DEV void backoff_long() const { __nanosleep(4000); }
            HG_DEV void backoff_short() const { __nanosleep(2000); }
            HG_DEV void signal_exit() const { term.signal_exit(); }

            HG_DEV void on_stall(uint32_t round, const uint64_t* p1, const uint64_t* c1) const {
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
            }

            HG_DEV void on_round(uint32_t round, uint32_t prod1, uint32_t done1) const {

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
                    // The canon bucket's four parts, as fractions of the bucket. Without this the
                    // bucket reads as "canonicalization" while containing three other calls.
                    unsigned long long ir = 0, sg = 0, qe_ = 0, dd = 0;
                    if (phase_cycles) {
                        ir = phase_cycles[11]; sg = phase_cycles[12];
                        qe_ = phase_cycles[14]; dd = phase_cycles[15];
                    }
                    const unsigned long long cb = ir + sg + qe_ + dd;
                    printf("[hg_gpu PROGRESS] round=%u prod=%u done=%u | "
                           "match=%llu%% rewrite=%llu%% canonblk=%llu%% idle=%llu%% || "
                           "ir=%llu%% sig=%llu%% qe=%llu%% dedup=%llu%%\n",
                           round, prod1, done1,
                           tot ? 100ull * m  / tot : 0ull, tot ? 100ull * rw / tot : 0ull,
                           tot ? 100ull * cn / tot : 0ull, tot ? 100ull * id / tot : 0ull,
                           cb ? 100ull * ir  / cb : 0ull, cb ? 100ull * sg  / cb : 0ull,
                           cb ? 100ull * qe_ / cb : 0ull, cb ? 100ull * dd  / cb : 0ull);
                }
            }
        } dctx{term, found, rewrites_done, ds, phase_cycles,
               qe.enabled ? qe.next_raw_event : nullptr,
               (qe.enabled && qe.replay) ? &qe.tasks : nullptr, &ev.expand};

        hgcommon::term_detect_loop(dctx, p1, c1, p2, c2);
        return;
    }

    // Per-block IR scratch, carried across items: claimed on first use and re-claimed only
    // when a larger state arrives. Block-shared because the whole warp runs the
    // canonicalization together: the leader claims, every lane addresses the same slot.
    __shared__ uint32_t* ir_slot;
    __shared__ uint64_t  ir_slot_words;
    __shared__ MatchWorkItem mitem;
    __shared__ uint32_t task_base;
    __shared__ uint32_t task_count;
    __shared__ bool     have;
    __shared__ uint32_t claimed;
    __shared__ uint32_t child_sid;
    __shared__ uint32_t child_event;
    __shared__ uint32_t child_step;
    __shared__ KeptCopy child_kept;
    __shared__ uint32_t child_keyed;
    __shared__ StateId  child_parent;
    __shared__ bool     twin_taken;
    __shared__ uint64_t twin_h;
    __shared__ StateId  twin_rep;
    __shared__ bool     capture_go;
    __shared__ StateId  id_canonical;
    __shared__ bool     id_fresh;
    __shared__ uint64_t surv_shared[kLocalSurvivors];
    __shared__ uint32_t expand_base;
    __shared__ uint32_t expand_count;
    __shared__ bool     run_rule_inline;
    __shared__ bool     stalled;
    uint32_t idle_spins = 0;
    uint32_t idle_ns    = 64;   // thread 0's backoff state; reset whenever work is found

    // Phase attribution, accumulated in thread 0's registers and flushed once at exit so the
    // hot loop carries no extra atomics. See PersistentEvolveStats for what the four mean.
    unsigned long long acc_match = 0, acc_rewrite = 0, acc_canon = 0, acc_idle = 0,
                       acc_wait = 0;
    // Slots 11-15, the parts acc_canon spans, accumulated the same way and for the same reason.
    // Written in place they were five more global atomicAdds per RECORD, and the sixteen
    // counters are one 128-byte allocation, so every block's every record queued on one L2
    // line. The published totals are identical -- the same sums, added once per flush.
    unsigned long long acc_irkey = 0, acc_evkey = 0, acc_qe = 0, acc_dedup = 0;
    auto flush_cycles = [&] {
        if (threadIdx.x == 0 && phase_cycles) {
            atomicAdd(&phase_cycles[0], acc_match);
            atomicAdd(&phase_cycles[1], acc_rewrite);
            atomicAdd(&phase_cycles[2], acc_canon);
            atomicAdd(&phase_cycles[3], acc_idle);
            atomicAdd(&phase_cycles[4], acc_wait);
            atomicAdd(&phase_cycles[11], acc_irkey);
            atomicAdd(&phase_cycles[12], acc_evkey);
            atomicAdd(&phase_cycles[14], acc_qe);
            atomicAdd(&phase_cycles[15], acc_dedup);
            acc_match = acc_rewrite = acc_canon = acc_idle = acc_wait = 0;
            acc_irkey = acc_evkey = acc_qe = acc_dedup = 0;
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
                child_kept   = applied.kept;
                child_keyed  = applied.keyed;
                child_parent = rec.state_id;
                acc_rewrite += clock64() - t0b;
            }
            __syncthreads();
            // The child's kept edges, on every lane, before region 2 reads the child's slice.
            if (child_sid != INVALID_ID) copy_kept_edges(ds, child_kept, IrWarpAll{});
            __syncthreads();
            // Keyed rewrites (keyed.hpp): a child whose token set an earlier state holds takes that
            // state's canonical results, and the warp skips its IR. One thread checks; the flag is
            // shared so every lane takes the same branch below.
            if (threadIdx.x == 0) {
                twin_taken = false;
                if (child_sid != INVALID_ID && child_keyed != 0)
                    twin_taken = keyed_take_twin(ds, child_sid, child_parent, child_event,
                                                 child_keyed, arena, ir_slot, ir_slot_words,
                                                 need_ranks, qc.enabled != 0, dedup_map, forms,
                                                 twin_h, twin_rep);
            }
            __syncthreads();
            // Region 2 of the record. The two canonicalizations run on the WHOLE warp: every
            // lane enters the shared core together under the all-lanes policy, and the block
            // is one warp (kMatchBlockThreads), so the collectives' full mask holds. The
            // branches here read only shared or uniform values, so the lanes stay converged.
            // Downstream of the hashes, publishing and signatures are thread 0's, the expansion
            // capture runs on every lane, and identity and depth are thread 0's.
            {
                const unsigned long long t1 = clock64();
                // SPLIT THE canon BUCKET INTO ITS PARTS.
                //
                // acc_canon spans this whole region, so it has been reporting
                // "canonicalization" for a span that also stamps event signatures, drives
                // the quotient causal DP, captures the class-frame expansion and consults
                // dedup. A 99% reading was taken to mean individualization-refinement and does
                // not: an isolated measurement puts device IR at 62.9x the host on one state,
                // not the thousands the whole-block figure implied. Slots 11-15 name the
                // parts so the next question is asked of the right one.
                uint64_t h = 0;
                uint32_t* form = nullptr;
                uint32_t form_words = 0;
                ExactHashStatus key_st = ExactHashStatus::kOk;
                if (child_sid != INVALID_ID && !twin_taken) {
                    key_st = state_key_device(ds, child_sid, state_mode, arena, ir_slot,
                                              ir_slot_words, h, need_ranks, qc.enabled != 0,
                                              &form, &form_words, IrWarpAll{});
                }
                if (threadIdx.x == 0) acc_irkey += clock64() - t1;

                // The exact isomorphism hash is a different question from the mode's key and
                // coincides with it only in Full. Computed only when an event identity or a
                // transition key will read it (run_needs_exact_hash).
                uint64_t exact = h;
                ExactHashStatus ex_st = ExactHashStatus::kOk;
                uint32_t* eform = nullptr;
                uint32_t eform_words = 0;
                if (child_sid != INVALID_ID && key_st == ExactHashStatus::kOk && need_exact &&
                    state_mode != CanonicalizationMode::Full) {
                    ex_st = state_exact_hash_device(ds, child_sid, arena, ir_slot,
                                                    ir_slot_words, exact, need_ranks,
                                                    false, &eform, &eform_words, IrWarpAll{});
                }

                if (threadIdx.x == 0) {
                const MatchRecord& rec = found.at(claimed);
                const uint32_t step = rec.step;
                capture_go = false;

                // Expand the child only if it exists, the step budget allows it, its exact
                // hash is computable, and the exploration rule keeps it. The hash is the
                // dedup KEY, so a state whose hash could not be computed is not enqueued
                // under a coarser one -- 1-WL merges non-isomorphic states.
                if (child_sid != INVALID_ID) {
                    if (key_st != ExactHashStatus::kOk) {
                        ds.errors.record(error_kind_for(key_st));
                    } else {
                        // IDENTITY FIRST. In Full mode the class's key, claimed on the
                        // canonical form (state_claim_full), is the state's canonical hash, so it
                        // is claimed before the hash is published and everything keyed by the
                        // hash -- event identity, the quotient's classes -- reads the key.
                        if (twin_taken) {
                            h = twin_h;
                            exact = h;
                            id_canonical = dedup ? twin_rep : child_sid;
                            id_fresh = !dedup;
                        } else if (state_mode == CanonicalizationMode::Full) {
                            const StateClaim c =
                                state_claim_form(ds, child_sid, h & ds.canonical_key_mask, form,
                                                 form_words, dedup_map, forms);
                            h = c.key;
                            exact = h;
                            id_canonical = dedup ? c.canonical : child_sid;
                            id_fresh = dedup ? c.fresh : true;
                        } else if (state_mode == CanonicalizationMode::Automatic) {
                            const StateClaim c = state_claim_content(ds, child_sid, h, dedup_map);
                            h = c.key;
                            id_canonical = dedup ? c.canonical : child_sid;
                            id_fresh = dedup ? c.fresh : true;
                        } else {
                            const StateIdentity id =
                                state_identity(ds, child_sid, h, dedup_map, dedup);
                            id_canonical = id.canonical;
                            id_fresh = id.fresh;
                        }
                        // Publish before anything reads it: a transition OUT of this state
                        // needs it as an input hash, and that read happens on another block.
                        ds.state_canonical_hash[child_sid] = h;

                        if (need_exact) {
                            if (ex_st != ExactHashStatus::kOk) {
                                ds.errors.record(error_kind_for(ex_st));
                                exact = 0;
                            } else if (state_mode != CanonicalizationMode::Full) {
                                // The exact hash event identity reads, claimed on the IR form
                                // (the host's event_canonical_state_map_).
                                exact = state_claim_form(ds, child_sid, exact & ds.event_key_mask,
                                                         eform, eform_words, exact_map,
                                                         forms).key;
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
                            stamp_event_signature(ds, child_event, event_keys, event_map);
                            acc_evkey += clock64() - s1;
                        }

                        // Quotient causal: EVERY raw event registers its canonical transition,
                        // whether or not the child survives dedup below -- the host registers
                        // per raw event too. Both endpoint hashes and orbit tables exist at this
                        // point (the parent's from its own canon, the child's from the pass just
                        // above). The capture runs on every lane, between this part and the next.
                        capture_go = child_event != INVALID_ID;
                    }
                }
                } // threadIdx.x == 0
                __syncthreads();
                // The class frame's match record (qe_capture_expansion), on every lane. The
                // block's slice of the survivor scratch is indexed by blockIdx.
                if (capture_go) {
                    const MatchRecord& rec = found.at(claimed);
                    const unsigned long long s3 = (threadIdx.x == 0) ? clock64() : 0;
                    qe_capture_expansion(ds, qe, rec.state_id, child_sid, child_event,
                                         rec.rule_id, rec.step, blockIdx.x, surv_shared);
                    if (threadIdx.x == 0) acc_qe += clock64() - s3;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                const MatchRecord& rec = found.at(claimed);
                if (child_sid != INVALID_ID && key_st == ExactHashStatus::kOk) {
                        // IDENTITY, THEN DEPTH (explore_depth.hpp). The first arrival of a key
                        // is its canonical state; a fresh state the exploration coin or a cap
                        // refuses, under the budget or in a session, is claimed unexpanded, so
                        // no later path expands it. Every arrival registers under its parent,
                        // and one that lowers the canonical state's depth admits it and lowers
                        // its descendants.
                        {
                            const uint64_t s4 = clock64();
                            const StateIdentity id{id_canonical, id_fresh};
                            if (id.fresh && (child_step < max_steps || sess.enabled) &&
                                !state_retained(ds, child_sid, child_step, rec.state_id,
                                                explore_threshold_u32, explore_seed))
                                ev.claim(id.canonical);
                            DeviceExploreCtx xc{ds, ev, sess, max_steps,
                                                ev.frame_node + size_t(blockIdx.x) * ev.frame_levels,
                                                ev.frame_depth + size_t(blockIdx.x) * ev.frame_levels,
                                                ev.frame_levels};
                            const uint32_t d = hgcommon::explore_register_child(
                                xc, rec.state_id, id.canonical, child_step);
                            if (d != hgcommon::kExploreNoDepth) {
                                xc.admit(id.canonical, d);
                                if (!hgcommon::explore_relax(xc, id.canonical, d))
                                    ds.errors.record(ErrorKind::kScratchOverflow);
                            }
                            acc_dedup += clock64() - s4;
                        }
                }
                acc_canon += clock64() - t1;
                } // threadIdx.x == 0
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(rewrites_done, 1u);
                // 1024, which is what the detector's note beside the progress print already
                // states this to be. A flush is ten atomics on one 128-byte line, shared by
                // every block, so at eight it cost 1.25 per record -- and the reason the
                // interval exists at all is that a run which never finishes is still
                // attributable, which 1024 serves exactly as well as 8. A block leaving the
                // loop flushes on the way out either way (exit_requested, stalled), so a run
                // shorter than the interval loses nothing.
                if (++records_since_flush >= 1024u) {
                    flush_cycles();
                    records_since_flush = 0;
                }
            }
            __syncthreads();
            continue;
        }

        // EXPAND ENTRIES (ExploreView::expand), one per block: push the state's match items,
        // matching one on this block when the ring is full. Booked after every push.
        if (threadIdx.x == 0) {
            expand_count = ev.expand.claim(1u, expand_base);
            if (expand_count) {
                const ExpandEntry& e = ev.expand.await(expand_base);
                child_sid  = e.state;
                child_step = e.depth;
            }
        }
        __syncthreads();
        if (expand_count) {
            for (uint32_t r = 0; r < num_rules; ++r) {
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
                ev.expand.book(1u);
                idle_ns = 64;
                idle_spins = 0;
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

        // REPLAY TASKS, a warp at a time (QeView::tasks). Lane 0 claims up to 32 consecutive
        // tasks with one CAS on the cursor, each lane runs one, and lane 0 books how many ran
        // after every lane's work, its appends included, is fenced. The block is one warp.
        if (qe.enabled && qe.replay) {
            if (threadIdx.x == 0) task_count = qe.tasks.claim(kMatchBlockThreads, task_base);
            __syncthreads();
            if (task_count) {
                const unsigned long long tQ = (threadIdx.x == 0) ? clock64() : 0;
                if (threadIdx.x < task_count) {
                    const QeTask& t = qe.tasks.await(task_base + threadIdx.x);
                    qe_apply(ds, qe, qe.instances.at(t.rec), qe.matches.at(t.match), t.hash,
                             t.depth);
                    __threadfence();
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    qe.tasks.book(task_count);
                    idle_ns = 64;
                    idle_spins = 0;
                    acc_canon += clock64() - tQ;
                    acc_qe    += clock64() - tQ;
                }
                __syncthreads();
                continue;
            }
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

// The RingBuffer's sequence init; declared in ring_buffer.hpp, whose class cannot hold a
// kernel because the header reaches host-only translation units. External linkage: every
// RingBuffer constructor everywhere calls this.
void ring_seq_ramp_device(uint64_t* seq, uint32_t n) {
    const uint32_t block = 256;
    k_seq_ramp<<<(n + block - 1) / block, block>>>(seq, n);
    HG_CUDA_CHECK(cudaGetLastError(), "ring seq ramp launch");
}

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

// A launch's ring, dedup maps and detector. Built on first use at the sizes the launch asks for
// and rebuilt only when a later launch asks for a different size; each launch clears what it
// takes. A PersistentEvolver's config never shrinks, so after its first run every launch reuses
// them and makes no cudaMalloc or cudaFree call for them (13 per run before, 0.26 ms of the
// 3.7 ms floor of wpp at 2 steps being the maps' and ring's frees alone).
struct EngineState::PersistentScratch {
    std::unique_ptr<RingBuffer<MatchWorkItem>> ring;
    std::unique_ptr<DedupMap> canonical;
    std::unique_ptr<DedupMap> event_ids;
    std::unique_ptr<TerminationDetector> term;
    std::unique_ptr<ExploreState> explore;
    std::unique_ptr<Pool<uint32_t>> forms;   // canonical-form records (state_claim_form)
    std::unique_ptr<DedupMap> exact;         // exact hash -> record, under None and Automatic
    std::unique_ptr<DedupMap> keyed_rewrites;   // KeyedView::rewrites
    std::unique_ptr<DedupMap> keyed_twins;      // KeyedView::twins
    uint32_t* keyed_words = nullptr;             // KeyedView::words
    uint32_t* explore_frames = nullptr;
    size_t    explore_frame_words = 0;
    ~PersistentScratch() {
        if (explore_frames) cudaFree(explore_frames);
        if (keyed_words) cudaFree(keyed_words);
    }
};

void EngineState::PersistentScratchFree::operator()(PersistentScratch* p) const { delete p; }

EngineState::PersistentScratch& EngineState::persistent_scratch() const {
    if (!persistent_scratch_) persistent_scratch_.reset(new PersistentScratch);
    return *persistent_scratch_;
}

namespace {

template <class T>
RingBuffer<T>& reuse_ring(std::unique_ptr<RingBuffer<T>>& slot, uint32_t capacity) {
    if (slot && slot->capacity() == capacity) slot->clear();
    else { slot.reset(); slot = std::make_unique<RingBuffer<T>>(capacity); }
    return *slot;
}

RingBuffer<MatchWorkItem>& reuse_ring(const EngineState& engine, uint32_t capacity) {
    return reuse_ring(engine.persistent_scratch().ring, capacity);
}

DedupMap& reuse_map(std::unique_ptr<DedupMap>& slot, uint32_t capacity) {
    if (slot && slot->capacity() == capacity) slot->clear();
    else { slot.reset(); slot = std::make_unique<DedupMap>(capacity); }
    return *slot;
}

TerminationDetector& reuse_term(const EngineState& engine, uint32_t num_roles = 1) {
    auto& slot = engine.persistent_scratch().term;
    if (slot && slot->num_roles() == num_roles) slot->clear();
    else { slot.reset(); slot = std::make_unique<TerminationDetector>(num_roles); }
    return *slot;
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

    // Engine-lifetime grow-only scratch: allocating these per call was API-call overhead on
    // the per-call floor.
    EngineState::LaunchScratch& sc =
        engine.launch_scratch(num_rules, static_cast<uint32_t>(states.size()));
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice, 0), "rules copy");
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice, 0), "states copy");

    uint32_t cap = 2;
    while (cap < num_items) cap <<= 1;
    RingBuffer<MatchWorkItem>& queue = reuse_ring(engine, cap);

    {
        const uint32_t block = 128;
        const uint32_t seed_grid = (num_items + block - 1) / block;
        k_seed_match_queue<<<seed_grid, block>>>(queue.view(), sc.states,
                                                 static_cast<uint32_t>(states.size()), num_rules,
                                                 /*step=*/0u);
        HG_CUDA_CHECK(cudaDeviceSynchronize(), "seed sync");
    }

    // Deliberately FEWER blocks than items: each one loops, which is the whole difference from
    // launching one block per item.
    const uint32_t grid = blocks ? blocks : 64;

    k_persistent_match<<<grid, kMatchBlockThreads>>>(engine.device(), sc.rules,
                                                     queue.view(), out.view());
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "persistent match sync");
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

    // Engine-lifetime grow-only scratch; see run_persistent_match.
    EngineState::LaunchScratch& sc =
        engine.launch_scratch(num_rules, static_cast<uint32_t>(states.size()));
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice, 0), "rules copy");
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice, 0), "states copy");

    uint32_t cap = 2;
    while (cap < num_items) cap <<= 1;
    RingBuffer<MatchWorkItem>& match_q = reuse_ring(engine, cap);
    {
        const uint32_t block = 128;
        const uint32_t seed_grid = (num_items + block - 1) / block;
        k_seed_match_queue<<<seed_grid, block>>>(match_q.view(), sc.states,
                                                 static_cast<uint32_t>(states.size()), num_rules,
                                                 step);
        HG_CUDA_CHECK(cudaDeviceSynchronize(), "seed sync");
    }

    HG_CUDA_CHECK(cudaMemset(sc.cursor, 0, sizeof(uint32_t)), "cursor clear");

    TerminationDetector& term = reuse_term(engine);
    term.mark_pushed_host(kRoleMatch, num_items);

    // Block 0 is the detector, so at least two blocks are needed for any work to happen.
    const uint32_t grid_req = blocks ? blocks : default_persistent_grid();
    const uint32_t grid = grid_req < 2 ? 2 : grid_req;
    k_persistent_match_rewrite<<<grid, kMatchBlockThreads>>>(
        engine.device(), sc.rules, match_q.view(), scratch_matches.view(),
        sc.cursor, term.view(), step);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "persistent match+rewrite sync");

    stats.matches_found = scratch_matches.size_host();
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
                                            const QcView* qc_in,
                                            const QeView* qe_in,
                                            SessionView* session,
                                            uint32_t start_step,
                                            bool read_stats) {
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

    // Engine-lifetime grow-only scratch; see run_persistent_match.
    EngineState::LaunchScratch& sc =
        engine.launch_scratch(num_rules, static_cast<uint32_t>(roots.size()));
    DeviceRule* d_rules = sc.rules;
    // Async from pageable memory, as upload_initial_states: staged on return, ordered before
    // the kernels by the stream.
    HG_CUDA_CHECK(cudaMemcpyAsync(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice, 0), "rules copy");

    StateId* d_states = sc.states;
    HG_CUDA_CHECK(cudaMemcpyAsync(d_states, roots.data(), sizeof(StateId) * roots.size(),
                     cudaMemcpyHostToDevice, 0), "states copy");

    // The ring holds work in flight, not the whole evolution: a run that outgrows it does not
    // fail, it runs the excess inline on the pushing block. Sized to the match pool so the
    // inline path is an escape valve rather than the normal case.
    uint32_t cap = 2;
    while (cap < num_seed) cap <<= 1;
    while (cap < scratch_matches.capacity() && cap < (1u << 20)) cap <<= 1;
    RingBuffer<MatchWorkItem>& match_q = reuse_ring(engine, cap);

    // The canonical map is the dedup key store for the whole run. Sized to the state pool: one
    // entry per state is the worst case, and the map must not fill, because a full map would
    // silently start admitting duplicates.
    // A SESSION OWNS ITS IDENTITY. Rebuilt per call otherwise, which is what a one-shot run
    // wants and what makes a second call re-derive everything as new.
    SessionView sess_v{};
    if (session) sess_v = *session;
    const bool dbgt = std::getenv("HG_GPU_DBG_TIME") != nullptr;
    auto t_maps0 = std::chrono::steady_clock::now();
    EngineState::PersistentScratch& ps = engine.persistent_scratch();

    // Canonical-form records for Full-mode identity and, under None and Automatic, for the exact
    // hash event identity reads (state_claim_form). A session keeps its own across calls, as it
    // keeps its dedup map; a one-shot run starts empty.
    typename Pool<uint32_t>::DeviceView forms_v;
    if (session) {
        forms_v = sess_v.forms;
    } else {
        const uint32_t fw = engine.config().canonical_form_words;
        if (ps.forms && ps.forms->capacity() == fw) ps.forms->reset();
        else ps.forms = std::make_unique<Pool<uint32_t>>(fw);
        forms_v = ps.forms->view();
    }

    // Exploration depth (explore_depth.hpp). A session keeps its depths, claims and child lists
    // across calls and consumes its expand log per call; a one-shot run starts from nothing.
    ExploreView ev;
    if (session) {
        ev = sess_v.explore;
        explore_reset_async(ev, /*full=*/false);
    } else {
        const uint32_t ms = engine.config().max_states, me = engine.config().max_events;
        if (ps.explore && ps.explore->max_states() == ms && ps.explore->max_events() == me)
            ps.explore->clear();
        else
            ps.explore = std::make_unique<ExploreState>(ms, me);
        ev = ps.explore->view();
    }
    // The walk's frames: one per level it can descend, per block. A frame is pushed only for a
    // state the walk lowered, one level deeper than the last, and only states under the budget
    // have children, so max_steps + 2 bounds a walk.
    const uint32_t grid_req = blocks ? blocks : default_persistent_grid();
    const uint32_t grid = grid_req < 2 ? 2 : grid_req;
    {
        const uint32_t levels = max_steps + 2u;
        const size_t words = size_t(grid) * levels * 2u;
        if (ps.explore_frame_words < words) {
            if (ps.explore_frames) cudaFree(ps.explore_frames);
            HG_CUDA_CHECK(cudaMalloc(&ps.explore_frames, sizeof(uint32_t) * words),
                          "explore frames alloc");
            ps.explore_frame_words = words;
        }
        ev.frame_node   = ps.explore_frames;
        ev.frame_depth  = ps.explore_frames + size_t(grid) * levels;
        ev.frame_levels = levels;
    }
    DedupMap* canonical_owner =
        session ? nullptr : &reuse_map(ps.canonical, engine.config().max_states * 2u);

    // Signature -> first event with it. Sized off the event budget rather than the state one:
    // an evolution has as many applications as it has matches, which is not bounded by its
    // state count.
    //
    // Sized to nothing when no event identity is being computed. At the default config this map
    // is 2^18 slots, and allocating plus clearing it is milliseconds on runs that take tens --
    // a cost charged to every run for a mode most do not select. The stamp sites are all behind
    // `event_keys != EVENT_SIG_NONE`, so the small map is never touched.
    const bool want_event_ids = (event_keys != EVENT_SIG_NONE);
    DedupMap* owned_event_ids =
        session ? nullptr
                : &reuse_map(ps.event_ids, want_event_ids ? engine.config().max_events * 2u : 8u);
    if (want_event_ids) engine.ensure_event_identity();

    // Exact hash -> record, claimed under None and Automatic when an event identity or a
    // transition key reads the exact hash; eight slots otherwise, never touched.
    DedupMap::DeviceView exact_v;
    if (session) {
        exact_v = sess_v.exact;
    } else {
        const DeviceState dsx = engine.device();
        const bool want_exact =
            state_mode != CanonicalizationMode::Full &&
            run_needs_exact_hash(event_keys, dsx.transition_rate, dsx.num_rule_weights,
                                 dsx.matches_per_state_rule);
        exact_v = reuse_map(ps.exact, want_exact ? engine.config().max_states * 2u : 8u).view();
    }

    // Keyed rewrites (keyed.hpp), when hgcommon::keyed_rewrites_apply admits the run: the
    // DeviceState the kernels are launched with carries the rewrite and twin maps, sized so neither
    // fills (a rewrite per event, a twin claim per state), and the state words. A session keeps its
    // own across calls; a one-shot run starts empty and ARMED.
    DeviceState dsk = engine.device();
    const bool keyed = hgcommon::keyed_rewrites_apply(
        engine.config().keyed_rewrites, state_mode == CanonicalizationMode::Full,
        /*positional_events=*/false,
        hgcommon::run_reads_rank_tuples(event_keys, dsk.transition_rate, dsk.num_rule_weights,
                                        dsk.matches_per_state_rule));
    if (keyed) {
        engine.ensure_keyed();
        dsk = engine.device();
        if (session) {
            dsk.keyed.rewrites = sess_v.keyed_rewrites;
            dsk.keyed.twins = sess_v.keyed_twins;
            dsk.keyed.words = sess_v.keyed_words;
        } else {
            if (!ps.keyed_words)
                HG_CUDA_CHECK(cudaMalloc(&ps.keyed_words, sizeof(uint32_t) * 4), "keyed words alloc");
            const uint32_t init[4] = {KEYED_ARMED, 0, 0, 0};
            HG_CUDA_CHECK(cudaMemcpyAsync(ps.keyed_words, init, sizeof(init),
                                          cudaMemcpyHostToDevice, 0),
                          "keyed words init");
            dsk.keyed.rewrites =
                reuse_map(ps.keyed_rewrites, engine.config().max_events * 2u).view();
            dsk.keyed.twins = reuse_map(ps.keyed_twins, engine.config().max_states * 2u).view();
            dsk.keyed.words = ps.keyed_words;
        }
        dsk.keyed.claim_limit = engine.config().keyed_claim_limit;
        dsk.keyed.sum_mask = engine.config().keyed_sum_mask;
        dsk.keyed.enabled = 1;
    }

    const double t_maps = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_maps0).count();
    auto t_alloc0 = std::chrono::steady_clock::now();

    // Every buffer is taken HERE, before the first kernel goes out: the scratch's rare grow
    // path calls cudaMalloc, which may synchronize the device, and the evolution's contract
    // is memory traffic at the start and end only, with ONE synchronization -- after the last
    // kernel.
    uint32_t* d_cursor = sc.cursor;
    HG_CUDA_CHECK(cudaMemset(d_cursor, 0, sizeof(uint32_t) * 2), "cursor clear");
    uint32_t* d_rewrites_done = d_cursor + 1;

    // 5 top-level phases + apply_one_match's 6 sub-stretches (see rewrite.hpp).
    unsigned long long* d_phase_cycles = sc.phase_cycles;
    HG_CUDA_CHECK(cudaMemset(d_phase_cycles, 0, sizeof(unsigned long long) * 16), "phase cycles clear");

    TerminationDetector& term = reuse_term(engine);

    // The whole evolution is a launch CHAIN on one stream: root hashing appends each root to
    // the expand log, which the detector counts, and the evolve kernel consumes it. Stream order
    // carries every dependency, so the host synchronizes exactly once, after the last kernel,
    // and reads nothing back before that.
    const double t_alloc = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_alloc0).count();
    auto t_seed0 = std::chrono::steady_clock::now();

    arena.reset();

    // THE ENGINE MUST FIT IN PHYSICAL VRAM. Under WDDM (Windows, WSL) cudaMalloc does not fail
    // past the device's memory: the driver pages device memory to system memory and every
    // kernel runs about 100x slower (measured: bigpath n=112 at 3 steps, 23,902 of 24,564 MiB
    // used, over 300 s where n=96 took 2.5 s). Every allocation of this run is done here, so a
    // free amount under 1/64 of the device means it did not fit; the run stops and the caller
    // gets the last partial result (run_with_growth, kDeviceOutOfMemory).
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b != 0 &&
            free_b < total_b / 64) {
            throw std::runtime_error(
                "the engine does not fit in device memory (" + std::to_string(free_b >> 20) +
                " of " + std::to_string(total_b >> 20) + " MiB free after allocation)");
        }
        cudaGetLastError();
    }
    // CONTINUING rather than starting: the frontier already holds hashed, deduplicated states,
    // so it is seeded straight into the queue at each entry's own recorded depth. The
    // root path would re-hash them and, worse, consult dedup -- which they already satisfy, so
    // nothing would expand.
    if (start_step > 0 && session) {
        // A continuation raised the depth bound: drive what the old bound left standing. On the
        // default stream, so it completes before the frontier below is expanded; the replay's
        // rendezvous makes the order of the two irrelevant to the answer.
        if (qe.enabled && (qe.replay || qe.multiplicity) && qe.work_slices) {
            const uint32_t rblock = 64;
            k_qe_redrive<<<(qe.work_slices + rblock - 1) / rblock, rblock>>>(
                engine.device(), qe, start_step);
        }
        const uint32_t block = 128;
        const uint32_t seed_grid = (sess_v.frontier_cap + block - 1) / block;
        if (seed_grid) {
            k_seed_frontier<<<seed_grid, block>>>(
                engine.device(), ev, sess_v.frontier, sess_v.frontier_step,
                sess_v.frontier_count, sess_v.frontier_cap);
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
        const DeviceState dsv = dsk;
        k_seed_root_hashes<<<(n + block - 1) / block, block>>>(
            dsv, d_states, n,
            session ? sess_v.states : canonical_owner->view(), state_mode,
            run_needs_exact_hash(event_keys, dsv.transition_rate, dsv.num_rule_weights,
                                 dsv.matches_per_state_rule),
            run_needs_edge_ranks(event_keys, qe.enabled != 0, dsv.transition_rate,
                                 dsv.num_rule_weights, dsv.matches_per_state_rule),
            arena.view(), qc, qe, ev, forms_v, exact_v);
    }

    // Block 0 is the detector, so at least two blocks are needed for any work to happen.
    const double t_seed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_seed0).count();
    if (dbgt)
        std::fprintf(stderr, "[persistent setup] dedup_maps=%.2f allocs=%.2f seed=%.2f (ms)\n",
                     t_maps, t_alloc, t_seed);

    k_persistent_evolve<<<grid, kMatchBlockThreads>>>(
        dsk, d_rules, num_rules, match_q.view(), scratch_matches.view(),
        d_cursor, d_rewrites_done,
        session ? sess_v.states : canonical_owner->view(), dedup,
        explore_threshold_u32, explore_seed, max_steps, state_mode, event_keys,
        session ? sess_v.events : owned_event_ids->view(), exact_v,
        arena.view(), term.view(), qc, qe, d_phase_cycles, sess_v, ev, forms_v);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "persistent evolve sync");
    if (!read_stats) return stats;

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
    for (int i = 0; i < 5; ++i) stats.cycles_canon_sub[i] = phase[11 + i];
    if (keyed)
        HG_CUDA_CHECK(cudaMemcpy(&stats.keyed_twins, dsk.keyed.words + 3, sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost),
                      "keyed twins read");

    // The buffers are the engine's grow-only launch scratch and outlive this run.
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

// ---- ExploreState ------------------------------------------------------------------------

// Zero the published flags of the entries the last launch appended; the counter is read on the
// device, so the host never learns how many there were.
__global__ void k_expand_log_clear(ExploreView v) {
    const uint32_t n = min(*v.expand.items.counter, v.expand.items.capacity);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        v.expand.items.at(i).published = 0u;
}

void explore_reset_async(const ExploreView& v, bool full) {
    k_expand_log_clear<<<64, 256>>>(v);
    HG_CUDA_CHECK(cudaMemsetAsync(v.expand.items.counter, 0, sizeof(uint32_t)),
                  "expand log counter clear");
    HG_CUDA_CHECK(cudaMemsetAsync(v.expand.cursor, 0, sizeof(uint32_t) * 2u),
                  "expand log cursor clear");
    if (!full) return;
    HG_CUDA_CHECK(cudaMemsetAsync(v.depth, 0xFF, sizeof(uint32_t) * v.max_states),
                  "explore depth clear");
    HG_CUDA_CHECK(cudaMemsetAsync(v.claimed, 0, sizeof(uint32_t) * v.max_states),
                  "explore claim clear");
    HG_CUDA_CHECK(cudaMemsetAsync(v.children.heads, 0xFF, sizeof(uint32_t) * v.children.num_keys),
                  "explore child heads clear");
    HG_CUDA_CHECK(cudaMemsetAsync(v.children.pool.counter, 0, sizeof(uint32_t)),
                  "explore child pool clear");
}

ExploreState::ExploreState(uint32_t max_states, uint32_t max_events)
    : max_states_(max_states), max_events_(max_events),
      children_(max_states, max_events), expand_(max_states) {
    HG_CUDA_CHECK(cudaMalloc(&words_, sizeof(uint32_t) * (2ull * max_states + 2u)),
                  "explore words alloc");
    HG_CUDA_CHECK(cudaMemset(expand_.view().data, 0, sizeof(ExpandEntry) * size_t(max_states)),
                  "expand log init");
    clear();
}

ExploreState::~ExploreState() { if (words_) cudaFree(words_); }

void ExploreState::clear() { explore_reset_async(view(), /*full=*/true); }

ExploreView ExploreState::view() const {
    ExploreView v;
    v.depth         = words_;
    v.claimed       = words_ + max_states_;
    v.children      = children_.view();
    v.expand.items  = expand_.view();
    v.expand.cursor = words_ + 2ull * max_states_;
    v.expand.done   = words_ + 2ull * max_states_ + 1u;
    v.max_states    = max_states_;
    return v;
}

SessionState::SessionState(uint32_t max_states, uint32_t max_events): states_(max_states * 2u), events_(max_events * 2u), explore_(max_states, max_events), forms_(max_states * 32u), exact_(max_states * 2u), keyed_rewrites_(max_events * 2u), keyed_twins_(max_states * 2u), cap_(max_states) {
        states_.clear();
        events_.clear();
        exact_.clear();
        keyed_rewrites_.clear();
        keyed_twins_.clear();
        HG_CUDA_CHECK(cudaMalloc(&keyed_words_, sizeof(uint32_t) * 4), "session keyed words alloc");
        const uint32_t keyed_init[4] = {KEYED_ARMED, 0, 0, 0};
        HG_CUDA_CHECK(cudaMemcpy(keyed_words_, keyed_init, sizeof(keyed_init), cudaMemcpyHostToDevice),
                      "session keyed words init");
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
        if (keyed_words_) cudaFree(keyed_words_);
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
        v.explore        = explore_.view();
        v.forms          = forms_.view();
        v.exact          = exact_.view();
        v.keyed_rewrites = keyed_rewrites_.view();
        v.keyed_twins    = keyed_twins_.view();
        v.keyed_words    = keyed_words_;
        return v;
    }

}  // namespace gpu
}  // namespace HG_NAMESPACE
