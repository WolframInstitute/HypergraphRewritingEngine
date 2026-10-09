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

// MaxStatesPerStep: the device twin of the host's select_step. Under the cap a step's
// transitions are held in a candidate pool until every piece of work that can still produce one
// has finished, and then the N lowest-ranked are appended to the rewrite pool. ds.step_pending[s]
// counts that work for step s: the token of step s - 1's selection, its selected rewrites (until
// each child is registered), step s's expand entries and its match items. Each unit is booked
// before the work it stands for can be seen and released after the work is done, so the count
// reaches zero once, after the last of them. Whichever block releases the last unit runs the
// selection, before it reports its own work done, so termination cannot be detected while
// candidates are held. Nothing waits.
// A full-capture run that explores with a probability below 1 under None or Automatic states keys
// its coin on the creating transition (explore_key), so it reads the parent's ranks and exact
// hash. Under Full states the coin reads the child's class hash.
__host__ __device__ inline uint32_t explore_reads_ranks(const DeviceState& ds, bool dedup,
                                                        CanonicalizationMode state_mode) {
    return (ds.exploration_probability < 1.0 && !dedup &&
            state_mode != CanonicalizationMode::Full) ? 1u : 0u;
}

struct StepSelectScratch {
    uint64_t* rank;    // [cap] the step's candidate ranks
    uint32_t* idx;     // [cap] their candidate-pool indices
    uint32_t* words;   // [2] candidates gathered, scan start
    uint32_t  cap;
};

__device__ inline void step_book(const DeviceState& ds, uint32_t d, uint32_t n) {
    if (ds.max_states_per_step != 0u && d < ds.step_slots) atomicAdd(&ds.step_pending[d], n);
}
// Under quotient exploration the ExplorationProbability coin is drawn when a class is claimed for
// expansion, keyed on its canonical hash, whichever path claims it: the host's
// claim_canonical_for_expansion. A refused class stays claimed and is not expanded.
__device__ inline bool explore_admits(const DeviceState& ds, StateId canonical) {
    return ds.exploration_probability >= 1.0 ||
           hgcommon::explore_survives(ds.state_canonical_hash[canonical], ds.sampling_seed,
                                      ds.exploration_probability);
}

// True when this release brings step d's count to zero.
__device__ inline bool step_release(const DeviceState& ds, uint32_t d, uint32_t n) {
    if (ds.max_states_per_step == 0u || d >= ds.step_slots) return false;
    __threadfence();   // the released work's effects before the count that admits the selection
    return atomicSub(&ds.step_pending[d], n) == n;
}

// The steps below the first one with work hold tokens nothing would release: their selections
// have no candidates. Released from the bottom, before the persistent loop starts.
__global__ void k_step_release_empty(const __grid_constant__ DeviceState ds) {
    for (uint32_t s = 0; s + 1 < ds.step_slots && ds.step_pending[s] == 0u; ++s)
        ds.step_pending[s + 1] -= 1u;
}

// Selection of step s on the calling block: rank the step's candidates, find the N-th smallest
// rank by radix selection, and append every candidate ranked at or below it to `found`: the N
// lowest and every candidate tied with the N-th (hgcommon::rank_cut_keeps, the host's
// cap_keep_count). The selected rewrites are booked on step s + 1 before any of them is visible.
__device__ __noinline__ void select_step(const DeviceState& ds, uint32_t s,
                                         typename Pool<MatchRecord>::DeviceView cand,
                                         typename Pool<MatchRecord>::DeviceView found,
                                         StepSelectScratch sel) {
    const uint32_t tid = threadIdx.x, nt = blockDim.x;
    __shared__ uint32_t s_cnt;
    __shared__ uint32_t s_remaining;
    __shared__ uint64_t s_prefix;
    // THE SCAN STARTS AT sel.words[1]: below it every candidate belongs to a step already
    // selected. A candidate of step s + 1 or later is produced from a state step s's selection
    // creates, after this scan reads the counter, or was seeded before the run (a session
    // frontier at mixed depths); the lowest index of those this scan passes is where the next
    // one starts, so a step's scan covers what was produced since the selection before it.
    __shared__ uint32_t s_next_start;
    const uint32_t start = sel.words[1];
    const uint32_t n = min(*cand.counter, cand.capacity);
    if (tid == 0) { sel.words[0] = 0u; s_next_start = n; }
    __syncthreads();
    for (uint32_t i = start + tid; i < n; i += nt) {
        const MatchRecord& r = cand.at(i);
        if (r.step != s) {
            if (r.step > s) atomicMin(&s_next_start, i);
            continue;
        }
        EdgeId edges[kMaxPatternEdges];
        for (uint32_t k = 0; k < kMaxPatternEdges; ++k) edges[k] = r.matched_edges[k];
        const uint64_t key = transition_key_device(ds, r.state_id, r.rule_id, edges, r.num_edges);
        const uint32_t at = atomicAdd(&sel.words[0], 1u);
        if (at < sel.cap) {
            sel.rank[at] = hgcommon::transition_rank(key, ds.sampling_seed);
            sel.idx[at] = i;
        }
    }
    __syncthreads();
    if (tid == 0) sel.words[1] = s_next_start;
    const uint32_t m = min(sel.words[0], sel.cap);
    const uint32_t cap_n = ds.max_states_per_step;
    hgcommon::RankCut cut{~0ULL, true};
    uint32_t take = m;
    if (m > cap_n) {
        if (tid == 0) { s_prefix = 0; s_remaining = cap_n; }
        uint64_t mask = 0;
        for (int bit = 63; bit >= 0; --bit) {
            const uint64_t b = 1ULL << bit;
            if (tid == 0) s_cnt = 0;
            __syncthreads();
            const uint64_t prefix = s_prefix;
            uint32_t local = 0;
            for (uint32_t i = tid; i < m; i += nt) {
                const uint64_t r = sel.rank[i];
                if ((r & mask) == prefix && !(r & b)) ++local;
            }
            atomicAdd(&s_cnt, local);
            __syncthreads();
            if (tid == 0 && s_remaining > s_cnt) { s_remaining -= s_cnt; s_prefix |= b; }
            mask |= b;
            __syncthreads();
        }
        cut = hgcommon::RankCut{s_prefix, false};   // the N-th smallest rank
        // The candidates kept, the N lowest and the ties with the N-th, counted for the booking.
        if (tid == 0) s_cnt = 0;
        __syncthreads();
        uint32_t local = 0;
        for (uint32_t i = tid; i < m; i += nt)
            if (hgcommon::rank_cut_keeps(cut, sel.rank[i])) ++local;
        atomicAdd(&s_cnt, local);
        __syncthreads();
        take = s_cnt;
    }
    if (tid == 0) step_book(ds, s + 1u, take);
    __syncthreads();
    for (uint32_t i = tid; i < m; i += nt) {
        if (!hgcommon::rank_cut_keeps(cut, sel.rank[i])) continue;
        const uint32_t k = found.claim();
        if (k == Pool<MatchRecord>::kInvalid) {
            ds.errors.record(ErrorKind::kMatchPoolFull);
            step_release(ds, s + 1u, 1u);   // the token is still held, so this is not the last
            continue;
        }
        const MatchRecord& src = cand.at(sel.idx[i]);
        MatchRecord& dst = found.at(k);
        dst.rule_id = src.rule_id;
        dst.state_id = src.state_id;
        dst.step = src.step;
        dst.num_edges = src.num_edges;
        for (uint32_t e = 0; e < kMaxPatternEdges; ++e) dst.matched_edges[e] = src.matched_edges[e];
        publish_match(dst);
    }
    __syncthreads();
}

// Step s's selection on the calling block, then each step whose count its token release brings
// to zero (a step with no candidates).
__device__ void run_step_selections(const DeviceState& ds, uint32_t s,
                                    typename Pool<MatchRecord>::DeviceView cand,
                                    typename Pool<MatchRecord>::DeviceView found,
                                    StepSelectScratch sel) {
    __shared__ uint32_t s_next;
    for (;;) {
        select_step(ds, s, cand, found, sel);
        if (threadIdx.x == 0) s_next = step_release(ds, s + 1u, 1u) ? 1u : 0u;
        __syncthreads();
        if (!s_next) return;
        ++s;
    }
}

__global__ void k_seed_frontier(const __grid_constant__ DeviceState ds, ExploreView ev, const StateId* ids,
                                const uint32_t* steps, const uint32_t* count, uint32_t cap,
                                bool dedup) {
    const uint32_t live = min(*count, cap);
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= live) return;
    const StateId s = ids[tid];
    // A state lowered under the old budget later in the run that recorded it was expanded
    // then, and holds the claim.
    if (!ev.claim(s)) return;
    if (dedup && !explore_admits(ds, s)) return;
    // Depth is PER ENTRY: after a steered Step the frontier mixes entries stranded by
    // different budgets. The state's own depth is the smallest any path reached it by.
    uint32_t d = steps[tid];
    const uint32_t known = ev.depth[s];
    if (known < d) d = known;
    step_book(ds, d, 1u);
    if (!ev.expand.append(ExpandEntry{s, d, 0u})) {
        ds.errors.record(ErrorKind::kStatePoolFull);
        step_release(ds, d, 1u);
    }
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
__device__ ExactHashStatus state_key_device(const DeviceState& ds, StateId sid,
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
// One thread per driver: the points the previous run's bound left standing. `slices` drivers:
// the multiplicity arena's slices when the cascade runs, otherwise one per replay lane.
__global__ void k_qe_redrive(const __grid_constant__ DeviceState ds, QeView qe, uint32_t old_bound,
                             uint32_t slices) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= slices) return;
    qe_redrive(ds, qe, old_bound, tid, slices);
}

// Record `s` on a session's frontier at `step`: the budget refused it and a continuation resumes
// from it. Past the capacity the entry is dropped and reported.
__device__ inline void session_frontier_append(const DeviceState& ds, const SessionView& sess,
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

__global__ void k_seed_root_hashes(const __grid_constant__ DeviceState ds, const StateId* roots, uint32_t num_roots,
                                   DedupMap::DeviceView map, CanonicalizationMode state_mode,
                                   bool need_exact, bool need_ranks, DeviceArena::View arena,
                                   QcView qc, QeView qe, ExploreView ev,
                                   typename Pool<uint32_t>::DeviceView forms,
                                   DedupMap::DeviceView exact_map, uint32_t max_steps,
                                   SessionView sess) {
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
        if (ds.record_invariants && claim.canonical == sid)
            record_state_invariants_device(ds, sid, form, form_words, arena, slot, slot_words);
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
            const StateClaim ec = state_claim_form(ds, sid, exact & ds.event_key_mask, eform,
                                                   eform_words, exact_map, forms);
            exact = ec.key;
            if (ds.record_invariants && ec.canonical == sid)
                record_state_invariants_device(ds, sid, eform, eform_words, arena, slot,
                                               slot_words);
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
            if (max_steps) ev.claim(claim.canonical);
        }
    } else if (key == 0) {
        ds.errors.record(ErrorKind::kUncomputedStateHash);   // keep it; see the kind
    } else {
        const auto r = map.insert_if_absent(key, sid);
        if (!r.inserted && !r.overflowed) {
            atomicMin(&ev.depth[r.value], 0u);
            if (max_steps) ev.claim(r.value);
        }
    }
    atomicMin(&ev.depth[sid], 0u);
    // A budget of 0 expands nothing: the root is hashed and recorded on a session's frontier.
    if (max_steps == 0) {
        session_frontier_append(ds, sess, sid, 0u);
        return;
    }
    ev.claim(sid);
    step_book(ds, 0u, 1u);
    if (!ev.expand.append(ExpandEntry{sid, 0u, 0u})) {
        ds.errors.record(ErrorKind::kStatePoolFull);
        step_release(ds, 0u, 1u);
    }
}


// This engine's face for hgcommon/explore_depth_core.hpp, run by a block's thread 0 over the
// block's frame slice. A state admitted under the budget is claimed and appended to the expand
// log; at or past it, recorded on a session's frontier unclaimed.
struct DeviceExploreCtx {
    using Node = uint32_t;
    const DeviceState&       ds;
    ExploreView&       ev;
    const SessionView& sess;
    uint32_t           max_steps;
    uint32_t*          frame_node;
    uint32_t*          frame_depth;
    uint32_t           levels;
    bool               dedup;      // quotient exploration: the coin is drawn at the claim
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
        if (dedup && !explore_admits(ds, s)) return;
        step_book(ds, d, 1u);
        if (!ev.expand.append(ExpandEntry{s, d, 0u})) {
            ds.errors.record(ErrorKind::kStatePoolFull);
            step_release(ds, d, 1u);
        }
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

// Reserve consecutive unconsumed records below the readable count, starting at the cursor:
// returns how many, the first at `base`; 0 when there is none yet. `limit(at, available)` caps
// the count for the run that would start at record `at` with `available` records readable.
//
// The reservation is a CAS rather than an unconditional bump, because the cursor is shared and
// a bump has nothing to undo with: a block that bumped past the end and then subtracted can
// have its subtraction cancel a DIFFERENT block's successful claim, which both hands the same
// record to two blocks and strands the one in between. A stranded record is never rewritten,
// so `rewrites_done` never reaches the record count and the run does not terminate.
template <class Limit>
__device__ __forceinline__ uint32_t claim_next_records(
        uint32_t* cursor, const typename Pool<MatchRecord>::DeviceView& found, Limit limit,
        uint32_t& base) {
    uint32_t cur = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*cursor)
                       .load(cuda::memory_order_relaxed);
    for (;;) {
        const uint32_t readable = readable_records(found, cuda::memory_order_relaxed);
        if (cur >= readable) return 0;
        const uint32_t take = min(limit(cur, readable - cur), readable - cur);
        if (take == 0) return 0;
        const uint32_t prev = atomicCAS(cursor, cur, cur + take);
        if (prev == cur) { base = cur; return take; }
        cur = prev;
    }
}

// The next unconsumed record index, or INVALID_ID when there is none yet.
__device__ __forceinline__ uint32_t claim_next_record(
        uint32_t* cursor, const typename Pool<MatchRecord>::DeviceView& found) {
    const auto one = [](uint32_t, uint32_t) { return 1u; };
    uint32_t base = INVALID_ID;
    return claim_next_records(cursor, found, one, base) ? base : INVALID_ID;
}

// ---- stage 1: the match role alone ------------------------------------------------------
//
// One block per popped item -- the shape match_state_rule already wants. Only thread 0 touches
// the queue, so a pop is one claim per block rather than a race between its threads.
//
// Exit when the queue is empty. That is exact for a queue seeded once and never grown: no work
// can appear after a failed pop. It is NOT the rule stage 2 uses.
__global__ void k_persistent_match(const __grid_constant__ DeviceState ds,
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
        const __grid_constant__ DeviceState ds,
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
            const DeviceState&                              ds;

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

// The readable records below which a block claims one at a time: a short backlog is spread over
// the grid, where a batch would leave the other blocks idle.
constexpr uint32_t kLaneBatchBacklog = 64;
// The records a batch claims: up to kBatchRecords, and up to 32 once the block has found work in
// kDeepStreak consecutive iterations. A batch's children are canonicalised on tiles of
// min(kMaxTile, the largest power of two at most 32 / records) lanes: 32 records run one per
// lane, which pays when every block is busy; a block coming off idle keeps tiles, whose children
// finish sooner.
constexpr uint32_t kBatchRecords = 8;
constexpr uint32_t kDeepStreak = 16;
constexpr uint32_t kMaxTile = 4;
// A child whose twin has not published waits on it (keyed_take_twin) when it has more than this
// many edges; a smaller child runs its own IR, which costs less than the wait.
constexpr uint32_t kFollowEdges = 32;
// Consecutive iterations that found work before a block claims a batch: a block coming off idle,
// or off the matching that produced a burst, takes one record, so the burst spreads over the
// grid.
constexpr uint32_t kBatchBusyStreak = 2;

// A rewritten child's canonical identity: its hash (IR, or a twin's through keyed rewrites), its
// class claim, the published hash and exact hash, and its event's signature. One body for a child
// canonicalised by a batch tile (IrTile) and by the whole warp (IrWarpAll, a single record): par.leader() runs the claims, and what the caller branches on crosses to every
// lane.
struct ChildIdentity {
    StateId canonical = INVALID_ID;
    StateId rep = INVALID_ID;   // the class representative under Full and Automatic states
    bool fresh = false;
    bool capture = false;   // the event's class-frame capture runs (qe_capture_expansion)
    bool ok = false;        // the child has a hash; its identity and depth are registered
    bool deferred = false;  // the child waits on a twin (kFollowing): its record is not done yet
};

template <class Par>
__device__ __forceinline__ ChildIdentity canonicalise_child(
        const DeviceState& ds, StateId sid, EventId evt, StateId parent, uint32_t keyed,
        CanonicalizationMode state_mode, EventSignatureKeys event_keys, bool need_ranks,
        bool need_exact, bool want_orbits, bool dedup, DeviceArena::View arena, uint32_t*& slot,
        uint64_t& slot_words, DedupMap::DeviceView dedup_map, DedupMap::DeviceView exact_map,
        DedupMap::DeviceView event_map, typename Pool<uint32_t>::DeviceView forms,
        typename RingBuffer<uint32_t>::DeviceView ready, unsigned long long& acc_irkey,
        unsigned long long& acc_evkey, Par par) {
    const unsigned long long t1 = clock64();
    // Keyed rewrites (keyed.hpp): a child whose token set an earlier state holds takes that
    // state's canonical results and skips its IR, or, when they are not published yet and the child
    // is large (kFollowEdges), waits on that state and is completed from `ready`. One thread
    // checks; the verdict crosses.
    uint64_t twin_h = 0;
    StateId twin_rep = INVALID_ID;
    uint32_t twin = static_cast<uint32_t>(TwinResult::kNone);
    if (keyed != 0 && par.leader()) {
        const bool may_follow = ds.state_edge_slices[sid].count > kFollowEdges;
        twin = static_cast<uint32_t>(keyed_take_twin(ds, sid, parent, evt, keyed, arena, slot,
                                                     slot_words, need_ranks, want_orbits,
                                                     dedup_map, forms, twin_h, twin_rep,
                                                     may_follow));
    }
    par.sync();
    twin = par.bcast(twin);
    if (twin == static_cast<uint32_t>(TwinResult::kFollowing)) {
        ChildIdentity out;
        out.deferred = true;
        return out;
    }

    uint64_t h = 0;
    uint32_t* form = nullptr;
    uint32_t form_words = 0;
    ExactHashStatus key_st = ExactHashStatus::kOk;
    if (!twin)
        key_st = state_key_device(ds, sid, state_mode, arena, slot, slot_words, h, need_ranks,
                                  want_orbits, &form, &form_words, par);
    if (par.leader()) acc_irkey += clock64() - t1;

    // The exact isomorphism hash is a different question from the mode's key and coincides with
    // it only in Full. Computed only when an event identity or a transition key will read it.
    uint64_t exact = h;
    ExactHashStatus ex_st = ExactHashStatus::kOk;
    uint32_t* eform = nullptr;
    uint32_t eform_words = 0;
    if (!twin && key_st == ExactHashStatus::kOk && need_exact &&
        state_mode != CanonicalizationMode::Full)
        ex_st = state_exact_hash_device(ds, sid, arena, slot, slot_words, exact, need_ranks,
                                        false, &eform, &eform_words, par);

    uint32_t canonical = INVALID_ID, rep = INVALID_ID, fresh = 0, capture = 0, ok = 0;
    // "StepStatistics": whether this state created its class's record, and the IR canonical form
    // the record's geometry reads.
    uint32_t want_record = 0, record_form_words = 0;
    const uint32_t* record_form = nullptr;
    if (par.leader()) {
        if (key_st != ExactHashStatus::kOk) {
            // The hash is the dedup KEY, so a state whose hash could not be computed is not
            // enqueued under a coarser one.
            ds.errors.record(error_kind_for(key_st));
        } else {
            ok = 1;
            // IDENTITY FIRST. In Full mode the class's key, claimed on the canonical form, is the
            // state's canonical hash, so it is claimed before the hash is published and everything
            // keyed by the hash -- event identity, the quotient's classes -- reads the key.
            if (twin) {
                h = twin_h;
                exact = h;
                rep = twin_rep;
                canonical = dedup ? twin_rep : sid;
                fresh = dedup ? 0u : 1u;
            } else if (state_mode == CanonicalizationMode::Full) {
                const StateClaim c = state_claim_form(ds, sid, h & ds.canonical_key_mask, form,
                                                      form_words, dedup_map, forms);
                if (ds.record_invariants && c.canonical == sid) {
                    want_record = 1;
                    record_form = form;
                    record_form_words = form_words;
                }
                h = c.key;
                exact = h;
                rep = c.canonical;
                canonical = dedup ? c.canonical : sid;
                fresh = (dedup ? c.fresh : true) ? 1u : 0u;
            } else if (state_mode == CanonicalizationMode::Automatic) {
                const StateClaim c = state_claim_content(ds, sid, h, dedup_map);
                h = c.key;
                rep = c.canonical;
                canonical = dedup ? c.canonical : sid;
                fresh = (dedup ? c.fresh : true) ? 1u : 0u;
            } else {
                const StateIdentity id = state_identity(ds, sid, h, dedup_map, dedup);
                canonical = id.canonical;
                fresh = id.fresh ? 1u : 0u;
            }
            // Publish before anything reads it: a transition OUT of this state needs it as an
            // input hash, and that read happens on another block.
            ds.state_canonical_hash[sid] = h;

            if (need_exact) {
                if (ex_st != ExactHashStatus::kOk) {
                    ds.errors.record(error_kind_for(ex_st));
                    exact = 0;
                } else if (state_mode != CanonicalizationMode::Full) {
                    // The exact hash event identity reads, claimed on the IR form (the host's
                    // event_canonical_state_map_).
                    const StateClaim ec = state_claim_form(ds, sid, exact & ds.event_key_mask,
                                                           eform, eform_words, exact_map, forms);
                    exact = ec.key;
                    if (ds.record_invariants && ec.canonical == sid) {
                        want_record = 1;
                        record_form = eform;
                        record_form_words = eform_words;
                    }
                }
                ds.state_exact_hash[sid] = exact;
            }

            // The event identity, where both halves exist: the input hash, published when the
            // parent was created, and the output hash just computed. Built from the EXACT hashes,
            // never the mode's key (SPEC.md sec 4).
            if (event_keys != EVENT_SIG_NONE && evt != INVALID_ID) {
                const uint64_t s1 = clock64();
                stamp_event_signature(ds, evt, event_keys, event_map);
                acc_evkey += clock64() - s1;
            }
            // Quotient causal: EVERY raw event registers its canonical transition, whether or not
            // the child survives dedup -- the host registers per raw event too.
            capture = evt != INVALID_ID ? 1u : 0u;
        }
        // The children waiting on this one go to `ready`: its results are published, or its key
        // failed and they run their own. A twin has this state's edge count, so only a state
        // above kFollowEdges can have followers.
        if (keyed != 0 && ds.state_edge_slices[sid].count > kFollowEdges)
            keyed_close_followers(ds, sid, ready);
    }
    if (ds.record_invariants && par.bcast(want_record)) {
        record_form = reinterpret_cast<const uint32_t*>(
            par.bcast64(reinterpret_cast<uint64_t>(record_form)));
        record_form_words = par.bcast(record_form_words);
        record_state_invariants_device(ds, sid, record_form, record_form_words, arena, slot,
                                       slot_words, par);
    }
    ChildIdentity out;
    out.canonical = par.bcast(canonical);
    out.rep = par.bcast(rep);
    out.fresh = par.bcast(fresh) != 0;
    out.capture = par.bcast(capture) != 0;
    out.ok = par.bcast(ok) != 0;
    return out;
}

// The key ExplorationProbability's coin draws on, the host's: the class's canonical hash under
// quotient exploration (claim_canonical_for_expansion) and under full capture with Full states,
// the creating transition's key under full capture with None or Automatic states (the rewrite
// site).
__device__ inline uint64_t explore_key(const DeviceState& ds, bool dedup, bool full_states,
                                       StateId canonical, const MatchRecord* rec) {
    if (dedup || full_states) return ds.state_canonical_hash[canonical];
    EdgeId edges[kMaxPatternEdges];
    for (uint32_t k = 0; k < kMaxPatternEdges; ++k) edges[k] = rec->matched_edges[k];
    return transition_key_device(ds, rec->state_id, rec->rule_id, edges, rec->num_edges);
}

// IDENTITY, THEN DEPTH (explore_depth.hpp) for one child with a hash, on thread 0. The first
// arrival of a key is its canonical state; a fresh state the exploration coin refuses, under the
// budget or in a session, is claimed unexpanded, so no later path expands it. Every arrival
// registers under its parent, and one that lowers the canonical state's depth admits it and
// lowers its descendants. `rec` is the rewritten record, read for the coin's key under None and
// Automatic states; `rep` is the child's class representative under Full states.
__device__ __forceinline__ void register_child(
        const DeviceState& ds, ExploreView& ev, const SessionView& sess, StateId sid,
        StateId parent, StateId canonical, bool fresh, uint32_t step, uint32_t max_steps,
        bool dedup, CanonicalizationMode state_mode, StateId rep, const MatchRecord* rec) {
    // Full capture: each state is registered once, as it is created, and the coin is drawn then.
    // Under Full states it is keyed on the child's class hash, and a class holding an initial
    // state (its representative is a root, at depth 0) is expanded without it, as quotient
    // exploration expands it (the host's initial_classes_). Under None and Automatic it is keyed
    // on the creating transition. Under quotient exploration it is drawn at the claim (admit), so
    // a class first reached past the budget and later under it is drawn when it is claimed.
    const bool full_states = state_mode == CanonicalizationMode::Full;
    if (!dedup && fresh && (full_states || rec != nullptr) && (step < max_steps || sess.enabled) &&
        ds.exploration_probability < 1.0) {
        bool initial_class = false;
        if (full_states && rep < ev.max_states) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> d(ev.depth[rep]);
            initial_class = d.load(cuda::memory_order_relaxed) == 0u;
        }
        if (!initial_class &&
            !hgcommon::explore_survives(explore_key(ds, dedup, full_states, canonical, rec),
                                        ds.sampling_seed, ds.exploration_probability))
            ev.claim(canonical);
    }
    DeviceExploreCtx xc{ds, ev, sess, max_steps,
                        ev.frame_node + size_t(blockIdx.x) * ev.frame_levels,
                        ev.frame_depth + size_t(blockIdx.x) * ev.frame_levels, ev.frame_levels,
                        dedup};
    const uint32_t d = hgcommon::explore_register_child(xc, parent, canonical, step);
    if (d != hgcommon::kExploreNoDepth) {
        xc.admit(canonical, d);
        if (!hgcommon::explore_relax(xc, canonical, d))
            ds.errors.record(ErrorKind::kScratchOverflow);
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
        const __grid_constant__ DeviceState ds,
        const DeviceRule* rules,
        uint32_t num_rules,
        typename RingBuffer<MatchWorkItem>::DeviceView match_q,
        typename Pool<MatchRecord>::DeviceView found,
        uint32_t* consume_cursor,
        uint32_t* rewrites_done,
        DedupMap::DeviceView dedup_map,
        bool dedup,
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
        typename Pool<uint32_t>::DeviceView forms,
        typename RingBuffer<uint32_t>::DeviceView ready,
        uint32_t* regions,             // region_words of IR scratch per block, from its start
        uint32_t region_words,
        typename Pool<MatchRecord>::DeviceView cand,   // MaxStatesPerStep's held candidates
        StepSelectScratch step_sel) {
    // Under MaxStatesPerStep a match goes to the step's candidates, not to the rewrite pool.
    const typename Pool<MatchRecord>::DeviceView match_out =
        ds.max_states_per_step != 0u ? cand : found;

    // Ranks are the reconstruction's frame alignment, Automatic's signature, AND the transition
    // draw's key. One predicate answers it for the roots and for every child; see its note.
    const bool need_ranks = run_needs_edge_ranks(event_keys, qe.enabled != 0,
                                                 ds.transition_rate, ds.num_rule_weights,
                                                 (hgcommon::drain_selects(ds.matches_per_state_rule,
                                        ds.max_successor_states_per_parent,
                                        ds.max_states_per_step) |
                                         explore_reads_ranks(ds, dedup, state_mode)));
    const bool need_exact = ds.record_invariants || run_needs_exact_hash(event_keys, ds.transition_rate,
                                                 ds.num_rule_weights, (hgcommon::drain_selects(ds.matches_per_state_rule,
                                        ds.max_successor_states_per_parent,
                                        ds.max_states_per_step) |
                                         explore_reads_ranks(ds, dedup, state_mode)));

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
            const DeviceState&                              ds;
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
    __shared__ bool     have_ready;   // a child from `ready` this iteration
    __shared__ uint32_t claimed;         // the first record the block claimed
    __shared__ uint32_t claimed_count;   // how many, up to 32
    __shared__ uint32_t claimed_more;    // records 2..claimed_count, consecutive from here
    // Each tile's IR scratch for the children it canonicalises (a batch's tiles, IrTile), claimed
    // from the arena on first use and grown by grow_ir_slot.
    __shared__ uint32_t* tile_slot[32];
    __shared__ uint64_t  tile_slot_words[32];
    __shared__ uint32_t child_sid;
    __shared__ uint32_t child_step;
    __shared__ KeptCopy child_kept;   // the kept edges of the child the warp copies
    __shared__ uint32_t child_event;
    __shared__ uint32_t child_keyed;
    __shared__ StateId  child_parent;
    __shared__ uint32_t child_rule;
    __shared__ uint32_t child_pstep;
    __shared__ uint64_t surv_shared[kLocalSurvivors];
    __shared__ uint32_t expand_base;
    __shared__ uint32_t expand_count;
    __shared__ bool     run_rule_inline;
    __shared__ bool     stalled;
    uint32_t idle_spins = 0;
    uint32_t idle_ns    = 64;   // thread 0's backoff state; reset whenever work is found
    uint32_t busy_streak = 0;   // thread 0's consecutive iterations that found work, to 255

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
    tile_slot[threadIdx.x] = nullptr;
    tile_slot_words[threadIdx.x] = 0;
    __syncthreads();

    // The warp canonicalises the child in the child_* shared words (its kept edges copied) and
    // registers it: the class-frame capture on every lane, identity and depth on thread 0.
    auto finish_warp_child = [&]() {
        const unsigned long long t1 = clock64();
        ChildIdentity id{};
        if (child_sid != INVALID_ID)
            id = canonicalise_child(ds, child_sid, child_event, child_parent, child_keyed,
                                    state_mode, event_keys, need_ranks, need_exact,
                                    qc.enabled != 0, dedup, arena, ir_slot, ir_slot_words,
                                    dedup_map, exact_map, event_map, forms, ready, acc_irkey,
                                    acc_evkey, IrWarpAll{});
        if (id.capture) {
            const unsigned long long s3 = (threadIdx.x == 0) ? clock64() : 0;
            qe_capture_expansion(ds, qe, child_parent, child_sid, child_event, child_rule,
                                 child_pstep, blockIdx.x, surv_shared);
            if (threadIdx.x == 0) acc_qe += clock64() - s3;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (id.ok) {
                const uint64_t s4 = clock64();
                register_child(ds, ev, sess, child_sid, child_parent, id.canonical, id.fresh,
                               child_step, max_steps, dedup, state_mode, id.rep,
                               have_ready ? nullptr : &found.at(claimed));
                acc_dedup += clock64() - s4;
            }
            acc_canon += clock64() - t1;
        }
        return id;
    };

    for (;;) {
        // Rewrite first: it drains what matching produced, and letting the pool run ahead
        // unboundedly is what makes it overflow. A block claims one record; when the IR slots of
        // more children like it fit the block's region (ir_slot_shape at max_edge_arity),
        // kLaneBatchBacklog records are readable and the block found work in its last
        // kBatchBusyStreak iterations, it claims up to kBatchRecords - 1 more in one exchange, or
        // 31 after kDeepStreak. A short burst spreads over the grid. The record is read only after
        // it is claimed.
        if (threadIdx.x == 0) {
            claimed = claim_next_record(consume_cursor, found);
            claimed_count = 0;
            if (claimed != INVALID_ID) {
                claimed_count = 1;
                const MatchRecord& r = found.at(claimed);
                await_match(r);
                const DeviceRule& rule = rules[r.rule_id];
                const uint32_t child = ds.state_edge_slices[r.state_id].count +
                                       rule.num_rhs_edges - rule.num_lhs_edges;
                // As many records as tiles whose child's IR slot fits a share of the region.
                const uint64_t need =
                    ir_slot_shape(child, child * ds.max_edge_arity, ds.ir_depth, ds.ir_generators)
                        .stride();
                const uint64_t fit = need ? region_words / need : 32u;
                if (fit > 1u && busy_streak >= kBatchBusyStreak) {
                    const bool deep = busy_streak >= kDeepStreak;
                    const auto more = [fit, deep](uint32_t, uint32_t available) {
                        if (available < kLaneBatchBacklog) return 0u;
                        const uint64_t cap = deep ? 32u : kBatchRecords;
                        return static_cast<uint32_t>((fit < cap ? fit : cap) - 1u);
                    };
                    claimed_count += claim_next_records(consume_cursor, found, more, claimed_more);
                }
            }
            // No record: a READY CHILD (keyed_close_followers), one whose rewrite is applied and
            // that waited on its twin, runs as a single record from its canonicalisation on.
            have_ready = false;
            uint32_t e = INVALID_ID;
            if (claimed_count == 0 && ds.keyed.enabled && ready.try_pop(e)) {
                have_ready = true;
                claimed_count = 1;
                const DeviceEvent& x = ds.event_pool.at(e);
                child_sid = x.output_state;
                child_event = e;
                child_step = x.step;
                child_parent = x.input_state;
                child_rule = x.rule;
                child_pstep = x.step - 1u;
                child_keyed = x.rewrite_id | hgcommon::REWRITE_TWIN_CANDIDATE;
            }
            // This claim's IR scratch: the block's region, whole for the warp, and for a batch a
            // share per tile. A state larger than its share claims from the pool (grow_ir_slot).
            if (claimed_count) {
                uint32_t* region = regions + size_t(blockIdx.x) * region_words;
                ir_slot = region;
                ir_slot_words = region_words;
                const uint32_t share = (region_words / claimed_count) & ~1u;
                for (uint32_t t = 0; t < claimed_count; ++t) {
                    tile_slot[t] = region + size_t(t) * share;
                    tile_slot_words[t] = share;
                }
            }
        }
        __syncthreads();

        // REWRITE the claimed records: a single one on the whole warp, a batch on its tiles.
        if (claimed_count) {
            const uint32_t lane = threadIdx.x;
            if (lane == 0) {
                idle_ns = 64;
                busy_streak += busy_streak < 255u ? 1u : 0u;
                idle_spins = 0;            // consecutive, not cumulative -- see the guard below
            }
            // Records whose child waits on a twin (TwinResult::kFollowing): done when completed
            // from `ready`. Thread 0's.
            uint32_t waiting = 0;
            if (claimed_count == 1) {
                // ONE RECORD: thread 0 applies it, and the whole warp copies and canonicalises the
                // child. A ready child is applied and copied already.
                if (lane == 0 && !have_ready) {
                    const unsigned long long t0 = clock64();
                    const MatchRecord& rec = found.at(claimed);
                    await_match(rec);
                    const unsigned long long t0b = clock64();
                    acc_wait += t0b - t0;
                    // The event carries the depth of the state it PRODUCES -- see the note in
                    // k_persistent_match_rewrite. The exploration depth below is the same value.
                    const AppliedMatch a = apply_one_match(
                        ds, rules, rec, rec.step + 1u, phase_cycles ? phase_cycles + 5 : nullptr);
                    child_sid = a.state;
                    child_event = a.event;
                    child_step = rec.step + 1u;
                    child_kept = a.kept;
                    child_keyed = a.keyed;
                    child_parent = rec.state_id;
                    child_rule = rec.rule_id;
                    child_pstep = rec.step;
                    acc_rewrite += clock64() - t0b;
                }
                __syncthreads();
                if (!have_ready && child_sid != INVALID_ID)
                    copy_kept_edges(ds, child_kept, IrWarpAll{});
                __syncthreads();
                const ChildIdentity id = finish_warp_child();
                if (lane == 0) waiting = id.deferred ? 1u : 0u;
                __syncthreads();
            } else {
                // A BATCH: a record per tile of T lanes (kMaxTile). The tile's leader applies it,
                // and the tile copies and canonicalises the child (IrTile) in its share of the
                // block's region.
                // The class-frame captures (every lane together) and the identity and depth
                // registration (thread 0: the walk's frames are per block) then run child by
                // child.
                const uint32_t fair = 1u << (31u - __clz(32u / claimed_count));
                const uint32_t T = fair < kMaxTile ? fair : kMaxTile;
                const IrTile Tile{T};
                const uint32_t tile = lane / T;
                const bool tlead = (lane & (T - 1u)) == 0u;
                StateId sid = INVALID_ID, parent = INVALID_ID;
                EventId evt = INVALID_ID;
                uint32_t cstep = 0, keyed = 0, rule = 0, pstep = 0;
                KeptCopy kept{};
                {
                    const unsigned long long t0 = clock64();
                    if (tlead && tile < claimed_count) {
                        const MatchRecord& rec =
                            found.at(tile == 0 ? claimed : claimed_more + tile - 1u);
                        await_match(rec);
                        if (lane == 0) acc_wait += clock64() - t0;
                        parent = rec.state_id;
                        rule = rec.rule_id;
                        pstep = rec.step;
                        // The event carries the depth of the state it PRODUCES -- see the note
                        // in k_persistent_match_rewrite. The exploration depth below is the same.
                        cstep = rec.step + 1u;
                        const AppliedMatch a = apply_one_match(
                            ds, rules, rec, cstep, phase_cycles ? phase_cycles + 5 : nullptr);
                        sid = a.state;
                        evt = a.event;
                        keyed = a.keyed;
                        kept = a.kept;
                    }
                    if (lane == 0) acc_rewrite += clock64() - t0;
                }
                __syncwarp();
                // The tile leader's child on every lane of its tile.
                sid = __shfl_sync(0xFFFFFFFFu, sid, 0, T);
                evt = __shfl_sync(0xFFFFFFFFu, evt, 0, T);
                parent = __shfl_sync(0xFFFFFFFFu, parent, 0, T);
                keyed = __shfl_sync(0xFFFFFFFFu, keyed, 0, T);
                cstep = __shfl_sync(0xFFFFFFFFu, cstep, 0, T);
                kept.src_offset = __shfl_sync(0xFFFFFFFFu, kept.src_offset, 0, T);
                kept.src_count = __shfl_sync(0xFFFFFFFFu, kept.src_count, 0, T);
                kept.dst_offset = __shfl_sync(0xFFFFFFFFu, kept.dst_offset, 0, T);
                kept.n_consumed = __shfl_sync(0xFFFFFFFFu, kept.n_consumed, 0, T);
                #pragma unroll
                for (uint32_t i = 0; i < kMaxPatternEdges; ++i)
                    kept.consumed[i] = __shfl_sync(0xFFFFFFFFu, kept.consumed[i], 0, T);
                if (sid != INVALID_ID) copy_kept_edges(ds, kept, Tile);

                const unsigned long long t1 = clock64();
                ChildIdentity id{};
                if (sid != INVALID_ID)
                    id = canonicalise_child(ds, sid, evt, parent, keyed, state_mode, event_keys,
                                            need_ranks, need_exact, qc.enabled != 0, dedup, arena,
                                            tile_slot[tile], tile_slot_words[tile], dedup_map,
                                            exact_map, event_map, forms, ready, acc_irkey,
                                            acc_evkey, Tile);
                __syncwarp();

                // The class frame's match record (qe_capture_expansion), every lane together,
                // child by child. The block's slice of the survivor scratch is indexed by blockIdx.
                for (uint32_t caps = __ballot_sync(0xFFFFFFFFu, tlead && id.capture); caps;
                     caps &= caps - 1u) {
                    const uint32_t c = __ffs(caps) - 1u;
                    const unsigned long long s3 = (lane == 0) ? clock64() : 0;
                    qe_capture_expansion(ds, qe, __shfl_sync(0xFFFFFFFFu, parent, c),
                                         __shfl_sync(0xFFFFFFFFu, sid, c),
                                         __shfl_sync(0xFFFFFFFFu, evt, c),
                                         __shfl_sync(0xFFFFFFFFu, rule, c),
                                         __shfl_sync(0xFFFFFFFFu, pstep, c), blockIdx.x,
                                         surv_shared);
                    if (lane == 0) acc_qe += clock64() - s3;
                    __syncthreads();
                }

                // Identity, then depth (register_child), child by child on thread 0.
                for (uint32_t oks = __ballot_sync(0xFFFFFFFFu, tlead && id.ok); oks;
                     oks &= oks - 1u) {
                    const uint32_t c = __ffs(oks) - 1u;
                    const StateId ccanon = __shfl_sync(0xFFFFFFFFu, id.canonical, c);
                    const StateId crep = __shfl_sync(0xFFFFFFFFu, id.rep, c);
                    const bool cfresh = __shfl_sync(0xFFFFFFFFu, id.fresh ? 1u : 0u, c) != 0;
                    const StateId csid = __shfl_sync(0xFFFFFFFFu, sid, c);
                    const StateId cparent = __shfl_sync(0xFFFFFFFFu, parent, c);
                    const uint32_t ccstep = __shfl_sync(0xFFFFFFFFu, cstep, c);
                    if (lane == 0) {
                        const uint64_t s4 = clock64();
                        const uint32_t ctile = c / T;
                        register_child(ds, ev, sess, csid, cparent, ccanon, cfresh, ccstep,
                                       max_steps, dedup, state_mode, crep,
                                       &found.at(ctile == 0 ? claimed : claimed_more + ctile - 1u));
                        acc_dedup += clock64() - s4;
                    }
                    __syncwarp();
                }
                if (lane == 0) acc_canon += clock64() - t1;
                const uint32_t w = __popc(__ballot_sync(0xFFFFFFFFu, tlead && id.deferred));
                if (lane == 0) waiting = w;
                __syncthreads();
            }

            // A selected record's unit on its child's step goes once the child is registered.
            if (ds.max_states_per_step != 0u && !have_ready) {
                __shared__ uint32_t s_sel_step;
                if (threadIdx.x == 0) {
                    s_sel_step = INVALID_ID;
                    for (uint32_t t = 0; t < claimed_count; ++t) {
                        const uint32_t d =
                            found.at(t == 0 ? claimed : claimed_more + t - 1u).step + 1u;
                        if (step_release(ds, d, 1u)) s_sel_step = d;
                    }
                }
                __syncthreads();
                if (s_sel_step != INVALID_ID)
                    run_step_selections(ds, s_sel_step, cand, found, step_sel);
            }
            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(rewrites_done, claimed_count - waiting);
                // 1024, which is what the detector's note beside the progress print already
                // states this to be. A flush is ten atomics on one 128-byte line, shared by
                // every block, so at eight it cost 1.25 per record -- and the reason the
                // interval exists at all is that a run which never finishes is still
                // attributable, which 1024 serves exactly as well as 8. A block leaving the
                // loop flushes on the way out either way (exit_requested, stalled), so a run
                // shorter than the interval loses nothing.
                records_since_flush += claimed_count - waiting;
                if (records_since_flush >= 1024u) {
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
                step_book(ds, child_step, num_rules);
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
                    match_state_rule(ds, rules, child_sid, r, child_step, match_out);
                __syncthreads();
                if (threadIdx.x == 0 && run_rule_inline) acc_match += clock64() - tA;
                if (run_rule_inline && ds.max_states_per_step != 0u) {
                    __shared__ uint32_t s_inline_sel;
                    if (threadIdx.x == 0) s_inline_sel = step_release(ds, child_step, 1u);
                    __syncthreads();
                    if (s_inline_sel) run_step_selections(ds, child_step, cand, found, step_sel);
                }
            }
            if (ds.max_states_per_step != 0u) {
                __shared__ uint32_t s_entry_sel;
                if (threadIdx.x == 0) s_entry_sel = step_release(ds, child_step, 1u);
                __syncthreads();
                if (s_entry_sel) run_step_selections(ds, child_step, cand, found, step_sel);
            }
            if (threadIdx.x == 0) {
                __threadfence();
                ev.expand.book(1u);
                idle_ns = 64;
                busy_streak += busy_streak < 255u ? 1u : 0u;
                idle_spins = 0;
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) have = match_q.try_pop(mitem);
        __syncthreads();
        if (have) {
            const unsigned long long tA = (threadIdx.x == 0) ? clock64() : 0;
            match_state_rule(ds, rules, mitem.state_id, mitem.rule_id, mitem.step, match_out);
            __syncthreads();
            if (ds.max_states_per_step != 0u) {
                __shared__ uint32_t s_item_sel;
                if (threadIdx.x == 0) s_item_sel = step_release(ds, mitem.step, 1u);
                __syncthreads();
                if (s_item_sel) run_step_selections(ds, mitem.step, cand, found, step_sel);
            }
            if (threadIdx.x == 0) {
                term.mark_completed(kRoleMatch);
                idle_ns = 64;
                busy_streak += busy_streak < 255u ? 1u : 0u;
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
                    busy_streak += busy_streak < 255u ? 1u : 0u;
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
                busy_streak = 0;
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

uint32_t persistent_ring_capacity(uint64_t items) {
    if (items > (1ull << 31))
        throw std::length_error("a persistent work queue of " + std::to_string(items) +
                                " items is past 2^31 slots");
    uint64_t cap = 2;
    while (cap < items) cap <<= 1;
    return static_cast<uint32_t>(cap);
}

size_t persistent_kernels_stack_bytes() {
    size_t need = 0;
    auto take = [&](const void* k) {
        cudaFuncAttributes a{};
        HG_CUDA_CHECK(cudaFuncGetAttributes(&a, k), "persistent kernel attributes");
        need = std::max<size_t>(need, a.localSizeBytes);
    };
    take(reinterpret_cast<const void*>(&k_persistent_evolve));
    take(reinterpret_cast<const void*>(&k_persistent_match));
    take(reinterpret_cast<const void*>(&k_persistent_match_rewrite));
    take(reinterpret_cast<const void*>(&k_qe_redrive));
    take(reinterpret_cast<const void*>(&k_seed_root_hashes));
    take(reinterpret_cast<const void*>(&k_seed_frontier));
    return need;
}

uint64_t device_resident_threads() {
    static uint64_t cached = 0;
    if (cached) return cached;
    int sms = 0, per_sm = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0) !=
            cudaSuccess ||
        sms <= 0 || per_sm <= 0) {
        cudaGetLastError();   // do not leave a sticky status behind for the next launch
        sms = 32;
        per_sm = 2048;        // the largest any supported architecture holds
    }
    cached = static_cast<uint64_t>(sms) * static_cast<uint64_t>(per_sm);
    return cached;
}

// A launch's ring, dedup maps and detector. Built on first use at the sizes the launch asks for
// and rebuilt only when a later launch asks for a different size; each launch clears what it
// takes. A PersistentEvolver reuses its engine while the engine's config covers the run, and
// then every launch reuses them and makes no cudaMalloc or cudaFree call for them (13 per run before, 0.26 ms of the
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
    std::unique_ptr<RingBuffer<uint32_t>> keyed_ready;   // children waiting on a twin, ready
    uint32_t* keyed_words = nullptr;             // KeyedView::words
    uint32_t* explore_frames = nullptr;
    size_t    explore_frame_words = 0;
    // MaxStatesPerStep: the held candidates and the selection's ranks, indices and two counters.
    std::unique_ptr<Pool<MatchRecord>> step_cand;
    uint64_t* step_rank = nullptr;
    uint32_t* step_idx = nullptr;
    uint32_t* step_words = nullptr;
    uint32_t  step_cap = 0;
    ~PersistentScratch() {
        if (step_rank) cudaFree(step_rank);
        if (step_idx) cudaFree(step_idx);
        if (step_words) cudaFree(step_words);
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

// With a batch, a reused structure's counters are cleared at the batch's flush.
template <class T>
RingBuffer<T>& reuse_ring(std::unique_ptr<RingBuffer<T>>& slot, uint32_t capacity,
                          ClearBatch* batch = nullptr) {
    if (slot && slot->capacity() == capacity) slot->clear(batch);
    else { slot.reset(); slot = std::make_unique<RingBuffer<T>>(capacity); }
    return *slot;
}

RingBuffer<MatchWorkItem>& reuse_ring(const EngineState& engine, uint32_t capacity,
                                      ClearBatch* batch = nullptr) {
    return reuse_ring(engine.persistent_scratch().ring, capacity, batch);
}

DedupMap& reuse_map(std::unique_ptr<DedupMap>& slot, uint32_t capacity,
                    ClearBatch* batch = nullptr) {
    if (slot && slot->capacity() == capacity) slot->clear(batch);
    else { slot.reset(); slot = std::make_unique<DedupMap>(capacity); }
    return *slot;
}

TerminationDetector& reuse_term(const EngineState& engine, uint32_t num_roles = 1,
                                ClearBatch* batch = nullptr) {
    auto& slot = engine.persistent_scratch().term;
    if (slot && slot->num_roles() == num_roles) slot->clear(batch);
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
    const uint32_t cap = persistent_ring_capacity(uint64_t{num_rules} * states.size());
    const uint32_t num_items = static_cast<uint32_t>(uint64_t{num_rules} * states.size());

    // Engine-lifetime grow-only scratch: allocating these per call was API-call overhead on
    // the per-call floor.
    EngineState::LaunchScratch& sc =
        engine.launch_scratch(num_rules, static_cast<uint32_t>(states.size()));
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice, 0), "rules copy");
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice, 0), "states copy");

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
    const uint32_t cap = persistent_ring_capacity(uint64_t{num_rules} * states.size());
    const uint32_t num_items = static_cast<uint32_t>(uint64_t{num_rules} * states.size());

    // Engine-lifetime grow-only scratch; see run_persistent_match.
    EngineState::LaunchScratch& sc =
        engine.launch_scratch(num_rules, static_cast<uint32_t>(states.size()));
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice, 0), "rules copy");
    HG_CUDA_CHECK(cudaMemcpyAsync(sc.states, states.data(), sizeof(StateId) * states.size(),
                     cudaMemcpyHostToDevice, 0), "states copy");

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
                                            CanonicalizationMode state_mode,
                                            EventSignatureKeys event_keys,
                                            uint32_t blocks,
                                            const QcView* qc_in,
                                            const QeView* qe_in,
                                            SessionView* session,
                                            uint32_t start_step,
                                            bool read_stats) {
    PersistentEvolveStats stats;
    // With no rule the roots are the whole evolution: they are hashed and recorded, and no match
    // is found (the host's evolve with an empty rule set, 41e1ba83).
    if (roots.empty()) return stats;

    QcView qc{};
    if (qc_in) qc = *qc_in;
    QeView qe{};
    if (qe_in) qe = *qe_in;

    // Every clear of this launch's scratch goes to one batch, flushed by one kernel launch
    // before the first kernel that reads any of it (after arena.reset below).
    ClearBatch clears;

    // Records are consumed while they are still being produced, so their publication flags
    // must start clear. The scheduler that relies on the flag is the one that clears it.
    scratch_matches.reset_and_clear(&clears);

    const uint32_t num_rules = static_cast<uint32_t>(rules.size());
    const uint32_t seed_cap = persistent_ring_capacity(uint64_t{num_rules} * roots.size());
    const uint32_t num_seed  = static_cast<uint32_t>(uint64_t{num_rules} * roots.size());

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
    uint32_t cap = seed_cap;
    while (cap < scratch_matches.capacity() && cap < (1u << 20)) cap <<= 1;
    RingBuffer<MatchWorkItem>& match_q = reuse_ring(engine, cap, &clears);

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
        if (ps.forms && ps.forms->capacity() == fw) ps.forms->reset(&clears);
        else ps.forms = std::make_unique<Pool<uint32_t>>(fw);
        forms_v = ps.forms->view();
    }

    // Exploration depth (explore_depth.hpp). A session keeps its depths, claims and child lists
    // across calls and consumes its expand log per call; a one-shot run starts from nothing.
    ExploreView ev;
    if (session) {
        ev = sess_v.explore;
        explore_reset_async(ev, /*full=*/false, &clears);
    } else {
        const uint32_t ms = engine.config().max_states, me = engine.config().max_events;
        if (ps.explore && ps.explore->max_states() == ms && ps.explore->max_events() == me)
            ps.explore->clear(&clears);
        else
            ps.explore = std::make_unique<ExploreState>(ms, me);
        ev = ps.explore->view();
    }
    // The walk's frames: one per level it can descend, per block. A frame is pushed only for a
    // state the walk lowered, one level deeper than the last, and only states under the budget
    // have children, so max_steps + 2 bounds a walk.
    const uint32_t grid_req = blocks ? blocks : default_persistent_grid();
    const uint32_t grid = grid_req < 2 ? 2 : grid_req;
    // The arena is one region per block, from its start, and the pool behind them: a block's IR
    // scratch is its region, laid out anew for every claim (a single record takes it whole, a
    // batch divides it), and a state larger than its share of the region claims from the pool.
    // Sixteen seventeenths of the arena are regions (persistent_arena_words), each an even word
    // count.
    const uint32_t region_words = static_cast<uint32_t>(
        std::min<uint64_t>((arena.capacity_words() * 16u / 17u / grid) & ~1ull, 0xFFFFFFFEull));
    DeviceArena::View pool_v = arena.view();
    pool_v.base += uint64_t(region_words) * grid;
    pool_v.capacity -= uint64_t(region_words) * grid;
    {
        // A walk descends one level per state it lowers, along a parent-child chain of distinct
        // states, so it is bounded by the state budget as well as the depth. Without
        // deduplication every state is registered once, as it is created, and is never lowered
        // again: its walk is the one frame of its own (empty) child list.
        const uint32_t chain = std::min(max_steps, engine.config().max_states);
        const uint32_t levels = dedup ? chain + 2u : 2u;
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
        session ? nullptr : &reuse_map(ps.canonical, engine.config().max_states * 2u, &clears);

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
                : &reuse_map(ps.event_ids, want_event_ids ? engine.config().max_events * 2u : 8u,
                             &clears);
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
            (dsx.record_invariants || run_needs_exact_hash(event_keys, dsx.transition_rate, dsx.num_rule_weights,
                                 (hgcommon::drain_selects(dsx.matches_per_state_rule,
                                        dsx.max_successor_states_per_parent,
                                        dsx.max_states_per_step) |
                                         explore_reads_ranks(dsx, dedup, state_mode))));
        exact_v = reuse_map(ps.exact, want_exact ? engine.config().max_states * 2u : 8u, &clears)
                      .view();
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
                                        (hgcommon::drain_selects(dsk.matches_per_state_rule,
                                        dsk.max_successor_states_per_parent,
                                        dsk.max_states_per_step) |
                                         explore_reads_ranks(dsk, dedup, state_mode))));
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
                reuse_map(ps.keyed_rewrites, engine.config().max_events * 2u, &clears).view();
            dsk.keyed.twins =
                reuse_map(ps.keyed_twins, engine.config().max_states * 2u, &clears).view();
            dsk.keyed.words = ps.keyed_words;
        }
        dsk.keyed.claim_limit = engine.config().keyed_claim_limit;
        dsk.keyed.sum_mask = engine.config().keyed_sum_mask;
        dsk.keyed.enabled = 1;
    }
    // Children waiting on a twin, handed on when it publishes (keyed_close_followers): at most
    // one per state, so a ring of max_states (to a power of two) never fills. Two slots when the
    // run does not key.
    uint32_t ready_cap = 2;
    if (keyed)
        while (ready_cap < engine.config().max_states && ready_cap < (1u << 31)) ready_cap <<= 1;
    const auto ready_v = reuse_ring(ps.keyed_ready, ready_cap, &clears).view();

    const double t_maps = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_maps0).count();
    auto t_alloc0 = std::chrono::steady_clock::now();

    // Every buffer is taken HERE, before the first kernel goes out: the scratch's rare grow
    // path calls cudaMalloc, which may synchronize the device, and the evolution's contract
    // is memory traffic at the start and end only, with ONE synchronization -- after the last
    // kernel.
    uint32_t* d_cursor = sc.cursor;
    clears.add(d_cursor, sizeof(uint32_t) * 2, 0);
    uint32_t* d_rewrites_done = d_cursor + 1;

    // 5 top-level phases + apply_one_match's 6 sub-stretches (see rewrite.hpp).
    unsigned long long* d_phase_cycles = sc.phase_cycles;
    clears.add(d_phase_cycles, sizeof(unsigned long long) * 16, 0);

    TerminationDetector& term = reuse_term(engine, 1, &clears);

    // The whole evolution is a launch CHAIN on one stream: root hashing appends each root to
    // the expand log, which the detector counts, and the evolve kernel consumes it. Stream order
    // carries every dependency, so the host synchronizes exactly once, after the last kernel,
    // and reads nothing back before that.
    const double t_alloc = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_alloc0).count();
    auto t_seed0 = std::chrono::steady_clock::now();

    arena.reset(&clears);
    clears.flush();

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
    // MaxStatesPerStep's candidate pool (as large as the rewrite pool), selection scratch, and the
    // tokens: step_pending was cleared by set_sampling, and every step from 1 holds one.
    typename Pool<MatchRecord>::DeviceView cand_v = scratch_matches.view();
    StepSelectScratch step_sel{};
    if (dsk.max_states_per_step != 0u) {
        const uint32_t cap = scratch_matches.capacity();
        if (!ps.step_cand || ps.step_cand->capacity() != cap) {
            ps.step_cand = std::make_unique<Pool<MatchRecord>>(cap);
            if (ps.step_rank) cudaFree(ps.step_rank);
            if (ps.step_idx) cudaFree(ps.step_idx);
            if (!ps.step_words)
                HG_CUDA_CHECK(cudaMalloc(&ps.step_words, sizeof(uint32_t) * 2u), "step words");
            HG_CUDA_CHECK(cudaMalloc(&ps.step_rank, sizeof(uint64_t) * cap), "step ranks");
            HG_CUDA_CHECK(cudaMalloc(&ps.step_idx, sizeof(uint32_t) * cap), "step indices");
            ps.step_cap = cap;
        }
        ps.step_cand->reset_and_clear(&clears);
        clears.add(ps.step_words, sizeof(uint32_t) * 2u, 0);
        cand_v = ps.step_cand->view();
        step_sel = StepSelectScratch{ps.step_rank, ps.step_idx, ps.step_words, ps.step_cap};
        std::vector<uint32_t> tokens(dsk.step_slots, 1u);
        tokens[0] = 0u;
        HG_CUDA_CHECK(cudaMemcpy(dsk.step_pending, tokens.data(), sizeof(uint32_t) * tokens.size(),
                                 cudaMemcpyHostToDevice), "step tokens");
    }
    // CONTINUING rather than starting: the frontier already holds hashed, deduplicated states,
    // so it is seeded straight into the queue at each entry's own recorded depth. The
    // root path would re-hash them and, worse, consult dedup -- which they already satisfy, so
    // nothing would expand.
    if (start_step > 0 && session) {
        // A continuation raised the depth bound: drive what the old bound left standing. On the
        // default stream, so it completes before the frontier below is expanded; the replay's
        // rendezvous makes the order of the two irrelevant to the answer.
        const uint32_t slices = qe.multiplicity ? qe.work_slices
                                                : default_persistent_grid() * kMatchBlockThreads;
        if (qe.enabled && (qe.replay || qe.multiplicity) && slices) {
            const uint32_t rblock = 64;
            k_qe_redrive<<<(slices + rblock - 1) / rblock, rblock>>>(engine.device(), qe,
                                                                      start_step, slices);
        }
        const uint32_t block = 128;
        const uint32_t seed_grid = (sess_v.frontier_cap + block - 1) / block;
        if (seed_grid) {
            k_seed_frontier<<<seed_grid, block>>>(
                engine.device(), ev, sess_v.frontier, sess_v.frontier_step,
                sess_v.frontier_count, sess_v.frontier_cap, dedup);
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
        // An opening run starts the session's frontier: entries an earlier opening at a budget
        // of 0 recorded are the roots seeded again below.
        if (session)
            HG_CUDA_CHECK(cudaMemsetAsync(sess_v.frontier_count, 0, sizeof(uint32_t)),
                          "session frontier open");
        const uint32_t block = 64;
        const uint32_t n = static_cast<uint32_t>(roots.size());
        // The device view is taken once: the rank predicate reads the run's sampling parameters
        // out of it, and it must be the SAME view the kernel is handed.
        const DeviceState dsv = dsk;
        k_seed_root_hashes<<<(n + block - 1) / block, block>>>(
            dsv, d_states, n,
            session ? sess_v.states : canonical_owner->view(), state_mode,
            dsv.record_invariants || run_needs_exact_hash(event_keys, dsv.transition_rate, dsv.num_rule_weights,
                                 (hgcommon::drain_selects(dsv.matches_per_state_rule,
                                        dsv.max_successor_states_per_parent,
                                        dsv.max_states_per_step) |
                                         explore_reads_ranks(dsv, dedup, state_mode))),
            run_needs_edge_ranks(event_keys, qe.enabled != 0, dsv.transition_rate,
                                 dsv.num_rule_weights, (hgcommon::drain_selects(dsv.matches_per_state_rule,
                                        dsv.max_successor_states_per_parent,
                                        dsv.max_states_per_step) |
                                         explore_reads_ranks(dsv, dedup, state_mode))),
            pool_v, qc, qe, ev, forms_v, exact_v, max_steps, sess_v);
    }

    // MaxStatesPerStep: every step from 1 holds its token (the previous step's selection), then
    // the empty steps below the first one with work give theirs back.
    if (dsk.max_states_per_step != 0u) k_step_release_empty<<<1, 1>>>(dsk);

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
        max_steps, state_mode, event_keys,
        session ? sess_v.events : owned_event_ids->view(), exact_v,
        pool_v, term.view(), qc, qe, d_phase_cycles, sess_v, ev, forms_v, ready_v,
        arena.view().base, region_words, cand_v, step_sel);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "persistent evolve sync");
    stats.explore_depth = ev.depth;
    if (!read_stats) return stats;

    // states_after and canonical_events are both slots of the engine's counter block, so one
    // transfer fetches them instead of two. The pool and arena counters belong to other objects
    // and still cost a call each.
    const auto ctr = engine.counters_snapshot_host();
    stats.matches_found    = scratch_matches.size_host();
    stats.states_after     = ctr.states;
    // The regions are every block's whether used or not; the pool counts what was claimed.
    stats.arena_words_used = uint64_t(region_words) * grid + arena.used_words_host();
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
    // A region of share_words per holder, and a sixteenth of that again as the pool a state larger
    // than its region claims from.
    return static_cast<uint64_t>(holders) * static_cast<uint64_t>(share_words) * 17u / 16u;
}

// ---- ExploreState ------------------------------------------------------------------------

// Zero the published flags of the entries the last launch appended; the counter is read on the
// device, so the host never learns how many there were.
__global__ void k_expand_log_clear(ExploreView v) {
    const uint32_t n = min(*v.expand.items.counter, v.expand.items.capacity);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        v.expand.items.at(i).published = 0u;
}

// The regions go to `batch` when one is given, flushed after this launch; otherwise they are
// cleared here.
void explore_reset_async(const ExploreView& v, bool full, ClearBatch* batch) {
    k_expand_log_clear<<<64, 256>>>(v);
    ClearBatch own;
    ClearBatch& b = batch ? *batch : own;
    b.add(v.expand.items.counter, sizeof(uint32_t), 0);
    b.add(v.expand.cursor, sizeof(uint32_t) * 2u, 0);
    if (full) {
        b.add(v.depth, sizeof(uint32_t) * v.max_states, 0xFF);
        b.add(v.claimed, sizeof(uint32_t) * v.max_states, 0);
        b.add(v.children.heads, sizeof(uint32_t) * v.children.num_keys, 0xFF);
        b.add(v.children.pool.counter, sizeof(uint32_t), 0);
    }
    if (!batch) own.flush();
}

// The destructors below free their raw device buffers with this, and so do the constructors when
// their bodies throw: the destructor does not run for an object whose constructor threw, so a
// failed cudaMalloc would otherwise keep every buffer allocated before it for the life of the
// process.
#define HG_FREE_DEVICE(p) if (p) { cudaFree(p); p = nullptr; }
#define HG_SESSION_STATE_DEVICE_BUFFERS(X) X(frontier_) X(step_) X(count_) X(keyed_words_)

ExploreState::ExploreState(uint32_t max_states, uint32_t max_events)
    : max_states_(max_states), max_events_(max_events),
      children_(max_states, max_events), expand_(max_states) {
    try {
        HG_CUDA_CHECK(cudaMalloc(&words_, sizeof(uint32_t) * (2ull * max_states + 2u)),
                      "explore words alloc");
        HG_CUDA_CHECK(cudaMemset(expand_.view().data, 0, sizeof(ExpandEntry) * size_t(max_states)),
                      "expand log init");
        clear();
    } catch (...) {
        HG_FREE_DEVICE(words_)
        throw;
    }
}

ExploreState::~ExploreState() { HG_FREE_DEVICE(words_) }

void ExploreState::clear(ClearBatch* batch) { explore_reset_async(view(), /*full=*/true, batch); }

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
      try {
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
      } catch (...) {
        HG_SESSION_STATE_DEVICE_BUFFERS(HG_FREE_DEVICE)
        throw;
      }
    }

SessionState::~SessionState() { HG_SESSION_STATE_DEVICE_BUFFERS(HG_FREE_DEVICE) }

#undef HG_SESSION_STATE_DEVICE_BUFFERS
#undef HG_FREE_DEVICE

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
