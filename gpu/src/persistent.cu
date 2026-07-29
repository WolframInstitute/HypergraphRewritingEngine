// Device-resident scheduling: workers that pull work from a queue rather than being launched
// once per phase per step. See gpu/include/hg_gpu/persistent.hpp and
// docs/GPU_PERSISTENT_DESIGN.md.
//
// Its own translation unit, not appended to match.cu, and that is a memory decision rather
// than a stylistic one: match.cu already costs several GB to compile, and adding one more
// kernel to it took a single nvcc to 8 GB. This machine is shared, so a translation unit that
// cannot be compiled within a safe ceiling is a defect whether or not it links.

#include "hg_gpu/persistent.hpp"

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

// Records a claiming consumer may safely read. The pool's counter counts CLAIMS, and a claim
// past the end returns kInvalid without writing, so the counter can exceed the capacity while
// only the first `capacity` slots hold anything. Reading up to the raw counter would read past
// the allocation.
__device__ __forceinline__ uint32_t readable_records(
        const typename Pool<MatchRecord>::DeviceView& found) {
    const uint32_t claimed = *found.counter;
    return claimed < found.capacity ? claimed : found.capacity;
}

// Reserve the next unconsumed record index, or INVALID_ID when there is none yet.
//
// The reservation is a CAS rather than an unconditional bump, because the cursor is shared and
// a bump has nothing to undo with: a block that bumped past the end and then subtracted can
// have its subtraction cancel a DIFFERENT block's successful claim, which both hands the same
// record to two blocks and strands the one in between. A stranded record is never rewritten,
// so the run does not terminate.
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
        for (;;) {
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
                apply_one_match(ds, rules, rec, rec.step);
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

}  // namespace hg_gpu
