// Device-resident scheduling, stage 1: the MATCH role over a queue seeded once.
// See gpu/include/hg_gpu/persistent.hpp and docs/GPU_PERSISTENT_DESIGN.md.
//
// Its own translation unit, not appended to match.cu, and that is a memory decision rather
// than a stylistic one: match.cu already costs about 5 GB to compile, and adding one more
// kernel to it took a single nvcc to 8 GB. This machine is shared, so a translation unit that
// cannot be compiled within a safe ceiling is a defect regardless of whether it links.

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
                                   uint32_t num_rules) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states * num_rules) return;
    MatchWorkItem item;
    item.state_id = states[tid / num_rules];
    item.rule_id  = tid - (tid / num_rules) * num_rules;
    queue.try_push(item);   // capacity >= item count, so this cannot fail here
}

// One block per popped item -- the shape match_state_rule already wants. Only thread 0 touches
// the queue, so a pop is one claim per block rather than a race between its threads.
//
// Exit when the queue is empty. That is exact for a queue seeded once and never grown: no work
// can appear after a failed pop. It is NOT the rule the full model uses, where a failed pop may
// only mean a lull -- that is what TerminationDetector's stable-observation window is for, and
// it wires in when the roles start feeding each other.
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

        match_state_rule(ds, rules, item.state_id, item.rule_id, out);
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
                                            static_cast<uint32_t>(states.size()), num_rules);
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

}  // namespace hg_gpu
