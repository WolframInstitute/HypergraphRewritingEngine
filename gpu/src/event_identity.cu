#include "hg_gpu/event_identity.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hg_gpu {
namespace {

// One thread per state, grid-stride. Each claims its own arena slot, sized from its own state,
// and keeps it across iterations so it re-claims only when it needs a larger one.
__global__ void k_fill_exact_and_ranks(DeviceState ds, uint32_t lo, uint32_t hi,
                                       bool key_is_exact, bool want_ranks, bool want_orbits,
                                       DeviceArena::View arena) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    uint32_t* slot = nullptr;
    uint64_t  slot_words = 0;

    for (uint32_t sid = lo + tid; sid < hi; sid += stride) {
        if (sid >= ds.max_states) continue;
        uint64_t exact = 0;
        // Nothing to compute when the state mode already produced the exact hash AND neither
        // ranks nor orbits are wanted: the key IS the exact hash, so a second pass would
        // recompute it.
        if (key_is_exact && !want_ranks && !want_orbits) {
            ds.state_exact_hash[sid] = ds.state_canonical_hash[sid];
            continue;
        }
        const ExactHashStatus st =
            state_exact_hash_device(ds, sid, arena, slot, slot_words, exact, want_ranks,
                                    want_orbits);
        if (st != ExactHashStatus::kOk) {
            ds.errors.record(error_kind_for(st));
            continue;
        }
        ds.state_exact_hash[sid] = exact;
    }
}

__global__ void k_stamp_events(DeviceState ds, uint32_t lo, uint32_t hi,
                               EventSignatureKeys keys, DedupMap::DeviceView event_map) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t live = ds.event_pool.size();

    for (uint32_t eid = lo + tid; eid < hi; eid += stride) {
        if (eid >= live || eid >= ds.event_pool.capacity) continue;
        DeviceEvent& ev = ds.event_pool.at(eid);
        if (ev.id == INVALID_ID) continue;
        const StateId in_s  = ev.input_state;
        const StateId out_s = ev.output_state;
        const uint64_t in_h  = (in_s  < ds.max_states) ? ds.state_exact_hash[in_s]  : 0ull;
        const uint64_t out_h = (out_s < ds.max_states) ? ds.state_exact_hash[out_s] : 0ull;
        stamp_event_signature(ds, eid, keys, in_h, out_h, in_s, out_s, ev.step, ev.rule,
                              event_map);
    }
}

}  // namespace

void fill_event_identity_inputs(EngineState& engine, uint32_t lo, uint32_t hi,
                                EventSignatureKeys keys, bool key_is_exact,
                                DeviceArena& arena, bool want_orbits) {
    if ((keys == hgcommon::EVENT_SIG_NONE && !want_orbits) || hi <= lo) return;
    const bool want_ranks = event_keys_need_ranks(keys);
    if (want_ranks) engine.ensure_edge_ranks();

    // Grid CAPPED, and the kernel is grid-stride so the cap costs coverage nothing.
    //
    // Every thread here claims its own arena slot and grows it to the largest state it handles,
    // so concurrent slot holders -- and therefore arena demand -- scale with the LAUNCH, not
    // with the device. The arena this kernel draws from is sized for default_persistent_grid()
    // holders (see persistent_arena_words), so the launch is bounded to that many THREADS:
    // grid x block <= the holder budget the arena was sized for.
    const int block = 128;
    const int want  = static_cast<int>(((hi - lo) + block - 1) / block);
    const int cap_threads = static_cast<int>(default_persistent_grid());
    const int cap   = cap_threads / block > 0 ? cap_threads / block : 1;
    const int grid  = want < 1 ? 1 : (want > cap ? cap : want);
    k_fill_exact_and_ranks<<<grid, block>>>(
        engine.device(), lo, hi, key_is_exact, want_ranks, want_orbits, arena.view());
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "fill_event_identity_inputs sync");
}

void stamp_event_identity_range(EngineState& engine, uint32_t lo, uint32_t hi,
                                EventSignatureKeys keys, DedupMap& event_map) {
    if (keys == hgcommon::EVENT_SIG_NONE || hi <= lo) return;
    engine.ensure_event_identity();

    const int block = 128;
    const int grid  = static_cast<int>(((hi - lo) + block - 1) / block);
    k_stamp_events<<<grid > 0 ? grid : 1, block>>>(
        engine.device(), lo, hi, keys, event_map.view());
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "stamp_event_identity_range sync");
}

}  // namespace hg_gpu
