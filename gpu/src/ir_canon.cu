#include "hgcommon/namespace.hpp"
#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/cuda_check.hpp"
#include "hgcommon/ir_core.hpp" // the canonical hash itself, shared with the host

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

// Slot geometry for one thread: every field a flattened state needs, then the shared core's
// scratch behind it. Sized from the states that will use it, never from a constant, because a
// state past a fixed bound would have to be keyed by the 1-WL hash and that MERGES
// non-isomorphic states.
//
// The slot lives in global memory: shared memory cannot hold the search's per-level
// partitions, and the core wants one contiguous span.
struct IrSlotShape {
    uint32_t cap_verts = 0;
    uint32_t cap_edges = 0;
    uint32_t cap_occs  = 0;
    // Generator rows the scratch is sized for. Must match the budget handed to the core, or
    // the search would write past what this slot reserved.
    uint32_t generators = hgcommon::IR_DEVICE_GENERATORS;
    uint32_t depth     = 0;

    HG_HD uint32_t ea_words()   const { return (cap_edges + 3) / 4; }
    HG_HD uint32_t eoff_words() const { return cap_edges + 1; }
    // Three cap_edges spans sit between the flattened state and the core's scratch: the ranks
    // the core reports, the CSR slot each flattened edge came from, and the per-edge orbits.
    // The slot map exists because flattening skips edges the slice still holds, so flat index
    // j and slot k diverge and neither ranks nor orbits can be scattered back without it. All
    // three are dwarfed by ir_scratch_words.
    HG_HD uint32_t rank_words() const { return 3u * cap_edges; }
    HG_HD uint64_t words() const {
        return ea_words() + eoff_words() + cap_occs + cap_verts + rank_words()
             + hgcommon::ir_scratch_words(cap_verts, cap_edges, cap_occs, depth,
                                          generators)
             + 8;
    }
    // Even, so every slot base keeps the 8-byte alignment the pool starts with.
    HG_HD uint64_t stride() const { return (words() + 1ull) & ~1ull; }
};


// Flatten a state's CSR slice to the core's convention: local vertex indices assigned in SORTED
// VERTEX ID order, which is what the host does. Returns false if the state exceeds the slot's
// bounds.
//
// The numbering is load-bearing and encounter order is NOT sufficient. It yields the same HASH,
// because the hash is taken over the winning canonical form, but it does not yield the same
// RANKS: on a state with a nontrivial automorphism group the canonical labelling is a COSET, and
// the core's within-cell tie-break is what picks a representative out of it. That tie-break reads
// the vertex numbering, so two numberings pick two different representatives -- each internally
// valid, and disagreeing about which edge holds which rank.
//
// `flat_to_slot`, when non-null, receives the CSR slot each flattened edge came from, which is
// what lets a caller scatter the core's per-edge ranks back onto the slice.
//
// Under a fanning policy the WRITES are the leader's: the flattening is a sequential scan
// whose outputs (n_edges, total_occ, and each edge's spans) each depend on how many edges the
// scan has kept so far. The span pointers are pure layout arithmetic and land on every lane;
// the counts cross by shuffle so the return and everything derived from it stay uniform.
template <class Par>
__device__ bool flatten_state(DeviceState ds, StateId sid, uint32_t* slot,
                              const IrSlotShape& shape,
                              uint8_t*& ea, uint32_t*& eoff, uint32_t*& ev,
                              uint32_t& n_edges, uint32_t& n_verts, uint32_t& total_occ,
                              VertexId*& verts_local, uint32_t* flat_to_slot, Par par)
{
    ea   = reinterpret_cast<uint8_t*>(slot);
    eoff = slot + shape.ea_words();
    ev   = eoff + shape.eoff_words();
    verts_local = ev + shape.cap_occs;

    n_edges = 0; n_verts = 0; total_occ = 0;
    if (sid >= ds.max_states) return false;
    uint32_t ok = 1u;
    if (par.leader()) {
        StateEdgeSlice sl = ds.state_edge_slices[sid];
        const uint32_t num_edges_live = ds.edge_pool.size();

        for (uint32_t k = 0; k < sl.count; ++k) {
            if (n_edges >= shape.cap_edges) { ok = 0u; break; }
            EdgeId eid = ds.state_edge_ids[sl.offset + k];
            if (eid >= num_edges_live) continue;
            if (eid >= ds.edge_pool.capacity) continue;

            const Edge& e = ds.edge_pool.at(eid);
            if (e.arity == 0 || e.arity > kMaxArity) continue;
            if (e.vertex_offset + e.arity > ds.vertex_pool.capacity) continue;

            eoff[n_edges] = total_occ;
            ea[n_edges]   = e.arity;
            if (flat_to_slot) flat_to_slot[n_edges] = k;
            ++n_edges;
            for (uint8_t p = 0; p < e.arity; ++p) {
                if (total_occ >= shape.cap_occs) { ok = 0u; break; }
                ev[total_occ++] = ds.vertex_pool.at(e.vertex_offset + p);
            }
            if (!ok) break;
        }

        // SORTED-UNIQUE renumbering through the shared rule (hgcommon::ir_renumber_sorted); the
        // convention note lives there. verts_local is sized cap_verts = total_occ + 1, so it
        // holds one copy of every occurrence as the helper's scratch.
        if (ok) n_verts = hgcommon::ir_renumber_sorted(ev, total_occ, verts_local);
    }
    par.sync();
    ok        = par.bcast(ok);
    n_edges   = par.bcast(n_edges);
    total_occ = par.bcast(total_occ);
    n_verts   = par.bcast(n_verts);
    return ok != 0u;
}

}  // namespace

// Exact canonical hash of ONE state, sized and allocated entirely on device.
//
// The batched entry point measures a range on the host and shapes one slot to the largest
// state in it. A persistent loop has no range -- states arrive continuously and the largest is
// not knowable before the run -- so this sizes the slot from THIS state's own edge and
// occurrence counts and takes it from a device arena. No host in the loop, and no fixed
// per-state ceiling, which is what keeps the 1-WL fallback out of the exact path entirely.
//
// `slot` and `slot_words` are the caller's reusable scratch, carried across items so a worker
// claims again only when it needs a LARGER slot. Returns false when the arena is exhausted;
// the caller records that as a capacity overflow and returns partial work, because growing the
// arena would need the host.
//
// `want_ranks` additionally writes each edge's canonical RANK into ds.state_edge_rank at the
// edge's own CSR slot. It rides on the pass the hash already runs, so Automatic event identity
// costs no extra canonicalization -- only the scatter. Slots the flattening skipped keep
// UINT32_MAX, which the signature site counts rather than silently substitutes.
template <class Par>
__device__ ExactHashStatus state_exact_hash_device(DeviceState ds, StateId sid,
                                                   DeviceArena::View arena,
                                                   uint32_t*& slot, uint64_t& slot_words,
                                                   uint64_t& out_hash, bool want_ranks,
                                                   bool want_orbits, Par par) {
    // Measure this state: exact counts, not a bound. Occurrences are summed rather than taken
    // as edges * kMaxArity, which is 8x loose on the arity-2 edges real rules produce.
    uint32_t n_edges = 0, total_occ = 0;
    {
        if (sid >= ds.max_states) { out_hash = 0; return ExactHashStatus::kOk; }
        StateEdgeSlice sl = ds.state_edge_slices[sid];
        const uint32_t live = ds.edge_pool.size();
        for (uint32_t k = 0; k < sl.count; ++k) {
            EdgeId eid = ds.state_edge_ids[sl.offset + k];
            if (eid >= live || eid >= ds.edge_pool.capacity) continue;
            const Edge& e = ds.edge_pool.at(eid);
            if (e.arity == 0 || e.arity > kMaxArity) continue;
            ++n_edges; total_occ += e.arity;
        }
    }
    // No edges to canonicalize: the empty state's hash is the reserved one both engines agree
    // on, not 0 -- 0 is what this array holds for "not computed yet".
    if (n_edges == 0) {
        out_hash = hgcommon::EMPTY_STATE_CANONICAL_HASH;
        return ExactHashStatus::kOk;
    }

    IrSlotShape shape;
    shape.cap_edges = n_edges + 1;
    shape.cap_occs  = total_occ + 1;
    shape.cap_verts = total_occ + 1;   // every occurrence could be a distinct vertex
    shape.depth     = ds.ir_depth;
    shape.generators = ds.ir_generators;

    const uint64_t need = shape.stride();
    if (need > slot_words) {
        // Grow. The previous slot is abandoned rather than freed -- a bump arena has no free,
        // and with each block growing at most to its own peak the waste is bounded. The claim
        // is the leader's -- one arena bump per state, not one per lane -- and under a fanning
        // policy `slot`/`slot_words` refer to block-shared storage, so the leader's write is
        // every lane's after the sync; the verdict crosses by shuffle so a refusal returns on
        // every lane together, with the caller's old slot intact.
        uint32_t got = 0;
        if (par.leader()) {
            uint32_t* bigger = arena.claim(need);
            if (bigger) { slot = bigger; slot_words = need; got = 1u; }
        }
        par.sync();
        got = par.bcast(got);
        if (!got) return ExactHashStatus::kArenaExhausted;
    }

    // The slot's layout follows from the shape alone, so the rank/slot/orbit spans can be
    // addressed before the flattening runs and filled by that same pass.
    const bool ranks  = want_ranks && ds.state_edge_rank != nullptr;
    const bool orbits = want_orbits && ds.state_edge_orbit != nullptr;
    uint32_t* rank_buf = slot + shape.ea_words() + shape.eoff_words()
                       + shape.cap_occs + shape.cap_verts;
    uint32_t* flat_to_slot = rank_buf + shape.cap_edges;
    uint32_t* orbit_buf = flat_to_slot + shape.cap_edges;

    uint8_t* ea; uint32_t* eoff; uint32_t* ev; VertexId* verts_local;
    uint32_t fn_edges, n_verts, fn_occ;
    if (!flatten_state(ds, sid, slot, shape, ea, eoff, ev, fn_edges, n_verts, fn_occ,
                       verts_local, (ranks || orbits) ? flat_to_slot : nullptr, par)) {
        // Cannot happen: the shape was sized from this state's own counts. Reported under its
        // own status rather than silently hashing something else, and deliberately NOT as an
        // arena failure -- growing the config would not fix it, so calling it retryable would
        // send the host into six pointless doublings.
        return ExactHashStatus::kMalformedState;
    }

    // Depth 1 is "root only": it sizes for the state that is discrete after refinement, which
    // is the common one, and reports IR_NEED_DEPTH for the rest. Only then is the full depth
    // worth the scratch.
    uint32_t* scratch = orbit_buf + shape.cap_edges;
    auto run_at = [&](uint32_t depth) {
        return hgcommon::ir_canonical_hash(ea, eoff, ev, fn_edges, n_verts, fn_occ,
                                           scratch, depth, ranks ? rank_buf : nullptr,
                                           shape.generators,
                                           orbits ? orbit_buf : nullptr, nullptr,
                                           nullptr, nullptr, nullptr, par);
    };
    hgcommon::IrResult r = run_at(1);
    if (r.status == hgcommon::IR_NEED_DEPTH && shape.depth > 1) r = run_at(shape.depth);

    // THE HOST ESCALATES DEPTH WITHIN THE CALL; THE DEVICE MUST TOO.
    //
    // ir_canonicalization.cpp:75 tries {1, 8, IR_MAX_DEPTH_DEFAULT} per state. The device tried
    // only {1, ds.ir_depth} and, past that, returned kDepthExceeded -- which is classed retryable,
    // so the WRAPPER answered by doubling the config and RE-RUNNING THE ENTIRE EVOLUTION, up to
    // seven attempts. One state needing a deeper search therefore re-does every other state's
    // work, repeatedly.
    //
    // Observed directly on disc-l3a2g2r2: at depth 4 the run reports "overflow on IR search depth
    // (retryable: grow config) -- retrying (attempt 2/7)" and completes; at depth 5 the same
    // restart loop is what the run never gets out of, with each attempt larger than the last.
    //
    // The slot is the reason this was not simply a bigger constant: the arena slot was sized from
    // shape.depth, so going deeper needs a bigger slot. Re-shape, re-claim, re-flatten, retry --
    // paid only by the states that actually need it, which are rare.
    // Deeper rungs: the shipped default, then the true bound (a search individualises at most
    // one vertex per level, so n_verts + 1 admits every state -- the host ladder matches).
    const uint32_t deep_rungs[2] = {hgcommon::IR_MAX_DEPTH_DEFAULT, n_verts + 1};
    for (uint32_t ri = 0; ri < 2 && r.status == hgcommon::IR_NEED_DEPTH; ++ri) {
        if (deep_rungs[ri] <= shape.depth) continue;
        IrSlotShape deep = shape;
        deep.depth = deep_rungs[ri];
        const uint64_t deep_need = deep.stride();
        uint32_t got = 0;
        if (par.leader()) {
            uint32_t* deep_slot = arena.claim(deep_need);
            if (deep_slot) { slot = deep_slot; slot_words = deep_need; got = 1u; }
        }
        par.sync();
        got = par.bcast(got);
        if (!got) break;
        shape = deep;
        rank_buf = slot + shape.ea_words() + shape.eoff_words()
                 + shape.cap_occs + shape.cap_verts;
        flat_to_slot = rank_buf + shape.cap_edges;
        orbit_buf = flat_to_slot + shape.cap_edges;
        scratch = orbit_buf + shape.cap_edges;
        if (!flatten_state(ds, sid, slot, shape, ea, eoff, ev, fn_edges, n_verts, fn_occ,
                           verts_local, (ranks || orbits) ? flat_to_slot : nullptr, par)) break;
        r = run_at(shape.depth);
    }
    if (r.status == hgcommon::IR_NEED_DEPTH) return ExactHashStatus::kDepthExceeded;
    // Orbits fused over a truncated generator table are too fine, and the quotient
    // reconstruction slots on them. Report rather than publish them.
    if (r.status == hgcommon::IR_NEED_GENERATORS) return ExactHashStatus::kGeneratorsExceeded;
    out_hash = r.hash;

    // The scatters fan: each CSR slot is written by exactly one lane (flat_to_slot is
    // injective), and the policy's sync between the prefill and the scatter orders the two.
    if (ranks) {
        StateEdgeSlice sl = ds.state_edge_slices[sid];
        par.fan(sl.count, [&](uint32_t k) { ds.state_edge_rank[sl.offset + k] = UINT32_MAX; });
        par.fan(fn_edges, [&](uint32_t j) {
            ds.state_edge_rank[sl.offset + flat_to_slot[j]] = rank_buf[j];
        });
    }
    if (orbits) {
        StateEdgeSlice sl = ds.state_edge_slices[sid];
        par.fan(sl.count, [&](uint32_t k) { ds.state_edge_orbit[sl.offset + k] = UINT32_MAX; });
        par.fan(fn_edges, [&](uint32_t j) {
            ds.state_edge_orbit[sl.offset + flat_to_slot[j]] = orbit_buf[j];
        });
        if (par.leader()) {
            uint32_t num_orbits = 0;
            for (uint32_t j = 0; j < fn_edges; ++j)
                if (orbit_buf[j] + 1 > num_orbits) num_orbits = orbit_buf[j] + 1;
            ds.state_num_orbits[sid] = num_orbits;
        }
        par.sync();
    }
    if (ranks || orbits) __threadfence();
    return ExactHashStatus::kOk;
}

// The two callers' policies, instantiated here so the template body stays in this translation
// unit: every one-thread-per-state caller uses the serial policy, the persistent kernel the
// warp one.
template __device__ ExactHashStatus state_exact_hash_device<hgcommon::IrSerial>(
    DeviceState, StateId, DeviceArena::View, uint32_t*&, uint64_t&, uint64_t&, bool, bool,
    hgcommon::IrSerial);
template __device__ ExactHashStatus state_exact_hash_device<IrWarpAll>(
    DeviceState, StateId, DeviceArena::View, uint32_t*&, uint64_t&, uint64_t&, bool, bool,
    IrWarpAll);

namespace {

// Grid-stride over a state range, one thread per state at a time. The RULE is
// state_exact_hash_device above; this is only a launch shape, so a range and a device-resident
// loop cannot drift apart on what a state's exact hash is.
//
// A state the exact path cannot key leaves its hash at 0 -- which the readers already treat as
// "not computed", and which kUncomputedStateHash reports -- rather than taking a coarser key.
__global__ void k_exact_hash_range(DeviceState ds, uint32_t lo, uint32_t hi, uint64_t* out,
                                   DeviceArena::View arena) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    uint32_t* slot = nullptr;
    uint64_t slot_words = 0;
    for (uint32_t i = lo + tid; i < hi; i += stride) {
        uint64_t h = 0;
        const ExactHashStatus st =
            state_exact_hash_device(ds, i, arena, slot, slot_words, h, false, false);
        if (st != ExactHashStatus::kOk) { ds.errors.record(error_kind_for(st)); h = 0; }
        out[i - lo] = h;
    }
}

// Largest (edge count, occurrence count) over a state range, so the ARENA can be sized to the
// batch. One cheap pass: the alternative is bounding occurrences by edges * kMaxArity, which is
// 8x loose on the arity-2 edges real rules produce, and the depth blocks scale with it.
__global__ void k_measure_states(DeviceState ds, uint32_t lo, uint32_t hi, uint32_t* out_max) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t num_edges_live = ds.edge_pool.size();
    for (uint32_t i = lo + tid; i < hi; i += stride) {
        if (i >= ds.max_states) continue;
        StateEdgeSlice sl = ds.state_edge_slices[i];
        uint32_t n = 0, occ = 0;
        for (uint32_t k = 0; k < sl.count; ++k) {
            EdgeId eid = ds.state_edge_ids[sl.offset + k];
            if (eid >= num_edges_live || eid >= ds.edge_pool.capacity) continue;
            const Edge& e = ds.edge_pool.at(eid);
            if (e.arity == 0 || e.arity > kMaxArity) continue;
            ++n; occ += e.arity;
        }
        atomicMax(&out_max[0], n);
        atomicMax(&out_max[1], occ);
    }
}

// Memory the arena may take for this launch. Slot size grows with state size, so this trades
// CONCURRENCY against state size rather than refusing large states: a batch of big states runs
// with fewer resident threads, never with a coarser hash.
constexpr uint64_t kExactHashArenaBudgetBytes = 512ull << 20;   // 512 MB
constexpr uint32_t kExactHashRangeThreads     = 1024;

}  // namespace

void compute_state_ir_hashes_range(EngineState& engine, uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device) {
    if (hi <= lo) return;
    const uint32_t n = hi - lo;

    // Measure the batch so the arena covers a thread's largest claim. A thread whose state
    // needs more than the arena can give records kIRArenaExhausted and its hash stays 0; the
    // measurement is what keeps that from happening for a reason the host could have known.
    uint32_t* d_max = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_max, sizeof(uint32_t) * 2), "measure alloc");
    HG_CUDA_CHECK(cudaMemset(d_max, 0, sizeof(uint32_t) * 2), "measure clear");
    {
        const uint32_t block = 128;
        const uint32_t grid = (n + block - 1) / block;
        k_measure_states<<<grid ? grid : 1, block>>>(engine.device(), lo, hi, d_max);
        HG_CUDA_CHECK(cudaDeviceSynchronize(), "measure sync");
    }
    uint32_t h_max[2] = {0, 0};
    HG_CUDA_CHECK(cudaMemcpy(h_max, d_max, sizeof(h_max), cudaMemcpyDeviceToHost), "measure copy");
    cudaFree(d_max);

    IrSlotShape shape;
    shape.cap_edges  = h_max[0] + 1;
    shape.cap_occs   = h_max[1] + 1;
    shape.cap_verts  = h_max[1] + 1;    // every occurrence could be a distinct vertex
    shape.depth      = engine.config().ir_depth;
    shape.generators = engine.config().ir_generators;
    const uint64_t slot_words = shape.stride();

    uint32_t threads = n < kExactHashRangeThreads ? n : kExactHashRangeThreads;
    const uint64_t affordable =
        kExactHashArenaBudgetBytes / (slot_words * sizeof(uint32_t) + 1);
    if (affordable == 0) threads = 1;                 // one slot, however big it is
    else if (threads > affordable) threads = static_cast<uint32_t>(affordable);

    DeviceArena& arena = engine.ir_arena(slot_words * threads + 2ull * threads);
    arena.reset();
    const uint32_t block = threads < 64 ? threads : 64;
    const uint32_t grid = (threads + block - 1) / block;
    k_exact_hash_range<<<grid, block>>>(engine.device(), lo, hi, out_hashes_device, arena.view());
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "k_exact_hash_range sync");
}

uint64_t compute_state_ir_hash_host(EngineState& engine, StateId sid) {
    uint64_t* d = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d, sizeof(uint64_t)), "alloc");
    compute_state_ir_hashes_range(engine, sid, sid + 1, d);
    uint64_t h = 0;
    HG_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
    cudaFree(d);
    return h;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE