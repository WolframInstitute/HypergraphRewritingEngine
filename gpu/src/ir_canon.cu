#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/wl_hash.hpp"   // wl_hash_state_device — the size-tolerant fallback
#include "hgcommon/ir_core.hpp" // the canonical hash itself, shared with the host

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hg_gpu {

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
                                          hgcommon::IR_DEVICE_GENERATORS)
             + 8;
    }
    // Even, so every slot base keeps the 8-byte alignment the pool starts with.
    HG_HD uint64_t stride() const { return (words() + 1ull) & ~1ull; }
};

// Depth the device attempts. A state discrete after refinement needs none of it; this bounds
// what a state that does search may use, and the pool is sized for it.
constexpr uint32_t kIRDeviceDepth = 8;

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
__device__ bool flatten_state(DeviceState ds, StateId sid, uint32_t* slot,
                              const IrSlotShape& shape,
                              uint8_t*& ea, uint32_t*& eoff, uint32_t*& ev,
                              uint32_t& n_edges, uint32_t& n_verts, uint32_t& total_occ,
                              VertexId*& verts_local, uint32_t* flat_to_slot = nullptr)
{
    ea   = reinterpret_cast<uint8_t*>(slot);
    eoff = slot + shape.ea_words();
    ev   = eoff + shape.eoff_words();
    verts_local = ev + shape.cap_occs;

    n_edges = 0; n_verts = 0; total_occ = 0;
    if (sid >= ds.max_states) return false;
    StateEdgeSlice sl = ds.state_edge_slices[sid];
    const uint32_t num_edges_live = ds.edge_pool.size();

    for (uint32_t k = 0; k < sl.count; ++k) {
        if (n_edges >= shape.cap_edges) return false;
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
            if (total_occ >= shape.cap_occs) return false;
            VertexId v = ds.vertex_pool.at(e.vertex_offset + p);
            uint32_t vi = 0xFFFFFFFFu;
            for (uint32_t i = 0; i < n_verts; ++i) if (verts_local[i] == v) { vi = i; break; }
            if (vi == 0xFFFFFFFFu) {
                if (n_verts >= shape.cap_verts) return false;
                vi = n_verts++;
                verts_local[vi] = v;
            }
            ev[total_occ++] = v;   // raw id; renumbered below, once the vertex set is known
        }
    }

    // Renumber vertices by SORTED id, which is what the host does
    // (hypergraph.cpp: sort, unique, lower_bound). Encounter order would be cheaper and gives
    // the same HASH, but it does not give the same RANKS: on a state with a nontrivial
    // automorphism group the canonical labelling is a coset, the core's within-cell tie-break is
    // what selects a representative from it, and that tie-break reads the vertex numbering. Two
    // numberings pick two different representatives, so the ranks differ while both are
    // internally valid -- and an event identity keyed on ranks then disagrees between devices.
    //
    // Insertion sort because n_verts is a state's vertex count, the array is already in the
    // slot, and one thread owns it.
    for (uint32_t i = 1; i < n_verts; ++i) {
        const VertexId key = verts_local[i];
        uint32_t j = i;
        while (j > 0 && verts_local[j - 1] > key) { verts_local[j] = verts_local[j - 1]; --j; }
        verts_local[j] = key;
    }
    for (uint32_t i = 0; i < total_occ; ++i) {
        // Binary search: verts_local is sorted and holds every vertex that appears.
        uint32_t lo = 0, hi = n_verts;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (verts_local[mid] < ev[i]) lo = mid + 1; else hi = mid;
        }
        ev[i] = lo;
    }
    return true;
}

// Grid-stride over the requested state range. One thread canonicalizes one state at a time,
// exactly as the shared WL core is driven: the parallelism in this computation is ACROSS
// states, and the pool is bounded by the grid rather than by the range.
__global__ void k_ir_canon_range(DeviceState ds, uint32_t lo, uint32_t hi,
                                 uint64_t* out, uint32_t* pool, uint64_t slot_words,
                                 IrSlotShape shape, uint32_t* degraded)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    // A null pool means the host could not afford even one slot for this batch's state size.
    // Every state then keeps the 1-WL hash, and the host has already counted them.
    if (pool == nullptr) {
        for (uint32_t i = lo + tid; i < hi; i += stride) {
            out[i - lo] = wl_hash_state_device(ds, i);
            ds.errors.record(ErrorKind::kIRDegradedToWL);
        }
        return;
    }
    uint32_t* slot = pool + tid * slot_words;

    for (uint32_t i = lo + tid; i < hi; i += stride) {
        const StateId sid = i;
        uint8_t* ea; uint32_t* eoff; uint32_t* ev; VertexId* verts_local;
        uint32_t n_edges, n_verts, total_occ;

        if (!flatten_state(ds, sid, slot, shape, ea, eoff, ev, n_edges, n_verts, total_occ,
                           verts_local)) {
            // Larger than the slot's bounds, so the dedup key for this state is the 1-WL
            // hash. That is NOT a correct dedup key. Isomorphism-invariance is one
            // directional: WL never separates isomorphic states, but it does MERGE
            // non-isomorphic ones -- tools/ir_vs_wl collides on the prism against K3,3, six
            // vertices, and on the rook's graph against Shrikhande. Nothing bounds how often
            // an evolution reaches such a state, so the count below is the only honest
            // signal, and it is a report of a defect rather than of a tuning parameter.
            out[i - lo] = wl_hash_state_device(ds, sid);
            atomicAdd(degraded, 1u);
            ds.errors.record(ErrorKind::kIRDegradedToWL);
            continue;
        }
        if (n_edges == 0) { out[i - lo] = 0; continue; }

        uint32_t* scratch = verts_local + shape.cap_verts + shape.rank_words();
        hgcommon::IrResult r{0, hgcommon::IR_NEED_DEPTH, 0};
        for (uint32_t depth = 1; depth <= shape.depth; depth *= shape.depth) {
            r = hgcommon::ir_canonical_hash(ea, eoff, ev, n_edges, n_verts, total_occ,
                                            scratch, depth, nullptr,
                                            hgcommon::IR_DEVICE_GENERATORS);
            if (r.status != hgcommon::IR_NEED_DEPTH) break;
        }
        if (r.status == hgcommon::IR_NEED_DEPTH) {
            // An individualization path deeper than the pool is sized for. Same hazard as
            // above: the fallback key can merge non-isomorphic states.
            out[i - lo] = wl_hash_state_device(ds, sid);
            atomicAdd(degraded, 1u);
            ds.errors.record(ErrorKind::kIRDegradedToWL);
        } else {
            out[i - lo] = r.hash;
        }
    }
}

// Largest (edge count, occurrence count) over a state range, so the slot can be sized to the
// batch instead of to a constant. One cheap pass: the alternative is bounding occurrences by
// edges * kMaxArity, which is 8x loose on the arity-2 edges that dominate real rules.
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

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::ir_canon ") + what + ": " +
                                 cudaGetErrorString(err));
    }
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
__device__ ExactHashStatus state_exact_hash_device(DeviceState ds, StateId sid,
                                                   DeviceArena::View arena,
                                                   uint32_t*& slot, uint64_t& slot_words,
                                                   uint64_t& out_hash, bool want_ranks,
                                                   bool want_orbits) {
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
    if (n_edges == 0) { out_hash = 0; return ExactHashStatus::kOk; }

    IrSlotShape shape;
    shape.cap_edges = n_edges + 1;
    shape.cap_occs  = total_occ + 1;
    shape.cap_verts = total_occ + 1;   // every occurrence could be a distinct vertex
    shape.depth     = kIRDeviceDepth;

    const uint64_t need = shape.stride();
    if (need > slot_words) {
        // Grow. The previous slot is abandoned rather than freed -- a bump arena has no free,
        // and with each block growing at most to its own peak the waste is bounded.
        uint32_t* bigger = arena.claim(need);
        if (!bigger) return ExactHashStatus::kArenaExhausted;
        slot = bigger;
        slot_words = need;
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
                       verts_local, (ranks || orbits) ? flat_to_slot : nullptr)) {
        // Cannot happen: the shape was sized from this state's own counts. Reported under its
        // own status rather than silently hashing something else, and deliberately NOT as an
        // arena failure -- growing the config would not fix it, so calling it retryable would
        // send the host into six pointless doublings.
        return ExactHashStatus::kMalformedState;
    }

    uint32_t* scratch = orbit_buf + shape.cap_edges;
    hgcommon::IrResult r{0, hgcommon::IR_NEED_DEPTH, 0};
    for (uint32_t depth = 1; depth <= shape.depth; depth *= shape.depth) {
        r = hgcommon::ir_canonical_hash(ea, eoff, ev, fn_edges, n_verts, fn_occ,
                                        scratch, depth, ranks ? rank_buf : nullptr,
                                        hgcommon::IR_DEVICE_GENERATORS,
                                        orbits ? orbit_buf : nullptr, nullptr);
        if (r.status != hgcommon::IR_NEED_DEPTH) break;
    }
    if (r.status == hgcommon::IR_NEED_DEPTH) return ExactHashStatus::kDepthExceeded;
    out_hash = r.hash;

    if (ranks) {
        StateEdgeSlice sl = ds.state_edge_slices[sid];
        for (uint32_t k = 0; k < sl.count; ++k) ds.state_edge_rank[sl.offset + k] = UINT32_MAX;
        for (uint32_t j = 0; j < fn_edges; ++j)
            ds.state_edge_rank[sl.offset + flat_to_slot[j]] = rank_buf[j];
    }
    if (orbits) {
        StateEdgeSlice sl = ds.state_edge_slices[sid];
        for (uint32_t k = 0; k < sl.count; ++k) ds.state_edge_orbit[sl.offset + k] = UINT32_MAX;
        uint32_t num_orbits = 0;
        for (uint32_t j = 0; j < fn_edges; ++j) {
            ds.state_edge_orbit[sl.offset + flat_to_slot[j]] = orbit_buf[j];
            if (orbit_buf[j] + 1 > num_orbits) num_orbits = orbit_buf[j] + 1;
        }
        ds.state_num_orbits[sid] = num_orbits;
    }
    if (ranks || orbits) __threadfence();
    return ExactHashStatus::kOk;
}

// Memory the scratch pool may take. Slot size grows with state size, so this trades
// concurrency against state size rather than refusing large states: a batch of big states
// runs with fewer resident threads, not with a coarser hash.
static constexpr uint64_t kIRPoolBudgetBytes = 512ull << 20;   // 512 MB
static constexpr uint32_t kIRMaxThreads      = 1024;

static uint32_t g_last_degraded_states = 0;

void compute_state_ir_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device) {
    if (hi <= lo) return;
    const uint32_t n = hi - lo;

    // Measure the batch, then size the slot to it. The previous fixed bounds meant any state
    // past them was deduplicated by the 1-WL hash, which MERGES non-isomorphic states -- a
    // wrong dedup key, not a coarser one (tools/ir_vs_wl collides on six-vertex graphs). A
    // state's size is knowable here, so nothing has to be given up for it.
    uint32_t* d_max = nullptr;
    check(cudaMalloc(&d_max, sizeof(uint32_t) * 2), "measure alloc");
    check(cudaMemset(d_max, 0, sizeof(uint32_t) * 2), "measure clear");
    {
        const uint32_t block = 128;
        const uint32_t grid = (n + block - 1) / block;
        k_measure_states<<<grid ? grid : 1, block>>>(engine.device(), lo, hi, d_max);
        check(cudaDeviceSynchronize(), "measure sync");
    }
    uint32_t h_max[2] = {0, 0};
    check(cudaMemcpy(h_max, d_max, sizeof(h_max), cudaMemcpyDeviceToHost), "measure copy");
    cudaFree(d_max);

    const uint32_t max_edges = h_max[0], max_occs = h_max[1];
    if (max_edges == 0) {
        // Every state in the range is empty; nothing to canonicalize.
        check(cudaMemset(out_hashes_device, 0, sizeof(uint64_t) * n), "empty range clear");
        g_last_degraded_states = 0;
        return;
    }

    IrSlotShape shape;
    shape.cap_edges = max_edges + 1;
    shape.cap_occs  = max_occs + 1;
    shape.cap_verts = max_occs + 1;      // every occurrence could be a distinct vertex
    shape.depth     = kIRDeviceDepth;

    const uint64_t slot_words = shape.stride();
    const uint64_t slot_bytes = slot_words * sizeof(uint32_t);

    uint32_t threads = n < kIRMaxThreads ? n : kIRMaxThreads;
    const uint64_t affordable = kIRPoolBudgetBytes / (slot_bytes ? slot_bytes : 1);
    if (affordable == 0) {
        // A single slot exceeds the whole budget. Rather than silently hand back a wrong
        // dedup key, refuse: the caller sees the count and the states keep the 1-WL hash.
        k_ir_canon_range<<<1, 1>>>(engine.device(), lo, hi, out_hashes_device,
                                   nullptr, 0, IrSlotShape{}, nullptr);
        check(cudaDeviceSynchronize(), "degenerate range sync");
        g_last_degraded_states = n;
        return;
    }
    if (threads > affordable) threads = static_cast<uint32_t>(affordable);

    uint32_t* pool = nullptr;
    check(cudaMalloc(&pool, size_t(threads) * slot_bytes), "pool alloc");
    uint32_t* degraded = nullptr;
    check(cudaMalloc(&degraded, sizeof(uint32_t)), "degraded alloc");
    check(cudaMemset(degraded, 0, sizeof(uint32_t)), "degraded clear");

    const uint32_t block = threads < 64 ? threads : 64;
    const uint32_t grid = (threads + block - 1) / block;
    k_ir_canon_range<<<grid, block>>>(engine.device(), lo, hi, out_hashes_device,
                                      pool, slot_words, shape, degraded);
    check(cudaDeviceSynchronize(), "k_ir_canon_range sync");

    uint32_t h_degraded = 0;
    check(cudaMemcpy(&h_degraded, degraded, sizeof(uint32_t), cudaMemcpyDeviceToHost),
          "degraded copy");
    cudaFree(degraded);
    cudaFree(pool);
    g_last_degraded_states = h_degraded;
}

uint32_t last_ir_degraded_states() { return g_last_degraded_states; }

uint64_t compute_state_ir_hash_host(const EngineState& engine, StateId sid) {
    uint64_t* d = nullptr;
    check(cudaMalloc(&d, sizeof(uint64_t)), "alloc");
    compute_state_ir_hashes_range(engine, sid, sid + 1, d);
    uint64_t h = 0;
    check(cudaMemcpy(&h, d, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
    cudaFree(d);
    return h;
}

}  // namespace hg_gpu
