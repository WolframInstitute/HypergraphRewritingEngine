#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/wl_hash.hpp"   // wl_hash_state_device — the size-tolerant fallback
#include "hgcommon/ir_core.hpp" // the canonical hash itself, shared with the host

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hg_gpu {

// States in the last range whose exact hash could not be produced within the device
// scratch bounds, so they carry the coarser 1-WL hash instead. Read via
// last_ir_degraded_states so a caller can report the degradation rather than absorb it.
static uint32_t g_last_degraded_states = 0;

namespace {

// Per-state bounds a scratch slot is sized for. A state past any of them takes the 1-WL
// fallback and is counted (see the header).
constexpr uint32_t kMaxIRVerts = 128;
constexpr uint32_t kMaxIREdges = 128;
constexpr uint32_t kMaxIROccs  = 256;
// Per-thread global-memory slot: the flattened state, then the shared core's scratch.
// Shared memory cannot hold the search's per-level partitions, and the core wants one
// contiguous span, so the slot is carved from a device pool the launcher owns.
HG_HD inline uint64_t ir_slot_words(uint32_t max_verts, uint32_t max_edges,
                                    uint32_t max_occs, uint32_t depth) {
    return (max_edges + 3) / 4          // ea, as bytes rounded to a word
         + (max_edges + 1)              // eoff
         + max_occs                     // ev
         + max_verts                    // verts_local: the thread's local-index table
         + hgcommon::ir_scratch_words(max_verts, max_edges, max_occs, depth,
                                      hgcommon::IR_DEVICE_GENERATORS)
         + 8;                           // alignment slack
}

// Slot stride, rounded so every slot base keeps the 8-byte alignment the pool starts with --
// a uint64 view of a 4-byte-aligned slot faults on the device.
HG_HD inline uint64_t ir_slot_stride(uint32_t max_verts, uint32_t max_edges,
                                     uint32_t max_occs, uint32_t depth) {
    return (ir_slot_words(max_verts, max_edges, max_occs, depth) + 1ull) & ~1ull;
}

// Depth the device attempts. A state discrete after refinement needs none of it; this bounds
// what a state that does search may use, and the pool is sized for it.
constexpr uint32_t kIRDeviceDepth = 8;

// Flatten a state's CSR slice to the core's convention: local vertex indices in encounter
// order (which the core's result does not depend on -- the tie-break it uses only orders
// vertices inside a cell). Returns false if the state exceeds the slot's bounds.
__device__ bool flatten_state(DeviceState ds, StateId sid, uint32_t* slot,
                              uint8_t*& ea, uint32_t*& eoff, uint32_t*& ev,
                              uint32_t& n_edges, uint32_t& n_verts, uint32_t& total_occ,
                              VertexId*& verts_local)
{
    ea   = reinterpret_cast<uint8_t*>(slot);
    eoff = slot + (kMaxIREdges + 3) / 4;
    ev   = eoff + (kMaxIREdges + 1);
    verts_local = ev + kMaxIROccs;

    n_edges = 0; n_verts = 0; total_occ = 0;
    if (sid >= ds.max_states) return false;
    StateEdgeSlice sl = ds.state_edge_slices[sid];
    const uint32_t num_edges_live = ds.edge_pool.counter ? *ds.edge_pool.counter : 0u;

    for (uint32_t k = 0; k < sl.count; ++k) {
        if (n_edges >= kMaxIREdges) return false;
        EdgeId eid = ds.state_edge_ids[sl.offset + k];
        if (eid >= num_edges_live) continue;
        if (eid >= ds.edge_pool.capacity) continue;

        const Edge& e = ds.edge_pool.at(eid);
        if (e.arity == 0 || e.arity > kMaxArity) continue;
        if (e.vertex_offset + e.arity > ds.vertex_pool.capacity) continue;

        eoff[n_edges] = total_occ;
        ea[n_edges]   = e.arity;
        ++n_edges;
        for (uint8_t p = 0; p < e.arity; ++p) {
            if (total_occ >= kMaxIROccs) return false;
            VertexId v = ds.vertex_pool.at(e.vertex_offset + p);
            uint32_t vi = 0xFFFFFFFFu;
            for (uint32_t i = 0; i < n_verts; ++i) if (verts_local[i] == v) { vi = i; break; }
            if (vi == 0xFFFFFFFFu) {
                if (n_verts >= kMaxIRVerts) return false;
                vi = n_verts++;
                verts_local[vi] = v;
            }
            ev[total_occ++] = vi;
        }
    }
    return true;
}

// Grid-stride over the requested state range. One thread canonicalizes one state at a time,
// exactly as the shared WL core is driven: the parallelism in this computation is ACROSS
// states, and the pool is bounded by the grid rather than by the range.
__global__ void k_ir_canon_range(DeviceState ds, uint32_t lo, uint32_t hi,
                                 uint64_t* out, uint32_t* pool, uint64_t slot_words,
                                 uint32_t* degraded)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    uint32_t* slot = pool + tid * slot_words;

    for (uint32_t i = lo + tid; i < hi; i += stride) {
        const StateId sid = i;
        uint8_t* ea; uint32_t* eoff; uint32_t* ev; VertexId* verts_local;
        uint32_t n_edges, n_verts, total_occ;

        if (!flatten_state(ds, sid, slot, ea, eoff, ev, n_edges, n_verts, total_occ,
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
            continue;
        }
        if (n_edges == 0) { out[i - lo] = 0; continue; }

        uint32_t* scratch = verts_local + kMaxIRVerts;
        hgcommon::IrResult r{0, hgcommon::IR_NEED_DEPTH, 0};
        for (uint32_t depth = 1; depth <= kIRDeviceDepth; depth *= kIRDeviceDepth) {
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
        } else {
            out[i - lo] = r.hash;
        }
    }
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::ir_canon ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

}  // namespace

// Threads resident at once. The scratch pool is one slot per thread, so this trades memory
// (about 48 KB a slot at kIRDeviceDepth) against occupancy.
static constexpr uint32_t kIRMaxThreads = 1024;

void compute_state_ir_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device) {
    if (hi <= lo) return;
    const uint32_t n = hi - lo;
    const uint32_t threads = n < kIRMaxThreads ? n : kIRMaxThreads;
    const uint64_t slot_words = ir_slot_stride(kMaxIRVerts, kMaxIREdges, kMaxIROccs,
                                               kIRDeviceDepth);

    uint32_t* pool = nullptr;
    check(cudaMalloc(&pool, size_t(threads) * slot_words * sizeof(uint32_t)), "pool alloc");
    uint32_t* degraded = nullptr;
    check(cudaMalloc(&degraded, sizeof(uint32_t)), "degraded alloc");
    check(cudaMemset(degraded, 0, sizeof(uint32_t)), "degraded clear");

    const uint32_t block = 64;
    const uint32_t grid = (threads + block - 1) / block;
    k_ir_canon_range<<<grid, block>>>(engine.device(), lo, hi, out_hashes_device,
                                      pool, slot_words, degraded);
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
