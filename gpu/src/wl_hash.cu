#include "hg_gpu/wl_hash.hpp"
#include "hgcommon/wl_core.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hg_gpu {

namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime  = 1099511628211ULL;

__host__ __device__ inline uint64_t fnv_mix(uint64_t h, uint64_t x) {
    h ^= x;
    h *= kFnvPrime;
    return h;
}

__device__ void insertion_sort_u64(uint64_t* a, uint32_t n) {
    for (uint32_t i = 1; i < n; ++i) {
        uint64_t key = a[i];
        uint32_t j = i;
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
}

}  // namespace (close anon; wl_hash_state_device needs external linkage)

__device__ uint64_t wl_hash_state_device(DeviceState ds, StateId sid) {
    if (sid >= ds.max_states) return 0;
    StateEdgeSlice sl = ds.state_edge_slices[sid];

    // 1. Collect edges in this state from the CSR slice.
    EdgeId   local_edges[kMaxWlEdges];
    uint32_t n_edges = 0;
    uint32_t num_edges_live = ds.edge_pool.counter ? *ds.edge_pool.counter : 0u;
    for (uint32_t k = 0; k < sl.count && n_edges < kMaxWlEdges; ++k) {
        EdgeId eid = ds.state_edge_ids[sl.offset + k];
        if (eid >= num_edges_live) continue;
        if (eid >= ds.edge_pool.capacity) continue;
        local_edges[n_edges++] = eid;
    }
    if (n_edges == 0) return 0;

    // 2. Collect unique vertices (linear-probe dedup; n is small in practice).
    VertexId local_verts[kMaxWlVertices];
    uint32_t n_verts = 0;
    for (uint32_t e = 0; e < n_edges; ++e) {
        const Edge& edge = ds.edge_pool.at(local_edges[e]);
        if (edge.arity == 0 || edge.arity > kMaxArity) continue;
        if (edge.vertex_offset + edge.arity > ds.vertex_pool.capacity) continue;
        for (uint8_t i = 0; i < edge.arity; ++i) {
            VertexId v = ds.vertex_pool.at(edge.vertex_offset + i);
            bool found = false;
            for (uint32_t j = 0; j < n_verts; ++j) {
                if (local_verts[j] == v) { found = true; break; }
            }
            if (!found) {
                if (n_verts >= kMaxWlVertices) return 0;  // overflow sentinel
                local_verts[n_verts++] = v;
            }
        }
    }

    // Helper: vertex → local index (linear search; small n).
    auto local_index = [&] (VertexId v) -> uint32_t {
        for (uint32_t j = 0; j < n_verts; ++j) {
            if (local_verts[j] == v) return j;
        }
        return n_verts;  // unreachable after the dedup above
    };

    // Flatten to local-index edges and hash with the shared WL core, so this
    // matches the CPU WLHash bit-for-bit (one implementation, no drift).
    uint8_t  ea  [kMaxWlEdges];
    uint32_t eoff[kMaxWlEdges];
    uint32_t ev  [kMaxWlEdges * kMaxArity];
    uint32_t ev_n = 0;
    for (uint32_t e = 0; e < n_edges; ++e) {
        const Edge& edge = ds.edge_pool.at(local_edges[e]);
        eoff[e] = ev_n;
        ea[e]   = edge.arity;
        for (uint8_t i = 0; i < edge.arity; ++i)
            ev[ev_n++] = local_index(ds.vertex_pool.at(edge.vertex_offset + i));
    }

    // Per-thread scratch (fixed-size local memory, bounded by kMaxWl*).
    constexpr uint32_t kOccCap = kMaxWlEdges * kMaxArity;
    uint64_t cur[kMaxWlVertices], nxt[kMaxWlVertices], dscr[kMaxWlVertices];
    uint32_t occ_off[kMaxWlVertices + 1], occ_edge[kOccCap];
    uint8_t  occ_pos[kOccCap];
    uint64_t nbr[kOccCap];

    return hgcommon::wl_canonical_hash(
        ea, eoff, ev, n_edges, n_verts, hgcommon::WL_MAX_REFINE_ITERS,
        cur, nxt, occ_off, occ_edge, occ_pos, nbr, kOccCap, dscr, nullptr);
}

namespace {

__global__ void k_wl_hash_states(DeviceState ds, uint32_t num_states, uint64_t* out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;
    out[tid] = wl_hash_state_device(ds, tid);
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::wl_hash ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

}  // namespace

void compute_state_wl_hashes(const EngineState& engine,
                             uint32_t  num_states,
                             uint64_t* out_hashes_device) {
    if (num_states == 0) return;
    int block = 64;
    int grid  = (int)((num_states + block - 1) / block);
    k_wl_hash_states<<<grid, block>>>(engine.device(), num_states, out_hashes_device);
    check(cudaDeviceSynchronize(), "k_wl_hash_states sync");
}

uint64_t compute_state_wl_hash_host(const EngineState& engine, StateId sid) {
    uint64_t* d = nullptr;
    check(cudaMalloc(&d, sizeof(uint64_t)), "alloc");
    compute_state_wl_hashes(engine, sid + 1, d);
    uint64_t h = 0;
    check(cudaMemcpy(&h, d + sid, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
    cudaFree(d);
    return h;
}


// ---------------------------------------------------------------------------
// Content-ordered (non-isomorphic) hash for CanonicalizationMode::Automatic.
// Mirrors CPU compute_content_ordered_hash: edge count, then per edge (edge-id
// order — the CSR is sorted by edge id) arity + ACTUAL vertex ids + a sentinel.
__device__ uint64_t content_hash_state_device(DeviceState ds, StateId sid) {
    if (sid >= ds.max_states) return 0;
    StateEdgeSlice sl = ds.state_edge_slices[sid];
    uint32_t live = ds.edge_pool.counter ? *ds.edge_pool.counter : 0u;
    uint32_t ne = 0;
    for (uint32_t k = 0; k < sl.count; ++k) {
        EdgeId eid = ds.state_edge_ids[sl.offset + k];
        if (eid < live && eid < ds.edge_pool.capacity) ++ne;
    }
    uint64_t h = hgcommon::fnv_hash(hgcommon::FNV_OFFSET, hgcommon::mix64(ne));
    for (uint32_t k = 0; k < sl.count; ++k) {
        EdgeId eid = ds.state_edge_ids[sl.offset + k];
        if (eid >= live || eid >= ds.edge_pool.capacity) continue;
        const Edge& e = ds.edge_pool.at(eid);
        h = hgcommon::fnv_hash(h, hgcommon::mix64(static_cast<uint64_t>(e.arity)));
        for (uint8_t i = 0; i < e.arity; ++i)
            h = hgcommon::fnv_hash(h, hgcommon::mix64(
                    static_cast<uint64_t>(ds.vertex_pool.at(e.vertex_offset + i))));
        h = hgcommon::fnv_hash(h, 0xDEADBEEFCAFEBABEull);
    }
    return h;
}

namespace {
__global__ void k_content_hash_range(DeviceState ds, uint32_t lo, uint32_t hi, uint64_t* out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = hi - lo;
    if (tid >= n) return;
    out[tid] = content_hash_state_device(ds, lo + tid);
}
}  // namespace

void compute_state_content_hashes_range(const EngineState& engine,
                                        uint32_t lo, uint32_t hi,
                                        uint64_t* out_hashes_device) {
    if (hi <= lo) return;
    uint32_t n = hi - lo;
    int block = 64, grid = (int)((n + block - 1) / block);
    k_content_hash_range<<<grid, block>>>(engine.device(), lo, hi, out_hashes_device);
    check(cudaDeviceSynchronize(), "k_content_hash_range sync");
}

}  // namespace hg_gpu
