#include "hg_gpu/wl_hash.hpp"

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

    // 3. Initial colors: FNV(FNV_OFFSET, degree, per-occurrence(arity, pos))
    uint64_t colors    [kMaxWlVertices];
    uint64_t new_colors[kMaxWlVertices];
    for (uint32_t v = 0; v < n_verts; ++v) colors[v] = kFnvOffset;

    for (uint32_t e = 0; e < n_edges; ++e) {
        const Edge& edge = ds.edge_pool.at(local_edges[e]);
        for (uint8_t i = 0; i < edge.arity; ++i) {
            VertexId v = ds.vertex_pool.at(edge.vertex_offset + i);
            uint32_t li = local_index(v);
            colors[li] = fnv_mix(colors[li], static_cast<uint64_t>(edge.arity));
            colors[li] = fnv_mix(colors[li], static_cast<uint64_t>(i));
        }
    }

    // 4. Refinement iterations.
    // Per-vertex "neighbour hashes" scratch: upper-bounded by sum of (arity-1)
    // over all edges containing v. Tight bound ≤ n_edges*(kMaxArity-1).
    constexpr uint32_t kMaxNeighborEntries = kMaxWlEdges * (kMaxArity - 1);
    uint64_t neighbour[kMaxNeighborEntries];

    for (uint32_t iter = 0; iter < kMaxWlRefineIters; ++iter) {
        bool changed = false;

        for (uint32_t v = 0; v < n_verts; ++v) {
            uint64_t h = colors[v];
            uint32_t n_neighbour = 0;

            for (uint32_t e = 0; e < n_edges; ++e) {
                const Edge& edge = ds.edge_pool.at(local_edges[e]);
                if (edge.arity == 0 || edge.arity > kMaxArity) continue;
                if (edge.vertex_offset + edge.arity > ds.vertex_pool.capacity) continue;
                // Find v's position in this edge (if any).
                uint8_t my_pos = 0xFF;
                for (uint8_t i = 0; i < edge.arity; ++i) {
                    if (ds.vertex_pool.at(edge.vertex_offset + i) == local_verts[v]) {
                        my_pos = i;
                        break;
                    }
                }
                if (my_pos == 0xFF) continue;

                for (uint8_t k = 0; k < edge.arity; ++k) {
                    if (k == my_pos) continue;
                    VertexId u = ds.vertex_pool.at(edge.vertex_offset + k);
                    uint32_t ui = local_index(u);
                    if (n_neighbour < kMaxNeighborEntries) {
                        neighbour[n_neighbour++] = fnv_mix(colors[ui], static_cast<uint64_t>(k));
                    }
                }
            }

            insertion_sort_u64(neighbour, n_neighbour);
            for (uint32_t i = 0; i < n_neighbour; ++i) h = fnv_mix(h, neighbour[i]);

            if (h != colors[v]) changed = true;
            new_colors[v] = h;
        }

        for (uint32_t v = 0; v < n_verts; ++v) colors[v] = new_colors[v];
        if (!changed) break;
    }

    // 5. Final graph hash = FNV over sorted colour multiset.
    insertion_sort_u64(colors, n_verts);
    uint64_t result = kFnvOffset;
    result = fnv_mix(result, static_cast<uint64_t>(n_verts));
    result = fnv_mix(result, static_cast<uint64_t>(n_edges));
    for (uint32_t v = 0; v < n_verts; ++v) result = fnv_mix(result, colors[v]);
    return result;
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

namespace {
__global__ void k_wl_hash_range(DeviceState ds, uint32_t lo, uint32_t hi, uint64_t* out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = hi - lo;
    if (tid >= n) return;
    out[tid] = wl_hash_state_device(ds, lo + tid);
}
}  // namespace

void compute_state_wl_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device) {
    if (hi <= lo) return;
    uint32_t n = hi - lo;
    int block = 64;
    int grid  = (int)((n + block - 1) / block);
    k_wl_hash_range<<<grid, block>>>(engine.device(), lo, hi, out_hashes_device);
    check(cudaDeviceSynchronize(), "k_wl_hash_range sync");
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

}  // namespace hg_gpu
