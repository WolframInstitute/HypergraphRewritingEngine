#include "hg_gpu/vertex_curvature.hpp"
#include "hg_gpu/cuda_check.hpp"
#include "hg_gpu/ir_canon.hpp"   // IrTile, the warp policy the geometry runs under
#include "hgcommon/state_geometry_core.hpp"

// One warp per state runs hgcommon::sg_state_geometry with the per-vertex curvature output, in a
// slice of scratch of its own. A state whose geometry needs more than the slice records the
// bytes it needs; the host runs those states again with larger slices and fewer warps, and leaves
// any that do not fit a quarter of the free device memory to its caller.

#include <algorithm>
#include <cmath>
#include <cstring>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

constexpr uint32_t kWarpsPerBlock = 4;

// One state's edges as hgcommon's EdgeList.
struct CsrEdges {
    const uint32_t* edge_verts;
    const uint32_t* verts;
    uint32_t first, n;
    __device__ uint32_t count() const { return n; }
    __device__ uint32_t arity(uint32_t i) const {
        return edge_verts[first + i + 1] - edge_verts[first + i];
    }
    __device__ uint32_t at(uint32_t i, uint32_t k) const { return verts[edge_verts[first + i] + k]; }
};

__global__ void k_vertex_curvature(const uint32_t* state_edges, const uint32_t* edge_verts,
                                   const uint32_t* verts, const uint32_t* todo, uint32_t ntodo,
                                   uint32_t* next, unsigned char* scratch, uint64_t slice,
                                   double* k, const uint64_t* k_off, uint8_t* done,
                                   uint64_t* need) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t warp = static_cast<uint64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    unsigned char* mine = scratch + warp * slice;
    IrTile par;
    for (;;) {
        uint32_t i = 0;
        if (lane == 0) i = atomicAdd(next, 1u);
        i = __shfl_sync(0xffffffffu, i, 0);
        if (i >= ntodo) break;
        const uint32_t s = todo[i];
        const CsrEdges el{edge_verts, verts, state_edges[s], state_edges[s + 1] - state_edges[s]};
        hgcommon::SgGeometry g;
        size_t needed = 0;
        const bool ok = hgcommon::sg_state_geometry(el, mine, slice, g, nullptr, 0, needed,
                                                    hgcommon::SG_OLLIVIER, par, nullptr, 0,
                                                    nullptr, k + k_off[s]);
        if (lane == 0) {
            if (ok) done[s] = 1;
            else need[s] = needed;
        }
        __syncwarp();
    }
}

template <class T>
T* upload(const std::vector<T>& h, std::vector<void*>& owned) {
    T* d = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)), "cudaMalloc");
    owned.push_back(d);
    if (!h.empty())
        HG_CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice),
                      "cudaMemcpy");
    return d;
}

}  // namespace

bool vertex_curvature_device(const std::vector<uint32_t>& state_edges,
                             const std::vector<uint32_t>& edge_verts,
                             const std::vector<uint32_t>& verts,
                             const std::vector<uint32_t>& vertex_count,
                             std::vector<double>& k, std::vector<uint64_t>& k_off,
                             std::vector<uint8_t>& done) {
    const uint32_t states = static_cast<uint32_t>(vertex_count.size());
    k_off.assign(states + 1, 0);
    for (uint32_t s = 0; s < states; ++s) k_off[s + 1] = k_off[s] + vertex_count[s];
    k.assign(k_off[states], std::nan(""));
    done.assign(states, 0);
    if (states == 0) return true;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        cudaGetLastError();
        return false;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaDeviceProp prop{};
    HG_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    std::vector<void*> owned;
    struct Free {
        std::vector<void*>& p;
        ~Free() { for (void* x : p) cudaFree(x); }
    } free_all{owned};
    const uint32_t* d_state_edges = upload(state_edges, owned);
    const uint32_t* d_edge_verts = upload(edge_verts, owned);
    const uint32_t* d_verts = upload(verts, owned);
    const uint64_t* d_k_off = upload(k_off, owned);
    double* d_k = upload(k, owned);
    uint8_t* d_done = upload(done, owned);
    std::vector<uint64_t> need(states, 0);
    uint64_t* d_need = upload(need, owned);
    uint32_t* d_next = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_next, sizeof(uint32_t)), "cudaMalloc");
    owned.push_back(d_next);

    // The first slice holds a state of 64 edges of arity 2 at maximum degree 8; each later pass
    // takes the largest size a state reported.
    uint64_t slice = hgcommon::sg_build_bytes(128, 64) + hgcommon::sg_metric_bytes(128, 8, 64, 32);
    std::vector<uint32_t> todo(states);
    for (uint32_t s = 0; s < states; ++s) todo[s] = s;
    const uint64_t budget = free_bytes / 4;
    for (int pass = 0; pass < 4 && !todo.empty(); ++pass) {
        slice = (slice + 255) & ~uint64_t{255};
        if (slice > budget) break;
        const uint64_t want = static_cast<uint64_t>(prop.multiProcessorCount) * 16;
        const uint64_t fit = budget / slice;
        uint64_t warps = std::min<uint64_t>({fit, want, todo.size()});
        warps = std::max<uint64_t>(1, (warps + kWarpsPerBlock - 1) / kWarpsPerBlock) * kWarpsPerBlock;
        if (warps * slice > budget) warps = budget / slice / kWarpsPerBlock * kWarpsPerBlock;
        if (warps == 0) break;
        unsigned char* d_scratch = nullptr;
        HG_CUDA_CHECK(cudaMalloc(&d_scratch, warps * slice), "cudaMalloc");
        owned.push_back(d_scratch);
        uint32_t* d_todo = upload(todo, owned);
        HG_CUDA_CHECK(cudaMemset(d_next, 0, sizeof(uint32_t)), "cudaMemset");
        k_vertex_curvature<<<static_cast<uint32_t>(warps / kWarpsPerBlock), 32 * kWarpsPerBlock>>>(
            d_state_edges, d_edge_verts, d_verts, d_todo, static_cast<uint32_t>(todo.size()),
            d_next, d_scratch, slice, d_k, d_k_off, d_done, d_need);
        HG_CUDA_CHECK(cudaGetLastError(), "k_vertex_curvature");
        HG_CUDA_CHECK(cudaMemcpy(done.data(), d_done, states, cudaMemcpyDeviceToHost),
                      "cudaMemcpy");
        HG_CUDA_CHECK(cudaMemcpy(need.data(), d_need, states * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy");
        std::vector<uint32_t> rest;
        uint64_t largest = 0;
        for (uint32_t s : todo)
            if (!done[s]) {
                rest.push_back(s);
                largest = std::max(largest, need[s]);
            }
        todo.swap(rest);
        slice = std::max(largest, slice + 1);
        cudaFree(d_scratch);
        owned.erase(std::find(owned.begin(), owned.end(), static_cast<void*>(d_scratch)));
    }
    HG_CUDA_CHECK(cudaMemcpy(k.data(), d_k, k.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    return true;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
