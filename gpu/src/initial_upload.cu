#include "hgcommon/namespace.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/edge_signature.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// Kernel that, for every edge in [0, num_edges), pushes (signature_hash →
// edge_id) into the signature index and (each vertex → edge_id) into the
// vertex inverted index.
__global__ void k_init_indices(const __grid_constant__ DeviceState ds, uint32_t num_edges) {
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;

    Edge& e = ds.edge_pool.at(eid);
    ds.signature_index.insert(eid, e.signature);
    auto vertex_at = [&](uint8_t k) { return ds.vertex_pool.at(e.vertex_offset + k); };
    for (uint8_t i = 0; i < e.arity; ++i) {
        if (!first_occurrence(vertex_at, i)) continue;
        ds.vertex_inverted_index.insert(vertex_at(i), eid);
    }
}

uint32_t upload_initial_states(EngineState& engine,
                               const std::vector<std::vector<std::vector<VertexId>>>& initial_states) {
    const uint32_t M = static_cast<uint32_t>(initial_states.size());
    if (M == 0) return 0u;

    DeviceState ds = engine.device();
    const EngineConfig& cfg = engine.config();

    std::vector<VertexId>      flat_vertices;
    std::vector<Edge>          edges;
    std::vector<EdgeId>        all_ids;      // state_edge_ids: each state's contiguous run
    std::vector<StateEdgeSlice> slices;
    slices.reserve(M);
    VertexId max_vertex = 0;

    for (uint32_t s = 0; s < M; ++s) {
        const auto& state_edges = initial_states[s];
        const uint32_t slice_off = static_cast<uint32_t>(all_ids.size());
        for (const auto& tuple : state_edges) {
            if (tuple.size() > kMaxArity)
                throw std::runtime_error("upload_initial_states: edge arity exceeds kMaxArity");
            Edge e{};
            e.arity         = static_cast<uint8_t>(tuple.size());
            e.vertex_offset = static_cast<uint32_t>(flat_vertices.size());
            e.signature     = signature_hash_from_vertices(tuple.data(), e.arity);
            e.creator_event = INVALID_ID;
            e.step          = 0;
            const EdgeId eid = static_cast<EdgeId>(edges.size());
            edges.push_back(e);
            all_ids.push_back(eid);            // ascending: state's edges are a contiguous run
            for (VertexId v : tuple) {
                if (v > max_vertex) max_vertex = v;
                flat_vertices.push_back(v);
            }
        }
        slices.push_back(StateEdgeSlice{slice_off,
                          static_cast<uint32_t>(all_ids.size()) - slice_off});
    }

    const uint32_t n_edges = static_cast<uint32_t>(edges.size());
    if (n_edges > cfg.max_edges) throw std::runtime_error("upload_initial_states: max_edges exceeded");
    if (flat_vertices.size() > cfg.max_vertex_slots)
        throw std::runtime_error("upload_initial_states: max_vertex_slots exceeded");
    if (max_vertex >= cfg.max_vertices && n_edges > 0)
        throw std::runtime_error("upload_initial_states: vertex id exceeds max_vertices");
    if (all_ids.size() > cfg.max_state_edge_total)
        throw std::runtime_error("upload_initial_states: initial edge count exceeds max_state_edge_total");
    if (M > cfg.max_states) throw std::runtime_error("upload_initial_states: max_states exceeded");

    // ASYNC FROM PAGEABLE MEMORY: each call returns once the runtime has staged the source, so
    // the host locals below may go out of scope, and stream order puts every copy before the
    // run's kernels. A synchronous cudaMemcpy costs about 25 us per call here, the async one a
    // few; nine of them ran per evolve.
    if (n_edges > 0) {
        HG_CUDA_CHECK(cudaMemcpyAsync(ds.vertex_pool.data, flat_vertices.data(),
                         sizeof(VertexId) * flat_vertices.size(), cudaMemcpyHostToDevice, 0),
              "upload vertex_pool");
        HG_CUDA_CHECK(cudaMemcpyAsync(ds.edge_pool.data, edges.data(),
                         sizeof(Edge) * edges.size(), cudaMemcpyHostToDevice, 0),
              "upload edge_pool");
        uint32_t vp_count = static_cast<uint32_t>(flat_vertices.size());
        HG_CUDA_CHECK(cudaMemcpyAsync(ds.vertex_pool.counter, &vp_count, sizeof(uint32_t), cudaMemcpyHostToDevice, 0),
              "set vertex_pool counter");
        HG_CUDA_CHECK(cudaMemcpyAsync(ds.edge_pool.counter, &n_edges, sizeof(uint32_t), cudaMemcpyHostToDevice, 0),
              "set edge_pool counter");
        uint32_t hi = max_vertex + 1;
        HG_CUDA_CHECK(cudaMemcpyAsync(ds.vertex_high_water, &hi, sizeof(uint32_t), cudaMemcpyHostToDevice, 0),
              "set vertex_high_water");
        HG_CUDA_CHECK(cudaMemcpyAsync(ds.state_edge_ids, all_ids.data(),
                         sizeof(EdgeId) * all_ids.size(), cudaMemcpyHostToDevice, 0),
              "upload state_edge_ids");
        uint32_t ids_cnt = static_cast<uint32_t>(all_ids.size());
        HG_CUDA_CHECK(cudaMemcpyAsync(ds.state_edge_ids_counter, &ids_cnt, sizeof(uint32_t), cudaMemcpyHostToDevice, 0),
              "set state_edge_ids_counter");
    }
    HG_CUDA_CHECK(cudaMemcpyAsync(ds.state_edge_slices, slices.data(),
                     sizeof(StateEdgeSlice) * slices.size(), cudaMemcpyHostToDevice, 0),
          "upload state slices");
    HG_CUDA_CHECK(cudaMemcpyAsync(ds.state_count, &M, sizeof(uint32_t), cudaMemcpyHostToDevice, 0),
          "set state_count");

    if (n_edges > 0 && engine.maintain_indices()) {
        int block = 128;
        int grid  = (int)((n_edges + block - 1) / block);
        k_init_indices<<<grid, block>>>(ds, n_edges);
        HG_CUDA_CHECK(cudaDeviceSynchronize(), "k_init_indices sync");
    }
    return M;
}

StateId upload_initial_state(EngineState& engine,
                             const std::vector<std::vector<VertexId>>& initial_edges) {
    upload_initial_states(engine, {initial_edges});
    return 0u;
}

// Bulk (re)build of the signature and vertex-inverted indices from the edge
// pool. Runs once when lazy index maintenance flips on: edges created while
// maintenance was off are absent from the indices, and incremental inserts
// resume after this call, so every edge appears in its buckets exactly once.
void rebuild_indices(EngineState& engine, uint32_t num_edges) {
    if (num_edges == 0) return;
    int block = 128;
    int grid  = (int)((num_edges + block - 1) / block);
    k_init_indices<<<grid, block>>>(engine.device(), num_edges);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "rebuild_indices sync");
}

}  // namespace gpu
}  // namespace HG_NAMESPACE