#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/edge_signature.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace hg_gpu {

// Kernel that, for every edge in [0, num_edges), pushes (signature_hash →
// edge_id) into the signature index and (each vertex → edge_id) into the
// vertex inverted index.
__global__ void k_init_indices(DeviceState ds, uint32_t num_edges) {
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;

    Edge& e = ds.edge_pool.at(eid);
    ds.signature_index.insert(eid, e.signature);
    for (uint8_t i = 0; i < e.arity; ++i) {
        VertexId v = ds.vertex_pool.at(e.vertex_offset + i);
        ds.vertex_inverted_index.insert(v, eid);
    }
}

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::upload_initial_state ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

}  // namespace

StateId upload_initial_state(EngineState& engine,
                             const std::vector<std::vector<VertexId>>& initial_edges) {
    const uint32_t n = static_cast<uint32_t>(initial_edges.size());
    if (n == 0) {
        // Empty initial state — still create state 0 (an empty state) and
        // bump state_count to 1.
        uint32_t sc = 1;
        check(cudaMemcpy(engine.device().state_count, &sc, sizeof(uint32_t),
                         cudaMemcpyHostToDevice),
              "set state_count for empty initial");
        return 0u;
    }

    DeviceState ds = engine.device();
    const EngineConfig& cfg = engine.config();

    // 1. Flatten host-side vertex tuples.
    std::vector<VertexId> flat_vertices;
    std::vector<Edge>     edges;
    edges.reserve(n);
    flat_vertices.reserve(n * 4);
    VertexId max_vertex = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const auto& tuple = initial_edges[i];
        if (tuple.size() > kMaxArity) {
            throw std::runtime_error("upload_initial_state: edge arity exceeds kMaxArity");
        }
        Edge e{};
        e.arity         = static_cast<uint8_t>(tuple.size());
        e.vertex_offset = static_cast<uint32_t>(flat_vertices.size());
        e.signature     = signature_hash_from_vertices(tuple.data(), e.arity);
        e.creator_event = INVALID_ID;
        e.step          = 0;
        edges.push_back(e);
        for (VertexId v : tuple) {
            if (v > max_vertex) max_vertex = v;
            flat_vertices.push_back(v);
        }
    }

    if (n > cfg.max_edges) throw std::runtime_error("upload_initial_state: max_edges exceeded");
    if (flat_vertices.size() > cfg.max_vertex_slots)
        throw std::runtime_error("upload_initial_state: max_vertex_slots exceeded");
    if (max_vertex >= cfg.max_vertices)
        throw std::runtime_error("upload_initial_state: vertex id exceeds max_vertices");

    // 2. Bulk upload vertex tuples and edges to their pools, then advance
    //    the atomic counters by direct host-side writes.
    check(cudaMemcpy(ds.vertex_pool.data, flat_vertices.data(),
                     sizeof(VertexId) * flat_vertices.size(), cudaMemcpyHostToDevice),
          "upload vertex_pool");
    check(cudaMemcpy(ds.edge_pool.data, edges.data(),
                     sizeof(Edge) * edges.size(), cudaMemcpyHostToDevice),
          "upload edge_pool");

    uint32_t vp_count = static_cast<uint32_t>(flat_vertices.size());
    uint32_t ep_count = n;
    check(cudaMemcpy(ds.vertex_pool.counter, &vp_count, sizeof(uint32_t), cudaMemcpyHostToDevice),
          "set vertex_pool counter");
    check(cudaMemcpy(ds.edge_pool.counter,   &ep_count, sizeof(uint32_t), cudaMemcpyHostToDevice),
          "set edge_pool counter");

    uint32_t hi = max_vertex + 1;
    check(cudaMemcpy(ds.vertex_high_water, &hi, sizeof(uint32_t), cudaMemcpyHostToDevice),
          "set vertex_high_water");

    // 3. Populate state 0's CSR slice: offset=0, count=n, ids=[0,1,...,n-1].
    if (n > cfg.max_state_edge_total) {
        throw std::runtime_error("upload_initial_state: initial edge count exceeds max_state_edge_total");
    }
    std::vector<EdgeId> initial_ids(n);
    for (uint32_t i = 0; i < n; ++i) initial_ids[i] = i;
    check(cudaMemcpy(ds.state_edge_ids, initial_ids.data(),
                     sizeof(EdgeId) * n, cudaMemcpyHostToDevice),
          "upload state 0 edge ids");
    StateEdgeSlice slice0{0u, n};
    check(cudaMemcpy(ds.state_edge_slices, &slice0, sizeof(StateEdgeSlice),
                     cudaMemcpyHostToDevice),
          "upload state 0 slice");
    uint32_t ids_cnt = n;
    check(cudaMemcpy(ds.state_edge_ids_counter, &ids_cnt, sizeof(uint32_t),
                     cudaMemcpyHostToDevice),
          "set state_edge_ids_counter");

    uint32_t sc = 1;
    check(cudaMemcpy(ds.state_count, &sc, sizeof(uint32_t), cudaMemcpyHostToDevice),
          "set state_count");

    // 4. Populate indices via kernel.
    int block = 128;
    int grid  = (int)((n + block - 1) / block);
    k_init_indices<<<grid, block>>>(ds, n);
    check(cudaDeviceSynchronize(), "k_init_indices sync");

    return 0u;
}

}  // namespace hg_gpu
