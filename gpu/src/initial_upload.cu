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

    if (n_edges > 0) {
        check(cudaMemcpy(ds.vertex_pool.data, flat_vertices.data(),
                         sizeof(VertexId) * flat_vertices.size(), cudaMemcpyHostToDevice),
              "upload vertex_pool");
        check(cudaMemcpy(ds.edge_pool.data, edges.data(),
                         sizeof(Edge) * edges.size(), cudaMemcpyHostToDevice),
              "upload edge_pool");
        uint32_t vp_count = static_cast<uint32_t>(flat_vertices.size());
        check(cudaMemcpy(ds.vertex_pool.counter, &vp_count, sizeof(uint32_t), cudaMemcpyHostToDevice),
              "set vertex_pool counter");
        check(cudaMemcpy(ds.edge_pool.counter, &n_edges, sizeof(uint32_t), cudaMemcpyHostToDevice),
              "set edge_pool counter");
        uint32_t hi = max_vertex + 1;
        check(cudaMemcpy(ds.vertex_high_water, &hi, sizeof(uint32_t), cudaMemcpyHostToDevice),
              "set vertex_high_water");
        check(cudaMemcpy(ds.state_edge_ids, all_ids.data(),
                         sizeof(EdgeId) * all_ids.size(), cudaMemcpyHostToDevice),
              "upload state_edge_ids");
        uint32_t ids_cnt = static_cast<uint32_t>(all_ids.size());
        check(cudaMemcpy(ds.state_edge_ids_counter, &ids_cnt, sizeof(uint32_t), cudaMemcpyHostToDevice),
              "set state_edge_ids_counter");
    }
    check(cudaMemcpy(ds.state_edge_slices, slices.data(),
                     sizeof(StateEdgeSlice) * slices.size(), cudaMemcpyHostToDevice),
          "upload state slices");
    check(cudaMemcpy(ds.state_count, &M, sizeof(uint32_t), cudaMemcpyHostToDevice),
          "set state_count");

    if (n_edges > 0 && engine.maintain_indices()) {
        int block = 128;
        int grid  = (int)((n_edges + block - 1) / block);
        k_init_indices<<<grid, block>>>(ds, n_edges);
        check(cudaDeviceSynchronize(), "k_init_indices sync");
    }
    return M;
}

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
    if (engine.maintain_indices()) {
        k_init_indices<<<grid, block>>>(ds, n);
    }
    check(cudaDeviceSynchronize(), "k_init_indices sync");

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
    check(cudaDeviceSynchronize(), "rebuild_indices sync");
}

}  // namespace hg_gpu
