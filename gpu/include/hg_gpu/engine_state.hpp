#pragma once

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/errors.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/hash_table.hpp"
#include "hg_gpu/lock_free_list.hpp"
#include "hg_gpu/signature_index.hpp"
#include "hg_gpu/types.hpp"
#include "hg_gpu/vertex_inverted_index.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hg_gpu {

// EngineConfig is declared in evolve.hpp so host-only translation units
// (the bench harness, the differential test driver) can include it without
// pulling in cuda_runtime.h. The full definition lives there; this header
// transitively re-exports it via #include "hg_gpu/evolve.hpp".

// Device-side POD passed to kernels. All pointers refer to memory owned by
// EngineState (host side); EngineState's lifetime brackets every kernel run
// that uses it.
struct DeviceState {
    // Pools
    typename Pool<VertexId>::DeviceView vertex_pool;
    typename Pool<Edge>::DeviceView     edge_pool;

    // State storage — CSR per-state edge lists.
    // state_edge_slices[sid] = {offset, count} into state_edge_ids.
    // See StateEdgeSlice comment in types.hpp.
    StateEdgeSlice* state_edge_slices;       // [max_states]
    EdgeId*         state_edge_ids;          // [state_edge_ids_capacity]
    uint32_t*       state_edge_ids_counter;  // device atomic; next slot in state_edge_ids
    uint32_t        state_edge_ids_capacity;
    uint32_t        max_states;

    // State allocator (atomic-bumped)
    uint32_t* state_count;            // device atomic; current num_states

    // Vertex allocator (atomic-bumped fresh-vertex counter)
    uint32_t* vertex_high_water;      // monotonic max VertexId issued + 1

    // Indices
    SignatureIndex::DeviceView        signature_index;
    VertexInvertedIndex::DeviceView   vertex_inverted_index;

    // Events and causal/branchial structures
    typename Pool<DeviceEvent>::DeviceView          event_pool;
    typename Pool<DeviceCausalEdge>::DeviceView     causal_edge_pool;
    typename Pool<DeviceBranchialEdge>::DeviceView  branchial_edge_pool;

    // Per-edge producer slot (atomic EventId, INVALID_ID if not yet produced)
    EventId* edge_producer;
    // Per-edge consumer list (LockFreeList keyed by EdgeId)
    typename LockFreeList<EventId>::DeviceView edge_consumers;
    // Per-state event list (LockFreeList keyed by raw StateId) — used by
    // branchial scan to find prior sibling events from the same input state
    // Branchial co-consumer index: bucket = hash(state, edge) & (num_keys - 1),
    // entry packs (event << 32 | edge). Bucket collisions across states are
    // disambiguated by the input-state check in register_branchial.
    typename LockFreeList<uint64_t>::DeviceView branchial_index;

    // Dedup maps. causal_triple_dedup ensures exactly one CausalEdge record
    // per (p, c, shared_edge) triple (preserves multiplicity across distinct
    // shared edges). causal_pair_dedup tracks whether the (p, c) pair has
    // been seen at all — used by online TR so the first-time-reachability
    // check is skipped for subsequent edges between the same pair.
    ConcurrentMap<uint64_t, uint32_t>::DeviceView causal_triple_dedup;
    ConcurrentMap<uint64_t, uint32_t>::DeviceView causal_pair_dedup;
    ConcurrentMap<uint64_t, uint32_t>::DeviceView branchial_pair_dedup;

    // Transitive reduction: Desc[event] (events reachable from event) and
    // Anc[event] (events that reach event). Lists are iterable; sets are
    // O(1) "is x in Desc[e]".
    typename LockFreeList<EventId>::DeviceView    desc_list;
    typename LockFreeList<EventId>::DeviceView    anc_list;
    ConcurrentMap<uint64_t, uint32_t>::DeviceView desc_set;
    ConcurrentMap<uint64_t, uint32_t>::DeviceView anc_set;

    // Flags
    bool tr_enabled;

    // Index maintenance is lazy. Small states are matched by scanning their own
    // CSR slice, so the signature and vertex-inverted indices are read only once
    // some state exceeds slice_scan_max_edges. Until then inserts are skipped;
    // the rewrite kernel raises *needs_indices when it publishes a larger state,
    // and the host rebuilds both indices from the edge pool before the next
    // match launch, then keeps them maintained.
    uint32_t  slice_scan_max_edges;
    uint32_t  maintain_indices;   // 0/1, host-set, read per launch
    uint32_t* needs_indices;      // device flag, raised by the rewrite kernel

    // Error channel: kernels record overflow reasons here instead of silently
    // bailing on partial work. Host inspects after every kernel sync.
    DeviceErrors::DeviceView errors;
};

class EngineState {
public:
    explicit EngineState(EngineConfig cfg)
        : cfg_(cfg)
        , vertex_pool_(cfg.max_vertex_slots)
        , edge_pool_(cfg.max_edges)
        , signature_index_(cfg.sig_index_buckets, cfg.sig_index_pool)
        , vertex_inverted_index_(cfg.max_vertices, cfg.inverted_pool)
        , event_pool_(cfg.max_events)
        , causal_edge_pool_(cfg.max_causal_edges)
        , branchial_edge_pool_(cfg.max_branchial_edges)
        , edge_consumers_(cfg.max_edges, cfg.edge_consumer_nodes)
        , branchial_index_(cfg.branchial_index_buckets, cfg.branchial_index_nodes)
        , causal_triple_dedup_(cfg.causal_triple_slots)
        , causal_pair_dedup_(cfg.causal_pair_slots)
        , branchial_pair_dedup_(cfg.branchial_pair_slots)
        , desc_list_(cfg.max_events, cfg.tr_desc_nodes)
        , anc_list_(cfg.max_events,  cfg.tr_anc_nodes)
        , desc_set_(cfg.tr_desc_slots)
        , anc_set_(cfg.tr_anc_slots)
    {
        slice_scan_max_edges_ = cfg.slice_scan_max_edges;
        check(cudaMalloc(&state_edge_slices_,
              sizeof(StateEdgeSlice) * cfg_.max_states),
              "EngineState state_edge_slices alloc");
        check(cudaMalloc(&state_edge_ids_,
              sizeof(EdgeId) * cfg_.max_state_edge_total),
              "EngineState state_edge_ids alloc");
        check(cudaMalloc(&state_edge_ids_counter_, sizeof(uint32_t)),
              "EngineState state_edge_ids_counter alloc");
        check(cudaMalloc(&state_count_,       sizeof(uint32_t)), "EngineState state_count alloc");
        check(cudaMalloc(&needs_indices_,     sizeof(uint32_t)), "EngineState needs_indices alloc");
        check(cudaMalloc(&vertex_high_water_, sizeof(uint32_t)), "EngineState vertex_high_water alloc");
        check(cudaMalloc(&edge_producer_,     sizeof(EventId) * cfg_.max_edges),
              "EngineState edge_producer alloc");
        clear();
    }

    ~EngineState() {
        if (state_edge_slices_)      cudaFree(state_edge_slices_);
        if (state_edge_ids_)         cudaFree(state_edge_ids_);
        if (state_edge_ids_counter_) cudaFree(state_edge_ids_counter_);
        if (state_count_)            cudaFree(state_count_);
        if (vertex_high_water_)      cudaFree(vertex_high_water_);
        if (edge_producer_)          cudaFree(edge_producer_);
    }

    EngineState(const EngineState&)            = delete;
    EngineState& operator=(const EngineState&) = delete;

    DeviceState device() const {
        DeviceState d;
        d.vertex_pool             = vertex_pool_.view();
        d.edge_pool               = edge_pool_.view();
        d.state_edge_slices       = state_edge_slices_;
        d.state_edge_ids          = state_edge_ids_;
        d.state_edge_ids_counter  = state_edge_ids_counter_;
        d.state_edge_ids_capacity = cfg_.max_state_edge_total;
        d.max_states              = cfg_.max_states;
        d.state_count             = state_count_;
        d.vertex_high_water       = vertex_high_water_;
        d.signature_index         = signature_index_.view();
        d.vertex_inverted_index   = vertex_inverted_index_.view();
        d.event_pool              = event_pool_.view();
        d.causal_edge_pool        = causal_edge_pool_.view();
        d.branchial_edge_pool     = branchial_edge_pool_.view();
        d.edge_producer           = edge_producer_;
        d.edge_consumers          = edge_consumers_.view();
        d.branchial_index         = branchial_index_.view();
        d.causal_triple_dedup     = causal_triple_dedup_.view();
        d.causal_pair_dedup       = causal_pair_dedup_.view();
        d.branchial_pair_dedup    = branchial_pair_dedup_.view();
        d.desc_list               = desc_list_.view();
        d.anc_list                = anc_list_.view();
        d.desc_set                = desc_set_.view();
        d.anc_set                 = anc_set_.view();
        d.tr_enabled              = tr_enabled_;
        d.slice_scan_max_edges    = slice_scan_max_edges_;
        d.maintain_indices        = maintain_indices_ ? 1u : 0u;
        d.needs_indices           = needs_indices_;
        d.errors                  = errors_.view();
        return d;
    }

    // Error channel — sync + drain into the caller's warnings list. Call
    // after every kernel launch that writes to DeviceState. Non-throwing:
    // capacity overflows are warnings, not errors. (Genuine driver
    // failures inside the d2h still throw std::runtime_error.)
    void collect_warnings_into(std::vector<OverflowWarning>& out,
                               const char* context) {
        errors_.collect_warnings_into(out, context);
    }

    // Legacy fail-fast variant for unit tests. Production code should use
    // collect_warnings_into instead.
    void throw_on_errors(const char* context) const {
        errors_.throw_if_any(context);
    }
    void clear_errors() { errors_.clear(); }

    void set_tr_enabled(bool enabled) { tr_enabled_ = enabled; }

    uint32_t config_slice_scan_max_edges() const { return slice_scan_max_edges_; }
    void set_maintain_indices(bool on) { maintain_indices_ = on; }
    bool maintain_indices() const { return maintain_indices_; }
    bool needs_indices_host() const {
        uint32_t v = 0;
        check(cudaMemcpy(&v, needs_indices_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "EngineState needs_indices read");
        return v != 0;
    }

    void clear() {
        check(cudaMemset(state_edge_slices_, 0,
              sizeof(StateEdgeSlice) * cfg_.max_states),
              "EngineState clear state_edge_slices");
        check(cudaMemset(state_edge_ids_counter_, 0, sizeof(uint32_t)),
              "EngineState clear state_edge_ids_counter");
        check(cudaMemset(state_count_,       0, sizeof(uint32_t)), "EngineState clear state_count");
        check(cudaMemset(needs_indices_,     0, sizeof(uint32_t)), "EngineState clear needs_indices");
        check(cudaMemset(vertex_high_water_, 0, sizeof(uint32_t)), "EngineState clear vertex_high_water");
        // edge_producer init to INVALID_ID (0xFF bytes).
        check(cudaMemset(edge_producer_, 0xFF, sizeof(EventId) * cfg_.max_edges),
              "EngineState clear edge_producer");
        vertex_pool_.reset();
        edge_pool_.reset();
        signature_index_.clear();
        vertex_inverted_index_.clear();
        event_pool_.reset();
        causal_edge_pool_.reset();
        branchial_edge_pool_.reset();
        edge_consumers_.clear();
        branchial_index_.clear();
        causal_triple_dedup_.clear();
        causal_pair_dedup_.clear();
        branchial_pair_dedup_.clear();
        desc_list_.clear();
        anc_list_.clear();
        desc_set_.clear();
        anc_set_.clear();
        errors_.clear();
    }

    const EngineConfig& config() const { return cfg_; }

    // ------------------------------------------------------------------
    // Host-side inspection helpers (slow; for tests / final readout only)
    // ------------------------------------------------------------------

    uint32_t num_edges_host() const  { return edge_pool_.size_host(); }
    uint32_t num_states_host() const {
        uint32_t v = 0;
        cudaMemcpy(&v, state_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return v;
    }
    uint32_t vertex_high_water_host() const {
        uint32_t v = 0;
        cudaMemcpy(&v, vertex_high_water_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return v;
    }

    Edge edge_at_host(EdgeId eid) const {
        Edge e{};
        cudaMemcpy(&e, edge_pool_view_data() + eid, sizeof(Edge), cudaMemcpyDeviceToHost);
        return e;
    }

    std::vector<VertexId> edge_vertices_host(EdgeId eid) const {
        Edge e = edge_at_host(eid);
        std::vector<VertexId> out(e.arity);
        cudaMemcpy(out.data(), vertex_pool_view_data() + e.vertex_offset,
                   sizeof(VertexId) * e.arity, cudaMemcpyDeviceToHost);
        return out;
    }

    // Read back every state's edge-vertex-tuple list from the device via
    // four bulk cudaMemcpy calls (slices, ids, edges, vertices) then
    // reconstructs on host. O(total state-edge slots) on the wire rather
    // than the O(max_states × max_edges/32) bitset readback.
    std::vector<std::vector<std::vector<VertexId>>> all_state_edges_host() const {
        uint32_t n_states = num_states_host();
        std::vector<std::vector<std::vector<VertexId>>> out(n_states);
        if (n_states == 0) return out;

        uint32_t n_edges      = edge_pool_.size_host();
        uint32_t n_vert_slots = vertex_pool_.size_host();
        uint32_t n_id_slots   = 0;
        cudaMemcpy(&n_id_slots, state_edge_ids_counter_, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        std::vector<Edge>           edges(n_edges);
        std::vector<VertexId>       verts(n_vert_slots);
        std::vector<StateEdgeSlice> slices(n_states);
        std::vector<EdgeId>         ids(n_id_slots);

        if (n_edges > 0) {
            cudaMemcpy(edges.data(), edge_pool_.view().data,
                       sizeof(Edge) * n_edges, cudaMemcpyDeviceToHost);
        }
        if (n_vert_slots > 0) {
            cudaMemcpy(verts.data(), vertex_pool_.view().data,
                       sizeof(VertexId) * n_vert_slots, cudaMemcpyDeviceToHost);
        }
        cudaMemcpy(slices.data(), state_edge_slices_,
                   sizeof(StateEdgeSlice) * n_states, cudaMemcpyDeviceToHost);
        if (n_id_slots > 0) {
            cudaMemcpy(ids.data(), state_edge_ids_,
                       sizeof(EdgeId) * n_id_slots, cudaMemcpyDeviceToHost);
        }

        for (uint32_t s = 0; s < n_states; ++s) {
            const StateEdgeSlice& sl = slices[s];
            if (static_cast<size_t>(sl.offset) + sl.count > ids.size()) continue;
            for (uint32_t k = 0; k < sl.count; ++k) {
                EdgeId eid = ids[sl.offset + k];
                if (eid >= n_edges) continue;
                const Edge& e = edges[eid];
                std::vector<VertexId> vs(e.arity);
                for (uint8_t i = 0; i < e.arity; ++i) {
                    vs[i] = verts[e.vertex_offset + i];
                }
                out[s].push_back(std::move(vs));
            }
        }
        return out;
    }

    // Read back one state's EdgeId list.
    std::vector<EdgeId> state_edges_host(StateId sid) const {
        StateEdgeSlice sl{0, 0};
        cudaMemcpy(&sl, state_edge_slices_ + sid, sizeof(StateEdgeSlice),
                   cudaMemcpyDeviceToHost);
        std::vector<EdgeId> out(sl.count);
        if (sl.count > 0) {
            cudaMemcpy(out.data(), state_edge_ids_ + sl.offset,
                       sizeof(EdgeId) * sl.count, cudaMemcpyDeviceToHost);
        }
        return out;
    }

    // Friend access for kernels that need raw pointers (rare; prefer DeviceView).
    Edge*     edge_pool_view_data()    const { return edge_pool_.view().data; }
    VertexId* vertex_pool_view_data()  const { return vertex_pool_.view().data; }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::EngineState ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    EngineConfig                       cfg_;
    Pool<VertexId>                     vertex_pool_;
    Pool<Edge>                         edge_pool_;
    StateEdgeSlice*                    state_edge_slices_      = nullptr;
    EdgeId*                            state_edge_ids_         = nullptr;
    uint32_t*                          state_edge_ids_counter_ = nullptr;
    uint32_t*                          state_count_            = nullptr;
    uint32_t*                          vertex_high_water_      = nullptr;
    SignatureIndex                     signature_index_;
    VertexInvertedIndex                vertex_inverted_index_;
    Pool<DeviceEvent>                  event_pool_;
    Pool<DeviceCausalEdge>             causal_edge_pool_;
    Pool<DeviceBranchialEdge>          branchial_edge_pool_;
    EventId*                           edge_producer_ = nullptr;
    LockFreeList<EventId>              edge_consumers_;
    LockFreeList<uint64_t>             branchial_index_;
    ConcurrentMap<uint64_t, uint32_t>  causal_triple_dedup_;
    ConcurrentMap<uint64_t, uint32_t>  causal_pair_dedup_;
    ConcurrentMap<uint64_t, uint32_t>  branchial_pair_dedup_;
    LockFreeList<EventId>              desc_list_;
    LockFreeList<EventId>              anc_list_;
    ConcurrentMap<uint64_t, uint32_t>  desc_set_;
    ConcurrentMap<uint64_t, uint32_t>  anc_set_;
    DeviceErrors                       errors_;
    bool                               tr_enabled_ = false;
    uint32_t slice_scan_max_edges_ = 256;
    bool maintain_indices_ = true;
    uint32_t* needs_indices_ = nullptr;

public:
    // Host readers for tests / EvolveResult population.
    uint32_t num_events_host()          const { return event_pool_.size_host(); }
    uint32_t num_causal_edges_host()    const { return causal_edge_pool_.size_host(); }
    uint32_t num_branchial_edges_host() const { return branchial_edge_pool_.size_host(); }

    std::vector<DeviceEvent> events_host() const {
        uint32_t n = num_events_host();
        std::vector<DeviceEvent> out(n);
        if (n > 0) cudaMemcpy(out.data(), event_pool_.view().data,
                              sizeof(DeviceEvent) * n, cudaMemcpyDeviceToHost);
        return out;
    }
    std::vector<DeviceCausalEdge> causal_edges_host() const {
        uint32_t n = num_causal_edges_host();
        std::vector<DeviceCausalEdge> out(n);
        if (n > 0) cudaMemcpy(out.data(), causal_edge_pool_.view().data,
                              sizeof(DeviceCausalEdge) * n, cudaMemcpyDeviceToHost);
        return out;
    }
    std::vector<DeviceBranchialEdge> branchial_edges_host() const {
        uint32_t n = num_branchial_edges_host();
        std::vector<DeviceBranchialEdge> out(n);
        if (n > 0) cudaMemcpy(out.data(), branchial_edge_pool_.view().data,
                              sizeof(DeviceBranchialEdge) * n, cudaMemcpyDeviceToHost);
        return out;
    }
};

}  // namespace hg_gpu
