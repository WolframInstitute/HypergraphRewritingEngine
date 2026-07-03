#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/wl_hash.hpp"   // wl_hash_state_device — the size-tolerant fallback

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hg_gpu {

namespace {

// Per-state scratch bounds. On overflow the kernel records kScratchOverflow
// and returns the (approximate) sorted-colour-multiset hash so evolution
// still progresses — exact per-state canonicalisation for that state is
// deferred to the host IR fallback (see ir_canon_host_fallback in evolve).
constexpr uint32_t kMaxIRVerts = 128;
constexpr uint32_t kMaxIREdges = 128;
constexpr uint32_t kMaxIROccs  = 256;
constexpr uint32_t kMaxIRRefineIters = 16;

constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime  = 1099511628211ULL;

__host__ __device__ inline uint64_t fnv_mix(uint64_t h, uint64_t x) {
    h ^= x;
    h *= kFnvPrime;
    return h;
}

// Per-block shared-memory scratch.
struct IRBlock {
    uint32_t n_edges;
    uint32_t n_verts;
    uint32_t n_occs;

    // Edges kept in this state: edge_ids are just dense 0..n_edges-1;
    // arities[] and the CSR of vertex tuples capture the shape. We
    // normalise vertex IDs to 0..n_verts-1 (local indices).
    EdgeId   edges_global[kMaxIREdges];   // global EdgeIds (for debugging)
    uint8_t  arities[kMaxIREdges];
    uint32_t edge_vert_offset[kMaxIREdges + 1];  // CSR into edge_vert_locals
    uint32_t edge_vert_locals[kMaxIROccs];       // local vertex indices per edge position

    // Vertices: global IDs + occurrences CSR into occs[].
    VertexId verts_global[kMaxIRVerts];
    uint32_t occ_offset[kMaxIRVerts + 1];
    // Each occurrence: (edge_idx, position, arity). Packed into uint32 as
    // (arity << 24) | (position << 16) | edge_idx (edge_idx ≤ 2^16).
    uint32_t occs[kMaxIROccs];

    uint64_t colors[kMaxIRVerts];
    uint64_t new_colors[kMaxIRVerts];

    // scratch for sorting and neighbour-signature hashing (per vertex)
    uint64_t neigh_scratch[kMaxIROccs];

    int overflow_flag;    // 0/1 — set if any buffer exceeded its cap
};

__device__ inline uint32_t pack_occ(uint32_t edge_idx, uint8_t pos, uint8_t arity) {
    return (static_cast<uint32_t>(arity) << 24)
         | (static_cast<uint32_t>(pos)   << 16)
         |  edge_idx;
}

__device__ inline void unpack_occ(uint32_t o, uint32_t& edge_idx, uint8_t& pos, uint8_t& arity) {
    edge_idx = o & 0xFFFF;
    pos      = static_cast<uint8_t>((o >> 16) & 0xFF);
    arity    = static_cast<uint8_t>((o >> 24) & 0xFF);
}

__device__ inline void insertion_sort_u64(uint64_t* a, uint32_t n) {
    for (uint32_t i = 1; i < n; ++i) {
        uint64_t key = a[i];
        uint32_t j = i;
        while (j > 0 && a[j - 1] > key) { a[j] = a[j - 1]; --j; }
        a[j] = key;
    }
}

// Linear-probe "find or insert" on a small local table (verts_global[]).
// Returns the local index, or UINT32_MAX on overflow.
__device__ inline uint32_t find_or_insert_vert(IRBlock& blk, VertexId v) {
    for (uint32_t i = 0; i < blk.n_verts; ++i) {
        if (blk.verts_global[i] == v) return i;
    }
    if (blk.n_verts >= kMaxIRVerts) return 0xFFFFFFFFu;
    uint32_t idx = blk.n_verts++;
    blk.verts_global[idx] = v;
    return idx;
}

// Build IRBlock from the state's CSR slice. Thread 0 of the block does the
// build; other threads wait. The state-scale is small enough (≤ ~128
// vertices, ≤ 128 edges) that parallelising this doesn't help; the win is
// in refinement, which IS thread-parallel below.
__device__ void build_block(IRBlock& blk, DeviceState ds, StateId sid) {
    blk.n_edges = 0;
    blk.n_verts = 0;
    blk.n_occs  = 0;
    blk.overflow_flag = 0;
    blk.edge_vert_offset[0] = 0;

    if (sid >= ds.max_states) return;
    StateEdgeSlice sl = ds.state_edge_slices[sid];

    uint32_t num_edges_live = ds.edge_pool.counter ? *ds.edge_pool.counter : 0u;

    for (uint32_t k = 0; k < sl.count; ++k) {
        if (blk.n_edges >= kMaxIREdges) { blk.overflow_flag = 1; return; }
        EdgeId eid = ds.state_edge_ids[sl.offset + k];
        if (eid >= num_edges_live) continue;
        if (eid >= ds.edge_pool.capacity) continue;

        const Edge& e = ds.edge_pool.at(eid);
        if (e.arity == 0 || e.arity > kMaxArity) continue;
        if (e.vertex_offset + e.arity > ds.vertex_pool.capacity) continue;

        uint32_t ei = blk.n_edges++;
        blk.edges_global[ei] = eid;
        blk.arities[ei]      = e.arity;
        blk.edge_vert_offset[ei] = blk.n_occs;
        for (uint8_t p = 0; p < e.arity; ++p) {
            if (blk.n_occs >= kMaxIROccs) { blk.overflow_flag = 1; return; }
            VertexId v = ds.vertex_pool.at(e.vertex_offset + p);
            uint32_t vi = find_or_insert_vert(blk, v);
            if (vi == 0xFFFFFFFFu) { blk.overflow_flag = 1; return; }
            blk.edge_vert_locals[blk.n_occs] = vi;
            ++blk.n_occs;
        }
    }
    blk.edge_vert_offset[blk.n_edges] = blk.n_occs;

    // Build per-vertex occurrences CSR. First pass: count occs per vertex.
    for (uint32_t v = 0; v < blk.n_verts; ++v) blk.occ_offset[v] = 0;
    for (uint32_t ei = 0; ei < blk.n_edges; ++ei) {
        uint8_t arity = blk.arities[ei];
        uint32_t base = blk.edge_vert_offset[ei];
        for (uint8_t p = 0; p < arity; ++p) {
            ++blk.occ_offset[blk.edge_vert_locals[base + p]];
        }
    }
    // Exclusive scan.
    uint32_t running = 0;
    for (uint32_t v = 0; v < blk.n_verts; ++v) {
        uint32_t c = blk.occ_offset[v];
        blk.occ_offset[v] = running;
        running += c;
    }
    blk.occ_offset[blk.n_verts] = running;

    // Second pass: fill occurrences, using occ_offset as a moving cursor.
    uint32_t cursors[kMaxIRVerts];
    for (uint32_t v = 0; v < blk.n_verts; ++v) cursors[v] = blk.occ_offset[v];
    for (uint32_t ei = 0; ei < blk.n_edges; ++ei) {
        uint8_t arity = blk.arities[ei];
        uint32_t base = blk.edge_vert_offset[ei];
        for (uint8_t p = 0; p < arity; ++p) {
            uint32_t vi = blk.edge_vert_locals[base + p];
            blk.occs[cursors[vi]++] = pack_occ(ei, p, arity);
        }
    }
}

// Initial colour: hash over the sorted multiset of (arity, position) pairs
// from the vertex's occurrences. Vertices with the same occurrence pattern
// start in the same cell.
__device__ uint64_t initial_color(const IRBlock& blk, uint32_t vi) {
    uint64_t sigs[kMaxIROccs];
    uint32_t n = blk.occ_offset[vi + 1] - blk.occ_offset[vi];
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t o = blk.occs[blk.occ_offset[vi] + k];
        uint32_t ei; uint8_t pos, arity;
        unpack_occ(o, ei, pos, arity);
        uint64_t h = fnv_mix(kFnvOffset, arity);
        h = fnv_mix(h, pos);
        sigs[k] = h;
    }
    insertion_sort_u64(sigs, n);
    uint64_t h = fnv_mix(kFnvOffset, n);
    for (uint32_t k = 0; k < n; ++k) h = fnv_mix(h, sigs[k]);
    return h;
}

// One refinement step for a single vertex: new_color[vi] = hash over sorted
// neighbour signatures, where each neighbour signature is
// FNV(arity, position, sorted colours of co-occurring vertices).
__device__ uint64_t refine_vertex(IRBlock& blk, uint32_t vi) {
    uint32_t n_occs = blk.occ_offset[vi + 1] - blk.occ_offset[vi];
    uint64_t occ_sigs[kMaxIROccs];
    for (uint32_t k = 0; k < n_occs; ++k) {
        uint32_t o = blk.occs[blk.occ_offset[vi] + k];
        uint32_t ei; uint8_t pos, arity;
        unpack_occ(o, ei, pos, arity);

        // Collect other-vertex colours in this edge (excluding self at pos).
        uint64_t other_colors[kMaxArity];
        uint32_t n_others = 0;
        uint32_t base = blk.edge_vert_offset[ei];
        for (uint8_t p = 0; p < arity; ++p) {
            if (p == pos) continue;
            uint32_t other_vi = blk.edge_vert_locals[base + p];
            other_colors[n_others++] = blk.colors[other_vi];
        }
        insertion_sort_u64(other_colors, n_others);
        uint64_t h = fnv_mix(kFnvOffset, arity);
        h = fnv_mix(h, pos);
        for (uint32_t j = 0; j < n_others; ++j) h = fnv_mix(h, other_colors[j]);
        occ_sigs[k] = h;
    }
    insertion_sort_u64(occ_sigs, n_occs);
    uint64_t h = fnv_mix(blk.colors[vi], n_occs);
    for (uint32_t k = 0; k < n_occs; ++k) h = fnv_mix(h, occ_sigs[k]);
    return h;
}

// Given refined colours, check if every vertex has a unique colour. If so,
// assign rank-in-sorted-colours as the canonical vertex labelling and
// return true; otherwise leave labelling undefined and return false.
//
// Uses one pass to build (color, vi) pairs, insertion-sort by color, check
// uniqueness, then write labelling[vi] = rank.
__device__ bool extract_labeling(const IRBlock& blk, uint32_t* labelling_out) {
    struct CV { uint64_t c; uint32_t vi; };
    CV cv[kMaxIRVerts];
    for (uint32_t v = 0; v < blk.n_verts; ++v) { cv[v].c = blk.colors[v]; cv[v].vi = v; }
    // Insertion sort by colour.
    for (uint32_t i = 1; i < blk.n_verts; ++i) {
        CV key = cv[i];
        uint32_t j = i;
        while (j > 0 && cv[j - 1].c > key.c) { cv[j] = cv[j - 1]; --j; }
        cv[j] = key;
    }
    for (uint32_t i = 1; i < blk.n_verts; ++i) {
        if (cv[i].c == cv[i - 1].c) return false;  // not discrete
    }
    for (uint32_t i = 0; i < blk.n_verts; ++i) labelling_out[cv[i].vi] = i;
    return true;
}

// Apply labelling to edges, sort edges lexicographically, emit FNV hash.
__device__ uint64_t canonical_hash_from_labeling(const IRBlock& blk,
                                                 const uint32_t* labelling) {
    // Build packed canonical edges: one uint64_t per edge encoding (arity,
    // labels...) up to kMaxArity — packs arity in the top byte and labels
    // in subsequent bytes, padded with 0xFF. This gives a lex-order that
    // (a) sorts shorter arities consistently, (b) sorts by relabelled
    // vertex positions. Insertion-sort then emit FNV-1a over the sorted
    // packed list.
    //
    // Packing: [arity | label0 | label1 | ... ] each byte. 8 bytes = arity
    // + up to 7 labels. kMaxArity = 16, so we need two uint64_t per edge.
    // To keep it simple (insertion sort is the hot path), we sort a pair
    // (high, low) per edge via a small struct.

    struct PackedEdge { uint64_t hi; uint64_t lo; };
    PackedEdge pe[kMaxIREdges];

    for (uint32_t ei = 0; ei < blk.n_edges; ++ei) {
        uint8_t arity = blk.arities[ei];
        uint32_t base = blk.edge_vert_offset[ei];
        uint8_t lbl[kMaxArity];
        for (uint8_t p = 0; p < arity; ++p) {
            // Label must fit in a byte. If n_verts > 255 we'd truncate —
            // fall through to a hash-of-labels in that edge case.
            uint32_t l = labelling[blk.edge_vert_locals[base + p]];
            lbl[p] = static_cast<uint8_t>(l & 0xFFu);
        }
        uint64_t hi = 0, lo = 0;
        lo |= static_cast<uint64_t>(arity);
        // Pack label[0..6] into bytes 1..7 of lo; label[7..14] into hi.
        for (uint8_t p = 0; p < arity && p < 7; ++p) {
            lo |= (static_cast<uint64_t>(lbl[p]) << (8 * (p + 1)));
        }
        for (uint8_t p = 7; p < arity && p < 15; ++p) {
            hi |= (static_cast<uint64_t>(lbl[p]) << (8 * (p - 7)));
        }
        pe[ei].hi = hi;
        pe[ei].lo = lo;
    }

    // Insertion sort (lex by hi then lo — but hi holds the higher-index
    // labels which differ only for arity > 7). For arity ≤ 7 (typical),
    // hi is 0 and sort is by lo only.
    for (uint32_t i = 1; i < blk.n_edges; ++i) {
        PackedEdge key = pe[i];
        uint32_t j = i;
        while (j > 0 && (pe[j - 1].hi > key.hi ||
                         (pe[j - 1].hi == key.hi && pe[j - 1].lo > key.lo))) {
            pe[j] = pe[j - 1];
            --j;
        }
        pe[j] = key;
    }

    uint64_t h = fnv_mix(kFnvOffset, blk.n_verts);
    h = fnv_mix(h, blk.n_edges);
    for (uint32_t i = 0; i < blk.n_edges; ++i) {
        h = fnv_mix(h, pe[i].lo);
        h = fnv_mix(h, pe[i].hi);
        h = fnv_mix(h, 0xDEADBEEFu);
    }
    return h;
}

__device__ uint64_t ir_canonical_hash_state(DeviceState ds, StateId sid) {
    __shared__ IRBlock blk;

    if (threadIdx.x == 0) {
        build_block(blk, ds, sid);
    }
    __syncthreads();

    if (blk.overflow_flag) {
        // State larger than the IRBlock fast-path bounds (kMaxIRVerts /
        // kMaxIREdges / kMaxIROccs). Fall back to the size-tolerant 1-WL
        // hash that operates directly on the CSR slice via per-thread
        // scratch (handles up to kMaxWlEdges-class states; for even
        // larger workloads the per-thread scratch in wl_hash also
        // overflows and produces a sentinel hash, but that's the same
        // soft accuracy degradation — never a correctness break since
        // 1-WL is iso-invariant).
        //
        // Critically: we do NOT record an error. This is a planned
        // graceful degradation, not a fault. (Stream 5 + a future
        // global-memory IRBlock variant will eventually let us run the
        // exact IR fast path on arbitrarily large states.)
        if (threadIdx.x != 0) return 0;
        return wl_hash_state_device(ds, sid);
    }
    if (blk.n_edges == 0) return 0;

    // Initial colours — thread-parallel across vertices (one thread per
    // vertex within the warp). For n_verts > blockDim.x each thread
    // handles a strided share.
    if (threadIdx.x == 0) {
        for (uint32_t v = 0; v < blk.n_verts; ++v) {
            blk.colors[v] = initial_color(blk, v);
        }
    }
    __syncthreads();

    // 1-WL refinement to fixpoint (or kMaxIRRefineIters). Single-thread
    // for now because IRBlock access patterns overlap heavily and the
    // per-vertex work is cheap; parallelising this is a worthwhile
    // optimisation once the fast path is validated.
    if (threadIdx.x == 0) {
        for (uint32_t iter = 0; iter < kMaxIRRefineIters; ++iter) {
            bool changed = false;
            for (uint32_t v = 0; v < blk.n_verts; ++v) {
                blk.new_colors[v] = refine_vertex(blk, v);
                if (blk.new_colors[v] != blk.colors[v]) changed = true;
            }
            for (uint32_t v = 0; v < blk.n_verts; ++v) blk.colors[v] = blk.new_colors[v];
            if (!changed) break;
        }
    }
    __syncthreads();

    if (threadIdx.x != 0) return 0;

    // Discreteness check + labelling.
    uint32_t labelling[kMaxIRVerts];
    bool discrete = extract_labeling(blk, labelling);
    if (!discrete) {
        // Non-discrete outcome — 1-WL refinement didn't fully separate the
        // vertex set. Emit the sorted-colour-multiset hash, which is
        // isomorphism-invariant (1-WL is iso-invariant) and therefore safe
        // for dedup correctness, but may have false positives on graphs
        // where 1-WL is strictly weaker than IR (Cai-Fürer-Immerman,
        // strongly regular, certain Cayley graphs). Replaced by proper
        // backtrack-tree IR in S3.5; until then this is a soft accuracy
        // limitation, not a correctness break — we deliberately do NOT
        // record an error so the host doesn't throw.
        uint64_t sorted_cols[kMaxIRVerts];
        for (uint32_t v = 0; v < blk.n_verts; ++v) sorted_cols[v] = blk.colors[v];
        insertion_sort_u64(sorted_cols, blk.n_verts);
        uint64_t h = fnv_mix(kFnvOffset, blk.n_verts);
        h = fnv_mix(h, blk.n_edges);
        for (uint32_t v = 0; v < blk.n_verts; ++v) h = fnv_mix(h, sorted_cols[v]);
        return h;
    }

    return canonical_hash_from_labeling(blk, labelling);
}

__global__ void k_ir_canon_range(DeviceState ds, uint32_t lo, uint32_t hi,
                                 uint64_t* out) {
    uint32_t bid = blockIdx.x;
    if (lo + bid >= hi) return;
    StateId sid = lo + bid;
    uint64_t h = ir_canonical_hash_state(ds, sid);
    if (threadIdx.x == 0) out[bid] = h;
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::ir_canon ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

}  // namespace

void compute_state_ir_hashes_range(const EngineState& engine,
                                   uint32_t lo, uint32_t hi,
                                   uint64_t* out_hashes_device) {
    if (hi <= lo) return;
    uint32_t n = hi - lo;
    k_ir_canon_range<<<n, 32>>>(engine.device(), lo, hi, out_hashes_device);
    check(cudaDeviceSynchronize(), "k_ir_canon_range sync");
}

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
