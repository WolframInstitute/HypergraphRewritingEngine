#pragma once

#include <cstdint>
#include "hgcommon/wl_core.hpp"
#include <cstddef>
#include <algorithm>
#include <vector>
#include <atomic>
#include <unordered_map>

#include "types.hpp"
#include "bitset.hpp"
#include "arena.hpp"
#include "scratch_alloc.hpp"
#include "segmented_array.hpp"
#include "concurrent_map.hpp"
#include "lock_free_list.hpp"

namespace hypergraph {

// =============================================================================
// WLHash: Weisfeiler-Lehman style hashing for hypergraph isomorphism
// =============================================================================
//
// This implements WL (color refinement) style hashing.
//
// WL hashing works by:
// 1. Assigning initial colors based on vertex degree/position structure
// 2. Iteratively refining colors by aggregating neighbor colors
// 3. Converging when colors stabilize (no changes between iterations)
//
// Key characteristics:
// - Iterative message passing (all vertices update simultaneously)
// - Polynomial time: O(V^2) per iteration, typically O(log V) iterations
// - Approximate: may not distinguish all non-isomorphic graphs (false-positive collisions)
//
// Thread safety:
// - Edge registration: Lock-free via LockFreeList
// - State hash computation: Thread-safe, uses shared read-only data
// - Hash caching: Lock-free via ConcurrentMap
//
// Note: Common types (EdgeOccurrence, VertexHashCache, EdgeCorrespondence,
// EventSignature, FNV constants) are defined in types.hpp

// Forward declarations
class Hypergraph;

// =============================================================================
// WLHash: Main class
// =============================================================================

class WLHash {
public:
    // FNV constants are defined in types.hpp
    static constexpr size_t MAX_REFINEMENT_DEPTH = 100;

    explicit WLHash(ConcurrentHeterogeneousArena* arena)
        : arena_(arena)
    {}

    // Non-copyable
    WLHash(const WLHash&) = delete;
    WLHash& operator=(const WLHash&) = delete;


    // =========================================================================
    // State Canonical Hash Computation
    // =========================================================================

    // Shared WL body. When out_cache != nullptr it also fills the per-vertex
    // subtree-hash cache (which feeds edge-correspondence recovery between
    // isomorphic states) from the PERSISTENT arena, because those colours outlive
    // the call. When out_cache == nullptr it hashes only and makes ZERO persistent
    // allocation — the common state-dedup path. One implementation for both, so the
    // two paths can never drift. All temporary storage is per-worker scratch.
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t compute_state_hash_impl(
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        VertexHashCache* out_cache
    ) const {
        if (state_edges.empty()) {
            return 0;
        }

        auto _wl_mark = worker_scratch().mark();

        // Collect vertices and track min/max for dense index optimization
        ArenaVector<VertexId> vertices(worker_scratch());
        VertexId max_vertex = 0;
        VertexId min_vertex = UINT32_MAX;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                vertices.push_back(verts[i]);
                if (verts[i] > max_vertex) max_vertex = verts[i];
                if (verts[i] < min_vertex) min_vertex = verts[i];
            }
        });

        // Remove duplicates
        std::sort(vertices.begin(), vertices.end());
        auto new_end = std::unique(vertices.begin(), vertices.end());
        vertices.resize(new_end - vertices.begin());

        if (vertices.size() == 0) {
            worker_scratch().release(_wl_mark);
            return 0;
        }

        const size_t num_vertices = vertices.size();
        const VertexId range = max_vertex - min_vertex + 1;

        // Always use arena-allocated dense array for O(1) lookup without hash overhead
        // 1M vertices = 8MB which is acceptable for performance
        ArenaVector<size_t> vertex_index(worker_scratch(), range);
        vertex_index.resize(range);
        for (size_t i = 0; i < num_vertices; ++i) {
            vertex_index[vertices[i] - min_vertex] = i;
        }

        // Direct array lookup - no hash overhead
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            return vertex_index[v - min_vertex];
        };

        // Flatten to local-index edges (local index = sorted-vertex position) and
        // hash with the shared hgcommon WL core — ONE implementation shared with
        // the GPU port, so their WL hashes are identical by construction.
        size_t n_edges = 0, total_occ = 0;
        state_edges.for_each([&](EdgeId eid){ ++n_edges; total_occ += edge_arities[eid]; });

        ArenaVector<uint8_t>  ea(worker_scratch(), n_edges);         ea.resize(n_edges);
        ArenaVector<uint32_t> eoff(worker_scratch(), n_edges);       eoff.resize(n_edges);
        ArenaVector<uint32_t> ev(worker_scratch(), total_occ);       ev.resize(total_occ);
        {
            size_t e = 0, off = 0;
            state_edges.for_each([&](EdgeId eid){
                uint8_t arity = edge_arities[eid];
                const VertexId* verts = edge_vertices[eid];
                eoff[e] = static_cast<uint32_t>(off);
                ea[e]   = arity;
                for (uint8_t i = 0; i < arity; ++i)
                    ev[off++] = static_cast<uint32_t>(get_vertex_idx(verts[i]));
                ++e;
            });
        }

        // Core scratch (transient, per-worker arena).
        size_t nbr_cap = total_occ ? total_occ : 1;
        ArenaVector<uint64_t> cur(worker_scratch(), num_vertices);        cur.resize(num_vertices);
        ArenaVector<uint64_t> nxt(worker_scratch(), num_vertices);        nxt.resize(num_vertices);
        ArenaVector<uint64_t> dscr(worker_scratch(), num_vertices);       dscr.resize(num_vertices);
        ArenaVector<uint32_t> occ_off(worker_scratch(), num_vertices + 1); occ_off.resize(num_vertices + 1);
        ArenaVector<uint32_t> occ_edge(worker_scratch(), total_occ);      occ_edge.resize(total_occ);
        ArenaVector<uint8_t>  occ_pos(worker_scratch(), total_occ);       occ_pos.resize(total_occ);
        ArenaVector<uint64_t> nbr(worker_scratch(), nbr_cap);             nbr.resize(nbr_cap);

        // Only when a cache is requested do the final per-vertex colours land in the
        // PERSISTENT arena (they must outlive this call for edge correspondence).
        // Hash-only callers pass nullptr and allocate nothing persistent — this was
        // the leak: every dedup hash used to strand two arrays in the never-reclaimed
        // arena.
        uint64_t* out_colours = nullptr;
        if (out_cache) {
            out_cache->count = static_cast<uint32_t>(num_vertices);
            out_cache->vertices = arena_->allocate_array<VertexId>(out_cache->count);
            out_cache->hashes   = arena_->allocate_array<uint64_t>(out_cache->count);
            for (size_t i = 0; i < num_vertices; ++i) out_cache->vertices[i] = vertices[i];
            out_colours = out_cache->hashes;
        }

        uint64_t hash = hgcommon::wl_canonical_hash(
            ea.data(), eoff.data(), ev.data(),
            static_cast<uint32_t>(n_edges), static_cast<uint32_t>(num_vertices), hgcommon::WL_MAX_REFINE_ITERS,
            cur.data(), nxt.data(), occ_off.data(), occ_edge.data(), occ_pos.data(),
            nbr.data(), static_cast<uint32_t>(nbr_cap), dscr.data(), out_colours);

        worker_scratch().release(_wl_mark);
        return hash;
    }

    // Cached: hash + per-vertex colour cache (for edge correspondence).
    template<typename VertexAccessor, typename ArityAccessor>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_with_cache(
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        VertexHashCache cache{};
        uint64_t hash = compute_state_hash_impl(state_edges, edge_vertices, edge_arities, &cache);
        return {hash, cache};
    }

    // Hash only: ZERO persistent allocation (the common state-dedup path).
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t compute_state_hash(
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        return compute_state_hash_impl(state_edges, edge_vertices, edge_arities, nullptr);
    }


    // =========================================================================
    // Edge Correspondence (O(E) algorithm)
    // =========================================================================

    // Find edge correspondence between two isomorphic states
    // Uses vertex subtree hashes - no O(E^2) search needed
    template<typename VertexAccessor, typename ArityAccessor>
    EdgeCorrespondence find_edge_correspondence(
        const SparseBitset& state1_edges,
        const SparseBitset& state2_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        EdgeCorrespondence result;

        // Compute vertex hashes for both states
        auto [hash1, cache1] = compute_state_hash_with_cache(state1_edges, edge_vertices, edge_arities);
        auto [hash2, cache2] = compute_state_hash_with_cache(state2_edges, edge_vertices, edge_arities);

        // Quick check: must have same canonical hash
        if (hash1 != hash2) {
            return result;  // Not isomorphic
        }

        // Must have same number of vertices
        if (cache1.count != cache2.count) {
            return result;
        }

        // Build edge signature -> list of edges map for state2
        // Edge signature = tuple of vertex subtree hashes (in order)
        // Multiple edges can have the same signature (automorphic edges)
        SUMap<uint64_t, SVec<EdgeId>> edge2_by_sig;   // worker-local scratch

        state2_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            uint64_t sig = compute_edge_signature(verts, arity, cache2);
            edge2_by_sig[sig].push_back(eid);
        });

        // Collect edges from state1
        SVec<EdgeId> edges1;
        state1_edges.for_each([&](EdgeId eid) {
            edges1.push_back(eid);
        });

        // Collect edges from state2
        size_t edge2_count = 0;
        state2_edges.for_each([&](EdgeId) { ++edge2_count; });

        if (edges1.size() != edge2_count) {
            return result;  // Different edge counts
        }

        // Allocate result arrays
        result.count = static_cast<uint32_t>(edges1.size());
        result.state1_edges = arena_->allocate_array<EdgeId>(result.count);
        result.state2_edges = arena_->allocate_array<EdgeId>(result.count);

        // Track which edges from state2 have been used
        SUMap<uint64_t, size_t> sig_next_idx;  // automorphic edges: next available (worker scratch)

        // Find correspondence for each edge in state1
        for (size_t i = 0; i < edges1.size(); ++i) {
            EdgeId e1 = edges1[i];
            uint8_t arity = edge_arities[e1];
            const VertexId* verts = edge_vertices[e1];

            uint64_t sig = compute_edge_signature(verts, arity, cache1);

            auto e2_it = edge2_by_sig.find(sig);
            if (e2_it == edge2_by_sig.end()) {
                // No matching edge found - states not actually isomorphic
                result.valid = false;
                return result;
            }

            // For automorphic edges (same signature), use next available
            // This is valid because automorphic edges are interchangeable
            size_t& idx = sig_next_idx[sig];
            if (idx >= e2_it->second.size()) {
                // More edges in state1 with this signature than in state2
                result.valid = false;
                return result;
            }

            result.state1_edges[i] = e1;
            result.state2_edges[i] = e2_it->second[idx];
            ++idx;
        }

        result.valid = true;
        return result;
    }

    // =========================================================================
    // Event Signature Computation (for Full deduplication)
    // =========================================================================

    // Compute event signature for Full deduplication mode
    template<typename VertexAccessor, typename ArityAccessor>
    EventSignature compute_event_signature(
        uint64_t input_state_hash,
        uint64_t output_state_hash,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& input_cache,
        const VertexHashCache& output_cache,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        EventSignature sig;
        sig.input_state_hash = input_state_hash;
        sig.output_state_hash = output_state_hash;

        // Compute consumed edges signature
        sig.consumed_edges_sig = compute_edge_set_signature(
            consumed_edges, num_consumed, input_cache, edge_vertices, edge_arities);

        // Compute produced edges signature
        sig.produced_edges_sig = compute_edge_set_signature(
            produced_edges, num_produced, output_cache, edge_vertices, edge_arities);

        return sig;
    }

private:
    ConcurrentHeterogeneousArena* arena_;

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    // Use fnv_hash from types.hpp - alias for compatibility
    static uint64_t fnv_combine(uint64_t h, uint64_t value) {
        return fnv_hash(h, value);
    }

    // Insertion sort for small arrays (faster than std::sort for n < 16)
    template<typename Iter>
    static void insertion_sort(Iter begin, Iter end) {
        for (Iter i = begin; i != end; ++i) {
            auto key = *i;
            Iter j = i;
            while (j != begin && *(j - 1) > key) {
                *j = *(j - 1);
                --j;
            }
            *j = key;
        }
    }

    // splitmix64 finaliser — strong avalanche so a multiset SUM of these is a
    // collision-resistant order-independent hash.
    static uint64_t mix64(uint64_t z) {
        z += 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    // Canonical state hash as a COMMUTATIVE multiset combine of per-vertex colours
    // and per-edge hashes. Order-independent => no sort needed to canonicalise, and
    // an incremental update can patch only the changed vertex/edge contributions
    // (O(delta)). Discrimination equals the sorted form (both are functions of the
    // colour multiset + edge-hash multiset). Optionally outputs the running sums +
    // edge count so callers can patch incrementally.
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t build_canonical_hash_dense(
        const ArenaVector<uint64_t>& vertex_hashes,
        const ArenaVector<VertexId>& vertices,
        VertexId min_vertex,
        const ArenaVector<size_t>& vertex_index,
        const SparseBitset& state_edges,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities,
        uint64_t* out_vsum = nullptr, uint64_t* out_esum = nullptr, size_t* out_m = nullptr
    ) const {
        uint64_t vsum = 0;
        for (size_t i = 0; i < vertices.size(); ++i) vsum += mix64(vertex_hashes[i]);
        uint64_t esum = 0; size_t m = 0;
        state_edges.for_each([&](EdgeId eid) {
            ++m;
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            uint64_t eh = FNV_OFFSET; eh = fnv_combine(eh, arity);
            for (uint8_t i = 0; i < arity; ++i) eh = fnv_combine(eh, vertex_hashes[vertex_index[verts[i] - min_vertex]]);
            esum += mix64(eh);
        });
        if (out_vsum) *out_vsum = vsum;
        if (out_esum) *out_esum = esum;
        if (out_m) *out_m = m;
        uint64_t hash = FNV_OFFSET;
        hash = fnv_combine(hash, vertices.size());
        hash = fnv_combine(hash, m);
        hash = fnv_combine(hash, vsum);
        hash = fnv_combine(hash, esum);
        return hash;
    }

    // Compute edge signature from vertex hashes
    uint64_t compute_edge_signature(
        const VertexId* vertices,
        uint8_t arity,
        const VertexHashCache& cache
    ) const {
        uint64_t sig = FNV_OFFSET;
        sig = fnv_combine(sig, arity);

        for (uint8_t i = 0; i < arity; ++i) {
            uint64_t vh = cache.lookup(vertices[i]);
            sig = fnv_combine(sig, vh);
        }

        return sig;
    }

    // Compute signature for a set of edges
    template<typename VertexAccessor, typename ArityAccessor>
    uint64_t compute_edge_set_signature(
        const EdgeId* edges,
        uint8_t count,
        const VertexHashCache& cache,
        const VertexAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) const {
        // Collect individual edge signatures (worker-local scratch)
        SVec<uint64_t> edge_sigs;
        edge_sigs.reserve(count);

        for (uint8_t i = 0; i < count; ++i) {
            EdgeId eid = edges[i];
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];

            edge_sigs.push_back(compute_edge_signature(verts, arity, cache));
        }

        // Sort for canonical ordering (order of edges in event shouldn't matter)
        insertion_sort(edge_sigs.begin(), edge_sigs.end());

        // Combine into single signature
        uint64_t sig = FNV_OFFSET;
        sig = fnv_combine(sig, count);
        for (uint64_t es : edge_sigs) {
            sig = fnv_combine(sig, es);
        }

        return sig;
    }
};

}  // namespace hypergraph
