#pragma once

#include <cstdint>

#include "types.hpp"
#include "signature.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"
#include "bitset.hpp"

namespace hypergraph {

// =============================================================================
// SignatureIndex
// =============================================================================
// Maps edge signatures to lists of edge IDs.
// This is the primary index for candidate generation during pattern matching.
//
// Thread safety: Lock-free via ConcurrentMap and LockFreeList.
// - add_edge: Called by rewrite tasks when creating new edges
// - for_each_candidate: Called by match tasks during candidate generation
//
// The index stores all edges from the hypergraph. Queries filter
// by state (SparseBitset) to get edges present in a specific state.

class SignatureIndex {
    // Map: signature hash → list of edges with that exact signature
    ConcurrentMap<uint64_t, LockFreeList<EdgeId>*> by_signature_;

public:
    SignatureIndex() = default;

    // Non-copyable (contains pointers to arena-allocated lists)
    SignatureIndex(const SignatureIndex&) = delete;
    SignatureIndex& operator=(const SignatureIndex&) = delete;

    // Add edge to the index
    // Called when edge is created in hypergraph
    void add_edge(EdgeId eid, const EdgeSignature& sig, ConcurrentHeterogeneousArena& arena) {
        uint64_t hash = sig.hash();

        // Get or create list for this signature
        auto result = by_signature_.lookup(hash);
        LockFreeList<EdgeId>* list = nullptr;

        if (result.has_value()) {
            list = result.value();
        } else {
            // Create new list
            list = arena.create<LockFreeList<EdgeId>>();
            auto [existing, inserted] = by_signature_.insert_if_absent(hash, list);
            if (!inserted) {
                // Another thread created it first, use theirs
                list = existing;
            }
        }

        // Add edge to list
        list->push(eid, arena);
    }

    // Get all edges with exact signature, filtered by state
    template<typename Visitor>
    void for_each_edge_with_signature(
        const EdgeSignature& sig,
        const SparseBitset& state_edges,
        Visitor&& visit
    ) const {
        uint64_t hash = sig.hash();
        auto result = by_signature_.lookup(hash);
        if (!result.has_value()) return;

        LockFreeList<EdgeId>* list = result.value();
        list->for_each([&](EdgeId eid) {
            if (state_edges.contains(eid)) {
                visit(eid);
            }
        });
    }

    // Get candidate edges for a pattern signature, filtered by state
    // This enumerates all compatible data signatures and collects edges
    template<typename Visitor>
    void for_each_candidate(
        const EdgeSignature& pattern_sig,
        const SparseBitset& state_edges,
        Visitor&& visit
    ) const {
        // Enumerate all data signatures compatible with pattern signature
        struct Context {
            const SignatureIndex* self;
            const SparseBitset* state_edges;
            Visitor* visitor;
        };

        Context ctx{this, &state_edges, &visit};

        enumerate_compatible_signatures(
            pattern_sig,
            [](const EdgeSignature& data_sig, void* user_data) {
                auto* ctx = static_cast<Context*>(user_data);
                ctx->self->for_each_edge_with_signature(
                    data_sig, *ctx->state_edges,
                    *ctx->visitor
                );
            },
            &ctx
        );
    }

    // Get candidate edges using pre-computed compatible signatures (faster)
    // Use this when the same pattern signature is queried repeatedly
    template<typename Visitor>
    void for_each_candidate_cached(
        const CompatibleSignatureCache& sig_cache,
        const SparseBitset& state_edges,
        Visitor&& visit
    ) const {
        sig_cache.for_each([&](const EdgeSignature& data_sig) {
            for_each_edge_with_signature(data_sig, state_edges, visit);
        });
    }

};

// =============================================================================
// InvertedVertexIndex
// =============================================================================
// Maps vertices to lists of edges containing that vertex.
// Essential for candidate generation when pattern variables are bound.
//
// Thread safety: Lock-free via SegmentedArray and LockFreeList.
// - Vertices are allocated sequentially (via GlobalCounters)
// - SegmentedArray grows lock-free as new vertices are added
// - Each vertex's edge list is a LockFreeList

class InvertedVertexIndex {
    // vertex_id → list of edges containing that vertex
    // Using ConcurrentMap for lock-free, wait-free access
    // EMPTY_KEY = 0xFFFFFFFE, LOCKED_KEY = 0xFFFFFFFF (both are INVALID_ID-ish values)
    static constexpr VertexId EMPTY_VERTEX = 0xFFFFFFFE;
    static constexpr VertexId LOCKED_VERTEX = 0xFFFFFFFF;
    ConcurrentMap<VertexId, LockFreeList<EdgeId>*, EMPTY_VERTEX, LOCKED_VERTEX> vertex_to_edges_;

public:
    InvertedVertexIndex() = default;

    // Non-copyable
    InvertedVertexIndex(const InvertedVertexIndex&) = delete;
    InvertedVertexIndex& operator=(const InvertedVertexIndex&) = delete;

    // Add edge to the index
    // Called when edge is created in hypergraph
    void add_edge(
        EdgeId eid,
        const VertexId* vertices,
        uint8_t arity,
        ConcurrentHeterogeneousArena& arena
    ) {
        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = vertices[i];

            // Get or create list for this vertex (lock-free)
            auto result = vertex_to_edges_.lookup(v);
            LockFreeList<EdgeId>* list = nullptr;

            if (result.has_value()) {
                list = result.value();
            } else {
                // Create new list
                list = arena.create<LockFreeList<EdgeId>>();
                auto [existing, inserted] = vertex_to_edges_.insert_if_absent(v, list);
                if (!inserted) {
                    // Another thread created it first, use theirs
                    // (our list is wasted but that's fine - arena memory)
                    list = existing;
                }
            }

            // Add edge to vertex's list
            list->push(eid, arena);
        }
    }

    // Get all edges containing vertex, filtered by state
    template<typename Visitor>
    void for_each_edge(
        VertexId v,
        const SparseBitset& state_edges,
        Visitor&& visit
    ) const {
        auto result = vertex_to_edges_.lookup(v);
        if (!result.has_value()) return;

        result.value()->for_each([&](EdgeId eid) {
            if (state_edges.contains(eid)) {
                visit(eid);
            }
        });
    }

    // Get edges containing vertex at specific position

    // Intersect: edges containing ALL specified vertices, filtered by state
    // This is the key operation for candidate generation with bound variables
    //
    // EdgeAccessor: functor returning edge data with .vertices and .arity members
    // This allows O(arity) vertex containment check instead of O(edges_per_vertex) list scan
    template<typename EdgeAccessor, typename Visitor>
    void for_each_edge_containing_all(
        const VertexId* vertices,
        uint8_t count,
        const SparseBitset& state_edges,
        const EdgeAccessor& get_edge,
        Visitor&& visit
    ) const {
        if (count == 0) return;

        // Seed the scan from the bound vertex with the SHORTEST occurrence list.
        // The yielded set (edges in state containing ALL bound vertices) is the same
        // whichever bound vertex's list is walked; only the walk length differs. The
        // inverted index is global and append-only over the whole evolution, so a hub
        // vertex owns a huge occurrence list; seeding from the rarest bound vertex
        // avoids a near-full-history scan per query.
        //
        // If any bound vertex has no occurrence list at all, it appears in no edge, so
        // the intersection is empty and we can stop.
        constexpr uint32_t PROBE_CAP = 64;  // bounds probe cost; long lists clamp here

        uint8_t seed_idx = 0;
        const LockFreeList<EdgeId>* seed_list = nullptr;
        uint32_t seed_len = UINT32_MAX;

        for (uint8_t i = 0; i < count; ++i) {
            auto result = vertex_to_edges_.lookup(vertices[i]);
            if (!result.has_value()) return;  // vertex occurs nowhere -> empty intersection
            const LockFreeList<EdgeId>* list = result.value();

            // Probe length only far enough to decide whether this list beats the
            // current best, capped so a hub list is never fully walked just to measure
            // it. Total probing is O(count * min(shortest_len, PROBE_CAP)).
            uint32_t bound = seed_len < PROBE_CAP ? seed_len : PROBE_CAP;
            uint32_t len = 0;
            list->for_each_while([&](EdgeId) {
                ++len;
                return len < bound;
            });

            if (seed_list == nullptr || len < seed_len) {
                seed_len = len;
                seed_idx = i;
                seed_list = list;
            }
        }

        // Iterate the seed vertex's edge list; keep edges present in state that also
        // contain every OTHER bound vertex. This is O(arity * count) per edge instead
        // of an O(edges_per_vertex * count) list membership scan.
        seed_list->for_each([&](EdgeId eid) {
            if (!state_edges.contains(eid)) return;

            const auto& edge = get_edge(eid);
            bool contains_all = true;

            for (uint8_t i = 0; i < count && contains_all; ++i) {
                if (i == seed_idx) continue;  // seed vertex is present by construction
                VertexId required_v = vertices[i];
                bool found = false;
                for (uint8_t j = 0; j < edge.arity && !found; ++j) {
                    if (edge.vertices[j] == required_v) {
                        found = true;
                    }
                }
                if (!found) contains_all = false;
            }

            if (contains_all) {
                visit(eid);
            }
        });
    }

    // Get count of edges containing vertex
    uint32_t edge_count(VertexId v) const {
        auto result = vertex_to_edges_.lookup(v);
        if (!result.has_value()) return 0;

        uint32_t count = 0;
        result.value()->for_each([&](EdgeId) { count++; });
        return count;
    }

    // Get count of edges containing vertex that are in state
    uint32_t edge_count_in_state(VertexId v, const SparseBitset& state_edges) const {
        auto result = vertex_to_edges_.lookup(v);
        if (!result.has_value()) return 0;

        uint32_t count = 0;
        result.value()->for_each([&](EdgeId eid) {
            if (state_edges.contains(eid)) count++;
        });
        return count;
    }

    // Number of vertices tracked
    size_t num_vertices() const {
        return vertex_to_edges_.size();
    }
};

// =============================================================================
// Combined Index for Pattern Matching
// =============================================================================
// Wraps both indices and provides the primary API for candidate generation

class PatternMatchingIndex {
    SignatureIndex signature_index_;
    InvertedVertexIndex inverted_index_;

public:
    PatternMatchingIndex() = default;

    // Non-copyable
    PatternMatchingIndex(const PatternMatchingIndex&) = delete;
    PatternMatchingIndex& operator=(const PatternMatchingIndex&) = delete;

    // Add edge to both indices
    void add_edge(
        EdgeId eid,
        const VertexId* vertices,
        uint8_t arity,
        ConcurrentHeterogeneousArena& arena
    ) {
        EdgeSignature sig = EdgeSignature::from_edge(vertices, arity);
        signature_index_.add_edge(eid, sig, arena);
        inverted_index_.add_edge(eid, vertices, arity, arena);
    }

    // Access individual indices
    const SignatureIndex& signature_index() const { return signature_index_; }
    const InvertedVertexIndex& inverted_index() const { return inverted_index_; }

    SignatureIndex& signature_index() { return signature_index_; }
    InvertedVertexIndex& inverted_index() { return inverted_index_; }
};

}  // namespace hypergraph
