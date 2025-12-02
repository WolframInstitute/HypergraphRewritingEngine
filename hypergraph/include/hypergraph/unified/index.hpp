#pragma once

#include <cstdint>

#include "types.hpp"
#include "signature.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"
#include "bitset.hpp"

namespace hypergraph::unified {

// =============================================================================
// SignatureIndex
// =============================================================================
// Maps edge signatures to lists of edge IDs.
// This is the primary index for candidate generation during pattern matching.
//
// Thread safety: Lock-free via ConcurrentMap and LockFreeList.
// - add_edge: Called by rewrite tasks when creating new edges
// - get_candidates: Called by match tasks during candidate generation
//
// The index stores all edges from the unified hypergraph. Queries filter
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
    // Called when edge is created in unified hypergraph
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

        result.value()->for_each([&](EdgeId eid) {
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

    // Get candidates into array (for use with stack arrays)
    uint32_t get_candidates(
        const EdgeSignature& pattern_sig,
        const SparseBitset& state_edges,
        EdgeId* out,
        uint32_t max_count
    ) const {
        uint32_t count = 0;
        for_each_candidate(pattern_sig, state_edges, [&](EdgeId eid) {
            if (count < max_count) {
                out[count++] = eid;
            }
        });
        return count;
    }

    // Estimate candidate count for matching order computation
    // Returns count for exact signature only (lower bound)
    uint32_t estimate_candidates(
        const EdgeSignature& pattern_sig,
        const SparseBitset& state_edges
    ) const {
        // For efficiency, we estimate based on exact signature match only
        // This is a lower bound but fast to compute
        uint64_t hash = pattern_sig.hash();
        auto result = by_signature_.lookup(hash);
        if (!result.has_value()) return 0;

        // Count edges present in state
        uint32_t count = 0;
        result.value()->for_each([&](EdgeId eid) {
            if (state_edges.contains(eid)) {
                count++;
            }
        });
        return count;
    }

    // Get number of distinct signatures in index
    size_t num_signatures() const {
        return by_signature_.size();
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
    SegmentedArray<LockFreeList<EdgeId>> vertex_to_edges_;

public:
    InvertedVertexIndex() = default;

    // Non-copyable
    InvertedVertexIndex(const InvertedVertexIndex&) = delete;
    InvertedVertexIndex& operator=(const InvertedVertexIndex&) = delete;

    // Add edge to the index
    // Called when edge is created in unified hypergraph
    void add_edge(
        EdgeId eid,
        const VertexId* vertices,
        uint8_t arity,
        ConcurrentHeterogeneousArena& arena
    ) {
        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = vertices[i];

            // Ensure vertex entry exists (thread-safe)
            vertex_to_edges_.ensure_size(v + 1, arena);

            // Add edge to vertex's list
            vertex_to_edges_[v].push(eid, arena);
        }
    }

    // Get all edges containing vertex, filtered by state
    template<typename Visitor>
    void for_each_edge(
        VertexId v,
        const SparseBitset& state_edges,
        Visitor&& visit
    ) const {
        if (v >= vertex_to_edges_.size()) return;

        vertex_to_edges_[v].for_each([&](EdgeId eid) {
            if (state_edges.contains(eid)) {
                visit(eid);
            }
        });
    }

    // Get edges containing vertex at specific position
    // Requires access to edge data to check position
    template<typename Visitor, typename EdgeAccessor>
    void for_each_edge_at_position(
        VertexId v,
        uint8_t position,
        const SparseBitset& state_edges,
        const EdgeAccessor& get_edge,
        Visitor&& visit
    ) const {
        for_each_edge(v, state_edges, [&](EdgeId eid) {
            const auto& edge = get_edge(eid);
            if (position < edge.arity && edge.vertices[position] == v) {
                visit(eid);
            }
        });
    }

    // Intersect: edges containing ALL specified vertices, filtered by state
    // This is the key operation for candidate generation with bound variables
    template<typename Visitor>
    void for_each_edge_containing_all(
        const VertexId* vertices,
        uint8_t count,
        const SparseBitset& state_edges,
        Visitor&& visit
    ) const {
        if (count == 0) return;

        // Find vertex with smallest edge list (for efficiency)
        VertexId smallest_v = vertices[0];
        uint32_t smallest_count = UINT32_MAX;

        for (uint8_t i = 0; i < count; ++i) {
            VertexId v = vertices[i];
            if (v >= vertex_to_edges_.size()) return;  // No edges possible

            uint32_t v_count = 0;
            vertex_to_edges_[v].for_each([&](EdgeId) { v_count++; });

            if (v_count < smallest_count) {
                smallest_count = v_count;
                smallest_v = v;
            }
        }

        // Iterate through smallest list, filter by other vertices
        for_each_edge(smallest_v, state_edges, [&](EdgeId eid) {
            // Check if edge contains all other vertices
            bool contains_all = true;
            for (uint8_t i = 0; i < count && contains_all; ++i) {
                if (vertices[i] == smallest_v) continue;

                // Check if this vertex's list contains eid
                bool found = false;
                vertex_to_edges_[vertices[i]].for_each([&](EdgeId other_eid) {
                    if (other_eid == eid) found = true;
                });
                if (!found) contains_all = false;
            }

            if (contains_all) {
                visit(eid);
            }
        });
    }

    // Get count of edges containing vertex
    uint32_t edge_count(VertexId v) const {
        if (v >= vertex_to_edges_.size()) return 0;

        uint32_t count = 0;
        vertex_to_edges_[v].for_each([&](EdgeId) { count++; });
        return count;
    }

    // Get count of edges containing vertex that are in state
    uint32_t edge_count_in_state(VertexId v, const SparseBitset& state_edges) const {
        if (v >= vertex_to_edges_.size()) return 0;

        uint32_t count = 0;
        vertex_to_edges_[v].for_each([&](EdgeId eid) {
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

}  // namespace hypergraph::unified
