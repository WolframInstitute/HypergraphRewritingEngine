#pragma once
#include "hgcommon/core.hpp"
#include <atomic>
#include <algorithm>
#include <string>
#include "hgcommon/namespace.hpp"

#include <cstdint>

#include "types.hpp"
#include "signature.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"
#include "bitset.hpp"

namespace HG_NAMESPACE {
namespace engine {

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
    void add_edge(EdgeId eid, const EdgeSignature& sig, ConcurrentHeterogeneousArena& arena);

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
                // Named apart from the `ctx` this call passes as user_data: the lambda is
                // captureless (it must convert to a function pointer), so the outer one is not
                // reachable here and one name for both only invites the reader to think it is.
                auto* c = static_cast<Context*>(user_data);
                c->self->for_each_edge_with_signature(
                    data_sig, *c->state_edges,
                    *c->visitor
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
        // If the pattern edge has more compatible data signatures than the cache can
        // hold (Bell(arity) > MAX_CACHED_SIGS), the cache is INCOMPLETE — enumerate the
        // full compatible set live so no matching edge is skipped. Correctness over the
        // (rare, high-arity) fast path.
        if (sig_cache.overflowed) {
            // Wrap in a fresh by-value lambda so for_each_candidate deduces a value
            // Visitor (it stores a Visitor* internally and cannot take a reference type).
            for_each_candidate(sig_cache.source_pattern_sig, state_edges,
                               [&](EdgeId eid) { visit(eid); });
            return;
        }
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
    static constexpr size_t kMissWitness = 8;
    std::atomic<size_t> lookup_misses_{0};
    std::atomic<size_t> lookup_miss_retry_hits_{0};
    std::atomic<size_t> empty_seed_walks_{0};
    std::atomic<size_t> miss_witness_count_{0};
    std::string miss_witness_[kMissWitness];
    void note_lookup_miss(VertexId v) const {
        auto* self = const_cast<InvertedVertexIndex*>(this);
        self->lookup_misses_.fetch_add(1, std::memory_order_relaxed);
        const auto retry = vertex_to_edges_.lookup(v);
        size_t len = 0;
        if (retry.has_value()) {
            self->lookup_miss_retry_hits_.fetch_add(1, std::memory_order_relaxed);
            retry.value()->for_each([&](EdgeId) { ++len; });
        }
        const size_t slot = self->miss_witness_count_.fetch_add(1, std::memory_order_acq_rel);
        if (slot < kMissWitness)
            self->miss_witness_[slot] = "lookup-miss v=" + std::to_string(v) + " retry=" +
                (retry.has_value() ? "found(" + std::to_string(len) + " edges)" : "miss") +
                " map=" + std::to_string(vertex_to_edges_.size());
    }
    void note_empty_walk(VertexId v) const {
        auto* self = const_cast<InvertedVertexIndex*>(this);
        self->empty_seed_walks_.fetch_add(1, std::memory_order_relaxed);
        size_t len = 0;
        const auto retry = vertex_to_edges_.lookup(v);
        if (retry.has_value()) retry.value()->for_each([&](EdgeId) { ++len; });
        const size_t slot = self->miss_witness_count_.fetch_add(1, std::memory_order_acq_rel);
        if (slot < kMissWitness)
            self->miss_witness_[slot] = "empty-walk v=" + std::to_string(v) + " retry-walk=" + std::to_string(len);
    }
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
    );

    // Get all edges containing vertex, filtered by state
    // Whether the index holds a list for `v` at this moment. A validator's probe: an edge that
    // a full rematch finds and the task-based path did not, checked against what the index
    // answers for that edge's vertices right after the miss.
    bool has_vertex(VertexId v) const { return vertex_to_edges_.lookup(v).has_value(); }

    // A LOOKUP THAT MISSES A BOUND VERTEX IS A DEFECT: every bound vertex came off an edge that
    // is in the index, so its list exists. Stats builds count the misses in the candidate
    // walks, retry the lookup at once, count the retries that then succeed, and keep the first
    // few as text -- which is what separates a transient answer from a permanent one.
    size_t lookup_misses() const { return lookup_misses_.load(std::memory_order_relaxed); }
    size_t lookup_miss_retry_hits() const { return lookup_miss_retry_hits_.load(std::memory_order_relaxed); }
    size_t empty_seed_walks() const { return empty_seed_walks_.load(std::memory_order_relaxed); }
    std::string miss_witness() const {
        std::string out;
        const size_t n = std::min(miss_witness_count_.load(std::memory_order_acquire), kMissWitness);
        for (size_t i = 0; i < n; ++i) out += miss_witness_[i] + " ";
        return out;
    }

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

        uint8_t seed_idx = 0;
        const LockFreeList<EdgeId>* seed_list = nullptr;

        if (count == 1) {
            // Exactly one bound vertex: there is no choice of seed, so skip the
            // length-probing pass. Its occurrence list IS the scan seed (the dominant
            // chain-rule case, where each later edge shares a single bound variable).
            auto result = vertex_to_edges_.lookup(vertices[0]);
            if (!result.has_value()) {
                HG_STAT(note_lookup_miss(vertices[0]));
                return;  // vertex occurs nowhere -> empty intersection
            }
            seed_list = result.value();
            // seed_idx stays 0
        } else {
            // Seed the scan from the bound vertex with the SHORTEST occurrence list.
            // The yielded set (edges in state containing ALL bound vertices) is the same
            // whichever bound vertex's list is walked; only the walk length differs. The
            // inverted index is global and append-only over the whole evolution, so a hub
            // vertex owns a huge occurrence list; seeding from the rarest bound vertex
            // avoids a near-full-history scan per query.
            //
            // If any bound vertex has no occurrence list at all, it appears in no edge,
            // so the intersection is empty and we can stop.
            constexpr uint32_t PROBE_CAP = 64;  // bounds probe cost; long lists clamp here

            uint32_t seed_len = UINT32_MAX;

            for (uint8_t i = 0; i < count; ++i) {
                auto result = vertex_to_edges_.lookup(vertices[i]);
                if (!result.has_value()) {
                    HG_STAT(note_lookup_miss(vertices[i]));
                    return;  // vertex occurs nowhere -> empty intersection
                }
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
        }

        // Iterate the seed vertex's edge list; keep edges present in state that also
        // contain every OTHER bound vertex. This is O(arity * count) per edge instead
        // of an O(edges_per_vertex * count) list membership scan.
        [[maybe_unused]] size_t walked = 0;   // stats builds: nodes the seed walk visited
        seed_list->for_each([&](EdgeId eid) {
            HG_STAT(++walked);
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
                // Thread the already-fetched edge to the visitor so it need not refetch.
                visit(eid, edge);
            }
        });
        HG_STAT(if (walked == 0) note_empty_walk(vertices[seed_idx]));
    }

    // Get count of edges containing vertex
    uint32_t edge_count(VertexId v) const;

    // Get count of edges containing vertex that are in state
    uint32_t edge_count_in_state(VertexId v, const SparseBitset& state_edges) const;

    // Number of vertices tracked
    size_t num_vertices() const;
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
    );

    // Access individual indices
    const SignatureIndex& signature_index() const;
    const InvertedVertexIndex& inverted_index() const;

    SignatureIndex& signature_index();
    InvertedVertexIndex& inverted_index();
};

}  // namespace engine
}  // namespace HG_NAMESPACE