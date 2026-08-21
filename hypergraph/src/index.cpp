#include "hypergraph/index.hpp"

// The bodies behind index.hpp. The query side (for_each_candidate, for_each_edge,
// for_each_edge_with_signature, the intersection walks) is templated on the visitor and
// stays in the header; what is here is the maintenance side plus the accessors.
//
// The three defaulted default constructors also stay in their classes: defining a defaulted
// special member out of class makes it user-provided, which is a property change for no gain.

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// SignatureIndex
// =============================================================================

void SignatureIndex::add_edge(EdgeId eid, const EdgeSignature& sig,
                              ConcurrentHeterogeneousArena& arena) {
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

// =============================================================================
// InvertedVertexIndex
// =============================================================================

void InvertedVertexIndex::add_edge(
    EdgeId eid,
    const VertexId* vertices,
    uint8_t arity,
    ConcurrentHeterogeneousArena& arena
) {
    for (uint8_t i = 0; i < arity; ++i) {
        VertexId v = vertices[i];

        // ONE ENTRY PER (VERTEX, EDGE), not one per occurrence. An edge that carries the
        // same vertex at two positions -- (2,2), or the middle vertex of a self-referential
        // ternary -- would otherwise sit twice in that vertex's list, and every query
        // seeded from it hands the edge to the join twice. The join binds it twice, emits
        // the same match twice, and the duplicate is absorbed downstream by the match
        // record's content dedup, so the cost is invisible in the results and real in the
        // work. It is also ORDER-DEPENDENT: which pattern position enumerates from a
        // repeated vertex depends on the join order, so two connected orders over one state
        // emit different numbers of matches for the same match set (path2 on a 12-edge
        // state: 29 emissions from one order and 26 from another, 25 distinct under both --
        // tools/join_order_counts, `diff` mode).
        //
        // The list length is also read as a selectivity estimate when choosing which bound
        // vertex to seed an intersection from, and a repeated vertex inflates that estimate
        // above the number of edges it can actually yield.
        bool already = false;
        for (uint8_t j = 0; j < i; ++j) {
            if (vertices[j] == v) { already = true; break; }
        }
        if (already) continue;

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

uint32_t InvertedVertexIndex::edge_count(VertexId v) const {
    auto result = vertex_to_edges_.lookup(v);
    if (!result.has_value()) return 0;

    uint32_t count = 0;
    result.value()->for_each([&](EdgeId) { count++; });
    return count;
}

uint32_t InvertedVertexIndex::edge_count_in_state(VertexId v,
                                                  const SparseBitset& state_edges) const {
    auto result = vertex_to_edges_.lookup(v);
    if (!result.has_value()) return 0;

    uint32_t count = 0;
    result.value()->for_each([&](EdgeId eid) {
        if (state_edges.contains(eid)) count++;
    });
    return count;
}

size_t InvertedVertexIndex::num_vertices() const {
    return vertex_to_edges_.size();
}

// =============================================================================
// PatternMatchingIndex
// =============================================================================

void PatternMatchingIndex::add_edge(
    EdgeId eid,
    const VertexId* vertices,
    uint8_t arity,
    ConcurrentHeterogeneousArena& arena
) {
    EdgeSignature sig = EdgeSignature::from_edge(vertices, arity);
    signature_index_.add_edge(eid, sig, arena);
    inverted_index_.add_edge(eid, vertices, arity, arena);
}

const SignatureIndex& PatternMatchingIndex::signature_index() const { return signature_index_; }
const InvertedVertexIndex& PatternMatchingIndex::inverted_index() const { return inverted_index_; }

SignatureIndex& PatternMatchingIndex::signature_index() { return signature_index_; }
InvertedVertexIndex& PatternMatchingIndex::inverted_index() { return inverted_index_; }

}  // namespace engine
}  // namespace HG_NAMESPACE
