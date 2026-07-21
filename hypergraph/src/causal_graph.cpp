// causal_graph.cpp - Implementation of CausalGraph class

#include "hypergraph/causal_graph.hpp"
#include "hypergraph/scratch_alloc.hpp"

namespace hypergraph {

// =============================================================================
// Edge Causal Tracking
// =============================================================================

EdgeCausalInfo* CausalGraph::get_or_create_edge_producer(EdgeId edge) {
    return &edge_producers_.get_or_default(edge, *arena_);
}

LockFreeList<EventId>* CausalGraph::get_or_create_edge_consumers(EdgeId edge) {
    return &edge_consumers_.get_or_default(edge, *arena_);
}

CausalGraph::DescAncSet* CausalGraph::get_or_create_desc(EventId event) {
    uint64_t key = event;

    auto result = desc_.lookup(key);
    if (result.has_value()) {
        return *result;
    }

    // Per-event descendant closures are small for the vast majority of events; start
    // tiny and let the set resize the rare large ones, rather than paying a full
    // default-capacity table per event.
    // create_untracked: the set's tables live in the arena and are bulk-reclaimed, so
    // its destructor is a no-op — skip the per-object destructor registration (a
    // shared-list CAS) that would otherwise contend on this high-frequency path.
    auto* new_set = arena_->template create_untracked<DescAncSet>(DESC_ANC_SET_INITIAL_CAPACITY, arena_);
    auto [existing, inserted] = desc_.insert_if_absent(key, new_set);
    return inserted ? new_set : existing;
}

bool CausalGraph::is_reachable_via_desc(EventId producer, EventId consumer) const {
    if (producer == consumer) return true;

    auto desc_result = desc_.lookup(producer);
    if (!desc_result.has_value()) return false;

    return (*desc_result)->contains(encode_id(consumer));
}

LockFreeList<EventId>* CausalGraph::get_or_create_state_events(StateId state) {
    uint64_t key = state;

    auto result = state_events_.lookup(key);
    if (result.has_value()) {
        return *result;
    }

    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = state_events_.insert_if_absent(key, new_list);
    return inserted ? new_list : existing;
}

LockFreeList<EventId>* CausalGraph::get_or_create_state_edge_events(StateId state, EdgeId edge) {
    uint64_t key = (static_cast<uint64_t>(state) << 32) | edge;

    auto result = state_edge_events_.lookup(key);
    if (result.has_value()) {
        return *result;
    }

    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = state_edge_events_.insert_if_absent(key, new_list);
    return inserted ? new_list : existing;
}

bool CausalGraph::set_edge_producer(EdgeId edge, EventId producer) {
    EdgeCausalInfo* info = get_or_create_edge_producer(edge);
    LockFreeList<EventId>* consumers = get_or_create_edge_consumers(edge);

    EventId expected = INVALID_ID;
    bool was_set = info->producer.compare_exchange_strong(
        expected, producer,
        std::memory_order_release,
        std::memory_order_acquire
    );

    if (was_set) {
        consumers->for_each([&](EventId consumer) {
            add_causal_edge(producer, consumer, edge);
        });
    }

    return was_set;
}

void CausalGraph::add_edge_consumer(EdgeId edge, EventId consumer) {
    EdgeCausalInfo* info = get_or_create_edge_producer(edge);
    LockFreeList<EventId>* consumers = get_or_create_edge_consumers(edge);

    consumers->push(consumer, *arena_);

    EventId producer = info->producer.load(std::memory_order_acquire);
    if (producer != INVALID_ID) {
        add_causal_edge(producer, consumer, edge);
    }
}

EventId CausalGraph::get_edge_producer(EdgeId edge) const {
    // get() returns null for an edge whose producer slot has not been materialized
    // (out of range, or its segment not yet installed) -- such an edge has no producer.
    const EdgeCausalInfo* info = edge_producers_.get(edge);
    if (!info) return INVALID_ID;
    return info->producer.load(std::memory_order_acquire);
}

// =============================================================================
// Branchial Tracking
// =============================================================================

// =============================================================================
// Graph Access
// =============================================================================

void CausalGraph::add_causal_edge(EventId producer, EventId consumer, EdgeId edge) {
    if (transitive_reduction_enabled_.load(std::memory_order_relaxed)) {
        uint64_t pair_key = (static_cast<uint64_t>(producer) << 32) | consumer;
        auto existing_pair = seen_causal_event_pairs_.lookup(pair_key);

        if (!existing_pair.has_value()) {
            if (is_reachable_via_desc(producer, consumer)) {
                num_redundant_edges_skipped_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }

    uint64_t triple_key = 14695981039346656037ULL;
    triple_key ^= producer;
    triple_key *= 1099511628211ULL;
    triple_key ^= consumer;
    triple_key *= 1099511628211ULL;
    triple_key ^= edge;
    triple_key *= 1099511628211ULL;

    auto [_, inserted] = seen_causal_triples_.insert_if_absent(triple_key, true);
    if (inserted) {
        causal_edges_.push(CausalEdge(producer, consumer, edge), *arena_);
        num_causal_edges_.fetch_add(1, std::memory_order_relaxed);

#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        VIZ_EMIT_CAUSAL_EDGE(producer, consumer, edge);
#endif

        uint64_t pair_key = (static_cast<uint64_t>(producer) << 32) | consumer;
        auto [_2, pair_inserted] = seen_causal_event_pairs_.insert_if_absent(pair_key, true);
        if (pair_inserted) {
            num_causal_event_pairs_.fetch_add(1, std::memory_order_relaxed);
            // Maintain the closure once per unique event pair. The union is
            // pair-dependent and idempotent, so folding it once (rather than once per
            // parallel edge of the same pair) yields the identical final Desc closure.
            if (transitive_reduction_enabled_.load(std::memory_order_relaxed)) {
                update_transitive_closure(producer, consumer);
            }
        }
    }
}

void CausalGraph::update_transitive_closure(EventId producer, EventId consumer) {
    // {consumer} ∪ Desc[consumer], held ENCODED (id+1; 0 is the set's empty sentinel),
    // with the consumer at index 0 so the reconvergence skip can test it first. Desc
    // is read via lookup, not get_or_create: a terminal consumer has no Desc set yet,
    // and materializing an empty one would waste an allocation per leaf event.
    SVec<uint32_t> descendants_to_add;  // encoded ids
    auto dc = desc_.lookup(consumer);
    descendants_to_add.reserve(1 + (dc.has_value() ? (*dc)->size() : 0));
    descendants_to_add.push_back(encode_id(consumer));
    if (dc.has_value()) {
        (*dc)->for_each([&](uint32_t enc_d) {
            descendants_to_add.push_back(enc_d);
        });
    }

    // Ancestors {producer} ∪ Anc[producer] by backward BFS over recorded direct
    // predecessors. The kept (reduced) causal edges preserve reachability, so this
    // yields exactly the ancestor set. Worker-local scratch -> per-worker arena.
    SVec<EventId> ancestors;
    ancestors.reserve(8);
    ancestors.push_back(producer);
    SUSet<EventId> visited;
    visited.insert(producer);
    for (size_t head = 0; head < ancestors.size(); ++head) {
        LockFreeList<EventId>* pl = preds_.get(ancestors[head]);
        if (pl) {
            pl->for_each([&](EventId pred) {
                if (visited.insert(pred).second) ancestors.push_back(pred);
            });
        }
    }

    for (EventId a : ancestors) {
        DescAncSet* desc_a = get_or_create_desc(a);
        // Insert the consumer first. If it was already in Desc[a], then Desc[consumer]
        // is already ⊆ Desc[a] (co-maintained), so the remaining descendants add
        // nothing -- skip the inner loop. This collapses the O(|A|·|D|) probe storm to
        // O(|A|) for reconvergent ancestors.
        if (!desc_a->insert(descendants_to_add[0])) continue;
        for (size_t k = 1; k < descendants_to_add.size(); ++k) {
            desc_a->insert(descendants_to_add[k]);
        }
    }

    // Record the direct predecessor edge for future ancestor BFS. Once per pair
    // (this runs once per unique event pair), so preds_ holds no duplicates.
    preds_.get_or_default(consumer, *arena_).push(producer, *arena_);
}

void CausalGraph::add_branchial_edge(EventId e1, EventId e2, EdgeId shared) {
    branchial_edges_.push(BranchialEdge(e1, e2, shared), *arena_);
    num_branchial_edges_.fetch_add(1, std::memory_order_relaxed);

#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
    VIZ_EMIT_BRANCHIAL_EDGE(e1, e2, 0);
#endif
}

// =============================================================================
// Utility
// =============================================================================

std::vector<CausalEdge> CausalGraph::get_causal_edges() const {
    std::vector<CausalEdge> result;
    for_each_causal_edge([&](const CausalEdge& e) {
        result.push_back(e);
    });
    return result;
}

std::vector<BranchialEdge> CausalGraph::get_branchial_edges() const {
    std::vector<BranchialEdge> result;
    for_each_branchial_edge([&](const BranchialEdge& e) {
        result.push_back(e);
    });
    return result;
}

}  // namespace hypergraph
