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

bool CausalGraph::is_reachable(EventId producer, EventId consumer) const {
    if (producer == consumer) return true;
    // Event ids increase along every causal edge, so an ancestor's id is strictly
    // smaller than its descendant's: a producer with id >= its consumer's cannot
    // reach it, and any node with id < producer's is out of the search cone.
    if (producer >= consumer) return false;

    // Backward BFS from consumer over the reduced predecessor adjacency, searching
    // for producer and pruning to ids >= producer. Scratch lives in the calling
    // worker's arena (bulk-reclaimed per task).
    SVec<EventId> stack;
    SUSet<EventId> visited;
    stack.push_back(consumer);
    visited.insert(consumer);
    while (!stack.empty()) {
        EventId x = stack.back();
        stack.pop_back();
        const LockFreeList<EventId>* pl = preds_.get(x);
        if (!pl) continue;
        bool found = false;
        pl->for_each([&](EventId q) {
            if (found) return;
            if (q == producer) { found = true; return; }
            // q < producer can neither be producer nor have it as an ancestor; skip.
            if (q > producer && visited.insert(q).second) stack.push_back(q);
        });
        if (found) return true;
    }
    return false;
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
            if (is_reachable(producer, consumer)) {
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
            // Record the kept edge in the reduced adjacency once per unique event
            // pair, so preds_ holds no duplicate producers for a consumer.
            if (transitive_reduction_enabled_.load(std::memory_order_relaxed)) {
                record_reduced_edge(producer, consumer);
            }
        }
    }
}

void CausalGraph::record_reduced_edge(EventId producer, EventId consumer) {
    // preds_[consumer] is written only by consumer's own thread (invariant 1) and
    // this runs once per unique event pair, so it holds no duplicate producers.
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
