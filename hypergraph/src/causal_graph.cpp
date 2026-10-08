#include "hgcommon/core.hpp"
#include "hgcommon/rendezvous.hpp"
#include "hgcommon/reach_core.hpp"
#include "hgcommon/namespace.hpp"
// causal_graph.cpp - Implementation of CausalGraph class

#include "hypergraph/causal_graph.hpp"
#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "hypergraph/scratch_alloc.hpp"

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// Edge Causal Tracking
// =============================================================================

LockFreeList<EventId>* CausalGraph::get_or_create_edge_producers(CanonicalEdgeKey edge_key) {
    if (key_is_edge_id(edge_key))
        return &edge_producers_by_id_->slot(static_cast<uint32_t>(edge_key.value), *arena_);
    auto result = edge_producers_.lookup(edge_key.value);
    if (result.has_value()) return *result;
    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = edge_producers_.insert_if_absent(edge_key.value, new_list);
    return inserted ? new_list : existing;
}

LockFreeList<EventId>* CausalGraph::get_or_create_edge_consumers(CanonicalEdgeKey edge_key) {
    if (key_is_edge_id(edge_key))
        return &edge_consumers_by_id_->slot(static_cast<uint32_t>(edge_key.value), *arena_);
    auto result = edge_consumers_.lookup(edge_key.value);
    if (result.has_value()) return *result;
    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = edge_consumers_.insert_if_absent(edge_key.value, new_list);
    return inserted ? new_list : existing;
}

bool CausalGraph::is_reachable(EventId producer, EventId consumer) const {
    // Full capture mints an event after the events that produced its inputs, so ids increase
    // along every causal edge and the search is pruned to ids above the producer.
    auto ctx = make_scratch_reach_ctx([this](uint32_t x, auto&& f) {
        if (const LockFreeList<EventId>* pl = preds_.find(x)) pl->for_each(f);
    });
    return hgcommon::reach_backward(ctx, producer, consumer, true);
}

LockFreeList<EventId>* CausalGraph::get_or_create_state_events(StateId state) {
    const uint64_t key = id_key(state);

    auto result = state_events_.lookup(key);
    if (result.has_value()) {
        return *result;
    }

    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = state_events_.insert_if_absent(key, new_list);
    return inserted ? new_list : existing;
}

LockFreeList<EventId>* CausalGraph::get_or_create_state_edge_events(StateId state, EdgeId edge) {
    const uint64_t key = id_key(state, edge);

    auto result = state_edge_events_.lookup(key);
    if (result.has_value()) {
        return *result;
    }

    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = state_edge_events_.insert_if_absent(key, new_list);
    return inserted ? new_list : existing;
}

bool CausalGraph::set_edge_producer(CanonicalEdgeKey edge_key, EventId producer, EdgeId raw_edge) {
    LockFreeList<EventId>* producers = get_or_create_edge_producers(edge_key);
    LockFreeList<EventId>* consumers = get_or_create_edge_consumers(edge_key);

    // Producer side of the symmetric rendezvous: publish self into the producer set,
    // then scan the consumer set. The producer set is append-only and de-duplicated by
    // the (producer,consumer) triple dedup downstream, so re-adding the same producer is
    // harmless; we report novelty for callers/tests that care.
    bool newly_added = true;
    producers->for_each([&](EventId p) { if (p == producer) newly_added = false; });

    // Producer side of the symmetric rendezvous, and the barrier between the halves is what
    // makes at least one side see the other for every (producer, consumer) pair.
    hgcommon::rendezvous<hgcommon::rv::EdgeProducerConsumer>(
        [&] { producers->push(producer, *arena_); },
        [&] { consumers->for_each([&](EventId consumer) {
                  HG_STAT(producer_side_emissions_.fetch_add(1, std::memory_order_relaxed));
                  add_causal_edge(producer, consumer, raw_edge);
              }); });

    return newly_added;
}

void CausalGraph::consume_edges(const CanonicalEdgeKey* keys, const EdgeId* raw_edges,
                                uint8_t n, EventId consumer) {
    // ONE READ decides the order AND supplies the edges. The online transitive reduction is
    // exact only if every in-edge of an event arrives from that event's own thread in
    // DESCENDING producer id: a closer producer's edge recorded first is what lets a farther
    // producer's edge be found redundant. Deciding the order from one lookup and emitting from
    // a second read of the same concurrently growing map let the two disagree -- an edge
    // ordered as having no producer, then emitted with one, arrived after everything placed
    // before it. Measured: exactly one such edge per firing of CausalTrExactnessTest, and one
    // surplus edge kept.
    struct InEdge { EventId producer; uint8_t idx; };
    InEdge in_edges[hgcommon::MAX_PATTERN_EDGES * hgcommon::MAX_IN_EDGE_PRODUCERS];
    uint32_t count = 0;
    for (uint8_t i = 0; i < n; ++i) {
        LockFreeList<EventId>* producers = get_or_create_edge_producers(keys[i]);
        LockFreeList<EventId>* consumers = get_or_create_edge_consumers(keys[i]);
        // Consumer side of the symmetric rendezvous: publish self, then read the producers.
        uint32_t met = 0;
        hgcommon::rendezvous<hgcommon::rv::EdgeProducerConsumer>(
            [&] { consumers->push(consumer, *arena_); },
            [&] { producers->for_each([&](EventId producer) {
                      if (met < hgcommon::MAX_IN_EDGE_PRODUCERS) in_edges[count++] = InEdge{producer, i};
                      ++met;
                  }); });
        // A raw edge has exactly one producer; a canonical edge orbit under quotient can have
        // several. Past the bound the surplus producers' edges are not recorded, and that is
        // counted as the capacity overflow it is rather than treated as a run that found less.
        if (met > hgcommon::MAX_IN_EDGE_PRODUCERS)
            HG_STAT(in_edge_producers_truncated_.fetch_add(1, std::memory_order_relaxed));
    }
    std::sort(in_edges, in_edges + count, [](const InEdge& a, const InEdge& b) {
#if defined(HG_CALIBRATE_IN_EDGE_ORDER_ASCENDING)
        // THE DEFECT, reinstated for verification/genmc/causal_in_edge_order.cpp: the farther
        // producer's edge is judged before the closer one's is present, so the path that would
        // make it redundant is not there yet and it is kept for good.
        if (a.producer != b.producer) return a.producer < b.producer;
#else
        if (a.producer != b.producer) return a.producer > b.producer;
#endif
        return a.idx < b.idx;
    });
    for (uint32_t k = 0; k < count; ++k)
        add_causal_edge(in_edges[k].producer, consumer, raw_edges[in_edges[k].idx]);
}



// =============================================================================
// Branchial Tracking
// =============================================================================

// =============================================================================
// Graph Access
// =============================================================================

void CausalGraph::add_causal_edge(EventId producer, EventId consumer, EdgeId edge) {
    if (transitive_reduction_enabled_.load(std::memory_order_relaxed)) {
        const uint64_t pair_key = causal_pair_key(producer, consumer);
        if (!seen_causal_event_pairs_.contains(pair_key)) {
            // HG_CALIBRATE_TR_NEVER_SKIP answers the redundancy question "no" every time, so
            // every offered pair is kept and the kept set is the full relation rather than its
            // reduction. The determinism gate's per-run tr_surplus check must then fail; a gate
            // that stays green under it is not testing the reduction.
#if defined(HG_CALIBRATE_TR_NEVER_SKIP)
            if (false) {
#else
            if (is_reachable(producer, consumer)) {
#endif
                HG_STAT(num_redundant_edges_skipped_.fetch_add(1, std::memory_order_relaxed));
                return;
            }
        }
    }

    uint64_t triple_key = hgcommon::FNV_OFFSET;
    triple_key = hgcommon::fnv_hash(triple_key, producer);
    triple_key = hgcommon::fnv_hash(triple_key, consumer);
    triple_key = hgcommon::fnv_hash(triple_key, edge);
    // The set reserves 0 and ~0 and refuses either from a worker thread.
    triple_key = hgcommon::avoid_reserved_keys(triple_key);

    if (seen_causal_triples_.insert(triple_key)) {
        causal_edges_[list_worker()].list.push(CausalEdge(producer, consumer, edge), *arena_);
        count_one(&CountSlot::causal_edges);

#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        VIZ_EMIT_CAUSAL_EDGE(producer, consumer, edge);
#endif

        const uint64_t pair_key = causal_pair_key(producer, consumer);
        if (seen_causal_event_pairs_.insert(pair_key)) {
            count_one(&CountSlot::causal_event_pairs);
            // Record the kept edge in the reduced adjacency once per unique event
            // pair, so preds_ holds no duplicate producers for a consumer.
            if (transitive_reduction_enabled_.load(std::memory_order_relaxed))
                record_reduced_edge(producer, consumer);
        }
    }
}

void CausalGraph::record_reduced_edge(EventId producer, EventId consumer) {
    // preds_[consumer] is written only by consumer's own thread (invariant 1) and
    // this runs once per unique event pair, so it holds no duplicate producers.
    preds_.slot(consumer, *arena_).push(producer, *arena_);
}

void CausalGraph::add_branchial_edge(EventId e1, EventId e2, EdgeId shared) {
    branchial_edges_[list_worker()].list.push(BranchialEdge(e1, e2, shared), *arena_);
    count_one(&CountSlot::branchial_edges);

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


// =============================================================================
// Construction, configuration and counters
// =============================================================================

uint64_t CausalGraph::causal_pair_key(EventId producer, EventId consumer) {
    return id_key(producer, consumer);
}

CausalGraph::CausalGraph() : arena_(nullptr) {}

CausalGraph::CausalGraph(ConcurrentHeterogeneousArena* arena) : arena_(arena) {}

void CausalGraph::set_transitive_reduction(bool enabled) {
    transitive_reduction_enabled_.store(enabled, std::memory_order_relaxed);
}

bool CausalGraph::transitive_reduction_enabled() const {
    return transitive_reduction_enabled_.load(std::memory_order_relaxed);
}

void CausalGraph::set_arena(ConcurrentHeterogeneousArena* arena) {
    arena_ = arena;
    seen_causal_triples_.set_arena(arena);
    seen_causal_event_pairs_.set_arena(arena);
    seen_branchial_pairs_.set_arena(arena);
    state_events_.set_arena(arena);
    state_edge_events_.set_arena(arena);
    edge_producers_.set_arena(arena);
    edge_consumers_.set_arena(arena);
}

// =============================================================================
// Branchial recording
// =============================================================================

void CausalGraph::record_state_event(EventId event, StateId input_state) {
    get_or_create_state_events(input_state)->push(event, *arena_);
}

void CausalGraph::record_branchial_overlaps(
    EventId event,
    StateId input_state,
    const EdgeId* consumed_edges,
    uint8_t num_consumed
) {
    // Inverted index: for each consumed edge, publish this event into that edge's
    // co-consumer bucket, then scan the same bucket. Per bucket this is
    // "add first, then check", so both events of a pair see each other (whichever
    // scans the shared bucket second finds the first); seen_branchial_pairs_
    // dedups the double add. Work is proportional to the actual number of
    // co-consumers, replacing the O(events^2) pairwise scan of the whole state's
    // event list (one bucket lookup per consumed edge, not two).
    for (uint8_t i = 0; i < num_consumed; ++i) {
        EdgeId shared = consumed_edges[i];
        LockFreeList<EventId>* bucket = get_or_create_state_edge_events(input_state, shared);
        bucket->push(event, *arena_);
        bucket->for_each([&](EventId other_event) {
            if (other_event == event) return;  // Skip self
            EventId e1 = std::min(event, other_event);
            EventId e2 = std::max(event, other_event);
            if (seen_branchial_pairs_.insert(id_key(e1, e2))) {
                add_branchial_edge(e1, e2, shared);
            }
        });
    }
}

// =============================================================================
// Statistics
// =============================================================================

size_t CausalGraph::num_causal_edges() const {
    return count_total(&CountSlot::causal_edges);
}

size_t CausalGraph::num_causal_event_pairs() const {
    return count_total(&CountSlot::causal_event_pairs);
}

size_t CausalGraph::num_branchial_pairs_claimed() const {
    return seen_branchial_pairs_.count_enumerated();
}

size_t CausalGraph::num_branchial_edges() const {
    return count_total(&CountSlot::branchial_edges);
}

#if HG_ENGINE_STATS
size_t CausalGraph::num_redundant_edges_skipped() const {
    return num_redundant_edges_skipped_.load(std::memory_order_relaxed);
}
#endif

}  // namespace engine
}  // namespace HG_NAMESPACE
