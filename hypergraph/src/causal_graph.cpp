#include "hgcommon/transitive_reduction.hpp"
#include "hgcommon/namespace.hpp"
// causal_graph.cpp - Implementation of CausalGraph class

#include "hypergraph/causal_graph.hpp"
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
    auto result = edge_producers_.lookup(edge_key.value);
    if (result.has_value()) return *result;
    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = edge_producers_.insert_if_absent(edge_key.value, new_list);
    return inserted ? new_list : existing;
}

LockFreeList<EventId>* CausalGraph::get_or_create_edge_consumers(CanonicalEdgeKey edge_key) {
    auto result = edge_consumers_.lookup(edge_key.value);
    if (result.has_value()) return *result;
    auto* new_list = arena_->template create<LockFreeList<EventId>>();
    auto [existing, inserted] = edge_consumers_.insert_if_absent(edge_key.value, new_list);
    return inserted ? new_list : existing;
}

bool CausalGraph::is_reachable(EventId producer, EventId consumer) const {
    if (producer == consumer) return true;
    // Both shortcuts below hold only while ids increase along every causal edge. When they do,
    // an ancestor's id is strictly smaller than its descendant's, so a producer with id >= its
    // consumer's cannot reach it and any node with id < producer's is out of the cone. The
    // quotient reconstruction emits between canonical ids, which are not monotonic, and there
    // the walk must run unpruned or it misses paths that exist.
    const bool topo = ids_are_topological_.load(std::memory_order_relaxed);
    if (topo && producer >= consumer) return false;

    // Backward BFS from consumer over the reduced predecessor adjacency, searching
    // for producer and pruning to ids >= producer. Scratch lives in the calling
    // worker's arena (bulk-reclaimed per task).
    SVec<EventId> stack;
    ScratchIdSet visited;
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
            if ((!topo || q > producer) && visited.insert(q)) stack.push_back(q);
        });
        if (found) return true;
    }
    return false;
}

// The reduction of the STORED relation. A pair (p,c) is redundant iff c is reachable from p by
// a path of length >= 2 in that relation. The relation is a set and a DAG's transitive
// reduction is unique, so this answer does not depend on the schedule that produced it -- which
// is the whole reason it exists: the incremental rule needs an arrival discipline the quotient
// reconstruction cannot provide.
std::vector<std::pair<EventId, EventId>> CausalGraph::reduced_pairs() const {
    std::vector<std::pair<EventId, EventId>> kept;
    hgcommon::tr_reduce(
        [&](auto&& add) {
            causal_edges_.for_each([&](const CausalEdge& e) { add(e.producer, e.consumer); });
        },
        [&](uint32_t p, uint32_t c) { kept.emplace_back(p, c); });
    return kept;
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
    producers->push(producer, *arena_);

    // StoreLoad barrier: pushing self (store to producers.head_) must be ordered before
    // scanning consumers (load of consumers.head_). Paired with the identical fence in
    // add_edge_consumer this makes the handshake sequentially consistent, so at least
    // one side sees the other for every (producer,consumer) pair -- no store-buffer miss.
    std::atomic_thread_fence(std::memory_order_seq_cst);

    consumers->for_each([&](EventId consumer) {
        add_causal_edge(producer, consumer, raw_edge);
    });

    return newly_added;
}

void CausalGraph::add_edge_consumer(CanonicalEdgeKey edge_key, EventId consumer, EdgeId raw_edge) {
    LockFreeList<EventId>* producers = get_or_create_edge_producers(edge_key);
    LockFreeList<EventId>* consumers = get_or_create_edge_consumers(edge_key);

    // Consumer side of the symmetric rendezvous: publish self into the consumer set,
    // then scan the producer set and emit an edge from every producer.
    consumers->push(consumer, *arena_);

    // StoreLoad barrier, mirror of set_edge_producer -- see there.
    std::atomic_thread_fence(std::memory_order_seq_cst);

    producers->for_each([&](EventId producer) {
        add_causal_edge(producer, consumer, raw_edge);
    });
}

void CausalGraph::propagate_producers(CanonicalEdgeKey from_key, CanonicalEdgeKey to_key,
                                      EdgeId raw_edge) {
    // A surviving edge carries its producers across a rewrite: whoever produced the
    // parent orbit (from_key) also "produces" the same edge in the child orbit (to_key),
    // because it is literally the same edge instance passing through. Register each as a
    // producer of to_key so a downstream consumer of that orbit rendezvous with the
    // edge's original creators, not just events that freshly produced into the child.
    // Reuses set_edge_producer so the rendezvous + (producer,consumer) dedup are shared.
    auto result = edge_producers_.lookup(from_key.value);
    if (!result.has_value()) return;
    (*result)->for_each([&](EventId p) {
        set_edge_producer(to_key, p, raw_edge);
    });
}

EventId CausalGraph::get_edge_producer(CanonicalEdgeKey edge_key) const {
    // A key with no producer set materialized has no producer.
    auto result = edge_producers_.lookup(edge_key.value);
    if (!result.has_value()) return INVALID_ID;
    // The largest producer id in the set is the closest (latest) producer; returning it
    // is a deterministic function of the order-independent set. INVALID_ID if empty.
    EventId best = INVALID_ID;
    (*result)->for_each([&](EventId p) {
        if (best == INVALID_ID || p > best) best = p;
    });
    return best;
}

// =============================================================================
// Branchial Tracking
// =============================================================================

// =============================================================================
// Graph Access
// =============================================================================

void CausalGraph::add_causal_edge(EventId producer, EventId consumer, EdgeId edge) {
    // Incremental reduction only where its preconditions hold. Where they do not, every pair is
    // stored and the reduction is computed on read, which is exact and schedule-independent
    // because the stored relation is a set and a DAG's transitive reduction is unique.
    if (transitive_reduction_enabled_.load(std::memory_order_relaxed) &&
        ids_are_topological_.load(std::memory_order_relaxed)) {
        const uint64_t pair_key = causal_pair_key(producer, consumer);
        if (!seen_causal_event_pairs_.contains(pair_key)) {
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
    if (triple_key == 0) triple_key = 1;   // never the set's EMPTY sentinel

    if (seen_causal_triples_.insert(triple_key)) {
        causal_edges_.push(CausalEdge(producer, consumer, edge), *arena_);
        num_causal_edges_.fetch_add(1, std::memory_order_relaxed);

#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        VIZ_EMIT_CAUSAL_EDGE(producer, consumer, edge);
#endif

        const uint64_t pair_key = causal_pair_key(producer, consumer);
        if (seen_causal_event_pairs_.insert(pair_key)) {
            num_causal_event_pairs_.fetch_add(1, std::memory_order_relaxed);
            // Record the kept edge in the reduced adjacency once per unique event
            // pair, so preds_ holds no duplicate producers for a consumer.
            if (transitive_reduction_enabled_.load(std::memory_order_relaxed) &&
                ids_are_topological_.load(std::memory_order_relaxed)) {
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

bool CausalGraph::reduces_on_read() const {
    return transitive_reduction_enabled_.load(std::memory_order_relaxed) &&
           !ids_are_topological_.load(std::memory_order_relaxed);
}

void CausalGraph::set_ids_are_topological(bool on) {
    ids_are_topological_.store(on, std::memory_order_relaxed);
}

bool CausalGraph::ids_are_topological() const {
    return ids_are_topological_.load(std::memory_order_relaxed);
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
    // Reducing on read means the stored relation is the FULL one, so the live edge count is
    // what the filtered iteration yields, not the admitted-triple counter.
    if (reduces_on_read()) {
        size_t n = 0;
        for_each_causal_edge([&](const CausalEdge&) { ++n; });
        return n;
    }
    return num_causal_edges_.load(std::memory_order_relaxed);
}

size_t CausalGraph::num_causal_event_pairs() const {
    if (reduces_on_read()) return reduced_pairs().size();
    return num_causal_event_pairs_.load(std::memory_order_relaxed);
}

size_t CausalGraph::num_branchial_pairs_claimed() const {
    return seen_branchial_pairs_.count_enumerated();
}

size_t CausalGraph::num_branchial_edges() const {
    return num_branchial_edges_.load(std::memory_order_relaxed);
}

size_t CausalGraph::num_redundant_edges_skipped() const {
    return num_redundant_edges_skipped_.load(std::memory_order_relaxed);
}

}  // namespace engine
}  // namespace HG_NAMESPACE
