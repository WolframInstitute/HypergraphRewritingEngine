#include "hypergraph/types.hpp"

namespace HG_NAMESPACE {
namespace engine {

const char* AbortedException::what() const noexcept { return "Operation aborted"; }

// =============================================================================
// VariableBinding
// =============================================================================

VariableBinding::VariableBinding() : bound_mask(0) {
    std::memset(bindings, 0xFF, sizeof(bindings));  // Initialize to INVALID_ID
}

bool VariableBinding::is_bound(uint8_t var_index) const {
    return (bound_mask & (1u << var_index)) != 0;
}

VertexId VariableBinding::get(uint8_t var_index) const {
    return bindings[var_index];
}

void VariableBinding::bind(uint8_t var_index, VertexId vertex) {
    bindings[var_index] = vertex;
    bound_mask |= (1u << var_index);
}

void VariableBinding::unbind(uint8_t var_index) {
    bindings[var_index] = INVALID_ID;
    bound_mask &= ~(1u << var_index);
}

bool VariableBinding::empty() const {
    return bound_mask == 0;
}

uint8_t VariableBinding::count() const {
    return static_cast<uint8_t>(hgcommon::popcount(bound_mask));
}

// =============================================================================
// Edge
// =============================================================================

Edge::Edge(EdgeId id_, const VertexId* src, uint8_t arity_, VertexId* spill,
           EventId creator, uint32_t step_)
    : id(id_)
    , arity(arity_)
    , creator_event(creator)
    , step(step_)
{
    VertexId* dst = (arity_ <= INLINE_ARITY) ? inline_vertices : spill;
    for (uint8_t i = 0; i < arity_; ++i) dst[i] = src[i];
    vertices = dst;
}

Edge::Edge()
    : id(INVALID_ID)
    , arity(0)
    , creator_event(INVALID_ID)
    , step(0)
    , vertices(nullptr)
{}

Edge::Edge(const Edge& o)
    : id(o.id)
    , arity(o.arity)
    , creator_event(o.creator_event)
    , step(o.step)
{
    copy_vertices_from(o);
}

Edge& Edge::operator=(const Edge& o) {
    id = o.id;
    arity = o.arity;
    creator_event = o.creator_event;
    step = o.step;
    copy_vertices_from(o);
    return *this;
}

Edge::Edge(Edge&& o) noexcept : Edge(static_cast<const Edge&>(o)) {}

Edge& Edge::operator=(Edge&& o) noexcept { return *this = static_cast<const Edge&>(o); }

void Edge::copy_vertices_from(const Edge& o) {
    if (o.vertices == o.inline_vertices) {
        for (uint8_t i = 0; i < arity; ++i) inline_vertices[i] = o.inline_vertices[i];
        vertices = inline_vertices;
    } else {
        vertices = o.vertices;  // arena-owned, shared
    }
}

// =============================================================================
// Event
// =============================================================================

Event::Event(EventId id_, StateId input, StateId output, RuleIndex rule,
             EdgeId* consumed, uint8_t n_consumed,
             EdgeId* produced, uint8_t n_produced,
             EventId canonical_id)
    : id(id_)
    , input_state(input)
    , output_state(output)
    , rule_index(rule)
    , consumed_edges(consumed)
    , produced_edges(produced)
    , num_consumed(n_consumed)
    , num_produced(n_produced)
    , canonical_event_id(canonical_id)
    , signature(0)
{}

Event::Event()
    : id(INVALID_ID)
    , input_state(INVALID_ID)
    , output_state(INVALID_ID)
    , rule_index(0)
    , consumed_edges(nullptr)
    , produced_edges(nullptr)
    , num_consumed(0)
    , num_produced(0)
    , canonical_event_id(INVALID_ID)
    , signature(0)
{}

bool Event::is_canonical() const { return canonical_event_id == INVALID_ID; }

// =============================================================================
// State
// =============================================================================

State::State(StateId id_, SparseBitset&& edge_set, uint32_t step_,
             uint64_t hash, EventId parent, StateId canonical)
    : id(id_)
    , edges(std::move(edge_set))
    , step(step_)
    , canonical_hash(hash)
    , parent_event(parent)
    , canonical_id(canonical == INVALID_ID ? id_ : canonical)
    , explore_depth(INVALID_ID)
    , expanded(0)
    , vertex_index(nullptr)
    , vertex_index_size(0)
    , delta_edges(nullptr)
    , num_delta_edges(0)
    , parent_state(INVALID_ID)
{}

State::State()
    : id(INVALID_ID)
    , edges()
    , step(0)
    , canonical_hash(0)
    , parent_event(INVALID_ID)
    , canonical_id(INVALID_ID)
    , explore_depth(INVALID_ID)
    , expanded(0)
    , vertex_index(nullptr)
    , vertex_index_size(0)
    , delta_edges(nullptr)
    , num_delta_edges(0)
    , parent_state(INVALID_ID)
{}

State::State(State&& other) noexcept
    : id(other.id)
    , edges(std::move(other.edges))
    , step(other.step)
    , canonical_hash(other.canonical_hash)
    , parent_event(other.parent_event)
    , canonical_id(other.canonical_id)
    , explore_depth(other.explore_depth)
    , expanded(other.expanded)
    , vertex_index(other.vertex_index)
    , vertex_index_size(other.vertex_index_size)
    , delta_edges(other.delta_edges)
    , num_delta_edges(other.num_delta_edges)
    , parent_state(other.parent_state)
{
    other.id = INVALID_ID;
}

State& State::operator=(State&& other) noexcept {
    if (this != &other) {
        id = other.id;
        edges = std::move(other.edges);
        step = other.step;
        canonical_hash = other.canonical_hash;
        parent_event = other.parent_event;
        canonical_id = other.canonical_id;
        explore_depth = other.explore_depth;
        expanded = other.expanded;
        vertex_index = other.vertex_index;
        vertex_index_size = other.vertex_index_size;
        delta_edges = other.delta_edges;
        num_delta_edges = other.num_delta_edges;
        parent_state = other.parent_state;
        other.id = INVALID_ID;
    }
    return *this;
}

// =============================================================================
// GlobalCounters
// =============================================================================

VertexId GlobalCounters::alloc_vertex() {
    return next_vertex.fetch_add(1, std::memory_order_relaxed);
}

EdgeId GlobalCounters::alloc_edge() {
    return next_edge.fetch_add(1, std::memory_order_relaxed);
}

StateId GlobalCounters::alloc_state() {
    return next_state.fetch_add(1, std::memory_order_relaxed);
}

EventId GlobalCounters::alloc_event() {
    return next_event.fetch_add(1, std::memory_order_release);
}

void GlobalCounters::reset() {
    next_vertex.store(0, std::memory_order_relaxed);
    next_edge.store(0, std::memory_order_relaxed);
    next_state.store(0, std::memory_order_relaxed);
    next_event.store(0, std::memory_order_relaxed);
}

// =============================================================================
// CausalEdge / BranchialEdge
// =============================================================================

CausalEdge::CausalEdge(EventId p, EventId c, EdgeId e)
    : producer(p), consumer(c), edge(e) {}

CausalEdge::CausalEdge() : producer(INVALID_ID), consumer(INVALID_ID), edge(INVALID_ID) {}

bool CausalEdge::operator==(const CausalEdge& other) const {
    return producer == other.producer && consumer == other.consumer;
}

BranchialEdge::BranchialEdge(EventId e1, EventId e2, EdgeId se)
    : event1(e1 < e2 ? e1 : e2)
    , event2(e1 < e2 ? e2 : e1)
    , shared_edge(se) {}

BranchialEdge::BranchialEdge()
    : event1(INVALID_ID), event2(INVALID_ID), shared_edge(INVALID_ID) {}

bool BranchialEdge::operator==(const BranchialEdge& other) const {
    return event1 == other.event1 && event2 == other.event2;
}

// =============================================================================
// EventSignature
// =============================================================================

uint64_t EventSignature::hash() const {
    uint64_t h = FNV_OFFSET;
    h = fnv_hash(h, input_state_hash);
    h = fnv_hash(h, output_state_hash);
    h = fnv_hash(h, consumed_edges_sig);
    h = fnv_hash(h, produced_edges_sig);
    return h;
}


// =============================================================================
// quotient_types.hpp
// =============================================================================
//
// types.hpp includes quotient_types.hpp and nothing else does, so the quotient identity
// records' bodies land here rather than in a file of their own.

uint64_t QcEventContent::triple_hash() const {
    return hgcommon::qr_content_hash(from_class, to_class, rule);
}

uint32_t EdgeOrbitTable::index_of(EdgeId e) const {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (edges[mid] < e) lo = mid + 1; else hi = mid;
    }
    return (lo < n && edges[lo] == e) ? lo : n;
}

uint32_t EdgeOrbitTable::orbit_of(EdgeId e) const {
    const uint32_t i = index_of(e);
    return i < n ? orbit[i] : 0;
}

uint32_t EdgeOrbitTable::slot_of(EdgeId e) const {
    const uint32_t i = index_of(e);
    return (i < n && slot) ? slot[i] : 0;
}

uint32_t CanonicalTransition::consumed(uint32_t i) const { return consumed_orbits[i]; }
uint32_t CanonicalTransition::produced(uint32_t i) const { return produced_orbits[i]; }
uint32_t CanonicalTransition::surv_from(uint32_t i) const { return surv_from_orbits[i]; }
uint32_t CanonicalTransition::surv_to(uint32_t i) const { return surv_to_orbits[i]; }

uint32_t SlotMatch::consumed(uint32_t i) const { return consumed_slots[i]; }
uint32_t SlotMatch::produced(uint32_t i) const { return produced_slots[i]; }
uint32_t SlotMatch::surv_from(uint32_t i) const { return surv_from_slot[i]; }
uint32_t SlotMatch::surv_to(uint32_t i) const { return surv_to_slot[i]; }
const uint32_t* SlotMatch::consumed_ptr() const { return consumed_slots; }
const uint32_t* SlotMatch::produced_ptr() const { return produced_slots; }

}  // namespace engine
}  // namespace HG_NAMESPACE
