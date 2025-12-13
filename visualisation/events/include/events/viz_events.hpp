#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

namespace viz {

// Event types emitted by the hypergraph rewriting engine
enum class VizEventType : uint8_t {
    StateCreated,      // New state created from rewrite
    HyperedgeData,     // Hyperedge data for a state (sent after StateCreated)
    MatchFound,        // Pattern match found in a state
    RewriteApplied,    // Rewrite rule applied
    CausalEdge,        // Causal edge between events
    BranchialEdge,     // Branchial edge between states
    EvolutionComplete  // Evolution finished
};

// Maximum edges we can store inline in an event
static constexpr size_t MAX_EVENT_EDGES = 8;

// Sentinel value for "no source state" (genesis events)
static constexpr uint64_t VIZ_NO_SOURCE_STATE = UINT64_MAX;

// State creation event data
struct StateCreatedData {
    uint64_t state_id;           // Canonical hash of the state
    uint64_t parent_state_id;    // Parent state (0 if root)
    uint32_t generation;         // Evolution step number
    uint32_t edge_count;         // Number of edges in this state
    uint32_t vertex_count;       // Number of vertices
    uint8_t  padding[4];
};
static_assert(sizeof(StateCreatedData) == 32, "StateCreatedData must be 32 bytes");

// Maximum vertices per hyperedge in an event (fits in 48 bytes with overhead)
static constexpr size_t MAX_HYPEREDGE_VERTICES = 8;

// Hyperedge data event - streams individual edges after StateCreated
struct HyperedgeData {
    uint64_t state_id;           // Which state this edge belongs to
    uint32_t edge_index;         // Edge index within the state
    uint8_t  vertex_count;       // Number of vertices in this edge
    uint8_t  padding[3];
    uint32_t vertices[MAX_HYPEREDGE_VERTICES];  // Vertex IDs
};
static_assert(sizeof(HyperedgeData) == 48, "HyperedgeData must be 48 bytes");

// Match found event data
struct MatchFoundData {
    uint64_t state_id;           // State where match was found
    uint32_t rule_index;         // Which rule matched
    uint32_t edge_count;         // Number of matched edges
    uint32_t matched_edges[MAX_EVENT_EDGES]; // Edge indices that matched
};
static_assert(sizeof(MatchFoundData) <= 48, "MatchFoundData too large");

// Rewrite applied event data
struct RewriteAppliedData {
    uint64_t source_state_id;    // State before rewrite
    uint64_t target_state_id;    // State after rewrite (canonical)
    uint32_t rule_index;         // Which rule was applied
    uint32_t event_id;           // Raw event identifier (for tracking)
    uint32_t canonical_event_id; // Canonical event ID (for deduplication)
    uint16_t destroyed_edges;    // Number of edges destroyed
    uint16_t created_edges;      // Number of edges created
};
static_assert(sizeof(RewriteAppliedData) == 32, "RewriteAppliedData must be 32 bytes");

// Causal edge event data
struct CausalEdgeData {
    uint64_t producer_event_id;  // Event that produced the edge
    uint64_t consumer_event_id;  // Event that consumed the edge
    uint32_t edge_id;            // The hyperedge connecting them
    uint8_t  padding[4];
};
static_assert(sizeof(CausalEdgeData) == 24, "CausalEdgeData must be 24 bytes");

// Branchial edge event data
struct BranchialEdgeData {
    uint64_t event_a_id;         // First event
    uint64_t event_b_id;         // Second event
    uint32_t generation;         // Evolution step
    uint8_t  padding[4];
};
static_assert(sizeof(BranchialEdgeData) == 24, "BranchialEdgeData must be 24 bytes");

// Evolution complete event data
struct EvolutionCompleteData {
    uint64_t total_states;       // Total states generated
    uint64_t total_events;       // Total rewrite events
    uint32_t max_generation;     // Deepest generation reached
    uint32_t final_state_count;  // Number of terminal states
};
static_assert(sizeof(EvolutionCompleteData) == 24, "EvolutionCompleteData must be 24 bytes");

// Union of all event data types - keep it compact
union VizEventData {
    StateCreatedData   state_created;
    HyperedgeData      hyperedge;
    MatchFoundData     match_found;
    RewriteAppliedData rewrite_applied;
    CausalEdgeData     causal_edge;
    BranchialEdgeData  branchial_edge;
    EvolutionCompleteData evolution_complete;
    uint8_t raw[48];  // Ensure minimum size for union
};

// Main event structure - POD, cache-line friendly
struct alignas(64) VizEvent {
    VizEventType type;
    uint8_t      reserved[7];    // Padding for alignment
    uint64_t     timestamp;      // Monotonic timestamp (nanoseconds or tick count)
    VizEventData data;

    // Factory methods for creating events
    static VizEvent make_state_created(uint64_t state_id, uint64_t parent_id,
                                       uint32_t generation, uint32_t edge_count,
                                       uint32_t vertex_count) {
        VizEvent e{};
        e.type = VizEventType::StateCreated;
        e.timestamp = get_timestamp();
        e.data.state_created.state_id = state_id;
        e.data.state_created.parent_state_id = parent_id;
        e.data.state_created.generation = generation;
        e.data.state_created.edge_count = edge_count;
        e.data.state_created.vertex_count = vertex_count;
        return e;
    }

    static VizEvent make_hyperedge(uint64_t state_id, uint32_t edge_index,
                                   const uint32_t* vertices, uint8_t vertex_count) {
        VizEvent e{};
        e.type = VizEventType::HyperedgeData;
        e.timestamp = get_timestamp();
        e.data.hyperedge.state_id = state_id;
        e.data.hyperedge.edge_index = edge_index;
        e.data.hyperedge.vertex_count = vertex_count;
        uint8_t count = vertex_count < MAX_HYPEREDGE_VERTICES ? vertex_count : MAX_HYPEREDGE_VERTICES;
        for (uint8_t i = 0; i < count; ++i) {
            e.data.hyperedge.vertices[i] = vertices[i];
        }
        return e;
    }

    static VizEvent make_rewrite_applied(uint64_t source_id, uint64_t target_id,
                                         uint32_t rule_index, uint32_t event_id,
                                         uint32_t canonical_event_id,
                                         uint32_t destroyed, uint32_t created) {
        VizEvent e{};
        e.type = VizEventType::RewriteApplied;
        e.timestamp = get_timestamp();
        e.data.rewrite_applied.source_state_id = source_id;
        e.data.rewrite_applied.target_state_id = target_id;
        e.data.rewrite_applied.rule_index = rule_index;
        e.data.rewrite_applied.event_id = event_id;
        e.data.rewrite_applied.canonical_event_id = canonical_event_id;
        e.data.rewrite_applied.destroyed_edges = static_cast<uint16_t>(destroyed);
        e.data.rewrite_applied.created_edges = static_cast<uint16_t>(created);
        return e;
    }

    static VizEvent make_causal_edge(uint64_t producer, uint64_t consumer, uint32_t edge_id) {
        VizEvent e{};
        e.type = VizEventType::CausalEdge;
        e.timestamp = get_timestamp();
        e.data.causal_edge.producer_event_id = producer;
        e.data.causal_edge.consumer_event_id = consumer;
        e.data.causal_edge.edge_id = edge_id;
        return e;
    }

    static VizEvent make_branchial_edge(uint64_t event_a, uint64_t event_b, uint32_t gen) {
        VizEvent e{};
        e.type = VizEventType::BranchialEdge;
        e.timestamp = get_timestamp();
        e.data.branchial_edge.event_a_id = event_a;
        e.data.branchial_edge.event_b_id = event_b;
        e.data.branchial_edge.generation = gen;
        return e;
    }

    static VizEvent make_evolution_complete(uint64_t total_states, uint64_t total_events,
                                            uint32_t max_gen, uint32_t final_count) {
        VizEvent e{};
        e.type = VizEventType::EvolutionComplete;
        e.timestamp = get_timestamp();
        e.data.evolution_complete.total_states = total_states;
        e.data.evolution_complete.total_events = total_events;
        e.data.evolution_complete.max_generation = max_gen;
        e.data.evolution_complete.final_state_count = final_count;
        return e;
    }

private:
    static uint64_t get_timestamp() {
        // Simple monotonic counter - could be replaced with rdtsc or clock_gettime
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
};

static_assert(sizeof(VizEvent) == 64, "VizEvent must be 64 bytes (one cache line)");
static_assert(alignof(VizEvent) == 64, "VizEvent must be cache-line aligned");

} // namespace viz
