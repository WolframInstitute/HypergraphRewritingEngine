#pragma once

#include "viz_events.hpp"
#include "mpsc_ring_buffer.hpp"

namespace viz {

// Global event sink for visualization
// The engine calls emit_* functions; visualization consumes from the buffer.
// When HYPERGRAPH_ENABLE_VISUALIZATION is not defined, all functions compile to no-ops.
class VizEventSink {
public:
    // Set the event buffer that will receive events
    // Must be called before evolution starts
    static void set_buffer(MPSCRingBuffer<VizEvent>* buffer) {
        buffer_ = buffer;
    }

    // Clear the buffer (disconnect visualization)
    static void clear_buffer() {
        buffer_ = nullptr;
    }

    // Check if visualization is active
    static bool is_active() {
        return buffer_ != nullptr;
    }

    // Get the current buffer (for testing/diagnostics)
    static MPSCRingBuffer<VizEvent>* get_buffer() {
        return buffer_;
    }

    // Emit a state created event
    static void emit_state_created(uint64_t state_id, uint64_t parent_id,
                                   uint32_t generation, uint32_t edge_count,
                                   uint32_t vertex_count) {
        if (buffer_) {
            auto event = VizEvent::make_state_created(state_id, parent_id,
                                                      generation, edge_count, vertex_count);
            buffer_->try_push(event);  // Drop on overflow
        }
    }

    // Emit a hyperedge data event (streams individual edges after StateCreated)
    static void emit_hyperedge(uint64_t state_id, uint32_t edge_index,
                               const uint32_t* vertices, uint8_t vertex_count) {
        if (buffer_) {
            auto event = VizEvent::make_hyperedge(state_id, edge_index, vertices, vertex_count);
            buffer_->try_push(event);
        }
    }

    // Emit a match found event
    static void emit_match_found(uint64_t state_id, uint32_t rule_index,
                                 const uint32_t* matched_edges, uint32_t edge_count) {
        if (buffer_) {
            VizEvent event{};
            event.type = VizEventType::MatchFound;
            event.data.match_found.state_id = state_id;
            event.data.match_found.rule_index = rule_index;
            event.data.match_found.edge_count = edge_count;
            // Copy edge indices (up to MAX_EVENT_EDGES)
            uint32_t copy_count = edge_count < MAX_EVENT_EDGES ? edge_count : MAX_EVENT_EDGES;
            for (uint32_t i = 0; i < copy_count; ++i) {
                event.data.match_found.matched_edges[i] = matched_edges[i];
            }
            buffer_->try_push(event);
        }
    }

    // Emit a rewrite applied event
    static void emit_rewrite_applied(uint64_t source_state_id, uint64_t target_state_id,
                                     uint32_t rule_index, uint32_t event_id,
                                     uint32_t canonical_event_id,
                                     uint32_t destroyed_edges, uint32_t created_edges) {
        if (buffer_) {
            auto event = VizEvent::make_rewrite_applied(source_state_id, target_state_id,
                                                        rule_index, event_id,
                                                        canonical_event_id,
                                                        destroyed_edges, created_edges);
            buffer_->try_push(event);
        }
    }

    // Emit a causal edge event
    static void emit_causal_edge(uint64_t producer_event_id, uint64_t consumer_event_id,
                                 uint32_t edge_id) {
        if (buffer_) {
            auto event = VizEvent::make_causal_edge(producer_event_id, consumer_event_id, edge_id);
            buffer_->try_push(event);
        }
    }

    // Emit a branchial edge event
    // Parameters are EVENT IDs, not state IDs - branchial edges connect events that share consumed edges
    static void emit_branchial_edge(uint64_t event_a_id, uint64_t event_b_id, uint32_t generation) {
        if (buffer_) {
            auto event = VizEvent::make_branchial_edge(event_a_id, event_b_id, generation);
            buffer_->try_push(event);
        }
    }

    // Emit evolution complete event
    static void emit_evolution_complete(uint64_t total_states, uint64_t total_events,
                                        uint32_t max_generation, uint32_t final_state_count) {
        if (buffer_) {
            auto event = VizEvent::make_evolution_complete(total_states, total_events,
                                                           max_generation, final_state_count);
            buffer_->try_push(event);
        }
    }

    // Statistics
    static uint64_t events_emitted() { return events_emitted_; }
    static uint64_t events_dropped() { return events_dropped_; }

private:
    static inline MPSCRingBuffer<VizEvent>* buffer_ = nullptr;
    static inline std::atomic<uint64_t> events_emitted_{0};
    static inline std::atomic<uint64_t> events_dropped_{0};
};

} // namespace viz

// Convenience macros that compile to no-ops when visualization is disabled
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION

#define VIZ_EMIT_STATE_CREATED(state_id, parent_id, gen, edges, verts) \
    ::viz::VizEventSink::emit_state_created(state_id, parent_id, gen, edges, verts)

#define VIZ_EMIT_HYPEREDGE(state_id, edge_idx, vertices, vertex_count) \
    ::viz::VizEventSink::emit_hyperedge(state_id, edge_idx, vertices, vertex_count)

#define VIZ_EMIT_MATCH_FOUND(state_id, rule_idx, edges, count) \
    ::viz::VizEventSink::emit_match_found(state_id, rule_idx, edges, count)

#define VIZ_EMIT_REWRITE_APPLIED(src_state, tgt_state, rule_idx, event_id, canonical_event_id, destroyed, created) \
    ::viz::VizEventSink::emit_rewrite_applied(src_state, tgt_state, rule_idx, event_id, canonical_event_id, destroyed, created)

#define VIZ_EMIT_CAUSAL_EDGE(producer, consumer, edge_id) \
    ::viz::VizEventSink::emit_causal_edge(producer, consumer, edge_id)

#define VIZ_EMIT_BRANCHIAL_EDGE(event_a, event_b, gen) \
    ::viz::VizEventSink::emit_branchial_edge(event_a, event_b, gen)

#define VIZ_EMIT_EVOLUTION_COMPLETE(total_states, total_events, max_gen, final_count) \
    ::viz::VizEventSink::emit_evolution_complete(total_states, total_events, max_gen, final_count)

#define VIZ_IS_ACTIVE() ::viz::VizEventSink::is_active()

#else

#define VIZ_EMIT_STATE_CREATED(state_id, parent_id, gen, edges, verts) ((void)0)
#define VIZ_EMIT_HYPEREDGE(state_id, edge_idx, vertices, vertex_count) ((void)0)
#define VIZ_EMIT_MATCH_FOUND(state_id, rule_idx, edges, count) ((void)0)
#define VIZ_EMIT_REWRITE_APPLIED(src_state, tgt_state, rule_idx, event_id, canonical_event_id, destroyed, created) ((void)0)
#define VIZ_EMIT_CAUSAL_EDGE(producer, consumer, edge_id) ((void)0)
#define VIZ_EMIT_BRANCHIAL_EDGE(event_a, event_b, gen) ((void)0)
#define VIZ_EMIT_EVOLUTION_COMPLETE(total_states, total_events, max_gen, final_count) ((void)0)
#define VIZ_IS_ACTIVE() false

#endif
