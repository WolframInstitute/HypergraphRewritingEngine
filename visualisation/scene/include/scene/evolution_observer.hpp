#pragma once

#include "hypergraph_data.hpp"
#include <events/viz_events.hpp>
#include <events/mpsc_ring_buffer.hpp>
#include <events/viz_event_sink.hpp>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <chrono>

namespace viz::scene {

// Callback for when new states are added (for layout updates)
using StateAddedCallback = std::function<void(StateId)>;
using EventAddedCallback = std::function<void(EventId)>;

// Observer that consumes events from the engine and builds Evolution
// Call process_events() each frame to drain pending events
class EvolutionObserver {
public:
    EvolutionObserver() {
        // Connect to the global event sink
        VizEventSink::set_buffer(&event_buffer_);
    }

    ~EvolutionObserver() {
        // Disconnect from event sink
        VizEventSink::clear_buffer();
    }

    // Non-copyable, non-movable (owns the buffer)
    EvolutionObserver(const EvolutionObserver&) = delete;
    EvolutionObserver& operator=(const EvolutionObserver&) = delete;

    // Process all pending events, updating the evolution structure
    // Returns number of events processed
    size_t process_events() {
        return event_buffer_.drain([this](const VizEvent& event) {
            process_event(event);
        });
    }

    // Process up to max_count events (for rate limiting)
    size_t process_events(size_t max_count) {
        return event_buffer_.drain([this](const VizEvent& event) {
            process_event(event);
        }, max_count);
    }

    // Process events until time budget is exhausted (in milliseconds)
    // Returns number of events processed
    size_t process_events_timed(double budget_ms) {
        using clock = std::chrono::steady_clock;
        auto start = clock::now();
        auto budget = std::chrono::duration<double, std::milli>(budget_ms);
        size_t count = 0;

        while (true) {
            VizEvent event;
            if (!event_buffer_.try_pop(event)) {
                break;  // No more events
            }
            process_event(event);
            ++count;

            // Check time budget periodically (every 64 events to reduce overhead)
            if ((count & 63) == 0) {
                if (clock::now() - start >= budget) {
                    break;
                }
            }
        }
        return count;
    }

    // Get the evolution structure (incrementally built)
    Evolution& get_evolution() { return evolution_; }
    const Evolution& get_evolution() const { return evolution_; }

    // Set callbacks for incremental updates
    void set_state_added_callback(StateAddedCallback cb) { on_state_added_ = std::move(cb); }
    void set_event_added_callback(EventAddedCallback cb) { on_event_added_ = std::move(cb); }

    // Check if evolution is complete
    bool is_complete() const { return evolution_complete_; }

    // Get statistics
    uint64_t total_events_processed() const { return total_events_processed_; }
    uint32_t max_generation() const { return max_generation_; }
    size_t pending_causal_count() const { return pending_causal_.size(); }
    size_t pending_branchial_count() const { return pending_branchial_.size(); }

    // Clear and reset for new evolution
    // WARNING: Only call when evolution thread is stopped!
    void reset() {
        // First drain any remaining events in the buffer
        event_buffer_.drain([](const VizEvent&) {});
        // Then reset the buffer itself
        event_buffer_.reset();

        evolution_ = Evolution{};
        engine_state_to_viz_.clear();
        engine_event_to_viz_.clear();
        seen_canonical_events_.clear();
        state_vertex_maps_.clear();
        pending_hypergraphs_.clear();
        pending_rewrites_.clear();
        pending_hyperedges_.clear();
        pending_causal_.clear();
        pending_branchial_.clear();
        evolution_complete_ = false;
        total_events_processed_ = 0;
        max_generation_ = 0;
    }

    // Register a hypergraph for a state (called externally to provide internal structure)
    // This is needed because VizEvent only contains IDs, not full hypergraph data
    void register_hypergraph(uint64_t engine_state_id, const Hypergraph& hg) {
        pending_hypergraphs_[engine_state_id] = hg;
    }

private:
    MPSCRingBuffer<VizEvent> event_buffer_;
    Evolution evolution_;

    // Mapping from engine IDs to visualization IDs
    std::unordered_map<uint64_t, StateId> engine_state_to_viz_;
    std::unordered_map<uint64_t, EventId> engine_event_to_viz_;       // raw event → viz event
    std::unordered_set<uint64_t> seen_canonical_events_;              // canonical event IDs already visualized

    // Pending hypergraphs waiting to be associated with states
    std::unordered_map<uint64_t, Hypergraph> pending_hypergraphs_;

    // Per-state vertex remapping: global engine vertex ID -> local 0-indexed ID
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, VertexId>> state_vertex_maps_;

    // Pending queues for out-of-order events
    // When RewriteApplied arrives before StateCreated, queue it
    std::vector<RewriteAppliedData> pending_rewrites_;
    // When HyperedgeData arrives before StateCreated, queue it
    std::vector<HyperedgeData> pending_hyperedges_;
    // When CausalEdge arrives before both events exist, queue it
    std::vector<CausalEdgeData> pending_causal_;
    // When BranchialEdge arrives before both states exist, queue it
    std::vector<BranchialEdgeData> pending_branchial_;

    // Callbacks
    StateAddedCallback on_state_added_;
    EventAddedCallback on_event_added_;

    // Statistics
    uint64_t total_events_processed_ = 0;
    uint32_t max_generation_ = 0;
    bool evolution_complete_ = false;

    void process_event(const VizEvent& event) {
        ++total_events_processed_;

        switch (event.type) {
            case VizEventType::StateCreated:
                handle_state_created(event.data.state_created);
                break;
            case VizEventType::HyperedgeData:
                handle_hyperedge_data(event.data.hyperedge);
                break;
            case VizEventType::MatchFound:
                handle_match_found(event.data.match_found);
                break;
            case VizEventType::RewriteApplied:
                handle_rewrite_applied(event.data.rewrite_applied);
                break;
            case VizEventType::CausalEdge:
                handle_causal_edge(event.data.causal_edge);
                break;
            case VizEventType::BranchialEdge:
                handle_branchial_edge(event.data.branchial_edge);
                break;
            case VizEventType::EvolutionComplete:
                handle_evolution_complete(event.data.evolution_complete);
                break;
        }
    }

    void handle_state_created(const StateCreatedData& data) {
        // Check if we already have this state
        if (engine_state_to_viz_.count(data.state_id)) {
            return;  // Duplicate event, ignore
        }

        // Create hypergraph - check if we have pending data
        Hypergraph hg;
        auto pending_it = pending_hypergraphs_.find(data.state_id);
        if (pending_it != pending_hypergraphs_.end()) {
            hg = std::move(pending_it->second);
            pending_hypergraphs_.erase(pending_it);
        } else {
            // No detailed hypergraph data - create placeholder
            hg.vertex_count = data.vertex_count;
            // Edges will be empty - can be filled later via register_hypergraph
        }

        bool is_initial = (data.parent_state_id == 0 && data.generation == 0);
        StateId viz_id = evolution_.add_state(hg, is_initial, data.generation);
        engine_state_to_viz_[data.state_id] = viz_id;

        if (data.generation > max_generation_) {
            max_generation_ = data.generation;
        }

        if (on_state_added_) {
            on_state_added_(viz_id);
        }

        // Process any pending events that were waiting for this state
        process_pending_for_state(data.state_id);
    }

    // Try to process pending events that may now be resolvable
    void process_pending_for_state(uint64_t new_state_id) {
        // Process pending hyperedges for this state
        auto it = pending_hyperedges_.begin();
        while (it != pending_hyperedges_.end()) {
            if (it->state_id == new_state_id) {
                handle_hyperedge_data_internal(*it);
                it = pending_hyperedges_.erase(it);
            } else {
                ++it;
            }
        }

        // Try to process pending rewrites that involve this state
        auto rit = pending_rewrites_.begin();
        while (rit != pending_rewrites_.end()) {
            if (try_process_rewrite(*rit)) {
                rit = pending_rewrites_.erase(rit);
            } else {
                ++rit;
            }
        }

        // Try to process pending branchial edges
        auto bit = pending_branchial_.begin();
        while (bit != pending_branchial_.end()) {
            if (try_process_branchial(*bit)) {
                bit = pending_branchial_.erase(bit);
            } else {
                ++bit;
            }
        }
    }

    // Try to process pending causal edges (called after event creation)
    void process_pending_causal_for_event(uint64_t new_event_id) {
        auto it = pending_causal_.begin();
        while (it != pending_causal_.end()) {
            if (try_process_causal(*it)) {
                it = pending_causal_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Try to process pending branchial edges (called after event creation)
    // Branchial edges depend on events existing, not states
    void process_pending_branchial_for_event(uint64_t new_event_id) {
        auto it = pending_branchial_.begin();
        while (it != pending_branchial_.end()) {
            if (try_process_branchial(*it)) {
                it = pending_branchial_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void handle_hyperedge_data(const HyperedgeData& data) {
        // Find the visualization state for this engine state
        auto it = engine_state_to_viz_.find(data.state_id);
        if (it == engine_state_to_viz_.end()) {
            // State not yet created - queue for later
            pending_hyperedges_.push_back(data);
            return;
        }

        handle_hyperedge_data_internal(data);
    }

    void handle_hyperedge_data_internal(const HyperedgeData& data) {
        auto it = engine_state_to_viz_.find(data.state_id);
        if (it == engine_state_to_viz_.end()) {
            return;  // Still not ready (shouldn't happen if called correctly)
        }

        StateId viz_id = it->second;
        if (viz_id >= evolution_.states.size()) {
            return;  // Invalid state ID
        }

        // Add edge to the state's hypergraph
        // Remap global vertex IDs to local 0-indexed IDs for this state
        State& state = evolution_.states[viz_id];
        std::vector<VertexId> vertices;
        vertices.reserve(data.vertex_count);

        // Get or create local vertex map for this state
        auto& vertex_map = state_vertex_maps_[data.state_id];

        for (uint8_t i = 0; i < data.vertex_count; ++i) {
            uint32_t global_v = data.vertices[i];
            auto [map_it, inserted] = vertex_map.try_emplace(global_v, static_cast<VertexId>(vertex_map.size()));
            vertices.push_back(map_it->second);
        }

        state.hypergraph.add_edge(vertices);
    }

    void handle_match_found(const MatchFoundData& data) {
        // Match events are informational - can be used for highlighting
        // For now, just track that matches exist
        (void)data;
    }

    void handle_rewrite_applied(const RewriteAppliedData& data) {
        if (!try_process_rewrite(data)) {
            // States not yet registered - queue for later
            pending_rewrites_.push_back(data);
        }
    }

    // Returns true if successfully processed, false if needs to wait
    bool try_process_rewrite(const RewriteAppliedData& data) {
        // Handle genesis events: source_state_id == VIZ_NO_SOURCE_STATE means no input state
        // Genesis events are synthetic events that "produce" initial state edges
        bool is_genesis = (data.source_state_id == viz::VIZ_NO_SOURCE_STATE);

        StateId src_viz = 0;  // For genesis, source is "null"
        if (!is_genesis) {
            auto src_it = engine_state_to_viz_.find(data.source_state_id);
            if (src_it == engine_state_to_viz_.end()) {
                return false;  // Not ready yet
            }
            src_viz = src_it->second;
        }

        auto tgt_it = engine_state_to_viz_.find(data.target_state_id);
        if (tgt_it == engine_state_to_viz_.end()) {
            return false;  // Not ready yet
        }

        StateId tgt_viz = tgt_it->second;

        // Check if this canonical event has already been visualized
        // This is the KEY deduplication step - multiple raw events may share
        // the same canonical_event_id (meaning they're equivalent events)
        bool is_canonical_event = seen_canonical_events_.insert(data.canonical_event_id).second;

        EventId event_id;
        if (is_canonical_event) {
            // First time seeing this canonical event - create the viz Event
            // For genesis events, we create a self-loop event (initial state -> initial state)
            // This keeps the event in the system for causal edge tracking
            event_id = is_genesis
                ? evolution_.add_genesis_event(tgt_viz)
                : evolution_.add_event(src_viz, tgt_viz);

            // Map canonical event ID so duplicate raw events can find this viz event
            engine_event_to_viz_[data.canonical_event_id] = event_id;

            if (on_event_added_) {
                on_event_added_(event_id);
            }
        } else {
            // Duplicate canonical event - look up the existing viz event ID
            auto canonical_it = engine_event_to_viz_.find(data.canonical_event_id);
            if (canonical_it != engine_event_to_viz_.end()) {
                event_id = canonical_it->second;
            } else {
                // Should not happen since we map canonical_event_id above
                // But if it does, use the raw event mapping as fallback
                event_id = is_genesis
                    ? evolution_.add_genesis_event(tgt_viz)
                    : evolution_.add_event(src_viz, tgt_viz);
            }
        }

        // ALWAYS map raw event_id → viz event
        // This is critical for causal/branchial edge resolution which uses raw event IDs
        engine_event_to_viz_[data.event_id] = event_id;

        // Process any pending causal edges that were waiting for this event
        process_pending_causal_for_event(data.event_id);

        // Process any pending branchial edges that were waiting for events
        // (branchial edges depend on events existing, not states)
        process_pending_branchial_for_event(data.event_id);

        return true;
    }

    void handle_causal_edge(const CausalEdgeData& data) {
        if (!try_process_causal(data)) {
            pending_causal_.push_back(data);
        }
    }

    bool try_process_causal(const CausalEdgeData& data) {
        auto prod_it = engine_event_to_viz_.find(data.producer_event_id);
        auto cons_it = engine_event_to_viz_.find(data.consumer_event_id);

        if (prod_it == engine_event_to_viz_.end() || cons_it == engine_event_to_viz_.end()) {
            return false;  // Not ready yet
        }

        evolution_.add_causal_edge(prod_it->second, cons_it->second);
        return true;
    }

    void handle_branchial_edge(const BranchialEdgeData& data) {
        if (!try_process_branchial(data)) {
            pending_branchial_.push_back(data);
        }
    }

    bool try_process_branchial(const BranchialEdgeData& data) {
        auto a_it = engine_event_to_viz_.find(data.event_a_id);
        auto b_it = engine_event_to_viz_.find(data.event_b_id);

        if (a_it == engine_event_to_viz_.end() || b_it == engine_event_to_viz_.end()) {
            return false;  // Events not ready yet
        }

        const Event* event_a = evolution_.get_event(a_it->second);
        const Event* event_b = evolution_.get_event(b_it->second);

        if (!event_a || !event_b) {
            return false;
        }

        StateId state_a = event_a->output_state;
        StateId state_b = event_b->output_state;

        if (state_a >= evolution_.states.size() || state_b >= evolution_.states.size()) {
            return false;
        }

        evolution_.add_branchial_edge(state_a, state_b);
        return true;
    }

    void handle_evolution_complete(const EvolutionCompleteData& data) {
        evolution_complete_ = true;
        max_generation_ = data.max_generation;

        // Final flush: try to resolve all remaining pending edges
        // This handles race conditions where events arrived out-of-order
        flush_pending_queues();
    }

    // Retry all pending queues - called at evolution end to resolve stragglers
    void flush_pending_queues() {
        // Multiple passes may be needed if dependencies chain
        bool made_progress = true;
        int max_passes = 10;  // Prevent infinite loops

        while (made_progress && max_passes-- > 0) {
            made_progress = false;

            // Try pending rewrites
            auto rit = pending_rewrites_.begin();
            while (rit != pending_rewrites_.end()) {
                if (try_process_rewrite(*rit)) {
                    rit = pending_rewrites_.erase(rit);
                    made_progress = true;
                } else {
                    ++rit;
                }
            }

            // Try pending causal edges
            auto cit = pending_causal_.begin();
            while (cit != pending_causal_.end()) {
                if (try_process_causal(*cit)) {
                    cit = pending_causal_.erase(cit);
                    made_progress = true;
                } else {
                    ++cit;
                }
            }

            // Try pending branchial edges
            auto bit = pending_branchial_.begin();
            while (bit != pending_branchial_.end()) {
                if (try_process_branchial(*bit)) {
                    bit = pending_branchial_.erase(bit);
                    made_progress = true;
                } else {
                    ++bit;
                }
            }
        }
    }
};

} // namespace viz::scene
