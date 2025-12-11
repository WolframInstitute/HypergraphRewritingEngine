// Hypergraph data structures for visualization
// These are decoupled from the rewriting engine - can be populated from mock data or real engine

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

namespace viz::scene {

// Type aliases for clarity
using VertexId = uint32_t;
using EdgeId = uint32_t;
using StateId = uint32_t;
using EventId = uint32_t;

// A hyperedge in the internal hypergraph representation
// Stored as ordered sequence: {1, 2, 2, 3} means edges 1→2, 2→2 (self-loop), 2→3
struct Hyperedge {
    EdgeId id;
    std::vector<VertexId> vertices;  // Ordered sequence

    // Derived info (computed after layout)
    bool has_self_loop() const {
        for (size_t i = 0; i + 1 < vertices.size(); ++i) {
            if (vertices[i] == vertices[i + 1]) return true;
        }
        return false;
    }

    // Get all self-loop positions (index where v[i] == v[i+1])
    std::vector<size_t> get_self_loop_indices() const {
        std::vector<size_t> loops;
        for (size_t i = 0; i + 1 < vertices.size(); ++i) {
            if (vertices[i] == vertices[i + 1]) loops.push_back(i);
        }
        return loops;
    }

    // Arity of the hyperedge (number of vertices)
    size_t arity() const { return vertices.size(); }
};

// A hypergraph (collection of hyperedges sharing vertices)
struct Hypergraph {
    std::vector<Hyperedge> edges;
    uint32_t vertex_count = 0;  // Max vertex ID + 1 (for layout sizing)
    std::vector<VertexId> used_vertices;  // Actually used vertices (for rendering)

    // Add a hyperedge, auto-updating vertex_count and used_vertices
    void add_edge(const std::vector<VertexId>& vertices) {
        Hyperedge e;
        e.id = static_cast<EdgeId>(edges.size());
        e.vertices = vertices;
        for (auto v : vertices) {
            if (v >= vertex_count) vertex_count = v + 1;
            // Track unique used vertices
            bool found = false;
            for (auto uv : used_vertices) {
                if (uv == v) { found = true; break; }
            }
            if (!found) used_vertices.push_back(v);
        }
        edges.push_back(std::move(e));
    }

    // Get all unique vertices that are actually used in edges
    std::vector<VertexId> get_vertices() const {
        return used_vertices;
    }

    // Get count of actually used vertices
    uint32_t num_used_vertices() const {
        return static_cast<uint32_t>(used_vertices.size());
    }
};

// A state in the multiway system (contains a hypergraph)
struct State {
    StateId id;
    StateId canonical_id;      // If canonicalized, points to canonical representative
    Hypergraph hypergraph;
    bool is_initial = false;

    bool is_canonical() const { return id == canonical_id; }
};

// An event (transition between states via rule application)
struct Event {
    EventId id;
    StateId input_state;
    StateId output_state;

    // Which edges were consumed/produced (indices into hypergraph)
    std::vector<EdgeId> consumed_edges;
    std::vector<EdgeId> produced_edges;

    // Multiplicity (for bundled edges between same states)
    uint32_t multiplicity = 1;
};

// Edge types in the evolution graph
enum class EvolutionEdgeType {
    Event,      // State → State (via event)
    Causal,     // Event → Event (causal dependency)
    Branchial   // State ↔ State (same step, common ancestry)
};

// An edge in the evolution graph (between states or events)
struct EvolutionEdge {
    uint32_t id;
    EvolutionEdgeType type;

    // Source/target depend on type:
    // Event: source_state → target_state
    // Causal: source_event → target_event
    // Branchial: state1 ↔ state2 (undirected)
    uint32_t source;
    uint32_t target;

    // For bundled edges
    uint32_t multiplicity = 1;
    std::vector<EventId> bundled_events;  // If this is a bundle
};

// The complete evolution data structure
struct Evolution {
    std::vector<State> states;
    std::vector<Event> events;
    std::vector<EvolutionEdge> evolution_edges;

    // Indices for quick lookup
    std::unordered_map<StateId, size_t> state_index;
    std::unordered_map<EventId, size_t> event_index;

    // Add a state
    StateId add_state(const Hypergraph& hg, bool is_initial = false) {
        StateId id = static_cast<StateId>(states.size());
        State s;
        s.id = id;
        s.canonical_id = id;  // Self-canonical by default
        s.hypergraph = hg;
        s.is_initial = is_initial;
        state_index[id] = states.size();
        states.push_back(std::move(s));
        return id;
    }

    // Add an event between states
    EventId add_event(StateId input, StateId output,
                      const std::vector<EdgeId>& consumed = {},
                      const std::vector<EdgeId>& produced = {}) {
        EventId id = static_cast<EventId>(events.size());
        Event e;
        e.id = id;
        e.input_state = input;
        e.output_state = output;
        e.consumed_edges = consumed;
        e.produced_edges = produced;
        event_index[id] = events.size();
        events.push_back(std::move(e));

        // Also add an evolution edge
        EvolutionEdge edge;
        edge.id = static_cast<uint32_t>(evolution_edges.size());
        edge.type = EvolutionEdgeType::Event;
        edge.source = input;
        edge.target = output;
        evolution_edges.push_back(edge);

        return id;
    }

    // Add a genesis event for an initial state
    // Genesis events have no input state (they "create" the initial state from nothing)
    // We store them as events with input_state == output_state (self-loop on initial state)
    // but don't create an evolution edge for them (they're invisible in the state graph)
    EventId add_genesis_event(StateId initial_state) {
        EventId id = static_cast<EventId>(events.size());
        Event e;
        e.id = id;
        e.input_state = initial_state;   // Self-reference (no actual input)
        e.output_state = initial_state;  // Output is the initial state
        // Genesis events produce all edges in the initial state, consume none
        event_index[id] = events.size();
        events.push_back(std::move(e));
        // Note: No evolution edge for genesis events - they're not rendered as state transitions
        return id;
    }

    // Add a causal edge between events
    void add_causal_edge(EventId source, EventId target) {
        EvolutionEdge edge;
        edge.id = static_cast<uint32_t>(evolution_edges.size());
        edge.type = EvolutionEdgeType::Causal;
        edge.source = source;
        edge.target = target;
        evolution_edges.push_back(edge);
    }

    // Add a branchial edge between states
    void add_branchial_edge(StateId state1, StateId state2) {
        EvolutionEdge edge;
        edge.id = static_cast<uint32_t>(evolution_edges.size());
        edge.type = EvolutionEdgeType::Branchial;
        edge.source = state1;
        edge.target = state2;
        evolution_edges.push_back(edge);
    }

    // Set canonical mapping (raw state → canonical state)
    void set_canonical(StateId raw, StateId canonical) {
        if (auto it = state_index.find(raw); it != state_index.end()) {
            states[it->second].canonical_id = canonical;
        }
    }

    // Get state by ID
    const State* get_state(StateId id) const {
        auto it = state_index.find(id);
        return it != state_index.end() ? &states[it->second] : nullptr;
    }

    // Get event by ID
    const Event* get_event(EventId id) const {
        auto it = event_index.find(id);
        return it != event_index.end() ? &events[it->second] : nullptr;
    }
};

} // namespace viz::scene
