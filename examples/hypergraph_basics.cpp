/**
 * Basic Hypergraph Usage Example
 *
 * Demonstrates fundamental hypergraph operations using the API:
 * - Creating states with edges
 * - Querying graph structure
 * - Iterating over edges and vertices
 */

#include <hypergraph/hypergraph.hpp>
#include <iostream>
#include <set>

using namespace hypergraph;

int main() {
    std::cout << "=== Basic Hypergraph Usage Example ===\n\n";

    // Create a hypergraph
    Hypergraph hg;
    std::cout << "Created empty hypergraph\n";
    std::cout << "Initial states: " << hg.num_states() << ", edges: " << hg.num_edges() << "\n\n";

    // In the API, edges belong to states. Let's create a state with some edges.
    std::cout << "Creating initial state with edges:\n";

    // Create edges directly using internal methods
    EdgeId edge1 = hg.create_edge({1, 2});
    std::cout << "  Created edge {1, 2} with ID: " << edge1 << "\n";

    EdgeId edge2 = hg.create_edge({2, 3});
    std::cout << "  Created edge {2, 3} with ID: " << edge2 << "\n";

    EdgeId edge3 = hg.create_edge({3, 1});
    std::cout << "  Created edge {3, 1} with ID: " << edge3 << "\n";

    // Create a hyperedge (more than 2 vertices)
    EdgeId hyperedge = hg.create_edge({1, 2, 3, 4});
    std::cout << "  Created hyperedge {1, 2, 3, 4} with ID: " << hyperedge << "\n\n";

    // Create a state from these edges
    std::vector<EdgeId> edge_ids = {edge1, edge2, edge3, hyperedge};
    StateId state = hg.create_state(edge_ids.data(), static_cast<uint32_t>(edge_ids.size()));
    std::cout << "Created state " << state << " with " << edge_ids.size() << " edges\n\n";

    // Query edge structure
    std::cout << "Edge details:\n";
    for (EdgeId eid : edge_ids) {
        const auto& edge = hg.get_edge(eid);
        std::cout << "  Edge " << eid << ": {";
        for (uint8_t i = 0; i < edge.arity; ++i) {
            std::cout << edge.vertices[i];
            if (i < edge.arity - 1) std::cout << ", ";
        }
        std::cout << "} (arity=" << (int)edge.arity << ")\n";
    }
    std::cout << "\n";

    // Count unique vertices
    std::set<VertexId> unique_vertices;
    for (EdgeId eid : edge_ids) {
        const auto& edge = hg.get_edge(eid);
        for (uint8_t i = 0; i < edge.arity; ++i) {
            unique_vertices.insert(edge.vertices[i]);
        }
    }

    std::cout << "Graph structure:\n";
    std::cout << "  Unique vertices: " << unique_vertices.size() << "\n";
    std::cout << "  Total edges: " << hg.num_edges() << "\n";
    std::cout << "  Total states: " << hg.num_states() << "\n\n";

    // Print vertices in the state
    std::cout << "Vertices in state: {";
    bool first = true;
    for (VertexId v : unique_vertices) {
        if (!first) std::cout << ", ";
        std::cout << v;
        first = false;
    }
    std::cout << "}\n\n";

    // Demonstrate state iteration
    const auto& state_data = hg.get_state(state);
    std::cout << "Edges in state " << state << ":\n";
    state_data.edges.for_each([&](EdgeId eid) {
        const auto& edge = hg.get_edge(eid);
        std::cout << "  Edge " << eid << ": {";
        for (uint8_t i = 0; i < edge.arity; ++i) {
            std::cout << edge.vertices[i];
            if (i < edge.arity - 1) std::cout << ", ";
        }
        std::cout << "}\n";
    });

    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}
