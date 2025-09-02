/**
 * Basic Hypergraph Usage Example
 * 
 * Demonstrates fundamental hypergraph operations:
 * - Creating vertices and edges
 * - Querying graph structure
 * - Basic traversal operations
 */

#include <hypergraph/hypergraph.hpp>
#include <iostream>

using namespace hypergraph;

int main() {
    std::cout << "=== Basic Hypergraph Usage Example ===\n\n";
    
    // Create an empty hypergraph
    Hypergraph graph;
    std::cout << "Created empty hypergraph\n";
    std::cout << "Initial vertices: " << graph.num_vertices() << ", edges: " << graph.num_edges() << "\n\n";
    
    // Add edges (vertices are created automatically)
    std::cout << "Adding edges:\n";
    EdgeId edge1 = graph.add_edge({1, 2});
    std::cout << "  Added edge {1, 2} with ID: " << edge1 << "\n";
    
    EdgeId edge2 = graph.add_edge({2, 3});
    std::cout << "  Added edge {2, 3} with ID: " << edge2 << "\n";
    
    EdgeId edge3 = graph.add_edge({3, 1});
    std::cout << "  Added edge {3, 1} with ID: " << edge3 << "\n";
    
    // Add a hyperedge (more than 2 vertices)
    EdgeId hyperedge = graph.add_edge({1, 2, 3, 4});
    std::cout << "  Added hyperedge {1, 2, 3, 4} with ID: " << hyperedge << "\n\n";
    
    // Query graph structure
    std::cout << "Graph structure:\n";
    std::cout << "  Vertices: " << graph.num_vertices() << "\n";
    std::cout << "  Edges: " << graph.num_edges() << "\n\n";
    
    // Find edges containing specific vertices
    std::cout << "Edges containing vertex 1:\n";
    auto edges_with_1 = graph.edges_containing(1);
    for (EdgeId edge_id : edges_with_1) {
        const auto* edge = graph.get_edge(edge_id);
        if (edge) {
            std::cout << "  Edge " << edge_id << ": {";
            for (std::size_t i = 0; i < edge->arity(); ++i) {
                std::cout << edge->vertex(i);
                if (i < edge->arity() - 1) std::cout << ", ";
            }
            std::cout << "}\n";
        }
    }
    std::cout << "\n";
    
    // Find edges within radius
    std::cout << "Edges within radius 1 of vertex 2:\n";
    auto nearby_edges = graph.edges_within_radius(2, 1);
    for (EdgeId edge_id : nearby_edges) {
        const auto* edge = graph.get_edge(edge_id);
        if (edge) {
            std::cout << "  Edge " << edge_id << ": {";
            for (std::size_t i = 0; i < edge->arity(); ++i) {
                std::cout << edge->vertex(i);
                if (i < edge->arity() - 1) std::cout << ", ";
            }
            std::cout << "}\n";
        }
    }
    std::cout << "\n";
    
    // Vertex degree (number of edges containing each vertex)
    std::cout << "Vertex degrees:\n";
    for (VertexId vertex : graph.vertices()) {
        auto incident_edges = graph.edges_containing(vertex);
        std::cout << "  Vertex " << vertex << ": degree " << incident_edges.size() << "\n";
    }
    std::cout << "\n";
    
    // Remove an edge
    std::cout << "Removing edge " << edge2 << "\n";
    graph.remove_edge(edge2);
    std::cout << "After removal: " << graph.num_vertices() << " vertices, " << graph.num_edges() << " edges\n";
    
    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}