#ifndef HYPERGRAPH_HYPERGRAPH_HPP
#define HYPERGRAPH_HYPERGRAPH_HPP

#include <hypergraph/vertex.hpp>
#include <hypergraph/hyperedge.hpp>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

namespace hypergraph {

/**
 * Directed hypergraph with ordered vertices in hyperedges.
 * Supports efficient lookup of edges by vertex participation.
 */
class Hypergraph {
private:
    std::vector<Hyperedge> edges_;
    std::unordered_set<VertexId> vertices_;
    
    // Index: vertex -> edges containing that vertex
    std::unordered_map<VertexId, std::vector<EdgeId>> vertex_to_edges_;
    
    EdgeId next_edge_id_ = 0;
    VertexId next_vertex_id_ = 0;
    
    void update_indices_add_edge(const Hyperedge& edge) {
        for (VertexId vertex_id : edge.vertices()) {
            vertex_to_edges_[vertex_id].push_back(edge.id());
            vertices_.insert(vertex_id);
        }
    }
    
    void update_indices_remove_edge(const Hyperedge& edge) {
        for (VertexId vertex_id : edge.vertices()) {
            auto& edge_list = vertex_to_edges_[vertex_id];
            edge_list.erase(std::remove(edge_list.begin(), edge_list.end(), edge.id()), 
                           edge_list.end());
            
            // Remove vertex if no edges reference it
            if (edge_list.empty()) {
                vertex_to_edges_.erase(vertex_id);
                vertices_.erase(vertex_id);
            }
        }
    }
    
public:
    Hypergraph() = default;
    
    // Copy constructor
    Hypergraph(const Hypergraph& other) 
        : edges_(other.edges_)
        , vertices_(other.vertices_)
        , vertex_to_edges_(other.vertex_to_edges_)
        , next_edge_id_(other.next_edge_id_)
        , next_vertex_id_(other.next_vertex_id_) {}
    
    // Assignment operator
    Hypergraph& operator=(const Hypergraph& other) {
        if (this != &other) {
            edges_ = other.edges_;
            vertices_ = other.vertices_;
            vertex_to_edges_ = other.vertex_to_edges_;
            next_edge_id_ = other.next_edge_id_;
            next_vertex_id_ = other.next_vertex_id_;
        }
        return *this;
    }
    
    // Add edge with specified vertices
    EdgeId add_edge(const std::vector<VertexId>& vertices) {
        if (vertices.empty()) {
            throw std::invalid_argument("Cannot add edge with no vertices");
        }
        
        EdgeId edge_id = next_edge_id_++;
        Hyperedge edge(edge_id, vertices);
        
        edges_.push_back(edge);
        update_indices_add_edge(edge);
        
        // Update next_vertex_id to be beyond any vertex we've seen
        for (VertexId v : vertices) {
            next_vertex_id_ = std::max(next_vertex_id_, v + 1);
        }
        
        return edge_id;
    }
    
    // Add edge with initializer list
    EdgeId add_edge(std::initializer_list<VertexId> vertices) {
        return add_edge(std::vector<VertexId>(vertices));
    }
    
    // Remove edge by ID
    bool remove_edge(EdgeId edge_id) {
        auto it = std::find_if(edges_.begin(), edges_.end(),
            [edge_id](const Hyperedge& e) { return e.id() == edge_id; });
        
        if (it == edges_.end()) {
            return false;
        }
        
        update_indices_remove_edge(*it);
        edges_.erase(it);
        return true;
    }
    
    // Remove edges by structural match
    std::vector<EdgeId> remove_edges_matching(const std::vector<VertexId>& vertices) {
        std::vector<EdgeId> removed;
        
        auto it = edges_.begin();
        while (it != edges_.end()) {
            if (it->vertices() == vertices) {
                removed.push_back(it->id());
                update_indices_remove_edge(*it);
                it = edges_.erase(it);
            } else {
                ++it;
            }
        }
        
        return removed;
    }
    
    // Create new vertex
    VertexId create_vertex() {
        return next_vertex_id_++;
    }
    
    // Accessors
    const std::vector<Hyperedge>& edges() const { return edges_; }
    const std::unordered_set<VertexId>& vertices() const { return vertices_; }
    std::size_t num_edges() const { return edges_.size(); }
    std::size_t num_vertices() const { return vertices_.size(); }
    
    // Get edge by ID
    const Hyperedge* get_edge(EdgeId edge_id) const {
        auto it = std::find_if(edges_.begin(), edges_.end(),
            [edge_id](const Hyperedge& e) { return e.id() == edge_id; });
        return (it != edges_.end()) ? &(*it) : nullptr;
    }
    
    // Get edges containing a vertex
    std::vector<EdgeId> edges_containing(VertexId vertex_id) const {
        auto it = vertex_to_edges_.find(vertex_id);
        return (it != vertex_to_edges_.end()) ? it->second : std::vector<EdgeId>{};
    }
    
    // Check if vertex exists
    bool has_vertex(VertexId vertex_id) const {
        return vertices_.count(vertex_id) > 0;
    }
    
    // Check if edge exists
    bool has_edge(EdgeId edge_id) const {
        return get_edge(edge_id) != nullptr;
    }
    
    // Get edges within radius of a vertex
    std::unordered_set<EdgeId> edges_within_radius(VertexId center_vertex, std::size_t radius) const {
        if (radius == 0) {
            return {};
        }
        
        std::unordered_set<EdgeId> result;
        std::unordered_set<VertexId> visited_vertices;
        std::unordered_set<VertexId> current_vertices = {center_vertex};
        
        for (std::size_t r = 0; r < radius && !current_vertices.empty(); ++r) {
            std::unordered_set<VertexId> next_vertices;
            
            for (VertexId v : current_vertices) {
                if (visited_vertices.insert(v).second) {  // If not visited before
                    auto edge_ids = edges_containing(v);
                    for (EdgeId edge_id : edge_ids) {
                        result.insert(edge_id);
                        
                        // Add all vertices from this edge to next round
                        const Hyperedge* edge = get_edge(edge_id);
                        if (edge) {
                            for (VertexId next_v : edge->vertices()) {
                                if (visited_vertices.count(next_v) == 0) {
                                    next_vertices.insert(next_v);
                                }
                            }
                        }
                    }
                }
            }
            
            current_vertices = std::move(next_vertices);
        }
        
        return result;
    }
    
    // Equality comparison
    bool operator==(const Hypergraph& other) const {
        // Quick size check
        if (edges_.size() != other.edges_.size() || vertices_.size() != other.vertices_.size()) {
            return false;
        }
        
        // Check if all edges match structurally (ignoring IDs)
        std::vector<std::vector<VertexId>> our_edges, other_edges;
        
        for (const auto& edge : edges_) {
            our_edges.push_back(edge.vertices());
        }
        for (const auto& edge : other.edges_) {
            other_edges.push_back(edge.vertices());
        }
        
        std::sort(our_edges.begin(), our_edges.end());
        std::sort(other_edges.begin(), other_edges.end());
        
        return our_edges == other_edges;
    }
    
    bool operator!=(const Hypergraph& other) const {
        return !(*this == other);
    }
    
    // Create a deep copy with fresh IDs
    Hypergraph clone() const {
        Hypergraph copy;
        
        // Map old vertex IDs to new ones
        std::unordered_map<VertexId, VertexId> vertex_map;
        for (VertexId old_id : vertices_) {
            vertex_map[old_id] = copy.create_vertex();
        }
        
        // Add edges with mapped vertices
        for (const auto& edge : edges_) {
            std::vector<VertexId> mapped_vertices;
            for (VertexId old_vertex : edge.vertices()) {
                mapped_vertices.push_back(vertex_map[old_vertex]);
            }
            copy.add_edge(mapped_vertices);
        }
        
        return copy;
    }
    
    // Clear all edges and vertices
    void clear() {
        edges_.clear();
        vertices_.clear();
        vertex_to_edges_.clear();
        next_edge_id_ = 0;
        next_vertex_id_ = 0;
    }
};

} // namespace hypergraph

#endif // HYPERGRAPH_HYPERGRAPH_HPP