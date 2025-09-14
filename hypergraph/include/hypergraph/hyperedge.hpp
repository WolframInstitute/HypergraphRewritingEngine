#ifndef HYPERGRAPH_HYPEREDGE_HPP
#define HYPERGRAPH_HYPEREDGE_HPP

#include <hypergraph/vertex.hpp>
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hypergraph {

using EdgeId = std::size_t;

constexpr EdgeId INVALID_EDGE = std::numeric_limits<std::size_t>::max();

/**
 * Directed hyperedge with ordered vertices.
 * Vertices are ordered, meaning {A,B,C} != {B,A,C}
 * Can have arity 1, 2, 3, or more vertices.
 */
class Hyperedge {
private:
    EdgeId id_;
    std::vector<VertexId> vertices_;

public:
    explicit Hyperedge(EdgeId edge_id = INVALID_EDGE)
        : id_(edge_id) {}

    Hyperedge(EdgeId edge_id, const std::vector<VertexId>& vertices)
        : id_(edge_id), vertices_(vertices) {
        if (vertices_.empty()) {
            throw std::invalid_argument("Hyperedge must have at least one vertex");
        }
    }

    Hyperedge(EdgeId edge_id, std::initializer_list<VertexId> vertices)
        : id_(edge_id), vertices_(vertices) {
        if (vertices_.empty()) {
            throw std::invalid_argument("Hyperedge must have at least one vertex");
        }
    }

    // Accessors
    EdgeId id() const { return id_; }
    const std::vector<VertexId>& vertices() const { return vertices_; }
    std::size_t arity() const { return vertices_.size(); }
    bool is_valid() const { return id_ != INVALID_EDGE && !vertices_.empty(); }

    // Vertex access
    VertexId vertex(std::size_t index) const {
        if (index >= vertices_.size()) {
            throw std::out_of_range("Vertex index out of range");
        }
        return vertices_[index];
    }

    VertexId operator[](std::size_t index) const {
        return vertices_[index];
    }

    // Iterators
    auto begin() const { return vertices_.begin(); }
    auto end() const { return vertices_.end(); }

    // Check if edge contains a vertex
    bool contains(VertexId vertex_id) const {
        return std::find(vertices_.begin(), vertices_.end(), vertex_id) != vertices_.end();
    }

    // Equality comparison (order matters!)
    bool operator==(const Hyperedge& other) const {
        return id_ == other.id_ && vertices_ == other.vertices_;
    }

    bool operator!=(const Hyperedge& other) const {
        return !(*this == other);
    }

    // Less than for ordering (first by arity, then lexicographically by vertices)
    bool operator<(const Hyperedge& other) const {
        if (arity() != other.arity()) {
            return arity() < other.arity();
        }
        return vertices_ < other.vertices_;
    }

    // Structural equality (ignores edge ID)
    bool structurally_equal(const Hyperedge& other) const {
        return vertices_ == other.vertices_;
    }
};

} // namespace hypergraph

namespace std {
/* template<>
struct hash<hypergraph::Hyperedge> {
    std::size_t operator()(const hypergraph::Hyperedge& edge) const {
        std::size_t hash_value = 0;
        std::hash<hypergraph::VertexId> vertex_hasher;

        // Combine hashes of all vertices in order
        for (const auto& vertex : edge.vertices()) {
            hash_value ^= vertex_hasher(vertex) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }

        return hash_value;
    }
}; */
}

#endif // HYPERGRAPH_HYPEREDGE_HPP