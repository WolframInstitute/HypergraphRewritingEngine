#ifndef HYPERGRAPH_VERTEX_HPP
#define HYPERGRAPH_VERTEX_HPP

#include <cstddef>
#include <functional>
#include <limits>

namespace hypergraph {

using VertexId = std::size_t;

constexpr VertexId INVALID_VERTEX = std::numeric_limits<std::size_t>::max();

struct Vertex {
    VertexId id;
    
    explicit Vertex(VertexId vertex_id = INVALID_VERTEX) : id(vertex_id) {}
    
    bool operator==(const Vertex& other) const {
        return id == other.id;
    }
    
    bool operator!=(const Vertex& other) const {
        return !(*this == other);
    }
    
    bool operator<(const Vertex& other) const {
        return id < other.id;
    }
    
    bool is_valid() const {
        return id != INVALID_VERTEX;
    }
};

} // namespace hypergraph

namespace std {
template<>
struct hash<hypergraph::Vertex> {
    std::size_t operator()(const hypergraph::Vertex& v) const {
        return std::hash<hypergraph::VertexId>{}(v.id);
    }
};
}

#endif // HYPERGRAPH_VERTEX_HPP