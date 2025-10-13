#ifndef HYPERGRAPH_TYPES_HPP
#define HYPERGRAPH_TYPES_HPP

#include <cstddef>
#include <limits>
#include <functional>

namespace hypergraph {

// Global vertex and edge types for multiway system
using GlobalVertexId = std::size_t;
using GlobalEdgeId = std::size_t;

// State and event types for evolution system
// Strong type for StateID
struct StateID {
    std::size_t value;
    explicit constexpr StateID(std::size_t v = 0) : value(v) {}
    constexpr bool operator==(const StateID& other) const { return value == other.value; }
    constexpr bool operator!=(const StateID& other) const { return value != other.value; }
    constexpr bool operator<(const StateID& other) const { return value < other.value; }
};

using EventId = std::size_t;

// Special invalid values for new types only
constexpr GlobalVertexId INVALID_GLOBAL_VERTEX = std::numeric_limits<GlobalVertexId>::max();
constexpr GlobalEdgeId INVALID_GLOBAL_EDGE = std::numeric_limits<GlobalEdgeId>::max();
constexpr StateID INVALID_STATE{std::numeric_limits<std::size_t>::max()};
constexpr EventId INVALID_EVENT = std::numeric_limits<EventId>::max();

// Hash strategy types for runtime selection
enum class HashStrategyType {
    CANONICALIZATION,  // O(n!) exact canonical form
    UNIQUENESS_TREE    // O(n^7) polynomial-time hash
};


} // namespace hypergraph

// Hash functions for the strong types
namespace std {
    template<>
    struct hash<hypergraph::StateID> {
        std::size_t operator()(const hypergraph::StateID& id) const {
            return std::hash<std::size_t>{}(id.value);
        }
    };
}

#endif // HYPERGRAPH_TYPES_HPP