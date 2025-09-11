#ifndef HYPERGRAPH_TYPES_HPP
#define HYPERGRAPH_TYPES_HPP

#include <cstddef>
#include <limits>

namespace hypergraph {

// Global vertex and edge types for multiway system
using GlobalVertexId = std::size_t;
using GlobalEdgeId = std::size_t;

// State and event types for evolution system
using StateId = std::size_t;
using EventId = std::size_t;

// Special invalid values for new types only
constexpr GlobalVertexId INVALID_GLOBAL_VERTEX = std::numeric_limits<GlobalVertexId>::max();
constexpr GlobalEdgeId INVALID_GLOBAL_EDGE = std::numeric_limits<GlobalEdgeId>::max();
constexpr StateId INVALID_STATE = std::numeric_limits<StateId>::max();
constexpr EventId INVALID_EVENT = std::numeric_limits<EventId>::max();

} // namespace hypergraph

#endif // HYPERGRAPH_TYPES_HPP