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
// Strong types to prevent mixing canonical and raw StateIds
struct CanonicalStateId {
    std::size_t value;
    explicit constexpr CanonicalStateId(std::size_t v = 0) : value(v) {}
    constexpr bool operator==(const CanonicalStateId& other) const { return value == other.value; }
    constexpr bool operator!=(const CanonicalStateId& other) const { return value != other.value; }
    constexpr bool operator<(const CanonicalStateId& other) const { return value < other.value; }
};

struct RawStateId {
    std::size_t value;
    explicit constexpr RawStateId(std::size_t v = 0) : value(v) {}
    constexpr bool operator==(const RawStateId& other) const { return value == other.value; }
    constexpr bool operator!=(const RawStateId& other) const { return value != other.value; }
    constexpr bool operator<(const RawStateId& other) const { return value < other.value; }
};

using EventId = std::size_t;

// Special invalid values for new types only
constexpr GlobalVertexId INVALID_GLOBAL_VERTEX = std::numeric_limits<GlobalVertexId>::max();
constexpr GlobalEdgeId INVALID_GLOBAL_EDGE = std::numeric_limits<GlobalEdgeId>::max();
constexpr CanonicalStateId INVALID_CANONICAL_STATE{std::numeric_limits<std::size_t>::max()};
constexpr RawStateId INVALID_RAW_STATE{std::numeric_limits<std::size_t>::max()};
constexpr EventId INVALID_EVENT = std::numeric_limits<EventId>::max();

// Legacy constant for backward compatibility during transition
constexpr std::size_t INVALID_STATE = std::numeric_limits<std::size_t>::max();

} // namespace hypergraph

// Hash functions for the strong types
namespace std {
    template<>
    struct hash<hypergraph::CanonicalStateId> {
        std::size_t operator()(const hypergraph::CanonicalStateId& id) const {
            return std::hash<std::size_t>{}(id.value);
        }
    };

    template<>
    struct hash<hypergraph::RawStateId> {
        std::size_t operator()(const hypergraph::RawStateId& id) const {
            return std::hash<std::size_t>{}(id.value);
        }
    };
}

#endif // HYPERGRAPH_TYPES_HPP