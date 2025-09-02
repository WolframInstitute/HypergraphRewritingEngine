#ifndef HYPERGRAPH_PATTERN_MATCHING_HPP
#define HYPERGRAPH_PATTERN_MATCHING_HPP

#include <hypergraph/hypergraph.hpp>
#include <hypergraph/canonicalization.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace hypergraph {

/**
 * Pattern vertex that can be either concrete or variable.
 * Variable vertices can match any concrete vertex in the target hypergraph.
 */
struct PatternVertex {
    enum Type { CONCRETE, VARIABLE };

    Type type;
    VertexId id;  // For concrete vertices, this is the actual vertex ID
                  // For variable vertices, this is a variable identifier

    PatternVertex(Type t, VertexId vertex_id) : type(t), id(vertex_id) {}

    static PatternVertex concrete(VertexId vertex_id) {
        return PatternVertex(CONCRETE, vertex_id);
    }

    static PatternVertex variable(VertexId var_id) {
        return PatternVertex(VARIABLE, var_id);
    }

    bool is_concrete() const { return type == CONCRETE; }
    bool is_variable() const { return type == VARIABLE; }

    bool operator==(const PatternVertex& other) const {
        return type == other.type && id == other.id;
    }

    bool operator!=(const PatternVertex& other) const {
        return !(*this == other);
    }
};

/**
 * Pattern edge with pattern vertices.
 */
struct PatternEdge {
    std::vector<PatternVertex> vertices;

    PatternEdge(const std::vector<PatternVertex>& pattern_vertices)
        : vertices(pattern_vertices) {}

    PatternEdge(std::initializer_list<PatternVertex> pattern_vertices)
        : vertices(pattern_vertices) {}

    std::size_t arity() const { return vertices.size(); }

    bool has_variables() const {
        for (const auto& v : vertices) {
            if (v.is_variable()) return true;
        }
        return false;
    }

    std::unordered_set<VertexId> get_variables() const {
        std::unordered_set<VertexId> vars;
        for (const auto& v : vertices) {
            if (v.is_variable()) {
                vars.insert(v.id);
            }
        }
        return vars;
    }
};

/**
 * Pattern hypergraph for matching against concrete hypergraphs.
 */
class PatternHypergraph {
private:
    std::vector<PatternEdge> edges_;
    std::unordered_set<VertexId> concrete_vertices_;
    std::unordered_set<VertexId> variable_vertices_;

public:
    PatternHypergraph() = default;

    void add_edge(const PatternEdge& edge) {
        edges_.push_back(edge);
        for (const auto& vertex : edge.vertices) {
            if (vertex.is_concrete()) {
                concrete_vertices_.insert(vertex.id);
            } else {
                variable_vertices_.insert(vertex.id);
            }
        }
    }


    const std::vector<PatternEdge>& edges() const { return edges_; }
    const std::unordered_set<VertexId>& concrete_vertices() const { return concrete_vertices_; }
    const std::unordered_set<VertexId>& variable_vertices() const { return variable_vertices_; }

    std::size_t num_edges() const { return edges_.size(); }
    std::size_t num_concrete_vertices() const { return concrete_vertices_.size(); }
    std::size_t num_variable_vertices() const { return variable_vertices_.size(); }

    bool has_variables() const { return !variable_vertices_.empty(); }
};

/**
 * Variable assignment mapping pattern variables to concrete vertices.
 */
struct VariableAssignment {
    std::unordered_map<VertexId, VertexId> variable_to_concrete;

    // Apply assignment to get concrete vertex for pattern vertex
    std::optional<VertexId> resolve(const PatternVertex& pattern_vertex) const {
        if (pattern_vertex.is_concrete()) {
            return pattern_vertex.id;
        }

        auto it = variable_to_concrete.find(pattern_vertex.id);
        return (it != variable_to_concrete.end()) ?
               std::optional<VertexId>(it->second) : std::nullopt;
    }

    // Assign variable to concrete vertex
    bool assign(VertexId variable, VertexId concrete) {
        auto it = variable_to_concrete.find(variable);
        if (it != variable_to_concrete.end()) {
            return it->second == concrete;  // Check consistency
        }
        variable_to_concrete[variable] = concrete;
        return true;
    }

    // Check if assignment is complete for given variables
    bool is_complete(const std::unordered_set<VertexId>& variables) const {
        for (VertexId var : variables) {
            if (variable_to_concrete.find(var) == variable_to_concrete.end()) {
                return false;
            }
        }
        return true;
    }

    void clear() { variable_to_concrete.clear(); }
};

/**
 * Match result containing the matched edges and variable assignment.
 */
struct PatternMatch {
    std::vector<EdgeId> matched_edges;  // Edges in target that match pattern
    std::unordered_map<std::size_t, EdgeId> edge_map;  // Pattern edge index -> Data edge ID
    VariableAssignment assignment;      // Variable assignments
    VertexId anchor_vertex;             // Vertex around which match was found

    bool is_valid() const { return !matched_edges.empty(); }

    /**
     * Get the data edge that matches a specific pattern edge.
     */
    EdgeId get_matched_edge(std::size_t pattern_edge_idx) const {
        auto it = edge_map.find(pattern_edge_idx);
        return (it != edge_map.end()) ? it->second : static_cast<EdgeId>(-1);
    }
};

/**
 * Pattern matching engine with radius-based search.
 */
class PatternMatcher {
private:
    /**
     * Check if a concrete edge matches a pattern edge under given assignment.
     */
    bool edge_matches(const Hyperedge& concrete_edge,
                     const PatternEdge& pattern_edge,
                     VariableAssignment& assignment) const;

    /**
     * Find all possible matches for a pattern starting from an anchor vertex.
     */
    std::vector<PatternMatch> find_matches_from_anchor(
        const Hypergraph& target,
        const PatternHypergraph& pattern,
        VertexId anchor_vertex,
        std::size_t search_radius) const;

    /**
     * Recursively try to match remaining pattern edges.
     */
    bool match_remaining_edges(
        const Hypergraph& target,
        const std::vector<PatternEdge>& remaining_pattern_edges,
        const std::unordered_set<EdgeId>& available_edges,
        std::vector<EdgeId>& matched_edges,
        VariableAssignment& assignment) const;

    /**
     * Generate all possible variable assignments for an edge match.
     */
    std::vector<VariableAssignment> generate_assignments(
        const Hyperedge& concrete_edge,
        const PatternEdge& pattern_edge,
        const VariableAssignment& base_assignment) const;

public:
    /**
     * Find all matches of pattern in target hypergraph around a specific vertex.
     * Uses radius-based search to limit scope.
     */
    std::vector<PatternMatch> find_matches_around(
        const Hypergraph& target,
        const PatternHypergraph& pattern,
        VertexId anchor_vertex,
        std::size_t search_radius) const;

    /**
     * Find all matches of pattern in target hypergraph.
     * This can be expensive for large hypergraphs.
     */
    std::vector<PatternMatch> find_all_matches(
        const Hypergraph& target,
        const PatternHypergraph& pattern) const;

    /**
     * Check if pattern matches at specific location with given assignment.
     */
    bool matches_at(
        const Hypergraph& target,
        const PatternHypergraph& pattern,
        const VariableAssignment& assignment) const;

    /**
     * Find first match of pattern around anchor vertex (faster for single match).
     */
    std::optional<PatternMatch> find_first_match_around(
        const Hypergraph& target,
        const PatternHypergraph& pattern,
        VertexId anchor_vertex,
        std::size_t search_radius) const;
};

} // namespace hypergraph

#endif // HYPERGRAPH_PATTERN_MATCHING_HPP