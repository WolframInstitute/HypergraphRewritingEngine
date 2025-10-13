#ifndef HYPERGRAPH_PATTERN_MATCHING_HPP
#define HYPERGRAPH_PATTERN_MATCHING_HPP

#include <hypergraph/hypergraph.hpp>
#include <hypergraph/canonicalization.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <set>
#include <algorithm>
#include <functional>

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

// Forward declaration for EdgeSignatureIndex (defined later in this file)
class EdgeSignatureIndex;

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
     * Uses EdgeSignatureIndex for fast candidate filtering via inverted index.
     */
    bool match_remaining_edges(
        const Hypergraph& target,
        const std::vector<PatternEdge>& remaining_pattern_edges,
        const std::unordered_set<EdgeId>& available_edges,
        std::vector<EdgeId>& matched_edges,
        VariableAssignment& assignment,
        const EdgeSignatureIndex& edge_index) const;

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

/**
 * Pattern signature representing variable arrangement (e.g., {0,0}, {0,1}, {0,1,2}).
 * This is used for partitioning edges by their variable patterns.
 */
struct PatternSignature {
    std::vector<std::size_t> variable_pattern;  // e.g., {0,0} for self-loop, {0,1} for distinct

    PatternSignature() = default;
    PatternSignature(const std::vector<std::size_t>& pattern);

    bool operator==(const PatternSignature& other) const;
    bool operator<(const PatternSignature& other) const;
    std::size_t hash() const;
};

/**
 * Signature of a hyperedge for fast pattern matching.
 * Includes vertex incidence information as per HGMatch paper.
 * Supports both concrete edges and pattern edges with variables.
 */
class EdgeSignature {
private:
    std::multiset<VertexId> concrete_labels_;  // Concrete vertex IDs (used as labels)
    std::size_t num_variables_;                // Number of variable vertices
    std::size_t arity_;                        // Total number of vertices

    // HGMatch-style vertex incidence information
    std::unordered_map<VertexId, std::vector<EdgeId>> vertex_incidence_;  // vertex -> incident edges
    std::unordered_map<VertexId, std::size_t> vertex_degrees_;             // vertex -> degree
    std::vector<std::size_t> variable_positions_;                          // Which positions are variables

public:
    EdgeSignature();

    /**
     * Create signature from a concrete hyperedge with incidence information.
     * Uses vertex IDs directly as labels.
     */
    static EdgeSignature from_concrete_edge(const Hyperedge& edge,
                                           const Hypergraph* hypergraph = nullptr);

    /**
     * Create signature from a pattern edge (may contain variables).
     * Uses vertex IDs directly as labels.
     */
    static EdgeSignature from_pattern_edge(const PatternEdge& edge);

    /**
     * Check if this signature is compatible with a pattern signature.
     */
    bool is_compatible_with_pattern(const EdgeSignature& pattern) const;

    bool operator==(const EdgeSignature& other) const;
    bool operator!=(const EdgeSignature& other) const;

    // Getters
    std::size_t arity() const;
    std::size_t num_variables() const;
    const std::multiset<VertexId>& concrete_labels() const;
    const std::unordered_map<VertexId, std::vector<EdgeId>>& vertex_incidence() const;
    const std::unordered_map<VertexId, std::size_t>& vertex_degrees() const;
    const std::vector<std::size_t>& variable_positions() const;

    /**
     * Generate all applicable pattern signatures for this concrete edge.
     */
    std::vector<PatternSignature> generate_pattern_signatures() const;

    /**
     * Get a hash value for use in unordered containers.
     */
    std::size_t hash() const;

private:
    /**
     * Check if a variable pattern is compatible with concrete labels.
     */
    bool is_pattern_compatible(const std::vector<std::size_t>& pattern,
                              const std::vector<VertexId>& sorted_labels) const;
};

/**
 * Multi-level index for edge signatures with pattern-specific partitioning.
 * Edges are partitioned by pattern signatures (e.g., {0,0}, {0,1}) for fast lookup.
 */
class EdgeSignatureIndex {
private:
    // Level 1: Partition by arity
    struct ArityPartition {
        // Level 2: Partition by pattern signature (e.g., {0,0}, {0,1})
        struct PatternPartition {
            // Level 3: Map from concrete signature to edge IDs
            std::unordered_map<std::size_t, std::vector<EdgeId>> signature_to_edges;

            // Store actual signatures for compatibility checking
            std::unordered_map<std::size_t, EdgeSignature> hash_to_signature;
        };

        std::unordered_map<std::size_t, PatternPartition> by_pattern_signature;
    };

    std::unordered_map<std::size_t, ArityPartition> by_arity_;

    // HGMatch-style inverted hyperedge index: he(v)
    // Maps: vertex_id -> [edge_ids]
    // Enables fast lookup: "which edges contain vertex v?"
    // Signature filtering happens after intersection
    std::unordered_map<VertexId, std::vector<EdgeId>> inverted_index_;

    // Quick lookup map: edge_id -> signature_hash for efficient signature retrieval
    std::unordered_map<EdgeId, std::size_t> edge_to_signature_hash_;

public:
    /**
     * Add an edge with its signature to the index.
     * Also updates inverted index for HGMatch-style candidate generation.
     */
    void add_edge(EdgeId edge_id, const EdgeSignature& signature);

    /**
     * Find all edges compatible with a pattern signature.
     */
    std::vector<EdgeId> find_compatible_edges(const EdgeSignature& pattern) const;

    /**
     * Get edges with exact signature match (for concrete edges).
     */
    std::vector<EdgeId> get_edges_with_signature(const EdgeSignature& signature) const;

    /**
     * HGMatch Algorithm 4: Get all edges incident to vertex v.
     * Returns he(v) from inverted index.
     * @param vertex Vertex ID to lookup
     * @return Const reference to vector of edge IDs incident to vertex (optimization #3)
     */
    const std::vector<EdgeId>& get_incident_edges(VertexId vertex) const;

    /**
     * HGMatch Algorithm 4: Generate candidates by intersecting incident edge sets.
     * For vertices in incident_vertices, compute ∩ he(vi).
     * Then filters by available_edges constraint.
     * @param incident_vertices Set of vertices to intersect over
     * @param available_edges Only consider edges in this set (for partial matches)
     * @return Edges incident to ALL vertices and in available_edges
     */
    std::vector<EdgeId> generate_candidates_by_intersection(
        const std::vector<VertexId>& incident_vertices,
        const std::unordered_set<EdgeId>& available_edges) const;

    /**
     * Clear the index (including inverted index).
     */
    void clear();

    /**
     * Get total number of indexed edges.
     */
    std::size_t size() const;
};

} // namespace hypergraph

#endif // HYPERGRAPH_PATTERN_MATCHING_HPP