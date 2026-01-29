#ifndef HYPERGRAPH_CANONICAL_TYPES_HPP
#define HYPERGRAPH_CANONICAL_TYPES_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <cstdint>

#include "types.hpp"

namespace hypergraph {

constexpr VertexId INVALID_VERTEX = INVALID_ID;

struct CanonicalForm {
    std::vector<std::vector<VertexId>> edges;
    VertexId vertex_count;

    bool operator==(const CanonicalForm& other) const {
        return vertex_count == other.vertex_count && edges == other.edges;
    }

    bool operator!=(const CanonicalForm& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "CanonicalForm(vertices=" << vertex_count << ", edges=[";
        for (std::size_t i = 0; i < edges.size(); ++i) {
            oss << "[";
            for (std::size_t j = 0; j < edges[i].size(); ++j) {
                oss << edges[i][j];
                if (j < edges[i].size() - 1) oss << ",";
            }
            oss << "]";
            if (i < edges.size() - 1) oss << ", ";
        }
        oss << "])";
        return oss.str();
    }
};

struct VertexMapping {
    std::unordered_map<VertexId, VertexId> original_to_canonical;
    std::vector<VertexId> canonical_to_original;

    std::unordered_map<std::size_t, std::size_t> original_edge_to_canonical;
    std::vector<std::size_t> canonical_edge_to_original;

    VertexId map_vertex(VertexId original) const {
        auto it = original_to_canonical.find(original);
        return (it != original_to_canonical.end()) ? it->second : INVALID_VERTEX;
    }

    VertexId get_original(VertexId canonical) const {
        return (canonical < canonical_to_original.size()) ?
               canonical_to_original[canonical] : INVALID_VERTEX;
    }

    std::size_t map_edge(std::size_t original_idx) const {
        auto it = original_edge_to_canonical.find(original_idx);
        return (it != original_edge_to_canonical.end()) ? it->second : static_cast<std::size_t>(-1);
    }
};

struct CanonicalizationResult {
    CanonicalForm canonical_form;
    VertexMapping vertex_mapping;

    static bool are_isomorphic(const CanonicalizationResult& a, const CanonicalizationResult& b) {
        return a.canonical_form == b.canonical_form;
    }
};

} // namespace hypergraph

#endif // HYPERGRAPH_CANONICAL_TYPES_HPP
