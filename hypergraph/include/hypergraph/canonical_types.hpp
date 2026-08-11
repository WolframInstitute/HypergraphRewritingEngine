#pragma once
#include "hgcommon/namespace.hpp"

#include <vector>
#include <unordered_map>
#include <cstddef>

#include "types.hpp"

namespace HG_NAMESPACE {
namespace engine {

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

}  // namespace engine
}  // namespace HG_NAMESPACE