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

    bool operator==(const CanonicalForm& other) const;
    bool operator!=(const CanonicalForm& other) const;

};

struct VertexMapping {
    std::unordered_map<VertexId, VertexId> original_to_canonical;
    std::vector<VertexId> canonical_to_original;

    std::unordered_map<std::size_t, std::size_t> original_edge_to_canonical;
    std::vector<std::size_t> canonical_edge_to_original;

    VertexId map_vertex(VertexId original) const;
    VertexId get_original(VertexId canonical) const;
    std::size_t map_edge(std::size_t original_idx) const;
};

struct CanonicalizationResult {
    CanonicalForm canonical_form;
    VertexMapping vertex_mapping;

    static bool are_isomorphic(const CanonicalizationResult& a, const CanonicalizationResult& b);
};

}  // namespace engine
}  // namespace HG_NAMESPACE