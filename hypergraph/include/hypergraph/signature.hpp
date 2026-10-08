#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>
#include <cstring>

#include "types.hpp"
#include "zero_value_init.hpp"
#include "hgcommon/signature_core.hpp"

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// Constants
// =============================================================================

using hgcommon::MAX_ARITY;

// =============================================================================
// EdgeSignature
// =============================================================================
// Describes the vertex repetition pattern of an edge.
// Since we have no vertex labels, this is our analog to HGMatch's label-based
// signature partitioning.
//
// Examples:
//   Edge {3, 3, 4} → Signature [0, 0, 1] (positions 0,1 same; position 2 different)
//   Edge {5, 6, 8} → Signature [0, 1, 2] (all positions different)
//   Edge {1, 1, 1} → Signature [0, 0, 0] (all positions same)
//   Edge {a, b, a} → Signature [0, 1, 0] (positions 0,2 same; position 1 different)

struct EdgeSignature {
    // Value-initialised, as the device port's is (hg_gpu/edge_signature.hpp): a signature is
    // copied whole, and the bytes past `arity` are then a defined value rather than
    // indeterminate ones every copy reads.
    // Both initialisers are zero, and zero_value_init_v says so below.
    uint8_t arity = 0;
    uint8_t pattern[MAX_ARITY] = {};  // Vertex repetition pattern

    // Compute signature from edge vertices
    static EdgeSignature from_edge(const VertexId* vertices, uint8_t arity);

    // Compute signature from pattern variable indices
    // Pattern edge stores variable indices directly, so we compute signature
    // from the variable repetition pattern
    // Body in signature.cpp: runs once per rule at registration, never per state.
    static EdgeSignature from_pattern(const uint8_t* vars, uint8_t arity);

    // Compute hash for signature (for use in ConcurrentMap)
    uint64_t hash() const;

    bool operator==(const EdgeSignature& other) const;
    bool operator!=(const EdgeSignature& other) const;

    // Number of distinct vertices (max label + 1)
    uint8_t num_distinct() const;
};
template <> inline constexpr bool zero_value_init_v<EdgeSignature> = true;

// =============================================================================
// Signature Compatibility
// =============================================================================
// Check if a data edge signature is compatible with a pattern signature.
//
// Compatibility rules:
// - Pattern [0, 1] matches data [0, 0] and [0, 1] (non-distinct variables)
// - Pattern [0, 0] matches data [0, 0] only (same-variable constraint)
//
// The rule: wherever the pattern has the same variable at two positions,
// the data edge must have the same vertex at those positions.
// But if the pattern has different variables, the data edge can have
// either the same or different vertices (non-distinct variable semantics).

bool signature_compatible(const EdgeSignature& data_sig,
                          const EdgeSignature& pattern_sig);

}  // namespace engine
}  // namespace HG_NAMESPACE