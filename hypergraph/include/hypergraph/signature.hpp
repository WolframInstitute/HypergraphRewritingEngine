#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>
#include <cstring>

#include "types.hpp"
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
    uint8_t arity;
    uint8_t pattern[MAX_ARITY];  // Vertex repetition pattern

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

// =============================================================================
// Signature Enumeration
// =============================================================================
// Enumerate all data signatures compatible with a pattern signature.
// Useful for index lookups when we need to iterate over compatible partitions.

// Callback type for signature enumeration
using SignatureVisitor = void(*)(const EdgeSignature&, void* user_data);

// Enumerate all compatible data signatures for a pattern signature
// This generates all signatures where:
// - Positions with same pattern variable have same signature label
// - Positions with different pattern variables may have same or different labels
//
// Example: Pattern [0, 1] → generates [0, 0] and [0, 1]
// Example: Pattern [0, 0] → generates [0, 0] only
// Example: Pattern [0, 1, 0] → generates [0, 0, 0], [0, 1, 0]

namespace detail {

// Recursive helper to enumerate all set partitions
// merged_to[i] is the partition id for class i (must be <= max partition seen so far + 1)
void enumerate_partitions_recursive(
    uint8_t num_classes,
    uint8_t current_class,
    uint8_t* merged_to,
    uint8_t max_partition_used,  // highest partition id used so far
    const EdgeSignature& pattern_sig,
    const uint8_t* var_to_class,  // pattern variable -> class id
    SignatureVisitor visitor,
    void* user_data
);

}  // namespace detail

void enumerate_compatible_signatures(
    const EdgeSignature& pattern_sig,
    SignatureVisitor visitor,
    void* user_data
);

// =============================================================================
// CompatibleSignatureCache
// =============================================================================
// Pre-computed cache of compatible data signatures for a pattern signature.
// This avoids repeated Bell number enumeration during matching.
//
// For a pattern like [0, 1] (2 distinct vars), stores the 2 compatible sigs:
// [0, 0] and [0, 1]. These are computed once at rule initialization.

struct CompatibleSignatureCache {
    static constexpr uint8_t MAX_CACHED_SIGS = 64;  // Bell(5)=52, Bell(6)=203

    EdgeSignature signatures[MAX_CACHED_SIGS];
    EdgeSignature source_pattern_sig;   // kept for the overflow fallback (re-enumerate live)
    uint8_t count = 0;
    bool overflowed = false;            // Bell(arity) > MAX_CACHED_SIGS => cache is INCOMPLETE

    CompatibleSignatureCache() = default;

    // Build cache from pattern signature
    // Body in signature.cpp: runs once per rule at registration, never per state.
    static CompatibleSignatureCache from_pattern(const EdgeSignature& pattern_sig);

    // Iterate over cached signatures
    template<typename Visitor>
    void for_each(Visitor&& visit) const {
        for (uint8_t i = 0; i < count; ++i) {
            visit(signatures[i]);
        }
    }
};

}  // namespace engine
}  // namespace HG_NAMESPACE