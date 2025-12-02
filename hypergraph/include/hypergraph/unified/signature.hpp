#pragma once

#include <cstdint>
#include <cstring>

#include "types.hpp"

// Debug logging - enabled with -DENABLE_DEBUG_OUTPUT
#ifdef ENABLE_DEBUG_OUTPUT
    #include <cstdio>
    #define DEBUG_LOG_SIG(fmt, ...) printf("[DEBUG][SIG] " fmt "\n", ##__VA_ARGS__)
#else
    #define DEBUG_LOG_SIG(fmt, ...) ((void)0)
#endif

namespace hypergraph::unified {

// =============================================================================
// Constants
// =============================================================================

constexpr uint8_t MAX_ARITY = 16;

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
    static EdgeSignature from_edge(const VertexId* vertices, uint8_t arity) {
        EdgeSignature sig;
        sig.arity = arity;
        std::memset(sig.pattern, 0, MAX_ARITY);

        if (arity == 0) return sig;

        // Map first occurrence of each vertex to incrementing label
        uint8_t next_label = 0;
        VertexId seen[MAX_ARITY];
        uint8_t labels[MAX_ARITY];

        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = vertices[i];

            // Check if vertex already seen
            uint8_t label = next_label;
            for (uint8_t j = 0; j < next_label; ++j) {
                if (seen[j] == v) {
                    label = labels[j];
                    break;
                }
            }

            // If new vertex, assign new label
            if (label == next_label) {
                seen[next_label] = v;
                labels[next_label] = next_label;
                next_label++;
            }

            sig.pattern[i] = label;
        }

        return sig;
    }

    // Compute signature from pattern variable indices
    // Pattern edge stores variable indices directly, so we compute signature
    // from the variable repetition pattern
    static EdgeSignature from_pattern(const uint8_t* vars, uint8_t arity) {
        EdgeSignature sig;
        sig.arity = arity;
        std::memset(sig.pattern, 0, MAX_ARITY);

        if (arity == 0) return sig;

        // Map first occurrence of each variable to incrementing label
        uint8_t next_label = 0;
        uint8_t seen_vars[MAX_ARITY];
        uint8_t var_labels[MAX_ARITY];

        for (uint8_t i = 0; i < arity; ++i) {
            uint8_t var = vars[i];

            // Check if variable already seen
            uint8_t label = next_label;
            for (uint8_t j = 0; j < next_label; ++j) {
                if (seen_vars[j] == var) {
                    label = var_labels[j];
                    break;
                }
            }

            // If new variable, assign new label
            if (label == next_label) {
                seen_vars[next_label] = var;
                var_labels[next_label] = next_label;
                next_label++;
            }

            sig.pattern[i] = label;
        }

        return sig;
    }

    // Compute hash for signature (for use in ConcurrentMap)
    uint64_t hash() const {
        // FNV-1a hash
        uint64_t h = 14695981039346656037ULL;
        h ^= arity;
        h *= 1099511628211ULL;
        for (uint8_t i = 0; i < arity; ++i) {
            h ^= pattern[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    bool operator==(const EdgeSignature& other) const {
        if (arity != other.arity) return false;
        for (uint8_t i = 0; i < arity; ++i) {
            if (pattern[i] != other.pattern[i]) return false;
        }
        return true;
    }

    bool operator!=(const EdgeSignature& other) const {
        return !(*this == other);
    }

    // Number of distinct vertices (max label + 1)
    uint8_t num_distinct() const {
        uint8_t max_label = 0;
        for (uint8_t i = 0; i < arity; ++i) {
            if (pattern[i] > max_label) max_label = pattern[i];
        }
        return arity > 0 ? max_label + 1 : 0;
    }
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

inline bool signature_compatible(const EdgeSignature& data_sig,
                                 const EdgeSignature& pattern_sig) {
    // Arities must match
    if (data_sig.arity != pattern_sig.arity) return false;

    // For each pair of positions in the pattern:
    // If pattern has same variable, data must have same vertex
    for (uint8_t i = 0; i < pattern_sig.arity; ++i) {
        for (uint8_t j = i + 1; j < pattern_sig.arity; ++j) {
            if (pattern_sig.pattern[i] == pattern_sig.pattern[j]) {
                // Pattern requires same variable at positions i and j
                // Data edge must have same vertex at those positions
                if (data_sig.pattern[i] != data_sig.pattern[j]) {
                    return false;
                }
            }
            // Note: if pattern has different variables, we don't constrain data
            // This implements non-distinct variable semantics
        }
    }

    return true;
}

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
inline void enumerate_partitions_recursive(
    uint8_t num_classes,
    uint8_t current_class,
    uint8_t* merged_to,
    uint8_t max_partition_used,  // highest partition id used so far
    const EdgeSignature& pattern_sig,
    const uint8_t* var_to_class,  // pattern variable -> class id
    SignatureVisitor visitor,
    void* user_data
) {
    if (current_class == num_classes) {
        // All classes assigned - generate signature
        EdgeSignature result;
        result.arity = pattern_sig.arity;
        std::memset(result.pattern, 0, MAX_ARITY);

        for (uint8_t i = 0; i < pattern_sig.arity; ++i) {
            uint8_t pvar = pattern_sig.pattern[i];
            uint8_t pclass = var_to_class[pvar];
            result.pattern[i] = merged_to[pclass];
        }

        DEBUG_LOG_SIG("Generated signature: [%d, %d] from merged_to [%d, %d]",
                      result.pattern[0], result.pattern[1],
                      merged_to[0], num_classes > 1 ? merged_to[1] : 0);

        visitor(result, user_data);
        return;
    }

    // Try assigning current_class to each existing partition (0..max_partition_used)
    // or to a new partition (max_partition_used + 1)
    for (uint8_t partition = 0; partition <= max_partition_used + 1; ++partition) {
        merged_to[current_class] = partition;

        uint8_t new_max = max_partition_used;
        if (partition > max_partition_used) {
            new_max = partition;
        }

        enumerate_partitions_recursive(
            num_classes, current_class + 1, merged_to, new_max,
            pattern_sig, var_to_class, visitor, user_data
        );
    }
}

}  // namespace detail

inline void enumerate_compatible_signatures(
    const EdgeSignature& pattern_sig,
    SignatureVisitor visitor,
    void* user_data
) {
    if (pattern_sig.arity == 0) {
        visitor(pattern_sig, user_data);
        return;
    }

    // Find equivalence classes in pattern (positions that share a variable)
    // Pattern variable → class id
    uint8_t var_to_class[MAX_ARITY];
    std::memset(var_to_class, 0xFF, MAX_ARITY);

    uint8_t num_classes = 0;

    for (uint8_t i = 0; i < pattern_sig.arity; ++i) {
        uint8_t pvar = pattern_sig.pattern[i];
        if (var_to_class[pvar] == 0xFF) {
            var_to_class[pvar] = num_classes;
            num_classes++;
        }
    }

    DEBUG_LOG_SIG("enumerate_compatible_signatures: arity=%d, num_classes=%d",
                  pattern_sig.arity, num_classes);

    // Enumerate all set partitions of {0, 1, ..., num_classes-1}
    // Each partition represents a way that distinct pattern variables can
    // collapse to the same data vertex
    uint8_t merged_to[MAX_ARITY];
    std::memset(merged_to, 0, MAX_ARITY);

    // First class always goes to partition 0
    merged_to[0] = 0;

    if (num_classes == 1) {
        // Only one class - just one signature possible
        EdgeSignature result;
        result.arity = pattern_sig.arity;
        std::memset(result.pattern, 0, MAX_ARITY);
        for (uint8_t i = 0; i < pattern_sig.arity; ++i) {
            result.pattern[i] = 0;  // All map to same partition
        }
        visitor(result, user_data);
        return;
    }

    detail::enumerate_partitions_recursive(
        num_classes, 1, merged_to, 0,  // start from class 1, class 0 is in partition 0
        pattern_sig, var_to_class, visitor, user_data
    );
}

// Count compatible signatures (useful for matching order estimation)
inline uint32_t count_compatible_signatures(const EdgeSignature& pattern_sig) {
    // Bell number B(n) where n = number of distinct variables in pattern
    // For small n (up to MAX_ARITY = 16), we can precompute

    // First, find number of distinct pattern variables
    uint8_t num_distinct = pattern_sig.num_distinct();

    // Bell numbers B(0) to B(16)
    static const uint32_t bell[] = {
        1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147,
        115975, 678570, 4213597, 27644437, 190899322, 1382958545, 0  // B(16) overflows uint32
    };

    return num_distinct < 16 ? bell[num_distinct] : UINT32_MAX;
}

}  // namespace hypergraph::unified
