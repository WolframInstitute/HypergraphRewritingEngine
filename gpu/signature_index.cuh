#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "types.cuh"

namespace hypergraph::gpu {

// =============================================================================
// Edge Signature (GPU version)
// =============================================================================
// Describes vertex repetition pattern - analog to unified/signature.hpp
// Example: Edge {3,3,4} → pattern [0,0,1]
// Example: Edge {a,b,a} → pattern [0,1,0]

struct DeviceEdgeSignature {
    uint8_t arity;
    uint8_t pattern[MAX_EDGE_ARITY];

    // Compute signature from edge vertices
    __device__ static DeviceEdgeSignature from_edge(
        const uint32_t* vertices,
        uint8_t arity
    ) {
        DeviceEdgeSignature sig;
        sig.arity = arity;

        if (arity == 0) return sig;

        // Map first occurrence of each vertex to incrementing label
        uint8_t next_label = 0;
        uint32_t seen[MAX_EDGE_ARITY];
        uint8_t labels[MAX_EDGE_ARITY];

        for (uint8_t i = 0; i < arity; ++i) {
            uint32_t v = vertices[i];

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

    // Compute signature from pattern variables (for rules)
    __device__ static DeviceEdgeSignature from_pattern(
        const uint8_t* vars,
        uint8_t arity
    ) {
        DeviceEdgeSignature sig;
        sig.arity = arity;

        if (arity == 0) return sig;

        uint8_t next_label = 0;
        uint8_t seen_vars[MAX_EDGE_ARITY];
        uint8_t var_labels[MAX_EDGE_ARITY];

        for (uint8_t i = 0; i < arity; ++i) {
            uint8_t var = vars[i];

            uint8_t label = next_label;
            for (uint8_t j = 0; j < next_label; ++j) {
                if (seen_vars[j] == var) {
                    label = var_labels[j];
                    break;
                }
            }

            if (label == next_label) {
                seen_vars[next_label] = var;
                var_labels[next_label] = next_label;
                next_label++;
            }

            sig.pattern[i] = label;
        }

        return sig;
    }

    // Compute hash for signature
    __device__ uint64_t hash() const {
        uint64_t h = 14695981039346656037ULL;
        h ^= arity;
        h *= 1099511628211ULL;
        for (uint8_t i = 0; i < arity; ++i) {
            h ^= pattern[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Check equality
    __device__ bool operator==(const DeviceEdgeSignature& other) const {
        if (arity != other.arity) return false;
        for (uint8_t i = 0; i < arity; ++i) {
            if (pattern[i] != other.pattern[i]) return false;
        }
        return true;
    }

    // Number of distinct vertices in signature
    __device__ uint8_t num_distinct() const {
        uint8_t max_label = 0;
        for (uint8_t i = 0; i < arity; ++i) {
            if (pattern[i] > max_label) max_label = pattern[i];
        }
        return arity > 0 ? max_label + 1 : 0;
    }
};

// =============================================================================
// Signature Compatibility (GPU version)
// =============================================================================
// Check if a data edge signature is compatible with a pattern signature.
// Pattern [0,1] matches data [0,0] and [0,1]
// Pattern [0,0] matches data [0,0] only

__device__ inline bool signature_compatible(
    const DeviceEdgeSignature& data_sig,
    const DeviceEdgeSignature& pattern_sig
) {
    if (data_sig.arity != pattern_sig.arity) return false;

    // For each pair of positions in pattern:
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
        }
    }

    return true;
}

// =============================================================================
// Signature Index (GPU version)
// =============================================================================
// Maps signature hashes to edge lists using a simple open-addressed hash table.
// This is simpler than the CPU version - uses arrays instead of lock-free lists.

// Each bucket stores: signature hash, start index in edge array, count
struct SignatureBucket {
    uint64_t sig_hash;      // Signature hash (0 = empty)
    uint32_t edge_start;    // Start index in edges array
    uint32_t edge_count;    // Number of edges with this signature
};

// Index structure
struct DeviceSignatureIndex {
    SignatureBucket* buckets;   // Hash table buckets
    uint32_t* edge_ids;         // Flat array of edge IDs, indexed by bucket
    uint32_t num_buckets;
    uint32_t num_edges;

    // Find bucket for signature hash (linear probing)
    __device__ int32_t find_bucket(uint64_t sig_hash) const {
        if (sig_hash == 0) sig_hash = 1;  // 0 reserved for empty

        uint32_t idx = sig_hash % num_buckets;
        for (uint32_t i = 0; i < num_buckets; ++i) {
            uint32_t probe = (idx + i) % num_buckets;
            if (buckets[probe].sig_hash == sig_hash) {
                return probe;
            }
            if (buckets[probe].sig_hash == 0) {
                return -1;  // Not found
            }
        }
        return -1;
    }

    // Get edges with given signature that are in state
    // Returns count of edges written to output
    __device__ uint32_t get_edges_for_signature(
        uint64_t sig_hash,
        const uint64_t* state_bitmap,
        EdgeId* output,
        uint32_t max_output
    ) const {
        int32_t bucket_idx = find_bucket(sig_hash);
        if (bucket_idx < 0) return 0;

        const SignatureBucket& bucket = buckets[bucket_idx];
        uint32_t count = 0;

        for (uint32_t i = 0; i < bucket.edge_count && count < max_output; ++i) {
            EdgeId eid = edge_ids[bucket.edge_start + i];
            // Check if edge is in state
            uint32_t word = eid / 64;
            uint32_t bit = eid % 64;
            if (state_bitmap[word] & (1ULL << bit)) {
                output[count++] = eid;
            }
        }

        return count;
    }
};

// =============================================================================
// Inverted Vertex Index (GPU version)
// =============================================================================
// Maps vertices to edges containing them (CSR format)

struct DeviceInvertedVertexIndex {
    uint32_t* row_offsets;      // [num_vertices + 1] Start offset per vertex
    uint32_t* edge_ids;         // Flat array of edge IDs
    uint32_t num_vertices;
    uint32_t num_entries;

    // Get edges containing vertex that are in state
    __device__ uint32_t get_edges_for_vertex(
        VertexId vertex,
        const uint64_t* state_bitmap,
        EdgeId* output,
        uint32_t max_output
    ) const {
        if (vertex >= num_vertices) return 0;

        uint32_t start = row_offsets[vertex];
        uint32_t end = row_offsets[vertex + 1];
        uint32_t count = 0;

        for (uint32_t i = start; i < end && count < max_output; ++i) {
            EdgeId eid = edge_ids[i];
            // Check if edge is in state
            uint32_t word = eid / 64;
            uint32_t bit = eid % 64;
            if (state_bitmap[word] & (1ULL << bit)) {
                output[count++] = eid;
            }
        }

        return count;
    }
};

// =============================================================================
// Host-side Index Builder
// =============================================================================

class GPUSignatureIndexBuilder {
public:
    void init(uint32_t max_edges, uint32_t max_vertices);
    void destroy();

    // Build indices from host data
    void build(
        const std::vector<std::vector<uint32_t>>& edges,
        DeviceSignatureIndex& sig_index,
        DeviceInvertedVertexIndex& inv_index
    );

private:
    // Host staging buffers
    std::vector<SignatureBucket> h_buckets_;
    std::vector<uint32_t> h_edge_ids_;
    std::vector<uint32_t> h_inv_offsets_;
    std::vector<uint32_t> h_inv_edges_;

    uint32_t max_edges_;
    uint32_t max_vertices_;
};

}  // namespace hypergraph::gpu
