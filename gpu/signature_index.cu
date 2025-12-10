#include "signature_index.cuh"
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace hypergraph::gpu {

// =============================================================================
// Host-side Signature Computation (mirrors GPU version)
// =============================================================================

struct HostEdgeSignature {
    uint8_t arity;
    uint8_t pattern[MAX_EDGE_ARITY];

    static HostEdgeSignature from_edge(const std::vector<uint32_t>& vertices) {
        HostEdgeSignature sig;
        sig.arity = static_cast<uint8_t>(vertices.size());

        if (sig.arity == 0) return sig;

        uint8_t next_label = 0;
        std::vector<uint32_t> seen;
        std::vector<uint8_t> labels;

        for (uint8_t i = 0; i < sig.arity; ++i) {
            uint32_t v = vertices[i];

            uint8_t label = next_label;
            for (size_t j = 0; j < seen.size(); ++j) {
                if (seen[j] == v) {
                    label = labels[j];
                    break;
                }
            }

            if (label == next_label) {
                seen.push_back(v);
                labels.push_back(next_label);
                next_label++;
            }

            sig.pattern[i] = label;
        }

        return sig;
    }

    uint64_t hash() const {
        uint64_t h = 14695981039346656037ULL;
        h ^= arity;
        h *= 1099511628211ULL;
        for (uint8_t i = 0; i < arity; ++i) {
            h ^= pattern[i];
            h *= 1099511628211ULL;
        }
        return h;
    }
};

// =============================================================================
// GPUSignatureIndexBuilder Implementation
// =============================================================================

void GPUSignatureIndexBuilder::init(uint32_t max_edges, uint32_t max_vertices) {
    max_edges_ = max_edges;
    max_vertices_ = max_vertices;
}

void GPUSignatureIndexBuilder::destroy() {
    h_buckets_.clear();
    h_edge_ids_.clear();
    h_inv_offsets_.clear();
    h_inv_edges_.clear();
}

void GPUSignatureIndexBuilder::build(
    const std::vector<std::vector<uint32_t>>& edges,
    DeviceSignatureIndex& sig_index,
    DeviceInvertedVertexIndex& inv_index
) {
    // =========================================================================
    // Build Signature Index
    // =========================================================================

    // Group edges by signature hash
    std::unordered_map<uint64_t, std::vector<uint32_t>> sig_to_edges;

    for (size_t eid = 0; eid < edges.size(); ++eid) {
        HostEdgeSignature sig = HostEdgeSignature::from_edge(edges[eid]);
        uint64_t hash = sig.hash();
        if (hash == 0) hash = 1;  // 0 reserved for empty buckets
        sig_to_edges[hash].push_back(static_cast<uint32_t>(eid));
    }

    // Build hash table (open addressed, ~2x load factor)
    uint32_t num_signatures = static_cast<uint32_t>(sig_to_edges.size());
    uint32_t num_buckets = std::max(num_signatures * 2, 16u);

    h_buckets_.resize(num_buckets);
    for (auto& bucket : h_buckets_) {
        bucket.sig_hash = 0;
        bucket.edge_start = 0;
        bucket.edge_count = 0;
    }

    // Build flat edge array
    h_edge_ids_.clear();
    h_edge_ids_.reserve(edges.size());

    for (const auto& [hash, edge_list] : sig_to_edges) {
        // Find bucket (linear probing)
        uint32_t idx = hash % num_buckets;
        while (h_buckets_[idx].sig_hash != 0) {
            idx = (idx + 1) % num_buckets;
        }

        h_buckets_[idx].sig_hash = hash;
        h_buckets_[idx].edge_start = static_cast<uint32_t>(h_edge_ids_.size());
        h_buckets_[idx].edge_count = static_cast<uint32_t>(edge_list.size());

        for (uint32_t eid : edge_list) {
            h_edge_ids_.push_back(eid);
        }
    }

    // Upload to device
    sig_index.num_buckets = num_buckets;
    sig_index.num_edges = static_cast<uint32_t>(edges.size());

    CUDA_CHECK(cudaMalloc(&sig_index.buckets, num_buckets * sizeof(SignatureBucket)));
    CUDA_CHECK(cudaMemcpy(sig_index.buckets, h_buckets_.data(),
                          num_buckets * sizeof(SignatureBucket), cudaMemcpyHostToDevice));

    if (!h_edge_ids_.empty()) {
        CUDA_CHECK(cudaMalloc(&sig_index.edge_ids, h_edge_ids_.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(sig_index.edge_ids, h_edge_ids_.data(),
                              h_edge_ids_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    } else {
        sig_index.edge_ids = nullptr;
    }

    // =========================================================================
    // Build Inverted Vertex Index
    // =========================================================================

    // Find max vertex
    uint32_t max_vertex = 0;
    for (const auto& edge : edges) {
        for (uint32_t v : edge) {
            max_vertex = std::max(max_vertex, v + 1);
        }
    }

    // Count edges per vertex
    std::vector<uint32_t> vertex_counts(max_vertex, 0);
    for (const auto& edge : edges) {
        for (uint32_t v : edge) {
            vertex_counts[v]++;
        }
    }

    // Build row offsets (prefix sum)
    h_inv_offsets_.resize(max_vertex + 1);
    h_inv_offsets_[0] = 0;
    for (uint32_t v = 0; v < max_vertex; ++v) {
        h_inv_offsets_[v + 1] = h_inv_offsets_[v] + vertex_counts[v];
    }

    uint32_t total_entries = h_inv_offsets_[max_vertex];

    // Build edge arrays
    h_inv_edges_.resize(total_entries);
    std::fill(vertex_counts.begin(), vertex_counts.end(), 0);  // Reuse as write indices

    for (size_t eid = 0; eid < edges.size(); ++eid) {
        for (uint32_t v : edges[eid]) {
            uint32_t offset = h_inv_offsets_[v] + vertex_counts[v];
            h_inv_edges_[offset] = static_cast<uint32_t>(eid);
            vertex_counts[v]++;
        }
    }

    // Upload to device
    inv_index.num_vertices = max_vertex;
    inv_index.num_entries = total_entries;

    CUDA_CHECK(cudaMalloc(&inv_index.row_offsets, (max_vertex + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(inv_index.row_offsets, h_inv_offsets_.data(),
                          (max_vertex + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    if (total_entries > 0) {
        CUDA_CHECK(cudaMalloc(&inv_index.edge_ids, total_entries * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(inv_index.edge_ids, h_inv_edges_.data(),
                              total_entries * sizeof(uint32_t), cudaMemcpyHostToDevice));
    } else {
        inv_index.edge_ids = nullptr;
    }
}

}  // namespace hypergraph::gpu
