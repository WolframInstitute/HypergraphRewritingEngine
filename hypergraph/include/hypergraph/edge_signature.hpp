#ifndef HYPERGRAPH_EDGE_SIGNATURE_HPP
#define HYPERGRAPH_EDGE_SIGNATURE_HPP

#include <hypergraph/hypergraph.hpp>
#include <hypergraph/pattern_matching.hpp>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>

namespace hypergraph {

// Type for vertex labels (can be customized based on application needs)
using VertexLabel = std::size_t;

/**
 * Pattern signature representing variable arrangement (e.g., {0,0}, {0,1}, {0,1,2}).
 * This is used for partitioning edges by their variable patterns.
 */
struct PatternSignature {
    std::vector<std::size_t> variable_pattern;  // e.g., {0,0} for self-loop, {0,1} for distinct
    
    PatternSignature() = default;
    PatternSignature(const std::vector<std::size_t>& pattern) : variable_pattern(pattern) {}
    
    bool operator==(const PatternSignature& other) const {
        return variable_pattern == other.variable_pattern;
    }
    
    bool operator<(const PatternSignature& other) const {
        return variable_pattern < other.variable_pattern;
    }
    
    std::size_t hash() const {
        std::size_t h = 0;
        for (auto var : variable_pattern) {
            h ^= std::hash<std::size_t>{}(var) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

/**
 * Signature of a hyperedge for fast pattern matching.
 * Includes vertex incidence information as per HGMatch paper.
 * Supports both concrete edges and pattern edges with variables.
 */
class EdgeSignature {
private:
    std::multiset<VertexLabel> concrete_labels_;  // Concrete vertex labels
    std::size_t num_variables_;                   // Number of variable vertices
    std::size_t arity_;                           // Total number of vertices
    
    // HGMatch-style vertex incidence information
    std::unordered_map<VertexId, std::vector<EdgeId>> vertex_incidence_;  // vertex -> incident edges
    std::unordered_map<VertexId, std::size_t> vertex_degrees_;             // vertex -> degree
    std::vector<std::size_t> variable_positions_;                          // Which positions are variables
    
public:
    EdgeSignature() : num_variables_(0), arity_(0) {}
    
    /**
     * Create signature from a concrete hyperedge with incidence information.
     */
    static EdgeSignature from_concrete_edge(const Hyperedge& edge, 
                                           const std::function<VertexLabel(VertexId)>& label_func,
                                           const Hypergraph* hypergraph = nullptr) {
        EdgeSignature sig;
        sig.arity_ = edge.arity();
        sig.num_variables_ = 0;
        
        for (VertexId v : edge.vertices()) {
            VertexLabel label = label_func(v);
            sig.concrete_labels_.insert(label);
            
            // Add vertex incidence information as per HGMatch paper
            if (hypergraph) {
                auto incident_edges = hypergraph->edges_containing(v);
                sig.vertex_incidence_[v] = incident_edges;
                sig.vertex_degrees_[v] = incident_edges.size();
            }
        }
        
        return sig;
    }
    
    /**
     * Create signature from a pattern edge (may contain variables).
     */
    static EdgeSignature from_pattern_edge(const PatternEdge& edge,
                                          const std::function<VertexLabel(VertexId)>& label_func) {
        EdgeSignature sig;
        sig.arity_ = edge.arity();
        sig.num_variables_ = 0;
        
        for (std::size_t pos = 0; pos < edge.vertices.size(); ++pos) {
            const PatternVertex& pv = edge.vertices[pos];
            if (pv.is_variable()) {
                sig.num_variables_++;
                sig.variable_positions_.push_back(pos);
            } else {
                sig.concrete_labels_.insert(label_func(pv.id));
            }
        }
        
        return sig;
    }
    
    /**
     * Check if this signature is compatible with a pattern signature.
     * Pattern can match if:
     * 1. Arity matches exactly
     * 2. All concrete labels in pattern exist in this signature
     * 3. Remaining positions can be filled by variables
     */
    bool is_compatible_with_pattern(const EdgeSignature& pattern) const {
        // Arity must match exactly
        if (arity_ != pattern.arity_) {
            return false;
        }
        
        // Check if all pattern's concrete labels can be matched
        std::multiset<VertexLabel> remaining_labels = concrete_labels_;
        
        for (VertexLabel label : pattern.concrete_labels_) {
            auto it = remaining_labels.find(label);
            if (it == remaining_labels.end()) {
                return false;  // Pattern requires label we don't have
            }
            remaining_labels.erase(it);
        }
        
        // Check if variables can cover remaining positions
        std::size_t unmatched_positions = remaining_labels.size();
        return unmatched_positions <= pattern.num_variables_;
    }
    
    /**
     * Check exact equality (for concrete edges).
     */
    bool operator==(const EdgeSignature& other) const {
        return arity_ == other.arity_ &&
               num_variables_ == other.num_variables_ &&
               concrete_labels_ == other.concrete_labels_;
    }
    
    bool operator!=(const EdgeSignature& other) const {
        return !(*this == other);
    }
    
    // Getters
    std::size_t arity() const { return arity_; }
    std::size_t num_variables() const { return num_variables_; }
    const std::multiset<VertexLabel>& concrete_labels() const { return concrete_labels_; }
    const std::unordered_map<VertexId, std::vector<EdgeId>>& vertex_incidence() const { return vertex_incidence_; }
    const std::unordered_map<VertexId, std::size_t>& vertex_degrees() const { return vertex_degrees_; }
    const std::vector<std::size_t>& variable_positions() const { return variable_positions_; }
    
    /**
     * Generate all applicable pattern signatures for this concrete edge.
     * E.g., edge {3,3} generates {0,0} and {0,1}, edge {3,4} generates only {0,1}.
     */
    std::vector<PatternSignature> generate_pattern_signatures() const {
        std::vector<PatternSignature> patterns;
        
        if (arity_ == 0) return patterns;
        
        // Convert concrete labels to indices for pattern generation
        std::vector<VertexLabel> sorted_labels(concrete_labels_.begin(), concrete_labels_.end());
        
        // Generate all possible variable patterns for this arity
        std::function<void(std::vector<std::size_t>&, std::size_t)> generate_patterns = 
            [&](std::vector<std::size_t>& current_pattern, std::size_t pos) {
                if (pos == arity_) {
                    // Check if this pattern is compatible with our concrete edge
                    if (is_pattern_compatible(current_pattern, sorted_labels)) {
                        patterns.emplace_back(current_pattern);
                    }
                    return;
                }
                
                // Try reusing existing variables
                std::size_t max_var = 0;
                for (auto v : current_pattern) {
                    max_var = std::max(max_var, v);
                }
                
                for (std::size_t var = 0; var <= max_var + 1; ++var) {
                    current_pattern.push_back(var);
                    generate_patterns(current_pattern, pos + 1);
                    current_pattern.pop_back();
                }
            };
        
        std::vector<std::size_t> current_pattern;
        generate_patterns(current_pattern, 0);
        
        return patterns;
    }
    
    /**
     * Check if a variable pattern is compatible with concrete labels.
     */
    bool is_pattern_compatible(const std::vector<std::size_t>& pattern, 
                              const std::vector<VertexLabel>& sorted_labels) const {
        if (pattern.size() != sorted_labels.size()) return false;
        
        std::unordered_map<std::size_t, VertexLabel> var_binding;
        std::unordered_set<VertexLabel> used_labels;
        
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            std::size_t var = pattern[i];
            VertexLabel label = sorted_labels[i];
            
            auto it = var_binding.find(var);
            if (it != var_binding.end()) {
                // Variable already bound - must match
                if (it->second != label) return false;
            } else {
                // New variable binding
                if (used_labels.count(label)) {
                    // Label already used by another variable - incompatible
                    return false;
                }
                var_binding[var] = label;
                used_labels.insert(label);
            }
        }
        
        return true;
    }
    
    /**
     * Get a hash value for use in unordered containers.
     */
    std::size_t hash() const {
        std::size_t h = 0;
        h ^= std::hash<std::size_t>{}(arity_) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<std::size_t>{}(num_variables_) + 0x9e3779b9 + (h << 6) + (h >> 2);
        
        for (VertexLabel label : concrete_labels_) {
            h ^= std::hash<VertexLabel>{}(label) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        
        return h;
    }
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
    
public:
    /**
     * Add an edge with its signature to the index.
     * The edge is added to ALL applicable pattern partitions.
     */
    void add_edge(EdgeId edge_id, const EdgeSignature& signature) {
        auto& arity_partition = by_arity_[signature.arity()];
        
        // Generate all pattern signatures for this concrete edge
        auto pattern_signatures = signature.generate_pattern_signatures();
        
        for (const auto& pattern_sig : pattern_signatures) {
            auto& pattern_partition = arity_partition.by_pattern_signature[pattern_sig.hash()];
            
            std::size_t sig_hash = signature.hash();
            pattern_partition.signature_to_edges[sig_hash].push_back(edge_id);
            pattern_partition.hash_to_signature[sig_hash] = signature;
        }
    }
    
    /**
     * Find all edges compatible with a pattern signature.
     * This is the key function for gathering applicable signatures given a partial embedding.
     */
    std::vector<EdgeId> find_compatible_edges(const EdgeSignature& pattern) const {
        std::vector<EdgeId> result;
        
        // Only look at edges with matching arity
        auto arity_it = by_arity_.find(pattern.arity());
        if (arity_it == by_arity_.end()) {
            return result;
        }
        
        const auto& arity_partition = arity_it->second;
        
        // Create pattern signature from the pattern edge
        PatternSignature pattern_sig(pattern.variable_positions());
        auto pattern_it = arity_partition.by_pattern_signature.find(pattern_sig.hash());
        
        if (pattern_it == arity_partition.by_pattern_signature.end()) {
            return result;
        }
        
        const auto& pattern_partition = pattern_it->second;
        
        // Check each signature in this pattern partition
        for (const auto& [sig_hash, edges] : pattern_partition.signature_to_edges) {
            const EdgeSignature& signature = pattern_partition.hash_to_signature.at(sig_hash);
            
            if (signature.is_compatible_with_pattern(pattern)) {
                result.insert(result.end(), edges.begin(), edges.end());
            }
        }
        
        return result;
    }
    
    /**
     * Get edges with exact signature match (for concrete edges).
     */
    std::vector<EdgeId> get_edges_with_signature(const EdgeSignature& signature) const {
        auto arity_it = by_arity_.find(signature.arity());
        if (arity_it == by_arity_.end()) {
            return {};
        }
        
        // Search all pattern partitions for this signature
        std::vector<EdgeId> result;
        for (const auto& [pattern_hash, pattern_partition] : arity_it->second.by_pattern_signature) {
            std::size_t sig_hash = signature.hash();
            auto edges_it = pattern_partition.signature_to_edges.find(sig_hash);
            if (edges_it != pattern_partition.signature_to_edges.end()) {
                result.insert(result.end(), edges_it->second.begin(), edges_it->second.end());
            }
        }
        
        return result;
    }
    
    /**
     * Clear the index.
     */
    void clear() {
        by_arity_.clear();
    }
    
    /**
     * Get total number of indexed edges.
     */
    std::size_t size() const {
        std::size_t total = 0;
        std::unordered_set<EdgeId> counted_edges;  // Avoid double counting
        
        for (const auto& [arity, arity_partition] : by_arity_) {
            for (const auto& [pattern_hash, pattern_partition] : arity_partition.by_pattern_signature) {
                for (const auto& [sig_hash, edges] : pattern_partition.signature_to_edges) {
                    for (EdgeId eid : edges) {
                        counted_edges.insert(eid);
                    }
                }
            }
        }
        
        return counted_edges.size();
    }
};

} // namespace hypergraph

// Hash specializations
namespace std {
template<>
struct hash<hypergraph::EdgeSignature> {
    std::size_t operator()(const hypergraph::EdgeSignature& sig) const {
        return sig.hash();
    }
};

template<>
struct hash<hypergraph::PatternSignature> {
    std::size_t operator()(const hypergraph::PatternSignature& sig) const {
        return sig.hash();
    }
};
}

#endif // HYPERGRAPH_EDGE_SIGNATURE_HPP