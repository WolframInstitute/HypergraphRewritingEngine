#include <hypergraph/pattern_matching.hpp>
#include <hypergraph/debug_log.hpp>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace hypergraph {

bool PatternMatcher::edge_matches(const Hyperedge& concrete_edge,
                                 const PatternEdge& pattern_edge,
                                 VariableAssignment& assignment) const {
    if (concrete_edge.arity() != pattern_edge.arity()) {
        return false;
    }

    // Try to match each position
    VariableAssignment temp_assignment = assignment;

    for (std::size_t i = 0; i < pattern_edge.arity(); ++i) {
        const PatternVertex& pattern_vertex = pattern_edge.vertices[i];
        VertexId concrete_vertex = concrete_edge.vertex(i);

        if (pattern_vertex.is_concrete()) {
            if (pattern_vertex.id != concrete_vertex) {
                return false;  // Concrete mismatch
            }
        } else {
            // Variable vertex - check assignment consistency
            if (!temp_assignment.assign(pattern_vertex.id, concrete_vertex)) {
                return false;  // Inconsistent assignment
            }
        }
    }

    // If we get here, the match is valid
    assignment = temp_assignment;
    return true;
}

std::vector<VariableAssignment> PatternMatcher::generate_assignments(
    const Hyperedge& concrete_edge,
    const PatternEdge& pattern_edge,
    const VariableAssignment& base_assignment) const {

    std::vector<VariableAssignment> assignments;

    if (concrete_edge.arity() != pattern_edge.arity()) {
        return assignments;  // Empty - no match possible
    }

    // For simplicity, we'll use a direct matching approach
    // In a full implementation, you might want permutation-based matching
    VariableAssignment assignment = base_assignment;
    if (edge_matches(concrete_edge, pattern_edge, assignment)) {
        assignments.push_back(assignment);
    }

    return assignments;
}

bool PatternMatcher::match_remaining_edges(
    const Hypergraph& target,
    const std::vector<PatternEdge>& remaining_pattern_edges,
    const std::unordered_set<EdgeId>& available_edges,
    std::vector<EdgeId>& matched_edges,
    VariableAssignment& assignment,
    const EdgeSignatureIndex& edge_index) const {

    if (remaining_pattern_edges.empty()) {
        return true;  // All edges matched
    }

    const PatternEdge& current_pattern = remaining_pattern_edges[0];
    std::vector<PatternEdge> remaining(remaining_pattern_edges.begin() + 1,
                                      remaining_pattern_edges.end());

    // HGMatch optimization: Use inverted index to get candidate edges
    // Collect concrete vertices from current assignment that appear in pattern
    // Use unordered_set to automatically deduplicate (important for self-loops)
    std::unordered_set<VertexId> incident_vertices_set;
    for (const auto& pattern_vertex : current_pattern.vertices) {
        if (pattern_vertex.is_variable()) {
            auto resolved = assignment.resolve(pattern_vertex);
            if (resolved.has_value()) {
                incident_vertices_set.insert(resolved.value());
            }
        } else {
            incident_vertices_set.insert(pattern_vertex.id);
        }
    }

    std::vector<VertexId> incident_vertices(incident_vertices_set.begin(),
                                            incident_vertices_set.end());

    // Get candidates via inverted index intersection (HGMatch Algorithm 4)
    std::vector<EdgeId> candidates;
    if (!incident_vertices.empty()) {
        candidates = edge_index.generate_candidates_by_intersection(
            incident_vertices, available_edges);
    } else {
        // No vertices assigned yet - fall back to checking all available edges
        candidates.assign(available_edges.begin(), available_edges.end());
    }

    // Try to match current pattern edge with each candidate
    for (EdgeId edge_id : candidates) {
        const Hyperedge* concrete_edge = target.get_edge(edge_id);
        if (!concrete_edge) continue;

        VariableAssignment temp_assignment = assignment;
        if (edge_matches(*concrete_edge, current_pattern, temp_assignment)) {
            // This edge matches - try to match remaining edges
            matched_edges.push_back(edge_id);

            std::unordered_set<EdgeId> remaining_available = available_edges;
            remaining_available.erase(edge_id);

            if (match_remaining_edges(target, remaining, remaining_available,
                                    matched_edges, temp_assignment, edge_index)) {
                assignment = temp_assignment;
                return true;  // Found complete match
            }

            // Backtrack
            matched_edges.pop_back();
        }
    }

    return false;  // No match found
}

std::vector<PatternMatch> PatternMatcher::find_matches_from_anchor(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    VertexId anchor_vertex,
    std::size_t search_radius) const {

    std::vector<PatternMatch> matches;

    if (!target.has_vertex(anchor_vertex)) {
        return matches;  // Anchor doesn't exist
    }

    // Get edges within radius of anchor
    auto nearby_edges = target.edges_within_radius(anchor_vertex, search_radius);

    if (nearby_edges.size() < pattern.num_edges()) {
        return matches;  // Not enough edges to match pattern
    }

    // Build EdgeSignatureIndex for fast candidate filtering (HGMatch optimization)
    EdgeSignatureIndex edge_index;
    for (EdgeId eid : nearby_edges) {
        const Hyperedge* edge = target.get_edge(eid);
        if (edge) {
            EdgeSignature sig = EdgeSignature::from_concrete_edge(*edge, &target);
            edge_index.add_edge(eid, sig);
        }
    }

    // Try to match pattern edges
    const auto& pattern_edges = pattern.edges();
    if (pattern_edges.empty()) {
        return matches;  // Empty pattern
    }

    // For each possible assignment of the first pattern edge
    for (EdgeId edge_id : nearby_edges) {
        const Hyperedge* concrete_edge = target.get_edge(edge_id);
        if (!concrete_edge) continue;

        // Try to match first pattern edge
        auto assignments = generate_assignments(*concrete_edge, pattern_edges[0],
                                              VariableAssignment{});

        for (const auto& initial_assignment : assignments) {
            std::vector<EdgeId> matched_edges = {edge_id};
            VariableAssignment assignment = initial_assignment;

            // Try to match remaining pattern edges
            std::vector<PatternEdge> remaining_patterns(pattern_edges.begin() + 1,
                                                       pattern_edges.end());
            std::unordered_set<EdgeId> available_edges = nearby_edges;
            available_edges.erase(edge_id);

            if (match_remaining_edges(target, remaining_patterns, available_edges,
                                    matched_edges, assignment, edge_index)) {
                // Found a complete match
                PatternMatch match;
                match.matched_edges = matched_edges;
                match.assignment = assignment;
                match.anchor_vertex = anchor_vertex;

                // Build edge_map: pattern edge index -> target edge ID
                for (std::size_t i = 0; i < matched_edges.size(); ++i) {
                    match.edge_map[i] = matched_edges[i];
                }

                matches.push_back(match);
            }
        }
    }

    return matches;
}

std::vector<PatternMatch> PatternMatcher::find_matches_around(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    VertexId anchor_vertex,
    std::size_t search_radius) const {

    return find_matches_from_anchor(target, pattern, anchor_vertex, search_radius);
}

std::vector<PatternMatch> PatternMatcher::find_all_matches(
    const Hypergraph& target,
    const PatternHypergraph& pattern) const {

    // Pure exhaustive pattern matching on the entire hypergraph
    std::vector<PatternMatch> all_matches;

    // Try every possible starting point for pattern matching
    for (VertexId vertex : target.vertices()) {
        auto matches = find_matches_from_anchor(target, pattern, vertex, target.num_edges());
        all_matches.insert(all_matches.end(), matches.begin(), matches.end());
    }

    // Remove duplicates (same set of matched edges)
    std::sort(all_matches.begin(), all_matches.end(),
        [](const PatternMatch& a, const PatternMatch& b) {
            return a.matched_edges < b.matched_edges;
        });

    all_matches.erase(
        std::unique(all_matches.begin(), all_matches.end(),
            [](const PatternMatch& a, const PatternMatch& b) {
                return a.matched_edges == b.matched_edges;
            }),
        all_matches.end());

    return all_matches;
}

bool PatternMatcher::matches_at(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    const VariableAssignment& assignment) const {

    // Check if assignment is complete
    if (!assignment.is_complete(pattern.variable_vertices())) {
        return false;
    }

    // For each pattern edge, check if corresponding concrete edge exists
    for (const auto& pattern_edge : pattern.edges()) {
        std::vector<VertexId> concrete_vertices;

        for (const auto& pattern_vertex : pattern_edge.vertices) {
            auto resolved = assignment.resolve(pattern_vertex);
            if (!resolved) {
                return false;  // Unresolved variable
            }
            concrete_vertices.push_back(*resolved);
        }

        // Check if this concrete edge exists in target
        bool found = false;
        for (const auto& target_edge : target.edges()) {
            if (target_edge.vertices() == concrete_vertices) {
                found = true;
                break;
            }
        }

        if (!found) {
            return false;  // Pattern edge not found in target
        }
    }

    return true;
}

std::optional<PatternMatch> PatternMatcher::find_first_match_around(
    const Hypergraph& target,
    const PatternHypergraph& pattern,
    VertexId anchor_vertex,
    std::size_t search_radius) const {

    auto matches = find_matches_around(target, pattern, anchor_vertex, search_radius);
    return matches.empty() ? std::nullopt : std::optional<PatternMatch>(matches[0]);
}

// EdgeSignature implementations

PatternSignature::PatternSignature(const std::vector<std::size_t>& pattern) : variable_pattern(pattern) {}

bool PatternSignature::operator==(const PatternSignature& other) const {
    return variable_pattern == other.variable_pattern;
}

bool PatternSignature::operator<(const PatternSignature& other) const {
    return variable_pattern < other.variable_pattern;
}

std::size_t PatternSignature::hash() const {
    std::size_t h = 0;
    for (auto var : variable_pattern) {
        h ^= std::hash<std::size_t>{}(var) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

EdgeSignature::EdgeSignature() : num_variables_(0), arity_(0) {}

EdgeSignature EdgeSignature::from_concrete_edge(const Hyperedge& edge,
                                       const Hypergraph* hypergraph) {
    EdgeSignature sig;
    sig.arity_ = edge.arity();
    sig.num_variables_ = 0;

    for (VertexId v : edge.vertices()) {
        sig.concrete_labels_.insert(v);  // Use VertexId directly as label

        // Add vertex incidence information as per HGMatch paper
        if (hypergraph) {
            auto incident_edges = hypergraph->edges_containing(v);
            sig.vertex_incidence_[v] = incident_edges;
            sig.vertex_degrees_[v] = incident_edges.size();
        }
    }

    return sig;
}

EdgeSignature EdgeSignature::from_pattern_edge(const PatternEdge& edge) {
    EdgeSignature sig;
    sig.arity_ = edge.arity();
    sig.num_variables_ = 0;

    for (std::size_t pos = 0; pos < edge.vertices.size(); ++pos) {
        const PatternVertex& pv = edge.vertices[pos];
        if (pv.is_variable()) {
            sig.num_variables_++;
            sig.variable_positions_.push_back(pos);
        } else {
            sig.concrete_labels_.insert(pv.id);  // Use VertexId directly
        }
    }

    return sig;
}

bool EdgeSignature::is_compatible_with_pattern(const EdgeSignature& pattern) const {
    // Arity must match exactly
    if (arity_ != pattern.arity_) {
        DEBUG_LOG("[SIGNATURE] Arity mismatch: concrete=%zu vs pattern=%zu", arity_, pattern.arity_);
        return false;
    }

    // Check if all pattern's concrete labels can be matched
    std::multiset<VertexId> remaining_labels = concrete_labels_;

    for (VertexId label : pattern.concrete_labels_) {
        auto it = remaining_labels.find(label);
        if (it == remaining_labels.end()) {
            DEBUG_LOG("[SIGNATURE] Missing required label: %zu", label);
            return false;  // Pattern requires label we don't have
        }
        remaining_labels.erase(it);
    }

    // Check if variables can cover remaining positions
    std::size_t unmatched_positions = remaining_labels.size();
    if (unmatched_positions > pattern.num_variables_) {
        DEBUG_LOG("[SIGNATURE] Too many unmatched positions: %zu > %zu variables",
                 unmatched_positions, pattern.num_variables_);
        return false;
    }

    // OPTIMIZATION: Use vertex degree information for fast filtering
    // If both signatures have degree information, check degree compatibility
    if (!vertex_degrees_.empty() && !pattern.vertex_degrees_.empty()) {
        // For each vertex in the pattern with known degree requirements
        for (const auto& [pattern_vertex, pattern_degree] : pattern.vertex_degrees_) {
            // Check if any vertex in the concrete edge has a compatible degree
            bool found_compatible_vertex = false;
            for (const auto& [concrete_vertex, concrete_degree] : vertex_degrees_) {
                // Concrete vertex must have at least as many connections as pattern requires
                // This is a necessary (but not sufficient) condition for matching
                if (concrete_degree >= pattern_degree) {
                    found_compatible_vertex = true;
                    break;
                }
            }

            if (!found_compatible_vertex) {
                DEBUG_LOG("[SIGNATURE] Degree incompatibility: pattern vertex %zu requires degree %zu, "
                         "but no concrete vertex has sufficient degree",
                         pattern_vertex, pattern_degree);
                return false;
            }
        }
    }

    DEBUG_LOG("[SIGNATURE] Compatibility check: concrete_labels=%zu, pattern_concrete=%zu, unmatched=%zu, pattern_vars=%zu -> COMPATIBLE",
             concrete_labels_.size(), pattern.concrete_labels_.size(),
             unmatched_positions, pattern.num_variables_);

    return true;
}

bool EdgeSignature::operator==(const EdgeSignature& other) const {
    return arity_ == other.arity_ &&
           num_variables_ == other.num_variables_ &&
           concrete_labels_ == other.concrete_labels_;
}

bool EdgeSignature::operator!=(const EdgeSignature& other) const {
    return !(*this == other);
}

std::size_t EdgeSignature::arity() const { return arity_; }
std::size_t EdgeSignature::num_variables() const { return num_variables_; }
const std::multiset<VertexId>& EdgeSignature::concrete_labels() const { return concrete_labels_; }
const std::unordered_map<VertexId, std::vector<EdgeId>>& EdgeSignature::vertex_incidence() const { return vertex_incidence_; }
const std::unordered_map<VertexId, std::size_t>& EdgeSignature::vertex_degrees() const { return vertex_degrees_; }
const std::vector<std::size_t>& EdgeSignature::variable_positions() const { return variable_positions_; }

std::vector<PatternSignature> EdgeSignature::generate_pattern_signatures() const {
    std::vector<PatternSignature> patterns;

    if (arity_ == 0) return patterns;

    // Optimization #6: Reserve based on Bell number estimate
    patterns.reserve(std::min(std::size_t(64), std::size_t(1) << (arity_ - 1)));

    // Convert concrete labels to indices for pattern generation
    std::vector<VertexId> sorted_labels(concrete_labels_.begin(), concrete_labels_.end());

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

bool EdgeSignature::is_pattern_compatible(const std::vector<std::size_t>& pattern,
                          const std::vector<VertexId>& sorted_labels) const {
    if (pattern.size() != sorted_labels.size()) return false;

    std::unordered_map<std::size_t, VertexId> var_binding;

    for (std::size_t i = 0; i < pattern.size(); ++i) {
        std::size_t var = pattern[i];
        VertexId label = sorted_labels[i];

        auto it = var_binding.find(var);
        if (it != var_binding.end()) {
            // Variable already bound - must match
            if (it->second != label) return false;
        } else {
            // New variable binding - allow multiple variables to bind to same label (for self-loops)
            var_binding[var] = label;
        }
    }

    return true;
}

std::size_t EdgeSignature::hash() const {
    std::size_t h = 0;
    h ^= std::hash<std::size_t>{}(arity_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::size_t>{}(num_variables_) + 0x9e3779b9 + (h << 6) + (h >> 2);

    for (VertexId label : concrete_labels_) {
        h ^= std::hash<VertexId>{}(label) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }

    return h;
}

// EdgeSignatureIndex implementations

void EdgeSignatureIndex::add_edge(EdgeId edge_id, const EdgeSignature& signature) {
    auto& arity_partition = by_arity_[signature.arity()];

    // Generate all pattern signatures for this concrete edge
    auto pattern_signatures = signature.generate_pattern_signatures();

    std::size_t sig_hash = signature.hash();

    for (const auto& pattern_sig : pattern_signatures) {
        auto& pattern_partition = arity_partition.by_pattern_signature[pattern_sig.hash()];

        pattern_partition.signature_to_edges[sig_hash].push_back(edge_id);
        pattern_partition.hash_to_signature[sig_hash] = signature;
    }

    // Build inverted index: for each UNIQUE vertex in this edge, add to he(v)
    // Use a set to deduplicate vertices (important for self-loops)
    std::unordered_set<VertexId> unique_vertices(
        signature.concrete_labels().begin(),
        signature.concrete_labels().end()
    );
    for (VertexId vertex : unique_vertices) {
        auto& edge_list = inverted_index_[vertex];
        edge_list.push_back(edge_id);
        // Keep sorted for fast set intersection (optimization #1)
        // Only sort if we have multiple elements (common case: append-only)
        if (edge_list.size() > 1 && edge_list[edge_list.size() - 2] > edge_id) {
            std::sort(edge_list.begin(), edge_list.end());
        }
    }

    // Store edge -> signature_hash mapping for quick signature lookup
    edge_to_signature_hash_[edge_id] = sig_hash;
}

std::vector<EdgeId> EdgeSignatureIndex::find_compatible_edges(const EdgeSignature& pattern) const {
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

    // Use a set to deduplicate edges that may appear in multiple compatible signatures
    std::unordered_set<EdgeId> unique_edges;

    // Optimization #6: Reserve space to avoid rehashing
    std::size_t total_estimate = 0;
    for (const auto& [sig_hash, edges] : pattern_partition.signature_to_edges) {
        total_estimate += edges.size();
    }
    unique_edges.reserve(total_estimate);

    // Check each signature in this pattern partition
    for (const auto& [sig_hash, edges] : pattern_partition.signature_to_edges) {
        const EdgeSignature& signature = pattern_partition.hash_to_signature.at(sig_hash);

        if (signature.is_compatible_with_pattern(pattern)) {
            unique_edges.insert(edges.begin(), edges.end());
        }
    }

    // Convert set back to vector
    result.assign(unique_edges.begin(), unique_edges.end());

    return result;
}

std::vector<EdgeId> EdgeSignatureIndex::get_edges_with_signature(const EdgeSignature& signature) const {
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

void EdgeSignatureIndex::clear() {
    by_arity_.clear();
    inverted_index_.clear();
    edge_to_signature_hash_.clear();
}

std::size_t EdgeSignatureIndex::size() const {
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

const std::vector<EdgeId>& EdgeSignatureIndex::get_incident_edges(VertexId vertex) const {
    static const std::vector<EdgeId> empty;
    auto vertex_it = inverted_index_.find(vertex);
    if (vertex_it == inverted_index_.end()) {
        return empty;  // Return reference to static empty vector
    }

    return vertex_it->second;  // Return reference
}

std::vector<EdgeId> EdgeSignatureIndex::generate_candidates_by_intersection(
    const std::vector<VertexId>& incident_vertices,
    const std::unordered_set<EdgeId>& available_edges) const {

    if (incident_vertices.empty()) {
        return std::vector<EdgeId>(available_edges.begin(), available_edges.end());
    }

    // Optimization #4: Sort vertices by selectivity (fewest incident edges first)
    // This dramatically reduces the working set size early in the intersection
    std::vector<VertexId> sorted_vertices = incident_vertices;
    std::sort(sorted_vertices.begin(), sorted_vertices.end(),
        [this](VertexId a, VertexId b) {
            return get_incident_edges(a).size() < get_incident_edges(b).size();
        });

    // Step 1: Get edges incident to first vertex as starting set (already sorted!)
    auto first_incident = get_incident_edges(sorted_vertices[0]);

    // Convert to sorted vector for std::set_intersection
    // If available_edges is non-empty, filter first_incident by it
    std::vector<EdgeId> candidates;
    if (!available_edges.empty()) {
        candidates.reserve(std::min(first_incident.size(), available_edges.size()));
        for (EdgeId eid : first_incident) {
            if (available_edges.find(eid) != available_edges.end()) {
                candidates.push_back(eid);
            }
        }

        if (candidates.empty()) {
            return {};
        }
    } else {
        candidates = first_incident;
    }

    // Step 2: Intersect with edges incident to each subsequent vertex
    // Use std::set_intersection since all vectors are sorted
    std::vector<EdgeId> temp_result;
    for (std::size_t i = 1; i < sorted_vertices.size(); ++i) {
        auto incident_edges = get_incident_edges(sorted_vertices[i]);

        // std::set_intersection requires sorted ranges (which we maintain)
        temp_result.clear();
        temp_result.reserve(std::min(candidates.size(), incident_edges.size()));

        std::set_intersection(
            candidates.begin(), candidates.end(),
            incident_edges.begin(), incident_edges.end(),
            std::back_inserter(temp_result)
        );

        // Early exit if intersection is empty
        if (temp_result.empty()) {
            return {};
        }

        // Swap for next iteration (avoids copy)
        std::swap(candidates, temp_result);
    }

    // Return candidates - edge_matches() will do the full structural verification
    return candidates;
}

} // namespace hypergraph

// Hash specializations
namespace std {
/* std::size_t hash<hypergraph::EdgeSignature>::operator()(const hypergraph::EdgeSignature& sig) const {
    return sig.hash();
}

std::size_t hash<hypergraph::PatternSignature>::operator()(const hypergraph::PatternSignature& sig) const {
    return sig.hash();
} */
}