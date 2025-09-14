#include <hypergraph/canonicalization.hpp>
#include <hypergraph/debug_log.hpp>
#include <sstream>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>
#include <unordered_set>

namespace hypergraph {

std::string CanonicalForm::to_string() const {
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

// DelDup[list_List] := Module[{alphabet}, 
//   alphabet = DeleteDuplicates[Flatten[list]];
//   list /. Thread[alphabet -> Range[Length[alphabet]]]]
template<typename VertexType>
std::vector<std::vector<VertexType>> del_dup(const std::vector<std::vector<VertexType>>& list) {
    // alphabet = DeleteDuplicates[Flatten[list]]
    std::vector<VertexType> alphabet;
    std::set<VertexType> seen;
    for (const auto& edge : list) {
        for (auto v : edge) {
            if (seen.insert(v).second) {
                alphabet.push_back(v);
            }
        }
    }
    
    // list /. Thread[alphabet -> Range[Length[alphabet]]]
    // Create replacement rules: alphabet[0] -> 0, alphabet[1] -> 1, etc.
    std::map<VertexType, VertexType> replacement_rules;
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        replacement_rules[alphabet[i]] = static_cast<VertexType>(i);
    }
    
    // Apply replacement rules to list
    std::vector<std::vector<VertexType>> result;
    for (const auto& edge : list) {
        std::vector<VertexType> new_edge;
        for (auto v : edge) {
            new_edge.push_back(replacement_rules[v]);
        }
        result.push_back(new_edge);
    }
    return result;
}

// Helper function to count unique vertices in a collection of edges
template<typename VertexType>
std::size_t count_unique_vertices(const std::vector<std::vector<VertexType>>& edges) {
    std::set<VertexType> vertices;
    for (const auto& edge : edges) {
        for (auto v : edge) {
            vertices.insert(v);
        }
    }
    return vertices.size();
}

// MiserTermsInTuples[tup_List] := Module[{gather, gat, size, seqs},
//   gather = Gather[tup];
//   size = Length[gather];
//   seqs = {#} & /@ Range[size];
//   gat = First /@ gather;
//   Do[seqs = Flatten[With[{grow = #, new = Complement[Range[size], #]},
//         Append[grow, #] & /@ new] & /@
//       First[SplitBy[SortBy[seqs, Length[Union[Flatten[gat[[#]]]]] &],
//         Length[Union[Flatten[gat[[#]]]]] &]], 1], {k, 1, size - 1}];
//   First[SplitBy[SortBy[Union[Flatten[gather[[#]], 1] & /@ seqs], DelDup[#] &], DelDup[#] &]]]
template<typename VertexType>
std::vector<std::vector<VertexType>> miser_terms_in_tuples(const std::vector<std::vector<VertexType>>& tup) {
    if (tup.empty()) return {};

    // gather = Gather[tup] - groups identical elements together
    std::map<std::vector<VertexType>, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < tup.size(); ++i) {
        groups[tup[i]].push_back(i);
    }

    std::vector<std::vector<std::size_t>> gather;
    std::vector<std::vector<VertexType>> gat;
    for (const auto& [edge, indices] : groups) {
        gather.push_back(indices);
        gat.push_back(edge);
    }

    std::size_t size = gather.size();
    if (size <= 1) return tup;

    // seqs = {{1}, {2}, ..., {size}} but 0-indexed
    std::vector<std::vector<std::size_t>> seqs;
    for (std::size_t i = 0; i < size; ++i) {
        seqs.push_back({i});
    }

    // Do[..., {k, 1, size - 1}]
    for (std::size_t k = 1; k < size; ++k) {
        // This is the critical part - we need to generate all possible extensions
        // and pick the ones that minimize vertex count at each step

        std::vector<std::vector<std::size_t>> new_seqs;

        // For each current sequence, try extending it with each unused index
        for (const auto& seq : seqs) {
            std::set<std::size_t> used(seq.begin(), seq.end());

            for (std::size_t i = 0; i < size; ++i) {
                if (used.find(i) == used.end()) {
                    std::vector<std::size_t> extended = seq;
                    extended.push_back(i);
                    new_seqs.push_back(extended);
                }
            }
        }

        // Sort by vertex count (SortBy[seqs, Length[Union[Flatten[gat[[#]]]]] &])
        std::sort(new_seqs.begin(), new_seqs.end(), [&](const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
            std::set<VertexType> a_verts, b_verts;
            for (auto idx : a) {
                for (auto v : gat[idx]) {
                    a_verts.insert(v);
                }
            }
            for (auto idx : b) {
                for (auto v : gat[idx]) {
                    b_verts.insert(v);
                }
            }
            return a_verts.size() < b_verts.size();
        });

        // Keep only those with minimum vertex count (First[SplitBy[...]])
        if (new_seqs.empty()) break;

        std::set<VertexType> min_verts;
        for (auto idx : new_seqs[0]) {
            for (auto v : gat[idx]) {
                min_verts.insert(v);
            }
        }
        std::size_t min_count = min_verts.size();

        std::vector<std::vector<std::size_t>> filtered_seqs;
        for (const auto& seq : new_seqs) {
            std::set<VertexType> seq_verts;
            for (auto idx : seq) {
                for (auto v : gat[idx]) {
                    seq_verts.insert(v);
                }
            }
            if (seq_verts.size() == min_count) {
                filtered_seqs.push_back(seq);
            } else {
                break; // Already sorted
            }
        }

        seqs = filtered_seqs;
    }

    // Convert sequence indices back to actual edge lists
    std::vector<std::vector<std::vector<VertexType>>> arrangements;
    for (const auto& seq : seqs) {
        std::vector<std::vector<VertexType>> arrangement;
        for (auto group_idx : seq) {
            for (auto edge_idx : gather[group_idx]) {
                arrangement.push_back(tup[edge_idx]);
            }
        }
        arrangements.push_back(arrangement);
    }

    // Remove duplicates
    std::sort(arrangements.begin(), arrangements.end());
    arrangements.erase(std::unique(arrangements.begin(), arrangements.end()), arrangements.end());

    // Sort by DelDup result
    std::sort(arrangements.begin(), arrangements.end(), [](const auto& a, const auto& b) {
        return del_dup(a) < del_dup(b);
    });

    // Return first arrangement
    if (arrangements.empty()) return tup;
    return arrangements[0];
}

// CanonicalizeParts[list_List] := Module[{parts, canonicalparts},
//   parts = GatherBy[ReverseSort[list], Length];
//   canonicalparts = MiserTermsInTuples[#] & /@ parts;
//   Flatten[DelDup[First[SortBy[Tuples[canonicalparts], DelDup[Flatten[#]] &]]], 1]]
template<typename VertexType>
std::vector<std::vector<VertexType>> canonicalize_parts(const std::vector<std::vector<VertexType>>& list) {
    if (list.empty()) return {};
    
    // parts = GatherBy[ReverseSort[list], Length]
    auto reverse_sorted = list;
    std::sort(reverse_sorted.rbegin(), reverse_sorted.rend());
    
    std::map<std::size_t, std::vector<std::vector<VertexType>>> parts;
    for (const auto& edge : reverse_sorted) {
        parts[edge.size()].push_back(edge);
    }
    
    // canonicalparts = MiserTermsInTuples[#] & /@ parts
    std::vector<std::vector<std::vector<VertexType>>> canonical_parts;
    for (auto& [arity, part] : parts) {
        canonical_parts.push_back(miser_terms_in_tuples(part));
    }
    
    // Tuples[canonicalparts] - generate all combinations
    std::vector<std::vector<std::vector<VertexType>>> all_tuples;
    std::function<void(std::size_t, std::vector<std::vector<VertexType>>&)> generate_tuples;
    generate_tuples = [&](std::size_t part_idx, std::vector<std::vector<VertexType>>& current) {
        if (part_idx >= canonical_parts.size()) {
            all_tuples.push_back(current);
            return;
        }
        
        // For now, each canonical_part has only one arrangement
        std::vector<std::vector<VertexType>> extended = current;
        for (const auto& edge : canonical_parts[part_idx]) {
            extended.push_back(edge);
        }
        generate_tuples(part_idx + 1, extended);
    };
    
    std::vector<std::vector<VertexType>> empty_tuple;
    generate_tuples(0, empty_tuple);
    
    if (all_tuples.empty()) return {};
    
    // SortBy[Tuples[canonicalparts], DelDup[Flatten[#]] &]
    std::sort(all_tuples.begin(), all_tuples.end(), [](const auto& a, const auto& b) {
        return del_dup(a) < del_dup(b);
    });
    
    // First[...] - take lexicographically smallest
    auto best_tuple = all_tuples[0];
    
    // DelDup[...]
    auto result = del_dup(best_tuple);
    
    // Flatten[..., 1] - already flat since we're working with edges
    return result;
}

template<typename VertexType>
std::vector<std::vector<VertexType>> Canonicalizer::wolfram_canonical_hypergraph(
    const std::vector<std::vector<VertexType>>& edges,
    VertexMapping& mapping) const {
    
    if (edges.empty()) {
        return {};
    }
    
    // CanonicalHypergraph[list_] := CanonicalizeParts[list]
    auto result = canonicalize_parts(edges);
    
    // Build vertex mapping
    mapping.canonical_to_original.clear();
    mapping.original_to_canonical.clear();
    
    std::set<VertexType> vertex_set;
    for (const auto& edge : edges) {
        for (auto v : edge) {
            vertex_set.insert(v);
        }
    }
    std::vector<VertexType> orig_vertices(vertex_set.begin(), vertex_set.end());
    std::sort(orig_vertices.begin(), orig_vertices.end());
    
    mapping.canonical_to_original.resize(orig_vertices.size());
    for (std::size_t i = 0; i < orig_vertices.size(); ++i) {
        mapping.canonical_to_original[i] = orig_vertices[i];
        mapping.original_to_canonical[orig_vertices[i]] = static_cast<VertexType>(i);
    }
    
    return result;
}

std::vector<std::vector<std::size_t>> Canonicalizer::edges_to_size_t(
    const std::vector<std::vector<VertexId>>& edges) const {
    
    std::vector<std::vector<std::size_t>> result;
    for (const auto& edge : edges) {
        std::vector<std::size_t> new_edge;
        for (auto v : edge) {
            new_edge.push_back(static_cast<std::size_t>(v));
        }
        result.push_back(new_edge);
    }
    return result;
}

CanonicalizationResult Canonicalizer::canonicalize(const Hypergraph& hg) const {
    CanonicalizationResult result;
    
    if (hg.num_vertices() == 0) {
        result.canonical_form.vertex_count = 0;
        return result;
    }
    
    // Extract edges as vectors
    std::vector<std::vector<VertexId>> edges;
    for (const Hyperedge& edge : hg.edges()) {
        std::vector<VertexId> edge_vertices;
        for (VertexId v : edge.vertices()) {
            edge_vertices.push_back(v);
        }
        edges.push_back(edge_vertices);
    }
    
    // Apply Wolfram canonicalization
    result.canonical_form.edges = wolfram_canonical_hypergraph(edges, result.vertex_mapping);
    result.canonical_form.vertex_count = result.vertex_mapping.canonical_to_original.size();
    
    return result;
}

template<typename VertexType>
CanonicalizationResult Canonicalizer::canonicalize_edges(const std::vector<std::vector<VertexType>>& edges) const {
    CanonicalizationResult result;
    
    if (edges.empty()) {
        result.canonical_form.vertex_count = 0;
        return result;
    }
    
    // Apply Wolfram canonicalization directly
    result.canonical_form.edges = wolfram_canonical_hypergraph(edges, result.vertex_mapping);
    result.canonical_form.vertex_count = result.vertex_mapping.canonical_to_original.size();
    
    return result;
}

// Explicit template instantiations for the types we use
// Note: VertexId and GlobalVertexId are both std::size_t, so only instantiate once
template CanonicalizationResult Canonicalizer::canonicalize_edges<VertexId>(const std::vector<std::vector<VertexId>>& edges) const;

template std::vector<std::vector<VertexId>> Canonicalizer::wolfram_canonical_hypergraph<VertexId>(
    const std::vector<std::vector<VertexId>>& edges, VertexMapping& mapping) const;

} // namespace hypergraph