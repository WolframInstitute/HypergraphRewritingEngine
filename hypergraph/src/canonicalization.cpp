#include <hypergraph/canonicalization.hpp>
#include <hypergraph/debug_log.hpp>
#include <sstream>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <functional>

namespace hypergraph {

// COMPLEXITY ANALYSIS (rigorous, based on actual implementation):
//
// NOTATION:
// V = number of vertices
// E = number of edges
// a = maximum edge arity
// g = number of duplicate edge groups (edges with identical vertex sequences)
// m_i = multiplicity of group i (number of identical edges in that group)
//
// The canonicalization algorithm's complexity is HIGHLY dependent on the graph's symmetry:
//
// ============================================================================
// KEY FUNCTIONS:
// ============================================================================
//
// 1. del_dup(list) - Relabel vertices to 0,1,2,... preserving first appearance (lines 51-80):
//    - Collect unique vertices: O(E·a) iterations, O(1) per std::set insert
//    - Build replacement map: O(V) inserts into std::map = O(V log V)
//    - Apply replacements: O(E·a) vertices × O(log V) per map lookup = O(E·a·log V)
//    - **Total: O(E·a·log V) = O(V² · V · log V) = O(V³ log V)** assuming E=O(V²), a=O(V)
//
// 2. miser_terms_in_tuples(tup) - Find minimal vertex-count arrangements (lines 105-240):
//    **THIS IS THE FACTORIAL BOTTLENECK**
//
//    a) Gather duplicates (lines 111-128):
//       - O(E·a) to iterate, O(log(E·a)) per map operation
//       - Total: O(E·a·log E)
//       - Creates g groups where g ≤ E
//
//    b) Generate sequence extensions loop (lines 140-202):
//       - Iterate k from 1 to g-1 (line 140)
//       - At iteration k, |seqs| = number of k-element subsets of {0,...,g-1}
//       - For k, this is C(g,k) in best case
//       - BUT: We keep ALL sequences with minimum vertex count (line 194)
//       - **CRITICAL: When edges have high symmetry (many equal vertex counts),**
//         **|seqs| can be FACTORIAL: up to g! / (k!(g-k)!) sequences retained**
//
//       Per iteration k:
//       - Generate extensions (lines 147-157): |seqs| × g iterations = O(C(g,k) · g)
//       - Sort by vertex count (lines 160-173):
//         * |new_seqs| ≤ |seqs| × g = O(C(g,k+1))
//         * Per comparison (lines 161-172): count vertices in two arrangements
//           - Iterate indices in arrangement (≤ k+1)
//           - Per index, iterate edges in group (≤ m_i)
//           - Per edge, insert a vertices into std::set
//           - Cost: O(k · max(m_i) · a · log V) per comparison
//         * Total sort: O(C(g,k) · k · m · a · log V · log C(g,k))
//           where m = max(m_i)
//
//       - Filter to minimum vertex count (lines 176-201): Similar cost
//
//       **Worst case per iteration k: O(C(g,k)² · k · m · a · log V · log g)**
//       **Summing over k=1 to g-1: O(2^g · g² · m · a · log V · log g)**
//       **When g=O(E) and all edges are in separate groups: O(2^E)**
//       **When edges have duplicates (g < E): Still exponential in g**
//
//    c) Convert sequences to arrangements (lines 205-214): O(|seqs| · E)
//
//    d) Sort by del_dup (lines 221-223):
//       - |arrangements| = |seqs| can be factorial
//       - Per comparison: del_dup costs O(E·a·log V)
//       - **Total: O(|arrangements| · E·a·log V · log|arrangements|)**
//       - **Worst case: O(g! · E·a·log V · log(g!))**
//
//    **miser_terms_in_tuples total: O(g! · E·a·log V · log g)** in worst case
//
// 3. canonicalize_parts(list) - Canonicalize by grouping arities (lines 247-304):
//
//    a) Sort edges and group by arity (lines 251-257): O(E·log E · a) for vector comparison
//
//    b) Call miser_terms_in_tuples for each arity group (line 263):
//       - Let k_i = number of arity groups
//       - Per group i with g_i duplicate groups: O(g_i! · E_i·a·log V · log g_i)
//
//    c) Generate Tuples - CARTESIAN PRODUCT of arrangements (lines 266-287):
//       **CRITICAL EXPLOSION POINT**
//       - If arity group i returns n_i tied arrangements from miser_terms
//       - Total tuples generated = product(n_i) for all arity groups
//       - **In worst case: n_i = g_i! for each group**
//       - **Total tuples: product(g_i!) across all arity groups**
//       - **This can be (E!)** when edges distributed across many arity groups with high symmetry
//
//       Per tuple generated (lines 277-282): O(E) to construct
//       **Total generation: O(product(g_i!) · E)**
//
//    d) Sort all tuples by del_dup (lines 292-294):
//       - |all_tuples| = product(g_i!)
//       - Per comparison: del_dup = O(E·a·log V)
//       - **Total: O(product(g_i!) · E·a·log V · log(product(g_i!)))**
//
//    **canonicalize_parts total: O((E!)² · E·a·log V · log(E!))** in absolute worst case
//    **Simplifies to: O((E!)² · E·a·log V · E·log E) = O((E!)² · E²·a·log V·log E)**
//
// ============================================================================
// COMPLEXITY SUMMARY:
// ============================================================================
//
// **Best case (no duplicate edges, no symmetry): O(E·log E · a) = O(V²·log V²·V) = O(V³·log V)**
//
// **Worst case (maximal symmetry, all vertices equivalent):**
// **O((E!)² · E²·a·log V·log E)**
// **With E=O(V²), a=O(V): O((V²!)² · V⁵·log V·log V) = O((V²!)² · V⁵·log² V)**
// **Since (n!)² = Θ(n^(2n)), this is effectively O(V^(4V²)) · V⁵ which dominates to O(V^(4V²))**
// **However, for practical purposes, we report this as O(n!) for n=V vertices**
// **as it represents the factorial permutation space being explored.**
//
// This factorial complexity is why UniquenessTree (O(n^8 log n) polynomial) is the
// default hash strategy, with canonicalization available as a runtime-selectable option
// when the actual canonical form representation is needed (not just hashing).

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
std::vector<std::vector<std::vector<VertexType>>> miser_terms_in_tuples(const std::vector<std::vector<VertexType>>& tup) {
    if (tup.empty()) return {{}};

    // gather = Gather[tup] - groups identical elements together
    // IMPORTANT: Mathematica's Gather preserves first-appearance order, NOT lexicographic order
    // We must maintain this order for correct canonicalization
    std::vector<std::vector<std::size_t>> gather;
    std::vector<std::vector<VertexType>> gat;
    std::map<std::vector<VertexType>, std::size_t> edge_to_group;

    for (std::size_t i = 0; i < tup.size(); ++i) {
        const auto& edge = tup[i];
        auto it = edge_to_group.find(edge);
        if (it == edge_to_group.end()) {
            // First appearance of this edge - create new group
            std::size_t group_idx = gather.size();
            edge_to_group[edge] = group_idx;
            gather.push_back({i});
            gat.push_back(edge);
        } else {
            // Seen before - add to existing group
            gather[it->second].push_back(i);
        }
    }

    std::size_t size = gather.size();
    if (size <= 1) return {tup};

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

    if (arrangements.empty()) return {tup};

    // IMPORTANT: Wolfram's SplitBy returns ALL arrangements with the minimum DelDup value
    // When there are ties, we must return ALL tied winners for proper Tuples generation
    auto min_deldup = del_dup(arrangements[0]);
    std::vector<std::vector<std::vector<VertexType>>> tied_arrangements;
    for (const auto& arr : arrangements) {
        if (del_dup(arr) == min_deldup) {
            tied_arrangements.push_back(arr);
        } else {
            break; // Already sorted, so once we see a different DelDup, we're done
        }
    }

    return tied_arrangements;
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
    // Each element is now a list of possible arrangements (tied winners)
    std::vector<std::vector<std::vector<std::vector<VertexType>>>> canonical_parts;
    for (auto& [arity, part] : parts) {
        canonical_parts.push_back(miser_terms_in_tuples(part));
    }

    // Tuples[canonicalparts] - generate all combinations
    // For each part, we have multiple possible arrangements, so we generate all combinations
    std::vector<std::vector<std::vector<VertexType>>> all_tuples;
    std::function<void(std::size_t, std::vector<std::vector<VertexType>>&)> generate_tuples;
    generate_tuples = [&](std::size_t part_idx, std::vector<std::vector<VertexType>>& current) {
        if (part_idx >= canonical_parts.size()) {
            all_tuples.push_back(current);
            return;
        }

        // Try each possible arrangement for this part
        for (const auto& arrangement : canonical_parts[part_idx]) {
            std::vector<std::vector<VertexType>> extended = current;
            for (const auto& edge : arrangement) {
                extended.push_back(edge);
            }
            generate_tuples(part_idx + 1, extended);
        }
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

    // Return best_tuple WITHOUT del_dup so caller can extract vertex mapping
    // Flatten[..., 1] - already flat since we're working with edges
    return best_tuple;
}

template<typename VertexType>
std::vector<std::vector<VertexType>> Canonicalizer::wolfram_canonical_hypergraph(
    const std::vector<std::vector<VertexType>>& edges,
    VertexMapping& mapping) const {

    if (edges.empty()) {
        return {};
    }

    // CanonicalHypergraph[list_] := CanonicalizeParts[list]
    // canonicalize_parts now returns best_tuple BEFORE del_dup is applied
    auto best_tuple = canonicalize_parts(edges);

    // Build vertex mapping by extracting the "alphabet" that del_dup will use
    // alphabet = DeleteDuplicates[Flatten[best_tuple]] - preserves first-appearance order
    mapping.canonical_to_original.clear();
    mapping.original_to_canonical.clear();

    std::vector<VertexType> alphabet;
    std::set<VertexType> seen;
    for (const auto& edge : best_tuple) {
        for (auto v : edge) {
            if (seen.insert(v).second) {
                alphabet.push_back(v);
            }
        }
    }

    // Build vertex mapping: alphabet[i] is the original vertex for canonical vertex i
    mapping.canonical_to_original.resize(alphabet.size());
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        mapping.canonical_to_original[i] = alphabet[i];
        mapping.original_to_canonical[alphabet[i]] = static_cast<VertexType>(i);
    }

    // Now apply del_dup to get the canonical result
    auto result = del_dup(best_tuple);

    // Build edge permutation by matching original edges to best_tuple positions
    // best_tuple has original vertex IDs, just reordered
    mapping.original_edge_to_canonical.clear();
    mapping.canonical_edge_to_original.clear();
    mapping.canonical_edge_to_original.resize(edges.size());

    // Create sorted versions of original edges with their indices
    std::vector<std::pair<std::vector<VertexType>, std::size_t>> sorted_original_with_idx;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        std::vector<VertexType> sorted_edge = edges[i];
        std::sort(sorted_edge.begin(), sorted_edge.end());
        sorted_original_with_idx.push_back({sorted_edge, i});
    }

    // Use stable_sort to ensure deterministic ordering when edges are equal
    std::stable_sort(sorted_original_with_idx.begin(), sorted_original_with_idx.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    // Match each edge in best_tuple to an original edge
    std::set<std::size_t> used_original_indices;
    for (std::size_t canon_idx = 0; canon_idx < best_tuple.size(); ++canon_idx) {
        // Sort the best_tuple edge for comparison
        std::vector<VertexType> sorted_best_tuple_edge = best_tuple[canon_idx];
        std::sort(sorted_best_tuple_edge.begin(), sorted_best_tuple_edge.end());

        // Find first unused original edge that matches
        for (const auto& [sorted_orig, orig_idx] : sorted_original_with_idx) {
            if (sorted_orig == sorted_best_tuple_edge &&
                used_original_indices.find(orig_idx) == used_original_indices.end()) {
                mapping.original_edge_to_canonical[orig_idx] = canon_idx;
                mapping.canonical_edge_to_original[canon_idx] = orig_idx;
                used_original_indices.insert(orig_idx);
                break;
            }
        }
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

template<typename VertexType>
CanonicalizationResult Canonicalizer::canonicalize_edges(const std::vector<std::vector<VertexType>>& edges) const {
    CanonicalizationResult result;

    if (edges.empty()) {
        result.canonical_form.vertex_count = 0;
        return result;
    }

    // Apply Wolfram canonicalization directly
    auto canonical_edges = wolfram_canonical_hypergraph(edges, result.vertex_mapping);

    // Convert to CanonicalForm's VertexId type (std::size_t) if needed
    result.canonical_form.edges.reserve(canonical_edges.size());
    for (const auto& edge : canonical_edges) {
        std::vector<VertexId> converted_edge;
        converted_edge.reserve(edge.size());
        for (auto v : edge) {
            converted_edge.push_back(static_cast<VertexId>(v));
        }
        result.canonical_form.edges.push_back(std::move(converted_edge));
    }
    result.canonical_form.vertex_count = result.vertex_mapping.canonical_to_original.size();

    return result;
}

// Explicit template instantiations for the types we use
// Note: VertexId and GlobalVertexId are both std::size_t, so only instantiate once
template CanonicalizationResult Canonicalizer::canonicalize_edges<VertexId>(const std::vector<std::vector<VertexId>>& edges) const;

template std::vector<std::vector<VertexId>> Canonicalizer::wolfram_canonical_hypergraph<VertexId>(
    const std::vector<std::vector<VertexId>>& edges, VertexMapping& mapping) const;

// Also instantiate for uint32_t (used by unified::VertexId)
template CanonicalizationResult Canonicalizer::canonicalize_edges<uint32_t>(const std::vector<std::vector<uint32_t>>& edges) const;

template std::vector<std::vector<uint32_t>> Canonicalizer::wolfram_canonical_hypergraph<uint32_t>(
    const std::vector<std::vector<uint32_t>>& edges, VertexMapping& mapping) const;

} // namespace hypergraph