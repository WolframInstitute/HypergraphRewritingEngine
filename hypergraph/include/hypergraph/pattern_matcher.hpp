#pragma once

#include <cstdint>
#include <cstring>
#include <atomic>
#include <functional>

#include "types.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "index.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"

namespace hypergraph {

// =============================================================================
// Pattern Matching Tasks (HGMatch Dataflow Model)
// =============================================================================
//
// Following v1's execution model:
// - SCAN/EXPAND execute synchronously (recursive depth-first within single task)
// - Only SINK→REWRITE spawns new jobs to the job system
// - This is memory-efficient (bounded by O(pattern_edges * candidates))
//
// Task types:
// - SCAN: Find initial candidates via signature partition
// - EXPAND: Extend partial match with next pattern edge
// - SINK: Process complete match, spawn REWRITE task
// - REWRITE: Apply rule, create new state, spawn next evolution step

// PartialMatch is defined in pattern.hpp

// =============================================================================
// Candidate Validation
// =============================================================================

// Validate candidate edge against pattern edge, extending binding
// Returns true if validation succeeds, binding is modified in place
inline bool validate_candidate(
    const VertexId* edge_vertices,
    uint8_t edge_arity,
    const PatternEdge& pattern_edge,
    VariableBinding& binding
) {
    if (edge_arity != pattern_edge.arity) return false;

    for (uint8_t i = 0; i < edge_arity; ++i) {
        VertexId actual = edge_vertices[i];
        uint8_t var = pattern_edge.var_at(i);

        if (binding.is_bound(var)) {
            if (actual != binding.get(var)) {
                return false;
            }
        } else {
            binding.bind(var, actual);
        }
    }
    return true;
}

// =============================================================================
// Pattern Matching Context
// =============================================================================
// Shared context for all tasks in a matching session.

template<typename EdgeAccessor, typename SignatureAccessor = std::function<const EdgeSignature&(EdgeId)>>
struct PatternMatchingContext {
    // Rule being matched
    const RewriteRule* rule;
    uint16_t rule_index;

    // State being matched against
    StateId state_id;
    const SparseBitset* state_edges;  // Bitset of edges in this state

    // Indices for candidate generation
    const SignatureIndex* sig_index;
    const InvertedVertexIndex* inv_index;

    // Edge accessor
    EdgeAccessor get_edge;

    // Signature accessor (cached signatures for O(1) lookup)
    SignatureAccessor get_signature;

    // Per-edge pattern signatures and compatible-signature caches are read directly
    // from `rule` (immutable after RewriteRule::compute_var_counts) — no per-session
    // copy: the context used to embed EdgeSignature[16] + CompatibleSignatureCache[16]
    // (~17 KB) and memcpy them from the rule on every state x rule matching session.

    // Coordination
    std::atomic<bool>* should_terminate;
    std::atomic<size_t>* matches_found;
    size_t max_matches;

    // Match callback: called for each complete match
    // Signature: void(rule_index, edges_in_pattern_order, num_edges, binding, state_id)
    using MatchCallback = std::function<void(
        uint16_t, const EdgeId*, uint8_t, const VariableBinding&, StateId)>;
    MatchCallback on_match;

    // Match deduplication (optional)
    ConcurrentMap<uint64_t, MatchId>* match_dedup;

    PatternMatchingContext(
        const RewriteRule* r,
        uint16_t ridx,
        StateId sid,
        const SparseBitset* edges,
        const SignatureIndex* sig,
        const InvertedVertexIndex* inv,
        EdgeAccessor accessor,
        SignatureAccessor sig_accessor,
        MatchCallback callback
    )
        : rule(r)
        , rule_index(ridx)
        , state_id(sid)
        , state_edges(edges)
        , sig_index(sig)
        , inv_index(inv)
        , get_edge(accessor)
        , get_signature(sig_accessor)
        , should_terminate(nullptr)
        , matches_found(nullptr)
        , max_matches(SIZE_MAX)
        , on_match(callback)
        , match_dedup(nullptr)
    {}
};

// =============================================================================
// Candidate Generation (HGMatch Algorithm 4)
// =============================================================================

template<typename EdgeAccessor, typename CandidateCallback>
void generate_candidates(
    const PatternEdge& pattern_edge,
    const EdgeSignature& pattern_sig,
    const CompatibleSignatureCache& sig_cache,  // Pre-computed compatible signatures
    const VariableBinding& binding,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    const EdgeAccessor& get_edge,
    CandidateCallback&& on_candidate
) {
    // Collect bound vertices and their required positions
    VertexId bound_vertices[MAX_ARITY];
    uint8_t bound_positions[MAX_ARITY];
    uint8_t num_bound = 0;

    for (uint8_t i = 0; i < pattern_edge.arity; ++i) {
        uint8_t var = pattern_edge.var_at(i);
        if (binding.is_bound(var)) {
            bound_vertices[num_bound] = binding.get(var);
            bound_positions[num_bound] = i;
            num_bound++;
        }
    }

    if (num_bound == 0) {
        if (pattern_sig.num_distinct() == pattern_edge.arity) {
            // All-distinct pattern edge: it imposes no vertex-repetition constraint, so
            // its compatible data signatures are every set-partition of the arity
            // (Bell(k)), whose per-signature edge-lists re-union to exactly the arity-k
            // edges present in this state. The signature index holds whole-evolution
            // history keyed by signature; drawing candidates from it walks that global
            // history filtered by the state bitset. Scan this state's own edges once and
            // keep those of matching arity — the same candidate set in one pass.
            // validate_candidate re-checks arity downstream.
            const uint8_t want_arity = pattern_edge.arity;
            state_edges.for_each([&](EdgeId eid) {
                const auto& edge = get_edge(eid);
                if (edge.arity == want_arity) {
                    on_candidate(eid, edge);
                }
            });
        } else {
            // Repeated-variable seed edge: the signature level genuinely prunes, so scan
            // the compatible signature partition using the pre-computed cache.
            sig_index.for_each_candidate_cached(sig_cache, state_edges, [&](EdgeId eid) {
                on_candidate(eid, get_edge(eid));
            });
        }
    } else {
        // Have bound variables: use inverted index intersection. The intersection has
        // already fetched each edge to test containment; it hands that edge to us.
        inv_index.for_each_edge_containing_all(
            bound_vertices, num_bound, state_edges, get_edge,
            [&](EdgeId eid, const auto& edge) {
                // Check bound vertices at the required positions. No signature test here:
                // validate_candidate (run by on_candidate) binds each variable on first
                // occurrence and checks equality on repeat, which enforces exactly the
                // repetition constraint signature_compatible would — at O(arity) instead
                // of O(arity^2) — and rejects every edge the signature test would.
                bool valid = true;
                for (uint8_t i = 0; i < num_bound && valid; ++i) {
                    if (edge.vertices[bound_positions[i]] != bound_vertices[i]) {
                        valid = false;
                    }
                }

                if (valid) {
                    on_candidate(eid, edge);
                }
            }
        );
    }
}

// =============================================================================
// EXPAND Task (Recursive)
// =============================================================================
// Extends partial match with next pattern edge.
// Executes synchronously (depth-first) - does not spawn jobs.

template<typename EdgeAccessor, typename SignatureAccessor>
void expand_match(
    PatternMatchingContext<EdgeAccessor, SignatureAccessor>& ctx,
    PartialMatch& partial
) {
    // Check termination
    if (ctx.should_terminate && ctx.should_terminate->load()) return;

    // Check if complete
    if (partial.is_complete()) {
        // Convert to pattern order
        EdgeId edges_in_order[MAX_PATTERN_EDGES];
        partial.to_pattern_order(edges_in_order);

        // Deduplication check (optional)
        if (ctx.match_dedup) {
            MatchIdentity identity(ctx.rule_index, edges_in_order, ctx.rule->num_lhs_edges);
            auto [existing, inserted] = ctx.match_dedup->insert_if_absent(
                identity.hash(), static_cast<MatchId>(0));
            if (!inserted) return;  // Already found
        }

        // Report match
        if (ctx.on_match) {
            ctx.on_match(
                ctx.rule_index,
                edges_in_order,
                ctx.rule->num_lhs_edges,
                partial.binding,
                ctx.state_id
            );
        }

        // Track count
        if (ctx.matches_found) {
            size_t count = ctx.matches_found->fetch_add(1) + 1;
            if (count >= ctx.max_matches && ctx.should_terminate) {
                ctx.should_terminate->store(true);
            }
        }
        return;
    }

    // Next pattern edge to match, in the rule's optimized join order.
    uint8_t pattern_idx = ctx.rule->match_order[partial.num_matched];
    if (pattern_idx >= ctx.rule->num_lhs_edges) return;

    const PatternEdge& pattern_edge = ctx.rule->lhs[pattern_idx];
    const EdgeSignature& pattern_sig = ctx.rule->lhs_sig[pattern_idx];
    const CompatibleSignatureCache& sig_cache = ctx.rule->lhs_cache[pattern_idx];

    // Generate candidates. Depth-first backtracking mutates a single PartialMatch in
    // place — bind the new variables directly into partial.binding, push the matched
    // edge, recurse, then unbind and pop on return — instead of copying the ~356 B of
    // binding + partial per surviving candidate at every recursion level.
    generate_candidates(
        pattern_edge, pattern_sig, sig_cache,
        partial.binding, *ctx.state_edges,
        *ctx.sig_index, *ctx.inv_index, ctx.get_edge,
        [&](EdgeId candidate, const auto& edge) {
            // Check termination
            if (ctx.should_terminate && ctx.should_terminate->load()) return;

            // Skip if already matched
            if (partial.contains_edge(candidate)) return;

            // Bind directly into partial.binding, recording which variables are bound
            // before this edge so the exact set of newly-bound variables can be undone.
            const uint32_t pre_mask = partial.binding.bound_mask;

            if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, partial.binding)) {
                // validate_candidate may bind some variables before hitting a mismatch;
                // restore partial.binding to its pre-edge state before trying the next.
                for (uint32_t newly = partial.binding.bound_mask & ~pre_mask; newly; newly &= newly - 1) {
                    partial.binding.unbind(static_cast<uint8_t>(hgcommon::ctz(newly)));
                }
                return;
            }

            // Push the matched edge/order and recurse on the same PartialMatch.
            partial.match_order[partial.num_matched] = pattern_idx;
            partial.matched_edges[partial.num_matched] = candidate;
            partial.num_matched++;

            expand_match(ctx, partial);

            // Pop the edge and unbind the variables this edge bound.
            partial.num_matched--;
            for (uint32_t newly = partial.binding.bound_mask & ~pre_mask; newly; newly &= newly - 1) {
                partial.binding.unbind(static_cast<uint8_t>(hgcommon::ctz(newly)));
            }
        }
    );
}

// =============================================================================
// SCAN Task
// =============================================================================
// Finds initial candidates for first pattern edge, then calls expand_match.
// Executes synchronously.

template<typename EdgeAccessor, typename SignatureAccessor>
void scan_pattern(
    PatternMatchingContext<EdgeAccessor, SignatureAccessor>& ctx
) {
    if (ctx.rule->num_lhs_edges == 0) return;

    // Seed the join with the rule's most-constrained edge (match_order[0]).
    const uint8_t first_pidx = ctx.rule->match_order[0];
    const PatternEdge& first_edge = ctx.rule->lhs[first_pidx];
    const EdgeSignature& first_sig = ctx.rule->lhs_sig[first_pidx];
    const CompatibleSignatureCache& first_cache = ctx.rule->lhs_cache[first_pidx];

    // Generate candidates for first edge
    generate_candidates(
        first_edge, first_sig, first_cache,
        VariableBinding{}, *ctx.state_edges,
        *ctx.sig_index, *ctx.inv_index, ctx.get_edge,
        [&](EdgeId candidate, const auto& edge) {
            // Check termination
            if (ctx.should_terminate && ctx.should_terminate->load()) return;

            VariableBinding binding;

            if (!validate_candidate(edge.vertices, edge.arity, first_edge, binding)) {
                return;
            }

            // Create initial partial match
            PartialMatch partial;
            partial.num_pattern_edges = ctx.rule->num_lhs_edges;
            partial.add_match(first_pidx, candidate, binding);

            // Single-edge pattern: complete match
            if (ctx.rule->num_lhs_edges == 1) {
                EdgeId edges_in_order[MAX_PATTERN_EDGES];
                partial.to_pattern_order(edges_in_order);

                // Deduplication
                if (ctx.match_dedup) {
                    MatchIdentity identity(ctx.rule_index, edges_in_order, 1);
                    auto [existing, inserted] = ctx.match_dedup->insert_if_absent(
                        identity.hash(), static_cast<MatchId>(0));
                    if (!inserted) return;
                }

                if (ctx.on_match) {
                    ctx.on_match(ctx.rule_index, edges_in_order, 1, binding, ctx.state_id);
                }

                if (ctx.matches_found) {
                    size_t count = ctx.matches_found->fetch_add(1) + 1;
                    if (count >= ctx.max_matches && ctx.should_terminate) {
                        ctx.should_terminate->store(true);
                    }
                }
            } else {
                // Multi-edge pattern: expand
                expand_match(ctx, partial);
            }
        }
    );
}

// =============================================================================
// Public API
// =============================================================================

// Find all matches for a rule in a state
template<typename EdgeAccessor, typename SignatureAccessor, typename MatchCallback>
void find_matches(
    const RewriteRule& rule,
    uint16_t rule_index,
    StateId state_id,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    EdgeAccessor get_edge,
    SignatureAccessor get_signature,
    MatchCallback&& on_match,
    std::atomic<bool>* should_terminate = nullptr,
    std::atomic<size_t>* matches_found = nullptr,
    size_t max_matches = SIZE_MAX,
    ConcurrentMap<uint64_t, MatchId>* match_dedup = nullptr
) {
    PatternMatchingContext<EdgeAccessor, SignatureAccessor> ctx(
        &rule, rule_index, state_id, &state_edges,
        &sig_index, &inv_index, get_edge, get_signature,
        std::forward<MatchCallback>(on_match)
    );

    ctx.should_terminate = should_terminate;
    ctx.matches_found = matches_found;
    ctx.max_matches = max_matches;
    ctx.match_dedup = match_dedup;

    scan_pattern(ctx);
}

// Backward-compatible overload: computes signatures on-the-fly
// Use the version with SignatureAccessor for better performance
template<typename EdgeAccessor, typename MatchCallback>
void find_matches(
    const RewriteRule& rule,
    uint16_t rule_index,
    StateId state_id,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    EdgeAccessor get_edge,
    MatchCallback&& on_match,
    std::atomic<bool>* should_terminate = nullptr,
    std::atomic<size_t>* matches_found = nullptr,
    size_t max_matches = SIZE_MAX,
    ConcurrentMap<uint64_t, MatchId>* match_dedup = nullptr
) {
    // Create a signature accessor that computes on-the-fly
    auto compute_signature = [&get_edge](EdgeId eid) -> EdgeSignature {
        const auto& edge = get_edge(eid);
        return EdgeSignature::from_edge(edge.vertices, edge.arity);
    };

    find_matches(rule, rule_index, state_id, state_edges,
                 sig_index, inv_index, get_edge, compute_signature,
                 std::forward<MatchCallback>(on_match),
                 should_terminate, matches_found, max_matches, match_dedup);
}

// =============================================================================
// Delta Matching - Only find NEW matches involving produced edges
// =============================================================================
// For match forwarding optimization: new matches must include at least one
// produced edge. We start pattern matching from produced edges only, which
// dramatically reduces the search space.
//
// For a k-edge pattern, we try each produced edge at each pattern position.
// Deduplication handles overlaps when multiple produced edges are in one match.

template<typename EdgeAccessor, typename SignatureAccessor>
void scan_pattern_from_edge(
    PatternMatchingContext<EdgeAccessor, SignatureAccessor>& ctx,
    EdgeId starting_edge,
    uint8_t pattern_position
) {
    if (ctx.rule->num_lhs_edges == 0) return;

    const PatternEdge& pattern_edge = ctx.rule->lhs[pattern_position];
    const auto& edge = ctx.get_edge(starting_edge);

    // Validate the starting edge matches the pattern at this position
    VariableBinding binding;
    if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, binding)) {
        return;
    }

    // Check signature compatibility using cached signature
    const EdgeSignature& data_sig = ctx.get_signature(starting_edge);
    if (!signature_compatible(data_sig, ctx.rule->lhs_sig[pattern_position])) {
        return;
    }

    // Create initial partial match at the specified position
    PartialMatch partial;
    partial.num_pattern_edges = ctx.rule->num_lhs_edges;
    partial.add_match(pattern_position, starting_edge, binding);

    // Single-edge pattern: complete match
    if (ctx.rule->num_lhs_edges == 1) {
        EdgeId edges_in_order[MAX_PATTERN_EDGES];
        partial.to_pattern_order(edges_in_order);

        // Deduplication
        if (ctx.match_dedup) {
            MatchIdentity identity(ctx.rule_index, edges_in_order, 1);
            auto [existing, inserted] = ctx.match_dedup->insert_if_absent(
                identity.hash(), static_cast<MatchId>(0));
            if (!inserted) return;
        }

        if (ctx.on_match) {
            ctx.on_match(ctx.rule_index, edges_in_order, 1, binding, ctx.state_id);
        }

        if (ctx.matches_found) {
            size_t count = ctx.matches_found->fetch_add(1) + 1;
            if (count >= ctx.max_matches && ctx.should_terminate) {
                ctx.should_terminate->store(true);
            }
        }
    } else {
        // Multi-edge pattern: expand from this starting point
        expand_match(ctx, partial);
    }
}

// Find matches that include at least one of the produced edges
// This is used for delta matching: only search for NEW patterns
template<typename EdgeAccessor, typename SignatureAccessor, typename MatchCallback>
void find_delta_matches(
    const RewriteRule& rule,
    uint16_t rule_index,
    StateId state_id,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    EdgeAccessor get_edge,
    SignatureAccessor get_signature,
    MatchCallback&& on_match,
    const EdgeId* produced_edges,
    uint8_t num_produced,
    std::atomic<bool>* should_terminate = nullptr,
    std::atomic<size_t>* matches_found = nullptr,
    size_t max_matches = SIZE_MAX,
    ConcurrentMap<uint64_t, MatchId>* match_dedup = nullptr
) {
    if (num_produced == 0) return;

    PatternMatchingContext<EdgeAccessor, SignatureAccessor> ctx(
        &rule, rule_index, state_id, &state_edges,
        &sig_index, &inv_index, get_edge, get_signature,
        std::forward<MatchCallback>(on_match)
    );

    ctx.should_terminate = should_terminate;
    ctx.matches_found = matches_found;
    ctx.max_matches = max_matches;
    ctx.match_dedup = match_dedup;

    // For each produced edge, try it at each pattern position
    // This ensures we find all matches that include at least one produced edge
    for (uint8_t p = 0; p < num_produced; ++p) {
        EdgeId produced = produced_edges[p];

        // Skip if edge not in state (shouldn't happen, but safety check)
        if (!state_edges.contains(produced)) continue;

        for (uint8_t pos = 0; pos < rule.num_lhs_edges; ++pos) {
            if (should_terminate && should_terminate->load()) return;

            scan_pattern_from_edge(ctx, produced, pos);
        }
    }
}

}  // namespace hypergraph
