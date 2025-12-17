#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "types.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "unified_hypergraph.hpp"
#include "hypergraph/debug_log.hpp"

namespace hypergraph::unified {

// =============================================================================
// RewriteResult
// =============================================================================
// Result of applying a rewrite rule to a match.

struct RewriteResult {
    StateId new_state;         // The canonical state ID (for deduplication)
    StateId raw_state;         // The raw state ID we created (with actual produced edges)
    EventId event;             // The event recording this rewrite (INVALID_ID if state existed)
    EventId canonical_event;   // The canonical event ID (same as event if is_canonical_event)
    EdgeId produced_edges[MAX_PATTERN_EDGES];  // Edges created by the rewrite
    uint8_t num_produced;      // Number of edges produced
    bool success;              // Whether rewrite succeeded
    bool was_new_state;        // true if new canonical state, false if existing canonical found
    bool is_canonical_event;   // true if this event is canonical (first with this signature)

    RewriteResult()
        : new_state(INVALID_ID)
        , raw_state(INVALID_ID)
        , event(INVALID_ID)
        , canonical_event(INVALID_ID)
        , num_produced(0)
        , success(false)
        , was_new_state(false)
        , is_canonical_event(false)
    {
        std::memset(produced_edges, 0xFF, sizeof(produced_edges));
    }
};

// =============================================================================
// Rewriter
// =============================================================================
// Applies rewrite rules to matches, creating new states.
//
// The rewrite operation:
// 1. Takes an input state and a match
// 2. Removes matched (consumed) edges from the state
// 3. Creates new edges from the RHS pattern using the binding
// 4. Allocates fresh vertices for new RHS variables
// 5. Creates the new state and event
//
// Thread safety: Fully thread-safe. Multiple rewrites can execute concurrently.

class Rewriter {
    UnifiedHypergraph* hg_;

public:
    explicit Rewriter(UnifiedHypergraph* hg) : hg_(hg) {}

    // Apply a match to create a new state
    RewriteResult apply(
        const RewriteRule& rule,
        StateId input_state,
        const EdgeId* matched_edges,
        uint8_t num_matched,
        const VariableBinding& binding,
        uint32_t output_step = 0
    ) {
        RewriteResult result;

        // Validate input_state
        uint32_t num_states = hg_->num_states();
        if (input_state >= num_states) {
            DEBUG_LOG("ERROR: Rewriter::apply input_state=%u >= num_states=%u",
                      input_state, num_states);
            abort();
        }

        // Get input state's edge set
        const SparseBitset& input_edges = hg_->get_state_edges(input_state);

        // VALIDATION: Check that all matched edges exist in input state
        // If they don't, this match was incorrectly forwarded and is invalid
        for (uint8_t i = 0; i < num_matched; ++i) {
            if (!input_edges.contains(matched_edges[i])) {
                // Match is invalid for this state - edges don't exist
                // This can happen due to forwarding bugs
                return result;  // Return empty result (match not applied)
            }
        }

        // Build new edge set: copy input, remove consumed
        SparseBitset new_edges;
        input_edges.for_each([&](EdgeId eid) {
            new_edges.set(eid, hg_->arena());
        });

        // Remove consumed edges
        for (uint8_t i = 0; i < num_matched; ++i) {
            new_edges.clear(matched_edges[i]);
        }

        // Allocate fresh vertices for new RHS variables
        VertexId fresh_vertex_map[MAX_VARS];
        std::memset(fresh_vertex_map, 0xFF, sizeof(fresh_vertex_map));

        uint32_t new_var_mask = rule.new_var_mask();
        while (new_var_mask) {
            uint8_t var = __builtin_ctz(new_var_mask);
            fresh_vertex_map[var] = hg_->alloc_vertex();
            new_var_mask &= new_var_mask - 1;
        }

        // Create new edges from RHS pattern
        result.num_produced = 0;
        for (uint8_t i = 0; i < rule.num_rhs_edges; ++i) {
            const PatternEdge& rhs_edge = rule.rhs[i];

            // Resolve vertices for this edge
            VertexId vertices[MAX_ARITY];
            for (uint8_t j = 0; j < rhs_edge.arity; ++j) {
                uint8_t var = rhs_edge.var_at(j);

                if (binding.is_bound(var)) {
                    // Variable from LHS - use binding
                    vertices[j] = binding.get(var);
                } else if (fresh_vertex_map[var] != INVALID_ID) {
                    // Fresh variable - use allocated vertex
                    vertices[j] = fresh_vertex_map[var];
                } else {
                    // Error: variable not bound and not fresh
                    // This shouldn't happen with valid rules
                    return result;
                }
            }

            // Create the edge (producer will be set after event is created)
            EdgeId eid = hg_->create_edge(vertices, rhs_edge.arity, INVALID_ID, output_step);
            result.produced_edges[result.num_produced++] = eid;
            new_edges.set(eid, hg_->arena());
        }

        // Compute canonical hash for new state
        // Use incremental path when IncrementalUniquenessTree strategy is selected
        auto [canonical_hash, vertex_cache] = hg_->compute_canonical_hash_incremental(
            new_edges,
            input_state,  // Parent state for incremental computation
            matched_edges, num_matched,  // Consumed edges
            result.produced_edges, result.num_produced  // Produced edges
        );

        // Try to create or get existing canonical state
        auto [canonical_id, raw_id, was_new] = hg_->create_or_get_canonical_state(
            std::move(new_edges),
            canonical_hash,
            output_step,
            INVALID_ID  // Will be updated when event is created
        );

        // Store cache for the new state (for future incremental computation)
        hg_->store_state_cache(raw_id, vertex_cache);

        result.new_state = canonical_id;
        result.raw_state = raw_id;
        result.was_new_state = was_new;

        // Create the event (even for duplicate states - we want to track all paths)
        // IMPORTANT: Use raw_state (not new_state/canonical_id) as output_state
        // The raw_state contains the actual produced edges, while new_state is
        // the canonical representative which may be a different state.
        // This is critical for ByStateAndEdges event canonicalization which needs
        // to find edge correspondence between the output_state and canonical_output.
        auto event_result = hg_->create_event(
            input_state,
            result.raw_state,  // Use raw state that contains produced edges
            rule.index,
            matched_edges,
            num_matched,
            result.produced_edges,
            result.num_produced,
            binding
        );
        result.event = event_result.event_id;
        result.canonical_event = event_result.canonical_event_id;
        result.is_canonical_event = event_result.is_canonical;

        // =====================================================================
        // Online Causal/Branchial Tracking
        // =====================================================================

        // Register produced edges (set this event as producer)
        for (uint8_t i = 0; i < result.num_produced; ++i) {
            hg_->set_edge_producer(result.produced_edges[i], result.event);
        }

        // Register consumed edges (add this event as consumer)
        // This triggers causal edge creation via rendezvous pattern
        //
        // IMPORTANT: For correct online transitive reduction, we must process
        // edges in DESCENDING order by producer event ID. This ensures edges
        // from closer (newer) producers are added first, propagating transitive
        // closure to farther (older) producers before checking their edges.
        //
        // Example: If P1→P2 path exists and consumer C has edges from both:
        // - Add P2→C first: Desc[P1] gets C (via Anc[P2] containing P1)
        // - Add P1→C second: Check Desc[P1] → C found → SKIP (correct!)
        // Wrong order would store P1→C before P2→C updates Desc[P1].

        // Collect (producer_id, edge_index) pairs for sorting
        std::vector<std::pair<EventId, uint8_t>> sorted_consumed;
        sorted_consumed.reserve(num_matched);
        for (uint8_t i = 0; i < num_matched; ++i) {
            EventId producer = hg_->get_edge_producer(matched_edges[i]);
            sorted_consumed.emplace_back(producer, i);
        }

        // Sort by producer ID DESCENDING (newest producers first)
        // INVALID_ID producers (initial edges with no producer) sort to end
        std::sort(sorted_consumed.begin(), sorted_consumed.end(),
            [](const auto& a, const auto& b) {
                if (a.first == INVALID_ID) return false;
                if (b.first == INVALID_ID) return true;
                return a.first > b.first;
            });

        // Add causal edges in sorted order
        for (const auto& [producer, idx] : sorted_consumed) {
            hg_->add_edge_consumer(matched_edges[idx], result.event);
        }

        // Register for branchial tracking (checks overlap with other events from same state)
        // Use the RAW input state ID for grouping (matching v1's behavior)
        // Branchial edges only connect events from the SAME actual state, not just canonically equivalent
        // Pass canonical_event_id to enable skipping branchial edges between equivalent events
        hg_->register_event_for_branchial(
            result.event, input_state, matched_edges, num_matched,
            event_result.canonical_event_id
        );

        result.success = true;
        return result;
    }

    // Apply a registered match
    RewriteResult apply(
        const RewriteRule& rule,
        StateId input_state,
        const Match& match,
        uint32_t output_step = 0
    ) {
        return apply(
            rule,
            input_state,
            match.matched_edges,
            match.num_edges,
            match.binding,
            output_step
        );
    }

    // Apply match by ID
    RewriteResult apply_match(
        const RewriteRule* rules,
        StateId input_state,
        MatchId match_id,
        uint32_t output_step = 0
    ) {
        const Match& match = hg_->get_match(match_id);
        const RewriteRule& rule = rules[match.rule_index];
        return apply(rule, input_state, match, output_step);
    }
};

// =============================================================================
// Convenience Function
// =============================================================================

// Apply a rewrite rule directly to a state with given matched edges
inline RewriteResult apply_rewrite(
    UnifiedHypergraph& hg,
    const RewriteRule& rule,
    StateId input_state,
    const EdgeId* matched_edges,
    uint8_t num_matched,
    const VariableBinding& binding,
    uint32_t output_step = 0
) {
    Rewriter rewriter(&hg);
    return rewriter.apply(rule, input_state, matched_edges, num_matched, binding, output_step);
}

}  // namespace hypergraph::unified
