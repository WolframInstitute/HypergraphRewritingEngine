// rewriter.cpp - Implementation of Rewriter class

#include "hypergraph/rewriter.hpp"

namespace hypergraph {

RewriteResult Rewriter::apply(
    const RewriteRule& rule,
    StateId input_state,
    const EdgeId* matched_edges,
    uint8_t num_matched,
    const VariableBinding& binding,
    uint32_t output_step
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

}  // namespace hypergraph
