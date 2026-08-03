// rewriter.cpp - Implementation of Rewriter class

#include "hypergraph/rewriter.hpp"
#include "hgcommon/portable_intrinsics.hpp"
#include "hgcommon/rewrite_core.hpp"

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

    // Validate input_state. A stale forwarded match can reach here with a state ID
    // beyond the current count. Return an empty result (success=false); callers
    // already handle this (see execute_rewrite_task's `if (rr.new_state == INVALID_ID)`).
    uint32_t num_states = hg_->num_states();
    if (input_state >= num_states) {
        DEBUG_LOG("WARN: Rewriter::apply input_state=%u >= num_states=%u (stale match?)",
                  input_state, num_states);
        return result;
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

    // Build the child edge set: copy the parent's chunks and clear the consumed
    // edges (produced edges are added below as they are created). Chunk-granular
    // copy (memcpy per chunk) instead of a per-edge rebuild -- a state's edge set is
    // never mutated after creation, so the parent's chunks are stable to copy from.
    SparseBitset new_edges = SparseBitset::derive(
        input_edges, matched_edges, num_matched, nullptr, 0, hg_->arena());

    // Fresh vertices for the variables that occur only in the RHS. Allocation is this
    // device's business; WHICH variables get them and IN WHAT ORDER is the rewrite's, so
    // that part is hgcommon's and the device runs the same one.
    const uint32_t new_var_mask = rule.new_var_mask();
    const uint8_t num_fresh = hgcommon::num_fresh_variables(new_var_mask);
    // One block, so the fresh ids are consecutive and this takes ONE atomic rather than one
    // per new variable -- and it gives the host the same shape the device's high-water bump
    // already had, so both scatter them through the same shared rule.
    const VertexId fresh_base = num_fresh ? hg_->alloc_vertices(num_fresh) : 0;

    VertexId fresh_by_var[MAX_VARS];
    std::memset(fresh_by_var, 0xFF, sizeof(fresh_by_var));
    hgcommon::assign_fresh_consecutive(new_var_mask, fresh_base, fresh_by_var);

    // Create new edges from RHS pattern
    result.num_produced = 0;
    for (uint8_t i = 0; i < rule.num_rhs_edges; ++i) {
        const PatternEdge& rhs_edge = rule.rhs[i];

        // Resolve vertices for this edge. A false return means the rule names a variable
        // that is neither matched nor new, which is a malformed rule, not a failed match.
        VertexId vertices[MAX_ARITY];
        if (!hgcommon::resolve_rhs_vertices(rhs_edge.vars, rhs_edge.arity,
                                            binding.bindings, fresh_by_var, vertices)) {
            return result;
        }

        // Create the edge (producer will be set after event is created)
        EdgeId eid = hg_->create_edge(vertices, rhs_edge.arity, INVALID_ID, output_step);
        result.produced_edges[result.num_produced++] = eid;
        new_edges.set(eid, hg_->arena());
    }

    // Create or get existing canonical state (canonical hash computed inside,
    // mode-aware). Pass the rewrite delta so the WL hash can be computed
    // incrementally from the parent (input_state) when incremental WL is enabled.
    auto [canonical_id, raw_id, was_new] = hg_->create_or_get_canonical_state(
        std::move(new_edges),
        output_step,
        INVALID_ID,  // Will be updated when event is created
        input_state,
        matched_edges, num_matched,
        result.produced_edges, result.num_produced
    );

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
        result.num_produced
    );
    result.event = event_result.event_id;
    result.canonical_event = event_result.canonical_event_id;
    result.is_canonical_event = event_result.is_canonical;

    // =====================================================================
    // Online Causal/Branchial Tracking
    // =====================================================================

    // Quotient mode: capture this event's canonical transition into the causal skeleton
    // (deduplicated); the depth-indexed producer-set reconstruction propagates over it.
    if (hg_->quotient_causal()) {
        hg_->register_quotient_transition(result.event);
    }

    const RecordSet rec = hg_->record_set();

    // Full-capture causal: the stable raw edge id is the rendezvous key, so a producer and
    // any later consumer of the same edge instance meet directly. Under quotient this whole
    // block is replaced by the online depth-indexed reconstruction driven from
    // register_quotient_transition above (which is why it is gated off here).
    //
    // Skipped entirely when the run does not record causal: the producer map is read by this
    // rendezvous and by nothing else, so not filling it costs nothing else.
    if (!hg_->quotient_causal() && rec.causal) {
    // Register produced edges (set this event as producer). The produced edges live in
    // result.raw_state (the output state); key each by its canonical edge identity so
    // that, under quotient, every parent producing the same canonical edge orbit meets its
    // consumers at one rendezvous key -- making the producer set the full, schedule-
    // independent attribution. Off quotient the key is just the raw edge id.
    CanonicalEdgeKey produced_keys[MAX_PATTERN_EDGES];
    hg_->causal_edge_keys(result.raw_state, result.produced_edges, result.num_produced,
                          produced_keys);
    for (uint8_t i = 0; i < result.num_produced; ++i) {
        hg_->set_edge_producer(produced_keys[i], result.event, result.produced_edges[i]);
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

    // The consumed edges live in input_state; key each by its canonical edge identity so a
    // consumer rendezvous with every producer of that canonical edge (not just the one
    // parent whose raw output became the representative).
    CanonicalEdgeKey consumed_keys[MAX_PATTERN_EDGES];
    hg_->causal_edge_keys(input_state, matched_edges, num_matched, consumed_keys);

    // Collect (producer_id, edge_index) pairs for sorting (per-worker scratch arena;
    // recycled after the task, so no heap allocation on the rewrite hot path)
    ArenaVector<std::pair<EventId, uint8_t>> sorted_consumed(worker_scratch(), num_matched);
    for (uint8_t i = 0; i < num_matched; ++i) {
        EventId producer = hg_->get_edge_producer(consumed_keys[i]);
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
        hg_->add_edge_consumer(consumed_keys[idx], result.event, matched_edges[idx]);
    }
    }  // end !quotient_causal (full-capture rendezvous)

    // Register for branchial tracking (checks overlap with other events from same state)
    // Use the RAW input state ID for grouping (matching v1's behavior)
    // Branchial edges only connect events from the SAME actual state, not just canonically equivalent
    // Pass canonical_event_id to enable skipping branchial edges between equivalent events
    if (rec.branchial) {
        hg_->register_event_for_branchial(
            result.event, input_state, matched_edges, num_matched,
            event_result.canonical_event_id
        );
    }

    result.success = true;
    return result;
}

}  // namespace hypergraph
