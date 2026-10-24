#include "hgcommon/namespace.hpp"
// rewriter.cpp - Implementation of Rewriter class

#include "hypergraph/rewriter.hpp"
#include "hgcommon/portable_intrinsics.hpp"
#include "hgcommon/rewrite_core.hpp"
#include "hgcommon/token_core.hpp"

namespace HG_NAMESPACE {
namespace engine {

RewriteResult Rewriter::apply(
    const RewriteRule& rule,
    StateId input_state,
    const EdgeId* matched_edges,
    uint8_t num_matched,
    const VariableBinding& binding,
    uint32_t output_step,
    bool inherited
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
            // A MATCH DROPPED HERE IS AN EVENT THAT NEVER HAPPENS, and the caller cannot tell:
            // it reads the empty result as "this rewrite produced nothing", gives back its
            // budget slots and moves on. The run then returns one event short, with no error and
            // no warning, which is exactly what non-determinism looks like when two thread counts
            // are compared. So it is counted, and the determinism gate asserts it is zero.
            //
            // It should be unreachable. A match is either produced by matching the very state it
            // is applied to, or FORWARDED to a child from its parent -- and forwarding is
            // supposed to carry only the matches that survive the parent's rewrite.
            hg_->note_invalid_match();
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
    // The rewrite's fresh vertices and its new edges, in one increment of the shared counter.
    EdgeId first_edge = 0;
    VertexId fresh_base = 0;
    if (num_fresh || rule.num_rhs_edges)
        hg_->alloc_edges_and_vertices(rule.num_rhs_edges, num_fresh, first_edge, fresh_base);

    // Keyed rewrites (hgcommon/token_core.hpp): from the run's first inherited match on, every
    // rewrite is interned; its state claims a twin when the rewrite may have been applied before.
    // `keyed_rewrite` is the rewrite id (0: none) with REWRITE_TWIN_CANDIDATE set when the state
    // claims a twin. ARMED plus an inherited match, or INTERNING, interns.
    uint32_t keyed_rewrite = 0;
    const uint8_t keyed = hg_->keyed_state();
    if (keyed + uint8_t(inherited) >= Hypergraph::KEYED_INTERNING) {
        if (keyed == Hypergraph::KEYED_ARMED) hg_->note_inherited_rewrite();
        bool repeated = false;
        const uint32_t rid = hg_->intern_rewrite(rule.index, matched_edges, num_matched, repeated);
        if (rid != hgcommon::REWRITE_ID_NONE)
            keyed_rewrite = rid | ((inherited || repeated) ? hgcommon::REWRITE_TWIN_CANDIDATE : 0u);
    }

    VertexId fresh_by_var[MAX_VARS];
    std::memset(fresh_by_var, 0xFF, sizeof(fresh_by_var));
    hgcommon::assign_fresh_consecutive(new_var_mask, fresh_base, fresh_by_var);

    // The event's id is taken before its produced edges, so each is created with its creator.
    const EventId event_id = hg_->reserve_event_id();
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
        EdgeId eid = hg_->create_edge_at(first_edge + i, vertices, rhs_edge.arity, event_id,
                                         output_step);
        result.produced_edges[result.num_produced++] = eid;
        new_edges.set(eid, hg_->arena());
    }
    // A known rewrite id gives the produced tokens now, before the state exists.
    if (keyed_rewrite) {
        const uint32_t rid = keyed_rewrite & ~hgcommon::REWRITE_TWIN_CANDIDATE;
        for (uint8_t i = 0; i < result.num_produced; ++i)
            hg_->cache_edge_token(result.produced_edges[i], hgcommon::token_produced(rid, i));
    }

    // Create or get existing canonical state (canonical hash computed inside,
    // mode-aware). Pass the rewrite delta so the WL hash can be computed
    // incrementally from the parent (input_state) when incremental WL is enabled.
    auto [canonical_id, raw_id, was_new] = hg_->create_or_get_canonical_state(
        std::move(new_edges),
        output_step,
        event_id,    // created below with this state as its output
        input_state,
        matched_edges, num_matched,
        result.produced_edges, result.num_produced,
        keyed_rewrite
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
    const auto event_result = record_event(
        event_id, input_state, result.raw_state, rule, matched_edges, num_matched,
        result.produced_edges, result.num_produced,
        keyed_rewrite & ~hgcommon::REWRITE_TWIN_CANDIDATE);
    result.event = event_result.event_id;
    result.canonical_event = event_result.canonical_event_id;
    result.is_canonical_event = event_result.is_canonical;
    result.success = true;
    return result;
}

Hypergraph::CreateEventResult Rewriter::record_event(EventId event_id, StateId input_state,
                                                     StateId output_state,
                                                     const RewriteRule& rule,
                                                     const EdgeId* matched_edges,
                                                     uint8_t num_matched,
                                                     const EdgeId* produced_edges,
                                                     uint8_t num_produced,
                                                     uint32_t rewrite_id) {
    const auto event_result = hg_->create_event_at(event_id, input_state, output_state,
                                                   rule.index, matched_edges, num_matched,
                                                   produced_edges, num_produced, rewrite_id);
    const EventId event = event_result.event_id;

    // =====================================================================
    // Online Causal/Branchial Tracking
    // =====================================================================

    // Quotient mode: capture this event's canonical transition into the causal skeleton
    // (deduplicated); the depth-indexed producer-set reconstruction propagates over it.
    // HG_HARNESS_DEFER_QUOTIENT_CAPTURE (GenMC only, verification/genmc/quotient_capture_*.cpp):
    // the harness calls register_quotient_transition itself, so the registration and the
    // capture are checked as separate harnesses.
#if !defined(HG_HARNESS_DEFER_QUOTIENT_CAPTURE)
    if (hg_->quotient_causal()) {
        hg_->register_quotient_transition(event);
    }
#endif

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
    // the output state; key each by its canonical edge identity so
    // that, under quotient, every parent producing the same canonical edge orbit meets its
    // consumers at one rendezvous key -- making the producer set the full, schedule-
    // independent attribution. Off quotient the key is just the raw edge id.
    CanonicalEdgeKey produced_keys[MAX_PATTERN_EDGES];
    hg_->causal_edge_keys(output_state, produced_edges, num_produced, produced_keys);
    for (uint8_t i = 0; i < num_produced; ++i) {
        hg_->set_edge_producer(produced_keys[i], event, produced_edges[i]);
    }

    // Register this event as a consumer of each consumed edge. The consumed edges live in
    // input_state; key each by its canonical edge identity so a consumer meets every producer
    // of that canonical edge, not just the one parent whose raw output became the
    // representative. The order the in-edges are recorded in is the reduction's business and
    // is decided in consume_edges.
    CanonicalEdgeKey consumed_keys[MAX_PATTERN_EDGES];
    hg_->causal_edge_keys(input_state, matched_edges, num_matched, consumed_keys);
    hg_->causal_graph().consume_edges(consumed_keys, matched_edges, num_matched, event);
    }  // end !quotient_causal (full-capture rendezvous)

    // Branchial, in two independent recordings. Grouping is by the RAW input state id: two
    // events branch when they consumed a common edge of the SAME actual state, not of a
    // canonically equivalent one, so event canonicalization does not enter either.
    if (rec.state_events) {
        hg_->record_state_event(event, input_state);
    }
    if (hgcommon::records_explored_branchial(rec, hg_->quotient_causal())) {
        hg_->record_branchial_overlaps(event, input_state, matched_edges, num_matched);
    }
    return event_result;
}

GenerationResult Rewriter::apply_generation(StateId input_state, const GenerationMatch* matches,
                                            uint32_t num_matches, uint32_t output_step) {
    GenerationResult result;
    if (num_matches == 0 || input_state >= hg_->num_states()) return result;
    const SparseBitset& input_edges = hg_->get_state_edges(input_state);
    std::vector<EdgeId> consumed;
    uint32_t num_produced = 0, num_fresh = 0;
    for (uint32_t m = 0; m < num_matches; ++m) {
        for (uint8_t i = 0; i < matches[m].num_edges; ++i) {
            if (!input_edges.contains(matches[m].edges[i])) {
                hg_->note_invalid_match();
                return result;
            }
            consumed.push_back(matches[m].edges[i]);
        }
        num_produced += matches[m].rule->num_rhs_edges;
        num_fresh += hgcommon::num_fresh_variables(matches[m].rule->new_var_mask());
    }
    SparseBitset new_edges = SparseBitset::derive(input_edges, consumed.data(), consumed.size(),
                                                  nullptr, 0, hg_->arena());
    EdgeId first_edge = 0;
    VertexId fresh_base = 0;
    if (num_produced || num_fresh)
        hg_->alloc_edges_and_vertices(num_produced, num_fresh, first_edge, fresh_base);

    // Each match's event id, produced edges and fresh vertices, in the order given.
    std::vector<EventId> events(num_matches);
    std::vector<EdgeId> produced(num_produced);
    uint32_t edge_at = 0;
    VertexId fresh_at = fresh_base;
    for (uint32_t m = 0; m < num_matches; ++m) {
        const RewriteRule& rule = *matches[m].rule;
        events[m] = hg_->reserve_event_id();
        VertexId fresh_by_var[MAX_VARS];
        std::memset(fresh_by_var, 0xFF, sizeof(fresh_by_var));
        hgcommon::assign_fresh_consecutive(rule.new_var_mask(), fresh_at, fresh_by_var);
        fresh_at += hgcommon::num_fresh_variables(rule.new_var_mask());
        for (uint8_t i = 0; i < rule.num_rhs_edges; ++i) {
            VertexId vertices[MAX_ARITY];
            if (!hgcommon::resolve_rhs_vertices(rule.rhs[i].vars, rule.rhs[i].arity,
                                                matches[m].binding->bindings, fresh_by_var,
                                                vertices))
                return result;
            const EdgeId eid = hg_->create_edge_at(first_edge + edge_at, vertices,
                                                   rule.rhs[i].arity, events[m], output_step);
            produced[edge_at++] = eid;
            new_edges.set(eid, hg_->arena());
        }
    }
    auto [canonical_id, raw_id, was_new] = hg_->create_or_get_canonical_state(
        std::move(new_edges), output_step, events[0], INVALID_ID, nullptr, 0, nullptr, 0, 0);
    result.new_state = canonical_id;
    result.raw_state = raw_id;
    result.was_new_state = was_new;
    edge_at = 0;
    for (uint32_t m = 0; m < num_matches; ++m) {
        const RewriteRule& rule = *matches[m].rule;
        record_event(events[m], input_state, raw_id, rule, matches[m].edges,
                     matches[m].num_edges, &produced[edge_at], rule.num_rhs_edges, 0);
        edge_at += rule.num_rhs_edges;
    }
    result.num_events = num_matches;
    return result;
}


// =============================================================================
// RewriteResult / Rewriter construction, and the free-function form of apply
// =============================================================================

RewriteResult::RewriteResult()
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

Rewriter::Rewriter(Hypergraph* hg) : hg_(hg) {}

RewriteResult apply_rewrite(
    Hypergraph& hg,
    const RewriteRule& rule,
    StateId input_state,
    const EdgeId* matched_edges,
    uint8_t num_matched,
    const VariableBinding& binding,
    uint32_t output_step
) {
    Rewriter rewriter(&hg);
    return rewriter.apply(rule, input_state, matched_edges, num_matched, binding, output_step);
}

}  // namespace engine
}  // namespace HG_NAMESPACE
