#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "types.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "hypergraph.hpp"
#include "hypergraph/debug_log.hpp"

namespace hypergraph {

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
    Hypergraph* hg_;

public:
    explicit Rewriter(Hypergraph* hg) : hg_(hg) {}

    // Apply a match to create a new state
    RewriteResult apply(
        const RewriteRule& rule,
        StateId input_state,
        const EdgeId* matched_edges,
        uint8_t num_matched,
        const VariableBinding& binding,
        uint32_t output_step = 0
    );

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
    Hypergraph& hg,
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

}  // namespace hypergraph
