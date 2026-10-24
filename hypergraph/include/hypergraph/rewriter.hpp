#pragma once
#include "hgcommon/namespace.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "types.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "hypergraph.hpp"
#include "hypergraph/debug_log.hpp"

namespace HG_NAMESPACE {
namespace engine {

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

    RewriteResult();
};

// One match of a generation (Rewriter::apply_generation).
struct GenerationMatch {
    const RewriteRule* rule;
    const EdgeId* edges;
    uint8_t num_edges;
    const VariableBinding* binding;
};

// What a generation made: the state, and how many events it recorded.
struct GenerationResult {
    StateId new_state = INVALID_ID;
    StateId raw_state = INVALID_ID;
    uint32_t num_events = 0;
    bool was_new_state = false;
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
    explicit Rewriter(Hypergraph* hg);

    // Apply a match to create a new state. `inherited` says every edge of the match predates the
    // input state, so its parent holds the match too and the rewrite may have been applied before
    // (hgcommon/token_core.hpp).
    RewriteResult apply(
        const RewriteRule& rule,
        StateId input_state,
        const EdgeId* matched_edges,
        uint8_t num_matched,
        const VariableBinding& binding,
        uint32_t output_step = 0,
        bool inherited = false
    );

    // A generation ("Generational"): matches of `input_state` that share no edge, applied
    // together. One state is made, the input state less every consumed edge and with every
    // produced edge, and one event per match from the input state to it, in the order given.
    // The produced edges and fresh vertices take consecutive ids in that order, so an edge's id
    // is its creation order. The state indexes all of its edges (no ancestry chain).
    GenerationResult apply_generation(StateId input_state, const GenerationMatch* matches,
                                      uint32_t num_matches, uint32_t output_step);

private:
    // Stores event `event_id` (from reserve_event_id) and registers it in the causal and
    // branchial relations. One body for apply and apply_generation.
    Hypergraph::CreateEventResult record_event(EventId event_id, StateId input_state,
                                               StateId output_state, const RewriteRule& rule,
                                               const EdgeId* consumed, uint8_t num_consumed,
                                               const EdgeId* produced, uint8_t num_produced,
                                               uint32_t rewrite_id);
};

// =============================================================================
// Convenience Function
// =============================================================================

// Apply a rewrite rule directly to a state with given matched edges
RewriteResult apply_rewrite(
    Hypergraph& hg,
    const RewriteRule& rule,
    StateId input_state,
    const EdgeId* matched_edges,
    uint8_t num_matched,
    const VariableBinding& binding,
    uint32_t output_step = 0
);

}  // namespace engine
}  // namespace HG_NAMESPACE