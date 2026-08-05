#pragma once
//
// One parsed job: everything `run_rewriting_core` reads out of the WXF envelope, and nothing it
// derives afterwards.
//
// WHY THIS EXISTS. `run_rewriting_core` is 1664 of `hypergraph_ffi.cpp`'s 1849 lines, and it
// resisted decomposition because every phase after the parse reads the same 38 locals -- which
// is why #12's op boundary had to be drawn by binding references and gating a block rather than
// by extracting a function. The parse is the one seam whose dependency runs ONE WAY: it writes
// these and reads nothing a later phase produces.
//
// A FIELD'S INITIALISER IS THE OPTION'S DEFAULT. That makes this the single place that answers
// "what happens if I omit it", which was previously spread through the declaration block of a
// 1600-line function.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "hypergraph/types.hpp"

namespace hgffi {

// The warning trail served under the "Warnings" result key, schema shared with the GPU backend
// (Kind/Count/Context) so the WL formatter handles both backends. Collects option-parse skips,
// analysis refusals, and the engine's own warnings.
struct FfiWarning {
    std::string kind;
    int64_t count;
    std::string context;
};

struct ParsedJob {
    std::vector<std::vector<std::vector<int64_t>>> initial_states_raw;
    std::vector<std::pair<std::string, std::vector<std::vector<std::vector<int64_t>>>>> parsed_rules_raw;
    int steps = 1;

    std::vector<FfiWarning> ffi_warnings;

    // Option values
    hypergraph::StateCanonicalizationMode state_canon_mode = hypergraph::StateCanonicalizationMode::None;  // Default: tree mode
    hypergraph::EventSignatureKeys event_signature_keys = hypergraph::EVENT_SIG_NONE;  // Default: no event canonicalization
    bool positional_event_identity = false;  // CanonicalizeEvents -> "Positional"
    bool show_genesis_events = false;
    bool show_progress = false;
    bool causal_transitive_reduction = true;
    size_t max_successor_states_per_parent = 0;
    size_t max_states_per_step = 0;
    double exploration_probability = 1.0;
    bool explore_from_canonical_states_only = false;  // Exploration deduplication
    bool quotient_initial_states = false;             // Collapse isomorphic initial states
    // ir_verification and return_canonical_states are derived from state_canon_mode == Full
    uint64_t random_seed = 0;     // 0: a fresh seed per run; nonzero fixes the sample
    bool uniform_random = false;  // Use uniform random match selection (reservoir sampling)
    size_t matches_per_step = 0;  // Matches per step in uniform random mode (0 = all)

    // Data selection flags - which components to include in output
    // By default all are included for backward compatibility
    bool include_states = true;
    bool include_canonical_hashes = false;  // Emit per-state IR canonical hash (CanonicalHash); stable across runs, for cross-run fusion
    bool include_events = true;
    bool include_events_minimal = false;  // Minimal event data: Id, InputState, OutputState only
    bool include_causal_edges = true;
    bool include_branchial_edges = true;       // Event-to-event (for Evolution*Branchial)
    bool include_branchial_state_edges = false; // State-to-state (for BranchialGraph) - overlap-based
    bool include_branchial_state_edges_all_siblings = false; // State-to-state all siblings (no overlap check)
    int branchial_step = 0;  // 0=All steps, positive=1-based step, negative=from end (-1=final)
    bool edge_deduplication = true;  // True: one edge per (from,to) pair; False: N edges for N shared hypergraph edges
    bool include_num_states = true;
    bool include_num_events = true;
    bool include_num_causal_edges = true;
    bool include_num_branchial_edges = true;
    bool include_global_edges = false;      // All edges created during evolution
    bool include_state_bitvectors = false;  // State edge sets as lists of edge IDs

    // GraphProperties option for graph-ready data output (list of properties)
    std::vector<std::string> graph_properties;  // e.g., {"StatesGraph", "CausalGraphStructure"}
    std::string canonicalize_states_mode = "None";  // Track actual mode string for effective ID computation

    // The session envelope. Empty `Op` means `Evolve`, which is the whole of today's protocol, so
    // a caller that sends neither key is served exactly as before. `Session` is the opaque handle;
    // 0 is "no session", the same reserved-zero discipline every other id space here follows.
    std::string session_op;
    uint64_t session_handle = 0;
};

}  // namespace hgffi
