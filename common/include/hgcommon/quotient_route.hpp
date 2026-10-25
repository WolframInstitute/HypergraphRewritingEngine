#pragma once
#include "hgcommon/namespace.hpp"
#include "hgcommon/event_core.hpp"

namespace HG_NAMESPACE {
namespace common {

// Whether quotient exploration runs when it is requested. It needs Full states. It does not run
// under MaxStatesPerStep: the step cap chooses among every transition of a step, raw copies
// included, and quotient exploration expands one raw state per class, at the class's shortest
// depth only. It does not run under "UndirectedEdges": the class frame maps a class's matches
// onto each instance through an identification that may reverse an edge, and a match's rule id
// names its orientation relative to the stored order of its edges (hgcommon/undirected_core.hpp).
// The FFI warns when it turns a request off.
constexpr bool quotient_exploration_applies(bool requested, bool full_states,
                                            uint64_t max_states_per_step, bool undirected) {
    return requested && full_states && max_states_per_step == 0 && !undirected;
}

// Whether a run asks for the quotient reconstruction: exploring canonical states only, or the
// Automatic event identity, whose edge ranks are read in the frame of the state's isomorphism
// class. Positional identity reads ranks in each state's own labeling and does not ask for it.
// The reconstruction is defined over canonical states and their edge orbits, which only Full
// state canonicalization computes, so a run takes it when this holds and states are Full.
constexpr bool quotient_route_requested(bool explore_canonical_states_only, bool positional,
                                        EventSignatureKeys event_keys) {
    return explore_canonical_states_only ||
           (!positional && event_keys == EVENT_SIG_AUTOMATIC);
}

}  // namespace common
}  // namespace HG_NAMESPACE
