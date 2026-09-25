#pragma once
#include "hgcommon/namespace.hpp"
#include "hgcommon/event_core.hpp"

namespace HG_NAMESPACE {
namespace common {

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
