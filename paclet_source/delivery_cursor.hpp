#pragma once
#include "hgcommon/namespace.hpp"
//
// WHAT A SESSION HAS ALREADY SENT, so a Step can report what it ADDED.
//
// WHY. A session avoids re-COMPUTING and, until this, not re-SENDING: every Step re-serialised
// the whole accumulated graph. tools/dev/session_step_cost.wls measured what that costs by
// timing a Step against a Query at the same depth -- Query does the same serialisation and no
// exploration -- and the ratio is ~1.0 at every depth past the first (24.8 MB and 1.9 s per
// Step at depth 7). So the reply IS the interactive cost, and the exploration is noise.
//
// THE DELTA IS A RECORD, NOT A DIFF, which is what the engine's shape already permits and what
// the non-phasic rule requires anyway. States are created once and their edge lists are
// immutable; effective ids are first-writer-wins; add_causal_edge only ever pushes. The one
// thing that CHANGES after delivery is a state's step, which try_lower_explore_depth may lower,
// and that is why a vertex carries a REVISION here rather than only presence.
//
// KEYED BY GRAPH PROPERTY, and that is load-bearing. A caller may ask for StatesGraphStructure
// on one Step and StatesGraph on the next; the second must receive the WHOLE graph at full
// payload, because it has never received those vertices in that shape. Keying the record by
// property is what makes "first requested at step k" mean "gets everything at step k".
//
// AN UNRECORDED PROPERTY IS NOT AN EMPTY ONE. A cursor that has never seen a property reports
// every vertex and edge as new, which is exactly a full delivery -- so the delta path and the
// full path are the same code with a different starting record, rather than two paths that
// agree until one is edited.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace HG_NAMESPACE {
namespace ffi {

class DeliveryCursor {
public:
    // True when this vertex has not been sent for this property, or was sent with a different
    // revision. Records it either way, so a caller loops over vertices once.
    bool take_vertex(const std::string& property, int64_t id, uint32_t revision);

    // True when this edge has not been sent for this property. Edges have no revision: an edge
    // is a pair of endpoints and a type, and none of the three changes after it is minted.
    bool take_edge(const std::string& property, uint64_t key);

    // Has this property ever been delivered? A caller needs this to decide whether a reply is a
    // delta to merge or a graph to replace -- the FIRST delivery of a property is a whole graph
    // even in delta mode.
    bool delivered_before(const std::string& property) const;

    // Forget everything. A caller that asks for a full delivery is asking to resynchronise, and
    // its next delta must be measured from that, not from what was sent before it.
    void reset();

private:
    struct PropertyRecord {
        std::unordered_map<int64_t, uint32_t> vertex_revision;
        std::unordered_set<uint64_t> edges;
    };
    std::unordered_map<std::string, PropertyRecord> by_property_;
};

}  // namespace ffi
}  // namespace HG_NAMESPACE
