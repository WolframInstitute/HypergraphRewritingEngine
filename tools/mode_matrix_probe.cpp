// The 3x3 identity surface, printed.
//
// SPEC.md sec 4 defines two INDEPENDENT axes, each a three-point refinement lattice. This
// prints every cell for every corpus case so the surface is visible rather than asserted:
// which cells differ, which coincide on a given workload, and by how much.
//
// It answers the question that inspection cannot: is a mode a genuinely distinct point, or has
// it silently become a duplicate of its neighbour? A gate built only from equalities between
// runs of the SAME configuration cannot see that, because a duplicated mode still equals
// itself. FeatureMatrix.IdentityLatticesAreOrdered asserts the ordering; this shows the values.

#include <cstdio>
#include <string>
#include <vector>

#include "reference/oracle_corpus.hpp"

using namespace hypergraph;

namespace {

struct StateAxis { StateCanonicalizationMode mode; const char* name; };
struct EventAxis { EventSignatureKeys keys;      const char* name; };

// Coarsest first on both axes, so a row or column that fails to widen is visible as a
// repeated number rather than having to be worked out.
const StateAxis kStates[] = {
    {StateCanonicalizationMode::Full,      "Full"},
    {StateCanonicalizationMode::Automatic, "Automatic"},
    {StateCanonicalizationMode::None,      "None"},
};
const EventAxis kEvents[] = {
    {EVENT_SIG_FULL,      "ByEndpointStates"},
    {EVENT_SIG_AUTOMATIC, "ByConsumedProducedEdges"},
    {EVENT_SIG_NONE,      "DistinctApplications"},
};

struct Cell { size_t states, events, causal, branchial; uint64_t raw_fallbacks; };

Cell run(const std::vector<RewriteRule>& rules,
         const std::vector<std::vector<VertexId>>& init,
         int steps, StateCanonicalizationMode sm, EventSignatureKeys ek) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(sm);
    hg.set_event_signature_keys(ek);
    ParallelEvolutionEngine engine(&hg, 1);
    for (const auto& r : rules) engine.add_rule(r);
    engine.evolve(init, steps);
    // num_events() reports the CANONICAL event count whenever an event signature is set,
    // and the raw count when it is not -- which is exactly the event-identity axis.
    return Cell{hg.num_canonical_states(), hg.num_events(),
                hg.causal_graph().num_causal_edges(), hg.causal_graph().num_branchial_edges(),
                hg.event_signature_raw_fallbacks()};
}

}  // namespace

int main() {
    std::printf("3x3 identity surface: state x event, one thread, counts = "
                "(canonical states, canonical events, causal edges, branchial edges), "
                "and raw= when an event signature fell back to a raw edge id\n");
    std::printf("Both axes run COARSEST FIRST, so counts must be non-decreasing left to right "
                "and top to bottom.\n\n");

    size_t cases = 0, state_sep = 0, event_sep = 0, dependent = 0;

    for (const auto& c : oracle::corpus()) {
        ++cases;
        std::printf("%s (steps=%d)\n", c.name, c.oracle_steps);
        std::printf("  %-12s", "");
        for (const auto& e : kEvents) std::printf(" %-26s", e.name);
        std::printf("\n");

        Cell grid[3][3];
        for (int si = 0; si < 3; ++si) {
            std::printf("  %-12s", kStates[si].name);
            for (int ei = 0; ei < 3; ++ei) {
                grid[si][ei] = run(c.rules, c.init, c.oracle_steps,
                                   kStates[si].mode, kEvents[ei].keys);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%zu/%zu/%zu/%zu%s",
                              grid[si][ei].states, grid[si][ei].events,
                              grid[si][ei].causal, grid[si][ei].branchial,
                              grid[si][ei].raw_fallbacks
                                  ? (" raw=" + std::to_string(grid[si][ei].raw_fallbacks)).c_str()
                                  : "");
                std::printf(" %-26s", buf);
            }
            std::printf("\n");
        }

        // SPEC sec 4: the axes are INDEPENDENT, so the event count must be constant down each
        // column -- changing how states are merged cannot change how many distinct events
        // there are. The reference has this property (MultiwayReference gives 8 events for
        // binary-growth under both Canonical and None); the engine did not, because it
        // resolved consumed/produced edges through a representative state.
        for (int ei = 0; ei < 3; ++ei) {
            if (grid[0][ei].events != grid[1][ei].events ||
                grid[1][ei].events != grid[2][ei].events) {
                std::printf("  AXES NOT INDEPENDENT under %s: events %zu/%zu/%zu "
                            "for state Full/Automatic/None\n",
                            kEvents[ei].name, grid[0][ei].events,
                            grid[1][ei].events, grid[2][ei].events);
                ++dependent;
            }
        }

        // A case separates an axis when moving along it changes the count at all.
        bool ssep = grid[0][0].states != grid[1][0].states
                 || grid[1][0].states != grid[2][0].states;
        bool esep = grid[0][0].events != grid[0][1].events
                 || grid[0][1].events != grid[0][2].events;
        if (ssep) ++state_sep;
        if (esep) ++event_sep;
        std::printf("  separates: state axis %s, event axis %s\n\n",
                    ssep ? "YES" : "no", esep ? "YES" : "no");
    }

    std::printf("%zu cases | state axis separated by %zu | event axis separated by %zu "
                "| axis-independence violations %zu\n",
                cases, state_sep, event_sep, dependent);
    if (state_sep == 0 || event_sep == 0) {
        std::printf("AXIS UNTESTED: no corpus case distinguishes one of the axes.\n");
        return 1;
    }
    return dependent ? 1 : 0;
}
