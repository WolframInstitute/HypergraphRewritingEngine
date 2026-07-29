// The CPU half of the presentation-order question the GPU probe answered.
//
// Measured on the device (tools/gpu_rank_order_probe.cu, 20 runs): on a directed 4-cycle the
// presentation order of a state's edges -- the order the canonicalizer receives them -- differs
// between block counts in 15 of 20 runs, and the canonical edge RANKS differ with it. Measured
// on the host (tools/event_signature_stability_probe.cpp, 72 runs per cell): the event signature
// VALUES never move.
//
// Both cannot be explained by the same rule, so one of two things is true of the CPU:
//
//   (1) its presentation order is stable, and its rank rule is order-DEPENDENT like the
//       device's -- in which case the fix is to make the device's presentation stable; or
//   (2) its presentation order also moves, and its rank rule is order-INDEPENDENT -- in which
//       case the fix is to change how the device assigns ranks.
//
// This prints, per canonical state hash, the presentation order as the canonicalizer sees it
// (vertices renumbered by first appearance, exactly as the flattening does) together with the
// ranks assigned. Comparing two worker counts separates (1) from (2) directly.

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "reference/oracle_corpus.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

struct Shot {
    std::string tuples;
    std::string ranks;
};

std::map<uint64_t, std::vector<Shot>> run_once(const oracle::Case& c, bool quotient,
                                               int steps, unsigned threads) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_event_signature_keys(EVENT_SIG_AUTOMATIC);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_explore_from_canonical_states_only(quotient);
    for (const auto& r : c.rules) e.add_rule(r);
    e.evolve(c.init, steps);

    std::map<uint64_t, std::vector<Shot>> out;
    for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
        const State& st = hg.get_state(sid);
        if (st.id == INVALID_ID) continue;

        Shot shot;
        std::map<VertexId, uint32_t> local;
        std::vector<EdgeId> eids;
        st.edges.for_each([&](EdgeId eid) { eids.push_back(eid); });
        for (EdgeId eid : eids) {
            const Edge& ed = hg.get_edge(eid);
            shot.tuples += "(";
            for (uint8_t i = 0; i < ed.arity; ++i) {
                auto [it, fresh] = local.emplace(ed.vertices[i],
                                                 static_cast<uint32_t>(local.size()));
                (void)fresh;
                if (i) shot.tuples += ",";
                shot.tuples += std::to_string(it->second);
            }
            shot.tuples += ")";
            const uint32_t r = hg.edge_rank_in_state(sid, eid);
            shot.ranks += (r == UINT32_MAX ? std::string("-") : std::to_string(r)) + " ";
        }
        out[hg.get_or_compute_canonical_hash(sid)].push_back(shot);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned wa = argc > 1 ? std::atoi(argv[1]) : 1;
    const unsigned wb = argc > 2 ? std::atoi(argv[2]) : 8;
    const int reps    = argc > 3 ? std::atoi(argv[3]) : 20;

    const std::vector<oracle::Case> corpus = oracle::corpus();
    for (const auto& c : corpus) {
        if (std::string(c.type) != "automorphism") continue;
        for (bool quotient : {false, true}) {
            int order_moved = 0, rank_moved = 0, table_moved = 0;
            for (int rep = 0; rep < reps; ++rep) {
                auto a = run_once(c, quotient, c.oracle_steps, wa);
                auto b = run_once(c, quotient, c.oracle_steps, wb);
                bool om = false, rm = false, tm = false;
                for (const auto& [hash, sa] : a) {
                    auto it = b.find(hash);
                    if (it == b.end()) continue;
                    std::vector<std::string> ta, tb, ra, rb;
                    for (const auto& s : sa) { ta.push_back(s.tuples); ra.push_back(s.ranks); }
                    for (const auto& s : it->second) {
                        tb.push_back(s.tuples); rb.push_back(s.ranks);
                    }
                    std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
                    std::sort(ra.begin(), ra.end()); std::sort(rb.begin(), rb.end());
                    if (ta != tb) om = true;

                    // A "-" is a state with NO rank table, which the CPU builds lazily -- only
                    // for states an event signature actually asked about. Whether a given state
                    // was asked is itself schedule-dependent, and that is a different fact from
                    // two tables disagreeing. Counted apart so one is not read as the other.
                    auto has_gap = [](const std::vector<std::string>& v) {
                        for (const auto& s : v) if (s.find('-') != std::string::npos) return true;
                        return false;
                    };
                    if (ra != rb) {
                        if (has_gap(ra) || has_gap(rb)) tm = true; else rm = true;
                    }
                }
                if (om) ++order_moved;
                if (rm) ++rank_moved;
                if (tm) ++table_moved;
            }
            std::printf("%s quotient=%d workers %u vs %u over %d reps: "
                        "presentation moved %d, ranks moved %d, table-presence moved %d\n",
                        c.name, quotient ? 1 : 0, wa, wb, reps,
                        order_moved, rank_moved, table_moved);
            std::fflush(stdout);
        }
    }
    return 0;
}
