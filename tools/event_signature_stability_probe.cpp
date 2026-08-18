// Does the CPU assign the SAME event signature values to the same evolution, run after run and
// worker count after worker count?
//
// The question exists because the GPU does not, on one measured state class. A canonical edge
// RANK is a position in a canonical LABELLING, and on a state whose automorphism group is
// nontrivial the canonical labelling is a coset: interchangeable edges can take each other's
// positions depending on the order the edges were presented in. The GPU's persistent scheduler
// was measured to permute the produced-edge ranks between 3 and 17 blocks on a directed 4-cycle
// while keeping the identity COUNT fixed.
//
// A single 1-worker-against-8-workers comparison passing does not establish that the CPU is
// stable -- it establishes that those two runs agreed. This runs every corpus case that has a
// nontrivial automorphism group across a spread of worker counts, repeatedly, and reports how
// many DISTINCT event fingerprints each cell ever produced. One means stable over the sample;
// more than one means the CPU permutes too, and by how often.
//
// Reports the state fingerprint alongside, so a cell where both move is separated from a cell
// where only the event identity does.

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "reference/golden_matrix.hpp"
#include "reference/oracle_corpus.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

struct Fingerprints {
    uint64_t states = 0;
    uint64_t events = 0;
    uint64_t event_count = 0;
};

Fingerprints run_once(const oracle::Case& c, StateCanonicalizationMode sm,
                      EventSignatureKeys ek, bool quotient, int steps, unsigned threads) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(sm);
    hg.set_event_signature_keys(ek);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_explore_from_canonical_states_only(quotient);
    for (const auto& r : c.rules) e.add_rule(r);
    e.evolve(c.init, steps);

    Fingerprints f;
    for (uint32_t sid = 0; sid < hg.num_published_states(); ++sid) {
        if (hg.get_state(sid).id == INVALID_ID) continue;
        f.states = golden::fold_fingerprint(f.states, hg.get_or_compute_canonical_hash(sid));
    }
    for (uint32_t eid = 0; eid < hg.num_published_events(); ++eid) {
        const Event& ev = hg.get_event(eid);
        if (ev.id == INVALID_ID || !ev.is_canonical()) continue;
        f.events = golden::fold_fingerprint(f.events, ev.signature);
    }
    f.event_count = hg.num_events();
    return f;
}

const char* mode_name(StateCanonicalizationMode m) { return golden::state_mode_name(m); }
const char* keys_name(EventSignatureKeys k)        { return golden::event_keys_name(k); }

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 12;
    const std::vector<unsigned> worker_counts = {1, 2, 3, 5, 8, 16};

    const std::vector<oracle::Case> corpus = oracle::corpus();
    const std::vector<StateCanonicalizationMode> state_modes = {
        StateCanonicalizationMode::None,
        StateCanonicalizationMode::Automatic,
        StateCanonicalizationMode::Full,
    };
    const std::vector<EventSignatureKeys> event_modes = {
        EVENT_SIG_NONE, EVENT_SIG_AUTOMATIC, EVENT_SIG_FULL,
    };

    std::printf("# event signature stability -- %d reps x %zu worker counts per cell\n",
                reps, worker_counts.size());
    std::printf("# case state event quotient distinct_event_fps distinct_state_fps "
                "distinct_event_counts\n");

    int unstable_cells = 0, total_cells = 0;
    for (const auto& c : corpus) {
        // The automorphism axis is the one this asks about. Restricted to it so the run is
        // short enough to repeat many times, which is what makes a "stable" reading mean
        // anything.
        if (std::string(c.type) != "automorphism") continue;

        for (auto sm : state_modes) {
            for (auto ek : event_modes) {
                for (bool quotient : {false, true}) {
                    std::set<uint64_t> event_fps, state_fps, counts;
                    for (int r = 0; r < reps; ++r) {
                        for (unsigned w : worker_counts) {
                            Fingerprints f = run_once(c, sm, ek, quotient, c.oracle_steps, w);
                            event_fps.insert(f.events);
                            state_fps.insert(f.states);
                            counts.insert(f.event_count);
                        }
                    }
                    ++total_cells;
                    const bool unstable = event_fps.size() > 1 || state_fps.size() > 1 ||
                                          counts.size() > 1;
                    if (unstable) ++unstable_cells;
                    std::printf("%s %s %s %d %zu %zu %zu%s\n",
                                c.name, mode_name(sm), keys_name(ek), quotient ? 1 : 0,
                                event_fps.size(), state_fps.size(), counts.size(),
                                unstable ? "   <-- UNSTABLE" : "");
                    std::fflush(stdout);
                }
            }
        }
    }

    std::printf("\n# %d of %d cells produced more than one fingerprint over %d reps x %zu "
                "worker counts\n", unstable_cells, total_cells, reps, worker_counts.size());
    return unstable_cells == 0 ? 0 : 1;
}
