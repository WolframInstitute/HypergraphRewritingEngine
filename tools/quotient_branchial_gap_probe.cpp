// tools/quotient_branchial_gap_probe.cpp
//
// Minimises the one configuration where the reconstruction's branchial count parted
// from full-capture's: rules "all-three", initial state "selfloop", depth 3.
//
// The discrepancy surfaced only when ConcurrentMap::insert_if_absent was made to WAIT
// on claimed slots while scanning the resize chain (audit item A2). Both sides of the
// comparison are printed at several thread counts, because the question that decides
// what to do about it is which side moved: full-capture is single-threaded and cannot
// race, so if the reconstruction alone changes with thread count the fault is a race in
// the reconstruction, whereas a stable disagreement at one thread is a systematic bug in
// how one of the two computes branchial.
//
// Build:
//   g++ -O2 -std=c++17 -Ihypergraph/include -Icommon/include -Ijob_system/include \
//       -Ilockfree_deque/include tools/quotient_branchial_gap_probe.cpp \
//       build/libhypergraph.a -o /tmp/qbg -pthread

#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <vector>
using namespace hypergraph;

using Rules = std::vector<RewriteRule>;
using Init = std::vector<std::vector<VertexId>>;

static Rules all_three() {
    return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
             make_rule(1).lhs({0,1}).rhs({1,0}).build(),
             make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() };
}

struct Counts { size_t events, pairs, branchial, states, canon; };

static Counts full_capture(int steps, int threads) {
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(false);
    for (const auto& r : all_three()) e.add_rule(r);
    Init in = {{0,0}}; e.evolve(in, steps);
    return { hg.num_events(), hg.causal_graph().num_causal_event_pairs(),
             hg.causal_graph().num_branchial_edges(), hg.num_states(),
             hg.num_canonical_states() };
}

static Counts reconstructed(int steps, int threads) {
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(true);
    hg.set_quotient_reconstruction(true);
    for (const auto& r : all_three()) e.add_rule(r);
    Init in = {{0,0}}; e.evolve(in, steps);
    return { hg.num_reconstructed_events(), hg.num_reconstructed_causal_pairs(false),
             hg.num_reconstructed_branchial(), hg.num_states(),
             hg.num_canonical_states() };
}

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 3;
    std::printf("all-three / selfloop / depth %d\n", steps);
    std::printf("  %-28s %7s %7s %10s %7s %7s\n",
                "arm", "events", "pairs", "branchial", "states", "canon");
    for (int t : {1, 2, 4, 8}) {
        Counts f = full_capture(steps, t);
        std::printf("  full-capture   threads=%-3d  %7zu %7zu %10zu %7zu %7zu\n",
                    t, f.events, f.pairs, f.branchial, f.states, f.canon);
    }
    for (int t : {1, 2, 4, 8}) {
        Counts r = reconstructed(steps, t);
        std::printf("  reconstruction threads=%-3d  %7zu %7zu %10zu %7zu %7zu\n",
                    t, r.events, r.pairs, r.branchial, r.states, r.canon);
    }
    return 0;
}
