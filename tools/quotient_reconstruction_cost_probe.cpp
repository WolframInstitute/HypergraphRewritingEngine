// What does exact quotient causal COST?
//
// #4 comes down to a decision, not an implementation: the engine's per-instance replay is exact
// but sits behind quotient_reconstruction_, which defaults FALSE. The replay walks every INSTANCE,
// so its cost is of the order of full capture -- which is what quotient exists to avoid. Whether
// the flag should default on therefore turns on how much it actually costs, and on whether the
// STATE-space saving survives even when the causal replay does not.
//
// Reported per workload:
//   states   quotient's saving, which the flag does NOT touch
//   causal   what the caller gets: qc_emit's subset with the flag off, the exact multiset with it on
//   time     the price of exactness
//
// Deliberately measured through the engine at several depths: the replay is per-instance, so its
// cost grows with the provenance count, and a single shallow number would understate it.
#include <chrono>
#include <cstdio>
#include <cstring>

#include "hgcommon/build_stamp.hpp"
#include <vector>
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
using namespace hypergraph;

struct R { size_t states, events, causal; double ms; };

static R run(const RewriteRule& rule, std::vector<std::vector<VertexId>> init,
             int steps, bool quotient, bool recon) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    if (quotient && recon) hg.set_quotient_reconstruction(true);
    ParallelEvolutionEngine e(&hg, 4);
    e.set_explore_from_canonical_states_only(quotient);
    e.add_rule(rule);
    const auto t0 = std::chrono::steady_clock::now();
    e.evolve(init, steps);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const size_t causal = (quotient && recon) ? hg.num_reconstructed_causal_edges()
                                             : hg.causal_graph().num_causal_edges();
    const size_t events = (quotient && recon) ? hg.num_reconstructed_events() : hg.num_events();
    return R{hg.num_canonical_states(), events, causal, ms};
}

int main(int argc, char** argv) {
    // The configuration record, first and alone on --build-info: paper_tables.py gates on it
    // before trusting any number this prints (hgcommon/build_stamp.hpp).
    static const char kBuildStamp[] = HG_BUILD_STAMP_LITERAL;
    if (argc > 1 && std::strcmp(argv[1], "--build-info") == 0) { std::printf("%s\n", kBuildStamp); return 0; }
    std::printf("%s\n", kBuildStamp);
    const int maxd = argc > 1 ? std::atoi(argv[1]) : 5;
    const RewriteRule rule =
        make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).build();
    const std::vector<std::vector<VertexId>> init = {{0,1},{1,2}};

    std::printf("# wolfram-2to4, 4 workers, Full canon. QUOT+recon uses the engine's own\n"
                "# num_reconstructed_*; QUOT alone reports what a caller receives today.\n\n");
    std::printf("%5s | %-22s | %-22s | %-22s\n", "steps",
                "FULL  st/ev/caus/ms", "QUOT  st/ev/caus/ms", "QUOT+recon st/ev/caus/ms");
    for (int d = 2; d <= maxd; ++d) {
        const R f = run(rule, init, d, false, false);
        const R q = run(rule, init, d, true,  false);
        const R r = run(rule, init, d, true,  true);
        std::printf("%5d | %4zu %5zu %5zu %6.1f | %4zu %5zu %5zu %6.1f | %4zu %5zu %5zu %6.1f  %s\n",
            d, f.states, f.events, f.causal, f.ms,
               q.states, q.events, q.causal, q.ms,
               r.states, r.events, r.causal, r.ms,
            (r.events == f.events && r.causal == f.causal) ? "exact" : "DIFFERS");
    }
    std::printf("\n# The states column is the quotient saving and is identical with and without\n"
                "# the flag: exactness costs causal-replay time, never state-space growth.\n");
    return 0;
}
