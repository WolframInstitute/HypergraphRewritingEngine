// Single-threaded representative multiway evolution, for deterministic profiling
// under callgrind/cachegrind (which serialize threads and simulate the cache, so
// their counts are immune to host-load noise). Drives the Wolfram rule to a fixed
// depth under a chosen state-canonicalization mode.
//
// Usage: profile_evolve [steps] [mode]   mode = none | auto | full   (default 5 full)
#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
using namespace hypergraph;

int main(int argc, char** argv) {
    int steps = argc > 1 ? std::atoi(argv[1]) : 5;
    const char* mode = argc > 2 ? argv[2] : "full";

    Hypergraph hg;
    StateCanonicalizationMode m = StateCanonicalizationMode::Full;
    if (!std::strcmp(mode, "none")) m = StateCanonicalizationMode::None;
    else if (!std::strcmp(mode, "auto")) m = StateCanonicalizationMode::Automatic;
    hg.set_state_canonicalization_mode(m);

    ParallelEvolutionEngine e(&hg, 1);  // single worker: clean per-function attribution
    // Wolfram rule {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}}
    e.add_rule(make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build());

    std::vector<std::vector<VertexId>> init = {{0u,1u},{0u,2u}};
    e.evolve(init, steps);

    std::printf("mode=%s steps=%d states=%u canonical=%u events=%u\n",
                mode, steps, hg.num_states(), hg.num_canonical_states(), hg.num_events());
    return 0;
}
