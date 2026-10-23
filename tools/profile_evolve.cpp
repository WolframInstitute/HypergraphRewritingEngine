// Single-threaded representative multiway evolution, for deterministic profiling
// under callgrind/cachegrind (which serialize threads and simulate the cache, so
// their counts are immune to host-load noise). Drives the Wolfram rule to a fixed
// depth under a chosen state-canonicalization mode.
//
// Usage: profile_evolve [steps] [mode] [threads] [edges_file matches_per_step seed]
//   mode = none | auto | full   (default 5 full 1)
// threads defaults to 1, which is what keeps per-function attribution clean; raise it to
// exercise the concurrent paths (contention, false sharing, the map rendezvous).
// With edges_file (one binary edge "a b" per line), the run starts from those edges under
// {{x,y},{y,z},{z,w},{w,v}} -> {{y,u},{u,v},{w,x},{x,u}} and keeps matches_per_step transitions
// per step ("UniformRandom" -> True, "MatchesPerStep" -> matches_per_step, "RandomSeed" -> seed).
#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace hypergraph;

int main(int argc, char** argv) {
    int steps = argc > 1 ? std::atoi(argv[1]) : 5;
    const char* mode = argc > 2 ? argv[2] : "full";
    int threads = argc > 3 ? std::atoi(argv[3]) : 1;
    const char* edges_file = argc > 4 ? argv[4] : nullptr;

    Hypergraph hg;
    StateCanonicalizationMode m = StateCanonicalizationMode::Full;
    if (!std::strcmp(mode, "none")) m = StateCanonicalizationMode::None;
    else if (!std::strcmp(mode, "auto")) m = StateCanonicalizationMode::Automatic;
    hg.set_state_canonicalization_mode(m);

    ParallelEvolutionEngine e(&hg, threads);
    std::vector<std::vector<VertexId>> init = {{0u,1u},{0u,2u}};
    if (edges_file) {
        std::FILE* f = std::fopen(edges_file, "r");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", edges_file); return 1; }
        init.clear();
        unsigned a, b;
        while (std::fscanf(f, "%u %u", &a, &b) == 2) init.push_back({a, b});
        std::fclose(f);
        // {{x,y},{y,z},{z,w},{w,v}} -> {{y,u},{u,v},{w,x},{x,u}}, x..v = 0..4, u = 5
        e.add_rule(make_rule(0).lhs({0,1}).lhs({1,2}).lhs({2,3}).lhs({3,4})
                       .rhs({1,5}).rhs({5,4}).rhs({3,0}).rhs({0,5}).build());
        e.set_matches_per_step(argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 1);
        e.set_random_seed(argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 1);
    } else {
        // Wolfram rule {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}}
        e.add_rule(make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build());
    }

    e.evolve(init, steps);

    std::printf("mode=%s steps=%d threads=%d edges=%zu states=%u canonical=%u events=%u\n",
                mode, steps, threads, init.size(), hg.num_states(), hg.num_canonical_states(),
                hg.num_events());
    return 0;
}
