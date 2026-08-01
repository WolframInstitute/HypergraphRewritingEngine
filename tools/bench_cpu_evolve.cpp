// The CPU twin of bench_gpu_evolve: identical workload (WPP rule, two-edge init, Full state
// canonicalization, quotient exploration), median-of-N wall time across a thread sweep.
//
// Exists because #72 (the GPU occupancy ceiling) needs a real CPU-vs-GPU baseline and none
// existed: bench_cpu_vs_gpu predates the current engine and crashes, and the two sides were
// never measured on the same workload with the same discipline. Wall clock drifts >10% on this
// box, so medians of many iterations, and the comparison is against bench_gpu_evolve's medians
// from the same session, not against stored numbers.
//
// Usage: bench_cpu_evolve [steps] [iters]   (default 6 20)

#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace hypergraph;

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 6;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;

    for (int threads : {1, 4, 8}) {
        std::vector<double> ms;
        size_t states = 0, raw = 0;
        for (int i = 0; i < iters; ++i) {
            Hypergraph g;
            g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine e(&g, threads);
            e.set_explore_from_canonical_states_only(true);
            e.add_rule(make_rule(0).lhs({0,1}).lhs({0,2})
                           .rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build());
            const auto t0 = std::chrono::steady_clock::now();
            e.evolve({{0,1},{0,2}}, steps);
            const auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            states = g.num_canonical_states();
            raw = g.num_states();
        }
        std::sort(ms.begin(), ms.end());
        // raw is the like-for-like twin of the GPU bench's result.states.size(); canonical is
        // what HGEvolve reports as NumStates. Print both so neither gets compared to the other.
        std::printf("threads=%d steps=%d canonical=%zu raw=%zu median_ms=%.3f min_ms=%.3f\n",
                    threads, steps, states, raw, ms[ms.size() / 2], ms.front());
    }
    return 0;
}
