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

#include "hgcommon/phase_timing.hpp"
#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace hypergraph;

// The thread counts to sweep. THE POINT OF A SCALING RUN IS WHERE IT STOPS SCALING, so the
// sweep has to reach the machine's width: stopping at 8 on a 32-thread host measures the easy
// half and reports the ratio there as if it were the answer. Any count above the host's
// hardware_concurrency is dropped rather than run, because oversubscription measures the
// scheduler.
static std::vector<int> thread_sweep(const char* spec) {
    std::vector<int> out;
    if (spec && *spec) {
        const std::string s(spec);
        size_t pos = 0;
        while (pos < s.size()) {
            const size_t comma = s.find(',', pos);
            const int t = std::atoi(s.substr(pos, comma - pos).c_str());
            if (t > 0) out.push_back(t);
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    } else {
        const int hw = static_cast<int>(std::thread::hardware_concurrency());
        for (int t : {1, 2, 4, 8, 16, 24, 32, 48, 64})
            if (t <= (hw > 0 ? hw : 8)) out.push_back(t);
        if (out.empty()) out.push_back(1);
    }
    return out;
}

// The same shapes bench_gpu_evolve measures, so a CPU row and a GPU row name the same workload.
// One workload is not a measurement: multi-rule and automorphic-initial shapes cost orders of
// magnitude more per state than the deep/narrow default, and only a corpus shows it.
struct Workload {
    const char* name;
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> init;
};

static std::vector<Workload> workloads() {
    return {
        {"wpp",       {make_rule(0).lhs({0,1}).lhs({0,2})
                          .rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
                      {{0,1},{0,2}}},
        {"binary",    {make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
                      {{0,1}}},
        {"wolfram24", {make_rule(0).lhs({0,1}).lhs({1,2})
                          .rhs({0,1}).rhs({1,3}).rhs({3,2}).rhs({2,0}).build()},
                      {{0,1},{1,2}}},
        {"triangle",  {make_rule(0).lhs({0,1}).lhs({1,2}).lhs({2,0})
                          .rhs({0,1}).rhs({1,2}).rhs({2,3}).rhs({3,0}).build()},
                      {{0,1},{1,2},{2,0}}},
        {"arity3",    {make_rule(0).lhs({0,1,2}).rhs({0,1,2}).rhs({2,3}).build()},
                      {{0,1,2}}},
        {"multirule", {make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).build(),
                       make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
                      {{0,1},{1,2}}},
        {"cycle4",    {make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).build()},
                      {{0,1},{1,2},{2,3},{3,0}}},
        {"multiroot", {make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).build()},
                      {{0,1},{1,2},{3,4},{4,5},{6,7},{7,8}}},
    };
}

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 6;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    const std::vector<int> sweep = thread_sweep(argc > 3 ? argv[3] : nullptr);
    const char* want = argc > 4 ? argv[4] : "wpp";
    const auto all = workloads();
    if (std::strcmp(want, "list") == 0) {
        for (const auto& w : all) std::printf("%s\n", w.name);
        return 0;
    }
    const Workload* sel = nullptr;
    for (const auto& w : all) if (std::strcmp(w.name, want) == 0) sel = &w;
    if (!sel) { std::fprintf(stderr, "unknown workload '%s' (try: list)\n", want); return 2; }

    double base_ms = 0.0;
    for (int threads : sweep) {
        std::vector<double> ms;
        size_t states = 0, raw = 0;
        for (int i = 0; i < iters; ++i) {
            Hypergraph g;
            g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            ParallelEvolutionEngine e(&g, threads);
            e.set_explore_from_canonical_states_only(true);
            for (const auto& r : sel->rules) e.add_rule(r);
            const auto t0 = std::chrono::steady_clock::now();
            e.evolve(sel->init, steps);
            const auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            states = g.num_canonical_states();
            raw = g.num_states();
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        if (base_ms == 0.0) base_ms = med;
        // raw is the like-for-like twin of the GPU bench's result.states.size(); canonical is
        // what HGEvolve reports as NumStates. Print both so neither gets compared to the other.
        //
        // Speedup and EFFICIENCY together: a speedup alone reads as success at any thread count,
        // while speedup/threads says how much of each added core the run actually used, and it
        // is the column that shows where scaling stops.
        std::printf("threads=%d steps=%d canonical=%zu raw=%zu median_ms=%.3f min_ms=%.3f "
                    "speedup=%.2f efficiency=%.2f\n",
                    threads, steps, states, raw, med, ms.front(),
                    base_ms / med, (base_ms / med) / threads);
    }
        if (hgcommon::phase_timing_compiled()) {
        uint64_t total = 0;
        for (uint32_t p = 0; p < static_cast<uint32_t>(hgcommon::Phase::Count); ++p)
            total += hgcommon::phase_cycles(static_cast<hgcommon::Phase>(p));
        if (total) {
            std::printf("phase cycles (summed over workers, fractions of their sum):\n");
            for (uint32_t p = 0; p < static_cast<uint32_t>(hgcommon::Phase::Count); ++p) {
                const auto ph = static_cast<hgcommon::Phase>(p);
                std::printf("  %-9s %6.2f%%  (%llu)\n", hgcommon::phase_name(ph),
                            100.0 * double(hgcommon::phase_cycles(ph)) / double(total),
                            (unsigned long long)hgcommon::phase_cycles(ph));
            }
        }
    }
    return 0;
}
