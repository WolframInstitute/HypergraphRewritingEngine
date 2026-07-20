// Focused GPU evolve timing harness: the deep/narrow regime (small initial state,
// many steps) where the per-step lockstep barrier floor dominates and CUDA-graph
// capture of the step chain should help most. Reports median/min wall time over N
// iterations plus the state/event counts (for a correctness cross-check).
//
// Usage: bench_gpu_evolve [steps] [iters]   (default 6 20)
#include "hg_gpu/evolve.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static hg_gpu::RewriteRule make_rule(std::vector<std::vector<uint8_t>> lhs,
                                     std::vector<std::vector<uint8_t>> rhs) {
    hg_gpu::RewriteRule r;
    r.lhs = std::move(lhs);
    r.rhs = std::move(rhs);
    uint8_t lm = 0; for (auto& e : r.lhs) for (auto v : e) lm = std::max<uint8_t>(lm, v);
    uint8_t rm = 0; for (auto& e : r.rhs) for (auto v : e) rm = std::max<uint8_t>(rm, v);
    r.num_lhs_vars = r.lhs.empty() ? 0 : static_cast<uint8_t>(lm + 1);
    r.num_rhs_vars = r.rhs.empty() ? 0 : static_cast<uint8_t>(rm + 1);
    return r;
}

int main(int argc, char** argv) {
    int steps = argc > 1 ? std::atoi(argv[1]) : 6;
    int iters = argc > 2 ? std::atoi(argv[2]) : 20;

    hg_gpu::EvolveInput in;
    in.rules = { make_rule({{0, 1}, {0, 2}}, {{0, 1}, {0, 3}, {1, 3}, {2, 3}}) };
    in.initial_state = {{0u, 1u}, {0u, 2u}};
    in.num_steps = static_cast<uint32_t>(steps);
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
    in.explore_from_canonical_states_only = true;

    auto r0 = hg_gpu::evolve(in);   // warmup (CUDA context, allocations)

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // (A) free evolve(): builds and destroys an Engine every call.
    std::vector<double> ta;
    for (int i = 0; i < iters; ++i) {
        auto a = std::chrono::steady_clock::now();
        auto r = hg_gpu::evolve(in);
        auto b = std::chrono::steady_clock::now();
        (void)r;
        ta.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }

    // (B) PersistentEvolver: allocations amortized across calls, with the same
    // grow-and-retry robustness as evolve().
    hg_gpu::PersistentEvolver evolver;
    auto rw = evolver.run(in);   // warmup (first run sizes the engine)
    std::vector<double> tb;
    for (int i = 0; i < iters; ++i) {
        auto a = std::chrono::steady_clock::now();
        auto r = evolver.run(in);
        auto b = std::chrono::steady_clock::now();
        (void)r;
        tb.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }

    std::printf("steps=%d states=%zu events=%zu | evolve()_median_ms=%.3f | "
                "Engine.run()_median_ms=%.3f (states=%zu) | speedup=%.2fx\n",
                steps, r0.states.size(), r0.events.size(),
                median(ta), median(tb), rw.states.size(), median(ta) / median(tb));
    return 0;
}
