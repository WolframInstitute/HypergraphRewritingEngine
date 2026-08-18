// GPU evolve timing: the deep/narrow regime (small initial state, many steps), where the work
// per step is small enough that scheduling overhead, not compute, sets the time.
//
// Usage: bench_gpu_evolve [steps] [iters] [mode] [workload]   (default 6 20 0 wpp)
//
// ONE WORKLOAD IS NOT A MEASUREMENT. Every shape below stresses the device differently: a rule
// whose left side is a single edge floods the matcher, an automorphic initial state makes
// individualization-refinement descend where it usually stops at depth one, a multi-rule set
// multiplies the queue traffic per state, and a reductive rule shrinks states so the frontier
// dies rather than grows. An optimisation measured on one of them is tuned to one of them.
// `bench_gpu_evolve <steps> <iters> <mode> list` prints the names.
//   mode 0  evolve() against PersistentEvolver
//   mode 1  PersistentEvolver alone (clean steady state for profiling)
//   mode 2  one row for the grid this process is running at
//
// MODE 2 ASKS WHICH RESOURCE THE PERSISTENT LOOP IS BOUND BY. Its blocks do not retire, so the
// grid IS the worker count, and grid size and queue contention predict opposite curves: if the
// bound is occupancy, time falls as blocks rise until the device saturates; if it is contention
// on the shared work cursors, time flattens early or climbs, because every added worker is more
// traffic on the same lines. The default (persistent.cu default_persistent_grid) is eight blocks
// per SM, chosen off this measurement; re-run it after any change to the queue or the work item.
//
// The grid is process-global and resolved ONCE -- worker count and IR arena slots both derive
// from it and must agree -- so ONE PROCESS MEASURES ONE GRID. Sweep from the shell:
//   for b in 128 256 512 1024 2048 3072; do HG_GPU_PERSISTENT_BLOCKS=$b ./bench_gpu_evolve 7 5 2; done
// Each row prints the grid it ran at, read from the same variable the engine reads. A loop that
// setenv'd inside one process would report the value it just set while every iteration ran at
// the first grid, and that flat curve reads exactly like a real result.
#include "hg_gpu/evolve.hpp"
#include "corpus_gen.hpp"
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct Workload {
    const char* name;
    std::vector<hg_gpu::RewriteRule> rules;
    std::vector<std::vector<uint32_t>> init;
};

// Shapes drawn from the corpus tools/cost_matrix.cpp proves exactness on, so a timing here and
// an exactness row there name the same thing.
static std::vector<Workload> workloads() {
    return {
        // The deep/narrow default: two-edge left side, growing right side, two-edge initial.
        {"wpp",        {make_rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})}, {{0,1},{0,2}}},
        // Single-edge left side: every edge in the state is a candidate, so the matcher floods.
        {"binary",     {make_rule({{0,1}}, {{0,2},{2,1}})},                    {{0,1}}},
        // Wolfram 2->4, the shape most of the published models use.
        {"wolfram24",  {make_rule({{0,1},{1,2}}, {{0,1},{1,3},{3,2},{2,0}})},  {{0,1},{1,2}}},
        // Cyclic left side: three edges, no acyclic join order, worst case for the matcher.
        {"triangle",   {make_rule({{0,1},{1,2},{2,0}}, {{0,1},{1,2},{2,3},{3,0}})},
                       {{0,1},{1,2},{2,0}}},
        // Mixed arity on both sides.
        {"arity3",     {make_rule({{0,1,2}}, {{0,1,2},{2,3}})},                {{0,1,2}}},
        // Two rules over the same state: queue traffic per state doubles and the two compete.
        {"multirule",  {make_rule({{0,1},{1,2}}, {{0,1},{1,3},{3,2}}),
                        make_rule({{0,1}}, {{0,2},{2,1}})},                    {{0,1},{1,2}}},
        // Automorphic initial state: the canonicalizer cannot stop at depth one, which is where
        // the device spends 91% of its time even on the easy shapes.
        {"cycle4",     {make_rule({{0,1},{1,2}}, {{0,1},{1,3},{3,2}})},
                       {{0,1},{1,2},{2,3},{3,0}}},
        // Several roots, so the frontier starts wide instead of narrow.
        {"multiroot",  {make_rule({{0,1},{1,2}}, {{0,1},{1,3},{3,2}})},
                       {{0,1},{1,2},{3,4},{4,5},{6,7},{7,8}}},
    };
}

int main(int argc, char** argv) {
    int steps = argc > 1 ? std::atoi(argv[1]) : 6;
    int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    int mode  = argc > 3 ? std::atoi(argv[3]) : 0;
    const char* want = argc > 4 ? argv[4] : "wpp";

    const auto all = workloads();
    if (std::strcmp(want, "list") == 0) {
        for (const auto& w : all) std::printf("%s\n", w.name);
        return 0;
    }
    // The generated corpus, shared with bench_cpu_evolve through corpus_gen.hpp.
    static std::vector<std::string> gen_names;
    std::vector<Workload> generated;
    for (const auto& g : corpus::corpus()) {
        gen_names.push_back(g.name);
        Workload w;
        w.name = nullptr;   // filled below, once gen_names has stopped reallocating
        for (const auto& r : g.rules) {
            std::vector<std::vector<uint8_t>> lhs, rhs;
            for (const auto& e : r.lhs) lhs.push_back(std::vector<uint8_t>(e.begin(), e.end()));
            for (const auto& e : r.rhs) rhs.push_back(std::vector<uint8_t>(e.begin(), e.end()));
            w.rules.push_back(make_rule(std::move(lhs), std::move(rhs)));
        }
        for (const auto& e : g.init) w.init.push_back(std::vector<uint32_t>(e.begin(), e.end()));
        generated.push_back(std::move(w));
    }
    for (size_t i = 0; i < generated.size(); ++i) generated[i].name = gen_names[i].c_str();

    if (std::strcmp(want, "corpus") == 0) {
        for (const auto& w : generated) std::printf("%s\n", w.name);
        return 0;
    }

    const Workload* sel = nullptr;
    for (const auto& w : all) if (std::strcmp(w.name, want) == 0) sel = &w;
    for (const auto& w : generated) if (std::strcmp(w.name, want) == 0) sel = &w;
    if (!sel) { std::fprintf(stderr, "unknown workload '%s' (try: list, corpus)\n", want); return 2; }

    hg_gpu::EvolveInput in;
    in.rules = sel->rules;
    in.initial_state = sel->init;
    in.num_steps = static_cast<uint32_t>(steps);
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
    // Quotient exploration is what couples the device to the reconstruction whose cost grows
    // exponentially while its output grows linearly, so the two must be separable at the bench to
    // attribute any device cost to one or the other. HG_BENCH_QUOTIENT=0 explores raw.
    const char* q = std::getenv("HG_BENCH_QUOTIENT");
    in.explore_from_canonical_states_only = !(q && q[0] == '0');

    // The raw unfolding is recovered by a replay whose cost is the RAW answer's, not the
    // canonical one's. HG_BENCH_RAW=0 records none of it, which is the configuration a caller
    // wanting states and their transitions actually runs.
    const char* rawenv = std::getenv("HG_BENCH_RAW");
    if (rawenv && rawenv[0] == '0')
        in.record.causal = in.record.branchial = in.record.raw_events = false;

    auto r0 = hg_gpu::evolve(in);   // warmup (CUDA context, allocations)

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // Mode 2: is the persistent loop's cost set by its GRID SIZE or by CONTENTION on the work
    // queue? The two predict different curves -- grid size falls until occupancy saturates,
    // contention flattens early or rises as more workers hit the same cursors.
    if (mode == 2) {
        hg_gpu::PersistentEvolver ev;
        auto warm = ev.run(in);
        std::vector<double> t;
        for (int i = 0; i < iters; ++i) {
            auto a = std::chrono::steady_clock::now();
            auto r = ev.run(in);
            auto b = std::chrono::steady_clock::now();
            (void)r;
            t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        }
        const char* grid = std::getenv("HG_GPU_PERSISTENT_BLOCKS");
        std::printf("%8s   %9.3f   %6zu   %6zu   (steps=%d)\n",
                    grid ? grid : "8/SM", median(t),
                    warm.states.size(), warm.events.size(), steps);
        return 0;
    }

    // (A) free evolve(): builds and destroys an Engine every call.
    std::vector<double> ta;
    if (mode == 0) {
        for (int i = 0; i < iters; ++i) {
            auto a = std::chrono::steady_clock::now();
            auto r = hg_gpu::evolve(in);
            auto b = std::chrono::steady_clock::now();
            (void)r;
            ta.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        }
    }

    // (B) PersistentEvolver: allocations amortized across calls, with the same
    // grow-and-retry robustness as evolve().
    hg_gpu::PersistentEvolver evolver;
    auto rw = evolver.run(in);   // sizes the engine

    // RAMP THE CLOCK BEFORE TIMING ANYTHING.
    //
    // The device idles at 210 MHz against a 3,150 MHz maximum and only ramps under sustained
    // load, so a measurement that begins on a cold clock is taken at a fraction of the speed the
    // same code reaches warm. Measured: a cold first run read 79.2 ms where warm runs of the
    // identical command read 8.1, and six consecutive warm runs then held a +-4% spread with two
    // other tenants busy on the CPU. The clock, not host contention, was the whole of that
    // variance.
    //
    // One warm-up CALL does not fix it: a workload that completes in 3.4 ms leaves the device
    // idle again before the timed loop starts. So the warm-up is bounded by TIME rather than by
    // iterations, and every workload gets the same ramp regardless of how fast it is.
    //
    // Locking the clock with `nvidia-smi --lock-gpu-clocks` would be better and needs root, which
    // is not available here; this reaches the same steady state by keeping the device busy.
    {
        const auto ramp_deadline = std::chrono::steady_clock::now() +
                                   std::chrono::milliseconds(400);
        while (std::chrono::steady_clock::now() < ramp_deadline) (void)evolver.run(in);
    }
    std::vector<double> tb;
    for (int i = 0; i < iters; ++i) {
        auto a = std::chrono::steady_clock::now();
        auto r = evolver.run(in);
        auto b = std::chrono::steady_clock::now();
        (void)r;
        tb.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }

    // A TRUNCATED RUN IS NOT A MEASUREMENT. A capacity overflow is a warning here by design:
    // the kernels keep running on the partial budget and the result carries whatever was
    // computed, so a bench that reads only states.size() reports a time for a workload the
    // device never finished. The CPU bench hid exactly this on disc-l3a2g2r2 depth 5, where
    // the edge ceiling pinned raw states at 838,861 and made the counts race-dependent.
    // Printed for both timed results, since each has its own budget.
    auto report_warnings = [](const char* which, const auto& res) {
        for (const auto& w : res.warnings)
            std::printf("  WARNING(%s): %s x%u%s%s\n", which,
                        error_kind_name(w.kind), w.count,
                        w.context.empty() ? "" : " -- ", w.context.c_str());
    };
    if (mode == 0) report_warnings("evolve", r0);
    report_warnings("persistent", rw);

    if (mode == 0) {
        std::printf("steps=%d states=%zu events=%zu | evolve()_median_ms=%.3f | "
                    "PersistentEvolver_median_ms=%.3f (states=%zu) | speedup=%.2fx\n",
                    steps, r0.states.size(), r0.events.size(),
                    median(ta), median(tb), rw.states.size(), median(ta) / median(tb));
    } else {
        std::printf("steps=%d states=%zu events=%zu | PersistentEvolver_median_ms=%.3f\n",
                    steps, rw.states.size(), rw.events.size(), median(tb));
    }
    return 0;
}
