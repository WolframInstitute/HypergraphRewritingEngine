// Does removing the per-step barrier pay, and on what SHAPE of evolution?
//
// A single benchmark number cannot answer this, because the barrier's cost is not a constant --
// it is one full device synchronisation per step, charged against however much work that step
// had. So the answer has to be a curve over shape, and the two ends of it predict opposite
// results:
//
//   DEEP-NARROW   many steps, few states per step. The sync is charged against almost no work,
//                 so this is where the persistent scheduler should win most.
//   SHALLOW-WIDE  few steps, many states per step. Each phase already fills the device, the
//                 barrier costs little, and the persistent scheduler's queue traffic, CAS
//                 contention and idle spinning could make it LOSE.
//
// A win at one end and a loss at the other is a real answer: it says the step loop cannot simply
// be deleted and the default has to be shape-dependent. A win at both ends says it can.
//
// INSTRUMENT. Wall clock on this box drifts more than 10% run to run, which is larger than the
// effect being measured, so:
//   - timing is CUDA events around the run, not host wall clock;
//   - the two schedulers are INTERLEAVED within each repetition rather than run in blocks, so a
//     thermal or clock drift over the run hits both arms equally;
//   - min-of-N is reported, because the minimum is the least contaminated sample -- everything
//     that perturbs a run adds time;
//   - the spread (min, median, max) is printed too. A ratio whose arms overlap in spread is not
//     a measurement, and printing only the mean would hide that.
//
// Every run is checked for warnings and for equal state counts across schedulers. A run that
// silently overflowed is faster for a reason that is not scheduling, and comparing it would be
// measuring the overflow.

#include "hg_gpu/evolve.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace hg_gpu;

namespace {

struct Shape {
    const char* name;
    std::vector<std::vector<VertexId>> init;
    uint32_t steps;
    // Quotient exploration expands each canonical class once, which BOUNDS the width. That is
    // what makes depth an independent variable: with full capture a growth rule's width explodes
    // with depth, so a "deeper" workload is also a wider one and the shape axis is not isolated.
    bool quotient;
    const char* why;
};

RewriteRule chain_rule() {
    RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;
    return r;
}

struct Sample {
    float ms = 0.0f;
    size_t states = 0;
    size_t events = 0;
    bool clean = true;
};

Sample time_once(const EvolveInput& in) {
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // The engine is constructed inside the timed region deliberately: it is part of what a
    // caller pays for a run, and the two schedulers allocate differently.
    cudaEventRecord(beg);
    EvolveResult res = evolve(in);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    Sample s;
    cudaEventElapsedTime(&s.ms, beg, end);
    s.states = res.states.size();
    s.events = res.events.size();
    s.clean  = res.warnings.empty();
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return s;
}

float min_of(const std::vector<float>& v) {
    return *std::min_element(v.begin(), v.end());
}

struct Stats { float lo, med, hi; };

Stats summarize(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return Stats{v.front(), v[v.size() / 2], v.back()};
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    // "warm" skips the cold table, which is long enough to crowd out the warm pass.
    const bool warm_only = argc > 2 && std::string(argv[2]) == "warm";

    // Shapes are built by trading depth against initial width at roughly comparable total work,
    // so the comparison is across SHAPE and not merely across size.
    auto wide_init = [](size_t n) {
        std::vector<std::vector<VertexId>> e;
        for (size_t i = 0; i < n; ++i)
            e.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
        return e;
    };

    // A zero-step run does the allocation and none of the evolution, so it measures the fixed
    // cost each scheduler charges before any work happens. Without it a constant difference in
    // setup is indistinguishable from a per-step scheduling difference, and at these sizes the
    // setup is the larger of the two -- which is exactly how the first version of this benchmark
    // reported the persistent scheduler as uniformly slower when the arms differed by an
    // allocation it made and the other did not.
    std::vector<Shape> shapes = {
        // EVERY shape is quotient-explored. Under full capture a growth rule's width explodes
        // with depth, so a deeper workload is also a wider one and the shape axis is not
        // isolated -- and past a few steps it outgrows the default pools, so both arms pay a
        // grow-and-retry that re-runs the whole evolution and swamps the difference being
        // measured. Quotient bounds the width, which is what makes depth and width independent
        // variables here.
        {"setup-only",    wide_init(8),   0, true, "allocation with no evolution -- the fixed cost of each arm"},
        // The wide end FIRST: it is the shape the persistent scheduler is most likely to LOSE
        // on, and a sweep that runs out of time before reaching its own counter-case has only
        // measured where it wins.
        {"wide-3",        wide_init(160), 3, true, "wide and shallow -- each step already fills the device"},
        {"wide-6",        wide_init(160), 6, true, "same width, more steps"},
        {"narrow-8",      wide_init(3),   8, true, "the depth sweep: width bounded, steps the only variable"},
        {"narrow-16",     wide_init(3),  16, true, "2x the steps"},
        {"narrow-24",     wide_init(3),  24, true, "3x the steps -- barrier cost scales with this"},
        {"narrow-32",     wide_init(3),  32, true, "4x the steps"},
        {"deep-narrow",   wide_init(3),   9, false, "many steps, little work per step, full capture"},
        {"medium",        wide_init(10),  6, false, "between the two ends"},
        {"shallow-wide",  wide_init(90),  3, false, "few steps, each already filling the device"},
    };

    const RewriteRule r = chain_rule();

    std::printf("# persistent vs level-synchronous, by evolution shape\n");
    std::printf("# CUDA events, interleaved arms, %d reps, min-of-N (spread shown)\n", reps);
    std::printf("# ratio > 1 means the persistent scheduler is FASTER\n\n");
    std::printf("%-14s %6s %8s %8s | %-22s | %-22s | %6s\n",
                "shape", "steps", "states", "events",
                "level-sync ms lo/med/hi", "persistent ms lo/med/hi", "ratio");

    for (const Shape& sh : shapes) {
        if (warm_only) break;
        EvolveInput base;
        base.rules = {r};
        base.initial_state = sh.init;
        base.num_steps = sh.steps;
        base.canonicalization = CanonicalizationMode::Full;
        base.explore_from_canonical_states_only = sh.quotient;

        std::vector<float> ls, ps;
        Sample last_ls, last_ps;
        bool mismatch = false;
        // Per ARM, not one flag for both. An overflow means the run paid a grow-and-retry, which
        // re-runs the whole evolution and swamps the difference being measured -- so the number
        // is not a scheduling measurement at all, and which arm it happened in is the first thing
        // a reader needs. A single combined flag says a row is contaminated without saying how.
        bool dirty_ls = false, dirty_ps = false;

        for (int i = 0; i < reps; ++i) {
            // Interleaved, and alternating which arm goes first so neither systematically pays
            // for the other's cache and clock state.
            EvolveInput a = base; a.persistent_scheduler = false;
            EvolveInput b = base; b.persistent_scheduler = true;
            Sample sa, sb;
            if (i % 2 == 0) { sa = time_once(a); sb = time_once(b); }
            else            { sb = time_once(b); sa = time_once(a); }

            ls.push_back(sa.ms);
            ps.push_back(sb.ms);
            last_ls = sa; last_ps = sb;
            if (sa.states != sb.states) mismatch = true;
            if (!sa.clean) dirty_ls = true;
            if (!sb.clean) dirty_ps = true;
        }

        const Stats a = summarize(ls), b = summarize(ps);
        std::printf("%-14s %6u %8zu %8zu | %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f | %6.2fx%s%s\n",
                    sh.name, sh.steps, last_ls.states, last_ls.events,
                    a.lo, a.med, a.hi, b.lo, b.med, b.hi,
                    b.lo > 0.0f ? a.lo / b.lo : 0.0f,
                    mismatch ? "  STATE-COUNT MISMATCH" : "",
                    dirty_ls && dirty_ps ? "  BOTH ARMS OVERFLOWED (ratio meaningless)"
                    : dirty_ls           ? "  LEVEL-SYNC OVERFLOWED (ratio meaningless)"
                    : dirty_ps           ? "  PERSISTENT OVERFLOWED (ratio meaningless)"
                                         : "");
        std::fflush(stdout);
    }

    std::printf("\n# why each shape is here:\n");
    for (const Shape& sh : shapes) std::printf("#   %-14s %s\n", sh.name, sh.why);

    // WARM ENGINE. Every row above builds a fresh Engine per timed run, so it measures
    // cold-start and charges each arm its full allocation every time. That is not how the
    // engine is used interactively: PersistentEvolver reuses one Engine across calls, and the
    // paclet drives it that way, so the allocation is paid once and amortised over a session.
    //
    // The two regimes answer different questions and the cold one cannot see a change that
    // moves cost out of the per-call path. Timed from the SECOND call onward, since the first
    // is the one that allocates.
    std::printf("\n# warm engine: one Engine, repeated run() calls, first call excluded\n");
    std::printf("%-14s %6s | %-14s | %-14s | %6s\n",
                "shape", "steps", "level-sync ms", "persistent ms", "ratio");

    for (const Shape& sh : shapes) {
        // One representative from each end. The full-capture shapes are skipped because they
        // outgrow the pools and pay a grow-and-retry, which is not a scheduling measurement.
        const bool representative = std::string(sh.name) == "wide-3" ||
                                    std::string(sh.name) == "narrow-16";
        if (!representative) continue;
        EvolveInput base;
        base.rules = {r};
        base.initial_state = sh.init;
        base.num_steps = sh.steps;
        base.canonicalization = CanonicalizationMode::Full;
        base.explore_from_canonical_states_only = sh.quotient;

        auto warm = [&](bool persistent) {
            EvolveInput in = base;
            in.persistent_scheduler = persistent;
            EngineConfig cfg = config_from_input(in);
            Engine engine(cfg);
            engine.run(in);                       // first call: allocates, not timed
            std::vector<float> v;
            for (int i = 0; i < reps; ++i) {
                cudaEvent_t b, e;
                cudaEventCreate(&b); cudaEventCreate(&e);
                cudaEventRecord(b);
                engine.run(in);
                cudaEventRecord(e);
                cudaEventSynchronize(e);
                float ms = 0.0f; cudaEventElapsedTime(&ms, b, e);
                cudaEventDestroy(b); cudaEventDestroy(e);
                v.push_back(ms);
            }
            return min_of(v);
        };

        const float ls = warm(false), ps = warm(true);
        std::printf("%-14s %6u | %14.2f | %14.2f | %6.2fx\n",
                    sh.name, sh.steps, ls, ps, ps > 0.0f ? ls / ps : 0.0f);
        std::fflush(stdout);
    }
    return 0;
}
