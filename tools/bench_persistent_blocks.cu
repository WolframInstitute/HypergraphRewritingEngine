// Is the persistent scheduler's loss on WIDE evolutions inherent, or just its grid size?
//
// bench_scheduler_shape measured a two-sided result: the persistent scheduler wins increasingly
// with depth (2.19x at 32 steps) and LOSES about 29% on wide-shallow shapes (0.71x). A loss at
// the wide end would mean the step loop cannot be deleted, so it is worth knowing which of two
// causes it is:
//
//   TUNING       run_persistent_evolve defaults to `blocks ? blocks : 33`, and evolve.cu never
//                passes a value. At kMatchBlockThreads = 32 that is ~1056 threads. This device
//                has 128 SMs and holds far more. Meanwhile the level-synchronous path launches
//                grid-stride kernels sized to the frontier, so on a wide step it uses the whole
//                device while the persistent one uses a slice of it.
//   INHERENT     the queue itself is the bottleneck -- every worker CASes the same head and tail,
//                and past some width that contention costs more than the barrier saved.
//
// The two predict different curves. If it is tuning, time falls as blocks rise until occupancy
// saturates. If it is contention, time flattens early or rises, because more workers means more
// CAS traffic on the same two cursors.
//
// Timed with CUDA events, min-of-N, and the level-synchronous arm measured in the same process
// for reference. Every run's state count is compared against the reference: a configuration that
// is fast because it dropped work is not fast.

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/rewrite.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace hg_gpu;

namespace {

RewriteRule chain_rule() {
    RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;
    return r;
}

std::vector<std::vector<VertexId>> path_init(size_t n) {
    std::vector<std::vector<VertexId>> e;
    for (size_t i = 0; i < n; ++i)
        e.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
    return e;
}

struct Timed { float ms; size_t states; bool clean; };

// One persistent run at a chosen block count, through the same primitives evolve.cu drives.
Timed run_persistent(const EvolveInput& in, const RewriteRule& r, uint32_t blocks) {
    EngineConfig cfg = config_from_input(in);

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);

    EngineState engine(cfg);
    upload_initial_state(engine, in.initial_state);
    std::vector<DeviceRule> rules = {make_device_rule(r)};
    Pool<MatchRecord> matches(cfg.max_states * 8u);
    matches.reset();
    DeviceArena arena(static_cast<uint64_t>(cfg.max_states) * 64ull);

    run_persistent_evolve(engine, rules, /*roots=*/{0u}, in.num_steps, matches, arena,
                          /*dedup=*/in.explore_from_canonical_states_only,
                          0xFFFFFFFFu, 0, in.canonicalization,
                          hgcommon::EVENT_SIG_NONE, blocks);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    Timed t;
    cudaEventElapsedTime(&t.ms, beg, end);
    t.states = engine.num_states_host();
    std::vector<OverflowWarning> w;
    engine.collect_warnings_into(w, "block sweep");
    t.clean = w.empty();
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return t;
}

Timed run_lockstep(const EvolveInput& in) {
    EvolveInput a = in;
    a.persistent_scheduler = false;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    EvolveResult res = evolve(a);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    Timed t;
    cudaEventElapsedTime(&t.ms, beg, end);
    t.states = res.states.size();
    t.clean = res.warnings.empty();
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return t;
}

float min_of(std::vector<float> v) {
    return *std::min_element(v.begin(), v.end());
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;

    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    std::printf("# device has %d SMs; kMatchBlockThreads = 32; current default grid is 33\n", sms);
    std::printf("# CUDA events, min-of-%d\n\n", reps);

    struct Case { const char* name; size_t width; uint32_t steps; };
    const std::vector<Case> cases = {
        {"wide-3",   160, 3},   // the shape the persistent scheduler loses on
        {"narrow-32",  3, 32},  // the shape it wins on -- check the fix does not cost here
    };
    const std::vector<uint32_t> block_counts = {33, 64, 128, 256, 512, 1024};

    const RewriteRule r = chain_rule();

    for (const Case& c : cases) {
        EvolveInput in;
        in.rules = {r};
        in.initial_state = path_init(c.width);
        in.num_steps = c.steps;
        in.canonicalization = CanonicalizationMode::Full;
        in.explore_from_canonical_states_only = true;

        std::vector<float> ls;
        Timed ref{};
        for (int i = 0; i < reps; ++i) { ref = run_lockstep(in); ls.push_back(ref.ms); }
        const float ls_ms = min_of(ls);
        std::printf("%s (width %zu, %u steps): level-synchronous %.2f ms, %zu states%s\n",
                    c.name, c.width, c.steps, ls_ms, ref.states,
                    ref.clean ? "" : "  OVERFLOWED");
        std::printf("  %8s %10s %8s %s\n", "blocks", "ms", "ratio", "states");

        for (uint32_t b : block_counts) {
            std::vector<float> v;
            Timed t{};
            for (int i = 0; i < reps; ++i) { t = run_persistent(in, r, b); v.push_back(t.ms); }
            const float ms = min_of(v);
            std::printf("  %8u %10.2f %7.2fx %zu%s%s\n", b, ms, ls_ms / ms, t.states,
                        t.states != ref.states ? "  STATE MISMATCH" : "",
                        t.clean ? "" : "  OVERFLOWED");
            std::fflush(stdout);
        }
        std::printf("\n");
    }
    return 0;
}
