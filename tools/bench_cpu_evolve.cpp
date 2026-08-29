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

#include "corpus_gen.hpp"
#include "hgcommon/phase_timing.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/bitset.hpp"
#include "hypergraph/pattern_matcher.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <hypergraph/ir_canonicalization.hpp>

using namespace hypergraph;

// HG_CAPACITY_SCALE multiplies every append-only array's segment size. A workload that exceeds
// the default ceiling returns a TRUNCATED evolution with a warning, and its counts are then
// decided by the arrival race rather than by the rules -- so a benchmark row taken from one is
// measuring the ceiling. Raising the scale is how a run gets the whole evolution.
static uint32_t capacity_scale_from_env() {
    const char* s = std::getenv("HG_CAPACITY_SCALE");
    const int v = s ? std::atoi(s) : 1;
    return v > 0 ? static_cast<uint32_t>(v) : 1u;
}

// The thread counts to sweep. THE POINT OF A SCALING RUN IS WHERE IT STOPS SCALING, so the
// sweep has to reach the machine's width: stopping at 8 on a 32-thread host measures the easy
// half and reports the ratio there as if it were the answer. Any count above the host's
// hardware_concurrency is dropped rather than run, because oversubscription measures the
// scheduler.
// One comma-list parser, used by the thread sweep and by the CPU set. Both take the same
// spelling ("1,2,4") and neither wants a second copy of the loop.
static std::vector<int> parse_int_list(const char* spec, bool positive_only) {
    std::vector<int> out;
    if (!spec || !*spec) return out;
    const std::string s(spec);
    size_t pos = 0;
    while (pos < s.size()) {
        const size_t comma = s.find(',', pos);
        const int t = std::atoi(s.substr(pos, comma - pos).c_str());
        if (t > 0 || (!positive_only && t == 0)) out.push_back(t);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

static std::vector<int> thread_sweep(const char* spec) {
    std::vector<int> out;
    if (spec && *spec) {
        out = parse_int_list(spec, /*positive_only=*/true);
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
        // TWO COMPONENTS OF TWO EDGES EACH, which the generated corpus does not build: its
        // Disconnected shape numbers every edge's variables apart, so disc-lNa2 is N components
        // of ONE edge and each component's match set is "every edge of this arity". A component
        // of one edge costs one scan to enumerate, so the product is the output and the join is
        // already output-optimal on it. A component of TWO edges has a join of its own, and the
        // schedule re-runs that join once per partial match of the components before it. This is
        // the shape the disconnected-LHS warning is about, and nothing measured it.
        {"disc2x2",   {make_rule(0).lhs({0,1}).lhs({1,2}).lhs({3,4}).lhs({4,5})
                          .rhs({0,1}).rhs({1,2}).rhs({3,4}).rhs({4,5}).rhs({2,6}).build()},
                      {{0,1},{1,2},{3,4},{4,5}}},
    };
}

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 6;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    const std::vector<int> sweep = thread_sweep(argc > 3 ? argv[3] : nullptr);
    const char* want = argc > 4 ? argv[4] : "wpp";
    // argv[5]: logical CPUs to bind the workers to, e.g. "0,2,4,6,8,10,12,14". Empty leaves
    // placement to the operating system, which is the default and what every earlier run used.
    //
    // A THREAD COUNT IS NOT A QUANTITY OF COMPUTE UNLESS THE CORES ARE THE SAME. On a
    // heterogeneous part the nth thread may be a core that does 0.59 of the work the first one
    // does, so a speedup column taken across the mix divides by a number that means nothing.
    // Naming a homogeneous set is how this bench produces a curve that can be published.
    std::vector<unsigned> worker_cpus;
    for (int c : parse_int_list(argc > 5 ? argv[5] : nullptr, /*positive_only=*/false))
        worker_cpus.push_back(static_cast<unsigned>(c));
    const auto all = workloads();
    if (std::strcmp(want, "list") == 0) {
        for (const auto& w : all) std::printf("%s\n", w.name);
        return 0;
    }
    // The generated corpus, shared with bench_gpu_evolve through corpus_gen.hpp so a CPU row and
    // a GPU row name the same workload by construction.
    std::vector<Workload> generated;
    for (const auto& g : corpus::corpus()) {
        Workload w;
        w.name = g.name.c_str();
        for (const auto& r : g.rules) {
            auto b = make_rule(static_cast<uint16_t>(w.rules.size()));
            for (const auto& e : r.lhs) b.lhs(std::vector<VertexId>(e.begin(), e.end()));
            for (const auto& e : r.rhs) b.rhs(std::vector<VertexId>(e.begin(), e.end()));
            w.rules.push_back(b.build());
        }
        for (const auto& e : g.init) w.init.push_back(std::vector<VertexId>(e.begin(), e.end()));
        generated.push_back(std::move(w));
    }
    static std::vector<std::string> gen_names;
    for (const auto& g : corpus::corpus()) gen_names.push_back(g.name);
    for (size_t i = 0; i < generated.size(); ++i) generated[i].name = gen_names[i].c_str();

    if (std::strcmp(want, "corpus") == 0) {
        for (const auto& w : generated) std::printf("%s\n", w.name);
        return 0;
    }

    // GROWTH CLASSIFICATION. A workload whose work does not increase with depth cannot distinguish
    // two engines, and cannot show a scaling curve either: it measures their per-call floor and
    // nothing else. Reporting a CPU/GPU ratio or a thread-scaling ratio from one is measuring the
    // harness. This mode reports every generated workload at two depths so the corpus can be
    // checked rather than assumed, and prints canonical and raw counts at each.
    //
    // THE VERDICT IS ON RAW STATES, NOT CANONICAL ONES. A workload can hold its canonical count
    // nearly fixed while exploring thousands of raw states -- star-l3a2g1r2 reaches 22 canonical
    // over 6071 raw at depth three -- so the canonical count answers a different question than the
    // one asked here, and answers it wrongly for any workload whose states are largely isomorphic.
    //
    // AND IT IS UNCAPPED. A state cap decides the outcome: it truncates the deeper run toward the
    // shallower one, which reads as exactly the saturation this mode exists to detect. Depths two
    // and three are what keep it cheap without one -- a rule that cannot fire reports equal counts
    // there, and a rule that reaches a fixed point stops increasing there. The floor rejects the
    // THE VERDICT IS TERMINATION, NOT RATE. A workload is rejected when it stops producing states
    // -- the rule cannot fire, or the system reaches a fixed point -- because from that step on
    // every deeper run repeats the same work and the wall time is the per-call floor. A workload
    // that grows slowly is NOT rejected: slow growth is a workload property the corpus is supposed
    // to contain, and it is where added threads have least to work with, which is exactly the
    // region a scaling curve has to cover rather than exclude.
    if (std::strcmp(want, "corpusgrow") == 0) {
        for (const auto& w : generated) {
            size_t lo_c = 0, hi_c = 0, lo_r = 0, hi_r = 0;
            for (int d : {2, 3}) {
                Hypergraph g(capacity_scale_from_env());
                g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
                hgcommon::RecordSet rs;
                rs.causal = rs.branchial = rs.state_events = rs.raw_events = false;
                g.set_record_set(rs);
                ParallelEvolutionEngine e(&g, 8, ParallelEvolutionEngine::ExecutionMode::Parallel, worker_cpus);
                // Default on: expanding one representative per isomorphism class is the whole
            // point of canonical exploration. The knob exists because that restriction
            // interacts with the depth budget -- a class first claimed at one depth and later
            // reached at a shallower one must be re-expanded with the larger remaining budget
            // -- and comparing the two settings is what shows whether that accounting is right.
            {
                const char* co = std::getenv("HG_BENCH_CANON_ONLY");
                e.set_explore_from_canonical_states_only(!(co && co[0] == '0'));
            }
                for (const auto& r : w.rules) e.add_rule(r);
                e.evolve(w.init, d);
                (d == 2 ? lo_c : hi_c) = g.num_canonical_states();
                (d == 2 ? lo_r : hi_r) = g.num_states();
            }
            std::printf("%-20s d2=%-5zu/%-6zu d3=%-6zu/%-8zu %s\n",
                        w.name, lo_c, lo_r, hi_c, hi_r,
                        (hi_r > lo_r) ? "GROWS" : "dead");
        }
        return 0;
    }

    const Workload* sel = nullptr;
    for (const auto& w : all) if (std::strcmp(w.name, want) == 0) sel = &w;
    for (const auto& w : generated) if (std::strcmp(w.name, want) == 0) sel = &w;
    if (!sel) { std::fprintf(stderr, "unknown workload '%s' (try: list, corpus)\n", want); return 2; }

    double base_ms = 0.0;
    for (int threads : sweep) {
        std::vector<double> ms;
        size_t states = 0, raw = 0;
        size_t hg_arena_hw = 0, hg_arena_used = 0;  // the hypergraph arena's share, read per run
        for (int i = 0; i < iters; ++i) {
            Hypergraph g(capacity_scale_from_env());
            g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            // Same knob as bench_gpu_evolve, so a CPU row and a GPU row record the same
            // artifacts. Without it the CPU would be reconstructing the raw unfolding while the
            // GPU was not, and the ratio between them would be measuring the record set rather
            // than the two devices.
            {
                hgcommon::RecordSet rs = g.record_set();
                if (const char* raw_env = std::getenv("HG_BENCH_RAW"); raw_env && raw_env[0] == '0')
                    rs.causal = rs.branchial = rs.state_events = rs.raw_events = false;
                // The three records have different costs and only one drives the per-instance
                // replay, so an all-or-nothing switch cannot say which is being paid for. Same
                // spelling as bench_gpu_evolve, so a host row and a device row select the same
                // records -- comparing two engines under two different record sets measures the
                // record sets.
                if (const char* v = std::getenv("HG_BENCH_CAUSAL"))    rs.causal     = v[0] != '0';
                if (const char* v = std::getenv("HG_BENCH_BRANCHIAL")) rs.branchial  = v[0] != '0';
                if (const char* v = std::getenv("HG_BENCH_RAWEVENTS")) rs.raw_events = v[0] != '0';
                g.set_record_set(rs);
            }
            ParallelEvolutionEngine e(&g, threads, ParallelEvolutionEngine::ExecutionMode::Parallel, worker_cpus);
            // QUOTIENT IS NOT THE ENGINE'S DEFAULT. explore_from_canonical_states_only_ is false
            // in ParallelEvolutionEngine; this bench turned it on unconditionally, so every
            // number it has produced describes the quotient path and none describes the one a
            // caller gets without asking. Overridable like every other record-set switch beside
            // it, and still on by default so existing comparisons stay comparable.
            {
                const char* co = std::getenv("HG_BENCH_CANON_ONLY");
                e.set_explore_from_canonical_states_only(!(co && co[0] == '0'));
            }
            for (const auto& r : sel->rules) e.add_rule(r);
            const auto t0 = std::chrono::steady_clock::now();
            e.evolve(sel->init, steps);
            const auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            hg_arena_hw = g.arena().block_bytes_high_water();
            hg_arena_used = g.arena().bytes_allocated();
#if HG_ENGINE_STATS
            if (std::getenv("HG_BENCH_ALLOC_PROFILE") && i == iters - 1)
                hg::engine::arena_alloc_profile_dump(stdout, 24);
#endif
            // A TRUNCATED RUN IS NOT A MEASUREMENT. Hitting a container ceiling makes the
            // engine return valid partial work with a warning rather than throwing, so a bench
            // that reads only the counts reports a number for a workload it never finished --
            // and above the ceiling WHICH states got in is decided by the arrival race, so the
            // counts vary between runs and between thread counts for a reason that has nothing
            // to do with the engine's concurrency. Every warning is printed, once per run.
            for (const std::string& w : e.warnings())
                std::printf("  WARNING: %s\n", w.c_str());
            // A REFUSED BINDING IS NOT A PINNED RUN. The work still completes and the timing
            // still prints, so nothing else here would reveal that the cores were the
            // scheduler's choice after all.
            if (!worker_cpus.empty() && e.worker_pin_failures() != 0)
                std::printf("  WARNING: %zu worker(s) could not bind to the requested CPU set; "
                            "this run is NOT pinned\n", e.worker_pin_failures());
            // Twin of the device bench's line: the reconstruction's size, so the two engines can
            // be compared on work done and not only on time.
            // S1: the quotient branchial fan-out. visits/scans is the mean applications per
            // instance (m). The scan costs sum m^2 while the relation is sum m(m-1)/2, so a
            // large m says an inverted index by (instance, slot) -- what the direct path
            // already uses -- would turn the scan linear in the co-consumers it actually finds.
            {
                const size_t sc = g.applied_scans(), vi = g.applied_visits();
                std::printf("  fanout: scans=%zu visits=%zu mean_m=%.2f\n",
                            sc, vi, sc ? double(vi) / double(sc) : 0.0);
            }
            std::printf("  recon: causal_pairs=%zu reduced_pairs=%zu branchial=%zu\n",
                        g.num_reconstructed_causal_pairs(false),
                        g.num_reconstructed_causal_pairs(true),
                        g.num_reconstructed_branchial());
            // WHAT THE REPLAY PAID FOR AGAINST WHAT IT KEPT. Every (instance, match) pair the
            // cross product offers takes a claim, and the width test that rejects a pair whose
            // capture and instance disagree on the class width runs against the pair after it.
            // claims/events is therefore the share of claims spent on pairs that mint nothing,
            // and it is the only number that says whether the order of those two tests matters.
            std::printf("  replay: claims=%zu events=%zu captured=%zu instances=%zu\n",
                        g.applied_claims(), g.num_reconstructed_events(),
                        g.captured_matches(), g.reconstruction_instances());
            {
                const auto ir = g.ir_work();
                std::printf("  ir: calls=%llu searched=%llu leaves=%llu nodes=%llu "
                            "leaves/searched=%.1f nodes/searched=%.1f mean_depth=%.2f "
                            "retries=%llu fallbacks=%llu\n",
                            (unsigned long long)ir.calls, (unsigned long long)ir.searched,
                            (unsigned long long)ir.leaves, (unsigned long long)ir.nodes,
                            ir.searched ? double(ir.leaves) / double(ir.searched) : 0.0,
                            ir.searched ? double(ir.nodes) / double(ir.searched) : 0.0,
                            ir.searched ? double(ir.depth_sum) / double(ir.searched) : 0.0,
                            (unsigned long long)ir.retries, (unsigned long long)ir.fallbacks);
            }
            states = g.num_canonical_states();
            raw = g.num_states();
            // Discriminates a dedup defect from a COUNTING defect. num_canonical_states is
            // count_unique() over the map's resize chain; this is the same quantity taken
            // independently, straight from the states' published canonical hashes. They must
            // agree: a state's hash is what keyed it, so the number of distinct hashes IS the
            // number of isomorphism classes found. A gap means the chain walk counts one key
            // twice, not that the engine failed to merge two states.
            if (const char* v = std::getenv("HG_BENCH_VERIFY_DEDUP"); v && v[0] == '1') {
                std::unordered_set<uint64_t> distinct;
                const size_t n = g.num_states();
                for (size_t s2 = 0; s2 < n; ++s2)
                    distinct.insert(g.get_state(static_cast<StateId>(s2)).canonical_hash);
                // The independent oracle: re-canonicalize one representative per stored hash
                // through the unbounded implementation, which shares no code with the bounded
                // core's search. If it collapses hashes the engine kept apart, the classes are
                // genuinely fewer than the engine found and the bounded key is at fault. If it
                // agrees, the states really are pairwise non-isomorphic and the surplus was
                // produced upstream of canonicalization.
                std::unordered_map<uint64_t, StateId> rep;
                for (size_t s2 = 0; s2 < n; ++s2)
                    rep.emplace(g.get_state(static_cast<StateId>(s2)).canonical_hash,
                                static_cast<StateId>(s2));
                std::unordered_set<uint64_t> oracle;
                IRCanonicalizer oracle_ir;
                for (const auto& [h, sid] : rep) {
                    std::vector<std::vector<VertexId>> ev;
                    g.get_state(sid).edges.for_each([&](EdgeId eid) {
                        const auto& ed = g.get_edge(eid);
                        ev.emplace_back(ed.vertices, ed.vertices + ed.arity);
                    });
                    oracle.insert(oracle_ir.compute_canonical_hash(ev));
                }
                // Dump the class set so two runs can be compared as SETS, not just counts.
                // Equal counts would not prove equal sets, and a subset relation is what
                // distinguishes "one run missed classes" from "the runs explored different
                // regions" -- different defects with different fixes.
                if (const char* dp = std::getenv("HG_BENCH_DUMP_CLASSES")) {
                    std::vector<uint64_t> sorted(oracle.begin(), oracle.end());
                    std::sort(sorted.begin(), sorted.end());
                    if (FILE* f = std::fopen(dp, "w")) {
                        for (uint64_t h : sorted) std::fprintf(f, "%llu\n", (unsigned long long)h);
                        std::fclose(f);
                    }
                }
                std::printf("  verify: count_unique=%zu distinct_hashes=%zu raw=%zu "
                            "oracle_classes=%zu\n",
                            states, distinct.size(), n, oracle.size());
            }
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
        // The arenas' share of the resident set's high-water mark, so a footprint that grows
        // with the worker count is split between the arena blocks (per-worker high-water) and
        // everything else (allocator arenas, stacks, tables).
        std::printf("threads=%d steps=%d canonical=%zu raw=%zu median_ms=%.3f min_ms=%.3f "
                    "speedup=%.2f efficiency=%.2f arena_block_hw_mb=%.1f scratch_hw_mb=%.1f hg_arena_hw_mb=%.1f hg_arena_used_mb=%.1f\n",
                    threads, steps, states, raw, med, ms.front(),
                    base_ms / med, (base_ms / med) / threads,
                    static_cast<double>(hg::engine::arena_block_bytes_high_water()) / (1024.0 * 1024.0),
                    static_cast<double>(hg::engine::arena_scratch_block_bytes_high_water()) / (1024.0 * 1024.0),
                    static_cast<double>(hg_arena_hw) / (1024.0 * 1024.0),
                    static_cast<double>(hg_arena_used) / (1024.0 * 1024.0));
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
#ifdef HG_BITSET_STATS
    hg::engine::bitset_stats_report(want);
#endif
#ifdef HG_MATCH_BRANCH_STATS
    // CANDIDATE-BRANCH COVERAGE, reported so a corpus can be CHECKED to reach all three rather
    // than assumed to. Order matches the branches in pattern_matcher.hpp.
    std::fprintf(stderr,
        "[matchbranch:%s] arity_scan=%llu repeated_var_seed=%llu bound_intersect=%llu\n", want,
        (unsigned long long)hg::engine::match_branch_count(0),
        (unsigned long long)hg::engine::match_branch_count(1),
        (unsigned long long)hg::engine::match_branch_count(2));
#endif
    return 0;
}
