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

    // GROWTH CLASSIFICATION. A workload whose canonical state count does not increase with depth
    // cannot distinguish two engines: it measures their per-call floor and nothing else. Running
    // such a workload and reporting a CPU/GPU ratio from it is measuring the harness. This mode
    // reports the state count at two depths for every generated workload so the corpus can be
    // filtered to the ones that actually evolve, and prints the ratio that decides it.
    if (std::strcmp(want, "corpusgrow") == 0) {
        for (const auto& w : generated) {
            size_t lo = 0, hi = 0;
            for (int d : {3, 6}) {
                Hypergraph g;
                g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
                hgcommon::RecordSet rs;
                rs.causal = rs.branchial = rs.state_events = rs.raw_events = false;
                g.set_record_set(rs);
                ParallelEvolutionEngine e(&g, 8);
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
                (d == 3 ? lo : hi) = g.num_canonical_states();
            }
            std::printf("%-18s d3=%-6zu d6=%-8zu %s\n", w.name, lo, hi,
                        (hi > lo * 2 && hi >= 20) ? "GROWS" : "flat");
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
        for (int i = 0; i < iters; ++i) {
            Hypergraph g;
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
            ParallelEvolutionEngine e(&g, threads);
            e.set_explore_from_canonical_states_only(true);
            for (const auto& r : sel->rules) e.add_rule(r);
            const auto t0 = std::chrono::steady_clock::now();
            e.evolve(sel->init, steps);
            const auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            // A TRUNCATED RUN IS NOT A MEASUREMENT. Hitting a container ceiling makes the
            // engine return valid partial work with a warning rather than throwing, so a bench
            // that reads only the counts reports a number for a workload it never finished --
            // and above the ceiling WHICH states got in is decided by the arrival race, so the
            // counts vary between runs and between thread counts for a reason that has nothing
            // to do with the engine's concurrency. Every warning is printed, once per run.
            for (const std::string& w : e.warnings())
                std::printf("  WARNING: %s\n", w.c_str());
            // Twin of the device bench's line: the reconstruction's size, so the two engines can
            // be compared on work done and not only on time.
            std::printf("  recon: causal_pairs=%zu reduced_pairs=%zu\n",
                        g.num_reconstructed_causal_pairs(false),
                        g.num_reconstructed_causal_pairs(true));
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
