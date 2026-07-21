// Wall-clock timing harness + multi-thread scaling for the multiway engine.
//
// Sibling to tools/cost_matrix.cpp: where the cost matrix proves memory/heap
// wins and oracle-exactness, this harness proves WALL-CLOCK wins and shows how
// well we keep cores busy. It reuses reference/oracle_corpus.hpp verbatim, so
// every timed workload is one the oracle gate already certifies exact — no
// fabricated benchmarks.
//
// What it reports, per corpus case:
//   * end-to-end evolution wall time (steady_clock), MEDIAN over >=5 repeats
//     after a discarded warmup, plus median-absolute-deviation (robust spread);
//   * us per canonical state and canonical states/sec;
//   * multi-thread scaling at 1/2/4/8/16 workers: speedup vs 1 thread and
//     parallel efficiency (are we actually using the cores?).
// Full-mode canonical-state count is a graph invariant, so it MUST be identical
// across thread counts; the harness checks this and flags any drift as
// non-deterministic (an oracle-consistency guard).
//
// Usage:
//   timing_harness [--repeats N] [--threads a,b,c] [--out CSV] [--compare CSV]
//   timing_harness --heavy <caseIdx> <steps> <threads>   // single run, for perf
//
// Baseline / regression:
//   timing_harness --out benchmark_results/timing_baseline.csv   # capture
//   timing_harness --compare benchmark_results/timing_baseline.csv# prove deltas

#include "../reference/oracle_corpus.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace hypergraph;
using clk = std::chrono::steady_clock;

namespace {

// Per-case measurement depth: the deepest step count whose single-thread run
// completes in ~1s here, so 5+ repeats x 5 thread counts stay in a few seconds
// each. Calibrated against the corpus; heavy growers (binary-growth, wolfram,
// arity3, disconnected causal blow-up) sit near their 1s knee, trivial/saturating
// cases run deep and cheap. Parallel to oracle::corpus() order.
const int kMeasureSteps[] = {8, 6, 6, 10, 12, 12, 14, 8, 12, 6};

struct RunResult {
    double ms;
    size_t canonical_states;
    size_t events;
};

// One full evolution from scratch at the given worker count; returns wall ms and
// the invariant counts. A fresh Hypergraph+engine each call so timing includes
// nothing carried over between repeats.
RunResult run_once(const oracle::Case& c, int steps, unsigned threads) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_transitive_reduction(true);
    for (const auto& r : c.rules) engine.add_rule(r);

    auto t0 = clk::now();
    engine.evolve(c.init, steps);
    auto t1 = clk::now();

    RunResult r;
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.canonical_states = hg.num_canonical_states();
    r.events = hg.num_events();
    return r;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Median absolute deviation: median(|x_i - median(x)|). Robust spread, no mean.
double mad(const std::vector<double>& v) {
    double m = median(v);
    std::vector<double> dev;
    dev.reserve(v.size());
    for (double x : v) dev.push_back(std::fabs(x - m));
    return median(std::move(dev));
}

struct Row {
    std::string name;
    std::string type;
    int steps;
    unsigned threads;
    double median_ms;
    double mad_ms;
    size_t canonical_states;
    size_t events;
    double us_per_state;
    double states_per_sec;
    double speedup_vs1;
    double efficiency;
};

std::string csv_key(const std::string& name, unsigned threads) {
    return name + "@" + std::to_string(threads);
}

std::map<std::string, Row> load_csv(const char* path) {
    std::map<std::string, Row> out;
    FILE* f = std::fopen(path, "r");
    if (!f) return out;
    char line[1024];
    bool header = true;
    while (std::fgets(line, sizeof(line), f)) {
        if (header) { header = false; continue; }
        Row r;
        char name[128], type[128];
        unsigned th;
        if (std::sscanf(line,
                "%127[^,],%127[^,],%d,%u,%lf,%lf,%zu,%zu,%lf,%lf,%lf,%lf",
                name, type, &r.steps, &th, &r.median_ms, &r.mad_ms,
                &r.canonical_states, &r.events, &r.us_per_state,
                &r.states_per_sec, &r.speedup_vs1, &r.efficiency) == 12) {
            r.name = name;
            r.type = type;
            r.threads = th;
            out[csv_key(r.name, th)] = r;
        }
    }
    std::fclose(f);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    int repeats = 5;
    std::vector<unsigned> thread_counts = {1, 2, 4, 8, 16};
    const char* out_path = nullptr;
    const char* compare_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--compare") && i + 1 < argc) {
            compare_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            thread_counts.clear();
            char* s = argv[++i];
            for (char* tok = std::strtok(s, ","); tok; tok = std::strtok(nullptr, ","))
                thread_counts.push_back((unsigned)std::atoi(tok));
        } else if (!std::strcmp(argv[i], "--heavy") && i + 3 < argc) {
            // Single evolution for an external profiler (perf/callgrind) to
            // attach to; no repeats, no CSV. Prints the invariant counts so the
            // profiled run is identifiable.
            int idx = std::atoi(argv[i + 1]);
            int steps = std::atoi(argv[i + 2]);
            unsigned th = (unsigned)std::atoi(argv[i + 3]);
            auto cases = oracle::corpus();
            const auto& c = cases.at(idx);
            RunResult r = run_once(c, steps, th);
            std::printf("HEAVY %s steps=%d threads=%u canon=%zu events=%zu ms=%.2f\n",
                        c.name, steps, th, r.canonical_states, r.events, r.ms);
            return 0;
        }
    }

    auto cases = oracle::corpus();
    std::vector<Row> rows;

    std::printf("wall-clock timing harness  (repeats=%d, warmup=1, steady_clock)\n", repeats);
    std::printf("%-18s %-20s %5s %3s %11s %9s %11s %13s %8s %7s\n",
                "case", "type", "steps", "T", "median_ms", "mad_ms",
                "canon", "states/s", "speedup", "eff");
    std::printf("%s\n", std::string(120, '-').c_str());

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const auto& c = cases[ci];
        int steps = kMeasureSteps[ci];

        double base_ms = 0.0;       // 1-thread median, for speedup
        size_t base_canon = 0;      // invariant reference across thread counts
        bool determinism_ok = true;

        for (unsigned th : thread_counts) {
            run_once(c, steps, th);  // warmup, discarded

            std::vector<double> samples;
            samples.reserve(repeats);
            RunResult last{};
            for (int r = 0; r < repeats; ++r) {
                last = run_once(c, steps, th);
                samples.push_back(last.ms);
            }
            double med = median(samples);
            double md = mad(samples);

            if (th == thread_counts.front()) {
                base_ms = med;
                base_canon = last.canonical_states;
            } else if (last.canonical_states != base_canon) {
                determinism_ok = false;  // invariant drifted -> non-deterministic
            }

            Row row;
            row.name = c.name;
            row.type = c.type;
            row.steps = steps;
            row.threads = th;
            row.median_ms = med;
            row.mad_ms = md;
            row.canonical_states = last.canonical_states;
            row.events = last.events;
            row.us_per_state = last.canonical_states
                ? (med * 1000.0 / (double)last.canonical_states) : 0.0;
            row.states_per_sec = med > 0.0
                ? (double)last.canonical_states / (med / 1000.0) : 0.0;
            row.speedup_vs1 = med > 0.0 ? base_ms / med : 0.0;
            row.efficiency = th ? row.speedup_vs1 / (double)th : 0.0;
            rows.push_back(row);

            std::printf("%-18s %-20s %5d %3u %11.2f %9.2f %11zu %13.0f %7.2fx %6.0f%%\n",
                        row.name.c_str(), row.type.c_str(), row.steps, row.threads,
                        row.median_ms, row.mad_ms, row.canonical_states,
                        row.states_per_sec, row.speedup_vs1, row.efficiency * 100.0);
        }
        if (!determinism_ok)
            std::printf("  *** %s: canonical-state count varies across thread counts (NON-DETERMINISTIC)\n",
                        c.name);
        std::printf("\n");
    }

    if (out_path) {
        FILE* f = std::fopen(out_path, "w");
        if (!f) {
            std::fprintf(stderr, "cannot open %s for write\n", out_path);
            return 2;
        }
        std::fprintf(f, "case,type,steps,threads,median_ms,mad_ms,canonical_states,"
                        "events,us_per_state,states_per_sec,speedup_vs1,efficiency\n");
        for (const auto& r : rows) {
            std::fprintf(f, "%s,%s,%d,%u,%.4f,%.4f,%zu,%zu,%.4f,%.2f,%.4f,%.4f\n",
                         r.name.c_str(), r.type.c_str(), r.steps, r.threads,
                         r.median_ms, r.mad_ms, r.canonical_states, r.events,
                         r.us_per_state, r.states_per_sec, r.speedup_vs1, r.efficiency);
        }
        std::fclose(f);
        std::printf("wrote baseline CSV: %s (%zu rows)\n", out_path, rows.size());
    }

    if (compare_path) {
        auto base = load_csv(compare_path);
        if (base.empty()) {
            std::fprintf(stderr, "compare: no rows loaded from %s\n", compare_path);
            return 2;
        }
        std::printf("\ncompare vs %s  (negative ms delta = faster)\n", compare_path);
        std::printf("%-18s %3s %11s %11s %9s %11s\n",
                    "case", "T", "base_ms", "now_ms", "d_ms", "d_ms%");
        std::printf("%s\n", std::string(72, '-').c_str());
        for (const auto& r : rows) {
            auto it = base.find(csv_key(r.name, r.threads));
            if (it == base.end()) continue;
            double b = it->second.median_ms;
            double d = r.median_ms - b;
            double dp = b > 0.0 ? 100.0 * d / b : 0.0;
            std::printf("%-18s %3u %11.2f %11.2f %9.2f %10.1f%%\n",
                        r.name.c_str(), r.threads, b, r.median_ms, d, dp);
        }
    }

    return 0;
}
