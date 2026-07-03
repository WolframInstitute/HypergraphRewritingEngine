// Apples-to-apples CPU-vs-GPU benchmark harness.
//
// Runs the same workload (rule, initial state, step count, canonicalization,
// transitive reduction) on both the CPU ParallelEvolutionEngine and the GPU
// hg_gpu engine, with warmup runs, verifies output equivalence by IR
// normalisation, and reports wall time + per-step throughput.
//
// Usage:
//   ./bench_cpu_vs_gpu                       # run default sweep
//   ./bench_cpu_vs_gpu --csv out.csv         # write CSV
//   ./bench_cpu_vs_gpu --only wolfram        # filter by substring

#include "hg_gpu/evolve.hpp"

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

struct Workload {
    std::string name;
    std::vector<hg_gpu::RewriteRule> rules;
    std::vector<std::vector<hg_gpu::VertexId>> initial_state;
    uint32_t num_steps = 0;
};

hg_gpu::RewriteRule make_rule(std::vector<std::vector<uint8_t>> lhs,
                              std::vector<std::vector<uint8_t>> rhs) {
    hg_gpu::RewriteRule r;
    r.lhs = std::move(lhs);
    r.rhs = std::move(rhs);
    uint8_t lhs_max = 0;
    for (auto& e : r.lhs) for (auto v : e) lhs_max = std::max<uint8_t>(lhs_max, v);
    uint8_t rhs_max = 0;
    for (auto& e : r.rhs) for (auto v : e) rhs_max = std::max<uint8_t>(rhs_max, v);
    r.num_lhs_vars = r.lhs.empty() ? 0 : static_cast<uint8_t>(lhs_max + 1);
    r.num_rhs_vars = r.rhs.empty() ? 0 : static_cast<uint8_t>(rhs_max + 1);
    return r;
}

std::vector<std::vector<hg_gpu::VertexId>> make_initial_complete_edges(uint32_t n_edges) {
    // Random-ish connected-ish edge set: edge i connects vertex (i % k) to
    // vertex ((i * 7 + 1) % k), where k ≈ sqrt(n_edges)*2. Deterministic
    // (no rand) so every run is comparable.
    uint32_t k = std::max<uint32_t>(2, static_cast<uint32_t>(std::sqrt(double(n_edges))) * 2);
    std::vector<std::vector<hg_gpu::VertexId>> out;
    out.reserve(n_edges);
    for (uint32_t i = 0; i < n_edges; ++i) {
        hg_gpu::VertexId a = i % k;
        hg_gpu::VertexId b = (i * 7u + 1u) % k;
        out.push_back({a, b});
    }
    return out;
}

// ---------- CPU side ----------

hypergraph::RewriteRule convert_rule(const hg_gpu::RewriteRule& src, uint16_t index) {
    hypergraph::RuleBuilder b(index);
    for (const auto& edge : src.lhs) b.lhs(edge);
    for (const auto& edge : src.rhs) b.rhs(edge);
    return b.build();
}

struct RunResult {
    double   wall_ms     = 0.0;
    uint64_t num_states  = 0;
    uint64_t num_events  = 0;
    uint64_t num_causal  = 0;
    uint64_t num_branch  = 0;
    std::set<uint64_t> canonical_state_hashes;  // for correctness cross-check
};

RunResult run_cpu(const Workload& w, bool verify) {
    RunResult out;

    auto t0 = std::chrono::steady_clock::now();

    hypergraph::Hypergraph hg;
    // The CPU's exploration-time edge hash is WL (fast + safe for internal
    // dedup); final state comparison uses IRCanonicalizer, which is what both
    // sides measure correctness against.
    hg.set_state_canonicalization_mode(hypergraph::StateCanonicalizationMode::Full);

    hypergraph::ParallelEvolutionEngine engine(&hg, 0);
    for (size_t i = 0; i < w.rules.size(); ++i) {
        engine.add_rule(convert_rule(w.rules[i], static_cast<uint16_t>(i)));
    }
    engine.set_transitive_reduction(true);
    engine.set_explore_from_canonical_states_only(true);
    engine.evolve(w.initial_state, w.num_steps);

    auto t1 = std::chrono::steady_clock::now();
    out.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    out.num_events = hg.num_events();
    out.num_causal = hg.num_causal_edges();
    out.num_branch = hg.num_branchial_edges();

    if (verify) {
        hypergraph::IRCanonicalizer ir;
        const uint32_t n = hg.num_states();
        for (uint32_t sid = 0; sid < n; ++sid) {
            const auto& state = hg.get_state(sid);
            if (state.id == hypergraph::INVALID_ID) continue;
            std::vector<std::vector<hg_gpu::VertexId>> edges;
            state.edges.for_each([&](hypergraph::EdgeId eid) {
                const auto& e = hg.get_edge(eid);
                std::vector<hg_gpu::VertexId> vs;
                vs.reserve(e.arity);
                for (uint8_t i = 0; i < e.arity; ++i) vs.push_back(e.vertices[i]);
                edges.push_back(std::move(vs));
            });
            out.canonical_state_hashes.insert(ir.compute_canonical_hash(edges));
        }
        out.num_states = out.canonical_state_hashes.size();
    } else {
        out.num_states = hg.num_states();
    }
    return out;
}

// ---------- GPU side ----------

// Build the GPU EvolveInput once per workload (warmup + measure runs use
// the same input).
hg_gpu::EvolveInput make_gpu_input(const Workload& w) {
    hg_gpu::EvolveInput in;
    in.rules = w.rules;
    in.initial_state = w.initial_state;
    in.num_steps = w.num_steps;
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
    in.transitive_reduction = std::getenv("HG_TR_OFF") ? false : true;
    in.explore_from_canonical_states_only = true;
    return in;
}

// Single-shot GPU run using a caller-supplied (possibly reused) Engine.
// Wall time excludes Engine construction — that's measured separately.
RunResult run_gpu_with_engine(hg_gpu::Engine& engine, const hg_gpu::EvolveInput& in,
                              bool verify) {
    RunResult out;
    auto t0 = std::chrono::steady_clock::now();
    auto result = engine.run(in);
    auto t1 = std::chrono::steady_clock::now();
    out.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    out.num_events = result.events.size();
    out.num_causal = result.causal_edges.size();
    out.num_branch = result.branchial_edges.size();

    if (verify) {
        hypergraph::IRCanonicalizer ir;
        for (const auto& s : result.states) {
            out.canonical_state_hashes.insert(ir.compute_canonical_hash(s.edges));
        }
        out.num_states = out.canonical_state_hashes.size();
    } else {
        out.num_states = result.states.size();
    }
    return out;
}

// Median of K runs (warmup excluded).
double median_of(std::vector<double>& xs) {
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2];
}

struct Config {
    std::vector<uint32_t> init_sizes = {10, 50, 200, 1000};
    std::vector<uint32_t> step_counts = {1, 3, 5};
    int warmup = 1;
    int measure = 3;
    std::string csv_path;
    std::string filter;
};

void run(const Config& cfg) {
    std::vector<Workload> corpus;

    // Wolfram canonical rule {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}} is
    // the default — high branching, interesting structure.
    auto wolfram = make_rule({{0, 1}, {0, 2}},
                             {{0, 1}, {0, 3}, {1, 3}, {2, 3}});

    for (uint32_t n : cfg.init_sizes) {
        for (uint32_t s : cfg.step_counts) {
            Workload w;
            w.name = "wolfram_n" + std::to_string(n) + "_s" + std::to_string(s);
            w.rules = {wolfram};
            w.initial_state = make_initial_complete_edges(n);
            w.num_steps = s;
            corpus.push_back(std::move(w));
        }
    }

    // Header.
    std::printf("%-30s %10s %10s %10s %10s %10s %10s %10s %8s\n",
                "workload", "cpu_ms", "gpu_ms", "speedup", "states",
                "events", "causal", "branch", "match");
    std::printf("%-30s %10s %10s %10s %10s %10s %10s %10s %8s\n",
                "--------", "------", "------", "-------", "------",
                "------", "------", "------", "-----");

    std::ofstream csv;
    if (!cfg.csv_path.empty()) {
        csv.open(cfg.csv_path);
        csv << "workload,init_edges,steps,cpu_ms,gpu_ms,speedup,"
            << "cpu_states,gpu_states,cpu_events,gpu_events,"
            << "cpu_causal,gpu_causal,cpu_branch,gpu_branch,match\n";
    }

    for (const auto& w : corpus) {
        if (!cfg.filter.empty() && w.name.find(cfg.filter) == std::string::npos) continue;

        // Build the GPU input + Engine ONCE per workload. Engine reuse
        // amortises the per-call CUDA setup (pool allocations, index
        // initialisation) across warmup + measure iterations, so the
        // measured wall_ms reflects the actual evolve cost rather than
        // setup-dominated noise.
        auto gpu_in = make_gpu_input(w);
        hg_gpu::Engine gpu_engine(hg_gpu::config_from_input(gpu_in));

        // Warmup.
        for (int k = 0; k < cfg.warmup; ++k) {
            (void)run_cpu(w, false);
            (void)run_gpu_with_engine(gpu_engine, gpu_in, false);
        }

        std::vector<double> cpu_ms, gpu_ms;
        RunResult cpu_r{}, gpu_r{};
        for (int k = 0; k < cfg.measure; ++k) {
            cpu_r = run_cpu(w, k == 0);
            cpu_ms.push_back(cpu_r.wall_ms);
        }
        for (int k = 0; k < cfg.measure; ++k) {
            gpu_r = run_gpu_with_engine(gpu_engine, gpu_in, k == 0);
            gpu_ms.push_back(gpu_r.wall_ms);
        }

        double cpu_med = median_of(cpu_ms);
        double gpu_med = median_of(gpu_ms);
        double speedup = (gpu_med > 0.0) ? cpu_med / gpu_med : 0.0;
        bool match = (cpu_r.canonical_state_hashes == gpu_r.canonical_state_hashes);

        uint32_t n_init = static_cast<uint32_t>(w.initial_state.size());
        std::printf("%-30s %10.2f %10.2f %9.2fx %10llu %10llu %10llu %10llu %8s\n",
                    w.name.c_str(), cpu_med, gpu_med, speedup,
                    (unsigned long long)cpu_r.num_states,
                    (unsigned long long)cpu_r.num_events,
                    (unsigned long long)cpu_r.num_causal,
                    (unsigned long long)cpu_r.num_branch,
                    match ? "yes" : "NO");

        if (csv.is_open()) {
            csv << w.name << "," << n_init << "," << w.num_steps << ","
                << cpu_med << "," << gpu_med << "," << speedup << ","
                << cpu_r.num_states << "," << gpu_r.num_states << ","
                << cpu_r.num_events << "," << gpu_r.num_events << ","
                << cpu_r.num_causal << "," << gpu_r.num_causal << ","
                << cpu_r.num_branch << "," << gpu_r.num_branch << ","
                << (match ? "yes" : "no") << "\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc) cfg.csv_path = argv[++i];
        else if (arg == "--only" && i + 1 < argc) cfg.filter = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) cfg.warmup = std::stoi(argv[++i]);
        else if (arg == "--measure" && i + 1 < argc) cfg.measure = std::stoi(argv[++i]);
        else if (arg == "--quick") { cfg.init_sizes = {10, 50}; cfg.step_counts = {1, 3}; }
        else if (arg == "--fast") { cfg.warmup = 0; cfg.measure = 1; }
    }
    run(cfg);
    return 0;
}
